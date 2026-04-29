"""Distill the Cellpose-SAM image encoder into a CPU-friendly student.

The script trains a CPU-friendly student encoder to match the feature map
produced by the CPSAM/SAM encoder neck. The default student is a compact
depthwise-separable CNN; an optional MobileNetV3-small backbone can be selected
with --student-backbone mobilenet-v3-small. Teacher features are computed on
the fly for each batch, avoiding the need to store large feature tensors on
disk.

Example:
    python tools/train_cpsam_encoder_distill.py --data-dir /path/to/images --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

# Some Windows conda environments used by the GUI load more than one OpenMP
# runtime through torch/cv2/nd2 dependencies. Set before importing numpy/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cellpose import io, models
from cellpose.transforms import normalize_img, random_rotate_and_resize
from cellpose.vit_sam import Transformer
from guv_app.services.image_service import ImageService


IMAGE_SUFFIXES = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".nd2",
    ".lif",
    ".npy",
}


@dataclass
class DistillConfig:
    data_dir: str | None
    file_list: str | None
    train_root_dirs: tuple[str, ...]
    output_dir: str
    cpsam_model: str
    tile_size: int
    epochs: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    student_backbone: str
    student_width: int
    student_depth: int
    mobilenet_weights: str
    mobilenet_weights_path: str | None
    mobilenet_tap_layer: int
    init_student: str | None
    resume_student: str | None
    sampling_mode: str
    tile_overlap: float
    augment_scale_range: float
    augment_rotate: bool
    augment_flip: bool
    seed: int
    teacher_device: str
    train_device: str
    amp: bool
    normalize_percentiles: tuple[float, float]
    feature_loss_weight: float
    output_loss_weight: float
    train_cpsam_head: bool


@dataclass(frozen=True)
class FrameRef:
    filename: str
    frame_id: str | None = None
    source_group: str = "default"


@dataclass(frozen=True)
class TileSpec:
    ref_index: int
    y0: int
    x0: int


@dataclass(frozen=True)
class ImageFileRef:
    path: Path
    source_group: str


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CPSAMEncoderStudent(nn.Module):
    """Small CPU-oriented encoder with total stride 8 and 256 output channels."""

    def __init__(self, width: int = 64, depth: int = 2, out_channels: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
        stages = []
        channels = width
        for stride in (2, 2):
            next_channels = min(out_channels, channels * 2)
            stages.extend(
                [
                    nn.Conv2d(channels, next_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(
                        next_channels,
                        next_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=next_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            channels = next_channels
            for _ in range(depth):
                stages.append(DepthwiseSeparableBlock(channels))
        self.body = nn.Sequential(*stages)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(self.stem(x)))


class MobileNetV3EncoderStudent(nn.Module):
    """MobileNetV3-small encoder with a CPSAM feature-map adapter."""

    def __init__(
        self,
        weights: str = "imagenet",
        weights_path: str | None = None,
        tap_layer: int = 3,
        out_channels: int = 256,
        target_stride: int = 8,
    ):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except Exception as exc:
            raise ImportError(
                "The MobileNetV3 student requires torchvision. Install torchvision "
                "or use --student-backbone compact."
            ) from exc

        if weights == "imagenet":
            tv_weights = MobileNet_V3_Small_Weights.DEFAULT
        elif weights == "none":
            tv_weights = None
        else:
            raise ValueError("--mobilenet-weights must be 'imagenet' or 'none'.")

        model = mobilenet_v3_small(weights=tv_weights)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            state = checkpoint.get("model_state_dict", state)
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
                if torch.is_tensor(value)
            }
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    "mobilenet checkpoint load: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        features = list(model.features.children())
        if tap_layer < 0 or tap_layer >= len(features):
            raise ValueError(
                f"--mobilenet-tap-layer must be between 0 and {len(features) - 1}."
            )
        self.encoder = nn.Sequential(*features[: tap_layer + 1])
        self.target_stride = int(target_stride)

        with torch.no_grad():
            probe = torch.zeros(1, 3, 256, 256)
            in_channels = int(self.encoder(probe).shape[1])

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = (
            max(1, x.shape[-2] // self.target_stride),
            max(1, x.shape[-1] // self.target_stride),
        )
        features = self.encoder(x)
        adapted = self.adapter(features)
        if adapted.shape[-2:] != target_hw:
            adapted = F.interpolate(
                adapted,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return adapted


class CPSAMReadoutHead(nn.Module):
    """CPSAM neck-feature readout copied from the teacher."""

    def __init__(self, out: nn.Conv2d, w2: torch.Tensor, ps: int):
        super().__init__()
        self.out = copy.deepcopy(out)
        self.W2 = nn.Parameter(w2.detach().clone(), requires_grad=False)
        self.ps = int(ps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.out(features)
        return F.conv_transpose2d(x, self.W2, stride=self.ps, padding=0)


class ImageTileDataset(Dataset):
    def __init__(
        self,
        refs: Sequence[FrameRef],
        tile_size: int,
        normalize_percentiles: tuple[float, float],
        samples_per_epoch: int,
        sampling_mode: str,
        tile_overlap: float,
        augment_scale_range: float,
        augment_rotate: bool,
        augment_flip: bool,
    ):
        if not refs:
            raise ValueError("No training image files found.")
        self.refs = list(refs)
        self.image_service = None
        self.tile_size = int(tile_size)
        self.normalize_percentiles = normalize_percentiles
        self.samples_per_epoch = int(samples_per_epoch)
        self.sampling_mode = sampling_mode
        self.tile_overlap = float(tile_overlap)
        self.augment_scale_range = float(augment_scale_range)
        self.augment_rotate = bool(augment_rotate)
        self.augment_flip = bool(augment_flip)
        self.tile_specs = (
            self._build_tile_specs() if sampling_mode == "overlap-grid" else []
        )
        self.ref_groups = self._group_indices_by_source(self.refs)
        self.group_names = sorted(self.ref_groups)
        self.tile_groups = self._group_tile_specs_by_source() if self.tile_specs else {}

    def __len__(self) -> int:
        if self.sampling_mode == "overlap-grid":
            if not self.tile_groups:
                return 0
            return max(len(indices) for indices in self.tile_groups.values()) * len(self.tile_groups)
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.sampling_mode == "overlap-grid":
            spec = self._balanced_tile_spec(index)
            ref = self.refs[spec.ref_index]
            image = load_image(ref, self._image_service())
            image = to_channels_last_3(image)
            tile = crop_tile(image, self.tile_size, spec.y0, spec.x0)
            tile = normalize_tile_cellpose(tile, self.normalize_percentiles)
            tile = np.moveaxis(tile, -1, 0).astype(np.float32, copy=False)
            return torch.from_numpy(tile)

        ref = self._balanced_ref(index)
        image = load_image(ref, self._image_service())
        image = to_channels_last_3(image)
        image = normalize_tile_cellpose(image, self.normalize_percentiles)
        tile = augment_cellpose_style(
            image,
            self.tile_size,
            self.augment_scale_range,
            self.augment_rotate,
            self.augment_flip,
        )
        tile = np.moveaxis(tile, -1, 0).astype(np.float32, copy=False)
        return torch.from_numpy(tile)

    def _image_service(self) -> ImageService:
        if self.image_service is None:
            self.image_service = ImageService()
        return self.image_service

    @staticmethod
    def _group_indices_by_source(refs: Sequence[FrameRef]) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for index, ref in enumerate(refs):
            groups.setdefault(ref.source_group, []).append(index)
        return groups

    def _balanced_ref(self, index: int) -> FrameRef:
        group_name = self.group_names[index % len(self.group_names)]
        candidates = self.ref_groups[group_name]
        ref_index = random.choice(candidates)
        return self.refs[ref_index]

    def _group_tile_specs_by_source(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for spec_index, spec in enumerate(self.tile_specs):
            group_name = self.refs[spec.ref_index].source_group
            groups.setdefault(group_name, []).append(spec_index)
        return groups

    def _balanced_tile_spec(self, index: int) -> TileSpec:
        group_names = sorted(self.tile_groups)
        group_name = group_names[index % len(group_names)]
        candidates = self.tile_groups[group_name]
        within_group_index = (index // len(group_names)) % len(candidates)
        return self.tile_specs[candidates[within_group_index]]

    def _build_tile_specs(self) -> list[TileSpec]:
        specs: list[TileSpec] = []
        for ref_index, ref in enumerate(self.refs):
            image = to_channels_last_3(load_image(ref, self._image_service()))
            h, w = image.shape[:2]
            for y0, x0 in overlap_tile_origins(h, w, self.tile_size, self.tile_overlap):
                specs.append(TileSpec(ref_index=ref_index, y0=y0, x0=x0))
        if not specs:
            raise ValueError("No overlap-grid tiles could be created.")
        return specs


def load_image(ref: FrameRef, image_service: ImageService) -> np.ndarray:
    path = Path(ref.filename)
    if ref.frame_id is not None:
        arr = image_service.load_frame(ref.filename, ref.frame_id)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = arr.item()
        if isinstance(arr, dict):
            for key in ("image", "raw_image", "img", "data"):
                if key in arr:
                    return np.asarray(arr[key])
            raise ValueError(f"No image-like key found in {path}")
        return np.asarray(arr)
    else:
        arr = image_service.load_image(ref.filename)
    if arr is None:
        raise ValueError(f"Could not load image/frame: {ref}")
    return np.asarray(arr)


def to_channels_last_3(image: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(image))
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = np.moveaxis(arr, 0, -1)
        elif arr.shape[-1] > 8:
            # Treat first plane as a stack/time axis and keep one plane.
            arr = arr[0, ..., None] if arr.shape[0] > 8 else arr[..., :1]
    else:
        while arr.ndim > 3:
            arr = arr[0]
        return to_channels_last_3(arr)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 2:
        arr = np.concatenate([arr, arr[..., :1]], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr.astype(np.float32, copy=False)


def random_tile(image: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    pad_y = max(0, tile_size - h)
    pad_x = max(0, tile_size - w)
    if pad_y or pad_x:
        image = np.pad(
            image,
            ((0, pad_y), (0, pad_x), (0, 0)),
            mode="reflect" if h > 1 and w > 1 else "edge",
        )
        h, w = image.shape[:2]
    y0 = random.randint(0, h - tile_size)
    x0 = random.randint(0, w - tile_size)
    tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
    if random.random() < 0.5:
        tile = tile[:, ::-1]
    if random.random() < 0.5:
        tile = tile[::-1]
    k = random.randint(0, 3)
    if k:
        tile = np.rot90(tile, k)
    return np.ascontiguousarray(tile)


def crop_tile(image: np.ndarray, tile_size: int, y0: int, x0: int) -> np.ndarray:
    h, w = image.shape[:2]
    pad_y = max(0, y0 + tile_size - h)
    pad_x = max(0, x0 + tile_size - w)
    if pad_y or pad_x:
        image = np.pad(
            image,
            ((0, pad_y), (0, pad_x), (0, 0)),
            mode="reflect" if h > 1 and w > 1 else "edge",
        )
    return np.ascontiguousarray(image[y0 : y0 + tile_size, x0 : x0 + tile_size])


def overlap_tile_origins(h: int, w: int, tile_size: int, overlap: float) -> list[tuple[int, int]]:
    overlap = min(0.95, max(0.0, float(overlap)))
    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    def axis_origins(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        origins = list(range(0, max(1, length - tile_size + 1), stride))
        last = length - tile_size
        if origins[-1] != last:
            origins.append(last)
        return origins

    return [(y, x) for y in axis_origins(h) for x in axis_origins(w)]


def normalize_tile_cellpose(tile: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    tile = tile.astype(np.float32, copy=False)
    return normalize_img(
        tile,
        normalize=True,
        percentile=tuple(percentiles),
        norm3D=False,
        axis=-1,
    ).astype(np.float32, copy=False)


def augment_cellpose_style(
    image: np.ndarray,
    tile_size: int,
    scale_range: float,
    rotate: bool,
    do_flip: bool,
) -> np.ndarray:
    chw = np.moveaxis(image, -1, 0).astype(np.float32, copy=False)
    imgi, _, _ = random_rotate_and_resize(
        [chw],
        Y=None,
        scale_range=scale_range,
        xy=(tile_size, tile_size),
        do_flip=do_flip,
        rotate=rotate,
    )
    return np.moveaxis(imgi[0], 0, -1)


def teacher_encoder_forward(teacher: Transformer, x: torch.Tensor) -> torch.Tensor:
    enc = teacher.encoder
    z = enc.patch_embed(x)
    if enc.pos_embed is not None:
        z = z + enc.pos_embed
    for block in enc.blocks:
        z = block(z)
    return enc.neck(z.permute(0, 3, 1, 2))


def cpsam_readout(features: torch.Tensor, out: nn.Module, w2: torch.Tensor, ps: int) -> torch.Tensor:
    x = out(features)
    return F.conv_transpose2d(x, w2, stride=ps, padding=0)


def feature_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    if student_features.shape[-2:] != teacher_features.shape[-2:]:
        student_features = F.interpolate(
            student_features,
            size=teacher_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    mse = F.mse_loss(student_features, teacher_features)
    cos = 1.0 - F.cosine_similarity(
        student_features.flatten(2),
        teacher_features.flatten(2),
        dim=1,
    ).mean()
    return mse + 0.1 * cos


def output_distill_loss(student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
    if student_output.shape[-2:] != teacher_output.shape[-2:]:
        student_output = F.interpolate(
            student_output,
            size=teacher_output.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    flow_mse = F.mse_loss(student_output[:, :-1], teacher_output[:, :-1])
    cellprob_mse = F.mse_loss(student_output[:, -1:], teacher_output[:, -1:])

    def gradient_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dy = x[..., 1:, :] - x[..., :-1, :]
        dx = x[..., :, 1:] - x[..., :, :-1]
        return dy, dx

    s_dy, s_dx = gradient_xy(student_output[:, -1:])
    t_dy, t_dx = gradient_xy(teacher_output[:, -1:])
    edge_mse = F.mse_loss(s_dy, t_dy) + F.mse_loss(s_dx, t_dx)
    return flow_mse + 2.0 * cellprob_mse + 0.25 * edge_mse


def is_candidate_image_file(path: Path) -> bool:
    stem = path.stem.lower()
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and stem != "classes"
        and stem != "flows"
        and not stem.endswith("_classes")
        and not stem.endswith("_flows")
        and "_masks" not in stem
        and "_seg" not in stem
        and "_pred" not in stem
    )


def discover_files(
    data_dir: str | None,
    file_list: str | None,
    train_root_dirs: Sequence[str] = (),
) -> list[ImageFileRef]:
    files: list[ImageFileRef] = []
    if file_list:
        with open(file_list, "r", encoding="utf-8") as handle:
            for line in handle:
                raw_path = line.strip()
                if not raw_path:
                    continue
                path = Path(raw_path.split("::", 1)[0])
                if is_candidate_image_file(path):
                    files.append(ImageFileRef(path, "file-list"))
    if data_dir:
        root = Path(data_dir)
        files.extend(
            ImageFileRef(path, str(root.resolve()))
            for path in root.rglob("*")
            if is_candidate_image_file(path)
        )
    for root_dir in train_root_dirs:
        root = Path(root_dir)
        train_dirs = []
        if root.name.lower() == "train" and root.is_dir():
            train_dirs.append(root)
        if root.exists():
            train_dirs.extend(
                path
                for path in root.rglob("*")
                if path.is_dir() and path.name.lower() == "train"
            )
        for train_dir in train_dirs:
            group = str(train_dir.resolve())
            files.extend(
                ImageFileRef(path, group)
                for path in train_dir.rglob("*")
                if is_candidate_image_file(path)
            )
    unique = []
    seen = set()
    for file_ref in files:
        resolved = file_ref.path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(ImageFileRef(resolved, file_ref.source_group))
    return unique


def discover_frame_refs(
    data_dir: str | None,
    file_list: str | None,
    train_root_dirs: Sequence[str] = (),
) -> list[FrameRef]:
    files = discover_files(data_dir, file_list, train_root_dirs)
    image_service = ImageService()
    refs: list[FrameRef] = []
    for file_ref in files:
        path = file_ref.path
        if path.suffix.lower() == ".npy":
            refs.append(FrameRef(str(path), None, file_ref.source_group))
            continue
        try:
            frame_refs = image_service.build_frame_references(str(path))
        except Exception:
            frame_refs = []
        if frame_refs:
            for frame_ref in frame_refs:
                base, frame_id = image_service.split_image_reference(frame_ref)
                refs.append(FrameRef(base, frame_id, file_ref.source_group))
        else:
            refs.append(FrameRef(str(path), None, file_ref.source_group))
    return refs


def build_teacher(model_path: str, device: torch.device, tile_size: int) -> Transformer:
    if model_path == "cpsam":
        model_path = models.cache_CPSAM_model_path()
    teacher = Transformer(bsize=tile_size, rdrop=0.0).to(device)
    teacher.load_model(model_path, device=device, strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def build_student(config: DistillConfig) -> nn.Module:
    if config.student_backbone == "compact":
        return CPSAMEncoderStudent(
            width=config.student_width,
            depth=config.student_depth,
            out_channels=256,
        )
    if config.student_backbone == "mobilenet-v3-small":
        return MobileNetV3EncoderStudent(
            weights=config.mobilenet_weights,
            weights_path=config.mobilenet_weights_path,
            tap_layer=config.mobilenet_tap_layer,
            out_channels=256,
            target_stride=8,
        )
    raise ValueError(f"Unknown student backbone: {config.student_backbone}")


def build_student_head_from_teacher(teacher: Transformer, device: torch.device) -> CPSAMReadoutHead:
    return CPSAMReadoutHead(teacher.out, teacher.W2, teacher.ps).to(device)


def save_checkpoint(
    output_dir: Path,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DistillConfig,
    epoch: int,
    loss: float,
    student_head: nn.Module | None = None,
    best: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "cpsam_encoder_student_best.pt" if best else f"cpsam_encoder_student_epoch_{epoch:04d}.pt"
    path = output_dir / name
    payload = {
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "loss": loss,
        "feature_target": "cpsam Transformer.encoder.neck output",
        "output_target": "cpsam Transformer.forward output maps",
    }
    if student_head is not None:
        payload["student_head_state_dict"] = student_head.state_dict()
    torch.save(payload, path)
    return path


def load_student_checkpoint(
    path: str,
    student: nn.Module,
    student_head: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("student_state_dict", checkpoint)
    missing, unexpected = student.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"student checkpoint load: missing={missing}, unexpected={unexpected}")
    if student_head is not None and "student_head_state_dict" in checkpoint:
        head_missing, head_unexpected = student_head.load_state_dict(
            checkpoint["student_head_state_dict"],
            strict=False,
        )
        if head_missing or head_unexpected:
            print(
                "student head checkpoint load: "
                f"missing={head_missing}, unexpected={head_unexpected}"
            )
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def train(config: DistillConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    refs = discover_frame_refs(config.data_dir, config.file_list, config.train_root_dirs)
    print(f"discovered {len(refs)} image/frame references")
    group_counts: dict[str, int] = {}
    for ref in refs:
        group_counts[ref.source_group] = group_counts.get(ref.source_group, 0) + 1
    if len(group_counts) > 1:
        print(f"balanced sampling across {len(group_counts)} source groups")
        for group_name, count in sorted(group_counts.items()):
            print(f"  {group_name}: {count} image/frame references")
    samples_per_epoch = config.steps_per_epoch * config.batch_size
    dataset = ImageTileDataset(
        refs,
        tile_size=config.tile_size,
        normalize_percentiles=config.normalize_percentiles,
        samples_per_epoch=samples_per_epoch,
        sampling_mode=config.sampling_mode,
        tile_overlap=config.tile_overlap,
        augment_scale_range=config.augment_scale_range,
        augment_rotate=config.augment_rotate,
        augment_flip=config.augment_flip,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    teacher_device = torch.device(config.teacher_device)
    train_device = torch.device(config.train_device)
    teacher = build_teacher(config.cpsam_model, teacher_device, config.tile_size)
    student = build_student(config).to(train_device)
    student_head = build_student_head_from_teacher(teacher, train_device)
    if not config.train_cpsam_head:
        student_head.eval()
        for param in student_head.parameters():
            param.requires_grad_(False)

    trainable_params = list(student.parameters())
    if config.train_cpsam_head:
        trainable_params.extend(param for param in student_head.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    start_epoch = 0
    if config.init_student:
        print(f"initializing student from {config.init_student}")
        load_student_checkpoint(
            config.init_student,
            student,
            student_head=student_head,
            optimizer=None,
            device=train_device,
            strict=False,
        )
    if config.resume_student:
        print(f"resuming student/optimizer from {config.resume_student}")
        start_epoch = load_student_checkpoint(
            config.resume_student,
            student,
            student_head=student_head,
            optimizer=optimizer,
            device=train_device,
            strict=False,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and train_device.type == "cuda")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "distill_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    best_loss = math.inf
    for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
        student.train()
        if config.train_cpsam_head:
            student_head.train()
        else:
            student_head.eval()
        losses = []
        feature_losses = []
        output_losses = []
        progress = tqdm(loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            batch = batch.to(train_device, non_blocking=True)
            with torch.no_grad():
                teacher_batch = batch.to(teacher_device, non_blocking=True)
                teacher_features = teacher_encoder_forward(teacher, teacher_batch)
                teacher_output = None
                if config.output_loss_weight:
                    teacher_output = cpsam_readout(
                        teacher_features,
                        teacher.out,
                        teacher.W2,
                        teacher.ps,
                    )
                teacher_features = teacher_features.to(train_device, non_blocking=True)
                if teacher_output is not None:
                    teacher_output = teacher_output.to(train_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=train_device.type,
                enabled=config.amp and train_device.type == "cuda",
            ):
                student_features = student(batch)
                f_loss = feature_loss(student_features, teacher_features)
                loss = config.feature_loss_weight * f_loss
                o_loss = None
                if config.output_loss_weight:
                    student_output = student_head(student_features)
                    o_loss = output_distill_loss(student_output, teacher_output)
                    loss = loss + config.output_loss_weight * o_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            value = float(loss.detach().cpu())
            losses.append(value)
            feature_losses.append(float(f_loss.detach().cpu()))
            postfix = {"loss": f"{value:.5f}", "feat": f"{feature_losses[-1]:.5f}"}
            if o_loss is not None:
                output_losses.append(float(o_loss.detach().cpu()))
                postfix["out"] = f"{output_losses[-1]:.5f}"
            progress.set_postfix(**postfix)

        mean_loss = float(np.mean(losses)) if losses else math.inf
        mean_feature = float(np.mean(feature_losses)) if feature_losses else math.inf
        mean_output = float(np.mean(output_losses)) if output_losses else 0.0
        print(
            f"epoch {epoch}: loss={mean_loss:.6f} "
            f"feature={mean_feature:.6f} output={mean_output:.6f}"
        )
        save_checkpoint(output_dir, student, optimizer, config, epoch, mean_loss, student_head, best=False)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = save_checkpoint(output_dir, student, optimizer, config, epoch, mean_loss, student_head, best=True)
            print(f"saved best checkpoint: {best_path}")


def parse_args(argv: Sequence[str] | None = None) -> DistillConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Directory of training images.")
    parser.add_argument("--file-list", default=None, help="Text file containing one image path per line.")
    parser.add_argument(
        "--train-root-dirs",
        nargs="+",
        default=(),
        help=(
            "Root directories to search for descendant directories named 'train'. "
            "Compatible images are collected recursively from those train directories only."
        ),
    )
    parser.add_argument("--output-dir", default="distilled_cpsam_encoder", help="Checkpoint output directory.")
    parser.add_argument("--cpsam-model", default="cpsam", help="'cpsam' or path to a CPSAM checkpoint.")
    parser.add_argument("--tile-size", type=int, default=256, help="Square tile size. CPSAM default is 256.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--student-backbone",
        default="compact",
        choices=("compact", "mobilenet-v3-small"),
        help="Student encoder architecture. 'mobilenet-v3-small' uses a MobileNetV3 encoder plus CPSAM adapter.",
    )
    parser.add_argument("--student-width", type=int, default=48)
    parser.add_argument("--student-depth", type=int, default=1)
    parser.add_argument(
        "--mobilenet-weights",
        default="imagenet",
        choices=("imagenet", "none"),
        help="Initial weights for --student-backbone mobilenet-v3-small.",
    )
    parser.add_argument(
        "--mobilenet-weights-path",
        default=None,
        help="Optional local MobileNetV3 checkpoint loaded after --mobilenet-weights initialization.",
    )
    parser.add_argument(
        "--mobilenet-tap-layer",
        type=int,
        default=3,
        help="Inclusive torchvision MobileNetV3-small feature layer used before the CPSAM adapter. Default 3 gives stride-8 features.",
    )
    parser.add_argument("--init-student", default=None, help="Initialize student weights from a distilled checkpoint.")
    parser.add_argument("--resume-student", default=None, help="Resume student and optimizer from a distilled checkpoint.")
    parser.add_argument(
        "--sampling-mode",
        default="cellpose-random",
        choices=("cellpose-random", "overlap-grid"),
        help="Use Cellpose random_rotate_and_resize augmentation or deterministic overlap grid tiles.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.1, help="Overlap fraction for --sampling-mode overlap-grid.")
    parser.add_argument("--augment-scale-range", type=float, default=1.0, help="Scale range passed to Cellpose random_rotate_and_resize.")
    parser.add_argument("--no-augment-rotate", action="store_true", help="Disable random rotation augmentation.")
    parser.add_argument("--no-augment-flip", action="store_true", help="Disable random flip augmentation.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision for student training.")
    parser.add_argument("--normalize-percentiles", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument(
        "--feature-loss-weight",
        type=float,
        default=1.0,
        help="Weight for matching CPSAM encoder neck features.",
    )
    parser.add_argument(
        "--output-loss-weight",
        type=float,
        default=1.0,
        help="Weight for matching CPSAM final flow/cellprob output maps.",
    )
    parser.add_argument(
        "--train-cpsam-head",
        action="store_true",
        help="Train and save a copied CPSAM readout head with the student encoder.",
    )
    args = parser.parse_args(argv)
    if args.data_dir is None and args.file_list is None and not args.train_root_dirs:
        parser.error("Provide --data-dir, --file-list, and/or --train-root-dirs.")
    return DistillConfig(
        data_dir=args.data_dir,
        file_list=args.file_list,
        train_root_dirs=tuple(args.train_root_dirs),
        output_dir=args.output_dir,
        cpsam_model=args.cpsam_model,
        tile_size=args.tile_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        student_backbone=args.student_backbone,
        student_width=args.student_width,
        student_depth=args.student_depth,
        mobilenet_weights=args.mobilenet_weights,
        mobilenet_weights_path=args.mobilenet_weights_path,
        mobilenet_tap_layer=args.mobilenet_tap_layer,
        init_student=args.init_student,
        resume_student=args.resume_student,
        sampling_mode=args.sampling_mode,
        tile_overlap=args.tile_overlap,
        augment_scale_range=args.augment_scale_range,
        augment_rotate=not args.no_augment_rotate,
        augment_flip=not args.no_augment_flip,
        seed=args.seed,
        teacher_device=args.teacher_device,
        train_device=args.train_device,
        amp=bool(args.amp),
        normalize_percentiles=tuple(args.normalize_percentiles),
        feature_loss_weight=float(args.feature_loss_weight),
        output_loss_weight=float(args.output_loss_weight),
        train_cpsam_head=bool(args.train_cpsam_head),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    train(config)


if __name__ == "__main__":
    main()
