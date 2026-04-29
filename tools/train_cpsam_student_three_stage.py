"""Three-stage training for a larger distilled CPSAM student.

Stages:
1. Match CPSAM encoder-neck features.
2. Match CPSAM final flow/cellprob output maps and train the copied readout head.
3. Fine-tune student + readout head on available ``*_masks.tif``,
   ``*_masks.png`` or GUI ``*_seg.npy`` masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cellpose import dynamics
from cellpose.cpsam_student import MobileNetV3LargeEncoderStudent
from cellpose.train import _loss_fn_seg
from cellpose.transforms import random_rotate_and_resize
from guv_app.services.image_service import ImageService
from guv_app.services.training_dataset_service import _load_seg_npy_compat
from tools.train_cpsam_encoder_distill import (
    CPSAMReadoutHead,
    ImageTileDataset,
    build_teacher,
    cpsam_readout,
    discover_frame_refs,
    feature_loss,
    load_student_checkpoint,
    normalize_tile_cellpose,
    output_distill_loss,
    save_checkpoint,
    teacher_encoder_forward,
    to_channels_last_3,
)


@dataclass
class ThreeStageConfig:
    data_dir: str | None
    file_list: str | None
    train_root_dirs: tuple[str, ...]
    output_dir: str
    cpsam_model: str
    tile_size: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    student_backbone: str
    student_width: int
    student_depth: int
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
    stage1_epochs: int
    stage2_epochs: int
    stage3_epochs: int
    stage1_feature_weight: float
    stage2_feature_weight: float
    stage2_output_weight: float
    stage2_flow_direction_weight: float
    stage3_seg_weight: float
    stage3_flow_direction_weight: float
    stage3_flow_device: str
    stage3_flow_cache_dir: str | None
    mobilenet_weights: str
    mobilenet_weights_path: str | None
    mobilenet_tap_layer: int
    mobilenet_adapter_width: int
    profile_batches: int


class SupervisedFlowDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[tuple],
        tile_size: int,
        normalize_percentiles: tuple[float, float],
        samples_per_epoch: int,
        augment_scale_range: float,
        augment_rotate: bool,
        augment_flip: bool,
        flow_cache_dir: str | Path | None = None,
    ):
        if not pairs:
            raise ValueError("No supervised image/mask pairs found.")
        self.pairs = list(pairs)
        self.tile_size = int(tile_size)
        self.normalize_percentiles = normalize_percentiles
        self.samples_per_epoch = int(samples_per_epoch)
        self.augment_scale_range = float(augment_scale_range)
        self.augment_rotate = bool(augment_rotate)
        self.augment_flip = bool(augment_flip)
        self.flow_cache_dir = Path(flow_cache_dir) if flow_cache_dir else None
        self.image_service = None
        self.groups = self._groups_by_source()
        self.group_names = sorted(self.groups)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        group_name = self.group_names[index % len(self.group_names)]
        pair_index = random.choice(self.groups[group_name])
        ref, seg_path = self.pairs[pair_index]

        image = self._load_ref_image(ref)
        image = to_channels_last_3(image)
        image = normalize_tile_cellpose(image, self.normalize_percentiles)
        flows = self._load_label_flows(seg_path)

        chw = np.moveaxis(image, -1, 0).astype(np.float32, copy=False)
        imgi, lbl, _ = random_rotate_and_resize(
            [chw],
            Y=[flows],
            scale_range=self.augment_scale_range,
            xy=(self.tile_size, self.tile_size),
            do_flip=self.augment_flip,
            rotate=self.augment_rotate,
        )
        return torch.from_numpy(imgi[0].astype(np.float32, copy=False)), torch.from_numpy(lbl[0].astype(np.float32, copy=False))

    def _image_service(self) -> ImageService:
        if self.image_service is None:
            self.image_service = ImageService()
        return self.image_service

    def _load_ref_image(self, ref):
        if ref.frame_id is not None:
            arr = self._image_service().load_frame(ref.filename, ref.frame_id)
        else:
            arr = self._image_service().load_image(ref.filename)
        if arr is None:
            raise ValueError(f"Could not load supervised image/frame: {ref}")
        return np.asarray(arr)

    def _groups_by_source(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for index, (ref, _seg_path) in enumerate(self.pairs):
            groups.setdefault(ref.source_group, []).append(index)
        return groups

    def _load_label_flows(self, label_path: str | Path) -> np.ndarray:
        cache_path = flow_cache_path(label_path, self.flow_cache_dir)
        if cache_path is not None and cache_path.exists():
            try:
                return np.load(cache_path).astype(np.float32, copy=False)
            except Exception:
                pass

        masks = load_label_mask(label_path)
        flows = dynamics.labels_to_flows([masks], device=torch.device("cpu"))[0].astype(np.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npy")
            try:
                np.save(tmp_path, flows)
                os.replace(tmp_path, cache_path)
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
        return flows


def flow_cache_path(label_path: str | Path, flow_cache_dir: str | Path | None) -> Path | None:
    if flow_cache_dir is None:
        return None
    label_path = Path(label_path)
    try:
        stat = label_path.stat()
        key_text = f"{label_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        key_text = str(label_path)
    key = hashlib.sha1(key_text.encode("utf-8", errors="replace")).hexdigest()
    return Path(flow_cache_dir) / f"{key}.npy"


def compute_label_flows(label_path: str | Path, flow_device: torch.device) -> np.ndarray:
    masks = load_label_mask(label_path)
    return dynamics.labels_to_flows([masks], device=flow_device)[0].astype(np.float32)


def precompute_supervised_flow_cache(pairs: Sequence[tuple], flow_cache_dir: str | Path, flow_device: torch.device) -> None:
    flow_cache_dir = Path(flow_cache_dir)
    flow_cache_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for _ref, label_path in pairs:
        cache_path = flow_cache_path(label_path, flow_cache_dir)
        if cache_path is not None and not cache_path.exists():
            missing.append(label_path)
    if not missing:
        print(f"supervised flow cache already populated for {len(pairs)} labels")
        return

    print(f"precomputing {len(missing)} supervised flow targets on {flow_device}")
    for label_path in tqdm(missing, desc="precompute supervised flows", leave=False):
        cache_path = flow_cache_path(label_path, flow_cache_dir)
        if cache_path is None or cache_path.exists():
            continue
        flows = compute_label_flows(label_path, flow_device)
        tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npy")
        try:
            np.save(tmp_path, flows)
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise


def load_label_mask(label_path: str | Path) -> np.ndarray:
    label_path = Path(label_path)
    if label_path.suffix.lower() == ".npy":
        dat = _load_seg_npy_compat(str(label_path))
        masks = np.squeeze(dat.get("masks"))
    elif label_path.suffix.lower() in (".tif", ".tiff"):
        masks = np.squeeze(tifffile.imread(label_path))
    elif label_path.suffix.lower() == ".png":
        with Image.open(label_path) as img:
            masks = np.squeeze(np.asarray(img))
    else:
        raise ValueError(f"Unsupported label file type: {label_path}")
    if masks.ndim != 2:
        raise ValueError(f"Expected 2D masks in {label_path}, got {masks.shape}")
    return masks.astype(np.int32, copy=False)


def _image_label_stems(filename: str) -> list[str]:
    base = os.path.splitext(filename)[0]
    stems = [base]
    for suffix in ("_img", "_image"):
        if base.lower().endswith(suffix):
            stems.append(base[: -len(suffix)])
    unique = []
    seen = set()
    for stem in stems:
        key = os.path.normcase(stem)
        if key not in seen:
            seen.add(key)
            unique.append(stem)
    return unique


def find_label_path(ref) -> Path | None:
    base = os.path.splitext(ref.filename)[0]
    if ref.frame_id:
        candidates = [
            Path(f"{base}__{ref.frame_id}_masks.tif"),
            Path(f"{base}__{ref.frame_id}_masks.tiff"),
            Path(f"{base}__{ref.frame_id}_masks.png"),
            Path(f"{base}__{ref.frame_id}_seg.npy"),
        ]
    else:
        candidates = []
        for stem in _image_label_stems(ref.filename):
            candidates.extend(
                [
                    Path(f"{stem}_masks.tif"),
                    Path(f"{stem}_masks.tiff"),
                    Path(f"{stem}_masks.png"),
                    Path(f"{stem}_seg.npy"),
                ]
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_supervised_pairs(refs: Sequence) -> list[tuple]:
    pairs = []
    for ref in refs:
        label_path = find_label_path(ref)
        if label_path is not None:
            pairs.append((ref, label_path))
    return pairs


def build_student(config: ThreeStageConfig) -> torch.nn.Module:
    return MobileNetV3LargeEncoderStudent(
        weights=config.mobilenet_weights,
        weights_path=config.mobilenet_weights_path,
        tap_layer=config.mobilenet_tap_layer,
        out_channels=256,
        target_stride=8,
        adapter_width=config.mobilenet_adapter_width,
    )


def build_student_head_from_teacher(teacher, device: torch.device) -> CPSAMReadoutHead:
    return CPSAMReadoutHead(teacher.out, teacher.W2, teacher.ps).to(device)


def weighted_flow_direction_loss(student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
    s_flow = student_output[:, -3:-1]
    t_flow = teacher_output[:, -3:-1]
    cos = F.cosine_similarity(s_flow, t_flow, dim=1, eps=1e-6)
    weight = torch.sigmoid(teacher_output[:, -1]).detach()
    denom = weight.sum().clamp_min(1.0)
    return ((1.0 - cos) * weight).sum() / denom


def supervised_flow_direction_loss(lbl: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = 5.0 * lbl[:, -2:]
    cos = F.cosine_similarity(y[:, -3:-1], target, dim=1, eps=1e-6)
    weight = (lbl[:, -3] > 0.5).float() if lbl.shape[1] >= 3 else (lbl[:, 0] > 0.5).float()
    denom = weight.sum().clamp_min(1.0)
    return ((1.0 - cos) * weight).sum() / denom


def make_distill_loader(config: ThreeStageConfig, refs) -> DataLoader:
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
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def make_supervised_loader(config: ThreeStageConfig, pairs) -> DataLoader:
    samples_per_epoch = config.steps_per_epoch * config.batch_size
    configured_cache_dir = getattr(config, "stage3_flow_cache_dir", None)
    flow_cache_dir = Path(configured_cache_dir) if configured_cache_dir else Path(config.output_dir) / "supervised_flow_cache"
    flow_device = torch.device(getattr(config, "stage3_flow_device", "cpu"))
    if flow_device.type != "cpu":
        precompute_supervised_flow_cache(pairs, flow_cache_dir, flow_device)
    dataset = SupervisedFlowDataset(
        pairs,
        tile_size=config.tile_size,
        normalize_percentiles=config.normalize_percentiles,
        samples_per_epoch=samples_per_epoch,
        augment_scale_range=config.augment_scale_range,
        augment_rotate=config.augment_rotate,
        augment_flip=config.augment_flip,
        flow_cache_dir=flow_cache_dir,
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def make_optimizer(config: ThreeStageConfig, student, student_head, train_head: bool):
    params = list(student.parameters())
    if train_head:
        for param in student_head.parameters():
            param.requires_grad_(True)
        params.extend(param for param in student_head.parameters() if param.requires_grad)
    else:
        for param in student_head.parameters():
            param.requires_grad_(False)
    return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def save_stage_checkpoint(output_dir, stage_name, student, student_head, optimizer, config, epoch, loss, best=False):
    stage_dir = Path(output_dir) / stage_name
    path = save_checkpoint(stage_dir, student, optimizer, config, epoch, loss, student_head, best=best)
    if best:
        final_path = Path(output_dir) / "cpsam_encoder_student_best.pt"
        final_path.write_bytes(path.read_bytes())
    return path


def run_feature_stage(config, loader, teacher, student, student_head, train_device, teacher_device, start_epoch=0):
    optimizer = make_optimizer(config, student, student_head, train_head=False)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and train_device.type == "cuda")
    best_loss = math.inf
    for epoch in range(start_epoch + 1, start_epoch + config.stage1_epochs + 1):
        student.train()
        student_head.eval()
        losses = []
        progress = tqdm(loader, desc=f"stage1 feature {epoch}/{config.stage1_epochs}", leave=False)
        iterator = iter(progress)
        batch_index = 0
        while True:
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            load_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            load_seconds = time.perf_counter() - load_start
            batch_index += 1

            transfer_start = time.perf_counter()
            batch = batch.to(train_device, non_blocking=True)
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_start

            with torch.no_grad():
                teacher_start = time.perf_counter()
                teacher_batch = batch.to(teacher_device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=teacher_device.type,
                    enabled=config.amp and teacher_device.type == "cuda",
                ):
                    teacher_features = teacher_encoder_forward(teacher, teacher_batch)
                teacher_features = teacher_features.to(train_device, non_blocking=True)
                if train_device.type == "cuda" or teacher_device.type == "cuda":
                    torch.cuda.synchronize()
                teacher_seconds = time.perf_counter() - teacher_start
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=train_device.type, enabled=config.amp and train_device.type == "cuda"):
                student_start = time.perf_counter()
                student_features = student(batch)
                loss = config.stage1_feature_weight * feature_loss(student_features, teacher_features)
                if train_device.type == "cuda":
                    torch.cuda.synchronize()
                student_seconds = time.perf_counter() - student_start
            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - backward_start
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.5f}")
            if batch_index <= getattr(config, "profile_batches", 0):
                print(
                    "PROFILE stage1 batch "
                    f"{batch_index}: load={load_seconds:.3f}s "
                    f"transfer={transfer_seconds:.3f}s "
                    f"teacher={teacher_seconds:.3f}s "
                    f"student_loss={student_seconds:.3f}s "
                    f"backward_step={backward_seconds:.3f}s "
                    f"batch_shape={tuple(batch.shape)} "
                    f"batch_device={batch.device} "
                    f"teacher_device={next(teacher.parameters()).device} "
                    f"student_device={next(student.parameters()).device}"
                )
        mean_loss = float(np.mean(losses)) if losses else math.inf
        print(f"stage1 epoch {epoch}: loss={mean_loss:.6f}")
        save_stage_checkpoint(config.output_dir, "stage1_feature", student, student_head, optimizer, config, epoch, mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_stage_checkpoint(config.output_dir, "stage1_feature", student, student_head, optimizer, config, epoch, mean_loss, best=True)


def run_output_stage(config, loader, teacher, student, student_head, train_device, teacher_device):
    optimizer = make_optimizer(config, student, student_head, train_head=True)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and train_device.type == "cuda")
    best_loss = math.inf
    for epoch in range(1, config.stage2_epochs + 1):
        student.train()
        student_head.train()
        losses = []
        progress = tqdm(loader, desc=f"stage2 output {epoch}/{config.stage2_epochs}", leave=False)
        iterator = iter(progress)
        batch_index = 0
        while True:
            import time
            load_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            load_seconds = time.perf_counter() - load_start
            batch_index += 1
            transfer_start = time.perf_counter()
            batch = batch.to(train_device, non_blocking=True)
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_start
            with torch.no_grad():
                teacher_start = time.perf_counter()
                teacher_batch = batch.to(teacher_device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=teacher_device.type,
                    enabled=config.amp and teacher_device.type == "cuda",
                ):
                    teacher_features = teacher_encoder_forward(teacher, teacher_batch)
                    teacher_output = cpsam_readout(teacher_features, teacher.out, teacher.W2, teacher.ps)
                teacher_features = teacher_features.to(train_device, non_blocking=True)
                teacher_output = teacher_output.to(train_device, non_blocking=True)
                if train_device.type == "cuda" or teacher_device.type == "cuda":
                    torch.cuda.synchronize()
                teacher_seconds = time.perf_counter() - teacher_start
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=train_device.type, enabled=config.amp and train_device.type == "cuda"):
                student_start = time.perf_counter()
                student_features = student(batch)
                student_output = student_head(student_features)
                f_loss = feature_loss(student_features, teacher_features)
                o_loss = output_distill_loss(student_output, teacher_output)
                d_loss = weighted_flow_direction_loss(student_output, teacher_output)
                loss = (
                    config.stage2_feature_weight * f_loss
                    + config.stage2_output_weight * o_loss
                    + config.stage2_flow_direction_weight * d_loss
                )
                if train_device.type == "cuda":
                    torch.cuda.synchronize()
                student_seconds = time.perf_counter() - student_start
            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - backward_start
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.5f}", out=f"{float(o_loss.detach().cpu()):.5f}", dir=f"{float(d_loss.detach().cpu()):.5f}")
            if batch_index <= getattr(config, "profile_batches", 0):
                print(
                    "PROFILE stage2 batch "
                    f"{batch_index}: load={load_seconds:.3f}s "
                    f"transfer={transfer_seconds:.3f}s "
                    f"teacher={teacher_seconds:.3f}s "
                    f"student_loss={student_seconds:.3f}s "
                    f"backward_step={backward_seconds:.3f}s "
                    f"batch_shape={tuple(batch.shape)} "
                    f"batch_device={batch.device}"
                )
        mean_loss = float(np.mean(losses)) if losses else math.inf
        print(f"stage2 epoch {epoch}: loss={mean_loss:.6f}")
        save_stage_checkpoint(config.output_dir, "stage2_output", student, student_head, optimizer, config, epoch, mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_stage_checkpoint(config.output_dir, "stage2_output", student, student_head, optimizer, config, epoch, mean_loss, best=True)


def run_supervised_stage(config, loader, student, student_head, train_device):
    optimizer = make_optimizer(config, student, student_head, train_head=True)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and train_device.type == "cuda")
    best_loss = math.inf
    for epoch in range(1, config.stage3_epochs + 1):
        student.train()
        student_head.train()
        losses = []
        progress = tqdm(loader, desc=f"stage3 supervised {epoch}/{config.stage3_epochs}", leave=False)
        iterator = iter(progress)
        batch_index = 0
        while True:
            load_start = time.perf_counter()
            try:
                batch, lbl = next(iterator)
            except StopIteration:
                break
            load_seconds = time.perf_counter() - load_start
            batch_index += 1
            transfer_start = time.perf_counter()
            batch = batch.to(train_device, non_blocking=True)
            lbl = lbl.to(train_device, non_blocking=True)
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_start
            optimizer.zero_grad(set_to_none=True)
            forward_start = time.perf_counter()
            with torch.amp.autocast(device_type=train_device.type, enabled=config.amp and train_device.type == "cuda"):
                y = student_head(student(batch))
                seg_loss = _loss_fn_seg(lbl, y, train_device)
                dir_loss = supervised_flow_direction_loss(lbl, y)
                loss = config.stage3_seg_weight * seg_loss + config.stage3_flow_direction_weight * dir_loss
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if train_device.type == "cuda":
                torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - backward_start
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.5f}", seg=f"{float(seg_loss.detach().cpu()):.5f}", dir=f"{float(dir_loss.detach().cpu()):.5f}")
            if batch_index <= getattr(config, "profile_batches", 0):
                print(
                    "PROFILE stage3 batch "
                    f"{batch_index}: load={load_seconds:.3f}s "
                    f"transfer={transfer_seconds:.3f}s "
                    f"forward_loss={forward_seconds:.3f}s "
                    f"backward_step={backward_seconds:.3f}s "
                    f"batch_shape={tuple(batch.shape)} "
                    f"label_shape={tuple(lbl.shape)} "
                    f"batch_device={batch.device}"
                )
        mean_loss = float(np.mean(losses)) if losses else math.inf
        print(f"stage3 epoch {epoch}: loss={mean_loss:.6f}")
        save_stage_checkpoint(config.output_dir, "stage3_supervised", student, student_head, optimizer, config, epoch, mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_stage_checkpoint(config.output_dir, "stage3_supervised", student, student_head, optimizer, config, epoch, mean_loss, best=True)


def train(config: ThreeStageConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    refs = discover_frame_refs(config.data_dir, config.file_list, config.train_root_dirs)
    print(f"discovered {len(refs)} image/frame references")
    if not refs:
        raise ValueError("No training images discovered.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "three_stage_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    teacher_device = torch.device(config.teacher_device)
    train_device = torch.device(config.train_device)
    teacher = build_teacher(config.cpsam_model, teacher_device, config.tile_size)
    student = build_student(config).to(train_device)
    student_head = build_student_head_from_teacher(teacher, train_device)

    if config.init_student:
        print(f"initializing student from {config.init_student}")
        load_student_checkpoint(config.init_student, student, student_head=student_head, device=train_device, strict=False)
    if config.resume_student:
        print(f"resuming student/head from {config.resume_student}")
        load_student_checkpoint(config.resume_student, student, student_head=student_head, device=train_device, strict=False)

    distill_loader = make_distill_loader(config, refs)
    if config.stage1_epochs > 0:
        run_feature_stage(config, distill_loader, teacher, student, student_head, train_device, teacher_device)
    if config.stage2_epochs > 0:
        run_output_stage(config, distill_loader, teacher, student, student_head, train_device, teacher_device)

    pairs = discover_supervised_pairs(refs)
    print(f"discovered {len(pairs)} supervised image/mask pairs")
    if config.stage3_epochs > 0:
        if not pairs:
            print("stage3 requested, but no matching *_masks.tif/_masks.tiff/_masks.png or *_seg.npy labels were found; skipping supervised fine-tuning")
        else:
            supervised_loader = make_supervised_loader(config, pairs)
            run_supervised_stage(config, supervised_loader, student, student_head, train_device)


def parse_args(argv: Sequence[str] | None = None) -> ThreeStageConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--train-root-dirs", nargs="+", default=())
    parser.add_argument("--output-dir", default="distilled_cpsam_encoder_three_stage")
    parser.add_argument("--cpsam-model", default="cpsam")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--init-student", default=None)
    parser.add_argument("--resume-student", default=None)
    parser.add_argument("--sampling-mode", default="cellpose-random", choices=("cellpose-random", "overlap-grid"))
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--augment-scale-range", type=float, default=1.0)
    parser.add_argument("--no-augment-rotate", action="store_true")
    parser.add_argument("--no-augment-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--normalize-percentiles", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--stage1-epochs", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--stage3-epochs", type=int, default=80)
    parser.add_argument("--stage1-feature-weight", type=float, default=1.0)
    parser.add_argument("--stage2-feature-weight", type=float, default=0.05)
    parser.add_argument("--stage2-output-weight", type=float, default=5.0)
    parser.add_argument("--stage2-flow-direction-weight", type=float, default=1.0)
    parser.add_argument("--stage3-seg-weight", type=float, default=1.0)
    parser.add_argument("--stage3-flow-direction-weight", type=float, default=0.5)
    parser.add_argument(
        "--stage3-flow-device",
        default="cpu",
        help="Device used to precompute supervised label flow targets before stage 3, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--stage3-flow-cache-dir",
        default=None,
        help="Directory for cached supervised flow .npy files. Defaults to <output-dir>/supervised_flow_cache.",
    )
    parser.add_argument("--mobilenet-weights", default="imagenet", choices=("imagenet", "none"))
    parser.add_argument("--mobilenet-weights-path", default=None)
    parser.add_argument("--mobilenet-tap-layer", type=int, default=6)
    parser.add_argument("--mobilenet-adapter-width", type=int, default=384)
    parser.add_argument("--profile-batches", type=int, default=0, help="Print timing breakdown for the first N batches of each stage.")
    args = parser.parse_args(argv)
    if args.data_dir is None and args.file_list is None and not args.train_root_dirs:
        parser.error("Provide --data-dir, --file-list, and/or --train-root-dirs.")
    return ThreeStageConfig(
        data_dir=args.data_dir,
        file_list=args.file_list,
        train_root_dirs=tuple(args.train_root_dirs),
        output_dir=args.output_dir,
        cpsam_model=args.cpsam_model,
        tile_size=args.tile_size,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        student_backbone="mobilenet-v3-large",
        student_width=0,
        student_depth=0,
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
        feature_loss_weight=args.stage2_feature_weight,
        output_loss_weight=args.stage2_output_weight,
        train_cpsam_head=True,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage1_feature_weight=args.stage1_feature_weight,
        stage2_feature_weight=args.stage2_feature_weight,
        stage2_output_weight=args.stage2_output_weight,
        stage2_flow_direction_weight=args.stage2_flow_direction_weight,
        stage3_seg_weight=args.stage3_seg_weight,
        stage3_flow_direction_weight=args.stage3_flow_direction_weight,
        stage3_flow_device=args.stage3_flow_device,
        stage3_flow_cache_dir=args.stage3_flow_cache_dir,
        mobilenet_weights=args.mobilenet_weights,
        mobilenet_weights_path=args.mobilenet_weights_path,
        mobilenet_tap_layer=args.mobilenet_tap_layer,
        mobilenet_adapter_width=args.mobilenet_adapter_width,
        profile_batches=args.profile_batches,
    )


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
