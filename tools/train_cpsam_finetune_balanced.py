"""Fine-tune CPSAM on mixed labeled datasets with persistent splits.

This script discovers labeled image/frame pairs under arbitrary directory trees,
creates permanent train/validation/test manifests, balances training by source
group through weighted inclusion, and fine-tunes CPSAM with only the last encoder
blocks plus output heads trainable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
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
from PIL import Image

from cellpose import dynamics
from cellpose import models as cp_models
from cellpose import train as cellpose_train
from cellpose.models import CellposeModel
from cellpose.semantic_class_weights import (
    compute_class_weights_from_class_maps,
    infer_semantic_nclasses_from_net,
)
from cellpose.semantic_label_utils import build_classes_map_from_masks, sanitize_class_map
from cellpose.training_mode_utils import configure_trainable_params
from guv_app.services.image_service import ImageService
from guv_app.services.training_dataset_service import _load_seg_npy_compat
from tools.train_cpsam_encoder_distill import (
    FrameRef,
    is_candidate_image_file,
)
from tools.train_cpsam_student_three_stage import _image_label_stems


@dataclass(frozen=True)
class LabeledRef:
    image: str
    frame_id: str | None
    label: str
    source_group: str

    @property
    def image_ref(self) -> str:
        return f"{self.image}::{self.frame_id}" if self.frame_id else self.image


@dataclass
class SplitManifest:
    seed: int
    val_ratio: float
    test_ratio: float
    records: list[dict]


LABEL_SUFFIXES = (
    "_masks.tif",
    "_masks.tiff",
    "_masks.png",
    "_mask.tif",
    "_mask.tiff",
    "_mask.png",
    "_labels.tif",
    "_labels.tiff",
    "_labels.png",
    "_label.tif",
    "_label.tiff",
    "_label.png",
    "_cp_masks.tif",
    "_cp_masks.tiff",
    "_cp_masks.png",
    "_seg.npy",
)


NPZ_IMAGE_KEYS = ("x", "images", "image", "imgs")
NPZ_MASK_KEYS = ("y", "masks", "mask", "labels", "label")


def to_channels_last_preserve(image: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(image))
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = np.moveaxis(arr, 0, -1)
        elif arr.shape[-1] > 8:
            # Treat first plane as stack/time when the last dimension cannot be channels.
            arr = arr[0, ..., None] if arr.shape[0] > 8 else arr[..., :1]
    else:
        while arr.ndim > 3:
            arr = arr[0]
        return to_channels_last_preserve(arr)
    return arr.astype(np.float32, copy=False)


def make_three_channel_view(image: np.ndarray, channels: tuple[int, ...] | None) -> np.ndarray:
    arr = to_channels_last_preserve(image)
    n_channels = arr.shape[-1]
    if channels is None:
        selected = arr[..., : min(3, n_channels)]
    else:
        selected = arr[..., list(channels)]
    if selected.ndim == 2:
        selected = selected[..., None]
    if selected.shape[-1] == 1:
        selected = np.repeat(selected, 3, axis=-1)
    elif selected.shape[-1] == 2:
        selected = np.concatenate([selected, selected[..., :1]], axis=-1)
    elif selected.shape[-1] > 3:
        selected = selected[..., :3]
    return selected.astype(np.float32, copy=False)


def channel_view_specs(
    image: np.ndarray,
    mode: str,
    max_all_channel_combos: int,
    rng: random.Random,
) -> list[tuple[str, tuple[int, ...] | None]]:
    n_channels = to_channels_last_preserve(image).shape[-1]
    if mode == "none" or n_channels <= 1:
        return [("all", None)]
    if mode != "single-and-all":
        raise ValueError("--channel-sampling-mode must be 'single-and-all' or 'none'")

    all_name = "all" if n_channels <= 3 else "all_first3"
    specs: list[tuple[str, tuple[int, ...] | None]] = [(all_name, None)]
    specs.extend((f"ch{idx}", (idx,)) for idx in range(n_channels))
    if n_channels > 3 and max_all_channel_combos > 0:
        combos = set()
        attempts = max_all_channel_combos * 8
        while len(combos) < max_all_channel_combos and attempts > 0:
            combo = tuple(sorted(rng.sample(range(n_channels), 3)))
            if combo != tuple(range(3)):
                combos.add(combo)
            attempts -= 1
        specs.extend((f"combo{','.join(map(str, combo))}", combo) for combo in sorted(combos))
    return specs


def _pick_npz_key(keys: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower_to_key = {key.lower(): key for key in keys}
    for candidate in candidates:
        if candidate in lower_to_key:
            return lower_to_key[candidate]
    for key in keys:
        key_lower = key.lower()
        if any(candidate in key_lower for candidate in candidates):
            return key
    return None


def _npz_member_shape(path: Path, key: str) -> tuple[int, ...] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(f"{key}.npy") as member:
                version = np.lib.format.read_magic(member)
                shape, _fortran_order, _dtype = np.lib.format._read_array_header(member, version)
                return tuple(int(dim) for dim in shape)
    except Exception:
        return None


def inspect_npz_dataset(path: Path) -> tuple[str, str, int] | None:
    try:
        with np.load(path, allow_pickle=True, mmap_mode="r") as data:
            keys = list(data.keys())
            image_key = _pick_npz_key(keys, NPZ_IMAGE_KEYS)
            mask_key = _pick_npz_key(keys, NPZ_MASK_KEYS)
            if image_key is None or mask_key is None:
                print(f"warning: could not infer image/mask keys in {path}; keys={keys}")
                return None
            image_shape = _npz_member_shape(path, image_key)
            mask_shape = _npz_member_shape(path, mask_key)
            if image_shape is not None and mask_shape is not None:
                n_images = int(image_shape[0])
                n_masks = int(mask_shape[0])
            else:
                n_images = int(len(data[image_key]))
                n_masks = int(len(data[mask_key]))
            if n_masks != n_images:
                print(
                    f"warning: image/mask length mismatch in {path}: "
                    f"{image_key}={n_images}, {mask_key}={n_masks}"
                )
                return None
            return image_key, mask_key, n_images
    except Exception as exc:
        print(f"warning: could not inspect npz dataset {path}: {exc}")
        return None


def _parse_npz_frame_id(frame_id: str | None) -> tuple[int, str | None, str | None]:
    if not frame_id or not frame_id.startswith("NPZ"):
        raise ValueError(f"Expected NPZ frame id, got {frame_id!r}")
    # Current format is NPZ{index}; keep optional key slots for forward compatibility.
    payload = frame_id[3:]
    parts = payload.split("|")
    return int(parts[0]), parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None


def _npz_cache_path(path: Path, key: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(f"{path.resolve()}::{key}".encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{path.stem}_{key}_{digest}.npy"


def _cached_npz_member(path: Path, key: str, cache_dir: str | Path | None) -> Path | None:
    if not cache_dir:
        return None
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = _npz_cache_path(path, key, cache_root)
    if cached.exists():
        return cached
    tmp = cached.with_suffix(".tmp")
    member_name = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member_name not in archive.namelist():
            return None
        with archive.open(member_name) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp.replace(cached)
    return cached


def load_npz_item(path: str | Path, frame_id: str | None, kind: str, npz_cache_dir: str | Path | None = None) -> np.ndarray:
    index, image_key_hint, mask_key_hint = _parse_npz_frame_id(frame_id)
    path = Path(path)
    with np.load(path, allow_pickle=True, mmap_mode="r") as data:
        keys = list(data.keys())
        if kind == "image":
            key = image_key_hint or _pick_npz_key(keys, NPZ_IMAGE_KEYS)
        else:
            key = mask_key_hint or _pick_npz_key(keys, NPZ_MASK_KEYS)
        if key is None:
            raise ValueError(f"Could not infer {kind} key in npz file {path}; keys={keys}")
    cached = _cached_npz_member(path, key, npz_cache_dir)
    if cached is not None:
        return np.asarray(np.load(cached, mmap_mode="r")[index])
    raise ValueError(
        f"Refusing to load {path}:{key}[{index}] directly because .npz archives are not "
        "random-access. Provide --npz-cache-dir on fast scratch/ephemeral storage."
    )


def find_label_path(ref: FrameRef) -> Path | None:
    base = os.path.splitext(ref.filename)[0]
    candidates: list[Path] = []
    if ref.frame_id:
        for suffix in LABEL_SUFFIXES:
            candidates.append(Path(f"{base}__{ref.frame_id}{suffix}"))
    else:
        for stem in _image_label_stems(ref.filename):
            for suffix in LABEL_SUFFIXES:
                candidates.append(Path(f"{stem}{suffix}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def source_group_for_path(path: Path, root: Path) -> str:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return str(path.parent.resolve())
    lower_parts = [part.lower() for part in rel_parts]
    if "train" in lower_parts:
        train_index = lower_parts.index("train")
        if train_index > 0:
            return str((root / Path(*rel_parts[:train_index])).resolve())
        return str(root.resolve())
    if len(rel_parts) > 1:
        return str((root / rel_parts[0]).resolve())
    return str(root.resolve())


def discover_frame_refs_in_tree(root_dir: str) -> list[FrameRef]:
    root = Path(root_dir)
    if not root.exists():
        print(f"warning: root does not exist, skipping: {root}")
        return []
    image_service = ImageService()
    refs: list[FrameRef] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".npz":
            inspected = inspect_npz_dataset(path)
            if inspected is None:
                continue
            image_key, mask_key, n_images = inspected
            source_group = source_group_for_path(path, root)
            for index in range(n_images):
                refs.append(FrameRef(str(path.resolve()), f"NPZ{index}|{image_key}|{mask_key}", source_group))
            continue
        if not is_candidate_image_file(path):
            continue
        source_group = source_group_for_path(path, root)
        if path.suffix.lower() == ".npy":
            refs.append(FrameRef(str(path.resolve()), None, source_group))
            continue
        try:
            frame_refs = image_service.build_frame_references(str(path))
        except Exception:
            frame_refs = []
        if frame_refs:
            for frame_ref in frame_refs:
                base, frame_id = image_service.split_image_reference(frame_ref)
                refs.append(FrameRef(str(Path(base).resolve()), frame_id, source_group))
        else:
            refs.append(FrameRef(str(path.resolve()), None, source_group))
    return refs


def discover_labeled_refs(root_dirs: Sequence[str], data_dirs: Sequence[str] = ()) -> list[LabeledRef]:
    refs = []
    for root_dir in root_dirs:
        refs.extend(discover_frame_refs_in_tree(root_dir))
    if data_dirs:
        for data_dir in data_dirs:
            refs.extend(discover_frame_refs_in_tree(data_dir))
    labeled: list[LabeledRef] = []
    seen = set()
    for ref in refs:
        label_path = Path(ref.filename) if Path(ref.filename).suffix.lower() == ".npz" else find_label_path(ref)
        if label_path is None:
            continue
        key = (str(Path(ref.filename).resolve()), ref.frame_id, str(label_path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        labeled.append(
            LabeledRef(
                image=str(Path(ref.filename).resolve()),
                frame_id=ref.frame_id,
                label=str(label_path.resolve()),
                source_group=ref.source_group,
            )
        )
    return labeled


def split_labeled_refs(
    refs: Sequence[LabeledRef],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> list[dict]:
    rng = random.Random(seed)
    by_group: dict[str, list[LabeledRef]] = {}
    for ref in refs:
        by_group.setdefault(ref.source_group, []).append(ref)

    records: list[dict] = []
    for group, group_refs in sorted(by_group.items()):
        shuffled = list(group_refs)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_test = min(max(1, n_test), n - 2)
            n_val = min(max(1, n_val), n - n_test - 1)
        else:
            n_test = 0
            n_val = 0
        split_for_index = []
        split_for_index.extend(["test"] * n_test)
        split_for_index.extend(["val"] * n_val)
        split_for_index.extend(["train"] * (n - n_test - n_val))
        for ref, split in zip(shuffled, split_for_index):
            records.append(
                {
                    "split": split,
                    "image": ref.image,
                    "frame_id": ref.frame_id or "",
                    "label": ref.label,
                    "source_group": ref.source_group,
                }
            )
    records.sort(key=lambda row: (row["split"], row["source_group"], row["image"], row["frame_id"], row["label"]))
    return records


def write_split_manifest(path: Path, manifest: SplitManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "frame_id", "label", "source_group"])
        writer.writeheader()
        writer.writerows(manifest.records)


def load_split_manifest(path: Path) -> SplitManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest(
        seed=int(payload["seed"]),
        val_ratio=float(payload["val_ratio"]),
        test_ratio=float(payload["test_ratio"]),
        records=list(payload["records"]),
    )


def parse_path_maps(items: Sequence[str]) -> list[tuple[str, str]]:
    maps: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --path-map {item!r}; expected FROM=TO")
        src, dst = item.split("=", 1)
        maps.append((src.replace("\\", "/").rstrip("/"), dst.rstrip("\\/")))
    return maps


def map_path(value: str | Path, path_maps: Sequence[tuple[str, str]]) -> Path:
    raw = str(value)
    path = Path(raw)
    if path.exists():
        return path
    comparable = raw.replace("\\", "/")
    for src, dst in path_maps:
        if comparable.startswith(src):
            suffix = comparable[len(src):].lstrip("/")
            candidate = Path(dst) / Path(*suffix.split("/"))
            if candidate.exists():
                return candidate
    return path


def remap_manifest_records(records: Sequence[dict], path_maps: Sequence[tuple[str, str]]) -> tuple[list[dict], int]:
    if not path_maps:
        return [dict(row) for row in records], 0
    remapped: list[dict] = []
    n_changed = 0
    for row in records:
        updated = dict(row)
        for key in ("image", "label"):
            mapped = map_path(updated[key], path_maps)
            mapped_text = str(mapped)
            if mapped_text != str(updated[key]):
                updated[key] = mapped_text
                n_changed += 1
        remapped.append(updated)
    return remapped, n_changed


def coerce_mask_2d(masks: np.ndarray, *, npz_mask_channel: str, label_path: Path, allow_channel_select: bool) -> np.ndarray:
    masks = np.squeeze(masks)
    if masks.ndim == 3 and not allow_channel_select and masks.shape[-1] in (3, 4):
        if np.array_equal(masks[..., 0], masks[..., 1]) and np.array_equal(masks[..., 0], masks[..., 2]):
            masks = masks[..., 0]
    if masks.ndim == 3 and allow_channel_select:
        if masks.shape[-1] <= 8:
            if npz_mask_channel == "first":
                masks = masks[..., 0]
            elif npz_mask_channel == "last":
                masks = masks[..., -1]
            elif npz_mask_channel == "max":
                masks = np.max(masks, axis=-1)
            else:
                try:
                    masks = masks[..., int(npz_mask_channel)]
                except Exception as exc:
                    raise ValueError(f"Invalid --npz-mask-channel {npz_mask_channel!r}") from exc
        elif masks.shape[0] <= 8:
            if npz_mask_channel == "first":
                masks = masks[0]
            elif npz_mask_channel == "last":
                masks = masks[-1]
            elif npz_mask_channel == "max":
                masks = np.max(masks, axis=0)
            else:
                try:
                    masks = masks[int(npz_mask_channel)]
                except Exception as exc:
                    raise ValueError(f"Invalid --npz-mask-channel {npz_mask_channel!r}") from exc
    if masks.ndim != 2:
        raise ValueError(f"Expected 2D mask labels in {label_path}, got {masks.shape}")
    return masks


def load_mask(
    label_path: str | Path,
    frame_id: str | None = None,
    npz_mask_channel: str = "last",
    npz_cache_dir: str | Path | None = None,
) -> np.ndarray:
    label_path = Path(label_path)
    suffix = label_path.suffix.lower()
    if suffix == ".npz":
        masks = load_npz_item(label_path, frame_id, "mask", npz_cache_dir=npz_cache_dir)
    elif suffix == ".npy":
        dat = _load_seg_npy_compat(str(label_path))
        masks = dat.get("masks")
    elif suffix in (".tif", ".tiff"):
        masks = tifffile.imread(label_path)
    elif suffix == ".png":
        with Image.open(label_path) as img:
            masks = np.asarray(img)
    else:
        raise ValueError(f"Unsupported label file type: {label_path}")
    masks = coerce_mask_2d(
        masks,
        npz_mask_channel=npz_mask_channel,
        label_path=label_path,
        allow_channel_select=suffix == ".npz",
    )
    return masks.astype(np.int32, copy=False)


def load_label_components(
    label_path: str | Path,
    frame_id: str | None = None,
    npz_mask_channel: str = "last",
    npz_cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an instance mask and optional semantic class map from a label file."""
    label_path = Path(label_path)
    suffix = label_path.suffix.lower()
    class_map = None
    class_names = None
    classes = None
    if suffix == ".npz":
        masks = load_npz_item(label_path, frame_id, "mask", npz_cache_dir=npz_cache_dir)
    elif suffix == ".npy":
        dat = _load_seg_npy_compat(str(label_path))
        masks = dat.get("masks")
        class_map = dat.get("classes_map")
        classes = dat.get("classes")
        class_names = dat.get("class_names")
    elif suffix in (".tif", ".tiff"):
        masks = tifffile.imread(label_path)
        for cls_suffix in ("_classes.tif", "_classes.tiff", "_classes.png"):
            cls_path = label_path.with_name(label_path.stem.replace("_masks", "").replace("_mask", "") + cls_suffix)
            if cls_path.exists():
                class_map = tifffile.imread(cls_path) if cls_path.suffix.lower() in (".tif", ".tiff") else np.asarray(Image.open(cls_path))
                break
    elif suffix == ".png":
        with Image.open(label_path) as img:
            masks = np.asarray(img)
        cls_path = label_path.with_name(label_path.stem.replace("_masks", "").replace("_mask", "") + "_classes.png")
        if cls_path.exists():
            with Image.open(cls_path) as cls_img:
                class_map = np.asarray(cls_img)
    else:
        raise ValueError(f"Unsupported label file type: {label_path}")

    if masks is None:
        raise ValueError(f"Label file has no masks: {label_path}")
    masks = coerce_mask_2d(
        masks,
        npz_mask_channel=npz_mask_channel,
        label_path=label_path,
        allow_channel_select=suffix == ".npz",
    ).astype(np.int32, copy=False)
    if class_map is None and classes is not None:
        class_map = build_classes_map_from_masks(masks, classes)
    if class_map is not None:
        class_map = sanitize_class_map(
            class_map,
            masks=masks,
            classes=classes,
            class_names=class_names,
        )
    return masks, class_map


def insert_class_map(flow_labels: Sequence[np.ndarray], class_maps: Sequence[np.ndarray | None]) -> list[np.ndarray]:
    """Insert semantic class maps after the mask channel in precomputed flow labels."""
    updated: list[np.ndarray] = []
    for idx, lbl in enumerate(flow_labels):
        cmap = class_maps[idx] if idx < len(class_maps) else None
        try:
            arr = np.asarray(lbl)
            if arr.ndim != 3:
                updated.append(arr)
                continue
            if cmap is not None:
                cmap = np.squeeze(np.asarray(cmap))
            if cmap is None or getattr(cmap, "ndim", 0) != 2 or tuple(cmap.shape) != tuple(arr.shape[1:]):
                cmap_arr = np.zeros(arr.shape[1:], dtype=np.int64)
            else:
                cmap_arr = np.rint(cmap).astype(np.int64, copy=False)
                cmap_arr = np.clip(cmap_arr, 0, max(0, int(np.max(cmap_arr))))
            if arr.shape[0] >= 5:
                combined = np.concatenate((arr[:1], cmap_arr[np.newaxis, ...], arr[2:]), axis=0)
            else:
                combined = np.concatenate((arr[:1], cmap_arr[np.newaxis, ...], arr[1:]), axis=0)
            updated.append(combined.astype(np.float32, copy=False))
        except Exception:
            updated.append(np.asarray(lbl))
    return updated


def valid_class_maps(class_maps: Sequence[np.ndarray | None]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for class_map in class_maps:
        if class_map is None:
            continue
        try:
            cmap = np.squeeze(np.asarray(class_map))
            if cmap.ndim == 2:
                out.append(np.rint(cmap).astype(np.int64, copy=False))
        except Exception:
            continue
    return out


def load_image_ref(
    image_service: ImageService,
    image: str,
    frame_id: str | None,
    npz_cache_dir: str | Path | None = None,
) -> np.ndarray:
    if Path(image).suffix.lower() == ".npz":
        return to_channels_last_preserve(load_npz_item(image, frame_id, "image", npz_cache_dir=npz_cache_dir))
    if frame_id:
        arr = image_service.load_frame(image, frame_id)
    else:
        arr = image_service.load_image(image)
    if arr is None:
        raise ValueError(f"Could not load image/frame: {image}::{frame_id}")
    return to_channels_last_preserve(np.asarray(arr))


def records_for_split(records: Sequence[dict], split: str) -> list[dict]:
    return [row for row in records if row["split"] == split]


def source_balanced_probs(records: Sequence[dict], mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode != "source":
        raise ValueError("--balance-mode must be 'source' or 'none'")
    by_group: dict[str, int] = {}
    for row in records:
        by_group[row["source_group"]] = by_group.get(row["source_group"], 0) + 1
    if not records or not by_group:
        return None
    n_groups = len(by_group)
    probs = np.array(
        [1.0 / (n_groups * by_group[row["source_group"]]) for row in records],
        dtype=np.float64,
    )
    probs /= probs.sum()
    return probs


def limit_records(records: Sequence[dict], limit: int, seed: int) -> list[dict]:
    records = list(records)
    if limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = {}
    for row in records:
        by_group.setdefault(row["source_group"], []).append(row)
    selected: list[dict] = []
    groups = sorted(by_group)
    per_group = max(1, limit // max(1, len(groups)))
    for group in groups:
        rows = list(by_group[group])
        rng.shuffle(rows)
        selected.extend(rows[: min(per_group, len(rows))])
    remaining = [row for row in records if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    rng.shuffle(selected)
    return selected[:limit]


def parse_record_caps(items: Sequence[str]) -> list[tuple[str, int]]:
    caps: list[tuple[str, int]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid record cap {item!r}; expected MATCH=LIMIT")
        match, limit_text = item.split("=", 1)
        match = match.strip()
        if not match:
            raise ValueError(f"Invalid record cap {item!r}; MATCH cannot be empty")
        try:
            limit = int(limit_text)
        except ValueError as exc:
            raise ValueError(f"Invalid record cap {item!r}; LIMIT must be an integer") from exc
        if limit < 0:
            raise ValueError(f"Invalid record cap {item!r}; LIMIT must be >= 0")
        caps.append((match.lower(), limit))
    return caps


def record_matches_cap(row: dict, match: str) -> bool:
    image = str(row.get("image", "")).replace("\\", "/").lower()
    label = str(row.get("label", "")).replace("\\", "/").lower()
    source_group = str(row.get("source_group", "")).replace("\\", "/").lower()
    return match in image or match in label or match in source_group


def limit_records_by_path_caps(records: Sequence[dict], caps: Sequence[tuple[str, int]], seed: int) -> list[dict]:
    selected = list(records)
    for index, (match, limit) in enumerate(caps):
        matched = [row for row in selected if record_matches_cap(row, match)]
        unmatched = [row for row in selected if not record_matches_cap(row, match)]
        capped = limit_records(matched, limit, seed + 1000 + index)
        print(f"path cap {match!r}: {len(matched)} -> {len(capped)} training records")
        selected = unmatched + capped
    return selected


def load_records(
    records: Sequence[dict],
    npz_mask_channel: str,
    channel_sampling_mode: str,
    max_all_channel_combos: int,
    seed: int,
    npz_cache_dir: str | Path | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray | None], list[str], list[dict], list[str]]:
    image_service = ImageService()
    rng = random.Random(seed)
    data: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    class_maps: list[np.ndarray | None] = []
    files: list[str] = []
    valid_records: list[dict] = []
    invalid: list[str] = []
    for index, row in enumerate(records, start=1):
        frame_id = row.get("frame_id") or None
        try:
            image = load_image_ref(image_service, row["image"], frame_id, npz_cache_dir=npz_cache_dir)
            mask, class_map = load_label_components(
                row["label"],
                frame_id=frame_id,
                npz_mask_channel=npz_mask_channel,
                npz_cache_dir=npz_cache_dir,
            )
            specs = channel_view_specs(image, channel_sampling_mode, max_all_channel_combos, rng)
            for view_name, channels in specs:
                data.append(make_three_channel_view(image, channels))
                labels.append(mask)
                class_maps.append(class_map)
                file_ref = f"{row['image']}::{frame_id}" if frame_id else row["image"]
                files.append(f"{file_ref}::channels={view_name}")
                valid_row = dict(row)
                valid_row["channel_view"] = view_name
                valid_row["source_group"] = f"{row['source_group']}|channels:{view_name}"
                valid_records.append(valid_row)
        except Exception as exc:
            invalid.append(f"{row.get('image')}::{frame_id or ''} label={row.get('label')} error={exc}")
        if index % 250 == 0:
            print(f"loaded {index}/{len(records)} records")
    if invalid:
        print(f"skipped {len(invalid)} invalid records")
        for line in invalid[:20]:
            print(f"  {line}")
    return data, labels, class_maps, files, valid_records, invalid


def summarize(records: Sequence[dict]) -> None:
    by_split: dict[str, int] = {}
    by_group: dict[tuple[str, str], int] = {}
    for row in records:
        split = row["split"]
        group = row["source_group"]
        by_split[split] = by_split.get(split, 0) + 1
        by_group[(split, group)] = by_group.get((split, group), 0) + 1
    print("split counts:")
    for split, count in sorted(by_split.items()):
        print(f"  {split}: {count}")
    print("source counts by split:")
    for (split, group), count in sorted(by_group.items()):
        print(f"  {split} | {group}: {count}")


def build_net(base_model: str, device: torch.device, semantic_classes: int = 0):
    model_kwargs = {}
    if semantic_classes > 0:
        model_kwargs["semantic_classes"] = semantic_classes
    model = CellposeModel(pretrained_model=base_model, gpu=device.type == "cuda", **model_kwargs)
    return model.net.to(device)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dirs", nargs="+", default=())
    parser.add_argument("--data-dirs", nargs="*", default=())
    parser.add_argument("--output-dir", default="cpsam_finetune_balanced")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--redo-splits", action="store_true")
    parser.add_argument(
        "--path-map",
        nargs="*",
        default=[r"/rds/general/user/mfletch1/home=X:\home"],
        help=(
            "Optional path prefix maps for existing split manifests, e.g. "
            "/rds/general/user/name/home=X:\\home. Applied only when original paths are missing."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--base-model", default="cpsam")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, default=0.5)
    parser.add_argument("--seg-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--semantic-classes",
        type=int,
        default=0,
        help=(
            "Number of semantic classes including background. Use 4 for three "
            "non-background classes, giving 7 output maps with CPSAM."
        ),
    )
    parser.add_argument("--unfreeze-blocks", type=int, default=9)
    parser.add_argument("--balance-mode", default="source", choices=("source", "none"))
    parser.add_argument("--max-train-records", type=int, default=0, help="Optional cap on unique training records loaded into memory.")
    parser.add_argument(
        "--max-train-records-by-path",
        nargs="*",
        default=(),
        help=(
            "Optional per-path training caps as MATCH=LIMIT, applied before "
            "--max-train-records. MATCH is compared against image, label, and "
            "source_group paths, e.g. FoundationTrain=500 cpsamOODtest=1500."
        ),
    )
    parser.add_argument("--max-val-records", type=int, default=0, help="Optional cap on validation records loaded into memory.")
    parser.add_argument("--nimg-per-epoch", type=int, default=0, help="Images sampled per epoch. Defaults to number of unique loaded training records.")
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help=(
            "Stop training when validation loss does not improve. Validation is "
            "evaluated by the underlying Cellpose loop at epoch 0, epoch 5, then every 10 epochs."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Number of validation checks without improvement before early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss decrease required to count as an improvement.",
    )
    parser.add_argument(
        "--channel-sampling-mode",
        default="single-and-all",
        choices=("single-and-all", "none"),
        help="For multichannel images, train on all available input channels plus each single channel.",
    )
    parser.add_argument(
        "--max-all-channel-combos",
        type=int,
        default=2,
        help="For images with more than 3 channels, add this many random 3-channel combination views in addition to single channels.",
    )
    parser.add_argument(
        "--channel-sampling-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same channel view expansion to validation records.",
    )
    parser.add_argument(
        "--npz-mask-channel",
        default="last",
        help="For 3D mask arrays in .npz datasets, select first, last, max, or a numeric channel index.",
    )
    parser.add_argument(
        "--npz-cache-dir",
        default=None,
        help="Directory for extracted .npz member .npy files. Required to train from large .npz archives efficiently.",
    )
    parser.add_argument("--use-validation-as-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--print-frozen-layers",
        action="store_true",
        help="After applying --unfreeze-blocks, print parameters whose requires_grad flag is False.",
    )
    parser.add_argument(
        "--print-trainable-layers",
        action="store_true",
        help="After applying --unfreeze-blocks, print parameters whose requires_grad flag is True.",
    )
    parser.add_argument(
        "--trainability-report-only",
        action="store_true",
        help="Build the model, apply --unfreeze-blocks, print the trainability report, then exit before loading data.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.semantic_classes < 0:
        parser.error("--semantic-classes must be >= 0")
    try:
        args.max_train_records_by_path = parse_record_caps(args.max_train_records_by_path)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.root_dirs and not args.data_dirs and not args.trainability_report_only:
        parser.error("Provide --root-dirs and/or --data-dirs")
    return args


def print_trainability_report(net, *, print_frozen: bool = False, print_trainable: bool = False) -> None:
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    frozen_names = []
    trainable_names = []

    for name, param in net.named_parameters():
        n_params = int(param.numel())
        total_params += n_params
        if param.requires_grad:
            trainable_params += n_params
            trainable_names.append((name, n_params))
        else:
            frozen_params += n_params
            frozen_names.append((name, n_params))

    print(
        "trainability summary: "
        f"total_params={total_params:,} trainable={trainable_params:,} frozen={frozen_params:,}"
    )

    if hasattr(net, "encoder") and hasattr(net.encoder, "blocks"):
        print("encoder block trainability:")
        for index, block in enumerate(net.encoder.blocks):
            block_params = list(block.named_parameters())
            n_total = sum(int(param.numel()) for _, param in block_params)
            n_trainable = sum(int(param.numel()) for _, param in block_params if param.requires_grad)
            if n_trainable == 0:
                state = "frozen"
            elif n_trainable == n_total:
                state = "trainable"
            else:
                state = "mixed"
            print(f"  encoder.blocks.{index:02d}: {state} ({n_trainable:,}/{n_total:,} params trainable)")

    if print_frozen:
        print("parameters with requires_grad=False:")
        for name, n_params in frozen_names:
            print(f"  frozen {name} ({n_params:,} params)")

    if print_trainable:
        print("parameters with requires_grad=True:")
        for name, n_params in trainable_names:
            print(f"  trainable {name} ({n_params:,} params)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else output_dir / "cpsam_finetune_splits.json"

    if args.trainability_report_only:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        net = build_net(args.base_model, device, args.semantic_classes)
        trainability_info = configure_trainable_params(
            net,
            use_lora=False,
            lora_blocks=None,
            unfreeze_blocks=args.unfreeze_blocks,
            logger=None,
        )
        print(
            "applied trainability policy: "
            f"encoder_blocks={trainability_info.get('total_encoder_blocks', 0)} "
            f"unfrozen_encoder_blocks={trainability_info.get('applied_unfreeze_blocks', 0)}"
        )
        print_trainability_report(
            net,
            print_frozen=True if not args.print_trainable_layers else args.print_frozen_layers,
            print_trainable=args.print_trainable_layers,
        )
        return 0

    if split_manifest_path.exists() and not args.redo_splits:
        manifest = load_split_manifest(split_manifest_path)
        print(f"loaded existing split manifest: {split_manifest_path}")
    else:
        labeled = discover_labeled_refs(args.root_dirs, args.data_dirs)
        if not labeled:
            raise ValueError("No labeled image/frame pairs discovered.")
        records = split_labeled_refs(labeled, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        manifest = SplitManifest(
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            records=records,
        )
        write_split_manifest(split_manifest_path, manifest)
        print(f"wrote split manifest: {split_manifest_path}")
        print(f"wrote split CSV: {split_manifest_path.with_suffix('.csv')}")

    path_maps = parse_path_maps(args.path_map)
    mapped_records, n_mapped_paths = remap_manifest_records(manifest.records, path_maps)
    if n_mapped_paths:
        print(f"remapped {n_mapped_paths} manifest paths using --path-map")
        manifest = SplitManifest(
            seed=manifest.seed,
            val_ratio=manifest.val_ratio,
            test_ratio=manifest.test_ratio,
            records=mapped_records,
        )

    summarize(manifest.records)
    train_records = records_for_split(manifest.records, "train")
    val_records = records_for_split(manifest.records, "val")
    test_records = records_for_split(manifest.records, "test")
    train_records_capped = limit_records_by_path_caps(train_records, args.max_train_records_by_path, args.seed)
    train_records_loaded = limit_records(train_records_capped, args.max_train_records, args.seed)
    val_records_loaded = limit_records(val_records, args.max_val_records, args.seed + 1)
    print(f"training records after cap: {len(train_records)} -> {len(train_records_loaded)}")
    print(f"held-out validation records: {len(val_records)}")
    print(f"held-out test records: {len(test_records)}")

    if args.dry_run:
        return 0

    train_data, train_labels, train_class_maps, train_files, valid_train_records, train_invalid = load_records(
        train_records_loaded,
        args.npz_mask_channel,
        args.channel_sampling_mode,
        args.max_all_channel_combos,
        args.seed,
        args.npz_cache_dir,
    )
    val_data, val_labels, val_class_maps, val_files, valid_val_records, val_invalid = load_records(
        val_records_loaded,
        args.npz_mask_channel,
        args.channel_sampling_mode if args.channel_sampling_val else "none",
        args.max_all_channel_combos,
        args.seed + 1,
        args.npz_cache_dir,
    )
    if not train_data:
        raise ValueError("No valid training data loaded.")
    train_probs = source_balanced_probs(valid_train_records, args.balance_mode)
    if train_probs is not None:
        print(f"source-balanced sampling enabled across {len(set(row['source_group'] for row in valid_train_records))} valid groups")
    else:
        print("source-balanced sampling disabled")
    print(f"valid training records loaded: {len(train_data)}/{len(train_records_loaded)}")
    print(f"valid validation records loaded: {len(val_data)}/{len(val_records_loaded)}")
    if len(train_data) != len(train_labels):
        raise ValueError(f"Internal loader error: train data/labels length mismatch ({len(train_data)} != {len(train_labels)})")
    if len(train_data) != len(train_class_maps):
        raise ValueError(
            f"Internal loader error: train data/class-map length mismatch ({len(train_data)} != {len(train_class_maps)})"
        )
    if train_probs is not None and len(train_probs) != len(train_data):
        raise ValueError(f"Internal loader error: train probabilities/data length mismatch ({len(train_probs)} != {len(train_data)})")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    net = build_net(args.base_model, device, args.semantic_classes)
    model_name = args.model_name or f"cpsam_finetune_last{args.unfreeze_blocks}_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"training model={model_name} on device={device}")
    if args.semantic_classes > 0:
        print(f"semantic segmentation enabled: classes_including_background={args.semantic_classes}")
    else:
        print("semantic segmentation disabled: training class-agnostic CPSAM outputs")
    print(f"unfreezing last {args.unfreeze_blocks} encoder blocks plus CPSAM heads")
    trainability_info = configure_trainable_params(
        net,
        use_lora=False,
        lora_blocks=None,
        unfreeze_blocks=args.unfreeze_blocks,
        logger=None,
    )
    print(
        "applied trainability policy: "
        f"encoder_blocks={trainability_info.get('total_encoder_blocks', 0)} "
        f"unfrozen_encoder_blocks={trainability_info.get('applied_unfreeze_blocks', 0)}"
    )
    if args.print_frozen_layers or args.print_trainable_layers:
        print_trainability_report(
            net,
            print_frozen=args.print_frozen_layers,
            print_trainable=args.print_trainable_layers,
        )
    if not hasattr(net, "diam_mean") or net.diam_mean is None or not torch.is_tensor(net.diam_mean):
        net.diam_mean = torch.nn.Parameter(torch.tensor([30.0], device=device), requires_grad=False)
    if not hasattr(net, "diam_labels") or net.diam_labels is None or not torch.is_tensor(net.diam_labels):
        net.diam_labels = torch.nn.Parameter(torch.tensor([30.0], device=device), requires_grad=False)
    normalize_params = dict(cp_models.normalize_default)
    normalize_params["normalize"] = True
    class_weights = None
    train_labels_for_training = train_labels
    val_labels_for_training = val_labels
    if args.semantic_classes > 0:
        valid_train_class_maps = valid_class_maps(train_class_maps)
        max_class_id = max((int(np.max(cm)) for cm in valid_train_class_maps), default=-1)
        inferred_nclasses = infer_semantic_nclasses_from_net(net)
        class_weights = compute_class_weights_from_class_maps(
            valid_train_class_maps,
            nclasses=args.semantic_classes if args.semantic_classes > 0 else inferred_nclasses,
        )
        print(
            "semantic class-weight inputs: "
            f"valid_class_maps={len(valid_train_class_maps)} max_class_id={max_class_id} "
            f"inferred_nclasses={inferred_nclasses}"
        )
        if class_weights is not None:
            print(f"semantic class weights ({len(class_weights)} classes incl. background): {class_weights.tolist()}")
        else:
            print("semantic class weights unavailable; using unweighted cross entropy")
        print("precomputing semantic flow labels with class-map channel")
        train_flow_labels = dynamics.labels_to_flows(train_labels, files=None, device=device)
        train_labels_for_training = insert_class_map(train_flow_labels, train_class_maps)
        if val_labels:
            val_flow_labels = dynamics.labels_to_flows(val_labels, files=None, device=device)
            val_labels_for_training = insert_class_map(val_flow_labels, val_class_maps)
    model_path, train_losses, val_losses = cellpose_train.train_seg(
        net,
        train_data=train_data,
        train_labels=train_labels_for_training,
        train_probs=train_probs,
        test_data=val_data if args.use_validation_as_test and val_data else None,
        test_labels=val_labels_for_training if args.use_validation_as_test and val_data else None,
        train_files=None,
        test_files=None,
        normalize=normalize_params,
        min_train_masks=0,
        batch_size=args.batch_size,
        bsize=args.bsize,
        rescale=False,
        scale_range=args.scale_range,
        save_path=str(output_dir),
        nimg_per_epoch=args.nimg_per_epoch if args.nimg_per_epoch > 0 else len(train_data),
        nimg_test_per_epoch=len(val_data) if args.use_validation_as_test and val_data else 0,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        early_stop=args.early_stop,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        model_name=model_name,
        seg_loss_weight=args.seg_loss_weight,
        class_weights=class_weights,
    )
    result_path = output_dir / "training_result.json"
    result_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "train_losses": train_losses.tolist() if hasattr(train_losses, "tolist") else train_losses,
                "validation_losses": val_losses.tolist() if hasattr(val_losses, "tolist") else val_losses,
                "split_manifest": str(split_manifest_path),
                "n_train_records": len(train_data),
                "n_requested_train_records": len(train_records_loaded),
                "max_train_records_by_path": args.max_train_records_by_path,
                "n_val_records": len(val_data),
                "n_requested_val_records": len(val_records_loaded),
                "n_test_records": len(test_records),
                "semantic_classes": args.semantic_classes,
                "semantic_class_weights": class_weights.tolist() if class_weights is not None else None,
                "semantic_class_weighted_loss": bool(class_weights is not None),
                "semantic_valid_class_maps": len(valid_class_maps(train_class_maps)) if args.semantic_classes > 0 else 0,
                "npz_mask_channel": args.npz_mask_channel,
                "npz_cache_dir": args.npz_cache_dir,
                "channel_sampling_mode": args.channel_sampling_mode,
                "max_all_channel_combos": args.max_all_channel_combos,
                "channel_sampling_val": args.channel_sampling_val,
                "early_stop": args.early_stop,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "invalid_train_records": train_invalid,
                "invalid_val_records": val_invalid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"training finished: {model_path}")
    print(f"held-out test split remains unused for final evaluation: {len(test_records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
