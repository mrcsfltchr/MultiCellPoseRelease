#!/usr/bin/env python3
"""
Evaluate semantic instance segmentation on a labeled test set.

This is an updated/packaged variant of the logic from:
  ../cellpose/tools/eval_inst_seg.py

It computes:
1) Class-aware instance metrics (TP/FP/FN, F1 at IoU thresholds)
2) Semantic pixel metrics (overall acc, foreground acc, mean IoU on FG classes)

Ground-truth inputs per image root:
  - Preferred: <root>_seg.npy (uses "masks" and optional "classes_map"/"classes")
  - Optional:  <root>_masks.tif/.tiff
  - Optional:  <root>_classes.tif/.tiff

Example:
  python scripts/eval_semantic_inst_seg.py \
    --dir C:/data/test \
    --model C:/Users/me/.cellpose/models/my_model \
    --iou 0.5 0.75 \
    --csv eval.csv \
    --conf-json eval_confusion.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile as tiff

from cellpose import io as cp_io
from cellpose import models as cp_models
from cellpose import utils as cp_utils


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2", ".lif", ".nrrd"}
DERIVED_SUBSTRINGS = ("_masks", "_classes", "_flows")

try:
    # Keep evaluator aligned with GUI defaults.
    from guv_app.services.segmentation_service import _LEGACY_EVAL_DEFAULTS as GUI_EVAL_DEFAULTS
except Exception:
    GUI_EVAL_DEFAULTS = {
        "cellprob_threshold": -0.5,
        "flow_threshold": 1.0,
        "do_3D": False,
        "niter": 0,
        "stitch_threshold": 0.0,
        "anisotropy": 1.0,
        "flow3D_smooth": 0.0,
        "min_size": 15,
        "max_size_fraction": 1.0,
    }


def list_base_images(folder: Path) -> list[Path]:
    images: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem_l = p.stem.lower()
        if any(s in stem_l for s in DERIVED_SUBSTRINGS):
            continue
        images.append(p)
    return sorted(images)


def read_eval_image(path: Path) -> np.ndarray:
    """Read only the first logical frame (S0/T0) when possible.

    This avoids loading full multi-series/time stacks into memory during eval.
    """
    try:
        series_key, series_count, time_count = cp_io.get_series_time_info(str(path))
        frame_parts = []
        if series_key and int(series_count or 1) > 1:
            frame_parts.append(f"{series_key}0")
        if int(time_count or 1) > 1:
            frame_parts.append("T0")
        frame_id = "_".join(frame_parts) if frame_parts else None
        frame = cp_io.read_image_frame(str(path), frame_id)
        if frame is not None and frame.array is not None:
            return np.asarray(frame.array)
    except Exception:
        pass
    return np.asarray(cp_io.imread(str(path)))


def iter_eval_frames(path: Path):
    """Yield (frame_id, image_array) using ImageReader APIs.

    For single-frame images, frame_id is None.
    """
    # TIFF RGB/multichannel should be treated as a single image, not a frame series.
    # Some readers expose channel planes as pseudo-series (S0/S1/...), which is not
    # desired for eval on channel images like (H, W, 3).
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            dat = cp_io.read_image_data(str(path))
            arr = np.asarray(dat.array) if dat is not None and dat.array is not None else None
            if arr is not None and arr.ndim == 3:
                # channels-last RGB/multichannel
                if arr.shape[-1] <= 4 and arr.shape[0] > 8 and arr.shape[1] > 8:
                    yield None, arr
                    return
                # channels-first RGB/multichannel -> convert to channels-last
                if arr.shape[0] <= 4 and arr.shape[1] > 8 and arr.shape[2] > 8:
                    yield None, np.moveaxis(arr, 0, -1)
                    return
        except Exception:
            pass

    try:
        series_key, series_count, time_count = cp_io.get_series_time_info(str(path))
        series_count = int(series_count or 1)
        time_count = int(time_count or 1)
    except Exception:
        series_key, series_count, time_count = None, 1, 1

    if series_count <= 1 and time_count <= 1:
        frame = cp_io.read_image_frame(str(path), None)
        if frame is not None and frame.array is not None:
            yield None, np.asarray(frame.array)
            return
        yield None, read_eval_image(path)
        return

    for s in range(max(1, series_count)):
        for t in range(max(1, time_count)):
            parts = []
            if series_key and series_count > 1:
                parts.append(f"{series_key}{s}")
            if time_count > 1:
                parts.append(f"T{t}")
            frame_id = "_".join(parts) if parts else None
            try:
                frame = cp_io.read_image_frame(str(path), frame_id)
            except Exception:
                frame = None
            if frame is None or frame.array is None:
                continue
            yield frame_id, np.asarray(frame.array)


def _label_stems(base: Path, frame_id: Optional[str]) -> list[str]:
    stems = [base.stem]
    if frame_id:
        stems.insert(0, f"{base.stem}__{frame_id}")
    return stems


def load_gt_masks(base: Path, frame_id: Optional[str] = None) -> Optional[np.ndarray]:
    for stem in _label_stems(base, frame_id):
        seg_path = base.with_suffix("").with_name(stem + "_seg.npy")
        if seg_path.exists():
            try:
                dat = np.load(seg_path, allow_pickle=True).item()
                masks = dat.get("masks")
                if masks is not None:
                    masks = np.asarray(masks)
                    if masks.ndim == 3:
                        masks = masks.squeeze()
                    return masks.astype(np.int32, copy=False)
            except Exception:
                pass
        for suf in ("_masks.tif", "_masks.tiff"):
            tif_path = base.with_suffix("").with_name(stem + suf)
            if tif_path.exists():
                try:
                    masks = tiff.imread(str(tif_path))
                    if masks.ndim == 3:
                        masks = masks.squeeze()
                    return masks.astype(np.int32, copy=False)
                except Exception:
                    pass
    return None


def load_gt_classes_map(
    base: Path,
    gt_masks: Optional[np.ndarray],
    frame_id: Optional[str] = None,
) -> Optional[np.ndarray]:
    for stem in _label_stems(base, frame_id):
        # Prefer classes embedded in *_seg.npy so evaluation follows Trainer labels.
        seg_path = base.with_suffix("").with_name(stem + "_seg.npy")
        if not seg_path.exists():
            pass
        else:
            try:
                dat = np.load(seg_path, allow_pickle=True).item()
                class_map = dat.get("classes_map")
                if class_map is not None:
                    class_map = np.asarray(class_map)
                    if class_map.ndim == 3:
                        class_map = class_map.squeeze()
                    return class_map.astype(np.int32, copy=False)
                if gt_masks is not None and "classes" in dat:
                    classes = np.asarray(dat["classes"])
                    class_map = classes[gt_masks]
                    return class_map.astype(np.int32, copy=False)
            except Exception:
                return None

        # Fallback: sidecar class map TIFF (legacy datasets)
        for suf in ("_classes.tif", "_classes.tiff"):
            tif_path = base.with_suffix("").with_name(stem + suf)
            if tif_path.exists():
                try:
                    class_map = tiff.imread(str(tif_path))
                    if class_map.ndim == 3:
                        class_map = class_map.squeeze()
                    return class_map.astype(np.int32, copy=False)
                except Exception:
                    pass
    return None


def get_instance_classes(mask_img: np.ndarray, class_map: Optional[np.ndarray]) -> np.ndarray:
    max_id = int(mask_img.max()) if mask_img is not None else 0
    out = np.zeros(max_id + 1, dtype=np.int32)
    if class_map is None or max_id == 0:
        return out

    # Keep class assignment behavior equivalent to GUI mask class determination
    # in guv_app.services.segmentation_service.postprocess_classes.
    # Additionally handle channel-encoded class maps (H,W,C or C,H,W) by
    # converting them to class IDs before majority vote per instance.
    def _channel_map_to_ids(arr: np.ndarray) -> np.ndarray:
        cls = np.argmax(arr, axis=-1).astype(np.int32) + 1
        try:
            bg = np.all(arr <= 0, axis=-1)
        except Exception:
            bg = (np.sum(arr, axis=-1) == 0)
        cls[bg] = 0
        return cls

    masks_arr = np.asarray(mask_img)
    classes_arr = np.asarray(class_map)

    if masks_arr.ndim == 3 and masks_arr.shape[0] == 1:
        masks_arr = masks_arr[0]
    if classes_arr.ndim == 3 and classes_arr.shape[0] == 1:
        classes_arr = classes_arr[0]

    mh, mw = int(masks_arr.shape[0]), int(masks_arr.shape[1])
    if classes_arr.ndim == 3 and classes_arr.shape[:2] == masks_arr.shape and 1 < classes_arr.shape[2] <= 16:
        classes_arr = _channel_map_to_ids(classes_arr)
    elif classes_arr.ndim == 3 and classes_arr.shape[1:] == masks_arr.shape and 1 < classes_arr.shape[0] <= 16:
        classes_arr = _channel_map_to_ids(np.moveaxis(classes_arr, 0, -1))
    if classes_arr.shape != masks_arr.shape:
        try:
            classes_arr = _coerce_image_for_2d_eval(classes_arr, masks_arr.shape)
        except Exception:
            pass
    if classes_arr.ndim == 3 and classes_arr.shape[:2] == masks_arr.shape and 1 < classes_arr.shape[2] <= 16:
        classes_arr = _channel_map_to_ids(classes_arr)
    if classes_arr.shape != masks_arr.shape:
        try:
            import cv2
            classes_arr = cv2.resize(
                classes_arr.astype(np.int32),
                (mw, mh),
                interpolation=cv2.INTER_NEAREST,
            )
        except Exception:
            try:
                from skimage.transform import resize as sk_resize
                classes_arr = sk_resize(
                    classes_arr,
                    masks_arr.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.int32)
            except Exception:
                return out

    nmask = int(masks_arr.max())
    for mid in range(1, nmask + 1):
        vals = classes_arr[masks_arr == mid]
        vals = vals[vals > 0]
        if vals.size > 0:
            counts = np.bincount(vals.astype(np.int64))
            out[mid] = int(np.argmax(counts))
    return out


def masks_to_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_flat = pred.ravel().astype(np.int64, copy=False)
    gt_flat = gt.ravel().astype(np.int64, copy=False)

    p_ids, p_counts = np.unique(pred_flat[pred_flat > 0], return_counts=True)
    g_ids, g_counts = np.unique(gt_flat[gt_flat > 0], return_counts=True)
    if p_ids.size == 0 or g_ids.size == 0:
        return np.zeros((p_ids.size, g_ids.size), dtype=np.float32)

    p_index = {int(pid): i for i, pid in enumerate(p_ids)}
    g_index = {int(gid): j for j, gid in enumerate(g_ids)}

    fg = (pred_flat > 0) & (gt_flat > 0)
    intersections = np.zeros((len(p_ids), len(g_ids)), dtype=np.float64)
    if np.any(fg):
        pair = np.stack((pred_flat[fg], gt_flat[fg]), axis=1)
        uniq_pair, inter_counts = np.unique(pair, axis=0, return_counts=True)
        for (pid, gid), cnt in zip(uniq_pair, inter_counts):
            i = p_index.get(int(pid))
            j = g_index.get(int(gid))
            if i is not None and j is not None:
                intersections[i, j] = float(cnt)

    union = p_counts[:, None].astype(np.float64) + g_counts[None, :].astype(np.float64) - intersections
    iou = np.divide(
        intersections,
        union,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=union > 0,
    )
    return iou.astype(np.float32, copy=False)


def _aligned_instance_class_vectors(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    pred_inst_classes: np.ndarray,
    gt_inst_classes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map mask-id-indexed class arrays onto IoU matrix row/column order.

    IoU rows/cols are ordered by sorted unique instance IDs (excluding 0).
    """
    pred_ids = np.unique(pred_masks[pred_masks > 0]).astype(np.int64)
    gt_ids = np.unique(gt_masks[gt_masks > 0]).astype(np.int64)

    pred_vec = np.zeros(len(pred_ids), dtype=np.int32)
    gt_vec = np.zeros(len(gt_ids), dtype=np.int32)

    if pred_inst_classes is not None and len(pred_inst_classes) > 0 and len(pred_ids) > 0:
        valid = pred_ids < len(pred_inst_classes)
        pred_vec[valid] = pred_inst_classes[pred_ids[valid]].astype(np.int32, copy=False)
    if gt_inst_classes is not None and len(gt_inst_classes) > 0 and len(gt_ids) > 0:
        valid = gt_ids < len(gt_inst_classes)
        gt_vec[valid] = gt_inst_classes[gt_ids[valid]].astype(np.int32, copy=False)

    return pred_vec, gt_vec


def greedy_match_iou(
    iou: np.ndarray,
    thr: float,
    pred_cls: Optional[np.ndarray],
    gt_cls: Optional[np.ndarray],
) -> Tuple[int, int, int, float]:
    if iou.size == 0:
        return 0, iou.shape[0], iou.shape[1], 0.0
    used_pred = set()
    used_gt = set()
    matched_ious: list[float] = []
    flat = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=True)

    for val, i, j in flat:
        if val < thr:
            break
        if i in used_pred or j in used_gt:
            continue
        if pred_cls is not None and gt_cls is not None:
            pc = int(pred_cls[i]) if i < len(pred_cls) else -1
            gc = int(gt_cls[j]) if j < len(gt_cls) else -1
            if pc != gc:
                used_pred.add(i)
                used_gt.add(j)
                continue
        used_pred.add(i)
        used_gt.add(j)
        matched_ious.append(val)

    tp = len(matched_ious)
    fp = iou.shape[0] - len(used_pred)
    fn = iou.shape[1] - len(used_gt)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return tp, fp, fn, mean_iou


def greedy_match_iou_class_counts(
    iou: np.ndarray,
    thr: float,
    pred_cls: Optional[np.ndarray],
    gt_cls: Optional[np.ndarray],
) -> Tuple[dict[int, int], dict[int, int]]:
    """Return per-class FP/FN counts (instance-level), excluding class 0."""
    fp_cls: dict[int, int] = {}
    fn_cls: dict[int, int] = {}
    if iou.size == 0:
        return fp_cls, fn_cls

    used_pred = set()
    used_gt = set()
    flat = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=True)

    for val, i, j in flat:
        if val < thr:
            break
        if i in used_pred or j in used_gt:
            continue
        pc = int(pred_cls[i]) if pred_cls is not None and i < len(pred_cls) else 0
        gc = int(gt_cls[j]) if gt_cls is not None and j < len(gt_cls) else 0
        if pc != gc:
            used_pred.add(i)
            used_gt.add(j)
            if pc > 0:
                fp_cls[pc] = fp_cls.get(pc, 0) + 1
            if gc > 0:
                fn_cls[gc] = fn_cls.get(gc, 0) + 1
            continue
        used_pred.add(i)
        used_gt.add(j)

    # Unmatched predictions are FP
    if pred_cls is not None:
        for i in range(iou.shape[0]):
            if i in used_pred:
                continue
            c = int(pred_cls[i]) if i < len(pred_cls) else 0
            if c > 0:
                fp_cls[c] = fp_cls.get(c, 0) + 1

    # Unmatched GT are FN
    if gt_cls is not None:
        for j in range(iou.shape[1]):
            if j in used_gt:
                continue
            c = int(gt_cls[j]) if j < len(gt_cls) else 0
            if c > 0:
                fn_cls[c] = fn_cls.get(c, 0) + 1

    return fp_cls, fn_cls


def greedy_match_fp_reason_counts(
    iou: np.ndarray,
    thr: float,
    pred_cls: Optional[np.ndarray],
    gt_cls: Optional[np.ndarray],
) -> Tuple[dict[int, int], dict[int, int]]:
    """Return per-class FP breakdown:
    - fp_mismatch_cls[c]: matched above IoU thr but class-mismatched
    - fp_low_iou_unmatched_cls[c]: unmatched predictions (including low-IoU)
    """
    fp_mismatch_cls: dict[int, int] = {}
    fp_low_iou_unmatched_cls: dict[int, int] = {}
    if iou.size == 0:
        return fp_mismatch_cls, fp_low_iou_unmatched_cls

    used_pred = set()
    used_gt = set()
    flat = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=True)

    for val, i, j in flat:
        if val < thr:
            break
        if i in used_pred or j in used_gt:
            continue
        pc = int(pred_cls[i]) if pred_cls is not None and i < len(pred_cls) else 0
        gc = int(gt_cls[j]) if gt_cls is not None and j < len(gt_cls) else 0
        used_pred.add(i)
        used_gt.add(j)
        if pc != gc and pc > 0:
            fp_mismatch_cls[pc] = fp_mismatch_cls.get(pc, 0) + 1

    # Anything not consumed above threshold is unmatched/low-IoU FP
    if pred_cls is not None:
        for i in range(iou.shape[0]):
            if i in used_pred:
                continue
            c = int(pred_cls[i]) if i < len(pred_cls) else 0
            if c > 0:
                fp_low_iou_unmatched_cls[c] = fp_low_iou_unmatched_cls.get(c, 0) + 1

    return fp_mismatch_cls, fp_low_iou_unmatched_cls


def create_classwise_tp_fp_fn_masks(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    iou: np.ndarray,
    thr: float,
    pred_cls: Optional[np.ndarray],
    gt_cls: Optional[np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    """Create per-class TP/FP/FN mask layers (instance-level matching).

    Returns:
      {
        class_id: {
          "tp": uint8 mask (pred instances correctly matched to same-class GT),
          "fp": uint8 mask (pred instances unmatched or class-mismatched),
          "fn": uint8 mask (GT instances unmatched or class-mismatched),
        }, ...
      }

    Matching policy is intentionally aligned with greedy_match_iou_class_counts:
    - Greedy one-to-one matching by IoU descending.
    - Matches above `thr` consume both instances even when class-mismatched.
    - Class mismatch contributes FP(pred_class) + FN(gt_class).
    """
    pred_masks = _coerce_masks_to_2d_int32(pred_masks)
    gt_masks = _coerce_masks_to_2d_int32(gt_masks)
    h, w = int(pred_masks.shape[0]), int(pred_masks.shape[1])

    # Row/column instance IDs must match the ordering used by masks_to_iou.
    pred_ids = np.unique(pred_masks[pred_masks > 0]).astype(np.int64)
    gt_ids = np.unique(gt_masks[gt_masks > 0]).astype(np.int64)

    out: dict[int, dict[str, np.ndarray]] = {}

    def _ensure_class(c: int):
        c = int(c)
        if c <= 0:
            return None
        if c not in out:
            out[c] = {
                "tp": np.zeros((h, w), dtype=np.uint8),
                "fp": np.zeros((h, w), dtype=np.uint8),
                "fn": np.zeros((h, w), dtype=np.uint8),
            }
        return out[c]

    def _pred_class(i: int) -> int:
        if pred_cls is None:
            return 0
        # Accept either IoU-row-aligned class vectors (len == len(pred_ids))
        # or mask-id-indexed vectors (len > max mask id).
        if len(pred_cls) == len(pred_ids):
            return int(pred_cls[i]) if i < len(pred_cls) else 0
        pid = int(pred_ids[i]) if i < len(pred_ids) else -1
        if 0 <= pid < len(pred_cls):
            return int(pred_cls[pid])
        return int(pred_cls[i]) if i < len(pred_cls) else 0

    def _gt_class(j: int) -> int:
        if gt_cls is None:
            return 0
        # Accept either IoU-col-aligned class vectors (len == len(gt_ids))
        # or mask-id-indexed vectors (len > max mask id).
        if len(gt_cls) == len(gt_ids):
            return int(gt_cls[j]) if j < len(gt_cls) else 0
        gid = int(gt_ids[j]) if j < len(gt_ids) else -1
        if 0 <= gid < len(gt_cls):
            return int(gt_cls[gid])
        return int(gt_cls[j]) if j < len(gt_cls) else 0

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    flat = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=True)

    # Greedy matching stage (same policy as count metrics).
    for val, i, j in flat:
        if val < thr:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        pc = _pred_class(i)
        gc = _gt_class(j)
        pid = int(pred_ids[i]) if i < len(pred_ids) else 0
        gid = int(gt_ids[j]) if j < len(gt_ids) else 0
        if pc == gc and pc > 0:
            dst = _ensure_class(pc)
            if dst is not None and pid > 0:
                dst["tp"][pred_masks == pid] = 1
        else:
            # class mismatch => FP for predicted class, FN for GT class
            if pc > 0 and pid > 0:
                dst = _ensure_class(pc)
                if dst is not None:
                    dst["fp"][pred_masks == pid] = 1
            if gc > 0 and gid > 0:
                dst = _ensure_class(gc)
                if dst is not None:
                    dst["fn"][gt_masks == gid] = 1

    # Unmatched predictions => FP in their class.
    for i in range(iou.shape[0]):
        if i in used_pred:
            continue
        pc = _pred_class(i)
        pid = int(pred_ids[i]) if i < len(pred_ids) else 0
        if pc > 0 and pid > 0:
            dst = _ensure_class(pc)
            if dst is not None:
                dst["fp"][pred_masks == pid] = 1

    # Unmatched GT => FN in their class.
    for j in range(iou.shape[1]):
        if j in used_gt:
            continue
        gc = _gt_class(j)
        gid = int(gt_ids[j]) if j < len(gt_ids) else 0
        if gc > 0 and gid > 0:
            dst = _ensure_class(gc)
            if dst is not None:
                dst["fn"][gt_masks == gid] = 1

    return out


def plot_classwise_tp_fp_fn_overlay(
    image: np.ndarray,
    class_layers: dict[int, dict[str, np.ndarray]],
    class_id: int,
    alpha: float = 0.35,
    figsize: tuple[int, int] = (7, 7),
):
    """Quick matplotlib overlay for one class (TP=green, FP=red, FN=blue)."""
    import matplotlib.pyplot as plt

    cid = int(class_id)
    if cid not in class_layers:
        raise ValueError(f"class_id={cid} not found in class_layers")
    lay = class_layers[cid]

    img = np.asarray(image)
    if img.ndim == 2:
        base = np.stack([img, img, img], axis=-1).astype(np.float32)
    else:
        base = np.asarray(img).astype(np.float32)
        if base.ndim == 3 and base.shape[-1] > 3:
            base = base[..., :3]
    bmin, bmax = float(base.min()), float(base.max())
    if bmax > bmin:
        base = (base - bmin) / (bmax - bmin)
    else:
        base = np.zeros_like(base, dtype=np.float32)

    overlay = np.zeros_like(base, dtype=np.float32)
    overlay[..., 1] = np.asarray(lay["tp"], dtype=np.float32)  # green
    overlay[..., 0] = np.maximum(overlay[..., 0], np.asarray(lay["fp"], dtype=np.float32))  # red
    overlay[..., 2] = np.maximum(overlay[..., 2], np.asarray(lay["fn"], dtype=np.float32))  # blue

    vis = np.clip((1.0 - alpha) * base + alpha * overlay, 0.0, 1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(vis)
    ax.set_title(f"Class {cid}: TP(green), FP(red), FN(blue)")
    ax.axis("off")
    return fig, ax


def _class_counts_to_str(counts: dict[int, int]) -> str:
    if not counts:
        return ""
    items = [(int(k), int(v)) for k, v in counts.items() if int(k) > 0 and int(v) > 0]
    if not items:
        return ""
    items.sort(key=lambda x: x[0])
    # Avoid ":" because spreadsheet tools often auto-cast "1:17" as time.
    return "|".join(f"c{k}={v}" for k, v in items)


def instance_confusion_matrix_counts(
    iou: np.ndarray,
    pred_cls: Optional[np.ndarray],
    gt_cls: Optional[np.ndarray],
    thr: float,
) -> Tuple[np.ndarray, list[int], dict[int, int], dict[int, int], dict[int, int]]:
    """Instance-level confusion matrix (rows=GT class, cols=Pred class).

    Class 0 is reserved for background:
    - unmatched GT instance contributes to (gt_class, 0)  -> FN
    - unmatched Pred instance contributes to (0, pred_class) -> FP
    """
    pred_classes = []
    gt_classes = []
    if pred_cls is not None:
        pred_classes = [int(c) for c in np.unique(pred_cls) if int(c) > 0]
    if gt_cls is not None:
        gt_classes = [int(c) for c in np.unique(gt_cls) if int(c) > 0]
    classes = sorted(set([0] + pred_classes + gt_classes))
    idx = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=np.int64)

    if iou.size == 0:
        tp_by_class = {c: 0 for c in classes if c > 0}
        fp_by_class = {c: 0 for c in classes if c > 0}
        fn_by_class = {c: 0 for c in classes if c > 0}
        return mat, classes, tp_by_class, fp_by_class, fn_by_class

    used_pred = set()
    used_gt = set()
    flat = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    flat.sort(key=lambda x: x[0], reverse=True)

    # Matched pairs above threshold (including class-mismatch matches).
    for val, i, j in flat:
        if val < thr:
            break
        if i in used_pred or j in used_gt:
            continue
        pc = int(pred_cls[i]) if pred_cls is not None and i < len(pred_cls) else 0
        gc = int(gt_cls[j]) if gt_cls is not None and j < len(gt_cls) else 0
        if gc < 0:
            gc = 0
        if pc < 0:
            pc = 0
        if gc not in idx:
            classes.append(gc)
            idx[gc] = len(classes) - 1
            mat = np.pad(mat, ((0, 1), (0, 0)), mode="constant")
        if pc not in idx:
            classes.append(pc)
            idx[pc] = len(classes) - 1
            mat = np.pad(mat, ((0, 0), (0, 1)), mode="constant")
        mat[idx[gc], idx[pc]] += 1
        used_pred.add(i)
        used_gt.add(j)

    # Unmatched predictions -> background row (FP)
    for i in range(iou.shape[0]):
        if i in used_pred:
            continue
        pc = int(pred_cls[i]) if pred_cls is not None and i < len(pred_cls) else 0
        if pc < 0:
            pc = 0
        if pc not in idx:
            classes.append(pc)
            idx[pc] = len(classes) - 1
            mat = np.pad(mat, ((0, 0), (0, 1)), mode="constant")
        mat[idx[0], idx[pc]] += 1

    # Unmatched GT -> background column (FN)
    for j in range(iou.shape[1]):
        if j in used_gt:
            continue
        gc = int(gt_cls[j]) if gt_cls is not None and j < len(gt_cls) else 0
        if gc < 0:
            gc = 0
        if gc not in idx:
            classes.append(gc)
            idx[gc] = len(classes) - 1
            mat = np.pad(mat, ((0, 1), (0, 0)), mode="constant")
        mat[idx[gc], idx[0]] += 1

    # Derive per-class TP/FP/FN (excluding class 0).
    tp_by_class: dict[int, int] = {}
    fp_by_class: dict[int, int] = {}
    fn_by_class: dict[int, int] = {}
    for c in classes:
        if c == 0:
            continue
        ci = idx[c]
        tp = int(mat[ci, ci])
        fp = int(mat[:, ci].sum() - tp)
        fn = int(mat[ci, :].sum() - tp)
        tp_by_class[c] = tp
        fp_by_class[c] = fp
        fn_by_class[c] = fn

    return mat, classes, tp_by_class, fp_by_class, fn_by_class


def pixel_confusion_matrix_counts(
    gt_cm: np.ndarray,
    pred_cm: np.ndarray,
) -> Tuple[np.ndarray, list[int], dict[int, int], dict[int, int], dict[int, int]]:
    """Pixel-level confusion matrix (rows=GT class, cols=Pred class)."""
    classes = np.union1d(np.unique(gt_cm), np.unique(pred_cm)).astype(np.int64)
    g = gt_cm.ravel().astype(np.int64, copy=False)
    p = pred_cm.ravel().astype(np.int64, copy=False)
    g_idx = np.searchsorted(classes, g)
    p_idx = np.searchsorted(classes, p)
    n = classes.size
    flat = g_idx * n + p_idx
    counts = np.bincount(flat, minlength=n * n)
    mat = counts.reshape(n, n).astype(np.int64, copy=False)
    class_list = [int(c) for c in classes.tolist()]
    idx = {c: i for i, c in enumerate(class_list)}
    tp_by_class: dict[int, int] = {}
    fp_by_class: dict[int, int] = {}
    fn_by_class: dict[int, int] = {}
    for c in class_list:
        if c == 0:
            continue
        ci = idx[c]
        tp = int(mat[ci, ci])
        fp = int(mat[:, ci].sum() - tp)
        fn = int(mat[ci, :].sum() - tp)
        tp_by_class[c] = tp
        fp_by_class[c] = fp
        fn_by_class[c] = fn
    return mat, class_list, tp_by_class, fp_by_class, fn_by_class


def _coerce_image_for_2d_eval(img: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    """Coerce ND image data to a single 2D (or 2D+channels) sample for 2D GT.

    This avoids accidentally sending full Z/T stacks into eval when labels are 2D.
    """
    arr = np.asarray(img)
    h, w = int(gt_shape[0]), int(gt_shape[1])

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # YXC
        if arr.shape[0] == h and arr.shape[1] == w:
            return arr
        # CYX -> YXC
        if arr.shape[1] == h and arr.shape[2] == w and arr.shape[0] <= 4:
            return np.moveaxis(arr, 0, -1)
        # ZYX (or similar): pick first plane matching GT size
        if arr.shape[1] == h and arr.shape[2] == w:
            return arr[0]
        if arr.shape[-2] == h and arr.shape[-1] == w:
            return arr.reshape(-1, h, w)[0]

    # ndim >= 4
    if arr.shape[-2] == h and arr.shape[-1] == w:
        return arr.reshape(-1, h, w)[0]

    if arr.shape[0] == h and arr.shape[1] == w:
        tmp = arr
        while tmp.ndim > 3:
            tmp = tmp[..., 0]
        return tmp

    # Last-resort squeeze/fallback
    tmp = np.squeeze(arr)
    if tmp.ndim > 3 and tmp.shape[-2] == h and tmp.shape[-1] == w:
        return tmp.reshape(-1, h, w)[0]
    if tmp.ndim > 3:
        while tmp.ndim > 3:
            tmp = tmp[0]
    return tmp


def run_model_on_image(
    model,
    img: np.ndarray,
    diameter: float,
    tile: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    out = None
    channel_axis = None
    arr_img = np.asarray(img)
    if arr_img.ndim == 3:
        if arr_img.shape[-1] <= 4:
            channel_axis = -1
        elif arr_img.shape[0] <= 4:
            channel_axis = 0

    normalize_params = dict(cp_models.normalize_default)
    normalize_params["normalize"] = True
    eval_diameter = None if diameter is None or float(diameter) <= 0 else float(diameter)

    try:
        kwargs = dict(GUI_EVAL_DEFAULTS)
        kwargs.update(
            dict(
                diameter=eval_diameter,
                normalize=normalize_params,
                channel_axis=channel_axis,
                z_axis=None,
                progress=None,
            )
        )
        if tile:
            # Optional override: this deviates from strict GUI defaults.
            kwargs["tile"] = True
            kwargs["tile_overlap"] = 0.1
        out = model.eval(img, **kwargs)
    except TypeError:
        kwargs = dict(GUI_EVAL_DEFAULTS)
        kwargs.update(
            dict(
                diameter=eval_diameter,
                normalize=normalize_params,
                channel_axis=channel_axis,
                z_axis=None,
                progress=None,
            )
        )
        if tile:
            kwargs["tile"] = True
            kwargs["tile_overlap"] = 0.1
        out = model.eval(img, **kwargs)

    if isinstance(out, (tuple, list)):
        masks = out[0] if len(out) > 0 else out
        styles = out[2] if len(out) > 2 else None
    else:
        masks = out
        styles = None

    masks = np.asarray(masks)
    if masks.ndim == 3:
        masks = masks.squeeze()
    masks = masks.astype(np.int32, copy=False)

    pred_cm = None
    try:
        if styles is not None:
            arr = np.asarray(styles)
            if arr.ndim >= 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim >= 3 and arr.shape[-1] > 1:
                pred_cm = np.argmax(arr, axis=-1).astype(np.int32)
            elif arr.ndim == 2:
                pred_cm = np.rint(arr).astype(np.int32)
    except Exception:
        pred_cm = None

    return masks, pred_cm


def _coerce_masks_to_2d_int32(masks: np.ndarray) -> np.ndarray:
    """Force model mask output into a plain 2D int32 ndarray."""
    arr = masks
    if isinstance(arr, list) and len(arr) == 1:
        arr = arr[0]
    arr = np.asarray(arr)

    # Some backends can return object arrays wrapping a single ndarray.
    while isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        try:
            arr = np.asarray(arr.item())
        except Exception:
            break

    if arr.dtype == object:
        raise ValueError(f"Unsupported mask dtype=object with shape {arr.shape}")

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim > 3:
        arr = np.squeeze(arr)

    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D masks, got shape {arr.shape}")

    return arr.astype(np.int32, copy=False)


def save_pred_npy(
    image_path: Path,
    frame_id: Optional[str],
    masks: np.ndarray,
    classes_map: Optional[np.ndarray],
    instance_classes: np.ndarray,
    diameter: float,
    pred_dir: Optional[Path] = None,
) -> Path:
    """Save prediction payload in MultiCellPose-compatible *_pred.npy format."""
    base_stem = image_path.stem
    frame_suffix = cp_io.frame_id_to_suffix(frame_id)
    out_dir = pred_dir if pred_dir is not None else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_stem}{frame_suffix}_pred.npy"

    masks_i32 = _coerce_masks_to_2d_int32(masks)
    nmask = int(masks_i32.max()) if masks_i32.size else 0
    classes = np.zeros(nmask + 1, dtype=np.int16)
    if instance_classes is not None:
        lim = min(len(classes), len(instance_classes))
        if lim > 1:
            classes[1:lim] = np.asarray(instance_classes[1:lim], dtype=np.int16)

    dat = {
        "masks": masks_i32,
        "filename": str(image_path),
        "flows": None,
        "ismanual": np.zeros(nmask, dtype=bool),
        "chan_choose": [0, 0],
        "classes": classes,
        "classes_map": None if classes_map is None else np.asarray(classes_map).astype(np.int32, copy=False),
        "class_names": None,
        "class_colors": None,
        "diameter": None if diameter is None or float(diameter) <= 0 else float(diameter),
    }
    try:
        outlines = cp_utils.masks_to_outlines(masks_i32)
        dat["outlines"] = outlines * masks_i32
    except Exception:
        pass

    np.save(str(out_path), dat)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate semantic instance segmentation on labeled test data")
    ap.add_argument("--dir", required=True, help="Directory containing test images and labels")
    ap.add_argument("--model", required=True, help="Model path/id for CellposeModel pretrained_model")
    ap.add_argument("--diameter", type=float, default=0.0, help="Diameter hint (0=auto)")
    ap.add_argument("--tile", action="store_true", help="Enable tiled inference")
    ap.add_argument("--iou", type=float, nargs="+", default=[0.5, 0.75], help="IoU thresholds")
    ap.add_argument("--csv", default=None, help="Optional output CSV for per-image metrics")
    ap.add_argument("--conf-json", default=None, help="Optional output JSON with confusion matrices")
    ap.add_argument(
        "--conf-level",
        choices=("instance", "pixel"),
        default="instance",
        help="Confusion-matrix level for --conf-json output",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    ap.add_argument("--verbose", action="store_true", help="Print per-image progress/timing")
    ap.add_argument(
        "--save-pred",
        action="store_true",
        help="Save per-image predictions to MultiCellPose-compatible *_pred.npy files",
    )
    ap.add_argument(
        "--pred-dir",
        default=None,
        help="Optional output directory for *_pred.npy files (default: alongside each image)",
    )
    args = ap.parse_args()

    folder = Path(args.dir).resolve()
    if not folder.is_dir():
        raise SystemExit(f"ERROR: not a directory: {folder}")
    pred_dir = Path(args.pred_dir).resolve() if args.pred_dir else None

    model = cp_models.CellposeModel(gpu=(not args.cpu), pretrained_model=args.model)
    images = list_base_images(folder)
    if not images:
        raise SystemExit("No base images found in directory.")
    print(f"Found {len(images)} candidate images in {folder}")
    if args.tile:
        print("NOTE: --tile enabled (this adds tile/tile_overlap beyond strict GUI eval defaults).")

    base_headers = [
        "image",
        "n_pred",
        "n_gt",
        "mean_iou_match",
        *[f"F1@{t:.2f}" for t in args.iou],
        "gt_class2_pred_class1",
        "gt_class1_pred_class2",
        "fp_class_mismatch_total",
        "fp_low_iou_unmatched_total",
    ]
    sem_headers = [
        "sem_acc_all",
        "sem_acc_fg",
        "sem_mIoU_fg",
        "sem_n_classes_gt",
    ]
    rows = []
    conf_records = []
    sum_f1 = np.zeros(len(args.iou), dtype=np.float64)
    sum_miou = 0.0
    n_eval = 0
    dataset_class_ids: set[int] = set()

    total_t_load = 0.0
    total_t_coerce = 0.0
    total_t_eval = 0.0
    total_t_metrics = 0.0

    for i, img_path in enumerate(images, start=1):
        any_frame_seen = False
        for frame_id, raw_img in iter_eval_frames(img_path):
            any_frame_seen = True
            label_key = f"{img_path.name}::{frame_id}" if frame_id else img_path.name
            gt = load_gt_masks(img_path, frame_id=frame_id)
            if gt is None:
                if args.verbose:
                    print(f"[{i}/{len(images)}] {label_key}: no labels found, skipping")
                continue

            t0 = time.time()
            if args.verbose:
                print(f"[{i}/{len(images)}] {label_key}: loading image")

            t_load0 = time.time()
            img = np.asarray(raw_img)
            t_load = time.time() - t_load0
            total_t_load += t_load

            t_coerce0 = time.time()
            img = _coerce_image_for_2d_eval(np.asarray(img), gt.shape)
            t_coerce = time.time() - t_coerce0
            total_t_coerce += t_coerce
            if args.verbose:
                print(f"[{i}/{len(images)}] {label_key}: eval on shape {tuple(np.asarray(img).shape)}")

            t_eval0 = time.time()
            pred, pred_cm = run_model_on_image(
                model=model,
                img=img,
                diameter=args.diameter,
                tile=args.tile,
            )
            t_eval = time.time() - t_eval0
            total_t_eval += t_eval
            pred = _coerce_masks_to_2d_int32(pred)

            # For non-semantic models (no predicted class map), treat every
            # detected instance as class 1 for class-aware instance scoring.
            if pred_cm is None:
                pred_cm = np.where(pred > 0, 1, 0).astype(np.int32, copy=False)

            t_metrics0 = time.time()
            gt_cm = load_gt_classes_map(img_path, gt_masks=gt, frame_id=frame_id)
            gt_inst_cls = get_instance_classes(gt, gt_cm)
            pred_inst_cls = get_instance_classes(pred, pred_cm)
            pred_cls_vec, gt_cls_vec = _aligned_instance_class_vectors(
                pred_masks=pred,
                gt_masks=gt,
                pred_inst_classes=pred_inst_cls,
                gt_inst_classes=gt_inst_cls,
            )
            if args.save_pred:
                saved_pred_path = save_pred_npy(
                    image_path=img_path,
                    frame_id=frame_id,
                    masks=pred,
                    classes_map=pred_cm,
                    instance_classes=pred_inst_cls,
                    diameter=args.diameter,
                    pred_dir=pred_dir,
                )
                if args.verbose:
                    print(f"[{i}/{len(images)}] {label_key}: saved {saved_pred_path.name}")
            for c in np.unique(gt_inst_cls):
                c = int(c)
                if c > 0:
                    dataset_class_ids.add(c)

            iou = masks_to_iou(pred, gt)
            max_thr = max(args.iou)
            _, _, _, mean_iou = greedy_match_iou(iou, max_thr, pred_cls_vec, gt_cls_vec)
            fp_cls, fn_cls = greedy_match_iou_class_counts(iou, max_thr, pred_cls_vec, gt_cls_vec)
            fp_mismatch_cls, fp_low_iou_unmatched_cls = greedy_match_fp_reason_counts(
                iou=iou,
                thr=max_thr,
                pred_cls=pred_cls_vec,
                gt_cls=gt_cls_vec,
            )
            conf_mat_inst, conf_classes_inst, tp_cls_csv, _, _ = instance_confusion_matrix_counts(
                iou=iou,
                pred_cls=pred_cls_vec,
                gt_cls=gt_cls_vec,
                thr=max_thr,
            )
            c2_to_c1 = 0
            c1_to_c2 = 0
            try:
                idx_map = {int(c): ix for ix, c in enumerate(conf_classes_inst)}
                if 2 in idx_map and 1 in idx_map:
                    c2_to_c1 = int(conf_mat_inst[idx_map[2], idx_map[1]])
                if 1 in idx_map and 2 in idx_map:
                    c1_to_c2 = int(conf_mat_inst[idx_map[1], idx_map[2]])
            except Exception:
                c2_to_c1 = 0
                c1_to_c2 = 0
            for c in fp_cls.keys():
                if int(c) > 0:
                    dataset_class_ids.add(int(c))
            for c in fn_cls.keys():
                if int(c) > 0:
                    dataset_class_ids.add(int(c))
            for c in tp_cls_csv.keys():
                if int(c) > 0:
                    dataset_class_ids.add(int(c))
            for c in fp_mismatch_cls.keys():
                if int(c) > 0:
                    dataset_class_ids.add(int(c))
            for c in fp_low_iou_unmatched_cls.keys():
                if int(c) > 0:
                    dataset_class_ids.add(int(c))
            f1_list = []
            for thr in args.iou:
                tp, fp, fn, _ = greedy_match_iou(iou, thr, pred_cls_vec, gt_cls_vec)
                denom = (2 * tp + fp + fn)
                f1_list.append((2 * tp) / denom if denom > 0 else 0.0)

            pix_tp_cls: dict[int, int] = {}
            pix_fp_cls: dict[int, int] = {}
            pix_fn_cls: dict[int, int] = {}
            sem_acc_all = ""
            sem_acc_fg = ""
            sem_miou_fg = ""
            sem_ncls = ""
            if gt_cm is not None and pred_cm is not None:
                pc = np.asarray(pred_cm)
                if pc.ndim == 3:
                    pc = pc.squeeze()
                if gt_cm.shape == pc.shape:
                    sem_acc_all = float((gt_cm == pc).mean())
                    fg = gt_cm > 0
                    if int(fg.sum()) > 0:
                        sem_acc_fg = float((gt_cm[fg] == pc[fg]).mean())
                    fg_classes = np.unique(gt_cm)
                    fg_classes = fg_classes[fg_classes > 0]
                    ious = []
                    for c in fg_classes:
                        g = (gt_cm == c)
                        p = (pc == c)
                        inter = np.logical_and(g, p).sum()
                        uni = g.sum() + p.sum() - inter
                        if uni > 0:
                            ious.append(inter / uni)
                    sem_miou_fg = float(np.mean(ious)) if ious else 0.0
                    sem_ncls = int(len(fg_classes))

                    # Always compute pixel-level TP/FP/FN for CSV output.
                    _pix_mat, _pix_classes, pix_tp_cls, pix_fp_cls, pix_fn_cls = pixel_confusion_matrix_counts(
                        gt_cm=gt_cm,
                        pred_cm=pc,
                    )
                    for c in _pix_classes:
                        ci = int(c)
                        if ci > 0:
                            dataset_class_ids.add(ci)

                    if args.conf_level == "instance":
                        conf_mat, conf_classes, tp_cls, fp_cls_cm, fn_cls_cm = instance_confusion_matrix_counts(
                            iou=iou,
                            pred_cls=pred_cls_vec,
                            gt_cls=gt_cls_vec,
                            thr=max_thr,
                        )
                    else:
                        conf_mat, conf_classes, tp_cls, fp_cls_cm, fn_cls_cm = pixel_confusion_matrix_counts(
                            gt_cm=gt_cm,
                            pred_cm=pc,
                        )
                    conf_records.append(
                        {
                            "image": label_key,
                            "level": args.conf_level,
                            "matching_iou_threshold": float(max_thr) if args.conf_level == "instance" else None,
                            "classes": conf_classes,
                            "matrix": np.asarray(conf_mat, dtype=int).tolist(),
                            "tp_by_class": {str(k): int(v) for k, v in tp_cls.items()},
                            "fp_by_class": {str(k): int(v) for k, v in fp_cls_cm.items()},
                            "fn_by_class": {str(k): int(v) for k, v in fn_cls_cm.items()},
                        }
                    )

            rows.append(
                {
                    "image": label_key,
                    "n_pred": int(pred.max()),
                    "n_gt": int(gt.max()),
                    "mean_iou_match": float(mean_iou),
                    **{f"F1@{thr:.2f}": float(v) for thr, v in zip(args.iou, f1_list)},
                    "gt_class2_pred_class1": int(c2_to_c1),
                    "gt_class1_pred_class2": int(c1_to_c2),
                    "fp_class_mismatch_total": int(sum(fp_mismatch_cls.values())),
                    "fp_low_iou_unmatched_total": int(sum(fp_low_iou_unmatched_cls.values())),
                    "tp_counts": {int(k): int(v) for k, v in tp_cls_csv.items()},
                    "fp_counts": {int(k): int(v) for k, v in fp_cls.items()},
                    "fn_counts": {int(k): int(v) for k, v in fn_cls.items()},
                    "fp_mismatch_counts": {int(k): int(v) for k, v in fp_mismatch_cls.items()},
                    "fp_low_iou_unmatched_counts": {int(k): int(v) for k, v in fp_low_iou_unmatched_cls.items()},
                    "pixel_tp_counts": {int(k): int(v) for k, v in pix_tp_cls.items()},
                    "pixel_fp_counts": {int(k): int(v) for k, v in pix_fp_cls.items()},
                    "pixel_fn_counts": {int(k): int(v) for k, v in pix_fn_cls.items()},
                    "sem_acc_all": sem_acc_all,
                    "sem_acc_fg": sem_acc_fg,
                    "sem_mIoU_fg": sem_miou_fg,
                    "sem_n_classes_gt": sem_ncls,
                }
            )
            sum_miou += float(mean_iou)
            sum_f1 += np.asarray(f1_list, dtype=np.float64)
            n_eval += 1
            t_metrics = time.time() - t_metrics0
            total_t_metrics += t_metrics
            if args.verbose:
                dt = time.time() - t0
                print(
                    f"[{i}/{len(images)}] {label_key}: done in {dt:.2f}s "
                    f"(load={t_load:.2f}s, coerce={t_coerce:.2f}s, eval={t_eval:.2f}s, metrics={t_metrics:.2f}s)"
                )

        if not any_frame_seen and args.verbose:
            print(f"[{i}/{len(images)}] {img_path.name}: no readable frames, skipping")

    if n_eval == 0:
        raise SystemExit("No labeled images with valid GT masks found.")

    print(f"\nEvaluation summary (n={n_eval}):")
    print(f"  mean IoU (matched @max thr): {sum_miou / n_eval:.4f}")
    for thr, val in zip(args.iou, (sum_f1 / n_eval).tolist()):
        print(f"  F1@{thr:.2f}: {val:.4f}")
    print(
        "  timing totals: "
        f"load={total_t_load:.2f}s, coerce={total_t_coerce:.2f}s, "
        f"eval={total_t_eval:.2f}s, metrics={total_t_metrics:.2f}s"
    )
    print(
        "  timing/image: "
        f"load={total_t_load / n_eval:.2f}s, coerce={total_t_coerce / n_eval:.2f}s, "
        f"eval={total_t_eval / n_eval:.2f}s, metrics={total_t_metrics / n_eval:.2f}s"
    )

    if args.csv:
        import csv

        class_headers = []
        for c in sorted(dataset_class_ids):
            class_headers.append(f"tp_class_{c}")
            class_headers.append(f"fp_class_{c}")
            class_headers.append(f"fn_class_{c}")
            class_headers.append(f"fp_mismatch_class_{c}")
            class_headers.append(f"fp_low_iou_unmatched_class_{c}")
            class_headers.append(f"pixel_tp_class_{c}")
            class_headers.append(f"pixel_fp_class_{c}")
            class_headers.append(f"pixel_fn_class_{c}")
        headers = [*base_headers, *class_headers, *sem_headers]

        out_rows = []
        for rec in rows:
            row = [
                rec["image"],
                rec["n_pred"],
                rec["n_gt"],
                rec["mean_iou_match"],
                *[rec[f"F1@{thr:.2f}"] for thr in args.iou],
                rec.get("gt_class2_pred_class1", 0),
                rec.get("gt_class1_pred_class2", 0),
                rec.get("fp_class_mismatch_total", 0),
                rec.get("fp_low_iou_unmatched_total", 0),
            ]
            tp_counts = rec.get("tp_counts", {})
            fp_counts = rec.get("fp_counts", {})
            fn_counts = rec.get("fn_counts", {})
            fp_mismatch_counts = rec.get("fp_mismatch_counts", {})
            fp_low_iou_unmatched_counts = rec.get("fp_low_iou_unmatched_counts", {})
            pix_tp_counts = rec.get("pixel_tp_counts", {})
            pix_fp_counts = rec.get("pixel_fp_counts", {})
            pix_fn_counts = rec.get("pixel_fn_counts", {})
            for c in sorted(dataset_class_ids):
                row.append(int(tp_counts.get(c, 0)))
                row.append(int(fp_counts.get(c, 0)))
                row.append(int(fn_counts.get(c, 0)))
                row.append(int(fp_mismatch_counts.get(c, 0)))
                row.append(int(fp_low_iou_unmatched_counts.get(c, 0)))
                row.append(int(pix_tp_counts.get(c, 0)))
                row.append(int(pix_fp_counts.get(c, 0)))
                row.append(int(pix_fn_counts.get(c, 0)))
            row.extend(
                [
                    rec["sem_acc_all"],
                    rec["sem_acc_fg"],
                    rec["sem_mIoU_fg"],
                    rec["sem_n_classes_gt"],
                ]
            )
            out_rows.append(row)

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(out_rows)
        print(f"Saved per-image metrics to {args.csv}")

    if args.conf_json and conf_records:
        with open(args.conf_json, "w", encoding="utf-8") as f:
            json.dump(conf_records, f, indent=2)
        print(f"Saved confusion matrices to {args.conf_json}")


if __name__ == "__main__":
    main()
