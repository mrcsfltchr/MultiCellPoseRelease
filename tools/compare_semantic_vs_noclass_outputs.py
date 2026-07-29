"""Compare semantic and non-semantic model outputs on the MCP_paper test set.

The non-semantic model is treated exactly as a 3-output CellposeSAM-style model:
every predicted object is assigned class 1. The semantic model uses the class map
returned by model.eval when available. Class-wise metrics are computed at the
instance level with one-to-one greedy IoU matching.

Example:
    python tools/compare_semantic_vs_noclass_outputs.py ^
        --test-dir X:\\home\\MCP_paper\\test ^
        --semantic-model X:\\home\\MCP_paper\\semantic_full_balanced_training\\guvpose_semantic_full_balanced ^
        --noclass-model X:\\home\\MCP_paper\\memory_replay_scaling_ood500_total2000\\replay01500\\guvpose_replay_ood500_replay01500 ^
        --output-dir X:\\home\\MCP_paper\\semantic_vs_noclass_output_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cellpose import models as cp_models
from scripts.eval_semantic_inst_seg import (
    _coerce_image_for_2d_eval,
    get_instance_classes,
    iter_eval_frames,
    load_gt_classes_map,
    load_gt_masks,
    masks_to_iou,
)
from tools.evaluate_standardized_test_models import discover_records, infer_channel_axis, rel_path


CLASS_COLORS = {
    0: (0.25, 0.25, 0.25),
    1: (0.05, 0.62, 0.46),
    2: (0.85, 0.37, 0.00),
    3: (0.45, 0.41, 0.69),
    4: (0.80, 0.47, 0.65),
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_path: str
    force_class_one: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default=r"X:\home\MCP_paper\test")
    parser.add_argument(
        "--semantic-model",
        default=r"X:\home\MCP_paper\semantic_full_balanced_training\guvpose_semantic_full_balanced",
    )
    parser.add_argument(
        "--noclass-model",
        default=(
            r"X:\home\MCP_paper\memory_replay_scaling_ood500_total2000"
            r"\replay01500\guvpose_replay_ood500_replay01500"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=r"X:\home\MCP_paper\semantic_vs_noclass_output_comparison",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Evaluate at most this many records. 0 means evaluate all records.",
    )
    parser.add_argument(
        "--max-visual-records",
        type=int,
        default=20,
        help="Save side-by-side overlay PNGs for at most this many evaluated records.",
    )
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--diameter", type=float, default=0.0)
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--flow-threshold", type=float, default=1.0)
    parser.add_argument("--cellprob-threshold", type=float, default=-0.5)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--max-size-fraction", type=float, default=0.4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return float(tp / denom) if denom else float("nan")


def recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return float(tp / denom) if denom else float("nan")


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return float(2 * p * r / (p + r)) if np.isfinite(p + r) and (p + r) > 0 else float("nan")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def make_model(model_ref: str, cpu: bool):
    resolved = Path(model_ref)
    model_arg = str(resolved) if resolved.exists() else model_ref
    gpu = bool(torch.cuda.is_available() and not cpu)
    return cp_models.CellposeModel(gpu=gpu, pretrained_model=model_arg)


def parse_semantic_class_map(styles, shape: tuple[int, int]) -> np.ndarray | None:
    if styles is None:
        return None
    try:
        arr = np.asarray(styles)
    except Exception:
        return None
    if arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[:2] == shape and arr.shape[-1] > 1:
        return np.argmax(arr, axis=-1).astype(np.int32)
    if arr.ndim == 3 and arr.shape[1:] == shape and arr.shape[0] > 1:
        return np.argmax(arr, axis=0).astype(np.int32)
    if arr.ndim == 2:
        cm = np.rint(arr).astype(np.int32)
        if cm.shape == shape:
            return cm
    try:
        cm = _coerce_image_for_2d_eval(arr, shape)
    except Exception:
        return None
    if cm.shape == shape:
        return np.rint(cm).astype(np.int32)
    return None


def run_model_on_image(model, img: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None]:
    normalize_params = dict(cp_models.normalize_default)
    normalize_params["normalize"] = not args.no_normalize
    eval_diameter = None if args.diameter is None or args.diameter <= 0 else float(args.diameter)
    masks, _flows, styles = model.eval(
        np.asarray(img),
        diameter=eval_diameter,
        normalize=normalize_params,
        channel_axis=infer_channel_axis(np.asarray(img)),
        z_axis=None,
        augment=bool(args.augment or args.tile),
        batch_size=int(args.batch_size),
        bsize=int(args.bsize),
        tile_overlap=float(args.tile_overlap),
        flow_threshold=float(args.flow_threshold),
        cellprob_threshold=float(args.cellprob_threshold),
        min_size=int(args.min_size),
        max_size_fraction=float(args.max_size_fraction),
        progress=None,
    )
    pred = np.asarray(masks)
    if pred.ndim == 3:
        pred = np.squeeze(pred)
    pred = pred.astype(np.int32, copy=False)
    return pred, parse_semantic_class_map(styles, pred.shape)


def instance_class_vector(mask: np.ndarray, class_map: np.ndarray | None) -> np.ndarray:
    ids = np.unique(mask[mask > 0]).astype(np.int64)
    classes_by_id = get_instance_classes(mask, class_map)
    out = np.zeros(len(ids), dtype=np.int32)
    for i, mid in enumerate(ids):
        if 0 <= int(mid) < len(classes_by_id):
            out[i] = int(classes_by_id[int(mid)])
    return out


def class_counts(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_classes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
) -> tuple[dict[int, dict[str, int]], dict[tuple[int, int], int]]:
    iou = masks_to_iou(pred, gt)
    class_ids = sorted({int(c) for c in pred_classes if c > 0} | {int(c) for c in gt_classes if c > 0})
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_ids}
    confusion: dict[tuple[int, int], int] = {}

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    pairs = [(float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    pairs.sort(key=lambda item: item[0], reverse=True)
    for val, pi, gi in pairs:
        if val < iou_threshold:
            break
        if pi in used_pred or gi in used_gt:
            continue
        pc = int(pred_classes[pi]) if pi < len(pred_classes) else 0
        gc = int(gt_classes[gi]) if gi < len(gt_classes) else 0
        confusion[(gc, pc)] = confusion.get((gc, pc), 0) + 1
        if pc == gc and pc > 0:
            counts.setdefault(pc, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        else:
            if pc > 0:
                counts.setdefault(pc, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            if gc > 0:
                counts.setdefault(gc, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
        used_pred.add(pi)
        used_gt.add(gi)

    for pi, pc_raw in enumerate(pred_classes):
        if pi not in used_pred and int(pc_raw) > 0:
            pc = int(pc_raw)
            counts.setdefault(pc, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            confusion[(0, pc)] = confusion.get((0, pc), 0) + 1
    for gi, gc_raw in enumerate(gt_classes):
        if gi not in used_gt and int(gc_raw) > 0:
            gc = int(gc_raw)
            counts.setdefault(gc, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            confusion[(gc, 0)] = confusion.get((gc, 0), 0) + 1
    return counts, confusion


def display_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr[np.isfinite(arr)], [1, 99]) if np.any(np.isfinite(arr)) else (0.0, 1.0)
    if hi <= lo:
        return np.zeros(arr.shape[:2] + (3,), dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    fg = m > 0
    b = np.zeros_like(fg, dtype=bool)
    b[:-1, :] |= fg[:-1, :] & (m[:-1, :] != m[1:, :])
    b[1:, :] |= fg[1:, :] & (m[1:, :] != m[:-1, :])
    b[:, :-1] |= fg[:, :-1] & (m[:, :-1] != m[:, 1:])
    b[:, 1:] |= fg[:, 1:] & (m[:, 1:] != m[:, :-1])
    return b


def overlay_instances(base: np.ndarray, mask: np.ndarray, inst_classes: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = base.copy()
    for mid in np.unique(mask[mask > 0]).astype(np.int64):
        cls = int(inst_classes[mid]) if 0 <= int(mid) < len(inst_classes) else 0
        color = np.asarray(CLASS_COLORS.get(cls, (0.5, 0.5, 0.5)), dtype=np.float32)
        region = mask == mid
        out[region] = (1.0 - alpha) * out[region] + alpha * color
    boundary = mask_boundaries(mask)
    out[boundary] = 1.0
    return np.clip(out, 0.0, 1.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "comparison_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(test_dir)
    if not records:
        raise ValueError(f"No labeled records found under {test_dir}")
    if args.max_records and args.max_records > 0 and args.max_records < len(records):
        rng = random.Random(args.sample_seed)
        records = sorted(rng.sample(records, args.max_records), key=lambda r: (str(r.image_path), r.frame_id or ""))

    specs = [
        ModelSpec("semantic", args.semantic_model, force_class_one=False),
        ModelSpec("noclass_all_class1", args.noclass_model, force_class_one=True),
    ]
    models = {spec.label: make_model(spec.model_path, bool(args.cpu)) for spec in specs}

    per_image_rows: list[dict] = []
    confusion_rows: list[dict] = []
    aggregate: dict[tuple[str, int], dict[str, int]] = {}
    visual_payloads: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for idx, record in enumerate(records, start=1):
        raw_img = next(img for fid, img in iter_eval_frames(record.image_path) if fid == record.frame_id)
        gt = load_gt_masks(record.image_path, frame_id=record.frame_id)
        if gt is None:
            continue
        gt = np.asarray(gt).astype(np.int32, copy=False)
        if gt.ndim == 3:
            gt = np.squeeze(gt)
        image = _coerce_image_for_2d_eval(raw_img, gt.shape)
        gt_cm = load_gt_classes_map(record.image_path, gt, frame_id=record.frame_id)
        if gt_cm is None:
            gt_cm = np.where(gt > 0, 1, 0).astype(np.int32, copy=False)
        gt_inst_classes_by_id = get_instance_classes(gt, gt_cm)
        gt_vec = instance_class_vector(gt, gt_cm)
        record_key = f"{rel_path(record.image_path, test_dir)}::{record.frame_id or 'frame0'}"
        print(f"[{idx}/{len(records)}] {record_key}", flush=True)

        if idx <= int(args.max_visual_records):
            visual_payloads[record_key] = {"GT": (gt, gt_inst_classes_by_id)}

        for spec in specs:
            pred, pred_cm = run_model_on_image(models[spec.label], image, args)
            if pred.shape != gt.shape:
                pred = _coerce_image_for_2d_eval(pred, gt.shape).astype(np.int32, copy=False)
            if spec.force_class_one or pred_cm is None:
                pred_cm = np.where(pred > 0, 1, 0).astype(np.int32, copy=False)
            pred_inst_classes_by_id = get_instance_classes(pred, pred_cm)
            pred_vec = instance_class_vector(pred, pred_cm)
            counts, conf = class_counts(pred, gt, pred_vec, gt_vec, float(args.iou_threshold))

            all_classes = sorted(set(counts) | {int(c) for c in gt_vec if c > 0} | {int(c) for c in pred_vec if c > 0})
            for cls in all_classes:
                c = counts.setdefault(int(cls), {"tp": 0, "fp": 0, "fn": 0})
                aggregate.setdefault((spec.label, int(cls)), {"tp": 0, "fp": 0, "fn": 0})
                for key in ("tp", "fp", "fn"):
                    aggregate[(spec.label, int(cls))][key] += int(c[key])
                per_image_rows.append(
                    {
                        "model": spec.label,
                        "image": rel_path(record.image_path, test_dir),
                        "frame_id": record.frame_id or "",
                        "class_id": int(cls),
                        "tp": int(c["tp"]),
                        "fp": int(c["fp"]),
                        "fn": int(c["fn"]),
                        "precision": precision(int(c["tp"]), int(c["fp"])),
                        "recall": recall(int(c["tp"]), int(c["fn"])),
                        "f1": f1_score(int(c["tp"]), int(c["fp"]), int(c["fn"])),
                        "n_gt_instances_class": int(np.sum(gt_vec == int(cls))),
                        "n_pred_instances_class": int(np.sum(pred_vec == int(cls))),
                    }
                )
            for (gt_cls, pred_cls), n in sorted(conf.items()):
                confusion_rows.append(
                    {
                        "model": spec.label,
                        "image": rel_path(record.image_path, test_dir),
                        "frame_id": record.frame_id or "",
                        "gt_class": int(gt_cls),
                        "pred_class": int(pred_cls),
                        "count": int(n),
                    }
                )
            if idx <= int(args.max_visual_records):
                visual_payloads[record_key][spec.label] = (pred, pred_inst_classes_by_id)

        if idx <= int(args.max_visual_records):
            base = display_image(image)
            panels = [
                ("Raw", base),
                ("GT", overlay_instances(base, *visual_payloads[record_key]["GT"])),
                ("Semantic", overlay_instances(base, *visual_payloads[record_key]["semantic"])),
                ("No-class, all class 1", overlay_instances(base, *visual_payloads[record_key]["noclass_all_class1"])),
            ]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
            for ax, (title, panel) in zip(axes, panels):
                ax.imshow(panel)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(record_key)
            fig.savefig(vis_dir / f"{idx:03d}_{safe_name(record_key)}.png", dpi=180)
            plt.close(fig)

    summary_rows = []
    for (model_label, cls), c in sorted(aggregate.items()):
        tp, fp, fn = int(c["tp"]), int(c["fp"]), int(c["fn"])
        summary_rows.append(
            {
                "model": model_label,
                "class_id": int(cls),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision(tp, fp),
                "recall": recall(tp, fn),
                "f1": f1_score(tp, fp, fn),
            }
        )

    write_csv(out_dir / "per_image_class_metrics.csv", per_image_rows)
    write_csv(out_dir / "summary_by_model_class.csv", summary_rows)
    write_csv(out_dir / "instance_confusion_by_model_image.csv", confusion_rows)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"\nWrote {out_dir / 'summary_by_model_class.csv'}", flush=True)
    print(f"Wrote visual comparisons to {vis_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
