"""Run standardized instance-segmentation tests on a labeled dataset.

This evaluates one or more Cellpose/MultiCellPose models against labeled test
images and reports class-1 object detection metrics. All predicted objects are
treated as class 1. Ground-truth class 2 objects are not counted as targets;
predictions overlapping class 2 objects are counted as false detections.

Default example:
    python tools/evaluate_standardized_test_models.py \
        --test-dir X:\\home\\MCP_paper\\test \
        --models cpsam guvpose cpsam_encoder_student_best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".nd2", ".lif", ".nrrd"}
DERIVED_SUBSTRINGS = ("_masks", "_mask", "_classes", "_class", "_flows", "_seg", "_pred")


@dataclass(frozen=True)
class EvalRecord:
    image_path: Path
    frame_id: str | None
    label_key: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default=r"X:\home\MCP_paper\test")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cpsam", "guvpose", "cpsam_encoder_student_best.pt"],
        help="Model names or paths. Names are resolved under --model-dir when present.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(Path.home() / ".cellpose" / "models"),
        help="Directory containing named model files.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-prefix", default=None)
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.75, 0.9])
    parser.add_argument(
        "--class2-iou-threshold",
        type=float,
        default=0.5,
        help="A class-1 prediction with IoU >= this value to any class-2 GT object is forced to FP.",
    )
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--misdetect-class", type=int, default=2)
    parser.add_argument(
        "--ignore-classes",
        action="store_true",
        help=(
            "Class-agnostic evaluation: treat every ground-truth instance as a target object, "
            "ignore class maps/classes, and do not force predictions overlapping class 2 to false positives."
        ),
    )
    parser.add_argument("--diameter", type=float, default=0.0, help="Diameter hint; 0 means model default/auto.")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--flow-threshold", type=float, default=1.0)
    parser.add_argument("--cellprob-threshold", type=float, default=-0.5)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--max-size-fraction", type=float, default=0.4)
    parser.add_argument("--augment", action="store_true", help="Use Cellpose test-time augmentation.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable Cellpose image normalization.")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for all models.")
    parser.add_argument(
        "--student-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force CPU for model names containing 'student'. Enabled by default.",
    )
    parser.add_argument("--save-pred", action="store_true", help="Save predicted masks as .npy arrays.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def is_base_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
        return False
    stem = path.stem.lower()
    return not any(token in stem for token in DERIVED_SUBSTRINGS)


def candidate_images(test_dir: Path) -> list[Path]:
    return sorted(path for path in test_dir.rglob("*") if is_base_image(path))


def discover_records(test_dir: Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for image_path in candidate_images(test_dir):
        for frame_id, _raw_img in iter_eval_frames(image_path):
            gt = load_gt_masks(image_path, frame_id=frame_id)
            if gt is None:
                continue
            label_key = f"{image_path.name}::{frame_id}" if frame_id else image_path.name
            records.append(EvalRecord(image_path=image_path, frame_id=frame_id, label_key=label_key))
    return records


def resolve_model(model_name: str, model_dir: Path) -> Path | str:
    path = Path(model_name).expanduser()
    if path.exists():
        return path.resolve()
    candidate = model_dir / model_name
    if candidate.exists():
        return candidate.resolve()
    return model_name


def model_uses_cpu(model_name: str, force_cpu: bool, student_cpu: bool) -> bool:
    if force_cpu:
        return True
    return student_cpu and "student" in model_name.lower()


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def infer_channel_axis(img: np.ndarray) -> int | None:
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 2, 3, 4):
            return -1
        if arr.shape[0] in (1, 2, 3, 4):
            return 0
    return None


def run_model_on_image_standardized(
    model,
    img: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None]:
    normalize_params = dict(cp_models.normalize_default)
    normalize_params["normalize"] = not args.no_normalize
    eval_diameter = None if args.diameter is None or args.diameter <= 0 else float(args.diameter)
    masks, flows, _styles = model.eval(
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
    cellprob = None
    if isinstance(flows, (list, tuple)) and len(flows) > 2:
        cellprob = np.asarray(flows[2])
    return np.asarray(masks), cellprob


def relabel_subset(mask: np.ndarray, instance_classes: np.ndarray, wanted_class: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for mid in np.unique(mask[mask > 0]).astype(np.int64):
        if mid < len(instance_classes) and int(instance_classes[mid]) == int(wanted_class):
            out[mask == mid] = next_id
            next_id += 1
    return out


def prediction_class2_blocklist(
    pred: np.ndarray,
    gt_class2: np.ndarray,
    threshold: float,
) -> set[int]:
    pred_ids = np.unique(pred[pred > 0])
    if pred_ids.size == 0 or int(gt_class2.max()) == 0:
        return set()
    iou = masks_to_iou(pred, gt_class2)
    blocked: set[int] = set()
    for row_index, pred_id in enumerate(pred_ids.astype(np.int64)):
        if iou.shape[1] > 0 and float(iou[row_index].max()) >= float(threshold):
            blocked.add(int(pred_id))
    return blocked


def remove_prediction_ids(pred: np.ndarray, blocked_ids: set[int]) -> np.ndarray:
    if not blocked_ids:
        return pred
    out = pred.copy()
    for pred_id in blocked_ids:
        out[out == pred_id] = 0
    return relabel_positive(out)


def relabel_positive(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for mid in np.unique(mask[mask > 0]).astype(np.int64):
        out[mask == mid] = next_id
        next_id += 1
    return out


def greedy_counts(iou: np.ndarray, threshold: float) -> tuple[int, int, int, float]:
    if iou.size == 0:
        n_pred, n_gt = iou.shape
        return 0, n_pred, n_gt, 0.0
    candidates = [
        (float(iou[i, j]), i, j)
        for i in range(iou.shape[0])
        for j in range(iou.shape[1])
        if float(iou[i, j]) >= float(threshold)
    ]
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched_ious: list[float] = []
    for val, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matched_ious.append(val)
    tp = len(matched_ious)
    fp = iou.shape[0] - tp
    fn = iou.shape[1] - tp
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return tp, fp, fn, mean_iou


def ap_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom else 1.0


def precision_from_counts(tp: int, fp: int) -> float:
    denom = tp + fp
    return float(tp / denom) if denom else 1.0


def recall_from_counts(tp: int, fn: int) -> float:
    denom = tp + fn
    return float(tp / denom) if denom else 1.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = precision_from_counts(tp, fp)
    recall = recall_from_counts(tp, fn)
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom else 0.0


def evaluate_one_record(
    model,
    record: EvalRecord,
    test_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray | None]:
    frame_iter = iter_eval_frames(record.image_path)
    raw_img = None
    for frame_id, arr in frame_iter:
        if frame_id == record.frame_id:
            raw_img = arr
            break
    if raw_img is None:
        raise ValueError(f"Could not reload frame {record.frame_id!r} from {record.image_path}")

    gt_masks = load_gt_masks(record.image_path, frame_id=record.frame_id)
    if gt_masks is None:
        raise ValueError("No ground-truth masks found")
    gt_masks = np.asarray(gt_masks, dtype=np.int32)
    if gt_masks.ndim == 3:
        gt_masks = np.squeeze(gt_masks)
    if args.ignore_classes:
        gt_target = relabel_positive(gt_masks)
        gt_misdetect = np.zeros_like(gt_target, dtype=np.int32)
    else:
        class_map = load_gt_classes_map(record.image_path, gt_masks, frame_id=record.frame_id)
        gt_classes = get_instance_classes(gt_masks, class_map)
        if class_map is None:
            gt_classes[1:] = int(args.target_class)

        gt_target = relabel_subset(gt_masks, gt_classes, args.target_class)
        gt_misdetect = relabel_subset(gt_masks, gt_classes, args.misdetect_class)
    image = _coerce_image_for_2d_eval(np.asarray(raw_img), gt_masks.shape)
    pred, _pred_cm = run_model_on_image_standardized(model, image, args)
    pred = np.asarray(pred, dtype=np.int32)
    if pred.ndim == 3:
        pred = np.squeeze(pred)
    if pred.shape != gt_masks.shape:
        raise ValueError(f"prediction shape {pred.shape} != GT shape {gt_masks.shape}")

    blocked_ids = set() if args.ignore_classes else prediction_class2_blocklist(pred, gt_misdetect, args.class2_iou_threshold)
    pred_for_target = remove_prediction_ids(pred, blocked_ids)
    iou = masks_to_iou(pred_for_target, gt_target)

    row: dict[str, object] = {
        "image": rel_path(record.image_path, test_dir),
        "frame_id": record.frame_id or "",
        "label_key": record.label_key,
        "n_gt_all": int(max(0, len(np.unique(gt_masks)) - 1)),
        "n_gt_class1": int(max(0, len(np.unique(gt_target)) - 1)),
        "n_gt_class2": int(max(0, len(np.unique(gt_misdetect)) - 1)),
        "n_pred_raw": int(max(0, len(np.unique(pred)) - 1)),
        "n_pred_eval_after_class2_block": int(max(0, len(np.unique(pred_for_target)) - 1)),
        "pred_overlap_class2_fp": int(len(blocked_ids)),
    }
    matched_ious = []
    for threshold in args.iou_thresholds:
        tp, fp_without_blocked, fn, mean_iou = greedy_counts(iou, float(threshold))
        fp = fp_without_blocked + len(blocked_ids)
        tag = threshold_tag(float(threshold))
        row[f"tp_{tag}"] = int(tp)
        row[f"fp_{tag}"] = int(fp)
        row[f"fn_{tag}"] = int(fn)
        row[f"ap_{tag}"] = ap_from_counts(tp, fp, fn)
        row[f"precision_{tag}"] = precision_from_counts(tp, fp)
        row[f"recall_{tag}"] = recall_from_counts(tp, fn)
        row[f"f1_{tag}"] = f1_from_counts(tp, fp, fn)
        row[f"mean_matched_iou_{tag}"] = float(mean_iou)
        matched_ious.append(float(mean_iou))
    row["mean_matched_iou"] = float(np.mean(matched_ious)) if matched_ious else 0.0
    row["mean_ap"] = float(np.mean([row[f"ap_{threshold_tag(float(t))}"] for t in args.iou_thresholds]))
    return row, pred


def summarize_rows(model_name: str, model_ref: str, rows: list[dict], args: argparse.Namespace) -> dict:
    summary: dict[str, object] = {
        "model_name": model_name,
        "model_ref": model_ref,
        "n_images": len(rows),
        "n_gt_class1": int(sum(int(row["n_gt_class1"]) for row in rows)),
        "n_gt_class2": int(sum(int(row["n_gt_class2"]) for row in rows)),
        "n_pred_raw": int(sum(int(row["n_pred_raw"]) for row in rows)),
        "pred_overlap_class2_fp": int(sum(int(row["pred_overlap_class2_fp"]) for row in rows)),
    }
    mean_ap_values = []
    for threshold in args.iou_thresholds:
        tag = threshold_tag(float(threshold))
        tp = int(sum(int(row[f"tp_{tag}"]) for row in rows))
        fp = int(sum(int(row[f"fp_{tag}"]) for row in rows))
        fn = int(sum(int(row[f"fn_{tag}"]) for row in rows))
        ap = ap_from_counts(tp, fp, fn)
        mean_ap_values.append(ap)
        summary[f"tp_{tag}"] = tp
        summary[f"fp_{tag}"] = fp
        summary[f"fn_{tag}"] = fn
        summary[f"ap_{tag}"] = ap
        summary[f"precision_{tag}"] = precision_from_counts(tp, fp)
        summary[f"recall_{tag}"] = recall_from_counts(tp, fn)
        summary[f"f1_{tag}"] = f1_from_counts(tp, fp, fn)
        summary[f"mean_image_ap_{tag}"] = float(np.mean([float(row[f"ap_{tag}"]) for row in rows])) if rows else None
    summary["map"] = float(np.mean(mean_ap_values)) if mean_ap_values else None
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    test_dir = Path(args.test_dir).resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else test_dir / "standardized_model_evaluation"
    prefix = args.results_prefix or time.strftime("standardized_eval_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(test_dir)
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise ValueError(f"No labeled test records found under {test_dir}")
    print(f"discovered {len(records)} labeled test records under {test_dir}")

    all_rows: list[dict] = []
    summaries: list[dict] = []
    failures: list[dict] = []
    for model_name in args.models:
        model_ref = resolve_model(model_name, model_dir)
        use_cpu = model_uses_cpu(model_name, args.cpu, args.student_cpu)
        print(f"evaluating {model_name} -> {model_ref} on {'CPU' if use_cpu else 'GPU if available'}")
        model = cp_models.CellposeModel(gpu=(not use_cpu) and torch.cuda.is_available(), pretrained_model=str(model_ref))
        model_rows: list[dict] = []
        pred_dir = output_dir / f"{prefix}_predictions" / model_name.replace("\\", "_").replace("/", "_") if args.save_pred else None
        if pred_dir is not None:
            pred_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(records, start=1):
            try:
                row, pred = evaluate_one_record(model, record, test_dir, args)
                row = {"model": model_name, "model_ref": str(model_ref), **row}
                model_rows.append(row)
                all_rows.append(row)
                if pred_dir is not None and pred is not None:
                    safe = rel_path(record.image_path, test_dir).replace("\\", "__").replace("/", "__")
                    suffix = f"__{record.frame_id}" if record.frame_id else ""
                    np.save(pred_dir / f"{Path(safe).stem}{suffix}_pred_masks.npy", pred.astype(np.int32, copy=False))
            except Exception as exc:
                failures.append({
                    "model": model_name,
                    "image": str(record.image_path),
                    "frame_id": record.frame_id or "",
                    "error": str(exc),
                })
                if args.verbose:
                    print(f"failed {model_name} {record.label_key}: {exc}")
            if args.verbose or index % 25 == 0:
                print(f"  {model_name}: {index}/{len(records)} records")
        summaries.append(summarize_rows(model_name, str(model_ref), model_rows, args))

    per_image_csv = output_dir / f"{prefix}_per_image.csv"
    summary_csv = output_dir / f"{prefix}_summary.csv"
    summary_json = output_dir / f"{prefix}_summary.json"
    write_csv(per_image_csv, all_rows)
    write_csv(summary_csv, summaries)
    payload = {
        "test_dir": str(test_dir),
        "model_dir": str(model_dir),
        "iou_thresholds": args.iou_thresholds,
        "target_class": args.target_class,
        "misdetect_class": args.misdetect_class,
        "ignore_classes": bool(args.ignore_classes),
        "class2_iou_threshold": args.class2_iou_threshold,
        "n_records": len(records),
        "summaries": summaries,
        "failures": failures,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote per-image metrics: {per_image_csv}")
    print(f"wrote summary CSV: {summary_csv}")
    print(f"wrote summary JSON: {summary_json}")
    print(json.dumps(summaries, indent=2))
    if failures:
        print(f"warning: {len(failures)} evaluation failures; see summary JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
