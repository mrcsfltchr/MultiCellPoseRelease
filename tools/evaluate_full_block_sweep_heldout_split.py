"""Evaluate full-diverse encoder-block sweep models on their held-out split.

This uses the same instance matching and summary definitions as
``evaluate_standardized_test_models.py``, but loads records from the persistent
train/val/test split manifest created by ``train_cpsam_finetune_balanced.py``.

Default example:
    python tools/evaluate_full_block_sweep_heldout_split.py \
        --sweep-root X:\\home\\FoundationTrain\\cpsam_finetune_block_sweep \
        --path-map /rds/general/user/mfletch1/home=X:\\home
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
from guv_app.services.image_service import ImageService
from scripts.eval_semantic_inst_seg import (
    _coerce_image_for_2d_eval,
    get_instance_classes,
    load_gt_classes_map,
    masks_to_iou,
)
from tools.evaluate_standardized_test_models import (
    ap_from_counts,
    f1_from_counts,
    greedy_counts,
    infer_channel_axis,
    prediction_class2_blocklist,
    precision_from_counts,
    recall_from_counts,
    relabel_positive,
    relabel_subset,
    remove_prediction_ids,
    resolve_model,
    run_model_on_image_standardized,
    summarize_rows,
    threshold_tag,
)
from tools.train_cpsam_finetune_balanced import (
    load_image_ref,
    load_mask,
    load_split_manifest,
    records_for_split,
)


DEFAULT_BLOCKS = ("09", "12", "15", "18", "full")


@dataclass(frozen=True)
class HeldoutRecord:
    image_path: Path
    label_path: Path
    frame_id: str | None
    source_group: str
    original_image: str
    original_label: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default=r"X:\home\FoundationTrain\cpsam_finetune_block_sweep")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit model paths/names.")
    parser.add_argument("--model-prefix", default="guvpose_full_blocks")
    parser.add_argument("--blocks", nargs="*", default=list(DEFAULT_BLOCKS))
    parser.add_argument("--model-dir", default=str(Path.home() / ".cellpose" / "models"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-prefix", default=None)
    parser.add_argument(
        "--path-map",
        nargs="*",
        default=[r"/rds/general/user/mfletch1/home=X:\home"],
        help="Optional path prefix maps, e.g. /rds/general/user/name/home=X:\\home. Applied only when original path is missing.",
    )
    parser.add_argument("--npz-mask-channel", default="last")
    parser.add_argument("--npz-cache-dir", default=None)
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.75, 0.9])
    parser.add_argument("--class2-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--misdetect-class", type=int, default=2)
    parser.add_argument("--ignore-classes", action="store_true")
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
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--student-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-pred", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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


def block_sort_key(block: str) -> tuple[int, int]:
    text = str(block).lower()
    if text == "full":
        return (1, 999)
    return (0, int(text))


def discover_models(sweep_root: Path, model_prefix: str, blocks: Sequence[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for block in blocks:
        block_label = "full" if str(block).lower() == "full" else f"{int(block):02d}"
        model_name = f"{model_prefix}{block_label}"
        candidate = sweep_root / f"blocks{block_label}" / model_name
        if candidate.exists():
            out.append((model_name, str(candidate)))
        else:
            out.append((model_name, model_name))
    out.sort(key=lambda item: block_sort_key(item[0].replace(model_prefix, "")))
    return out


def limit_records(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    return records[:limit]


def load_records(args: argparse.Namespace) -> list[HeldoutRecord]:
    sweep_root = Path(args.sweep_root)
    manifest_path = Path(args.split_manifest) if args.split_manifest else sweep_root / "shared_full_dataset_splits.json"
    manifest = load_split_manifest(manifest_path)
    raw_records = limit_records(records_for_split(manifest.records, args.split), args.max_records, args.seed)
    path_maps = parse_path_maps(args.path_map)
    out: list[HeldoutRecord] = []
    for row in raw_records:
        frame_id = row.get("frame_id") or None
        out.append(
            HeldoutRecord(
                image_path=map_path(row["image"], path_maps),
                label_path=map_path(row["label"], path_maps),
                frame_id=frame_id,
                source_group=row.get("source_group", ""),
                original_image=row["image"],
                original_label=row["label"],
            )
        )
    return out


def rel_path_or_original(path: Path, root: Path, original: str) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return original


def model_uses_cpu(model_name: str, force_cpu: bool, student_cpu: bool) -> bool:
    return bool(force_cpu or (student_cpu and "student" in model_name.lower()))


def evaluate_one_record(
    model,
    record: HeldoutRecord,
    image_service: ImageService,
    args: argparse.Namespace,
    display_root: Path,
) -> tuple[dict, np.ndarray | None]:
    raw_img = load_image_ref(
        image_service,
        str(record.image_path),
        record.frame_id,
        npz_cache_dir=args.npz_cache_dir,
    )
    gt_masks = load_mask(
        record.label_path,
        frame_id=record.frame_id,
        npz_mask_channel=args.npz_mask_channel,
        npz_cache_dir=args.npz_cache_dir,
    )
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
        "image": rel_path_or_original(record.image_path, display_root, record.original_image),
        "frame_id": record.frame_id or "",
        "label_key": f"{Path(record.original_image).name}::{record.frame_id}" if record.frame_id else Path(record.original_image).name,
        "label": str(record.label_path),
        "source_group": record.source_group,
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
    sweep_root = Path(args.sweep_root)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_root / "heldout_split_standardized_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_prefix or time.strftime("heldout_split_eval_%Y%m%d_%H%M%S")
    model_dir = Path(args.model_dir).expanduser().resolve()

    records = load_records(args)
    if not records:
        raise ValueError(f"No records found for split={args.split!r}")
    print(f"loaded {len(records)} held-out {args.split} records")

    if args.models:
        model_specs = [(Path(model).name, model) for model in args.models]
    else:
        model_specs = discover_models(sweep_root, args.model_prefix, args.blocks)
    print("models:")
    for name, ref in model_specs:
        print(f"  {name}: {ref}")
    if args.dry_run:
        print("dry run complete; no models were loaded.")
        for record in records[:5]:
            print(f"  record image={record.image_path} frame={record.frame_id or ''} label={record.label_path}")
        return 0

    display_root = sweep_root.parent
    image_service = ImageService()
    all_rows: list[dict] = []
    summaries: list[dict] = []
    failures: list[dict] = []

    for model_name, model_ref_arg in model_specs:
        model_ref = resolve_model(model_ref_arg, model_dir)
        use_cpu = model_uses_cpu(model_name, args.cpu, args.student_cpu)
        print(f"evaluating {model_name} -> {model_ref} on {'CPU' if use_cpu else 'GPU if available'}")
        model = cp_models.CellposeModel(gpu=(not use_cpu) and torch.cuda.is_available(), pretrained_model=str(model_ref))
        model_rows: list[dict] = []
        pred_dir = output_dir / f"{prefix}_predictions" / model_name.replace("\\", "_").replace("/", "_") if args.save_pred else None
        if pred_dir is not None:
            pred_dir.mkdir(parents=True, exist_ok=True)

        for index, record in enumerate(records, start=1):
            try:
                row, pred = evaluate_one_record(model, record, image_service, args, display_root)
                row = {"model": model_name, "model_ref": str(model_ref), **row}
                model_rows.append(row)
                all_rows.append(row)
                if pred_dir is not None and pred is not None:
                    safe = str(row["image"]).replace("\\", "__").replace("/", "__").replace(":", "_")
                    suffix = f"__{record.frame_id}" if record.frame_id else ""
                    np.save(pred_dir / f"{Path(safe).stem}{suffix}_pred_masks.npy", pred.astype(np.int32, copy=False))
            except Exception as exc:
                failures.append({
                    "model": model_name,
                    "image": str(record.image_path),
                    "frame_id": record.frame_id or "",
                    "label": str(record.label_path),
                    "error": str(exc),
                })
                if args.verbose:
                    print(f"failed {model_name} {record.original_image}::{record.frame_id or ''}: {exc}")
            if args.verbose or index % 50 == 0:
                print(f"  {model_name}: {index}/{len(records)} records")
        summaries.append(summarize_rows(model_name, str(model_ref), model_rows, args))

    per_image_csv = output_dir / f"{prefix}_per_image.csv"
    summary_csv = output_dir / f"{prefix}_summary.csv"
    summary_json = output_dir / f"{prefix}_summary.json"
    write_csv(per_image_csv, all_rows)
    write_csv(summary_csv, summaries)
    payload = {
        "sweep_root": str(sweep_root),
        "split_manifest": str(Path(args.split_manifest) if args.split_manifest else sweep_root / "shared_full_dataset_splits.json"),
        "split": args.split,
        "model_dir": str(model_dir),
        "iou_thresholds": args.iou_thresholds,
        "target_class": args.target_class,
        "misdetect_class": args.misdetect_class,
        "ignore_classes": bool(args.ignore_classes),
        "class2_iou_threshold": args.class2_iou_threshold,
        "n_records": len(records),
        "path_map": args.path_map,
        "summaries": summaries,
        "failures": failures,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote per-image metrics: {per_image_csv}")
    print(f"wrote summary CSV: {summary_csv}")
    print(f"wrote summary JSON: {summary_json}")
    if failures:
        print(f"warning: {len(failures)} evaluation failures; see summary JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
