"""Evaluate a fine-tuned CPSAM model on a held-out split manifest.

This intentionally runs outside training so the manifest test split remains
untouched until final model selection is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from cellpose import metrics
from cellpose.models import CellposeModel
from guv_app.services.image_service import ImageService
from tools.train_cpsam_finetune_balanced import (
    channel_view_specs,
    load_image_ref,
    load_mask,
    load_split_manifest,
    make_three_channel_view,
    records_for_split,
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="Path to the trained CPSAM model.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory. Used to find training_result.json and the default split manifest.",
    )
    parser.add_argument("--split-manifest", default=None, help="Path to cpsam_finetune_splits.json.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--results-prefix", default=None)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8, help="Tile batch size used during Cellpose eval.")
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--npz-mask-channel", default="last")
    parser.add_argument("--npz-cache-dir", default=None)
    parser.add_argument(
        "--channel-sampling-mode",
        default="single-and-all",
        choices=("single-and-all", "none"),
    )
    parser.add_argument("--max-all-channel-combos", type=int, default=2)
    parser.add_argument("--ap-thresholds", nargs="+", type=float, default=(0.5, 0.75, 0.9))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.model_path is None and output_dir is not None:
        result_path = output_dir / "training_result.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            args.model_path = payload.get("model_path")
    if args.split_manifest is None and output_dir is not None:
        args.split_manifest = str(output_dir / "cpsam_finetune_splits.json")
    if not args.model_path:
        parser.error("Provide --model-path or --output-dir containing training_result.json")
    if not args.split_manifest:
        parser.error("Provide --split-manifest or --output-dir")
    return args


def _limit_records(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    return records[:limit]


def _safe_mean(values: Sequence[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _label_overlap(masks_true: np.ndarray, masks_pred: np.ndarray) -> np.ndarray:
    true = masks_true.astype(np.int64, copy=False).ravel()
    pred = masks_pred.astype(np.int64, copy=False).ravel()
    n_pred = int(pred.max()) + 1 if pred.size else 1
    n_true = int(true.max()) + 1 if true.size else 1
    encoded = true * n_pred + pred
    overlap = np.bincount(encoded, minlength=n_true * n_pred)
    return overlap.reshape((n_true, n_pred))


def aggregated_jaccard_index_one(masks_true: np.ndarray, masks_pred: np.ndarray) -> float:
    masks_true = masks_true.astype(np.int32, copy=False)
    masks_pred = masks_pred.astype(np.int32, copy=False)
    union = int(np.logical_or(masks_true > 0, masks_pred > 0).sum())
    if union == 0:
        return 1.0
    if int(masks_true.max()) == 0 or int(masks_pred.max()) == 0:
        return 0.0
    _iout, preds = metrics.mask_ious(masks_true, masks_pred)
    inds = np.arange(0, int(masks_true.max()), 1, int)
    matched = preds > 0
    if not np.any(matched):
        return 0.0
    overlap = _label_overlap(masks_true, masks_pred)
    matched_overlap = overlap[inds[matched] + 1, preds[matched].astype(int)]
    return float(matched_overlap.sum() / union)


def _summary_payload(args, records: Sequence[dict], rows: Sequence[dict], failures: Sequence[str], thresholds: Sequence[float]) -> dict:
    by_view: dict[str, list[dict]] = {}
    for row in rows:
        by_view.setdefault(str(row["channel_view"]), []).append(row)
    return {
        "model_path": str(args.model_path),
        "split_manifest": str(args.split_manifest),
        "split": args.split,
        "n_records_requested": len(records),
        "n_channel_views_evaluated": len(rows),
        "n_failures": len(failures),
        "failures": list(failures[:50]),
        "thresholds": list(thresholds),
        "overall": {
            "aji_mean": _safe_mean([row["aji"] for row in rows]),
            **{
                f"ap_{threshold:g}_mean": _safe_mean(
                    [row[f"ap_{threshold:g}".replace(".", "p")] for row in rows]
                )
                for threshold in thresholds
            },
        },
        "by_channel_view": {
            view: {
                "n": len(view_rows),
                "aji_mean": _safe_mean([row["aji"] for row in view_rows]),
                **{
                    f"ap_{threshold:g}_mean": _safe_mean(
                        [row[f"ap_{threshold:g}".replace(".", "p")] for row in view_rows]
                    )
                    for threshold in thresholds
                },
            }
            for view, view_rows in sorted(by_view.items())
        },
    }


def evaluate(args) -> tuple[Path, Path]:
    split_manifest = load_split_manifest(Path(args.split_manifest))
    records = records_for_split(split_manifest.records, args.split)
    records = _limit_records(records, args.max_records, args.seed)
    if not records:
        raise ValueError(f"No records found for split={args.split!r} in {args.split_manifest}")

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_prefix or f"{args.split}_evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}_summary.json"

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    model = CellposeModel(gpu=use_gpu, pretrained_model=args.model_path)
    image_service = ImageService()
    rng = random.Random(args.seed)
    thresholds = [float(t) for t in args.ap_thresholds]

    rows: list[dict] = []
    failures: list[str] = []
    for index, row in enumerate(records, start=1):
        frame_id = row.get("frame_id") or None
        try:
            image = load_image_ref(
                image_service,
                row["image"],
                frame_id,
                npz_cache_dir=args.npz_cache_dir,
            )
            mask_true = load_mask(
                row["label"],
                frame_id=frame_id,
                npz_mask_channel=args.npz_mask_channel,
                npz_cache_dir=args.npz_cache_dir,
            )
            specs = channel_view_specs(
                image,
                args.channel_sampling_mode,
                args.max_all_channel_combos,
                rng,
            )
            for view_name, channels in specs:
                image_view = make_three_channel_view(image, channels)
                mask_pred, _flows, _styles = model.eval(
                    image_view,
                    batch_size=args.batch_size,
                    channel_axis=-1,
                    normalize=True,
                    flow_threshold=args.flow_threshold,
                    cellprob_threshold=args.cellprob_threshold,
                    min_size=args.min_size,
                    bsize=args.bsize,
                    tile_overlap=args.tile_overlap,
                )
                mask_pred = np.asarray(mask_pred).astype(np.int32, copy=False)
                if mask_pred.shape != mask_true.shape:
                    raise ValueError(
                        f"prediction shape {mask_pred.shape} != label shape {mask_true.shape}"
                    )
                ap, tp, fp, fn = metrics.average_precision(
                    [mask_true.astype(np.int32, copy=False)],
                    [mask_pred],
                    threshold=thresholds,
                )
                aji = aggregated_jaccard_index_one(mask_true, mask_pred)
                metric_row = {
                    "split": args.split,
                    "image": row["image"],
                    "frame_id": frame_id or "",
                    "label": row["label"],
                    "source_group": row.get("source_group", ""),
                    "channel_view": view_name,
                    "n_true": int(max(0, len(np.unique(mask_true)) - 1)),
                    "n_pred": int(max(0, len(np.unique(mask_pred)) - 1)),
                    "aji": float(aji),
                }
                for j, threshold in enumerate(thresholds):
                    tag = f"{threshold:g}".replace(".", "p")
                    metric_row[f"ap_{tag}"] = float(ap[0, j])
                    metric_row[f"tp_{tag}"] = int(tp[0, j])
                    metric_row[f"fp_{tag}"] = int(fp[0, j])
                    metric_row[f"fn_{tag}"] = int(fn[0, j])
                rows.append(metric_row)
        except Exception as exc:
            failures.append(
                f"{row.get('image')}::{frame_id or ''} label={row.get('label')} error={exc}"
            )
        if index % 50 == 0:
            print(f"evaluated {index}/{len(records)} records ({len(rows)} channel views)")

    summary = _summary_payload(args, records, rows, failures, thresholds)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not rows:
        print(f"wrote failure summary: {json_path}")
        print("No evaluation rows were produced. First failures:")
        for failure in failures[:20]:
            print(f"  {failure}")
        raise ValueError(
            f"No evaluation rows were produced. {len(failures)} records failed; "
            f"see {json_path}"
        )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote per-image metrics: {csv_path}")
    print(f"wrote summary metrics: {json_path}")
    print(json.dumps(summary["overall"], indent=2))
    return csv_path, json_path


def main(argv: Sequence[str] | None = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
