"""Train CPSAM fine-tunes to test dataset diversity at matched OOD set sizes.

The experiment compares single-modality training pools against a diverse pool
sampled across cpsamOODtest modalities. Each condition uses the same requested
number of cpsamOODtest training records and is evaluated on MCP_paper/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_cpsam_finetune_balanced import (  # noqa: E402
    SplitManifest,
    discover_labeled_refs,
    load_split_manifest,
    load_mask,
    parse_path_maps,
    remap_manifest_records,
    split_labeled_refs,
    write_split_manifest,
)


DEFAULT_MODALITIES = ("Fluorescence", "Confocal", "PhaseContrast")


@dataclass(frozen=True)
class SplitPools:
    train: list[dict]
    val: list[dict]
    test: list[dict]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ood-root", default=r"X:\home\cpsamOODtest")
    parser.add_argument("--mcp-test-dir", default=r"X:\home\MCP_paper\test")
    parser.add_argument("--output-root", default=r"X:\home\MCP_paper\dataset_diversity_experiment")
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument(
        "--modality-split-root",
        default=None,
        help=(
            "Optional root containing previous modality split manifests, e.g. "
            "MCP_paper/modality_encoder_block_sweep. If present, each modality "
            "loads <root>/<modality>/<split-block-dir>/guvpose_<modality>_<split-block-dir>_splits.json."
        ),
    )
    parser.add_argument("--split-block-dir", default="blocks09")
    parser.add_argument(
        "--path-map",
        nargs="*",
        default=[r"/rds/general/user/mfletch1/home=X:\home"],
        help="Optional path prefix maps for manifests created on another filesystem.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200],
        help="Target training-set size, interpreted according to --match-unit.",
    )
    parser.add_argument(
        "--match-unit",
        choices=("objects", "records"),
        default="objects",
        help="Match conditions by ground-truth object count or by labeled image/frame records.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--base-model", default="cpsam")
    parser.add_argument("--model-prefix", default="guvpose_diversity")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, default=0.5)
    parser.add_argument("--seg-loss-weight", type=float, default=0.1)
    parser.add_argument("--unfreeze-blocks", type=int, default=9)
    parser.add_argument("--nimg-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=300)
    parser.add_argument("--balance-mode", choices=("source", "none"), default="source")
    parser.add_argument("--channel-sampling-mode", choices=("single-and-all", "none"), default="single-and-all")
    parser.add_argument("--max-all-channel-combos", type=int, default=2)
    parser.add_argument("--channel-sampling-val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--npz-mask-channel", default="last")
    parser.add_argument("--npz-cache-dir", default=None)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-records", type=int, default=0)
    parser.add_argument("--ignore-classes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def pools_from_modality(root: Path, modality: str, seed: int, val_ratio: float, test_ratio: float) -> SplitPools:
    refs = discover_labeled_refs([str(root / modality)])
    records = split_labeled_refs(refs, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    return SplitPools(
        train=[row for row in records if row["split"] == "train"],
        val=[row for row in records if row["split"] == "val"],
        test=[row for row in records if row["split"] == "test"],
    )


def existing_modality_manifest(split_root: Path, modality: str, block_dir: str) -> Path | None:
    block_path = split_root / modality / block_dir
    candidates = sorted(block_path.glob(f"*_{modality}_{block_dir}_splits.json"))
    if not candidates:
        candidates = sorted(block_path.glob("*_splits.json"))
    return candidates[0] if candidates else None


def load_or_discover_pools(args: argparse.Namespace, modality: str, offset: int) -> SplitPools:
    if args.modality_split_root:
        manifest_path = existing_modality_manifest(Path(args.modality_split_root), modality, args.split_block_dir)
        if manifest_path and manifest_path.exists():
            manifest = load_split_manifest(manifest_path)
            records, n_mapped = remap_manifest_records(manifest.records, parse_path_maps(args.path_map))
            if n_mapped:
                print(f"remapped {n_mapped} paths in {manifest_path}")
            print(f"loaded {modality} split manifest: {manifest_path}")
            return SplitPools(
                train=[dict(row) for row in records if row.get("split") == "train"],
                val=[dict(row) for row in records if row.get("split") == "val"],
                test=[dict(row) for row in records if row.get("split") == "test"],
            )
        print(f"warning: no split manifest found for {modality} under {args.modality_split_root}; discovering labels")
    return pools_from_modality(Path(args.ood_root), modality, args.seeds[0] + 100 + offset, args.val_ratio, args.test_ratio)


def sample_rows(rows: Sequence[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    if n <= 0:
        return []
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def record_key(row: dict) -> str:
    return "||".join([str(row.get("image", "")), str(row.get("frame_id") or ""), str(row.get("label", ""))])


def count_mask_objects(row: dict, npz_mask_channel: str, npz_cache_dir: str | Path | None) -> int:
    mask = load_mask(
        row["label"],
        frame_id=row.get("frame_id") or None,
        npz_mask_channel=npz_mask_channel,
        npz_cache_dir=npz_cache_dir,
    )
    return int(len(np.unique(mask[mask > 0])))


def add_object_counts_to_pools(args: argparse.Namespace, pools: dict[str, SplitPools]) -> dict[str, SplitPools]:
    if args.match_unit != "objects":
        return pools

    cache_path = Path(args.output_root) / "ground_truth_object_counts.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    changed = False
    counted = 0
    out: dict[str, SplitPools] = {}
    for modality, pool in pools.items():
        split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for split_name, rows in (("train", pool.train), ("val", pool.val), ("test", pool.test)):
            for row in rows:
                updated = dict(row)
                key = record_key(updated)
                if key not in cache:
                    cache[key] = count_mask_objects(updated, args.npz_mask_channel, args.npz_cache_dir)
                    changed = True
                    if len(cache) % 25 == 0:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
                        print(f"updated object-count cache: {cache_path} ({len(cache)} records)")
                updated["object_count"] = int(cache[key])
                split_rows[split_name].append(updated)
                counted += 1
                if counted % 100 == 0:
                    print(f"counted ground-truth objects for {counted} records")
        out[modality] = SplitPools(train=split_rows["train"], val=split_rows["val"], test=split_rows["test"])

    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"wrote object-count cache: {cache_path}")
    return out


def row_object_count(row: dict) -> int:
    return int(row.get("object_count", 0))


def sample_rows_by_object_count(rows: Sequence[dict], target_objects: int, rng: random.Random) -> list[dict]:
    rows = [row for row in rows if row_object_count(row) > 0]
    if target_objects <= 0 or not rows:
        return []
    total_available = sum(row_object_count(row) for row in rows)
    if total_available <= target_objects:
        return list(rows)

    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected: list[dict] = []
    total = 0
    skipped: list[dict] = []
    for row in shuffled:
        count = row_object_count(row)
        if total + count <= target_objects:
            selected.append(row)
            total += count
        else:
            skipped.append(row)
    if not selected and skipped:
        return [min(skipped, key=lambda row: abs(row_object_count(row) - target_objects))]

    best = selected
    best_gap = abs(target_objects - total)
    selected_keys = {record_key(row) for row in selected}
    for row in shuffled:
        if record_key(row) in selected_keys:
            continue
        gap = abs(target_objects - (total + row_object_count(row)))
        if gap < best_gap:
            best = selected + [row]
            best_gap = gap
    return best


def allocate_equal(total: int, names: Sequence[str], capacities: dict[str, int]) -> dict[str, int]:
    allocation = {name: 0 for name in names}
    remaining = int(total)
    active = [name for name in names if capacities.get(name, 0) > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for name in list(active):
            if remaining <= 0:
                break
            take = min(share, capacities[name] - allocation[name], remaining)
            if take > 0:
                allocation[name] += take
                remaining -= take
                progressed = True
            if allocation[name] >= capacities[name]:
                active.remove(name)
        if not progressed:
            break
    return allocation


def allocate_equal_objects(total: int, names: Sequence[str], capacities: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("Training size must be non-negative.")
    allocation = {name: 0 for name in names}
    remaining = int(total)
    active = [name for name in names if capacities.get(name, 0) > 0]
    while remaining > 0 and active:
        share = max(1, int(math.ceil(remaining / len(active))))
        progressed = False
        for name in list(active):
            if remaining <= 0:
                break
            take = min(share, capacities[name] - allocation[name], remaining)
            if take > 0:
                allocation[name] += take
                remaining -= take
                progressed = True
            if allocation[name] >= capacities[name]:
                active.remove(name)
        if not progressed:
            break
    return allocation


def diverse_sample(
    pools: dict[str, SplitPools],
    total: int,
    rng: random.Random,
    match_unit: str,
) -> tuple[list[dict], dict[str, int]]:
    capacities = {name: len(pool.train) for name, pool in pools.items()}
    if match_unit == "objects":
        capacities = {name: sum(row_object_count(row) for row in pool.train) for name, pool in pools.items()}
        allocation = allocate_equal_objects(total, list(pools), capacities)
    else:
        allocation = allocate_equal(total, list(pools), capacities)
    rows: list[dict] = []
    for name, count in allocation.items():
        if match_unit == "objects":
            rows.extend(sample_rows_by_object_count(pools[name].train, count, rng))
        else:
            rows.extend(sample_rows(pools[name].train, count, rng))
    rng.shuffle(rows)
    return rows, allocation


def fixed_validation_records(pools: dict[str, SplitPools], max_val_records: int, seed: int) -> list[dict]:
    rows = [row for pool in pools.values() for row in pool.val]
    if max_val_records and max_val_records > 0:
        rows = sample_rows(rows, min(max_val_records, len(rows)), random.Random(seed))
    return rows


def write_manifest(path: Path, train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], seed: int, val_ratio: float, test_ratio: float) -> None:
    records: list[dict] = []
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        for row in rows:
            record = dict(row)
            record["split"] = split
            records.append(record)
    records.sort(key=lambda row: (row["split"], row["source_group"], row["image"], row.get("frame_id") or "", row["label"]))
    write_split_manifest(path, SplitManifest(seed=seed, val_ratio=val_ratio, test_ratio=test_ratio, records=records))


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def train_model(args: argparse.Namespace, run_dir: Path, manifest_path: Path, model_name: str) -> Path:
    model_path = run_dir / model_name
    if args.skip_training:
        return model_path
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "train_cpsam_finetune_balanced.py"),
        "--root-dirs",
        str(args.ood_root),
        "--output-dir",
        str(run_dir),
        "--split-manifest",
        str(manifest_path),
        "--base-model",
        str(args.base_model),
        "--model-name",
        model_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--bsize",
        str(args.bsize),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--scale-range",
        str(args.scale_range),
        "--seg-loss-weight",
        str(args.seg_loss_weight),
        "--unfreeze-blocks",
        str(args.unfreeze_blocks),
        "--max-train-records",
        "0",
        "--max-val-records",
        str(args.max_val_records),
        "--nimg-per-epoch",
        str(args.nimg_per_epoch),
        "--balance-mode",
        str(args.balance_mode),
        "--channel-sampling-mode",
        str(args.channel_sampling_mode),
        "--max-all-channel-combos",
        str(args.max_all_channel_combos),
        "--npz-mask-channel",
        str(args.npz_mask_channel),
    ]
    cmd.append("--channel-sampling-val" if args.channel_sampling_val else "--no-channel-sampling-val")
    if args.npz_cache_dir:
        cmd.extend(["--npz-cache-dir", str(Path(args.npz_cache_dir) / model_name)])
    if args.early_stop:
        cmd.extend([
            "--early-stop",
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--early-stop-min-delta",
            str(args.early_stop_min_delta),
        ])
    if args.cpu:
        cmd.append("--cpu")
    run_command(cmd, args.dry_run)
    return model_path


def evaluate_model(args: argparse.Namespace, model_path: Path, model_name: str, eval_dir: Path) -> Path:
    summary_csv = eval_dir / f"{model_name}_mcp_paper_summary.csv"
    if args.skip_evaluation:
        return summary_csv
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_standardized_test_models.py"),
        "--test-dir",
        str(args.mcp_test_dir),
        "--models",
        str(model_path),
        "--output-dir",
        str(eval_dir),
        "--results-prefix",
        f"{model_name}_mcp_paper",
        "--batch-size",
        str(args.eval_batch_size),
        "--max-records",
        str(args.eval_max_records),
    ]
    if args.ignore_classes:
        cmd.append("--ignore-classes")
    if args.cpu:
        cmd.append("--cpu")
    run_command(cmd, args.dry_run)
    return summary_csv


def read_summary(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in summary CSV: {path}")
    return rows[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pools: dict[str, SplitPools] = {}
    for offset, modality in enumerate(args.modalities):
        pool = load_or_discover_pools(args, modality, offset)
        pools[modality] = pool
        print(f"{modality}: train={len(pool.train)} val={len(pool.val)} test={len(pool.test)}")
    pools = add_object_counts_to_pools(args, pools)
    if args.match_unit == "objects":
        for modality, pool in pools.items():
            print(
                f"{modality}: train_objects={sum(row_object_count(row) for row in pool.train)} "
                f"val_objects={sum(row_object_count(row) for row in pool.val)} "
                f"test_objects={sum(row_object_count(row) for row in pool.test)}"
            )

    fixed_val = fixed_validation_records(pools, args.max_val_records, args.seeds[0] + 999)
    fixed_test = [row for pool in pools.values() for row in pool.test]
    result_rows: list[dict] = []
    manifest_rows: list[dict] = []

    conditions = [("single", modality) for modality in args.modalities] + [("diverse", "all")]
    for seed in args.seeds:
        for train_size in args.train_sizes:
            for condition_type, condition_name in conditions:
                rng = random.Random(seed + train_size * 1009 + sum(ord(ch) for ch in condition_name))
                if condition_type == "single":
                    allocation = {modality: 0 for modality in args.modalities}
                    if args.match_unit == "objects":
                        selected = sample_rows_by_object_count(pools[condition_name].train, train_size, rng)
                        allocation[condition_name] = sum(row_object_count(row) for row in selected)
                    else:
                        selected = sample_rows(pools[condition_name].train, train_size, rng)
                        allocation[condition_name] = len(selected)
                else:
                    selected, allocation = diverse_sample(pools, train_size, rng, args.match_unit)

                actual_train_size = len(selected)
                actual_train_objects = sum(row_object_count(row) for row in selected)
                size_label = actual_train_objects if args.match_unit == "objects" else actual_train_size
                model_name = f"{args.model_prefix}_{condition_type}_{condition_name}_{args.match_unit}{size_label:04d}_seed{seed}"
                run_dir = output_root / condition_type / condition_name / f"{args.match_unit}{size_label:04d}" / f"seed{seed}"
                eval_dir = run_dir / "evaluation"
                run_dir.mkdir(parents=True, exist_ok=True)
                eval_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = run_dir / f"{model_name}_splits.json"
                write_manifest(manifest_path, selected, fixed_val, fixed_test, seed, args.val_ratio, args.test_ratio)

                meta = {
                    "model_name": model_name,
                    "condition_type": condition_type,
                    "condition_name": condition_name,
                    "match_unit": args.match_unit,
                    "requested_train_size": train_size,
                    "actual_train_records": actual_train_size,
                    "actual_train_objects": actual_train_objects,
                    "seed": seed,
                    "manifest": str(manifest_path),
                    "n_val_records": len(fixed_val),
                    "n_internal_test_records": len(fixed_test),
                    **{f"train_{name}": int(allocation.get(name, 0)) for name in args.modalities},
                    **{f"train_records_{name}": sum(1 for row in selected if row in pools[name].train) for name in args.modalities},
                    **{f"train_objects_{name}": sum(row_object_count(row) for row in selected if row in pools[name].train) for name in args.modalities},
                }
                (run_dir / "diversity_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                manifest_rows.append(meta)

                print("\n" + "=" * 80)
                print(json.dumps(meta, indent=2))
                model_path = train_model(args, run_dir, manifest_path, model_name)
                summary_csv = evaluate_model(args, model_path, model_name, eval_dir)
                if not args.dry_run and not args.skip_evaluation:
                    result_rows.append({
                        **meta,
                        "test_dataset": "mcp_paper",
                        "test_dir": str(args.mcp_test_dir),
                        "summary_csv": str(summary_csv),
                        **read_summary(summary_csv),
                    })
                    write_csv(output_root / "dataset_diversity_results.csv", result_rows)

    write_csv(output_root / "dataset_diversity_manifests.csv", manifest_rows)
    if result_rows:
        write_csv(output_root / "dataset_diversity_results.csv", result_rows)
    print(f"wrote experiment outputs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
