"""Run CPSAM fine-tuning while increasing FoundationTrain memory replay.

Each run keeps a fixed number of cpsamOODtest training records and varies the
number of FoundationTrain training records. A per-run split manifest is written
so the exact training composition is persistent and inspectable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_cpsam_finetune_balanced import (
    SplitManifest,
    discover_labeled_refs,
    split_labeled_refs,
    write_split_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-root", default=r"X:\home\FoundationTrain")
    parser.add_argument("--ood-root", default=r"X:\home\cpsamOODtest")
    parser.add_argument("--output-root", default=r"X:\home\MCP_paper\memory_replay_scaling")
    parser.add_argument(
        "--shared-split-manifest",
        default=None,
        help="Optional existing split manifest covering FoundationTrain and cpsamOODtest. If provided, avoids rediscovery.",
    )
    parser.add_argument("--replay-counts", nargs="+", type=int, default=[0, 250, 500, 1000, 2000, 4000, 8000])
    parser.add_argument(
        "--replay-multipliers",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional FoundationTrain replay sizes expressed as multiples of the selected "
            "cpsamOODtest train-set size. Overrides --replay-counts when provided."
        ),
    )
    parser.add_argument(
        "--ood-train-records",
        type=int,
        default=0,
        help="Fixed number of cpsamOODtest train records. 0 uses all available cpsamOODtest train records.",
    )
    parser.add_argument("--base-model", default="cpsam")
    parser.add_argument("--model-prefix", default="guvpose_replay")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, default=0.5)
    parser.add_argument("--seg-loss-weight", type=float, default=0.1)
    parser.add_argument("--unfreeze-blocks", type=int, default=9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-val-records", type=int, default=300)
    parser.add_argument("--nimg-per-epoch", type=int, default=0)
    parser.add_argument("--balance-mode", choices=("source", "none"), default="source")
    parser.add_argument("--channel-sampling-mode", choices=("single-and-all", "none"), default="single-and-all")
    parser.add_argument("--max-all-channel-combos", type=int, default=2)
    parser.add_argument("--channel-sampling-val", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--npz-mask-channel", default="last")
    parser.add_argument("--npz-cache-dir", default=None)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--redo-splits", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    return args, extra


def normalized_path_prefix(value: str | Path) -> str:
    return str(value).replace("\\", "/").lower().rstrip("/")


def source_contains(row: dict, root_prefix: str, root_name: str) -> bool:
    image_text = str(row["image"]).replace("\\", "/").lower()
    group_text = str(row.get("source_group", "")).replace("\\", "/").lower()
    root_parts = [part for part in root_prefix.lower().strip("/").split("/") if part]
    marker_tail = root_parts[-1] if len(root_parts) < 2 else "/".join(root_parts[-2:])
    if marker_tail.endswith(":"):
        marker_tail = root_name.lower().strip("/")
    marker = f"/{marker_tail}"
    return (
        image_text.startswith(root_prefix)
        or group_text.startswith(root_prefix)
        or marker in image_text
        or marker in group_text
    )


def stratified_sample(records: list[dict], limit: int, seed: int, *, zero_means_all: bool = False) -> list[dict]:
    records = list(records)
    if limit == 0 and not zero_means_all:
        return []
    if limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = {}
    for row in records:
        by_group.setdefault(str(row.get("source_group", "")), []).append(row)
    selected: list[dict] = []
    groups = sorted(by_group)
    per_group = max(1, limit // max(1, len(groups)))
    for group in groups:
        rows = list(by_group[group])
        rng.shuffle(rows)
        selected.extend(rows[: min(per_group, len(rows))])
    selected_keys = {record_key(row) for row in selected}
    remaining = [row for row in records if record_key(row) not in selected_keys]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    rng.shuffle(selected)
    return selected[:limit]


def record_key(row: dict) -> tuple:
    return (row.get("split"), row.get("image"), row.get("frame_id"), row.get("label"))


def subset_manifest(
    base_manifest: SplitManifest,
    foundation_root_prefix: str,
    ood_root_prefix: str,
    foundation_root_name: str,
    ood_root_name: str,
    replay_count: int,
    ood_train_records: int,
    seed: int,
) -> tuple[SplitManifest, dict]:
    train = [row for row in base_manifest.records if row["split"] == "train"]
    non_train = [row for row in base_manifest.records if row["split"] != "train"]
    foundation_train = [row for row in train if source_contains(row, foundation_root_prefix, foundation_root_name)]
    ood_train = [row for row in train if source_contains(row, ood_root_prefix, ood_root_name)]

    sampled_ood = stratified_sample(ood_train, ood_train_records, seed + 101, zero_means_all=True)
    sampled_foundation = stratified_sample(
        foundation_train,
        replay_count,
        seed + 202 + int(replay_count),
        zero_means_all=False,
    )
    selected = sampled_ood + sampled_foundation
    selected.sort(key=lambda row: (row["source_group"], row["image"], row.get("frame_id") or "", row["label"]))
    records = [dict(row) for row in selected] + [dict(row) for row in non_train]
    records.sort(key=lambda row: (row["split"], row["source_group"], row["image"], row.get("frame_id") or "", row["label"]))

    manifest = SplitManifest(
        seed=base_manifest.seed,
        val_ratio=base_manifest.val_ratio,
        test_ratio=base_manifest.test_ratio,
        records=records,
    )
    meta = {
        "available_foundation_train_records": len(foundation_train),
        "available_ood_train_records": len(ood_train),
        "requested_foundation_replay_records": replay_count,
        "requested_ood_train_records": ood_train_records,
        "selected_foundation_train_records": len(sampled_foundation),
        "selected_ood_train_records": len(sampled_ood),
        "selected_train_records": len(selected),
        "heldout_records_preserved": len(non_train),
    }
    return manifest, meta


def replay_label(count: int) -> str:
    return f"replay{int(count):05d}"


def replay_counts_from_multipliers(
    base_manifest: SplitManifest,
    ood_root_prefix: str,
    ood_root_name: str,
    ood_train_records: int,
    multipliers: Sequence[float],
) -> tuple[list[int], dict[int, float], int, int]:
    train = [row for row in base_manifest.records if row["split"] == "train"]
    ood_train = [row for row in train if source_contains(row, ood_root_prefix, ood_root_name)]
    basis_count = len(ood_train) if ood_train_records <= 0 else min(int(ood_train_records), len(ood_train))
    if basis_count <= 0:
        raise ValueError("Cannot derive replay counts from multipliers because no OOD train records were found.")

    counts: list[int] = []
    multiplier_by_count: dict[int, float] = {}
    for multiplier in multipliers:
        if multiplier < 0:
            raise ValueError(f"Replay multipliers must be non-negative, got {multiplier}")
        count = int(math.floor(basis_count * float(multiplier) + 0.5))
        if count not in multiplier_by_count:
            counts.append(count)
            multiplier_by_count[count] = float(multiplier)
    return counts, multiplier_by_count, basis_count, len(ood_train)


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if any(ch.isspace() for ch in str(part)) else str(part) for part in cmd)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = parse_args(argv)
    foundation_root = Path(args.foundation_root)
    ood_root = Path(args.ood_root)
    foundation_root_prefix = normalized_path_prefix(foundation_root)
    ood_root_prefix = normalized_path_prefix(ood_root)
    foundation_root_name = foundation_root.name
    ood_root_name = ood_root.name
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shared_manifest_path = Path(args.shared_split_manifest) if args.shared_split_manifest else output_root / "shared_foundation_ood_splits.json"
    if shared_manifest_path.exists() and (args.shared_split_manifest or not args.redo_splits):
        payload = json.loads(shared_manifest_path.read_text(encoding="utf-8"))
        base_manifest = SplitManifest(
            seed=int(payload["seed"]),
            val_ratio=float(payload["val_ratio"]),
            test_ratio=float(payload["test_ratio"]),
            records=list(payload["records"]),
        )
        print(f"loaded existing shared split manifest: {shared_manifest_path}")
    else:
        labeled = discover_labeled_refs([str(foundation_root), str(ood_root)])
        base_records = split_labeled_refs(
            labeled,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        base_manifest = SplitManifest(
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            records=base_records,
        )
        write_split_manifest(shared_manifest_path, base_manifest)
        print(f"wrote shared split manifest: {shared_manifest_path}")
        print(f"wrote shared split CSV: {shared_manifest_path.with_suffix('.csv')}")

    replay_counts = [int(count) for count in args.replay_counts]
    multiplier_by_count: dict[int, float] = {}
    multiplier_basis_count = None
    available_ood_train_records = None
    if args.replay_multipliers:
        replay_counts, multiplier_by_count, multiplier_basis_count, available_ood_train_records = replay_counts_from_multipliers(
            base_manifest,
            ood_root_prefix,
            ood_root_name,
            int(args.ood_train_records),
            args.replay_multipliers,
        )
        print(
            "derived replay counts from multipliers: "
            f"basis_train_records={multiplier_basis_count} "
            f"available_ood_train_records={available_ood_train_records} "
            f"counts={replay_counts}"
        )

    runs = []
    for replay_count in replay_counts:
        label = replay_label(replay_count)
        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest, meta = subset_manifest(
            base_manifest,
            foundation_root_prefix,
            ood_root_prefix,
            foundation_root_name,
            ood_root_name,
            int(replay_count),
            int(args.ood_train_records),
            int(args.seed),
        )
        split_manifest = run_dir / f"{args.model_prefix}_{label}_splits.json"
        write_split_manifest(split_manifest, manifest)
        meta_path = run_dir / f"{args.model_prefix}_{label}_composition.json"
        meta_payload = {
            **meta,
            "foundation_root": str(foundation_root),
            "ood_root": str(ood_root),
            "split_manifest": str(split_manifest),
            "replay_multiplier": multiplier_by_count.get(int(replay_count)),
            "replay_multiplier_basis_train_records": multiplier_basis_count,
            "available_ood_train_records_for_multiplier_basis": available_ood_train_records,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

        model_name = f"{args.model_prefix}_{label}"
        npz_cache_dir = str(Path(args.npz_cache_dir) / label) if args.npz_cache_dir else None
        script = Path(__file__).resolve().parent / "train_cpsam_finetune_balanced.py"
        cmd = [
            sys.executable,
            str(script),
            "--root-dirs",
            str(foundation_root),
            str(ood_root),
            "--output-dir",
            str(run_dir),
            "--split-manifest",
            str(split_manifest),
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
            "--val-ratio",
            str(args.val_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
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
        if npz_cache_dir:
            cmd.extend(["--npz-cache-dir", npz_cache_dir])
        if args.early_stop:
            cmd.extend(["--early-stop"])
        cmd.extend(
            [
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--early-stop-min-delta",
                str(args.early_stop_min_delta),
            ]
        )
        if args.cpu:
            cmd.append("--cpu")
        cmd.extend(extra)
        runs.append({"label": label, "model_name": model_name, "command": cmd, "composition": meta_payload})

    manifest_path = output_root / "memory_replay_scaling_manifest.json"
    manifest_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print(f"wrote replay sweep manifest: {manifest_path}")

    for run in runs:
        print("\n" + "=" * 80)
        print(
            f"training {run['model_name']} "
            f"(FoundationTrain replay={run['composition']['selected_foundation_train_records']}, "
            f"cpsamOODtest={run['composition']['selected_ood_train_records']})"
        )
        print(quote_cmd(run["command"]))
        if not args.dry_run:
            subprocess.run(run["command"], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
