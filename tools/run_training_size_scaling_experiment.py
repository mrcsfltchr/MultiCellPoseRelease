"""Train/evaluate CPSAM fine-tunes over increasing training-set sizes.

This experiment samples OOD/GUV training images equally from modality folders
under ``cpsamOODtest`` and adds FoundationTrain replay images at the same
FoundationTrain:OOD ratio observed in the full balanced training pool. Each
sample size is fine-tuned, then evaluated on both cyto2 and MCP paper test
sets with the standardized evaluator.

The script writes:
    - one split manifest per training size
    - trained model weights
    - standardized evaluation CSVs for each test set
    - a combined scaling_results.csv
    - metric-vs-training-size PNG plots
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tools.train_cpsam_finetune_balanced import (
    LabeledRef,
    SplitManifest,
    discover_labeled_refs,
    load_split_manifest,
    split_labeled_refs,
    write_split_manifest,
)


DEFAULT_MODALITIES = ("Fluorescence", "Confocal", "PhaseContrast", "MultiObject")


@dataclass(frozen=True)
class SplitPools:
    train: list[dict]
    val: list[dict]
    test: list[dict]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-dir", default=r"X:\home\FoundationTrain")
    parser.add_argument("--ood-root", default=r"X:\home\cpsamOODtest")
    parser.add_argument(
        "--full-split-manifest",
        default=None,
        help=(
            "Optional existing full balanced cpsam_finetune_splits.json. When supplied, "
            "the script samples from this manifest instead of rediscovering all labels, "
            "and the FoundationTrain/OOD replay ratio is taken from its train split."
        ),
    )
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--cyto2-test-dir", default=r"X:\home\FoundationTrain\cyto2\test")
    parser.add_argument("--mcp-test-dir", default=r"X:\home\MCP_paper\test")
    parser.add_argument("--output-root", default=r"X:\home\cpsamOODtest\training_size_scaling")
    parser.add_argument(
        "--ood-train-sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Number of cpsamOODtest training records sampled equally across modality folders.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--base-model", default="cpsam")
    parser.add_argument("--model-prefix", default="guvpose_scale")
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
    parser.add_argument("--npz-mask-channel", default="last")
    parser.add_argument("--npz-cache-dir", default=None)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-records", type=int, default=0)
    parser.add_argument(
        "--ignore-classes",
        "--class-agnostic",
        dest="ignore_classes",
        action="store_true",
        help="Forward class-agnostic evaluation to evaluate_standardized_test_models.py.",
    )
    parser.add_argument(
        "--respect-classes",
        dest="ignore_classes",
        action="store_false",
        help="Forward class-aware evaluation to evaluate_standardized_test_models.py.",
    )
    parser.set_defaults(ignore_classes=False)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--x-axis",
        choices=("total_training_images", "ood_training_images"),
        default="total_training_images",
    )
    parser.add_argument("--metrics", nargs="+", default=["f1_0p5", "ap_0p5", "map"])
    return parser.parse_args(argv)


def dict_to_ref(row: dict) -> LabeledRef:
    return LabeledRef(
        image=str(row["image"]),
        frame_id=str(row["frame_id"]) if row.get("frame_id") else None,
        label=str(row["label"]),
        source_group=str(row["source_group"]),
    )


def pools_from_refs(refs: Sequence[LabeledRef], seed: int, val_ratio: float, test_ratio: float) -> SplitPools:
    records = split_labeled_refs(refs, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    return SplitPools(
        train=[row for row in records if row["split"] == "train"],
        val=[row for row in records if row["split"] == "val"],
        test=[row for row in records if row["split"] == "test"],
    )


def discover_pools(args: argparse.Namespace) -> tuple[SplitPools, dict[str, SplitPools]]:
    if args.full_split_manifest:
        return pools_from_full_manifest(args)

    print(f"discovering FoundationTrain labels under {args.foundation_dir}")
    foundation_refs = discover_labeled_refs([args.foundation_dir])
    foundation = pools_from_refs(foundation_refs, args.seed, args.val_ratio, args.test_ratio)
    print(f"FoundationTrain: train={len(foundation.train)} val={len(foundation.val)} test={len(foundation.test)}")

    modality_pools: dict[str, SplitPools] = {}
    for offset, modality in enumerate(args.modalities):
        modality_dir = Path(args.ood_root) / modality
        print(f"discovering {modality} labels under {modality_dir}")
        refs = discover_labeled_refs([str(modality_dir)])
        pools = pools_from_refs(refs, args.seed + 100 + offset, args.val_ratio, args.test_ratio)
        modality_pools[modality] = pools
        print(f"{modality}: train={len(pools.train)} val={len(pools.val)} test={len(pools.test)}")
    return foundation, modality_pools


def _norm_path_text(value: object) -> str:
    return str(value or "").replace("/", "\\").lower()


def _under_root(row: dict, root: str) -> bool:
    root_norm = _norm_path_text(root).rstrip("\\")
    image = _norm_path_text(row.get("image"))
    source = _norm_path_text(row.get("source_group"))
    return image.startswith(root_norm) or source.startswith(root_norm)


def _in_modality(row: dict, ood_root: str, modality: str) -> bool:
    root_norm = _norm_path_text(Path(ood_root) / modality).rstrip("\\")
    image = _norm_path_text(row.get("image"))
    source = _norm_path_text(row.get("source_group"))
    return image.startswith(root_norm) or source.startswith(root_norm)


def _pools_from_rows(rows: Sequence[dict]) -> SplitPools:
    return SplitPools(
        train=[dict(row) for row in rows if row.get("split") == "train"],
        val=[dict(row) for row in rows if row.get("split") == "val"],
        test=[dict(row) for row in rows if row.get("split") == "test"],
    )


def pools_from_full_manifest(args: argparse.Namespace) -> tuple[SplitPools, dict[str, SplitPools]]:
    manifest_path = Path(args.full_split_manifest)
    manifest = load_split_manifest(manifest_path)
    rows = [dict(row) for row in manifest.records]
    foundation_rows = [row for row in rows if _under_root(row, args.foundation_dir)]
    foundation = _pools_from_rows(foundation_rows)
    print(
        f"loaded FoundationTrain pools from {manifest_path}: "
        f"train={len(foundation.train)} val={len(foundation.val)} test={len(foundation.test)}"
    )

    modality_pools: dict[str, SplitPools] = {}
    for modality in args.modalities:
        modality_rows = [row for row in rows if _in_modality(row, args.ood_root, modality)]
        pools = _pools_from_rows(modality_rows)
        modality_pools[modality] = pools
        print(
            f"loaded {modality} pools from {manifest_path}: "
            f"train={len(pools.train)} val={len(pools.val)} test={len(pools.test)}"
        )
    if not foundation.train:
        raise ValueError(f"No FoundationTrain training rows matched {args.foundation_dir} in {manifest_path}")
    if not any(pools.train for pools in modality_pools.values()):
        raise ValueError(f"No OOD modality training rows matched {args.ood_root} in {manifest_path}")
    return foundation, modality_pools


def split_by_source(rows: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row["source_group"]), []).append(row)
    return out


def sample_rows(rows: Sequence[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    if n <= 0:
        return []
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def allocate_equal(total: int, names: Sequence[str], capacities: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("Training size must be non-negative.")
    allocation = {name: 0 for name in names}
    remaining = int(total)
    active = [name for name in names if capacities.get(name, 0) > 0]
    while remaining > 0 and active:
        base = max(1, remaining // len(active))
        progressed = False
        for name in list(active):
            if remaining <= 0:
                break
            available = capacities[name] - allocation[name]
            take = min(base, available, remaining)
            if take > 0:
                allocation[name] += take
                remaining -= take
                progressed = True
            if allocation[name] >= capacities[name]:
                active.remove(name)
        if not progressed:
            break
    return allocation


def select_balanced_training_records(
    foundation: SplitPools,
    modality_pools: dict[str, SplitPools],
    ood_train_size: int,
    foundation_per_ood: float,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    capacities = {name: len(pools.train) for name, pools in modality_pools.items()}
    allocation = allocate_equal(ood_train_size, list(modality_pools), capacities)
    selected: list[dict] = []
    for name, n in allocation.items():
        selected.extend(sample_rows(modality_pools[name].train, n, rng))

    actual_ood = len(selected)
    n_foundation = min(len(foundation.train), int(round(actual_ood * foundation_per_ood)))
    selected.extend(sample_rows(foundation.train, n_foundation, rng))
    counts = {f"ood_{name}": int(n) for name, n in allocation.items()}
    counts["ood_training_images"] = int(actual_ood)
    counts["foundation_replay_images"] = int(n_foundation)
    counts["total_training_images"] = int(actual_ood + n_foundation)
    return selected, counts


def fixed_validation_records(
    foundation: SplitPools,
    modality_pools: dict[str, SplitPools],
    max_val_records: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    rows.extend(foundation.val)
    for pools in modality_pools.values():
        rows.extend(pools.val)
    if max_val_records and max_val_records > 0:
        rng = random.Random(seed)
        rows = sample_rows(rows, min(max_val_records, len(rows)), rng)
    return rows


def write_manifest(path: Path, train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], args: argparse.Namespace) -> None:
    records: list[dict] = []
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        for row in rows:
            record = dict(row)
            record["split"] = split
            records.append(record)
    records.sort(key=lambda row: (row["split"], row["source_group"], row["image"], row["frame_id"], row["label"]))
    manifest = SplitManifest(seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio, records=records)
    write_split_manifest(path, manifest)


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
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
        str(args.foundation_dir),
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


def evaluate_model(
    args: argparse.Namespace,
    model_path: Path,
    model_name: str,
    test_name: str,
    test_dir: Path,
    eval_dir: Path,
) -> Path:
    prefix = f"{model_name}_{test_name}"
    summary_csv = eval_dir / f"{prefix}_summary.csv"
    if args.skip_evaluation:
        return summary_csv
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_standardized_test_models.py"),
        "--test-dir",
        str(test_dir),
        "--models",
        str(model_path),
        "--output-dir",
        str(eval_dir),
        "--results-prefix",
        prefix,
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


def read_summary(summary_csv: Path) -> dict:
    with summary_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in summary CSV: {summary_csv}")
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


def plot_results(results_csv: Path, output_dir: Path, metrics: Sequence[str], x_axis: str) -> None:
    df = pd.read_csv(results_csv)
    if df.empty:
        raise ValueError(f"No rows in {results_csv}")
    for metric in metrics:
        if metric not in df.columns:
            print(f"warning: metric {metric!r} not in results CSV, skipping")
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
        for test_name, sub in df.groupby("test_dataset"):
            sub = sub.sort_values(x_axis)
            ax.plot(sub[x_axis], sub[metric], marker="o", linewidth=1.8, label=test_name)
        ax.set_xlabel(x_axis.replace("_", " "))
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs training-set size")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        out = output_dir / f"{metric}_vs_{x_axis}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "scaling_results.csv"

    if args.plot_only:
        plot_results(results_csv, output_root, args.metrics, args.x_axis)
        return 0

    foundation, modality_pools = discover_pools(args)
    full_ood_train = sum(len(pools.train) for pools in modality_pools.values())
    if full_ood_train <= 0:
        raise ValueError("No OOD modality training records were found.")
    foundation_per_ood = len(foundation.train) / float(full_ood_train)
    print(
        "full balanced replay ratio: "
        f"FoundationTrain/OOD = {len(foundation.train)}/{full_ood_train} = {foundation_per_ood:.4f}"
    )

    fixed_val = fixed_validation_records(foundation, modality_pools, args.max_val_records, args.seed + 999)
    fixed_test = foundation.test + [row for pools in modality_pools.values() for row in pools.test]
    manifest_meta: list[dict] = []
    result_rows: list[dict] = []

    for ood_train_size in args.ood_train_sizes:
        model_name = f"{args.model_prefix}_{ood_train_size:05d}"
        run_dir = output_root / model_name
        eval_dir = run_dir / "evaluation"
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_rows, counts = select_balanced_training_records(
            foundation,
            modality_pools,
            int(ood_train_size),
            foundation_per_ood,
            args.seed + int(ood_train_size),
        )
        manifest_path = run_dir / f"{model_name}_splits.json"
        write_manifest(manifest_path, train_rows, fixed_val, fixed_test, args)
        meta = {
            "model_name": model_name,
            "manifest": str(manifest_path),
            "foundation_per_ood": foundation_per_ood,
            "n_val_records": len(fixed_val),
            "n_internal_test_records": len(fixed_test),
            **counts,
        }
        manifest_meta.append(meta)
        (run_dir / "scale_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("\n" + "=" * 80)
        print(json.dumps(meta, indent=2))
        model_path = train_model(args, run_dir, manifest_path, model_name)
        for test_name, test_dir in (
            ("cyto2", Path(args.cyto2_test_dir)),
            ("mcp_paper", Path(args.mcp_test_dir)),
        ):
            summary_csv = evaluate_model(args, model_path, model_name, test_name, test_dir, eval_dir)
            if args.dry_run:
                continue
            summary = read_summary(summary_csv)
            result_rows.append(
                {
                    **meta,
                    "test_dataset": test_name,
                    "test_dir": str(test_dir),
                    "summary_csv": str(summary_csv),
                    **summary,
                }
            )
            write_csv(results_csv, result_rows)

    write_csv(output_root / "scale_manifests_summary.csv", manifest_meta)
    if not args.dry_run and result_rows:
        write_csv(results_csv, result_rows)
        plot_results(results_csv, output_root, args.metrics, args.x_axis)
    print(f"wrote experiment outputs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
