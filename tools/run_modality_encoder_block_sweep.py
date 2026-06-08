"""Run modality-specific CPSAM fine-tunes across encoder block counts.

Each run trains on one cpsamOODtest modality subfolder and one encoder
unfreeze setting. Model names encode both factors:

    guvpose_<Modality>_blocks09
    guvpose_<Modality>_blocks12
    guvpose_<Modality>_blocks15
    guvpose_<Modality>_blocks18
    guvpose_<Modality>_blocksfull

The ``full`` setting is passed to the balanced finetune script as a large
``--unfreeze-blocks`` value and is clamped to the actual encoder depth by
``cellpose.training_mode_utils.configure_trainable_params``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_MODALITIES = ("Fluorescence", "Confocal", "PhaseContrast", "MultiObject")


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=r"X:\home\cpsamOODtest")
    parser.add_argument("--output-root", default=r"X:\home\MCP_paper\modality_encoder_block_sweep")
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--blocks", nargs="+", default=["9", "12", "15", "18", "full"])
    parser.add_argument("--full-blocks-value", type=int, default=999)
    parser.add_argument("--base-model", default="cpsam")
    parser.add_argument("--model-prefix", default="guvpose")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--batch-size-by-block",
        default="",
        help=(
            "Optional comma-separated overrides such as "
            "'9:24,12:12,15:8,18:6,full:2'. Keys match --blocks entries."
        ),
    )
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, default=0.5)
    parser.add_argument("--seg-loss-weight", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=0)
    parser.add_argument("--nimg-per-epoch", type=int, default=0)
    parser.add_argument("--balance-mode", choices=("source", "none"), default="source")
    parser.add_argument("--channel-sampling-mode", choices=("single-and-all", "none"), default="single-and-all")
    parser.add_argument("--max-all-channel-combos", type=int, default=2)
    parser.add_argument("--channel-sampling-val", action=argparse.BooleanOptionalAction, default=True)
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


def parse_block(block: str, full_value: int) -> tuple[str, int]:
    text = str(block).strip().lower()
    if text == "full":
        return "full", int(full_value)
    value = int(text)
    if value < 0:
        raise ValueError(f"Block count must be non-negative or 'full', got {block!r}")
    return f"{value:02d}", value


def parse_batch_size_by_block(spec: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if not spec:
        return overrides
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --batch-size-by-block item {item!r}; expected key:value")
        key, value = item.split(":", 1)
        key = key.strip().lower()
        if key != "full":
            key = str(int(key))
        overrides[key] = int(value)
    return overrides


def batch_size_for_block(block: str, default_batch_size: int, overrides: dict[str, int]) -> int:
    key = str(block).strip().lower()
    if key != "full":
        key = str(int(key))
    return int(overrides.get(key, default_batch_size))


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if any(ch.isspace() for ch in part) else part for part in cmd)


def build_command(
    args: argparse.Namespace,
    extra: list[str],
    modality: str,
    block_label: str,
    unfreeze_blocks: int,
    batch_size: int,
) -> tuple[list[str], dict]:
    modality_dir = Path(args.root_dir) / modality
    output_dir = Path(args.output_root) / modality / f"blocks{block_label}"
    split_manifest = output_dir / f"{args.model_prefix}_{modality}_blocks{block_label}_splits.json"
    model_name = f"{args.model_prefix}_{modality}_blocks{block_label}"
    script = Path(__file__).resolve().parent / "train_cpsam_finetune_balanced.py"

    cmd = [
        sys.executable,
        str(script),
        "--root-dirs",
        str(modality_dir),
        "--output-dir",
        str(output_dir),
        "--split-manifest",
        str(split_manifest),
        "--base-model",
        str(args.base_model),
        "--model-name",
        model_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
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
        str(unfreeze_blocks),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--seed",
        str(args.seed),
        "--max-train-records",
        str(args.max_train_records),
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
        cmd.extend(["--npz-cache-dir", str(Path(args.npz_cache_dir) / modality / f"blocks{block_label}")])
    if args.early_stop:
        cmd.extend(
            [
                "--early-stop",
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--early-stop-min-delta",
                str(args.early_stop_min_delta),
            ]
        )
    if args.cpu:
        cmd.append("--cpu")
    if args.redo_splits:
        cmd.append("--redo-splits")
    cmd.extend(extra)

    meta = {
        "modality": modality,
        "block_label": block_label,
        "unfreeze_blocks_arg": unfreeze_blocks,
        "batch_size": batch_size,
        "model_name": model_name,
        "modality_dir": str(modality_dir),
        "output_dir": str(output_dir),
        "split_manifest": str(split_manifest),
        "command": cmd,
    }
    return cmd, meta


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runs = []
    batch_overrides = parse_batch_size_by_block(args.batch_size_by_block)
    for modality in args.modalities:
        modality_dir = Path(args.root_dir) / modality
        if not modality_dir.exists():
            raise FileNotFoundError(f"Modality directory does not exist: {modality_dir}")
        for block in args.blocks:
            block_label, unfreeze_blocks = parse_block(block, args.full_blocks_value)
            block_batch_size = batch_size_for_block(block, args.batch_size, batch_overrides)
            cmd, meta = build_command(args, extra, modality, block_label, unfreeze_blocks, block_batch_size)
            Path(meta["output_dir"]).mkdir(parents=True, exist_ok=True)
            runs.append(meta)

    manifest_path = output_root / "modality_encoder_block_sweep_manifest.json"
    manifest_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print(f"wrote sweep manifest: {manifest_path}")

    for run in runs:
        print("\n" + "=" * 80)
        print(f"training {run['model_name']} from {run['modality_dir']}")
        print(quote_cmd(run["command"]))
        if not args.dry_run:
            subprocess.run(run["command"], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
