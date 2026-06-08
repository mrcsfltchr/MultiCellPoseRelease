"""Run modality-specific CPSAM fine-tuning experiments.

This is a thin launcher around ``tools/train_cpsam_finetune_balanced.py`` for
testing how much CPSAM benefits from fine-tuning on highly correlated modality
subsets. Each modality is trained independently and gets a deterministic model
name of the form ``guvpose_<modality>``.

Example:
    python tools/run_modality_specific_cpsam_finetunes.py \
        --root-dir X:\\home\\cpsamOODtest \
        --epochs 300 \
        --batch-size 8 \
        --unfreeze-blocks 9 \
        --npz-cache-dir X:\\ephemeral\\cpsam_modality_npz_cache
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_MODALITIES = ("Fluorescence", "Confocal", "PhaseContrast", "MultiObject")


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=r"X:\home\cpsamOODtest")
    parser.add_argument("--output-root", default=r"X:\home\cpsamOODtest\modality_finetunes")
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--base-model", default="cpsam")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, default=0.5)
    parser.add_argument("--seg-loss-weight", type=float, default=0.1)
    parser.add_argument("--unfreeze-blocks", type=int, default=9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=0)
    parser.add_argument("--nimg-per-epoch", type=int, default=0)
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Pass --early-stop to train_cpsam_finetune_balanced.py.",
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
    parser.add_argument("--balance-mode", choices=("source", "none"), default="source")
    parser.add_argument("--channel-sampling-mode", choices=("single-and-all", "none"), default="single-and-all")
    parser.add_argument("--max-all-channel-combos", type=int, default=2)
    parser.add_argument("--npz-mask-channel", default="last")
    parser.add_argument("--npz-cache-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--redo-splits", action="store_true")
    parser.add_argument("--launcher-dry-run", action="store_true", help="Print commands without running training.")
    parser.add_argument(
        "--install-to-cellpose-models",
        action="store_true",
        help="Copy each finished model file to ~/.cellpose/models/guvpose_<modality>.",
    )
    parser.add_argument(
        "--cellpose-model-dir",
        default=str(Path.home() / ".cellpose" / "models"),
        help="Destination used with --install-to-cellpose-models.",
    )
    args, extra = parser.parse_known_args(argv)
    return args, extra


def build_command(args: argparse.Namespace, modality: str, extra: list[str]) -> tuple[list[str], Path, Path]:
    root = Path(args.root_dir)
    output_root = Path(args.output_root)
    modality_dir = root / modality
    output_dir = output_root / modality
    model_name = f"guvpose_{modality}"
    script = Path(__file__).resolve().parent / "train_cpsam_finetune_balanced.py"

    cmd = [
        sys.executable,
        str(script),
        "--root-dirs",
        str(modality_dir),
        "--output-dir",
        str(output_dir),
        "--split-manifest",
        str(output_dir / f"{model_name}_splits.json"),
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
        str(args.max_train_records),
        "--max-val-records",
        str(args.max_val_records),
        "--nimg-per-epoch",
        str(args.nimg_per_epoch),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
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
        cmd.extend(["--npz-cache-dir", str(Path(args.npz_cache_dir) / modality)])
    if args.cpu:
        cmd.append("--cpu")
    if args.redo_splits:
        cmd.append("--redo-splits")
    if args.early_stop:
        cmd.append("--early-stop")
    cmd.extend(extra)
    return cmd, modality_dir, output_dir / model_name


def install_model(model_path: Path, destination_dir: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model does not exist: {model_path}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / model_path.name
    shutil.copy2(model_path, destination)
    print(f"installed {model_path} -> {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = parse_args(argv)
    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    for modality in args.modalities:
        cmd, modality_dir, model_path = build_command(args, modality, extra)
        if not modality_dir.exists():
            raise FileNotFoundError(f"Modality directory does not exist: {modality_dir}")
        print("\n" + "=" * 80)
        print(f"modality: {modality}")
        print(f"model: guvpose_{modality}")
        print("command:")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        if args.launcher_dry_run:
            continue
        subprocess.run(cmd, check=True)
        if args.install_to_cellpose_models:
            install_model(model_path, Path(args.cellpose_model_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
