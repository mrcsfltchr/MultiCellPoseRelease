"""Evaluate baseline CPSAM on the held-out cpsamOODtest split.

This is a thin launcher around ``evaluate_full_block_sweep_heldout_split.py``.
It uses the persistent full balanced train/val/test manifest, filters the held
out split to records from cpsamOODtest, and evaluates the unfine-tuned CPSAM
model with class-agnostic metrics by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default=r"X:\home\FoundationTrain\cpsam_finetune_block_sweep")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--source-contains", nargs="*", default=["cpsamOODtest"])
    parser.add_argument(
        "--output-dir",
        default=r"X:\home\MCP_paper\heldout_split_standardized_evaluation\noclasses",
    )
    parser.add_argument(
        "--fallback-output-dir",
        default=r"heldout_split_standardized_evaluation\noclasses",
    )
    parser.add_argument("--results-prefix", default="cpsam_baseline_cpsamOODtest_heldout_eval")
    parser.add_argument("--model", default="cpsam")
    parser.add_argument("--model-dir", default=str(Path.home() / ".cellpose" / "models"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--respect-classes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    args.extra_eval_args = extra
    return args


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if any(ch.isspace() for ch in str(part)) else str(part) for part in cmd)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evaluator = Path(__file__).resolve().parent / "evaluate_full_block_sweep_heldout_split.py"
    cmd = [
        sys.executable,
        str(evaluator),
        "--sweep-root",
        str(args.sweep_root),
        "--split",
        str(args.split),
        "--models",
        str(args.model),
        "--model-dir",
        str(args.model_dir),
        "--output-dir",
        str(args.output_dir),
        "--fallback-output-dir",
        str(args.fallback_output_dir),
        "--results-prefix",
        str(args.results_prefix),
        "--batch-size",
        str(args.batch_size),
        "--bsize",
        str(args.bsize),
        "--max-records",
        str(args.max_records),
    ]
    if args.split_manifest:
        cmd.extend(["--split-manifest", str(args.split_manifest)])
    if args.source_contains:
        cmd.append("--source-contains")
        cmd.extend(str(item) for item in args.source_contains)
    cmd.append("--respect-classes" if args.respect_classes else "--class-agnostic")
    if args.cpu:
        cmd.append("--cpu")
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(args.extra_eval_args)

    print(quote_cmd(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
