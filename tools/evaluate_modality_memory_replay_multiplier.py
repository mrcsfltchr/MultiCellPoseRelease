"""Evaluate single-modality memory replay multiplier models.

This launcher discovers models produced by
``jobs/train_modality_memory_replay_multiplier_scaling.pbs`` and evaluates
each one with ``tools/evaluate_standardized_test_models.py``. It runs one model
per subprocess so interrupted evaluation can be resumed with ``--skip-existing``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


MODEL_RE = re.compile(
    r"^guvpose_modality_replay_(?P<modality>.+?)_replay(?P<replay_count>\d+)$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        default=r"X:\home\MCP_paper\modality_memory_replay_multiplier_scaling",
    )
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-tag", required=True)
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=None,
        help="Optional subset, e.g. Fluorescence Confocal PhaseContrast.",
    )
    parser.add_argument(
        "--replay-counts",
        nargs="*",
        type=int,
        default=None,
        help="Optional replay-count subset, e.g. 0 14 28. Counts are model filename counts.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0)
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
    parser.set_defaults(ignore_classes=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    args.extra_eval_args = extra
    return args


def discover_models(
    model_root: Path,
    modalities: set[str] | None,
    replay_counts: set[int] | None,
) -> list[tuple[str, int, Path]]:
    models: list[tuple[str, int, Path]] = []
    for path in model_root.rglob("guvpose_modality_replay_*_replay*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".csv", ".txt", ".log"}:
            continue
        match = MODEL_RE.match(path.name)
        if not match:
            continue
        modality = match.group("modality")
        replay_count = int(match.group("replay_count"))
        if modalities is not None and modality not in modalities:
            continue
        if replay_counts is not None and replay_count not in replay_counts:
            continue
        models.append((modality, replay_count, path))
    models.sort(key=lambda row: (row[0], row[1], str(row[2])))
    return models


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(
        f'"{part}"' if any(ch.isspace() for ch in str(part)) else str(part)
        for part in cmd
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir) if args.output_dir else model_root / f"{args.results_tag}_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = set(args.modalities) if args.modalities else None
    replay_counts = {int(item) for item in args.replay_counts} if args.replay_counts else None
    models = discover_models(model_root, modalities, replay_counts)
    if not models:
        raise ValueError(f"No guvpose_modality_replay_*_replay* model files found under {model_root}")

    print(f"discovered {len(models)} model files")
    print(f"evaluation outputs: {output_dir}")

    evaluator = Path(__file__).resolve().parent / "evaluate_standardized_test_models.py"
    for modality, replay_count, model_path in models:
        model_name = model_path.name
        prefix = f"{model_name}_{args.results_tag}"
        per_image_csv = output_dir / f"{prefix}_per_image.csv"
        summary_csv = output_dir / f"{prefix}_summary.csv"
        if args.skip_existing and per_image_csv.exists() and summary_csv.exists():
            print(f"skipping existing evaluation: {model_name}")
            continue

        cmd = [
            sys.executable,
            str(evaluator),
            "--test-dir",
            str(args.test_dir),
            "--models",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--results-prefix",
            prefix,
            "--batch-size",
            str(args.batch_size),
            "--bsize",
            str(args.bsize),
            "--max-records",
            str(args.max_records),
        ]
        if args.ignore_classes:
            cmd.append("--ignore-classes")
        if args.cpu:
            cmd.append("--cpu")
        cmd.extend(args.extra_eval_args)

        print("\n" + "=" * 80)
        print(f"evaluating {model_name} ({modality}, replay={replay_count})")
        print(quote_cmd(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print(f"evaluation outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
