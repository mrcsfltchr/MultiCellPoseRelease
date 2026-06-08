"""Compare dataset diversity using Cellpose-style feature correlations.

The Cellpose / Cellpose-SAM papers use image-level style/feature vectors to
visualize how similar test images are to training images. This script creates
the same kind of train-vs-test correlation plots for two datasets, for example
your foundation training data and the cyto2 dataset.

By default this script uses pooled CPSAM encoder-neck features instead of the
``styles`` returned by this repository's CPSAM Transformer, because the current
Transformer implementation returns random style vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cellpose import models as cp_models
from tools.cpsam_feature_similarity_pairs import (  # noqa: E402
    ImageRecord,
    discover_records,
    extract_cpsam_feature_map,
    load_frame,
    pooled_feature_vector,
    prepare_image,
    stable_id,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two image datasets using train-vs-test feature correlation "
            "plots similar to Cellpose-SAM Figure 2."
        )
    )
    parser.add_argument("--dataset-a-name", default="training")
    parser.add_argument("--dataset-a-train-dirs", nargs="+", required=True)
    parser.add_argument("--dataset-a-test-dirs", nargs="*", default=None)
    parser.add_argument("--dataset-b-name", default="cyto2")
    parser.add_argument("--dataset-b-train-dirs", nargs="+", required=True)
    parser.add_argument("--dataset-b-test-dirs", nargs="*", default=None)
    parser.add_argument("--output-dir", default="paper/dataset_style_correlations")
    parser.add_argument(
        "--feature-backend",
        choices=("cpsam-neck", "cellpose-style"),
        default="cpsam-neck",
        help=(
            "cpsam-neck uses pooled CPSAM encoder-neck features. cellpose-style "
            "uses model.eval(..., compute_masks=False) styles, which are random "
            "for this repository's CPSAM Transformer."
        ),
    )
    parser.add_argument(
        "--model",
        default="cpsam",
        help="Cellpose model name or path used for feature extraction.",
    )
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto if omitted.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=0.2,
        help="Held-out fraction used when a dataset test directory is not supplied.",
    )
    parser.add_argument("--max-train-images", type=int, default=500)
    parser.add_argument("--max-test-images", type=int, default=250)
    parser.add_argument(
        "--channel-mode",
        choices=("first3", "mean", "channel"),
        default="first3",
        help="How to collapse/select channels before feature extraction.",
    )
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discover images recursively.",
    )
    parser.add_argument(
        "--sort-heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort test rows by maximum train correlation in heatmaps.",
    )
    return parser.parse_args(argv)


def choose_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_records(records: list[ImageRecord], max_records: int | None, seed: int) -> list[ImageRecord]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=max_records, replace=False)
    return [records[int(i)] for i in sorted(idx)]


def split_records(
    records: list[ImageRecord], fraction: float, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if len(records) < 2:
        raise ValueError("At least two records are required when splitting a dataset.")
    fraction = min(max(float(fraction), 0.01), 0.9)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_test = max(1, int(round(len(records) * fraction)))
    test_idx = set(int(i) for i in order[:n_test])
    train = [r for i, r in enumerate(records) if i not in test_idx]
    test = [r for i, r in enumerate(records) if i in test_idx]
    return train, test


def discover_dataset_records(
    train_dirs: Iterable[str],
    test_dirs: Iterable[str] | None,
    split_fraction: float,
    seed: int,
    recursive: bool,
) -> tuple[list[ImageRecord], list[ImageRecord], bool]:
    train = discover_records([Path(p) for p in train_dirs], recursive=recursive)
    if test_dirs:
        test = discover_records([Path(p) for p in test_dirs], recursive=recursive)
        return train, test, False
    split_train, split_test = split_records(train, split_fraction, seed)
    return split_train, split_test, True


def normalize_feature(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def infer_channel_axis(arr: np.ndarray) -> int | None:
    if arr.ndim == 2:
        return None
    if arr.ndim == 3:
        if arr.shape[-1] <= 4:
            return -1
        if arr.shape[0] <= 4:
            return 0
    return None


def extract_cellpose_style_vector(model: cp_models.CellposeModel, record: ImageRecord, args: argparse.Namespace) -> np.ndarray:
    image = load_frame(record)
    channel_axis = infer_channel_axis(np.asarray(image))
    _masks, _flows, styles = model.eval(
        image,
        batch_size=args.batch_size,
        bsize=args.bsize,
        compute_masks=False,
        channel_axis=channel_axis,
    )
    style = np.asarray(styles)
    if style.ndim > 1:
        style = style.reshape(-1, style.shape[-1]).mean(axis=0)
    return normalize_feature(style)


@torch.no_grad()
def extract_cpsam_neck_vector(
    net: torch.nn.Module,
    record: ImageRecord,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    image = load_frame(record)
    prepared = prepare_image(image, args.bsize, args.channel_mode, args.channel_index)
    feature_map = extract_cpsam_feature_map(net, prepared, device)
    return pooled_feature_vector(feature_map).astype(np.float32, copy=False)


def target_bsize(net: torch.nn.Module, requested: int) -> int:
    encoder = getattr(net, "encoder", None)
    ps = int(getattr(net, "ps", 8))
    pos_embed = getattr(encoder, "pos_embed", None)
    if pos_embed is not None and getattr(pos_embed, "ndim", 0) >= 3:
        return int(pos_embed.shape[1]) * ps
    return requested


def extract_features(
    name: str,
    records: list[ImageRecord],
    model: cp_models.CellposeModel,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    total = len(records)
    for i, record in enumerate(records, start=1):
        try:
            if args.feature_backend == "cellpose-style":
                vec = extract_cellpose_style_vector(model, record, args)
            else:
                vec = extract_cpsam_neck_vector(model.net, record, device, args)
        except Exception as exc:
            print(f"[{name}] skipped {stable_id(record)}: {exc}")
            continue
        vectors.append(vec)
        rows.append(
            {
                "dataset": name,
                "record_id": stable_id(record),
                "path": str(record.path),
                "frame": "" if record.frame_id is None else str(record.frame_id),
                "source": str(record.group_id),
            }
        )
        if i % 50 == 0 or i == total:
            print(f"[{name}] extracted {i}/{total} records")
    if not vectors:
        raise ValueError(f"No valid feature vectors extracted for {name}.")
    return np.stack(vectors, axis=0), rows


def pearson_correlation_matrix(test_vectors: np.ndarray, train_vectors: np.ndarray) -> np.ndarray:
    test = np.asarray(test_vectors, dtype=np.float32)
    train = np.asarray(train_vectors, dtype=np.float32)
    test = test - test.mean(axis=1, keepdims=True)
    train = train - train.mean(axis=1, keepdims=True)
    test = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    return test @ train.T


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(path: Path, corr: np.ndarray, title: str, sort_rows: bool) -> None:
    matrix = corr
    if sort_rows:
        order = np.argsort(np.max(matrix, axis=1))[::-1]
        matrix = matrix[order]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("training images")
    ax.set_ylabel("test images")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_distribution(path: Path, summaries: list[dict[str, object]]) -> None:
    names = sorted({str(row["dataset"]) for row in summaries})
    values = [
        [float(row["max_train_correlation"]) for row in summaries if row["dataset"] == name]
        for name in names
    ]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.boxplot(values, labels=names, showfliers=False)
    for i, vals in enumerate(values, start=1):
        if vals:
            x = np.full(len(vals), i, dtype=np.float32)
            jitter = np.random.default_rng(3).normal(0, 0.035, size=len(vals))
            ax.scatter(x + jitter, vals, s=8, alpha=0.35)
    ax.set_ylabel("maximum train-image correlation")
    ax.set_title("Nearest training-set feature correlation")
    ax.set_ylim(-1, 1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyze_dataset(
    dataset_name: str,
    train_records: list[ImageRecord],
    test_records: list[ImageRecord],
    model: cp_models.CellposeModel,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    train_records = sample_records(train_records, args.max_train_images, args.seed)
    test_records = sample_records(test_records, args.max_test_images, args.seed + 1)
    print(f"{dataset_name}: using {len(train_records)} train and {len(test_records)} test records")

    train_vectors, train_rows = extract_features(f"{dataset_name}:train", train_records, model, device, args)
    test_vectors, test_rows = extract_features(f"{dataset_name}:test", test_records, model, device, args)
    corr = pearson_correlation_matrix(test_vectors, train_vectors)

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "train_vectors.npy", train_vectors)
    np.save(dataset_dir / "test_vectors.npy", test_vectors)
    np.save(dataset_dir / "test_vs_train_correlation.npy", corr)
    write_csv(dataset_dir / "train_records.csv", train_rows)
    write_csv(dataset_dir / "test_records.csv", test_rows)
    plot_heatmap(
        dataset_dir / "test_vs_train_correlation_heatmap.png",
        corr,
        f"{dataset_name}: test vs train feature correlation",
        args.sort_heatmap,
    )

    summary_rows: list[dict[str, object]] = []
    for i, row in enumerate(test_rows):
        values = corr[i]
        summary_rows.append(
            {
                "dataset": dataset_name,
                "record_id": row["record_id"],
                "path": row["path"],
                "frame": row["frame"],
                "source": row["source"],
                "max_train_correlation": float(values.max()),
                "mean_train_correlation": float(values.mean()),
                "median_train_correlation": float(np.median(values)),
            }
        )
    write_csv(dataset_dir / "test_correlation_summary.csv", summary_rows)
    dataset_summary = {
        "dataset": dataset_name,
        "n_train": int(train_vectors.shape[0]),
        "n_test": int(test_vectors.shape[0]),
        "feature_dim": int(train_vectors.shape[1]),
        "mean_max_train_correlation": float(np.mean([r["max_train_correlation"] for r in summary_rows])),
        "median_max_train_correlation": float(np.median([r["max_train_correlation"] for r in summary_rows])),
        "mean_mean_train_correlation": float(np.mean([r["mean_train_correlation"] for r in summary_rows])),
    }
    return summary_rows, dataset_summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"feature backend: {args.feature_backend}")
    print(f"model: {args.model}")
    print(f"device: {device}")

    model = cp_models.CellposeModel(gpu=device.type == "cuda", pretrained_model=args.model)
    model.net.to(device)
    model.net.eval()
    if args.feature_backend == "cpsam-neck":
        args.bsize = target_bsize(model.net, args.bsize)
        print(f"using CPSAM feature extraction bsize={args.bsize}")
    elif model.net.__class__.__name__.lower() == "transformer":
        print(
            "WARNING: this repository's CPSAM Transformer returns random style "
            "vectors. Prefer --feature-backend cpsam-neck for CPSAM models."
        )

    datasets = []
    for name, train_dirs, test_dirs, seed_offset in (
        (args.dataset_a_name, args.dataset_a_train_dirs, args.dataset_a_test_dirs, 0),
        (args.dataset_b_name, args.dataset_b_train_dirs, args.dataset_b_test_dirs, 1000),
    ):
        train_records, test_records, split_generated = discover_dataset_records(
            train_dirs,
            test_dirs,
            args.split_fraction,
            args.seed + seed_offset,
            args.recursive,
        )
        datasets.append((name, train_records, test_records, split_generated))
        print(
            f"{name}: discovered {len(train_records)} train and {len(test_records)} "
            f"test records ({'generated split' if split_generated else 'explicit split'})"
        )

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "feature_backend": args.feature_backend,
        "model": args.model,
        "device": str(device),
        "bsize": args.bsize,
        "datasets": [],
    }
    for name, train_records, test_records, split_generated in datasets:
        rows, summary = analyze_dataset(name, train_records, test_records, model, device, args, output_dir)
        all_rows.extend(rows)
        summaries.append(summary)
        manifest["datasets"].append(
            {
                **summary,
                "split_generated": split_generated,
                "train_dirs": [
                    str(p)
                    for p in (args.dataset_a_train_dirs if name == args.dataset_a_name else args.dataset_b_train_dirs)
                ],
                "test_dirs": [
                    str(p)
                    for p in (
                        args.dataset_a_test_dirs
                        if name == args.dataset_a_name
                        else args.dataset_b_test_dirs
                    )
                    or []
                ],
            }
        )

    write_csv(output_dir / "combined_test_correlation_summary.csv", all_rows)
    write_csv(output_dir / "dataset_summary.csv", summaries)
    plot_distribution(output_dir / "max_train_correlation_comparison.png", all_rows)
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
