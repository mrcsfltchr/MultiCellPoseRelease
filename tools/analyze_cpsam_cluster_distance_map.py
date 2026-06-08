"""Relate CPSAM feature-cluster train distance to detection mAP.

This script extracts pooled CPSAM encoder features for training and test images,
clusters the test images in that feature space, then computes each test
cluster's nearest cosine distance to the training features:

    min_cosine_distance_to_train = 1 - max cosine_similarity(cluster_centroid, train_image)

It joins the cluster assignments to a per-image evaluator CSV, computes pooled
mAP per cluster, and writes tables for plotting correlation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellpose import models as cp_models
from tools.cpsam_feature_similarity_pairs import (
    ImageRecord,
    discover_records,
    extract_cpsam_feature_map,
    load_frame,
    pooled_feature_vector,
    prepare_image,
    stable_id,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-image-dirs", nargs="+", required=True)
    parser.add_argument("--test-image-dirs", nargs="+", required=True)
    parser.add_argument("--eval-csv", required=True, help="Per-image CSV from evaluate_standardized_test_models.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-model", default="cpsam")
    parser.add_argument("--eval-model", default=None, help="Optional model name to select from a multi-model eval CSV.")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--cluster-method", choices=("cosine-kmeans", "source-group"), default="cosine-kmeans")
    parser.add_argument(
        "--center-train-mean",
        action="store_true",
        help=(
            "Subtract the global mean pooled embedding vector computed from the training set "
            "from both train and test features before clustering/distance analysis."
        ),
    )
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--channel-mode", choices=("first3", "mean", "channel"), default="first3")
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def cosine_kmeans(x: np.ndarray, n_clusters: int, seed: int, max_iter: int = 100) -> np.ndarray:
    x = normalize_rows(np.asarray(x, dtype=np.float32))
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    k = max(1, min(int(n_clusters), n))
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int32)
    for _ in range(max_iter):
        new_labels = np.argmax(x @ centroids.T, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members) == 0:
                centroids[cluster_id] = x[int(rng.integers(0, n))]
            else:
                centroids[cluster_id] = members.mean(axis=0)
        centroids = normalize_rows(centroids)
    return labels


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def rel_to_any_root(path: Path, roots: Sequence[Path]) -> str:
    resolved = path.resolve()
    best = str(resolved)
    for root in roots:
        try:
            rel = str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
        if len(rel) < len(best):
            best = rel
    return best.replace("\\", "/")


def eval_key(image_value: str, frame_value: object = "") -> tuple[str, str]:
    image = str(image_value).replace("\\", "/").lower()
    frame = "" if frame_value is None or (isinstance(frame_value, float) and math.isnan(frame_value)) else str(frame_value)
    if frame.lower() == "nan":
        frame = ""
    return image, frame


def record_eval_key(record: ImageRecord, test_roots: Sequence[Path]) -> tuple[str, str]:
    return eval_key(rel_to_any_root(record.path, test_roots), record.frame_id or "")


def read_eval_rows(path: Path, eval_model: str | None) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        candidates = []
        if path.parent.exists():
            candidates = sorted(path.parent.glob("*per_image*.csv")) or sorted(path.parent.glob("*.csv"))
        message = f"Evaluation CSV not found: {path}"
        if candidates:
            shown = "\n  ".join(str(candidate) for candidate in candidates[:10])
            message += f"\nAvailable candidate CSVs:\n  {shown}"
        raise FileNotFoundError(message)
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if eval_model and row.get("model") != eval_model:
                continue
            key = eval_key(row.get("image", ""), row.get("frame_id", ""))
            rows[key] = row
    return rows


def metric_tags(row: dict[str, str]) -> list[str]:
    tags = []
    for key in row:
        if key.startswith("tp_"):
            tag = key[3:]
            if f"fp_{tag}" in row and f"fn_{tag}" in row:
                tags.append(tag)
    return sorted(tags)


def float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def ap_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 1.0


def discover_limited_records(paths: Sequence[str], recursive: bool, max_records: int, seed: int) -> list[ImageRecord]:
    records = discover_records(paths, recursive=recursive)
    if max_records and max_records > 0 and len(records) > max_records:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(records), size=max_records, replace=False).tolist())
        records = [records[i] for i in keep]
    return records


def target_bsize_for_net(net, requested_bsize: int) -> int:
    if getattr(net, "student_encoder", None) is None and hasattr(net, "encoder") and getattr(net.encoder, "pos_embed", None) is not None:
        return int(net.encoder.pos_embed.shape[1] * net.ps)
    return int(requested_bsize)


def extract_vectors(
    records: Sequence[ImageRecord],
    net,
    device: torch.device,
    bsize: int,
    channel_mode: str,
    channel_index: int,
    label: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    vectors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        image = prepare_image(load_frame(record), bsize=bsize, channel_mode=channel_mode, channel_index=channel_index)
        fmap = extract_cpsam_feature_map(net, image, device)
        vec = pooled_feature_vector(fmap)
        vectors.append(vec)
        rows.append({
            "record_id": stable_id(record),
            "split": label,
            "path": str(record.path),
            "frame_id": record.frame_id or "",
            "group_id": record.group_id,
            "feature_map_shape": "x".join(map(str, fmap.shape)),
            "feature_vector_dim": int(vec.shape[0]),
        })
        if idx % 25 == 0:
            print(f"extracted {label} features for {idx}/{len(records)} images")
    return np.vstack(vectors).astype(np.float32), rows


def assign_clusters(records: Sequence[ImageRecord], vectors: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.cluster_method == "source-group":
        group_to_id: dict[str, int] = {}
        labels = []
        for record in records:
            if record.group_id not in group_to_id:
                group_to_id[record.group_id] = len(group_to_id)
            labels.append(group_to_id[record.group_id])
        return np.asarray(labels, dtype=np.int32)
    return cosine_kmeans(vectors, args.n_clusters, args.seed)


def subtract_train_mean(
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mean = np.asarray(train_vectors, dtype=np.float32).mean(axis=0, keepdims=True)
    return (
        np.asarray(train_vectors, dtype=np.float32) - train_mean,
        np.asarray(test_vectors, dtype=np.float32) - train_mean,
        train_mean.squeeze(0),
    )


def summarize_clusters(
    test_records: Sequence[ImageRecord],
    test_vectors: np.ndarray,
    train_vectors: np.ndarray,
    labels: np.ndarray,
    eval_rows: dict[tuple[str, str], dict[str, str]],
    test_roots: Sequence[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test_vectors = normalize_rows(test_vectors)
    train_vectors = normalize_rows(train_vectors)
    image_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    exemplar_row = next(iter(eval_rows.values())) if eval_rows else {}
    tags = metric_tags(exemplar_row)
    if not tags and "mean_ap" in exemplar_row:
        tags = []

    for record, vec, cluster_id in zip(test_records, test_vectors, labels):
        key = record_eval_key(record, test_roots)
        row = eval_rows.get(key)
        image_rows.append({
            "record_id": stable_id(record),
            "cluster_id": int(cluster_id),
            "image": key[0],
            "frame_id": key[1],
            "path": str(record.path),
            "group_id": record.group_id,
            "has_eval": row is not None,
            "mean_ap": float_value(row, "mean_ap", float("nan")) if row else float("nan"),
        })

    for cluster_id in sorted(np.unique(labels).astype(int)):
        member_idx = np.where(labels == cluster_id)[0]
        centroid = normalize_rows(test_vectors[member_idx].mean(axis=0, keepdims=True))[0]
        max_cos = float(np.max(train_vectors @ centroid))
        cluster_eval = [eval_rows.get(record_eval_key(test_records[i], test_roots)) for i in member_idx]
        cluster_eval = [row for row in cluster_eval if row is not None]

        pooled_aps = []
        pooled_metrics: dict[str, object] = {}
        for tag in tags:
            tp = sum(float_value(row, f"tp_{tag}") for row in cluster_eval)
            fp = sum(float_value(row, f"fp_{tag}") for row in cluster_eval)
            fn = sum(float_value(row, f"fn_{tag}") for row in cluster_eval)
            ap = ap_from_counts(tp, fp, fn)
            pooled_metrics[f"tp_{tag}"] = int(tp)
            pooled_metrics[f"fp_{tag}"] = int(fp)
            pooled_metrics[f"fn_{tag}"] = int(fn)
            pooled_metrics[f"ap_{tag}"] = ap
            pooled_aps.append(ap)
        mean_image_ap = float(np.nanmean([float_value(row, "mean_ap", float("nan")) for row in cluster_eval])) if cluster_eval else float("nan")
        cluster_rows.append({
            "cluster_id": int(cluster_id),
            "n_test_images": int(len(member_idx)),
            "n_eval_images": int(len(cluster_eval)),
            "nearest_train_cosine_similarity": max_cos,
            "min_cosine_distance_to_train": 1.0 - max_cos,
            "pooled_map": float(np.mean(pooled_aps)) if pooled_aps else mean_image_ap,
            "mean_image_map": mean_image_ap,
            **pooled_metrics,
        })
    return image_rows, cluster_rows


def cluster_centroid_distance_rows(
    test_vectors: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, object]]:
    test_vectors = normalize_rows(test_vectors)
    centroids: dict[int, np.ndarray] = {}
    for cluster_id in sorted(np.unique(labels).astype(int)):
        member_idx = np.where(labels == cluster_id)[0]
        centroid = normalize_rows(test_vectors[member_idx].mean(axis=0, keepdims=True))[0]
        centroids[int(cluster_id)] = centroid

    rows: list[dict[str, object]] = []
    cluster_ids = sorted(centroids)
    for i, cluster_a in enumerate(cluster_ids):
        for cluster_b in cluster_ids[i + 1 :]:
            similarity = float(np.dot(centroids[cluster_a], centroids[cluster_b]))
            rows.append({
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "cosine_similarity": similarity,
                "cosine_distance": 1.0 - similarity,
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = discover_limited_records(args.train_image_dirs, args.recursive, args.max_train_images, args.seed)
    test_records = discover_limited_records(args.test_image_dirs, args.recursive, args.max_test_images, args.seed + 1)
    if not train_records:
        raise ValueError("No training image records found.")
    if not test_records:
        raise ValueError("No test image records found.")
    print(f"train records: {len(train_records)}")
    print(f"test records: {len(test_records)}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = cp_models.CellposeModel(gpu=device.type == "cuda", pretrained_model=args.feature_model)
    net = model.net.to(device)
    bsize = target_bsize_for_net(net, args.bsize)
    if bsize != args.bsize:
        print(f"using model positional-embedding input size {bsize} instead of requested --bsize {args.bsize}")

    train_vectors, train_meta = extract_vectors(train_records, net, device, bsize, args.channel_mode, args.channel_index, "train")
    test_vectors, test_meta = extract_vectors(test_records, net, device, bsize, args.channel_mode, args.channel_index, "test")
    train_mean_norm = None
    train_mean_vector = np.zeros(train_vectors.shape[1], dtype=np.float32)
    if args.center_train_mean:
        train_vectors, test_vectors, train_mean_vector = subtract_train_mean(train_vectors, test_vectors)
        train_mean_norm = float(np.linalg.norm(train_mean_vector))
        print(f"subtracted training-set mean embedding vector; mean norm before subtraction={train_mean_norm:.6f}")
    labels = assign_clusters(test_records, test_vectors, args)
    eval_rows = read_eval_rows(Path(args.eval_csv), args.eval_model)
    image_rows, cluster_rows = summarize_clusters(
        test_records,
        test_vectors,
        train_vectors,
        labels,
        eval_rows,
        [Path(p) for p in args.test_image_dirs],
    )
    centroid_distance_rows = cluster_centroid_distance_rows(test_vectors, labels)

    write_csv(output_dir / "feature_records.csv", train_meta + test_meta)
    write_csv(output_dir / "test_image_cluster_assignments.csv", image_rows)
    write_csv(output_dir / "cluster_distance_vs_map.csv", cluster_rows)
    write_csv(output_dir / "cluster_centroid_cosine_distances.csv", centroid_distance_rows)
    np.savez_compressed(
        output_dir / "feature_vectors.npz",
        train=train_vectors,
        test=test_vectors,
        test_cluster_labels=labels,
        train_mean_embedding=train_mean_vector,
        centered_on_train_mean=np.asarray([bool(args.center_train_mean)]),
    )

    valid = [row for row in cluster_rows if int(row["n_eval_images"]) > 0 and not math.isnan(float(row["pooled_map"]))]
    distances = np.asarray([float(row["min_cosine_distance_to_train"]) for row in valid], dtype=float)
    maps = np.asarray([float(row["pooled_map"]) for row in valid], dtype=float)
    summary = {
        "feature_model": args.feature_model,
        "eval_model": args.eval_model,
        "cluster_method": args.cluster_method,
        "center_train_mean": bool(args.center_train_mean),
        "train_mean_embedding_norm": train_mean_norm,
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "n_clusters": len(cluster_rows),
        "n_clusters_with_eval": len(valid),
        "pearson_distance_vs_pooled_map": pearson(distances, maps),
        "spearman_distance_vs_pooled_map": spearman(distances, maps),
        "output_files": {
            "cluster_table": "cluster_distance_vs_map.csv",
            "cluster_centroid_distances": "cluster_centroid_cosine_distances.csv",
            "image_assignments": "test_image_cluster_assignments.csv",
            "feature_records": "feature_records.csv",
            "feature_vectors": "feature_vectors.npz",
        },
    }
    (output_dir / "cluster_distance_vs_map_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'cluster_distance_vs_map.csv'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
