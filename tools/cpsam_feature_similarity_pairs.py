"""Explore CPSAM feature-space similarity for same-source and different-source image pairs.

This module extracts CPSAM encoder-neck features from images, pools each
``(256, 32, 32)`` feature map into one 256-dimensional vector, and samples image
pairs from the same inferred source group versus different groups.

The feature tensor is the same representation produced in ``cellpose/vit_sam.py``:
patch embedding -> positional embedding -> transformer blocks -> ``encoder.neck``.
For 256x256 inputs and patch size 8, the feature map is ``(B, 256, 32, 32)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellpose import models as cp_models
from cellpose import transforms
from scripts.eval_semantic_inst_seg import iter_eval_frames


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".nd2", ".lif", ".nrrd"}
DERIVED_SUBSTRINGS = ("_masks", "_classes", "_flows", "_seg", "_pred")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    frame_id: str | None
    group_id: str
    label: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="cpsam")
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--same-pairs", type=int, default=100)
    parser.add_argument("--different-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--channel-mode", choices=("first3", "mean", "channel"), default="first3")
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def is_base_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
        return False
    stem = path.stem.lower()
    return not any(token in stem for token in DERIVED_SUBSTRINGS)


def infer_group_id(path: Path, frame_id: str | None = None) -> str:
    """Infer an acquisition/name-pattern group for same-source pairing.

    Multi-frame records from the same stack share the image stem. For separate
    files, numeric runs are collapsed so names like ``sample_xy01_tile_03`` and
    ``sample_xy02_tile_08`` group together without merging unrelated folders.
    """
    stem = path.stem.lower()
    stem = re.sub(r"\d+", "#", stem)
    stem = re.sub(r"[_\-]+$", "", stem)
    return str(path.parent.resolve() / stem)


def discover_records(image_dirs: Sequence[str | Path], recursive: bool = True) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for root_like in image_dirs:
        root = Path(root_like)
        paths = root.rglob("*") if recursive else root.glob("*")
        for path in sorted(paths):
            if not is_base_image(path):
                continue
            for frame_id, _arr in iter_eval_frames(path):
                group_id = infer_group_id(path, frame_id)
                suffix = f"::{frame_id}" if frame_id else ""
                records.append(ImageRecord(path=path, frame_id=frame_id, group_id=group_id, label=f"{path}{suffix}"))
    return records


def load_frame(record: ImageRecord) -> np.ndarray:
    for frame_id, arr in iter_eval_frames(record.path):
        if frame_id == record.frame_id:
            return np.asarray(arr)
    raise ValueError(f"Could not load frame {record.frame_id!r} from {record.path}")


def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
        return np.moveaxis(arr, 0, -1)
    if arr.ndim == 3:
        return arr
    while arr.ndim > 3:
        arr = arr[0]
    return _to_channels_last(arr)


def prepare_image(arr: np.ndarray, bsize: int = 256, channel_mode: str = "first3", channel_index: int = 0) -> np.ndarray:
    arr = _to_channels_last(arr).astype(np.float32, copy=False)
    if channel_mode == "mean":
        arr = arr.mean(axis=-1, keepdims=True)
    elif channel_mode == "channel":
        idx = min(max(0, int(channel_index)), arr.shape[-1] - 1)
        arr = arr[..., idx : idx + 1]
    else:
        arr = arr[..., : min(3, arr.shape[-1])]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 2:
        arr = np.concatenate([arr, arr[..., :1]], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = transforms.normalize_img(arr, invert=False)
    if arr.shape[0] != bsize or arr.shape[1] != bsize:
        pil_channels = []
        for c in range(arr.shape[-1]):
            ch = arr[..., c]
            ch_min = float(np.nanmin(ch))
            ch_max = float(np.nanmax(ch))
            if ch_max > ch_min:
                ch = (ch - ch_min) / (ch_max - ch_min)
            pil = Image.fromarray(np.clip(ch * 255.0, 0, 255).astype(np.uint8))
            pil = pil.resize((bsize, bsize), Image.Resampling.BILINEAR)
            pil_channels.append(np.asarray(pil).astype(np.float32) / 255.0)
        arr = np.stack(pil_channels, axis=-1)
    return arr.astype(np.float32, copy=False)


def extract_cpsam_feature_map(net, image_256: np.ndarray, device: torch.device) -> np.ndarray:
    """Return CPSAM neck features as ``(256, 32, 32)`` for a 256x256 input."""
    tensor = torch.from_numpy(image_256).permute(2, 0, 1).unsqueeze(0).float().to(device)
    net.eval()
    with torch.no_grad():
        if getattr(net, "student_encoder", None) is not None:
            features = net.student_encoder(tensor)
        else:
            enc = net.encoder
            x = enc.patch_embed(tensor)
            if enc.pos_embed is not None:
                x = x + enc.pos_embed
            for blk in enc.blocks:
                x = blk(x)
            features = enc.neck(x.permute(0, 3, 1, 2))
    return features.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def pooled_feature_vector(feature_map: np.ndarray) -> np.ndarray:
    vec = np.asarray(feature_map, dtype=np.float32).mean(axis=(1, 2))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(vec_a, vec_b) / denom)


def stable_id(record: ImageRecord) -> str:
    text = f"{record.path.resolve()}::{record.frame_id or ''}"
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def sample_pairs(records: Sequence[ImageRecord], n_same: int, n_different: int, seed: int) -> list[tuple[str, int, int]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        groups.setdefault(record.group_id, []).append(idx)

    same_candidates = [(a, b) for idxs in groups.values() if len(idxs) > 1 for n, a in enumerate(idxs) for b in idxs[n + 1 :]]
    different_candidates = [
        (i, j)
        for i in range(len(records))
        for j in range(i + 1, len(records))
        if records[i].group_id != records[j].group_id
    ]
    pairs: list[tuple[str, int, int]] = []
    for label, candidates, n_pairs in (("same_source", same_candidates, n_same), ("different_source", different_candidates, n_different)):
        if not candidates:
            continue
        chosen = rng.choice(len(candidates), size=min(n_pairs, len(candidates)), replace=False)
        pairs.extend((label, *candidates[int(i)]) for i in chosen)
    rng.shuffle(pairs)
    return pairs


def save_preview(path: Path, image_256: np.ndarray) -> None:
    arr = np.asarray(image_256)
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def run_analysis(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.image_dirs, recursive=args.recursive)
    if args.max_images and args.max_images > 0:
        rng = np.random.default_rng(args.seed)
        if len(records) > args.max_images:
            keep = sorted(rng.choice(len(records), size=args.max_images, replace=False).tolist())
            records = [records[i] for i in keep]
    if len(records) < 2:
        raise ValueError("Need at least two image records for pair similarity analysis.")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = cp_models.CellposeModel(gpu=device.type == "cuda", pretrained_model=args.model)
    net = model.net.to(device)
    target_bsize = int(args.bsize)
    if getattr(net, "student_encoder", None) is None and hasattr(net, "encoder") and getattr(net.encoder, "pos_embed", None) is not None:
        target_bsize = int(net.encoder.pos_embed.shape[1] * net.ps)
    if target_bsize != int(args.bsize):
        print(f"using model positional-embedding input size {target_bsize} instead of requested --bsize {args.bsize}")

    features: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        image = prepare_image(load_frame(record), bsize=target_bsize, channel_mode=args.channel_mode, channel_index=args.channel_index)
        feature_map = extract_cpsam_feature_map(net, image, device)
        vec = pooled_feature_vector(feature_map)
        rid = stable_id(record)
        features[rid] = vec
        preview_path = preview_dir / f"{rid}.png"
        save_preview(preview_path, image)
        metadata_rows.append({
            "record_id": rid,
            "path": str(record.path),
            "frame_id": record.frame_id or "",
            "group_id": record.group_id,
            "preview": str(preview_path),
            "feature_map_shape": "x".join(map(str, feature_map.shape)),
            "feature_vector_dim": int(vec.shape[0]),
        })
        if idx % 25 == 0:
            print(f"extracted features for {idx}/{len(records)} images")

    record_by_id = {stable_id(record): record for record in records}
    pairs = sample_pairs(records, args.same_pairs, args.different_pairs, args.seed)
    pair_rows: list[dict[str, object]] = []
    for pair_type, i, j in pairs:
        ra, rb = records[i], records[j]
        ida, idb = stable_id(ra), stable_id(rb)
        pair_rows.append({
            "pair_type": pair_type,
            "record_a": ida,
            "record_b": idb,
            "path_a": str(record_by_id[ida].path),
            "frame_a": record_by_id[ida].frame_id or "",
            "path_b": str(record_by_id[idb].path),
            "frame_b": record_by_id[idb].frame_id or "",
            "group_a": record_by_id[ida].group_id,
            "group_b": record_by_id[idb].group_id,
            "preview_a": str(preview_dir / f"{ida}.png"),
            "preview_b": str(preview_dir / f"{idb}.png"),
            "cosine_similarity": cosine_similarity(features[ida], features[idb]),
        })

    write_csv(output_dir / "cpsam_feature_records.csv", metadata_rows)
    write_csv(output_dir / "cpsam_feature_pair_similarity.csv", pair_rows)
    np.savez_compressed(output_dir / "cpsam_feature_vectors.npz", **features)
    summary = {
        "model": args.model,
        "device": str(device),
        "n_records": len(records),
        "n_pairs": len(pair_rows),
        "feature_map_shape": metadata_rows[0]["feature_map_shape"] if metadata_rows else None,
        "feature_vector_dim": metadata_rows[0]["feature_vector_dim"] if metadata_rows else None,
        "source_lines": {
            "cellpose/vit_sam.py": "Transformer.forward lines 62-78 produce encoder.patch_embed -> pos_embed -> encoder.blocks -> encoder.neck",
            "scripts/extract_cpsam_features.py": "extract_encoder_features lines 72-90 implement the same encoder-only path",
        },
    }
    (output_dir / "cpsam_feature_similarity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'cpsam_feature_pair_similarity.csv'}")
    print(f"feature maps are {summary['feature_map_shape']}; pooled vectors are {summary['feature_vector_dim']}D")


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
    run_analysis(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
