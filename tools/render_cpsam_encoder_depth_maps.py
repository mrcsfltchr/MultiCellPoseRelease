"""Render CPSAM/GUVPose flow and cell-probability maps at encoder depths.

This is a diagnostic for ViT encoder depth. For each selected image and model,
the script runs:

    patch_embed -> pos_embed -> first N encoder blocks -> encoder.neck -> Cellpose head

for the requested depths, plus the model's normal final output. It writes raw
output maps and PNG previews for flow and cell probability.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellpose import models as cp_models
from scripts.eval_semantic_inst_seg import iter_eval_frames
from tools.cpsam_feature_similarity_pairs import prepare_image


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".nd2", ".lif", ".nrrd"}
DERIVED_SUBSTRINGS = ("_masks", "_mask", "_classes", "_class", "_flows", "_seg", "_pred")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dirs", nargs="+", default=[r"X:\home\FoundationTrain"])
    parser.add_argument("--output-dir", default="paper/encoder_depth_diagnostics")
    parser.add_argument("--models", nargs="+", default=["cpsam", "guvpose"])
    parser.add_argument("--depths", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=19)
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


def discover_images(roots: Sequence[str | Path], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for root_like in roots:
        root = Path(root_like)
        iterator = root.rglob("*") if recursive else root.glob("*")
        paths.extend(path for path in iterator if is_base_image(path))
    return sorted(set(paths))


def read_first_frame(path: Path) -> np.ndarray:
    for _frame_id, arr in iter_eval_frames(path):
        return np.asarray(arr)
    raise ValueError(f"No readable frame found in {path}")


def target_bsize_for_net(net, requested_bsize: int = 256) -> int:
    if getattr(net, "student_encoder", None) is None and hasattr(net, "encoder") and getattr(net.encoder, "pos_embed", None) is not None:
        return int(net.encoder.pos_embed.shape[1] * net.ps)
    return int(requested_bsize)


def output_after_depth(net, tensor: torch.Tensor, depth: int | None) -> torch.Tensor:
    """Return raw Cellpose output maps as (C, H, W).

    ``depth=None`` uses the model's standard full forward path.
    """
    net.eval()
    with torch.no_grad():
        if depth is None:
            out = net(tensor)[0]
            return out.squeeze(0).detach().cpu()

        if getattr(net, "student_encoder", None) is not None:
            raise ValueError("Intermediate depths are not defined for attached student encoders.")
        enc = net.encoder
        x = enc.patch_embed(tensor)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed
        n_blocks = len(enc.blocks)
        depth = max(0, min(int(depth), n_blocks))
        for blk in enc.blocks[:depth]:
            x = blk(x)
        features = enc.neck(x.permute(0, 3, 1, 2))
        out = net.out(features)
        out = F.conv_transpose2d(out, net.W2, stride=net.ps, padding=0)
        return out.squeeze(0).detach().cpu()


def robust_norm(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    p0, p1 = np.percentile(finite, [lo, hi])
    if p1 <= p0:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - p0) / (p1 - p0), 0, 1)


def save_gray(path: Path, arr: np.ndarray) -> None:
    img = (robust_norm(arr) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_flow_rgb(path: Path, flow_y: np.ndarray, flow_x: np.ndarray) -> None:
    angle = np.arctan2(flow_y, flow_x)
    mag = np.sqrt(flow_y**2 + flow_x**2)
    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = robust_norm(mag)
    rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=-1))
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    out = np.zeros(hsv.shape, dtype=np.float32)
    choices = [
        np.stack([v, t, p], axis=-1),
        np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1),
        np.stack([v, p, q], axis=-1),
    ]
    for idx, choice in enumerate(choices):
        out[i == idx] = choice[i == idx]
    return np.clip(out, 0, 1)


def safe_stem(path: Path, index: int) -> str:
    return f"{index:02d}_{path.stem}".replace(" ", "_").replace(".", "_")


def save_input_preview(path: Path, image: np.ndarray) -> None:
    arr = np.clip(np.asarray(image), 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(args.train_dirs, args.recursive)
    if not images:
        raise ValueError(f"No training images found in {args.train_dirs}")
    rng = np.random.default_rng(args.seed)
    if len(images) > args.n_images:
        images = [images[i] for i in sorted(rng.choice(len(images), size=args.n_images, replace=False).tolist())]
    print(f"selected {len(images)} images")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    manifest_rows: list[dict[str, object]] = []
    for model_name in args.models:
        print(f"loading {model_name} on {device}")
        model = cp_models.CellposeModel(gpu=device.type == "cuda", pretrained_model=model_name)
        net = model.net.to(device)
        bsize = target_bsize_for_net(net)
        n_blocks = len(net.encoder.blocks) if hasattr(net, "encoder") and hasattr(net.encoder, "blocks") else 0
        valid_depths = [depth for depth in args.depths if depth <= n_blocks]
        if len(valid_depths) != len(args.depths):
            print(f"{model_name}: using depths {valid_depths}; model has {n_blocks} encoder blocks")

        for image_index, image_path in enumerate(images, start=1):
            image = prepare_image(
                read_first_frame(image_path),
                bsize=bsize,
                channel_mode=args.channel_mode,
                channel_index=args.channel_index,
            )
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
            stem = safe_stem(image_path, image_index)
            image_out_dir = output_dir / model_name / stem
            image_out_dir.mkdir(parents=True, exist_ok=True)
            save_input_preview(image_out_dir / "input_256.png", image)

            for depth_label, depth in [(f"block{d:02d}", d) for d in valid_depths] + [("final", None)]:
                maps = output_after_depth(net, tensor, depth).numpy()
                if maps.shape[0] < 3:
                    raise ValueError(f"Expected at least 3 output channels, got {maps.shape}")
                flow_y, flow_x, cellprob = maps[0], maps[1], maps[2]
                np.savez_compressed(
                    image_out_dir / f"{depth_label}_raw_maps.npz",
                    output_maps=maps.astype(np.float32, copy=False),
                    flow_y=flow_y.astype(np.float32, copy=False),
                    flow_x=flow_x.astype(np.float32, copy=False),
                    cellprob=cellprob.astype(np.float32, copy=False),
                    depth=-1 if depth is None else int(depth),
                )
                save_flow_rgb(image_out_dir / f"{depth_label}_flow_rgb.png", flow_y, flow_x)
                save_gray(image_out_dir / f"{depth_label}_flow_magnitude.png", np.sqrt(flow_y**2 + flow_x**2))
                save_gray(image_out_dir / f"{depth_label}_cellprob.png", cellprob)
                manifest_rows.append({
                    "model": model_name,
                    "image": str(image_path),
                    "output_dir": str(image_out_dir),
                    "depth_label": depth_label,
                    "depth": "final" if depth is None else int(depth),
                    "n_encoder_blocks": n_blocks,
                    "raw_maps": str(image_out_dir / f"{depth_label}_raw_maps.npz"),
                    "flow_rgb": str(image_out_dir / f"{depth_label}_flow_rgb.png"),
                    "flow_magnitude": str(image_out_dir / f"{depth_label}_flow_magnitude.png"),
                    "cellprob": str(image_out_dir / f"{depth_label}_cellprob.png"),
                })
            print(f"rendered {model_name}: {image_path.name}")

    manifest_path = output_dir / "encoder_depth_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
