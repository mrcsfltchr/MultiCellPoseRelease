"""Convert DisGUVery GUI exports to labelled instance-mask arrays.

DisGUVery exports vesicle detections as CSV files. Hough detections contain
circle centers/radii, template detections contain square-box centers/sizes, and
floodfill detections also export a labelled ``*_mask.tiff`` file. This adapter
renders those outputs into integer-labelled ``*_pred_masks.npy`` files suitable
for standardized evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, help="Directory containing DisGUVery exported CSV files.")
    parser.add_argument("--image-dir", required=True, help="Directory containing the original images.")
    parser.add_argument("--output-dir", required=True, help="Directory for labelled prediction masks.")
    parser.add_argument(
        "--method",
        choices=("auto", "hough", "template", "floodfill"),
        default="auto",
        help="Detection method. 'auto' infers from the CSV header or matching floodfill mask.",
    )
    parser.add_argument("--template-render", choices=("disk", "square"), default="disk")
    parser.add_argument("--recursive", action="store_true", help="Search input folders recursively.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def iter_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    yield from (root.rglob(pattern) if recursive else root.glob(pattern))


def read_image_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim < 2:
        raise ValueError(f"Image has no 2D shape: {path}")
    return int(arr.shape[0]), int(arr.shape[1])


def read_label_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        mask = np.asarray(im)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return relabel(mask.astype(np.int32, copy=False))


def relabel(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for value in np.unique(mask):
        if int(value) == 0:
            continue
        out[mask == value] = next_id
        next_id += 1
    return out


def parse_csv(path: Path) -> tuple[str, np.ndarray]:
    header = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        header = first.lstrip("#").strip().lower()
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.size == 0:
        data = np.zeros((0, 4), dtype=float)
    elif data.ndim == 1:
        data = data.reshape(1, -1)

    if "radius" in header:
        method = "hough"
    elif "matching score" in header or "size" in header:
        method = "template"
    elif "axis major" in header:
        method = "floodfill"
    else:
        method = "hough" if data.shape[1] == 4 else "template"
    return method, data


def strip_disguvery_suffix(stem: str) -> str:
    for suffix in ("_detected_vesicles", "_vesicles", "_detection", "_detections"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_image(image_dir: Path, csv_path: Path, recursive: bool) -> Path:
    base = strip_disguvery_suffix(csv_path.stem)
    candidates = []
    for ext in IMG_EXTS:
        candidates.extend(iter_files(image_dir, base + ext, recursive))
    if not candidates:
        raise FileNotFoundError(f"No original image found for {csv_path.name}; looked for stem {base!r}")
    return sorted(candidates)[0]


def render_disk(mask: np.ndarray, xc: float, yc: float, radius: float, value: int) -> None:
    h, w = mask.shape
    radius = max(0.0, float(radius))
    x0 = max(0, int(np.floor(xc - radius)))
    x1 = min(w, int(np.ceil(xc + radius + 1)))
    y0 = max(0, int(np.floor(yc - radius)))
    y1 = min(h, int(np.ceil(yc + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    region = (xx - float(xc)) ** 2 + (yy - float(yc)) ** 2 <= radius**2
    sub = mask[y0:y1, x0:x1]
    sub[region] = value


def render_square(mask: np.ndarray, xc: float, yc: float, size: float, value: int) -> None:
    h, w = mask.shape
    half = max(0.0, float(size) / 2.0)
    x0 = max(0, int(np.floor(xc - half)))
    x1 = min(w, int(np.ceil(xc + half)))
    y0 = max(0, int(np.floor(yc - half)))
    y1 = min(h, int(np.ceil(yc + half)))
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = value


def render_csv(data: np.ndarray, method: str, shape: tuple[int, int], template_render: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.int32)
    if data.size == 0:
        return mask
    for out_id, row in enumerate(data, start=1):
        if len(row) < 4:
            continue
        xc = float(row[1])
        yc = float(row[2])
        if method == "hough":
            render_disk(mask, xc, yc, float(row[3]), out_id)
        elif method == "template":
            size = float(row[3])
            if template_render == "square":
                render_square(mask, xc, yc, size, out_id)
            else:
                render_disk(mask, xc, yc, size / 2.0, out_id)
        elif method == "floodfill":
            render_disk(mask, xc, yc, float(row[3]) / 2.0, out_id)
        else:
            raise ValueError(f"Unsupported method: {method}")
    return relabel(mask)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_dir = Path(args.export_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    csv_paths = sorted(iter_files(export_dir, "*.csv", args.recursive))
    for csv_path in csv_paths:
        if csv_path.name.endswith(("_radial_profiles.csv", "_angular_profiles.csv")):
            continue
        method, data = parse_csv(csv_path)
        if args.method != "auto":
            method = args.method

        image_path = find_image(image_dir, csv_path, args.recursive)
        output_path = output_dir / f"{strip_disguvery_suffix(csv_path.stem)}_pred_masks.npy"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite")

        mask_file = csv_path.with_name(csv_path.stem + "_mask.tiff")
        if method == "floodfill" and mask_file.exists():
            mask = read_label_mask(mask_file)
        else:
            mask = render_csv(data, method, read_image_shape(image_path), args.template_render)

        np.save(output_path, mask.astype(np.int32, copy=False))
        rows.append({
            "csv": str(csv_path),
            "image": str(image_path),
            "output": str(output_path),
            "method": method,
            "n_instances": int(mask.max()),
        })

    manifest_path = output_dir / "disguvery_adapter_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["csv", "image", "output", "method", "n_instances"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "disguvery_adapter_summary.json").write_text(json.dumps({"n_files": len(rows)}, indent=2))
    print(f"converted {len(rows)} DisGUVery exports")
    print(f"wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
