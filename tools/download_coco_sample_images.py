"""Download a small fixed set of COCO validation images for diagnostics.

The images are pulled from the public COCO 2017 validation image host. This is
intended only as a compact natural-image control set for feature diagnostics,
not as a full COCO benchmark.
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path
from typing import Sequence


COCO_VAL2017_IDS = [
    "000000000139",
    "000000000285",
    "000000000632",
    "000000000724",
    "000000000785",
    "000000000802",
    "000000000872",
    "000000000885",
    "000000001000",
    "000000001268",
    "000000001296",
    "000000001353",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper/coco_sample_images")
    parser.add_argument("--n-images", type=int, default=len(COCO_VAL2017_IDS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = COCO_VAL2017_IDS[: max(0, min(int(args.n_images), len(COCO_VAL2017_IDS)))]

    rows = []
    for image_id in image_ids:
        url = f"http://images.cocodataset.org/val2017/{image_id}.jpg"
        path = output_dir / f"coco_val2017_{image_id}.jpg"
        if path.exists() and not args.overwrite:
            print(f"keeping existing {path}")
        else:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        rows.append({"image_id": image_id, "url": url, "path": str(path)})

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "url", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} COCO sample images to {output_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
