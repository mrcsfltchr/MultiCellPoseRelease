"""Create a small no-download RGB image set using torchvision FakeData.

PyTorch/torchvision do not bundle real ImageNet images. ``torchvision.datasets``
does include ``FakeData``, which produces deterministic PIL RGB images with the
same kind of tensor/image interface as torchvision datasets and requires no
network access. These images are useful as a control set for code-path and
feature-space diagnostics, but they should not be described as natural-image or
ImageNet examples.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from torchvision.datasets import FakeData
from torchvision.transforms import ToPILImage


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper/pytorch_compatible_fake_images")
    parser.add_argument("--n-images", type=int, default=12)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--random-offset", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FakeData(
        size=int(args.n_images),
        image_size=(int(args.channels), int(args.height), int(args.width)),
        num_classes=int(args.num_classes),
        random_offset=int(args.random_offset),
        transform=None,
        target_transform=None,
    )
    to_pil = ToPILImage()
    rows = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        if not hasattr(image, "save"):
            image = to_pil(image)
        path = output_dir / f"torchvision_fake_{index:03d}_class_{int(label):04d}.png"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        image.save(path)
        rows.append({
            "index": index,
            "label": int(label),
            "path": str(path),
            "note": "torchvision.datasets.FakeData synthetic RGB image; not real ImageNet",
        })

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "label", "path", "note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} synthetic torchvision-compatible images to {output_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
