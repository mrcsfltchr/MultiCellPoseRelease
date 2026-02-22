#!/usr/bin/env python3
"""
Prepare foundation training dataset from multiple sources.

This script collects data from various subdirectories and formats it into
a unified training directory suitable for CPSAM foundation training, applying
dataset sampling weights similar to the cellpose-SAM paper.

Handles:
- TissueNet .npz files (train, val, test)
- Omnipose bacterial datasets (fluorescent and phase)
- Worm datasets (standard and high-res)
- Mixed file formats (.tif, .png)
- Various label formats (_masks.tif, _masks.png, _seg.npy)

Usage:
    python scripts/prepare_foundation_dataset.py \
        --input-dir X:/home/FoundationTrain \
        --output-dir data/foundation_cpsam \
        --create-train-weights \
        --extract-tissuenet \
        --sample-strategy balanced
"""
import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dataset sampling probabilities from cellpose-SAM paper
# These control how often each dataset is sampled during training
DATASET_SAMPLING_WEIGHTS = {
    "cyto2": 0.59,          # Cellpose cyto2 (most variable)
    "nuclei": 0.20,         # Cellpose nuclei
    "tissuenet": 0.08,      # TissueNet
    "livecell": 0.05,       # LiveCell
    "bact_fluor": 0.01,     # Fluorescent bacteria (Omnipose)
    "bact_phase": 0.02,     # Phase bacteria (Omnipose)
    "yeast_phc": 0.01,      # Phase contrast yeast
    "yeast_bf": 0.02,       # Bright-field yeast
    "deepbacs": 0.02,       # DeepBacs
    "worm": 0.00,           # Custom (not in paper, adjust as needed)
}


def extract_tissuenet_npz(
    npz_path: Path,
    output_dir: Path,
    dataset_name: str = "tissuenet",
    max_images: int = None,
) -> List[Tuple[Path, Path]]:
    """Extract images and masks from TissueNet .npz file.

    Args:
        npz_path: Path to .npz file
        output_dir: Directory to save extracted images
        dataset_name: Prefix for output files
        max_images: Maximum images to extract (None = all)

    Returns:
        List of (image_path, mask_path) tuples
    """
    logger.info(f"Extracting {npz_path.name}...")

    # Load npz file
    data = np.load(str(npz_path))

    # Determine keys (TissueNet typically has 'X' for images, 'y' for masks)
    # But check actual keys in case format differs
    available_keys = list(data.keys())
    logger.info(f"  Keys in npz: {available_keys}")

    # Common key variations
    img_key = None
    mask_key = None

    for key in available_keys:
        key_lower = key.lower()
        if key_lower in ['x', 'images', 'image', 'imgs']:
            img_key = key
        elif key_lower in ['y', 'masks', 'mask', 'labels', 'label']:
            mask_key = key

    if img_key is None or mask_key is None:
        logger.error(f"Could not identify image/mask keys in {npz_path.name}")
        logger.error(f"Available keys: {available_keys}")
        logger.error("Please check the .npz file structure manually")
        return []

    images = data[img_key]
    masks = data[mask_key]

    logger.info(f"  Images shape: {images.shape}, dtype: {images.dtype}")
    logger.info(f"  Masks shape: {masks.shape}, dtype: {masks.dtype}")

    # Limit number of images if requested
    n_images = len(images)
    if max_images is not None and max_images < n_images:
        indices = np.random.choice(n_images, max_images, replace=False)
        images = images[indices]
        masks = masks[indices]
        n_images = max_images

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save images and masks
    from cellpose import io

    pairs = []
    split_name = npz_path.stem.replace("tissuenet_v1.1_", "")  # train/val/test

    for i in tqdm(range(n_images), desc=f"  Extracting {split_name}"):
        # Create filenames
        img_id = f"{dataset_name}__{split_name}_{i:05d}"
        img_path = output_dir / f"{img_id}.tif"
        mask_path = output_dir / f"{img_id}_seg.npy"

        # Get image and mask
        img = images[i]
        mask = masks[i]

        # Handle different dimensionalities
        # TissueNet images are typically (H, W, C) or (H, W)
        # Masks are typically (H, W)

        # Save image
        io.imsave(str(img_path), img)

        # Save mask as _seg.npy (cellpose format)
        np.save(str(mask_path), {'masks': mask})

        pairs.append((img_path, mask_path))

    logger.info(f"  Extracted {len(pairs)} image-mask pairs")
    return pairs


def find_image_mask_pairs(
    directory: Path,
    dataset_name: str,
    recursive: bool = True,
) -> List[Tuple[Path, Path, str]]:
    """Find all image-mask pairs in a directory.

    Args:
        directory: Directory to search
        dataset_name: Dataset identifier
        recursive: Whether to search recursively

    Returns:
        List of (image_path, mask_path, image_id) tuples
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    pairs = []

    # Find all potential image files
    image_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    glob_pattern = "**/*" if recursive else "*"

    potential_images = []
    for ext in image_extensions:
        potential_images.extend(directory.glob(f"{glob_pattern}{ext}"))

    # Filter out mask/flow files
    image_files = [
        f for f in potential_images
        if not any(x in f.stem for x in ["_masks", "_flows", "_seg"])
    ]

    logger.info(f"  Found {len(image_files)} potential images in {directory.name}")

    # Find corresponding masks
    for img_path in image_files:
        mask_path = None

        # Try different mask naming conventions
        mask_candidates = [
            img_path.parent / f"{img_path.stem}_masks.tif",
            img_path.parent / f"{img_path.stem}_masks.png",
            img_path.parent / f"{img_path.stem}_seg.npy",
        ]

        for candidate in mask_candidates:
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is not None:
            # Create unique image ID
            rel_path = img_path.relative_to(directory)
            image_id = f"{dataset_name}__{rel_path.parent.name}_{img_path.stem}"
            image_id = image_id.replace("/", "_").replace("\\", "_")

            pairs.append((img_path, mask_path, image_id))
        else:
            logger.debug(f"  No mask found for {img_path.name}")

    logger.info(f"  Found {len(pairs)} image-mask pairs")
    return pairs


def copy_and_rename(
    pairs: List[Tuple[Path, Path, str]],
    output_dir: Path,
    create_symlinks: bool = False,
) -> int:
    """Copy image-mask pairs to output directory with standardized names.

    Args:
        pairs: List of (image_path, mask_path, image_id) tuples
        output_dir: Output directory
        create_symlinks: If True, create symlinks instead of copying

    Returns:
        Number of pairs copied
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path, mask_path, image_id in tqdm(pairs, desc="  Copying files"):
        try:
            # Determine output paths
            img_ext = img_path.suffix
            mask_ext = mask_path.suffix

            out_img_path = output_dir / f"{image_id}{img_ext}"
            out_mask_path = output_dir / f"{image_id}{mask_ext}"

            # Copy or symlink
            if create_symlinks:
                if not out_img_path.exists():
                    out_img_path.symlink_to(img_path.resolve())
                if not out_mask_path.exists():
                    out_mask_path.symlink_to(mask_path.resolve())
            else:
                if not out_img_path.exists():
                    shutil.copy2(img_path, out_img_path)
                if not out_mask_path.exists():
                    shutil.copy2(mask_path, out_mask_path)

            copied += 1

        except Exception as exc:
            logger.warning(f"Failed to copy {img_path.name}: {exc}")

    return copied


def create_sampling_weights_file(
    dataset_counts: Dict[str, int],
    sampling_weights: Dict[str, float],
    output_path: Path,
):
    """Create a sampling weights file for training.

    Args:
        dataset_counts: Number of images per dataset
        sampling_weights: Sampling probability per dataset
        output_path: Where to save the weights file
    """
    # Calculate per-image sampling weights
    total_images = sum(dataset_counts.values())

    weights_info = {
        "total_images": total_images,
        "dataset_counts": dataset_counts,
        "dataset_sampling_probs": sampling_weights,
        "notes": "Use these weights during training to balance dataset contributions",
    }

    # Calculate per-image weight for each dataset
    per_image_weights = {}
    for dataset, count in dataset_counts.items():
        if count > 0 and dataset in sampling_weights:
            # Weight = (target_prob) / (dataset_fraction)
            dataset_fraction = count / total_images
            target_prob = sampling_weights[dataset]
            per_image_weights[dataset] = target_prob / dataset_fraction
        else:
            per_image_weights[dataset] = 1.0

    weights_info["per_image_weights"] = per_image_weights

    # Save
    with open(output_path, 'w') as f:
        json.dump(weights_info, f, indent=2)

    logger.info(f"Saved sampling weights to {output_path}")
    logger.info("Per-image sampling weights:")
    for dataset, weight in per_image_weights.items():
        logger.info(f"  {dataset}: {weight:.4f}x")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare foundation training dataset from multiple sources"
    )
    p.add_argument(
        "--input-dir",
        default="X:/home/FoundationTrain",
        help="Root directory containing all datasets"
    )
    p.add_argument(
        "--output-dir",
        default="data/foundation_cpsam",
        help="Output directory for prepared dataset"
    )
    p.add_argument(
        "--extract-tissuenet",
        action="store_true",
        help="Extract TissueNet from .npz files (slow, ~2GB)"
    )
    p.add_argument(
        "--tissuenet-max-train",
        type=int,
        default=None,
        help="Max images to extract from TissueNet train set (default: all)"
    )
    p.add_argument(
        "--create-symlinks",
        action="store_true",
        help="Create symlinks instead of copying files (saves space)"
    )
    p.add_argument(
        "--create-train-weights",
        action="store_true",
        help="Create sampling weights file based on cellpose-SAM strategy"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "bact_fluor", "bact_phase", "worm", "worm_high_res", "tissuenet"],
        help="Which datasets to include (default: all)"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Track dataset statistics
    dataset_counts = defaultdict(int)
    all_datasets = args.datasets if "all" not in args.datasets else [
        "bact_fluor", "bact_phase", "worm", "worm_high_res", "tissuenet"
    ]

    # Process each dataset
    logger.info("=" * 60)
    logger.info("Processing datasets...")
    logger.info("=" * 60)

    # 1. Bacterial fluorescent (Omnipose)
    if "bact_fluor" in all_datasets:
        logger.info("\n[1/5] Processing bact_fluor...")
        pairs = find_image_mask_pairs(
            input_dir / "bact_fluor" / "train_sorted",
            "bact_fluor",
            recursive=True
        )
        copied = copy_and_rename(pairs, train_dir, args.create_symlinks)
        dataset_counts["bact_fluor"] = copied

        # Test set
        test_pairs = find_image_mask_pairs(
            input_dir / "bact_fluor" / "test_sorted",
            "bact_fluor",
            recursive=True
        )
        copy_and_rename(test_pairs, test_dir, args.create_symlinks)

    # 2. Bacterial phase (Omnipose)
    if "bact_phase" in all_datasets:
        logger.info("\n[2/5] Processing bact_phase...")
        pairs = find_image_mask_pairs(
            input_dir / "bact_phase" / "train_sorted",
            "bact_phase",
            recursive=True
        )
        copied = copy_and_rename(pairs, train_dir, args.create_symlinks)
        dataset_counts["bact_phase"] = copied

        # Test set
        test_pairs = find_image_mask_pairs(
            input_dir / "bact_phase" / "test_sorted",
            "bact_phase",
            recursive=True
        )
        copy_and_rename(test_pairs, test_dir, args.create_symlinks)

    # 3. Worm dataset
    if "worm" in all_datasets:
        logger.info("\n[3/5] Processing worm...")
        pairs = find_image_mask_pairs(
            input_dir / "worm" / "train",
            "worm",
            recursive=False
        )
        copied = copy_and_rename(pairs, train_dir, args.create_symlinks)
        dataset_counts["worm"] = copied

        # Test set
        test_pairs = find_image_mask_pairs(
            input_dir / "worm" / "test",
            "worm",
            recursive=False
        )
        copy_and_rename(test_pairs, test_dir, args.create_symlinks)

    # 4. Worm high-res dataset
    if "worm_high_res" in all_datasets:
        logger.info("\n[4/5] Processing worm_high_res...")
        pairs = find_image_mask_pairs(
            input_dir / "worm_high_res" / "train",
            "worm_high_res",
            recursive=False
        )
        copied = copy_and_rename(pairs, train_dir, args.create_symlinks)
        dataset_counts["worm_high_res"] = copied

        # Test set
        test_pairs = find_image_mask_pairs(
            input_dir / "worm_high_res" / "test",
            "worm_high_res",
            recursive=False
        )
        copy_and_rename(test_pairs, test_dir, args.create_symlinks)

    # 5. TissueNet (from .npz files)
    if "tissuenet" in all_datasets and args.extract_tissuenet:
        logger.info("\n[5/5] Processing tissuenet...")
        logger.info("  NOTE: This is slow (~5-10 minutes for full dataset)")

        # Extract train set
        train_npz = input_dir / "tissuenet_v1.1_train.npz"
        if train_npz.exists():
            pairs = extract_tissuenet_npz(
                train_npz,
                train_dir,
                "tissuenet",
                max_images=args.tissuenet_max_train
            )
            dataset_counts["tissuenet"] = len(pairs)

        # Extract test set
        test_npz = input_dir / "tissuenet_v1.1_test.npz"
        if test_npz.exists():
            extract_tissuenet_npz(test_npz, test_dir, "tissuenet")

        # Note: validation set can be merged into training or kept separate
        # val_npz = input_dir / "tissuenet_v1.1_val.npz"

    elif "tissuenet" in all_datasets and not args.extract_tissuenet:
        logger.info("\n[5/5] Skipping tissuenet (use --extract-tissuenet to include)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Dataset preparation complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training directory: {train_dir}")
    logger.info(f"Test directory: {test_dir}")
    logger.info("\nDataset counts:")

    total = 0
    for dataset, count in sorted(dataset_counts.items()):
        logger.info(f"  {dataset:20s}: {count:5d} images")
        total += count

    logger.info(f"  {'TOTAL':20s}: {total:5d} images")

    # Create sampling weights file if requested
    if args.create_train_weights and dataset_counts:
        weights_path = output_dir / "sampling_weights.json"
        create_sampling_weights_file(
            dict(dataset_counts),
            DATASET_SAMPLING_WEIGHTS,
            weights_path
        )

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review prepared data: ls {train_dir}")
    logger.info(f"  2. Train foundation model:")
    logger.info(f"     python scripts/train_cpsam_foundation.py \\")
    logger.info(f"         --train-dir {train_dir} \\")
    logger.info(f"         --foundation-training \\")
    logger.info(f"         --epochs 300")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
