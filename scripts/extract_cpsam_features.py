#!/usr/bin/env python
"""
Extract CPSAM encoder features for knowledge distillation.

This script runs images through the CPSAM encoder and saves the 256-channel
encoder features (after the neck, before the segmentation head) for later use
in knowledge distillation training.

Example usage:
    python scripts/extract_cpsam_features.py \
        --image-dir data/foundation_cpsam/train \
        --output-dir data/cpsam_features \
        --model cpsam \
        --batch-size 8 \
        --bsize 256

Output:
    - data/cpsam_features/features_<image_id>.npz  (one per image)
    - data/cpsam_features/manifest.json  (metadata for all features)
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from cellpose import io, transforms
from cellpose.models import CellposeModel

logger = logging.getLogger(__name__)


def extract_encoder_features(
    model_net,
    image: np.ndarray,
    bsize: int = 256,
    batch_size: int = 8,
    device: torch.device = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Extract 256-channel encoder features from CPSAM model.

    Args:
        model_net: The CPSAM network (Transformer)
        image: Input image (H, W) or (H, W, C)
        bsize: Tile size for processing
        batch_size: Number of tiles to process at once
        device: Torch device

    Returns:
        features: (256, H, W) encoder features
        feature_shape: (H, W) shape of features
    """
    if device is None:
        device = next(model_net.parameters()).device

    # Normalize and prepare image
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.shape[-1] < 3:
        # Pad to 3 channels if grayscale
        image = np.repeat(image, 3, axis=-1)

    # Normalize
    img = transforms.normalize_img(image, invert=False)

    # Convert to torch tensor (C, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Extract features (run through encoder only, skip segmentation head)
    model_net.eval()
    with torch.no_grad():
        # Run through patch embedding
        x = model_net.encoder.patch_embed(img_tensor)

        # Add positional embeddings
        if model_net.encoder.pos_embed is not None:
            x = x + model_net.encoder.pos_embed

        # Run through all encoder blocks
        for blk in model_net.encoder.blocks:
            x = blk(x)

        # Apply neck to get 256-channel features
        # neck expects (B, H, W, C) → outputs (B, 256, H, W)
        features = model_net.encoder.neck(x.permute(0, 3, 1, 2))

    # Convert to numpy (256, H, W)
    features = features.squeeze(0).cpu().numpy()

    return features, features.shape[1:]


def process_images(
    model: CellposeModel,
    image_files: List[Path],
    output_dir: Path,
    bsize: int = 256,
    batch_size: int = 8,
) -> List[Dict]:
    """Process all images and extract features.

    Args:
        model: CellposeModel instance
        image_files: List of image file paths
        output_dir: Directory to save features
        bsize: Tile size for processing
        batch_size: Batch size for processing

    Returns:
        List of feature metadata dicts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest = []

    for img_path in image_files:
        logger.info(f"Processing {img_path.name}...")

        try:
            # Load image
            img = io.imread(str(img_path))
            original_shape = img.shape[:2]  # (H, W)

            # Extract features
            features, feature_shape = extract_encoder_features(
                model.net,
                img,
                bsize=bsize,
                batch_size=batch_size,
                device=model.device,
            )

            # Save features
            image_id = img_path.stem  # filename without extension
            feature_file = f"features_{image_id}.npz"
            feature_path = output_dir / feature_file

            np.savez_compressed(
                str(feature_path),
                features=features,
                image_id=image_id,
                original_shape=original_shape,
            )

            # Add to manifest
            feature_manifest.append({
                "image_id": image_id,
                "image_path": str(img_path),
                "original_shape": list(original_shape),
                "feature_shape": list(feature_shape),
                "feature_file": feature_file,
            })

            logger.info(
                f"  Saved features: {features.shape} → {feature_file}"
            )

        except Exception as exc:
            logger.error(f"  Failed to process {img_path.name}: {exc}")
            continue

    return feature_manifest


def save_manifest(
    manifest: List[Dict],
    output_dir: Path,
    model_name: str,
    feature_dim: int = 256,
):
    """Save feature extraction manifest.

    Args:
        manifest: List of feature metadata dicts
        output_dir: Directory containing features
        model_name: Name of the model used for extraction
        feature_dim: Feature dimensionality (default: 256 for CPSAM)
    """
    manifest_data = {
        "model": model_name,
        "feature_dim": feature_dim,
        "num_images": len(manifest),
        "features": manifest,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Saved manifest: {manifest_path} ({len(manifest)} images)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract CPSAM encoder features for knowledge distillation"
    )
    p.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing images to process"
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save extracted features"
    )
    p.add_argument(
        "--model",
        default="cpsam",
        help="CPSAM model to use (default: cpsam)"
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for feature extraction"
    )
    p.add_argument(
        "--bsize",
        type=int,
        default=256,
        help="Tile size for processing (default: 256)"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
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

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = CellposeModel(
        pretrained_model=args.model,
        gpu=args.gpu,
    )

    # Find images
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return 1

    image_files = io.get_image_files(str(image_dir), mask_filter="_seg")
    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return 1

    logger.info(f"Found {len(image_files)} images")

    # Process images
    output_dir = Path(args.output_dir)
    manifest = process_images(
        model,
        [Path(f) for f in image_files],
        output_dir,
        bsize=args.bsize,
        batch_size=args.batch_size,
    )

    # Save manifest
    save_manifest(
        manifest,
        output_dir,
        model_name=args.model,
        feature_dim=256,
    )

    logger.info(f"Feature extraction complete: {len(manifest)} images processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
