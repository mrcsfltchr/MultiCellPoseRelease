#!/usr/bin/env python
"""
Utility functions for loading CPSAM encoder features.

Use this module to load pre-extracted CPSAM features during knowledge
distillation training.

Example usage:
    from scripts.load_cpsam_features import CPSAMFeatureLoader

    # Initialize loader
    loader = CPSAMFeatureLoader("data/cpsam_features")

    # Load features for a specific image
    features = loader.load_features("image_001")  # Returns (256, H, W) array

    # Get all available image IDs
    image_ids = loader.get_image_ids()

    # Iterate through all features
    for image_id, features, metadata in loader:
        print(f"{image_id}: {features.shape}")
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CPSAMFeatureLoader:
    """Loader for pre-extracted CPSAM encoder features."""

    def __init__(self, feature_dir: str):
        """Initialize feature loader.

        Args:
            feature_dir: Directory containing extracted features and manifest.json
        """
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        # Load manifest
        manifest_path = self.feature_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_path}. "
                "Did you run extract_cpsam_features.py?"
            )

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Create lookup dict: image_id -> feature metadata
        self._feature_map = {
            item["image_id"]: item
            for item in self.manifest["features"]
        }

        logger.info(
            f"Loaded feature manifest: {len(self._feature_map)} images, "
            f"feature_dim={self.manifest['feature_dim']}"
        )

    def get_image_ids(self) -> List[str]:
        """Get list of all available image IDs.

        Returns:
            List of image IDs (without extensions)
        """
        return list(self._feature_map.keys())

    def get_feature_metadata(self, image_id: str) -> Dict:
        """Get metadata for a specific image's features.

        Args:
            image_id: Image identifier (filename without extension)

        Returns:
            Dict with keys: image_path, original_shape, feature_shape, feature_file
        """
        if image_id not in self._feature_map:
            raise KeyError(f"No features found for image_id: {image_id}")
        return self._feature_map[image_id]

    def load_features(
        self,
        image_id: str,
        return_metadata: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, Dict]:
        """Load pre-extracted features for a specific image.

        Args:
            image_id: Image identifier (filename without extension)
            return_metadata: If True, also return metadata dict

        Returns:
            features: (256, H, W) encoder features
            metadata: (optional) Dict with original_shape, feature_shape, etc.
        """
        metadata = self.get_feature_metadata(image_id)

        # Load features from .npz file
        feature_path = self.feature_dir / metadata["feature_file"]
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        data = np.load(feature_path)
        features = data["features"]  # (256, H, W)

        if return_metadata:
            return features, metadata
        return features

    def load_batch(
        self,
        image_ids: List[str],
        return_metadata: bool = False
    ) -> List[np.ndarray] | Tuple[List[np.ndarray], List[Dict]]:
        """Load features for multiple images.

        Args:
            image_ids: List of image identifiers
            return_metadata: If True, also return list of metadata dicts

        Returns:
            features_list: List of (256, H, W) feature arrays
            metadata_list: (optional) List of metadata dicts
        """
        features_list = []
        metadata_list = []

        for image_id in image_ids:
            if return_metadata:
                features, metadata = self.load_features(image_id, return_metadata=True)
                features_list.append(features)
                metadata_list.append(metadata)
            else:
                features = self.load_features(image_id)
                features_list.append(features)

        if return_metadata:
            return features_list, metadata_list
        return features_list

    def __len__(self) -> int:
        """Return number of available feature sets."""
        return len(self._feature_map)

    def __iter__(self):
        """Iterate through all features.

        Yields:
            image_id: Image identifier
            features: (256, H, W) encoder features
            metadata: Metadata dict
        """
        for image_id in self.get_image_ids():
            features, metadata = self.load_features(image_id, return_metadata=True)
            yield image_id, features, metadata

    def __repr__(self):
        return (
            f"CPSAMFeatureLoader(feature_dir={self.feature_dir}, "
            f"num_images={len(self)}, feature_dim={self.manifest['feature_dim']})"
        )


def verify_features(feature_dir: str, num_samples: int = 5) -> bool:
    """Verify that extracted features can be loaded correctly.

    Args:
        feature_dir: Directory containing extracted features
        num_samples: Number of random samples to verify (default: 5)

    Returns:
        True if verification passes, False otherwise
    """
    logger.info(f"Verifying features in {feature_dir}")

    try:
        loader = CPSAMFeatureLoader(feature_dir)
        logger.info(f"  ✓ Manifest loaded: {len(loader)} images")

        # Check a few samples
        image_ids = loader.get_image_ids()
        num_samples = min(num_samples, len(image_ids))

        for i in range(num_samples):
            image_id = image_ids[i]
            features, metadata = loader.load_features(image_id, return_metadata=True)

            # Verify shape
            assert features.ndim == 3, f"Expected 3D features, got {features.ndim}D"
            assert features.shape[0] == 256, f"Expected 256 channels, got {features.shape[0]}"

            logger.info(
                f"  ✓ {image_id}: {features.shape} (original: {metadata['original_shape']})"
            )

        logger.info("Verification passed!")
        return True

    except Exception as exc:
        logger.error(f"Verification failed: {exc}")
        return False


if __name__ == "__main__":
    """Simple CLI for verifying extracted features."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Verify CPSAM extracted features"
    )
    parser.add_argument(
        "feature_dir",
        help="Directory containing extracted features"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of samples to verify (default: 5)"
    )

    args = parser.parse_args()

    success = verify_features(args.feature_dir, args.num_samples)
    sys.exit(0 if success else 1)
