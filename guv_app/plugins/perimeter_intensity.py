import logging
import numpy as np
import pandas as pd
from guv_app.plugins.interface import AnalysisPlugin
from guv_app.plugins.validator import validate_visualization_mask
from guv_app.plugins.basic_stats import (
    _local_background_values,
    _make_local_background_labels,
    _nanmean_columns,
)
from cellpose.utils import masks_to_edges

_logger = logging.getLogger(__name__)


def _make_band_mask(masks: np.ndarray, thickness: int, direction: str) -> np.ndarray:
    """Return a labeled band mask for the perimeter region.

    direction:
        "symmetric" – band straddles the mask edge (uses masks_to_edges).
        "inward"    – band lies strictly inside each mask, thickness px from edge.
        "outward"   – band lies strictly outside each mask, thickness px from edge.
    """
    from scipy.ndimage import distance_transform_edt

    if direction == "inward":
        # Distance of each foreground pixel to the nearest background pixel.
        # Pixels within [1, thickness] of the boundary form the inner band.
        fg_dist = distance_transform_edt(masks > 0)
        band = (fg_dist > 0) & (fg_dist <= thickness)
        return (masks * band).astype(masks.dtype)

    if direction == "outward":
        # Distance of each background pixel to the nearest foreground pixel,
        # plus the label of that nearest foreground pixel.
        bg_dist, nearest_idx = distance_transform_edt(masks == 0, return_indices=True)
        nearest_labels = masks[tuple(nearest_idx)]
        band = (bg_dist > 0) & (bg_dist <= thickness)
        return (nearest_labels * band).astype(masks.dtype)

    # "symmetric" – default: straddles the mask edge
    edges = masks_to_edges(masks, threshold=thickness)
    return (masks * edges.astype(masks.dtype)).astype(masks.dtype)


def _extract_channels(image: np.ndarray, masks: np.ndarray) -> list:
    """Return a list of arrays (one per channel) each matching masks spatial shape.

    For a single-channel or ambiguous image a list of length 1 is returned.
    The last non-spatial dimension is treated as the channel axis when detected.
    """
    img = np.asarray(image, dtype=np.float64)
    spatial = masks.shape[-2:] if masks.ndim == 3 else masks.shape  # (H,W)

    # Already matches masks shape – single channel
    if img.shape == masks.shape:
        return [img]

    img_sq = np.squeeze(img)
    if img_sq.shape == masks.shape:
        return [img_sq]

    if masks.ndim == 2:
        H, W = spatial
        if img_sq.ndim == 3:
            if img_sq.shape[:2] == (H, W):
                # (H, W, C)
                return [img_sq[:, :, c] for c in range(img_sq.shape[2])]
            if img_sq.shape[1:] == (H, W):
                # (C, H, W)
                return [img_sq[c] for c in range(img_sq.shape[0])]

    if masks.ndim == 3:
        Z, H, W = masks.shape

        # Singleton-3D mask (1, H, W): match against 2D multichannel image layouts.
        # The ravel order of (1,H,W) is identical to (H,W) so bincount aligns correctly.
        if Z == 1:
            if img_sq.ndim == 2 and img_sq.shape == (H, W):
                return [img_sq]
            if img_sq.ndim == 3:
                if img_sq.shape[:2] == (H, W):
                    # (H, W, C)
                    return [img_sq[:, :, c] for c in range(img_sq.shape[2])]
                if img_sq.shape[1:] == (H, W):
                    # (C, H, W)
                    return [img_sq[c] for c in range(img_sq.shape[0])]

        if img_sq.ndim == 4:
            if img_sq.shape[:3] == (Z, H, W):
                # (Z, H, W, C)
                return [img_sq[:, :, :, c] for c in range(img_sq.shape[3])]
            if img_sq.shape[1:] == (Z, H, W):
                # (C, Z, H, W)
                return [img_sq[c] for c in range(img_sq.shape[0])]

    # Fallback: collapse extra dims by mean to produce a single channel
    arr = img_sq
    while arr.ndim > masks.ndim:
        arr = arr.mean(axis=-1)
    if arr.shape == masks.shape:
        return [arr]

    raise ValueError(f"Cannot extract channels: image {image.shape} vs masks {masks.shape}")


def _label_means(perimeter_masks: np.ndarray, channel_img: np.ndarray,
                 mask_ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Compute per-label mean intensity for one channel within perimeter_masks."""
    max_id = int(mask_ids[-1])
    labels_flat = perimeter_masks.astype(np.int64, copy=False).ravel()
    intensities = channel_img.astype(np.float64, copy=False).ravel()
    sums = np.bincount(labels_flat, weights=intensities, minlength=max_id + 1)
    return sums[mask_ids] / counts[mask_ids]


class PerimeterIntensityPlugin(AnalysisPlugin):
    """
    Calculates intensity statistics on the perimeter band of masks.

    Reports per-channel mean intensities (perimeter_mean_intensity_ch0, ch1, …)
    and an overall mean across channels (perimeter_mean_intensity).
    Generates a visualization of the perimeter regions as a mask layer.
    """

    @property
    def name(self) -> str:
        return "Perimeter Intensity"

    def run(self, image: np.ndarray, masks: np.ndarray,
            classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if masks.max() == 0:
            return pd.DataFrame()

        thickness = kwargs.get('thickness', 1)
        direction = kwargs.get('direction', 'symmetric')
        bg_inner_gap = max(0, int(kwargs.get("background_inner_gap_px", 2)))
        bg_outer_radius = max(1, int(kwargs.get("background_outer_radius_px", 12)))
        bg_percentile = float(kwargs.get("background_percentile", 25.0))
        bg_percentile = min(100.0, max(0.0, bg_percentile))
        if bg_outer_radius <= bg_inner_gap:
            bg_outer_radius = bg_inner_gap + 1

        visualization_masks = kwargs.get("visualization_masks")
        if visualization_masks is not None:
            validate_visualization_mask(visualization_masks, masks)
            perimeter_masks = visualization_masks
        else:
            perimeter_masks = self.get_visualization_mask(masks, thickness, direction)

        try:
            channels = _extract_channels(image, masks)
        except ValueError as exc:
            _logger.warning("Perimeter Intensity: %s", exc)
            return pd.DataFrame()

        labels_flat = perimeter_masks.astype(np.int64, copy=False).ravel()
        max_id = int(labels_flat.max()) if labels_flat.size > 0 else 0
        if max_id == 0:
            return pd.DataFrame()

        counts = np.bincount(labels_flat, minlength=max_id + 1)
        mask_ids = np.arange(1, max_id + 1)
        valid = counts[mask_ids] > 0
        mask_ids = mask_ids[valid]
        if mask_ids.size == 0:
            return pd.DataFrame()

        df = pd.DataFrame({"label": mask_ids})
        bg_labels = _make_local_background_labels(masks, bg_inner_gap, bg_outer_radius)

        # Per-channel intensities
        channel_means = []
        bg_sub_channel_means = []
        for c, ch_img in enumerate(channels):
            means_c = _label_means(perimeter_masks, ch_img, mask_ids, counts)
            channel_means.append(means_c)
            bg_values = _local_background_values(
                bg_labels, ch_img, mask_ids, bg_percentile
            )
            bg_sub_c = means_c - bg_values
            bg_sub_channel_means.append(bg_sub_c)
            if len(channels) > 1:
                df[f"perimeter_mean_intensity_ch{c}"] = means_c
                df[f"perimeter_mean_intensity_bg_subtracted_ch{c}"] = bg_sub_c

        # Overall mean across channels
        df["perimeter_mean_intensity"] = np.mean(channel_means, axis=0)
        df["perimeter_mean_intensity_bg_subtracted"] = _nanmean_columns(bg_sub_channel_means)

        if classes is not None:
            def get_class(label):
                return classes[label] if label < len(classes) else 0
            df['class_id'] = df['label'].apply(get_class)

        return df

    def get_visualization_mask(self, masks: np.ndarray, thickness: int = 1,
                                direction: str = 'symmetric') -> np.ndarray:
        """
        Generates a mask array representing the perimeter band of each object.
        Validates that the output is compatible with the original segmentation.
        """
        perimeter_masks = _make_band_mask(masks, thickness, direction)
        validate_visualization_mask(perimeter_masks, masks)
        return perimeter_masks

    def get_parameter_definitions(self):
        return {
            "thickness": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "label": "Perimeter Thickness",
                "help": "Thickness of the perimeter band in pixels",
            },
            "direction": {
                "type": "enum",
                "default": "symmetric",
                "options": ["symmetric", "inward", "outward"],
                "label": "Dilation Direction",
                "help": (
                    "symmetric: band straddles the mask edge. "
                    "inward: band is inside the mask. "
                    "outward: band is outside the mask."
                ),
            },
            "background_inner_gap_px": {
                "type": "int",
                "default": 2,
                "min": 0,
                "max": 50,
                "label": "Background Gap (px)",
                "help": "Pixels to skip between each mask and its local background annulus.",
            },
            "background_outer_radius_px": {
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 200,
                "label": "Background Radius (px)",
                "help": "Outer radius of each local background annulus.",
            },
            "background_percentile": {
                "type": "float",
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "label": "Background Percentile",
                "help": "Robust percentile used as the local background level.",
            },
        }

    def visualize(self, image: np.ndarray, masks: np.ndarray,
                  classes: np.ndarray = None, **kwargs) -> np.ndarray:
        thickness = kwargs.get('thickness', 1)
        direction = kwargs.get('direction', 'symmetric')
        return self.get_visualization_mask(masks, thickness, direction)
