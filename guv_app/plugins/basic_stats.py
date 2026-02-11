import logging
import numpy as np
import pandas as pd
from guv_app.plugins.interface import AnalysisPlugin

_logger = logging.getLogger(__name__)


class BasicStatsPlugin(AnalysisPlugin):
    """
    Calculates basic per-mask intensity and morphology statistics.
    """
    @property
    def name(self) -> str:
        return "Basic Stats"

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if masks is None or masks.max() == 0:
            return pd.DataFrame()

        intensity_img = image
        if intensity_img is None:
            return pd.DataFrame()

        try:
            intensity_img = _match_intensity_to_masks(intensity_img, masks)
        except ValueError as exc:
            _logger.warning(f"Basic Stats: {exc}")
            return pd.DataFrame()

        labels = masks.astype(np.int32, copy=False)
        max_label = int(labels.max())
        if max_label <= 0:
            return pd.DataFrame()

        flat_labels = labels.ravel()
        flat_intensity = intensity_img.ravel()
        area = np.bincount(flat_labels, minlength=max_label + 1)
        intensity_sum = np.bincount(flat_labels, weights=flat_intensity, minlength=max_label + 1)
        mean_intensity = np.zeros_like(intensity_sum, dtype=np.float64)
        nonzero = area > 0
        mean_intensity[nonzero] = intensity_sum[nonzero] / area[nonzero]

        padded = np.pad(labels, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        boundary = (center != up) | (center != down) | (center != left) | (center != right)
        boundary_labels = center[boundary]
        perimeter = np.bincount(boundary_labels, minlength=max_label + 1).astype(np.float64)

        mask_ids = np.arange(1, max_label + 1, dtype=np.int32)
        df = pd.DataFrame({
            "mask_id": mask_ids,
            "area": area[1:],
            "perimeter": perimeter[1:],
            "mean_intensity": mean_intensity[1:],
        })

        if classes is not None and "mask_id" in df.columns:
            def get_class(mask_id):
                return int(classes[mask_id]) if mask_id < len(classes) else 0
            df["class_id"] = df["mask_id"].apply(get_class)

        return df


def _match_intensity_to_masks(intensity_img: np.ndarray, masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(intensity_img)
    if arr.shape == masks.shape:
        return arr
    arr = np.squeeze(arr)
    if arr.shape == masks.shape:
        return arr

    if masks.ndim == 2:
        if arr.ndim >= 3:
            shape = arr.shape
            yx_axes = []
            for idx, dim in enumerate(shape):
                if dim in masks.shape:
                    yx_axes.append(idx)
            if len(yx_axes) >= 2:
                a0, a1 = yx_axes[0], yx_axes[1]
                if (shape[a0], shape[a1]) != masks.shape and (shape[a1], shape[a0]) == masks.shape:
                    arr = np.moveaxis(arr, (a0, a1), (0, 1))
                else:
                    arr = np.moveaxis(arr, (a0, a1), (0, 1))
                while arr.ndim > 2:
                    arr = arr.mean(axis=-1)
        if arr.ndim == 2 and arr.shape != masks.shape:
            raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
        return arr

    if masks.ndim == 3:
        if arr.ndim == 3 and arr.shape[:2] == masks.shape[1:]:
            arr = arr.mean(axis=-1)
            arr = arr[np.newaxis, ...]
        if arr.ndim >= 4:
            shape = arr.shape
            zyx_axes = []
            for idx, dim in enumerate(shape):
                if dim in masks.shape:
                    zyx_axes.append(idx)
            if len(zyx_axes) >= 3:
                arr = np.moveaxis(arr, zyx_axes[:3], (0, 1, 2))
                while arr.ndim > 3:
                    arr = arr.mean(axis=-1)
        if arr.shape != masks.shape:
            raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
        return arr

    raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
