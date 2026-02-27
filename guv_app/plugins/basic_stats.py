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

    def get_parameter_definitions(self):
        return {
            "intensity_channel": {
                "type": "enum",
                "default": "all",
                "options": ["all", "1", "2", "3"],
                "label": "Intensity Channel",
                "help": "'all' averages channels. 1/2/3 selects R/G/B channel.",
            }
        }

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if masks is None or masks.max() == 0:
            return pd.DataFrame()

        intensity_img = image
        if intensity_img is None:
            return pd.DataFrame()

        intensity_channel = _parse_intensity_channel(kwargs.get("intensity_channel", "all"))
        channel_name = _format_intensity_channel_name(intensity_channel)
        try:
            intensity_img = _match_intensity_to_masks(
                intensity_img,
                masks,
                intensity_channel=intensity_channel,
            )
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
            "intensity_channel_name": channel_name,
        })

        if classes is not None and "mask_id" in df.columns:
            def get_class(mask_id):
                return int(classes[mask_id]) if mask_id < len(classes) else 0
            df["class_id"] = df["mask_id"].apply(get_class)

        return df


def _match_intensity_to_masks(
    intensity_img: np.ndarray,
    masks: np.ndarray,
    intensity_channel: int = -1,
) -> np.ndarray:
    arr = np.asarray(intensity_img)
    if masks.ndim == 3 and masks.shape[0] == 1 and arr.ndim == 2 and arr.shape == masks.shape[1:]:
        return arr[np.newaxis, ...]
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
                arr = _collapse_non_spatial_axes(arr, target_ndim=2, intensity_channel=intensity_channel)
        if arr.ndim == 2 and arr.shape != masks.shape:
            raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
        return arr

    if masks.ndim == 3:
        if arr.ndim == 3 and arr.shape[:2] == masks.shape[1:]:
            arr = _select_channel_axis_last(arr, intensity_channel=intensity_channel)
            arr = arr[np.newaxis, ...]
        if arr.ndim >= 4:
            shape = arr.shape
            zyx_axes = []
            for idx, dim in enumerate(shape):
                if dim in masks.shape:
                    zyx_axes.append(idx)
            if len(zyx_axes) >= 3:
                arr = np.moveaxis(arr, zyx_axes[:3], (0, 1, 2))
                arr = _collapse_non_spatial_axes(arr, target_ndim=3, intensity_channel=intensity_channel)
        if arr.shape != masks.shape:
            raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
        return arr

    raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")


def _select_channel_axis_last(arr: np.ndarray, intensity_channel: int) -> np.ndarray:
    if intensity_channel < 0:
        return arr.mean(axis=-1)
    if intensity_channel >= arr.shape[-1]:
        raise ValueError(
            f"Requested intensity_channel={intensity_channel}, but channel axis has size {arr.shape[-1]}"
        )
    return arr[..., intensity_channel]


def _collapse_non_spatial_axes(
    arr: np.ndarray,
    target_ndim: int,
    intensity_channel: int,
) -> np.ndarray:
    if arr.ndim <= target_ndim:
        return arr
    if intensity_channel >= 0:
        arr = _select_channel_axis_last(arr, intensity_channel=intensity_channel)
    while arr.ndim > target_ndim:
        arr = arr.mean(axis=-1)
    return arr


def _parse_intensity_channel(value) -> int:
    if value is None:
        return -1
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("all", ""):
            return -1
        if v in ("1", "2", "3"):
            return int(v) - 1
        try:
            return int(v)
        except ValueError as exc:
            raise ValueError(f"Invalid intensity_channel value: {value}") from exc
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Invalid intensity_channel value: {value}") from exc


def _format_intensity_channel_name(channel_index: int) -> str:
    if channel_index < 0:
        return "all"
    return f"ch{channel_index + 1}"
