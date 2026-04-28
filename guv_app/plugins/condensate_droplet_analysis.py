import logging

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from guv_app.plugins.interface import AnalysisPlugin
from guv_app.plugins.validator import validate_visualization_mask

_logger = logging.getLogger(__name__)


class CondensateDropletAnalysisPlugin(AnalysisPlugin):
    """
    Classifies droplets as homogeneous or condensate-containing and measures
    condensate area plus full-droplet per-channel mean intensities.
    """

    @property
    def name(self) -> str:
        return "Condensate Droplet Analysis"

    def get_parameter_definitions(self):
        return {
            "condensate_detection_channel": {
                "type": "enum",
                "default": "all",
                "options": ["all", "1", "2", "3"],
                "label": "Detection Channel",
                "help": (
                    "'all' uses the mean of all fluorescence channels for condensate "
                    "detection. 1/2/3 selects one channel."
                ),
            },
            "threshold_mad": {
                "type": "float",
                "default": 4.0,
                "min": 0.5,
                "max": 20.0,
                "label": "MAD Threshold",
                "help": "Candidate condensates are pixels this many robust sigma above the droplet median.",
            },
            "threshold_percentile": {
                "type": "float",
                "default": 90.0,
                "min": 50.0,
                "max": 99.9,
                "label": "Percentile Floor",
                "help": "Candidate threshold also must exceed this within-droplet percentile.",
            },
            "min_condensate_area_px": {
                "type": "int",
                "default": 8,
                "min": 1,
                "max": 100000,
                "label": "Min Area (px)",
                "help": "Minimum connected candidate area to count as a condensate.",
            },
            "min_condensate_fraction": {
                "type": "float",
                "default": 0.001,
                "min": 0.0,
                "max": 1.0,
                "label": "Min Area Fraction",
                "help": "Minimum condensate area divided by droplet area for classification.",
            },
            "max_circularity": {
                "type": "float",
                "default": 0.85,
                "min": 0.0,
                "max": 1.0,
                "label": "Max Circularity",
                "help": "Rejects round bright objects; lower values require more irregular condensate shapes.",
            },
            "edge_exclusion_px": {
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 50,
                "label": "Edge Exclusion (px)",
                "help": "Pixels excluded inside the droplet boundary during condensate detection.",
            },
            "visualization_mode": {
                "type": "enum",
                "default": "accepted_condensates",
                "options": ["accepted_condensates", "threshold_candidates"],
                "label": "Visualization Mode",
                "help": (
                    "accepted_condensates shows regions that pass area, fraction, and "
                    "irregularity filters. threshold_candidates shows bright thresholded "
                    "pixels before connected-component filtering."
                ),
            },
        }

    def run(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        classes: np.ndarray = None,
        **kwargs,
    ) -> pd.DataFrame:
        if image is None or masks is None or np.asarray(masks).max() == 0:
            return pd.DataFrame()

        labels = np.squeeze(np.asarray(masks)).astype(np.int32, copy=False)
        if labels.ndim > 2:
            labels = labels[0]
        if labels.ndim != 2 or int(labels.max()) <= 0:
            return pd.DataFrame()

        try:
            channels = _extract_channels(image, labels)
        except ValueError as exc:
            _logger.warning("Condensate Droplet Analysis: %s", exc)
            return pd.DataFrame()

        try:
            config = _parse_config(kwargs, channels)
        except ValueError as exc:
            _logger.warning("Condensate Droplet Analysis: %s", exc)
            return pd.DataFrame()

        flat_labels = labels.ravel()
        max_label = int(labels.max())
        area = np.bincount(flat_labels, minlength=max_label + 1)
        mask_ids = np.flatnonzero(area[1:] > 0).astype(np.int32) + 1
        visualization_masks = kwargs.get("visualization_masks")
        if visualization_masks is not None:
            try:
                condensate_area_by_id = _condensate_area_from_visualization(
                    visualization_masks, labels, max_label
                )
            except ValueError as exc:
                _logger.warning("Condensate Droplet Analysis: %s", exc)
                return pd.DataFrame()
        else:
            detection = _detect_condensates_fast(labels, config)
            condensate_area_by_id = detection["condensate_area_by_id"]
        channel_means = _channel_means_by_label(channels, flat_labels, area, max_label)

        rows = []
        for mask_id in mask_ids:
            droplet_area = int(area[mask_id])
            condensate_area = int(condensate_area_by_id[mask_id])
            condensate_fraction = condensate_area / droplet_area if droplet_area else 0.0
            has_condensate = (
                condensate_area >= config["min_area_px"]
                and condensate_fraction >= config["min_fraction"]
            )

            row = {
                "droplet_id": int(mask_id),
                "classification": "condensate" if has_condensate else "homogeneous",
                "contains_condensate": bool(has_condensate),
                "droplet_area": droplet_area,
                "condensate_area": condensate_area if has_condensate else 0,
                "condensate_area_fraction": condensate_fraction if has_condensate else 0.0,
            }
            for c_idx, means in enumerate(channel_means):
                row[f"droplet_mean_intensity_ch{c_idx + 1}"] = float(means[mask_id])
            if len(channels) > 1:
                row["droplet_mean_intensity"] = float(
                    np.nanmean([row[f"droplet_mean_intensity_ch{i + 1}"] for i in range(len(channels))])
                )
            else:
                row["droplet_mean_intensity"] = row["droplet_mean_intensity_ch1"]

            if classes is not None:
                row["class_id"] = int(classes[mask_id]) if mask_id < len(classes) else 0
            rows.append(row)

        return pd.DataFrame(rows)

    def visualize(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        classes: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        if image is None or masks is None or np.asarray(masks).max() == 0:
            return None

        labels = np.squeeze(np.asarray(masks)).astype(np.int32, copy=False)
        if labels.ndim > 2:
            labels = labels[0]
        if labels.ndim != 2 or int(labels.max()) <= 0:
            return None

        try:
            channels = _extract_channels(image, labels)
            config = _parse_config(kwargs, channels)
        except ValueError as exc:
            _logger.warning("Condensate Droplet Analysis visualization: %s", exc)
            return None

        mode = str(kwargs.get("visualization_mode", "accepted_condensates")).strip().lower()
        max_label = int(labels.max())
        area = np.bincount(labels.ravel(), minlength=max_label + 1)
        detection = _detect_condensates_fast(labels, config)

        if mode == "threshold_candidates":
            show = detection["candidate_mask"]
        else:
            condensate_area = detection["condensate_area_by_id"]
            fraction = np.zeros(max_label + 1, dtype=np.float64)
            valid = area > 0
            fraction[valid] = condensate_area[valid] / area[valid]
            keep_label = (
                (condensate_area >= config["min_area_px"])
                & (fraction >= config["min_fraction"])
            )
            show = detection["accepted_mask"] & keep_label[labels]

        out = np.where(show, labels, 0).astype(np.int32, copy=False)

        validate_visualization_mask(out, masks)
        return out


def _extract_channels(image: np.ndarray, labels_2d: np.ndarray) -> list:
    img = np.asarray(image, dtype=np.float64)
    labels_shape = labels_2d.shape
    img_sq = np.squeeze(img)

    if img_sq.shape == labels_shape:
        return [img_sq]

    if img_sq.ndim == 3:
        if img_sq.shape[:2] == labels_shape:
            return [img_sq[:, :, c] for c in range(img_sq.shape[2])]
        if img_sq.shape[1:] == labels_shape:
            return [img_sq[c] for c in range(img_sq.shape[0])]

    arr = img_sq
    while arr.ndim > 2:
        arr = arr.mean(axis=-1)
    if arr.shape == labels_shape:
        return [arr]

    raise ValueError(f"Cannot extract channels: image {image.shape} vs masks {labels_2d.shape}")


def _parse_detection_channel(value) -> int:
    if value is None:
        return -1
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("all", ""):
            return -1
        if v in ("1", "2", "3"):
            return int(v) - 1
        return int(v)
    return int(value)


def _make_detection_image(channels: list, detection_channel: int) -> np.ndarray:
    if detection_channel < 0:
        return np.nanmean(np.stack(channels, axis=0), axis=0)
    if detection_channel >= len(channels):
        raise ValueError(
            f"Requested condensate_detection_channel={detection_channel + 1}, "
            f"but image has {len(channels)} channel(s)"
        )
    return channels[detection_channel]


def _parse_config(kwargs, channels: list) -> dict:
    detection_channel = _parse_detection_channel(
        kwargs.get("condensate_detection_channel", "all")
    )
    detection_img = _make_detection_image(channels, detection_channel)
    threshold_percentile = min(
        99.9, max(50.0, float(kwargs.get("threshold_percentile", 90.0)))
    )
    return {
        "detection_img": detection_img,
        "threshold_mad": float(kwargs.get("threshold_mad", 4.0)),
        "threshold_percentile": threshold_percentile,
        "min_area_px": max(1, int(kwargs.get("min_condensate_area_px", 8))),
        "min_fraction": max(0.0, float(kwargs.get("min_condensate_fraction", 0.001))),
        "max_circularity": min(1.0, max(0.0, float(kwargs.get("max_circularity", 0.85)))),
        "edge_exclusion_px": max(0, int(kwargs.get("edge_exclusion_px", 1))),
    }


def _channel_means_by_label(channels: list, flat_labels: np.ndarray, area: np.ndarray, max_label: int) -> list:
    valid = area > 0
    means_by_channel = []
    for ch_img in channels:
        values = np.asarray(ch_img, dtype=np.float64).ravel()
        finite = np.isfinite(values)
        weighted_values = np.where(finite, values, 0.0)
        sums = np.bincount(flat_labels, weights=weighted_values, minlength=max_label + 1)
        counts = np.bincount(flat_labels, weights=finite.astype(np.float64), minlength=max_label + 1)
        means = np.full(max_label + 1, np.nan, dtype=np.float64)
        usable = valid & (counts > 0)
        means[usable] = sums[usable] / counts[usable]
        means_by_channel.append(means)
    return means_by_channel


def _condensate_area_from_visualization(
    visualization_masks: np.ndarray,
    labels: np.ndarray,
    max_label: int,
) -> np.ndarray:
    viz = np.squeeze(np.asarray(visualization_masks)).astype(np.int32, copy=False)
    if viz.ndim > 2:
        viz = viz[0]
    validate_visualization_mask(viz, labels)
    if viz.shape != labels.shape:
        if viz.size == labels.size:
            viz = viz.reshape(labels.shape)
        else:
            raise ValueError(
                f"Visualization mask shape {viz.shape} does not match labels shape {labels.shape}"
            )

    # Only count pixels that are still inside the same parent droplet. This
    # protects edited visualization masks from contributing to the wrong label.
    accepted = (viz > 0) & (labels == viz)
    return np.bincount(viz[accepted].ravel(), minlength=max_label + 1)


def _detect_condensates_fast(labels: np.ndarray, config: dict) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    detection_img = np.asarray(config["detection_img"], dtype=np.float64)
    max_label = int(labels.max())

    droplet_pixels = labels > 0
    finite = np.isfinite(detection_img)
    roi = _global_detection_roi(labels, config["edge_exclusion_px"])
    roi &= finite

    threshold, median = _thresholds_by_label(
        labels,
        detection_img,
        roi,
        max_label,
        config["threshold_mad"],
        config["threshold_percentile"],
    )
    threshold_at_pixel = threshold[labels]
    median_at_pixel = median[labels]
    use_inclusive = threshold_at_pixel > median_at_pixel
    candidate = roi & np.isfinite(threshold_at_pixel) & (
        (use_inclusive & (detection_img >= threshold_at_pixel))
        | (~use_inclusive & (detection_img > threshold_at_pixel))
    )
    candidate &= droplet_pixels

    component_labels, component_count = ndi.label(candidate)
    if component_count == 0:
        empty = np.zeros_like(labels, dtype=bool)
        return {
            "candidate_mask": empty,
            "accepted_mask": empty,
            "condensate_area_by_id": np.zeros(max_label + 1, dtype=np.int64),
        }

    component_area = np.bincount(component_labels.ravel(), minlength=component_count + 1)
    component_perimeter = _component_perimeters(component_labels, component_count)
    circularity = np.ones(component_count + 1, dtype=np.float64)
    has_perimeter = component_perimeter > 0
    circularity[has_perimeter] = (
        4.0
        * np.pi
        * component_area[has_perimeter]
        / (component_perimeter[has_perimeter] * component_perimeter[has_perimeter])
    )
    keep_component = (
        (component_area >= config["min_area_px"])
        & (circularity <= config["max_circularity"])
    )
    keep_component[0] = False
    accepted = keep_component[component_labels]
    condensate_area_by_id = np.bincount(
        labels[accepted].ravel(),
        minlength=max_label + 1,
    )

    return {
        "candidate_mask": candidate,
        "accepted_mask": accepted,
        "condensate_area_by_id": condensate_area_by_id,
    }


def _global_detection_roi(labels: np.ndarray, edge_exclusion_px: int) -> np.ndarray:
    droplet_pixels = labels > 0
    if edge_exclusion_px <= 0:
        return droplet_pixels.copy()

    dist_inside = ndi.distance_transform_edt(droplet_pixels)
    roi = droplet_pixels & (dist_inside > edge_exclusion_px)

    max_label = int(labels.max())
    original_counts = np.bincount(labels.ravel(), minlength=max_label + 1)
    roi_counts = np.bincount(labels[roi].ravel(), minlength=max_label + 1)
    empty_after_erosion = (original_counts > 0) & (roi_counts == 0)
    if empty_after_erosion.any():
        roi |= empty_after_erosion[labels]
    return roi


def _thresholds_by_label(
    labels: np.ndarray,
    detection_img: np.ndarray,
    roi: np.ndarray,
    max_label: int,
    threshold_mad: float,
    threshold_percentile: float,
) -> np.ndarray:
    flat_labels = labels[roi].ravel()
    values = detection_img[roi].ravel()
    median = _label_percentile_nearest(flat_labels, values, max_label, 50.0)

    median_at_sample = median[flat_labels]
    abs_dev = np.abs(values - median_at_sample)
    mad = _label_percentile_nearest(flat_labels, abs_dev, max_label, 50.0)
    percentile_floor = _label_percentile_nearest(
        flat_labels, values, max_label, threshold_percentile
    )

    counts = np.bincount(flat_labels, minlength=max_label + 1).astype(np.float64)
    sums = np.bincount(flat_labels, weights=values, minlength=max_label + 1)
    sums_sq = np.bincount(flat_labels, weights=values * values, minlength=max_label + 1)
    std = np.zeros(max_label + 1, dtype=np.float64)
    valid = counts > 0
    mean = np.zeros(max_label + 1, dtype=np.float64)
    mean[valid] = sums[valid] / counts[valid]
    variance = np.zeros(max_label + 1, dtype=np.float64)
    variance[valid] = np.maximum((sums_sq[valid] / counts[valid]) - mean[valid] ** 2, 0.0)
    std[valid] = np.sqrt(variance[valid])

    robust_threshold = np.full(max_label + 1, np.inf, dtype=np.float64)
    has_mad = np.isfinite(mad) & (mad > 0)
    robust_threshold[has_mad] = median[has_mad] + threshold_mad * 1.4826 * mad[has_mad]
    flat_but_variable = (~has_mad) & np.isfinite(median) & (std > 0)
    robust_threshold[flat_but_variable] = median[flat_but_variable]

    return np.maximum(robust_threshold, percentile_floor), median


def _label_percentile_nearest(
    labels_1d: np.ndarray,
    values_1d: np.ndarray,
    max_label: int,
    percentile: float,
) -> np.ndarray:
    out = np.full(max_label + 1, np.nan, dtype=np.float64)
    if labels_1d.size == 0:
        return out

    finite = (labels_1d > 0) & np.isfinite(values_1d)
    if not finite.any():
        return out

    sample_labels = labels_1d[finite].astype(np.int32, copy=False)
    sample_values = values_1d[finite].astype(np.float64, copy=False)
    order = np.lexsort((sample_values, sample_labels))
    sorted_labels = sample_labels[order]
    sorted_values = sample_values[order]

    counts = np.bincount(sorted_labels, minlength=max_label + 1)
    valid_ids = np.flatnonzero(counts > 0)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    q = np.clip(float(percentile) / 100.0, 0.0, 1.0)
    offsets = np.rint((counts[valid_ids] - 1) * q).astype(np.int64)
    out[valid_ids] = sorted_values[starts[valid_ids] + offsets]
    return out


def _component_perimeters(component_labels: np.ndarray, component_count: int) -> np.ndarray:
    padded = np.pad(component_labels, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1]
    boundary = (center > 0) & (
        (center != padded[:-2, 1:-1])
        | (center != padded[2:, 1:-1])
        | (center != padded[1:-1, :-2])
        | (center != padded[1:-1, 2:])
    )
    return np.bincount(center[boundary].ravel(), minlength=component_count + 1)
