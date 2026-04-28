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
                "help": (
                    "'all' saves mean_intensity (cross-channel mean) plus "
                    "mean_intensity_ch1, mean_intensity_ch2, ... columns. "
                    "1/2/3 selects a specific channel only."
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

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if masks is None or np.asarray(masks).max() == 0:
            return pd.DataFrame()
        if image is None:
            return pd.DataFrame()

        intensity_channel = _parse_intensity_channel(kwargs.get("intensity_channel", "all"))
        bg_inner_gap = max(0, int(kwargs.get("background_inner_gap_px", 2)))
        bg_outer_radius = max(1, int(kwargs.get("background_outer_radius_px", 12)))
        bg_percentile = float(kwargs.get("background_percentile", 25.0))
        bg_percentile = min(100.0, max(0.0, bg_percentile))
        if bg_outer_radius <= bg_inner_gap:
            bg_outer_radius = bg_inner_gap + 1

        # Squeeze masks to 2D for consistent morphology + intensity computation
        labels = np.squeeze(np.asarray(masks)).astype(np.int32, copy=False)
        if labels.ndim > 2:
            labels = labels[0]
        max_label = int(labels.max())
        if max_label <= 0:
            return pd.DataFrame()

        flat_labels = labels.ravel()
        area = np.bincount(flat_labels, minlength=max_label + 1)
        nonzero = area > 0

        # Perimeter via 4-connectivity boundary detection on 2D labels
        padded = np.pad(labels, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        boundary = (
            (center != padded[:-2, 1:-1]) |
            (center != padded[2:,  1:-1]) |
            (center != padded[1:-1, :-2]) |
            (center != padded[1:-1, 2:])
        )
        perimeter = np.bincount(center[boundary], minlength=max_label + 1).astype(np.float64)

        mask_ids = np.arange(1, max_label + 1, dtype=np.int32)
        df = pd.DataFrame({
            "mask_id": mask_ids,
            "area": area[1:],
            "perimeter": perimeter[1:],
        })
        bg_labels = _make_local_background_labels(labels, bg_inner_gap, bg_outer_radius)

        # ── Intensity ────────────────────────────────────────────────────────
        if intensity_channel == -1:
            # "all" mode: per-channel columns + cross-channel mean
            per_channel = _compute_per_channel_means(
                image, labels, flat_labels, area, nonzero, max_label
            )
            if per_channel is not None and len(per_channel) > 1:
                channel_slices = _extract_channel_slices(image, labels)
                overall = np.mean(np.stack(per_channel, axis=0), axis=0)
                df["mean_intensity"] = overall
                bg_sub_channels = []
                for c_idx, ch_means in enumerate(per_channel):
                    df[f"mean_intensity_ch{c_idx + 1}"] = ch_means
                    if channel_slices[c_idx] is None:
                        bg_values = np.full(mask_ids.shape, np.nan, dtype=np.float64)
                    else:
                        bg_values = _local_background_values(
                            bg_labels, channel_slices[c_idx], mask_ids, bg_percentile
                        )
                    bg_sub = ch_means - bg_values
                    bg_sub_channels.append(bg_sub)
                    df[f"mean_intensity_bg_subtracted_ch{c_idx + 1}"] = bg_sub
                df["mean_intensity_bg_subtracted"] = _nanmean_columns(bg_sub_channels)
            else:
                # Single-channel image or indeterminate axes — produce one column
                try:
                    intensity_img = _match_intensity_to_masks(image, masks, intensity_channel=-1)
                    flat_intensity = _ravel_to_match(intensity_img, labels)
                    isum = np.bincount(flat_labels, weights=flat_intensity, minlength=max_label + 1)
                    mean_intensity = np.zeros(max_label + 1, dtype=np.float64)
                    mean_intensity[nonzero] = isum[nonzero] / area[nonzero]
                    df["mean_intensity"] = mean_intensity[1:]
                    bg_values = _local_background_values(
                        bg_labels, np.squeeze(intensity_img), mask_ids, bg_percentile
                    )
                    df["mean_intensity_bg_subtracted"] = mean_intensity[1:] - bg_values
                except ValueError as exc:
                    _logger.warning("Basic Stats: %s", exc)
                    return pd.DataFrame()
            df["intensity_channel_name"] = "all"
        else:
            channel_name = _format_intensity_channel_name(intensity_channel)
            try:
                intensity_img = _match_intensity_to_masks(image, masks, intensity_channel=intensity_channel)
            except ValueError as exc:
                _logger.warning("Basic Stats: %s", exc)
                return pd.DataFrame()
            flat_intensity = _ravel_to_match(intensity_img, labels)
            isum = np.bincount(flat_labels, weights=flat_intensity, minlength=max_label + 1)
            mean_intensity = np.zeros(max_label + 1, dtype=np.float64)
            mean_intensity[nonzero] = isum[nonzero] / area[nonzero]
            df["mean_intensity"] = mean_intensity[1:]
            bg_values = _local_background_values(
                bg_labels, np.squeeze(intensity_img), mask_ids, bg_percentile
            )
            df["mean_intensity_bg_subtracted"] = mean_intensity[1:] - bg_values
            df["intensity_channel_name"] = channel_name

        # ── Object ID assignment ─────────────────────────────────────────────
        object_ids = _normalize_object_ids(kwargs.get("object_ids_by_mask"))
        has_existing_ids = object_ids is not None and int(np.max(object_ids)) > 0

        if has_existing_ids:
            df["object_id"] = df["mask_id"].apply(
                lambda mid: int(object_ids[mid]) if mid < len(object_ids) else 0
            )
        else:
            # Auto-assign: single channel → object_id = mask_id;
            # multi-channel → IoU matching across channels
            channel_index = int(kwargs.get("channel_index", 0))
            channel_segmentations = kwargs.get("channel_segmentations")
            auto_ids = _auto_assign_object_ids(masks, channel_index, channel_segmentations)
            df["object_id"] = df["mask_id"].apply(
                lambda mid: int(auto_ids[mid]) if auto_ids is not None and mid < len(auto_ids) else int(mid)
            )

        if classes is not None:
            def get_class(mask_id):
                return int(classes[mask_id]) if mask_id < len(classes) else 0
            df["class_id"] = df["mask_id"].apply(get_class)

        return df


# ── Intensity helpers ─────────────────────────────────────────────────────────

def _ravel_to_match(intensity_img: np.ndarray, labels_2d: np.ndarray) -> np.ndarray:
    """Return a float64 1-D array matching labels_2d.ravel() in length."""
    arr = np.squeeze(intensity_img).astype(np.float64)
    if arr.shape == labels_2d.shape:
        return arr.ravel()
    if arr.size == labels_2d.size:
        return arr.ravel()
    raise ValueError(
        f"Cannot align intensity shape {intensity_img.shape} with labels shape {labels_2d.shape}"
    )


def _extract_channel_slices(image, labels_2d):
    """
    Return per-channel 2D arrays matching labels_2d, or None for single-channel
    or ambiguous image layouts.
    """
    arr = np.squeeze(np.asarray(image))
    if arr.ndim < 3:
        return None

    if arr.shape[-1] <= 8 and arr.shape[0] > 8 and arr.shape[1] > 8:
        channel_slices = [arr[..., c] for c in range(arr.shape[-1])]
    elif arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
        channel_slices = [arr[c] for c in range(arr.shape[0])]
    else:
        return None

    if len(channel_slices) < 2:
        return None

    aligned = []
    for ch_arr in channel_slices:
        ch_2d = np.squeeze(ch_arr).astype(np.float64)
        if ch_2d.shape != labels_2d.shape:
            if ch_2d.size == labels_2d.size:
                ch_2d = ch_2d.reshape(labels_2d.shape)
            else:
                aligned.append(None)
                continue
        aligned.append(ch_2d)
    return aligned


def _compute_per_channel_means(image, labels_2d, flat_labels, area, nonzero, max_label):
    """
    Extract per-channel 2D arrays and compute per-mask mean for each.
    Returns a list of float64 arrays (length max_label), one per channel,
    or None if the image is single-channel.
    """
    arr = np.squeeze(np.asarray(image))
    if arr.ndim < 3:
        return None

    # Determine channel axis — use channels-last convention (shape[-1] <= 8),
    # falling back to channels-first (shape[0] <= 8).
    if arr.shape[-1] <= 8 and arr.shape[0] > 8 and arr.shape[1] > 8:
        channel_slices = [arr[..., c] for c in range(arr.shape[-1])]
    elif arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
        channel_slices = [arr[c] for c in range(arr.shape[0])]
    else:
        return None

    if len(channel_slices) < 2:
        return None

    per_channel = []
    for ch_arr in channel_slices:
        ch_2d = np.squeeze(ch_arr).astype(np.float64)
        if ch_2d.shape != labels_2d.shape:
            if ch_2d.size == labels_2d.size:
                ch_2d = ch_2d.reshape(labels_2d.shape)
            else:
                _logger.warning(
                    "Basic Stats: channel %d shape %s (size %d) does not match "
                    "labels shape %s (size %d) — channel will be reported as zeros.",
                    len(per_channel), ch_2d.shape, ch_2d.size,
                    labels_2d.shape, labels_2d.size,
                )
                per_channel.append(np.zeros(max_label, dtype=np.float64))
                continue
        ch_sum = np.bincount(flat_labels, weights=ch_2d.ravel(), minlength=max_label + 1)
        ch_mean = np.zeros(max_label + 1, dtype=np.float64)
        ch_mean[nonzero] = ch_sum[nonzero] / area[nonzero]
        per_channel.append(ch_mean[1:])

    return per_channel


def _make_local_background_labels(
    labels_2d: np.ndarray,
    inner_gap_px: int,
    outer_radius_px: int,
) -> np.ndarray:
    """
    Label local background pixels by nearest object.

    Only unlabeled pixels are eligible, so the annulus cannot include any
    segmented object. The distance-transform nearest-label assignment gives
    each object its local background ring.
    """
    from scipy.ndimage import distance_transform_edt

    labels = np.asarray(labels_2d, dtype=np.int32)
    bg_dist, nearest_idx = distance_transform_edt(labels == 0, return_indices=True)
    nearest_labels = labels[tuple(nearest_idx)]
    annulus = (labels == 0) & (bg_dist > inner_gap_px) & (bg_dist <= outer_radius_px)
    return (nearest_labels * annulus).astype(np.int32, copy=False)


def _local_background_values(
    background_labels: np.ndarray,
    channel_img: np.ndarray,
    mask_ids: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Return one robust local background value per mask id."""
    ch = np.squeeze(np.asarray(channel_img, dtype=np.float64))
    if ch.shape != background_labels.shape:
        if ch.size == background_labels.size:
            ch = ch.reshape(background_labels.shape)
        else:
            raise ValueError(
                f"Cannot align background channel shape {channel_img.shape} "
                f"with labels shape {background_labels.shape}"
            )

    bg_flat = background_labels.ravel()
    values_flat = ch.ravel()
    usable = (bg_flat > 0) & np.isfinite(values_flat)
    out = np.full(mask_ids.shape, np.nan, dtype=np.float64)
    if not usable.any():
        return out

    sample_labels = bg_flat[usable]
    sample_values = values_flat[usable]
    order = np.argsort(sample_labels, kind="stable")
    sample_labels = sample_labels[order]
    sample_values = sample_values[order]

    starts = np.r_[0, np.flatnonzero(np.diff(sample_labels)) + 1]
    ends = np.r_[starts[1:], sample_labels.size]
    id_to_pos = {int(mask_id): pos for pos, mask_id in enumerate(mask_ids)}
    for start, end in zip(starts, ends):
        label_id = int(sample_labels[start])
        pos = id_to_pos.get(label_id)
        if pos is not None:
            out[pos] = np.percentile(sample_values[start:end], percentile)
    return out


def _nanmean_columns(values) -> np.ndarray:
    """Mean across arrays, preserving NaN where every input is NaN."""
    stack = np.stack(values, axis=0)
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    out = np.full(stack.shape[1], np.nan, dtype=np.float64)
    has_value = counts > 0
    out[has_value] = np.nansum(stack[:, has_value], axis=0) / counts[has_value]
    return out


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


def _collapse_non_spatial_axes(arr: np.ndarray, target_ndim: int, intensity_channel: int) -> np.ndarray:
    if arr.ndim <= target_ndim:
        return arr
    if intensity_channel >= 0:
        arr = _select_channel_axis_last(arr, intensity_channel=intensity_channel)
    while arr.ndim > target_ndim:
        arr = arr.mean(axis=-1)
    return arr


# ── Object ID auto-assignment ─────────────────────────────────────────────────

def _auto_assign_object_ids(current_masks, channel_index, channel_segmentations):
    """
    Auto-assign object IDs.

    Single channel (no other channel masks available):
        object_id[mask_id] = mask_id  (1:1 assignment)

    Multiple channels (other channel masks found in channel_segmentations):
        IoU-based matching across channels → shared object IDs for overlapping masks.

    Returns an int32 array indexed by mask_id (length = max_mask_id + 1),
    where result[mask_id] = object_id.
    """
    masks_by_channel = {channel_index: np.squeeze(np.asarray(current_masks)).astype(np.int32)}

    if channel_segmentations:
        for key, state in channel_segmentations.items():
            ch_idx = int(key)
            if ch_idx == channel_index:
                continue
            ch_masks = state.get("masks") if isinstance(state, dict) else state
            if ch_masks is None:
                continue
            ch_masks_2d = np.squeeze(np.asarray(ch_masks)).astype(np.int32)
            if ch_masks_2d.ndim > 2:
                ch_masks_2d = ch_masks_2d[0]
            if ch_masks_2d.ndim == 2 and int(ch_masks_2d.max()) > 0:
                masks_by_channel[ch_idx] = ch_masks_2d

    max_id = int(masks_by_channel[channel_index].max())
    result = np.arange(max_id + 1, dtype=np.int32)  # default: object_id = mask_id

    if len(masks_by_channel) <= 1:
        _logger.debug(
            "Basic Stats: single channel detected — assigning object_id = mask_id for %d masks.",
            max_id,
        )
        return result

    # Multi-channel: IoU matching
    _logger.debug(
        "Basic Stats: matching masks across %d channels via IoU.", len(masks_by_channel)
    )
    try:
        assignment = _iou_match_channels(masks_by_channel)
        for mask_id in range(1, max_id + 1):
            result[mask_id] = assignment.get((channel_index, mask_id), mask_id)
    except Exception as exc:
        _logger.warning("Basic Stats: IoU object matching failed (%s); falling back to mask_id.", exc)

    return result


def _iou_match_channels(masks_by_channel, iou_threshold=0.3):
    """
    Union-Find IoU matching across all channel pairs.
    Returns {(channel_idx, mask_id): object_id}.
    """
    channels = sorted(masks_by_channel.keys())

    # ── Union-Find with path compression ────────────────────────────────────
    parent = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        # path compression
        cur = x
        while cur != root:
            nxt = parent.get(cur, cur)
            parent[cur] = root
            cur = nxt
        return root

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Initialise one node per (channel, mask_id)
    for ch_idx, ch_masks in masks_by_channel.items():
        for mid in range(1, int(ch_masks.max()) + 1):
            node = (ch_idx, mid)
            if node not in parent:
                parent[node] = node

    # Match every pair of channels
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            ch1, ch2 = channels[i], channels[j]
            _union_overlapping_masks(
                masks_by_channel[ch1], masks_by_channel[ch2],
                ch1, ch2, union, iou_threshold,
            )

    # Assign compact object IDs per connected component
    component_to_id = {}
    next_id = [1]
    result = {}
    for node in list(parent.keys()):
        root = find(node)
        if root not in component_to_id:
            component_to_id[root] = next_id[0]
            next_id[0] += 1
        result[node] = component_to_id[root]

    return result


def _union_overlapping_masks(m1, m2, ch1, ch2, union_fn, iou_threshold):
    """Find all (m1_id, m2_id) pairs with IoU >= threshold and union them."""
    f1 = m1.ravel().astype(np.int32)
    f2 = m2.ravel().astype(np.int32)
    if f1.shape != f2.shape:
        return

    max1 = int(f1.max())
    max2 = int(f2.max())
    if max1 == 0 or max2 == 0:
        return

    overlap = (f1 > 0) & (f2 > 0)
    if not overlap.any():
        return

    ids1 = f1[overlap].astype(np.int64)
    ids2 = f2[overlap].astype(np.int64)
    stride = int(max2) + 1
    pair_keys = ids1 * stride + ids2
    pair_counts = np.bincount(pair_keys, minlength=(max1 + 1) * stride)

    area1 = np.bincount(f1[f1 > 0], minlength=max1 + 1)
    area2 = np.bincount(f2[f2 > 0], minlength=max2 + 1)

    for pk in np.unique(pair_keys):
        id1 = int(pk // stride)
        id2 = int(pk % stride)
        if id1 <= 0 or id2 <= 0:
            continue
        intersection = int(pair_counts[pk])
        union_area = int(area1[id1]) + int(area2[id2]) - intersection
        if union_area <= 0:
            continue
        if intersection / union_area >= iou_threshold:
            union_fn((ch1, id1), (ch2, id2))


# ── Misc helpers ──────────────────────────────────────────────────────────────

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


def _normalize_object_ids(value):
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.int32)
    except Exception:
        return None
