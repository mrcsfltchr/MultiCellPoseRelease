import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from guv_app.plugins.interface import AnalysisPlugin
from guv_app.plugins.validator import validate_visualization_mask
try:
    import cv2
except ImportError:
    cv2 = None

_logger = logging.getLogger(__name__)


@dataclass
class _Detection:
    frame_index: int
    mask_id: int
    class_id: int
    area: float
    centroid_y: float
    centroid_x: float
    tracking_intensity: float
    mean_intensity: float
    measurement_intensities: Dict[str, float]
    perimeter: float
    circularity: float
    aspect_ratio: float


@dataclass
class _TrackState:
    track_id: int
    last_detection: _Detection
    last_frame_index: int
    age: int


class ObjectTrackingPlugin(AnalysisPlugin):
    """
    Assigns stable track IDs to segmented objects across time.

    Matching uses centroid displacement, area change, mean fluorescence intensity,
    and simple shape descriptors. New detections start new tracks; tracks can be
    recovered after short gaps controlled by max_frame_gap.
    """

    @property
    def name(self) -> str:
        return "Object Tracking"

    def __init__(self):
        self._sequence_key = None
        self._next_track_id = 1
        self._tracks: Dict[int, _TrackState] = {}
        self._track_history: Dict[int, List[Tuple[int, float, float]]] = {}
        self._last_frame_index = -1

    def get_parameter_definitions(self):
        return {
            "tracking_channels": {
                "type": "str",
                "default": "all",
                "label": "Tracking Channels",
                "help": "Channels used for object matching. Use 'all' or comma-separated one-based channels, e.g. 1,3.",
            },
            "measurement_channels": {
                "type": "str",
                "default": "all",
                "label": "Measurement Channels",
                "help": "Channels measured for each final track. Use 'all' or comma-separated one-based channels, e.g. 1,2.",
            },
            "max_displacement_px": {
                "type": "float",
                "default": 50.0,
                "min": 0.0,
                "max": 10000.0,
                "label": "Max Displacement (px)",
                "help": "Maximum centroid movement allowed between matched detections.",
            },
            "max_frame_gap": {
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 1000,
                "label": "Max Missing Frames",
                "help": "Number of missing frames over which an object can still be recovered.",
            },
            "match_threshold": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "label": "Match Threshold",
                "help": "Maximum weighted match cost accepted as the same object.",
            },
            "distance_weight": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "label": "Distance Weight",
            },
            "area_weight": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "label": "Area Weight",
            },
            "intensity_weight": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "label": "Intensity Weight",
            },
            "shape_weight": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 100.0,
                "label": "Shape Weight",
            },
            "class_constraint": {
                "type": "bool",
                "default": True,
                "label": "Match Same Class Only",
                "help": "When classes are available, only match detections with the same class ID.",
            },
            "min_area_px": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 100000000,
                "label": "Minimum Area (px)",
            },
            "show_track_history": {
                "type": "bool",
                "default": True,
                "label": "Show Track History",
                "help": "Draw each approved track history in the editable visualization overlay.",
            },
            "trail_length_frames": {
                "type": "int",
                "default": 30,
                "min": 1,
                "max": 10000,
                "label": "Trail Length (frames)",
            },
        }

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if masks is None:
            return pd.DataFrame()

        masks_arr = np.asarray(masks)
        if _is_mask_stack(masks_arr):
            return self._run_stack(image, masks_arr, classes=classes, **kwargs)

        filename = kwargs.get("filename")
        sequence_key, frame_index = _frame_context(filename, kwargs.get("frame_index"))
        if sequence_key is None:
            self._reset_state()
            sequence_key = "__single_frame__"
            frame_index = 0
        elif sequence_key != self._sequence_key or frame_index <= self._last_frame_index:
            self._reset_state()
            self._sequence_key = sequence_key

        masks2d = _to_2d_masks(masks_arr)
        if masks2d is None or int(masks2d.max()) <= 0:
            self._last_frame_index = frame_index
            return pd.DataFrame()

        image_channels = _align_channels_to_masks(image, masks2d)
        tracking_image = _combine_channels(image_channels, _tracking_channel_setting(kwargs))
        measurement_images = _measurement_channel_images(image_channels, kwargs.get("measurement_channels", "all"))
        detections = _extract_detections(
            tracking_image,
            measurement_images,
            masks2d,
            classes=classes,
            frame_index=frame_index,
            min_area=max(1, int(kwargs.get("min_area_px", 1))),
        )
        rows = self._assign_tracks(detections, **kwargs)
        self._last_frame_index = frame_index
        df = pd.DataFrame(rows)
        return _filter_approved_tracks(df, kwargs.get("approved_track_ids"))

    def visualize(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> np.ndarray:
        if masks is None:
            return None
        masks2d = _to_2d_masks(np.asarray(masks))
        if masks2d is None or int(masks2d.max()) <= 0:
            return None
        df = self.run(image, masks, classes=classes, **kwargs)
        out = np.zeros_like(masks2d, dtype=np.int32)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                track_id = int(row["track_id"])
                if bool(kwargs.get("show_track_history", True)):
                    _draw_track_history_label(
                        out,
                        self._track_history.get(track_id, []),
                        track_id,
                        trail_length=int(kwargs.get("trail_length_frames", 30)),
                    )
                _draw_current_track_marker(
                    out,
                    float(row["centroid_y"]),
                    float(row["centroid_x"]),
                    track_id,
                )
        validate_visualization_mask(out, masks2d, allow_new_labels=True)
        return out

    def _run_stack(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        self._reset_state()
        self._sequence_key = "__stack__"
        rows: List[dict] = []
        masks_stack = np.asarray(masks)
        for frame_index in range(masks_stack.shape[0]):
            masks2d = _to_2d_masks(masks_stack[frame_index])
            if masks2d is None:
                continue
            image_frame = _slice_image_frame(image, frame_index)
            image_channels = _align_channels_to_masks(image_frame, masks2d)
            tracking_image = _combine_channels(image_channels, _tracking_channel_setting(kwargs))
            measurement_images = _measurement_channel_images(image_channels, kwargs.get("measurement_channels", "all"))
            frame_classes = _slice_classes(classes, frame_index)
            detections = _extract_detections(
                tracking_image,
                measurement_images,
                masks2d,
                classes=frame_classes,
                frame_index=frame_index,
                min_area=max(1, int(kwargs.get("min_area_px", 1))),
            )
            rows.extend(self._assign_tracks(detections, **kwargs))
            self._last_frame_index = frame_index
        df = pd.DataFrame(rows)
        return _filter_approved_tracks(df, kwargs.get("approved_track_ids"))

    def _reset_state(self) -> None:
        self._sequence_key = None
        self._next_track_id = 1
        self._tracks = {}
        self._track_history = {}
        self._last_frame_index = -1

    def _assign_tracks(self, detections: List[_Detection], **kwargs) -> List[dict]:
        max_gap = max(0, int(kwargs.get("max_frame_gap", 1)))
        frame_index = detections[0].frame_index if detections else self._last_frame_index + 1
        active_tracks = {
            tid: tr for tid, tr in self._tracks.items()
            if frame_index - tr.last_frame_index <= max_gap + 1
        }
        self._tracks = dict(active_tracks)

        candidates = []
        for det_idx, det in enumerate(detections):
            for track_id, track in active_tracks.items():
                cost, dist = _match_cost(det, track.last_detection, frame_index - track.last_frame_index, **kwargs)
                if np.isfinite(cost):
                    candidates.append((cost, dist, det_idx, track_id))
        candidates.sort(key=lambda item: (item[0], item[1]))

        assigned_dets = set()
        assigned_tracks = set()
        assignments: Dict[int, Tuple[int, float, float, int]] = {}
        threshold = float(kwargs.get("match_threshold", 1.0))
        for cost, dist, det_idx, track_id in candidates:
            if cost > threshold or det_idx in assigned_dets or track_id in assigned_tracks:
                continue
            gap = frame_index - self._tracks[track_id].last_frame_index - 1
            assignments[det_idx] = (track_id, cost, dist, gap)
            assigned_dets.add(det_idx)
            assigned_tracks.add(track_id)

        rows = []
        for det_idx, det in enumerate(detections):
            if det_idx in assignments:
                track_id, cost, dist, gap = assignments[det_idx]
                prev = self._tracks[track_id]
                status = "gap_closed" if gap > 0 else "matched"
                age = prev.age + 1
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                cost = np.nan
                dist = np.nan
                gap = 0
                status = "new"
                age = 1

            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                last_detection=det,
                last_frame_index=det.frame_index,
                age=age,
            )
            self._track_history.setdefault(track_id, []).append(
                (int(det.frame_index), float(det.centroid_y), float(det.centroid_x))
            )
            rows.append(_row_from_detection(det, track_id, status, age, cost, dist, gap))
        return rows


def _is_mask_stack(masks: np.ndarray) -> bool:
    arr = np.asarray(masks)
    return arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[1] > 1 and arr.shape[2] > 1


def _filter_approved_tracks(df: pd.DataFrame, approved_track_ids) -> pd.DataFrame:
    if df is None or df.empty or approved_track_ids is None:
        return df
    ids = _parse_track_id_set(approved_track_ids)
    if not ids:
        return df.iloc[0:0].copy()
    return df[df["track_id"].astype(int).isin(ids)].copy()


def _draw_track_history_label(out: np.ndarray, history, track_id: int, trail_length: int = 30) -> None:
    if cv2 is None or not history:
        return
    ordered = sorted(history, key=lambda item: int(item[0]))
    if trail_length > 0:
        ordered = ordered[-trail_length:]
    if len(ordered) == 1:
        _, y, x = ordered[0]
        cv2.circle(out, (int(round(x)), int(round(y))), 2, int(track_id), -1, lineType=cv2.LINE_8)
        return
    points = [(int(round(x)), int(round(y))) for _, y, x in ordered]
    for p0, p1 in zip(points[:-1], points[1:]):
        cv2.line(out, p0, p1, int(track_id), 2, lineType=cv2.LINE_8)
    cv2.circle(out, points[-1], 2, int(track_id), -1, lineType=cv2.LINE_8)


def _draw_current_track_marker(out: np.ndarray, y: float, x: float, track_id: int) -> None:
    if cv2 is None:
        yy = int(round(y))
        xx = int(round(x))
        if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
            out[yy, xx] = int(track_id)
        return
    center = (int(round(x)), int(round(y)))
    cv2.line(out, (center[0] - 3, center[1]), (center[0] + 3, center[1]), int(track_id), 1, lineType=cv2.LINE_8)
    cv2.line(out, (center[0], center[1] - 3), (center[0], center[1] + 3), int(track_id), 1, lineType=cv2.LINE_8)


def _parse_track_id_set(value) -> set:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        out = set()
        for token in re.split(r"[,; ]+", text):
            if not token:
                continue
            try:
                out.add(int(token))
            except Exception:
                pass
        return out
    try:
        return {int(v) for v in value}
    except TypeError:
        try:
            return {int(value)}
        except Exception:
            return set()


def _to_2d_masks(masks: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(masks)
    if arr.ndim == 2:
        return arr.astype(np.int32, copy=False)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0].astype(np.int32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.int32, copy=False)
    _logger.warning("Object Tracking: unsupported mask shape %s", arr.shape)
    return None


def _frame_context(filename, explicit_frame_index=None) -> Tuple[Optional[str], int]:
    if explicit_frame_index is not None:
        try:
            explicit = int(explicit_frame_index)
        except Exception:
            explicit = 0
    else:
        explicit = None
    if not filename:
        return None, explicit if explicit is not None else 0

    text = str(filename)
    if "::" in text:
        base_key, frame_id = text.split("::", 1)
        sequence_key = _sequence_key_from_frame_id(base_key, frame_id)
        frame_index = _frame_index_from_id(frame_id)
        return sequence_key, explicit if explicit is not None else frame_index
    return text, explicit if explicit is not None else 0


def _sequence_key_from_frame_id(base_key: str, frame_id: str) -> str:
    tokens = re.findall(r"([A-Za-z])(\d+)", str(frame_id))
    if not any(axis.upper() == "T" for axis, _ in tokens):
        return f"{base_key}::{frame_id}"
    non_time = [f"{axis.upper()}{value}" for axis, value in tokens if axis.upper() != "T"]
    return f"{base_key}::{'_'.join(non_time)}" if non_time else base_key


def _frame_index_from_id(frame_id: str) -> int:
    matches = re.findall(r"([TtZz])(\d+)", str(frame_id))
    for axis, value in matches:
        if axis.upper() == "T":
            return int(value)
    if matches:
        return int(matches[-1][1])
    nums = re.findall(r"\d+", str(frame_id))
    return int(nums[-1]) if nums else 0


def _slice_image_frame(image, frame_index: int):
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim >= 3 and arr.shape[0] > frame_index and arr.shape[1] > 8 and arr.shape[2] > 8:
        return arr[frame_index]
    return image


def _slice_classes(classes, frame_index: int):
    if classes is None:
        return None
    arr = np.asarray(classes)
    if arr.ndim >= 2 and arr.shape[0] > frame_index:
        return arr[frame_index]
    return arr


def _align_channels_to_masks(image, masks2d: np.ndarray) -> List[np.ndarray]:
    if image is None:
        return []
    arr = np.asarray(image)
    if arr.ndim == 2:
        return [arr.astype(np.float64, copy=False)] if arr.shape == masks2d.shape else []
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return [arr.astype(np.float64, copy=False)] if arr.shape == masks2d.shape else []
    if arr.ndim != 3:
        return []

    channels = _extract_channels(arr, masks2d.shape)
    return channels


def _combine_channels(channels: List[np.ndarray], setting="all") -> Optional[np.ndarray]:
    selected = _select_channels(channels, setting)
    if not selected:
        return None
    if len(selected) == 1:
        return selected[0]
    return np.mean(np.stack(selected, axis=0), axis=0)


def _measurement_channel_images(channels: List[np.ndarray], setting="all") -> Dict[str, np.ndarray]:
    if not channels:
        return {}
    indices = _parse_channel_indices(setting, len(channels))
    return {f"mean_intensity_ch{idx + 1}": channels[idx] for idx in indices}


def _extract_channels(arr: np.ndarray, shape: Tuple[int, int]) -> List[np.ndarray]:
    if arr.shape[:2] == shape and arr.shape[-1] <= 8:
        return [arr[..., c].astype(np.float64, copy=False) for c in range(arr.shape[-1])]
    if arr.shape[1:] == shape and arr.shape[0] <= 8:
        return [arr[c].astype(np.float64, copy=False) for c in range(arr.shape[0])]
    return []


def _tracking_channel_setting(kwargs) -> str:
    if "tracking_channels" in kwargs and str(kwargs.get("tracking_channels", "")).strip():
        return kwargs.get("tracking_channels")
    return kwargs.get("intensity_channel", "all")


def _select_channels(channels: List[np.ndarray], setting="all") -> List[np.ndarray]:
    return [channels[idx] for idx in _parse_channel_indices(setting, len(channels))]


def _parse_channel_indices(setting, channel_count: int) -> List[int]:
    if channel_count <= 0:
        return []
    text = "" if setting is None else str(setting).strip().lower()
    if text in {"", "all", "*"}:
        return list(range(channel_count))
    indices = []
    for token in re.split(r"[,; ]+", text):
        if not token:
            continue
        idx = _parse_channel(token)
        if idx is None:
            continue
        if 0 <= idx < channel_count and idx not in indices:
            indices.append(idx)
    return indices or list(range(channel_count))


def _parse_channel(value) -> Optional[int]:
    if value is None or str(value).lower() == "all":
        return None
    try:
        return max(0, int(value) - 1)
    except Exception:
        return None


def _extract_detections(
    tracking_image: Optional[np.ndarray],
    measurement_images: Dict[str, np.ndarray],
    masks2d: np.ndarray,
    classes: np.ndarray = None,
    frame_index: int = 0,
    min_area: int = 1,
) -> List[_Detection]:
    max_label = int(masks2d.max())
    if max_label <= 0:
        return []
    flat = masks2d.ravel().astype(np.int64, copy=False)
    area = np.bincount(flat, minlength=max_label + 1).astype(np.float64)
    yy, xx = np.indices(masks2d.shape)
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=max_label + 1)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=max_label + 1)
    if tracking_image is not None:
        tracking_sum = np.bincount(flat, weights=np.asarray(tracking_image).ravel(), minlength=max_label + 1)
    else:
        tracking_sum = np.full(max_label + 1, np.nan, dtype=np.float64)
    measurement_sums = {
        name: np.bincount(flat, weights=np.asarray(image).ravel(), minlength=max_label + 1)
        for name, image in measurement_images.items()
    }

    perimeter = _perimeter_by_label(masks2d, max_label)
    detections = []
    for mask_id in np.flatnonzero(area > 0):
        if mask_id == 0 or area[mask_id] < min_area:
            continue
        region = masks2d == mask_id
        ys, xs = np.nonzero(region)
        height = max(1, int(ys.max() - ys.min() + 1))
        width = max(1, int(xs.max() - xs.min() + 1))
        aspect_ratio = float(width / height)
        circ = float(4.0 * np.pi * area[mask_id] / max(perimeter[mask_id] ** 2, 1.0))
        measured = {
            name: float(values[mask_id] / area[mask_id])
            for name, values in measurement_sums.items()
        }
        measured_mean = float(np.nanmean(list(measured.values()))) if measured else np.nan
        detections.append(
            _Detection(
                frame_index=int(frame_index),
                mask_id=int(mask_id),
                class_id=_safe_class(classes, int(mask_id)),
                area=float(area[mask_id]),
                centroid_y=float(sum_y[mask_id] / area[mask_id]),
                centroid_x=float(sum_x[mask_id] / area[mask_id]),
                tracking_intensity=float(tracking_sum[mask_id] / area[mask_id]) if tracking_image is not None else np.nan,
                mean_intensity=measured_mean,
                measurement_intensities=measured,
                perimeter=float(perimeter[mask_id]),
                circularity=circ,
                aspect_ratio=aspect_ratio,
            )
        )
    return detections


def _perimeter_by_label(masks2d: np.ndarray, max_label: int) -> np.ndarray:
    padded = np.pad(masks2d, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1]
    boundary = (
        (center != padded[:-2, 1:-1]) |
        (center != padded[2:, 1:-1]) |
        (center != padded[1:-1, :-2]) |
        (center != padded[1:-1, 2:])
    )
    return np.bincount(center[boundary].ravel(), minlength=max_label + 1).astype(np.float64)


def _safe_class(classes: np.ndarray, mask_id: int) -> int:
    if classes is None:
        return 0
    arr = np.asarray(classes)
    if arr.ndim != 1 or mask_id < 0 or mask_id >= len(arr):
        return 0
    return int(arr[mask_id])


def _match_cost(det: _Detection, prev: _Detection, gap: int, **kwargs) -> Tuple[float, float]:
    if bool(kwargs.get("class_constraint", True)) and det.class_id > 0 and prev.class_id > 0 and det.class_id != prev.class_id:
        return np.inf, np.inf

    max_disp = max(1e-6, float(kwargs.get("max_displacement_px", 50.0)))
    distance = float(np.hypot(det.centroid_y - prev.centroid_y, det.centroid_x - prev.centroid_x))
    allowed = max_disp * max(1, int(gap))
    if distance > allowed:
        return np.inf, distance

    dist_cost = distance / max_disp
    area_cost = abs(np.log((det.area + 1.0) / (prev.area + 1.0)))
    intensity_cost = _relative_difference(det.tracking_intensity, prev.tracking_intensity)
    shape_cost = (
        abs(det.circularity - prev.circularity) +
        abs(np.log((det.aspect_ratio + 1e-6) / (prev.aspect_ratio + 1e-6)))
    )

    cost = (
        float(kwargs.get("distance_weight", 1.0)) * dist_cost +
        float(kwargs.get("area_weight", 0.5)) * area_cost +
        float(kwargs.get("intensity_weight", 0.5)) * intensity_cost +
        float(kwargs.get("shape_weight", 0.3)) * shape_cost
    )
    return float(cost), distance


def _relative_difference(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return float(abs(a - b) / max(abs(a), abs(b), 1e-6))


def _row_from_detection(det: _Detection, track_id: int, status: str, age: int, cost: float, distance: float, gap: int) -> dict:
    row = {
        "frame_index": int(det.frame_index),
        "mask_id": int(det.mask_id),
        "track_id": int(track_id),
        "status": status,
        "track_age_frames": int(age),
        "gap_frames": int(gap),
        "match_cost": cost,
        "match_distance_px": distance,
        "class_id": int(det.class_id),
        "area": det.area,
        "centroid_y": det.centroid_y,
        "centroid_x": det.centroid_x,
        "tracking_mean_intensity": det.tracking_intensity,
        "mean_intensity": det.mean_intensity,
        "perimeter": det.perimeter,
        "circularity": det.circularity,
        "aspect_ratio": det.aspect_ratio,
    }
    row.update(det.measurement_intensities)
    return row
