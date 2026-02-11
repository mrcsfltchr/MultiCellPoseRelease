import logging
import os
import tempfile
import threading
import time
import numpy as np
from cellpose import io, transforms
try:
    import nd2 as nd2_lib
except Exception:
    nd2_lib = None

_logger = logging.getLogger(__name__)

class ImageService:
    """
    Service for handling image I/O and basic preprocessing.
    Leverages cellpose.io for robust file format support.
    """
    def __init__(self):
        self._nd2_dask_cache = {}
        self._nd2_meta_cache = {}
        self._frame_cache = {}
        self._save_lock = threading.Lock()
        self._save_cv = threading.Condition(self._save_lock)
        self._pending_saves = {}
        self._stop_save_worker = False
        self._save_debounce_s = 0.35
        self._save_worker = threading.Thread(
            target=self._save_worker_loop,
            name="image-save-worker",
            daemon=True,
        )
        self._save_worker.start()

    def _get_frame_cache(self, filename):
        return self._frame_cache.get(filename)

    def _set_frame_cache(self, filename, frames):
        self._frame_cache.clear()
        self._frame_cache[filename] = frames

    def load_image(self, filename, do_3D=False):
        """
        Loads an image using cellpose.io.imread.
        """
        _logger.info(f"Loading image: {filename}")
        try:
            data = io.read_image_data(filename)
            image = data.array

            if image is None:
                _logger.error(f"Failed to load image: {filename}")
                return None

            return image
        except Exception as e:
            _logger.error(f"Error loading image {filename}: {e}")
            return None

    def normalize_image(self, image_data):
        """
        Normalizes image data for display (0-1 range, 1st-99th percentile).
        """
        if image_data is None:
            return None
        try:
            arr = np.asarray(image_data)
        except Exception as e:
            _logger.error(f"Failed to convert image to array: {e}")
            return None
        if arr.ndim < 2:
            _logger.error(f"Unsupported image dimensions: {arr.shape}")
            return None
        if arr.dtype.kind == "c":
            arr = np.abs(arr)
        if arr.dtype.kind not in "fiu":
            try:
                arr = arr.astype(np.float32)
            except Exception as e:
                _logger.error(f"Failed to coerce image dtype {arr.dtype}: {e}")
                return None
        return transforms.normalize99(arr)

    def split_image_reference(self, filename):
        if "::" not in filename:
            return filename, None
        base, frame_id = filename.split("::", 1)
        return base, frame_id

    def build_image_reference(self, filename, frame_id):
        if frame_id:
            return f"{filename}::{frame_id}"
        return filename

    def build_frame_references(self, filename, series_index=None):
        series_key, series_count, time_count = self.get_series_time_info(filename)
        refs = []
        if time_count > 1:
            if series_key and series_count > 1:
                series_vals = [series_index] if series_index is not None else list(range(series_count))
                for s_idx in series_vals:
                    for t_idx in range(time_count):
                        frame_id = f"{series_key}{s_idx}_T{t_idx}"
                        refs.append(self.build_image_reference(filename, frame_id))
            else:
                for t_idx in range(time_count):
                    frame_id = f"T{t_idx}"
                    refs.append(self.build_image_reference(filename, frame_id))
            return refs
        if series_key and series_count > 1:
            series_vals = [series_index] if series_index is not None else list(range(series_count))
            for s_idx in series_vals:
                frame_id = f"{series_key}{s_idx}"
                refs.append(self.build_image_reference(filename, frame_id))
            return refs
        frames = self.iter_image_frames(filename, series_index=series_index)
        for frame in frames:
            refs.append(self.build_image_reference(filename, frame.frame_id))
        return refs

    def load_frame(self, filename, frame_id):
        if frame_id is None:
            return self.load_image(filename)
        frame = io.read_image_frame(filename, frame_id)
        if frame is not None:
            return frame.array
        frames = self.iter_image_frames(filename, frame_id=frame_id)
        for frame in frames:
            if frame.frame_id == frame_id:
                return frame.array
        _logger.error(f"Frame {frame_id} not found for {filename}")
        return None

    def build_frame_path(self, filename, frame_id, suffix):
        base = os.path.splitext(filename)[0]
        frame_suffix = io.frame_id_to_suffix(frame_id)
        return f"{base}{frame_suffix}{suffix}"

    def collect_training_pairs(self, folder_path, mask_suffix="_seg.npy", look_one_level_down=False):
        if not folder_path:
            return [], []
        try:
            image_files = io.get_image_files(folder_path, mask_suffix, look_one_level_down=look_one_level_down)
        except Exception as e:
            _logger.error(f"Failed to list training images in {folder_path}: {e}")
            return [], []
        train_files = []
        label_files = []
        for image_path in image_files:
            base, _ = os.path.splitext(image_path)
            label_path = f"{base}{mask_suffix}"
            if os.path.exists(label_path):
                train_files.append(image_path)
                label_files.append(label_path)
        return train_files, label_files

    def _build_axes_from_sizes(self, sizes, shape):
        if not sizes:
            return None
        order = "TPSZCYX"
        axes = [ax for ax in order if ax in sizes]
        if len(axes) != len(shape):
            return None
        return "".join(axes)

    def _axes_from_meta(self, meta, shape):
        axes = getattr(meta, "axes", None)
        if isinstance(axes, str) and len(axes) == len(shape):
            return axes
        sizes = getattr(meta, "sizes", None) or {}
        return self._build_axes_from_sizes(sizes, shape)

    def _axis_index_from_meta(self, meta, axis_key, shape):
        axes = self._axes_from_meta(meta, shape)
        if axes and axis_key in axes:
            return axes.index(axis_key)
        sizes = getattr(meta, "sizes", None) or {}
        if axis_key in sizes:
            target = int(sizes[axis_key])
            matches = [i for i, dim in enumerate(shape) if int(dim) == target]
            if len(matches) == 1:
                return matches[0]
        return None

    def _slice_axis(self, arr, axis_index, index):
        if axis_index is None:
            return arr
        try:
            return np.take(arr, index, axis=axis_index)
        except Exception:
            return arr

    def _normalize_channels_last(self, arr):
        if not hasattr(arr, "ndim"):
            return arr
        if arr.ndim == 3:
            if arr.shape[-1] <= 8 and arr.shape[0] > 8 and arr.shape[1] > 8:
                return arr
            if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
                return np.moveaxis(arr, 0, -1)
            if arr.shape[1] <= 8 and arr.shape[0] > 8 and arr.shape[2] > 8:
                return np.moveaxis(arr, 1, -1)
        return arr

    def _parse_frame_id(self, frame_id):
        if not frame_id:
            return {}
        parts = frame_id.split("_")
        parsed = {}
        for part in parts:
            if len(part) < 2:
                continue
            key = part[0]
            if key in ("S", "P", "T"):
                try:
                    parsed[key] = int(part[1:])
                except ValueError:
                    continue
        return parsed

    def parse_frame_id(self, frame_id):
        return self._parse_frame_id(frame_id)

    def get_series_time_info(self, filename):
        try:
            return io.get_series_time_info(filename)
        except Exception as e:
            _logger.error(f"Error reading image metadata {filename}: {e}")
            return None, 1, 1

    def iter_image_frames(self, filename, series_index=None, frame_id=None):
        """
        Iterates over frames in a file, returning a list of ImageFrame objects.
        """
        _logger.info(f"Loading image frames: {filename}")
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".nd2" and nd2_lib is not None:
            frames = self._iter_nd2_frames_lazy(filename, series_index=series_index, frame_id=frame_id)
            if frames is not None:
                return frames
        try:
            if ext in (".lif", ".tif", ".tiff"):
                cached = self._get_frame_cache(filename)
                if cached is not None:
                    base_frames = cached
                else:
                    base_frames = io.iter_image_frames(filename)
                    self._set_frame_cache(filename, base_frames)
            else:
                base_frames = io.iter_image_frames(filename)
        except Exception as e:
            _logger.error(f"Error loading image frames {filename}: {e}")
            image = self.load_image(filename)
            if image is None:
                return []
            meta = io.ImageMeta(axes=None, shape=tuple(image.shape), sizes={}, dtype=image.dtype)
            return [io.ImageFrame(image, meta, None)]

        parsed = self._parse_frame_id(frame_id) if frame_id else {}
        if "S" in parsed:
            series_index = parsed["S"]
        if "P" in parsed and series_index is None:
            series_index = parsed["P"]
        time_index_only = parsed.get("T")

        frames = []
        for base_frame in base_frames:
            base_id = base_frame.frame_id
            if series_index is not None and base_id:
                series_parsed = self._parse_frame_id(base_id)
                base_series = series_parsed.get("S", series_parsed.get("P"))
                if base_series is not None and base_series != series_index:
                    continue
            arr = base_frame.array
            if arr is None:
                continue
            meta = base_frame.meta
            sizes = getattr(meta, "sizes", None) or {}
            series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
            axis_series = None
            if series_key == "S":
                axis_series = self._axis_index_from_meta(meta, "S", tuple(arr.shape))
            elif series_key == "P":
                axis_series = self._axis_index_from_meta(meta, "P", tuple(arr.shape))
            if series_index is not None and axis_series is not None:
                arr = self._slice_axis(arr, axis_series, series_index)
                if not base_id and series_key:
                    base_id = f"{series_key}{series_index}"
            axis_time = self._axis_index_from_meta(meta, "T", tuple(arr.shape))
            time_count = int(sizes.get("T", 1))
            if time_count <= 1 and hasattr(arr, "shape"):
                if arr.ndim >= 4 and arr.shape[-1] <= 5:
                    time_count = int(arr.shape[0])
                    axis_time = 0
            if time_count > 1 and axis_time is not None:
                for t_index in range(time_count):
                    if time_index_only is not None and t_index != time_index_only:
                        continue
                    frame_arr = self._slice_axis(arr, axis_time, t_index)
                    frame_arr = self._normalize_channels_last(frame_arr)
                    if base_id:
                        frame_id_out = f"{base_id}_T{t_index}"
                    else:
                        frame_id_out = f"T{t_index}"
                    frames.append(io.ImageFrame(frame_arr, meta, frame_id_out))
            else:
                arr = self._normalize_channels_last(arr)
                frames.append(io.ImageFrame(arr, meta, base_id))
        return frames

    def _iter_nd2_frames_lazy(self, filename, series_index=None, frame_id=None):
        try:
            axes = None
            sizes = None
            cached = self._nd2_meta_cache.get(filename)
            if cached:
                axes = cached.get("axes")
                sizes = cached.get("sizes")
            with nd2_lib.ND2File(filename) as f:
                if axes is None:
                    axes = getattr(f, "axes", None)
                    axes = "".join(axes) if isinstance(axes, (list, tuple)) else axes
                if sizes is None:
                    sizes = getattr(f, "sizes", {}) or {}
                if axes is not None and sizes is not None:
                    self._nd2_meta_cache[filename] = {"axes": axes, "sizes": sizes}
                series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
                series_count = int(sizes.get(series_key, 1)) if series_key else 1
                time_count = int(sizes.get("T", 1))
                parsed = self._parse_frame_id(frame_id) if frame_id else {}
                if series_key and series_key in parsed:
                    series_index = parsed[series_key]
                time_index_only = parsed.get("T")

                def axis_index(key):
                    if axes and key in axes:
                        return axes.index(key)
                    return None

                z_index = 0
                if sizes.get("Z", 1) > 1:
                    z_index = 0

                series_axis = axis_index(series_key) if series_key else None

                dask_arr = None
                cache_entry = self._nd2_dask_cache.get(filename, {})
                if cache_entry.get("dask") is not None:
                    dask_arr = cache_entry.get("dask")
                if hasattr(f, "to_dask"):
                    try:
                        dask_arr = dask_arr or f.to_dask()
                    except Exception:
                        dask_arr = None
                if dask_arr is None and hasattr(nd2_lib, "imread"):
                    try:
                        dask_arr = nd2_lib.imread(filename, dask=True)
                    except Exception:
                        dask_arr = None
                if dask_arr is not None:
                    self._nd2_dask_cache[filename] = {"dask": dask_arr}
                    if axes is None:
                        inferred = self._infer_axes_from_sizes(sizes, getattr(dask_arr, "shape", None))
                        if inferred:
                            axes = inferred
                            self._nd2_meta_cache[filename] = {"axes": axes, "sizes": sizes}

                frames = []
                def _compute_slice(indexer):
                    arr = dask_arr[tuple(indexer)] if dask_arr is not None else f.asarray()[tuple(indexer)]
                    if hasattr(arr, "compute"):
                        arr = arr.compute()
                    return np.asarray(arr)

                if dask_arr is not None and axes:
                    if series_key and series_count > 1 and series_axis is None and hasattr(f, "get_frame_2D"):
                        dask_arr = None
                    if dask_arr is not None:
                        if series_key and series_count > 1:
                            series_vals = [series_index] if series_index is not None else list(range(series_count))
                        else:
                            series_vals = [None]
                        if time_count > 1:
                            time_vals = [time_index_only] if time_index_only is not None else list(range(time_count))
                        else:
                            time_vals = [None]
                        for s_idx in series_vals:
                            for t_idx in time_vals:
                                indexer = [slice(None)] * len(axes)
                                if series_key and s_idx is not None and series_axis is not None:
                                    indexer[series_axis] = s_idx
                                time_axis = axis_index("T")
                                if time_count > 1 and t_idx is not None and time_axis is not None:
                                    indexer[time_axis] = t_idx
                                if sizes.get("Z", 1) > 1 and axis_index("Z") is not None:
                                    indexer[axis_index("Z")] = z_index
                                arr = _compute_slice(indexer)
                                out_axes = [ax for ax, idx in zip(axes, indexer) if not isinstance(idx, int)]
                                if "C" in out_axes:
                                    c_idx = out_axes.index("C")
                                    if c_idx != len(out_axes) - 1:
                                        arr = np.moveaxis(arr, c_idx, -1)
                                        out_axes.pop(c_idx)
                                        out_axes.append("C")
                                arr = self._normalize_channels_last(arr)
                                if series_key and series_count > 1:
                                    if time_count > 1:
                                        frame_id_out = f"{series_key}{s_idx}_T{t_idx}"
                                    else:
                                        frame_id_out = f"{series_key}{s_idx}"
                                else:
                                    frame_id_out = f"T{t_idx}" if time_count > 1 else None
                                meta = io.ImageMeta(
                                    axes="".join(out_axes) if out_axes else None,
                                    shape=tuple(arr.shape),
                                    sizes=self._sizes_from_axes("".join(out_axes) if out_axes else None, tuple(arr.shape)),
                                    dtype=arr.dtype,
                                    series_index=s_idx if series_key and series_count > 1 else None,
                                )
                                frames.append(io.ImageFrame(arr, meta, frame_id_out))
                        return frames

                if hasattr(f, "get_frame_2D"):
                    import inspect
                    sig = inspect.signature(f.get_frame_2D)
                    params = set(sig.parameters.keys())
                    def _get_frame_2d(t_val, c_val, s_val):
                        kwargs = {}
                        if "t" in params and t_val is not None:
                            kwargs["t"] = int(t_val)
                        if "c" in params and c_val is not None:
                            kwargs["c"] = int(c_val)
                        if "z" in params and sizes.get("Z", 1) > 1:
                            kwargs["z"] = int(z_index)
                        if "p" in params and series_key == "P" and s_val is not None:
                            kwargs["p"] = int(s_val)
                        if "s" in params and series_key == "S" and s_val is not None:
                            kwargs["s"] = int(s_val)
                        return np.asarray(f.get_frame_2D(**kwargs))

                    if series_key and series_count > 1:
                        series_vals = [series_index] if series_index is not None else list(range(series_count))
                    else:
                        series_vals = [None]
                    if time_count > 1:
                        time_vals = [time_index_only] if time_index_only is not None else list(range(time_count))
                    else:
                        time_vals = [None]
                    c_count = int(sizes.get("C", 1))
                    for s_idx in series_vals:
                        for t_idx in time_vals:
                            if c_count > 1:
                                chans = [_get_frame_2d(t_idx, c_idx, s_idx) for c_idx in range(c_count)]
                                arr = np.stack(chans, axis=-1)
                            else:
                                arr = _get_frame_2d(t_idx, None, s_idx)
                            arr = self._normalize_channels_last(arr)
                            if series_key and series_count > 1:
                                if time_count > 1:
                                    frame_id_out = f"{series_key}{s_idx}_T{t_idx}"
                                else:
                                    frame_id_out = f"{series_key}{s_idx}"
                            else:
                                frame_id_out = f"T{t_idx}" if time_count > 1 else None
                            meta = io.ImageMeta(
                                axes="YXC" if arr.ndim == 3 else "YX",
                                shape=tuple(arr.shape),
                                sizes=self._sizes_from_axes("YXC" if arr.ndim == 3 else "YX", tuple(arr.shape)),
                                dtype=arr.dtype,
                                series_index=s_idx if series_key and series_count > 1 else None,
                            )
                            frames.append(io.ImageFrame(arr, meta, frame_id_out))
                    return frames
        except Exception as e:
            _logger.warning(f"ND2 lazy read failed for {filename}: {e}")
            return None
        return None

    def _load_nd2_frame_lazy(self, filename, series_index=None, time_index=None):
        try:
            with nd2_lib.ND2File(filename) as f:
                sizes = getattr(f, "sizes", {}) or {}
                series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
                if series_key and series_index is None:
                    series_index = 0
                if time_index is None and sizes.get("T", 1) > 1:
                    time_index = 0
                if hasattr(f, "get_frame_2D"):
                    import inspect
                    sig = inspect.signature(f.get_frame_2D)
                    params = set(sig.parameters.keys())

                    def _get_frame_2d(t_val, c_val, s_val):
                        kwargs = {}
                        if "t" in params and t_val is not None:
                            kwargs["t"] = int(t_val)
                        if "c" in params and c_val is not None:
                            kwargs["c"] = int(c_val)
                        if "z" in params and sizes.get("Z", 1) > 1:
                            kwargs["z"] = 0
                        if "p" in params and series_key == "P" and s_val is not None:
                            kwargs["p"] = int(s_val)
                        if "s" in params and series_key == "S" and s_val is not None:
                            kwargs["s"] = int(s_val)
                        return np.asarray(f.get_frame_2D(**kwargs))

                    c_count = int(sizes.get("C", 1))
                    if c_count > 1:
                        chans = [_get_frame_2d(time_index, c_idx, series_index) for c_idx in range(c_count)]
                        arr = np.stack(chans, axis=-1)
                    else:
                        arr = _get_frame_2d(time_index, None, series_index)
                    return self._normalize_channels_last(arr)
                if hasattr(nd2_lib, "imread"):
                    try:
                        dask_arr = nd2_lib.imread(filename, dask=True)
                    except Exception:
                        dask_arr = None
                    if dask_arr is None:
                        return None
                    axes = self._infer_axes_from_sizes(sizes, getattr(dask_arr, "shape", None))
                    if axes is None:
                        _logger.warning(f"ND2 axes inference failed for {filename}")
                        return None
                    def axis_index(key):
                        return axes.index(key) if key in axes else None
                    indexer = [slice(None)] * len(axes)
                    if series_key and series_index is not None:
                        s_axis = axis_index(series_key)
                        if s_axis is not None:
                            indexer[s_axis] = int(series_index)
                    if sizes.get("T", 1) > 1 and time_index is not None:
                        t_axis = axis_index("T")
                        if t_axis is not None:
                            indexer[t_axis] = int(time_index)
                    if sizes.get("Z", 1) > 1 and axis_index("Z") is not None:
                        indexer[axis_index("Z")] = 0
                    arr = dask_arr[tuple(indexer)]
                    if hasattr(arr, "compute"):
                        arr = arr.compute()
                    arr = np.asarray(arr)
                    if "C" in axes:
                        c_axis = axis_index("C")
                        if c_axis is not None and c_axis != arr.ndim - 1:
                            arr = np.moveaxis(arr, c_axis, -1)
                    return self._normalize_channels_last(arr)
        except Exception as e:
            _logger.warning(f"ND2 lazy frame load failed for {filename}: {e}")
        return None

    def _sizes_from_axes(self, axes, shape):
        if not axes or len(axes) != len(shape):
            return {}
        return {ax: int(dim) for ax, dim in zip(axes, shape)}

    def _infer_axes_from_sizes(self, sizes, shape):
        if not sizes or not shape:
            return None
        axes = [None] * len(shape)
        used = set()

        y_size = sizes.get("Y")
        x_size = sizes.get("X")
        if len(shape) >= 2 and y_size is not None and x_size is not None:
            tail = shape[-2:]
            if tail[0] == y_size and tail[1] == x_size:
                axes[-2], axes[-1] = "Y", "X"
                used.update({len(shape) - 2, len(shape) - 1})
            elif tail[0] == x_size and tail[1] == y_size:
                axes[-2], axes[-1] = "X", "Y"
                used.update({len(shape) - 2, len(shape) - 1})

        for key in ("T", "P", "S", "C", "Z"):
            target = sizes.get(key)
            if target is None:
                continue
            matches = [i for i, dim in enumerate(shape) if i not in used and dim == target]
            if len(matches) == 1:
                axes[matches[0]] = key
                used.add(matches[0])
            elif len(matches) > 1:
                idx = matches[0]
                axes[idx] = key
                used.add(idx)

        if any(ax is None for ax in axes):
            return None
        return "".join(axes)

    def save_segmentation(self, model, suffix="_seg.npy"):
        """
        Queues the current segmentation state for background save.
        """
        if not model.filename or model.masks is None:
            _logger.warning("No filename or masks to save.")
            return

        _logger.info(f"Queueing segmentation save for {model.filename} with suffix {suffix}")

        try:
            if suffix == "_seg.npy":
                payload = self._build_seg_payload(model)
                self._enqueue_save(("seg", payload["save_path"], payload))
            else:
                payload = self._build_pred_payload(model, suffix=suffix)
                self._enqueue_save(("pred", payload["save_path"], payload))
        except Exception as e:
            _logger.error(f"Failed to save segmentation: {e}")

    def save_prediction_with_classes(self, model):
        """Queues prediction data save to _pred.npy including assigned classes."""
        if not model.filename or model.masks is None:
            _logger.warning("No filename or masks to save.")
            return
        payload = self._build_pred_payload(model, suffix="_pred.npy")
        self._enqueue_save(("pred", payload["save_path"], payload))

    def _safe_array_copy(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        return value

    def _safe_list_copy(self, value):
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True).tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _safe_flow_copy(self, flows):
        if not isinstance(flows, list):
            return flows
        copied = []
        for item in flows:
            if isinstance(item, np.ndarray):
                copied.append(np.array(item, copy=True))
            else:
                copied.append(item)
        return copied

    def _build_common_payload(self, model):
        masks = np.array(model.masks, copy=True)
        nmask = int(masks.max()) if masks is not None else 0
        classes = getattr(model, "mask_classes", None)
        if classes is None:
            classes = np.zeros(nmask + 1, dtype=np.int16)
        else:
            classes = np.array(classes, copy=True)
        classes_map = getattr(model, "pred_classes_map", None)
        if classes_map is not None:
            classes_map = np.array(classes_map, copy=True)
        return {
            "filename": model.filename,
            "frame_id": model.frame_id,
            "raw_image": self._safe_array_copy(getattr(model, "raw_image", None)),
            "masks": masks,
            "flows": self._safe_flow_copy(getattr(model, "flows", None)),
            "ismanual": np.array(getattr(model, "ismanual", np.zeros(nmask, dtype=bool)), copy=True),
            "classes": classes,
            "classes_map": classes_map,
            "class_names": self._safe_list_copy(getattr(model, "class_names", None)),
            "class_colors": self._safe_list_copy(getattr(model, "class_colors", None)),
            "diameter": getattr(model, "diameter", None),
        }

    def _build_seg_payload(self, model):
        payload = self._build_common_payload(model)
        payload["save_path"] = self.build_frame_path(model.filename, model.frame_id, "_seg.npy")
        return payload

    def _build_pred_payload(self, model, suffix="_pred.npy"):
        payload = self._build_common_payload(model)
        payload["save_path"] = self.build_frame_path(model.filename, model.frame_id, suffix)
        payload["suffix"] = suffix
        return payload

    def _write_npy_atomic(self, filename, payload):
        directory = os.path.dirname(filename) or "."
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=directory, suffix=".npy") as tmp:
            tmp_name = tmp.name
        try:
            np.save(tmp_name, payload)
            os.replace(tmp_name, filename)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass

    def _ensure_valid_flows(self, masks, flows):
        valid_flows = False
        if isinstance(flows, list) and len(flows) > 0:
            if hasattr(flows[0], "ndim") and flows[0].ndim >= 2:
                valid_flows = True
        if valid_flows:
            return flows
        h, w = masks.shape[-2:]
        return [
            np.zeros((h, w, 3), dtype=np.uint8),
            np.zeros((2, h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32),
            [[]],
        ]

    def _write_seg_payload(self, payload):
        masks = payload["masks"]
        flows = self._ensure_valid_flows(masks, payload.get("flows"))
        save_path = payload["save_path"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_base = os.path.join(tmp_dir, "segtmp")
            io.masks_flows_to_seg(
                payload.get("raw_image"),
                masks.squeeze() if masks.ndim == 3 and masks.shape[0] == 1 else masks,
                flows,
                tmp_base,
                channels=[0, 0],
            )
            tmp_seg_path = tmp_base + "_seg.npy"
            dat = np.load(tmp_seg_path, allow_pickle=True).item()
        classes = payload["classes"]
        classes_map = payload.get("classes_map")
        if classes_map is None:
            try:
                classes_map = classes[masks.astype(np.int64, copy=False)]
            except Exception:
                classes_map = None
        dat["classes"] = classes
        dat["classes_map"] = classes_map
        dat["class_names"] = payload.get("class_names")
        dat["class_colors"] = payload.get("class_colors")
        self._write_npy_atomic(save_path, dat)
        _logger.info(f"Saved {save_path}")

    def _write_pred_payload(self, payload):
        masks = payload["masks"]
        nmask = int(masks.max()) if masks is not None else 0
        dat = {
            "masks": masks,
            "filename": payload["filename"],
            "flows": payload.get("flows"),
            "ismanual": payload.get("ismanual", np.zeros(nmask, dtype=bool)),
            "chan_choose": [0, 0],
            "classes": payload.get("classes"),
            "classes_map": payload.get("classes_map"),
            "class_names": payload.get("class_names"),
            "class_colors": payload.get("class_colors"),
            "diameter": payload.get("diameter"),
        }
        try:
            from cellpose import utils

            outlines = utils.masks_to_outlines(masks)
            dat["outlines"] = outlines * masks
        except Exception:
            pass
        filename = payload["save_path"]
        self._write_npy_atomic(filename, dat)
        _logger.info(f"Saved {filename}")

    def _enqueue_save(self, save_request):
        _mode, path, _payload = save_request
        with self._save_cv:
            # Keep only the latest snapshot per destination file.
            self._pending_saves[path] = save_request
            self._save_cv.notify()

    def _save_worker_loop(self):
        while True:
            with self._save_cv:
                while not self._pending_saves and not self._stop_save_worker:
                    self._save_cv.wait()
                if self._stop_save_worker:
                    return
            # debounce to coalesce rapid updates; newer edits replace pending request
            time.sleep(self._save_debounce_s)
            with self._save_cv:
                if self._stop_save_worker:
                    return
                batch = list(self._pending_saves.values())
                self._pending_saves.clear()
            for save_request in batch:
                try:
                    mode, _path, payload = save_request
                    if mode == "seg":
                        self._write_seg_payload(payload)
                    else:
                        self._write_pred_payload(payload)
                except Exception as e:
                    _logger.error(f"Background save failed: {e}")

    def save_visualization_mask(self, filename, frame_id, masks, plugin_name=None):
        if not filename or masks is None:
            return None
        out_path = self.build_frame_path(filename, frame_id, "_viz.npy")
        payload = {"masks": masks}
        if plugin_name:
            payload["plugin"] = plugin_name
        try:
            np.save(out_path, payload)
            return out_path
        except Exception as e:
            _logger.error(f"Failed to save visualization mask: {e}")
            return None

    def load_visualization_mask(self, filename, frame_id, plugin_name=None):
        if not filename:
            return None
        path = self.build_frame_path(filename, frame_id, "_viz.npy")
        if not os.path.exists(path):
            return None
        try:
            dat = np.load(path, allow_pickle=True).item()
            if plugin_name and dat.get("plugin") not in (None, plugin_name):
                return None
            return dat.get("masks")
        except Exception as e:
            _logger.error(f"Failed to load visualization mask: {e}")
            return None
