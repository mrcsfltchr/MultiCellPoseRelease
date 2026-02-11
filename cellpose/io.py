"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os, warnings, glob, shutil
from dataclasses import dataclass
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys
from tqdm import tqdm
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple
from .version import version_str
from roifile import ImagejRoi, roiwrite

try:
    from qtpy import QtGui, QtCore, Qt, QtWidgets
    from qtpy.QtWidgets import QMessageBox
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

try:
    import nd2
    ND2 = True
except:
    ND2 = False

try:
    from readlif.reader import LifFile
    LIF = True
except:
    LIF = False
try:
    import nrrd
    NRRD = True
except:
    NRRD = False

try:
    from google.cloud import storage
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

io_logger = logging.getLogger(__name__)
SUPPRESS_NON_TIFF_INFO = False
SUPPRESS_OVERWRITE_TILES_PROMPT = False


@dataclass(frozen=True)
class ImageMeta:
    axes: Optional[str]
    shape: Tuple[int, ...]
    sizes: Dict[str, int]
    dtype: Optional[np.dtype]
    series_index: Optional[int] = None


@dataclass(frozen=True)
class ImageData:
    array: np.ndarray
    meta: ImageMeta


@dataclass(frozen=True)
class ImageFrame:
    array: np.ndarray
    meta: ImageMeta
    frame_id: Optional[str]


class ImageReader:
    extensions: Tuple[str, ...] = ()

    def can_read(self, filename: str) -> bool:
        ext = os.path.splitext(filename)[-1].lower()
        return ext in self.extensions

    def read(self, filename: str) -> ImageData:
        raise NotImplementedError

    def read_frame(self, filename: str, frame_id: Optional[str]) -> Optional[ImageFrame]:
        data = self.read(filename)
        return ImageFrame(data.array, data.meta, None)

    def get_series_time_info(self, filename: str) -> Tuple[Optional[str], int, int]:
        data = self.read(filename)
        sizes = data.meta.sizes or {}
        series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
        series_count = int(sizes.get(series_key, 1)) if series_key else 1
        time_count = int(sizes.get("T", 1))
        return series_key, series_count, time_count
    def iter_frames(self, filename: str) -> Iterable[ImageFrame]:
        data = self.read(filename)
        yield ImageFrame(data.array, data.meta, None)


_READERS: List[ImageReader] = []
_ND2_WARN_BYTES = 500 * 1024 * 1024


def register_reader(reader: ImageReader) -> None:
    _READERS.append(reader)


def _get_reader(filename: str) -> Optional[ImageReader]:
    for reader in _READERS:
        if reader.can_read(filename):
            return reader
    return None


def _sizes_from_axes(axes: Optional[str], shape: Tuple[int, ...]) -> Dict[str, int]:
    if not axes or len(axes) != len(shape):
        return {}
    return {ax: int(dim) for ax, dim in zip(axes, shape)}


def _maybe_warn_large_nd2(sizes: Dict[str, int], dtype: Optional[np.dtype], filename: str) -> None:
    if not sizes:
        return
    try:
        dtype = np.dtype(dtype) if dtype is not None else np.dtype("uint16")
        total_elems = int(np.prod([int(v) for v in sizes.values()]))
        est_bytes = total_elems * dtype.itemsize
    except Exception:
        return
    if est_bytes >= _ND2_WARN_BYTES:
        est_mb = est_bytes / (1024 * 1024)
        io_logger.warning(
            f"ND2 file {filename} is large (~{est_mb:.1f} MB). Loading into memory may be slow or fail."
        )


def _build_axes_from_sizes(shape: Tuple[int, ...], sizes: Dict[str, int]) -> Optional[str]:
    if not sizes:
        return None
    order = "TPSZCYX"
    axes = [ax for ax in order if ax in sizes]
    if len(axes) != len(shape):
        return None
    return "".join(axes)


def _drop_axis(axes: Optional[str], axis: str) -> Optional[str]:
    if not axes:
        return axes
    return "".join([ax for ax in axes if ax != axis])


def frame_id_to_suffix(frame_id: Optional[str]) -> str:
    if not frame_id:
        return ""
    return f"__{frame_id}"


def parse_frame_id(frame_id: Optional[str]) -> Dict[str, int]:
    if not frame_id:
        return {}
    parts = frame_id.split("_")
    parsed: Dict[str, int] = {}
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


def logger_setup(cp_path=".cellpose", logfile_name="run.log", stdout_file_replacement=None):
    cp_dir = pathlib.Path.home().joinpath(cp_path)
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath(logfile_name)
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    handlers = [logging.FileHandler(log_file),]
    if stdout_file_replacement is not None:
        handlers.append(logging.FileHandler(stdout_file_replacement))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=handlers,
                    force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"WRITING LOG OUTPUT TO {log_file}")
    logger.info(version_str)

    return logger, log_file


from . import utils, plot, transforms

# helper function to check for a path; if it doesn't exist, make it
def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def outlines_to_text(base, outlines):
    with open(base + "_cp_outlines.txt", "w") as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ",".join(map(str, xy))
            f.write(xy_str)
            f.write("\n")


def _load_seg(*args, **kwargs):
    try:
        from .gui import io as gui_io
    except Exception as exc:
        raise AttributeError("cellpose.io._load_seg requires GUI components") from exc
    return gui_io._load_seg(*args, **kwargs)


def load_dax(filename):
    ### modified from ZhuangLab github:
    ### https://github.com/ZhuangLab/storm-analysis/blob/71ae493cbd17ddb97938d0ae2032d97a0eaa76b2/storm_analysis/sa_library/datareader.py#L156

    inf_filename = os.path.splitext(filename)[0] + ".inf"
    if not os.path.exists(inf_filename):
        io_logger.critical(
            f"ERROR: no inf file found for dax file {filename}, cannot load dax without it"
        )
        return None

    ### get metadata
    image_height, image_width = None, None
    # extract the movie information from the associated inf file
    size_re = re.compile(r"frame dimensions = ([\d]+) x ([\d]+)")
    length_re = re.compile(r"number of frames = ([\d]+)")
    endian_re = re.compile(r" (big|little) endian")

    with open(inf_filename, "r") as inf_file:
        lines = inf_file.read().split("\n")
        for line in lines:
            m = size_re.match(line)
            if m:
                image_height = int(m.group(2))
                image_width = int(m.group(1))
            m = length_re.match(line)
            if m:
                number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    bigendian = 1
                else:
                    bigendian = 0
    # set defaults, warn the user that they couldn"t be determined from the inf file.
    if not image_height:
        io_logger.warning("could not determine dax image size, assuming 256x256")
        image_height = 256
        image_width = 256

    ### load image
    img = np.memmap(filename, dtype="uint16",
                    shape=(number_frames, image_height, image_width))
    if bigendian:
        img = img.byteswap()
    img = np.array(img)

    return img


def imread(filename, return_first_tile=True, save_tiles=True):
    """
    Read in an image file with tif or image file type supported by cv2.

    Args:
        filename (str): The path to the image file.
        return_first_tile (bool): For ND2/LIF files, return the first 256x256 tile
            instead of the full image (default True, legacy behavior).

    Returns:
        numpy.ndarray: The image data as a NumPy array.

    Raises:
        None

    Raises an error if the image file format is not supported.

    Examples:
        >>> img = imread("image.tif")
    """
    # ensure that extension check is not case sensitive
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tif" or ext == ".tiff" or ext == ".flex":
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]["shape"]
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                io_logger.info(f"reading tiff with {ltif} planes")
                img = np.zeros((ltif, *shape), dtype=dtype)
                for i, page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)
        # Check for channels first (C, Y, X) and transpose to (Y, X, C)
        if img.ndim == 3 and img.shape[0] < 10 and img.shape[1] > 10 and img.shape[2] > 10:
            img = img.transpose(1, 2, 0)
        return img
    elif ext == ".dax":
        img = load_dax(filename)
        return img
    elif ext == ".nd2":
        if not ND2:
            io_logger.critical("ERROR: need to 'pip install nd2' to load in .nd2 file")
            return None
        # GUI-level tiling disabled; always return full image without saving tiles.
        save_tiles = False
        return_first_tile = False
        try:
            with nd2.ND2File(filename) as f:
                sizes = getattr(f, "sizes", {}) or {}
                dtype = getattr(f, "dtype", None)
                _maybe_warn_large_nd2(sizes, dtype, filename)
                n_channels = int(sizes.get("C", 1))
                n_series = int(sizes.get("T", 1)) * int(sizes.get("Z", 1)) * int(sizes.get("S", 1))
                img = f.asarray()
            img = np.squeeze(img)
            if img.ndim == 3 and 'C' in sizes and sizes['C'] > 1 and sizes.get('Z', 1) == 1 and sizes.get('T', 1) == 1:
                if img.shape[0] == sizes['C']:
                    img = img.transpose(1, 2, 0)
            tiled = None
            _maybe_warn_non_tiff(
                filename,
                kind="ND2",
                n_series=n_series,
                n_channels=n_channels,
                saved_tiles=bool(save_tiles),
            )
            return img
        except Exception as e:
            io_logger.critical(f"ERROR: could not read nd2 file ({e})")
            return None
    elif ext == ".lif":
        if not LIF:
            io_logger.critical("ERROR: need to 'pip install readlif' to load in .lif file")
            raise RuntimeError("readlif not installed")
        # GUI-level tiling disabled; always return full image without saving tiles.
        save_tiles = False
        return_first_tile = False
        try:
            lif_obj = LifFile(filename)
            if lif_obj is None:
                raise AttributeError("readlif.Reader / readlif.LifFile not available")
            stacks = []
            shapes = []
            # iterate through images in the LIF file
            iterator = lif_obj.get_iter_image()
            for img_info in iterator:
                arr = None
                try:
                    dims_n = getattr(img_info, "dims_n", None)
                    if dims_n:
                        io_logger.info(f"LIF image dims_n: {dims_n}")
                    if hasattr(img_info, "get_iter_c"):
                        chans = [np.asarray(p) for p in img_info.get_iter_c()]
                        arr = np.stack(chans, axis=-1) if len(chans) > 1 else chans[0]
                    elif hasattr(img_info, "get_frame"):
                        arr = np.asarray(img_info.get_frame())
                    else:
                        arr = np.asarray(img_info)
                except Exception:
                    arr = None
                if arr is not None:
                    stacks.append(arr)
                    shapes.append(arr.shape)

            if len(stacks) == 0:
                raise RuntimeError("no images found in LIF file")

            # determine channel options from first stack
            first_stack = stacks[0]
            io_logger.info(f"image dimensions: {first_stack.shape}")
            n_channels = 1
            if first_stack.ndim >= 3 and first_stack.shape[-1] <= 10:
                n_channels = first_stack.shape[-1]
                io_logger.info(f"decided last dimension are fluorescent channels")
            elif first_stack.ndim >= 3 and first_stack.shape[0] <= 10:
                n_channels = first_stack.shape[0]
            channels_to_save = _prompt_channel_split(n_channels)

            # tile and save each series separately; return first tile of first series
            first_tile = None
            exported_paths = []
            for i, arr in enumerate(stacks):
                arr = np.squeeze(arr)
                # print(arr)
                base_series = os.path.splitext(filename)[0] + f"_series{i}"
                tiled, paths = None, []
                exported_paths.extend(paths)
                if i == 0:
                    first_tile = tiled if return_first_tile else arr
                    if tiled is not None and tiled.ndim >= 3 and tiled.shape[-1] <= 10:
                        n_channels = tiled.shape[-1]

            if shapes:
                io_logger.info(f"LIF series shapes: {shapes}")

            _maybe_warn_non_tiff(
                filename,
                kind="LIF",
                n_series=len(stacks),
                n_channels=n_channels,
                saved_tiles=bool(save_tiles),
            )
            return first_tile
        except Exception as e:
            io_logger.critical(f"ERROR: could not read lif file ({e})")
            raise
    elif ext == ".nrrd":
        if not NRRD:
            io_logger.critical(
                "ERROR: need to 'pip install pynrrd' to load in .nrrd file")
            return None
        else:
            img, metadata = nrrd.read(filename)
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            return img
    elif ext != ".npy":
        try:
            img = cv2.imread(filename, -1)  #cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2, 1, 0]]
            return img
        except Exception as e:
            io_logger.critical("ERROR: could not read file, %s" % e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat["masks"]
            return masks
        except Exception as e:
            io_logger.critical("ERROR: could not read masks from file, %s" % e)
            return None


class _TiffReader(ImageReader):
    extensions = (".tif", ".tiff", ".flex")

    def _is_rgb_samples_axis(self, series, axes: Optional[str], shape: Tuple[int, ...]) -> bool:
        if not isinstance(axes, str) or len(axes) != len(shape):
            return False
        if "S" not in axes:
            return False
        s_idx = axes.index("S")
        s_dim = int(shape[s_idx])
        if s_dim not in (3, 4):
            return False
        try:
            pages = getattr(series, "pages", None)
            first_page = pages[0] if pages is not None and len(pages) > 0 else None
            photometric = getattr(first_page, "photometric", None)
            if photometric is None:
                return False
            # tifffile enum/name compatibility
            name = getattr(photometric, "name", None)
            if isinstance(name, str) and name.upper() == "RGB":
                return True
            return int(photometric) == 2
        except Exception:
            return False

    def _normalize_axes(self, series, axes: Optional[str], shape: Tuple[int, ...]) -> Optional[str]:
        if not isinstance(axes, str):
            return None
        if self._is_rgb_samples_axis(series, axes, shape):
            # TIFF RGB is often YXS; treat S as channel axis, not series axis.
            return axes.replace("S", "C")
        return axes

    def read(self, filename: str) -> ImageData:
        with tifffile.TiffFile(filename) as tif:
            series = tif.series[0]
            axes = getattr(series, "axes", None)
            arr = series.asarray()
        axes = self._normalize_axes(series, axes, tuple(arr.shape))
        meta = ImageMeta(
            axes=axes,
            shape=tuple(arr.shape),
            sizes=_sizes_from_axes(axes, tuple(arr.shape)),
            dtype=arr.dtype,
        )
        return ImageData(arr, meta)

    def read_frame(self, filename: str, frame_id: Optional[str]) -> Optional[ImageFrame]:
        if not frame_id:
            data = self.read(filename)
            return ImageFrame(_normalize_channels_last(data.array), data.meta, None)
        with tifffile.TiffFile(filename) as tif:
            series = tif.series[0]
            axes = getattr(series, "axes", None)
            if not isinstance(axes, str):
                arr = series.asarray()
                meta = ImageMeta(
                    axes=None,
                    shape=tuple(arr.shape),
                    sizes=_sizes_from_axes(None, tuple(arr.shape)),
                    dtype=arr.dtype,
                )
                return ImageFrame(_normalize_channels_last(arr), meta, None)
            axes = self._normalize_axes(series, axes, tuple(series.shape))

            parsed = parse_frame_id(frame_id)
            key = []
            for ax in axes:
                if ax in ("T", "Z", "S", "P"):
                    idx = parsed.get(ax)
                    if idx is None:
                        idx = 0
                    key.append(int(idx))
                else:
                    key.append(slice(None))
            key_tuple = tuple(key)
            try:
                arr = series.asarray(key=key_tuple)
            except Exception:
                arr = series.asarray()[key_tuple]
            arr = _normalize_channels_last(arr)
            new_axes = "".join(
                [ax for ax, k in zip(axes, key) if not isinstance(k, int)]
            )
            meta = ImageMeta(
                axes=new_axes,
                shape=tuple(arr.shape),
                sizes=_sizes_from_axes(new_axes, tuple(arr.shape)),
                dtype=arr.dtype,
            )
            return ImageFrame(arr, meta, frame_id)

    def get_series_time_info(self, filename: str) -> Tuple[Optional[str], int, int]:
        with tifffile.TiffFile(filename) as tif:
            series = tif.series[0]
            axes = getattr(series, "axes", None)
            shape = tuple(getattr(series, "shape", ()))
            axes = self._normalize_axes(series, axes, shape)
            sizes = _sizes_from_axes(axes, shape)
        series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
        series_count = int(sizes.get(series_key, 1)) if series_key else 1
        time_count = int(sizes.get("T", 1))
        return series_key, series_count, time_count


class _Nd2Reader(ImageReader):
    extensions = (".nd2",)

    def __init__(self):
        self._cache = {}

    def _get_cache(self, filename: str) -> Optional[Dict[str, object]]:
        return self._cache.get(filename)

    def _set_cache(self, filename: str, entry: Dict[str, object]) -> None:
        self._cache.clear()
        self._cache[filename] = entry

    def _load_meta(self, filename: str) -> Dict[str, object]:
        with nd2.ND2File(filename) as f:
            sizes = getattr(f, "sizes", {}) or {}
            dtype = getattr(f, "dtype", None)
            axes = getattr(f, "axes", None)
        axes = "".join(axes) if isinstance(axes, (list, tuple)) else axes
        entry = {"sizes": sizes, "dtype": dtype, "axes": axes}
        self._set_cache(filename, entry)
        return entry

    def _get_meta(self, filename: str) -> Dict[str, object]:
        entry = self._get_cache(filename)
        if entry is None:
            entry = self._load_meta(filename)
        return entry

    def _infer_axes_from_sizes(self, sizes: Dict[str, int], shape: Tuple[int, ...]) -> Optional[str]:
        return _build_axes_from_sizes(shape, sizes)

    def _get_frame_2d(self, f, sizes: Dict[str, int], series_key: Optional[str],
                      series_index: Optional[int], time_index: Optional[int]) -> Optional[np.ndarray]:
        if not hasattr(f, "get_frame_2D"):
            return None
        import inspect
        sig = inspect.signature(f.get_frame_2D)
        params = set(sig.parameters.keys())

        def _get_frame(t_val, c_val, s_val):
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
            chans = [_get_frame(time_index, c_idx, series_index) for c_idx in range(c_count)]
            arr = np.stack(chans, axis=-1)
        else:
            arr = _get_frame(time_index, None, series_index)
        return _normalize_channels_last(arr)

    def _load_dask_frame(self, filename: str, sizes: Dict[str, int], axes: Optional[str],
                         series_key: Optional[str], series_index: Optional[int],
                         time_index: Optional[int]) -> Optional[np.ndarray]:
        if not hasattr(nd2, "imread"):
            return None
        try:
            dask_arr = nd2.imread(filename, dask=True)
        except Exception:
            return None
        if dask_arr is None:
            return None
        if axes is None:
            axes = self._infer_axes_from_sizes(sizes, tuple(getattr(dask_arr, "shape", ())))
        if axes is None:
            return None

        def axis_index(key: str) -> Optional[int]:
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
        if sizes.get("Z", 1) > 1:
            z_axis = axis_index("Z")
            if z_axis is not None:
                indexer[z_axis] = 0
        arr = dask_arr[tuple(indexer)]
        if hasattr(arr, "compute"):
            arr = arr.compute()
        arr = np.asarray(arr)
        if "C" in axes:
            c_axis = axis_index("C")
            if c_axis is not None and c_axis != arr.ndim - 1:
                arr = np.moveaxis(arr, c_axis, -1)
        return _normalize_channels_last(arr)

    def _load_frame(self, f, filename: str, sizes: Dict[str, int], axes: Optional[str],
                    series_key: Optional[str], series_index: Optional[int],
                    time_index: Optional[int]) -> Optional[np.ndarray]:
        arr = self._get_frame_2d(f, sizes, series_key, series_index, time_index)
        if arr is not None:
            return arr
        arr = self._load_dask_frame(filename, sizes, axes, series_key, series_index, time_index)
        if arr is not None:
            return arr
        try:
            arr = f.asarray()
        except Exception:
            return None
        axes_local = "".join(axes) if isinstance(axes, (list, tuple)) else axes
        if not isinstance(axes_local, str) or len(axes_local) != arr.ndim:
            axes_local = self._infer_axes_from_sizes(sizes, tuple(arr.shape))
        if axes_local and series_index is not None:
            if "S" in axes_local:
                arr = np.take(arr, series_index, axis=axes_local.index("S"))
                axes_local = _drop_axis(axes_local, "S")
            elif "P" in axes_local:
                arr = np.take(arr, series_index, axis=axes_local.index("P"))
                axes_local = _drop_axis(axes_local, "P")
        if axes_local and time_index is not None and "T" in axes_local:
            arr = np.take(arr, time_index, axis=axes_local.index("T"))
            axes_local = _drop_axis(axes_local, "T")
        if axes_local and "C" in axes_local:
            c_idx = axes_local.index("C")
            if c_idx != len(axes_local) - 1:
                arr = np.moveaxis(arr, c_idx, -1)
        return _normalize_channels_last(arr)

    def read(self, filename: str) -> ImageData:
        if not ND2:
            raise RuntimeError("nd2 not installed")
        meta = self._get_meta(filename)
        sizes = meta.get("sizes", {}) or {}
        dtype = meta.get("dtype", None)
        axes = meta.get("axes", None)
        _maybe_warn_large_nd2(sizes, dtype, filename)
        with nd2.ND2File(filename) as f:
            arr = f.asarray()
        arr = np.squeeze(arr)
        axes = "".join(axes) if isinstance(axes, (list, tuple)) else axes
        if not isinstance(axes, str) or len(axes) != arr.ndim:
            axes = _build_axes_from_sizes(tuple(arr.shape), sizes)
        if axes and "C" in axes:
            c_idx = axes.index("C")
            if c_idx != len(axes) - 1:
                arr = np.moveaxis(arr, c_idx, -1)
                axes = axes.replace("C", "") + "C"
        meta = ImageMeta(
            axes=axes if isinstance(axes, str) else None,
            shape=tuple(arr.shape),
            sizes={k: int(v) for k, v in sizes.items()} if sizes else _sizes_from_axes(axes, tuple(arr.shape)),
            dtype=arr.dtype,
        )
        return ImageData(arr, meta)

    def get_series_time_info(self, filename: str) -> Tuple[Optional[str], int, int]:
        if not ND2:
            return None, 1, 1
        meta = self._get_meta(filename)
        sizes = meta.get("sizes", {}) or {}
        series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
        series_count = int(sizes.get(series_key, 1)) if series_key else 1
        time_count = int(sizes.get("T", 1))
        return series_key, series_count, time_count

    def iter_frames(self, filename: str) -> Iterable[ImageFrame]:
        if not ND2:
            raise RuntimeError("nd2 not installed")
        meta = self._get_meta(filename)
        sizes = meta.get("sizes", {}) or {}
        axes = meta.get("axes", None)
        series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
        series_count = int(sizes.get(series_key, 1)) if series_key else 1
        time_count = int(sizes.get("T", 1))
        series_vals = [None] if series_count <= 1 else list(range(series_count))
        time_vals = [None] if time_count <= 1 else list(range(time_count))
        with nd2.ND2File(filename) as f:
            for s_idx in series_vals:
                for t_idx in time_vals:
                    arr = self._load_frame(f, filename, sizes, axes, series_key, s_idx, t_idx)
                    if arr is None:
                        continue
                    if series_key and series_count > 1:
                        frame_id = f"{series_key}{s_idx}_T{t_idx}" if time_count > 1 else f"{series_key}{s_idx}"
                    else:
                        frame_id = f"T{t_idx}" if time_count > 1 else None
                    axes_out = "YXC" if arr.ndim == 3 else "YX"
                    meta_out = ImageMeta(
                        axes=axes_out,
                        shape=tuple(arr.shape),
                        sizes=_sizes_from_axes(axes_out, tuple(arr.shape)),
                        dtype=arr.dtype,
                        series_index=s_idx if series_key and series_count > 1 else None,
                    )
                    yield ImageFrame(arr, meta_out, frame_id)

    def read_frame(self, filename: str, frame_id: Optional[str]) -> Optional[ImageFrame]:
        if not ND2:
            raise RuntimeError("nd2 not installed")
        if frame_id is None:
            data = self.read(filename)
            return ImageFrame(data.array, data.meta, None)
        parsed = parse_frame_id(frame_id)
        series_index = parsed.get("S", parsed.get("P"))
        time_index = parsed.get("T")
        meta = self._get_meta(filename)
        sizes = meta.get("sizes", {}) or {}
        axes = meta.get("axes", None)
        series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
        with nd2.ND2File(filename) as f:
            arr = self._load_frame(f, filename, sizes, axes, series_key, series_index, time_index)
        if arr is None:
            return None
        axes_out = "YXC" if arr.ndim == 3 else "YX"
        meta_out = ImageMeta(
            axes=axes_out,
            shape=tuple(arr.shape),
            sizes=_sizes_from_axes(axes_out, tuple(arr.shape)),
            dtype=arr.dtype,
            series_index=series_index if series_key and sizes.get(series_key, 1) > 1 else None,
        )
        return ImageFrame(arr, meta_out, frame_id)


class _LifReader(ImageReader):
    extensions = (".lif",)

    def __init__(self):
        self._cache = {}

    def read(self, filename: str) -> ImageData:
        if not LIF:
            raise RuntimeError("readlif not installed")
        entry = self._get_cache(filename)
        if entry is None:
            entry = self._load_cache(filename)
        images = entry.get("images", [])
        first = images[0] if images else None
        if first is None:
            raise RuntimeError("no images found in LIF file")
        arr, axes = _lif_image_to_array(first)
        meta = ImageMeta(
            axes=axes,
            shape=tuple(arr.shape),
            sizes=_sizes_from_axes(axes, tuple(arr.shape)),
            dtype=arr.dtype,
            series_index=0,
        )
        return ImageData(arr, meta)

    def iter_frames(self, filename: str) -> Iterable[ImageFrame]:
        if not LIF:
            raise RuntimeError("readlif not installed")
        entry = self._get_cache(filename)
        if entry is None:
            entry = self._load_cache(filename)
        images = entry.get("images", [])
        for idx, img_info in enumerate(images):
            arr, axes = _lif_image_to_array(img_info)
            meta = ImageMeta(
                axes=axes,
                shape=tuple(arr.shape),
                sizes=_sizes_from_axes(axes, tuple(arr.shape)),
                dtype=arr.dtype,
                series_index=idx,
            )
            frame_id = f"S{idx}"
            yield ImageFrame(arr, meta, frame_id)

    def get_series_time_info(self, filename: str) -> Tuple[Optional[str], int, int]:
        if not LIF:
            return None, 1, 1
        entry = self._get_cache(filename)
        if entry is None:
            entry = self._load_cache(filename)
        images = entry.get("images", [])
        series_count = len(images)
        series_key = "S" if series_count > 1 else None
        time_count = 1
        if images:
            dims_n = getattr(images[0], "dims_n", None)
            if isinstance(dims_n, dict):
                for key in ("T", "t", 4):
                    if key in dims_n:
                        try:
                            time_count = int(dims_n[key])
                        except Exception:
                            time_count = 1
                        break
            if time_count <= 1:
                dims = getattr(images[0], "dims", None)
                t_val = getattr(dims, "t", None) if dims is not None else None
                if t_val is not None:
                    try:
                        time_count = int(t_val)
                    except Exception:
                        time_count = 1
        return series_key, series_count, time_count

    def read_frame(self, filename: str, frame_id: Optional[str]) -> Optional[ImageFrame]:
        if not LIF:
            raise RuntimeError("readlif not installed")
        if frame_id is None:
            data = self.read(filename)
            return ImageFrame(data.array, data.meta, None)
        parsed = parse_frame_id(frame_id)
        series_index = parsed.get("S", parsed.get("P", 0))
        time_index = parsed.get("T")
        entry = self._get_cache(filename)
        if entry is None:
            entry = self._load_cache(filename)
        images = entry.get("images", [])
        if not images:
            return None
        if series_index < 0 or series_index >= len(images):
            return None
        img_info = images[series_index]
        arr = self._load_lif_frame(img_info, time_index=time_index)
        if arr is None:
            return None
        arr = _normalize_channels_last(arr)
        meta = ImageMeta(
            axes="YXC" if arr.ndim == 3 else "YX",
            shape=tuple(arr.shape),
            sizes=_sizes_from_axes("YXC" if arr.ndim == 3 else "YX", tuple(arr.shape)),
            dtype=arr.dtype,
            series_index=series_index,
        )
        return ImageFrame(arr, meta, frame_id)

    def _get_cache(self, filename: str) -> Optional[Dict[str, object]]:
        return self._cache.get(filename)

    def _set_cache(self, filename: str, entry: Dict[str, object]) -> None:
        self._cache.clear()
        self._cache[filename] = entry

    def _load_cache(self, filename: str) -> Dict[str, object]:
        lif_obj = LifFile(filename)
        images = list(lif_obj.get_iter_image())
        entry = {"lif": lif_obj, "images": images}
        self._set_cache(filename, entry)
        return entry

    def _load_lif_frame(self, img_info, time_index: Optional[int] = None) -> Optional[np.ndarray]:
        if time_index is not None and hasattr(img_info, "get_frame"):
            if hasattr(img_info, "get_iter_c"):
                try:
                    c_count = len(list(img_info.get_iter_c()))
                except Exception:
                    c_count = 0
                chans = [
                    np.asarray(img_info.get_frame(t=int(time_index), c=int(c_idx)))
                    for c_idx in range(c_count)
                ]
                return np.stack(chans, axis=-1) if chans else None
            return np.asarray(img_info.get_frame(t=int(time_index)))
        if time_index is not None and hasattr(img_info, "get_iter_t"):
            for idx, frame in enumerate(img_info.get_iter_t()):
                if idx == int(time_index):
                    return np.asarray(frame)
        if hasattr(img_info, "get_iter_c"):
            chans = [np.asarray(p) for p in img_info.get_iter_c()]
            return np.stack(chans, axis=-1) if len(chans) > 1 else (chans[0] if chans else None)
        if hasattr(img_info, "get_frame"):
            return np.asarray(img_info.get_frame())
        try:
            return np.asarray(img_info)
        except Exception:
            return None


def _lif_image_to_array(img_info):
    def _lif_frame_to_array(frame_info):
        if hasattr(frame_info, "get_iter_c"):
            chans = [np.asarray(p) for p in frame_info.get_iter_c()]
            return np.stack(chans, axis=-1) if len(chans) > 1 else chans[0]
        if hasattr(frame_info, "get_frame"):
            return np.asarray(frame_info.get_frame())
        return np.asarray(frame_info)

    try:
        arr = _lif_frame_to_array(img_info)
    except Exception as exc:
        raise RuntimeError(f"failed to read LIF frame ({exc})") from exc
    axes = None
    dims_n = getattr(img_info, "dims_n", None)
    dims = getattr(img_info, "dims", None)
    if isinstance(dims_n, dict):
        t_count = None
        if dims is not None:
            t_count = getattr(dims, "t", None)
            t_count = int(t_count) if t_count is not None else None
        if t_count is None:
            for key in ("T", "t", 4):
                if key in dims_n:
                    t_count = int(dims_n[key])
                    break
        c_count = None
        if hasattr(img_info, "get_iter_c"):
            try:
                c_count = len(list(img_info.get_iter_c()))
            except Exception:
                c_count = None
        can_index_frames = hasattr(img_info, "get_frame")
        if t_count and t_count > 1 and c_count and c_count > 1 and can_index_frames and arr.ndim <= 3:
            try:
                t_frames = []
                for t_index in range(t_count):
                    c_frames = [
                        np.asarray(img_info.get_frame(t=t_index, c=c_index))
                        for c_index in range(c_count)
                    ]
                    t_frames.append(np.stack(c_frames, axis=-1))
                arr = np.stack(t_frames, axis=0)
            except Exception:
                can_index_frames = False
        if t_count and t_count > 1 and hasattr(img_info, "get_iter_t") and arr.ndim <= 3 and not can_index_frames:
            try:
                time_frames = [_lif_frame_to_array(frame) for frame in img_info.get_iter_t()]
                if time_frames:
                    arr = np.stack(time_frames, axis=0)
            except Exception as exc:
                raise RuntimeError(f"failed to read LIF timepoints ({exc})") from exc
    if isinstance(dims_n, dict):
        axes = _build_axes_from_sizes(tuple(arr.shape), dims_n)
    return arr, axes


def read_image_data(filename: str) -> ImageData:
    reader = _get_reader(filename)
    if reader is None:
        arr = imread(filename)
        meta = ImageMeta(
            axes=None,
            shape=tuple(arr.shape) if arr is not None else (),
            sizes={},
            dtype=arr.dtype if arr is not None else None,
        )
        return ImageData(arr, meta)
    return reader.read(filename)


def iter_image_frames(filename: str) -> List[ImageFrame]:
    reader = _get_reader(filename)
    if reader is None:
        data = read_image_data(filename)
        return [ImageFrame(data.array, data.meta, None)]
    return list(reader.iter_frames(filename))


def read_image_frame(filename: str, frame_id: Optional[str]) -> Optional[ImageFrame]:
    reader = _get_reader(filename)
    if reader is None:
        data = read_image_data(filename)
        return ImageFrame(data.array, data.meta, None)
    return reader.read_frame(filename, frame_id)


def get_series_time_info(filename: str) -> Tuple[Optional[str], int, int]:
    reader = _get_reader(filename)
    if reader is None:
        data = read_image_data(filename)
        sizes = data.meta.sizes or {}
        series_key = "S" if sizes.get("S", 1) > 1 else "P" if sizes.get("P", 1) > 1 else None
        series_count = int(sizes.get(series_key, 1)) if series_key else 1
        time_count = int(sizes.get("T", 1))
        return series_key, series_count, time_count
    return reader.get_series_time_info(filename)


register_reader(_TiffReader())
register_reader(_Nd2Reader())
register_reader(_LifReader())


def _infer_channel_axis(arr, lif_channel_axis=None):
    """Heuristic to find channel axis. lif_channel_axis can override."""
    if lif_channel_axis is not None:
        return lif_channel_axis
    if arr.ndim < 3:
        return None
    # guess: last axis if small, else first axis if small
    if arr.shape[-1] <= 10:
        return arr.ndim - 1
    if arr.shape[0] <= 10:
        return 0
    return None


def _normalize_channels_last(arr):
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


def _maybe_tile_image(img, tile_size=256, base_path=None, channels_to_save=None, return_paths=False, channel_axis=None,
                      progress_cb=None, tile_dir_name=None):
    """Split large images into non-overlapping tiles; optionally save each tile as a TIFF.

    Returns the first tile so the GUI treats it as a single image, while all tiles
    are written to disk in a sibling 'tifs' directory for browsing/labeling.
    """
    if img is None or img.ndim < 2:
        return (img, []) if return_paths else img

    # normalize axis order: move channel to last if specified/guessed
    ch_axis = _infer_channel_axis(img, channel_axis)
    arr = img
    if ch_axis is not None and ch_axis != arr.ndim - 1:
        arr = np.moveaxis(arr, ch_axis, -1)
        ch_axis = arr.ndim - 1

    # pick two spatial axes (largest two dims among non-channel axes)
    axes = list(range(arr.ndim - (1 if ch_axis is not None else 0)))
    lengths = sorted([(ax, arr.shape[ax]) for ax in axes], key=lambda x: x[1])
    spatial_axes = [ax for ax, _ in lengths[-2:]] if len(lengths) >= 2 else axes
    other_axes = [ax for ax in range(arr.ndim) if ax not in spatial_axes + ([ch_axis] if ch_axis is not None else [])]
    new_order = other_axes + spatial_axes + ([ch_axis] if ch_axis is not None else [])
    arr = np.transpose(arr, new_order)

    # spatial dims positions after transpose
    if ch_axis is not None:
        spatial0 = arr.shape[-3] if arr.ndim >= 3 else arr.shape[-2]
        spatial1 = arr.shape[-2]
    else:
        spatial0 = arr.shape[-2] if arr.ndim >= 2 else arr.shape[-1]
        spatial1 = arr.shape[-1] if arr.ndim >= 2 else arr.shape[-1]

    if spatial0 <= tile_size and spatial1 <= tile_size:
        return (arr, []) if return_paths else arr
    pad_h = (tile_size - spatial0 % tile_size) % tile_size
    pad_w = (tile_size - spatial1 % tile_size) % tile_size
    pad_width = [(0, 0)] * arr.ndim
    if ch_axis is not None:
        pad_width[-3] = (0, pad_h)
        pad_width[-2] = (0, pad_w)
    else:
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
    img_pad = np.pad(arr, pad_width, mode="reflect")
    if ch_axis is not None:
        h_pad, w_pad = img_pad.shape[-3], img_pad.shape[-2]
    else:
        h_pad, w_pad = img_pad.shape[-2], img_pad.shape[-1]
    tiles = []
    out_dir = None
    if base_path is not None:
        parent = os.path.dirname(base_path)
        base = os.path.splitext(os.path.basename(base_path))[0]
        if tile_dir_name:
            base = os.path.splitext(os.path.basename(tile_dir_name))[0]
        out_dir = os.path.join(parent, "tifs", base)
        os.makedirs(out_dir, exist_ok=True)
    for y in range(0, h_pad, tile_size):
        for x in range(0, w_pad, tile_size):
            if ch_axis is not None:
                sl = (slice(None),) * (img_pad.ndim - 3) + (slice(y, y + tile_size),
                                                           slice(x, x + tile_size),
                                                           slice(None))
            else:
                sl = (slice(None),) * (img_pad.ndim - 2) + (slice(y, y + tile_size),
                                                           slice(x, x + tile_size))
            tile = img_pad[sl]
            tiles.append(tile)
    tiles = np.stack(tiles, axis=0)
    saved_paths = []
    if out_dir is not None:
        # prompt before overwriting existing tiles
        existing = sorted(glob.glob(os.path.join(out_dir, "*tile_*.tif")))
        if existing:
            if not _prompt_overwrite_tiles(base_path, len(existing), out_dir=out_dir):
                io_logger.info(f"Skipping tile save for {base} (user declined overwrite).")
                # still return first tile for display
                return (tiles[0], saved_paths) if return_paths else tiles[0]
        total_tiles = len(tiles)
        for i, t in enumerate(tiles):
            if channels_to_save and t.ndim >= 3:
                chan_axis = -1 if t.shape[-1] <= 10 else 0
                for c in channels_to_save:
                    try:
                        t_chan = np.take(t, c, axis=chan_axis)
                    except Exception:
                        continue
                    if t_chan.ndim == 2:
                        t_save = np.stack([t_chan] * 3, axis=-1)
                    else:
                        t_save = t_chan
                    tif_path = os.path.join(out_dir, f"{base}_ch{c}_tile_{i:03d}.tif")
                try:
                    tifffile.imwrite(tif_path, t_save)
                    saved_paths.append(tif_path)
                except Exception as e:
                    io_logger.warning(f"Could not save tile {tif_path}: {e}")
            else:
                t_save = np.stack([t] * 3, axis=-1) if t.ndim == 2 else t
                tif_path = os.path.join(out_dir, f"{base}_tile_{i:03d}.tif")
                try:
                    tifffile.imwrite(tif_path, t_save)
                    saved_paths.append(tif_path)
                except Exception as e:
                    io_logger.warning(f"Could not save tile {tif_path}: {e}")
            if progress_cb is not None:
                try:
                    cont = progress_cb(i + 1, total_tiles)
                    if cont is False:
                        io_logger.warning("Tile saving cancelled by user.")
                        break
                except Exception:
                    pass
        io_logger.info(
            f"Tiled large image {os.path.basename(base_path)} into {len(tiles)} tiles of size {tile_size} and saved to {out_dir}."
        )
    # return the first tile to avoid treating the stack as a 3D volume in the GUI
    first_tile = tiles[0]
    # if channel split requested and available, show first selected channel
    if channels_to_save and first_tile.ndim >= 3:
        chan_axis = -1 if first_tile.shape[-1] <= 10 else 0
        try:
            first_tile = np.take(first_tile, channels_to_save[0], axis=chan_axis)
        except Exception:
            pass
    # convert grayscale to rgb for display
    if first_tile.ndim == 2:
        first_tile = np.stack([first_tile] * 3, axis=-1)
    if return_paths:
        return first_tile, saved_paths
    return first_tile


def _prompt_overwrite_tiles(base_path, n_existing, out_dir=None):
    """Ask user if existing tiles for this source should be overwritten."""
    if SUPPRESS_OVERWRITE_TILES_PROMPT:
        return True
    location = out_dir or os.path.join(os.path.dirname(base_path), "tifs")
    msg = (f"Tiles already exist for {os.path.basename(base_path)} "
           f"({n_existing} found in '{location}'). Overwrite them?")
    try:
        from qtpy.QtWidgets import QMessageBox
        reply = QMessageBox.question(None, "Overwrite tiles?", msg,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes
    except Exception:
        # headless or Qt missing: default to not overwriting
        io_logger.info(msg + " (no GUI prompt available; skipping overwrite)")
        return False


def _maybe_save_tif_copy(filename, img):
    """Save a TIFF copy to enable downstream labeling/training workflows."""
    try:
        base = os.path.splitext(filename)[0]
        tif_path = base + ".tif"
        if not os.path.exists(tif_path):
            tifffile.imwrite(tif_path, img)
            io_logger.info(f"Saved TIFF copy to {tif_path}")
    except Exception as e:
        io_logger.warning(f"Could not save TIFF copy for {filename}: {e}")


def _maybe_warn_non_tiff(filename, kind="LIF/ND2", n_series=1, n_channels=1, saved_tiles=True):
    """Show a GUI info box when opening non-TIFF containers (if GUI available)."""
    if SUPPRESS_NON_TIFF_INFO or not saved_tiles:
        return
    msg = f"Loaded {kind} file: {os.path.basename(filename)}\n"
    msg += f"Series/frames detected: {n_series} | Channels (best-effort): {n_channels}\n"
    if saved_tiles:
        msg += (
            "Images larger than 256x256 were tiled into 256x256 patches and a TIFF copy was saved "
            "alongside for labeling."
        )
    else:
        msg += "Images larger than 256x256 can be tiled on demand during labeling/training."
    io_logger.info(msg.replace("\n", " "))
    try:
        from qtpy.QtWidgets import QMessageBox
        QMessageBox.information(None, f"{kind} opened", msg)
    except Exception:
        pass


def _prompt_channel_split(n_channels):
    """Prompt user to select channels to split; returns list of channel indices or None."""
    if n_channels <= 1:
        return None
    if SUPPRESS_NON_TIFF_INFO or not GUI:
        return None
    try:
        from qtpy.QtWidgets import QInputDialog
        items = [f"Channel {i}" for i in range(n_channels)]
        text, ok = QInputDialog.getText(None, "Separate channels?",
                                        f"{n_channels} channels detected. Enter comma-separated channel indices to export (e.g., 0,1):")
        if ok and text.strip():
            sel = []
            for tok in text.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    idx = int(tok)
                    if 0 <= idx < n_channels:
                        sel.append(idx)
            return sel if sel else None
    except Exception:
        pass
    return None


def imread_2D(img_file, return_first_tile=True, save_tiles=True):
    """
    Read in a 2D image file and convert it to a 3-channel image. Attempts to do this for multi-channel and grayscale images.
    If the image has more than 3 channels, only the first 3 channels are kept.
    
    Args:
        img_file (str): The path to the image file.

    Returns:
        img_out (numpy.ndarray): The 3-channel image data as a NumPy array.
    """
    img = imread(img_file, return_first_tile=return_first_tile, save_tiles=save_tiles)
    return transforms.convert_image(img, do_3D=False)


def imread_3D(img_file, return_first_tile=True, save_tiles=True):
    """
    Read in a 3D image file and convert it to have a channel axis last automatically. Attempts to do this for multi-channel and grayscale images.

    If multichannel image, the channel axis is assumed to be the smallest dimension, and the z axis is the next smallest dimension. 
    Use `cellpose.io.imread()` to load the full image without selecting the z and channel axes. 
    
    Args:
        img_file (str): The path to the image file.

    Returns:
        img_out (numpy.ndarray): The image data as a NumPy array.
    """
    img = imread(img_file, return_first_tile=return_first_tile, save_tiles=save_tiles)

    dimension_lengths = list(img.shape)

    # grayscale images:
    if img.ndim == 3:
        channel_axis = None
        # guess at z axis:
        z_axis = np.argmin(dimension_lengths)

    elif img.ndim == 4:
        # guess at channel axis:
        channel_axis = np.argmin(dimension_lengths)

        # guess at z axis: 
        # set channel axis to max so argmin works:
        dimension_lengths[channel_axis] = max(dimension_lengths)
        z_axis = np.argmin(dimension_lengths)

    else: 
        raise ValueError(f'image shape error, 3D image must 3 or 4 dimensional. Number of dimensions: {img.ndim}')
    
    try:
        return transforms.convert_image(img, channel_axis=channel_axis, z_axis=z_axis, do_3D=True)
    except Exception as e:
        io_logger.critical("ERROR: could not read file, %s" % e)
        io_logger.critical("ERROR: Guessed z_axis: %s, channel_axis: %s" % (z_axis, channel_axis))
        return None

def remove_model(filename, delete=False):
    """ remove model from .cellpose custom model list """
    filename = os.path.split(filename)[-1]
    from . import models
    model_strings = models.get_user_models()
    if len(model_strings) > 0:
        with open(models.MODEL_LIST_PATH, "w") as textfile:
            for fname in model_strings:
                textfile.write(fname + "\n")
    else:
        # write empty file
        textfile = open(models.MODEL_LIST_PATH, "w")
        textfile.close()
    print(f"{filename} removed from custom model list")
    if delete:
        os.remove(os.fspath(models.MODEL_DIR.joinpath(fname)))
        print("model deleted")


def add_model(filename):
    """ add model to .cellpose models folder to use with GUI or CLI """
    from . import models
    fname = os.path.split(filename)[-1]
    try:
        shutil.copyfile(filename, os.fspath(models.MODEL_DIR.joinpath(fname)))
    except shutil.SameFileError:
        pass
    print(f"{filename} copied to models folder {os.fspath(models.MODEL_DIR)}")
    if fname not in models.get_user_models():
        # ensure file ends with newline before appending, so new entry is on its own line
        try:
            if os.path.exists(models.MODEL_LIST_PATH) and os.path.getsize(models.MODEL_LIST_PATH) > 0:
                with open(models.MODEL_LIST_PATH, "rb") as tf:
                    tf.seek(-1, os.SEEK_END)
                    last = tf.read(1)
                if last not in (b"\n", b"\r"):
                    with open(models.MODEL_LIST_PATH, "ab") as tf:
                        tf.write(b"\n")
        except Exception:
            pass
        with open(models.MODEL_LIST_PATH, "a", newline="\n") as textfile:
            textfile.write(fname + "\n")


def imsave(filename, arr):
    """
    Saves an image array to a file.

    Args:
        filename (str): The name of the file to save the image to.
        arr (numpy.ndarray): The image array to be saved.

    Returns:
        None
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tif" or ext == ".tiff":
        tifffile.imwrite(filename, data=arr, compression="zlib")
    else:
        if len(arr.shape) > 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, arr)


def get_image_files(folder, mask_filter, imf=None, look_one_level_down=False):
    """
    Finds all images in a folder and its subfolders (if specified) with the given file extensions.

    Args:
        folder (str): The path to the folder to search for images.
        mask_filter (str): The filter for mask files.
        imf (str, optional): The additional filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to search for images in subfolders. Defaults to False.

    Returns:
        list: A list of image file paths.

    Raises:
        ValueError: If no files are found in the specified folder.
        ValueError: If no images are found in the specified folder with the supported file extensions.
        ValueError: If no images are found in the specified folder without the mask or flow file endings.
    """
    # Exclude common derivative files from the image list, including classes maps
    mask_filters = [
        "_cp_output",
        "_flows",
        "_flows_0",
        "_flows_1",
        "_flows_2",
        "_cellprob",
        "_masks",
        "_classes",
        mask_filter,
    ]
    image_names = []
    if imf is None:
        imf = ""

    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*/")))
    folders.append(folder)
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".flex", ".dax", ".nd2", ".lif", ".nrrd"]
    l0 = 0
    al = 0
    for folder in folders:
        all_files = glob.glob(folder + "/*")
        al += len(all_files)
        for ext in exts:
            image_names.extend(glob.glob(folder + f"/*{imf}{ext}"))
            image_names.extend(glob.glob(folder + f"/*{imf}{ext.upper()}"))
        l0 += len(image_names)

    # return error if no files found
    if al == 0:
        raise ValueError("ERROR: no files in --dir folder ")
    elif l0 == 0:
        raise ValueError(
            "ERROR: no images in --dir folder with extensions .png, .jpg, .jpeg, .tif, .tiff, .flex"
        )

    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and
                      imfile[-len(mask_filter):] != mask_filter) or
                     len(imfile) <= len(mask_filter) for mask_filter in mask_filters])
        if len(imf) > 0:
            igood &= imfile[-len(imf):] == imf
        if igood:
            imn.append(im)

    image_names = imn

    # remove duplicates
    image_names = [*set(image_names)]
    image_names = natsorted(image_names)

    if len(image_names) == 0:
        raise ValueError(
            "ERROR: no images in --dir folder without _masks or _flows or _cellprob ending")

    return image_names

def get_label_files(image_names, mask_filter, imf=None):
    """
    Get the label files corresponding to the given image names and mask filter.

    Args:
        image_names (list): List of image names.
        mask_filter (str): Mask filter to be applied.
        imf (str, optional): Image file extension. Defaults to None.

    Returns:
        tuple: A tuple containing the label file names and flow file names (if present).
    """
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    # check for flows
    if os.path.exists(label_names0[0] + "_flows.tif"):
        flow_names = [label_names0[n] + "_flows.tif" for n in range(nimg)]
    else:
        flow_names = [label_names[n] + "_flows.tif" for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        io_logger.info(
            "not all flows are present, running flow generation for all images")
        flow_names = None

    # check for masks
    if mask_filter == "_seg.npy":
        label_names = [label_names[n] + mask_filter for n in range(nimg)]
        return label_names, None

    if os.path.exists(label_names[0] + mask_filter + ".tif"):
        label_names = [label_names[n] + mask_filter + ".tif" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".tiff"):
        label_names = [label_names[n] + mask_filter + ".tiff" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".png"):
        label_names = [label_names[n] + mask_filter + ".png" for n in range(nimg)]
    # TODO, allow _seg.npy
    #elif os.path.exists(label_names[0] + "_seg.npy"):
    #    io_logger.info("labels found as _seg.npy files, converting to tif")
    else:
        if not flow_names:
            raise ValueError("labels not provided with correct --mask_filter")
        else:
            label_names = None
    if not all([os.path.exists(label) for label in label_names]):
        if not flow_names:
            raise ValueError(
                "labels not provided for all images in train and/or test set")
        else:
            label_names = None

    return label_names, flow_names


def load_images_labels(tdir, mask_filter="_masks", image_filter=None,
                       look_one_level_down=False):
    """
    Loads images and corresponding labels from a directory.

    Args:
        tdir (str): The directory path.
        mask_filter (str, optional): The filter for mask files. Defaults to "_masks".
        image_filter (str, optional): The filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to look for files one level down. Defaults to False.

    Returns:
        tuple: A tuple containing a list of images, a list of labels, and a list of image names.
    """
    image_names = get_image_files(tdir, mask_filter, image_filter, look_one_level_down)
    nimg = len(image_names)

    # training data
    label_names, flow_names = get_label_files(image_names, mask_filter,
                                              imf=image_filter)

    images = []
    labels = []
    k = 0
    for n in range(nimg):
        if (os.path.isfile(label_names[n]) or
            (flow_names is not None and os.path.isfile(flow_names[0]))):
            image = imread(image_names[n])
            if label_names is not None:
                label = imread(label_names[n])
            if flow_names is not None:
                flow = imread(flow_names[n])
                if flow.shape[0] < 4:
                    label = np.concatenate((label[np.newaxis, :, :], flow), axis=0)
                else:
                    label = flow
            images.append(image)
            labels.append(label)
            k += 1
    io_logger.info(f"{k} / {nimg} images in {tdir} folder have labels")
    return images, labels, image_names

def load_train_test_data(train_dir, test_dir=None, image_filter=None,
                         mask_filter="_masks", look_one_level_down=False):
    """
    Loads training and testing data for a Cellpose model.

    Args:
        train_dir (str): The directory path containing the training data.
        test_dir (str, optional): The directory path containing the testing data. Defaults to None.
        image_filter (str, optional): The filter for selecting image files. Defaults to None.
        mask_filter (str, optional): The filter for selecting mask files. Defaults to "_masks".
        look_one_level_down (bool, optional): Whether to look for data in subdirectories of train_dir and test_dir. Defaults to False.

    Returns:
        images, labels, image_names, test_images, test_labels, test_image_names

    """
    images, labels, image_names = load_images_labels(train_dir, mask_filter,
                                                     image_filter, look_one_level_down)
    # testing data
    test_images, test_labels, test_image_names = None, None, None
    if test_dir is not None:
        test_images, test_labels, test_image_names = load_images_labels(
            test_dir, mask_filter, image_filter, look_one_level_down)

    return images, labels, image_names, test_images, test_labels, test_image_names


def masks_flows_to_seg(images, masks, flows, file_names, 
                       channels=None,
                       imgs_restore=None, restore_type=None, ratio=1.):
    """Save output of model eval to be loaded in GUI.

    Can be list output (run on multiple images) or single output (run on single image).

    Saved to file_names[k]+"_seg.npy".

    Args:
        images (list): Images input into cellpose.
        masks (list): Masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flows (list): Flows output from Cellpose.eval.
        file_names (list, str): Names of files of images.
        diams (float array): Diameters used to run Cellpose. Defaults to 30. TODO: remove this
        channels (list, int, optional): Channels used to run Cellpose. Defaults to None.

    Returns:
        None
    """

    if channels is None:
        channels = [0, 0]

    if isinstance(masks, list):
        if imgs_restore is None:
            imgs_restore = [None] * len(masks)
        if isinstance(file_names, str):
            file_names = [file_names] * len(masks)
        for k, [image, mask, flow, 
                # diam, 
                file_name, img_restore
               ] in enumerate(zip(images, masks, flows, 
                                #   diams, 
                                  file_names,
                                  imgs_restore)):
            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(image, mask, flow, file_name, 
                            #    diams=diam,
                               channels=channels_img, imgs_restore=img_restore,
                               restore_type=restore_type, ratio=ratio)
        return

    if len(channels) == 1:
        channels = channels[0]

    flowi = []
    if flows[0].ndim == 3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(
            cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,
                                                                            ...])
    else:
        flowi.append(flows[0])

    if flows[0].ndim == 3:
        cellprob = (np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(
            np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis, ...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis, ...]
    else:
        flowi.append(
            (np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(np.uint8))
        flowi.append((flows[1][0] / 10 * 127 + 127).astype(np.uint8))
    if len(flows) > 2:
        if len(flows) > 3:
            flowi.append(flows[3])
        else:
            flowi.append([])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis, ...]), axis=0))
    outlines = masks * utils.masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]

    dat = {
        "outlines":
            outlines.astype(np.uint16) if outlines.max() < 2**16 -
            1 else outlines.astype(np.uint32),
        "masks":
            masks.astype(np.uint16) if outlines.max() < 2**16 -
            1 else masks.astype(np.uint32),
        "chan_choose":
            channels,
        "ismanual":
            np.zeros(masks.max(), bool),
        "filename":
            file_names,
        "flows":
            flowi,
        "diameter":
            np.nan
    }
    if restore_type is not None and imgs_restore is not None:
        dat["restore"] = restore_type
        dat["ratio"] = ratio
        dat["img_restore"] = imgs_restore

    np.save(base + "_seg.npy", dat)

def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True)

        does not work for 3D images

    """
    save_masks(images, masks, flows, file_names, png=True)


def save_rois(masks, file_name, multiprocessing=None):
    """ save masks to .roi files in .zip archive for ImageJ/Fiji

    Args:
        masks (np.ndarray): masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels
        file_name (str): name to save the .zip file to

    Returns:
        None
    """
    outlines = utils.outlines_list(masks, multiprocessing=multiprocessing)
    nonempty_outlines = [outline for outline in outlines if len(outline)!=0]
    if len(outlines)!=len(nonempty_outlines):
        print(f"empty outlines found, saving {len(nonempty_outlines)} ImageJ ROIs to .zip archive.")
    rois = [ImagejRoi.frompoints(outline) for outline in nonempty_outlines]
    file_name = os.path.splitext(file_name)[0] + '_rois.zip'


    # Delete file if it exists; the roifile lib appends to existing zip files.
    # If the user removed a mask it will still be in the zip file
    if os.path.exists(file_name):
        os.remove(file_name)

    roiwrite(file_name, rois)


def save_masks(images, masks, flows, file_names, png=True, tif=False, channels=[0, 0],
               suffix="_cp_masks", save_flows=False, save_outlines=False, dir_above=False,
               in_folders=False, savedir=None, save_txt=False, save_mpl=False):
    """ Save masks + nicely plotted segmentation image to png and/or tiff.

    Can save masks, flows to different directories, if in_folders is True.

    If png, masks[k] for images[k] are saved to file_names[k]+"_cp_masks.png".

    If tif, masks[k] for images[k] are saved to file_names[k]+"_cp_masks.tif".

    If png and matplotlib installed, full segmentation figure is saved to file_names[k]+"_cp.png".

    Only tif option works for 3D data, and only tif option works for empty masks.

    Args:
        images (list): Images input into cellpose.
        masks (list): Masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flows (list): Flows output from Cellpose.eval.
        file_names (list, str): Names of files of images.
        png (bool, optional): Save masks to PNG. Defaults to True.
        tif (bool, optional): Save masks to TIF. Defaults to False.
        channels (list, int, optional): Channels used to run Cellpose. Defaults to [0,0].
        suffix (str, optional): Add name to saved masks. Defaults to "_cp_masks".
        save_flows (bool, optional): Save flows output from Cellpose.eval. Defaults to False.
        save_outlines (bool, optional): Save outlines of masks. Defaults to False.
        dir_above (bool, optional): Save masks/flows in directory above. Defaults to False.
        in_folders (bool, optional): Save masks/flows in separate folders. Defaults to False.
        savedir (str, optional): Absolute path where images will be saved. If None, saves to image directory. Defaults to None.
        save_txt (bool, optional): Save masks as list of outlines for ImageJ. Defaults to False.
        save_mpl (bool, optional): If True, saves a matplotlib figure of the original image/segmentation/flows. Does not work for 3D.
                This takes a long time for large images. Defaults to False.

    Returns:
        None
    """

    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif, suffix=suffix,
                       dir_above=dir_above, save_flows=save_flows,
                       save_outlines=save_outlines, savedir=savedir, save_txt=save_txt,
                       in_folders=in_folders, save_mpl=save_mpl)
        return

    if masks.ndim > 2 and not tif:
        raise ValueError("cannot save 3D outputs as PNG, use tif option instead")

    if masks.max() == 0:
        io_logger.warning("no masks found, will not save PNG or outlines")
        if not tif:
            return
        else:
            png = False
            save_outlines = False
            save_flows = False
            save_txt = False

    if savedir is None:
        if dir_above:
            savedir = Path(file_names).parent.parent.absolute(
            )  #go up a level to save in its own folder
        else:
            savedir = Path(file_names).parent.absolute()

    check_dir(savedir)

    basename = os.path.splitext(os.path.basename(file_names))[0]
    if in_folders:
        maskdir = os.path.join(savedir, "masks")
        outlinedir = os.path.join(savedir, "outlines")
        txtdir = os.path.join(savedir, "txt_outlines")
        flowdir = os.path.join(savedir, "flows")
    else:
        maskdir = savedir
        outlinedir = savedir
        txtdir = savedir
        flowdir = savedir

    check_dir(maskdir)

    exts = []
    if masks.ndim > 2:
        png = False
        tif = True
    if png:
        if masks.max() < 2**16:
            masks = masks.astype(np.uint16)
            exts.append(".png")
        else:
            png = False
            tif = True
            io_logger.warning(
                "found more than 65535 masks in each image, cannot save PNG, saving as TIF"
            )
    if tif:
        exts.append(".tif")

    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:
            imsave(os.path.join(maskdir, basename + suffix + ext), masks)

    if save_mpl and png and MATPLOTLIB and not min(images.shape) > 3:
        # Make and save original/segmentation/flows image

        img = images.copy()
        if img.ndim < 3:
            img = img[:, :, np.newaxis]
        elif img.shape[0] < 8:
            np.transpose(img, (1, 2, 0))

        fig = plt.figure(figsize=(12, 3))
        plot.show_segmentation(fig, img, masks, flows[0])
        fig.savefig(os.path.join(savedir, basename + "_cp_output" + suffix + ".png"),
                    dpi=300)
        plt.close(fig)

    # ImageJ txt outline files
    if masks.ndim < 3 and save_txt:
        check_dir(txtdir)
        outlines = utils.outlines_list(masks)
        outlines_to_text(os.path.join(txtdir, basename), outlines)

    # RGB outline images
    if masks.ndim < 3 and save_outlines:
        check_dir(outlinedir)
        outlines = utils.masks_to_outlines(masks)
        outX, outY = np.nonzero(outlines)
        img0 = transforms.normalize99(images)
        if img0.shape[0] < 4:
            img0 = np.transpose(img0, (1, 2, 0))
        if img0.shape[-1] < 3 or img0.ndim < 3:
            img0 = plot.image_to_rgb(img0, channels=channels)
        else:
            if img0.max() <= 50.0:
                img0 = np.uint8(np.clip(img0 * 255, 0, 1))
        imgout = img0.copy()
        imgout[outX, outY] = np.array([255, 0, 0])  #pure red
        imsave(os.path.join(outlinedir, basename + "_outlines" + suffix + ".png"),
               imgout)

    # save RGB flow picture
    if masks.ndim < 3 and save_flows:
        check_dir(flowdir)
        imsave(os.path.join(flowdir, basename + "_flows" + suffix + ".tif"),
               (flows[0] * (2**16 - 1)).astype(np.uint16))
        #save full flow data
        imsave(os.path.join(flowdir, basename + '_dP' + suffix + '.tif'), flows[1])
