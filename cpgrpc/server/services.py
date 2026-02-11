import os
import shutil
import time
import grpc
import queue
import threading
import re
import logging
import numpy as np
from concurrent import futures
from pathlib import Path

_logger = logging.getLogger(__name__)

try:
    from grpc_tools import protoc  # noqa: F401
    HAS_PROTOC = True
except Exception:
    HAS_PROTOC = False

# Attempt to import generated stubs; if missing, try to generate at import-time
def _ensure_generated():
    try:
        from . import cellpose_remote_pb2  # noqa: F401
        from . import cellpose_remote_pb2_grpc  # noqa: F401
        return True
    except Exception:
        pass
    if not HAS_PROTOC:
        return False
    proto_path = Path(__file__).parent.parent / "protos" / "cellpose_remote.proto"
    out_dir = Path(__file__).parent
    cmd = [
        "-I", str(proto_path.parent),
        "--python_out", str(out_dir),
        "--grpc_python_out", str(out_dir),
        str(proto_path),
    ]
    from grpc_tools import protoc as _protoc
    ok = _protoc.main(["protoc", *cmd]) == 0
    return ok


GENERATED = _ensure_generated()

from . import cellpose_remote_pb2 as pb2  # type: ignore
from . import cellpose_remote_pb2_grpc as pb2_grpc  # type: ignore
from cellpose import remote_config
from cellpose.semantic_class_weights import (
    compute_class_weights_from_class_maps,
    extract_class_maps_from_labels,
    infer_semantic_nclasses_from_net,
)
from cellpose.semantic_label_utils import (
    build_classes_map_from_masks,
    sanitize_class_map,
)
from cellpose.training_mode_utils import configure_trainable_params


def _get_quota_bytes() -> int:
    try:
        gb = float(os.environ.get("CELLPOSE_USER_QUOTA_GB", "30"))
    except Exception:
        gb = 30.0
    return int(gb * 1024 * 1024 * 1024)


def _dir_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            total += p.stat().st_size
        except Exception:
            continue
    return total


def _ensure_quota(user_root: Path, bytes_needed: int) -> None:
    quota = _get_quota_bytes()
    used = _dir_size_bytes(user_root)
    if used + bytes_needed > quota:
        raise RuntimeError(
            f"Storage quota exceeded: used={used} bytes, requested={bytes_needed} bytes, quota={quota} bytes"
        )


def _sanitize_user_id(value: str) -> str:
    if not value:
        return "anonymous"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return safe or "anonymous"


def _metadata_dict(context):
    return dict(context.invocation_metadata() or [])


def _get_user_id(context):
    meta = _metadata_dict(context)
    user = meta.get("x-user") or meta.get("x-user-id") or meta.get("user")
    if isinstance(user, bytes):
        try:
            user = user.decode("utf-8", errors="ignore")
        except Exception:
            user = ""
    return _sanitize_user_id(user or "")


class HealthServicer(pb2_grpc.HealthServicer):
    def Check(self, request, context):  # noqa: N802 (gRPC naming)
        return pb2.HealthCheckResponse(status="SERVING")


class FileService(pb2_grpc.FileServiceServicer):
    _CLEANUP_PROJECT = "_admin"
    _CLEANUP_RELPATH = "__clear_user_jobs__"
    _MODEL_PROJECT = "_admin_models"

    def __init__(self, root: str, model_root: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        cfg = remote_config.load_remote_config()
        model_root = model_root or os.environ.get("CELLPOSE_LOCAL_MODELS_PATH") or cfg.get("model_root")
        if model_root:
            self.model_root = Path(model_root)
        else:
            self.model_root = Path.home().joinpath(".cellpose", "models")
        self.model_root.mkdir(parents=True, exist_ok=True)

    def _win_longpath(self, path: str) -> str:
        if os.name != "nt":
            return path
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if path.startswith("\\\\?\\") or path.startswith("\\\\.\\"):
            return path
        if path.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path.lstrip("\\")
        return "\\\\?\\" + path

    def _user_root(self, context):
        user_id = _get_user_id(context)
        user_root = self.root / "users" / user_id
        user_root.mkdir(parents=True, exist_ok=True)
        return user_root

    def _clear_user_jobs(self, context) -> int:
        user_root = self._user_root(context)
        train_jobs_root = user_root / "train_jobs"
        if not train_jobs_root.exists():
            return 0
        removed = 0
        try:
            for item in train_jobs_root.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    try:
                        item.unlink()
                    except Exception:
                        pass
                removed += 1
        except Exception as e:
            _logger.error(f"failed clearing train_jobs ({e})")
        return removed

    def _project_root(self, context, project_id: str):
        proj = Path(project_id)
        if proj.is_absolute() or ".." in proj.parts:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Invalid project_id")
        proj_dir = self._user_root(context) / project_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        return proj_dir

    def _safe_relpath(self, context, relpath: str):
        rel = Path(relpath)
        if rel.is_absolute() or ".." in rel.parts:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Invalid relpath")
        return relpath

    def Upload(self, request_iterator, context):  # stream -> unary
        total = 0
        project = None
        relpath = None
        fh = None
        user_root = None
        try:
            for chunk in request_iterator:
                project = project or chunk.project_id
                relpath = relpath or chunk.relpath
                if not project or not relpath:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing project_id/relpath")
                if (
                    project == self._CLEANUP_PROJECT
                    and relpath == self._CLEANUP_RELPATH
                ):
                    for _ in request_iterator:
                        pass
                    removed = self._clear_user_jobs(context)
                    _logger.info(f"cleared {removed} train job entries for user")
                    return pb2.UploadReply(uri="clear://train_jobs", bytes_received=0)
                if project == self._MODEL_PROJECT:
                    filename = Path(relpath).name
                    if not filename:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing model filename")
                    target = self.model_root / filename
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if fh is None:
                            fh = open(os.fspath(target), "w+b")
                            _logger.info(f"model upload start name={filename}")
                    except Exception as e:
                        _logger.error(f"model upload open failed target={target} ({e})")
                        raise
                    if chunk.data:
                        fh.write(chunk.data)
                        total += len(chunk.data)
                    continue
                user_root = self._user_root(context)
                proj_dir = self._project_root(context, project)
                relpath = self._safe_relpath(context, relpath)
                target = proj_dir / relpath
                try:
                    if os.name == "nt":
                        os.makedirs(self._win_longpath(os.fspath(target.parent.resolve())), exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    _logger.error(f"failed to create parent for {target} ({e})")
                    raise
                if fh is None:
                    mode = "r+b" if target.exists() else "w+b"
                    try:
                        path_str = os.fspath(target)
                        if os.name == "nt":
                            path_str = self._win_longpath(path_str)
                        fh = open(path_str, mode)
                    except Exception as e:
                        _logger.error(
                            "upload open failed "
                            f"target={target} parent_exists={target.parent.exists()} "
                            f"cwd={os.getcwd()} len={len(os.fspath(target))} "
                            f"project={project} relpath={relpath} ({e})"
                        )
                        raise
                    _logger.info(f"upload start project={project} relpath={relpath}")
                if chunk.offset:
                    fh.seek(chunk.offset)
                if chunk.data:
                    try:
                        _ensure_quota(user_root, total + len(chunk.data))
                    except RuntimeError as exc:
                        context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
                    fh.write(chunk.data)
                    total += len(chunk.data)
            if fh is not None:
                fh.flush()
            if project and relpath:
                _logger.info(f"upload complete project={project} relpath={relpath} bytes={total}")
            if project == self._MODEL_PROJECT:
                _logger.info(f"model upload complete name={Path(relpath).name} bytes={total}")
                return pb2.UploadReply(uri=f"model://{Path(relpath).name}", bytes_received=total)
            uri = f"file://{(proj_dir / relpath).as_posix()}"
            return pb2.UploadReply(uri=uri, bytes_received=total)
        finally:
            if fh is not None:
                fh.close()

    def Download(self, request, context): # unary -> stream
        try:
            if request.uri.startswith("list://"):
                from urllib.parse import urlparse, parse_qs
                import json
                parsed = urlparse(request.uri)
                project_id = "/".join(
                    p for p in [parsed.netloc, parsed.path.lstrip("/")] if p
                )
                prefix = ""
                query = parse_qs(parsed.query or "")
                if "prefix" in query and query["prefix"]:
                    prefix = query["prefix"][0]
                if not project_id:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing project_id")
                proj_dir = self._project_root(context, project_id).resolve()
                user_root = self._user_root(context).resolve()
                if user_root not in proj_dir.parents and proj_dir != user_root:
                    context.abort(grpc.StatusCode.PERMISSION_DENIED, "Access denied")
                if not proj_dir.exists():
                    payload = {"files": []}
                else:
                    files = []
                    for f in proj_dir.rglob("*"):
                        if not f.is_file():
                            continue
                        relpath = os.fspath(f.relative_to(proj_dir))
                        if prefix and not relpath.replace("\\", "/").startswith(prefix):
                            continue
                        try:
                            stat = f.stat()
                            files.append(
                                {
                                    "relpath": relpath,
                                    "mtime": int(stat.st_mtime),
                                    "size": int(stat.st_size),
                                    "uri": f"file://{f.as_posix()}",
                                }
                            )
                        except Exception:
                            continue
                    payload = {"files": files}
                data = json.dumps(payload).encode("utf-8")
                yield pb2.FileChunk(data=data)
                return

            if not request.uri.startswith("file://"):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid URI scheme")
            
            file_path = Path(request.uri.replace("file://", ""))
            user_root = self._user_root(context).resolve()
            if user_root not in file_path.resolve().parents and file_path.resolve() != user_root:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "Access denied")

            path_str = os.fspath(file_path)
            if os.name == "nt":
                path_str = self._win_longpath(path_str)
            with open(path_str, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    yield pb2.FileChunk(data=chunk)
        except FileNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, "File not found")
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"An error occurred: {e}")


class InferenceJob:
    def __init__(self, request, user_root, user_id):
        self.request = request
        self.user_root = user_root
        self.user_id = user_id
        self.updates = queue.Queue()
        self.error = None


class InferenceManager:
    def __init__(self, service):
        self.service = service
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, request, user_root, user_id):
        job = InferenceJob(request, user_root, user_id)
        self.queue.put(job)
        return job

    def _worker(self):
        while True:
            job = self.queue.get()
            try:
                for update in self.service._run_inference_job(job.request, job.user_root, job.user_id):
                    job.updates.put(update)
            except Exception as exc:
                job.error = exc
                job.updates.put(pb2.JobUpdate(progress=0, stage="error", message=str(exc)))
            finally:
                job.updates.put(None)


class InferenceService(pb2_grpc.InferenceServiceServicer):
    def __init__(self, storage_root: str):
        self.root = Path(storage_root)
        cfg = remote_config.load_remote_config()
        model_root = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH") or cfg.get("model_root")
        if model_root:
            self.model_root = Path(model_root)
        else:
            self.model_root = Path.home().joinpath(".cellpose", "models")
        self.model_root.mkdir(parents=True, exist_ok=True)
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(self.model_root)
        replay_root = os.environ.get("CELLPOSE_REPLAY_DATASET") or cfg.get("replay_root")
        if replay_root:
            self.replay_root = Path(replay_root)
        else:
            self.replay_root = Path.home().joinpath(".cellpose", "replay")
        replay_sample = os.environ.get("CELLPOSE_REPLAY_SAMPLE_SIZE") or cfg.get("replay_sample_size", "100")
        self.replay_sample_size = int(replay_sample)

        try:
            from cellpose import models as _models
            _models.MODEL_DIR = Path(os.environ["CELLPOSE_LOCAL_MODELS_PATH"])
            _models.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            _models.MODEL_LIST_PATH = os.fspath(_models.MODEL_DIR.joinpath("gui_models.txt"))
        except Exception:
            pass
        try:
            from cellpose import models as _models
            model_dir = Path(_models.MODEL_DIR)
            model_files = []
            if model_dir.exists():
                for f in os.listdir(model_dir):
                    if f.endswith(".pth") or f.endswith(".pt") or f == "cpsam" or f.startswith("cpsamGUV") or f.startswith("cpsam_inst"):
                        model_files.append(f)
            self.model_files = sorted(set(model_files))
            _logger.info(f"model directory: {model_dir}")
            _logger.info(f"available models: {self.model_files}")
        except Exception as exc:
            _logger.warning(f"could not list models on startup: {exc}")
        self.replay_items = self._scan_replay_items()
        self.training_manager = TrainingManager(self)
        self.inference_manager = InferenceManager(self)

    def _user_root(self, context):
        user_id = _get_user_id(context)
        user_root = self.root / "users" / user_id
        user_root.mkdir(parents=True, exist_ok=True)
        return user_root, user_id

    def _ensure_user_path(self, context, path_value):
        user_root, _user_id = self._user_root(context)
        p = Path(path_value).resolve()
        if user_root.resolve() not in p.parents and p != user_root.resolve():
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Access denied")
        return p

    def _resolve_item_path(self, job_root, path_value):
        p = Path(path_value)
        if p.is_absolute():
            return p
        try:
            resolved = p.resolve()
            if resolved.exists():
                return resolved
        except Exception:
            pass
        return job_root / p

    def _scan_replay_items(self):
        try:
            from cellpose.io import get_image_files
        except Exception:
            return []
        if not self.replay_root.exists():
            _logger.info(f"replay dataset not found at {self.replay_root}")
            return []
        images = get_image_files(os.fspath(self.replay_root), "_masks")
        items = []
        for img_path in images:
            base, _ext = os.path.splitext(img_path)
            seg = base + "_seg.npy"
            if os.path.isfile(seg):
                items.append({"image": img_path, "seg_npy": seg})
                continue
            masks = base + "_masks.tif"
            flows = base + "_flows.tif"
            classes = base + "_classes.tif"
            if not os.path.isfile(masks) and os.path.isfile(base + "_masks.tiff"):
                masks = base + "_masks.tiff"
            if not os.path.isfile(flows) and os.path.isfile(base + "_flows.tiff"):
                flows = base + "_flows.tiff"
            if not os.path.isfile(classes) and os.path.isfile(base + "_classes.tiff"):
                classes = base + "_classes.tiff"
            if os.path.isfile(masks) and os.path.isfile(flows):
                item = {"image": img_path, "masks": masks, "flows": flows}
                if os.path.isfile(classes):
                    item["classes"] = classes
                items.append(item)
                continue
            if os.path.isfile(masks):
                item = {"image": img_path, "masks": masks}
                if os.path.isfile(classes):
                    item["classes"] = classes
                items.append(item)
        _logger.info(f"replay dataset items: {len(items)} from {self.replay_root}")
        return items

    def _scan_project_items(self, project_root):
        try:
            from cellpose.io import get_image_files
        except Exception:
            return []
        images = get_image_files(os.fspath(project_root), "_masks")
        items = []
        for img_path in images:
            base, _ext = os.path.splitext(img_path)
            seg = base + "_seg.npy"
            if os.path.isfile(seg):
                items.append({"image": img_path, "seg_npy": seg})
                continue
            masks = base + "_masks.tif"
            flows = base + "_flows.tif"
            classes = base + "_classes.tif"
            if not os.path.isfile(masks) and os.path.isfile(base + "_masks.tiff"):
                masks = base + "_masks.tiff"
            if not os.path.isfile(flows) and os.path.isfile(base + "_flows.tiff"):
                flows = base + "_flows.tiff"
            if not os.path.isfile(classes) and os.path.isfile(base + "_classes.tiff"):
                classes = base + "_classes.tiff"
            if os.path.isfile(masks) and os.path.isfile(flows):
                item = {"image": img_path, "masks": masks, "flows": flows}
                if os.path.isfile(classes):
                    item["classes"] = classes
                items.append(item)
                continue
            if os.path.isfile(masks):
                item = {"image": img_path, "masks": masks}
                if os.path.isfile(classes):
                    item["classes"] = classes
                items.append(item)
        return items

    def ListModels(self, request, context):
        from cellpose import models
        import os
        model_ids = []
        model_ids.extend(models.MODEL_NAMES)
        model_dir = models.MODEL_DIR
        if model_dir.exists():
            for f in os.listdir(model_dir):
                if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".bin"):
                    model_ids.append(f)
                elif os.path.isfile(os.fspath(model_dir / f)) and f != "gui_models.txt":
                    model_ids.append(f)
        user_root, user_id = self._user_root(context)
        user_model_dir = user_root / "models"
        if user_model_dir.exists():
            for f in os.listdir(user_model_dir):
                if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".bin"):
                    model_ids.append(f)
                elif os.path.isfile(os.fspath(user_model_dir / f)) and f != "gui_models.txt":
                    model_ids.append(f)
        for model_name in getattr(self, "model_files", []):
            if model_name not in model_ids:
                model_ids.append(model_name)
        _logger.info(f"ListModels user={user_id} from {model_dir} -> {model_ids}")
        return pb2.ListModelsResponse(model_ids=model_ids)

    def _resolve_model_id(self, model_id: str, user_root: Path | None = None) -> str:
        if not model_id:
            return model_id
        try:
            p = Path(model_id)
            if p.is_absolute() and p.exists():
                return os.fspath(p)
        except Exception:
            pass
        if user_root is not None:
            user_model_dir = user_root / "models"
            base = user_model_dir / model_id
            if base.exists():
                return os.fspath(base)
            for ext in (".pth", ".pt", ".bin"):
                candidate = user_model_dir / f"{model_id}{ext}"
                if candidate.exists():
                    return os.fspath(candidate)
        base = self.model_root / model_id
        if base.exists():
            return os.fspath(base)
        for ext in (".pth", ".pt", ".bin"):
            candidate = self.model_root / f"{model_id}{ext}"
            if candidate.exists():
                return os.fspath(candidate)
        return model_id

    def Run(self, request, context):  # unary -> stream
        user_root, user_id = self._user_root(context)
        if request.model_id == "__train__":
            job = self.training_manager.submit(request, user_root, user_id)
            while True:
                update = job.updates.get()
                if update is None:
                    break
                yield update
            if job.error:
                context.abort(grpc.StatusCode.INTERNAL, f"Training failed: {job.error}")
            return

        job = self.inference_manager.submit(request, user_root, user_id)
        while True:
            update = job.updates.get()
            if update is None:
                break
            yield update
        if job.error:
            context.abort(grpc.StatusCode.INTERNAL, f"Inference failed: {job.error}")
        return

    def _run_inference_job(self, request, user_root, user_id):
        from cellpose import models, io, transforms, utils
        import json
        import numpy as np
        try:
            io.SUPPRESS_NON_TIFF_INFO = True
            io.SUPPRESS_OVERWRITE_TILES_PROMPT = True
            io.GUI = False
        except Exception:
            pass

        stages = ["queue", "preprocess", "infer", "postprocess", "done"]

        def _prepare_batch(images):
            prepped = []
            shape_hw = None
            for img in images:
                if img is None:
                    return None
                arr = img
                if arr.ndim == 2:
                    arr = arr[:, :, None]
                elif arr.ndim == 3:
                    if arr.shape[-1] <= 4:
                        pass
                    elif arr.shape[0] <= 4:
                        arr = arr.transpose(1, 2, 0)
                    else:
                        return None
                else:
                    return None
                if arr.shape[-1] != 3:
                    arr3 = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=arr.dtype)
                    copy_ch = min(3, arr.shape[-1])
                    arr3[..., :copy_ch] = arr[..., :copy_ch]
                    arr = arr3
                if shape_hw is None:
                    shape_hw = arr.shape[:2]
                elif arr.shape[:2] != shape_hw:
                    return None
                prepped.append(arr)
            return np.stack(prepped, axis=0)

        def _split_component(comp, idx, nimg):
            arr = comp
            if isinstance(arr, list):
                if idx < len(arr):
                    return arr[idx]
                return arr[0]
            if isinstance(arr, np.ndarray) and nimg > 1:
                if arr.ndim >= 4 and arr.shape[0] == nimg:
                    return arr[idx]
                if arr.ndim >= 4 and arr.shape[1] == nimg and arr.shape[0] in (2, 3):
                    return arr[:, idx]
                if arr.ndim == 3 and arr.shape[0] == nimg:
                    return arr[idx]
            return arr

        def _split_outputs(masks, flows, styles, nimg):
            if isinstance(masks, list):
                masks_list = list(masks)
            elif isinstance(masks, np.ndarray) and nimg > 1 and masks.ndim >= 3 and masks.shape[0] == nimg:
                masks_list = [masks[i] for i in range(nimg)]
            else:
                masks_list = [masks] * nimg

            if isinstance(styles, list):
                styles_list = list(styles)
            elif isinstance(styles, np.ndarray) and nimg > 1 and styles.shape[0] == nimg:
                styles_list = [styles[i] for i in range(nimg)]
            else:
                styles_list = [styles] * nimg

            flows_list = []
            if isinstance(flows, (list, tuple)) and len(flows) >= 3:
                flow0, flow1, flow2 = flows[:3]
                for i in range(nimg):
                    flows_list.append([
                        _split_component(flow0, i, nimg),
                        _split_component(flow1, i, nimg),
                        _split_component(flow2, i, nimg),
                    ])
            else:
                flows_list = [flows] * nimg
            return masks_list, flows_list, styles_list
        
        yield pb2.JobUpdate(progress=0, stage=stages[0], message="Inference job queued")

        # Stage: Preprocess
        yield pb2.JobUpdate(progress=20, stage=stages[1], message="Loading images")
        
        image_paths = []
        for uri in request.uris:
            if not uri.startswith("file://"):
                raise ValueError("Invalid URI scheme")
            p = Path(uri.replace("file://", "")).resolve()
            if user_root.resolve() not in p.parents and p != user_root.resolve():
                raise PermissionError("Access denied")
            image_paths.append(os.fspath(p))
        images = [io.imread(p) for p in image_paths]
        # Get parameters from request, with defaults
        diameter = request.diameter if request.HasField('diameter') else 30.0
        # Keep defaults aligned with legacy GUI eval settings for consistency.
        cellprob_threshold = request.cellprob_threshold if request.HasField('cellprob_threshold') else -0.5
        flow_threshold = request.flow_threshold if request.HasField('flow_threshold') else 1.0
        do_3D = request.do_3D if request.HasField('do_3D') else False
        niter = request.niter if request.HasField('niter') else 0
        stitch_threshold = request.stitch_threshold if request.HasField('stitch_threshold') else 0.0
        anisotropy = request.anisotropy if request.HasField('anisotropy') else 1.0
        flow3D_smooth = request.flow3D_smooth if request.HasField('flow3D_smooth') else 0.0
        min_size = request.min_size if request.HasField('min_size') else 15
        max_size_fraction = request.max_size_fraction if request.HasField('max_size_fraction') else 1.0
        normalize_params_str = request.normalize_params if request.HasField('normalize_params') else '{}'
        normalize_params = json.loads(normalize_params_str)
        z_axis = request.z_axis if request.HasField('z_axis') else None
        channel_axis = request.channel_axis if request.HasField('channel_axis') else None
        nimg = len(images)
        batch_channel_axis = channel_axis
        if nimg == 1:
            data = images[0]
        else:
            data = None
            if not do_3D:
                data = _prepare_batch(images)
                if data is not None:
                    batch_channel_axis = -1
                    yield pb2.JobUpdate(progress=30, stage=stages[1], message=f"Stacked {nimg} images for batched inference")
            if data is None:
                data = images

        # Stage: Infer
        yield pb2.JobUpdate(progress=40, stage=stages[2], message=f"Loading model: {request.model_id}")
        model_id = request.model_id
        if any(sep in model_id for sep in ("/", "\\", ":")):
            model_path = Path(model_id).resolve()
            allowed = [self.model_root.resolve(), (user_root / "models").resolve()]
            if not any(model_path == root or root in model_path.parents for root in allowed):
                raise PermissionError("Model path not allowed")
            model_id = os.fspath(model_path)
        resolved_model_id = self._resolve_model_id(model_id, user_root=user_root)
        if resolved_model_id != model_id:
            _logger.info(f"resolved model id {model_id} -> {resolved_model_id}")
        model = models.CellposeModel(gpu=True, pretrained_model=resolved_model_id)
        
        yield pb2.JobUpdate(progress=60, stage=stages[2], message="Running model evaluation")

        bsize = int(normalize_params.get("bsize", 256)) if isinstance(normalize_params, dict) else 256
        eval_out = model.eval(
            data, 
            batch_size=min(32, nimg),
            bsize=bsize,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold, 
            do_3D=do_3D, 
            niter=niter,
            normalize=normalize_params, 
            stitch_threshold=stitch_threshold,
            anisotropy=anisotropy, 
            flow3D_smooth=flow3D_smooth,
            min_size=min_size, 
            channel_axis=batch_channel_axis, 
            max_size_fraction=max_size_fraction,
            z_axis=z_axis
        )
        if isinstance(eval_out, (list, tuple)) and len(eval_out) >= 3:
            masks, flows, styles = eval_out[:3]
        else:
            raise ValueError("Unexpected model output; expected masks, flows, styles")

        masks_list, flows_list, styles_list = _split_outputs(masks, flows, styles, nimg)

        # Stage: Postprocess
        yield pb2.JobUpdate(progress=80, stage=stages[3], message="Processing results")
        
        for idx in range(nimg):
            masks_i = masks_list[idx]
            flows_i = flows_list[idx]
            styles_i = styles_list[idx]

            colors = None
            mask_classes = None
            pred_classes_map = None
            if styles_i is not None:
                arr = np.squeeze(styles_i)
                if arr.ndim >= 3 and arr.shape[-1] > 1:
                    pred_classes_map = np.argmax(arr, axis=-1).astype(np.int32)
                elif arr.ndim == 2:
                    pred_classes_map = arr.astype(np.int32)

            def _resize_class_map(class_map, target_shape):
                if class_map is None or target_shape is None:
                    return class_map
                if class_map.shape == target_shape:
                    return class_map
                try:
                    import cv2
                    return cv2.resize(
                        class_map.astype(np.int32),
                        (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                except Exception:
                    try:
                        from skimage.transform import resize as sk_resize
                        return sk_resize(
                            class_map,
                            target_shape,
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False,
                        ).astype(np.int32)
                    except Exception:
                        return None
            
            if pred_classes_map is not None:
                m2d = masks_i[0] if masks_i.ndim == 3 else masks_i
                pred_classes_map = _resize_class_map(pred_classes_map, m2d.shape)
                if pred_classes_map is None:
                    colors = None
                    mask_classes = None
                    pred_classes_map = None
                else:
                    nmask = int(m2d.max())
                    mask_classes = np.zeros(nmask + 1, dtype=np.int16)
                    cols = np.zeros((nmask, 3), dtype=np.uint8)
                    colormap = (np.random.rand(nmask, 3) * 255).astype(np.uint8)
                    for j in range(1, nmask + 1):
                        clsj = pred_classes_map[m2d == j]
                        if clsj.size > 0:
                            cls_vals = clsj[clsj > 0]
                            if cls_vals.size > 0:
                                counts = np.bincount(cls_vals.astype(np.int64))
                                cid = int(np.argmax(counts))
                                mask_classes[j] = cid
                                cols[j - 1] = colormap[j - 1]
                    colors = cols

            outlines = utils.masks_to_outlines(masks_i) * masks_i

            img_to_save = images[idx]
            if (not do_3D) and img_to_save.ndim == 4 and img_to_save.shape[0] == 1:
                img_to_save = img_to_save[0]

            dat = {
                "outlines": outlines.astype(np.uint16) if outlines.max() < 2**16 - 1 else outlines.astype(np.uint32),
                "masks": masks_i.astype(np.uint16) if masks_i.max() < 2**16 - 1 else masks_i.astype(np.uint32),
                "chan_choose": [channel_axis, 0] if channel_axis is not None else [0, 0],
                "ismanual": np.zeros(masks_i.max(), bool),
                "filename": image_paths[idx],
                "flows": flows_i,
                "diameter": diameter,
                "img": img_to_save,
                "normalize_params": normalize_params,
            }
            if colors is not None:
                dat["colors"] = colors
            if mask_classes is not None:
                dat["classes"] = mask_classes
            if pred_classes_map is not None:
                dat["classes_map"] = pred_classes_map.astype(np.int16)

            original_path = Path(image_paths[idx])
            output_path = original_path.with_name(f"{original_path.stem}_seg.npy")
            try:
                est = int(masks_i.nbytes)
                if isinstance(flows_i, (list, tuple)):
                    for comp in flows_i:
                        if isinstance(comp, np.ndarray):
                            est += int(comp.nbytes)
                elif hasattr(flows_i, "nbytes"):
                    est += int(flows_i.nbytes)
                if pred_classes_map is not None:
                    est += int(pred_classes_map.nbytes)
                _ensure_quota(user_root, est)
            except RuntimeError as exc:
                raise RuntimeError(str(exc))
            np.save(output_path, dat)
            
            result_uri = f"file://{output_path.as_posix()}"
            progress = 80 + int(15 * ((idx + 1) / max(1, nimg)))
            yield pb2.JobUpdate(progress=progress, stage=stages[3], message=f"Result saved to {result_uri}", result_uri=result_uri)

        # Stage: Done
        yield pb2.JobUpdate(progress=100, stage=stages[4], message="Inference complete")

    def _filter_masks_by_class(self, masks, classes_map, class_id):
        import numpy as np
        if masks is None or classes_map is None:
            return None
        keep_ids = [mid for mid in range(1, int(masks.max()) + 1)
                    if mid < len(classes_map) and int(classes_map[mid]) == int(class_id)]
        if len(keep_ids) == 0:
            return None
        keep = np.isin(masks, keep_ids)
        return np.where(keep, masks, 0).astype(masks.dtype)

    def _load_classes_map(self, dat, masks):
        cm = None
        if "classes_map" in dat:
            cm = dat["classes_map"]
            cm = cm.squeeze() if hasattr(cm, "squeeze") else cm
            cm = self._sanitize_classes_map(cm, dat, masks)
            if cm is not None:
                return cm
        if "classes" in dat:
            classes = dat.get("classes", None)
            if classes is not None and masks is not None:
                cm = build_classes_map_from_masks(masks, classes)
                cm = self._sanitize_classes_map(cm, dat, masks)
                if cm is not None:
                    return cm
        return None

    def _sanitize_classes_map(self, classes_map, dat, masks):
        return sanitize_class_map(
            classes_map,
            masks=masks,
            classes=dat.get("classes") if isinstance(dat, dict) else None,
            class_names=dat.get("class_names") if isinstance(dat, dict) else None,
        )

    def _labels_from_masks(self, masks, classes_map=None, device=None):
        import numpy as np
        from cellpose import dynamics
        flows = dynamics.labels_to_flows([masks], device=device)[0]
        if classes_map is not None:
            return np.concatenate((flows[:1], classes_map[np.newaxis, ...], flows[1:]), axis=0)
        return flows

    def _load_label_item(self, item, job_root, class_id=None, flow_device=None):
        from cellpose import io
        import numpy as np
        image_path = self._resolve_item_path(job_root, item["image"])
        img = io.imread(str(image_path))
        if "seg_npy" in item:
            dat = np.load(self._resolve_item_path(job_root, item["seg_npy"]), allow_pickle=True).item()
            masks = dat.get("masks", None)
            if masks is None:
                raise ValueError(f"missing masks in {item['seg_npy']}")
            if masks.ndim == 3:
                masks = masks.squeeze()
            classes_map = self._load_classes_map(dat, masks)
            if classes_map is not None:
                try:
                    unique_ids = np.unique(classes_map)
                    _logger.info(
                        "training labels %s unique class ids=%s",
                        item["seg_npy"],
                        unique_ids.tolist(),
                    )
                except Exception:
                    pass
            else:
                _logger.warning(
                    "training labels %s has no valid classes_map (semantic channel will be omitted)",
                    item["seg_npy"],
                )
            mask_classes = dat.get("classes", None)
            if mask_classes is None and classes_map is not None:
                nmask = int(masks.max())
                mask_classes = np.zeros(nmask + 1, dtype=np.int16)
                for j in range(1, nmask + 1):
                    clsj = classes_map[masks == j]
                    if clsj.size > 0:
                        cls_vals = clsj[clsj > 0]
                        if cls_vals.size > 0:
                            counts = np.bincount(cls_vals.astype(np.int64))
                            mask_classes[j] = int(np.argmax(counts))
            if class_id is not None and mask_classes is not None:
                masks = self._filter_masks_by_class(masks, mask_classes, class_id)
                if masks is None or masks.max() == 0:
                    return None, None
            labels = self._labels_from_masks(masks, classes_map=classes_map, device=flow_device)
            return img, labels

        if "masks" in item:
            masks = io.imread(str(self._resolve_item_path(job_root, item["masks"])))
            if masks.ndim == 3:
                masks = masks.squeeze()
            classes_map = None
            if "classes" in item and item["classes"]:
                try:
                    classes_path = self._resolve_item_path(job_root, item["classes"])
                    if Path(classes_path).exists():
                        classes_map = io.imread(str(classes_path))
                        if classes_map.ndim == 3:
                            classes_map = classes_map.squeeze()
                    else:
                        classes_map = None
                except Exception:
                    classes_map = None
            if class_id is not None and classes_map is not None:
                nmask = int(masks.max())
                mask_classes = np.zeros(nmask + 1, dtype=np.int16)
                for j in range(1, nmask + 1):
                    clsj = classes_map[masks == j]
                    if clsj.size > 0:
                        cls_vals = clsj[clsj > 0]
                        if cls_vals.size > 0:
                            counts = np.bincount(cls_vals.astype(np.int64))
                            mask_classes[j] = int(np.argmax(counts))
                masks = self._filter_masks_by_class(masks, mask_classes, class_id)
                if masks is None or masks.max() == 0:
                    return None, None
            if "flows" in item and item["flows"]:
                flows = io.imread(str(self._resolve_item_path(job_root, item["flows"])))
                if flows.ndim == 3 and flows.shape[0] < 4:
                    labels = np.concatenate((masks[np.newaxis, ...], flows), axis=0)
                else:
                    labels = flows
                if classes_map is not None:
                    labels = np.concatenate((labels[:1], classes_map[np.newaxis, ...], labels[1:]), axis=0)
            else:
                labels = self._labels_from_masks(masks, classes_map=classes_map, device=flow_device)
            return img, labels
        raise ValueError(f"unsupported label item: {item}")

    def _run_training_job(self, request, user_root, user_id):
        import json
        import numpy as np
        import shutil
        from cellpose import models, train, io
        try:
            io.SUPPRESS_NON_TIFF_INFO = True
            io.SUPPRESS_OVERWRITE_TILES_PROMPT = True
            io.GUI = False
        except Exception:
            pass

        stages = ["queue", "validate", "flows", "train", "save", "done"]
        yield pb2.JobUpdate(progress=0, stage=stages[0], message="Training job queued")

        _logger.info(f"training request user={user_id} project_id={request.project_id} uris={list(request.uris)}")
        if not request.uris:
            raise ValueError("missing manifest URI")
        manifest_uri = request.uris[0]
        if not manifest_uri.startswith("file://"):
            raise ValueError("Invalid manifest URI")

        manifest_path = Path(manifest_uri.replace("file://", ""))
        if user_root.resolve() not in manifest_path.resolve().parents and manifest_path.resolve() != user_root.resolve():
            raise PermissionError("Access denied")
        if not manifest_path.exists():
            raise FileNotFoundError("Manifest not found")

        job_root = manifest_path.parent
        train_jobs_root = user_root / "train_jobs"
        train_jobs_root.mkdir(parents=True, exist_ok=True)
        yield pb2.JobUpdate(progress=10, stage=stages[1], message="Loading manifest")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        _logger.info(f"training manifest loaded from {manifest_path}")
        _logger.info(f"training items={len(manifest.get('train', []))} test items={len(manifest.get('test', []))}")
        class_id = manifest.get("specialized_class_id", None)
        training_params = manifest.get("training_params", {})
        normalize_params = manifest.get("normalize_params", {})
        use_lora = bool(training_params.get("use_lora", False))
        base_model = manifest.get("base_model", "cpsam")
        if use_lora and os.path.basename(str(base_model)) != "cpsam":
            _logger.info(
                "LoRA enabled: overriding remote base model '%s' -> 'cpsam'.",
                base_model,
            )
            base_model = "cpsam"
        resolved_base_model = self._resolve_model_id(base_model, user_root=user_root)
        if resolved_base_model != base_model:
            _logger.info(f"resolved base model {base_model} -> {resolved_base_model}")
        base_model = resolved_base_model
        use_gpu = bool(manifest.get("use_gpu", True))
        al_finetune = bool(training_params.get("al_finetune", False))
        al_unfreeze_blocks = int(training_params.get("al_unfreeze_blocks", 9))
        min_train_masks = training_params.get("min_train_masks", 0)
        if min_train_masks not in (0, None):
            _logger.info(
                "GUI_INFO: Remote alignment: ignoring min_train_masks=%s and using 0.",
                min_train_masks,
            )
        min_train_masks = 0
        _logger.info(f"training params model={base_model} gpu={use_gpu} epochs={training_params.get('n_epochs')} lr={training_params.get('learning_rate')}")

        train_items = manifest.get("train", [])
        test_items = manifest.get("test", [])
        # Drop missing class map files to avoid hard failures
        def _strip_missing_classes(items, job_root):
            cleaned = []
            for item in items:
                if "classes" in item and item["classes"]:
                    try:
                        class_path = self._resolve_item_path(job_root, item["classes"])
                        if not Path(class_path).exists():
                            _logger.warning(f"missing classes file {class_path}, dropping classes for item")
                            item = dict(item)
                            item.pop("classes", None)
                    except Exception:
                        _logger.warning(f"error resolving classes file for item, dropping classes")
                        item = dict(item)
                        item.pop("classes", None)
                cleaned.append(item)
            return cleaned
        train_items = _strip_missing_classes(train_items, job_root)
        test_items = _strip_missing_classes(test_items, job_root)
        missing_classes = []
        for item in (train_items + test_items):
            cls = item.get("classes")
            if cls:
                try:
                    cls_path = self._resolve_item_path(job_root, cls)
                    if not Path(cls_path).exists():
                        missing_classes.append(str(cls_path))
                except Exception:
                    missing_classes.append(str(cls))
        if missing_classes:
            for path in missing_classes:
                yield pb2.JobUpdate(progress=5, stage=stages[1],
                                    message=f"Missing classes file skipped: {path}")
        def _strip_missing_files(items, label):
            cleaned = []
            removed = 0
            for item in items:
                try:
                    image_path = self._resolve_item_path(job_root, item.get("image", ""))
                except Exception:
                    image_path = ""
                if not image_path or not Path(image_path).exists():
                    removed += 1
                    continue
                new_item = dict(item)
                seg = new_item.get("seg_npy")
                if seg:
                    seg_path = self._resolve_item_path(job_root, seg)
                    if not Path(seg_path).exists():
                        new_item.pop("seg_npy", None)
                masks = new_item.get("masks")
                if masks:
                    masks_path = self._resolve_item_path(job_root, masks)
                    if not Path(masks_path).exists():
                        removed += 1
                        continue
                flows = new_item.get("flows")
                if flows:
                    flows_path = self._resolve_item_path(job_root, flows)
                    if not Path(flows_path).exists():
                        new_item.pop("flows", None)
                classes = new_item.get("classes")
                if classes:
                    classes_path = self._resolve_item_path(job_root, classes)
                    if not Path(classes_path).exists():
                        new_item.pop("classes", None)
                if "seg_npy" not in new_item and "masks" not in new_item:
                    removed += 1
                    continue
                cleaned.append(new_item)
            if removed:
                _logger.warning(f"dropped {removed} {label} items with missing files")
            return cleaned

        train_items = _strip_missing_files(train_items, "training")
        test_items = _strip_missing_files(test_items, "test")
        if manifest.get("aggregate_existing", False):
            try:
                existing_items = self._scan_project_items(job_root)
                if existing_items:
                    existing_set = set()
                    for item in train_items:
                        existing_set.add(item.get("image"))
                    for item in existing_items:
                        if item.get("image") not in existing_set:
                            train_items.append(item)
                    _logger.info(f"aggregated {len(existing_items)} existing items into training set")
            except Exception as exc:
                _logger.warning(f"failed to aggregate existing items ({exc})")
        if len(train_items) == 0:
            raise ValueError("Manifest has no training items")

        yield pb2.JobUpdate(progress=20, stage=stages[1], message="Validating training data")
        flow_device = None
        try:
            import torch
            if use_gpu:
                if torch.cuda.is_available():
                    flow_device = torch.device("cuda")
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    flow_device = torch.device("mps")
            if flow_device is None:
                flow_device = torch.device("cpu")
        except Exception:
            flow_device = None
        flow_items = [
            item for item in (train_items + test_items)
            if "seg_npy" in item or ("masks" in item and "flows" not in item)
        ]
        flow_total = len(flow_items)
        flow_done = 0
        train_data, train_labels, train_files = [], [], []
        for item in train_items:
            if "seg_npy" in item:
                flow_done += 1
                progress = 25 + int(10 * (flow_done / max(1, flow_total)))
                yield pb2.JobUpdate(
                    progress=progress,
                    stage=stages[2],
                    message=f"Computing flows ({flow_done}/{flow_total})",
                )
            try:
                img, lbl = self._load_label_item(item, job_root, class_id=class_id, flow_device=flow_device)
            except Exception as e:
                yield pb2.JobUpdate(progress=25, stage=stages[1],
                                    message=f"Skipping train item due to error: {e}")
                continue
            if img is None or lbl is None:
                continue
            train_data.append(img)
            train_labels.append(lbl)
            train_files.append(str(job_root / item["image"]))

        if len(train_files) == 0:
            raise ValueError("No valid training items found")

        test_data, test_labels, test_files = None, None, None
        if test_items:
            test_data, test_labels, test_files = [], [], []
            for item in test_items:
                if "seg_npy" in item:
                    flow_done += 1
                    progress = 25 + int(10 * (flow_done / max(1, flow_total)))
                    yield pb2.JobUpdate(
                        progress=progress,
                        stage=stages[2],
                        message=f"Computing flows ({flow_done}/{flow_total})",
                    )
                try:
                    img, lbl = self._load_label_item(item, job_root, class_id=class_id, flow_device=flow_device)
                except Exception as e:
                    yield pb2.JobUpdate(progress=30, stage=stages[1],
                                        message=f"Skipping test item due to error: {e}")
                    continue
                if img is None or lbl is None:
                    continue
                test_data.append(img)
                test_labels.append(lbl)
                test_files.append(str(job_root / item["image"]))

        # Ensure consistent semantic label channel count across all items.
        # Mixed [mask+flows] (4ch) and [mask+class+flows] (5ch) labels can crash
        # augmentation with "index ... out of bounds".
        try:
            import numpy as np
            all_labels = list(train_labels) + list(test_labels or [])
            has_class_channel = any(getattr(lbl, "ndim", 0) == 3 and lbl.shape[0] >= 5 for lbl in all_labels)
            if has_class_channel:
                def _ensure_class_channel(lbl):
                    if getattr(lbl, "ndim", 0) != 3:
                        return lbl
                    if lbl.shape[0] >= 5:
                        return lbl
                    bg_class = np.zeros(lbl.shape[1:], dtype=np.int64)
                    return np.concatenate((lbl[:1], bg_class[np.newaxis, ...], lbl[1:]), axis=0)

                train_labels = [_ensure_class_channel(lbl) for lbl in train_labels]
                if test_labels is not None:
                    test_labels = [_ensure_class_channel(lbl) for lbl in test_labels]
        except Exception:
            pass

        train_class_maps = extract_class_maps_from_labels(train_labels)

        semantic_classes = 0
        try:
            max_class = max((int(np.max(cm)) for cm in train_class_maps), default=-1)
            if max_class >= 1:
                semantic_classes = int(max_class + 1)
            else:
                semantic_classes = 0
        except Exception:
            semantic_classes = 0
        if os.path.basename(str(base_model)) == "cpsam" and semantic_classes > 0:
            semantic_classes = max(semantic_classes, 4)

        yield pb2.JobUpdate(progress=35, stage=stages[3], message="Initializing model")
        model = models.CellposeModel(
            gpu=use_gpu, pretrained_model=base_model,
            semantic_classes=semantic_classes if semantic_classes > 0 else None,
        )

        try:
            net = model.net
            if use_lora:
                if al_finetune:
                    n_lora_blocks = al_unfreeze_blocks
                else:
                    try:
                        raw_blocks = training_params.get("lora_blocks", None)
                        if raw_blocks is None:
                            raw_blocks = training_params.get("unfreeze_blocks", 9)
                        n_lora_blocks = 9 if raw_blocks is None else int(raw_blocks)
                    except Exception:
                        n_lora_blocks = 9
            elif hasattr(net, "encoder") and hasattr(net.encoder, "blocks"):
                n_unfreeze = al_unfreeze_blocks if al_finetune else int(training_params.get("unfreeze_blocks", 9))
            else:
                n_unfreeze = int(training_params.get("unfreeze_blocks", 9))

            mode_info = configure_trainable_params(
                net,
                use_lora=bool(use_lora),
                lora_blocks=n_lora_blocks if use_lora else None,
                unfreeze_blocks=n_unfreeze if not use_lora else None,
                logger=_logger,
            )
            if use_lora and mode_info.get("lora_info") is not None:
                lora_info = mode_info["lora_info"]
                _logger.info(
                    "GUI_INFO: LoRA injected into last %s/%s encoder blocks, converted_linear_layers=%s",
                    lora_info.get("applied_blocks"),
                    lora_info.get("total_blocks"),
                    lora_info.get("converted_linear_layers"),
                )
                try:
                    rep = models.lora_trainability_report(net)
                    _logger.info(
                        "GUI_INFO: LoRA params total=%s trainable=%s lora_trainable=%s out_trainable=%s",
                        rep["total_params"],
                        rep["trainable_params"],
                        rep["lora_trainable_params"],
                        rep["out_trainable_params"],
                    )
                    if rep["unexpected_encoder_trainable"]:
                        _logger.warning(
                            "GUI_WARN: unexpected trainable encoder params detected in LoRA mode (n=%s): %s",
                            len(rep["unexpected_encoder_trainable"]),
                            rep["unexpected_encoder_trainable"][:10],
                        )
                except Exception:
                    pass
        except Exception:
            pass

        replay_items = list(getattr(self, "replay_items", []))
        if replay_items:
            import random
            sample_n = min(self.replay_sample_size, len(replay_items))
            sample = random.sample(replay_items, sample_n)
            _logger.info(f"adding {sample_n} replay items from {self.replay_root}")
            for item in sample:
                img, lbl = self._load_label_item(item, self.replay_root, class_id=class_id, flow_device=flow_device)
                if img is None or lbl is None:
                    continue
                train_data.append(img)
                train_labels.append(lbl)
                train_files.append(str(self._resolve_item_path(self.replay_root, item["image"])))

        # Recompute class maps after optional replay augmentation.
        train_class_maps = extract_class_maps_from_labels(train_labels)

        class_weights = None
        try:
            nclasses_inferred = infer_semantic_nclasses_from_net(model.net)
            max_class_id = max((int(np.max(cm)) for cm in train_class_maps), default=-1)
            class_weights = compute_class_weights_from_class_maps(
                train_class_maps,
                nclasses=nclasses_inferred,
            )
            _logger.info(
                "GUI_INFO: class-weight inputs: valid_class_maps=%s max_class_id=%s inferred_nclasses=%s",
                int(len(train_class_maps)),
                int(max_class_id),
                nclasses_inferred,
            )
        except Exception:
            class_weights = None
        if class_weights is not None:
            _logger.info(
                "GUI_INFO: class weights (%s classes incl. bg): %s",
                int(len(class_weights)),
                class_weights,
            )
            _logger.info(
                "GUI_INFO: class-weight outputs: weight_vector_length=%s",
                int(len(class_weights)),
            )
        else:
            _logger.info(
                "GUI_INFO: class weights unavailable (no valid semantic class maps detected); using unweighted CE."
            )
            _logger.info(
                "GUI_INFO: class-weight outputs: weight_vector_length=0 (unweighted CE fallback)"
            )

        save_dir = train_jobs_root / manifest.get("job_id", "job")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_dir = save_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        yield pb2.JobUpdate(progress=50, stage=stages[3], message="Training model")
        model_name = training_params.get("model_name", "cpsam_remote")
        _logger.info("GUI_INFO: name of new model: %s", model_name)
        n_epochs = int(training_params.get("n_epochs", 100))
        learning_rate = float(training_params.get("learning_rate", 1e-5))
        weight_decay = float(training_params.get("weight_decay", 0.1))
        early_stop = bool(training_params.get("early_stop", False))
        batch_size = int(training_params.get("batch_size", 10))
        bsize = int(training_params.get("bsize", 256))
        rescale = bool(training_params.get("rescale", False))
        scale_range = float(training_params.get("scale_range", 0.5))
        out = train.train_seg(
            model.net, train_data=train_data, train_labels=train_labels,
            test_data=test_data, test_labels=test_labels, test_files=test_files,
            normalize=normalize_params, min_train_masks=min_train_masks,
            save_path=str(model_dir),
            nimg_per_epoch=len(train_data),
            nimg_test_per_epoch=len(test_data) if test_data else 0,
            batch_size=batch_size,
            bsize=bsize,
            rescale=rescale,
            scale_range=scale_range,
            learning_rate=learning_rate, weight_decay=weight_decay,
            n_epochs=n_epochs, early_stop=early_stop,
            model_name=model_name, class_weights=class_weights)
        if isinstance(out, (list, tuple)):
            model_path = out[0]
            train_losses = out[1] if len(out) > 1 else None
            test_losses = out[2] if len(out) > 2 else None
        else:
            model_path = out
            train_losses = None
            test_losses = None

        if train_losses is not None:
            try:
                for idx in range(0, len(train_losses), 5):
                    tl = float(train_losses[idx])
                    if test_losses is not None and idx < len(test_losses):
                        vl = float(test_losses[idx])
                        _logger.info(
                            "GUI_INFO: epoch %s/%s train_loss=%.4f test_loss=%.4f",
                            idx + 1,
                            n_epochs,
                            tl,
                            vl,
                        )
                    else:
                        _logger.info(
                            "GUI_INFO: epoch %s/%s train_loss=%.4f",
                            idx + 1,
                            n_epochs,
                            tl,
                        )
            except Exception:
                pass

        if use_lora:
            try:
                model.net.load_model(model_path, device=model.device)
                models.merge_and_remove_lora(model.net)
                import torch
                torch.save(model.net.state_dict(), model_path)
            except Exception:
                pass

        train_losses_path = save_dir / f"{Path(model_path).name}_train_losses.npy"
        try:
            needed = 0
            try:
                needed += int(os.path.getsize(model_path))
            except Exception:
                pass
            try:
                needed += int(train_losses.nbytes)
            except Exception:
                pass
            needed += 1024 * 1024
            _ensure_quota(user_root, needed)
        except RuntimeError as exc:
            raise RuntimeError(str(exc))
        np.save(str(train_losses_path), train_losses)

        meta_path = save_dir / f"{Path(model_path).name}_meta.json"
        meta = {
            "model_name": model_name,
            "save_path": str(model_path),
            "n_train": len(train_files),
            "n_test": len(test_files) if test_files else 0,
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        try:
            user_model_dir = user_root / "models"
            user_model_dir.mkdir(parents=True, exist_ok=True)
            dest = user_model_dir / Path(model_path).name
            shutil.copy2(model_path, dest)
        except Exception:
            dest = Path(model_path)

        artifacts = {
            "model_uri": f"file://{Path(dest).as_posix()}",
            "train_losses_uri": f"file://{train_losses_path.as_posix()}",
            "meta_uri": f"file://{meta_path.as_posix()}",
            "job_dir_uri": f"file://{save_dir.as_posix()}",
        }
        yield pb2.JobUpdate(progress=90, stage=stages[4], message="Saving artifacts")
        yield pb2.JobUpdate(progress=100, stage=stages[5],
                            message=json.dumps(artifacts),
                            result_uri=artifacts["model_uri"])


class TrainingJob:
    def __init__(self, request, user_root, user_id):
        self.request = request
        self.user_root = user_root
        self.user_id = user_id
        self.updates = queue.Queue()
        self.error = None


class TrainingManager:
    def __init__(self, service):
        self.service = service
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, request, user_root, user_id):
        job = TrainingJob(request, user_root, user_id)
        self.queue.put(job)
        return job

    def _worker(self):
        while True:
            job = self.queue.get()
            try:
                for update in self.service._run_training_job(job.request, job.user_root, job.user_id):
                    job.updates.put(update)
            except Exception as exc:
                job.error = exc
                job.updates.put(pb2.JobUpdate(progress=0, stage="error", message=str(exc)))
            finally:
                job.updates.put(None)


def serve(bind: str, storage_root: str, max_workers: int = 8, interceptors=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), interceptors=interceptors or [])
    pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    if GENERATED:
        cfg = remote_config.load_remote_config()
        pb2_grpc.add_FileServiceServicer_to_server(
            FileService(storage_root, model_root=cfg.get("model_root")),
            server,
        )
        pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(storage_root), server)
    server.add_insecure_port(bind)
    server.start()
    return server
