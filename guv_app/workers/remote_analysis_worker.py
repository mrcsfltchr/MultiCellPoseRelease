import os
import tempfile
import json
import numpy as np
from guv_app.workers.base_worker import BaseWorker
from guv_app.services.remote_service import RemoteConnectionService
from cpgrpc.server import cellpose_remote_pb2 as pb2
from PyQt6.QtCore import pyqtSignal
from guv_app.data_models.configs import BatchConfig
from guv_app.data_models.results import InferenceResult
from cellpose import io as cp_io
from cellpose import models

class RemoteAnalysisWorker(BaseWorker):
    finished = pyqtSignal()
    result_ready = pyqtSignal(InferenceResult)
    progress = pyqtSignal(int, str)

    def __init__(self, remote_service: RemoteConnectionService, image_service, folder_path: str, diameter: float, model_id: str, batch_config: BatchConfig, series_index=None, channel_index=None, file_list=None):
        super().__init__()
        self.remote_service = remote_service
        self.image_service = image_service
        self.folder_path = folder_path
        self.diameter = diameter
        self.model_id = model_id
        self.batch_config = batch_config
        self._is_running = True
        self.series_index = series_index
        self.channel_index = channel_index
        self.file_list = list(file_list) if file_list else None

    def run(self):
        try:
            from cellpose.io import get_image_files
        except ImportError:
            self.progress.emit(0, "cellpose not installed correctly")
            self.finished.emit()
            return
            
        if not self.remote_service or not self.remote_service.health_check():
            self.progress.emit(0, "Remote service not available or not connected")
            self.finished.emit()
            return

        # Use _masks filter to exclude mask files from processing list
        files = self.file_list if self.file_list is not None else get_image_files(self.folder_path, "_masks")
        if not files:
            self.progress.emit(0, "No images found to process.")
            self.finished.emit()
            return

        total_files = len(files)
        self.progress.emit(0, f"Found {total_files} images. Starting remote batch processing...")

        # 1. Upload All Images
        uris = []
        valid_files = []
        temp_paths = []
        for i, f in enumerate(files):
            if not self._is_running:
                break
            try:
                progress_val = int(((i + 1) / total_files) * 50)
                self.progress.emit(progress_val, f"Preparing {i+1}/{total_files}: {os.path.basename(f)}")
                refs = self.image_service.build_frame_references(f, series_index=self.series_index)
                if refs:
                    for ref in refs:
                        base, frame_id = self.image_service.split_image_reference(ref)
                        img = self.image_service.load_frame(base, frame_id)
                        if img is None:
                            continue
                        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
                            temp_path = tf.name
                        temp_paths.append(temp_path)
                        if self.channel_index is not None and hasattr(img, "ndim"):
                            if img.ndim == 3 and img.shape[2] > self.channel_index:
                                img = img[..., self.channel_index]
                        cp_io.imsave(temp_path, img)
                        uri = self.remote_service.upload_file("default", temp_path)
                        uris.append(uri)
                        valid_files.append((f, frame_id))
                    continue

                frames = self.image_service.iter_image_frames(f, series_index=self.series_index)
                if not frames:
                    continue
                for frame in frames:
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
                        temp_path = tf.name
                    temp_paths.append(temp_path)
                    img = frame.array
                    if self.channel_index is not None and hasattr(img, "ndim"):
                        if img.ndim == 3:
                            if img.shape[-1] > self.channel_index and img.shape[-1] <= 4:
                                img = img[..., self.channel_index]
                            elif img.shape[0] > self.channel_index and img.shape[0] <= 4:
                                img = img[self.channel_index, ...]
                            elif img.shape[2] > self.channel_index:
                                img = img[..., self.channel_index]
                    cp_io.imsave(temp_path, img)
                    uri = self.remote_service.upload_file("default", temp_path)
                    uris.append(uri)
                    valid_files.append((f, frame.frame_id))
            except Exception as e:
                self.progress.emit(progress_val, f"Failed to upload {os.path.basename(f)}: {e}")

        if not uris:
            self.progress.emit(0, "No images uploaded successfully.")
            self.finished.emit()
            return

        # 2. Run Inference on All Images (Server handles batching)
        self.progress.emit(50, f"Running inference on {len(uris)} images...")
        try:
            req = pb2.RunRequest(
                project_id="default",
                uris=uris,
                model_id=self.model_id,
                diameter=self.diameter,
                cellprob_threshold=-0.5,
                flow_threshold=1.0,
                do_3D=False,
                niter=0,
                stitch_threshold=0.0,
                anisotropy=1.0,
                flow3D_smooth=0.0,
                min_size=15,
                max_size_fraction=1.0,
                normalize_params=json.dumps({**models.normalize_default, "normalize": True}),
            )
            
            result_idx = 0
            for update in self.remote_service.run_inference(req):
                if not self._is_running:
                    break
                
                if update.message:
                    self.progress.emit(50, f"Remote: {update.message}")

                if update.result_uri:
                    # Match result to local file (server preserves order)
                    if result_idx < len(valid_files):
                        local_file, frame_id = valid_files[result_idx]
                        progress_val = 50 + int(((result_idx + 1) / len(valid_files)) * 50)
                        try:
                            output_path = self.image_service.build_frame_path(local_file, frame_id, "_pred.npy")
                            self.remote_service.download_file(update.result_uri, output_path)
                            self.progress.emit(progress_val, f"Finished {os.path.basename(local_file)}")
                            
                            # Load the data to return a standard result with classes
                            data = np.load(output_path, allow_pickle=True).item()
                            result = InferenceResult(
                                masks=data.get("masks"),
                                flows=data.get("flows"),
                                filename=local_file,
                                frame_id=frame_id,
                                is_saved=False,
                                classes=data.get("classes"),
                                classes_map=data.get("classes_map"),
                                class_names=data.get("class_names"),
                                class_colors=data.get("class_colors")
                            )
                            self.result_ready.emit(result)
                        except Exception as e:
                            self.progress.emit(progress_val, f"Failed to download result for {os.path.basename(local_file)}: {e}")
                        
                        result_idx += 1

        except Exception as e:
            self.progress.emit(50, f"Remote inference failed: {e}")
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        
        self.progress.emit(100, "Done")
        self.finished.emit()

    def stop(self):
        self._is_running = False
