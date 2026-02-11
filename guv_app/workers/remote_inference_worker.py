import os
import tempfile
import json
from cellpose import io as cp_io
from cellpose import models
from guv_app.workers.base_worker import BaseWorker
from guv_app.services.remote_service import RemoteConnectionService
from guv_app.data_models.results import InferenceResult
from cpgrpc.server import cellpose_remote_pb2 as pb2
from PyQt6.QtCore import pyqtSignal
import numpy as np

class RemoteInferenceWorker(BaseWorker):
    finished = pyqtSignal(InferenceResult)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, remote_service: RemoteConnectionService, image_service, local_filename: str, model_id: str, diameter: float, channel_index=None, frame_id=None):
        super().__init__()
        self.remote_service = remote_service
        self.image_service = image_service
        self.local_filename = local_filename
        self.model_id = model_id
        self.diameter = diameter
        self.channel_index = channel_index
        self.frame_id = frame_id

    def run(self):
        try:
            self.progress.emit(0, "Uploading image...")
            upload_path = self.local_filename
            temp_path = None
            if self.channel_index is not None or self.frame_id is not None:
                image = None
                if self.frame_id is not None:
                    image = self.image_service.load_frame(self.local_filename, self.frame_id)
                if image is None:
                    image = self.image_service.load_image(self.local_filename)
                if image is not None:
                    if self.channel_index is not None and image.ndim == 3:
                        if image.shape[-1] > self.channel_index and image.shape[-1] <= 4:
                            image = image[..., self.channel_index]
                        elif image.shape[0] > self.channel_index and image.shape[0] <= 4:
                            image = image[self.channel_index, ...]
                        elif image.shape[2] > self.channel_index:
                            image = image[..., self.channel_index]
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
                        temp_path = tf.name
                    cp_io.imsave(temp_path, image)
                    upload_path = temp_path
            uri = self.remote_service.upload_file("default", upload_path)
            self.progress.emit(10, f"Image uploaded to {uri}")

            req = pb2.RunRequest(
                project_id="default",
                uris=[uri],
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

            result_uri = None
            for update in self.remote_service.run_inference(req):
                # Scale server progress (0-100) to our UI range (10-100)
                ui_progress = 10 + int(update.progress * 0.9)
                self.progress.emit(ui_progress, f"{update.stage}: {update.message}")
                if update.result_uri:
                    result_uri = update.result_uri

            if result_uri:
                self.progress.emit(100, "Downloading result...")
                # Download the result
                output_path = self.image_service.build_frame_path(
                    self.local_filename,
                    self.frame_id,
                    "_pred.npy",
                )
                self.remote_service.download_file(result_uri, output_path)
                
                # Load the data to return a standard result
                data = np.load(output_path, allow_pickle=True).item()
                result = InferenceResult(
                    masks=data.get("masks"),
                    flows=data.get("flows"),
                    styles=None, # Styles might not be in saved npy
                    filename=self.local_filename,
                    frame_id=self.frame_id,
                    is_saved=True,
                    classes=data.get("classes"),
                    classes_map=data.get("classes_map"),
                    class_names=data.get("class_names"),
                    class_colors=data.get("class_colors")
                )
                self.finished.emit(result)
            else:
                self.error.emit("Remote inference finished but returned no result URI.")

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
