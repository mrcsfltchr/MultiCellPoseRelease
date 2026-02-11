import os
import json
import tempfile
import uuid
from PyQt6.QtCore import pyqtSignal, QObject
from urllib.parse import urlparse

from guv_app.data_models.configs import TrainingConfig, RemoteConfig
from guv_app.data_models.results import TrainingResult
from cpgrpc.client import client as grpc_client
from cpgrpc.server import cellpose_remote_pb2 as pb2
from cellpose import models, io as cellpose_io

class RemoteTrainingWorker(QObject):
    """
    Worker for running model training on a remote server.
    """
    finished = pyqtSignal(TrainingResult)
    progress = pyqtSignal(str, str)  # stage, message
    error = pyqtSignal(str)

    def __init__(self, config: TrainingConfig, remote_config: RemoteConfig):
        super().__init__()
        self.config = config
        self.remote_config = remote_config

    def run(self):
        """
        Entry-point routine that orchestrates execution.
        """
        try:
            channel = grpc_client.make_channel(self.remote_config.address, self.remote_config.insecure)
            
            project_id = f"train_job_{uuid.uuid4()}"
            
            self.progress.emit("setup", f"Created remote project: {project_id}")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Upload files
                self.progress.emit("upload", "Uploading training files...")
                train_local, train_labels_local = self._prepare_training_files(
                    temp_dir, self.config.train_files, self.config.train_labels_files
                )
                test_local, test_labels_local = self._prepare_training_files(
                    temp_dir, self.config.test_files, self.config.test_labels_files
                )
                train_uris = self._upload_files(channel, project_id, train_local)
                train_labels_uris = self._upload_files(channel, project_id, train_labels_local)
                test_uris = self._upload_files(channel, project_id, test_local)
                test_labels_uris = self._upload_files(channel, project_id, test_labels_local)

                # Create and upload manifest
                manifest_path = os.path.join(temp_dir, "manifest.json")
                manifest = self._create_manifest(train_uris, train_labels_uris, test_uris, test_labels_uris)
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f)
                
                self.progress.emit("upload", "Uploading manifest...")
                manifest_reply = grpc_client.upload_file(channel, project_id, manifest_path, relpath="manifest.json", token=self.remote_config.token)
                
                # Start training
                self.progress.emit("train", "Starting remote training job...")
                req = pb2.RunRequest(project_id=project_id, model_id="__train__", uris=[manifest_reply.uri])
                
                updates = grpc_client.run_inference(channel, req, token=self.remote_config.token)
                
                model_uri = None
                artifact_uris = None
                for update in updates:
                    self.progress.emit(update.stage, update.message)
                    if update.stage == "done":
                        try:
                            result_data = json.loads(update.message)
                            model_uri = result_data.get("model_uri")
                            artifact_uris = result_data
                        except json.JSONDecodeError:
                            self.error.emit(f"Failed to parse final training message: {update.message}")
                            return

                if not model_uri:
                    self.error.emit("Training completed, but no model URI was returned.")
                    return

                # Download result
                self.progress.emit("download", f"Downloading model from {model_uri}")
                save_root = self.config.save_path
                if not save_root:
                    try:
                        save_root = os.fspath(models.MODEL_DIR)
                    except Exception:
                        save_root = temp_dir
                os.makedirs(save_root, exist_ok=True)
                parsed_name = None
                try:
                    parsed = urlparse(model_uri)
                    parsed_name = os.path.basename(parsed.path)
                except Exception:
                    parsed_name = None
                filename = parsed_name or self.config.model_name or "trained_model.pth"
                if not os.path.splitext(filename)[1]:
                    filename = f"{filename}.pth"
                local_model_path = os.path.join(save_root, filename)
                grpc_client.download_file(channel, model_uri, local_model_path, token=self.remote_config.token)

                self.finished.emit(TrainingResult(model_path=local_model_path, artifact_uris=artifact_uris))

        except Exception as e:
            self.error.emit(str(e))

    def _upload_files(self, channel, project_id, files):
        relpaths = []
        if not files:
            return relpaths
        for f in files:
            filename = os.path.basename(f)
            self.progress.emit("upload", f"Uploading {filename}...")
            grpc_client.upload_file(channel, project_id, f, relpath=filename, token=self.remote_config.token)
            relpaths.append(filename)
        return relpaths

    def _prepare_training_files(self, temp_dir, image_files, label_files):
        if not image_files or not label_files:
            return [], []
        prepared_images = []
        prepared_labels = []
        for image_path, label_path in zip(image_files, label_files):
            if image_path and "::" in image_path:
                base_path, frame_id = image_path.split("::", 1)
                frame = cellpose_io.read_image_frame(base_path, frame_id)
                if frame is None or frame.array is None:
                    raise RuntimeError(f"Failed to read frame {frame_id} from {base_path}")
                base_name = os.path.splitext(os.path.basename(base_path))[0]
                safe_name = f"{base_name}__{frame_id}.tif"
                out_path = os.path.join(temp_dir, safe_name)
                cellpose_io.imsave(out_path, frame.array)
                prepared_images.append(out_path)
            else:
                prepared_images.append(image_path)
            prepared_labels.append(label_path)
        return prepared_images, prepared_labels

    def _create_manifest(self, train_relpaths, train_labels_relpaths, test_relpaths, test_labels_relpaths):
        train_items = [{"image": img, "seg_npy": lbl} for img, lbl in zip(train_relpaths, train_labels_relpaths)]
        test_items = [{"image": img, "seg_npy": lbl} for img, lbl in zip(test_relpaths, test_labels_relpaths)]

        normalize_params = dict(models.normalize_default)
        normalize_params["normalize"] = True
        return {
            "train": train_items,
            "test": test_items,
            "base_model": self.config.base_model,
            "use_gpu": True,
            "normalize_params": normalize_params,
            "training_params": {
                "model_name": self.config.model_name,
                "n_epochs": self.config.n_epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "bsize": self.config.bsize,
                "rescale": self.config.rescale,
                "scale_range": self.config.scale_range,
                "min_train_masks": self.config.min_train_masks,
                "use_lora": self.config.use_lora,
                "lora_blocks": self.config.lora_blocks,
                "unfreeze_blocks": self.config.unfreeze_blocks,
            }
        }
