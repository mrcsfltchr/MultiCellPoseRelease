import os
import time
from PyQt6.QtCore import QSettings
from urllib.parse import urlparse

from PyQt6.QtCore import pyqtSlot, QThread, QObject, pyqtSignal
from guv_app.controllers.main_controller import MainController
from guv_app.workers.training_worker import TrainingWorker
from guv_app.workers.remote_training_worker import RemoteTrainingWorker
from guv_app.data_models.configs import TrainingConfig
from guv_app.services.training_dataset_service import TrainingDatasetService
import logging
from cellpose.models import CellposeModel

_logger = logging.getLogger(__name__)


class _TrainingDiscoveryWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, dataset_service, folder_path):
        super().__init__()
        self._dataset_service = dataset_service
        self._folder_path = folder_path

    def run(self):
        try:
            try:
                files = self._dataset_service.list_training_images(self._folder_path)
            except Exception:
                files = []
            if not files:
                try:
                    files = self._dataset_service.list_training_images(
                        self._folder_path, look_one_level_down=True
                    )
                except Exception:
                    files = []
            self.finished.emit(files)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainerController(MainController):
    """
    Controller for the Trainer application.
    Inherits from MainController to reuse navigation, image loading, and drawing logic.
    """
    def __init__(self, model, view, services):
        super().__init__(model, view, services)
        self.worker = None
        self.thread = None
        self.discovery_worker = None
        self.discovery_thread = None
        self.training_dataset_service = services.get("training_dataset")
        if self.training_dataset_service is None:
            self.training_dataset_service = TrainingDatasetService()
        self._remote_training_active = False
        self._remote_train_start_ts = None
        self._pending_folder_path = None
        self._pending_use_remote = False

    def connect_signals(self):
        """Connects signals, including those from the base controller."""
        super().connect_signals()
        self.view.training_requested.connect(self.handle_start_training)
        if hasattr(self.view, "training_cancel_requested"):
            self.view.training_cancel_requested.connect(self.handle_cancel_training)

    @pyqtSlot()
    def handle_start_training(self):
        if self.thread is not None and self.thread.isRunning():
            self.view.statusBar().showMessage("Training already running.")
            return
        if self.discovery_thread is not None and self.discovery_thread.isRunning():
            self.view.statusBar().showMessage("Training preparation already running.")
            return
        if not self.model.filename:
            self.view.statusBar().showMessage("No image loaded for training.")
            return

        _logger.info("GUI_INFO: preparing training data from current folder")
        self.view.statusBar().showMessage("Preparing training data...")
        base, _ = self.image_service.split_image_reference(self.model.filename)
        self._pending_folder_path = os.path.dirname(base)
        self._pending_use_remote = self.is_remote_connected()
        self._start_training_discovery()
        return

    def _start_training_discovery(self):
        folder_path = self._pending_folder_path
        if not folder_path:
            self.view.statusBar().showMessage("No folder found for training data.")
            return
        self.discovery_worker = _TrainingDiscoveryWorker(self.training_dataset_service, folder_path)
        self.discovery_thread = QThread()
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_thread.started.connect(self.discovery_worker.run)
        self.discovery_worker.finished.connect(self._on_discovery_finished)
        self.discovery_worker.error.connect(self._on_discovery_error)
        self.discovery_worker.finished.connect(self.discovery_thread.quit)
        self.discovery_worker.error.connect(self.discovery_thread.quit)
        self.discovery_thread.finished.connect(self.discovery_worker.deleteLater)
        self.discovery_thread.finished.connect(self.discovery_thread.deleteLater)
        self.discovery_thread.start()

    def _on_discovery_error(self, message):
        self.view.statusBar().showMessage(f"Failed to list training images: {message}")
        self._cleanup_discovery_thread()

    def _on_discovery_finished(self, image_files):
        self._cleanup_discovery_thread()
        self._begin_training_with_images(image_files)

    def _begin_training_with_images(self, image_files):
        use_remote = self._pending_use_remote
        folder_path = self._pending_folder_path
        train_files = []
        test_files = []
        train_labels = None
        train_label_files = []
        test_label_files = []
        train_data = None
        test_data = None
        test_labels = None
        class_maps = None
        test_class_maps = None
        if not image_files:
            self.view.statusBar().showMessage("No labeled training data found in the folder.")
            return

        test_ratio = 0.0
        test_pick = None
        if hasattr(self.view, "prompt_test_set_files"):
            test_pick = self.view.prompt_test_set_files(folder_path)
            if test_pick is None:
                self.view.statusBar().showMessage("Training canceled.")
                return
        if test_pick is not None:
            test_files = list(test_pick)
            train_files = [p for p in image_files if p not in test_files]

        if not train_files and test_files and len(image_files) > 1 and hasattr(self.view, "prompt_train_test_split"):
            ratio = self.view.prompt_train_test_split(len(image_files))
            if ratio is None:
                self.view.statusBar().showMessage("Training canceled.")
                return
            test_ratio = ratio
            if test_ratio > 0:
                train_files, test_files = self.training_dataset_service.split_train_test(
                    image_files, test_ratio
                )
            else:
                train_files = image_files
                test_files = []

        if not test_files and len(image_files) > 1 and hasattr(self.view, "prompt_train_test_split"):
            ratio = self.view.prompt_train_test_split(len(image_files))
            if ratio is None:
                self.view.statusBar().showMessage("Training canceled.")
                return
            test_ratio = ratio

            if test_ratio > 0:
                train_files, test_files = self.training_dataset_service.split_train_test(
                    image_files, test_ratio
                )
            else:
                train_files = image_files
        elif not train_files:
            train_files = image_files

        if use_remote:
            train_files, train_label_files, missing = self.training_dataset_service.pair_images_with_labels(
                train_files,
                expand_series=True,
            )
            test_files, test_label_files, missing_test = self.training_dataset_service.pair_images_with_labels(
                test_files,
                expand_series=True,
            )
            missing_all = missing + missing_test
            if missing_all:
                if train_files or test_files:
                    self.view.statusBar().showMessage(
                        f"Skipping {len(missing_all)} unlabeled images; training on labeled data."
                    )
                else:
                    self.view.statusBar().showMessage(
                        f"Missing labels for {len(missing_all)} images. Fix labels before training."
                    )
                    return
            train_files, train_label_files, invalid = self.training_dataset_service.validate_training_pairs(
                train_files, train_label_files
            )
            test_files, test_label_files, invalid_test = self.training_dataset_service.validate_training_pairs(
                test_files, test_label_files
            )
            if invalid or invalid_test:
                for image_path, label_path, issues in invalid[:20]:
                    _logger.warning(
                        "Invalid training pair (train): image=%s label=%s issues=%s",
                        image_path,
                        label_path,
                        "; ".join(issues) if issues else "unknown",
                    )
                for image_path, label_path, issues in invalid_test[:20]:
                    _logger.warning(
                        "Invalid training pair (test): image=%s label=%s issues=%s",
                        image_path,
                        label_path,
                        "; ".join(issues) if issues else "unknown",
                    )

                if not train_files:
                    first_issue = "unknown"
                    if invalid:
                        _, bad_label, issues = invalid[0]
                        first_issue = f"{os.path.basename(bad_label)}: {'; '.join(issues) if issues else 'invalid label'}"
                    self.view.statusBar().showMessage(
                        f"No valid labeled training images found. First issue: {first_issue}"
                    )
                    return

                skipped_invalid = len(invalid) + len(invalid_test)
                self.view.statusBar().showMessage(
                    f"Skipping {skipped_invalid} invalid labeled items; training on {len(train_files)} valid images."
                )
        else:
            train_data, train_labels, train_files, class_maps, invalid_local = self.training_dataset_service.load_local_sets(
                train_files
            )
            invalid_local_test = []
            if test_files:
                test_data, test_labels, test_files, test_class_maps, invalid_local_test = self.training_dataset_service.load_local_sets(
                    test_files
                )
            if invalid_local or invalid_local_test:
                for image_path, label_path, issues in invalid_local[:20]:
                    _logger.warning(
                        "Invalid training pair (train): image=%s label=%s issues=%s",
                        image_path,
                        label_path,
                        "; ".join(issues) if issues else "unknown",
                    )
                for image_path, label_path, issues in invalid_local_test[:20]:
                    _logger.warning(
                        "Invalid training pair (test): image=%s label=%s issues=%s",
                        image_path,
                        label_path,
                        "; ".join(issues) if issues else "unknown",
                    )
                skipped_invalid = len(invalid_local) + len(invalid_local_test)
                self.view.statusBar().showMessage(
                    f"Skipping {skipped_invalid} invalid labeled items; training on valid images."
                )
            if not train_files:
                first_issue = "unknown"
                if invalid_local:
                    _, bad_label, issues = invalid_local[0]
                    first_issue = f"{os.path.basename(bad_label)}: {'; '.join(issues) if issues else 'invalid label'}"
                self.view.statusBar().showMessage(
                    f"No valid labeled training images found. First issue: {first_issue}"
                )
                return

        model_names = self._get_available_models_for_training()
        default_name = f"{self.model.current_model_id or 'cpsam'}_{time.strftime('%Y%m%d_%H%M%S')}"
        default_config = TrainingConfig(
            base_model=self.model.current_model_id or "cpsam",
            model_name=default_name,
            lora_blocks=9,
            unfreeze_blocks=9,
        )
        if hasattr(self.view, "prompt_training_config"):
            config = self.view.prompt_training_config(
                model_names,
                default_config,
                train_files,
                total_blocks=None,
            )
        else:
            config = default_config
        if config is None:
            self.view.statusBar().showMessage("Training canceled.")
            return

        config.train_files = train_files
        config.test_files = test_files
        if use_remote:
            config.train_labels_files = train_label_files
            config.test_labels_files = test_label_files
        else:
            config.train_labels_files = None
            config.test_labels_files = None

        if use_remote:
            if self._remote_training_active:
                self.view.statusBar().showMessage("Remote training already running.")
                return
            if not self.is_remote_connected():
                self.handle_training_error("Remote service not available or not connected")
                return
            remote_config = self.remote_service.get_config()
            self.worker = RemoteTrainingWorker(config, remote_config)
            self._remote_training_active = True
            self._remote_train_start_ts = time.time()
        else:
            self.worker = TrainingWorker(
                config,
                train_data=train_data,
                train_labels=train_labels,
                test_data=test_data,
                test_labels=test_labels,
                class_maps=class_maps,
                test_class_maps=test_class_maps if test_files else None,
                flow_labels=None,
                test_flow_labels=None,
            )

        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        if use_remote:
            self.worker.progress.connect(self.handle_training_progress_remote)
        else:
            self.worker.progress.connect(self.handle_training_progress_local)
        self.worker.finished.connect(self.handle_training_finished)
        self.worker.error.connect(self.handle_training_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._on_training_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.view.statusBar().showMessage("Training started...")

    def handle_training_progress_local(self, epoch, total_epochs, train_loss, test_loss):
        _logger.info(
            f"Training epoch {epoch + 1}/{total_epochs}: loss={train_loss:.4f} test={test_loss:.4f}"
        )
        self.view.statusBar().showMessage(
            f"Training epoch {epoch + 1}/{total_epochs}: loss={train_loss:.4f} test={test_loss:.4f}"
        )

    def handle_training_progress_remote(self, stage, message):
        safe_message = self._sanitize_remote_message(message)
        _logger.info(f"Training [{stage}]: {safe_message}")
        eta = self._get_train_eta_seconds()
        eta_msg = ""
        if eta > 0 and stage == "train":
            eta_msg = f" (ETA ~{int(eta // 60)}m)"
        self.view.statusBar().showMessage(f"Training [{stage}]: {safe_message}{eta_msg}")

    def handle_training_finished(self, result):
        model_label = os.path.basename(result.model_path) if result.model_path else "trained model"
        _logger.info("Training finished. Model saved.")
        self.view.statusBar().showMessage(f"Training finished. Model saved: {model_label}")
        trained_name = None
        try:
            if self.worker and hasattr(self.worker, "config"):
                trained_name = self.worker.config.model_name
        except Exception:
            trained_name = None
        # Register locally saved model (if available) so inference can find it.
        resolved_name = None
        try:
            if result.model_path and os.path.isfile(result.model_path):
                from cellpose import io as cellpose_io
                cellpose_io.add_model(result.model_path)
                resolved_name = os.path.basename(result.model_path)
        except Exception:
            resolved_name = None
        if trained_name and self.model_service:
            try:
                self.model_service.register_model(resolved_name or trained_name)
            except Exception:
                pass
            self.refresh_model_list(select_model_name=resolved_name or trained_name)
        self._record_train_duration()
        if result.artifact_uris and self.is_remote_connected():
            if hasattr(self.view, "prompt_download_training_artifacts"):
                folder = self.view.prompt_download_training_artifacts(list(result.artifact_uris.keys()))
            else:
                folder = None
            if folder:
                artifact_paths = {}
                for key, uri in result.artifact_uris.items():
                    if not uri:
                        continue
                    if key == "job_dir_uri":
                        continue
                    parsed = urlparse(uri)
                    name = os.path.basename(parsed.path) if parsed.scheme else os.path.basename(uri)
                    if not name:
                        continue
                    dest = os.path.join(folder, name)
                    try:
                        self.remote_service.download_file(uri, dest)
                        artifact_paths[key] = dest
                    except Exception as exc:
                        _logger.error(f"Failed to download {key}: {exc}")
                result.artifact_paths = artifact_paths
                # Register any downloaded model artifacts so inference can find them.
                try:
                    from cellpose import io as cellpose_io
                    for path in artifact_paths.values():
                        if path and os.path.isfile(path) and os.path.splitext(path)[1].lower() in (".pth", ".pt", ".bin"):
                            cellpose_io.add_model(path)
                            if self.model_service:
                                self.model_service.register_model(os.path.basename(path))
                    if self.model_service and artifact_paths:
                        # Prefer selecting the model artifact if present.
                        model_paths = [
                            p for p in artifact_paths.values()
                            if p and os.path.splitext(p)[1].lower() in (".pth", ".pt", ".bin")
                        ]
                        if model_paths:
                            self.refresh_model_list(select_model_name=os.path.basename(model_paths[0]))
                except Exception:
                    pass

    def _sanitize_remote_message(self, message: str) -> str:
        if not message:
            return message
        # Remove file URIs and obvious server paths from UI messages.
        try:
            import re
            msg = re.sub(r"file://\\?[^\\s]+", "[server artifact]", str(message))
            msg = re.sub(r"[A-Za-z]:\\\\[^\\s]+", "[server path]", msg)
            msg = msg.replace(".cellpose_server_data", "[server path]")
            return msg
        except Exception:
            return message

    def handle_training_error(self, error_message):
        _logger.error(f"Training failed: {error_message}")
        self.view.statusBar().showMessage(f"Training failed: {error_message}")

    def handle_cancel_training(self):
        if self.discovery_thread is not None and self.discovery_thread.isRunning():
            self._cleanup_discovery_thread()
        if self.thread is None or not self.thread.isRunning():
            self.view.statusBar().showMessage("No active training to cancel.")
            return
        if hasattr(self.worker, "request_cancel"):
            try:
                self.worker.request_cancel()
            except Exception:
                pass
        try:
            self.thread.terminate()
        except Exception:
            pass
        self.thread.wait(500)
        self.worker = None
        self.thread = None
        self._remote_training_active = False
        self._remote_train_start_ts = None
        self.view.statusBar().showMessage("Training canceled.")

    def _cleanup_discovery_thread(self):
        if self.discovery_thread is not None and self.discovery_thread.isRunning():
            try:
                self.discovery_thread.quit()
            except Exception:
                pass
            self.discovery_thread.wait(500)
        self.discovery_worker = None
        self.discovery_thread = None

    def cleanup_all_threads(self):
        self._cleanup_discovery_thread()
        super().cleanup_all_threads()

    def _on_training_thread_finished(self):
        self.worker = None
        self.thread = None
        self._remote_training_active = False
        self._remote_train_start_ts = None

    def _get_available_models_for_training(self):
        if self.is_remote_connected():
            if self.remote_service and hasattr(self.remote_service, "list_models"):
                try:
                    return self.remote_service.list_models()
                except Exception:
                    return []
            return []
        if self.model_service:
            return self.model_service.get_available_models()
        return []

    def _get_encoder_blocks_count(self, base_model):
        try:
            model = CellposeModel(pretrained_model=base_model, gpu=False)
            net = getattr(model, "net", None)
            if net and hasattr(net, "encoder") and hasattr(net.encoder, "blocks"):
                return len(net.encoder.blocks)
        except Exception:
            return None
        return None

    def _get_train_eta_seconds(self):
        settings = QSettings("GUVpose", "Trainer")
        return float(settings.value("remote_train/avg_seconds", 0.0))

    def _record_train_duration(self):
        if not self._remote_train_start_ts:
            return
        elapsed = max(0.0, time.time() - self._remote_train_start_ts)
        settings = QSettings("GUVpose", "Trainer")
        count = int(settings.value("remote_train/avg_count", 0))
        avg = float(settings.value("remote_train/avg_seconds", 0.0))
        new_avg = elapsed if count <= 0 else (avg * count + elapsed) / (count + 1)
        settings.setValue("remote_train/avg_seconds", new_avg)
        settings.setValue("remote_train/avg_count", count + 1)
