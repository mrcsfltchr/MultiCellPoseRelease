import logging
import sys
try:
    from PyQt6.QtCore import Qt, QThread, QObject
    from PyQt6.QtWidgets import QColorDialog, QFileDialog, QMessageBox
    from PyQt6 import sip
    import numpy as np
except ImportError:
    logging.error("PyQt6 not installed. Please install it with 'pip install PyQt6'")
    sys.exit(1)

import os
from cellpose import plot, transforms, utils
from guv_app.views.dialogs.ssh_dialog import SshLoginDialog, SshAdvancedDialog
from PyQt6.QtWidgets import QInputDialog
from guv_app.workers.inference_worker import InferenceWorker
from guv_app.workers.remote_inference_worker import RemoteInferenceWorker
from guv_app.workers.analysis_worker import AnalysisWorker
from guv_app.workers.remote_analysis_worker import RemoteAnalysisWorker
from guv_app.workers.statistics_worker import StatisticsWorker
from guv_app.workers.file_upload_worker import FileUploadWorker
from guv_app.workers.image_load_worker import ImageLoadWorker
from guv_app.data_models.configs import RemoteConfig, BatchConfig
from cpgrpc.server import cellpose_remote_pb2 as pb2
from cellpose import io

_logger = logging.getLogger(__name__)

class MainController(QObject):
    """
    The Main Controller for the application.
    """
    def __init__(self, model, view, services):
        """
        Initialize the controller and store references to the MVC components.
        """
        super().__init__()
        self.model = model
        self.view = view
        # Pass the model to the view so it can be updated
        self.view.set_model(self.model)
        
        self.image_service = services['image']
        self.segmentation_service = services['segmentation']
        self.remote_service = services.get('remote')
        self.model_service = services.get('model_management')
        self.analysis_service = services.get('analysis')
        # Assumes a dictionary of services is passed in
        
        # Worker thread state
        self.worker = None
        self.thread = None
        self.load_worker = None
        self.load_thread = None
        
        # Default mask load priority (Trainer/Base prefers ground truth)
        self.mask_load_priority = ['_seg.npy', '_pred.npy']

    def connect_signals(self):
        """
        Connect UI signals from the View to the controller's handler methods.
        """
        self.view.file_loaded.connect(self.handle_load_image)
        self.view.folder_selected.connect(self.handle_run_on_folder)
        self.view.navigate_next_requested.connect(self.handle_navigate_next)
        self.view.navigate_prev_requested.connect(self.handle_navigate_prev)
        self.view.connect_remote_requested.connect(self.handle_connect_remote_request)
        self.view.disconnect_remote_requested.connect(self.handle_disconnect_remote_request)
        self.view.add_ssh_hostname_requested.connect(self.handle_add_ssh_hostname)
        self.view.ssh_advanced_requested.connect(self.handle_ssh_advanced)
        self.view.upload_model_requested.connect(self.handle_upload_model)
        self.view.clear_remote_jobs_requested.connect(self.handle_clear_remote_jobs)
        self.view.control_panel.run_current_button.clicked.connect(self.handle_run_inference)
        self.view.control_panel.run_series_button.clicked.connect(self.handle_run_on_series)
        if hasattr(self.view.control_panel, "cancel_local_button"):
            self.view.control_panel.cancel_local_button.clicked.connect(self.handle_cancel_local_inference)
        self.view.toggle_masks_requested.connect(self.handle_toggle_masks_shortcut)
        self.view.toggle_outlines_requested.connect(self.handle_toggle_outlines_shortcut)
        self.view.toggle_color_mode_requested.connect(self.handle_toggle_color_mode_shortcut)
        self.view.toggle_visualization_requested.connect(self.handle_toggle_visualization_shortcut)
        self.view.brush_size_change_requested.connect(self.handle_brush_size_change)
        self.view.view_mode_step_requested.connect(self.handle_view_mode_step)
        self.view.color_mode_step_requested.connect(self.handle_color_mode_step)
        self.view.color_mode_set_requested.connect(self.handle_color_mode_set)
        self.view.finalize_stroke_requested.connect(self.handle_finalize_stroke)
        
        # Connect statistics button if it exists in the view
        if hasattr(self.view.control_panel, 'run_statistics_button'):
            self.view.control_panel.run_statistics_button.clicked.connect(self.handle_run_statistics)

        # Drawing and view toggles
        self.view.drawing_item.stroke_finished.connect(self.handle_add_mask_from_stroke)
        self.view.control_panel.masks_checkbox.stateChanged.connect(self.handle_toggle_masks)
        self.view.control_panel.outlines_checkbox.stateChanged.connect(self.handle_toggle_outlines)

        # Class management signals
        self.view.control_panel.add_class_button.clicked.connect(self.handle_add_new_class)
        self.view.control_panel.new_class_color_btn.clicked.connect(self.handle_pick_new_class_color)
        self.view.control_panel.class_dropdown.currentIndexChanged.connect(self.handle_set_current_class)
        self.view.control_panel.remove_class_button.clicked.connect(self.handle_remove_class)
        self.view.control_panel.color_by_class_checkbox.stateChanged.connect(self.handle_toggle_color_mode)
        self.view.control_panel.visualization_checkbox.stateChanged.connect(self.handle_toggle_visualization)
        self.view.control_panel.class_visibility_changed.connect(self.handle_class_visibility)
        self.view.drawing_item.assign_class_requested.connect(self.handle_assign_class)
        self.view.drawing_item.selection_rect_finished.connect(self.handle_select_masks_in_rect)
        self.view.drawing_item.clear_selection_requested.connect(self.handle_clear_selected_masks)
        self.view.drawing_item.delete_mask_requested.connect(self.handle_delete_mask)
        self.view.drawing_item.delete_stroke_finished.connect(self.handle_delete_masks_lasso)
        self.view.control_panel.delete_lasso_button.toggled.connect(self.handle_toggle_delete_lasso)

        # Saving signals
        self.view.save_requested.connect(self.handle_save_request)
        self.view.control_panel.autosave_checkbox.stateChanged.connect(self.handle_autosave_toggle)
        self.view.control_panel.model_dropdown.currentTextChanged.connect(self.handle_model_change)
        self.view.model_add_requested.connect(self.handle_add_custom_model)
        self.view.model_remove_requested.connect(self.handle_remove_custom_model)

        if self.model_service is not None and getattr(self.model_service, "models_updated", None) is not None:
            try:
                self.model_service.models_updated.connect(self._on_models_updated)
            except Exception:
                pass
        
        # Connect sliders if they exist
        if hasattr(self.view.control_panel, 'sliders'):
            for i, slider in enumerate(self.view.control_panel.sliders):
                slider.valueChanged.connect(lambda val, idx=i: self.handle_level_change(idx, val))
        if hasattr(self.view.control_panel, "view_mode_dropdown"):
            self.view.control_panel.view_mode_dropdown.currentIndexChanged.connect(
                self.handle_view_mode_change)
        if hasattr(self.view.control_panel, "color_mode_dropdown"):
            self.view.control_panel.color_mode_dropdown.currentIndexChanged.connect(
                self.handle_color_mode_change)

        # Initialize model list
        self.refresh_model_list()

    def is_remote_connected(self):
        """Checks if the remote service is available and connected."""
        return self.remote_service and self.remote_service.health_check()

    def refresh_model_list(self, state=None, select_model_name=None):
        """Refreshes the model dropdown based on local/remote state."""
        use_remote = self.is_remote_connected()
        if hasattr(self.view, "set_connection_mode"):
            self.view.set_connection_mode(use_remote)
        models_list = []
        
        if use_remote:
            if self.remote_service:
                try:
                    # Assuming remote_service has list_models method wrapping the gRPC call
                    if hasattr(self.remote_service, 'list_models'):
                        models_list = self.remote_service.list_models()
                    else:
                        _logger.warning("Remote service does not support listing models.")
                except Exception as e:
                    _logger.error(f"Failed to list remote models: {e}")
        else:
            if self.model_service:
                models_list = self.model_service.get_available_models()
        if select_model_name:
            if select_model_name not in models_list:
                models_list.append(select_model_name)
            current_index = models_list.index(select_model_name)
        else:
            current_index = 0

        self.view.control_panel.update_model_list(models_list, current_index)

    def _on_models_updated(self, model_name):
        if model_name:
            self.refresh_model_list(select_model_name=model_name)

    def handle_model_change(self, model_id):
        """Updates the active model ID in the state."""
        self.model.current_model_id = model_id
        _logger.info(f"Active model changed to: {model_id}")

    def handle_add_custom_model(self):
        if not self.model_service:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self.view,
            "Add model to GUI",
            "",
            "Model files (*.pth *.pt *.bin);;All files (*.*)",
        )
        if not filename:
            return
        model_name = self.model_service.add_model_from_path(filename)
        if model_name:
            self.refresh_model_list(select_model_name=model_name)
            self.view.statusBar().showMessage(f"Added custom model: {model_name}")
        else:
            self.view.statusBar().showMessage("Failed to add custom model.")

    def handle_remove_custom_model(self):
        if not self.model_service:
            return
        model_name = self.view.control_panel.model_dropdown.currentText()
        if not model_name:
            return
        resp = QMessageBox.question(
            self.view,
            "Remove model",
            f"Remove custom model '{model_name}' from the GUI list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        if self.model_service.remove_model(model_name):
            self.refresh_model_list()
            self.view.statusBar().showMessage(f"Removed custom model: {model_name}")
        else:
            self.view.statusBar().showMessage("Selected model is not a custom model.")

    def _trigger_autosave(self):
        if self.model.view_config.autosave_enabled:
            self.handle_save_request()

    def handle_save_request(self):
        """Handles a request to save the current segmentation."""
        _logger.info("Controller: Handling save request")
        self.image_service.save_segmentation(self.model)

    def handle_autosave_toggle(self, state):
        """Handles the autosave checkbox."""
        self.model.view_config.autosave_enabled = (Qt.CheckState(state) == Qt.CheckState.Checked)

    def handle_load_image(self, filename):
        """
        Handle the "Load Image" event from the view.
        """
        _logger.info(f"Controller: Handling image load for {filename}")
        if self.load_thread is not None and self.load_thread.isRunning():
            _logger.warning("Image load already running.")
            return
        base, frame_id = self.image_service.split_image_reference(filename)
        series_index = None
        if frame_id is None:
            series_key, series_count, time_count = self.image_service.get_series_time_info(base)
            if series_count > 1 and time_count > 1:
                label = series_key or "series"
                series_index = self.view.prompt_series_index(
                    series_count,
                    message=f"Select {label} series (0-{series_count - 1}):",
                )
                if series_index is None:
                    return
            elif time_count > 1 and series_count <= 1:
                series_index = 0
        self.load_worker = ImageLoadWorker(self.image_service, filename, series_index=series_index)
        self.load_thread = QThread()
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.loaded.connect(self._on_image_loaded)
        self.load_worker.error.connect(self._on_image_load_error)
        self.load_worker.loaded.connect(self.load_thread.quit)
        self.load_worker.error.connect(self.load_thread.quit)
        self.load_thread.finished.connect(self.load_worker.deleteLater)
        self.load_thread.finished.connect(self.load_thread.deleteLater)
        self.load_thread.start()
        self.view.set_progress_busy(True, "Loading image...")
        return

    def _cleanup_load_thread(self):
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.quit()
            self.load_thread.wait()
        self.load_worker = None
        self.load_thread = None

    def cleanup_all_threads(self):
        self._cleanup_load_thread()
        self._cleanup_thread()

    def _on_image_load_error(self, message):
        self.view.set_progress_busy(False)
        _logger.error(f"Controller: Failed to load image ({message})")
        self._cleanup_load_thread()

    def _on_image_loaded(self, image_data, filename, frame_id, frame_refs):
        self.view.set_progress_busy(False)
        self._cleanup_load_thread()
        if image_data is None:
            _logger.error(f"Controller: Failed to load image {filename}")
            return

        use_frame_refs = bool(frame_refs) and len(frame_refs) > 1
        if frame_id:
            parts = self.image_service.parse_frame_id(frame_id)
            series_index = parts.get("S", parts.get("P"))
            series_key, series_count, time_count = self.image_service.get_series_time_info(filename)
            refs_series_index = series_index
            if series_count > 1 and time_count <= 1:
                refs_series_index = None
                series_index = None
            frame_refs = self.image_service.build_frame_references(filename, series_index=refs_series_index)
            use_frame_refs = bool(frame_refs) and len(frame_refs) > 1
            if use_frame_refs:
                self.model.image_files = frame_refs
                ref = self.image_service.build_image_reference(filename, frame_id)
                try:
                    self.model.current_file_index = self.model.image_files.index(ref)
                except ValueError:
                    self.model.current_file_index = -1
            else:
                frame_refs = []
        if use_frame_refs:
            self.model.image_files = frame_refs
            ref = self.image_service.build_image_reference(filename, frame_id)
            try:
                self.model.current_file_index = self.model.image_files.index(ref)
            except ValueError:
                self.model.current_file_index = -1
        else:
            folder = os.path.dirname(filename)
            refresh_list = True
            if self.model.image_files:
                if os.path.dirname(self.model.image_files[0]) == folder:
                    refresh_list = False
            if refresh_list:
                try:
                    self.model.image_files = io.get_image_files(folder, '_masks')
                except Exception as e:
                    _logger.warning(f"Could not get image files for navigation: {e}")
                    self.model.image_files = []
            normalized_target = os.path.normcase(os.path.normpath(filename))
            normalized_files = [
                os.path.normcase(os.path.normpath(path)) for path in self.model.image_files
            ]
            try:
                self.model.current_file_index = normalized_files.index(normalized_target)
            except ValueError:
                self.model.current_file_index = -1

        normalized_data = self.image_service.normalize_image(image_data)
        if normalized_data is None:
            _logger.error(f"Controller: Unsupported image data for {filename}")
            return
        if hasattr(self.view, "control_panel"):
            if image_data.ndim == 3 and image_data.shape[2] > 1:
                channel_count = image_data.shape[2]
            else:
                channel_count = 1
            self.view.control_panel.update_inference_channel_options(channel_count)
            if channel_count > 1:
                current = int(self.model.view_config.channel_index)
                self.view.control_panel.inference_channel_dropdown.setCurrentIndex(
                    min(current + 1, channel_count)
                )
        self.model.frame_id = frame_id
        if frame_id:
            parts = self.image_service.parse_frame_id(frame_id)
            self.model.series_index = parts.get("S", parts.get("P"))
        self.model.update_image(normalized_data, filename)
        self.model.frame_refs = frame_refs if use_frame_refs else []
        if hasattr(self.view, "control_panel"):
            self.view.control_panel.run_series_button.setEnabled(bool(self.model.frame_refs))
            if hasattr(self.view, "run_plugin_series_button"):
                self.view.run_plugin_series_button.setEnabled(bool(self.model.frame_refs))
        if hasattr(self.view, "update_window_title"):
            self.view.update_window_title(filename, frame_id)
        if self.model.raw_image is not None and self.model.raw_image.ndim == 3:
            chan_count = self.model.raw_image.shape[2]
            if chan_count > 1:
                self.view.statusBar().showMessage(
                    f"Channel: {self.model.view_config.channel_index + 1}/{chan_count}"
                )
        self.init_sliders(normalized_data)
        self.refresh_class_list()
        self.handle_level_change(0, 0)
        self._autoload_masks(filename)

    def _autoload_masks(self, filename):
        """
        Automatically loads masks from files based on priority if they exist.
        """
        base = os.path.splitext(filename)[0]
        base_candidates = [base]
        try:
            from pathlib import Path
            path = Path(filename)
            if len(path.suffixes) > 1:
                trimmed = path
                for _ in range(len(path.suffixes)):
                    trimmed = trimmed.with_suffix("")
                    base_candidates.append(os.fspath(trimmed))
        except Exception:
            pass

        target_file = None
        frame_suffix = io.frame_id_to_suffix(getattr(self.model, "frame_id", None))
        require_frame_suffix = bool(frame_suffix)
        if require_frame_suffix:
            try:
                series_key, series_count, time_count = self.image_service.get_series_time_info(filename)
                if series_count <= 1 and time_count <= 1:
                    require_frame_suffix = False
            except Exception:
                pass
        for suffix in self.mask_load_priority:
            for candidate_base in base_candidates:
                candidates = []
                if frame_suffix:
                    candidates.append(candidate_base + frame_suffix + suffix)
                if not require_frame_suffix:
                    candidates.append(candidate_base + suffix)
                for candidate in candidates:
                    if os.path.exists(candidate):
                        target_file = candidate
                        break
                if target_file:
                    break
            if target_file:
                break
            
        if target_file:
            try:
                dat = np.load(target_file, allow_pickle=True).item()
                masks = dat.get('masks')
                if masks is not None:
                    classes = dat.get('classes')
                    # Handle dimensions similar to cellpose.gui.io
                    if masks.ndim == 3:
                        masks = masks.squeeze()
                    
                    # If outlines are present in the file, we could load them,
                    # but app_state.add_masks now auto-computes them to ensure consistency.
                    
                    # Ensure masks are at least 2D
                    if masks.ndim == 1:
                         # Handle edge case of flattened masks or empty
                         return

                    # Recover classes from classes_map if classes vector is missing/invalid.
                    if classes is None:
                        classes_map = dat.get("classes_map")
                        if classes_map is not None:
                            try:
                                masks_i64 = masks.astype(np.int64, copy=False)
                                classes_map = np.asarray(classes_map)
                                if classes_map.shape == masks.shape:
                                    nmask = int(masks_i64.max())
                                    recovered = np.zeros(nmask + 1, dtype=np.int16)
                                    for mid in range(1, nmask + 1):
                                        vals = classes_map[masks_i64 == mid]
                                        if vals.size:
                                            vals = vals.astype(np.int64, copy=False)
                                            vals = vals[vals >= 0]
                                            if vals.size:
                                                recovered[mid] = int(np.bincount(vals).argmax())
                                    classes = recovered
                            except Exception:
                                pass
                    
                    # Load class metadata if present
                    if 'class_names' in dat and dat['class_names'] is not None:
                        self.model.class_names = list(dat['class_names'])
                    if 'class_colors' in dat and dat['class_colors'] is not None:
                        self.model.class_colors = np.asarray(dat['class_colors'], dtype=np.uint8)
                        
                    success = self.model.add_masks(masks, classes=classes)
                    
                    if success:
                        msg = "predictions" if target_file.endswith('_pred.npy') else "segmentation"
                        self.view.statusBar().showMessage(f"Loaded {msg} from {os.path.basename(target_file)}")
                        self.refresh_class_list()
                    else:
                        self.view.statusBar().showMessage(f"Failed to load masks: Shape mismatch")
            except Exception as e:
                _logger.warning(f"Failed to autoload masks from {target_file}: {e}")
                self.view.statusBar().showMessage(f"Failed to load masks: {e}")

    def init_sliders(self, image):
        """Initializes slider ranges based on the loaded image."""
        if not hasattr(self.view.control_panel, 'sliders'):
            return
            
        # Determine if image is RGB or Grayscale
        is_rgb = image.ndim == 3 and image.shape[2] == 3
        
        # Block signals to prevent update loops during init
        for slider in self.view.control_panel.sliders:
            slider.blockSignals(True)
            
        if is_rgb:
            for i in range(3):
                if i == 0: self.view.control_panel.set_slider_label(0, "Red Channel")
                if i == 1: self.view.control_panel.set_slider_label(1, "Green Channel")
                if i == 2: self.view.control_panel.set_slider_label(2, "Blue Channel")
                chan = image[..., i]
                mn, mx = chan.min(), chan.max()
                # Scale to 0-255 for slider if float
                if image.dtype.kind == 'f':
                    self.view.control_panel.sliders[i].setRange(0, 1000)
                    self.view.control_panel.sliders[i].setValue((0, 1000))
                else:
                    self.view.control_panel.sliders[i].setRange(int(mn), int(mx))
                    self.view.control_panel.sliders[i].setValue((int(mn), int(mx)))
                self.view.control_panel.sliders[i].setEnabled(True)
        else:
            self.view.control_panel.set_slider_label(0, "Intensity")
            # Grayscale: use first slider, disable others
            mn, mx = image.min(), image.max()
            if image.dtype.kind == 'f':
                self.view.control_panel.sliders[0].setRange(0, 1000)
                self.view.control_panel.sliders[0].setValue((0, 1000))
            else:
                self.view.control_panel.sliders[0].setRange(int(mn), int(mx))
                self.view.control_panel.sliders[0].setValue((int(mn), int(mx)))
            
            self.view.control_panel.sliders[0].setEnabled(True)
            self.view.control_panel.sliders[1].setEnabled(False)
            self.view.control_panel.sliders[2].setEnabled(False)
            
        for slider in self.view.control_panel.sliders:
            slider.blockSignals(False)

    def handle_level_change(self, channel_idx, value):
        """Updates the image display based on slider values."""
        self._apply_display_modes()

    def handle_navigate_next(self):
        """Loads the next image in the current folder."""
        if self.model.image_files and self.model.current_file_index != -1:
            if self.model.current_file_index < len(self.model.image_files) - 1:
                next_file = self.model.image_files[self.model.current_file_index + 1]
                self.handle_load_image(next_file)
            else:
                self.view.statusBar().showMessage("End of folder reached.")

    def handle_navigate_prev(self):
        """Loads the previous image in the current folder."""
        if self.model.image_files and self.model.current_file_index != -1:
            if self.model.current_file_index > 0:
                prev_file = self.model.image_files[self.model.current_file_index - 1]
                self.handle_load_image(prev_file)
            else:
                self.view.statusBar().showMessage("Start of folder reached.")

    def handle_add_ssh_hostname(self):
        """
        Handle the "Add SSH Hostname" request from the view.
        """
        if not self.remote_service:
            _logger.warning("Remote service not available")
            return

        hostname, ok = QInputDialog.getText(self.view, "Add SSH Hostname", "Hostname:")
        if ok and hostname:
            self.remote_service._config.hostname = hostname
            _logger.info(f"SSH hostname set to: {hostname}")

    def handle_ssh_advanced(self):
        """
        Handle the "SSH Advanced" request from the view.
        """
        if not self.remote_service:
            _logger.warning("Remote service not available")
            return

        dialog = SshAdvancedDialog(self.view, config=self.remote_service._config)
        if dialog.exec():
            credentials = dialog.get_credentials()
            
            # Update config
            self.remote_service._config.hostname = credentials["host"]
            self.remote_service._config.username = credentials["username"]
            self.remote_service._config.password = credentials["password"]
            self.remote_service._config.ssh_port = credentials["port"]
            self.remote_service._config.key_path = credentials["key_path"]
            self.remote_service._config.ssh_local_port = credentials["ssh_local_port"]
            self.remote_service._config.ssh_remote_port = credentials["ssh_remote_port"]
            self.remote_service._config.ssh_remote_bind = credentials["ssh_remote_bind"]
            
            conn_creds = credentials.copy()
            conn_creds["use_key"] = bool(credentials["key_path"])
            
            if self.remote_service.connect(conn_creds):
                _logger.info("Connected to remote server")
                self.view.statusBar().showMessage("Connected to remote server")
                if hasattr(self.view, "set_connection_mode"):
                    self.view.set_connection_mode(True)
                self.refresh_model_list()
            else:
                _logger.error("Failed to connect to remote server")
                self.view.statusBar().showMessage("Failed to connect to remote server")
                if hasattr(self.view, "set_connection_mode"):
                    self.view.set_connection_mode(False)

    def handle_connect_remote_request(self):
        """
        Handle the "Connect to Remote" request from the view.
        """
        if not self.remote_service:
            _logger.warning("Remote service not available")
            return

        dialog = SshLoginDialog(self.view, config=self.remote_service._config)
        if dialog.exec():
            credentials = dialog.get_credentials()
            credentials["host"] = self.remote_service._config.hostname
            credentials["port"] = self.remote_service._config.ssh_port
            credentials["key_path"] = self.remote_service._config.key_path
            credentials["use_key"] = bool(credentials["key_path"])
            
            # SSH Tunneling parameters
            credentials["ssh_local_port"] = self.remote_service._config.ssh_local_port
            credentials["ssh_remote_port"] = self.remote_service._config.ssh_remote_port
            credentials["ssh_remote_bind"] = self.remote_service._config.ssh_remote_bind
            
            # Update config with entered credentials
            self.remote_service._config.username = credentials["username"]
            self.remote_service._config.password = credentials["password"]

            if self.remote_service.connect(credentials):
                _logger.info("Connected to remote server")
                self.view.statusBar().showMessage("Connected to remote server")
                if hasattr(self.view, "set_connection_mode"):
                    self.view.set_connection_mode(True)
                if hasattr(self.view, "set_activity_message"):
                    self.view.set_activity_message("Remote: connected")
                self.refresh_model_list()
            else:
                _logger.error("Failed to connect to remote server")
                self.view.statusBar().showMessage("Failed to connect to remote server")
                if hasattr(self.view, "set_connection_mode"):
                    self.view.set_connection_mode(False)
                if hasattr(self.view, "set_activity_message"):
                    self.view.set_activity_message("Remote: error")

    def handle_disconnect_remote_request(self):
        """
        Handle the "Disconnect from Remote" request from the view.
        """
        if not self.remote_service:
            _logger.warning("Remote service not available")
            return
        if self.thread is not None and self.thread.isRunning():
            self.view.statusBar().showMessage("Remote task running. Stop it before disconnecting.")
            return
        self.remote_service.disconnect()
        self.view.statusBar().showMessage("Disconnected from remote server.")
        if hasattr(self.view, "set_activity_message"):
            self.view.set_activity_message("Remote: disconnected")
        if hasattr(self.view, "set_connection_mode"):
            self.view.set_connection_mode(False)
        self.refresh_model_list()

    def handle_upload_model(self):
        """Handles uploading a custom model to the remote server."""
        if not self.is_remote_connected():
            self.view.statusBar().showMessage("Not connected to remote server.")
            return

        filename, _ = QFileDialog.getOpenFileName(self.view, "Select Model File")
        if filename:
            self.view.update_progress(0, f"Uploading model {os.path.basename(filename)}...")
            
            # Setup worker
            self.upload_worker = FileUploadWorker(self.remote_service, None, filename, is_model=True)
            self.thread = QThread()
            self.upload_worker.moveToThread(self.thread)
            
            # Connect signals
            self.thread.started.connect(self.upload_worker.run)
            self.upload_worker.progress.connect(lambda p: self.view.update_progress(p))
            self.upload_worker.finished.connect(self._on_upload_finished)
            self.upload_worker.error.connect(self._on_upload_error)
            
            self.thread.start()

    def _on_upload_finished(self):
        self.view.update_progress(0, "Model upload complete.")
        self.refresh_model_list()
        self._cleanup_thread()

    def _on_upload_error(self, error_message):
        self.view.update_progress(0, f"Upload failed: {error_message}")
        self._cleanup_thread()

    def handle_clear_remote_jobs(self):
        """Handles clearing remote training jobs."""
        if not self.is_remote_connected():
            self.view.statusBar().showMessage("Not connected to remote server.")
            return
            
        try:
            self.remote_service.clear_user_jobs()
            self.view.statusBar().showMessage("Remote training files cleared.")
        except Exception as e:
            self.view.statusBar().showMessage(f"Failed to clear remote jobs: {e}")

    def handle_run_inference(self):
        """
        Handle the "Run Inference" event from the view.
        """
        if self.is_remote_connected():
            self.handle_run_remote_inference()
        else:
            self.handle_run_local_inference()

    def handle_run_on_folder(self, folder_path):
        """
        Handle the "Run on Folder" event.
        """
        if self.thread is not None and self.thread.isRunning():
            _logger.warning("Analysis already running.")
            return

        _logger.info(f"Controller: Handling run on folder {folder_path}")
        diameter = self.view.control_panel.diameter_spinbox.value()
        channel_index = self.view.control_panel.get_inference_channel_index()
        use_remote = self.is_remote_connected()
        model_id = self.model.current_model_id
        self.view.set_progress_busy(True, "Running batch inference...")
        if use_remote:
            batch_config = BatchConfig()
            self.worker = RemoteAnalysisWorker(
                self.remote_service,
                self.image_service,
                folder_path,
                diameter,
                model_id,
                batch_config,
                channel_index=channel_index,
            )
        else:
            self.worker = AnalysisWorker(
                self.segmentation_service,
                self.image_service,
                folder_path,
                diameter,
                model_id,
                channel_index=channel_index,
            )
            if hasattr(self.view.control_panel, "cancel_local_button"):
                self.view.control_panel.cancel_local_button.setEnabled(True)
            
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self.model.handle_inference_result)
        self.worker.progress.connect(self.handle_analysis_progress)
        self.worker.finished.connect(self.handle_analysis_finished)
        if hasattr(self.worker, "canceled"):
            self.worker.canceled.connect(self.handle_local_inference_canceled)
            self.worker.canceled.connect(self._cleanup_thread)
        self.worker.finished.connect(self._cleanup_thread)
        self.thread.start()

    def handle_analysis_progress(self, percentage, message):
        _logger.info(f"Analysis progress: {percentage}% - {message}")
        self.view.update_progress(percentage, message)

    def handle_analysis_finished(self):
        _logger.info("Analysis finished")
        self.view.update_progress(100, "Analysis finished.")
        self.view.set_progress_busy(False)

    def handle_run_on_series(self):
        if self.thread is not None and self.thread.isRunning():
            _logger.warning("Analysis already running.")
            return
        filename = self.model.filename
        if not filename:
            self.view.statusBar().showMessage("No image loaded.")
            return
        if not getattr(self.model, "frame_refs", None):
            self.view.statusBar().showMessage("Current file has no multiple frames.")
            return
        diameter = self.view.control_panel.diameter_spinbox.value()
        channel_index = self.view.control_panel.get_inference_channel_index()
        model_id = self.model.current_model_id
        use_remote = self.is_remote_connected()
        series_index = self.model.series_index
        try:
            series_key, series_count, time_count = self.image_service.get_series_time_info(filename)
            if series_count > 1 and time_count <= 1:
                series_index = None
        except Exception:
            pass
        file_list = [filename]
        self.view.set_progress_busy(True, "Running series inference...")
        if use_remote:
            batch_config = BatchConfig()
            self.worker = RemoteAnalysisWorker(
                self.remote_service,
                self.image_service,
                "",
                diameter,
                model_id,
                batch_config,
                series_index=series_index,
                channel_index=channel_index,
                file_list=file_list,
            )
        else:
            self.worker = AnalysisWorker(
                self.segmentation_service,
                self.image_service,
                "",
                diameter,
                model_id,
                series_index=series_index,
                channel_index=channel_index,
                file_list=file_list,
            )
            if hasattr(self.view.control_panel, "cancel_local_button"):
                self.view.control_panel.cancel_local_button.setEnabled(True)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self.model.handle_inference_result)
        self.worker.progress.connect(self.handle_analysis_progress)
        self.worker.finished.connect(self.handle_analysis_finished)
        if hasattr(self.worker, "canceled"):
            self.worker.canceled.connect(self.handle_local_inference_canceled)
            self.worker.canceled.connect(self._cleanup_thread)
        self.worker.finished.connect(self._cleanup_thread)
        self.thread.start()

    def handle_run_statistics(self):
        """
        Handle the "Run Statistics" event.
        """
        _logger.info("Controller: Handling run statistics")
        
        if not self.analysis_service:
            _logger.warning("Analysis service not available")
            self.view.statusBar().showMessage("Analysis service not available.")
            return

        if not self.model.image_files:
            self.view.statusBar().showMessage("No images found to analyze.")
            return

        folder_path = os.path.dirname(self.model.image_files[0])
        series_index, cancelled = self._prompt_series_for_folder(folder_path)
        if cancelled:
            return
        
        # Instantiate plugins
        # In a full implementation, these might be selected via the UI
        from guv_app.plugins.perimeter_intensity import PerimeterIntensityPlugin
        plugins = [PerimeterIntensityPlugin()]
        
        plugin_params = {
            'thickness': 1 
        }

        self.stats_worker = StatisticsWorker(
            self.image_service, 
            self.analysis_service, 
            folder_path, 
            plugins=plugins, 
            plugin_params=plugin_params,
            series_index=series_index,
        )
        
        self.stats_thread = QThread()
        self.stats_worker.moveToThread(self.stats_thread)
        
        self.stats_thread.started.connect(self.stats_worker.run)
        self.stats_worker.progress.connect(lambda s: self.view.statusBar().showMessage(s))
        self.stats_worker.error.connect(lambda e: self.view.statusBar().showMessage(f"Error: {e}"))
        self.stats_worker.finished.connect(self.stats_thread.quit)
        self.stats_worker.finished.connect(self.stats_worker.deleteLater)
        self.stats_thread.finished.connect(self.stats_thread.deleteLater)
        
        self.stats_thread.start()

    def _prompt_series_for_folder(self, folder_path):
        try:
            files = io.get_image_files(folder_path, "_masks")
        except Exception:
            return None, False
        for path in files:
            ext = os.path.splitext(path)[1].lower()
            if ext not in (".nd2", ".lif"):
                continue
            series_key, series_count, time_count = self.image_service.get_series_time_info(path)
            if time_count > 1 and series_count > 1:
                choice = self.view.prompt_series_index(
                    series_count,
                    message=f"Select {series_key} series for timepoints (0-{series_count - 1}):",
                )
                if choice is None:
                    return None, True
                return choice, False
            if time_count > 1 and series_count <= 1:
                return 0, False
        return None, False

    def handle_run_local_inference(self):
        """
        Handle the "Run Local Inference" event.
        """
        if self.thread is not None and self.thread.isRunning():
            _logger.warning("Inference already running.")
            self.view.statusBar().showMessage("Inference already running.")
            return

        _logger.info("Controller: Handling run local inference")
        diameter = self.view.control_panel.diameter_spinbox.value()
        channel_index = self.view.control_panel.get_inference_channel_index()
        
        self.view.set_progress_busy(True, "Running local inference...")
        if hasattr(self.view.control_panel, "cancel_local_button"):
            self.view.control_panel.cancel_local_button.setEnabled(True)
        
        self.worker = InferenceWorker(
            self.segmentation_service,
            self.model.raw_image,
            diameter,
            self.model.current_model_id,
            channel_index=channel_index,
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.model.handle_inference_result)
        self.worker.canceled.connect(self.handle_local_inference_canceled)
        self.worker.canceled.connect(self._cleanup_thread)
        self.worker.finished.connect(self._cleanup_thread)
        self.thread.start()

    def handle_run_remote_inference(self):
        """
        Handle the "Run Remote Inference" event.
        """
        if self.thread is not None and self.thread.isRunning():
            _logger.warning("Inference already running.")
            self.view.statusBar().showMessage("Inference already running.")
            return

        if not self.is_remote_connected():
            _logger.warning("Remote service not available or not connected")
            self.view.statusBar().showMessage("Not connected to remote server.")
            return
        
        _logger.info("Controller: Handling run remote inference")
        
        diameter = self.view.control_panel.diameter_spinbox.value()
        channel_index = self.view.control_panel.get_inference_channel_index()
        model_id = self.model.current_model_id

        self.view.set_progress_busy(True, "Starting remote inference...")
        if hasattr(self.view, "set_activity_message"):
            self.view.set_activity_message("Remote: starting")

        # Start the inference worker (which now handles upload)
        self.worker = RemoteInferenceWorker(
            self.remote_service,
            self.image_service,
            self.model.filename,
            model_id,
            diameter,
            channel_index=channel_index,
            frame_id=self.model.frame_id,
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.handle_remote_progress)
        self.worker.error.connect(self._on_remote_inference_error)
        self.worker.finished.connect(self._on_remote_inference_finished)
        
        self.thread.start()

    def _cleanup_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait() # Ensure thread is finished
            
        self.worker = None
        self.thread = None
        self.view.set_progress_busy(False)
        self._set_cancel_button_enabled(False)

    def _on_remote_inference_finished(self, result):
        self.view.statusBar().showMessage("Inference complete. Loading results...")
        self.model.handle_inference_result(result)
        if hasattr(self.view, "set_activity_message"):
            self.view.set_activity_message("Remote: idle")
        self._cleanup_thread()

    def _on_remote_inference_error(self, error_message):
        self.view.update_progress(0, f"Remote Error: {error_message}")
        if hasattr(self.view, "set_activity_message"):
            self.view.set_activity_message("Remote: error")
        self._cleanup_thread()
        
    def handle_remote_progress(self, percentage, message):
        _logger.info(f"Remote progress: {percentage}% - {message}")
        self.view.update_progress(percentage, message)
        if hasattr(self.view, "set_activity_message"):
            self.view.set_activity_message(f"Remote: {message}")

    def handle_cancel_local_inference(self):
        if not self.worker or not self.thread or not self.thread.isRunning():
            return
        if not hasattr(self.worker, "request_cancel"):
            return
        self.worker.request_cancel()
        self._set_cancel_button_enabled(False)
        self.view.statusBar().showMessage("Cancel requested. Finishing current work...")

    def handle_local_inference_canceled(self):
        self.view.update_progress(0, "Local inference canceled.")
        self.view.statusBar().showMessage("Local inference canceled.")

    def _set_cancel_button_enabled(self, enabled):
        try:
            if sip.isdeleted(self.view) or sip.isdeleted(self.view.control_panel):
                return
        except Exception:
            pass
        try:
            button = getattr(self.view.control_panel, "cancel_local_button", None)
            if button is None:
                return
            if hasattr(sip, "isdeleted") and sip.isdeleted(button):
                return
            button.setEnabled(enabled)
        except Exception:
            pass

    def handle_toggle_masks(self, state):
        """Handles the masks visibility toggle."""
        self.model.view_config.masks_visible = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.model.trigger_view_update()

    def handle_toggle_outlines(self, state):
        """Handles the outlines visibility toggle."""
        self.model.view_config.outlines_visible = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.model.trigger_view_update()
        
    def handle_add_mask_from_stroke(self, points):
        """Handles the completion of a drawing stroke."""
        _logger.info(f"Controller: Stroke finished with {len(points)} points.")
        self.model.add_mask(points)
        self._trigger_autosave()

    def handle_pick_new_class_color(self):
        """Opens a color picker for the new class."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.view.control_panel.set_new_class_color(color.red(), color.green(), color.blue())

    def handle_add_new_class(self):
        """Adds a new class to the model."""
        max_classes = 4  # background + 3 object classes
        if len(self.model.class_names) >= max_classes:
            self.view.statusBar().showMessage(
                f"Max classes reached ({max_classes} including background)."
            )
            return
        new_name = self.view.control_panel.new_class_edit.text().strip()
        if not new_name:
            new_name = f"class {len(self.model.class_names) + 1}"
        
        # Get color and visibility from UI
        new_color_list = self.view.control_panel.current_new_class_color
        is_visible = self.view.control_panel.new_class_visible_chk.isChecked()
        
        self.model.class_names.append(new_name)
        new_color = np.array(new_color_list, dtype=np.uint8)
        self.model.class_colors = np.vstack([self.model.class_colors, new_color])
        self.model.view_config.class_visible.append(is_visible)
        self.view.control_panel.new_class_edit.clear()
        
        self.model.trigger_view_update()
        self.refresh_class_list()
        
    def handle_set_current_class(self, index):
        """Sets the current class for drawing."""
        if index >= 0:
            self.model.current_class = index + 1 # class IDs are 1-based

    def handle_remove_class(self):
        """Removes the currently selected class from the model."""
        class_id_to_remove = self.model.current_class
        self.model.remove_class(class_id_to_remove)
        self._trigger_autosave()
        self.refresh_class_list()

    def handle_toggle_color_mode(self, state):
        """Toggles the mask coloring mode."""
        self.model.view_config.color_by_class = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.model.trigger_view_update()

    def handle_toggle_visualization(self, state):
        """Toggles the visualization layer visibility."""
        self.model.view_config.show_visualization = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.model.trigger_view_update()

    def handle_toggle_masks_shortcut(self):
        self.view.control_panel.masks_checkbox.toggle()

    def handle_toggle_outlines_shortcut(self):
        self.view.control_panel.outlines_checkbox.toggle()

    def handle_toggle_color_mode_shortcut(self):
        self.view.control_panel.color_by_class_checkbox.toggle()

    def handle_toggle_visualization_shortcut(self):
        self.view.control_panel.visualization_checkbox.toggle()

    def handle_brush_size_change(self, delta):
        if not hasattr(self.view, "drawing_item"):
            return
        current = self.view.drawing_item.brush_size
        new_size = max(1, min(50, current + delta))
        if new_size == current:
            return
        self.view.drawing_item.set_brush_size(new_size)
        self.view.statusBar().showMessage(f"Brush size: {new_size}")

    def handle_view_mode_change(self, index):
        self.model.view_config.view_mode = index
        self._apply_display_modes()

    def handle_color_mode_change(self, index):
        img = self.model.raw_image
        if img is not None and img.ndim == 3 and img.shape[2] not in (1, 3):
            chan_count = img.shape[2]
            self._set_channel_index(index % chan_count)
            return
        self.model.view_config.color_mode = index
        self._apply_display_modes()

    def handle_view_mode_step(self, delta):
        if hasattr(self.view.control_panel, "view_mode_dropdown"):
            dropdown = self.view.control_panel.view_mode_dropdown
            count = dropdown.count()
            if count == 0:
                return
            dropdown.setCurrentIndex((dropdown.currentIndex() + delta) % count)
        else:
            self.model.view_config.view_mode = max(0, self.model.view_config.view_mode + delta)
            self._apply_display_modes()

    def handle_color_mode_step(self, delta):
        if self._handle_channel_step(delta):
            return
        if hasattr(self.view.control_panel, "color_mode_dropdown"):
            dropdown = self.view.control_panel.color_mode_dropdown
            count = dropdown.count()
            if count == 0:
                return
            dropdown.setCurrentIndex((dropdown.currentIndex() + delta) % count)
        else:
            self.model.view_config.color_mode = max(0, self.model.view_config.color_mode + delta)
            self._apply_display_modes()

    def handle_color_mode_set(self, color_index):
        if self._handle_channel_step(color_index):
            return
        current = self.model.view_config.color_mode
        target = 0 if current == color_index else color_index
        if hasattr(self.view.control_panel, "color_mode_dropdown"):
            self.view.control_panel.color_mode_dropdown.setCurrentIndex(target)
        else:
            self.model.view_config.color_mode = target
            self._apply_display_modes()

    def handle_finalize_stroke(self):
        if hasattr(self.view, "drawing_item") and self.view.drawing_item.in_stroke:
            self.view.drawing_item.end_stroke()

    def _apply_display_modes(self):
        if self.model.raw_image is None:
            return

        view_mode = self.model.view_config.view_mode
        if view_mode != 0:
            flow_view = self._render_flow_view(view_mode)
            if flow_view is not None:
                self.model.image_data = flow_view
                self.model.trigger_view_update()
                return
            self.model.view_config.view_mode = 0
            if hasattr(self.view.control_panel, "view_mode_dropdown"):
                self.view.control_panel.view_mode_dropdown.setCurrentIndex(0)

        img = self._apply_levels(self.model.raw_image)
        img = self._apply_color_mode(img, self.model.view_config.color_mode)
        self.model.image_data = self._to_uint8(img)
        self.model.trigger_view_update()

    def _apply_levels(self, image):
        img = image.copy()
        is_rgb = img.ndim == 3 and img.shape[2] == 3
        if not hasattr(self.view.control_panel, "sliders"):
            return np.clip(img, 0, 1)

        sliders = self.view.control_panel.sliders
        if is_rgb:
            for c in range(3):
                mn, mx = sliders[c].value()
                if img.dtype.kind == "f":
                    mn /= 1000.0
                    mx /= 1000.0
                rng = max(mx - mn, 1e-5)
                img[..., c] = (img[..., c] - mn) / rng
                img[..., c] = np.clip(img[..., c], 0, 1)
        else:
            mn, mx = sliders[0].value()
            if img.dtype.kind == "f":
                mn /= 1000.0
                mx /= 1000.0
            rng = max(mx - mn, 1e-5)
            img = (img - mn) / rng
            img = np.clip(img, 0, 1)

        return img

    def _apply_color_mode(self, image, color_mode):
        img = image
        is_rgb = img.ndim == 3 and img.shape[2] == 3
        is_multichannel = img.ndim == 3 and img.shape[2] not in (1, 3)
        if is_multichannel:
            chan_idx = int(self.model.view_config.channel_index)
            chan_idx = max(0, min(chan_idx, img.shape[2] - 1))
            return img[..., chan_idx]

        if color_mode == 0:
            return img
        if color_mode in (1, 2, 3):
            channel_idx = color_mode - 1
            if img.ndim == 3 and img.shape[2] > 1:
                channel_idx = min(channel_idx, img.shape[2] - 1)
                channel = img[..., channel_idx]
            else:
                channel = img
            rgb = np.zeros((*channel.shape, 3), dtype=channel.dtype)
            rgb[..., channel_idx] = channel
            return rgb
        if color_mode == 4:
            return img.mean(axis=-1) if is_rgb else img
        if color_mode == 5:
            gray = img.mean(axis=-1) if is_rgb else img
            gray = np.clip(gray, 0, 1)
            hsv = np.zeros((*gray.shape, 3), dtype=np.float32)
            hsv[..., 0] = gray * 0.75
            hsv[..., 1] = 1.0
            hsv[..., 2] = np.clip(gray * 1.1, 0, 1)
            return utils.hsv_to_rgb(hsv)

        return img

    def _handle_channel_step(self, delta):
        img = self.model.raw_image
        if img is None:
            return False
        if img.ndim == 3 and img.shape[2] not in (1, 3):
            chan_count = img.shape[2]
            idx = int(self.model.view_config.channel_index)
            if abs(delta) <= 3:
                new_idx = idx + delta
            else:
                new_idx = delta
            new_idx = max(0, min(new_idx, chan_count - 1))
            if new_idx != idx:
                self._set_channel_index(new_idx)
            return True
        return False

    def _set_channel_index(self, new_idx):
        img = self.model.raw_image
        if img is None or img.ndim != 3:
            return
        chan_count = img.shape[2]
        new_idx = max(0, min(int(new_idx), chan_count - 1))
        self.model.view_config.channel_index = new_idx
        self._apply_display_modes()
        self.view.statusBar().showMessage(f"Channel: {new_idx + 1}/{chan_count}")

    def _render_flow_view(self, view_mode):
        flows = self.model.flows
        if not flows or len(flows) < 2:
            return None

        if view_mode == 1:
            flow_img = flows[0]
            if flow_img is None:
                return None
            if flow_img.ndim == 4 and flow_img.shape[0] == 1:
                flow_img = flow_img[0]
            if flow_img.ndim == 3 and flow_img.shape[-1] == 3:
                return flow_img.astype(np.uint8)
            dP = flows[1] if len(flows) > 1 else None
            if dP is None:
                return None
            if dP.ndim == 4 and dP.shape[0] == 1:
                dP = dP[0]
            if dP.ndim == 3 and dP.shape[0] == 2:
                return plot.dx_to_circ(dP)
            return None

        if view_mode == 2:
            cellprob = flows[2] if len(flows) > 2 else None
            if cellprob is None:
                return None
            if cellprob.ndim == 3:
                cellprob = cellprob[0]
            cellprob = transforms.normalize99(cellprob)
            cellprob = np.clip(cellprob, 0, 1)
            return (cellprob * 255).astype(np.uint8)

        return None

    def _to_uint8(self, image):
        if image.dtype.kind == "f":
            image = np.clip(image, 0, 1)
            return (image * 255).astype(np.uint8)
        if image.dtype == np.uint8:
            return image
        image = image.astype(np.float32)
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def handle_class_visibility(self, class_index, is_visible):
        """Handles toggling the visibility of a specific class."""
        if class_index < len(self.model.view_config.class_visible):
            self.model.view_config.class_visible[class_index] = is_visible
            self.model.trigger_view_update()

    def handle_assign_class(self, y, x):
        """Handles a request to assign the current class to a clicked mask."""
        if self.model.cellpix is None:
            return
        
        # Check bounds and get mask ID (assuming 2D for now)
        if 0 <= y < self.model.Ly and 0 <= x < self.model.Lx:
            mask_id = self.model.cellpix[0, y, x]
            if mask_id > 0:
                selected_ids = self.model.get_selected_mask_ids()
                if selected_ids and mask_id in selected_ids:
                    for mid in selected_ids:
                        self.model.assign_class_to_mask(mid, self.model.current_class)
                else:
                    self.model.assign_class_to_mask(mask_id, self.model.current_class)
                self.model.trigger_view_update()
                if selected_ids and mask_id in selected_ids:
                    self.model.clear_selected_masks()
                self._trigger_autosave()

    def handle_select_masks_in_rect(self, rect):
        y0, x0, y1, x1 = rect
        count = self.model.select_masks_in_rect(y0, x0, y1, x1)
        if count <= 0:
            self.view.statusBar().showMessage("No masks selected.")
        else:
            self.view.statusBar().showMessage(f"Selected {count} masks. Shift+click to assign class.")
        self.model.trigger_view_update()

    def handle_clear_selected_masks(self):
        self.model.clear_selected_masks()
        self.model.trigger_view_update()

    def handle_delete_mask(self, y, x):
        """Handles a request to delete a mask at a specific point."""
        if self.model.view_config.show_visualization and self.model.visualization_masks is not None:
            self.model.remove_visualization_mask_at_point(y, x)
            return
        self.model.remove_mask_at_point(y, x)
        self._trigger_autosave()

    def handle_delete_masks_lasso(self, points):
        if not points:
            return
        if self.model.view_config.show_visualization and self.model.visualization_masks is not None:
            removed = self.model.remove_visualization_masks_in_polygon(points)
        else:
            removed = self.model.remove_masks_in_polygon(points)
        if removed:
            self._trigger_autosave()
            self.view.statusBar().showMessage(f"Deleted {removed} mask(s)")

    def handle_toggle_delete_lasso(self, enabled):
        if hasattr(self.view, "drawing_item"):
            self.view.drawing_item.set_delete_mode(enabled)
        if enabled:
            self.view.statusBar().showMessage("Delete lasso mode enabled")
        else:
            self.view.statusBar().showMessage("Delete lasso mode disabled")

    def refresh_class_list(self):
        """Updates the class dropdown and visibility UI in the view."""
        class_names = self.model.class_names
        if len(self.model.view_config.class_visible) < len(class_names):
            missing = len(class_names) - len(self.model.view_config.class_visible)
            self.model.view_config.class_visible.extend([True] * missing)
        # current_class is 1-based, dropdown is 0-based
        current_idx = max(0, self.model.current_class - 1)
        if current_idx >= len(class_names):
            current_idx = len(class_names) - 1
        self.view.control_panel.update_class_list(class_names, current_idx)
        self.view.control_panel.update_class_visibility_ui(class_names, self.model.view_config.class_visible)
