from PyQt6.QtCore import pyqtSlot, QThread
import numpy as np
import pandas as pd
from guv_app.controllers.main_controller import MainController
from guv_app.workers.analysis_worker import AnalysisWorker
from guv_app.workers.remote_analysis_worker import RemoteAnalysisWorker
from guv_app.workers.promote_worker import PromoteWorker
from guv_app.workers.statistics_worker import StatisticsWorker
from guv_app.services.analysis_service import AnalysisService
from guv_app.plugins.interface import AnalysisPlugin
from guv_app.data_models.configs import BatchConfig
from cellpose import io
import os

class AnalyzerController(MainController):
    def __init__(self, model, view, services):
        super().__init__(model, view, services)
        self.worker = None
        self.thread = None
        self.active_plugin = None
        self.active_plugin_params = {}
        self.pending_folder_path = None
        self.pending_folder_plugin_params = None
        self.pending_series_index = None
        self.pending_series_file = None
        self.pending_visualization_generation = False
        self.visualization_masks_by_file = {}
        # Analyzer prefers predictions
        self.mask_load_priority = ['_pred.npy', '_seg.npy']

    def connect_signals(self):
        super().connect_signals()
        if hasattr(self.view, "batch_folder_selected"):
            self.view.batch_folder_selected.connect(self.on_folder_selected)
        else:
            self.view.folder_selected.connect(self.on_folder_selected)
        self.view.start_analysis.connect(self.on_start_analysis)
        self.view.promote_requested.connect(self.handle_promote_request)
        self.view.export_csv_requested.connect(self.on_export_csv)
        self.view.run_plugin_requested.connect(self.on_run_plugin_visualization)
        if hasattr(self.view, "run_plugin_series_requested"):
            self.view.run_plugin_series_requested.connect(self.on_run_plugin_series)
        self.view.finalize_plugin_requested.connect(self.on_finalize_plugin_analysis)

    @pyqtSlot(str)
    def on_folder_selected(self, folder_path):
        # self.model.folder_path = folder_path # Store in model if needed
        self.view.set_folder_path(folder_path)
        
        # Load preview of first image
        try:
            self.model.image_files = io.get_image_files(folder_path, '_masks')
            self.model.current_file_index = -1
            if self.model.image_files:
                self.handle_load_image(self.model.image_files[0])
        except Exception as e:
            self.view.show_progress(f"Error loading folder preview: {e}")

    @pyqtSlot()
    def on_start_analysis(self):
        if self.thread is not None and self.thread.isRunning():
            self.view.show_progress("Analysis already running.")
            return

        folder_path = self.view.folder_label.text()
        if folder_path and os.path.isdir(folder_path):
            model_id = self.model.current_model_id
            
            # Get settings from the shared ControlPanel
            diameter = self.view.control_panel.diameter_spinbox.value()
            use_remote = self.is_remote_connected()
            channel_index = self.view.control_panel.get_inference_channel_index()
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
            
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.result_ready.connect(self.model.handle_inference_result)
            self.worker.progress.connect(self.handle_analysis_progress)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self._on_thread_finished)
            self.thread.finished.connect(self.thread.deleteLater)
            
            self.thread.start()
            self.view.set_analysis_running(True)
            self.view.set_progress_busy(True, "Running batch analysis...")

    def on_export_csv(self):
        """Calculates statistics for all masks in the folder and exports to CSV."""
        if self.thread is not None and self.thread.isRunning():
            self.view.show_progress("Analysis/Statistics already running.")
            return

        folder_path = self.view.folder_label.text()
        if not folder_path or not os.path.isdir(folder_path):
            self.view.show_progress("Invalid folder path.")
            return
        series_index, cancelled = self._prompt_series_for_folder(folder_path)
        if cancelled:
            return

        if self.analysis_service is None:
            self.analysis_service = AnalysisService()
        else:
            self.analysis_service.discover_plugins()

        if not self.analysis_service.plugins:
            self.view.show_progress("No analysis plugins available.")
            return

        plugin = self.view.prompt_plugin_selection(self.analysis_service.plugins)
        if plugin is None:
            return

        plugin_params = {}
        param_defs = plugin.get_parameter_definitions()
        if param_defs:
            params = self.view.prompt_plugin_parameters(plugin)
            if params is None:
                return
            plugin_params[plugin.name] = params

        if self._plugin_supports_visualization(plugin):
            if self.model.image_data is None or self.model.masks is None:
                self.view.show_progress("Load an image with masks to preview plugin visualization.")
                return

            self.active_plugin = plugin
            self.active_plugin_params = plugin_params.get(plugin.name, {})
            self.pending_folder_path = folder_path
            self.pending_folder_plugin_params = plugin_params
            self.pending_series_index = series_index
            self.pending_series_file = None
            self.visualization_masks_by_file = {}
            self.pending_visualization_generation = True
            self._start_statistics_worker(
                folder_path,
                plugins=[plugin],
                plugin_params=plugin_params,
                visualization_masks_by_file=None,
                series_index=series_index,
                visualize_only=True,
            )
            self.model.view_config.show_visualization = True
            self.view.control_panel.visualization_checkbox.setChecked(True)
            if hasattr(self.view, "set_plugin_hint_visible"):
                self.view.set_plugin_hint_visible(True)
            self.view.show_progress(
                f"{plugin.name} visualization generating for folder. Navigate images to review, then press Finalize Plugin Analysis."
            )
            return

        self.pending_folder_path = None
        self.pending_folder_plugin_params = None
        self._start_statistics_worker(
            folder_path,
            plugins=[plugin],
            plugin_params=plugin_params,
            visualization_masks_by_file=None,
            series_index=series_index,
        )

    def _on_thread_finished(self):
        self.view.set_analysis_running(False)
        self.view.set_progress_busy(False)
        self.worker = None
        self.thread = None
        if self.pending_visualization_generation:
            self.pending_visualization_generation = False
            self._prepare_visualization_for_current_image()

    def _store_visualization_for_current_file(self):
        if not self.model.filename or self.model.visualization_masks is None:
            return
        ref = self.image_service.build_image_reference(self.model.filename, self.model.frame_id)
        normalized = os.path.normcase(os.path.normpath(ref))
        self.visualization_masks_by_file[normalized] = np.array(self.model.visualization_masks, copy=True)
        plugin_name = self.active_plugin.name if self.active_plugin else None
        self.image_service.save_visualization_mask(
            self.model.filename,
            self.model.frame_id,
            self.model.visualization_masks,
            plugin_name=plugin_name,
        )

    def _prepare_visualization_for_current_image(self):
        if (not self.pending_folder_path and not self.pending_series_file) or self.active_plugin is None:
            return
        if self.model.image_data is None or self.model.masks is None:
            return
        if self.pending_series_file and self.model.filename != self.pending_series_file:
            return
        if self.analysis_service is None:
            self.analysis_service = AnalysisService()

        ref = self.image_service.build_image_reference(self.model.filename or "", self.model.frame_id)
        normalized = os.path.normcase(os.path.normpath(ref))
        if normalized in self.visualization_masks_by_file:
            self.model.set_visualization(self.visualization_masks_by_file[normalized])
            self.model.view_config.show_visualization = True
            self.view.control_panel.visualization_checkbox.setChecked(True)
            return

        plugin_name = self.active_plugin.name if self.active_plugin else None
        stored = self.image_service.load_visualization_mask(
            self.model.filename,
            self.model.frame_id,
            plugin_name=plugin_name,
        )
        if stored is not None:
            self.model.set_visualization(stored)
            self.visualization_masks_by_file[normalized] = np.array(stored, copy=True)
            self.model.view_config.show_visualization = True
            self.view.control_panel.visualization_checkbox.setChecked(True)
            return

        viz_mask = self.analysis_service.run_visualization(
            self.active_plugin,
            self.model.image_data,
            self.model.masks,
            classes=self.model.classes,
            plugin_params=self.active_plugin_params,
        )
        if viz_mask is None:
            return
        self.model.set_visualization(viz_mask)
        self._store_visualization_for_current_file()
        self.model.view_config.show_visualization = True
        self.view.control_panel.visualization_checkbox.setChecked(True)

    def _plugin_supports_visualization(self, plugin):
        try:
            return plugin.visualize.__func__ is not AnalysisPlugin.visualize
        except AttributeError:
            return False

    def _start_statistics_worker(self, folder_path, plugins, plugin_params, visualization_masks_by_file, series_index=None,
                                 visualize_only=False, image_files=None):
        self.worker = StatisticsWorker(
            self.image_service,
            self.analysis_service,
            folder_path,
            plugins=plugins,
            plugin_params=plugin_params,
            mask_suffix="_pred.npy",
            visualization_masks_by_file=visualization_masks_by_file,
            series_index=series_index,
            visualize_only=visualize_only,
            image_files=image_files,
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.view.show_progress)
        self.worker.error.connect(lambda e: self.view.show_progress(f"Error: {e}"))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.view.set_analysis_running(True)

    def on_run_plugin_visualization(self):
        """
        Runs a selected plugin on the currently loaded image and updates the view with the result.
        """
        if self.model.image_data is None:
            self.view.show_progress("No image loaded.")
            return
            
        if self.model.masks is None:
            self.view.show_progress("No masks available on current image.")
            return

        # Ensure service is ready
        if self.analysis_service is None:
            self.analysis_service = AnalysisService()

        # 1. Select Plugin
        selected_plugins, plugin_params = self.view.prompt_plugin_configuration(self.analysis_service.plugins)
        if not selected_plugins:
            return

        # For visualization, we typically only want to visualize one result at a time.
        # We'll take the first selected plugin.
        plugin = selected_plugins[0]
        params = plugin_params.get(plugin.name, {})
        self.active_plugin = plugin
        self.active_plugin_params = params

        self.view.show_progress(f"Running {plugin.name} on current image...")

        # 2. Run Analysis (Calculation) & Save
        try:
            # Use service to handle defaults and execution
            results = self.analysis_service.run_analysis(
                self.model.image_data, 
                self.model.masks, 
                classes=self.model.classes,
                filename=os.path.basename(self.model.filename) if self.model.filename else None,
                plugins=[plugin],
                plugin_params={plugin.name: params}
            )
            
            # Save results using the service
            if self.model.filename:
                saved_files = self.analysis_service.save_results(results, self.model.filename)
                for path in saved_files:
                    self.view.show_progress(f"Saved results to {os.path.basename(path)}")
            else:
                self.view.show_progress("Results calculated but not saved (no filename).")

        except Exception as e:
            self.view.show_progress(f"Error running analysis: {e}")

        # 3. Run Visualization
        try:
            viz_mask = self.analysis_service.run_visualization(
                plugin, self.model.image_data, self.model.masks, 
                classes=self.model.classes, plugin_params=params
            )
            
            if viz_mask is not None:
                # 4. Update View (by updating model)
                # We set the visualization masks (temporary, not saved)
                self.model.set_visualization(viz_mask)
                self.model.view_config.show_visualization = True
                self.view.control_panel.visualization_checkbox.setChecked(True)
                if hasattr(self.view, "set_plugin_hint_visible"):
                    self.view.set_plugin_hint_visible(True)
                self.view.show_progress(f"Visualization applied. Edit masks or reload image to restore original view.")
            else:
                self.view.show_progress(f"Plugin {plugin.name} does not support visualization.")
                
        except Exception as e:
            self.view.show_progress(f"Error running visualization: {e}")

    def on_run_plugin_series(self):
        if self.model.filename is None:
            self.view.show_progress("No image loaded.")
            return
        if not getattr(self.model, "frame_refs", None):
            self.view.show_progress("Current file has no multiple frames.")
            return
        if self.analysis_service is None:
            self.analysis_service = AnalysisService()
        selected_plugins, plugin_params = self.view.prompt_plugin_configuration(
            self.analysis_service.plugins
        )
        if not selected_plugins:
            return
        base_file = self.model.filename
        series_index = getattr(self.model, "series_index", None)
        plugin = selected_plugins[0]
        if self._plugin_supports_visualization(plugin):
            self.active_plugin = plugin
            self.active_plugin_params = plugin_params.get(plugin.name, {})
            self.pending_series_file = base_file
            self.pending_series_index = series_index
            self.pending_folder_path = None
            self.pending_folder_plugin_params = None
            self.visualization_masks_by_file = {}
            self.pending_visualization_generation = True
            self._start_statistics_worker(
                os.path.dirname(base_file),
                plugins=[plugin],
                plugin_params=plugin_params,
                visualization_masks_by_file=None,
                series_index=series_index,
                visualize_only=True,
                image_files=[base_file],
            )
            self.model.view_config.show_visualization = True
            self.view.control_panel.visualization_checkbox.setChecked(True)
            if hasattr(self.view, "set_plugin_hint_visible"):
                self.view.set_plugin_hint_visible(True)
            self.view.show_progress(
                f"{plugin.name} visualization generating for series. Navigate frames to review, then press Finalize Plugin Analysis."
            )
            return
        self.pending_folder_path = None
        self.pending_folder_plugin_params = None
        self._start_statistics_worker(
            os.path.dirname(base_file),
            plugins=selected_plugins,
            plugin_params=plugin_params,
            visualization_masks_by_file=None,
            series_index=series_index,
            visualize_only=False,
            image_files=[base_file],
        )
        self.view.show_progress(
            f"{plugin.name} analysis running for series in background."
        )

    def on_finalize_plugin_analysis(self):
        """
        Runs the active plugin analysis on the current image using the edited visualization mask.
        """
        if self.pending_series_file:
            plugin = self.active_plugin
            if plugin is None:
                self.view.show_progress("No active plugin selected.")
                return
            base_file = self.pending_series_file
            series_index = self.pending_series_index
            selected_plugins = [plugin]
            plugin_params = {plugin.name: dict(self.active_plugin_params or {})}
            frames = self.image_service.iter_image_frames(base_file, series_index=series_index)
            if not frames:
                self.view.show_progress("No frames found for this series.")
                return
            combined = {}
            for frame in frames:
                image = frame.array
                if image is None:
                    continue
                base = os.path.splitext(base_file)[0]
                frame_suffix = io.frame_id_to_suffix(frame.frame_id)
                pred_path = base + frame_suffix + "_pred.npy"
                seg_path = base + frame_suffix + "_seg.npy"
                masks = None
                classes = None
                if os.path.exists(pred_path):
                    dat = np.load(pred_path, allow_pickle=True).item()
                    masks = dat.get("masks")
                    classes = dat.get("classes")
                elif os.path.exists(seg_path):
                    dat = np.load(seg_path, allow_pickle=True).item()
                    masks = dat.get("masks")
                    classes = dat.get("classes")
                if masks is None:
                    continue
                viz_mask = self.image_service.load_visualization_mask(
                    base_file,
                    frame.frame_id,
                    plugin_name=plugin.name,
                )
                if viz_mask is not None:
                    plugin_params[plugin.name]["visualization_masks"] = viz_mask
                frame_name = os.path.basename(base_file)
                if frame.frame_id:
                    frame_name = f"{frame_name}::{frame.frame_id}"
                results = self.analysis_service.run_analysis(
                    image,
                    masks,
                    classes=classes,
                    filename=frame_name,
                    plugins=selected_plugins,
                    plugin_params=plugin_params,
                )
                for plugin_name, df in results.items():
                    if df is None or df.empty:
                        continue
                    combined.setdefault(plugin_name, []).append(df)
            if not combined:
                self.view.show_progress("No plugin results produced for this series.")
                return
            for plugin_name, frames_df in combined.items():
                merged = pd.concat(frames_df, ignore_index=True)
                safe_name = "".join(
                    x for x in plugin_name if x.isalnum() or x in "._- "
                ).replace(" ", "_")
                series_suffix = ""
                if series_index is not None:
                    series_key = "S"
                    try:
                        key, _, _ = self.image_service.get_series_time_info(base_file)
                        if key:
                            series_key = key
                    except Exception:
                        pass
                    series_suffix = f"__{series_key}{series_index}"
                out_path = f"{os.path.splitext(base_file)[0]}__series{series_suffix}_{safe_name}.csv"
                try:
                    merged.to_csv(out_path, index=False)
                    self.view.show_progress(f"Saved {os.path.basename(out_path)}")
                except Exception as exc:
                    self.view.show_progress(f"Failed to save {safe_name} CSV: {exc}")
            if hasattr(self.view, "set_plugin_hint_visible"):
                self.view.set_plugin_hint_visible(False)
            self.pending_series_file = None
            self.pending_series_index = None
            return
        if self.pending_folder_path:
            if self.model.visualization_masks is None:
                self.view.show_progress("No plugin visualization mask to finalize.")
                return
            if self.active_plugin is None:
                self.view.show_progress("No active plugin selected.")
                return

            folder_path = self.pending_folder_path
            plugin = self.active_plugin
            plugin_params = dict(self.pending_folder_plugin_params or {})

            if not self.model.filename:
                self.view.show_progress("No current filename to apply visualization override.")
                return

            self._store_visualization_for_current_file()
            visualization_masks_by_file = {}
            for key, mask in self.visualization_masks_by_file.items():
                visualization_masks_by_file[key] = {plugin.name: mask}
            self.pending_folder_path = None
            self.pending_folder_plugin_params = None
            series_index = self.pending_series_index
            self.pending_series_index = None

            if hasattr(self.view, "set_plugin_hint_visible"):
                self.view.set_plugin_hint_visible(False)

            self._start_statistics_worker(
                folder_path,
                plugins=[plugin],
                plugin_params=plugin_params,
                visualization_masks_by_file=visualization_masks_by_file,
                series_index=series_index,
            )
            return

        if self.model.image_data is None:
            self.view.show_progress("No image loaded.")
            return

        if self.model.masks is None:
            self.view.show_progress("No masks available on current image.")
            return

        if self.model.visualization_masks is None:
            self.view.show_progress("No plugin visualization mask to finalize.")
            return

        if self.analysis_service is None:
            self.analysis_service = AnalysisService()

        plugin = self.active_plugin
        params = dict(self.active_plugin_params or {})
        if plugin is None:
            selected_plugins, plugin_params = self.view.prompt_plugin_configuration(self.analysis_service.plugins)
            if not selected_plugins:
                return
            plugin = selected_plugins[0]
            params = plugin_params.get(plugin.name, {})
            self.active_plugin = plugin
            self.active_plugin_params = params

        params["visualization_masks"] = self.model.visualization_masks
        self.view.show_progress(f"Finalizing {plugin.name} analysis on current image...")

        try:
            results = self.analysis_service.run_analysis(
                self.model.image_data,
                self.model.masks,
                classes=self.model.classes,
                filename=os.path.basename(self.model.filename) if self.model.filename else None,
                plugins=[plugin],
                plugin_params={plugin.name: params},
            )

            if self.model.filename:
                saved_files = self.analysis_service.save_results(results, self.model.filename)
                for path in saved_files:
                    self.view.show_progress(f"Saved results to {os.path.basename(path)}")
                if hasattr(self.view, "set_plugin_hint_visible"):
                    self.view.set_plugin_hint_visible(False)
            else:
                self.view.show_progress("Results calculated but not saved (no filename).")
        except Exception as e:
            self.view.show_progress(f"Error finalizing analysis: {e}")

    def on_analysis_finished(self):
        self.view.set_analysis_running(False)

    def handle_analysis_progress(self, percentage, message):
        self.view.update_progress(percentage, message)
        self.view.show_progress(message)

    def handle_save_request(self):
        """Overrides MainController save to target _pred.npy for Analyzer."""
        # In Analyzer, standard save/autosave targets the prediction file
        self.image_service.save_prediction_with_classes(self.model)

    def handle_promote_request(self):
        """Promotes all predictions in the folder to ground truth labels (_seg.npy)."""
        folder_path = self.view.folder_label.text()
        if not folder_path or not os.path.isdir(folder_path):
            if self.model.filename:
                folder_path = os.path.dirname(self.model.filename)
        if not folder_path or not os.path.isdir(folder_path):
            self.view.show_progress("Select a valid folder to promote predictions.")
            return
        self.worker = PromoteWorker(folder_path)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.view.show_progress)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()
        self.view.set_analysis_running(True)

    def handle_load_image(self, filename):
        super().handle_load_image(filename)

    def _on_image_loaded(self, image_data, filename, frame_id, frame_refs):
        super()._on_image_loaded(image_data, filename, frame_id, frame_refs)
        self._prepare_visualization_for_current_image()

    def handle_add_mask_from_stroke(self, points):
        if (self.pending_folder_path or self.pending_series_file) and self.model.visualization_masks is not None:
            if self.model.add_visualization_mask(points):
                self._store_visualization_for_current_file()
            return
        super().handle_add_mask_from_stroke(points)

    def handle_delete_mask(self, y, x):
        super().handle_delete_mask(y, x)
        if (self.pending_folder_path or self.pending_series_file) and self.model.visualization_masks is not None:
            self._store_visualization_for_current_file()
