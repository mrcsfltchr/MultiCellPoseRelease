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
import uuid
from guv_app.workers.class_lock_tracking_worker import (
    ClassLockApplyWorker,
    ClassLockPreviewWorker,
)

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
        self.rejected_object_tracking_track_ids = set()
        self._class_lock_preview = None
        self._class_lock_applied_result = None
        self._class_lock_worker = None
        self._class_lock_thread = None
        self._class_lock_busy = False
        self._class_lock_error = False
        self._class_track_selection_mode = False
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
        if hasattr(self.view, "lock_class_track_requested"):
            self.view.lock_class_track_requested.connect(self.handle_lock_class_across_track)
        self.view.begin_class_track_selection_requested.connect(
            self.begin_class_track_selection
        )
        self.view.cancel_class_track_selection_requested.connect(
            self.cancel_class_track_selection
        )

    @pyqtSlot()
    def begin_class_track_selection(self):
        if self._class_lock_busy:
            return
        self._class_track_selection_mode = True
        self.model.clear_selected_masks()
        self.view.set_class_track_selection_mode(True, ())
        self.model.trigger_view_update()
        self.view.statusBar().showMessage(
            "Track selection active: click objects to toggle them; Esc cancels."
        )

    @pyqtSlot()
    def cancel_class_track_selection(self):
        if not self._class_track_selection_mode or self._class_lock_busy:
            return
        self._class_track_selection_mode = False
        self.model.clear_selected_masks()
        self.view.set_class_track_selection_mode(False, ())
        self.model.trigger_view_update()
        self.view.statusBar().showMessage("Track selection cancelled.")

    def handle_select_mask(self, y, x):
        if not self._class_track_selection_mode:
            return super().handle_select_mask(y, x)
        if self.model.cellpix is None or not (0 <= y < self.model.Ly and 0 <= x < self.model.Lx):
            return
        mask_id = int(self.model.cellpix[0, y, x])
        if mask_id <= 0:
            return
        if mask_id in self.model.selected_mask_ids:
            self.model.selected_mask_ids.remove(mask_id)
        else:
            self.model.selected_mask_ids.add(mask_id)
        selected = sorted(self.model.get_selected_mask_ids())
        self.view.set_class_track_selection_mode(True, selected)
        self.view.statusBar().showMessage(
            f"Selected {len(selected)} track{'s' if len(selected) != 1 else ''}; "
            "click another object or Apply."
        )
        self.model.trigger_view_update()

    @pyqtSlot()
    def handle_lock_class_across_track(self):
        if self._class_lock_busy:
            self.view.show_progress("A class-lock operation is already running.")
            return
        selected = sorted(self.model.get_selected_mask_ids())
        if not selected:
            self.view.show_progress("Select at least one object before applying a class across tracks.")
            return
        if not getattr(self.model, "frame_refs", None) or len(self.model.frame_refs) < 2:
            self.view.show_progress("The current image is not a multi-timepoint series.")
            return
        frame_specs = []
        for ref in self.model.frame_refs:
            base_file, frame_id = self.image_service.split_image_reference(ref)
            path = self.image_service.build_frame_path(base_file, frame_id, "_pred.npy")
            frame_specs.append((frame_id, path))
        diameter = float(self.view.control_panel.diameter_spinbox.value())
        self._class_lock_busy = True
        self._class_lock_error = False
        self._class_lock_preview = None
        self._class_lock_worker = ClassLockPreviewWorker(
            frame_specs=frame_specs,
            anchor_frame_id=self.model.frame_id,
            anchor_mask_ids=selected,
            target_class=int(self.model.current_class),
            max_displacement=max(5.0, 2.0 * diameter),
        )
        self._class_lock_thread = QThread()
        self._class_lock_worker.moveToThread(self._class_lock_thread)
        self._class_lock_thread.started.connect(self._class_lock_worker.run)
        self._class_lock_worker.progress.connect(self._show_class_lock_progress)
        self._class_lock_worker.preview_ready.connect(self._store_class_lock_preview)
        self._class_lock_worker.error.connect(self._show_class_lock_error)
        self._class_lock_worker.finished.connect(self._class_lock_thread.quit)
        self._class_lock_worker.finished.connect(self._class_lock_worker.deleteLater)
        self._class_lock_thread.finished.connect(self._on_class_lock_preview_finished)
        self._class_lock_thread.finished.connect(self._class_lock_thread.deleteLater)
        self.view.lock_class_track_button.setEnabled(False)
        self.view.set_progress_busy(True, "Building class-agnostic object track…")
        self._class_lock_thread.start()

    def _show_class_lock_progress(self, message):
        self.view.show_progress(message)
        self.view.statusBar().showMessage(message)

    def _show_class_lock_error(self, message):
        self._class_lock_error = True
        self.view.show_progress(f"Class-lock error: {message}")
        self.view.statusBar().showMessage(f"Class-lock error: {message}")

    def _store_class_lock_preview(self, preview):
        self._class_lock_preview = preview

    def _on_class_lock_preview_finished(self):
        preview = self._class_lock_preview
        self._class_lock_worker = None
        self._class_lock_thread = None
        self.view.set_progress_busy(False)
        if preview is None or self._class_lock_error:
            self._finish_class_lock()
            return
        tracks = preview["tracks"]
        rows = []
        total_frames = len(preview["frames"])
        total_unique = len({(m.frame_index, m.mask_id) for members in tracks.values() for m in members})
        for number, (anchor_id, members) in enumerate(tracks.items(), start=1):
            indices = {member.frame_index for member in members}
            scores = [m.score for m in members if m.frame_index != preview["anchor_index"]]
            before = preview["anchor_index"] - min(indices)
            after = max(indices) - preview["anchor_index"]
            lost_before = preview["anchor_index"] - before
            lost_after = total_frames - 1 - preview["anchor_index"] - after
            rows.append(
                f"Track {number} (mask {anchor_id}): {len(members)}/{total_frames} frames; "
                f"lost before/after {lost_before}/{lost_after}; "
                f"low-confidence {sum(score < 0.5 for score in scores)}"
            )
        message = (
            f"Assign class {preview['target_class']} to {len(tracks)} selected tracks?\n\n"
            + "\n".join(rows)
            + "\n\n"
            f"Unique object instances: {total_unique}\n"
            f"Shared-mask conflicts: {len(preview['conflicts'])}\n\n"
            "Tracking ignores the current semantic class.\n"
            f"Maximum displacement: {preview['max_displacement']:.1f} px/frame\n"
            "Mask geometry will not be changed."
        )
        if not self.view.confirm_class_track_override(message):
            self._class_lock_preview = None
            self.view.show_progress("Class-lock operation cancelled.")
            print("Class lock: user cancelled apply", flush=True)
            self._finish_class_lock()
            return
        print("Class lock: user confirmed apply; starting save worker", flush=True)
        self._start_class_lock_apply(preview)

    def _start_class_lock_apply(self, preview):
        self._class_lock_applied_result = None
        self._class_lock_error = False
        override_id = f"class-lock-{uuid.uuid4()}"
        self._class_lock_worker = ClassLockApplyWorker(preview, override_id)
        self._class_lock_thread = QThread()
        self._class_lock_worker.moveToThread(self._class_lock_thread)
        self._class_lock_thread.started.connect(self._class_lock_worker.run)
        self._class_lock_worker.progress.connect(self._show_class_lock_progress)
        self._class_lock_worker.applied.connect(self._store_class_lock_applied_result)
        self._class_lock_worker.error.connect(self._show_class_lock_error)
        self._class_lock_worker.finished.connect(self._class_lock_thread.quit)
        self._class_lock_worker.finished.connect(self._class_lock_worker.deleteLater)
        self._class_lock_thread.finished.connect(self._on_class_lock_apply_finished)
        self._class_lock_thread.finished.connect(self._class_lock_thread.deleteLater)
        self.view.lock_class_track_button.setEnabled(False)
        self.view.set_progress_busy(True, "Applying and verifying class corrections…")
        self._class_lock_thread.start()

    def _store_class_lock_applied_result(self, result):
        self._class_lock_applied_result = result

    def _on_class_lock_apply_finished(self):
        result = self._class_lock_applied_result
        self._class_lock_worker = None
        self._class_lock_thread = None
        self.view.set_progress_busy(False)
        self._class_lock_preview = None
        if result is None or self._class_lock_error:
            self._finish_class_lock()
            return
        target_class = int(result["target_class"])
        anchor_mask_ids = [int(value) for value in result["anchor_mask_ids"]]
        for anchor_mask_id in anchor_mask_ids:
            self.model.assign_class_to_mask(anchor_mask_id, target_class)
        self.model.persist_current_channel_state()
        self.model.clear_selected_masks()
        self.model.trigger_view_update()
        message = (
            f"Applied class {target_class} to {len(anchor_mask_ids)} tracks and "
            f"{result['verified_frames']} object instances "
            f"({result['override_id']})."
        )
        self.view.show_progress(message)
        self.view.statusBar().showMessage(message)
        self._class_track_selection_mode = False
        self.view.set_class_track_selection_mode(False, ())
        self._finish_class_lock()

    def _finish_class_lock(self):
        self._class_lock_busy = False
        self._class_lock_error = False
        self._class_lock_worker = None
        self._class_lock_thread = None
        self.view.lock_class_track_button.setEnabled(True)

    def cleanup_all_threads(self):
        thread = self._class_lock_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
        self._class_lock_worker = None
        self._class_lock_thread = None
        self._class_lock_busy = False
        super().cleanup_all_threads()

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
        # Always process all frames in each multi-image file (series_index=None)
        series_index = None

        if self.analysis_service is None:
            self.analysis_service = AnalysisService()
        else:
            self.analysis_service.discover_plugins(reload_modules=True)

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

        image_files = self._folder_image_files(folder_path)
        skip_folder_visualization = self._plugin_supports_visualization(plugin) and len(image_files) > 1

        if self._plugin_supports_visualization(plugin) and not skip_folder_visualization:
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
            self._prepare_plugin_review_display(plugin)
            if hasattr(self.view, "set_plugin_hint_visible"):
                self.view.set_plugin_hint_visible(True)
            self.view.show_progress(
                f"{plugin.name} visualization generating for folder. Navigate images to review, then press Finalize Plugin Analysis."
            )
            return

        if skip_folder_visualization:
            self.active_plugin = None
            self.active_plugin_params = {}
            self.pending_visualization_generation = False
            self.visualization_masks_by_file = {}
            if hasattr(self.view, "set_plugin_hint_visible"):
                self.view.set_plugin_hint_visible(False)
            self.view.show_progress(
                f"{plugin.name} supports visualization, but folder contains multiple image files; running direct CSV export."
            )

        self.pending_folder_path = None
        self.pending_folder_plugin_params = None
        self._start_statistics_worker(
            folder_path,
            plugins=[plugin],
            plugin_params=plugin_params,
            visualization_masks_by_file=None,
            series_index=series_index,
            image_files=image_files or None,
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
            viz_mask, changed = self._apply_rejected_tracks_to_visualization(self.visualization_masks_by_file[normalized])
            if changed:
                self.visualization_masks_by_file[normalized] = viz_mask
                self.image_service.save_visualization_mask(
                    self.model.filename,
                    self.model.frame_id,
                    viz_mask,
                    plugin_name=self.active_plugin.name if self.active_plugin else None,
                )
            self.model.set_visualization(viz_mask)
            self.model.view_config.show_visualization = True
            self.view.control_panel.visualization_checkbox.setChecked(True)
            self._prepare_plugin_review_display(self.active_plugin)
            return

        plugin_name = self.active_plugin.name if self.active_plugin else None
        stored = self.image_service.load_visualization_mask(
            self.model.filename,
            self.model.frame_id,
            plugin_name=plugin_name,
            reference_masks=self.model.masks,
            require_same_label=plugin_name == "Condensate Droplet Analysis",
            allow_labels_above_reference=plugin_name == "Object Tracking",
        )
        if stored is not None:
            stored, changed = self._apply_rejected_tracks_to_visualization(stored)
            self.model.set_visualization(stored)
            self.visualization_masks_by_file[normalized] = np.array(stored, copy=True)
            if changed:
                self.image_service.save_visualization_mask(
                    self.model.filename,
                    self.model.frame_id,
                    stored,
                    plugin_name=plugin_name,
                )
            self.model.view_config.show_visualization = True
            self.view.control_panel.visualization_checkbox.setChecked(True)
            self._prepare_plugin_review_display(self.active_plugin)
            return

        viz_params = dict(self.active_plugin_params or {})
        frame_name = os.path.basename(self.model.filename) if self.model.filename else None
        if frame_name and self.model.frame_id:
            frame_name = f"{frame_name}::{self.model.frame_id}"
        if frame_name:
            viz_params.setdefault("filename", frame_name)
        viz_mask = self.analysis_service.run_visualization(
            self.active_plugin,
            self.model.image_data,
            self.model.masks,
            classes=self.model.classes,
            plugin_params=viz_params,
        )
        if viz_mask is None:
            return
        viz_mask, _ = self._apply_rejected_tracks_to_visualization(viz_mask)
        self.model.set_visualization(viz_mask)
        self._store_visualization_for_current_file()
        self.model.view_config.show_visualization = True
        self.view.control_panel.visualization_checkbox.setChecked(True)
        self._prepare_plugin_review_display(self.active_plugin)

    def _plugin_supports_visualization(self, plugin):
        try:
            return plugin.visualize.__func__ is not AnalysisPlugin.visualize
        except AttributeError:
            return False

    def _prepare_plugin_review_display(self, plugin):
        if plugin is None or plugin.name != "Object Tracking":
            if hasattr(self.model, "set_visualization_color_by_label"):
                self.model.set_visualization_color_by_label(False)
            return
        if hasattr(self.model, "set_visualization_color_by_label"):
            self.model.set_visualization_color_by_label(True)
        self.model.view_config.color_by_class = False
        checkbox = getattr(getattr(self.view, "control_panel", None), "color_by_class_checkbox", None)
        if checkbox is not None and checkbox.isChecked():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self.model.trigger_view_update()

    def _is_object_tracking_review_active(self):
        return (
            self.active_plugin is not None
            and self.active_plugin.name == "Object Tracking"
            and self.pending_series_file is not None
            and self.model.visualization_masks is not None
            and self.model.view_config.show_visualization
        )

    def _track_ids_at_point(self, y, x):
        if not hasattr(self.model, "visualization_label_at_point"):
            return set()
        track_id = int(self.model.visualization_label_at_point(int(y), int(x)))
        return {track_id} if track_id > 0 else set()

    def _track_ids_in_polygon(self, points):
        if not hasattr(self.model, "visualization_labels_in_polygon"):
            return set()
        return self.model.visualization_labels_in_polygon(points)

    def _remove_track_ids_from_review(self, track_ids):
        track_ids = {int(i) for i in track_ids if int(i) > 0}
        if not track_ids or not self.pending_series_file:
            return 0
        self.rejected_object_tracking_track_ids.update(track_ids)
        frames = self.image_service.iter_image_frames(
            self.pending_series_file,
            series_index=self.pending_series_index,
        )
        changed = 0
        for frame in frames:
            ref = self.image_service.build_image_reference(self.pending_series_file, frame.frame_id)
            normalized = os.path.normcase(os.path.normpath(ref))
            viz = self.visualization_masks_by_file.get(normalized)
            if viz is None:
                viz = self.image_service.load_visualization_mask(
                    self.pending_series_file,
                    frame.frame_id,
                    plugin_name="Object Tracking",
                    allow_labels_above_reference=True,
                )
            if viz is None:
                continue
            updated = np.array(viz, copy=True)
            nonzero_before = int(np.count_nonzero(updated))
            updated[np.isin(updated, list(track_ids))] = 0
            if int(np.count_nonzero(updated)) == nonzero_before:
                continue
            changed += 1
            self.visualization_masks_by_file[normalized] = updated
            self.image_service.save_visualization_mask(
                self.pending_series_file,
                frame.frame_id,
                updated,
                plugin_name="Object Tracking",
            )
            if self.model.filename == self.pending_series_file and self.model.frame_id == frame.frame_id:
                self.model.set_visualization(updated)
        return changed

    def _apply_rejected_tracks_to_visualization(self, viz):
        if self.active_plugin is None or self.active_plugin.name != "Object Tracking":
            return viz, False
        rejected = {int(i) for i in getattr(self, "rejected_object_tracking_track_ids", set()) if int(i) > 0}
        if not rejected or viz is None:
            return viz, False
        updated = np.array(viz, copy=True)
        before = int(np.count_nonzero(updated))
        updated[np.isin(updated, list(rejected))] = 0
        return updated, int(np.count_nonzero(updated)) != before

    def _approved_track_ids_for_series(self, base_file, frames):
        approved = set()
        for frame in frames:
            ref = self.image_service.build_image_reference(base_file, frame.frame_id)
            normalized = os.path.normcase(os.path.normpath(ref))
            viz = self.visualization_masks_by_file.get(normalized)
            if viz is None:
                viz = self.image_service.load_visualization_mask(
                    base_file,
                    frame.frame_id,
                    plugin_name="Object Tracking",
                    allow_labels_above_reference=True,
                )
            if viz is None:
                continue
            viz, _ = self._apply_rejected_tracks_to_visualization(viz)
            ids = np.unique(np.asarray(viz))
            approved.update(int(i) for i in ids if int(i) > 0)
        return sorted(approved)

    def _get_analysis_image(self):
        """Return the pre-normalization image for intensity measurements.

        Priority: source_image (original pixel values) > raw_image
        (normalize99-processed) > image_data.  Using the pre-normalization
        image ensures that per-channel intensity values reflect actual
        fluorescence levels rather than globally-normalised ones.
        """
        if getattr(self.model, "source_image", None) is not None:
            return self.model.source_image
        if getattr(self.model, "raw_image", None) is not None:
            return self.model.raw_image
        return self.model.image_data

    def _augment_plugin_params_for_current_model(self, plugin_params, plugins):
        params = {name: dict(values) for name, values in (plugin_params or {}).items()}
        channel_index = int(self.model.get_current_channel_index())
        object_ids = self.model.get_object_ids_for_channel(channel_index)
        channel_segmentations = getattr(self.model, "channel_segmentations", None)
        for plugin in plugins or []:
            name = plugin.name
            if name not in params:
                params[name] = {}
            params[name].setdefault("channel_index", channel_index)
            params[name].setdefault("object_ids_by_mask", object_ids)
            params[name].setdefault("channel_segmentations", channel_segmentations)
        return params

    @staticmethod
    def _folder_image_files(folder_path):
        try:
            return io.get_image_files(folder_path, '_masks')
        except Exception:
            return []

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
        analysis_image = self._get_analysis_image()
        if analysis_image is None:
            self.view.show_progress("No image loaded.")
            return

        if self.model.masks is None:
            self.view.show_progress("No masks available on current image.")
            return

        # Ensure service is ready
        if self.analysis_service is None:
            self.analysis_service = AnalysisService()
        else:
            self.analysis_service.discover_plugins(reload_modules=True)
        names = ", ".join(sorted(self.analysis_service.plugins.keys()))
        self.view.show_progress(f"Discovered plugins: {names}" if names else "Discovered plugins: <none>")

        # 1. Select Plugin
        selected_plugins, plugin_params = self.view.prompt_plugin_configuration(self.analysis_service.plugins)
        if not selected_plugins:
            return

        # For visualization, we typically only want to visualize one result at a time.
        # We'll take the first selected plugin.
        plugin = selected_plugins[0]
        plugin_params = self._augment_plugin_params_for_current_model(plugin_params, [plugin])
        params = plugin_params.get(plugin.name, {})
        self.active_plugin = plugin
        self.active_plugin_params = params

        self.view.show_progress(f"Running {plugin.name} on current image...")

        # 2. Run Analysis (Calculation) & Save
        try:
            # Use service to handle defaults and execution
            results = self.analysis_service.run_analysis(
                analysis_image,
                self.model.masks,
                classes=self.model.classes,
                filename=os.path.basename(self.model.filename) if self.model.filename else None,
                plugins=[plugin],
                plugin_params={plugin.name: params}
            )

            # Save results using the service
            if self.model.filename:
                saved_files = self.analysis_service.save_results(results, self.model.filename, frame_id=self.model.frame_id)
                for path in saved_files:
                    self.view.show_progress(f"Saved results to {os.path.basename(path)}")
            else:
                self.view.show_progress("Results calculated but not saved (no filename).")

        except Exception as e:
            self.view.show_progress(f"Error running analysis: {e}")

        # 3. Run Visualization
        try:
            viz_params = dict(params)
            frame_name = os.path.basename(self.model.filename) if self.model.filename else None
            if frame_name and self.model.frame_id:
                frame_name = f"{frame_name}::{self.model.frame_id}"
            if frame_name:
                viz_params.setdefault("filename", frame_name)
            viz_mask = self.analysis_service.run_visualization(
                plugin, analysis_image, self.model.masks,
                classes=self.model.classes, plugin_params=viz_params
            )
            
            if viz_mask is not None:
                # 4. Update View (by updating model)
                # We set the visualization masks (temporary, not saved)
                self.model.set_visualization(viz_mask)
                self.model.view_config.show_visualization = True
                self.view.control_panel.visualization_checkbox.setChecked(True)
                self._prepare_plugin_review_display(plugin)
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
        else:
            self.analysis_service.discover_plugins(reload_modules=True)
        selected_plugins, plugin_params = self.view.prompt_plugin_configuration(
            self.analysis_service.plugins
        )
        if not selected_plugins:
            return
        base_file = self.model.filename
        series_index = getattr(self.model, "series_index", None)
        if base_file:
            try:
                _, series_count, time_count = self.image_service.get_series_time_info(base_file)
                if series_count > 1 and time_count <= 1:
                    series_index = None  # positions-only file: iterate all positions
            except Exception:
                pass
        plugin = selected_plugins[0]
        plugin_params = self._augment_plugin_params_for_current_model(plugin_params, [plugin])
        if self._plugin_supports_visualization(plugin):
            self.active_plugin = plugin
            self.active_plugin_params = plugin_params.get(plugin.name, {})
            if plugin.name == "Object Tracking":
                self.rejected_object_tracking_track_ids = set()
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
            self._prepare_plugin_review_display(plugin)
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
            self._store_visualization_for_current_file()
            frames = self.image_service.iter_image_frames(base_file, series_index=series_index)
            if not frames:
                self.view.show_progress("No frames found for this series.")
                return
            if plugin.name == "Object Tracking":
                approved_track_ids = self._approved_track_ids_for_series(base_file, frames)
                if not approved_track_ids:
                    self.view.show_progress("No approved Object Tracking tracks remain to save.")
                    return
                plugin_params[plugin.name]["approved_track_ids"] = approved_track_ids
            combined = {}
            for frame in frames:
                image = self.image_service.load_frame(base_file, frame.frame_id)
                if image is None:
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
                image = _align_image_to_masks(image, masks)
                if image is None:
                    continue
                viz_mask = self.image_service.load_visualization_mask(
                    base_file,
                    frame.frame_id,
                    plugin_name=plugin.name,
                    reference_masks=masks,
                    require_same_label=plugin.name == "Condensate Droplet Analysis",
                    allow_labels_above_reference=plugin.name == "Object Tracking",
                )
                if viz_mask is not None:
                    plugin_params[plugin.name]["visualization_masks"] = viz_mask
                else:
                    plugin_params[plugin.name].pop("visualization_masks", None)
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
                channel_suffix = ""
                if "intensity_channel_name" in merged.columns:
                    vals = [
                        str(v).strip() for v in merged["intensity_channel_name"].dropna().unique()
                        if str(v).strip()
                    ]
                    if len(vals) == 1:
                        token = "".join(c for c in vals[0] if c.isalnum() or c in "._-").lower()
                        if token:
                            channel_suffix = f"_{token}"
                    elif len(vals) > 1:
                        channel_suffix = "_multi_channel"
                out_path = f"{os.path.splitext(base_file)[0]}__series{series_suffix}_{safe_name}{channel_suffix}.csv"
                try:
                    if plugin_name == "Object Tracking":
                        from guv_app.plugins.object_tracking_timeseries_export import tracking_position_tables, tracking_timeseries_tables
                        fallback_name = f"{os.path.basename(os.path.splitext(base_file)[0])}_{safe_name}"
                        tables = tracking_timeseries_tables(
                            merged,
                            fallback_name=fallback_name,
                            intensity_columns="auto",
                        )
                        position_tables = tracking_position_tables(
                            merged,
                            fallback_name=fallback_name,
                        )
                        saved = []
                        if len(tables) == 1:
                            tables[0][1].to_csv(out_path, index=False)
                            saved.append(out_path)
                            pos_path = f"{os.path.splitext(out_path)[0]}_positions.csv"
                            position_tables[0][1].to_csv(pos_path, index=False)
                            saved.append(pos_path)
                        else:
                            root = os.path.splitext(base_file)[0]
                            for series_key, wide_df in tables:
                                label = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(series_key).strip()).strip("._") or "series"
                                path = f"{root}__series{series_suffix}_{safe_name}__{label}.csv"
                                wide_df.to_csv(path, index=False)
                                saved.append(path)
                            for series_key, position_df in position_tables:
                                label = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(series_key).strip()).strip("._") or "series"
                                path = f"{root}__series{series_suffix}_{safe_name}__{label}_positions.csv"
                                position_df.to_csv(path, index=False)
                                saved.append(path)
                        for path in saved:
                            self.view.show_progress(f"Saved {os.path.basename(path)}")
                    else:
                        merged.to_csv(out_path, index=False)
                        self.view.show_progress(f"Saved {os.path.basename(out_path)}")
                except Exception as exc:
                    self.view.show_progress(f"Failed to save {safe_name} CSV: {exc}")
            if hasattr(self.view, "set_plugin_hint_visible"):
                self.view.set_plugin_hint_visible(False)
            self.pending_series_file = None
            self.pending_series_index = None
            self.rejected_object_tracking_track_ids = set()
            return
        if self.pending_folder_path:
            if self.active_plugin is None:
                self.view.show_progress("No active plugin selected.")
                return

            folder_path = self.pending_folder_path
            plugin = self.active_plugin
            plugin_params = dict(self.pending_folder_plugin_params or {})
            series_index = self.pending_series_index

            # Persist any edits to the currently displayed visualization before
            # the worker consumes the folder-wide saved visualization masks.
            self._store_visualization_for_current_file()

            visualization_masks_by_file = {}
            for key, mask in self.visualization_masks_by_file.items():
                visualization_masks_by_file[key] = {plugin.name: mask}

            self.pending_folder_path = None
            self.pending_folder_plugin_params = None
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
            self.view.show_progress(
                f"Finalizing {plugin.name} analysis for all images in folder."
            )
            return

        analysis_image = self._get_analysis_image()
        if analysis_image is None:
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
            plugin_params = self._augment_plugin_params_for_current_model(plugin_params, [plugin])
            params = plugin_params.get(plugin.name, {})
            self.active_plugin = plugin
            self.active_plugin_params = params

        params["visualization_masks"] = self.model.visualization_masks
        self.view.show_progress(f"Finalizing {plugin.name} analysis on current image...")

        try:
            results = self.analysis_service.run_analysis(
                analysis_image,
                self.model.masks,
                classes=self.model.classes,
                filename=os.path.basename(self.model.filename) if self.model.filename else None,
                plugins=[plugin],
                plugin_params={plugin.name: params},
            )

            if self.model.filename:
                saved_files = self.analysis_service.save_results(results, self.model.filename, frame_id=self.model.frame_id)
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
        if self._is_object_tracking_review_active():
            track_ids = self._track_ids_at_point(y, x)
            changed = self._remove_track_ids_from_review(track_ids)
            if changed:
                label = ", ".join(str(i) for i in sorted(track_ids))
                self.view.statusBar().showMessage(f"Removed track {label} from {changed} frame(s).")
            elif not track_ids:
                self.view.statusBar().showMessage("No track selected in visualization.")
            return
        super().handle_delete_mask(y, x)
        if (self.pending_folder_path or self.pending_series_file) and self.model.visualization_masks is not None:
            self._store_visualization_for_current_file()

    def handle_delete_masks_lasso(self, points):
        if self._is_object_tracking_review_active():
            track_ids = self._track_ids_in_polygon(points)
            changed = self._remove_track_ids_from_review(track_ids)
            if changed:
                label = ", ".join(str(i) for i in sorted(track_ids))
                self.view.statusBar().showMessage(f"Removed tracks {label} from {changed} frame(s).")
            elif not track_ids:
                self.view.statusBar().showMessage("No tracks selected in visualization.")
            return
        super().handle_delete_masks_lasso(points)
        if (self.pending_folder_path or self.pending_series_file) and self.model.visualization_masks is not None:
            self._store_visualization_for_current_file()


def _align_image_to_masks(image, masks):
    if image is None or masks is None:
        return image
    img = np.asarray(image)
    mask_shape = np.squeeze(np.asarray(masks)).shape
    if len(mask_shape) > 2:
        mask_shape = mask_shape[-2:]
    if img.ndim >= 2 and tuple(img.shape[:2]) == tuple(mask_shape):
        return image
    if img.ndim == 3 and tuple(img.shape[1:]) == tuple(mask_shape):
        return np.moveaxis(img, 0, -1)
    if img.ndim >= 2 and tuple(img.shape[-2:]) == tuple(mask_shape):
        return image
    return None


def _to_2d_array(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    squeezed = np.squeeze(arr)
    return squeezed if squeezed.ndim == 2 else None
