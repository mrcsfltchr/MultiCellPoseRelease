import os
import numpy as np
import time
import pandas as pd
from PyQt6.QtCore import pyqtSignal
from guv_app.workers.base_worker import BaseWorker
from cellpose import io
from guv_app.plugins.interface import AnalysisPlugin
import logging

_logger = logging.getLogger(__name__)

class StatisticsWorker(BaseWorker):
    """
    Worker for running analysis plugins on a folder of images.
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, image_service, analysis_service, folder_path, plugins=None, plugin_params=None,
                 mask_suffix="_seg.npy", visualization_masks_by_file=None, series_index=None,
                 visualize_only=False, image_files=None):
        super().__init__()
        self.image_service = image_service
        self.analysis_service = analysis_service
        self.folder_path = folder_path
        self.plugins = plugins
        self.plugin_params = plugin_params
        self.mask_suffix = mask_suffix
        self.visualization_masks_by_file = visualization_masks_by_file
        self.series_index = series_index
        self.visualize_only = visualize_only
        self.image_files = image_files

    def run(self):
        try:
            self.progress.emit("Starting statistics analysis...")
            
            # Get files (filtering for images that likely have masks)
            # cellpose.io.get_image_files filters for image extensions and excludes internal files
            image_files = self.image_files
            if image_files is None:
                image_files = io.get_image_files(self.folder_path, '_masks')
            
            if not image_files:
                self.error.emit("No images found in the selected folder.")
                self.finished.emit()
                return

            all_results = {}
            total = len(image_files)
            
            for i, filename in enumerate(image_files):
                self.progress.emit(f"Processing {i+1}/{total}: {os.path.basename(filename)}")
                
                try:
                    frames = self.image_service.iter_image_frames(filename, series_index=self.series_index)
                    if not frames:
                        continue

                    for frame in frames:
                        frame_label = os.path.basename(filename)
                        if frame.frame_id:
                            frame_label = f"{frame_label}::{frame.frame_id}"
                        _logger.info("Plugin analysis: start frame %s", frame_label)
                        t_start = time.time()
                        image = frame.array
                        if image is None:
                            continue

                        base = os.path.splitext(filename)[0]
                        frame_suffix = io.frame_id_to_suffix(frame.frame_id)
                        seg_file = base + frame_suffix + self.mask_suffix
                        fallback_seg = base + self.mask_suffix

                        masks = None
                        classes = None

                        if os.path.exists(seg_file):
                            dat = np.load(seg_file, allow_pickle=True).item()
                            masks = dat.get('masks')
                            classes = dat.get('classes')
                        elif os.path.exists(fallback_seg):
                            dat = np.load(fallback_seg, allow_pickle=True).item()
                            masks = dat.get('masks')
                            classes = dat.get('classes')

                        if masks is None:
                            continue

                        plugin_params = self.plugin_params
                        if self.visualization_masks_by_file is not None:
                            ref = filename
                            if frame.frame_id:
                                ref = f"{filename}::{frame.frame_id}"
                            normalized = os.path.normcase(os.path.normpath(ref))
                            overrides = self.visualization_masks_by_file.get(normalized, {})
                            base_params = self.plugin_params or {}
                            plugin_params = {
                                name: dict(params) for name, params in base_params.items()
                            }
                            for plugin in self.plugins or []:
                                name = plugin.name
                                viz_mask = overrides.get(name)
                                if viz_mask is None and self._plugin_supports_visualization(plugin):
                                    viz_mask = self.analysis_service.run_visualization(
                                        plugin,
                                        image,
                                        masks,
                                        classes=classes,
                                        plugin_params=plugin_params.get(name),
                                    )
                                if viz_mask is not None:
                                    if name not in plugin_params:
                                        plugin_params[name] = {}
                                    plugin_params[name]["visualization_masks"] = viz_mask

                        frame_name = os.path.basename(filename)
                        if frame.frame_id:
                            frame_name = f"{frame_name}::{frame.frame_id}"

                        if self.visualize_only:
                            for plugin in self.plugins or []:
                                try:
                                    params = (plugin_params or {}).get(plugin.name, {})
                                    viz_mask = self.analysis_service.run_visualization(
                                        plugin,
                                        image,
                                        masks,
                                        classes=classes,
                                        plugin_params=params,
                                    )
                                    if viz_mask is None:
                                        continue
                                    self.image_service.save_visualization_mask(
                                        filename,
                                        frame.frame_id,
                                        viz_mask,
                                        plugin_name=plugin.name,
                                    )
                                except Exception as exc:
                                    _logger.warning(
                                        f"Visualization failed for {plugin.name} on {os.path.basename(filename)}: {exc}"
                                    )
                            _logger.info(
                                "Plugin analysis: visualization done for %s in %.2fs",
                                frame_label,
                                time.time() - t_start,
                            )
                            continue

                        results_dict = self.analysis_service.run_analysis(
                            image, masks, classes=classes, filename=frame_name,
                            plugins=self.plugins, plugin_params=plugin_params
                        )

                        for plugin_name, df in results_dict.items():
                            if df is None or df.empty:
                                continue
                            if 'plugin' not in df.columns:
                                df['plugin'] = plugin_name
                            all_results.setdefault(plugin_name, []).append(df)
                        _logger.info(
                            "Plugin analysis: finished frame %s in %.2fs",
                            frame_label,
                            time.time() - t_start,
                        )
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    # Don't fail the whole batch for one bad file
                    continue

            if self.visualize_only:
                self.progress.emit("Visualization masks generated. Review images and finalize plugin analysis.")
            elif all_results:
                series_suffix = ""
                if self.series_index is not None:
                    series_key = "S"
                    try:
                        for path in image_files:
                            ext = os.path.splitext(path)[1].lower()
                            if ext in (".nd2", ".lif"):
                                key, _, _ = self.image_service.get_series_time_info(path)
                                if key:
                                    series_key = key
                                break
                    except Exception:
                        pass
                    series_suffix = f"__{series_key}{self.series_index}"
                for plugin_name, dfs in all_results.items():
                    final_df = pd.concat(dfs, ignore_index=True)
                    safe_name = "".join(
                        x for x in plugin_name if x.isalnum() or x in "._- "
                    ).replace(" ", "_")
                    save_path = os.path.join(
                        self.folder_path,
                        f"statistics_results__{safe_name}{series_suffix}.csv",
                    )
                    final_df.to_csv(save_path, index=False)
                    self.progress.emit(f"Saved results to {os.path.basename(save_path)}")
            else:
                self.progress.emit("No analysis results generated (were masks present?).")

            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()

    @staticmethod
    def _plugin_supports_visualization(plugin):
        try:
            return plugin.visualize.__func__ is not AnalysisPlugin.visualize
        except AttributeError:
            return False
