
import os
from guv_app.workers.base_worker import BaseWorker
from guv_app.services.segmentation_service import SegmentationService
from guv_app.services.image_service import ImageService
from guv_app.data_models.results import InferenceResult
from PyQt6.QtCore import pyqtSignal
import numpy as np
from cellpose import utils

class AnalysisWorker(BaseWorker):
    finished = pyqtSignal()
    canceled = pyqtSignal()
    result_ready = pyqtSignal(InferenceResult)
    progress = pyqtSignal(int, str)

    def __init__(self, segmentation_service: SegmentationService, image_service: ImageService, folder_path: str, diameter: float, model_id: str, series_index=None, channel_index=None, file_list=None):
        super().__init__()
        self.segmentation_service = segmentation_service
        self.image_service = image_service
        self.folder_path = folder_path
        self.diameter = diameter
        self.model_id = model_id
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
            
        files = self.file_list if self.file_list is not None else get_image_files(self.folder_path, "")
        total_files = len(files)
        for i, f in enumerate(files):
            if self.is_cancel_requested():
                self.progress.emit(int((i / max(1, total_files)) * 100), "Canceled")
                self.canceled.emit()
                return
            self.progress.emit(int((i / total_files) * 100), f"Processing {i+1}/{total_files}: {os.path.basename(f)}")
            try:
                refs = self.image_service.build_frame_references(f, series_index=self.series_index)
                if refs:
                    for ref in refs:
                        if self.is_cancel_requested():
                            self.progress.emit(int((i / max(1, total_files)) * 100), "Canceled")
                            self.canceled.emit()
                            return
                        base, frame_id = self.image_service.split_image_reference(ref)
                        image = self.image_service.load_frame(base, frame_id)
                        if image is None:
                            continue
                        masks, flows, styles = self.segmentation_service.run_inference(
                            image, self.diameter, self.model_id, channel_index=self.channel_index)
                        if self.is_cancel_requested():
                            self.progress.emit(int((i / max(1, total_files)) * 100), "Canceled")
                            self.canceled.emit()
                            return
                        classes, classes_map = self.segmentation_service.postprocess_classes(masks, styles)
                        result = InferenceResult(
                            masks=masks,
                            flows=flows,
                            styles=styles,
                            filename=f,
                            frame_id=frame_id,
                            diameter=self.diameter,
                            classes=classes,
                            classes_map=classes_map
                        )
                        self.result_ready.emit(result)
                    continue

                frames = self.image_service.iter_image_frames(f, series_index=self.series_index)
                for frame in frames:
                    if self.is_cancel_requested():
                        self.progress.emit(int((i / max(1, total_files)) * 100), "Canceled")
                        self.canceled.emit()
                        return
                    image = frame.array
                    if image is None:
                        continue
                    masks, flows, styles = self.segmentation_service.run_inference(
                        image, self.diameter, self.model_id, channel_index=self.channel_index)
                    if self.is_cancel_requested():
                        self.progress.emit(int((i / max(1, total_files)) * 100), "Canceled")
                        self.canceled.emit()
                        return
                    classes, classes_map = self.segmentation_service.postprocess_classes(masks, styles)
                    result = InferenceResult(
                        masks=masks,
                        flows=flows,
                        styles=styles,
                        filename=f,
                        frame_id=frame.frame_id,
                        diameter=self.diameter,
                        classes=classes,
                        classes_map=classes_map
                    )
                    self.result_ready.emit(result)

            except Exception as e:
                self.progress.emit(int((i / total_files) * 100), f"Failed to process {f}: {e}")
        
        self.progress.emit(100, "Done")
        self.finished.emit()
