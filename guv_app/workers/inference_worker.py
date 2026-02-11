from guv_app.workers.base_worker import BaseWorker
from guv_app.services.segmentation_service import SegmentationService
from guv_app.data_models.results import InferenceResult
from PyQt6.QtCore import pyqtSignal
import numpy as np

class InferenceWorker(BaseWorker):
    finished = pyqtSignal(InferenceResult)
    canceled = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, segmentation_service: SegmentationService, image: np.ndarray, diameter: float, model_id: str, channel_index=None):
        super().__init__()
        self.segmentation_service = segmentation_service
        self.image = image
        self.diameter = diameter
        self.model_id = model_id
        self.channel_index = channel_index

    def run(self):
        if self.is_cancel_requested():
            self.canceled.emit()
            return
        masks, flows, styles = self.segmentation_service.run_inference(
            self.image, self.diameter, self.model_id, channel_index=self.channel_index)
        if self.is_cancel_requested():
            self.canceled.emit()
            return
        classes, classes_map = self.segmentation_service.postprocess_classes(masks, styles)
        
        result = InferenceResult(
            masks=masks, 
            flows=flows, 
            styles=styles, 
            diameter=self.diameter,
            classes=classes,
            classes_map=classes_map
        )
        self.finished.emit(result)
