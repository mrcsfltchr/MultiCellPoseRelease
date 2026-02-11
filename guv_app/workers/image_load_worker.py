from PyQt6.QtCore import pyqtSignal
from guv_app.workers.base_worker import BaseWorker


class ImageLoadWorker(BaseWorker):
    loaded = pyqtSignal(object, str, object, object)
    error = pyqtSignal(str)

    def __init__(self, image_service, filename, series_index=None):
        super().__init__()
        self.image_service = image_service
        self.filename = filename
        self.series_index = series_index

    def run(self):
        try:
            base, frame_id = self.image_service.split_image_reference(self.filename)
            if frame_id is not None:
                image = self.image_service.load_frame(base, frame_id)
                self.loaded.emit(image, base, frame_id, None)
                return

            frame_refs = self.image_service.build_frame_references(base, series_index=self.series_index)
            if frame_refs:
                first_ref = frame_refs[0]
                _, first_id = self.image_service.split_image_reference(first_ref)
                image = self.image_service.load_frame(base, first_id)
                if image is None:
                    raise RuntimeError(f"Failed to load frame {first_id} from {base}")
                self.loaded.emit(image, base, first_id, frame_refs)
                return

            image = self.image_service.load_image(base)
            self.loaded.emit(image, base, None, None)
        except Exception as exc:
            self.error.emit(str(exc))
