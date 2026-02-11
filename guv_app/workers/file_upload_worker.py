from PyQt6.QtCore import pyqtSignal
from guv_app.workers.base_worker import BaseWorker

class FileUploadWorker(BaseWorker):
    """
    Worker for uploading files to the remote server with progress tracking.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # Emits the resulting URI
    error = pyqtSignal(str)

    def __init__(self, remote_service, project_id, file_path, is_model=False):
        super().__init__()
        self.remote_service = remote_service
        self.project_id = project_id
        self.file_path = file_path
        self.is_model = is_model

    def run(self):
        try:
            if self.is_model:
                uri = self.remote_service.upload_model(self.file_path, self.progress.emit)
            else:
                uri = self.remote_service.upload_file(self.project_id, self.file_path, self.progress.emit)
            self.finished.emit(uri)
        except Exception as e:
            self.error.emit(str(e))
