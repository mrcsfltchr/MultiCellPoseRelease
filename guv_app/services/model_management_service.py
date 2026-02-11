import logging
import os

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    QObject = object
    pyqtSignal = lambda *args, **kwargs: None

from cellpose import models, io as cellpose_io

_logger = logging.getLogger(__name__)

class ModelManagementService(QObject):
    """
    Service for managing available segmentation models.
    """
    models_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # Standard Cellpose models
        self.core_models = ["cpsam", "cyto3", "nuclei", "cyto2", "tissuenet", "livecell", "cyto"]
        self._extra_models = []
        
    def get_available_models(self):
        """Returns a list of all available model names (core + user custom models)."""
        user_models = models.get_user_models()
        model_list = self.core_models + user_models + list(self._extra_models)
        seen = set()
        model_list = [m for m in model_list if not (m in seen or seen.add(m))]
        _logger.info("Available local models: %s", model_list)
        return model_list

    def register_model(self, model_name: str) -> None:
        if not model_name:
            return
        if model_name not in self._extra_models:
            self._extra_models.append(model_name)
        if self.models_updated is not None:
            try:
                self.models_updated.emit(model_name)
            except Exception:
                pass

    def add_model_from_path(self, filename: str) -> str | None:
        if not filename:
            return None
        try:
            cellpose_io.add_model(filename)
        except Exception as exc:
            _logger.error("Failed to add model: %s", exc)
            return None
        model_name = os.path.basename(filename)
        self.register_model(model_name)
        return model_name

    def remove_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        user_models = models.get_user_models()
        if model_name not in user_models:
            return False
        try:
            user_models = [m for m in user_models if m != model_name]
            with open(models.MODEL_LIST_PATH, "w", newline="\n") as textfile:
                for name in user_models:
                    textfile.write(name + "\n")
        except Exception as exc:
            _logger.error("Failed to remove model: %s", exc)
            return False
        if model_name in self._extra_models:
            self._extra_models.remove(model_name)
        if self.models_updated is not None:
            try:
                self.models_updated.emit(model_name)
            except Exception:
                pass
        return True
