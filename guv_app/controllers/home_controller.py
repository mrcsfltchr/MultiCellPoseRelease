import logging

from guv_app.controllers.analyzer_controller import AnalyzerController
from guv_app.controllers.training_controller import TrainerController
from guv_app.models.app_state import ApplicationStateModel
from guv_app.services.analysis_service import AnalysisService
from guv_app.services.image_service import ImageService
from guv_app.services.model_management_service import ModelManagementService
from guv_app.services.remote_service import RemoteConnectionService
from guv_app.services.segmentation_service import SegmentationService
from guv_app.services.training_dataset_service import TrainingDatasetService
from guv_app.views.analyzer_view import AnalyzerView
from guv_app.views.trainer_view import TrainerView
from PyQt6.QtCore import Qt

_logger = logging.getLogger(__name__)


class HomeController:
    def __init__(self, view, app):
        self.view = view
        self.app = app
        self._child_windows = []
        self._controllers = []
        self._analyzer_open = False
        self._trainer_open = False
        self._analyzer_view = None
        self._trainer_view = None
        self._model_service = ModelManagementService()

    def connect_signals(self):
        self.view.start_analyzer_requested.connect(self.launch_analyzer)
        self.view.start_trainer_requested.connect(self.launch_trainer)

    def launch_analyzer(self):
        if self._analyzer_open:
            if self._analyzer_view:
                self._raise_window(self._analyzer_view)
            return
        services = self._create_services()
        model = ApplicationStateModel()
        view = AnalyzerView()
        view.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        controller = AnalyzerController(model, view, services)
        controller.connect_signals()
        self._track_window(view, controller)
        self._analyzer_view = view
        self._analyzer_open = True
        if hasattr(self.view, "set_analyzer_enabled"):
            self.view.set_analyzer_enabled(False)
        view.destroyed.connect(self._on_analyzer_closed)
        view.show()

    def launch_trainer(self):
        if self._trainer_open:
            if self._trainer_view:
                self._raise_window(self._trainer_view)
            return
        services = self._create_services()
        model = ApplicationStateModel()
        view = TrainerView()
        view.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        controller = TrainerController(model, view, services)
        controller.connect_signals()
        self._track_window(view, controller)
        self._trainer_view = view
        self._trainer_open = True
        if hasattr(self.view, "set_trainer_enabled"):
            self.view.set_trainer_enabled(False)
        view.destroyed.connect(self._on_trainer_closed)
        view.show()

    def _track_window(self, view, controller):
        self._child_windows.append(view)
        self._controllers.append(controller)
        self.app.aboutToQuit.connect(controller.cleanup_all_threads)

    def _on_analyzer_closed(self, *args):
        self._analyzer_open = False
        self._analyzer_view = None
        if hasattr(self.view, "set_analyzer_enabled"):
            self.view.set_analyzer_enabled(True)

    def _on_trainer_closed(self, *args):
        self._trainer_open = False
        self._trainer_view = None
        if hasattr(self.view, "set_trainer_enabled"):
            self.view.set_trainer_enabled(True)

    def _raise_window(self, view):
        if view is None:
            return
        view.show()
        view.raise_()
        view.activateWindow()

    def _create_services(self):
        services = {}
        services["image"] = ImageService()
        services["segmentation"] = SegmentationService()
        services["remote"] = RemoteConnectionService()
        services["model_management"] = self._model_service
        services["analysis"] = AnalysisService()
        services["training_dataset"] = TrainingDatasetService()
        return services
