from PyQt6.QtCore import pyqtSignal, QObject
from guv_app.data_models.configs import TrainingConfig
from guv_app.data_models.results import TrainingResult
import os
import logging
import numpy as np
from guv_app.services import training_service as training_service_module
from guv_app.services.training_service import TrainingService
from cellpose.models import CellposeModel

_logger = logging.getLogger(__name__)

class TrainingWorker(QObject):
    """
    Worker for running model training on the local machine.
    Inherits from QObject to allow moving to a thread and using signals.
    """
    finished = pyqtSignal(TrainingResult)
    progress = pyqtSignal(int, int, float, float)  # epoch, total_epochs, train_loss, test_loss
    error = pyqtSignal(str)

    def __init__(
        self,
        config: TrainingConfig,
        train_data=None,
        train_labels=None,
        test_data=None,
        test_labels=None,
        class_maps=None,
        test_class_maps=None,
        flow_labels=None,
        test_flow_labels=None,
    ):
        super().__init__()
        self.config = config
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.class_maps = class_maps
        self.test_class_maps = test_class_maps
        self.flow_labels = flow_labels
        self.test_flow_labels = test_flow_labels

    def run(self):
        """
        Entry-point routine that orchestrates execution.
        """
        try:
            base_model = self.config.base_model
            if self.config.use_lora and os.path.basename(str(base_model)) != "cpsam":
                _logger.info(
                    "LoRA enabled: overriding local base model '%s' -> 'cpsam'.",
                    base_model,
                )
                base_model = "cpsam"

            net = None
            class_max = None
            if self.class_maps:
                for cmap in self.class_maps:
                    if cmap is None:
                        continue
                    try:
                        vmax = int(np.max(cmap))
                        if class_max is None or vmax > class_max:
                            class_max = vmax
                    except Exception:
                        continue
            if class_max is not None and class_max >= 1 and os.path.basename(str(base_model)) == "cpsam":
                net = training_service_module._initialize_class_net(nclasses=class_max + 1)
            else:
                model = CellposeModel(pretrained_model=base_model, gpu=True)
                net = model.net

            training_service = TrainingService(net=net)

            # 3. Define progress callback
            def progress_callback(epoch, total_epochs, train_loss, test_loss):
                self.progress.emit(epoch, total_epochs, train_loss, test_loss)

            # 4. Start training
            result = training_service.start_training(
                config=self.config,
                progress_callback=progress_callback,
                train_data=self.train_data,
                train_labels=self.train_labels,
                test_data=self.test_data,
                test_labels=self.test_labels,
                class_maps=self.class_maps,
                test_class_maps=self.test_class_maps,
                flow_labels=self.flow_labels,
                test_flow_labels=self.test_flow_labels,
            )

            # 5. Emit finished signal
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
