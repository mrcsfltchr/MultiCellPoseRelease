from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMenu, QMessageBox
from cellpose import io as cellpose_io

from guv_app.views.dialogs.training_dialog import TrainingConfigDialog
from .base_view import BaseMainView

class TrainerView(BaseMainView):
    """
    The view for the Trainer, inheriting from the BaseMainVew.
    """
    training_requested = pyqtSignal()
    training_cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        """
        Initializer that sets up instance state.
        """
        super().__init__(parent)
        self.setWindowTitle("MultiCellPose - Trainer")
        self.set_base_window_title("MultiCellPose - Trainer")
        self._add_training_menu()
        self.model_train_requested.connect(lambda: self.training_requested.emit())
        self.model_train_help_requested.connect(self.show_training_instructions)

    def _add_training_menu(self):
        menubar = self.menuBar()
        training_menu = QMenu("&Training", menubar)
        menubar.addMenu(training_menu)
        train_action = QAction("Train model...", self)
        train_action.setShortcut(QKeySequence("Ctrl+T"))
        train_action.triggered.connect(self.training_requested.emit)
        training_menu.addAction(train_action)
        cancel_action = QAction("Cancel training", self)
        cancel_action.setShortcut(QKeySequence("Ctrl+C"))
        cancel_action.triggered.connect(self.training_cancel_requested.emit)
        training_menu.addAction(cancel_action)

    def prompt_training_config(self, model_names, default_config, train_files, total_blocks=None):
        dialog = TrainingConfigDialog(
            model_names=model_names,
            default_config=default_config,
            train_files=train_files,
            total_blocks=total_blocks,
            parent=self,
        )
        if dialog.exec():
            return dialog.get_config()
        return None

    def prompt_train_test_split(self, image_count):
        if image_count < 2:
            return 0.0
        resp = QMessageBox.question(
            self,
            "Training split",
            "No test set specified. Split the dataset into train/test?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return 0.0
        percent, ok = QInputDialog.getInt(
            self,
            "Test set size",
            "Test set percentage:",
            value=20,
            min=1,
            max=50,
        )
        if not ok:
            return None
        return float(percent) / 100.0

    def prompt_test_set_files(self, folder_path):
        resp = QMessageBox.question(
            self,
            "Test set selection",
            "Select a test set now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if resp == QMessageBox.StandardButton.Cancel:
            return None
        if resp != QMessageBox.StandardButton.Yes:
            return []
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select test set folder (optional)",
            folder_path or "",
        )
        if folder:
            return self._collect_test_set_files(folder)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select test set images",
            folder_path or "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.nd2 *.lif);;All Files (*)",
        )
        return files or []

    def _collect_test_set_files(self, folder_path):
        return self._collect_image_files(folder_path)

    def _collect_image_files(self, folder_path):
        if not folder_path:
            return []
        try:
            return [str(p) for p in cellpose_io.get_image_files(folder_path, "_masks")]
        except Exception:
            return [str(p) for p in cellpose_io.get_image_files(folder_path, "_masks", look_one_level_down=True)]

    def prompt_download_training_artifacts(self, artifact_names):
        if not artifact_names:
            return None
        resp = QMessageBox.question(
            self,
            "Download training artifacts",
            "Training finished. Download model artifacts (losses/meta)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return None
        folder = QFileDialog.getExistingDirectory(self, "Select Download Folder")
        return folder or None

    def show_training_instructions(self):
        QMessageBox.information(
            self,
            "Training instructions",
            "See training instructions in cellpose/gui/guitrainhelpwindowtext.html",
        )
