from typing import Optional, Tuple, List, Dict

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QLabel,
    QDockWidget,
    QProgressBar,
    QToolButton,
    QHBoxLayout,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
)
from PyQt6 import sip
from PyQt6.QtGui import QAction

from .base_view import BaseMainView
from guv_app.views.dialogs.plugin_config_dialog import PluginConfigDialog, DynamicPluginConfigWidget


class AnalyzerView(BaseMainView):
    """
    The view for the Analyzer, inheriting from the BaseMainView.
    """

    batch_folder_selected = pyqtSignal(str)
    start_analysis = pyqtSignal()
    run_plugin_requested = pyqtSignal()
    run_plugin_series_requested = pyqtSignal()
    finalize_plugin_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MultiCellPose - Analyzer")
        self.set_base_window_title("MultiCellPose - Analyzer")
        self._add_masks_menu()

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.select_folder_button = QPushButton("Select Folder")
        self.run_plugin_button = QPushButton("Run Plugin on Current Image")
        self.run_plugin_series_button = QPushButton("Run Plugin on Multi-Image")
        self.run_plugin_series_button.setEnabled(False)
        self.run_plugin_folder_button = QPushButton("Run Plugins on Folder")
        self.finalize_plugin_button = QPushButton("Finalize Plugin Analysis")
        self.plugin_hint_label = QLabel(
            "Plugin visualization mode: edit masks, then press Finalize Plugin Analysis."
        )
        self.plugin_hint_label.setWordWrap(True)
        self.plugin_hint_label.setStyleSheet("color: #555555;")
        self.plugin_hint_label.setVisible(False)
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setVisible(False)
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)

        self.analysis_widget = QWidget()
        layout = QVBoxLayout(self.analysis_widget)
        layout.addWidget(self.select_folder_button)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.run_plugin_button)
        layout.addWidget(self.run_plugin_series_button)
        layout.addWidget(self.run_plugin_folder_button)
        layout.addWidget(self.finalize_plugin_button)
        layout.addWidget(self.plugin_hint_label)
        layout.addWidget(self.analysis_progress_bar)
        layout.addWidget(self.progress_text)

        self.analysis_dock = QDockWidget("Batch Analysis", self)
        self.analysis_dock.setWidget(self.analysis_widget)
        self.analysis_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.analysis_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._analysis_dock_collapsed = False
        self._analysis_dock_expanded_width = 320

        title_bar = QWidget(self.analysis_dock)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 2, 6, 2)
        title_layout.setSpacing(6)
        title_label = QLabel("Batch Analysis", title_bar)
        self._analysis_toggle_button = QToolButton(title_bar)
        self._analysis_toggle_button.setText("<")
        self._analysis_toggle_button.setToolTip("Collapse panel")
        self._analysis_toggle_button.clicked.connect(self._toggle_analysis_dock)
        title_layout.addWidget(title_label)
        title_layout.addStretch(1)
        title_layout.addWidget(self._analysis_toggle_button)
        self.analysis_dock.setTitleBarWidget(title_bar)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)

        self._connect_signals()

    def _connect_signals(self) -> None:
        self.select_folder_button.clicked.connect(self.select_folder)
        self.run_plugin_button.clicked.connect(self.run_plugin_requested.emit)
        self.run_plugin_series_button.clicked.connect(self.run_plugin_series_requested.emit)
        self.run_plugin_folder_button.clicked.connect(self.export_csv_requested.emit)
        self.finalize_plugin_button.clicked.connect(self.finalize_plugin_requested.emit)

    def _add_masks_menu(self) -> None:
        menubar = self.menuBar()
        masks_menu = menubar.addMenu('&Masks')
        promote_action = QAction('Promote predictions (pred) -> labels (seg)', self)
        promote_action.triggered.connect(self.promote_requested.emit)
        masks_menu.addAction(promote_action)

    def prompt_plugin_configuration(
        self,
        available_plugins: Dict[str, object],
    ) -> Tuple[Optional[List[object]], Optional[Dict[str, Dict[str, object]]]]:
        """
        Shows a dialog to select plugins and configure their parameters.
        Returns: (selected_plugins_list, plugin_params_dict) or (None, None)
        """
        dlg = PluginConfigDialog(available_plugins, self)
        if dlg.exec():
            return dlg.get_configuration()
        return None, None

    def prompt_plugin_selection(self, available_plugins: Dict[str, object]) -> Optional[object]:
        if not available_plugins:
            return None
        plugin_names = sorted(available_plugins.keys())
        if len(plugin_names) == 1:
            return available_plugins[plugin_names[0]]
        selection, ok = QInputDialog.getItem(
            self,
            "Select Plugin",
            "Plugin:",
            plugin_names,
            0,
            False,
        )
        if not ok or not selection:
            return None
        return available_plugins.get(selection)

    def prompt_plugin_parameters(self, plugin: object) -> Optional[Dict[str, object]]:
        definitions = plugin.get_parameter_definitions() if plugin else {}
        if not definitions:
            return {}
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Configure {plugin.name}")
        layout = QVBoxLayout(dialog)
        widget = DynamicPluginConfigWidget(definitions)
        layout.addWidget(widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            return widget.get_values()
        return None

    def set_folder_path(self, folder_path: str) -> None:
        if self.folder_label is None or sip.isdeleted(self.folder_label):
            return
        self.folder_label.setText(folder_path)

    def show_progress(self, message: str) -> None:
        if self.progress_text is None or sip.isdeleted(self.progress_text):
            return
        self.progress_text.append(message)

    def set_plugin_hint_visible(self, visible: bool) -> None:
        self.plugin_hint_label.setVisible(visible)

    def _toggle_analysis_dock(self) -> None:
        if self._analysis_dock_collapsed:
            self.analysis_dock.setMaximumWidth(16777215)
            self.analysis_dock.setMinimumWidth(0)
            self.analysis_dock.widget().setVisible(True)
            self._analysis_toggle_button.setText("<")
            self._analysis_toggle_button.setToolTip("Collapse panel")
            if self._analysis_dock_expanded_width:
                self.analysis_dock.resize(
                    self._analysis_dock_expanded_width,
                    self.analysis_dock.height(),
                )
        else:
            self._analysis_dock_expanded_width = max(200, self.analysis_dock.width())
            self.analysis_dock.widget().setVisible(False)
            self.analysis_dock.setMaximumWidth(28)
            self.analysis_dock.setMinimumWidth(28)
            self._analysis_toggle_button.setText(">")
            self._analysis_toggle_button.setToolTip("Expand panel")
        self._analysis_dock_collapsed = not self._analysis_dock_collapsed

    def set_analysis_running(self, running: bool) -> None:
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(not running)
        self.select_folder_button.setEnabled(not running)
        self.analysis_progress_bar.setVisible(running)
        if running:
            self.analysis_progress_bar.setRange(0, 0)
        else:
            self.analysis_progress_bar.setRange(0, 100)
            self.analysis_progress_bar.setValue(0)

    def select_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.batch_folder_selected.emit(folder_path)
