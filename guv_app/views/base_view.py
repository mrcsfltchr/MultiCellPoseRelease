import os
import logging
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStatusBar,
    QMenu,
    QMenuBar,
    QFileDialog,
    QProgressBar,
    QApplication,
    QLineEdit,
    QTextEdit,
    QAbstractSpinBox,
    QInputDialog,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6 import sip
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
import pyqtgraph as pg
import numpy as np
from cellpose import utils
from guv_app.views.widgets.drawing import DrawingItem
from guv_app.views.widgets.control_panel import ControlPanel

_logger = logging.getLogger(__name__)


class BaseMainView(QMainWindow):
    """
    The main application window containing all shared UI components.
    """
    # Signals used by Controller
    file_loaded = pyqtSignal(str)
    folder_selected = pyqtSignal(str)
    save_requested = pyqtSignal()
    navigate_next_requested = pyqtSignal()
    navigate_prev_requested = pyqtSignal()
    navigate_z_requested = pyqtSignal(int)
    connect_remote_requested = pyqtSignal()
    disconnect_remote_requested = pyqtSignal()
    add_ssh_hostname_requested = pyqtSignal()
    ssh_advanced_requested = pyqtSignal()
    upload_model_requested = pyqtSignal()
    clear_remote_jobs_requested = pyqtSignal()
    model_add_requested = pyqtSignal()
    model_remove_requested = pyqtSignal()
    model_train_requested = pyqtSignal()
    model_train_help_requested = pyqtSignal()
    export_csv_requested = pyqtSignal()
    promote_requested = pyqtSignal()
    import_seg_requested = pyqtSignal(str)
    toggle_masks_requested = pyqtSignal()
    toggle_outlines_requested = pyqtSignal()
    toggle_freeze_masks_requested = pyqtSignal()
    toggle_color_mode_requested = pyqtSignal()
    toggle_visualization_requested = pyqtSignal()
    move_selected_masks_requested = pyqtSignal(int, int)
    brush_size_change_requested = pyqtSignal(int)
    view_mode_step_requested = pyqtSignal(int)
    color_mode_step_requested = pyqtSignal(int)
    color_mode_set_requested = pyqtSignal(int)
    finalize_stroke_requested = pyqtSignal()
    toggle_association_mode_requested = pyqtSignal(bool)
    auto_associate_requested = pyqtSignal()
    link_association_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1200, 800)
        self._base_window_title = None
        self._is_remote = False
        self.remote_connect_action = None
        self._supported_extensions = {
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
            ".npy",
            ".nd2",
            ".lif",
            ".dax",
            ".nrrd",
            ".flex",
        }
        self.setAcceptDrops(True)

        # Central Widget & Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # 1. Image View (PyQtGraph)
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.view_box = self.graph_layout.addViewBox(row=0, col=0)
        self.view_box.setAspectLocked(True)
        self.view_box.invertY(True)

        # Image Item
        self.img_item = pg.ImageItem()
        self.img_item.setZValue(0)
        self.view_box.addItem(self.img_item)

        # Drawing Item (Masks)
        self.drawing_item = DrawingItem(parent=self)
        self.drawing_item.setZValue(10)
        self.view_box.addItem(self.drawing_item)

        self.association_lines_item = pg.PlotDataItem()
        self.association_lines_item.setZValue(20)
        self.association_lines_item.setPen(pg.mkPen((0, 255, 0), width=2))
        self.view_box.addItem(self.association_lines_item)
        self.association_selected_lines_item = pg.PlotDataItem()
        self.association_selected_lines_item.setZValue(21)
        self.association_selected_lines_item.setPen(pg.mkPen((0, 255, 255), width=3))
        self.view_box.addItem(self.association_selected_lines_item)
        self.association_points_item = pg.ScatterPlotItem(size=6, pxMode=True, pen=None, brush=pg.mkBrush(0, 255, 0, 220))
        self.association_points_item.setZValue(22)
        self.view_box.addItem(self.association_points_item)
        self.association_selected_points_item = pg.ScatterPlotItem(size=8, pxMode=True, pen=None, brush=pg.mkBrush(0, 255, 255, 240))
        self.association_selected_points_item.setZValue(23)
        self.view_box.addItem(self.association_selected_points_item)
        self.reference_selection_points_item = pg.ScatterPlotItem(
            size=14,
            pxMode=True,
            symbol="x",
            pen=pg.mkPen((0, 255, 255), width=3),
            brush=None,
        )
        self.reference_selection_points_item.setZValue(24)
        self.view_box.addItem(self.reference_selection_points_item)
        self.review_track_lines_item = pg.ImageItem()
        self.review_track_lines_item.setZValue(30)
        self.view_box.addItem(self.review_track_lines_item)
        self.review_track_points_item = pg.ScatterPlotItem(size=8, pxMode=True)
        self.review_track_points_item.setZValue(31)
        self.view_box.addItem(self.review_track_points_item)
        self.review_track_label_items = []

        self.main_layout.addWidget(self.graph_layout, stretch=3)

        # 2. Control Panel
        self.control_panel = ControlPanel(parent=self)
        self.control_panel.run_folder_button.clicked.connect(self.on_run_on_folder)
        self.main_layout.addWidget(self.control_panel, stretch=1)

        # Status Bar
        self.setStatusBar(QStatusBar())
        self.connection_label = QLabel("Local")
        self.connection_label.setStyleSheet("padding: 0 6px; color: #555555;")
        self.statusBar().addPermanentWidget(self.connection_label)
        self.activity_label = QLabel("")
        self.activity_label.setStyleSheet("padding: 0 6px; color: #444444;")
        self.statusBar().addPermanentWidget(self.activity_label)
        self.progress_bar = QProgressBar(self)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #3B82F6;
                width: 10px;
            }
        """)

        # Create Menu Bar
        self.create_menu_bar()
        self._setup_shortcuts()

    def update_progress(self, value, text=""):
        if self.progress_bar is None or sip.isdeleted(self.progress_bar):
            return
        if value is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)
        if text:
            self.statusBar().showMessage(text)
        elif value == 0:
            self.statusBar().clearMessage()

    def set_progress_busy(self, busy, text=""):
        if busy:
            self.update_progress(None, text=text)
        else:
            self.update_progress(0, text=text)

    def set_connection_mode(self, is_remote):
        if self.connection_label is None or sip.isdeleted(self.connection_label):
            return
        self._is_remote = bool(is_remote)
        if is_remote:
            self.connection_label.setText("Remote")
            self.connection_label.setStyleSheet("padding: 0 6px; color: #1F6F1F;")
        else:
            self.connection_label.setText("Local")
            self.connection_label.setStyleSheet("padding: 0 6px; color: #555555;")
        if self.remote_connect_action is not None:
            if self._is_remote:
                self.remote_connect_action.setText("&Stop remote connection")
            else:
                self.remote_connect_action.setText("&Connect to remote...")

    def set_activity_message(self, text):
        if self.activity_label is None or sip.isdeleted(self.activity_label):
            return
        self.activity_label.setText(text or "")

    def set_base_window_title(self, title):
        self._base_window_title = title
        self.setWindowTitle(title)

    def update_window_title(self, filename, frame_id=None):
        if not self._base_window_title:
            self._base_window_title = self.windowTitle()
        if filename:
            name = os.path.basename(filename)
            if frame_id:
                name = f"{name} [{frame_id}]"
            self.setWindowTitle(f"{self._base_window_title} - {name}")
        else:
            self.setWindowTitle(self._base_window_title)

    def on_run_on_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_selected.emit(folder_path)


    def create_menu_bar(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')

        load_action = QAction('&Load Image...', self)
        load_action.setShortcut('Ctrl+L')
        load_action.triggered.connect(self.on_load_image)
        file_menu.addAction(load_action)

        save_action = QAction('&Save Masks', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_requested.emit)
        file_menu.addAction(save_action)

        import_seg_action = QAction('Import &Segmentation...', self)
        import_seg_action.setShortcut('Ctrl+Shift+L')
        import_seg_action.triggered.connect(self.on_import_segmentation)
        file_menu.addAction(import_seg_action)

        tools_menu = menubar.addMenu('&Tools')
        self.association_mode_action = QAction('Association mode', self)
        self.association_mode_action.setCheckable(True)
        self.association_mode_action.toggled.connect(self.toggle_association_mode_requested.emit)
        tools_menu.addAction(self.association_mode_action)

        auto_associate_action = QAction('Auto-match current to previous channel', self)
        auto_associate_action.triggered.connect(self.auto_associate_requested.emit)
        tools_menu.addAction(auto_associate_action)

        link_association_action = QAction('Link selected masks across channels', self)
        link_association_action.triggered.connect(self.link_association_requested.emit)
        tools_menu.addAction(link_association_action)

        # Remote Menu
        remote_menu = menubar.addMenu('&Remote')

        self.remote_connect_action = QAction('&Connect to remote...', self)
        self.remote_connect_action.triggered.connect(self._handle_remote_action)
        remote_menu.addAction(self.remote_connect_action)

        add_ssh_action = QAction('Add SSH Hostname...', self)
        add_ssh_action.triggered.connect(self.add_ssh_hostname_requested.emit)
        remote_menu.addAction(add_ssh_action)

        ssh_adv_action = QAction('SSH Advanced...', self)
        ssh_adv_action.triggered.connect(self.ssh_advanced_requested.emit)
        remote_menu.addAction(ssh_adv_action)

        remote_menu.addSeparator()

        upload_action = QAction('&Upload model to server...', self)
        upload_action.triggered.connect(self.upload_model_requested.emit)
        remote_menu.addAction(upload_action)

        clear_jobs_action = QAction('&Clear remote training files...', self)
        clear_jobs_action.triggered.connect(self.clear_remote_jobs_requested.emit)
        remote_menu.addAction(clear_jobs_action)

        # Models Menu
        models_menu = menubar.addMenu("&Models")
        add_model_action = QAction("Add custom torch model to GUI...", self)
        add_model_action.triggered.connect(self.model_add_requested.emit)
        models_menu.addAction(add_model_action)

        remove_model_action = QAction("Remove selected custom model from GUI", self)
        remove_model_action.triggered.connect(self.model_remove_requested.emit)
        models_menu.addAction(remove_model_action)

        train_model_action = QAction("Train new model with image+masks in folder", self)
        train_model_action.triggered.connect(self.model_train_requested.emit)
        models_menu.addAction(train_model_action)

        train_help_action = QAction("Training instructions", self)
        train_help_action.triggered.connect(self.model_train_help_requested.emit)
        models_menu.addAction(train_help_action)

    def _handle_remote_action(self):
        if self._is_remote:
            self.disconnect_remote_requested.emit()
        else:
            self.connect_remote_requested.emit()


    def _setup_shortcuts(self):
        self._shortcuts = []
        self._add_shortcut(QKeySequence("X"), self._toggle_masks_shortcut)
        self._add_shortcut(QKeySequence("Z"), self._toggle_outlines_shortcut)
        self._add_shortcut(QKeySequence("F"), self._toggle_freeze_masks_shortcut)
        self._add_shortcut(QKeySequence("C"), self._toggle_color_mode_shortcut)
        self._add_shortcut(QKeySequence("K"), self._toggle_visualization_shortcut)
        self._add_shortcut(QKeySequence("D"), self._toggle_delete_lasso_shortcut)
        self._add_shortcut(QKeySequence(Qt.Key.Key_Left), self._navigate_prev_shortcut)
        self._add_shortcut(QKeySequence(Qt.Key.Key_Right), self._navigate_next_shortcut)
        self._add_shortcut(QKeySequence("A"), self._navigate_prev_shortcut)
        self._add_shortcut(QKeySequence(","), lambda: self._move_selected_masks_shortcut(-1, 0))
        self._add_shortcut(QKeySequence("."), lambda: self._move_selected_masks_shortcut(1, 0))
        self._add_shortcut(QKeySequence("J"), self._toggle_association_mode_shortcut)
        self._add_shortcut(QKeySequence("M"), self._auto_associate_shortcut)
        self._add_shortcut(QKeySequence("L"), self._link_association_shortcut)
        self._add_shortcut(QKeySequence("-"), lambda: self._zoom_view(1.1))
        self._add_shortcut(QKeySequence("="), lambda: self._zoom_view(0.9))
        self._add_shortcut(QKeySequence("+"), lambda: self._zoom_view(0.9))
        self._add_shortcut(QKeySequence(Qt.Key.Key_PageDown), lambda: self._view_mode_shortcut(1))
        self._add_shortcut(QKeySequence(Qt.Key.Key_PageUp), lambda: self._view_mode_shortcut(-1))
        self._add_shortcut(QKeySequence(Qt.Key.Key_Up), lambda: self._color_mode_shortcut(-1))
        self._add_shortcut(QKeySequence(Qt.Key.Key_Down), lambda: self._color_mode_shortcut(1))
        self._add_shortcut(QKeySequence("W"), lambda: self._color_mode_shortcut(-1))
        self._add_shortcut(QKeySequence("S"), lambda: self._color_mode_shortcut(1))
        self._add_shortcut(QKeySequence("R"), lambda: self._color_mode_toggle(1))
        self._add_shortcut(QKeySequence("G"), lambda: self._color_mode_toggle(2))
        self._add_shortcut(QKeySequence("B"), lambda: self._color_mode_toggle(3))
        self._add_shortcut(QKeySequence(Qt.Key.Key_BracketLeft), lambda: self._move_selected_masks_shortcut(0, -1))
        self._add_shortcut(QKeySequence(Qt.Key.Key_BracketRight), lambda: self._move_selected_masks_shortcut(0, 1))
        self._add_shortcut(QKeySequence("Q"), lambda: self._navigate_z_shortcut(-1))
        self._add_shortcut(QKeySequence("E"), lambda: self._navigate_z_shortcut(1))
        self._add_shortcut(QKeySequence(Qt.Key.Key_Return), self._finalize_stroke_shortcut)
        self._add_shortcut(QKeySequence(Qt.Key.Key_Enter), self._finalize_stroke_shortcut)

    def _add_shortcut(self, key_sequence, handler):
        shortcut = QShortcut(key_sequence, self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(handler)
        self._shortcuts.append(shortcut)

    def _should_ignore_shortcut(self, allow_in_stroke=False):
        if self.drawing_item.in_stroke and not allow_in_stroke:
            return True
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return False
        return isinstance(focus_widget, (QLineEdit, QTextEdit, QAbstractSpinBox))

    def _toggle_masks_shortcut(self):
        if not self._should_ignore_shortcut():
            self.toggle_masks_requested.emit()

    def _toggle_outlines_shortcut(self):
        if not self._should_ignore_shortcut():
            self.toggle_outlines_requested.emit()

    def _toggle_freeze_masks_shortcut(self):
        if not self._should_ignore_shortcut():
            self.toggle_freeze_masks_requested.emit()

    def _toggle_color_mode_shortcut(self):
        if not self._should_ignore_shortcut():
            self.toggle_color_mode_requested.emit()

    def _toggle_visualization_shortcut(self):
        if not self._should_ignore_shortcut():
            self.toggle_visualization_requested.emit()

    def _navigate_prev_shortcut(self):
        if not self._should_ignore_shortcut():
            self.navigate_prev_requested.emit()

    def _navigate_next_shortcut(self):
        if not self._should_ignore_shortcut():
            self.navigate_next_requested.emit()

    def _navigate_z_shortcut(self, delta):
        if not self._should_ignore_shortcut():
            self.navigate_z_requested.emit(delta)

    def _brush_size_shortcut(self, delta):
        if not self._should_ignore_shortcut():
            self.brush_size_change_requested.emit(delta)

    def _zoom_view(self, scale):
        if not self._should_ignore_shortcut():
            self.view_box.scaleBy([scale, scale])

    def _view_mode_shortcut(self, delta):
        if not self._should_ignore_shortcut():
            self.view_mode_step_requested.emit(delta)

    def _color_mode_shortcut(self, delta):
        if not self._should_ignore_shortcut():
            self.color_mode_step_requested.emit(delta)

    def _color_mode_toggle(self, color_index):
        if not self._should_ignore_shortcut():
            self.color_mode_set_requested.emit(color_index)

    def _finalize_stroke_shortcut(self):
        if not self._should_ignore_shortcut(allow_in_stroke=True):
            self.finalize_stroke_requested.emit()

    def _move_selected_masks_shortcut(self, dx, dy):
        if not self._should_ignore_shortcut():
            self.move_selected_masks_requested.emit(dx, dy)

    def _toggle_delete_lasso_shortcut(self):
        if self._should_ignore_shortcut():
            return
        if hasattr(self, "control_panel") and hasattr(self.control_panel, "delete_lasso_button"):
            self.control_panel.delete_lasso_button.toggle()

    def _toggle_association_mode_shortcut(self):
        if self._should_ignore_shortcut():
            return
        if hasattr(self, "association_mode_action"):
            self.association_mode_action.trigger()

    def _auto_associate_shortcut(self):
        if not self._should_ignore_shortcut():
            self.auto_associate_requested.emit()

    def _link_association_shortcut(self):
        if not self._should_ignore_shortcut():
            self.link_association_requested.emit()

    def set_association_mode_enabled(self, enabled):
        if hasattr(self, "association_mode_action"):
            self.association_mode_action.blockSignals(True)
            self.association_mode_action.setChecked(bool(enabled))
            self.association_mode_action.blockSignals(False)
        if hasattr(self, "drawing_item"):
            self.drawing_item.set_association_mode(enabled)

    def on_load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.npy *.nd2 *.lif *.dax *.nrrd)")
        if filename:
            self.file_loaded.emit(filename)

    def on_import_segmentation(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select segmentation", "", "Segmentation (*.npy)")
        if filename:
            self.import_seg_requested.emit(filename)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if any(self._is_supported_path(p) for p in paths):
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if not paths:
            event.ignore()
            return

        dirs = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if os.path.isfile(p) and self._is_supported_path(p)]

        if dirs and not files:
            self.folder_selected.emit(dirs[0])
            event.acceptProposedAction()
            return

        if files:
            if len(files) > 1:
                self.statusBar().showMessage("Multiple files dropped. Loading the first; use 'Run on folder' for batch.")
            self.file_loaded.emit(files[0])
            event.acceptProposedAction()
            return

        event.ignore()

    def _is_supported_path(self, path):
        if os.path.isdir(path):
            return True
        _, ext = os.path.splitext(path)
        return ext.lower() in self._supported_extensions

    def prompt_tiff_stack_interpretation(self, filename, plane_count):
        options = [
            "Separate channels in one image",
            "Separate positions/images",
        ]
        try:
            choice, ok = QInputDialog.getItem(
                self,
                "Interpret TIFF stack",
                (
                    f"The TIFF metadata does not define what the {plane_count} stacked planes mean.\n"
                    f"File: {filename}\n"
                    "Interpret the stack as:"
                ),
                options,
                0,
                False,
            )
            if not ok:
                return None
            if choice == options[0]:
                return "channels"
            if choice == options[1]:
                return "positions"
        except Exception:
            return None
        return None

    def prompt_series_index(self, max_series, message=None):
        if max_series <= 1:
            return 0
        if message is None:
            message = f"Series index (0-{max_series - 1}):"
        try:
            value, ok = QInputDialog.getInt(
                self,
                "Select series",
                message,
                0,
                0,
                max_series - 1,
                1,
            )
            if not ok:
                return None
            return value
        except Exception:
            return 0

    def set_model(self, model):
        self.model = model
        # Connect model signals to view updates if needed
        self.model.view_update_signal.connect(self.update_view)

    def update_view(self):
        if self.model.image_data is not None:
            self.display_image(self.model.image_data)

        if self.model.image_data is None:
            self._clear_association_overlay()
            self._clear_review_track_overlay()
            return

        h, w = self.model.image_data.shape[:2]
        mask_display = np.zeros((h, w, 4), dtype=np.uint8)
        z = 0
        pending_review_viz_masks = None
        pending_review_object_masks = None

        reference_idx = getattr(self.model, "previous_channel_index", None)
        current_idx = self.model.get_current_channel_index() if hasattr(self.model, "get_current_channel_index") else 0
        association_mode_enabled = bool(getattr(getattr(self, "association_mode_action", None), "isChecked", lambda: False)())
        ref_state = None
        ref_masks = None
        selected_reference_mask_id = getattr(self.model, "selected_reference_mask_id", None)
        if association_mode_enabled and reference_idx is not None and reference_idx != current_idx and hasattr(self.model, "channel_has_masks") and self.model.channel_has_masks(reference_idx):
            ref_state = self.model.get_channel_state(reference_idx)
            ref_masks = ref_state["masks"]
            ref_masks = ref_masks[z] if ref_masks.ndim == 3 else ref_masks
            ref_rgba = self._render_mask_layer(
                ref_masks,
                ref_state,
                alpha_val=45,
                force_color=np.array([255, 215, 0], dtype=np.uint8),
                selected_ids={int(selected_reference_mask_id)} if selected_reference_mask_id else set(),
                selection_color=np.array([0, 255, 255], dtype=np.uint8),
            )
            self._blit_rgba(mask_display, ref_rgba)

        has_viz = self.model.visualization_masks is not None and self.model.view_config.show_visualization
        label_review_viz = has_viz and bool(getattr(self.model, "visualization_color_by_label", False))
        if (self.model.cellpix is not None or has_viz) and (
            self.model.view_config.masks_visible
            or self.model.view_config.outlines_visible
            or has_viz
        ):
            source_masks = None
            is_outline_mode = False
            viz_masks = None
            if has_viz:
                if self.model.visualization_masks.ndim == 3:
                    if z < self.model.visualization_masks.shape[0]:
                        viz_masks = self.model.visualization_masks[z]
                else:
                    viz_masks = self.model.visualization_masks
            if label_review_viz:
                if self.model.cellpix is not None and self.model.view_config.masks_visible:
                    source_masks = self.model.cellpix[z]
                elif self.model.cellpix is not None and self.model.view_config.outlines_visible:
                    if self.model.outpix is not None:
                        source_masks = self.model.outpix[z]
                        is_outline_mode = True
            elif has_viz:
                source_masks = viz_masks
            elif self.model.cellpix is not None:
                source_masks = self.model.cellpix[z]
            if not has_viz and not self.model.view_config.masks_visible and self.model.view_config.outlines_visible:
                if self.model.outpix is not None:
                    source_masks = self.model.outpix[z]
                    is_outline_mode = True
            elif not has_viz and not self.model.view_config.masks_visible:
                source_masks = None

            if source_masks is not None and source_masks.max() > 0:
                current_state = self.model.get_channel_state(current_idx) if hasattr(self.model, "get_channel_state") else {
                    "mask_classes": self.model.mask_classes,
                    "instance_colors": self.model.instance_colors,
                }
                current_rgba = self._render_mask_layer(
                    source_masks,
                    current_state,
                    alpha_val=255 if is_outline_mode else 100,
                    selected_ids=self.model.get_selected_mask_ids() if hasattr(self.model, "get_selected_mask_ids") else set(),
                    selection_color=np.array([255, 255, 255], dtype=np.uint8),
                )
                self._blit_rgba(mask_display, current_rgba)
            if label_review_viz and viz_masks is not None:
                pending_review_viz_masks = viz_masks
                pending_review_object_masks = self.model.cellpix[z] if self.model.cellpix is not None else None
            else:
                self._clear_review_track_overlay()

        # Redraw the selected previous-channel mask on top so it remains visible
        # even when the current-channel masks overlap it.
        if (
            association_mode_enabled
            and ref_masks is not None
            and selected_reference_mask_id is not None
        ):
            selected_ref_masks = np.where(ref_masks == int(selected_reference_mask_id), ref_masks, 0)
            if selected_ref_masks.max() > 0:
                selected_ref_rgba = self._render_mask_layer(
                    selected_ref_masks,
                    ref_state,
                    alpha_val=70,
                    force_color=np.array([255, 215, 0], dtype=np.uint8),
                    selected_ids={int(selected_reference_mask_id)},
                    selection_color=np.array([0, 255, 255], dtype=np.uint8),
                )
                self._blit_rgba(mask_display, selected_ref_rgba)
            if hasattr(self.model, "get_mask_centroid"):
                center = self.model.get_mask_centroid(reference_idx, int(selected_reference_mask_id))
                if center is not None:
                    cy, cx = center
                    self.reference_selection_points_item.setData(x=[cx], y=[cy])
                else:
                    self.reference_selection_points_item.setData(x=[], y=[])
            else:
                self.reference_selection_points_item.setData(x=[], y=[])
        else:
            self.reference_selection_points_item.setData(x=[], y=[])

        segments = []
        if reference_idx is not None and hasattr(self.model, "get_association_line_segments"):
            segments = self.model.get_association_line_segments(reference_idx, current_idx)
        self._update_association_overlay(segments)

        self.drawing_item.setImage(mask_display, autoLevels=False)
        self.drawing_item.setLevels([0, 255])
        if pending_review_viz_masks is not None:
            self._update_review_track_overlay(pending_review_viz_masks, pending_review_object_masks)

    def _clear_association_overlay(self):
        for item in (
            getattr(self, "association_lines_item", None),
            getattr(self, "association_selected_lines_item", None),
            getattr(self, "association_points_item", None),
            getattr(self, "association_selected_points_item", None),
            getattr(self, "reference_selection_points_item", None),
        ):
            if item is None:
                continue
            try:
                item.setData([], [])
            except TypeError:
                item.setData(x=[], y=[])

    def _clear_review_track_overlay(self):
        if getattr(self, "review_track_lines_item", None) is not None:
            self.review_track_lines_item.clear()
        if getattr(self, "review_track_points_item", None) is not None:
            try:
                self.review_track_points_item.setData([])
            except TypeError:
                self.review_track_points_item.setData(x=[], y=[])
        for item in getattr(self, "review_track_label_items", []):
            try:
                self.view_box.removeItem(item)
            except Exception:
                pass
        self.review_track_label_items = []

    def _update_review_track_overlay(self, viz_masks, object_masks):
        self.drawing_item.setZValue(10)
        self.review_track_lines_item.setZValue(40)
        self.review_track_points_item.setZValue(41)
        viz = np.asarray(viz_masks)
        if viz.ndim != 2 or viz.max() <= 0:
            self._clear_review_track_overlay()
            return
        obj = np.asarray(object_masks) if object_masks is not None else np.zeros_like(viz)
        if obj.shape != viz.shape:
            obj = np.zeros_like(viz)

        rgba = np.zeros(viz.shape + (4,), dtype=np.uint8)
        current = (viz > 0) & (obj > 0)
        track_pixels = viz > 0
        for label in np.unique(viz[track_pixels]):
            label = int(label)
            if label <= 0:
                continue
            color = np.asarray(_review_track_color(label), dtype=np.uint8)
            label_pixels = track_pixels & (viz == label)
            current_pixels = current & (viz == label)
            if np.any(current_pixels):
                cy, cx = np.mean(np.nonzero(current_pixels), axis=1)
            else:
                cy, cx = np.mean(np.nonzero(label_pixels), axis=1)
            yy, xx = np.nonzero(label_pixels)
            dist = np.hypot(yy.astype(float) - float(cy), xx.astype(float) - float(cx))
            if dist.size and float(dist.max()) > 0:
                recency = 1.0 - (dist / float(dist.max()))
            else:
                recency = np.ones_like(dist, dtype=float)
            alpha = (70 + 185 * recency).astype(np.uint8)
            rgb = (color[None, :] * (0.65 + 0.35 * recency[:, None]) + 255 * (0.35 * (1.0 - recency[:, None]))).astype(np.uint8)
            rgba[yy, xx, :3] = rgb
            rgba[yy, xx, 3] = alpha
        self.review_track_lines_item.setImage(rgba, autoLevels=False)
        self.review_track_lines_item.setLevels([0, 255])

        spots = []
        for item in getattr(self, "review_track_label_items", []):
            try:
                self.view_box.removeItem(item)
            except Exception:
                pass
        self.review_track_label_items = []

        for label in np.unique(viz[current]):
            label = int(label)
            if label <= 0:
                continue
            yy, xx = np.nonzero(current & (viz == label))
            if yy.size == 0:
                continue
            y = float(np.mean(yy))
            x = float(np.mean(xx))
            color = _review_track_color(label)
            spots.append(
                {
                    "pos": (x, y),
                    "brush": pg.mkBrush(*color, 230),
                    "pen": pg.mkPen((255, 255, 255), width=1),
                    "size": 8,
                }
            )
            text = pg.TextItem(
                text=str(label),
                color=color,
                anchor=(0, 1),
                fill=pg.mkBrush(0, 0, 0, 150),
            )
            text.setPos(x + 5, y - 5)
            text.setZValue(42)
            self.view_box.addItem(text)
            self.review_track_label_items.append(text)
        self.review_track_points_item.setData(spots)

    def _update_association_overlay(self, segments):
        if not segments:
            self._clear_association_overlay()
            return
        line_x = []
        line_y = []
        selected_line_x = []
        selected_line_y = []
        point_x = []
        point_y = []
        selected_point_x = []
        selected_point_y = []
        for seg in segments:
            ref_y, ref_x = seg["reference_center"]
            cur_y, cur_x = seg["current_center"]
            if seg.get("selected"):
                selected_line_x.extend([ref_x, cur_x, np.nan])
                selected_line_y.extend([ref_y, cur_y, np.nan])
                selected_point_x.extend([ref_x, cur_x])
                selected_point_y.extend([ref_y, cur_y])
            else:
                line_x.extend([ref_x, cur_x, np.nan])
                line_y.extend([ref_y, cur_y, np.nan])
                point_x.extend([ref_x, cur_x])
                point_y.extend([ref_y, cur_y])
        self.association_lines_item.setData(x=line_x, y=line_y, connect="finite")
        self.association_selected_lines_item.setData(x=selected_line_x, y=selected_line_y, connect="finite")
        self.association_points_item.setData(x=point_x, y=point_y)
        self.association_selected_points_item.setData(x=selected_point_x, y=selected_point_y)

    def _blit_rgba(self, base, overlay):
        if overlay is None:
            return
        if overlay.shape[:2] != base.shape[:2]:
            _logger.warning(
                "Skipping mask overlay: image shape %s does not match mask shape %s. "
                "Re-run inference on this image to regenerate masks at the correct resolution.",
                base.shape[:2], overlay.shape[:2],
            )
            return
        mask = overlay[:, :, 3] > 0
        base[mask] = overlay[mask]

    def _render_mask_layer(self, source_masks, state, alpha_val, force_color=None, selected_ids=None, selection_color=None, color_by_label=False):
        if source_masks is None or source_masks.max() <= 0:
            return None
        max_id = int(source_masks.max())
        rgba = np.zeros(source_masks.shape + (4,), dtype=np.uint8)
        color_lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
        if force_color is not None:
            color_lut[1:] = np.asarray(force_color, dtype=np.uint8)
            vis_lut = np.ones(max_id + 1, dtype=bool)
            vis_lut[0] = False
        elif self.model.view_config.color_by_class and not color_by_label:
            mask_classes = np.asarray(state.get("mask_classes"), dtype=np.int16)
            if len(mask_classes) <= max_id:
                mask_classes = np.pad(mask_classes, (0, max_id - len(mask_classes) + 1))
            n_classes = len(self.model.class_colors)
            class_color_lut = np.zeros((n_classes + 1, 3), dtype=np.uint8)
            class_color_lut[1:] = self.model.class_colors
            safe_class_ids = np.clip(mask_classes[: max_id + 1], 0, n_classes)
            color_lut = class_color_lut[safe_class_ids]
            vis_lut = np.ones(max_id + 1, dtype=bool)
            vis_lut[0] = False
            for i, is_visible in enumerate(self.model.view_config.class_visible):
                class_id = i + 1
                if not is_visible:
                    vis_lut[mask_classes[: max_id + 1] == class_id] = False
            vis_lut[mask_classes[: max_id + 1] < 1] = False
        else:
            instance_colors = np.asarray(state.get("instance_colors"), dtype=np.uint8)
            if len(instance_colors) <= max_id:
                missing = max_id - len(instance_colors) + 1
                new_cols = np.random.randint(0, 255, (missing, 3), dtype=np.uint8)
                instance_colors = np.vstack([instance_colors, new_cols])
            color_lut = instance_colors[: max_id + 1]
            vis_lut = np.ones(max_id + 1, dtype=bool)
            vis_lut[0] = False

        rgb = color_lut[source_masks]
        is_visible = vis_lut[source_masks]
        rgba[is_visible, :3] = rgb[is_visible]
        rgba[is_visible, 3] = alpha_val

        if selected_ids:
            sel_labels = np.where(np.isin(source_masks, list(selected_ids)), source_masks, 0)
            if sel_labels.max() > 0:
                outlines = utils.masks_to_outlines(sel_labels)
                color = np.asarray(selection_color if selection_color is not None else np.array([255, 255, 255], dtype=np.uint8), dtype=np.uint8)
                rgba[outlines > 0, :3] = color
                rgba[outlines > 0, 3] = 255
        return rgba

    def display_image(self, image_data):
        self.img_item.setImage(image_data, autoLevels=False, levels=(0, 255))


def _review_track_color(track_id):
    palette = [
        (0, 180, 216),
        (255, 183, 3),
        (42, 157, 143),
        (231, 111, 81),
        (131, 118, 184),
        (239, 71, 111),
        (6, 214, 160),
        (255, 127, 80),
    ]
    return palette[(int(track_id) - 1) % len(palette)]
