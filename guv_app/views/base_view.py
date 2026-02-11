import os
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
    toggle_masks_requested = pyqtSignal()
    toggle_outlines_requested = pyqtSignal()
    toggle_color_mode_requested = pyqtSignal()
    toggle_visualization_requested = pyqtSignal()
    brush_size_change_requested = pyqtSignal(int)
    view_mode_step_requested = pyqtSignal(int)
    color_mode_step_requested = pyqtSignal(int)
    color_mode_set_requested = pyqtSignal(int)
    finalize_stroke_requested = pyqtSignal()

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
        self.view_box.addItem(self.img_item)
        
        # Drawing Item (Masks)
        self.drawing_item = DrawingItem(parent=self)
        self.view_box.addItem(self.drawing_item)
        
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
        self._add_shortcut(QKeySequence("C"), self._toggle_color_mode_shortcut)
        self._add_shortcut(QKeySequence("K"), self._toggle_visualization_shortcut)
        self._add_shortcut(QKeySequence("D"), self._toggle_delete_lasso_shortcut)
        self._add_shortcut(QKeySequence(Qt.Key.Key_Left), self._navigate_prev_shortcut)
        self._add_shortcut(QKeySequence(Qt.Key.Key_Right), self._navigate_next_shortcut)
        self._add_shortcut(QKeySequence("A"), self._navigate_prev_shortcut)
        self._add_shortcut(QKeySequence(","), lambda: self._brush_size_shortcut(-1))
        self._add_shortcut(QKeySequence("."), lambda: self._brush_size_shortcut(1))
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

    def _toggle_delete_lasso_shortcut(self):
        if self._should_ignore_shortcut():
            return
        if hasattr(self, "control_panel") and hasattr(self.control_panel, "delete_lasso_button"):
            self.control_panel.delete_lasso_button.toggle()

    def on_load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.npy *.nd2 *.lif *.dax *.nrrd)")
        if filename:
            self.file_loaded.emit(filename)

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
        # Refresh image/masks from model
        if self.model.image_data is not None:
            self.display_image(self.model.image_data)
            
        # Update DrawingItem with masks or empty RGBA buffer
        if self.model.image_data is not None:
            h, w = self.model.image_data.shape[:2]
            
            # Create RGBA buffer for masks/drawing
            # For now, just a transparent layer. 
            mask_display = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Render masks if available (allow visualization even if cellpix is missing)
            has_viz = self.model.visualization_masks is not None and self.model.view_config.show_visualization
            if (self.model.cellpix is not None or has_viz) and (
                self.model.view_config.masks_visible
                or self.model.view_config.outlines_visible
                or has_viz
            ):
                # Use current Z plane (assuming 0 for 2D for now)
                z = 0 
                if self.model.cellpix is None or z < self.model.cellpix.shape[0]:
                    # 1. Select source array
                    source_masks = None
                    is_outline_mode = False
                    
                    if has_viz:
                        # Visualization takes precedence over standard masks/outlines
                        if self.model.visualization_masks.ndim == 3:
                             if z < self.model.visualization_masks.shape[0]:
                                 source_masks = self.model.visualization_masks[z]
                        else:
                             source_masks = self.model.visualization_masks
                    else:
                        # Standard rendering logic
                        if self.model.cellpix is not None:
                            source_masks = self.model.cellpix[z]
                        
                        if not self.model.view_config.masks_visible and self.model.view_config.outlines_visible:
                            if self.model.outpix is not None:
                                source_masks = self.model.outpix[z]
                                is_outline_mode = True
                        elif not self.model.view_config.masks_visible:
                            # Neither visible
                            source_masks = None

                    if source_masks is not None and source_masks.max() > 0:
                        max_id = source_masks.max()
                        
                        # 2. Prepare Color Lookup Table (LUT)
                        # LUT size: max_id + 1. Index 0 is background.
                        color_lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
                        
                        if self.model.view_config.color_by_class:
                            # Map Mask ID -> Class ID -> Color
                            mask_classes = self.model.mask_classes
                            # Ensure mask_classes covers all mask IDs
                            if len(mask_classes) <= max_id:
                                mask_classes = np.pad(mask_classes, (0, max_id - len(mask_classes) + 1))
                            
                            # Create Class -> Color LUT
                            n_classes = len(self.model.class_colors)
                            class_color_lut = np.zeros((n_classes + 1, 3), dtype=np.uint8)
                            class_color_lut[1:] = self.model.class_colors # Index 1..N
                            
                            # Map mask IDs to Class IDs, then to Colors
                            # Clip class IDs to ensure they are within range of class_color_lut
                            safe_class_ids = np.clip(mask_classes[:max_id+1], 0, n_classes)
                            color_lut = class_color_lut[safe_class_ids]
                        else:
                            # Map Mask ID -> Instance Color
                            instance_colors = self.model.instance_colors
                            if len(instance_colors) <= max_id:
                                # Generate missing colors on the fly if needed (though model should handle this)
                                missing = max_id - len(instance_colors) + 1
                                new_cols = np.random.randint(0, 255, (missing, 3), dtype=np.uint8)
                                instance_colors = np.vstack([instance_colors, new_cols])
                                self.model.instance_colors = instance_colors
                            color_lut = instance_colors[:max_id+1]

                        # 3. Prepare Visibility Lookup Table
                        # Mask ID -> Visible (bool)
                        vis_lut = np.ones(max_id + 1, dtype=bool)
                        vis_lut[0] = False # Background always invisible
                        
                        # Apply class visibility
                        mask_classes = self.model.mask_classes
                        if len(mask_classes) <= max_id:
                             # Use -1 for unknown/unassigned class IDs so they can be hidden.
                             mask_classes = np.pad(
                                 mask_classes,
                                 (0, max_id - len(mask_classes) + 1),
                                 constant_values=-1
                             )
                        
                        class_vis_config = self.model.view_config.class_visible
                        # Iterate classes to disable invisible ones
                        for i, is_visible in enumerate(class_vis_config):
                            class_id = i + 1
                            if not is_visible:
                                # Find masks belonging to this class and hide them
                                # Note: This is a boolean array operation on the LUT
                                vis_lut[mask_classes[:max_id+1] == class_id] = False
                        # Hide any mask IDs that have invalid/unassigned class IDs.
                        invalid_class = mask_classes[:max_id+1] < 1
                        vis_lut[invalid_class] = False

                        # 4. Render
                        # Apply visibility to LUT (set invisible colors to 0)
                        # We handle alpha separately
                        
                        # Get pixels
                        mask_pixels = source_masks
                        
                        # Look up colors
                        rgb = color_lut[mask_pixels]
                        
                        # Look up visibility per pixel
                        is_visible = vis_lut[mask_pixels]
                        
                        # Assign to display buffer
                        mask_display[is_visible, :3] = rgb[is_visible]
                        
                        # Set Alpha
                        alpha_val = 255 if is_outline_mode else 100
                        mask_display[is_visible, 3] = alpha_val

                        selected_ids = self.model.get_selected_mask_ids() if hasattr(self.model, "get_selected_mask_ids") else set()
                        if selected_ids:
                            ref_masks = self.model.cellpix[z] if self.model.cellpix is not None else source_masks
                            if ref_masks is None:
                                ref_masks = source_masks
                            sel_labels = np.where(np.isin(ref_masks, list(selected_ids)), ref_masks, 0)
                            if sel_labels.max() > 0:
                                outlines = utils.masks_to_outlines(sel_labels)
                                mask_display[outlines > 0, :3] = 255
                                mask_display[outlines > 0, 3] = 255
            
            self.drawing_item.setImage(mask_display, autoLevels=False)
            self.drawing_item.setLevels([0, 255])

    def display_image(self, image_data):
        self.img_item.setImage(image_data, autoLevels=False, levels=(0, 255))
