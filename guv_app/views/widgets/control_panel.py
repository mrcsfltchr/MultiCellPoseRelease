try:
    from PyQt6.QtWidgets import (QWidget, QGridLayout, QCheckBox, QComboBox, QLineEdit, 
                                 QPushButton, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox, 
                                 QVBoxLayout, QHBoxLayout, QColorDialog, QScrollArea)
    from PyQt6.QtCore import pyqtSignal, Qt
    from PyQt6.QtGui import QColor
except ImportError:
    # Define fallback classes for non-GUI environments
    QWidget = object
    QScrollArea = object
    QGridLayout = object
    QVBoxLayout = object
    QHBoxLayout = object
    QCheckBox = object
    QComboBox = object
    QLineEdit = object
    QPushButton = object
    QLabel = object
    QGroupBox = object
    QSpinBox = object
    QDoubleSpinBox = object
    QColorDialog = object
    QColor = object
    pyqtSignal = lambda *args, **kwargs: (lambda **kwargs: None)
    Qt = type('Qt', (object,), {'CheckState': type('CheckState', (object,), {'Checked': 2})})()

try:
    from superqt import QRangeSlider
except ImportError:
    QRangeSlider = None

class ControlPanel(QScrollArea):
    """
    A widget to hold UI controls like checkboxes for toggling views and managing classes.
    """
    class_visibility_changed = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.content_widget = QWidget()
        self.setWidget(self.content_widget)
        self.layout = QGridLayout(self.content_widget)
        self.visibility_checkboxes = []
        self.current_new_class_color = [255, 0, 0] # Default red
        
        # --- View Toggles ---
        self.masks_checkbox = QCheckBox("Show Masks")
        self.masks_checkbox.setChecked(True)
        self.layout.addWidget(self.masks_checkbox, 0, 0, 1, 2)
        
        self.outlines_checkbox = QCheckBox("Show Outlines")
        self.outlines_checkbox.setChecked(False)
        self.layout.addWidget(self.outlines_checkbox, 1, 0, 1, 2)

        self.autosave_checkbox = QCheckBox("Enable Autosave")
        self.autosave_checkbox.setChecked(True)
        self.layout.addWidget(self.autosave_checkbox, 2, 0, 1, 2)

        # --- Class Management ---
        class_box = QGroupBox("Class Management")
        class_layout = QGridLayout()
        class_box.setLayout(class_layout)
        self.layout.addWidget(class_box, 3, 0, 1, 2)

        class_layout.addWidget(QLabel("Current Class:"), 0, 0)
        self.class_dropdown = QComboBox()
        class_layout.addWidget(self.class_dropdown, 0, 1)

        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("New class name")
        class_layout.addWidget(self.new_class_edit, 1, 0, 1, 2)

        # New Class Options Row
        new_class_options = QHBoxLayout()
        
        self.new_class_color_btn = QPushButton()
        self.new_class_color_btn.setFixedWidth(40)
        self.new_class_color_btn.setToolTip("Select Color for New Class")
        self.update_color_button_style()
        
        self.new_class_visible_chk = QCheckBox("Visible")
        self.new_class_visible_chk.setChecked(True)
        self.new_class_visible_chk.setToolTip("Make new class visible immediately")

        self.add_class_button = QPushButton("Add")
        
        new_class_options.addWidget(self.new_class_color_btn)
        new_class_options.addWidget(self.new_class_visible_chk)
        new_class_options.addWidget(self.add_class_button)
        
        class_layout.addLayout(new_class_options, 2, 0, 1, 2)
        
        self.remove_class_button = QPushButton("Remove Selected")
        class_layout.addWidget(self.remove_class_button, 3, 0, 1, 2)
        
        self.class_visibility_layout = QGridLayout()
        class_layout.addLayout(self.class_visibility_layout, 4, 0, 1, 2)

        self.delete_lasso_button = QPushButton("Delete Masks (Lasso)")
        self.delete_lasso_button.setCheckable(True)
        class_layout.addWidget(self.delete_lasso_button, 5, 0, 1, 2)


        # --- View Options ---
        view_box = QGroupBox("View Options")
        view_layout = QGridLayout()
        view_box.setLayout(view_layout)
        self.layout.addWidget(view_box, 4, 0, 1, 2)

        view_layout.addWidget(QLabel("View Mode:"), 0, 0)
        self.view_mode_dropdown = QComboBox()
        self.view_mode_dropdown.addItems(["Image", "GradXY", "Cellprob"])
        view_layout.addWidget(self.view_mode_dropdown, 0, 1)

        view_layout.addWidget(QLabel("Color Mode:"), 1, 0)
        self.color_mode_dropdown = QComboBox()
        self.color_mode_dropdown.addItems(["RGB", "Red", "Green", "Blue", "Gray", "Spectral"])
        view_layout.addWidget(self.color_mode_dropdown, 1, 1)

        self.color_by_class_checkbox = QCheckBox("Color by Class")
        self.color_by_class_checkbox.setChecked(True)
        view_layout.addWidget(self.color_by_class_checkbox, 2, 0, 1, 2)

        self.visualization_checkbox = QCheckBox("Show Visualization")
        self.visualization_checkbox.setChecked(True)
        view_layout.addWidget(self.visualization_checkbox, 3, 0, 1, 2)

        # --- Image Levels ---
        if QRangeSlider:
            levels_box = QGroupBox("Image Levels")
            levels_layout = QVBoxLayout()
            levels_box.setLayout(levels_layout)
            self.layout.addWidget(levels_box, 5, 0, 1, 2)
            
            self.sliders = []
            self.slider_labels = []
            colors = ["Red", "Green", "Blue"]
            for i, color in enumerate(colors):
                label = QLabel(f"{color} Channel")
                slider = QRangeSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 255) # Default, will be updated
                levels_layout.addWidget(label)
                levels_layout.addWidget(slider)
                self.sliders.append(slider)
                self.slider_labels.append(label)

        # --- Segmentation ---
        seg_box = QGroupBox("Segmentation")
        seg_layout = QGridLayout()
        seg_box.setLayout(seg_layout)
        self.layout.addWidget(seg_box, 6, 0, 1, 2)

        seg_layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_dropdown = QComboBox()
        seg_layout.addWidget(self.model_dropdown, 0, 1)

        seg_layout.addWidget(QLabel("Diameter:"), 1, 0)
        self.diameter_spinbox = QDoubleSpinBox()
        self.diameter_spinbox.setRange(0, 500)
        self.diameter_spinbox.setValue(30)
        seg_layout.addWidget(self.diameter_spinbox, 1, 1)

        seg_layout.addWidget(QLabel("Inference Channel:"), 2, 0)
        self.inference_channel_dropdown = QComboBox()
        self.inference_channel_dropdown.addItems(["Auto"])
        seg_layout.addWidget(self.inference_channel_dropdown, 2, 1)

        self.run_current_button = QPushButton("Run Inference on Current Image")
        seg_layout.addWidget(self.run_current_button, 3, 0, 1, 2)

        self.run_series_button = QPushButton("Run on Multi-Image")
        self.run_series_button.setEnabled(False)
        seg_layout.addWidget(self.run_series_button, 4, 0, 1, 2)

        self.run_folder_button = QPushButton("Run on Folder")
        seg_layout.addWidget(self.run_folder_button, 5, 0, 1, 2)

        self.cancel_local_button = QPushButton("Cancel Local Inference")
        self.cancel_local_button.setEnabled(False)
        seg_layout.addWidget(self.cancel_local_button, 6, 0, 1, 2)

    def update_color_button_style(self):
        r, g, b = self.current_new_class_color
        self.new_class_color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid white;")

    def set_new_class_color(self, r, g, b):
        self.current_new_class_color = [r, g, b]
        self.update_color_button_style()
        
    def set_slider_label(self, index, text):
        if hasattr(self, 'slider_labels') and 0 <= index < len(self.slider_labels):
            self.slider_labels[index].setText(text)

    def update_model_list(self, model_names, current_index=0):
        self.model_dropdown.blockSignals(True)
        self.model_dropdown.clear()
        self.model_dropdown.addItems(model_names)
        if 0 <= current_index < len(model_names):
            self.model_dropdown.setCurrentIndex(current_index)
        self.model_dropdown.blockSignals(False)

    def update_class_list(self, class_names, current_class_idx=0):
        self.class_dropdown.blockSignals(True)
        self.class_dropdown.clear()
        self.class_dropdown.addItems(class_names)
        self.class_dropdown.setCurrentIndex(current_class_idx)
        self.class_dropdown.blockSignals(False)

    def update_class_visibility_ui(self, class_names, visibility_states):
        for cb in self.visibility_checkboxes:
            self.class_visibility_layout.removeWidget(cb)
            cb.deleteLater()
        self.visibility_checkboxes = []

        for i, name in enumerate(class_names):
            cb = QCheckBox(name)
            is_visible = visibility_states[i] if i < len(visibility_states) else True
            cb.setChecked(is_visible)
            cb.stateChanged.connect(lambda state, index=i: self.class_visibility_changed.emit(index, Qt.CheckState(state) == Qt.CheckState.Checked))
            self.visibility_checkboxes.append(cb)
            self.class_visibility_layout.addWidget(cb, i, 0)

    def update_inference_channel_options(self, channel_count):
        channel_count = max(1, int(channel_count))
        current_text = self.inference_channel_dropdown.currentText()
        self.inference_channel_dropdown.blockSignals(True)
        self.inference_channel_dropdown.clear()
        self.inference_channel_dropdown.addItem("Auto")
        for idx in range(channel_count):
            self.inference_channel_dropdown.addItem(f"Channel {idx + 1}")
        if current_text:
            match_idx = self.inference_channel_dropdown.findText(current_text)
            if match_idx >= 0:
                self.inference_channel_dropdown.setCurrentIndex(match_idx)
        self.inference_channel_dropdown.blockSignals(False)

    def get_inference_channel_index(self):
        idx = self.inference_channel_dropdown.currentIndex()
        if idx <= 0:
            return None
        return idx - 1
