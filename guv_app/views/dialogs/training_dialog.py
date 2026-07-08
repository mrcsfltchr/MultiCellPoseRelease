import os
import math
from typing import List, Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from guv_app.data_models.configs import TrainingConfig


class TrainingConfigDialog(QDialog):
    def __init__(
        self,
        model_names: List[str],
        default_config: TrainingConfig,
        train_files: Optional[List[str]] = None,
        train_image_shapes: Optional[List[tuple]] = None,
        total_blocks: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Training settings")
        self._default_config = default_config

        layout = QVBoxLayout(self)
        train_files = train_files or []
        self._train_image_shapes = train_image_shapes or []
        self._train_count = max(1, len(train_files))
        info = QLabel(f"{len(train_files)} training images with _seg.npy found.")
        layout.addWidget(info)

        self.tabs = QTabWidget()
        basic_tab = QWidget()
        advanced_tab = QWidget()
        basic_form = QFormLayout()
        advanced_form = QFormLayout()

        self.base_model_combo = QComboBox()
        models = model_names if model_names else [default_config.base_model]
        self.base_model_combo.addItems(models)
        if default_config.base_model in models:
            self.base_model_combo.setCurrentText(default_config.base_model)
        advanced_form.addRow("Base model:", self.base_model_combo)

        self.model_name_edit = QLineEdit(default_config.model_name)
        basic_form.addRow("Model name:", self.model_name_edit)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(7)
        self.learning_rate_spin.setRange(0.0, 1.0)
        self.learning_rate_spin.setSingleStep(1e-5)
        self.learning_rate_spin.setValue(default_config.learning_rate)
        advanced_form.addRow("Learning rate:", self.learning_rate_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(1e-4)
        self.weight_decay_spin.setValue(default_config.weight_decay)
        advanced_form.addRow("Weight decay:", self.weight_decay_spin)

        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setRange(1, 100000)
        self.n_epochs_spin.setValue(default_config.n_epochs)
        basic_form.addRow("Epochs:", self.n_epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(default_config.batch_size)
        advanced_form.addRow("Batch size:", self.batch_size_spin)

        self.bsize_spin = QSpinBox()
        self.bsize_spin.setRange(64, 2048)
        self.bsize_spin.setSingleStep(32)
        self.bsize_spin.setValue(default_config.bsize)
        advanced_form.addRow("Crop size (bsize):", self.bsize_spin)

        self.nimg_per_epoch_spin = QSpinBox()
        self.nimg_per_epoch_spin.setRange(1, 10_000_000)
        suggested = (
            int(default_config.nimg_per_epoch)
            if default_config.nimg_per_epoch is not None
            else self._suggest_nimg_per_epoch(int(self.bsize_spin.value()))
        )
        self.nimg_per_epoch_spin.setValue(max(1, suggested))
        advanced_form.addRow("Images/crops per epoch:", self.nimg_per_epoch_spin)
        self.crop_capacity_label = QLabel("")
        advanced_form.addRow("Estimated crop capacity:", self.crop_capacity_label)

        self.rescale_checkbox = QCheckBox("Enable rescale by diameter")
        self.rescale_checkbox.setChecked(default_config.rescale)
        advanced_form.addRow("Rescale:", self.rescale_checkbox)

        self.scale_range_spin = QDoubleSpinBox()
        self.scale_range_spin.setDecimals(2)
        self.scale_range_spin.setRange(0.0, 2.0)
        self.scale_range_spin.setSingleStep(0.05)
        self.scale_range_spin.setValue(default_config.scale_range)
        advanced_form.addRow("Scale range:", self.scale_range_spin)

        self.seg_loss_weight_spin = QDoubleSpinBox()
        self.seg_loss_weight_spin.setDecimals(3)
        self.seg_loss_weight_spin.setRange(0.0, 10.0)
        self.seg_loss_weight_spin.setSingleStep(0.05)
        self.seg_loss_weight_spin.setValue(float(default_config.seg_loss_weight))
        advanced_form.addRow("Segmentation loss weight:", self.seg_loss_weight_spin)

        self.class_loss_weight_spin = QDoubleSpinBox()
        self.class_loss_weight_spin.setDecimals(3)
        self.class_loss_weight_spin.setRange(0.0, 100.0)
        self.class_loss_weight_spin.setSingleStep(0.5)
        self.class_loss_weight_spin.setValue(float(default_config.class_loss_weight))
        advanced_form.addRow("Class loss weight:", self.class_loss_weight_spin)

        self.min_masks_spin = QSpinBox()
        self.min_masks_spin.setRange(0, 100000)
        self.min_masks_spin.setValue(default_config.min_train_masks)
        advanced_form.addRow("Min masks per image:", self.min_masks_spin)

        max_blocks = 24
        self.unfreeze_blocks_spin = QSpinBox()
        self.unfreeze_blocks_spin.setRange(0, max_blocks)
        self.unfreeze_blocks_spin.setValue(default_config.unfreeze_blocks)
        advanced_form.addRow("Unfreeze blocks:", self.unfreeze_blocks_spin)

        self.use_lora_checkbox = QCheckBox("Use LoRA")
        self.use_lora_checkbox.setChecked(default_config.use_lora)
        advanced_form.addRow("LoRA:", self.use_lora_checkbox)

        self.lora_blocks_spin = QSpinBox()
        self.lora_blocks_spin.setRange(0, max_blocks)
        default_lora_blocks = (
            default_config.lora_blocks
            if default_config.lora_blocks is not None
            else default_config.unfreeze_blocks
        )
        self.lora_blocks_spin.setValue(int(default_lora_blocks))
        advanced_form.addRow("LoRA blocks to inject:", self.lora_blocks_spin)
        self.use_lora_checkbox.toggled.connect(self._on_lora_toggled)
        self._on_lora_toggled(self.use_lora_checkbox.isChecked())

        self.save_path_edit = QLineEdit(default_config.save_path or "")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_save_path)
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(browse_btn)
        basic_form.addRow("Save folder:", save_layout)

        basic_tab.setLayout(basic_form)
        advanced_tab.setLayout(advanced_form)
        self.tabs.addTab(basic_tab, "Basic")
        self.tabs.addTab(advanced_tab, "Advanced")
        self.tabs.setTabVisible(1, False)
        layout.addWidget(self.tabs)

        self.advanced_toggle = QPushButton("Show Advanced Settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced_tab)
        layout.addWidget(self.advanced_toggle)
        self.bsize_spin.valueChanged.connect(self._refresh_crop_capacity)
        self._refresh_crop_capacity()

        if train_files:
            layout.addWidget(QLabel("Training files (first 10):"))
            file_list = QListWidget()
            for path in train_files[:10]:
                file_list.addItem(os.path.basename(path))
            if len(train_files) > 10:
                file_list.addItem("...")
            file_list.setMinimumHeight(120)
            layout.addWidget(file_list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_save_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.save_path_edit.setText(folder)

    def _on_lora_toggled(self, checked: bool):
        self.lora_blocks_spin.setEnabled(bool(checked))
        self.unfreeze_blocks_spin.setEnabled(not bool(checked))

    def _toggle_advanced_tab(self, checked: bool):
        self.tabs.setTabVisible(1, bool(checked))
        self.advanced_toggle.setText(
            "Hide Advanced Settings" if checked else "Show Advanced Settings"
        )
        if checked:
            self.tabs.setCurrentIndex(1)
        else:
            self.tabs.setCurrentIndex(0)

    def _estimate_total_possible_crops(self, bsize: int) -> Optional[int]:
        if not self._train_image_shapes:
            return None
        total = 0
        for shape in self._train_image_shapes:
            if not shape:
                continue
            h = int(shape[0]) if len(shape) >= 1 else 0
            w = int(shape[1]) if len(shape) >= 2 else 0
            if h <= 0 or w <= 0:
                continue
            total += max(1, math.ceil(h / bsize) * math.ceil(w / bsize))
        return total if total > 0 else None

    def _suggest_nimg_per_epoch(self, bsize: int) -> int:
        total_crops = self._estimate_total_possible_crops(bsize)
        if total_crops is None:
            return self._train_count
        return max(self._train_count, int(round(0.25 * total_crops)))

    def _refresh_crop_capacity(self):
        bsize = int(self.bsize_spin.value())
        total_crops = self._estimate_total_possible_crops(bsize)
        if total_crops is None:
            self.crop_capacity_label.setText("unknown")
            return
        suggested = self._suggest_nimg_per_epoch(bsize)
        self.crop_capacity_label.setText(
            f"{total_crops} (@ bsize={bsize}), suggested default={suggested}"
        )

    def get_config(self) -> TrainingConfig:
        model_name = self.model_name_edit.text().strip() or self._default_config.model_name
        save_path = self.save_path_edit.text().strip() or None
        return TrainingConfig(
            base_model=self.base_model_combo.currentText(),
            model_name=model_name,
            learning_rate=float(self.learning_rate_spin.value()),
            weight_decay=float(self.weight_decay_spin.value()),
            n_epochs=int(self.n_epochs_spin.value()),
            batch_size=int(self.batch_size_spin.value()),
            bsize=int(self.bsize_spin.value()),
            nimg_per_epoch=int(self.nimg_per_epoch_spin.value()),
            rescale=bool(self.rescale_checkbox.isChecked()),
            scale_range=float(self.scale_range_spin.value()),
            seg_loss_weight=float(self.seg_loss_weight_spin.value()),
            class_loss_weight=float(self.class_loss_weight_spin.value()),
            min_train_masks=int(self.min_masks_spin.value()),
            use_lora=bool(self.use_lora_checkbox.isChecked()),
            lora_blocks=int(self.lora_blocks_spin.value()),
            unfreeze_blocks=int(self.unfreeze_blocks_spin.value()),
            save_path=save_path,
        )
