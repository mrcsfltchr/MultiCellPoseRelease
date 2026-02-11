from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QDialogButtonBox, 
                             QWidget, QGroupBox, QScrollArea, QLabel,
                             QFormLayout, QSpinBox, QDoubleSpinBox, 
                             QCheckBox, QLineEdit, QComboBox)

class DynamicPluginConfigWidget(QWidget):
    """
    A widget that automatically generates UI controls based on a parameter definition dictionary.
    """
    def __init__(self, definitions):
        super().__init__()
        self.definitions = definitions
        self.widgets = {}
        layout = QFormLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        for key, param in definitions.items():
            label_text = param.get('label', key)
            ptype = param.get('type', 'str')
            default = param.get('default')
            tooltip = param.get('help', '')
            
            widget = None
            if ptype == 'int':
                widget = QSpinBox()
                widget.setRange(param.get('min', -999999), param.get('max', 999999))
                if default is not None: widget.setValue(int(default))
            elif ptype == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(param.get('min', -float('inf')), param.get('max', float('inf')))
                if default is not None: widget.setValue(float(default))
            elif ptype == 'bool':
                widget = QCheckBox()
                if default is not None: widget.setChecked(bool(default))
            elif ptype == 'enum':
                widget = QComboBox()
                options = param.get('options', [])
                widget.addItems([str(o) for o in options])
                if default in options:
                    widget.setCurrentText(str(default))
            else: # str
                widget = QLineEdit()
                if default is not None: widget.setText(str(default))
            
            if widget:
                widget.setToolTip(tooltip)
                self.widgets[key] = widget
                layout.addRow(label_text, widget)

    def get_values(self):
        values = {}
        for key, widget in self.widgets.items():
            ptype = self.definitions[key].get('type', 'str')
            if ptype == 'int' or ptype == 'float':
                values[key] = widget.value()
            elif ptype == 'bool':
                values[key] = widget.isChecked()
            elif ptype == 'enum':
                values[key] = widget.currentText()
            else:
                values[key] = widget.text()
        return values

class PluginConfigDialog(QDialog):
    """
    Dialog for selecting analysis plugins and configuring their parameters.
    """
    def __init__(self, plugins, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Analysis Plugins")
        self.resize(400, 500)
        self.plugins = plugins # dict name -> instance
        self.plugin_widgets = {} # name -> widget
        self.checkboxes = {} # name -> groupbox (which acts as checkbox)
        
        self.layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        
        for name, plugin in self.plugins.items():
            # Use a checkable GroupBox for each plugin
            group = QGroupBox(name)
            group.setCheckable(True)
            group.setChecked(True) # Default to checked
            self.checkboxes[name] = group
            
            group_layout = QVBoxLayout(group)
            
            definitions = plugin.get_parameter_definitions()
            if definitions:
                widget = DynamicPluginConfigWidget(definitions)
                group_layout.addWidget(widget)
                self.plugin_widgets[name] = widget
            else:
                group_layout.addWidget(QLabel("No settings available."))
            
            self.content_layout.addWidget(group)
            
        self.content_layout.addStretch()
        scroll.setWidget(content)
        self.layout.addWidget(scroll)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def get_configuration(self):
        """Returns (selected_plugins_list, plugin_params_dict)"""
        selected = []
        params = {}
        for name, group in self.checkboxes.items():
            if group.isChecked():
                selected.append(self.plugins[name])
                if name in self.plugin_widgets:
                    widget = self.plugin_widgets[name]
                    if hasattr(widget, 'get_values'):
                        params[name] = widget.get_values()
        return selected, params
