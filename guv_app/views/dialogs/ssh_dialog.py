from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QDialogButtonBox, 
                             QLineEdit, QFormLayout, QSpinBox, QPushButton, 
                             QHBoxLayout, QFileDialog)

class SshLoginDialog(QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("SSH Login")
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        self.user_edit = QLineEdit()
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        
        form.addRow("Username:", self.user_edit)
        form.addRow("Password:", self.pass_edit)
        layout.addLayout(form)
        
        if config:
            self.user_edit.setText(config.username)
            self.pass_edit.setText(config.password)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_credentials(self):
        return {
            "username": self.user_edit.text(),
            "password": self.pass_edit.text()
        }

class SshAdvancedDialog(QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("SSH Advanced Configuration")
        self.resize(450, 400)
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        self.host_edit = QLineEdit()
        self.user_edit = QLineEdit()
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        
        self.key_path_edit = QLineEdit()
        self.key_browse_btn = QPushButton("Browse")
        self.key_browse_btn.clicked.connect(self.browse_key)
        key_layout = QHBoxLayout()
        key_layout.addWidget(self.key_path_edit)
        key_layout.addWidget(self.key_browse_btn)
        
        self.local_port_spin = QSpinBox()
        self.local_port_spin.setRange(1, 65535)
        self.local_port_spin.setValue(50051)
        
        self.remote_port_spin = QSpinBox()
        self.remote_port_spin.setRange(1, 65535)
        self.remote_port_spin.setValue(50051)
        
        self.remote_bind_edit = QLineEdit()
        self.remote_bind_edit.setText("localhost")
        
        if config:
            self.host_edit.setText(config.hostname)
            self.user_edit.setText(config.username)
            self.pass_edit.setText(config.password)
            self.port_spin.setValue(config.ssh_port)
            self.key_path_edit.setText(config.key_path)
            self.local_port_spin.setValue(config.ssh_local_port)
            self.remote_port_spin.setValue(config.ssh_remote_port)
            self.remote_bind_edit.setText(config.ssh_remote_bind)
        
        form.addRow("Hostname:", self.host_edit)
        form.addRow("Username:", self.user_edit)
        form.addRow("Password:", self.pass_edit)
        form.addRow("SSH Port:", self.port_spin)
        form.addRow("Identity File:", key_layout)
        form.addRow("Local Port (Tunnel):", self.local_port_spin)
        form.addRow("Remote Port (Service):", self.remote_port_spin)
        form.addRow("Remote Bind Address:", self.remote_bind_edit)
        
        layout.addLayout(form)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_key(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Identity File")
        if filename:
            self.key_path_edit.setText(filename)

    def get_credentials(self):
        return {
            "host": self.host_edit.text(),
            "username": self.user_edit.text(),
            "password": self.pass_edit.text(),
            "port": self.port_spin.value(),
            "key_path": self.key_path_edit.text(),
            "ssh_local_port": self.local_port_spin.value(),
            "ssh_remote_port": self.remote_port_spin.value(),
            "ssh_remote_bind": self.remote_bind_edit.text()
        }
