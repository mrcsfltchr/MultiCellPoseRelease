from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget


class HomeView(QMainWindow):
    start_analyzer_requested = pyqtSignal()
    start_trainer_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MultiCellPose - Home")
        self.resize(720, 420)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("MultiCellPose")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self.trainer_button = QPushButton("Trainer")
        self.analyzer_button = QPushButton("Analyzer")

        for btn in (self.trainer_button, self.analyzer_button):
            btn.setMinimumSize(220, 220)
            btn.setStyleSheet(
                "QPushButton { font-size: 16px; font-weight: 600; }"
            )
            button_row.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)

        self.trainer_button.clicked.connect(self.start_trainer_requested.emit)
        self.analyzer_button.clicked.connect(self.start_analyzer_requested.emit)

    def set_analyzer_enabled(self, enabled: bool):
        self.analyzer_button.setEnabled(bool(enabled))

    def set_trainer_enabled(self, enabled: bool):
        self.trainer_button.setEnabled(bool(enabled))

    def closeEvent(self, event):
        app = QApplication.instance()
        if app is not None:
            app.quit()
        event.accept()
