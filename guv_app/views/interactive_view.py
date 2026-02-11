from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QProgressBar
from .widgets.drawing import DrawingWidget
from .widgets.control_panel import ControlPanel

class InteractiveView(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MultiCellPose - Interactive")
        
        self.drawing_widget = DrawingWidget()
        self.control_panel = ControlPanel()
        
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        layout.addWidget(self.drawing_widget)
        layout.addWidget(self.control_panel)
        
        self.setCentralWidget(main_widget)
        
        # Add a progress bar to the status bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
