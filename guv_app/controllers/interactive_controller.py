from PyQt6.QtCore import QObject, pyqtSlot

class InteractiveController(QObject):
    def __init__(self, model, view, services):
        super().__init__()
        self.model = model
        self.view = view
        self.services = services
        
        # Connect signals from the view
        self.view.control_panel.run_current_button.clicked.connect(self.run_segmentation)
        self.view.drawing_widget.stroke_finished.connect(self.add_mask)
        self.view.control_panel.class_visibility_changed.connect(self.set_class_visibility)
        self.view.control_panel.add_class_button.clicked.connect(self.add_class)
        self.view.control_panel.remove_class_button.clicked.connect(self.remove_class)
        self.view.control_panel.class_dropdown.currentIndexChanged.connect(self.set_active_class)

    @pyqtSlot()
    def run_segmentation(self):
        # To be implemented
        pass

    @pyqtSlot(list)
    def add_mask(self, stroke):
        # To be implemented
        pass

    @pyqtSlot(int, bool)
    def set_class_visibility(self, class_index, visible):
        # To be implemented
        pass

    @pyqtSlot()
    def add_class(self):
        # To be implemented
        pass

    @pyqtSlot()
    def remove_class(self):
        # To be implemented
        pass

    @pyqtSlot(int)
    def set_active_class(self, class_index):
        # To be implemented
        pass
