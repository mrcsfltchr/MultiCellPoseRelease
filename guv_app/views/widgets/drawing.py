import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QGraphicsRectItem
import numpy as np

class DrawingItem(pg.ImageItem):
    """
    Graphics item for handling drawing interactions (masks, strokes).
    """
    stroke_finished = QtCore.pyqtSignal(object)
    assign_class_requested = QtCore.pyqtSignal(int, int)
    delete_mask_requested = QtCore.pyqtSignal(int, int)
    delete_stroke_finished = QtCore.pyqtSignal(object)
    selection_rect_finished = QtCore.pyqtSignal(object)
    clear_selection_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        # FIX: Do NOT pass 'parent' (QMainWindow) to QGraphicsObject constructor.
        super().__init__()
        self.parent_view = parent 
        
        # Enable mouse interaction
        self.setAcceptHoverEvents(True)
        
        # State for drawing
        self.current_stroke = []
        self.in_stroke = False
        self.brush_size = 3
        self.setDrawKernel(self.brush_size)
        self.scatter = None
        self.stroke_appended = False
        self.delete_mode = False
        self._select_start = None
        self._selecting = False
        self._selection_rect = None

    def mouseClickEvent(self, ev):
        ev.accept() # Prevent ViewBox menu from opening on right-click

        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.in_stroke:
                self.end_stroke()
            else:
                self.create_start(ev.pos())
                self.in_stroke = True
                self.stroke_appended = False
                self.current_stroke = []
                self.drawAt(ev.pos())
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = ev.pos()
            y, x = int(pos.y()), int(pos.x())
            
            # Handle modifier keys for actions defined in MainController
            if not self.in_stroke:
                if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                     self.delete_mask_requested.emit(y, x)
                elif ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                     self.assign_class_requested.emit(y, x)
                else:
                     self.clear_selection_requested.emit()

    def mouseDragEvent(self, ev):
        ev.accept()
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if ev.isStart():
            self._select_start = ev.pos()
            self._selecting = True
            self._ensure_selection_rect()
            self._update_selection_rect(self._select_start, self._select_start)
        elif ev.isFinish():
            if self._select_start is None:
                return
            end = ev.pos()
            y0, x0 = int(self._select_start.y()), int(self._select_start.x())
            y1, x1 = int(end.y()), int(end.x())
            self._select_start = None
            self._selecting = False
            self._clear_selection_rect()
            self.selection_rect_finished.emit((y0, x0, y1, x1))
        else:
            if self._select_start is None:
                return
            self._ensure_selection_rect()
            self._update_selection_rect(self._select_start, ev.pos())
        
    def hoverEvent(self, ev):
        if ev.isExit():
            return
        if self.in_stroke:
            self.drawAt(ev.pos())
            if self.is_at_start(ev.pos()):
                self.end_stroke()
        else:
            # Pass through hover if not drawing (optional)
            pass

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                          pen=pg.mkPen(color=(255, 0, 0),
                                                       width=self.brush_size),
                                          size=max(3 * 2,
                                                   self.brush_size * 1.8 * 2),
                                          brush=None)
        self.getViewBox().addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.brush_size * 3)
        thresh_in = max(3, self.brush_size * 1.8)
        # first check if you ever left the start
        if len(self.current_stroke) > 3:
            stroke = np.array(self.current_stroke)
            # Calculate distance from start (first point)
            # stroke is list of [y, x]
            start_pt = stroke[0]
            # current pos
            curr_pt = np.array([pos.y(), pos.x()])
            
            # Check if we are far enough to consider "left"
            # This is a simplified check compared to original which checks history
            dist = np.linalg.norm(stroke - start_pt, axis=1)
            has_left = np.any(dist > thresh_out)
            
            if has_left:
                # Check distance of current point to start
                curr_dist = np.linalg.norm(curr_pt - start_pt)
                if curr_dist < thresh_in:
                    return True
        return False

    def end_stroke(self):
        if self.scatter is not None:
            self.getViewBox().removeItem(self.scatter)
            self.scatter = None
        
        if not self.stroke_appended and len(self.current_stroke) > 0:
            if self.delete_mode:
                self.delete_stroke_finished.emit(self.current_stroke)
            else:
                self.stroke_finished.emit(self.current_stroke)
            self.stroke_appended = True
            
        self.current_stroke = []
        self.in_stroke = False

    def set_delete_mode(self, enabled):
        self.delete_mode = bool(enabled)


    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs, bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [
            int(np.floor(kernel.shape[0] / 2)),
            int(np.floor(kernel.shape[1] / 2))
        ]
        # Create a red mask for drawing feedback
        # RGBA: Red=255, Green=0, Blue=0, Alpha=100 (semi-transparent)
        self.strokemask = np.zeros((bs, bs, 4), dtype=np.uint8)
        self.strokemask[:, :, 0] = 255 
        self.strokemask[:, :, 3] = 100

    def set_brush_size(self, size):
        size = max(1, int(size))
        self.brush_size = size
        self.setDrawKernel(self.brush_size)

    def drawAt(self, pos):
        if self.image is None:
            return
            
        y, x = int(pos.y()), int(pos.x())
        self.current_stroke.append([y, x])
        
        # Visual feedback: draw into the image buffer
        # Note: This modifies the display image temporarily. 
        # The controller should refresh the view with the real mask after stroke finishes.
        
        dk = self.drawKernel
        kc = self.drawKernelCenter
        h, w = self.image.shape[:2]
        
        # Calculate bounds
        y_min = max(0, y - kc[0])
        y_max = min(h, y - kc[0] + dk.shape[0])
        x_min = max(0, x - kc[1])
        x_max = min(w, x - kc[1] + dk.shape[1])
        
        if y_min >= y_max or x_min >= x_max:
            return
            
        # Kernel bounds
        ky_min = y_min - (y - kc[0])
        ky_max = ky_min + (y_max - y_min)
        kx_min = x_min - (x - kc[1])
        kx_max = kx_min + (x_max - x_min)
        
        # Apply stroke mask to image (simple overwrite for feedback)
        # Assuming self.image is RGBA. If it's not, this might fail or need conversion.
        if self.image is not None and self.image.ndim == 3 and self.image.shape[2] == 4:
            # We use a masked update to only overwrite where strokemask has opacity
            s_mask = self.strokemask[ky_min:ky_max, kx_min:kx_max]
            # Only update pixels where the stroke mask is not transparent (alpha > 0)
            mask_indices = s_mask[:, :, 3] > 0
            self.image[y_min:y_max, x_min:x_max][mask_indices] = s_mask[mask_indices]
            self.updateImage()

    def updateImage(self):
        if hasattr(super(), "updateImage"):
            super().updateImage()
        else:
            self.render()
            self.update()

    def _ensure_selection_rect(self):
        if self._selection_rect is not None:
            return
        pen = pg.mkPen(color=(255, 255, 255), width=1)
        self._selection_rect = QGraphicsRectItem()
        self._selection_rect.setPen(pen)
        self._selection_rect.setBrush(pg.mkBrush(0, 0, 0, 0))
        self.getViewBox().addItem(self._selection_rect)

    def _update_selection_rect(self, start, end):
        if self._selection_rect is None:
            return
        x0, y0 = start.x(), start.y()
        x1, y1 = end.x(), end.y()
        left = min(x0, x1)
        top = min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        self._selection_rect.setRect(left, top, width, height)

    def _clear_selection_rect(self):
        if self._selection_rect is None:
            return
        self.getViewBox().removeItem(self._selection_rect)
        self._selection_rect = None
