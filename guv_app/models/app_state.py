import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import os
from guv_app.data_models.results import InferenceResult
from cellpose import io, utils
from PyQt6.QtCore import QObject, pyqtSignal

class ViewConfig:
    """
    Holds the state of the UI's view controls.
    """
    def __init__(self):
        self.masks_visible = True
        self.outlines_visible = False
        self.class_visible = [True] # Corresponds to the initial "class 1"
        self.color_by_class = True
        self.autosave_enabled = True
        self.show_visualization = True
        self.color_mode = 0
        self.view_mode = 0
        self.channel_index = 0

class ApplicationStateModel(QObject):
    """
    A central, observable class holding the application's shared state.
    """
    view_update_signal = pyqtSignal()

    def __init__(self):
        """
        Initializer that sets up instance state.
        """
        super().__init__()
        self.raw_image = None
        self.display_image = None
        self.image_data = None
        self.filename = None
        self.frame_id = None
        self.series_index = None
        self.NZ = 1
        self.Ly = 0
        self.Lx = 0
        
        # Drawing and mask state
        self.view_config = ViewConfig()
        self.cellpix = None # np.ndarray for mask instances
        self.visualization_masks = None # Temporary masks for visualization (not saved)
        self.outpix = None  # np.ndarray for outlines
        self.ncells = 0
        self.strokes = []
        
        # Class-related state
        self.class_names = ["class 1"]
        self.class_colors = np.array([[255, 0, 0]], dtype=np.uint8)
        self.mask_classes = np.zeros(1, dtype=np.int16) # Maps mask ID -> class ID
        self.instance_colors = np.zeros((1, 3), dtype=np.uint8) # Maps mask ID -> random color
        self.current_class = 1 # Currently selected class ID for drawing
        self.pred_classes_map = None # 2D/3D array of predicted classes per pixel
        self.selected_mask_ids = set()

        # Navigation state
        self.image_files = []
        self.current_file_index = -1
        self.frame_refs = []

        # Model state
        self.current_model_id = "cpsam"

        # --- Placeholders for saving compatibility ---
        self.flows = []
        self.ismanual = np.zeros(0, bool)
        # --- End Placeholders ---

        # Callback for view updates
        self.update_view_callback = None

    @property
    def masks(self):
        return self.cellpix

    @masks.setter
    def masks(self, value):
        self.cellpix = value

    @property
    def classes(self):
        return self.mask_classes

    @classes.setter
    def classes(self, value):
        self.mask_classes = value

    def generate_instance_colors(self, n_needed):
        """Generates random colors for new masks."""
        if n_needed <= 0:
            return
        new_colors = np.random.randint(0, 255, (n_needed, 3), dtype=np.uint8)
        self.instance_colors = np.vstack([self.instance_colors, new_colors])

    def update_image(self, image_data, filename):
        """Updates the image and resets mask state."""
        prev_channel_index = self.view_config.channel_index
        self.raw_image = image_data
        self.display_image = image_data.copy()
        self.image_data = self.display_image
        self.filename = filename
        
        shape = image_data.shape
        # Heuristic to detect 2D RGB vs 3D Z-stack
        # If 3D and last dim is small (<=5), it's channels, so NZ=1
        if image_data.ndim > 2 and not (image_data.ndim == 3 and shape[-1] <= 5):
            self.NZ, self.Ly, self.Lx = shape[0], shape[1], shape[2]
        else: # 2D image (Y, X, C) or (Y, X)
            self.NZ = 1
            self.Ly, self.Lx = shape[0], shape[1]
        channel_count = shape[2] if image_data.ndim == 3 else 1
        if channel_count > 1:
            self.view_config.channel_index = min(prev_channel_index, channel_count - 1)
        else:
            self.view_config.channel_index = 0
            
        # Reset mask data
        self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), dtype=np.uint16)
        self.visualization_masks = None
        self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), dtype=np.uint16)
        self.ncells = 0
        self.strokes = []
        self.mask_classes = np.zeros(1, dtype=np.int16)
        self.instance_colors = np.zeros((1, 3), dtype=np.uint8)
        self.pred_classes_map = None
        self.flows = []

        self.trigger_view_update()

    def set_visualization(self, masks):
        """Sets a temporary visualization mask that overrides the display."""
        self.visualization_masks = masks
        self.trigger_view_update()

    def add_mask(self, points):
        """Adds a new mask from a list of points."""
        if not points or self.cellpix is None or cv2 is None:
            return

        mask = np.zeros((self.Ly, self.Lx), np.uint8)
        # points are [y, x] (row, col) from DrawingItem
        # cv2.fillPoly expects [x, y] (col, row)
        # We need to swap coordinates
        pts = np.array([[p[1], p[0]] for p in points], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

        ref_masks = self.cellpix[0] if self.cellpix.ndim == 3 else self.cellpix
        overlap = (ref_masks > 0) & (mask > 0)
        if overlap.any():
            mask[overlap] = 0
            if mask.sum() < 10:
                return
        
        # Update outlines for this new mask
        # We can just compute the outline of the new mask and add it
        outline = utils.masks_to_outlines(mask)
        self.outpix[0, outline > 0] = self.ncells + 1

        self.ncells += 1
        self.cellpix[0, mask > 0] = self.ncells # Assuming 2D for now

        # Assign the current class to the new mask
        if len(self.mask_classes) <= self.ncells:
            self.mask_classes = np.append(self.mask_classes, np.zeros(self.ncells + 1 - len(self.mask_classes), dtype=np.int16))
        self.mask_classes[self.ncells] = self.current_class
        
        # Generate instance color
        if len(self.instance_colors) <= self.ncells:
            self.generate_instance_colors(self.ncells + 1 - len(self.instance_colors))

        self.visualization_masks = None # Clear visualization on edit
        self.trigger_view_update()

    def add_masks(self, masks, classes=None):
        """Adds new masks from a numpy array."""
        if self.cellpix is None:
            return False

        # Validate shapes to prevent mismatches (e.g. loading full mask for a tile)
        if masks.shape != self.cellpix.shape:
            # Handle 2D masks for 3D container (1, Y, X)
            if self.cellpix.ndim == 3 and masks.ndim == 2:
                if self.cellpix.shape[1:] == masks.shape:
                    masks = masks[np.newaxis, ...]
                else:
                    # Shape mismatch
                    print(f"GUI_WARN: Mask shape mismatch. Image: {self.cellpix.shape}, Masks: {masks.shape}")
                    return False
            else:
                print(f"GUI_WARN: Mask shape mismatch. Image: {self.cellpix.shape}, Masks: {masks.shape}")
                return False

        # Find the max existing mask ID to offset the new masks
        max_id = self.ncells
        
        # Add new masks, offsetting their IDs
        new_mask_ids = np.unique(masks[masks > 0])
        self.ncells += len(new_mask_ids)
        
        # Create a mapping from old mask IDs in the input array to new global IDs
        id_map = np.zeros(masks.max() + 1, dtype=np.uint16)
        for new_id, old_id in enumerate(new_mask_ids, start=max_id + 1):
            id_map[old_id] = new_id

        # Apply the mapping
        new_cellpix = id_map[masks]
        
        # Combine with existing masks
        self.cellpix[new_cellpix > 0] = new_cellpix[new_cellpix > 0]
        
        # Update outlines
        # Recomputing all outlines is safer to ensure consistency
        self.outpix = utils.masks_to_outlines(self.cellpix) * self.cellpix

        # Update class assignments for the new masks
        if len(self.mask_classes) <= self.ncells:
            self.mask_classes = np.append(self.mask_classes, np.zeros(self.ncells + 1 - len(self.mask_classes), dtype=np.int16))
        
        # Update instance colors
        if len(self.instance_colors) <= self.ncells:
            self.generate_instance_colors(self.ncells + 1 - len(self.instance_colors))

        if classes is not None:
            # classes is indexed by old mask ID. Map old IDs to new IDs.
            for old_id, new_id in enumerate(id_map):
                if old_id > 0 and new_id > 0 and old_id < len(classes):
                    self.mask_classes[new_id] = classes[old_id]
        else:
            for new_mask_id in range(max_id + 1, self.ncells + 1):
                # Unlabeled imports should default to class 1, not current UI class.
                self.mask_classes[new_mask_id] = 1

        self.visualization_masks = None # Clear visualization on load
        self.trigger_view_update()
        return True

    def add_visualization_mask(self, points):
        """Adds a visualization mask stroke using the underlying mask ID at the stroke start."""
        if not points or self.cellpix is None or cv2 is None:
            return False
        if self.visualization_masks is None:
            self.visualization_masks = np.zeros_like(self.cellpix)

        ref_masks = self.cellpix[0] if self.cellpix.ndim == 3 else self.cellpix
        y0, x0 = points[0]
        if not (0 <= y0 < ref_masks.shape[0] and 0 <= x0 < ref_masks.shape[1]):
            return False
        mask_id = int(ref_masks[y0, x0])
        if mask_id <= 0:
            return False

        viz_masks = self.visualization_masks[0] if self.visualization_masks.ndim == 3 else self.visualization_masks
        draw_mask = np.zeros((self.Ly, self.Lx), np.uint8)
        pts = np.array([[p[1], p[0]] for p in points], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(draw_mask, [pts], 1)
        viz_masks[draw_mask > 0] = mask_id

        if self.visualization_masks.ndim == 3:
            self.visualization_masks[0] = viz_masks
        else:
            self.visualization_masks = viz_masks
        self.trigger_view_update()
        return True

    def remove_class(self, class_id_to_remove):
        """
        Removes a class and re-assigns masks belonging to it.
        For simplicity, we'll re-assign them to class 1.
        """
        if class_id_to_remove > len(self.class_names) or class_id_to_remove <= 0 or len(self.class_names) <=1:
            return

        class_idx = class_id_to_remove - 1
        
        # Remove from lists
        del self.class_names[class_idx]
        self.class_colors = np.delete(self.class_colors, class_idx, axis=0)
        del self.view_config.class_visible[class_idx]

        # Re-assign masks of the removed class to class 1
        # And shift down class IDs for all classes above the removed one
        new_mask_classes = self.mask_classes.copy()
        for i in range(len(new_mask_classes)):
            if self.mask_classes[i] == class_id_to_remove:
                new_mask_classes[i] = 1 # Re-assign to class 1
            elif self.mask_classes[i] > class_id_to_remove:
                new_mask_classes[i] -= 1
        self.mask_classes = new_mask_classes

        # Adjust current class if it was the removed one or after
        if self.current_class == class_id_to_remove:
            self.current_class = 1
        elif self.current_class > class_id_to_remove:
            self.current_class -= 1
            
        self.visualization_masks = None
        self.trigger_view_update()


    def assign_class_to_mask(self, mask_id, class_id):
        """Assigns a new class ID to an existing mask ID."""
        if mask_id > 0 and mask_id < len(self.mask_classes):
            self.mask_classes[mask_id] = class_id

    def get_selected_mask_ids(self):
        return set(self.selected_mask_ids) if self.selected_mask_ids else set()

    def clear_selected_masks(self):
        self.selected_mask_ids = set()

    def select_masks_in_rect(self, y0, x0, y1, x1):
        if self.cellpix is None:
            self.selected_mask_ids = set()
            return 0
        ref_masks = self.cellpix[0] if self.cellpix.ndim == 3 else self.cellpix
        y_min, y_max = sorted((int(y0), int(y1)))
        x_min, x_max = sorted((int(x0), int(x1)))
        y_min = max(0, min(y_min, ref_masks.shape[0]))
        y_max = max(0, min(y_max, ref_masks.shape[0]))
        x_min = max(0, min(x_min, ref_masks.shape[1]))
        x_max = max(0, min(x_max, ref_masks.shape[1]))
        if y_min == y_max or x_min == x_max:
            self.selected_mask_ids = set()
            return 0
        ids = np.unique(ref_masks[y_min:y_max, x_min:x_max])
        ids = ids[ids > 0]
        self.selected_mask_ids = set(int(i) for i in ids)
        return len(self.selected_mask_ids)

    def remove_mask_at_point(self, y, x):
        """Removes the mask at the given coordinates."""
        if self.cellpix is None:
            return
        
        if 0 <= y < self.Ly and 0 <= x < self.Lx:
            mask_id = self.cellpix[0, y, x]
            if mask_id > 0:
                self.cellpix[self.cellpix == mask_id] = 0
                # Recompute outlines so deleted masks do not leave stale contour pixels.
                self.outpix = utils.masks_to_outlines(self.cellpix) * self.cellpix
                if self.selected_mask_ids:
                    self.selected_mask_ids.discard(int(mask_id))
                self.visualization_masks = None
                self.trigger_view_update()

    def remove_visualization_mask_at_point(self, y, x):
        """Removes a label from the visualization mask at the given coordinates."""
        if self.visualization_masks is None:
            return

        masks = self.visualization_masks
        if masks.ndim == 3:
            masks = masks[0]

        if 0 <= y < masks.shape[0] and 0 <= x < masks.shape[1]:
            mask_id = masks[y, x]
            if mask_id > 0:
                masks[masks == mask_id] = 0
                if self.visualization_masks.ndim == 3:
                    self.visualization_masks[0] = masks
                else:
                    self.visualization_masks = masks
                self.trigger_view_update()

    def _polygon_mask(self, points, shape):
        if cv2 is None:
            return None
        if not points:
            return None
        mask = np.zeros(shape, np.uint8)
        pts = np.array([[p[1], p[0]] for p in points], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)

    def remove_masks_in_polygon(self, points):
        if self.cellpix is None or cv2 is None:
            return 0
        poly = self._polygon_mask(points, (self.Ly, self.Lx))
        if poly is None:
            return 0
        masks = self.cellpix[0] if self.cellpix.ndim == 3 else self.cellpix
        ids = np.unique(masks[poly])
        ids = ids[ids > 0]
        if ids.size == 0:
            return 0
        for mid in ids:
            self.cellpix[self.cellpix == mid] = 0
        self.outpix = utils.masks_to_outlines(self.cellpix) * self.cellpix
        self.visualization_masks = None
        self.trigger_view_update()
        return int(ids.size)

    def remove_visualization_masks_in_polygon(self, points):
        if self.visualization_masks is None or cv2 is None:
            return 0
        poly = self._polygon_mask(points, (self.Ly, self.Lx))
        if poly is None:
            return 0
        masks = self.visualization_masks
        if masks.ndim == 3:
            ref = masks[0]
        else:
            ref = masks
        ids = np.unique(ref[poly])
        ids = ids[ids > 0]
        if ids.size == 0:
            return 0
        ref[np.isin(ref, ids)] = 0
        if masks.ndim == 3:
            self.visualization_masks[0] = ref
        else:
            self.visualization_masks = ref
        self.trigger_view_update()
        return int(ids.size)

    def clear_masks(self):
        """Resets all mask-related data."""
        if self.cellpix is not None:
            self.cellpix.fill(0)
        if self.outpix is not None:
            self.outpix.fill(0)
        self.ncells = 0
        self.strokes = []
        self.mask_classes = np.zeros(1, dtype=np.int16)
        self.instance_colors = np.zeros((1, 3), dtype=np.uint8)
        self.visualization_masks = None

    def trigger_view_update(self):
        """Manually triggers a view update via callback."""
        self.view_update_signal.emit()

    def handle_inference_result(self, result: InferenceResult):
        """
        Handles a completed inference result.
        If it's a single image inference (no filename or matches current), update the view.
        If it's a batch result (has filename), save it to disk.
        """
        # 1. Save to disk if it's a batch result and not already saved
        if result.filename and not result.is_saved:
            self.save_prediction(result)

        # 2. Update view if it matches the current image
        if result.masks is not None:
            filename_match = result.filename is None or result.filename == self.filename
            frame_match = (result.frame_id is None) or (result.frame_id == self.frame_id)
            if filename_match and frame_match:
                self.clear_masks()
                self.add_masks(result.masks, result.classes)
                self.flows = result.flows if result.flows is not None else []
                if result.classes_map is not None:
                    self.pred_classes_map = result.classes_map
                if result.class_names is not None:
                    self.class_names = list(result.class_names)
                if result.class_colors is not None:
                    self.class_colors = result.class_colors
                self.trigger_view_update()

    def save_prediction(self, result: InferenceResult):
        """Saves the inference result to a _pred.npy file."""
        if not result.filename or result.masks is None:
            return

        base, ext = os.path.splitext(result.filename)
        frame_suffix = io.frame_id_to_suffix(result.frame_id)
        dat = {
            "outlines": result.outlines if result.outlines is not None else (utils.masks_to_outlines(result.masks) * result.masks),
            "masks": result.masks,
            "chan_choose": [0, 0],
            "ismanual": np.zeros(result.masks.max(), bool),
            "filename": result.filename,
            "flows": result.flows,
            "diameter": result.diameter,
            "classes": result.classes if result.classes is not None else np.zeros(result.masks.max() + 1, dtype=np.int16),
            "classes_map": result.classes_map,
            "class_names": result.class_names,
            "class_colors": result.class_colors
        }
        np.save(f"{base}{frame_suffix}_pred.npy", dat)
