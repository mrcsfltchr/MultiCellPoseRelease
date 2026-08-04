import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
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
        self.masks_frozen = False
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
        self.raw_image = None        # normalize99-processed image used for display/inference
        self.source_image = None     # original pixel values before normalization; used for intensity measurements
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
        self.visualization_color_by_label = False
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
        self.selected_reference_mask_id = None
        self.channel_segmentations = {}
        self.previous_channel_index = None
        self.association_groups = {}
        self.mask_associations = {}
        self.next_association_object_id = 1

        # Navigation state
        self.image_files = []
        self.current_file_index = -1
        self.frame_refs = []
        self.z_count = 1
        self.current_z_index = 0

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

    def update_image(self, image_data, filename, source_image=None):
        """Updates the image and resets mask state.

        Args:
            image_data: The normalize99-processed image for display and inference.
            filename: Path to the source file.
            source_image: Optional original image before normalization.  When
                provided, intensity-measurement code uses this instead of
                image_data so that per-channel values reflect actual pixel
                intensities rather than the globally-normalized ones.
        """
        prev_channel_index = self.view_config.channel_index
        self.raw_image = image_data
        self.source_image = source_image  # None → callers will fall back to raw_image
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
        self.visualization_color_by_label = False
        self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), dtype=np.uint16)
        self.ncells = 0
        self.strokes = []
        self.mask_classes = np.zeros(1, dtype=np.int16)
        self.instance_colors = np.zeros((1, 3), dtype=np.uint8)
        self.pred_classes_map = None
        self.flows = []
        self.channel_segmentations = {}
        self.previous_channel_index = None
        self.selected_mask_ids = set()
        self.selected_reference_mask_id = None
        self.association_groups = {}
        self.mask_associations = {}
        self.next_association_object_id = 1

        self.trigger_view_update()

    def set_visualization(self, masks):
        """Sets a temporary visualization mask that overrides the display."""
        self.visualization_masks = masks
        self.trigger_view_update()

    def set_visualization_color_by_label(self, enabled):
        self.visualization_color_by_label = bool(enabled)
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
        self.visualization_color_by_label = False
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

        # Some inference outputs (e.g., non-semantic models) persist a classes
        # vector of all zeros. Treat this as "unlabeled" so GUI defaults each
        # instance to class 1 instead of hiding all masks as class 0.
        classes_arr = None
        if classes is not None:
            try:
                classes_arr = np.asarray(classes).reshape(-1)
                has_labeled_class = False
                for old_id in new_mask_ids:
                    oid = int(old_id)
                    if oid < len(classes_arr) and int(classes_arr[oid]) > 0:
                        has_labeled_class = True
                        break
                if not has_labeled_class:
                    classes_arr = None
            except Exception:
                classes_arr = None

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

        if classes_arr is not None:
            # classes is indexed by old mask ID. Map old IDs to new IDs.
            for old_id, new_id in enumerate(id_map):
                if old_id > 0 and new_id > 0 and old_id < len(classes_arr):
                    self.mask_classes[new_id] = classes_arr[old_id]
        else:
            for new_mask_id in range(max_id + 1, self.ncells + 1):
                # Unlabeled imports should default to class 1, not current UI class.
                self.mask_classes[new_mask_id] = 1

        self.visualization_masks = None # Clear visualization on load
        self.visualization_color_by_label = False
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

    def get_current_channel_index(self):
        if self.raw_image is not None and self.raw_image.ndim == 3 and self.raw_image.shape[2] not in (1, 3):
            return int(self.view_config.channel_index)
        return 0

    def get_channel_count(self):
        if self.raw_image is not None and self.raw_image.ndim == 3 and self.raw_image.shape[2] not in (1, 3):
            return int(self.raw_image.shape[2])
        return 1

    def _blank_channel_state(self):
        masks = np.zeros((self.NZ, self.Ly, self.Lx), dtype=np.uint16)
        return {
            "masks": masks,
            "outpix": np.zeros_like(masks),
            "mask_classes": np.zeros(1, dtype=np.int16),
            "instance_colors": np.zeros((1, 3), dtype=np.uint8),
            "pred_classes_map": None,
            "flows": [],
        }

    def _copy_channel_state(self, state):
        copied = {
            "masks": np.array(state.get("masks"), copy=True),
            "outpix": np.array(state.get("outpix"), copy=True),
            "mask_classes": np.array(state.get("mask_classes"), copy=True),
            "instance_colors": np.array(state.get("instance_colors"), copy=True),
            "pred_classes_map": None,
            "flows": [],
        }
        classes_map = state.get("pred_classes_map")
        if classes_map is not None:
            copied["pred_classes_map"] = np.array(classes_map, copy=True)
        flows = state.get("flows") or []
        copied["flows"] = [np.array(item, copy=True) if isinstance(item, np.ndarray) else item for item in flows]
        return copied

    def _state_from_current(self):
        return {
            "masks": np.array(self.cellpix, copy=True),
            "outpix": np.array(self.outpix, copy=True),
            "mask_classes": np.array(self.mask_classes, copy=True),
            "instance_colors": np.array(self.instance_colors, copy=True),
            "pred_classes_map": None if self.pred_classes_map is None else np.array(self.pred_classes_map, copy=True),
            "flows": [np.array(item, copy=True) if isinstance(item, np.ndarray) else item for item in (self.flows or [])],
        }

    def persist_current_channel_state(self):
        idx = self.get_current_channel_index()
        if self.cellpix is None:
            self.channel_segmentations[idx] = self._blank_channel_state()
        else:
            self.channel_segmentations[idx] = self._state_from_current()

    def restore_channel_state(self, channel_index):
        idx = int(channel_index)
        state = self.channel_segmentations.get(idx)
        if state is None:
            state = self._blank_channel_state()
            self.channel_segmentations[idx] = self._copy_channel_state(state)
        state = self._copy_channel_state(state)
        self.cellpix = state["masks"]
        self.outpix = state["outpix"]
        self.mask_classes = state["mask_classes"]
        self.instance_colors = state["instance_colors"]
        self.pred_classes_map = state.get("pred_classes_map")
        self.flows = state.get("flows") or []
        self.ncells = int(self.cellpix.max()) if self.cellpix is not None else 0
        self.visualization_masks = None
        self.selected_mask_ids = set()
        self.selected_reference_mask_id = None

    def _peek_channel_state(self, channel_index):
        idx = int(channel_index)
        if idx == self.get_current_channel_index() and self.cellpix is not None:
            return {
                "masks": self.cellpix,
                "outpix": self.outpix,
                "mask_classes": self.mask_classes,
                "instance_colors": self.instance_colors,
                "pred_classes_map": self.pred_classes_map,
                "flows": self.flows or [],
            }
        state = self.channel_segmentations.get(idx)
        if state is None:
            state = self._blank_channel_state()
        return state

    def get_channel_state(self, channel_index):
        idx = int(channel_index)
        if idx == self.get_current_channel_index() and self.cellpix is not None:
            return self._state_from_current()
        state = self.channel_segmentations.get(idx)
        if state is None:
            state = self._blank_channel_state()
        return self._copy_channel_state(state)

    def get_channel_masks_2d(self, channel_index):
        state = self._peek_channel_state(channel_index)
        masks = state["masks"]
        if masks is None:
            return None
        return masks[0] if masks.ndim == 3 else masks

    def channel_has_masks(self, channel_index):
        masks = self.get_channel_masks_2d(channel_index)
        return masks is not None and int(masks.max()) > 0

    def clear_reference_selection(self):
        self.selected_reference_mask_id = None

    def select_reference_mask_at_point(self, y, x):
        ref_idx = self.previous_channel_index
        if ref_idx is None:
            self.selected_reference_mask_id = None
            return 0
        masks = self.get_channel_masks_2d(ref_idx)
        if masks is None:
            self.selected_reference_mask_id = None
            return 0
        if 0 <= y < masks.shape[0] and 0 <= x < masks.shape[1]:
            mask_id = int(masks[y, x])
            self.selected_reference_mask_id = mask_id if mask_id > 0 else None
            return mask_id
        self.selected_reference_mask_id = None
        return 0

    def _remove_association_membership(self, channel_index, mask_id):
        key = (int(channel_index), int(mask_id))
        object_id = self.mask_associations.pop(key, None)
        if object_id is None:
            return
        members = self.association_groups.get(object_id)
        if not members:
            return
        members.pop(int(channel_index), None)
        if len(members) < 2:
            for ch, mid in list(members.items()):
                self.mask_associations.pop((int(ch), int(mid)), None)
            self.association_groups.pop(object_id, None)

    def clear_channel_associations(self, channel_index, mask_ids=None):
        if mask_ids is None:
            keys = [key for key in self.mask_associations if key[0] == int(channel_index)]
            for ch, mid in keys:
                self._remove_association_membership(ch, mid)
            return
        for mask_id in mask_ids:
            self._remove_association_membership(channel_index, int(mask_id))

    def associate_masks(self, reference_channel, reference_mask_id, current_channel, current_mask_id):
        ref_channel = int(reference_channel)
        cur_channel = int(current_channel)
        ref_mask = int(reference_mask_id)
        cur_mask = int(current_mask_id)
        if ref_channel == cur_channel or ref_mask <= 0 or cur_mask <= 0:
            return None
        self._remove_association_membership(ref_channel, ref_mask)
        self._remove_association_membership(cur_channel, cur_mask)
        object_id = int(self.next_association_object_id)
        self.next_association_object_id += 1
        members = {ref_channel: ref_mask, cur_channel: cur_mask}
        self.association_groups[object_id] = members
        self.mask_associations[(ref_channel, ref_mask)] = object_id
        self.mask_associations[(cur_channel, cur_mask)] = object_id
        return object_id

    def _compute_mask_stats(self, masks, mask_ids):
        ids = np.array(sorted({int(mid) for mid in mask_ids if int(mid) > 0}), dtype=np.int32)
        if masks is None or ids.size == 0:
            return {}
        present = np.isin(masks, ids)
        if not np.any(present):
            return {}
        ys, xs = np.nonzero(present)
        vals = masks[present].astype(np.int64, copy=False)
        unique_ids, inverse = np.unique(vals, return_inverse=True)
        sum_y = np.bincount(inverse, weights=ys.astype(np.float64))
        sum_x = np.bincount(inverse, weights=xs.astype(np.float64))
        counts = np.bincount(inverse).astype(np.float64)
        stats = {}
        for i, mask_id in enumerate(unique_ids):
            if counts[i] <= 0:
                continue
            stats[int(mask_id)] = {
                "centroid": (float(sum_y[i] / counts[i]), float(sum_x[i] / counts[i])),
                "area": float(counts[i]),
            }
        return stats

    def _compute_overlap_counts(self, reference_masks, current_masks):
        overlap = (reference_masks > 0) & (current_masks > 0)
        if not np.any(overlap):
            return {}
        pairs = np.stack((reference_masks[overlap], current_masks[overlap]), axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        return {
            (int(ref_mask), int(cur_mask)): int(count)
            for (ref_mask, cur_mask), count in zip(unique_pairs, counts)
            if int(ref_mask) > 0 and int(cur_mask) > 0
        }

    @staticmethod
    def _solve_assignment(cost_matrix, invalid_cost):
        if cost_matrix.size == 0:
            return []
        if linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            return [(int(r), int(c)) for r, c in zip(row_idx, col_idx) if cost_matrix[r, c] < invalid_cost]

        candidates = []
        for r in range(cost_matrix.shape[0]):
            for c in range(cost_matrix.shape[1]):
                cost = float(cost_matrix[r, c])
                if cost < invalid_cost:
                    candidates.append((cost, r, c))
        candidates.sort(key=lambda item: item[0])
        assignments = []
        used_rows = set()
        used_cols = set()
        for _cost, r, c in candidates:
            if r in used_rows or c in used_cols:
                continue
            assignments.append((int(r), int(c)))
            used_rows.add(r)
            used_cols.add(c)
        return assignments

    def auto_associate_with_previous_channel(self, iou_threshold=0.25):
        current_channel = self.get_current_channel_index()
        reference_channel = self.previous_channel_index
        if reference_channel is None or reference_channel == current_channel:
            return 0, "No previous channel with masks."
        current_masks = self.get_channel_masks_2d(current_channel)
        reference_masks = self.get_channel_masks_2d(reference_channel)
        if current_masks is None or reference_masks is None:
            return 0, "No masks available to match."
        if current_masks.shape != reference_masks.shape:
            return 0, "Channel mask shapes do not match."
        current_ids = np.unique(current_masks)
        current_ids = current_ids[current_ids > 0]
        reference_ids = np.unique(reference_masks)
        reference_ids = reference_ids[reference_ids > 0]
        if current_ids.size == 0 or reference_ids.size == 0:
            return 0, "No masks available to match."
        current_stats = self._compute_mask_stats(current_masks, current_ids)
        reference_stats = self._compute_mask_stats(reference_masks, reference_ids)
        if not current_stats or not reference_stats:
            return 0, "No masks available to match."

        all_areas = np.array(
            [stats["area"] for stats in reference_stats.values()] +
            [stats["area"] for stats in current_stats.values()],
            dtype=np.float64,
        )
        median_area = float(np.median(all_areas)) if all_areas.size else 1.0
        median_area = max(median_area, 1.0)
        typical_diameter = max(1.0, 2.0 * np.sqrt(median_area / np.pi))

        log_areas = np.log(np.clip(all_areas, 1.0, None))
        sigma_log_area = float(np.std(log_areas)) if log_areas.size else 1.0
        if sigma_log_area < 0.1:
            q75, q25 = np.percentile(log_areas, [75, 25]) if log_areas.size else (0.0, 0.0)
            sigma_log_area = max(float((q75 - q25) / 1.349), 0.25)

        overlap_counts = self._compute_overlap_counts(reference_masks, current_masks)

        reference_ids = np.array(sorted(reference_stats.keys()), dtype=np.int32)
        current_ids = np.array(sorted(current_stats.keys()), dtype=np.int32)
        invalid_cost = 1e9
        max_position_norm = 8.0
        min_area_ratio = 0.25
        max_area_ratio = 4.0
        max_total_cost = 5.0
        cost_matrix = np.full((reference_ids.size, current_ids.size), invalid_cost, dtype=np.float64)

        for r, ref_mask in enumerate(reference_ids):
            ref_area = float(reference_stats[int(ref_mask)]["area"])
            ref_center = reference_stats[int(ref_mask)]["centroid"]
            ref_log_area = np.log(max(ref_area, 1.0))
            for c, cur_mask in enumerate(current_ids):
                cur_area = float(current_stats[int(cur_mask)]["area"])
                area_ratio = cur_area / max(ref_area, 1.0)
                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue

                cur_center = current_stats[int(cur_mask)]["centroid"]
                pos_dist = np.hypot(cur_center[0] - ref_center[0], cur_center[1] - ref_center[1])
                pos_norm = pos_dist / typical_diameter
                if pos_norm > max_position_norm:
                    continue

                area_norm = abs(np.log(max(cur_area, 1.0)) - ref_log_area) / sigma_log_area
                inter = float(overlap_counts.get((int(ref_mask), int(cur_mask)), 0))
                union = ref_area + cur_area - inter
                iou = inter / union if union > 0 else 0.0

                # IoU is only a weak local-shape cue; position and size dominate.
                cost = (0.6 * pos_norm) + (0.3 * area_norm) + (0.1 * (1.0 - iou))
                if cost <= max_total_cost:
                    cost_matrix[r, c] = cost

        assignments = self._solve_assignment(cost_matrix, invalid_cost)
        if not assignments:
            return 0, "No plausible matches between channels."

        matched = 0
        for row_idx, col_idx in assignments:
            ref_mask = int(reference_ids[row_idx])
            cur_mask = int(current_ids[col_idx])
            self.associate_masks(reference_channel, ref_mask, current_channel, cur_mask)
            matched += 1
        return matched, f"Matched {matched} mask(s) from channel {reference_channel + 1} to {current_channel + 1}."

    def _compute_mask_centroids(self, masks, mask_ids):
        ids = np.array(sorted({int(mid) for mid in mask_ids if int(mid) > 0}), dtype=np.int32)
        if masks is None or ids.size == 0:
            return {}
        present = np.isin(masks, ids)
        if not np.any(present):
            return {}
        ys, xs = np.nonzero(present)
        vals = masks[present].astype(np.int64, copy=False)
        unique_ids, inverse = np.unique(vals, return_inverse=True)
        sum_y = np.bincount(inverse, weights=ys.astype(np.float64))
        sum_x = np.bincount(inverse, weights=xs.astype(np.float64))
        counts = np.bincount(inverse).astype(np.float64)
        centers = {}
        for i, mask_id in enumerate(unique_ids):
            if counts[i] > 0:
                centers[int(mask_id)] = (float(sum_y[i] / counts[i]), float(sum_x[i] / counts[i]))
        return centers

    def get_mask_centroid(self, channel_index, mask_id):
        mask_id = int(mask_id)
        if mask_id <= 0:
            return None
        masks = self.get_channel_masks_2d(channel_index)
        centers = self._compute_mask_centroids(masks, [mask_id])
        return centers.get(mask_id)

    def get_object_id_for_mask(self, channel_index, mask_id):
        mask_id = int(mask_id)
        if mask_id <= 0:
            return 0
        return int(self.mask_associations.get((int(channel_index), mask_id), 0) or 0)

    def get_object_ids_for_channel(self, channel_index=None):
        if channel_index is None:
            channel_index = self.get_current_channel_index()
        state = self.get_channel_state(channel_index)
        masks = state["masks"]
        if masks is None:
            return np.zeros(1, dtype=np.int32)
        max_mask_id = int(np.max(masks)) if np.size(masks) else 0
        object_ids = np.zeros(max_mask_id + 1, dtype=np.int32)
        for mask_id in range(1, max_mask_id + 1):
            object_ids[mask_id] = self.get_object_id_for_mask(channel_index, mask_id)
        return object_ids

    def get_association_line_segments(self, reference_channel=None, current_channel=None):
        if current_channel is None:
            current_channel = self.get_current_channel_index()
        if reference_channel is None:
            reference_channel = self.previous_channel_index
        if reference_channel is None or int(reference_channel) == int(current_channel):
            return []

        current_channel = int(current_channel)
        reference_channel = int(reference_channel)
        selected_object_ids = set()
        for mask_id in self.get_selected_mask_ids():
            object_id = self.mask_associations.get((current_channel, int(mask_id)))
            if object_id is not None:
                selected_object_ids.add(int(object_id))
        if self.selected_reference_mask_id is not None:
            object_id = self.mask_associations.get((reference_channel, int(self.selected_reference_mask_id)))
            if object_id is not None:
                selected_object_ids.add(int(object_id))

        relevant = []
        ref_ids = []
        cur_ids = []
        for object_id, members in sorted(self.association_groups.items()):
            ref_mask_id = members.get(reference_channel)
            cur_mask_id = members.get(current_channel)
            if ref_mask_id is None or cur_mask_id is None:
                continue
            relevant.append((int(object_id), int(ref_mask_id), int(cur_mask_id)))
            ref_ids.append(int(ref_mask_id))
            cur_ids.append(int(cur_mask_id))

        if not relevant:
            return []

        ref_masks = self.get_channel_masks_2d(reference_channel)
        cur_masks = self.get_channel_masks_2d(current_channel)
        ref_centers = self._compute_mask_centroids(ref_masks, ref_ids)
        cur_centers = self._compute_mask_centroids(cur_masks, cur_ids)

        segments = []
        for object_id, ref_mask_id, cur_mask_id in relevant:
            ref_center = ref_centers.get(ref_mask_id)
            cur_center = cur_centers.get(cur_mask_id)
            if ref_center is None or cur_center is None:
                continue
            segments.append({
                "object_id": object_id,
                "reference_channel": reference_channel,
                "current_channel": current_channel,
                "reference_mask_id": ref_mask_id,
                "current_mask_id": cur_mask_id,
                "reference_center": ref_center,
                "current_center": cur_center,
                "selected": object_id in selected_object_ids,
            })
        return segments

    def export_channel_segmentations(self):
        payload = {}
        current_idx = self.get_current_channel_index()
        seen = set(self.channel_segmentations.keys()) | {current_idx}
        for idx in sorted(seen):
            state = self.get_channel_state(idx)
            masks = state["masks"]
            payload[str(int(idx))] = {
                "masks": np.array(masks, copy=True),
                "classes": np.array(state.get("mask_classes"), copy=True),
                "classes_map": None if state.get("pred_classes_map") is None else np.array(state.get("pred_classes_map"), copy=True),
                "instance_colors": np.array(state.get("instance_colors"), copy=True),
            }
        return payload

    def load_channel_segmentations(self, payload, active_channel_index=0, switch_channel=True):
        self.channel_segmentations = {}
        if payload:
            for key, state in payload.items():
                idx = int(key)
                masks = np.array(state.get("masks"), copy=True)
                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]
                outpix = utils.masks_to_outlines(masks) * masks
                n_masks = int(masks.max())

                # Read classes — "mask_classes" is the canonical key; fall back to "classes"
                # for files written by older code versions.
                raw_classes = state.get("mask_classes")
                if raw_classes is None:
                    raw_classes = state.get("classes")
                if raw_classes is None:
                    mask_classes = np.zeros(n_masks + 1, dtype=np.int16)
                else:
                    mask_classes = np.array(raw_classes, dtype=np.int16, copy=True)
                # If all class labels are 0 (unlabeled / non-semantic model), default every
                # mask to class 1 so they are visible in color-by-class mode.  This mirrors
                # the behaviour of add_masks(), which also promotes unlabeled masks to class 1.
                if n_masks > 0 and not np.any(mask_classes[1 : n_masks + 1] > 0):
                    padded = np.ones(max(n_masks + 1, len(mask_classes)), dtype=np.int16)
                    padded[0] = 0
                    mask_classes = padded

                self.channel_segmentations[idx] = {
                    "masks": masks,
                    "outpix": outpix,
                    "mask_classes": mask_classes,
                    "instance_colors": np.array(state.get("instance_colors") if state.get("instance_colors") is not None else np.zeros((n_masks + 1, 3), dtype=np.uint8), copy=True),
                    "pred_classes_map": None if state.get("classes_map") is None else np.array(state.get("classes_map"), copy=True),
                    "flows": [],
                }
        display_channel = active_channel_index if switch_channel else self.get_current_channel_index()
        if self.get_channel_count() > 1:
            self.view_config.channel_index = max(0, min(int(display_channel), self.get_channel_count() - 1))
        self.restore_channel_state(display_channel)
        self.trigger_view_update()

    def export_associations(self):
        if not self.association_groups:
            return None
        groups = []
        for object_id, members in sorted(self.association_groups.items()):
            groups.append({
                "object_id": int(object_id),
                "members": [{"channel": int(ch), "mask_id": int(mid)} for ch, mid in sorted(members.items())],
            })
        return {
            "version": 1,
            "next_object_id": int(self.next_association_object_id),
            "groups": groups,
        }

    def load_associations(self, payload):
        self.association_groups = {}
        self.mask_associations = {}
        self.next_association_object_id = 1
        if not payload:
            return
        groups = payload.get("groups") or []
        for group in groups:
            object_id = int(group.get("object_id", self.next_association_object_id))
            members = {}
            for member in group.get("members") or []:
                ch = int(member.get("channel", -1))
                mid = int(member.get("mask_id", 0))
                if ch < 0 or mid <= 0:
                    continue
                members[ch] = mid
                self.mask_associations[(ch, mid)] = object_id
            if len(members) >= 2:
                self.association_groups[object_id] = members
                self.next_association_object_id = max(self.next_association_object_id, object_id + 1)
        self.next_association_object_id = max(self.next_association_object_id, int(payload.get("next_object_id", self.next_association_object_id)))

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
        self.visualization_color_by_label = False
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
        self.selected_reference_mask_id = None
        return len(self.selected_mask_ids)

    def remove_mask_at_point(self, y, x):
        """Removes the mask at the given coordinates."""
        if self.cellpix is None:
            return

        if 0 <= y < self.Ly and 0 <= x < self.Lx:
            mask_id = self.cellpix[0, y, x]
            if mask_id > 0:
                self.cellpix[self.cellpix == mask_id] = 0
                self.clear_channel_associations(self.get_current_channel_index(), [mask_id])
                # Recompute outlines so deleted masks do not leave stale contour pixels.
                self.outpix = utils.masks_to_outlines(self.cellpix) * self.cellpix
                if self.selected_mask_ids:
                    self.selected_mask_ids.discard(int(mask_id))
                if self.selected_reference_mask_id == int(mask_id):
                    self.selected_reference_mask_id = None
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

    def visualization_label_at_point(self, y, x):
        """Returns the visualization label at a point without mutating the mask."""
        masks = self.visualization_masks
        if masks is None:
            return 0
        if masks.ndim == 3:
            masks = masks[0]
        if 0 <= y < masks.shape[0] and 0 <= x < masks.shape[1]:
            return int(masks[y, x])
        return 0

    def visualization_labels_in_polygon(self, points):
        """Returns nonzero visualization labels inside a polygon without mutating the mask."""
        if self.visualization_masks is None or cv2 is None or not points:
            return set()
        masks = self.visualization_masks[0] if self.visualization_masks.ndim == 3 else self.visualization_masks
        poly = self._polygon_mask(points, masks.shape)
        if poly is None:
            return set()
        ids = np.unique(masks[poly])
        return {int(i) for i in ids if int(i) > 0}

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
        self.clear_channel_associations(self.get_current_channel_index(), ids)
        self.outpix = utils.masks_to_outlines(self.cellpix) * self.cellpix
        self.selected_mask_ids = set()
        self.selected_reference_mask_id = None
        self.visualization_masks = None
        self.visualization_color_by_label = False
        self.trigger_view_update()
        return int(ids.size)

    def translate_selected_masks(self, dx=0, dy=0):
        if self.cellpix is None or not self.selected_mask_ids:
            return False, "No masks selected."
        dx = int(dx)
        dy = int(dy)
        if dx == 0 and dy == 0:
            return False, "No movement requested."

        ref_masks = self.cellpix[0] if self.cellpix.ndim == 3 else self.cellpix
        selected_ids = np.array(sorted(int(mid) for mid in self.selected_mask_ids if int(mid) > 0), dtype=np.int32)
        if selected_ids.size == 0:
            return False, "No masks selected."

        selected_mask = np.isin(ref_masks, selected_ids)
        if not selected_mask.any():
            self.selected_mask_ids = set()
            return False, "Selected masks are empty."

        y_src_start = max(0, -dy)
        y_src_end = min(ref_masks.shape[0], ref_masks.shape[0] - dy) if dy >= 0 else ref_masks.shape[0]
        x_src_start = max(0, -dx)
        x_src_end = min(ref_masks.shape[1], ref_masks.shape[1] - dx) if dx >= 0 else ref_masks.shape[1]
        y_dst_start = max(0, dy)
        y_dst_end = y_dst_start + max(0, y_src_end - y_src_start)
        x_dst_start = max(0, dx)
        x_dst_end = x_dst_start + max(0, x_src_end - x_src_start)

        if y_src_start >= y_src_end or x_src_start >= x_src_end:
            return False, "Movement would place mask outside the image."

        selected_layer = np.where(selected_mask, ref_masks, 0)
        moved_layer = np.zeros_like(ref_masks)
        moved_layer[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = selected_layer[y_src_start:y_src_end, x_src_start:x_src_end]

        moved_nonzero = moved_layer > 0
        if not moved_nonzero.any():
            return False, "Movement would place mask outside the image."

        base_masks = np.where(selected_mask, 0, ref_masks)
        if np.any((base_masks > 0) & moved_nonzero):
            return False, "Movement blocked by another mask."

        merged = base_masks.copy()
        merged[moved_nonzero] = moved_layer[moved_nonzero]
        if self.cellpix.ndim == 3:
            self.cellpix[0] = merged
        else:
            self.cellpix = merged
        self.outpix = utils.masks_to_outlines(self.cellpix) * self.cellpix
        self.visualization_masks = None
        self.visualization_color_by_label = False
        self.trigger_view_update()
        return True, f"Moved {selected_ids.size} mask(s)."

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
        self.selected_mask_ids = set()
        self.selected_reference_mask_id = None
        self.clear_channel_associations(self.get_current_channel_index())

    def trigger_view_update(self):
        """Manually triggers a view update via callback."""
        self.view_update_signal.emit()

    def handle_inference_result(self, result: InferenceResult):
        """
        Handles a completed inference result.
        If it's a single image inference (no filename or matches current), update the view.
        If it's a batch result (has filename), save it to disk.
        """
        # 1. Update in-memory state first so save_prediction captures the latest channel data
        if result.masks is not None:
            filename_match = result.filename is None or result.filename == self.filename
            frame_match = (result.frame_id is None) or (result.frame_id == self.frame_id)
            if filename_match and frame_match:
                result_channel = (
                    result.channel_index
                    if result.channel_index is not None
                    else self.get_current_channel_index()
                )
                current_channel = self.get_current_channel_index()
                if result_channel == current_channel:
                    # Same channel as the live display — update live buffer directly.
                    # When masks are frozen, preserve existing masks and merge new ones
                    # on top (union).  Otherwise replace entirely.
                    if not self.view_config.masks_frozen:
                        self.clear_masks()
                    self.add_masks(result.masks, result.classes)
                    self.flows = result.flows if result.flows is not None else []
                    if result.classes_map is not None:
                        self.pred_classes_map = result.classes_map
                    if result.class_names is not None:
                        self.class_names = list(result.class_names)
                    if result.class_colors is not None:
                        self.class_colors = result.class_colors
                    self.persist_current_channel_state()
                else:
                    # Different channel — store into the correct slot without
                    # disrupting the live display or the current channel's masks.
                    self._store_result_in_channel(result, result_channel)
                self.trigger_view_update()

        # 2. Save to disk after in-memory state is updated
        if result.filename and not result.is_saved:
            self.save_prediction(result)

    @staticmethod
    def _merge_channel_states_frozen(existing_state, new_state):
        """Merges new_state masks on top of frozen existing_state masks.

        Pixels occupied by frozen masks are preserved; new inference mask
        pixels are added with offset IDs only where no frozen mask exists.
        Returns a merged channel-state dict, or new_state if merging is not
        possible (shape mismatch, missing arrays).
        """
        existing_masks = np.asarray(existing_state.get("masks")) if existing_state.get("masks") is not None else None
        new_masks = np.asarray(new_state.get("masks")) if new_state.get("masks") is not None else None
        if existing_masks is None or new_masks is None:
            return new_state
        if existing_masks.shape != new_masks.shape:
            return new_state  # shape mismatch — just use new result

        existing_max = int(existing_masks.max())
        merged = np.array(existing_masks, copy=True)
        n_new = int(new_masks.max())
        for label in range(1, n_new + 1):
            new_label = label + existing_max
            free_pixels = (new_masks == label) & (merged == 0)
            merged[free_pixels] = new_label
        merged_n = int(merged.max())

        merged_outpix = utils.masks_to_outlines(merged) * merged

        existing_mc = np.asarray(existing_state.get("mask_classes") or [0])
        existing_ic = np.asarray(existing_state.get("instance_colors") or np.zeros((1, 3), dtype=np.uint8))
        new_mc = np.asarray(new_state.get("mask_classes") or [0])
        new_ic = np.asarray(new_state.get("instance_colors") or np.zeros((1, 3), dtype=np.uint8))

        merged_mc = np.zeros(merged_n + 1, dtype=np.int16)
        merged_ic = np.zeros((merged_n + 1, 3), dtype=np.uint8)
        n_ex = max(len(existing_mc) - 1, 0)
        n_nw = max(len(new_mc) - 1, 0)
        if n_ex > 0:
            copy_ex = min(n_ex, merged_n)
            merged_mc[1:copy_ex + 1] = existing_mc[1:copy_ex + 1]
            merged_ic[1:copy_ex + 1] = existing_ic[1:copy_ex + 1]
        if n_nw > 0 and n_ex + n_nw <= merged_n:
            merged_mc[n_ex + 1:n_ex + 1 + n_nw] = new_mc[1:1 + n_nw]
            merged_ic[n_ex + 1:n_ex + 1 + n_nw] = new_ic[1:1 + n_nw]

        return {
            "masks": merged,
            "outpix": merged_outpix,
            "mask_classes": merged_mc,
            "instance_colors": merged_ic,
            "pred_classes_map": new_state.get("pred_classes_map"),
            "flows": new_state.get("flows", []),
        }

    def _store_result_in_channel(self, result: InferenceResult, channel_index: int):
        """Stores an inference result directly into a channel_segmentations slot.

        Used when the result is for a channel that is not currently displayed, so
        the live buffer (cellpix / instance_colors) must not be touched.
        """
        masks = result.masks
        if masks is None:
            return
        n = int(masks.max()) if masks.size else 0
        masks_3d = masks[np.newaxis, ...] if masks.ndim == 2 else np.array(masks, copy=True)
        outpix = utils.masks_to_outlines(masks_3d) * masks_3d
        raw_classes = result.classes
        if raw_classes is None or not np.any(np.asarray(raw_classes)[1:n + 1] > 0):
            # Default unlabeled masks to class 1 for visibility in color-by-class mode.
            mc = np.ones(n + 1, dtype=np.int16)
            mc[0] = 0
        else:
            mc = np.asarray(raw_classes, dtype=np.int16)
        instance_colors = np.random.randint(0, 255, (n + 1, 3), dtype=np.uint8)
        new_state = {
            "masks": masks_3d,
            "outpix": outpix,
            "mask_classes": mc,
            "instance_colors": instance_colors,
            "pred_classes_map": result.classes_map,
            "flows": result.flows or [],
        }
        # When frozen, merge new result on top of existing channel masks instead of replacing.
        if self.view_config.masks_frozen:
            existing = self.channel_segmentations.get(int(channel_index))
            if existing and existing.get("masks") is not None:
                new_state = self._merge_channel_states_frozen(existing, new_state)
        self.channel_segmentations[int(channel_index)] = new_state

    def save_prediction(self, result: InferenceResult):
        """Saves the inference result to a _pred.npy file.

        Always persists channel_segmentations so that both channels' masks survive
        across multiple inference runs on different channels for the same frame.
        Merges the result channel's state with any other channels already on disk.
        """
        if not result.filename or result.masks is None:
            return

        base, ext = os.path.splitext(result.filename)
        frame_suffix = io.frame_id_to_suffix(result.frame_id)
        pred_path = f"{base}{frame_suffix}_pred.npy"
        channel_index = result.channel_index if result.channel_index is not None else 0

        masks = result.masks
        n = int(masks.max()) if masks.size else 0
        masks_3d = masks[np.newaxis, ...] if masks.ndim == 2 else np.array(masks, copy=True)

        live_state = self.channel_segmentations.get(channel_index)
        is_current_frame_result = (
            result.filename == self.filename
            and (result.frame_id is None or result.frame_id == self.frame_id)
            and channel_index == self.get_current_channel_index()
        )
        live_masks = live_state.get("masks") if live_state else None
        live_shape = tuple(np.asarray(live_masks).shape) if live_masks is not None else None

        if is_current_frame_result and live_state is not None and live_shape == tuple(masks_3d.shape):
            this_channel_state = {
                k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                for k, v in live_state.items()
            }
            outpix_for_file = this_channel_state.get("outpix")
            classes_for_file = this_channel_state.get("mask_classes", result.classes)
        else:
            outpix_for_file = utils.masks_to_outlines(masks_3d) * masks_3d
            classes = result.classes if result.classes is not None else None
            if classes is None or not np.any(np.asarray(classes)[1:n + 1] > 0):
                mask_classes = np.ones(n + 1, dtype=np.int16)
                mask_classes[0] = 0
            else:
                mask_classes = np.asarray(classes, dtype=np.int16)
            classes_for_file = mask_classes
            this_channel_state = {
                "masks": masks_3d,
                "outpix": outpix_for_file,
                "mask_classes": mask_classes,
                "instance_colors": np.random.randint(0, 255, (n + 1, 3), dtype=np.uint8),
                "pred_classes_map": result.classes_map,
                "flows": result.flows or [],
            }

        channel_segs = {}
        try:
            if os.path.exists(pred_path):
                existing = np.load(pred_path, allow_pickle=True).item()
                existing_ch = existing.get("channel_segmentations") or {}
                for k, v in existing_ch.items():
                    try:
                        channel_segs[int(k)] = v
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

        if self.view_config.masks_frozen and channel_index in channel_segs:
            existing_state = channel_segs[channel_index]
            if existing_state.get("masks") is not None:
                this_channel_state = self._merge_channel_states_frozen(existing_state, this_channel_state)
        channel_segs[channel_index] = this_channel_state

        ismanual = np.zeros(n, bool)
        dat = {
            "outlines": result.outlines if result.outlines is not None else outpix_for_file,
            "masks": result.masks,
            "chan_choose": [0, 0],
            "ismanual": ismanual,
            "filename": result.filename,
            "frame_id": result.frame_id,
            "flows": result.flows,
            "diameter": result.diameter,
            "classes": classes_for_file,
            "classes_map": result.classes_map,
            "class_names": result.class_names,
            "class_colors": result.class_colors,
            "active_channel_index": channel_index,
            "channel_segmentations": {str(k): v for k, v in channel_segs.items()},
        }
        np.save(pred_path, dat)
