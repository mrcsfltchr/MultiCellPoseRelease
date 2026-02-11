import logging
import numpy as np
import pandas as pd
from guv_app.plugins.interface import AnalysisPlugin
from guv_app.plugins.validator import validate_visualization_mask
from cellpose.utils import masks_to_edges

_logger = logging.getLogger(__name__)

class PerimeterIntensityPlugin(AnalysisPlugin):
    """
    Calculates intensity statistics on the perimeter of masks.
    Generates a visualization of the perimeter regions as a mask layer.
    """
    @property
    def name(self) -> str:
        return "Perimeter Intensity"

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if masks.max() == 0:
            return pd.DataFrame()

        # Parameter defaults are handled by AnalysisService based on get_parameter_definitions
        thickness = kwargs.get('thickness', 1)
        
        visualization_masks = kwargs.get("visualization_masks")
        if visualization_masks is not None:
            validate_visualization_mask(visualization_masks, masks)
            perimeter_masks = visualization_masks
        else:
            # Generate perimeter masks (labeled with original IDs)
            perimeter_masks = self.get_visualization_mask(masks, thickness)
        
        # Ensure intensity image matches mask dimensions
        intensity_img = image
        try:
            intensity_img = _match_intensity_to_masks(intensity_img, masks)
        except ValueError as exc:
            _logger.warning(f"Perimeter Intensity: {exc}")
            return pd.DataFrame()
        
        labels = perimeter_masks.astype(np.int64, copy=False).ravel()
        intensities = intensity_img.astype(np.float64, copy=False).ravel()
        max_id = int(labels.max()) if labels.size > 0 else 0
        if max_id == 0:
            return pd.DataFrame()

        sums = np.bincount(labels, weights=intensities, minlength=max_id + 1)
        counts = np.bincount(labels, minlength=max_id + 1)

        mask_ids = np.arange(1, max_id + 1)
        valid = counts[mask_ids] > 0
        mask_ids = mask_ids[valid]
        if mask_ids.size == 0:
            return pd.DataFrame()

        means = sums[mask_ids] / counts[mask_ids]
        df = pd.DataFrame({
            "label": mask_ids,
            "perimeter_mean_intensity": means,
        })

        # Add class information if available
        if classes is not None:
            # classes array is indexed by mask_id. 
            # df['label'] contains mask_ids.
            def get_class(label):
                return classes[label] if label < len(classes) else 0
            df['class_id'] = df['label'].apply(get_class)

        return df


    def get_visualization_mask(self, masks: np.ndarray, thickness: int = 1) -> np.ndarray:
        """
        Generates a mask array representing the perimeters of the objects.
        Validates that the output is compatible with the original segmentation.
        """
        # masks_to_edges returns a boolean array where edges are True
        edges = masks_to_edges(masks, threshold=thickness)
        
        # Apply edges to the original masks to get labeled perimeters
        # Casting edges to masks.dtype ensures we get the label IDs back
        perimeter_masks = masks * edges.astype(masks.dtype)
        
        # Validate the generated mask against the original to ensure compatibility
        validate_visualization_mask(perimeter_masks, masks)
        
        return perimeter_masks

    def get_parameter_definitions(self):
        return {
            "thickness": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "label": "Perimeter Thickness",
                "help": "Thickness of the perimeter in pixels"
            }
        }

    def visualize(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> np.ndarray:
        thickness = kwargs.get('thickness', 1)
        return self.get_visualization_mask(masks, thickness)


def _match_intensity_to_masks(intensity_img: np.ndarray, masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(intensity_img)
    if masks.ndim == 3 and masks.shape[0] == 1 and arr.ndim == 2 and arr.shape == masks.shape[1:]:
        return arr[np.newaxis, ...]
    if arr.shape == masks.shape:
        return arr
    arr = np.squeeze(arr)
    if arr.shape == masks.shape:
        return arr

    if masks.ndim == 2:
        if arr.ndim >= 3:
            shape = arr.shape
            yx_axes = []
            for idx, dim in enumerate(shape):
                if dim in masks.shape:
                    yx_axes.append(idx)
            if len(yx_axes) >= 2:
                # pick the two axes that match mask dims best
                a0, a1 = yx_axes[0], yx_axes[1]
                if (shape[a0], shape[a1]) != masks.shape and (shape[a1], shape[a0]) == masks.shape:
                    arr = np.moveaxis(arr, (a0, a1), (0, 1))
                else:
                    arr = np.moveaxis(arr, (a0, a1), (0, 1))
                # reduce remaining axes (channels/time) by mean
                while arr.ndim > 2:
                    arr = arr.mean(axis=-1)
        if arr.ndim == 2 and arr.shape != masks.shape:
            raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
        return arr

    if masks.ndim == 3:
        if arr.ndim == 3 and arr.shape[:2] == masks.shape[1:]:
            arr = arr.mean(axis=-1)
            arr = arr[np.newaxis, ...]
        if arr.ndim >= 4:
            shape = arr.shape
            zyx_axes = []
            for idx, dim in enumerate(shape):
                if dim in masks.shape:
                    zyx_axes.append(idx)
            if len(zyx_axes) >= 3:
                arr = np.moveaxis(arr, zyx_axes[:3], (0, 1, 2))
                while arr.ndim > 3:
                    arr = arr.mean(axis=-1)
        if arr.shape != masks.shape:
            raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
        return arr

    raise ValueError(f"Intensity image shape {arr.shape} does not match masks {masks.shape}")
