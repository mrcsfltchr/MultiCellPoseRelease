import numpy as np
import pandas as pd
from guv_app.plugins.interface import AnalysisPlugin

try:
    from skimage.measure import regionprops_table
except ImportError:
    regionprops_table = None

class BasicStatsPlugin(AnalysisPlugin):
    """
    Calculates basic morphological and intensity statistics.
    """
    @property
    def name(self) -> str:
        return "Basic Statistics"

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if regionprops_table is None:
            raise ImportError("scikit-image is required for BasicStatsPlugin")

        if masks.max() == 0:
            return pd.DataFrame()

        # Ensure intensity image matches mask dimensions
        intensity_img = image
        if image.ndim == 3 and masks.ndim == 2:
            # If RGB/multichannel, take mean for intensity or use specific channel if provided
            # For basic stats, we'll just use the mean of channels to get a 2D intensity map
            intensity_img = image.mean(axis=2)
        
        props = ['label', 'area', 'mean_intensity', 'centroid']
        data = regionprops_table(masks, intensity_image=intensity_img, properties=props)
        
        df = pd.DataFrame(data)
        
        # Rename centroid columns for clarity
        if 'centroid-0' in df.columns:
            df.rename(columns={'centroid-0': 'centroid_y', 'centroid-1': 'centroid_x'}, inplace=True)
            if 'centroid-2' in df.columns:
                df.rename(columns={'centroid-2': 'centroid_z'}, inplace=True)

        # Add class information if available
        if classes is not None:
            # classes array is indexed by mask_id. 
            # df['label'] contains mask_ids.
            # We map label -> class_id
            def get_class(label):
                return classes[label] if label < len(classes) else 0
            df['class_id'] = df['label'].apply(get_class)

        return df