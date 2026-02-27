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

    def get_parameter_definitions(self):
        return {
            "intensity_channel": {
                "type": "enum",
                "default": "all",
                "options": ["all", "1", "2", "3"],
                "label": "Intensity Channel",
                "help": "'all' averages channels. 1/2/3 selects R/G/B channel.",
            }
        }

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if regionprops_table is None:
            raise ImportError("scikit-image is required for BasicStatsPlugin")

        if masks.max() == 0:
            return pd.DataFrame()

        # Ensure intensity image matches mask dimensions
        intensity_img = image
        intensity_channel = _parse_intensity_channel(kwargs.get("intensity_channel", "all"))
        channel_name = _format_intensity_channel_name(intensity_channel)
        if image.ndim == 2 and masks.ndim == 3 and masks.shape[0] == 1 and image.shape == masks.shape[1:]:
            intensity_img = image[np.newaxis, ...]
        elif image.ndim == 3 and masks.ndim == 2:
            if intensity_channel < 0:
                intensity_img = image.mean(axis=2)
            else:
                if intensity_channel >= image.shape[2]:
                    raise ValueError(
                        f"Requested intensity_channel={intensity_channel}, but image has {image.shape[2]} channels"
                    )
                intensity_img = image[..., intensity_channel]
        
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

        df["intensity_channel_name"] = channel_name
        return df


def _parse_intensity_channel(value) -> int:
    if value is None:
        return -1
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("all", ""):
            return -1
        if v in ("1", "2", "3"):
            return int(v) - 1
        try:
            return int(v)
        except ValueError as exc:
            raise ValueError(f"Invalid intensity_channel value: {value}") from exc
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Invalid intensity_channel value: {value}") from exc


def _format_intensity_channel_name(channel_index: int) -> str:
    if channel_index < 0:
        return "all"
    return f"ch{channel_index + 1}"
