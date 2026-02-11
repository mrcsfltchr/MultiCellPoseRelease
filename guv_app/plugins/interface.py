from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any

class AnalysisPlugin(ABC):
    """
    Abstract base class for analysis plugins.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The display name of the plugin.
        """
        pass

    @abstractmethod
    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        """
        Run the analysis on the given image and masks.

        Args:
            image: The input image (H, W) or (H, W, C).
            masks: The segmentation masks (H, W).
            classes: Optional array of class IDs corresponding to mask labels.
            **kwargs: Additional parameters (e.g. settings).

        Returns:
            pd.DataFrame: A DataFrame containing the analysis results.
        """
        pass

    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary defining the parameters for the plugin.
        
        The returned dictionary should map parameter keys to configuration dictionaries.
        Supported fields in configuration dictionary:
            - 'type': str, one of ['int', 'float', 'bool', 'str', 'enum']
            - 'default': Any, default value
            - 'label': str, display label (optional, defaults to key)
            - 'help': str, tooltip text (optional)
            - 'min': int/float, minimum value (for int/float)
            - 'max': int/float, maximum value (for int/float)
            - 'options': list, valid options (for enum)
        """
        return {}

    def visualize(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Optional: Generates a visualization mask (integer labels) based on the analysis.
        Returns None if visualization is not supported.
        """
        return None