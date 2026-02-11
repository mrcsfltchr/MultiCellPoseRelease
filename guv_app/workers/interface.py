from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class AnalysisPlugin(ABC):
    """
    Abstract base class for analysis plugins.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the analysis plugin."""
        pass

    @abstractmethod
    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        """
        Runs analysis on the given image and masks.
        Returns a DataFrame where each row corresponds to an object (or the whole image).
        """
        pass

    def visualize(self, parent, data: pd.DataFrame, **kwargs):
        """
        Optional method to visualize results.
        
        Args:
            parent: The parent widget (e.g. QWidget) to attach visualization to.
            data: The DataFrame returned by run() (or aggregated results).
            **kwargs: Additional arguments (e.g. image, masks if available).
        """
        pass