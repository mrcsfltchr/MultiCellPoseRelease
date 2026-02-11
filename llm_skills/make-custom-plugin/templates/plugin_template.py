import numpy as np
import pandas as pd

from guv_app.plugins.interface import AnalysisPlugin
from guv_app.plugins.validator import validate_visualization_mask


class ExamplePlugin(AnalysisPlugin):
    @property
    def name(self) -> str:
        return "Example Plugin"

    def get_parameter_definitions(self):
        return {
            "example_param": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 100,
                "label": "Example Parameter",
                "help": "Explain what this parameter controls.",
            }
        }

    def run(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> pd.DataFrame:
        if masks is None or np.max(masks) == 0:
            return pd.DataFrame()

        # Replace with real logic
        return pd.DataFrame({"mask_id": [1], "value": [0.0]})

    def visualize(self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray = None, **kwargs) -> np.ndarray:
        # Replace with real visualization logic
        viz = np.asarray(masks).astype(np.int32, copy=False)
        validate_visualization_mask(viz, np.asarray(masks))
        return viz
