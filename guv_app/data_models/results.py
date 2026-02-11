from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Any

@dataclass
class InferenceResult:
    masks: Optional[np.ndarray] = None
    flows: Optional[List[Any]] = None
    styles: Optional[np.ndarray] = None
    diams: Optional[float] = None
    filename: Optional[str] = None
    frame_id: Optional[str] = None
    is_saved: bool = False
    classes: Optional[np.ndarray] = None
    classes_map: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    class_colors: Optional[np.ndarray] = None
    outlines: Optional[np.ndarray] = None
    diameter: Optional[float] = None


@dataclass
class TrainingResult:
    model_path: str
    train_losses: Optional[List[float]] = None
    test_losses: Optional[List[float]] = None
    artifact_uris: Optional[dict] = None
    artifact_paths: Optional[dict] = None
