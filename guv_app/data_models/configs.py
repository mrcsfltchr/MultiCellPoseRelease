
from dataclasses import dataclass, field
from typing import Optional, List
from typing import List

@dataclass
class InferenceConfig:
    """Bundles all parameters needed to run segmentation."""
    diameter: float = 30.0
    model_path: str = "cpsam"

@dataclass
class TrainingConfig:
    """Bundles all parameters for training a new model."""
    base_model: str = "cpsam"
    model_name: str = "cellpose_model"
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    n_epochs: int = 300
    batch_size: int = 10
    min_train_masks: int = 0
    bsize: int = 256
    rescale: bool = False
    scale_range: float = 0.5
    use_lora: bool = False
    lora_blocks: Optional[int] = None
    unfreeze_blocks: int = 9
    save_path: Optional[str] = None
    train_files: List[str] = field(default_factory=list)
    train_labels_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    test_labels_files: List[str] = field(default_factory=list)

@dataclass
class ViewConfig:
    """Holds the state of the UI's view controls."""
    masks_visible: bool = True
    outlines_visible: bool = False
    color_by_class: bool = False
    autosave_enabled: bool = True
    class_visible: List[bool] = field(default_factory=list)

@dataclass
class RemoteConfig:
    """Configuration for Remote connection."""
    address: str = "localhost:50051"
    hostname: str = ""
    insecure: bool = True
    token: str = "dev-token"
    user_id: Optional[str] = None

    username: str = ""
    password: str = ""
    key_path: str = ""
    
    # SSH Tunneling defaults
    ssh_port: int = 22
    ssh_local_port: int = 50051
    ssh_remote_port: int = 50051
    ssh_remote_bind: str = "localhost"
    
@dataclass
class BatchConfig:
    batch_size: int = 32
    """Configuration for Batch processing."""
    pass
