"""
Configuration system for BlueMind OrganelleNet.
"""

import os
import sys
import copy
import yaml
from dataclasses import dataclass, field
from typing import List, Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Hardcoded Biological Constants (No longer needed in YAML)
# ---------------------------------------------------------------------------
atomic_to_channel = {
    # Complex
    3:  1,   # mito_mem
    4:  2,   # mito_lum
    16: 3,   # er_mem
    17: 4,   # er_lum
    6:  5,   # golgi_mem
    7:  6,   # golgi_lum
    # Simple
    13: 7,   # lyso_lum
    15: 8,   # ld_lum
    9:  9,   # ves_lum
    11: 10,  # endo_lum
    48: 11,  # perox_lum
    # Structural
    20: 12,  # ne_mem
    22: 13,  # np_out
    30: 14,  # mt_out
    # Background
    1:  0,   # ecs
    35: 0    # cyto
}

# Because we are using 14 sementic classes, these are the crops that have no these 14 classes. so pytorch dataset class
# converts them into -1. therefore, I removed them from training. 
UNWANTED_CROPS = {
    # Original structural removals
    "crop337", "crop247", "crop357",'crop358',
    
    # Original background-only crops
    "crop243", "crop56", "crop57", "crop58", "crop59", "crop54", "crop55", 
    "crop60", "crop61", "crop62", "crop63", "crop64", "crop65", "crop66", 
    "crop67", "crop68", "crop69", "crop70", "crop71", "crop72", "crop73", 
    "crop74", "crop75", "crop76", "crop77", "crop282", "crop25", "crop26", 
    "crop81", "crop82", "crop83", "crop84", "crop97", "crop98", "crop99",
    
    # Empty crops from Batch 1 & 2
    "crop257", "crop238", "crop94", "crop95", "crop96", "crop85", "crop86",
    "crop87", "crop88", "crop89", "crop90", "crop91", "crop92", "crop93",
    "crop423", "crop452", "crop472", "crop421", "crop179", "crop184", 
    "crop221", "crop229", "crop230", "crop231", "crop473", "crop289", 
    "crop354", "crop355", "crop356", "crop362", "crop366", "crop367", 
    "crop387", "crop408",
    
    # Empty crops from Final Batch
    "crop378", "crop379", "crop380", "crop381", "crop177",
    # Newly identified empty crops (Lipid Droplet, Nucleus, Peroxisome Parent Only)
    "crop324", "crop329", "crop336", "crop347", "crop348", "crop349", "crop351", 
    "crop353", "crop386", "crop407", "crop410", "crop411", "crop412", "crop413"
}

# ---------------------------------------------------------------------------
# Strict Type Schemas
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    root_dir: str
    jsons: str
    dataset: str
    train_crops_json: str
    val_crops_json: str
    test_crops_json: str
    checkpoint_dir: str


    def __post_init__(self):
        if not os.path.isabs(self.checkpoint_dir):
            self.checkpoint_dir = os.path.join(self.root_dir, self.checkpoint_dir)
        if not os.path.isabs(self.train_crops_json):
            self.train_crops_json = os.path.join(self.root_dir, self.train_crops_json)
        if not os.path.isabs(self.val_crops_json):
            self.val_crops_json = os.path.join(self.root_dir, self.val_crops_json)
        if not os.path.isabs(self.test_crops_json):
            self.test_crops_json = os.path.join(self.root_dir, self.test_crops_json)
        if not os.path.isabs(self.jsons):
            self.jsons = os.path.join(self.root_dir, 'all_jsons')

@dataclass
class DataConfig:
    patch_dim: int
    samples: int
    batch_size: int
    num_workers: int

@dataclass
class ModelConfig:
    spatial_dims: int
    in_channels: int
    out_channels: int
    channels: List[int]
    strides: List[int]

@dataclass
class TrainingConfig:
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    mixed_precision: bool
    early_stopping_patience: int
    print_freq: int
    entropy_masking: bool
    loss_function: str

@dataclass
class ArchConfig:
    unet: int
    swin: int
    resnet: int

@dataclass
class ExperimentConfig:
    experiment_name: str
    path: PathConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    rfs_weights: str
 

    # We use default_factory to automatically load the Python dictionary.
    # It will be universally applied to every experiment.
    semantic_map: Dict[int, int] = field(default_factory=lambda: atomic_to_channel)

    unwanted_crops: set[str] = field(
        default_factory=lambda: UNWANTED_CROPS
    )


# ---------------------------------------------------------------------------
# YAML Loading with Inheritance
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def _dict_to_config(raw: dict) -> ExperimentConfig:
    """Passes the raw YAML dictionaries into the strict Python schemas."""
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        path=PathConfig(**raw["path"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        rfs_weights= raw['rfs_weights']
     

    )

def load_config(config_path: str) -> ExperimentConfig:
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    parent_file = raw.pop("inherits", None)
    if parent_file:
        parent_path = os.path.join(config_dir, parent_file) if not os.path.isabs(parent_file) else parent_file
        with open(parent_path, "r") as f:
            parent_raw = yaml.safe_load(f) or {}
        parent_raw.pop("inherits", None)
        raw = _deep_merge(parent_raw, raw)

    return _dict_to_config(raw)