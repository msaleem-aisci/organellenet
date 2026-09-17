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
SEMANTIC_MAP_13 ={
            # 1. Mitochondria
            50: 1, 3: 1, 4: 1, 5: 1,
            
            # 2. Vesicles
            8: 2, 9: 2,
            
            # 3. Endosomes
            10: 3, 11: 3,
            
            # 4. Lysosomes
            12: 4, 13: 4,
            
            # 5. Lipid Droplets
            44: 5, 14: 5, 15: 5,
            
            # 6. Nucleus
            37: 6, 20: 6, 21: 6, 26: 6, 24: 6, 25: 6, 27: 6, 28: 6, 29: 6,
            
            # 7. Nuclear Pores
            22: 7, 23: 7,
            
            # 8. Microtubules
            30: 8, 36: 8,
            
            # 9. Peroxisomes
            49: 9, 47: 9, 48: 9,
            
            # 10. Golgi Apparatus
            6: 10, 7: 10,
            
            # 11. Endoplasmic Reticulum
            16: 11, 17: 11, 64: 11,
            
            # 12. ER Exit Sites
            18: 12, 19: 12,
            
            # 13. Background
            35: 0, 1: 0 
        }

# ---------------------------------------------------------------------------
# Strict Type Schemas
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    root_dir: str
    blueprint_json: str
    RFS_weights: str
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
        if not os.path.isabs(self.blueprint_json):
            self.blueprint_json = os.path.join(self.root_dir, 'all_jsons')

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
    blueprint_json: str
    rfs_weights: str
 

    # We use default_factory to automatically load the Python dictionary.
    # It will be universally applied to every experiment.
    semantic_map: Dict[int, int] = field(default_factory=lambda: SEMANTIC_MAP_13)

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
        blueprint_json = raw['blueprint_json']
        rfs_weights= raw['blueprint_json']
     

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