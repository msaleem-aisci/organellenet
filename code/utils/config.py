"""
Configuration system for BlueMind OrganelleNet.

Loads YAML config files into strict, type-enforced dataclasses.
The YAML file acts as the absolute source of truth; Python only enforces the schema.
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
# Strict Type Schemas (No default values - YAML must provide them)
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
        # We still need this purely to join the paths mathematically. 
        # YAML cannot perform os.path.join.
        if not os.path.isabs(self.checkpoint_dir):
            self.checkpoint_dir = os.path.join(self.root_dir, self.checkpoint_dir)
        if not os.path.isabs(self.train_crops_json):
            self.train_crops_json = os.path.join(self.root_dir, self.train_crops_json)
        if not os.path.isabs(self.val_crops_json):
            self.val_crops_json = os.path.join(self.root_dir, self.val_crops_json)
        if not os.path.isabs(self.test_crops_json):
            self.test_crops_json = os.path.join(self.root_dir, self.test_crops_json)


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
class ExperimentConfig:
    experiment_name: str
    path: PathConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    semantic_map: Dict[int, List[int]]


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
        semantic_map=raw["semantic_map"]
    )

def load_config(config_path: str) -> ExperimentConfig:


    config_path = os.path.abspath(config_path)
    
    config_dir = os.path.dirname(config_path)
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    parent_file = raw.pop("inherits", None)

    print(f"parent file path: {os.path.join(config_dir, parent_file)}")

    
    if parent_file:
        parent_path = os.path.join(config_dir, parent_file) if not os.path.isabs(parent_file) else parent_file
        with open(parent_path, "r") as f:
            parent_raw = yaml.safe_load(f) or {}
        parent_raw.pop("inherits", None)
        raw = _deep_merge(parent_raw, raw)

    return _dict_to_config(raw)