"""
Configuration system for OrganelleNet experiments.

Loads YAML config files with inheritance support (via 'inherits' key).
Merges child config on top of parent defaults to produce a flat ExperimentConfig.
"""

import os
import sys
import copy
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Project root resolution — allows imports from anywhere in the repo
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Default semantic → instance label mapping (shared across all baselines)
# ---------------------------------------------------------------------------
LABEL_MAP_30CLS =  {
            1:  0,   # ecs (background)
            35: 0,   # cyto (background)
            2:  1,   # pm
            3:  2,   # mito_mem
            4:  3,   # mito_lum
            5:  4,   # mito_ribo
            6:  5,   # golgi_mem
            7:  6,   # golgi_lum
            8:  7,   # ves_mem
            9:  8,   # ves_lum
            10: 9,   # endo_mem
            11: 10,  # endo_lum
            12: 11,  # lyso_mem
            13: 12,  # lyso_lum
            14: 13,  # ld_mem
            15: 14,  # ld_lum
            16: 15,  # er_mem
            17: 16,  # er_lum
            18: 17,  # eres_mem
            19: 18,  # eres_lum
            20: 19,  # ne_mem
            21: 20,  # ne_lum
            22: 21,  # np_out
            23: 22,  # np_in
            24: 23,  # hchrom
            26: 24,  # echrom
            28: 25,  # nucpl
            30: 26,  # mt_out
            36: 27,  # mt_in
            47: 28,  # perox_mem
            48: 29   # perox_lum
        }

CLASS_NAMES_30 = {
    0: "Background", 1: "pm", 2: "mito_mem", 3: "mito_lum",
    4: "mito_ribo", 5: "golgi_mem", 6: "golgi_lum", 7: "ves_mem",
    8: "ves_lum", 9: "endo_mem", 10: "endo_lum", 11: "lyso_mem", 12: "lyso_lum",
    13: "ld_mem", 14: "ld_lum", 15: "er_mem", 16: "er_lum", 17: "eres_mem",
    18: "eres_lum", 19: "ne_mem", 20: "ne_lum", 21: "np_out", 22: "np_in",
    23: "hchrom", 24: "echrom", 25: "nucpl", 26: "mt_out", 27: "mt_in",
    28: "perox_mem", 29: "perox_lum"
}

# Default class weights for the DiceCE loss (background-inclusive vector)
DEFAULT_WEIGHTS_30 = [
    0.1000, 0.1157, 0.1000, 0.1000, 5.0000, 0.9902, 0.7605, 2.0054, 4.3424, 0.3085, 
    0.1555, 3.5115, 0.9457, 5.0000, 1.4330, 0.1000, 0.1000, 5.0000, 5.0000, 0.9091, 
    0.8071, 5.0000, 5.0000, 0.1249, 5.0000, 0.1000, 1.0100, 2.6797, 5.0000, 3.2246
]

DEFAULT_WEIGHTS_14 = DEFAULT_WEIGHTS_13 + [1.7003]

target_classes = [
    "pm", "mito_mem", "mito_lum", "mito_ribo", "golgi_mem", "golgi_lum",
    "ves_mem", "ves_lum", "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
    "ld_mem", "ld_lum", "er_mem", "er_lum", "eres_mem", "eres_lum",
    "ne_mem", "ne_lum", "np_out", "np_in", "hchrom", "echrom",
    "nucpl", "mt_out", "mt_in", "perox_mem", "perox_lum"
]

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    main_dir: str = "/mnt/graid/codebases_other/cellmap-segmentation-challenge/saleem/organellenet"
    data_dir: str = "/mnt/graid/codebases_other/cellmap-segmentation-challenge/data"
    json_dir: str = ""   # auto-resolved

    def __post_init__(self):
        if not self.json_dir:
            self.json_dir = os.path.join(self.main_dir, "all_jsons")


@dataclass
class DataConfig:
    patch_dim: int = 128
    max_jitter: int = 48
    seed: int = 42
    num_classes: int = 30
    target_type: str = "labels"
    target_classes: List[str] = field(default_factory=lambda: target_classes)
    train_json: str = "train_anchors.json"
    val_json: str = "val_anchors.json"
    test_json: str = "test_anchors.json"




@dataclass
class ModelConfig:
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 30
    scale_conditioned: bool = False
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    kernel_size: int = 3
    up_kernel_size: int = 3
    num_res_units: int = 2
    act: str = "PRELU"
    norm: str = "INSTANCE"


@dataclass
class TrainingConfig:
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_mode: str = "min"     # 'min' for loss, 'max' for dice
    early_stop_patience: int = 12
    early_stop_metric: str = "val_loss"   # 'val_loss' or 'val_dice'
    loss_type: str = "dice_ce"
    accumulation_steps: int = 1
    print_freq: int = 1
    num_workers: int = 2
    sampler_balance_level: str = "raw"  # raw or organelle
    class_weights: Optional[List[float]] = None   # if None, uses defaults


@dataclass
class AugmentationConfig:
    """Configuration for the 3D EM augmentation pipeline."""
    enabled: bool = False
    # Geometric
    flip: bool = True
    rotate90: bool = True
    # Intensity (EM only)
    intensity_scale_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    intensity_shift_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
    gaussian_noise_std: float = 0.03
    gamma_range: List[float] = field(default_factory=lambda: [0.7, 1.5])
    # Elastic deformation
    elastic_enabled: bool = True
    elastic_alpha: float = 80.0
    elastic_sigma: float = 10.0
    elastic_prob: float = 0.3
    # Resolution augmentation (novel)
    resolution_aug_enabled: bool = False
    resolution_scale_range: List[float] = field(default_factory=lambda: [0.25, 1.0])
    resolution_aug_prob: float = 0.4


@dataclass
class ExperimentConfig:
    experiment_name: str = "unnamed-experiment"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Derived helpers (populated after load)
    label_map: Dict[int, int] = field(default_factory=dict, repr=False)
    class_names: Dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Auto-select label map and class names based on num_classes
        if not self.label_map:
            self.label_map = LABEL_MAP_30CLS
        if not self.class_names:
            self.class_names = CLASS_NAMES_30
        # Auto-select class weights if not provided
        if self.training.class_weights is None:
            self.training.class_weights = DEFAULT_WEIGHTS_30 
        # Auto-set scheduler mode from early_stop_metric
        if self.training.early_stop_metric == "val_dice":
            self.training.scheduler_mode = "max"
        else:
            self.training.scheduler_mode = "min"


# ---------------------------------------------------------------------------
# YAML Loading with Inheritance
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _dict_to_config(raw: dict) -> ExperimentConfig:
    """Convert a flat merged dict into an ExperimentConfig dataclass."""
    cfg = ExperimentConfig(
        experiment_name=raw.get("experiment_name", "unnamed-experiment"),
        paths=PathsConfig(**raw.get("paths", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        augmentation=AugmentationConfig(**raw.get("augmentation", {})),
    )
    return cfg


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load a YAML config, resolving inheritance via the 'inherits' key.

    If the config contains `inherits: base.yaml`, the base config is loaded
    first and the child config is merged on top.
    """
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Resolve inheritance
    parent_file = raw.pop("inherits", None)
    if parent_file:
        parent_path = os.path.join(config_dir, parent_file)
        if not os.path.isabs(parent_file):
            parent_path = os.path.join(config_dir, parent_file)
        with open(parent_path, "r") as f:
            parent_raw = yaml.safe_load(f) or {}
        # Remove inherits from parent (if any) to prevent recursive chains for now
        parent_raw.pop("inherits", None)
        raw = _deep_merge(parent_raw, raw)

    return _dict_to_config(raw)
