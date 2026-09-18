"""
Main training entry point for BlueMind OrganelleNet.
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.utils.config import load_config
from code.data.splits import split_handler
from code.data.zarr_utils import build_zarr_map
from code.data.dataset import MISO2DDataset
from code.data.sampler import create_rfs_sampler 
from code.training.losses import build_loss
from code.models.unet import build_model
from code.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train BlueMind OrganelleNet")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file (e.g., configs/base.yml)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Load the configuration
    print(f"Loading configuration from: {args.config}...\n")
    cfg = load_config(args.config)
    root_dir = cfg.path.root_dir
    
    # 2. Data Split Routing
    split_paths = split_handler(cfg.path.jsons, root_dir)
    ZARR_MAP = build_zarr_map(cfg.path.dataset)

    rfs_weights_path = os.path.join(cfg.path.jsons, cfg.path.RFS_weights)
    with open(rfs_weights_path, 'r') as f:
        rfs_weights_json = json.load(f)

    # 3. Dataset Instantiation
    train_dataset = MISO2DDataset(
        crops_json_path = split_paths['train_crops'], 
        zarr_map = ZARR_MAP,
    )

    val_dataset = MISO2DDataset(
        crops_json_path = split_paths['val_crops'], 
        zarr_map = ZARR_MAP,
    )

    # 4. Sampler and Dataloaders (Pulling parameters from cfg)
    train_sampler = create_rfs_sampler(
        crops_list=train_dataset.crops, 
        rfs_weights_dict=rfs_weights_json, 
        num_samples=cfg.data.samples  
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        sampler=train_sampler, 
        num_workers=cfg.data.num_workers,   
        prefetch_factor=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers,
        shuffle=False,    
        prefetch_factor=2,
        pin_memory=True
    )

    # 5. Model & Criterion Assembly
    model, device = build_model(cfg)
    criterion = build_loss(cfg)
    
    print("\n[Model and Loss Successfully Initialized]")

    # 6. Execute Training Pipeline
    trainer = Trainer(
        model=model,
        criterion=criterion,
        config=cfg,
        device=device
    )

    trainer.train(train_dataloader, val_dataloader, resume=True)


if __name__ == "__main__":
    main()