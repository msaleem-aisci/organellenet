"""
Main training entry point for BlueMind OrganelleNet.
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# train.py is located inside the 'code/' directory.
# We move one level up ("..") to reach the 'organellenet' root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader


from code.utils.config import load_config
from data.splits import split_handler
from data.zarr_utils import build_zarr_map, select_z_slice, extract_both_patches
from data.dataset import MISO2DDataset
from data.sampler import create_rfs_sampler 
from code.training.losses import build_loss
from code.models.unet import build_model


def parse_args():

    parser = argparse.ArgumentParser(description="Train BlueMind OrganelleNet")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file (e.g., configs/base.yml)")
    
    return parser.parse_args()

def main():

    args = parse_args()
    
    # 2. Load the configuration
    print(f"Loading configuration from: {args.config}...\n")
    cfg = load_config(args.config)
    root_dir = cfg.path.root_dir
    split_paths = split_handler(cfg.path.jsons, root_dir)


    ZARR_MAP = build_zarr_map(cfg.path.dataset)

  
    rfs_weights_path = os.path.join(cfg.path.jsons, cfg.rfs_weights)
   
    with open(rfs_weights_path, 'r') as f:
        rfs_weights_json = json.load(f)



    train_dataset = MISO2DDataset(
        crops_json_path = split_paths['train_crops'], 
        zarr_map = ZARR_MAP,
    )

    val_dataset = MISO2DDataset(
        crops_json_path = split_paths['val_crops'], 
        zarr_map = ZARR_MAP,
    )



    train_sampler = create_rfs_sampler(
        crops_list=train_dataset.crops, 
        rfs_weights_dict= rfs_weights_json, 
        num_samples=8000  
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=64, 
        sampler=train_sampler, 
        num_workers=4,   
        prefetch_factor=2,
        pin_memory=True
    )
    # val_dataloader = DataLoader(
    #     val_dataset, 
    #     batch_size=64, 
    #     num_workers=4,
    #     shuffle=False,    
    #     prefetch_factor=2,
    #     pin_memory=True
    # )

    model, device = build_model(cfg)
    criterion = build_loss(cfg)
    print(model)







    
    
   
if __name__ == "__main__":
    main()


