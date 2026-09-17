"""
Main training entry point for BlueMind OrganelleNet.
"""

import os
import sys
import argparse
import json

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


    ZARR_MAP = build_zarr_map(root_dir)

  
    rfs_weights_path = os.path.join(cfg.path.jsons, cfg.rfs_weights)
   
    with open(split_paths['train_crops'], 'r') as f:
        train_crops_json = json.load(f)

    with open(rfs_weights_path, 'r') as f:
        rfs_weights_json = json.load(f)

    train_dataset = MISO2DDataset(
        crops_json_path = split_paths['train_crops'], 
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

    # Optional: Quick loop test to verify tensor output shapes
    print("--- DataLoader Pipeline Test ---")
    for batch_idx, (em_tensors, label_tensors) in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"EM Tensor Shape:    {em_tensors.shape}  | Dtype: {em_tensors.dtype}")
        print(f"Label Tensor Shape: {label_tensors.shape} | Dtype: {label_tensors.dtype}")
        break # Test only the first batch



    
    
   
if __name__ == "__main__":
    main()


