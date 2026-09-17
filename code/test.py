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
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=64, 
        num_workers=4,
        shuffle=False,    
        prefetch_factor=2,
        pin_memory=True
    )

    # Optional: Quick loop test to verify tensor output shapes
    print("--- DataLoader Pipeline Test ---")
    print(f"Train Loader: {len(train_dataloader)}")
    print(f"Val Loader: {len(val_dataloader)}")


    

    for batch_idx, (em_batch, lbl_batch) in enumerate(train_dataloader):
        print(f"EM Batch Shape:    {em_batch.shape} | Type: {em_batch.dtype}")
        print(f"Label Batch Shape: {lbl_batch.shape} | Type: {lbl_batch.dtype}")

        # Extract the second item in the batch
        em_tensor = em_batch[1]
        lbl_tensor = lbl_batch[1]
        print(f"em_tensor {em_tensor.shape}")
        break

    # Convert to 2D numpy arrays: (128, 128)
    em_single = em_tensor.squeeze(0).numpy()
    lbl_single = lbl_tensor.numpy()

    patch_dim = 128
    mid_pt = patch_dim // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')

    # Plot entire 2D patch
    axes[0].imshow(em_single, cmap='gray')
    axes[0].set_title("EM Patch (2D)", color='white')
    axes[0].axis('off')

    # Handle the -1 ignore index for visualization (map to 0)
    visual_lbl = np.where(lbl_single == -1, 0, lbl_single)
    axes[1].imshow(visual_lbl, cmap='nipy_spectral', interpolation='nearest')
    axes[1].set_title("Label Patch (2D)", color='white')
    axes[1].axis('off')

    # Apply crosshairs at the spatial center of the 2D plane
    for ax in axes:
        ax.axhline(mid_pt, color='blue', linestyle='-', linewidth=1)
        ax.axvline(mid_pt, color='blue', linestyle='-', linewidth=1)

    output_image_path = "/kaggle/working/debug_batch.png"
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig) 

    print(f"Visualization saved to {output_image_path}")



    
    
   
if __name__ == "__main__":
    main()


