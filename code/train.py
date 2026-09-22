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
from code.training.trainer import Trainer
from code.utils.class_to_crop import Class_To_Crop

def parse_args():

    parser = argparse.ArgumentParser(description="Train BlueMind OrganelleNet")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file (e.g., configs/base.yml)")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Absolute path to a manual checkpoint file.")
    parser.add_argument("--resume_logs", type=str, default=None, help="Absolute path to a manual CSV logs.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--samples", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")

    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment Name")
    parser.add_argument("--entropy_masking", type=bool, default=None, help="Batch size")
    parser.add_argument("--loss_function", type=str, default=None, help="Loss Function")


    return parser.parse_args()

def main():

    args = parse_args()
    
    # 2. Load the configuration
    print(f"Loading configuration from: {args.config}...\n")
    cfg = load_config(args.config)


    if args.num_epochs is not None:
        cfg.training.num_epochs = args.num_epochs

    if args.samples is not None:
        cfg.data.samples = args.samples

    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name

    if args.entropy_masking is not None:
        cfg.training.entropy_masking = args.entropy_masking


    if args.loss_function is not None:
        cfg.training.loss_function = args.loss_function


    # if args.entropy_masking is not None:
    #     cfg.training.entropy_masking = args.entropy_masking

    root_dir = cfg.path.root_dir
    split_paths = split_handler(cfg.path.jsons, root_dir)

    print(f">> Exp Name: {cfg.experiment_name}")
    ZARR_MAP = build_zarr_map(cfg.path.dataset)

  

    
    ctc = Class_To_Crop(ZARR_MAP)
    print(f"Class Mapper: {ctc.lass_mapper()}")


    sys.exist(0)
    rfs_weights_path = os.path.join(cfg.path.jsons, cfg.rfs_weights)
   
    with open(rfs_weights_path, 'r') as f:
        rfs_weights_json = json.load(f)


    train_dataset = MISO2DDataset(
        crops_json_path = split_paths['train_crops'], 
        zarr_map = ZARR_MAP,
        class_map = cfg.semantic_map,
        unwanted_crops = cfg.unwanted_crops
    )

    val_dataset = MISO2DDataset(
        crops_json_path = split_paths['val_crops'], 
        zarr_map = ZARR_MAP,
        class_map = cfg.semantic_map,
        unwanted_crops = cfg.unwanted_crops
    )



    train_sampler = create_rfs_sampler(
        crops_list=train_dataset.crops, 
        rfs_weights_dict= rfs_weights_json, 
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

    print("Dataset is completed.")
    model, device = build_model(cfg)
    criterion = build_loss(cfg)
 

    trainer = Trainer(
        model=model,
        criterion=criterion,
        config=cfg,
        device=device
    )

    # Start the training loop
    print("\n[STARTING TRAINING PIPELINE]")
    trainer.train(train_dataloader, val_dataloader, args, resume=True)







    
    
   
if __name__ == "__main__":
    main()


