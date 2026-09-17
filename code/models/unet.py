"""
Model factory for OrganelleNet.

Builds MONAI UNet models from an ExperimentConfig, with optional
multi-GPU DataParallel wrapping.
"""

import sys
import os
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def build_model(config, device=None, multi_gpu=True):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc = config.model

  
    model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=13, # Update to 32 if you transition to the full CellMap benchmark
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=0, # CRITICAL: Disables residuals to match paper's standard U-Net
            norm=Norm.BATCH
            ).to(device)


    num_gpus = torch.cuda.device_count()
    print(f"Model built: {mc.out_channels}-class UNet | Device: {device} | GPUs: {num_gpus}")

    if multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapped model with DataParallel across {num_gpus} GPUs.")

    return model, device


def get_raw_model(model):
    """Unwrap DataParallel or Wrappers if present to access the raw model."""
    if isinstance(model, nn.DataParallel):
        model = model.module
    if hasattr(model, "base_model"):
        model = model.base_model
    return model
