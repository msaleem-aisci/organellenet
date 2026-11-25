"""
Loss function factory for OrganelleNet.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class BoundaryWeightedMSELoss(nn.Module):
    def __init__(self, boundary_weight=10.0, threshold=0.95):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.threshold = threshold

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction="none")
        weight_mask = torch.ones_like(target)
        weight_mask[torch.abs(target) < self.threshold] = self.boundary_weight
        return (mse * weight_mask).mean()


class MaskedDiceCELoss(nn.Module):

    def __init__(self, num_classes=30, weight=None, ignore_index=-1, include_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if weight is not None and len(weight) == num_classes - 1:
            # If 29 foreground weights are passed, prepend 0.1 for Background (index 0) for CE
            ce_weight = torch.cat([torch.tensor([0.1], device=weight.device, dtype=weight.dtype), weight])
            dice_weight = weight if not include_background else ce_weight
        else:
            ce_weight = weight
            dice_weight = weight[1:] if (weight is not None and not include_background) else weight

        self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)
        self.dice = DiceLoss(
            to_onehot_y=False,
            softmax=False,
            include_background=include_background,
            weight=dice_weight,
            batch=True,
        )

    def forward(self, logits, target):
        # target shape: [B, D, H, W] or [B, 1, D, H, W]
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)
        target = target.long()

        # 1. Cross-Entropy natively ignores ignore_index (-1)
        ce_loss = self.ce(logits, target)

        # 2. Mask out -1 voxels before one-hot encoding for Dice
        valid_mask = target != self.ignore_index  # [B, D, H, W]
        safe_target = torch.where(valid_mask, target, torch.zeros_like(target))

        target_one_hot = F.one_hot(safe_target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

        valid_mask_5d = valid_mask.unsqueeze(1).float()  # [B, 1, D, H, W]
        probs = torch.softmax(logits, dim=1) * valid_mask_5d
        target_one_hot = target_one_hot * valid_mask_5d

        dice_loss = self.dice(probs, target_one_hot)
        return dice_loss + ce_loss


def build_loss(config, device=None):
    """
    Build a Loss function from an ExperimentConfig.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_type = getattr(config.training, "loss_type", "dice_ce")

    if loss_type == "mse":
        criterion = BoundaryWeightedMSELoss(boundary_weight=10.0, threshold=0.95)
        print("Loss: BoundaryWeightedMSELoss (Weight: 10.0 | for SDT Regression)")
        return criterion
    elif loss_type == "smooth_l1":
        criterion = torch.nn.SmoothL1Loss()
        print("Loss: SmoothL1Loss (for Regression)")
        return criterion
    else:
        raw_weights = getattr(config.training, "class_weights", None)
        weights = (
            torch.tensor(raw_weights, dtype=torch.float32, device=device)
            if raw_weights is not None
            else None
        )
        num_classes = getattr(config.data, "num_classes", 30)

        criterion = MaskedDiceCELoss(
            num_classes=num_classes,
            weight=weights,
            ignore_index=-1,
            include_background=False,
        )

        w_info = f"{len(raw_weights)} classes" if weights is not None else "uniform"
        print(f"Loss: MaskedDiceCELoss (ignore_index=-1) | Class weights: {w_info}")
        return criterion