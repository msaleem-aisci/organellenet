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
    def __init__(self, boundary_weight=10.0, threshold=0.95, ignore_index=-100.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.threshold = threshold
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # 1. Create spatial mask (True where target is valid)
        valid_mask = (target != self.ignore_index).float()
        
        # 2. Compute raw unreduced MSE
        mse = F.mse_loss(pred, target, reduction='none')
        
        # 3. Compute boundary weights
        weight_mask = torch.ones_like(target)
        weight_mask[torch.abs(target) < self.threshold] = self.boundary_weight
        
        # 4. Apply masks
        masked_mse = mse * weight_mask * valid_mask
        
        # 5. Take the mean only over the valid pixels
        return masked_mse.sum() / valid_mask.sum().clamp(min=1)


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, ignore_index=-100.0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        valid_mask = (target != self.ignore_index).float()
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        masked_loss = loss * valid_mask
        return masked_loss.sum() / valid_mask.sum().clamp(min=1)


class MaskedDiceCELoss(nn.Module):
    def __init__(self, weight, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(include_background=False, softmax=False, to_onehot_y=False)
        
    def forward(self, logits, targets):
        # 1. Ensure targets are 4D [B, Z, Y, X] for CrossEntropy
        if targets.dim() == logits.dim():
            targets_4d = targets.squeeze(1)
        else:
            targets_4d = targets

        ce_loss = self.ce(logits, targets_4d.long())
        
        # 2. Create a 5D valid mask [B, 1, Z, Y, X] for correct broadcasting
        valid_mask = (targets_4d != self.ignore_index).unsqueeze(1).float()
        
        # 3. Safely zero out the -100 regions before one-hot encoding
        safe_targets = torch.where(
            targets_4d != self.ignore_index, 
            targets_4d, 
            torch.zeros_like(targets_4d)
        )
        
        probs = torch.softmax(logits, dim=1)
        
        # One-hot returns [B, Z, Y, X, C], permute to [B, C, Z, Y, X]
        targets_onehot = F.one_hot(safe_targets.long(), num_classes=logits.shape[1])
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()
        
        # 4. Apply the 5D mask (zeros out both intersection and union for ignored voxels)
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask
        
        dice_loss = self.dice(probs, targets_onehot)
        
        # MONAI DiceLoss computes: 1 - mean(dice). We reverse it to get the raw score.
        dice_score = 1.0 - dice_loss 
        
        return ce_loss + dice_loss, dice_score


def build_loss(config, device=None):
    """
    Build a Loss function from an ExperimentConfig.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_type = getattr(config.training, "loss_type", "dice_ce")

    if loss_type == "mse":
        criterion = BoundaryWeightedMSELoss(boundary_weight=10.0, threshold=0.95, ignore_index=-100.0)
        print(f"Loss: Masked BoundaryWeightedMSELoss (ignore_index=-100.0)")
        return criterion
        
    elif loss_type == "smooth_l1":
        criterion = MaskedSmoothL1Loss(ignore_index=-100.0)
        print(f"Loss: Masked SmoothL1Loss (ignore_index=-100.0)")
        return criterion
        
    else:
        weights = torch.tensor(config.training.class_weights, dtype=torch.float32).to(device)

        criterion = MaskedDiceCELoss(weight=weights, ignore_index=-100) # Ensure -100 here

        print(f"Loss: MaskedDiceCELoss (ignore_index=-100) | Class weights: {len(config.training.class_weights)} classes")
        return criterion