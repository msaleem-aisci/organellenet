"""
Loss function factory for BlueMind OrganelleNet.
Contains the custom BCE + Tversky loss with EMA Entropy Masking and standard DiceCE.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 1. Corrected BCE Function (AMP Safe)
# ---------------------------------------------------------------------------
def bce_loss_fn(logits, targets_one_hot, valid_mask, smooth=1e-6):
    mask = valid_mask.unsqueeze(1)
    
    # Use with_logits for safe autocast execution
    bce_raw = F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction='none')
    bce_masked = bce_raw * mask
    
    num_classes = logits.shape[1]
    return bce_masked.sum() / (mask.sum() * num_classes + smooth)


# ---------------------------------------------------------------------------
# 2. Corrected Tversky Function
# ---------------------------------------------------------------------------
def tversky_loss_fn(probs, targets_one_hot, valid_mask, alpha, beta, smooth=1e-6):
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)
    targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
    
    # Flatten mask and broadcast across the class dimension
    mask_flat = valid_mask.view(valid_mask.size(0), 1, -1)
    
    # Apply the weight multipliers directly to the overlap terms
    TruePos = (mask_flat * probs_flat * targets_flat).sum(dim=2)
    FalsePos = (mask_flat * (1 - targets_flat) * probs_flat).sum(dim=2)
    FalseNeg = (mask_flat * targets_flat * (1 - probs_flat)).sum(dim=2)
    
    tversky_index = (TruePos + smooth) / (TruePos + alpha * FalsePos + beta * FalseNeg + smooth)
    
    return 1.0 - tversky_index.mean()


# ---------------------------------------------------------------------------
# 3. Dice Function
# ---------------------------------------------------------------------------
def dice_loss_fn(probs, targets_one_hot, valid_mask, smooth=1e-6):
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)
    targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
    
    mask_flat = valid_mask.view(valid_mask.size(0), 1, -1)
    
    intersection = (mask_flat * probs_flat * targets_flat).sum(dim=2)
    denominator = (mask_flat * probs_flat).sum(dim=2) + (mask_flat * targets_flat).sum(dim=2)
    
    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
    
    return 1.0 - dice_score.mean()


class EMAEntropyMasking(nn.Module):
    def __init__(self, warmup_epochs=5, ema_momentum=0.95):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.ema_momentum = ema_momentum
        self.register_buffer("tau_low", torch.tensor(-1.0))
        self.register_buffer("tau_high", torch.tensor(-1.0))

    def forward(self, probs, valid_mask, current_epoch):
        if current_epoch < self.warmup_epochs:
            return valid_mask.float()

        # CRITICAL FIX: Detach probabilities to prevent memory explosion at warmup end
        p_detached = probs.detach()
        p_safe = torch.clamp(p_detached, 1e-6, 1.0 - 1e-6)
        
        pixel_entropy = -(p_safe * torch.log(p_safe) + (1.0 - p_safe) * torch.log(1.0 - p_safe))
        mean_entropy = pixel_entropy.mean(dim=1)

        valid_entropy_vals = mean_entropy[valid_mask]

        if valid_entropy_vals.numel() > 0 and self.training:
            batch_tau_low = torch.quantile(valid_entropy_vals, 0.30)
            batch_tau_high = torch.quantile(valid_entropy_vals, 0.90)

            if self.tau_low.item() < 0:
                self.tau_low.copy_(batch_tau_low)
                self.tau_high.copy_(batch_tau_high)
            else:
                self.tau_low.copy_(self.ema_momentum * self.tau_low + (1.0 - self.ema_momentum) * batch_tau_low)
                self.tau_high.copy_(self.ema_momentum * self.tau_high + (1.0 - self.ema_momentum) * batch_tau_high)

        weights = torch.ones_like(mean_entropy)

        if self.tau_low.item() >= 0:
            weights[mean_entropy >= self.tau_high] = 0.0
            informative_mask = (mean_entropy >= self.tau_low) & (mean_entropy < self.tau_high)
            weights[informative_mask] = 1.5

        return valid_mask.float() * weights


# ---------------------------------------------------------------------------
# 4. Main BCE + Tversky Wrapper
# ---------------------------------------------------------------------------
class BCE_Tversky(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, ignore_index=-1, warmup_epochs=5, entropy=False, ema_momentum=0.95):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

        if entropy:
            self.entropy_masker = EMAEntropyMasking(warmup_epochs=warmup_epochs, ema_momentum=ema_momentum)
        else:
            self.entropy_masker = None

    def forward(self, logits, targets, current_epoch): 
        num_classes = logits.shape[1]
        
        valid_mask = (targets != self.ignore_index)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs = torch.sigmoid(logits)
        
        # Conditionally apply entropy masking
        if self.entropy_masker is not None:
            weighted_mask = self.entropy_masker(probs, valid_mask, current_epoch)
        else:
            weighted_mask = valid_mask.float()
     
        # Pass 'logits' to BCE, keep 'probs' for Tversky
        bce = bce_loss_fn(logits, targets_one_hot, weighted_mask, self.smooth)
        tversky = tversky_loss_fn(probs, targets_one_hot, weighted_mask, self.alpha, self.beta, self.smooth)
        
        return bce + tversky


# ---------------------------------------------------------------------------
# 5. DiceCE Wrapper
# ---------------------------------------------------------------------------
class DiceCE(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets, current_epoch=None): 
        # current_epoch is accepted to maintain the same API signature as BCE_Tversky, 
        # even though DiceCE does not use a curriculum.
        num_classes = logits.shape[1]
        
        valid_mask = (targets != self.ignore_index)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)
        
        weighted_mask = valid_mask.float()
     
        bce = bce_loss_fn(logits, targets_one_hot, weighted_mask, self.smooth)
        dice = dice_loss_fn(probs, targets_one_hot, weighted_mask, self.smooth)
        
        return bce + dice


# ---------------------------------------------------------------------------
# 6. Factory Function
# ---------------------------------------------------------------------------
def build_loss(config, device=None):
    """
    Builds the loss function using parameters defined in the YAML config.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.training.loss_function == "DiceCE":
        criterion = DiceCE(
            ignore_index=-1
        )
        print("="*70)
        print(f"Loss Initialized: DiceCE | Device: {device}")
        print("="*70)
    else:
        criterion = BCE_Tversky(
            alpha=0.3, 
            beta=0.7, 
            ignore_index=-1, 
            warmup_epochs=config.training.warmup_epochs,
            entropy=config.training.entropy_masking
        )
        print("="*70)
        print(f"Loss Initialized: BCE_Tversky | Warmup Epochs: {config.training.warmup_epochs} | Entropy Masking: {config.training.entropy_masking} | Device: {device}")
        print("="*70)
    
    return criterion.to(device)