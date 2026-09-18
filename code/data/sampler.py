"""
Balanced sampling for class-imbalanced patch datasets.
"""

import sys
import os
import collections
import torch
from torch.utils.data import WeightedRandomSampler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)



def create_rfs_sampler(crops_list, rfs_weights_dict, num_samples=32000)-> WeightedRandomSampler:

    sample_weights = []
    for crop_meta in crops_list:
        crop_id = crop_meta["crop"]
        weight = rfs_weights_dict.get(crop_id, 1.0)
        sample_weights.append(weight)
        
    # Convert to PyTorch sampler
    weights_tensor = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=weights_tensor, 
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler