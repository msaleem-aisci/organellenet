"""
PyTorch Dataset for loading 3D EM/label patches from zarr volumes.

Configurable for different jitter levels, class counts, and label remapping.
"""

import os
import sys
import json
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.data.zarr_utils import extract_safe, select_z_slice, extract_both_patches, get_scale_trans
from code.data.augmentations import apply_augmentations






class MISO2DDataset(Dataset):
    def __init__(self, crops_json_path, zarr_map, patch_dim=256):
        self.patch_dim = patch_dim
        self.zarr_map = zarr_map
        
        unwanted_crops = {
            "crop243", "crop56", "crop57", "crop58", "crop59", "crop54", "crop55", 
            "crop60", "crop61", "crop62", "crop63", "crop64", "crop65", "crop66", 
            "crop67", "crop68", "crop69", "crop70", "crop71", "crop72", "crop73", 
            "crop74", "crop75", "crop76", "crop77", "crop282", "crop25", "crop26", 
            "crop81", "crop82", "crop83", "crop84", "crop97", "crop98", "crop99"
        }
        
        with open(crops_json_path, 'r') as f:
            all_crops = json.load(f)
            # Filter crops before loading them into the dataset list
            self.crops = [crop for crop in all_crops if crop["crop"] not in unwanted_crops]
            
        print(f"Loaded {len(self.crops)} valid crops. Excluded {len(all_crops) - len(self.crops)} empty crops.")
            
        semantic_to_instance_map = {
            # 1. Mitochondria
            50: 1, 3: 1, 4: 1, 5: 1,
            
            # 2. Vesicles
            8: 2, 9: 2,
            
            # 3. Endosomes
            10: 3, 11: 3,
            
            # 4. Lysosomes
            12: 4, 13: 4,
            
            # 5. Lipid Droplets
            44: 5, 14: 5, 15: 5,
            
            # 6. Nucleus
            37: 6, 20: 6, 21: 6, 26: 6, 24: 6, 25: 6, 27: 6, 28: 6, 29: 6,
            
            # 7. Nuclear Pores
            22: 7, 23: 7,
            
            # 8. Microtubules
            30: 8, 36: 8,
            
            # 9. Peroxisomes
            49: 9, 47: 9, 48: 9,
            
            # 10. Golgi Apparatus
            6: 10, 7: 10,
            
            # 11. Endoplasmic Reticulum
            16: 11, 17: 11, 64: 11,
            
            # 12. ER Exit Sites
            18: 12, 19: 12,
            
            # 13. Background
            35: 0, 1: 0 
        }

    
        
        # Fast lookup array for semantic mapping
        self.label_lookup = np.full(256, -1, dtype=np.int64)
    
        for semantic_id, instance_id in semantic_to_instance_map.items():
            self.label_lookup[semantic_id] = instance_id
            
        self.zarr_cache = {}

    def _get_zarr_handles(self, dataset, crop_id, em_scale, label_scale):
        cache_key = f"{dataset}_{crop_id}_{em_scale}_{label_scale}"
        if cache_key in self.zarr_cache:
            return self.zarr_cache[cache_key]
            
        valid_em_base = None
        valid_lbl_base = None
        
        # Resolve paths dynamically from zarr_map
        for zarr_path in self.zarr_map.get(dataset, []):
            base_recon = os.path.join(zarr_path, "recon-1")
            temp_em = os.path.join(base_recon, "em", "fibsem-uint8")
            temp_lbl = os.path.join(base_recon, "labels", "groundtruth", crop_id, "all")
            if os.path.exists(temp_em) and os.path.exists(temp_lbl):
                valid_em_base = temp_em
                valid_lbl_base = temp_lbl
                break
                
        if not valid_em_base: 
            raise FileNotFoundError(f"Missing Zarr paths for {crop_id} in dataset {dataset}")
            
        # # Open arrays targeting the exact matching resolution scales from the JSON
        # em_zarr = zarr.open(os.path.join(valid_em_base, str(em_scale)), mode='r')
        # label_zarr = zarr.open(os.path.join(valid_lbl_base, str(label_scale)), mode='r')
        
        # # Load spatial metadata to compute translation offsets
        # lbl_scale_arr, lbl_trans_arr = get_scale_trans(valid_lbl_base, str(label_scale))
        # _, em_trans_arr = get_scale_trans(valid_em_base, str(em_scale))
        
        # self.zarr_cache[cache_key] = (em_zarr, label_zarr, lbl_scale_arr, lbl_trans_arr, em_trans_arr)
        # return self.zarr_cache[cache_key]
       
    # Open fresh arrays directly (No Cache)
    
        em_zarr = zarr.open(os.path.join(valid_em_base, str(em_scale)), mode='r')
        label_zarr = zarr.open(os.path.join(valid_lbl_base, str(label_scale)), mode='r')
        
        lbl_scale_arr, lbl_trans_arr = get_scale_trans(valid_lbl_base, str(label_scale))
        _, em_trans_arr = get_scale_trans(valid_em_base, str(em_scale))
        
        return em_zarr, label_zarr, lbl_scale_arr, lbl_trans_arr, em_trans_arr

    def __len__(self):
        return len(self.crops)
        
    def __getitem__(self, idx):
        crop_meta = self.crops[idx]
        dataset = crop_meta["dataset"]
        crop_id = crop_meta["crop"]
        em_lvl = crop_meta["em_scale"]
        lbl_lvl = crop_meta["label_scale"]
        
        # 1. Fetch cached arrays and metadata vectors natively
        em_zarr, label_zarr, lbl_scale, lbl_trans, em_trans = self._get_zarr_handles(
            dataset, crop_id, em_lvl, lbl_lvl
        )
        
        # 2. Select Z-slice
        z_idx = select_z_slice(label_zarr)
        
        # 3. Extract 2D patches
        label_patch, em_patch = extract_both_patches(
            label_zarr, em_zarr, z_idx, 
            lbl_scale, lbl_trans, em_trans, 
            patch_dim=self.patch_dim
        )
        
        # 4. Semantic Remapping
        remapped_lbl = np.full_like(label_patch, fill_value=-1)
        valid_mask = (label_patch != -1)
        remapped_lbl[valid_mask] = self.label_lookup[label_patch[valid_mask]]
        
        # 5. Tensor Conversion
        em_tensor = torch.from_numpy(em_patch.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(remapped_lbl)
            
        return em_tensor, lbl_tensor