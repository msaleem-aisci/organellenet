import os
import sys
import json
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.data.zarr_utils import extract_safe
from code.data.augmentations import apply_augmentations

class PatchDataset(Dataset):
    """
    Dynamic 3D crop dataset that randomly extracts patches at runtime.
    Integrates 29-class atomic mapping, handles -1 unannotated regions, 
    and applies geometric/intensity augmentations.
    """
    def __init__(self, 
    json_path, 
    zarr_map,
    label_map, 
    patch_dim=128,
    augmentation_config=None, 
    target_type="labels", 
    num_classes=30,             
    scale_conditioned=False):
        
        self.patch_dim = patch_dim
        self.zarr_map = zarr_map
        self.augmentation_config = augmentation_config
        self.target_type = target_type
        self.num_classes = num_classes
        self.scale_conditioned = scale_conditioned
        self.zarr_cache = {}

        with open(json_path, 'r') as f:
            self.crops = json.load(f)

        # 29 Atomic Organelles Mapping
        semantic_atomic_to_channel =label_map

        # O(1) Lookup Array Initialization (-1 for unannotated/ignore)
        self.label_lookup = np.full(256, -1, dtype=np.int64)
        for semantic_id, class_id in semantic_atomic_to_channel.items():
            self.label_lookup[semantic_id] = class_id


        print(f"Dynamic Dataset initialized with {len(self.crops)} patches/crops. "
              f"Classes: {self.num_classes} | Augmentations enabled: {self.augmentation_config is not None and self.augmentation_config.enabled}")

    def get_scale_trans(self, path, level="s0"):
        scale = np.array([1.0, 1.0, 1.0])
        trans = np.array([0.0, 0.0, 0.0])
        if path is None: return scale, trans
        try:
            with open(f"{path}/.zattrs", 'r') as f:
                meta = json.load(f)
                multiscales = meta.get("multiscales", [{}])[0]
                for ds in multiscales.get("datasets", []):
                    if ds.get("path") == level:
                        for t in ds.get("coordinateTransformations", []):
                            if t.get("type") == "scale": scale = np.array(t["scale"])[-3:]
                            if t.get("type") == "translation": trans = np.array(t["translation"])[-3:]
                        return scale, trans
        except Exception:
            pass
        return scale, trans

    def _get_zarr_handles(self, dataset, crop_id, em_scale, label_scale):
        cache_key = f"{dataset}_{crop_id}_{em_scale}_{label_scale}"
        if cache_key in self.zarr_cache:
            return self.zarr_cache[cache_key]

        valid_em_base = None
        valid_lbl_base = None
        for zarr_path in self.zarr_map.get(dataset, []):
            base_recon = os.path.join(zarr_path, "recon-1")
            temp_em = os.path.join(base_recon, "em", "fibsem-uint8")
            temp_lbl = os.path.join(base_recon, "labels", "groundtruth", crop_id, "all")
            if os.path.exists(temp_em) and os.path.exists(temp_lbl):
                valid_em_base = temp_em
                valid_lbl_base = temp_lbl
                break

        if not valid_em_base: 
            raise FileNotFoundError(f"Missing Zarr paths for {crop_id}")

        em_zarr = zarr.open(os.path.join(valid_em_base, str(em_scale)), mode='r')
        label_zarr = zarr.open(os.path.join(valid_lbl_base, str(label_scale)), mode='r')

        all_scale_lbl, all_trans_lbl = self.get_scale_trans(valid_lbl_base, str(label_scale))
        em_scale_arr, em_trans_arr = self.get_scale_trans(valid_em_base, str(em_scale))

        self.zarr_cache[cache_key] = (em_zarr, label_zarr, all_scale_lbl, all_trans_lbl, em_scale_arr, em_trans_arr)
        return self.zarr_cache[cache_key]

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop_meta = self.crops[idx]
        dataset = crop_meta["dataset"]
        crop_id = crop_meta["crop"]
        em_lvl = crop_meta["em_scale"]
        lbl_lvl = crop_meta["label_scale"]

        em_zarr, label_zarr, lbl_scale, lbl_trans, em_scale, em_trans = self._get_zarr_handles(
            dataset, crop_id, em_lvl, lbl_lvl
        )

        lbl_shape = np.array(label_zarr.shape)
        half_patch = self.patch_dim // 2
        max_bounds = np.maximum(lbl_shape - half_patch, half_patch)
        patch_shape = [self.patch_dim, self.patch_dim, self.patch_dim]

        # 1. Dynamic Random Extraction Loop
        max_retries = 10
        for attempt in range(max_retries):
            l_center = np.array([np.random.randint(half_patch, limit + 1) for limit in max_bounds])

            phys_center = (l_center * lbl_scale) + lbl_trans
            e_center = np.round((phys_center - em_trans) / em_scale).astype(int)

            l_start = np.floor(l_center - half_patch).astype(int)
            e_start = np.floor(e_center - half_patch).astype(int)

            lbl_np = extract_safe(label_zarr, l_start.tolist(), patch_shape, pad_value=-1, out_dtype=np.int64)
            remapped_lbl = self.label_lookup[lbl_np]

            if np.any(remapped_lbl > 0) or attempt == max_retries - 1:
                break

        em_np = extract_safe(em_zarr, e_start.tolist(), patch_shape, pad_value=0)
        em_float = em_np.astype(np.float32) / 255.0

        # 2. Apply Augmentations
        if self.augmentation_config is not None and self.augmentation_config.enabled:
            em_float, remapped_lbl = apply_augmentations(
                em_float, remapped_lbl, self.augmentation_config
            )

        # 3. Generate Targets
        if self.target_type == "sdt":
            import scipy.ndimage as ndi
            resolution = crop_meta.get("resolution", [8.0, 8.0, 8.0])
            sdt_scale = 40.0 
            
            sdt_tensor = np.zeros((self.num_classes, self.patch_dim, self.patch_dim, self.patch_dim), dtype=np.float32)
            for c in range(self.num_classes):
                mask = (remapped_lbl == c)
                if not np.any(mask):
                    sdt_tensor[c] = -1.0
                else:
                    in_dist = ndi.distance_transform_edt(mask, sampling=resolution)
                    out_dist = ndi.distance_transform_edt(~mask, sampling=resolution)
                    sdt = in_dist - out_dist
                    sdt_tensor[c] = np.tanh(sdt / sdt_scale)
            
            lbl_tensor = torch.from_numpy(sdt_tensor)
        else:
            lbl_tensor = torch.from_numpy(remapped_lbl)

        em_tensor = torch.from_numpy(em_float).unsqueeze(0)
        
        # 4. Scale Conditioning
        if getattr(self, "scale_conditioned", False):
            res = crop_meta.get("resolution", [8.0, 8.0, 8.0])
            z_chan = torch.full_like(em_tensor, res[0])
            y_chan = torch.full_like(em_tensor, res[1])
            x_chan = torch.full_like(em_tensor, res[2])
            em_tensor = torch.cat([em_tensor, z_chan, y_chan, x_chan], dim=0)

        return em_tensor, lbl_tensor

    def get_patch_metadata(self, idx):
        return self.crops[idx]