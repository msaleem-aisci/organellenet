"""
Zarr I/O utilities for OrganelleNet.

Provides safe boundary-clamped patch extraction, zarr directory scanning,
coordinate metadata reading, and full-crop aligned volume extraction.
"""

import os
import sys
import json
import glob
import numpy as np
import zarr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def extract_safe(zarr_arr, start_coords, patch_shape, pad_value=0, out_dtype=None):

    arr_shape = zarr_arr.shape
    z_min, z_max = max(0, start_coords[0]), min(arr_shape[0], start_coords[0] + patch_shape[0])
    y_min, y_max = max(0, start_coords[1]), min(arr_shape[1], start_coords[1] + patch_shape[1])
    x_min, x_max = max(0, start_coords[2]), min(arr_shape[2], start_coords[2] + patch_shape[2])

    target_dtype = out_dtype if out_dtype is not None else zarr_arr.dtype
    patch = np.full(patch_shape, fill_value=pad_value, dtype=target_dtype)

    pz_min = z_min - start_coords[0]
    py_min = y_min - start_coords[1]
    px_min = x_min - start_coords[2]

    if z_max > z_min and y_max > y_min and x_max > x_min:
        patch[
            pz_min:pz_min + (z_max - z_min),
            py_min:py_min + (y_max - y_min),
            px_min:px_min + (x_max - x_min),
        ] = zarr_arr[z_min:z_max, y_min:y_max, x_min:x_max]

    return patch


def build_zarr_map(data_root: str) -> dict:
    zarr_map = {}
    search_pattern = os.path.join(data_root, "*", "*.zarr")

    for zarr_path in glob.glob(search_pattern):
        dataset_name = os.path.basename(zarr_path).replace(".zarr", "")
        if dataset_name not in zarr_map:
            zarr_map[dataset_name] = []
        if zarr_path not in zarr_map[dataset_name]:
            zarr_map[dataset_name].append(zarr_path)

    return zarr_map


def get_scale_trans(path:str, level:str ="s0"):
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
            if "coordinateTransformations" in multiscales:
                for t in multiscales["coordinateTransformations"]:
                    if t.get("type") == "scale": scale = np.array(t["scale"])[-3:]
                    if t.get("type") == "translation": trans = np.array(t["translation"])[-3:]
    except Exception:
        pass
    return scale, trans


def extract_aligned_volumes(dataset_base: str, crop_id: str = "crop234", em_scale: str = "s0"):
    """
    Extract spatially aligned EM and label 3D volumes for a crop.

    Uses zarr metadata (scale/translation) to compute the physical bounding
    box of the label crop, then extracts the corresponding EM sub-volume.

    Parameters
    ----------
    dataset_base : str
        Path to the dataset.zarr root (e.g., `.../jrc_cos7-1a.zarr`).
    crop_id : str
        Crop identifier (e.g., "crop234").
    em_scale : str
        The scale to load EM data from (e.g. "s0", "s1"). Default "s0".

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (em_volume, label_volume) — aligned 3D arrays.
    """
    print(f"--- Extracting 3D Volumes for {crop_id} at EM scale {em_scale} ---")

    lbl_base = os.path.join(dataset_base, "recon-1", "labels", "groundtruth", crop_id, "all")
    em_base = os.path.join(dataset_base, "recon-1", "em", "fibsem-uint8")

    # 1. Get Scales and Translations
    scale_lbl, trans_lbl = get_scale_trans(lbl_base, "s0")
    scale_em, trans_em = get_scale_trans(em_base, em_scale)

    # 2. Open Zarr Arrays (metadata only, no RAM used yet)
    lbl_zarr = zarr.open(os.path.join(lbl_base, "s0"), mode="r")
    em_zarr = zarr.open(os.path.join(em_base, em_scale), mode="r")

    lbl_shape = np.array(lbl_zarr.shape)

    # 3. Calculate Spatial Bounding Box for the EM Sub-volume
    phys_min = trans_lbl
    phys_max = (lbl_shape * scale_lbl) + trans_lbl

    e_min = np.round((phys_min - trans_em) / scale_em).astype(int)
    e_max = np.round((phys_max - trans_em) / scale_em).astype(int)

    print(f"Label Crop Shape: {lbl_shape}")
    print(f"Target EM Bounding Box: Z[{e_min[0]}:{e_max[0]}] Y[{e_min[1]}:{e_max[1]}] X[{e_min[2]}:{e_max[2]}]")

    # 4. Extract Data into RAM
    lbl_volume = np.array(lbl_zarr[:])
    em_volume = np.array(em_zarr[e_min[0]:e_max[0], e_min[1]:e_max[1], e_min[2]:e_max[2]])

    # 5. Fix any 1-voxel rounding disparities
    min_z = min(lbl_volume.shape[0], em_volume.shape[0])
    min_y = min(lbl_volume.shape[1], em_volume.shape[1])
    min_x = min(lbl_volume.shape[2], em_volume.shape[2])

    lbl_volume = lbl_volume[:min_z, :min_y, :min_x]
    em_volume = em_volume[:min_z, :min_y, :min_x]

    print(f"Extraction Complete. Final Aligned Shape: {em_volume.shape}")

    return em_volume, lbl_volume



def select_z_slice(crop_zarr):
    z_max = crop_zarr.shape[0]
    return np.random.randint(0, z_max)

def extract_both_patches(label_zarr, em_zarr, z_idx, lbl_scale, lbl_trans, em_trans, patch_dim=256):
    lbl_shape = label_zarr.shape
    y_max = max(0, lbl_shape[1] - patch_dim)
    x_max = max(0, lbl_shape[2] - patch_dim)
    
    offset = np.round((lbl_trans - em_trans) / lbl_scale).astype(int)
    z_off, y_off, x_off = offset[0], offset[1], offset[2]
    
    # 1. Single Random Extraction (No Retries)
    y_start = np.random.randint(0, y_max + 1)
    x_start = np.random.randint(0, x_max + 1)
    
    label_patch = label_zarr[z_idx, y_start:y_start+patch_dim, x_start:x_start+patch_dim]
    extracted_h, extracted_w = label_patch.shape
    
    label_patch = label_patch.astype(np.int64)
    
    if extracted_h != patch_dim or extracted_w != patch_dim:
        pad_h = patch_dim - extracted_h
        pad_w = patch_dim - extracted_w
        label_patch = np.pad(label_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
    
    em_z = z_idx + z_off
    em_y_start = y_start + y_off
    em_y_end = em_y_start + patch_dim
    em_x_start = x_start + x_off
    em_x_end = em_x_start + patch_dim
    
    em_patch = em_zarr[em_z, em_y_start:em_y_end, em_x_start:em_x_end]
    
    em_h, em_w = em_patch.shape
    if em_h != patch_dim or em_w != patch_dim:
        pad_h = patch_dim - em_h
        pad_w = patch_dim - em_w
        em_patch = np.pad(em_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
    return label_patch, em_patch