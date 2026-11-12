"""
Train / Val / Test split generation from patch blueprint JSONs.

Supports:
- Target class filtering
- Optional crop exclusion (e.g., holding out crop234 for testing)
- Configurable split ratios
"""

import os
import sys
import json
import random

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def prepare_splits(
    blueprint_json_path: str,
    output_dir: str,
    target_classes: list,
    split_ratios: list = None,
    excluded_crop: str = None,
    class_mapping: dict = None,
    strip_suffixes: bool = True,
    seed: int = 42,
) -> dict:
    """
    Filter a blueprint JSON by target classes, optionally exclude a crop,
    then split into train/val/test JSONs.

    Parameters
    ----------
    blueprint_json_path : str
        Path to the full patch blueprint JSON.
    output_dir : str
        Directory where train.json, val.json, test.json will be written.
    target_classes : list of str
        Organelle class names to retain (e.g., ['mito', 'ves', ...]).
    split_ratios : list of float, optional
        [train_frac, val_frac, test_frac]. Default [0.80, 0.10, 0.10].
    excluded_crop : str, optional
        If set, patches from this crop are excluded entirely (held out).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    dict
        Keys: 'train_path', 'val_path', 'test_path', 'stats' (dict with counts).
    """
    if split_ratios is None:
        split_ratios = [0.80, 0.10, 0.10]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load blueprint
    with open(blueprint_json_path, "r") as f:
        blueprint = json.load(f)

    target_set = set(target_classes)

    # 2. Filter by target classes and optionally exclude a crop
    filtered_patches = []
    test_patches = []
    excluded_datasets = set()

    for patch in blueprint:
        if excluded_crop and patch.get("crop") == excluded_crop:
            excluded_datasets.add(patch.get("dataset"))
            test_patches.append(patch)
        else:
            cls = patch.get("class", "")
            
            # 1. Explicit mapping override
            if class_mapping and cls in class_mapping:
                base_cls = class_mapping[cls]
            # 2. Dynamic suffix stripping (if enabled)
            elif strip_suffixes and "_" in cls and cls.rsplit("_", 1)[-1] in ["lum", "mem", "in", "out"]:
                base_cls = cls.rsplit("_", 1)[0]
            else:
                base_cls = cls

            if base_cls in target_set:
                filtered_patches.append(patch)

    print(f"Total original patches: {len(blueprint)}")
    print(f"Total filtered (train/val) patches: {len(filtered_patches)}")
    if excluded_crop and excluded_datasets:
        print(f"Held out {excluded_crop} for test set from dataset(s): {', '.join(excluded_datasets)}")
        print(f"Total holdout test patches: {len(test_patches)}")

    # 3. Save targets only patches
    targets_only_path = os.path.join(output_dir, "targets_only.json")
    with open(targets_only_path, "w") as f:
        json.dump(filtered_patches + test_patches, f, indent=4)

    # 4. Shuffle and split
    random.seed(seed)
    random.shuffle(filtered_patches)

    total_train_val = len(filtered_patches)
    # If no test_patches, we do standard 3-way split, else 2-way split of the remaining
    if len(test_patches) > 0:
        # Normalize train/val ratios to sum to 1
        train_frac = split_ratios[0] / (split_ratios[0] + split_ratios[1])
        train_end = int(total_train_val * train_frac)
        train_patches = filtered_patches[:train_end]
        val_patches = filtered_patches[train_end:]
    else:
        train_end = int(total_train_val * split_ratios[0])
        val_end = int(total_train_val * (split_ratios[0] + split_ratios[1]))
        train_patches = filtered_patches[:train_end]
        val_patches = filtered_patches[train_end:val_end]
        test_patches = filtered_patches[val_end:]

    # 5. Write splits
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")

    with open(train_path, "w") as f:
        json.dump(train_patches, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val_patches, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test_patches, f, indent=4)

    total = len(train_patches) + len(val_patches) + len(test_patches)

    stats = {
        "total": total,
        "train": len(train_patches),
        "val": len(val_patches),
        "test": len(test_patches),
    }

    print(f"Total Patches: {total}")
    print(f"-> Train: {stats['train']}")
    print(f"-> Val  : {stats['val']}")
    print(f"-> Test : {stats['test']}")

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "stats": stats,
    }



def prepare_splits1(
    blueprint_json_path: str,
    output_dir: str,
    target_classes: list,
    split_ratios: list = None,  # Kept for signature compatibility, but bypassed by hardcoded sets
    excluded_crop: str = None,  # Can still be used to force a specific crop out
    class_mapping: dict = None,
    strip_suffixes: bool = True,
    seed: int = 42,             # Kept for signature compatibility
) -> dict:
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Hardcoded Curated Anchor Crops
    train_crops_set = {
        "crop234", "crop236", "crop237", "crop239", "crop248", "crop252", "crop256", 
        "crop292", "crop235", "crop240", "crop241", "crop242", "crop245", "crop249", 
        "crop255", "crop258", "crop259", "crop291", "crop116", "crop117", "crop118", 
        "crop119", "crop130", "crop120", "crop121", "crop123", "crop134", "crop174", 
        "crop176", "crop185", "crop78", "crop1", "crop13", "crop14", "crop15", "crop16", 
        "crop18", "crop19", "crop23", "crop28", "crop3", "crop4", "crop6", "crop7", 
        "crop8", "crop100", "crop101", "crop102", "crop27", "crop33", "crop34", "crop50", 
        "crop51", "crop107", "crop112", "crop126", "crop180", "crop182", "crop35", 
        "crop36", "crop37", "crop43", "crop47", "crop109", "crop31", "crop32", "crop40", 
        "crop48", "crop49", "crop129", "crop140", "crop141", "crop146", "crop147", 
        "crop148", "crop149", "crop156", "crop158", "crop159", "crop160", "crop162", 
        "crop163", "crop164", "crop165", "crop166", "crop268", "crop269", "crop270", 
        "crop272", "crop273", "crop274", "crop276", "crop277", "crop278", "crop279", 
        "crop280", "crop298", "crop313", "crop321", "crop322", "crop323", "crop326", 
        "crop377", "crop333", "crop340", "crop341", "crop342", "crop368", "crop369", 
        "crop370", "crop376", "crop125", "crop131", "crop133", "crop136", "crop137", 
        "crop138", "crop139", "crop142", "crop115", "crop20", "crop22", "crop186", 
        "crop187", "crop188", "crop189", "crop201", "crop202", "crop203", "crop206", 
        "crop208", "crop209", "crop210", "crop211", "crop212", "crop213", "crop216", 
        "crop218", "crop190", "crop192", "crop193", "crop195", "crop196", "crop197", 
        "crop198", "crop199", "crop214", "crop220", "crop222", "crop224", "crop225", 
        "crop226", "crop227", "crop144", "crop145", "crop150", "crop151", "crop157", 
        "crop171", "crop172", "crop175", "crop183", "crop416", "crop178", "crop113", 
        "crop155"
    }

    val_crops_set = {
        'crop219', 'crop143', 'crop266', 'crop191', 'crop110', 
        'crop345', 'crop228', 'crop124', 'crop173', 'crop181', 
        'crop200', 'crop319', 'crop417', 'crop79',  'crop9'
    }

    test_crops_set = {
        'crop254', 'crop122', 'crop161', 'crop132', 'crop267', 
        'crop111', 'crop80',  'crop320', 'crop325', 'crop135', 
        'crop275', 'crop217', 'crop346', 'crop38',  'crop42'
    }
    
    # 2. Load blueprint
    with open(blueprint_json_path, "r") as f:
        blueprint = json.load(f)
        
    target_set = set(target_classes)
    train_patches, val_patches, test_patches = [], [], []
    
    # 3. Filter, Map, and Route
    for patch in blueprint:
        raw_cls = patch.get("class", "")
        
        # Apply semantic mapping
        mapped_cls = raw_cls
        if class_mapping and raw_cls in class_mapping:
            mapped_cls = class_mapping[raw_cls]
        elif strip_suffixes:
            mapped_cls = raw_cls.split('_')[0]
            
        # Check against target classes
        if mapped_cls in target_set:
            patch["class"] = mapped_cls
            crop_id = patch.get("crop")
            
            # Force holdout if excluded_crop matches
            if excluded_crop and crop_id == excluded_crop:
                test_patches.append(patch)
                continue
                
            # Route to hardcoded splits
            if crop_id in test_crops_set:
                test_patches.append(patch)
            elif crop_id in val_crops_set:
                val_patches.append(patch)
            elif crop_id in train_crops_set:
                train_patches.append(patch)

    # 4. Save Output JSONs
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")
    
    with open(train_path, "w") as f:
        json.dump(train_patches, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val_patches, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test_patches, f, indent=4)
        
    stats = {
        "total_patches": len(train_patches) + len(val_patches) + len(test_patches),
        "train_patches": len(train_patches),
        "val_patches": len(val_patches),
        "test_patches": len(test_patches)
    }
    
    print("=" * 60)
    print("Strict Target-Based Crop Split Complete")
    print("-" * 60)
    print(f"Total Patches: {stats['total_patches']}")
    print(f"-> Train: {stats['train_patches']}")
    print(f"-> Val  : {stats['val_patches']}")
    print(f"-> Test : {stats['test_patches']}")
    print("=" * 60)
    
    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "stats": stats
    }