

import os
import sys
import json

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)



def split_handler(blueprint_json_path: str, output_dir: str) -> dict:

    print(f"path: {blueprint_json_path}")
    sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)

    # Hardcoded Anchor Crops for Guaranteed Evaluation Representation
    val_crops = {
        'crop219', 'crop143', 'crop266', 'crop191', 'crop110', 
        'crop345', 'crop228', 'crop124', 'crop173', 'crop181', 
        'crop200', 'crop319', 'crop417', 'crop79',  'crop9', 'crop39'
    }

    test_crops = {
        'crop254', 'crop122', 'crop161', 'crop132', 'crop267', 
        'crop111', 'crop80',  'crop320', 'crop325', 'crop135', 
        'crop275', 'crop217', 'crop346', 'crop38',  'crop42', 'crop247'
    }

    print(f"Loading master blueprint from: {blueprint_json_path}")
    with open(blueprint_json_path, 'r') as f:
        blueprint = json.load(f)

    train_split = {}
    val_split = {}
    test_split = {}

    # Parse the blueprint solely to extract crop metadata
    for patch in blueprint:
        crop_id = patch.get("crop")
        
        # Define a clean metadata object devoid of specific centroid data
        crop_meta = {
            "crop": crop_id,
            "dataset": patch.get("dataset"),
            "em_scale": patch.get("em_scale"),
            "label_scale": patch.get("label_scale")
        }
        
        # Route the unique crops
        if crop_id in test_crops:
            if crop_id not in test_split:
                test_split[crop_id] = crop_meta
        elif crop_id in val_crops:
            if crop_id not in val_split:
                val_split[crop_id] = crop_meta
        else:
            if crop_id not in train_split:
                train_split[crop_id] = crop_meta

    # Convert back to lists for JSON saving
    train_list = list(train_split.values())
    val_list = list(val_split.values())
    test_list = list(test_split.values())

    print("\n" + "=" * 60)
    print("Unique Crop Split Complete")
    print("-" * 60)
    print(f"Total Train Crops: {len(train_list)}")
    print(f"Total Val Crops:   {len(val_list)}")
    print(f"Total Test Crops:  {len(test_list)}")
    print("=" * 60)

    train_path = os.path.join(output_dir, "train_crops.json")
    val_path = os.path.join(output_dir, "val_crops.json")
    test_path = os.path.join(output_dir, "test_crops.json")

    with open(train_path, 'w') as f:
        json.dump(train_list, f, indent=4)
        
    with open(val_path, 'w') as f:
        json.dump(val_list, f, indent=4)
        
    with open(test_path, 'w') as f:
        json.dump(test_list, f, indent=4)

    return {
        "train_crops": train_path,
        "val_crops": val_path,
        "test_crops": test_path
    }
