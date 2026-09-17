"""
Main training entry point for BlueMind OrganelleNet.
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# train.py is located inside the 'code/' directory.
# We move one level up ("..") to reach the 'organellenet' root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.utils.config import load_config

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train BlueMind OrganelleNet")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file (e.g., configs/base.yml)"
    )
    return parser.parse_args()

def main():
    # 1. Parse terminal command
    args = parse_args()
    
    # 2. Load the configuration
    print(f"Loading configuration from: {args.config}...\n")
    cfg = load_config(args.config)
    
    # 3. Print the verified schemas
    # Because we used @dataclass in config.py, Python automatically formats 
    # the print statements cleanly without needing custom formatting logic.
    print("=" * 60)
    print(f"=== EXPERIMENT: {cfg.experiment_name.upper()} ===")
    print("=" * 60)
    
    print("\n[PATH CONFIGURATION]")
    print(cfg.path)
    
    print("\n[DATA CONFIGURATION]")
    print(cfg.data)
    
    print("\n[MODEL CONFIGURATION]")
    print(cfg.model)
    
    print("\n[TRAINING CONFIGURATION]")
    print(cfg.training)
    
    print("\n[SEMANTIC MAP CLASSES DETECTED]")
    print(f"Total target classes defined: {len(cfg.semantic_map.keys())}")
    print("=" * 60)

if __name__ == "__main__":
    main()