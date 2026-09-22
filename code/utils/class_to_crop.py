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




class Class_To_Crop:
    def __init__(self, dataset):
        self.dataset = dataset
        
    
    def class_mapper(self):
        return "Hello from class to crop"