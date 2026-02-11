"""
Experiment 16 Utilities
=======================
Shared functions for device management and utilities.
"""

import os
import random
import numpy as np
import torch

# DirectML import
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_compute_device():
    """Returns best device for heavy computation (metrics)."""
    if HAS_DIRECTML and torch_directml.is_available():
        return torch_directml.device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_generation_device():
    """Returns device for text generation."""
    if HAS_DIRECTML and torch_directml.is_available():
        return torch_directml.device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
