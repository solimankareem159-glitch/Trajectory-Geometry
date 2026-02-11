"""
Script 01: Environment and Device Setup
=======================================
Configures DirectML device with graceful fallbacks and sets deterministic behavior.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import multiprocessing
from datetime import datetime

# Try DirectML import
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False
    print("WARNING: torch_directml not available")

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
    """
    Returns the best available device for computation.
    Priority: DirectML > CUDA > CPU
    """
    if HAS_DIRECTML and torch_directml.is_available():
        return torch_directml.device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_generation_device():
    """
    Returns device for text generation.
    May differ from compute device if DirectML is unstable for generation.
    """
    # For Pythia-70m, DirectML should be stable
    # But allow CPU fallback if needed
    if HAS_DIRECTML and torch_directml.is_available():
        return torch_directml.device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    print("="*60)
    print("01_env_and_device.py: Environment Setup")
    print("="*60)
    
    # Set seed
    set_seed(42)
    print("Random seed set to: 42")
    
    # Get device info
    cpu_count = multiprocessing.cpu_count()
    print(f"\nCPU cores: {cpu_count}")
    
    # Check DirectML
    print(f"\nDirectML available: {HAS_DIRECTML}")
    if HAS_DIRECTML:
        try:
            dml_available = torch_directml.is_available()
            print(f"DirectML is_available(): {dml_available}")
            if dml_available:
                device_name = torch_directml.device_name(0)
                print(f"DirectML device: {device_name}")
        except Exception as e:
            print(f"DirectML check failed: {e}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Get compute and generation devices
    compute_device = get_compute_device()
    generation_device = get_generation_device()
    
    print(f"\nCompute device: {compute_device}")
    print(f"Generation device: {generation_device}")
    
    # Package versions
    print(f"\nPackage versions:")
    print(f"  torch: {torch.__version__}")
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError:
        print(f"  transformers: NOT INSTALLED")
    
    # Save environment info
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "seed": 42,
        "cpu_cores": cpu_count,
        "directml_available": HAS_DIRECTML,
        "directml_is_available": HAS_DIRECTML and torch_directml.is_available() if HAS_DIRECTML else False,
        "directml_device": torch_directml.device_name(0) if HAS_DIRECTML and torch_directml.is_available() else None,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "compute_device": str(compute_device),
        "generation_device": str(generation_device),
        "torch_version": torch.__version__
    }
    
    output_path = "experiments/Experiment 16/data/env_info.json"
    with open(output_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print(f"\nEnvironment info saved to: {output_path}")
    print("\n[OK] Environment setup complete")

if __name__ == "__main__":
    main()
