"""
Script 00: Environment Check
============================
Detects DirectML availability, GPU, CPU cores, and sets seeds.
"""

import os
import sys
import multiprocessing
import numpy as np
import torch
import json
from datetime import datetime

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import get_device, ensure_dirs, EXP15_REPORTS_DIR

def main():
    ensure_dirs()
    report_path = os.path.join(EXP15_REPORTS_DIR, "run_metadata.json")
    
    print("="*60)
    print("00_env_check.py: Environment Discovery")
    print("="*60)
    
    # CPU
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU Cores: {cpu_count}")
    
    # GPU / DirectML
    try:
        import torch_directml
        dml_available = torch_directml.is_available()
        dml_device_name = torch_directml.device_name(0) if dml_available else "None"
    except ImportError:
        dml_available = False
        dml_device_name = "Not Installed"
        
    print(f"Torch Version: {torch.__version__}")
    print(f"DirectML Available: {dml_available}")
    print(f"DirectML Device: {dml_device_name}")
    
    device = get_device()
    print(f"Active Torch Device: {device}")
    
    # Seeds
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random Seed set to: {seed}")
    
    # Save Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "cpu_cores": cpu_count,
        "torch_version": torch.__version__,
        "directml_available": dml_available,
        "directml_device": dml_device_name,
        "seed": seed
    }
    
    with open(report_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nEnvironment metadata saved to {report_path}")

if __name__ == "__main__":
    main()
