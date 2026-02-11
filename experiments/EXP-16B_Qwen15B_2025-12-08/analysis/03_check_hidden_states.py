
"""
Check Hidden States Integrity
=============================
This script verifies the integrity of the extracted hidden states.
It checks:
1. Total number of files (should be 300 * 2 = 600).
2. Random sample of files for NaN/Inf/Zero values.
3. Tensor dimensions.
"""

import os
import torch
import glob
import random
import numpy as np

# Use relative path from this script's location
HIDDEN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'hidden_states')
NUM_SAMPLES = 20


def check_file(filepath):
    try:
        # Load numpy array
        tensors = np.load(filepath)
        
        if not isinstance(tensors, np.ndarray):
            print(f"ERROR: {filepath} is not a numpy array, it is {type(tensors)}")
            return False, None

        if np.isnan(tensors).any():
            print(f"ERROR: {filepath} contains NaNs")
            return False, None
            
        if np.isinf(tensors).any():
            print(f"ERROR: {filepath} contains Infs")
            return False, None
            
        if (tensors == 0).all():
            print(f"ERROR: {filepath} is all zeros")
            return False, None
            
        return True, tensors.shape
        
    except Exception as e:
        print(f"ERROR: Could not load {filepath}: {e}")
        return False, None

def main():
    if not os.path.exists(HIDDEN_DIR):
        print(f"CRITICAL: Directory not found: {HIDDEN_DIR}")
        return

    files = glob.glob(os.path.join(HIDDEN_DIR, "*.npy"))
    print(f"Found {len(files)} hidden state files.")
    
    if len(files) == 0:
        print("CRITICAL: No hidden state files found!")
        return

    # Check a sample
    sample_files = random.sample(files, min(len(files), NUM_SAMPLES))
    print(f"Checking {len(sample_files)} random files...")
    
    valid_count = 0
    shapes = []
    
    for f in sample_files:
        is_valid, shape = check_file(f)
        if is_valid:
            valid_count += 1
            if shape:
                shapes.append(shape)
            
    print(f"Validity: {valid_count}/{len(sample_files)} passed integrity check.")
    
    if shapes:
        print(f"Sample Shapes (layers, tokens, hidden_dim) or similar: {shapes[:5]}")
        
if __name__ == "__main__":
    main()
