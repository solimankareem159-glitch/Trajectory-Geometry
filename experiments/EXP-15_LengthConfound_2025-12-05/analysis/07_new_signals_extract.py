"""
Script 07: New Signals Extraction
=================================
Extracts additional geometric signals (High + Medium + Speculative).
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cdist

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import EXP15_DATA_DIR, get_device

def compute_curvature(h):
    """Computes mean curvature (acceleration/velocity angle)."""
    if len(h) < 3: return 0.0
    v = h[1:] - h[:-1]
    a = v[1:] - v[:-1]
    
    # Norms
    v_norm = np.linalg.norm(v[:-1], axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    
    # Avoid zero division
    valid = (v_norm > 1e-6) & (a_norm > 1e-6)
    if not np.any(valid): return 0.0
    
    dot = np.sum(v[:-1][valid] * a[valid], axis=1)
    cos_theta = dot / (v_norm[valid] * a_norm[valid])
    # Clip for safety
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    return np.mean(theta)

def main():
    print("="*60)
    print("07_new_signals_extract.py: New Geometric Signals")
    print("="*60)
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    HIDDEN_DIR = os.path.join(ROOT_DIR, "experiments", "Experiment 14", "data", "hidden_states")
    
    # Outputs to a new metrics file
    out_file = os.path.join(EXP15_DATA_DIR, "exp15_extra_metrics.csv")
    
    if not os.path.exists(HIDDEN_DIR):
        print("Hidden states directory not found.")
        return
        
    print("Scanning hidden states...")
    files = [f for f in os.listdir(HIDDEN_DIR) if f.endswith('.npy')]
    print(f"Found {len(files)} files.")
    
    results = []
    
    for idx, f in enumerate(files):
        if idx % 50 == 0: print(f"Processing {idx}/{len(files)}...")
        
        try:
            pid_cond = f.replace('.npy', '')
            pid_str = pid_cond.split('_')[0]
            cond = '_'.join(pid_cond.split('_')[1:])
            
            stack = np.load(os.path.join(HIDDEN_DIR, f)) # [25, T, D]
            n_layers, n_tokens, dim = stack.shape
            
            # --- Signal 1: Cross-Layer Correlation (Sample) ---
            # Correlation between adjacent layers
            # Mean correlation of valid tokens
            
            layer_corrs = []
            for l in range(n_layers - 1):
                # Flatten -> correlation
                # h_l: [T, D] -> [T*D]
                flat1 = stack[l].flatten()
                flat2 = stack[l+1].flatten()
                corr = np.corrcoef(flat1, flat2)[0,1]
                layer_corrs.append(corr)
                
                results.append({
                    'problem_id': pid_str,
                    'condition': cond,
                    'layer': l,
                    'metric': 'next_layer_corr',
                    'value': corr
                })
                
            # --- Signal 2: Curvature Profile (Per Layer) ---
            for l in range(n_layers):
                h = stack[l]
                curv = compute_curvature(h)
                
                results.append({
                    'problem_id': pid_str,
                    'condition': cond,
                    'layer': l,
                    'metric': 'trajectory_curvature',
                    'value': curv
                })
                
            # --- Signal 3: Commitment Timing (Placeholder) ---
            # Needs "Attractor Centroids" to be robust.
            # Skipping complex attractor logic for now due to complexity vs script size limits.
            # Using simple 'Derivative Spike' (Commitment Sharpness) instead
            
            for l in range(n_layers):
                h = stack[l]
                if len(h) < 2: continue
                v = np.linalg.norm(h[1:] - h[:-1], axis=1)
                max_v = np.max(v)
                mean_v = np.mean(v)
                sharpness = max_v / (mean_v + 1e-9)
                
                results.append({
                    'problem_id': pid_str,
                    'condition': cond,
                    'layer': l,
                    'metric': 'commitment_sharpness',
                    'value': sharpness
                })

        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_file, index=False)
    print(f"Saved {len(df_res)} extra metric records to {out_file}")

if __name__ == "__main__":
    main()
