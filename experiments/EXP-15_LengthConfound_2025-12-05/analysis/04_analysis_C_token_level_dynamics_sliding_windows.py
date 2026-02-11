"""
Script 04: Analysis C - Token Dynamics
======================================
Computes sliding-window geometric metrics to detect phase transitions.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import EXP15_DATA_DIR, EXP15_FIGURES_DIR, get_device

def compute_sliding_window_metrics(h, window=5):
    """Computes basic geometry over sliding window."""
    if len(h) < window: return None
    
    metrics = []
    # h shape: [T, D]
    
    # Pre-compute deltas
    delta = h[1:] - h[:-1]
    
    for t in range(len(h) - window + 1):
        chunk = h[t:t+window]
        chunk_delta = delta[t:t+window-1] if window > 1 else []
        
        # Radius of Gyration
        center = np.mean(chunk, axis=0)
        rg = np.sqrt(np.mean(np.sum((chunk - center)**2, axis=1)))
        
        # Speed (mean step size)
        speed = np.mean(np.linalg.norm(chunk_delta, axis=1)) if len(chunk_delta) > 0 else 0
        
        metrics.append({
            't_start': t,
            'window_rg': rg,
            'window_speed': speed
        })
        
    return pd.DataFrame(metrics)

def main():
    print("="*60)
    print("04_analysis_C.py: Token Dynamics")
    print("="*60)
    
    # We need raw hidden states! 
    # The unified metrics CSV only has summaries.
    # We must scan `Experiment 14/data/hidden_states/*.npy`
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    HIDDEN_DIR = os.path.join(ROOT_DIR, "experiments", "Experiment 14", "data", "hidden_states")
    
    if not os.path.exists(HIDDEN_DIR):
        print(f"Error: Hidden states dir {HIDDEN_DIR} not found.")
        return
    
    # Process a subset of interesting cases?
    # Or aggregate over all?
    # Analyzing ALL 600 sliding windows and saving them is heavy.
    # The prompt asks for "aggregate distributions" and "example panels".
    
    # Let's pick 20 random samples from each Group (G1, G2, G3, G4)
    # Use unified metrics to identify IDs
    
    input_file = os.path.join(EXP15_DATA_DIR, "unified_metrics.csv")
    df = pd.read_csv(input_file)
    
    # Select sample IDs
    sample_ids = []
    for grp in ['G1', 'G2', 'G3', 'G4']:
        grp_ids = df[df['group'] == grp]['problem_id'].unique()
        if len(grp_ids) > 10:
            selected = np.random.choice(grp_ids, 10, replace=False)
        else:
            selected = grp_ids
        for pid in selected:
            # Get condition
            cond = df[(df['group'] == grp) & (df['problem_id'] == pid)]['condition'].iloc[0]
            sample_ids.append((pid, cond, grp))
            
    print(f"Selected {len(sample_ids)} trajectories for deep dive.")
    
    all_window_stats = []
    
    for pid, cond, grp in sample_ids:
        filename = f"{pid}_{cond}.npy"
        path = os.path.join(HIDDEN_DIR, filename)
        if not os.path.exists(path): continue
        
        try:
            stack = np.load(path) # [25, T, D]
            # Use Layer 16
            h = stack[16]
            
            w_df = compute_sliding_window_metrics(h, window=5)
            if w_df is not None:
                w_df['group'] = grp
                w_df['problem_id'] = pid
                all_window_stats.append(w_df)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if not all_window_stats:
        print("No window stats computed.")
        return
        
    full_w_df = pd.concat(all_window_stats)
    
    # Normalize time? (Progress 0.0 to 1.0)
    # Or just raw token index?
    # Raw is better for "inflection points"
    
    # Plot Aggregate Regimes
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=full_w_df, x='t_start', y='window_rg', hue='group')
    plt.title("Sliding Window Radius of Gyration (Layer 16)")
    plt.xlabel("Token Index")
    plt.ylabel("Local Rg")
    plt.savefig(os.path.join(EXP15_FIGURES_DIR, "analysis_C_dynamics_aggregate.png"))
    plt.close()
    
    # Detect Inflections (Change points in speed/Rg)
    # Simple derivative threshold
    # For now, just save the window data for deeper analysis if valid
    full_w_df.to_csv(os.path.join(EXP15_DATA_DIR, "analysis_C_window_metrics.csv"), index=False)
    print("Saved window metrics.")

if __name__ == "__main__":
    main()
