"""
Salvaged Experiment 16: Compute Metrics (DirectML-Accelerated)
==============================================================
Computes geometric metrics on SALVAGED (truncated) hidden states.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress
from tqdm import tqdm

# Add scripts dir to path for utils
scripts_dir = "experiments/Experiment 16/scripts"
sys.path.insert(0, scripts_dir)
from exp16_utils import get_compute_device, set_seed

def safe_svd(X, device):
    """Compute SVD with DirectML/CPU fallback."""
    try:
        X_torch = torch.tensor(X, dtype=torch.float32).to(device)
        U, S, Vh = torch.linalg.svd(X_torch, full_matrices=False)
        return S.cpu().numpy()
    except Exception as e:
        return np.linalg.svd(X, compute_uv=False)

def compute_effective_dim(h, device):
    """Participation ratio from SVD."""
    if len(h) < 2:
        return np.nan
    n_tokens, hidden_dim = h.shape
    S = safe_svd(h.T, device)
    S = S + 1e-10
    S_norm = S / np.sum(S)
    return 1.0 / np.sum(S_norm ** 2)

def compute_metrics_for_trajectory(hidden, device):
    """Compute all metrics for a single trajectory."""
    n_layers, n_tokens, hidden_dim = hidden.shape
    
    if n_tokens < 2:
        return None
    
    metrics_all_layers = []
    
    for layer_idx in range(n_layers):
        h = hidden[layer_idx]
        delta = h[1:] - h[:-1]
        
        speed = np.mean(np.linalg.norm(delta, axis=1)) if len(delta) > 0 else 0.0
        
        if len(delta) >= 2:
            dots = np.sum(delta[:-1] * delta[1:], axis=1)
            norms = np.linalg.norm(delta[:-1], axis=1) * np.linalg.norm(delta[1:], axis=1)
            norms = np.clip(norms, 1e-10, None)
            dir_consistency = np.mean(dots / norms)
        else:
            dir_consistency = np.nan
        
        if len(delta) >= 4:
            early_speed = np.mean(np.linalg.norm(delta[:len(delta)//2], axis=1))
            late_speed = np.mean(np.linalg.norm(delta[len(delta)//2:], axis=1))
            stabilization = late_speed / (early_speed + 1e-10)
        else:
            stabilization = np.nan
        
        effective_dim = compute_effective_dim(h, device)
        
        centroid = np.mean(h, axis=0)
        rg = np.sqrt(np.mean(np.sum((h - centroid)**2, axis=1)))
        
        if n_tokens >= 4:
            displacements = np.cumsum(np.linalg.norm(delta, axis=1))
            time_steps = np.arange(1, len(displacements) + 1)
            if len(time_steps) > 1:
                try:
                    slope, _, _, _, _ = linregress(np.log(time_steps + 1), np.log(displacements + 1))
                    msd_exponent = slope
                except:
                    msd_exponent = np.nan
            else:
                msd_exponent = np.nan
        else:
            msd_exponent = np.nan
        
        if n_tokens >= 4:
            late_window = h[-min(4, n_tokens//4):]
            late_mean = np.mean(late_window, axis=0)
            dots = np.sum(h * late_mean, axis=1)
            norms = np.linalg.norm(h, axis=1) * np.linalg.norm(late_mean)
            cos_to_late = np.mean(dots / (norms + 1e-10))
        else:
            cos_to_late = np.nan
        
        metrics_all_layers.append({
            'layer': layer_idx,
            'speed': speed,
            'dir_consistency': dir_consistency,
            'stabilization': stabilization,
            'effective_dim': effective_dim,
            'radius_of_gyration': rg,
            'msd_exponent': msd_exponent,
            'cos_to_late_window': cos_to_late
        })
    
    if n_layers >= 2:
        alignments = []
        for l in range(n_layers - 1):
            h1 = hidden[l].flatten()
            h2 = hidden[l+1].flatten()
            dot = np.dot(h1, h2)
            norm = np.linalg.norm(h1) * np.linalg.norm(h2)
            if norm > 0:
                alignments.append(dot / norm)
        interlayer_align = np.mean(alignments) if alignments else np.nan
    else:
        interlayer_align = np.nan
    
    metrics_all_layers.append({
        'layer': -1,
        'speed': np.nan,
        'dir_consistency': np.nan,
        'stabilization': np.nan,
        'effective_dim': np.nan,
        'radius_of_gyration': np.nan,
        'msd_exponent': np.nan,
        'cos_to_late_window': np.nan,
        'interlayer_alignment': interlayer_align
    })
    
    return metrics_all_layers

def main():
    print("="*60)
    print("04_recompute_metrics_exp16.py: SALVAGE Metric Computation")
    print("="*60)
    
    set_seed(42)
    device = get_compute_device()
    print(f"Compute device: {device}")
    
    metadata = pd.read_csv("experiments/Experiment 16/data/metadata_reparsed.csv")
    print(f"\nLoaded {len(metadata)} records from metadata_reparsed.csv")
    
    all_metrics = []
    fallback_count = 0
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Computing clean metrics"):
        problem_id = row['problem_id']
        condition = row['condition']
        
        hidden_path = f"experiments/Experiment 16/data/hidden_states_clean/{problem_id}_{condition}.npy"
        
        if not os.path.exists(hidden_path):
            continue
        
        try:
            hidden = np.load(hidden_path).astype(np.float32)
            trajectory_metrics = compute_metrics_for_trajectory(hidden, device)
            
            if trajectory_metrics:
                for m in trajectory_metrics:
                    all_metrics.append({
                        'problem_id': problem_id,
                        'condition': condition,
                        'correct': row['correct_new'],
                        'response_length_tokens_clean': hidden.shape[1],
                        **m
                    })
        
        except Exception as e:
            fallback_count += 1
    
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv("experiments/Experiment 16/data/exp16_metrics_salvaged.csv", index=False)
    print(f"\n[OK] Salvaged Metrics saved: {len(df_metrics)} rows")

if __name__ == "__main__":
    main()
