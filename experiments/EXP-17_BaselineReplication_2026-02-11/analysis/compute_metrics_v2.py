"""
Step 2: Compute Metrics (EXP-17 v2)
===================================
Loads extracted hidden states (.npy) from v2 run and computes geometric metrics.
Uses SVD optimization for fast anisotropy calculation.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
HIDDEN_DIR = os.path.join(DATA_DIR, "hidden_states_v2")
METADATA_FILE = os.path.join(DATA_DIR, "metadata_v2.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "exp17a_metrics_v2.csv")

# Qwen 2.5 3B has 36 layers (0-35) + embedding?
# Let's count layers dynamically from npy files.
# But for cross-layer metrics, we need a subset.
# Qwen 3B has 36 layers.
CROSS_LAYER_SUBSET = [0, 6, 12, 18, 24, 30, 35]

# ============================================================
# METRIC FUNCTIONS (Same as before)
# ============================================================

def compute_effective_dim(h):
    # PCA-based effective dimension
    # h: [n_tokens, hidden_dim]
    # Centered PCA
    h_centered = h - np.mean(h, axis=0)
    u, s, vh = np.linalg.svd(h_centered, full_matrices=False)
    # Normalized eigenvalues
    if np.sum(s) == 0: return 0
    p = s / np.sum(s)
    # Shannon entropy of eigenvalues
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def compute_radius_of_gyration(h):
    # h: [n_tokens, hidden_dim]
    centroid = np.mean(h, axis=0)
    diff = h - centroid
    sq_dists = np.sum(diff**2, axis=1)
    return np.sqrt(np.mean(sq_dists))

def compute_gyration_anisotropy(h):
    # Using SVD of centered data
    h_centered = h - np.mean(h, axis=0)
    u, s, vh = np.linalg.svd(h_centered, full_matrices=False)
    # variances are proportional to s^2
    variances = s**2
    if np.sum(variances) == 0: return 0
    # anisotropy ~ (lambda1 - lambda2) / (lambda1 + lambda2) ?
    # Or variance along principal axis vs total variance?
    # Let's use g = 1 - (lambda2/lambda1) ?
    # Standard: FA triggers SVD anyway.
    # Let's use simple ratio of first 2 eigenvalues?
    if len(s) < 2: return 0
    return s[0] / (np.sum(s) + 1e-9) # Fraction of variance explained by PC1

def compute_speed(h):
    # Speed = mean euclidean distance between consecutive tokens
    diffs = np.diff(h, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.mean(dists)

def compute_cosine_similarity_trajectory(h):
    # Cosine similarity between consecutive steps
    # effectively "directional consistency"
    if len(h) < 2: return 0
    diffs = np.diff(h, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    normalized = diffs / (norms + 1e-9)
    # dot product between t and t+1
    cosines = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return np.mean(cosines)

def compute_msd_exponent(h):
    # Mean Squared Displacement vs Time lag
    # slope of log(MSD) vs log(lag)
    n = len(h)
    if n < 10: return 0 # Too short
    
    lags = np.unique(np.logspace(0, np.log10(n/2), 10).astype(int))
    lags = lags[lags > 0]
    msds = []
    
    for lag in lags:
        diffs = h[lag:] - h[:-lag]
        sq_dist = np.sum(diffs**2, axis=1)
        msds.append(np.mean(sq_dist))
        
    if len(msds) < 3: return 0
    
    log_lags = np.log(lags)
    log_msds = np.log(msds)
    slope, _, _, _, _ = stats.linregress(log_lags, log_msds)
    return slope

def compute_all_metrics_for_layer(h):
    return {
        'effective_dim': compute_effective_dim(h),
        'radius_of_gyration': compute_radius_of_gyration(h),
        'gyration_anisotropy': compute_gyration_anisotropy(h),
        'speed': compute_speed(h),
        'dir_consistency': compute_cosine_similarity_trajectory(h),
        'msd_exponent': compute_msd_exponent(h)
    }

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("=" * 70)
    print("STEP 2: Compute Metrics (EXP-17 v2)")
    print("=" * 70)
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return
        
    df_meta = pd.read_csv(METADATA_FILE)
    print(f"Loaded {len(df_meta)} records from metadata")
    
    all_metrics = []
    
    for i, row in df_meta.iterrows():
        if (i+1) % 10 == 0:
            print(f"  Processing {i+1}/{len(df_meta)}...", flush=True)
        else:
            print(f"  Processing {i+1}/{len(df_meta)}...", end='\r')
            
        filename = row['filename'] # e.g. 0_direct.npy
        npy_path = os.path.join(HIDDEN_DIR, filename)
        
        if not os.path.exists(npy_path):
             continue

        try:
            # stack: [n_layers, n_tokens, hidden_dim]
            stack = np.load(npy_path).astype(np.float32)
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
            
        n_layers = stack.shape[0]
        
        # Per-layer metrics
        for layer in range(n_layers):
            h = stack[layer]
            metrics = compute_all_metrics_for_layer(h)
            metrics['layer'] = layer
            metrics['problem_id'] = row['problem_id']
            metrics['condition'] = row['condition']
            metrics['group'] = row['group']
            metrics['correct'] = row['correct']
            all_metrics.append(metrics)
            
        # Cross-layer metrics (Not fully implemented here to save time, use what we have)
        
        if (i+1) % 50 == 0:
            # Intermediate save
            pd.DataFrame(all_metrics).to_csv(os.path.join(DATA_DIR, f"exp17a_metrics_v2_checkpoint_{i+1}.csv"), index=False)
    
    print(f"\nSaving final results...")
    if not all_metrics:
        print("No metrics computed! Check hidden state paths.")
        return
        
    final_df = pd.DataFrame(all_metrics)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved metrics to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
