"""
Step 2: Compute Metrics (EXP-16B Qwen 1.5B - CORRECTED)
=======================================================
Loads extracted hidden states (.npy) from EXP-16B_Qwen15B and computes geometric metrics.
Corrects previous error of targeting Pythia 70M folder.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = r"experiments/EXP-16B_Qwen15B_2025-12-08/data"
HIDDEN_DIR = os.path.join(DATA_DIR, "hidden_states")
METADATA_FILE = os.path.join(DATA_DIR, "exp16b_dataset.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "exp16b_full_metrics.csv")

# Qwen 1.5B has 28 layers (0-27).
ALL_LAYERS = list(range(28)) 

# ============================================================
# METRIC FUNCTIONS (Standard Suite)
# ============================================================

def compute_speed(h):
    if len(h) < 2: return np.nan
    delta = h[1:] - h[:-1]
    return float(np.mean(np.linalg.norm(delta, axis=1)))

def compute_dir_consistency(h):
    if len(h) < 3: return np.nan
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
    unit_delta = delta / norms
    mean_dir = np.mean(unit_delta, axis=0)
    return float(np.linalg.norm(mean_dir))

def compute_stabilization(h):
    if len(h) < 3: return np.nan
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1)
    if len(norms) < 2: return np.nan
    slope, _, _, _, _ = stats.linregress(np.arange(len(norms)), norms)
    return float(slope)

def compute_turning_angle(h):
    if len(h) < 3: return np.nan
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
    unit_delta = delta / norms
    cos_sims = np.sum(unit_delta[:-1] * unit_delta[1:], axis=1)
    cos_sims = np.clip(cos_sims, -1.0, 1.0)
    angles = np.arccos(cos_sims)
    return float(np.mean(angles))

def compute_dir_autocorr(h, lag=1):
    if len(h) < 3 + lag: return np.nan
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
    unit_delta = delta / norms
    if lag >= len(unit_delta): return np.nan
    cos_sims = np.sum(unit_delta[:-lag] * unit_delta[lag:], axis=1)
    cos_sims = np.clip(cos_sims, -1.0, 1.0)
    return float(np.mean(cos_sims))

def compute_tortuosity(h):
    if len(h) < 2: return np.nan
    delta = h[1:] - h[:-1]
    net_displacement = np.linalg.norm(np.sum(delta, axis=0))
    arc_length = np.sum(np.linalg.norm(delta, axis=1))
    if arc_length < 1e-9: return np.nan
    return float(net_displacement / arc_length)

def compute_effective_dim(h):
    if len(h) < 4: return np.nan
    delta = h[1:] - h[:-1]
    delta_centered = delta - np.mean(delta, axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(delta_centered, full_matrices=False)
        eigenvalues = (s ** 2) / len(delta)
        sum_sq = np.sum(eigenvalues) ** 2
        sq_sum = np.sum(eigenvalues ** 2)
        if sq_sum < 1e-12: return np.nan
        return float(sum_sq / sq_sum)
    except:
        return np.nan

def compute_cos_slope(h):
    if len(h) < 4: return np.nan
    h_final = h[-1]
    cos_to_final = []
    for t in range(len(h) - 1):
        norm_t = np.linalg.norm(h[t])
        norm_f = np.linalg.norm(h_final)
        if norm_t > 1e-10 and norm_f > 1e-10:
            cos = np.dot(h[t], h_final) / (norm_t * norm_f)
            cos_to_final.append(cos)
        else:
            cos_to_final.append(0.0)
    if len(cos_to_final) < 2: return np.nan
    slope, _, _, _, _ = stats.linregress(np.arange(len(cos_to_final)), cos_to_final)
    return float(slope)

def compute_dist_slope(h):
    if len(h) < 4: return np.nan
    h_final = h[-1]
    dist_to_final = [np.linalg.norm(h[t] - h_final) for t in range(len(h) - 1)]
    if len(dist_to_final) < 2: return np.nan
    slope, _, _, _, _ = stats.linregress(np.arange(len(dist_to_final)), dist_to_final)
    return float(slope)

def compute_early_late_ratio(h):
    if len(h) < 4: return np.nan
    h_final = h[-1]
    dist_to_final = [np.linalg.norm(h[t] - h_final) for t in range(len(h) - 1)]
    mid = len(dist_to_final) // 2
    early_mean = np.mean(dist_to_final[:mid]) if mid > 0 else 0.0
    late_mean = np.mean(dist_to_final[mid:]) if mid < len(dist_to_final) else 0.0
    eps = 1e-6 * (early_mean + late_mean + 1.0)
    ratio = (early_mean + eps) / (late_mean + eps)
    return float(np.clip(ratio, 1e-3, 1e3))

# --- New Metrics ---

def compute_radius_of_gyration(h):
    if len(h) < 2: return np.nan
    h_mean = np.mean(h, axis=0)
    rg_sq = np.mean(np.sum((h - h_mean) ** 2, axis=1))
    return float(np.sqrt(rg_sq))

def compute_gyration_anisotropy(h):
    """SVD Optimized Anisotropy"""
    if len(h) < 4: return np.nan
    h_centered = h - np.mean(h, axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(h_centered, full_matrices=False)
        eigenvalues = (s ** 2) / (len(h) - 1)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0: return np.nan
        p = eigenvalues / np.sum(eigenvalues)
        entropy = -np.sum(p * np.log(p + 1e-12))
        max_entropy = np.log(len(p))
        return float(entropy / max_entropy) if max_entropy > 0 else np.nan
    except:
        return np.nan

def compute_drift_to_spread(h):
    if len(h) < 2: return np.nan
    rg = compute_radius_of_gyration(h)
    if rg < 1e-9 or np.isnan(rg): return np.nan
    drift = np.linalg.norm(h[-1] - h[0])
    return float(drift / rg)

def compute_vel_autocorr(h, lag=1):
    if len(h) < 3 + lag: return np.nan
    delta = h[1:] - h[:-1]
    if lag >= len(delta): return np.nan
    autocorr = np.sum(delta[:-lag] * delta[lag:], axis=1)
    return float(np.mean(autocorr))

def compute_msd_exponent(h, max_tau=8):
    if len(h) < max_tau + 2: return np.nan
    taus = np.arange(1, min(max_tau + 1, len(h)))
    msd_vals = []
    for tau in taus:
        displacements = np.linalg.norm(h[tau:] - h[:-tau], axis=1) ** 2
        msd_vals.append(np.mean(displacements))
    
    log_tau = np.log(taus)
    log_msd = np.log(np.array(msd_vals) + 1e-12)
    
    try:
        slope, _, _, _, _ = stats.linregress(log_tau, log_msd)
        return float(slope)
    except:
        return np.nan

def compute_cos_to_running_mean(h):
    if len(h) < 4: return np.nan
    cos_vals = []
    running_sum = np.zeros_like(h[0])
    for t in range(len(h)):
        running_sum += h[t]
        running_mean = running_sum / (t + 1)
        norm_t = np.linalg.norm(h[t])
        norm_m = np.linalg.norm(running_mean)
        if norm_t > 1e-10 and norm_m > 1e-10:
            cos = np.dot(h[t], running_mean) / (norm_t * norm_m)
            cos_vals.append(cos)
    return float(np.mean(cos_vals)) if len(cos_vals) > 0 else np.nan

def compute_cos_to_late_window(h, window=8):
    if len(h) < window + 2: return np.nan
    late_mean = np.mean(h[-window:], axis=0)
    norm_late = np.linalg.norm(late_mean)
    if norm_late < 1e-10: return np.nan
    cos_vals = []
    for t in range(len(h) - window):
        norm_t = np.linalg.norm(h[t])
        if norm_t > 1e-10:
            cos = np.dot(h[t], late_mean) / (norm_t * norm_late)
            cos_vals.append(cos)
    return float(np.mean(cos_vals)) if len(cos_vals) > 0 else np.nan

def compute_time_to_commit(h, window=6, stride=2):
    if len(h) < window + 4: return np.nan
    rg_vals = []
    positions = []
    for start in range(0, len(h) - window + 1, stride):
        chunk = h[start:start+window]
        rg = compute_radius_of_gyration(chunk)
        if not np.isnan(rg):
            rg_vals.append(rg)
            positions.append(start + window // 2)
    
    if len(rg_vals) < 3: return np.nan
    rg_vals = np.array(rg_vals)
    diffs = rg_vals[:-1] - rg_vals[1:]  # Positive = drop
    max_drop_idx = np.argmax(diffs)
    return float(positions[max_drop_idx])

# --- Aggregation ---

def compute_all_metrics_for_layer(h):
    return {
        'speed': compute_speed(h),
        'dir_consistency': compute_dir_consistency(h),
        'stabilization': compute_stabilization(h),
        'turning_angle': compute_turning_angle(h),
        'dir_autocorr': compute_dir_autocorr(h, lag=1),
        'tortuosity': compute_tortuosity(h),
        'effective_dim': compute_effective_dim(h),
        'cos_slope': compute_cos_slope(h),
        'dist_slope': compute_dist_slope(h),
        'early_late_ratio': compute_early_late_ratio(h),
        'radius_of_gyration': compute_radius_of_gyration(h),
        'gyration_anisotropy': compute_gyration_anisotropy(h),
        'drift_to_spread': compute_drift_to_spread(h),
        'dir_autocorr_lag2': compute_dir_autocorr(h, lag=2),
        'dir_autocorr_lag4': compute_dir_autocorr(h, lag=4),
        'dir_autocorr_lag8': compute_dir_autocorr(h, lag=8),
        'vel_autocorr_lag1': compute_vel_autocorr(h, lag=1),
        'vel_autocorr_lag2': compute_vel_autocorr(h, lag=2),
        'vel_autocorr_lag4': compute_vel_autocorr(h, lag=4),
        'msd_exponent': compute_msd_exponent(h),
        'cos_to_running_mean': compute_cos_to_running_mean(h),
        'cos_to_late_window': compute_cos_to_late_window(h),
        'time_to_commit': compute_time_to_commit(h),
    }

def compute_cross_layer_metrics(stack, layers_to_compare):
    results = {}
    
    # Inter-layer alignment
    alignments = []
    for i in range(len(layers_to_compare) - 1):
        l1, l2 = layers_to_compare[i], layers_to_compare[i + 1]
        if l1 >= stack.shape[0] or l2 >= stack.shape[0]: continue
        
        h1, h2 = stack[l1], stack[l2]
        if len(h1) < 3: continue
        
        delta1 = h1[1:] - h1[:-1]
        delta2 = h2[1:] - h2[:-1]
        
        norms1 = np.linalg.norm(delta1, axis=1, keepdims=True) + 1e-9
        norms2 = np.linalg.norm(delta2, axis=1, keepdims=True) + 1e-9
        unit1 = delta1 / norms1
        unit2 = delta2 / norms2
        
        cos_sims = np.sum(unit1 * unit2, axis=1)
        alignments.append(np.mean(cos_sims))
    
    results['interlayer_alignment_mean'] = float(np.mean(alignments)) if alignments else np.nan
    
    return results

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("=" * 70)
    print("STEP 2: Compute Full Metrics (EXP-16B Qwen 1.5B Corrected)")
    print("=" * 70)
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return
        
    print(f"Loading metadata from {METADATA_FILE}...")
    dataset = []
    with open(METADATA_FILE, 'r') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    print(f"Loaded {len(dataset)} records from metadata")
    
    all_metrics = []
    
    # Cross layer subset - Adjusted for 28 layers (approx every 5th layer)
    cross_layer_subset = [0, 5, 10, 15, 20, 25, 27]
    
    for i, rec in enumerate(dataset):
        pid = rec['id']
        if (i+1) % 10 == 0:
            print(f"  Processing {i+1}/{len(dataset)}...", flush=True)
        else:
            print(f"  Processing {i+1}/{len(dataset)}...", end='\r')
            
        truth = rec['truth']
        
        conditions = ['direct', 'cot']
        
        for cond in conditions:
            # Metadata lookup
            info = rec[cond]
            # Standardize correct/hallucination logic based on file
            # Assuming file content is truth
            is_correct = info.get('correct', False)
            
            # Group assignment
            if cond == 'direct':
                group = 'G2' if is_correct else 'G1'
            else:
                group = 'G4' if is_correct else 'G3'
                
            # Filename loading: problem_{id:03d}_{condition}.npy
            filename = f"problem_{pid:03d}_{cond}.npy"
            npy_path = os.path.join(HIDDEN_DIR, filename)
            
            if not os.path.exists(npy_path):
                # Try simple format just in case
                alt_filename = f"{pid}_{cond}.npy"
                alt_path = os.path.join(HIDDEN_DIR, alt_filename)
                if os.path.exists(alt_path):
                    npy_path = alt_path
                else:
                    # Skip silently if not found (maybe not all processed?)
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
                metrics['group'] = group
                metrics['problem_id'] = pid
                metrics['condition'] = cond
                metrics['correct'] = is_correct
                all_metrics.append(metrics)
                
            # Cross-layer metrics
            cross_res = compute_cross_layer_metrics(stack, cross_layer_subset)
            cross_rec = {
                'layer': -1,
                'group': group,
                'problem_id': pid,
                'condition': cond,
                'correct': is_correct,
                **cross_res
            }
            all_metrics.append(cross_rec)
        
        if (i+1) % 50 == 0:
            # Intermediate save
            pd.DataFrame(all_metrics).to_csv(os.path.join(DATA_DIR, f"exp16b_full_metrics_checkpoint_{i+1}.csv"), index=False)
    
    print(f"\nSaving final results...")
    if not all_metrics:
        print("No metrics computed! Check hidden state paths.")
        return
        
    final_df = pd.DataFrame(all_metrics)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved metrics to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
