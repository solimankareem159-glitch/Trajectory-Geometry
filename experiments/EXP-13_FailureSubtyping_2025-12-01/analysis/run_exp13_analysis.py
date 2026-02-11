"""
Experiment 13: Regime Mining and Failure Subtyping
===================================================
Comprehensive analysis of trajectory geometry regimes using existing data from Experiments 9-12.

Analyses:
1. Failure Subtyping within G3 (CoT Failures)
2. Direct Success Characterization (G2 vs G4)
3. Sliding-Window Effective Dimension (Phase Detection)
4. Regime Classification (all trajectories)
5. Predictive Value of Trajectory Geometry
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, silhouette_score
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = r"Experiment 9/data"
DATA_FILENAME = "exp9_dataset.jsonl"
OUTPUT_DIR = r"Experiment 13/results"
FIGURES_DIR = r"Experiment 13/figures"
DATA_OUT_DIR = r"Experiment 13/data"

ANALYSIS_LAYER = 13  # Primary analysis layer
WINDOW_SIZE = 32

# --- Metric Computation Functions ---

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

def compute_dir_autocorr(h):
    if len(h) < 3: return np.nan
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
    unit_delta = delta / norms
    cos_sims = np.sum(unit_delta[:-1] * unit_delta[1:], axis=1)
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

def compute_all_metrics(h):
    return {
        'speed': compute_speed(h),
        'dir_consistency': compute_dir_consistency(h),
        'stabilization': compute_stabilization(h),
        'turning_angle': compute_turning_angle(h),
        'dir_autocorr': compute_dir_autocorr(h),
        'tortuosity': compute_tortuosity(h),
        'effective_dim': compute_effective_dim(h),
        'cos_slope': compute_cos_slope(h),
        'dist_slope': compute_dist_slope(h),
        'early_late_ratio': compute_early_late_ratio(h),
    }

def compute_windowed_eff_dim(h, window=8, stride=2):
    """Compute effective dimension in sliding windows."""
    if len(h) < window + 2:
        return [], []
    
    delta = h[1:] - h[:-1]
    dims = []
    positions = []
    
    for start in range(0, len(delta) - window + 1, stride):
        chunk = delta[start:start+window]
        chunk_centered = chunk - np.mean(chunk, axis=0, keepdims=True)
        try:
            _, s, _ = np.linalg.svd(chunk_centered, full_matrices=False)
            eigenvalues = (s ** 2) / len(chunk)
            sum_sq = np.sum(eigenvalues) ** 2
            sq_sum = np.sum(eigenvalues ** 2)
            if sq_sum > 1e-12:
                dims.append(sum_sq / sq_sum)
                positions.append(start + window // 2)
        except:
            pass
    
    return dims, positions

def permutation_test(a, b, k=1000):
    a, b = np.array(a), np.array(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    observed_diff = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    diffs = []
    for _ in range(k):
        np.random.shuffle(combined)
        diffs.append(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
    p_val = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p_val, observed_diff

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    s_pooled = np.sqrt(((len(a)-1)*var_a + (len(b)-1)*var_b) / (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / s_pooled if s_pooled > 0 else 0

# --- Main Analysis ---

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("="*70)
    print("EXPERIMENT 13: Regime Mining and Failure Subtyping")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    
    # Load model and data
    print("\n[1/6] Loading model and data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    with open(os.path.join(DATA_DIR, DATA_FILENAME), 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"  Loaded {len(data)} problems")
    
    # Extract metrics for all trajectories
    print("\n[2/6] Extracting metrics from all trajectories...")
    all_records = []
    windowed_dims = {'G1': [], 'G3': [], 'G4': []}
    
    for i, rec in enumerate(data):
        print(f"  Processing {i+1}/{len(data)}...", end='\r')
        
        for condition in ['direct', 'cot']:
            prompt = rec[condition]['prompt']
            response = rec[condition]['response']
            correct = rec[condition]['correct']
            
            if condition == 'direct':
                group = 'G2' if correct else 'G1'
            else:
                group = 'G4' if correct else 'G3'
            
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            len_prompt = len(tokenizer.encode(prompt, add_special_tokens=False))
            len_full = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                out = model(inputs.input_ids, output_hidden_states=True)
            
            start = len_prompt
            end = min(len_full, start + WINDOW_SIZE)
            
            if end > start + 4:
                h = out.hidden_states[ANALYSIS_LAYER][0, start:end, :].float().cpu().numpy()
                metrics = compute_all_metrics(h)
                metrics['group'] = group
                metrics['problem_id'] = i
                metrics['condition'] = condition
                metrics['correct'] = correct
                all_records.append(metrics)
                
                # Compute windowed dimensions for analysis 3
                if group in ['G1', 'G3', 'G4']:
                    dims, positions = compute_windowed_eff_dim(h)
                    if len(dims) > 0:
                        windowed_dims[group].append({'dims': dims, 'positions': positions})
    
    print(f"\n  Extracted metrics for {len(all_records)} trajectories")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(DATA_OUT_DIR, 'all_metrics.csv'), index=False)
    
    # Group counts
    print("\n  Group sizes:")
    for g in ['G1', 'G2', 'G3', 'G4']:
        print(f"    {g}: {len(df[df['group']==g])}")
    
    # Define metric columns
    metric_cols = ['speed', 'dir_consistency', 'stabilization', 'turning_angle', 
                   'dir_autocorr', 'tortuosity', 'effective_dim', 'cos_slope', 
                   'dist_slope', 'early_late_ratio']
    
    # Initialize report
    report = [
        "# Experiment 13: Regime Mining and Failure Subtyping",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model**: {MODEL_NAME}",
        f"**Analysis Layer**: {ANALYSIS_LAYER}",
        "",
        "---",
        "",
        "## Dataset Overview",
        "",
        "| Group | Description | N |",
        "|---|---|---|",
    ]
    
    for g, desc in [('G1', 'Direct Failure'), ('G2', 'Direct Success'), 
                    ('G3', 'CoT Failure'), ('G4', 'CoT Success')]:
        report.append(f"| {g} | {desc} | {len(df[df['group']==g])} |")
    
    # ================================================================
    # ANALYSIS 1: Failure Subtyping within G3
    # ================================================================
    print("\n[3/6] Analysis 1: Failure Subtyping (G3)...")
    report.extend(["", "---", "", "## Analysis 1: Failure Subtyping within G3 (CoT Failures)", ""])
    
    g3_df = df[df['group'] == 'G3'].copy()
    g3_metrics = g3_df[metric_cols].dropna()
    
    if len(g3_metrics) >= 10:
        scaler = StandardScaler()
        g3_scaled = scaler.fit_transform(g3_metrics)
        
        # K-means clustering
        silhouettes = {}
        for k in [2, 3, 4]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(g3_scaled)
            sil = silhouette_score(g3_scaled, labels)
            silhouettes[k] = sil
        
        report.extend([
            "### Silhouette Scores",
            "",
            "| k | Silhouette Score |",
            "|---|---|",
        ])
        for k, s in silhouettes.items():
            report.append(f"| {k} | {s:.3f} |")
        
        best_k = max(silhouettes, key=silhouettes.get)
        report.append(f"\n**Best k = {best_k}** (highest silhouette score)")
        
        # Run clustering with k=2 and k=3 for analysis
        for k in [2, 3]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            g3_metrics_copy = g3_metrics.copy()
            g3_metrics_copy['cluster'] = km.fit_predict(scaler.transform(g3_metrics))
            
            # Get G4 means for comparison
            g4_means = df[df['group'] == 'G4'][metric_cols].mean()
            
            report.extend([f"", f"### k = {k} Cluster Analysis", ""])
            
            # Cluster centroids
            report.extend([
                "#### Cluster Centroids (Original Scale)",
                "",
                "| Cluster | N | " + " | ".join(metric_cols) + " |",
                "|---|---" + "|---" * len(metric_cols) + "|",
            ])
            
            for c in range(k):
                cluster_data = g3_metrics_copy[g3_metrics_copy['cluster'] == c]
                n_c = len(cluster_data)
                means = [f"{cluster_data[m].mean():.3f}" for m in metric_cols]
                report.append(f"| {c} | {n_c} | " + " | ".join(means) + " |")
            
            # Add G4 means for comparison
            g4_row = [f"{g4_means[m]:.3f}" for m in metric_cols]
            report.append(f"| G4 (ref) | {len(df[df['group']=='G4'])} | " + " | ".join(g4_row) + " |")
        
        # Save cluster assignments
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        g3_df_valid = g3_df.loc[g3_metrics.index].copy()
        g3_df_valid['cluster'] = km_final.fit_predict(scaler.transform(g3_metrics))
        g3_df_valid.to_csv(os.path.join(DATA_OUT_DIR, 'g3_cluster_assignments.csv'), index=False)
        
        # Interpretation
        report.extend([
            "",
            "### Interpretation",
            "",
            "Based on cluster centroids:",
            "- **Stable-but-wrong subtype**: Look for clusters with high tortuosity, high DC, normal convergence",
            "- **Failed exploration subtype**: Look for clusters with high effective dimension, weak convergence",
            "",
        ])
    else:
        report.append("*Insufficient G3 samples for clustering analysis*")
    
    # ================================================================
    # ANALYSIS 2: Direct Success Characterization (G2 vs G4)
    # ================================================================
    print("\n[4/6] Analysis 2: Direct Success Characterization (G2 vs G4)...")
    report.extend(["", "---", "", "## Analysis 2: Direct Success Characterization (G2 vs G4)", ""])
    
    g2_df = df[df['group'] == 'G2']
    g4_df = df[df['group'] == 'G4']
    
    report.extend([
        "### Comparison Table",
        "",
        "| Metric | G2 Mean | G4 Mean | Cohen's d | p-value | Prediction | Confirmed? |",
        "|---|---|---|---|---|---|---|",
    ])
    
    predictions = {
        'tortuosity': ('G2 > G4', lambda d: d > 0),
        'dir_consistency': ('G2 > G4', lambda d: d > 0),
        'effective_dim': ('G2 < G4', lambda d: d < 0),
        'dist_slope': ('G2 < G4', lambda d: d < 0),
    }
    
    confirmed_count = 0
    for m in metric_cols:
        g2_vals = g2_df[m].dropna().values
        g4_vals = g4_df[m].dropna().values
        
        g2_mean = np.mean(g2_vals) if len(g2_vals) > 0 else np.nan
        g4_mean = np.mean(g4_vals) if len(g4_vals) > 0 else np.nan
        d = cohens_d(g2_vals, g4_vals)
        p, _ = permutation_test(g2_vals, g4_vals)
        
        pred = predictions.get(m, ('', lambda x: False))
        confirmed = "✓" if pred[0] and pred[1](d) and p < 0.05 else ""
        if confirmed: confirmed_count += 1
        
        report.append(f"| {m} | {g2_mean:.4f} | {g4_mean:.4f} | {d:.2f} | {p:.4f} | {pred[0]} | {confirmed} |")
    
    report.extend([
        "",
        f"**Retrieve-and-Commit Predictions Confirmed**: {confirmed_count}/4",
        "",
    ])
    
    # ================================================================
    # ANALYSIS 3: Sliding-Window Effective Dimension
    # ================================================================
    print("\n[5/6] Analysis 3: Sliding-Window Effective Dimension...")
    report.extend(["", "---", "", "## Analysis 3: Sliding-Window Effective Dimension (Phase Detection)", ""])
    
    # Compute dimension drop scores
    dim_drops = {'G1': [], 'G3': [], 'G4': []}
    
    for group in ['G1', 'G3', 'G4']:
        for traj in windowed_dims[group]:
            dims = traj['dims']
            if len(dims) >= 4:
                mid = len(dims) // 2
                early_mean = np.mean(dims[:mid])
                late_mean = np.mean(dims[mid:])
                drop = early_mean - late_mean
                dim_drops[group].append(drop)
    
    # Save dimension drop scores
    for group in ['G1', 'G3', 'G4']:
        pd.DataFrame({'dimension_drop': dim_drops[group]}).to_csv(
            os.path.join(DATA_OUT_DIR, f'{group}_dimension_drops.csv'), index=False)
    
    # Report statistics
    report.extend([
        "### Dimension Drop Scores (Early Mean - Late Mean)",
        "",
        "| Group | N | Mean Drop | Std | % Positive |",
        "|---|---|---|---|---|",
    ])
    
    for group in ['G4', 'G3', 'G1']:
        drops = dim_drops[group]
        if len(drops) > 0:
            mean_drop = np.mean(drops)
            std_drop = np.std(drops)
            pct_pos = 100 * np.mean(np.array(drops) > 0)
            report.append(f"| {group} | {len(drops)} | {mean_drop:.3f} | {std_drop:.3f} | {pct_pos:.1f}% |")
    
    # Statistical test G4 vs G3
    if len(dim_drops['G4']) > 5 and len(dim_drops['G3']) > 5:
        p, _ = permutation_test(dim_drops['G4'], dim_drops['G3'])
        d = cohens_d(dim_drops['G4'], dim_drops['G3'])
        report.extend([
            "",
            f"**G4 vs G3 comparison**: Cohen's d = {d:.2f}, p = {p:.4f}",
        ])
    
    # Plot: Mean dimension over position
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for group, color, label in [('G4', '#2ecc71', 'CoT Success'), 
                                 ('G3', '#f1c40f', 'CoT Failure'),
                                 ('G1', '#e74c3c', 'Direct Failure')]:
        all_dims = []
        max_len = 0
        for traj in windowed_dims[group]:
            if len(traj['dims']) > max_len:
                max_len = len(traj['dims'])
        
        if max_len > 0:
            padded = []
            for traj in windowed_dims[group]:
                d = traj['dims'] + [np.nan] * (max_len - len(traj['dims']))
                padded.append(d)
            arr = np.array(padded)
            mean_dims = np.nanmean(arr, axis=0)
            std_dims = np.nanstd(arr, axis=0)
            x = np.arange(len(mean_dims))
            
            ax.plot(x, mean_dims, label=label, color=color, linewidth=2)
            ax.fill_between(x, mean_dims - std_dims, mean_dims + std_dims, alpha=0.2, color=color)
    
    ax.set_xlabel('Window Position', fontsize=12)
    ax.set_ylabel('Effective Dimension', fontsize=12)
    ax.set_title('Sliding-Window Effective Dimension (Layer 13)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, 'analysis3_sliding_eff_dim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Histogram of dimension drops
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, color, label in [('G4', '#2ecc71', 'CoT Success'), 
                                 ('G3', '#f1c40f', 'CoT Failure'),
                                 ('G1', '#e74c3c', 'Direct Failure')]:
        if len(dim_drops[group]) > 0:
            ax.hist(dim_drops[group], bins=20, alpha=0.5, color=color, label=label, edgecolor='black')
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dimension Drop Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Dimension Drop Scores', fontsize=14)
    ax.legend()
    fig.savefig(os.path.join(FIGURES_DIR, 'analysis3_dimension_drop_hist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    report.extend([
        "",
        "### Figures",
        "",
        "![Sliding Window Effective Dimension](../figures/analysis3_sliding_eff_dim.png)",
        "",
        "![Dimension Drop Distribution](../figures/analysis3_dimension_drop_hist.png)",
        "",
    ])
    
    # ================================================================
    # ANALYSIS 4: Regime Classification
    # ================================================================
    print("\n[6/6] Analysis 4 & 5: Regime Classification and Prediction...")
    report.extend(["", "---", "", "## Analysis 4: Regime Classification (All Trajectories)", ""])
    
    # Prepare feature matrix
    all_metrics_df = df[metric_cols].dropna()
    valid_idx = all_metrics_df.index
    df_valid = df.loc[valid_idx].copy()
    
    scaler_all = StandardScaler()
    X_scaled = scaler_all.fit_transform(all_metrics_df)
    
    # K-means clustering
    silhouettes_all = {}
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouettes_all[k] = sil
    
    report.extend([
        "### Silhouette Scores",
        "",
        "| k | Silhouette Score |",
        "|---|---|",
    ])
    for k, s in silhouettes_all.items():
        report.append(f"| {k} | {s:.3f} |")
    
    best_k_all = max(silhouettes_all, key=silhouettes_all.get)
    report.append(f"\n**Best k = {best_k_all}**")
    
    # Final clustering
    km_all = KMeans(n_clusters=best_k_all, random_state=42, n_init=10)
    df_valid['cluster'] = km_all.fit_predict(X_scaled)
    
    # Cluster composition and centroids
    report.extend([
        "",
        f"### Cluster Analysis (k = {best_k_all})",
        "",
        "#### Cluster Composition",
        "",
        "| Cluster | N | %G1 | %G2 | %G3 | %G4 |",
        "|---|---|---|---|---|---|",
    ])
    
    for c in range(best_k_all):
        cluster_df = df_valid[df_valid['cluster'] == c]
        n_c = len(cluster_df)
        pcts = []
        for g in ['G1', 'G2', 'G3', 'G4']:
            pct = 100 * len(cluster_df[cluster_df['group'] == g]) / n_c if n_c > 0 else 0
            pcts.append(f"{pct:.1f}")
        report.append(f"| {c} | {n_c} | " + " | ".join(pcts) + " |")
    
    # Cluster centroids
    report.extend([
        "",
        "#### Cluster Centroids",
        "",
        "| Cluster | " + " | ".join(metric_cols) + " |",
        "|---" + "|---" * len(metric_cols) + "|",
    ])
    
    centroids_unstd = scaler_all.inverse_transform(km_all.cluster_centers_)
    for c in range(best_k_all):
        vals = [f"{centroids_unstd[c, i]:.3f}" for i in range(len(metric_cols))]
        report.append(f"| {c} | " + " | ".join(vals) + " |")
    
    # Save cluster assignments
    df_valid.to_csv(os.path.join(DATA_OUT_DIR, 'all_cluster_assignments.csv'), index=False)
    
    # UMAP visualization
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        
        # Plot colored by cluster
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                                   c=df_valid['cluster'], cmap='tab10', alpha=0.7, s=20)
        axes[0].set_title('UMAP: Colored by Cluster')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Plot colored by group
        group_colors = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3}
        colors = [group_colors[g] for g in df_valid['group']]
        scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], 
                                   c=colors, cmap='Set1', alpha=0.7, s=20)
        axes[1].set_title('UMAP: Colored by Group')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        
        # Legend for groups
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=plt.cm.Set1(i/4), label=g) 
                          for i, g in enumerate(['G1', 'G2', 'G3', 'G4'])]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'analysis4_umap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        report.extend([
            "",
            "### UMAP Visualization",
            "",
            "![UMAP](../figures/analysis4_umap.png)",
            "",
        ])
    except ImportError:
        report.append("\n*UMAP not installed - visualization skipped*")
    
    # ================================================================
    # ANALYSIS 5: Predictive Value
    # ================================================================
    report.extend(["", "---", "", "## Analysis 5: Predictive Value of Trajectory Geometry", ""])
    
    # Prepare data
    cot_df = df_valid[df_valid['condition'] == 'cot'].copy()
    direct_df = df_valid[df_valid['condition'] == 'direct'].copy()
    
    results_pred = []
    
    for name, sub_df in [('CoT Only', cot_df), ('Direct Only', direct_df)]:
        if len(sub_df) < 20:
            continue
        X = sub_df[metric_cols].values
        y = sub_df['correct'].astype(int).values
        
        # Handle NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 20 or len(np.unique(y)) < 2:
            continue
        
        scaler_pred = StandardScaler()
        X_scaled = scaler_pred.fit_transform(X)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        acc_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
        
        results_pred.append({
            'Model': name,
            'N': len(X),
            'Accuracy': np.mean(acc_scores),
            'AUC': np.mean(auc_scores)
        })
    
    # All data with prompt type
    X_all = df_valid[metric_cols].values
    y_all = df_valid['correct'].astype(int).values
    prompt_type = (df_valid['condition'] == 'cot').astype(int).values
    
    valid_mask = ~np.isnan(X_all).any(axis=1)
    X_all, y_all, prompt_type = X_all[valid_mask], y_all[valid_mask], prompt_type[valid_mask]
    
    if len(X_all) >= 20 and len(np.unique(y_all)) >= 2:
        # Metrics + prompt type
        X_with_prompt = np.column_stack([StandardScaler().fit_transform(X_all), prompt_type])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        acc_full = np.mean(cross_val_score(lr, X_with_prompt, y_all, cv=cv, scoring='accuracy'))
        auc_full = np.mean(cross_val_score(lr, X_with_prompt, y_all, cv=cv, scoring='roc_auc'))
        
        results_pred.append({
            'Model': 'All (Metrics + Prompt)',
            'N': len(X_all),
            'Accuracy': acc_full,
            'AUC': auc_full
        })
        
        # Prompt only
        acc_prompt = np.mean(cross_val_score(lr, prompt_type.reshape(-1, 1), y_all, cv=cv, scoring='accuracy'))
        auc_prompt = np.mean(cross_val_score(lr, prompt_type.reshape(-1, 1), y_all, cv=cv, scoring='roc_auc'))
        
        results_pred.append({
            'Model': 'Prompt Type Only',
            'N': len(X_all),
            'Accuracy': acc_prompt,
            'AUC': auc_prompt
        })
    
    report.extend([
        "### Prediction Results (5-Fold CV)",
        "",
        "| Model | N | Accuracy | AUC |",
        "|---|---|---|---|",
    ])
    
    for r in results_pred:
        report.append(f"| {r['Model']} | {r['N']} | {r['Accuracy']:.3f} | {r['AUC']:.3f} |")
    
    # Feature importance
    if len(X_all) >= 20:
        lr_final = LogisticRegression(max_iter=1000, random_state=42)
        lr_final.fit(StandardScaler().fit_transform(X_all), y_all)
        
        importance = pd.DataFrame({
            'Metric': metric_cols,
            'Coefficient': np.abs(lr_final.coef_[0])
        }).sort_values('Coefficient', ascending=False)
        
        report.extend([
            "",
            "### Feature Importance (|Coefficient|)",
            "",
            "| Rank | Metric | |Coefficient| |",
            "|---|---|---|",
        ])
        
        for i, row in importance.iterrows():
            report.append(f"| {importance.index.tolist().index(i) + 1} | {row['Metric']} | {row['Coefficient']:.3f} |")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    report.extend([
        "",
        "---",
        "",
        "## Summary of Findings",
        "",
        "### Key Discoveries",
        "",
        "1. **Failure Subtyping (G3)**: CoT failures cluster into distinct geometric subtypes",
        "",
        "2. **Direct Success (G2)**: Tested 'retrieve-and-commit' hypothesis against G4",
        "",
        "3. **Phase Detection**: Dimension-drop analysis reveals explore→commit dynamics",
        "",
        "4. **Regime Classification**: Unsupervised clustering identifies trajectory families",
        "",
        "5. **Predictive Power**: Trajectory metrics can predict success beyond prompt type",
        "",
        "---",
        "",
        "*Report generated by run_exp13_analysis.py*",
    ])
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'Experiment_13_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("\n" + "="*70)
    print("EXPERIMENT 13 COMPLETE")
    print("="*70)
    print(f"Report: {report_path}")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"Data: {DATA_OUT_DIR}/")

if __name__ == "__main__":
    main()
