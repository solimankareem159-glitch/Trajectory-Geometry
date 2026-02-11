"""
Step 3: Generate Report
=======================
Loads computed metrics (CSV), runs statistical tests, generates figures, and writes the report.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Configuration ---
DATA_DIR = r"experiments/Experiment 14/data"
INPUT_FILE = os.path.join(DATA_DIR, "exp14_metrics.csv")
OUTPUT_DIR = r"experiments/Experiment 14/results"
FIGURES_DIR = r"experiments/Experiment 14/figures"
REPORT_FILE = os.path.join(OUTPUT_DIR, "exp14_extended_metrics_report.md")

ALL_LAYERS = list(range(25))
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# ============================================================
# STAT FUNCTIONS
# ============================================================

def permutation_test(a, b, k=1000):
    a, b = np.array(a), np.array(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    observed_diff = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(k):
        np.random.shuffle(combined)
        diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if np.abs(diff) >= np.abs(observed_diff):
            count += 1
    return count / k

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    s_pooled = np.sqrt(((len(a)-1)*var_a + (len(b)-1)*var_b) / (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / s_pooled if s_pooled > 0 else 0

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("=" * 70)
    print("STEP 3: Generate Report")
    print("=" * 70)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found {INPUT_FILE}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading metrics...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records")
    
    # Filter for per-layer and cross-layer
    df_layers = df[df['layer'] >= 0].copy()
    df_cross = df[df['layer'] == -1].copy()
    
    per_layer_metrics = [
        'speed', 'dir_consistency', 'stabilization', 'turning_angle', 'dir_autocorr',
        'tortuosity', 'effective_dim', 'cos_slope', 'dist_slope', 'early_late_ratio',
        'radius_of_gyration', 'gyration_anisotropy', 'drift_to_spread',
        'dir_autocorr_lag2', 'dir_autocorr_lag4', 'dir_autocorr_lag8',
        'vel_autocorr_lag1', 'vel_autocorr_lag2', 'vel_autocorr_lag4',
        'msd_exponent', 'cos_to_running_mean', 'cos_to_late_window', 'time_to_commit',
        'recurrence_rate', 'determinism', 'laminarity', 'trapping_time', 'diagonal_entropy',
        'psd_slope', 'spectral_entropy'
    ]
    
    cross_layer_metrics = ['interlayer_alignment_mean', 'depth_accel_speed', 'depth_accel_tortuosity']
    
    # Pairwise Comparisons
    print("Running comparisons...")
    group_pairs = [
        ('G4', 'G1', 'CoT Success vs Direct Failure'),
        ('G4', 'G3', 'CoT Success vs CoT Failure'),
        ('G2', 'G1', 'Direct Success vs Direct Failure'),
        ('G4', 'G2', 'CoT Success vs Direct Success'),
        ('G3', 'G1', 'CoT Failure vs Direct Failure'),
        ('G2', 'G3', 'Direct Success vs CoT Failure'),
    ]
    
    comparison_results = []
    
    # Per-layer comparisons
    layers_to_analyze = sorted(df_layers['layer'].unique())
    
    count = 0
    total_comparisons = len(group_pairs) * len(layers_to_analyze) * len(per_layer_metrics)
    
    for g1, g2, desc in group_pairs:
        for layer in layers_to_analyze:
            df_g1 = df_layers[(df_layers['group'] == g1) & (df_layers['layer'] == layer)]
            df_g2 = df_layers[(df_layers['group'] == g2) & (df_layers['layer'] == layer)]
            
            for metric in per_layer_metrics:
                if metric not in df.columns: continue
                
                vals_g1 = df_g1[metric].dropna().values
                vals_g2 = df_g2[metric].dropna().values
                
                if len(vals_g1) < 5 or len(vals_g2) < 5:
                    continue
                
                d = cohens_d(vals_g1, vals_g2)
                p = permutation_test(vals_g1, vals_g2, k=500) # Reduce k for speed
                
                comparison_results.append({
                    'comparison': desc,
                    'g1': g1,
                    'g2': g2,
                    'layer': layer,
                    'metric': metric,
                    'g1_mean': np.mean(vals_g1),
                    'g2_mean': np.mean(vals_g2),
                    'cohens_d': d,
                    'p_value': p,
                    'significant': p < 0.05 and np.abs(d) > 0.5
                })
                count += 1
                if count % 1000 == 0:
                     print(f"  Comparison {count}...", end='\r')
    
    # Cross-layer comparisons
    for g1, g2, desc in group_pairs:
        df_g1 = df_cross[df_cross['group'] == g1]
        df_g2 = df_cross[df_cross['group'] == g2]
        
        for metric in cross_layer_metrics:
            if metric not in df.columns: continue
            
            vals_g1 = df_g1[metric].dropna().values
            vals_g2 = df_g2[metric].dropna().values
            
            if len(vals_g1) < 5 or len(vals_g2) < 5:
                continue
            
            d = cohens_d(vals_g1, vals_g2)
            p = permutation_test(vals_g1, vals_g2, k=500)
            
            comparison_results.append({
                'comparison': desc,
                'g1': g1,
                'g2': g2,
                'layer': 'cross',
                'metric': metric,
                'g1_mean': np.mean(vals_g1),
                'g2_mean': np.mean(vals_g2),
                'cohens_d': d,
                'p_value': p,
                'significant': p < 0.05 and np.abs(d) > 0.5
            })

    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(os.path.join(DATA_DIR, 'exp14_comparisons.csv'), index=False)
    
    # Figures
    print("\nGenerating figures...")
    
    colors = {'G1': '#e74c3c', 'G2': '#3498db', 'G3': '#f1c40f', 'G4': '#2ecc71'}
    labels = {'G1': 'Direct Failure', 'G2': 'Direct Success', 'G3': 'CoT Failure', 'G4': 'CoT Success'}
    
    # Fig 1: Evolution
    key_metrics_to_plot = [
        'speed', 'effective_dim', 'tortuosity', 'dir_consistency',
        'msd_exponent', 'radius_of_gyration', 'drift_to_spread',
        'recurrence_rate', 'laminarity', 'psd_slope'
    ]
    key_metrics_to_plot = [m for m in key_metrics_to_plot if m in df.columns]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(key_metrics_to_plot):
        if idx >= len(axes): break
        ax = axes[idx]
        for group in ['G1', 'G2', 'G3', 'G4']:
            means = []
            stds = []
            for layer in layers_to_analyze:
                vals = df_layers[(df_layers['group'] == group) & (df_layers['layer'] == layer)][metric].dropna().values
                if len(vals) > 0:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(layers_to_analyze, means, label=labels[group], color=colors[group], linewidth=2)
            ax.fill_between(layers_to_analyze, means - stds, means + stds, alpha=0.2, color=colors[group])
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'exp14_layer_evolution.png'))
    plt.close()
    
    # Heatmaps
    heatmap_files = []
    print("Generating heatmaps...")
    
    for g1, g2, desc in group_pairs:
        sub_df = comp_df[(comp_df['g1'] == g1) & (comp_df['g2'] == g2) & (comp_df['layer'] != 'cross')]
        
        if sub_df.empty:
            continue
            
        pivot = sub_df.pivot(index='metric', columns='layer', values='cohens_d')
        
        # Sort metrics by mean absolute effect size to put important ones on top
        mean_abs_d = pivot.abs().mean(axis=1).sort_values(ascending=False)
        pivot = pivot.reindex(mean_abs_d.index)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': "Cohen's d"})
        ax.set_title(f"Cohen's d: {desc} ({g1} vs {g2})")
        
        filename = f'exp14_heatmap_{g1.lower()}_{g2.lower()}.png'
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.tight_layout()
        fig.savefig(filepath)
        plt.close()
        heatmap_files.append((desc, filename))
            
    # Report Generation
    print("Writing report...")
    
    report = [
        "# Experiment 14: Extended Trajectory Geometry Analysis",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Key Findings",
        ""
    ]
    
    # Top effects for G4 vs G1 (Primary)
    top_effects = comp_df[(comp_df['g1'] == 'G4') & (comp_df['g2'] == 'G1') & 
                          (comp_df['significant'] == True) & (np.abs(comp_df['cohens_d']) > 1.0)]
    top_effects = top_effects.sort_values('cohens_d', key=abs, ascending=False).head(20)
    
    report.append("### Top Discriminators (G4 vs G1)")
    report.append("| Layer | Metric | Cohen's d |")
    report.append("|---|---|---|")
    for _, row in top_effects.iterrows():
        report.append(f"| {row['layer']} | {row['metric']} | {row['cohens_d']:.2f} |")

    # Success Comparison: G4 vs G2
    success_effects = comp_df[(comp_df['g1'] == 'G4') & (comp_df['g2'] == 'G2') & 
                              (comp_df['significant'] == True)]
    success_effects = success_effects.sort_values('cohens_d', key=abs, ascending=False).head(15)
    
    report.extend([
        "",
        "### Success Comparison: CoT vs Direct (G4 vs G2)",
        "Top differences between successful CoT and successful Direct answers.",
        "",
        "| Layer | Metric | Cohen's d | G4 Mean | G2 Mean |",
        "|---|---|---|---|---|",
    ])
    for _, row in success_effects.iterrows():
        report.append(f"| {row['layer']} | {row['metric']} | {row['cohens_d']:.2f} | {row['g1_mean']:.4f} | {row['g2_mean']:.4f} |")

    # Failure Comparison: G3 vs G1
    failure_effects = comp_df[(comp_df['g1'] == 'G3') & (comp_df['g2'] == 'G1') & 
                              (comp_df['significant'] == True)]
    failure_effects = failure_effects.sort_values('cohens_d', key=abs, ascending=False).head(15)
    
    report.extend([
        "",
        "### Failure Comparison: CoT vs Direct (G3 vs G1)",
        "Top differences between CoT failures and Direct failures.",
        "",
        "| Layer | Metric | Cohen's d | G3 Mean | G1 Mean |",
        "|---|---|---|---|---|",
    ])
    for _, row in failure_effects.iterrows():
        report.append(f"| {row['layer']} | {row['metric']} | {row['cohens_d']:.2f} | {row['g1_mean']:.4f} | {row['g2_mean']:.4f} |")
    
    report.extend([
        "",
        "## Visualizations",
        "",
        "### Layer Evolution",
        "![Layer Evolution](../figures/exp14_layer_evolution.png)",
        "",
        "### Heatmaps",
    ])
    
    for desc, filename in heatmap_files:
        report.append(f"**{desc}**")
        report.append(f"![{desc}](../figures/{filename})")
        report.append("")
        
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report))
        
    print(f"Done. Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
