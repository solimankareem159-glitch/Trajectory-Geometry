"""
EXP-17A: Analysis & Reporting
=============================
analyzes the geometric signatures of Qwen2.5-3B-Instruct.
Compares Direct Fail (G1) vs CoT Success (G4).
Replicates EXP-09 analysis protocol.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import json

# --- Configuration ---
DATA_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
METRICS_FILE = os.path.join(DATA_DIR, "exp17a_metrics.csv")
REPORT_FILE = os.path.join(DATA_DIR, "exp17a_report.md")

# metrics to interpret
KEY_METRICS = [
    'speed', 
    'turning_angle', 
    'stabilization', 
    'dir_consistency', 
    'radius_of_gyration',
    'gyration_anisotropy',
    'dimension_effective', # effective_dim
    'spectral_entropy',
    'depth_accel_speed'
]

def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0)

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("=" * 70)
    print("EXP-17A: Analysis")
    print("=" * 70)
    
    if not os.path.exists(METRICS_FILE):
        print(f"Error: Metrics file not found at {METRICS_FILE}")
        return
        
    df = pd.read_csv(METRICS_FILE)
    print(f"Loaded {len(df)} rows from {METRICS_FILE}")
    
    # Filter Groups
    # G1: Direct Fail (Condition=direct, Correct=False)
    # G4: CoT Success (Condition=cot, Correct=True)
    
    g1 = df[(df['condition'] == 'direct') & (df['correct'] == False)]
    g4 = df[(df['condition'] == 'cot') & (df['correct'] == True)]
    
    print(f"G1 (Direct Fail): {len(g1)} rows")
    print(f"G4 (CoT Success): {len(g4)} rows")
    
    if len(g1) == 0 or len(g4) == 0:
        print("Warning: One or both groups empty. Cannot perform comparison.")
        return

    # Analyze per layer
    layers = sorted(df['layer'].dropna().unique().astype(int))
    # Filter out -1 (cross-layer)
    layers = [l for l in layers if l >= 0]
    
    results = []
    
    for layer in layers:
        g1_l = g1[g1['layer'] == layer]
        g4_l = g4[g4['layer'] == layer]
        
        if len(g1_l) < 5 or len(g4_l) < 5:
            continue
            
        for metric in KEY_METRICS:
            if metric not in df.columns: continue
            
            vals1 = g1_l[metric].dropna()
            vals4 = g4_l[metric].dropna()
            
            if len(vals1) < 5 or len(vals4) < 5: continue
            
            # T-test
            t_stat, p_val = stats.ttest_ind(vals4, vals1, equal_var=False)
            d = cohen_d(vals4, vals1)
            
            results.append({
                'layer': layer,
                'metric': metric,
                'mean_g4': vals4.mean(),
                'mean_g1': vals1.mean(),
                'diff': vals4.mean() - vals1.mean(),
                'p_value': p_val,
                'cohen_d': d
            })
            
    # Cross-layer analysis
    g1_x = g1[g1['layer'] == -1]
    g4_x = g4[g4['layer'] == -1]
    
    if len(g1_x) > 5 and len(g4_x) > 5:
        for metric in g1_x.columns:
            if metric in ['layer', 'group', 'problem_id', 'condition', 'correct', 'Unnamed: 0']: continue
            if pd.api.types.is_numeric_dtype(g1_x[metric]):
                vals1 = g1_x[metric].dropna()
                vals4 = g4_x[metric].dropna()
                
                if len(vals1) < 5 or len(vals4) < 5: continue
                
                t_stat, p_val = stats.ttest_ind(vals4, vals1, equal_var=False)
                d = cohen_d(vals4, vals1)
                
                results.append({
                    'layer': 'Cross',
                    'metric': metric,
                    'mean_g4': vals4.mean(),
                    'mean_g1': vals1.mean(),
                    'diff': vals4.mean() - vals1.mean(),
                    'p_value': p_val,
                    'cohen_d': d
                })

    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        res_df.to_csv(os.path.join(DATA_DIR, "exp17a_stats.csv"), index=False)
        print(f"Saved stats to {DATA_DIR}/exp17a_stats.csv")
        
        # Generator Report
        generate_report(res_df, len(g1), len(g4))
    else:
        print("No significant results found or insufficient data.")

def generate_report(stats_df, n_g1, n_g4):
    with open(REPORT_FILE, 'w') as f:
        f.write("# EXP-17A: Baseline Replication Report\n\n")
        f.write(f"**Model:** Qwen/Qwen2.5-3B-Instruct\n")
        f.write(f"**Comparison:** G4 (CoT Success, n={n_g4}) vs G1 (Direct Fail, n={n_g1})\n\n")
        
        f.write("## Significant Differences (p < 0.001)\n\n")
        sig = stats_df[stats_df['p_value'] < 0.001].sort_values('cohen_d', ascending=False)
        
        if len(sig) > 0:
            f.write("| Layer | Metric | Cohen's d | Mean G4 | Mean G1 | p-value |\n")
            f.write("|---|---|---|---|---|---|\n")
            for _, r in sig.iterrows():
                f.write(f"| {r['layer']} | {r['metric']} | {r['cohen_d']:.2f} | {r['mean_g4']:.4f} | {r['mean_g1']:.4f} | {r['p_value']:.2e} |\n")
        else:
            f.write("No significant differences found at p < 0.001.\n")
            
        f.write("\n## Cross-Layer Metrics\n\n")
        cross = stats_df[stats_df['layer'] == 'Cross']
        if len(cross) > 0:
            f.write("| Metric | Cohen's d | Mean G4 | Mean G1 | p-value |\n")
            f.write("|---|---|---|---|---|\n")
            for _, r in cross.iterrows():
                f.write(f"| {r['metric']} | {r['cohen_d']:.2f} | {r['mean_g4']:.4f} | {r['mean_g1']:.4f} | {r['p_value']:.2e} |\n")

    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
