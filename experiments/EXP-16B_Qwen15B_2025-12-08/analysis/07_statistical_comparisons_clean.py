"""
Experiment 16B: Statistical Comparisons (Clean Reasoning States)
=================================================================
Perform comprehensive pairwise statistical comparisons across all groups.

Group Definitions:
- G1: Direct Answer, FAIL
- G2: Direct Answer, SUCCESS  
- G3: Chain-of-Thought, FAIL
- G4: Chain-of-Thought, SUCCESS

Comparisons (6 total):
1. G4 vs G1 (Primary: CoT Success vs Direct Fail)
2. G2 vs G1 (Direct Success vs Direct Fail)
3. G4 vs G3 (CoT Success vs CoT Fail)
4. G2 vs G3 (Cross-condition: Direct Success vs CoT Fail)
5. G4 vs G2 (CoT Success vs Direct Success)
6. G3 vs G1 (Cross-condition: CoT Fail vs Direct Fail)

Methods:
- Permutation test (10,000 shuffles) for p-values
- Cohen's d for effect size
- Bonferroni correction (α=0.05 / 6 = 0.0083)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import json

# Configuration
DATA_DIR = r"experiments/Experiment 16B/data"
METRICS_FILE = os.path.join(DATA_DIR, "exp16b_metrics_clean.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "exp16b_statistical_comparisons_clean.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "exp16b_statistical_summary_clean.json")

# Statistical parameters
N_PERMUTATIONS = 10000
BONFERRONI_N = 6  # 6 pairwise comparisons
ALPHA = 0.05 / BONFERRONI_N  # 0.0083

print(f"PID: {os.getpid()}", flush=True)
print("=" * 70)
print("Experiment 16B: Statistical Comparisons (Clean States)")
print("=" * 70)
print(f"Bonferroni-corrected alpha: {ALPHA:.4f}")

# Load metrics
print(f"\nLoading metrics from {METRICS_FILE}...")
df = pd.read_csv(METRICS_FILE)
print(f"Loaded {len(df)} metric rows")

# Identify all metric columns (exclude metadata)
metadata_cols = ['layer', 'problem_id', 'condition', 'correct', 'group']
metric_cols = [col for col in df.columns if col not in metadata_cols]
print(f"\nFound {len(metric_cols)} metrics")

# Define comparison pairs
comparisons = [
    ('G4', 'G1', 'CoT Success vs Direct Fail'),
    ('G2', 'G1', 'Direct Success vs Direct Fail'),
    ('G4', 'G3', 'CoT Success vs CoT Fail'),
    ('G2', 'G3', 'Direct Success vs CoT Fail'),
    ('G4', 'G2', 'CoT Success vs Direct Success'),
    ('G3', 'G1', 'CoT Fail vs Direct Fail')
]

# Permutation test function
def permutation_test(group_a, group_b, n_perm=10000):
    """Two-tailed permutation test for mean difference"""
    if len(group_a) == 0 or len(group_b) == 0:
        return np.nan
    
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    
    diffs = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        diffs.append(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
    
    p_val = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p_val

# Cohen's d
def cohens_d(group_a, group_b):
    """Standardized mean difference"""
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan
    
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    
    s_pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if s_pooled < 1e-12:
        return np.nan
    
    mean_diff = np.mean(group_a) - np.mean(group_b)
    return mean_diff / s_pooled

# Run comparisons
print(f"\nRunning pairwise comparisons...")
all_results = []

# Get unique layers (including -1 for cross-layer metrics)
layers = sorted(df['layer'].unique())

total_comparisons = len(comparisons) * len(layers) * len(metric_cols)
print(f"Total comparisons to compute: {total_comparisons}")

with tqdm(total=total_comparisons, desc="Comparing") as pbar:
    for group_a_name, group_b_name, comparison_label in comparisons:
        for layer in layers:
            # Filter data for this layer
            df_layer = df[df['layer'] == layer]
            
            df_a = df_layer[df_layer['group'] == group_a_name]
            df_b = df_layer[df_layer['group'] == group_b_name]
            
            for metric in metric_cols:
                # Extract values (drop NaNs)
                vals_a = df_a[metric].dropna().values
                vals_b = df_b[metric].dropna().values
                
                if len(vals_a) < 2 or len(vals_b) < 2:
                    pbar.update(1)
                    continue
                
                # Compute statistics
                mean_a = np.mean(vals_a)
                mean_b = np.mean(vals_b)
                std_a = np.std(vals_a, ddof=1)
                std_b = np.std(vals_b, ddof=1)
                
                # Permutation test
                p_val = permutation_test(vals_a, vals_b, N_PERMUTATIONS)
                
                # Effect size
                d = cohens_d(vals_a, vals_b)
                
                # Significance
                significant = p_val < ALPHA if not np.isnan(p_val) else False
                
                # Store result
                all_results.append({
                    'comparison': comparison_label,
                    'group_a': group_a_name,
                    'group_b': group_b_name,
                    'layer': layer,
                    'metric': metric,
                    'mean_a': mean_a,
                    'mean_b': mean_b,
                    'std_a': std_a,
                    'std_b': std_b,
                    'mean_diff': mean_a - mean_b,
                    'p_value': p_val,
                    'cohens_d': d,
                    'significant': significant,
                    'n_a': len(vals_a),
                    'n_b': len(vals_b)
                })
                
                pbar.update(1)

# Save results
print(f"\n\nSaving results to {OUTPUT_FILE}...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_FILE, index=False)

# Generate summary
print("Generating summary statistics...")
summary = {
    'total_comparisons': len(results_df),
    'bonferroni_alpha': ALPHA,
    'n_permutations': N_PERMUTATIONS,
    'significant_findings': {},
    'top_effects': {}
}

# Count significant findings per comparison
for comp_label in results_df['comparison'].unique():
    comp_df = results_df[results_df['comparison'] == comp_label]
    n_sig = len(comp_df[comp_df['significant'] == True])
    summary['significant_findings'][comp_label] = {
        'total': len(comp_df),
        'significant': n_sig,
        'percentage': (n_sig / len(comp_df) * 100) if len(comp_df) > 0 else 0
    }
    
    # Top 5 effects by abs(Cohen's d)
    top_effects = comp_df.nlargest(5, 'cohens_d', keep='all')[['metric', 'layer', 'cohens_d', 'p_value', 'significant']]
    summary['top_effects'][comp_label] = top_effects.to_dict('records')

# Save summary
with open(SUMMARY_FILE, 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for comp_label, stats_dict in summary['significant_findings'].items():
    print(f"\n{comp_label}:")
    print(f"  Total comparisons: {stats_dict['total']}")
    print(f"  Significant (p < {ALPHA:.4f}): {stats_dict['significant']} ({stats_dict['percentage']:.1f}%)")

print(f"\nOutputs:")
print(f"  Full results:  {OUTPUT_FILE}")
print(f"  Summary:       {SUMMARY_FILE}")
print("=" * 70)
