"""
Script 05: Statistical Tests and Replication Analysis
=====================================================
Runs 6 key comparisons and tests replication of Exp14 findings.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

def cohens_d(x, y):
    """Cohen's d effect size."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x,ddof=1) + (ny-1)*np.var(y,ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-10)

def permutation_test(x, y, n_perm=1000, seed=42):
    """Permutation test for mean difference."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    np.random.seed(seed)
    observed = np.mean(x) - np.mean(y)
    combined = np.concatenate([x, y])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:len(x)]) - np.mean(combined[len(x):])
        if abs(perm_diff) >= abs(observed):
            count += 1
    return count / n_perm

def main():
    print("="*60)
    print("05_stats_and_tests.py: Statistical Analysis")
    print("="*60)
    
    # Load data
    df = pd.read_csv("experiments/Experiment 16/data/exp16_metrics.csv")
    metadata = pd.read_csv("experiments/Experiment 16/data/metadata.csv")
    
    # Define groups
    # Merge metadata (drop old correct from metrics if exists)
    if 'correct' in df.columns:
        df = df.drop(columns=['correct'])
        
    df = df.merge(metadata[['problem_id', 'condition', 'correct']], 
                  on=['problem_id', 'condition'], how='left')
    
    df['group'] = 'Unknown'
    df.loc[(df['condition'] == 'direct') & (df['correct'] == False), 'group'] = 'G1'
    df.loc[(df['condition'] == 'direct') & (df['correct'] == True), 'group'] = 'G2'
    df.loc[(df['condition'] == 'cot') & (df['correct'] == False), 'group'] = 'G3'
    df.loc[(df['condition'] == 'cot') & (df['correct'] == True), 'group'] = 'G4'
    
    print(f"\nGroup counts:")
    print(df.groupby('group')['problem_id'].nunique())
    
    # Comparisons (comprehensive pairwise)
    comparisons = [
        # Within-regime (success vs failure)
        ("G4", "G3", "CoT Success vs CoT Failure"),
        ("G2", "G1", "Direct Success vs Direct Failure"),
        
        # Cross-regime (CoT vs Direct, same outcome)
        ("G4", "G2", "CoT Success vs Direct Success"),  # Why does CoT succeed?
        ("G3", "G1", "CoT Failure vs Direct Failure"),  # Why does CoT fail?
        
        # Mixed comparisons
        ("G4", "G1", "CoT Success vs Direct Failure"),
        ("G3", "G2", "CoT Failure vs Direct Success")
    ]
    
    metrics = ['speed', 'dir_consistency', 'effective_dim', 'radius_of_gyration', 'msd_exponent', 'cos_to_late_window']
    
    results = []
    
    for g1, g2, desc in tqdm(comparisons, desc="Comparisons"):
        for metric in metrics:
            for layer in df['layer'].unique():
                if layer == -1:  # Skip cross-layer summary for now
                    continue
                
                data_g1 = df[(df['group'] == g1) & (df['layer'] == layer)][metric].dropna()
                data_g2 = df[(df['group'] == g2) & (df['layer'] == layer)][metric].dropna()
                
                if len(data_g1) < 2 or len(data_g2) < 2:
                    continue
                
                d = cohens_d(data_g1, data_g2)
                p = permutation_test(data_g1.values, data_g2.values, n_perm=1000)
                
                results.append({
                    'comparison': desc,
                    'group1': g1,
                    'group2': g2,
                    'layer': layer,
                    'metric': metric,
                    'cohens_d': d,
                    'perm_p': p,
                    'significant': p < 0.05,
                    'n1': len(data_g1),
                    'n2': len(data_g2)
                })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiments/Experiment 16/data/exp16_comparisons.csv", index=False)
    print(f"\n[OK] Saved {len(df_results)} comparison results")
    
    # Summary stats
    print(f"\nSignificant effects (p<0.05): {df_results['significant'].sum()}/{len(df_results)}")

if __name__ == "__main__":
    main()
