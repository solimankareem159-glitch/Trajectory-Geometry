import pandas as pd
import numpy as np
from scipy import stats
import os
import argparse
import itertools

def get_cohen_d(group1, group2):
    """Compute Cohen's d between two groups."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    if len(g1) < 3 or len(g2) < 3:
        return np.nan, 0, 0
    
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    # d = (mean1 - mean2) / sqrt((var1 + var2)/2)
    pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
    if pooled_std == 0:
        d = 0
    else:
        d = (np.mean(g1) - np.mean(g2)) / pooled_std
        
    return d, p_val, t_stat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", type=str, default="experiments/EXP-19_Robustness_2026-02-14")
    args = parser.parse_args()
    
    data_dir = os.path.join(args.ssd_root, "data")
    metrics_path = os.path.join(data_dir, "all_metrics.csv")
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return
        
    df = pd.read_csv(metrics_path)
    
    # Load all metadata and merge
    model_keys = df['model'].unique()
    all_merged = []
    
    for m_key in model_keys:
        meta_path = os.path.join(data_dir, m_key, "metadata.csv")
        if os.path.exists(meta_path):
            m_meta = pd.read_csv(meta_path)
            m_metrics = df[df['model'] == m_key]
            merged = pd.merge(m_metrics, m_meta[['problem_id', 'condition', 'group', 'bin']], 
                             on=['problem_id', 'condition'])
            
            # Normalize layer depth [0, 1]
            max_l = merged['layer'].max()
            merged['relative_depth'] = merged['layer'] / max_l
            all_merged.append(merged)
            
    full_df = pd.concat(all_merged)
    
    # --- 1. Exhaustive Within-Model Pairwise ---
    pairwise_results = []
    groups = ['G1', 'G2', 'G3', 'G4']
    pairs = list(itertools.combinations(groups, 2))
    
    metric_cols = [c for c in full_df.columns if 'clean_' in c or 'full_' in c]
    
    print("Computing within-model pairwise comparisons...")
    for m_key in model_keys:
        m_df = full_df[full_df['model'] == m_key]
        layers = sorted(m_df['layer'].unique())
        
        for layer in layers:
            l_df = m_df[m_df['layer'] == layer]
            for g_a, g_b in pairs:
                df_a = l_df[l_df['group'] == g_a]
                df_b = l_df[l_df['group'] == g_b]
                
                if len(df_a) > 2 and len(df_b) > 2:
                    for col in metric_cols:
                        d, p, t = get_cohen_d(df_a[col], df_b[col])
                        if not np.isnan(d):
                            pairwise_results.append({
                                'model': m_key,
                                'layer': layer,
                                'group_a': g_a,
                                'group_b': g_b,
                                'metric': col,
                                'cohen_d': d,
                                'p_val': p,
                                'mean_a': np.mean(df_a[col]),
                                'mean_b': np.mean(df_b[col])
                            })
                            
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_path = os.path.join(data_dir, "exhaustive_pairwise.csv")
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"Saved {len(pairwise_df)} comparisons to {pairwise_path}")
    
    # --- 2. Identifying Architecture-Invariant Signatures ---
    # Signature: Metric where G4 vs G1 d > 2.0 in ALL architectures at some depth
    print("Searching for architecture-invariant signatures...")
    invariants = []
    
    # Focus on G4 vs G1 (Success vs Baseline Failure)
    g4g1 = pairwise_df[(pairwise_df['group_a'] == 'G1') & (pairwise_df['group_b'] == 'G4')] 
    # Note: Pairwise order might be G1 vs G4 or G4 vs G1 depending on combinations logic.
    # combinations(['G1','G2','G3','G4']) gives ('G1','G2'), ('G1','G3'), ('G1','G4') ...
    
    for metric in metric_cols:
        m_data = g4g1[g4g1['metric'] == metric]
        
        # Check if high effect exists in all models
        models_hit = m_data[m_data['cohen_d'].abs() > 2.0]['model'].unique()
        if len(models_hit) == len(model_keys):
            # Find the peak effect per model
            peak_effects = []
            for m_key in model_keys:
                peak = m_data[m_data['model'] == m_key].iloc[m_data[m_data['model'] == m_key]['cohen_d'].abs().argmax()]
                peak_effects.append(peak)
            
            invariants.append({
                'metric': metric,
                'avg_abs_cohen_d': np.mean([abs(p['cohen_d']) for p in peak_effects]),
                'min_abs_cohen_d': min([abs(p['cohen_d']) for p in peak_effects])
            })
            
    invariant_df = pd.DataFrame(invariants).sort_values('avg_abs_cohen_d', ascending=False)
    invariant_path = os.path.join(data_dir, "invariant_signatures.csv")
    invariant_df.to_csv(invariant_path, index=False)
    print(f"Detected {len(invariant_df)} architecture-invariant signatures.")
    
    # --- 3. Cross-Model Relative Layer Synchronization ---
    # Correlation of relative depth curves
    # To be implemented if needed for specific metrics.
    
if __name__ == "__main__":
    main()
