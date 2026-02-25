import pandas as pd
import numpy as np
from scipy import stats
import os
import argparse

def run_comparisons(metrics_df, metadata_df):
    """
    Perform the primary statistical comparison: CoT Success (G4) vs Direct Failure (G1).
    """
    # Merge metrics with metadata to get labels (correctness, etc.)
    # Note: metrics_df already has problem_id, condition, model, layer
    # We need to join with metadata_df to get 'group' (G1, G2, G3, G4)
    
    # But wait, group can vary by model. Metadata is per model.
    # Let's merge iteratively per model.
    
    results = []
    
    model_keys = metrics_df['model'].unique()
    for m_key in model_keys:
        m_metrics = metrics_df[metrics_df['model'] == m_key]
        
        # Load metadata for this model to get groups
        # Assuming metadata is available in the ssd_root/data/model_key
        # We'll pass metadata_df as a dict of dfs or similar
        m_meta = metadata_df[m_key]
        
        # Merge
        merged = pd.merge(m_metrics, m_meta[['problem_id', 'condition', 'group', 'bin']], 
                         on=['problem_id', 'condition'])
        
        layers = sorted(merged['layer'].unique())
        metric_cols = [c for c in merged.columns if 'clean_' in c or 'full_' in c]
        
        for layer in layers:
            l_df = merged[merged['layer'] == layer]
            
            # G4 vs G1 Comparison
            g4 = l_df[l_df['group'] == 'G4']
            g1 = l_df[l_df['group'] == 'G1']
            
            if len(g4) > 3 and len(g1) > 3:
                for col in metric_cols:
                    g4_vals = g4[col].dropna()
                    g1_vals = g1[col].dropna()
                    
                    if len(g4_vals) > 3 and len(g1_vals) > 3:
                        t_stat, p_val = stats.ttest_ind(g4_vals, g1_vals, equal_var=False)
                        cohen_d = (np.mean(g4_vals) - np.mean(g1_vals)) / np.sqrt((np.var(g4_vals) + np.var(g1_vals)) / 2)
                        
                        results.append({
                            'model': m_key,
                            'layer': layer,
                            'metric': col,
                            'g4_mean': np.mean(g4_vals),
                            'g1_mean': np.mean(g1_vals),
                            't_stat': t_stat,
                            'p_val': p_val,
                            'cohen_d': cohen_d,
                            'n_g4': len(g4_vals),
                            'n_g1': len(g1_vals)
                        })
                        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", type=str, default="experiments/EXP-19_Robustness_2026-02-14")
    args = parser.parse_args()
    
    data_dir = os.path.join(args.ssd_root, "data")
    metrics_path = os.path.join(data_dir, "all_metrics.csv")
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return
        
    metrics_df = pd.read_csv(metrics_path)
    
    # Load all metadata
    metadata_dfs = {}
    model_keys = metrics_df['model'].unique()
    for m_key in model_keys:
        meta_path = os.path.join(data_dir, m_key, "metadata.csv")
        if os.path.exists(meta_path):
            metadata_dfs[m_key] = pd.read_csv(meta_path)
            
    print("Running group comparisons (G4 vs G1)...")
    comparison_results = run_comparisons(metrics_df, metadata_dfs)
    
    if not comparison_results.empty:
        output_path = os.path.join(data_dir, "statistical_comparisons.csv")
        comparison_results.to_csv(output_path, index=False)
        print(f"Saved comparisons to {output_path}")
        
        # Print highlights (top Cohen's d per model)
        print("\n--- TOPS BY EFFECT SIZE (G4 vs G1) ---")
        for m_key in comparison_results['model'].unique():
            m_res = comparison_results[comparison_results['model'] == m_key]
            top = m_res.iloc[m_res['cohen_d'].abs().argsort()[-5:][::-1]]
            print(f"\nModel: {m_key}")
            print(top[['layer', 'metric', 'cohen_d', 'p_val']])
    else:
        print("No significant comparisons possible.")

if __name__ == "__main__":
    main()
