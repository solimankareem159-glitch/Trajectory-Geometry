
import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(x, y):
    """Computes Cohen's d effect size."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof + 1e-9)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, "results", "exp18_metrics.csv")
    report_path = os.path.join(base_dir, "results", "analysis_report.md")
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    df = pd.read_csv(metrics_path)
    
    # Define groups
    groups = {
        'G1': df[df['group'] == 'G1'],
        'G2': df[df['group'] == 'G2'],
        'G3': df[df['group'] == 'G3'],
        'G4': df[df['group'] == 'G4']
    }
    
    # Metrics to analyze (exclude metadata)
    metadata_cols = ['problem_id', 'condition', 'group', 'correct', 'filename', 'n_layers', 'n_tokens', 'hidden_dim', 'layer', 'interlayer_alignment']
    metric_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Comparisons to run
    comparisons = [('G1', 'G2'), ('G3', 'G4'), ('G1', 'G4'), ('G2', 'G4')]
    
    results = []
    
    # 1. Pairwise Comparisons across all layers
    print("Running pairwise comparisons...")
    for g1_name, g2_name in comparisons:
        g1_full = groups[g1_name]
        g2_full = groups[g2_name]
        
        layers = sorted(df['layer'].unique())
        for layer in layers:
            g1 = g1_full[g1_full['layer'] == layer]
            g2 = g2_full[g2_full['layer'] == layer]
            
            if len(g1) < 2 or len(g2) < 2: continue
            
            for m in metric_cols:
                v1 = g1[m].dropna()
                v2 = g2[m].dropna()
                
                if len(v1) < 2 or len(v2) < 2: continue
                
                # Statistics
                t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
                d = cohen_d(v1, v2)
                
                results.append({
                    'comparison': f"{g1_name} vs {g2_name}",
                    'layer': layer,
                    'metric': m,
                    'mean_g1': np.mean(v1),
                    'mean_g2': np.mean(v2),
                    'cohen_d': d,
                    'p_value': p_val
                })
                
    res_df = pd.DataFrame(results)
    
    # Identify top discriminators
    print("Identifying top discriminators...")
    top_results = res_df.sort_values(by='cohen_d', key=abs, ascending=False).head(50)
    
    # 2. Write Report
    with open(report_path, "w") as f:
        f.write("# Experiment 18: Statistical Analysis Report\n\n")
        f.write("## 1. Overview\n")
        f.write(f"Analyzed {len(metric_cols)} metrics across {len(layers)} layers for groups G1-G4.\n\n")
        
        f.write("## 2. Top Metric-Layer Discriminators (by Cohen's d)\n\n")
        f.write("| Comparison | Layer | Metric | Cohen's d | p-value | Mean G1 | Mean G2 |\n")
        f.write("|:---|:---|:---|:---|:---|:---|:---|\n")
        for idx, row in top_results.head(20).iterrows():
            f.write(f"| {row['comparison']} | {row['layer']} | {row['metric']} | {row['cohen_d']:.3f} | {row['p_value']:.2e} | {row['mean_g1']:.3f} | {row['mean_g2']:.3f} |\n")
            
        f.write("\n## 3. Layer-Wise Summary: Best Separation Layers\n")
        f.write("Across all comparisons, which layers show the highest mean absolute effect size?\n\n")
        layer_means = res_df.groupby('layer')['cohen_d'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
        f.write("| Layer | Avg |abs(Cohen's d)| |\n")
        f.write("|:---|:---|\n")
        for layer, val in layer_means.head(10).items():
            f.write(f"| {layer} | {val:.3f} |\n")
            
        f.write("\n## 4. Specific Regime Insights\n")
        
        # CoT Success (G4) vs CoT Failure (G3)
        f.write("\n### CoT Success (G4) vs CoT Failure (G3)\n")
        cot_comp = res_df[res_df['comparison'] == "G3 vs G4"].sort_values(by='cohen_d', key=abs, ascending=False)
        f.write("| Layer | Metric | Cohen's d | Description |\n")
        f.write("|:---|:---|:---|:---|\n")
        for idx, row in cot_comp.head(10).iterrows():
            f.write(f"| {row['layer']} | {row['metric']} | {row['cohen_d']:.3f} | |\n")

        # Direct vs CoT
        f.write("\n### Direct Success (G2) vs CoT Success (G4)\n")
        direct_cot = res_df[res_df['comparison'] == "G2 vs G4"].sort_values(by='cohen_d', key=abs, ascending=False)
        f.write("| Layer | Metric | Cohen's d | Description |\n")
        f.write("|:---|:---|:---|:---|\n")
        for idx, row in direct_cot.head(10).iterrows():
            f.write(f"| {row['layer']} | {row['metric']} | {row['cohen_d']:.3f} | |\n")

    print(f"Analysis saved to {report_path}")

if __name__ == "__main__":
    main()
