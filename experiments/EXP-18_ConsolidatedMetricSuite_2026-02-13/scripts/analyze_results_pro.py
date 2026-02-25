
import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof + 1e-9)

def generate_comparison_report(df, g1_name, g2_name, metric_cols, output_dir):
    layers = sorted(df['layer'].unique())
    results = []
    
    g1_full = df[df['group'] == g1_name]
    g2_full = df[df['group'] == g2_name]
    
    for layer in layers:
        g1 = g1_full[g1_full['layer'] == layer]
        g2 = g2_full[g2_full['layer'] == layer]
        for m in metric_cols:
            v1, v2 = g1[m].dropna(), g2[m].dropna()
            if len(v1) < 2 or len(v2) < 2: continue
            t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            d = cohen_d(v1, v2)
            results.append({
                'layer': layer, 'metric': m, 'mean_g1': np.mean(v1), 'mean_g2': np.mean(v2), 
                'cohen_d': d, 'p_value': p_val
            })
            
    res_df = pd.DataFrame(results)
    if res_df.empty: return
    
    filename = f"comparison_{g1_name}_{g2_name}.md"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(f"# Statistical Analysis: {g1_name} vs {g2_name}\n\n")
        f.write("## Top 20 Discriminators\n\n")
        f.write("| Layer | Metric | Cohen's d | p-value | Mean {g1_name} | Mean {g2_name} |\n")
        f.write("|:---|:---|:---|:---|:---|:---|\n")
        top = res_df.sort_values(by='cohen_d', key=abs, ascending=False).head(20)
        for _, row in top.iterrows():
            f.write(f"| {row['layer']} | {row['metric']} | {row['cohen_d']:.3f} | {row['p_value']:.2e} | {row['mean_g1']:.3f} | {row['mean_g2']:.3f} |\n")
            
        f.write("\n## Layer-wise Effect Size (Mean Absolute d)\n\n")
        l_summary = res_df.groupby('layer')['cohen_d'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
        for l, val in l_summary.head(5).items():
            f.write(f"- Layer {l}: {val:.3f}\n")
            
    print(f"Generated {filename}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, "results", "exp18_metrics_full.csv")
    output_dir = os.path.join(base_dir, "results")
    
    if not os.path.exists(metrics_path):
        print("Waiting for full metrics CSV...")
        return

    df = pd.read_csv(metrics_path)
    metadata_cols = ['problem_id', 'condition', 'group', 'correct', 'filename', 'n_layers', 'n_tokens', 'hidden_dim', 'layer']
    metric_cols = [c for c in df.columns if c not in metadata_cols]
    
    comparisons = [('G1', 'G2'), ('G3', 'G4'), ('G1', 'G4'), ('G2', 'G4')]
    for g1, g2 in comparisons:
        generate_comparison_report(df, g1, g2, metric_cols, output_dir)

if __name__ == "__main__":
    main()
