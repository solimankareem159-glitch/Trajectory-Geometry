
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP18B_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-18B_ScalingGeometry_2026-02-13")
RESULTS_DIR = os.path.join(EXP18B_DIR, "results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

MODELS = ["pythia_70m", "qwen_0_5b", "qwen_1_5b"]

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def analyze_model(model_name):
    csv_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics_57.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {model_name}: {csv_path} not found.")
        return

    print(f"Analyzing {model_name}...")
    df = pd.read_csv(csv_path)
    
    # Identify Groups
    # G1: Direct Failure
    # G2: Direct Success
    # G3: CoT Failure
    # G4: CoT Success
    
    # Logic if 'group' column is missing or needs inference
    # Default to inferring if not present
    if 'group' not in df.columns:
        conditions = []
        for _, row in df.iterrows():
            c = row['condition']
            corr = row['correct']
            if c == 'direct' and not corr: conditions.append('G1')
            elif c == 'direct' and corr: conditions.append('G2')
            elif c == 'cot' and not corr: conditions.append('G3')
            elif c == 'cot' and corr: conditions.append('G4')
            else: conditions.append('unknown')
        df['group'] = conditions

    # Metrics to analyze (all numeric columns except metadata)
    metadata_cols = ['problem_id', 'condition', 'correct', 'group', 'model', 'filename', 'layer', 'n_layers', 'n_tokens', 'hidden_dim']
    metric_cols = [c for c in df.columns if c not in metadata_cols and np.issubdtype(df[c].dtype, np.number)]
    
    report_lines = [f"# Statistical Analysis: {model_name}", ""]
    summary_stats = []
    
    # Pairwise Comparisons
    comparisons = [('G1', 'G4'), ('G2', 'G4'), ('G3', 'G4')]
    
    for gA, gB in comparisons:
        report_lines.append(f"\n## Comparison: {gA} vs {gB}")
        
        # Filter
        dfA = df[df['group'] == gA]
        dfB = df[df['group'] == gB]
        
        if dfA.empty or dfB.empty:
            report_lines.append(f"  Insufficient data: {gA}={len(dfA)}, {gB}={len(dfB)}")
            continue
            
        # Analyze per layer
        layers = sorted(df['layer'].unique())
        
        best_d_abs = 0
        best_stats = None
        
        pwal = ""
        pwal += "| Metric | Layer | Cohen's d | p-value | Mean A | Mean B |\n"
        pwal += "|---|---|---|---|---|---|\n"
        
        has_significant = False
        
        for m in metric_cols:
            for l in layers:
                vA = dfA[dfA['layer']==l][m].dropna()
                vB = dfB[dfB['layer']==l][m].dropna()
                
                if len(vA) < 3 or len(vB) < 3: continue
                
                d = cohens_d(vA, vB)
                t, p = ttest_ind(vA, vB, equal_var=False)
                
                # Check for significant large effect
                if abs(d) > 0.8 and p < 0.01:
                    has_significant = True
                    pwal += f"| {m} | {l} | {d:.3f} | {p:.3e} | {np.mean(vA):.3f} | {np.mean(vB):.3f} |\n"
                    
                    summary_stats.append({
                        'model': model_name,
                        'comparison': f"{gA}_vs_{gB}",
                        'metric': m,
                        'layer': l,
                        'd': d,
                        'p': p,
                        'mean_A': np.mean(vA),
                        'mean_B': np.mean(vB)
                    })
                    
                if abs(d) > best_d_abs:
                    best_d_abs = abs(d)
                    best_stats = (m, l, d)
        
        if has_significant:
            report_lines.append(pwal)
        else:
            report_lines.append("No large significant differences found (d > 0.8, p < 0.01).")

        if best_stats:
            report_lines.append(f"\n**Largest Effect**: {best_stats[0]} at Layer {best_stats[1]} (d={best_stats[2]:.3f})")
        report_lines.append("\n" + "-"*40 + "\n")

    # Save Report
    with open(os.path.join(REPORTS_DIR, f"{model_name}_stats.md"), "w") as f:
        f.write("\n".join(report_lines))
        
    # Save Summary CSV
    if summary_stats:
        pd.DataFrame(summary_stats).to_csv(os.path.join(REPORTS_DIR, f"{model_name}_summary_stats.csv"), index=False)
        print(f"Saved stats summary to {REPORTS_DIR}/{model_name}_summary_stats.csv")

def main():
    print("Starting Intra-Model Analysis...")
    for m in MODELS:
        analyze_model(m)
    print("Done.")

if __name__ == "__main__":
    main()
