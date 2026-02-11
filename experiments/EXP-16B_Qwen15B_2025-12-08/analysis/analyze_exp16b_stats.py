"""
EXP-16B Group Statistics (Qwen 1.5B)
====================================
Computes mean metrics for each group (G1, G2, G3, G4) at key layers.
Outputs a markdown table.
"""

import pandas as pd
import numpy as np
import os

DATA_FILE = r"experiments/EXP-16B_Qwen15B_2025-12-08/data/exp16b_full_metrics.csv"

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Filter for key layers
    key_layers = [5, 14, 25] # Early, Middle, Late (Total 28 layers)
    df_filtered = df[df['layer'].isin(key_layers)]
    
    metrics = ['speed', 'radius_of_gyration', 'effective_dim', 'gyration_anisotropy']
    
    print("## Group Descriptive Stats (Qwen 1.5B - Preliminary)")
    
    # Calculate means
    grouped = df_filtered.groupby(['layer', 'group'])[metrics].mean().reset_index()
    
    # Pivot for readability
    pivot_tables = {}
    for m in metrics:
        p = grouped.pivot(index='layer', columns='group', values=m)
        pivot_tables[m] = p
    
    # Print Markdown Table
    print("| Metric | Layer | G1 (Dir Fail) | G2 (Dir Succ) | G3 (CoT Fail) | G4 (CoT Succ) |")
    print("|---|---|---|---|---|---|")
    
    for m in metrics:
        if m in pivot_tables:
            pt = pivot_tables[m]
            for layer in key_layers:
                if layer in pt.index:
                    row = pt.loc[layer]
                    g1 = f"{row.get('G1', np.nan):.3f}"
                    g2 = f"{row.get('G2', np.nan):.3f}"
                    g3 = f"{row.get('G3', np.nan):.3f}"
                    g4 = f"{row.get('G4', np.nan):.3f}"
                    print(f"| **{m}** | {layer} | {g1} | {g2} | {g3} | {g4} |")
            print("|---|---|---|---|---|---|")

if __name__ == "__main__":
    main()
