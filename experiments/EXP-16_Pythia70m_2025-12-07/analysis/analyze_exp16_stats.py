"""
EXP-16 Group Statistics
=======================
Computes mean metrics for each group (G1, G2, G3, G4) at key layers.
Outputs a markdown table.
"""

import pandas as pd
import numpy as np
import os

DATA_FILE = r"experiments/EXP-16_Pythia70m_2025-12-07/data/exp16_full_metrics.csv"

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found")
        return
        
    df = pd.read_csv(DATA_FILE)
    
    # Filter groups
    groups = ['G1', 'G2', 'G3', 'G4']
    metrics = ['speed', 'radius_of_gyration', 'effective_dim', 'gyration_anisotropy', 'msd_exponent']
    
    # Key layers: Early, Mid, Late
    # Qwen 1.5B has 28 layers.
    layers = [5, 14, 25] # approx 20%, 50%, 90%
    
    print("## Group Descriptive Stats (Qwen 1.5B)")
    print("| Metric | Layer | G1 (Dir Fail) | G2 (Dir Succ) | G3 (CoT Fail) | G4 (CoT Succ) |")
    print("|---|---|---|---|---|---|")
    
    for m in metrics:
        if m not in df.columns: continue
        for l in layers:
            row = f"| **{m}** | {l} |"
            for g in groups:
                subset = df[(df['group'] == g) & (df['layer'] == l)]
                val = subset[m].mean()
                row += f" {val:.3f} |"
            print(row)
        print("|---|---|---|---|---|---|")

if __name__ == "__main__":
    main()
