"""
Cross-Model Comparison (EXP-14 vs EXP-16B)
==========================================
Compares Qwen 0.5B (EXP-14) and Qwen 1.5B (EXP-16B).
Focuses on CoT Success (G4) vs Direct Fail (G1) scaling trends.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Configuration
EXP14_FILE = r"experiments/EXP-14_UniversalSignature_2025-12-03/data/exp14_metrics.csv"
EXP16B_FILE = r"experiments/EXP-16B_Qwen15B_2025-12-08/data/exp16b_full_metrics.csv"
OUTPUT_DIR = r"experiments/EXP-16B_Qwen15B_2025-12-08/analysis/figures/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(EXP14_FILE):
        print("EXP-14 data missing!")
        return None, None
    if not os.path.exists(EXP16B_FILE):
        print("EXP-16B data missing!")
        return None, None
        
    df14 = pd.read_csv(EXP14_FILE)
    df16 = pd.read_csv(EXP16B_FILE)
    
    df14['Model'] = 'Qwen 0.5B'
    df16['Model'] = 'Qwen 1.5B'
    
    return df14, df16

def plot_scaling_trend(df14, df16, metric, group='G4', layer_ratio=True):
    """
    Plots metric vs Relative Depth for both models.
    """
    cols = ['layer', 'group', metric, 'Model']
    
    sub14 = df14[df14['group'] == group][cols].copy()
    sub16 = df16[df16['group'] == group][cols].copy()
    
    # Normalize layers (0-1)
    sub14['Relative Depth'] = sub14['layer'] / sub14['layer'].max()
    sub16['Relative Depth'] = sub16['layer'] / sub16['layer'].max()
    
    combined = pd.concat([sub14, sub16]).reset_index(drop=True)
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='Relative Depth', y=metric, hue='Model', data=combined, marker='o')
    plt.title(f"{metric} Scaling (Group {group})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"scaling_{group}_{metric}.png"))
    plt.close()
    print(f"Saved scaling_{group}_{metric}.png")

def main():
    df14, df16 = load_data()
    if df14 is None or df16 is None: return

    metrics = ['speed', 'radius_of_gyration', 'effective_dim', 'gyration_anisotropy']
    
    for m in metrics:
        if m in df14.columns and m in df16.columns:
            plot_scaling_trend(df14, df16, m, group='G4') # CoT Success
            plot_scaling_trend(df14, df16, m, group='G1') # Direct Fail
            plot_scaling_trend(df14, df16, m, group='G3') # CoT Fail

if __name__ == "__main__":
    main()
