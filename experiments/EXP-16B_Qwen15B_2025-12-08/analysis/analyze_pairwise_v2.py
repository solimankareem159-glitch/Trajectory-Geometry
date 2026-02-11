"""
EXP-16B Pairwise Analysis (Qwen 1.5B)
=====================================
Generates pairwise comparison plots for the 4 groups:
- G1: Direct Fail (Wandering?)
- G2: Direct Success (Rare?)
- G3: CoT Fail (Tunnel Vision?)
- G4: CoT Success

Comparing across layers and metrics.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Configuration
DATA_DIR = r"experiments/EXP-16B_Qwen15B_2025-12-08/data"
INPUT_FILE = os.path.join(DATA_DIR, "exp16b_full_metrics.csv")
OUTPUT_DIR = r"experiments/EXP-16B_Qwen15B_2025-12-08/analysis/figures/pairwise"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Waiting for {INPUT_FILE}...")
        return None
    return pd.read_csv(INPUT_FILE)

def plot_pairwise_violin(df, metric, layers=[5, 14, 25]):
    """
    Plots violin plots for a specific metric across selected layers.
    Groups: G1, G2, G3, G4.
    """
    subset = df[df['layer'].isin(layers)]
    if subset.empty: return

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='layer', y=metric, hue='group', data=subset, 
                   hue_order=['G1', 'G2', 'G3', 'G4'],
                   palette={'G1': '#e74c3c', 'G2': '#2ecc71', 'G3': '#e67e22', 'G4': '#3498db'},
                   inner='quartile')
    
    plt.title(f"EXP-16B (Qwen 1.5B) - {metric} Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"pairwise_{metric}.png"))
    plt.close()
    print(f"Saved pairwise_{metric}.png")

def main():
    df = load_data()
    if df is None: return

    metrics = [
        'speed', 'radius_of_gyration', 'effective_dim', 
        'gyration_anisotropy', 'msd_exponent', 'cos_to_late_window',
        'time_to_commit'
    ]

    for m in metrics:
        if m in df.columns:
            plot_pairwise_violin(df, m)
        else:
            print(f"Metric {m} not found in columns.")

if __name__ == "__main__":
    main()
