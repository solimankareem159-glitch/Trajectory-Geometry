"""
Compare Models: EXP-14 (0.5B) vs EXP-16 (1.5B)
==============================================
Compares trajectory metrics across model scales.
Normalizes depth to [0, 1] for fair comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_DIR_14 = r"experiments/EXP-14_UniversalSignature_2025-12-03/data"
DATA_DIR_16 = r"experiments/EXP-16_Pythia70m_2025-12-07/data"
OUTPUT_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/analysis/comparison_plots"

METRICS_TO_PLOT = [
    'speed', 'radius_of_gyration', 'gyration_anisotropy', 'effective_dim', 
    'msd_exponent', 'cos_to_late_window'
]

def load_data():
    print("Loading data...")
    df14 = pd.read_csv(os.path.join(DATA_DIR_14, "exp14_metrics.csv"))
    df16 = pd.read_csv(os.path.join(DATA_DIR_16, "exp16_full_metrics.csv"))
    
    df14['model'] = 'Qwen-0.5B'
    df16['model'] = 'Qwen-1.5B'
    
    # Filter for G4 (CoT Success)
    df14 = df14[(df14['group'] == 'G4') & (df14['layer'] != -1)]
    df16 = df16[(df16['group'] == 'G4') & (df16['layer'] != -1)]
    
    # Normalize Depth
    df14['depth'] = df14['layer'] / df14['layer'].max()
    df16['depth'] = df16['layer'] / df16['layer'].max()
    
    return pd.concat([df14, df16])

def plot_comparison(df):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    sns.set_theme(style="whitegrid")
    
    for metric in METRICS_TO_PLOT:
        if metric not in df.columns: continue
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='depth', y=metric, hue='model', marker='o')
        plt.title(f"{metric} Comparison (G4 Group)")
        plt.xlabel("Normalized Depth (0=Embed, 1=Output)")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"compare_{metric}.png"))
        plt.close()
        print(f"Saved {metric} plot")

def main():
    try:
        df = load_data()
        print(f"Loaded {len(df)} rows.")
        plot_comparison(df)
        print("Comparison complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
