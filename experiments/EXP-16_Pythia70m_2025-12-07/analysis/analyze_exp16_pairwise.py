"""
EXP-16 Pairwise Group Analysis
==============================
Analyzes Qwen 1.5B metrics to compare internal groups:
- G1: Direct Fail
- G2: Direct Success
- G4: CoT Success
- G3: CoT Fail

Generates plots for key metrics to identify patterns like:
- Wandering (High Radius, Low Consistency)
- Stable but Wrong (High Stability, Wrong Answer)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_FILE = r"experiments/EXP-16_Pythia70m_2025-12-07/data/exp16_full_metrics.csv"
OUTPUT_DIR = r"experiments/EXP-16_Pythia70m_2025-12-07/analysis/pairwise_plots"

METRICS = [
    'speed', 'radius_of_gyration', 'gyration_anisotropy', 'effective_dim',
    'msd_exponent', 'cos_to_late_window', 'dir_consistency', 
    'determinism', 'laminarity' # Recurrence metrics if available
]

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")
    return pd.read_csv(DATA_FILE)

def plot_metric(df, metric):
    plt.figure(figsize=(10, 6))
    
    # Filter out cross-layer rows (layer=-1)
    df_layer = df[df['layer'] != -1]
    
    # Plot mean with confidence interval
    sns.lineplot(data=df_layer, x='layer', y=metric, hue='group', style='condition', 
                 markers=True, dashes=False, palette='tab10')
    
    plt.title(f"{metric} by Group (Qwen 1.5B)")
    plt.ylabel(metric)
    plt.xlabel("Layer")
    plt.legend(title='Group')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, f"pairwise_{metric}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def main():
    print(f"PID: {os.getpid()}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        df = load_data()
        print(f"Loaded {len(df)} rows. Groups: {df['group'].unique()}")
        
        # Check standard metrics
        for m in METRICS:
            if m in df.columns:
                plot_metric(df, m)
            else:
                print(f"Skipping {m} (not in columns)")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
