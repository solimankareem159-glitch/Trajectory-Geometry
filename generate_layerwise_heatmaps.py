
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Style Configuration ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid", {"grid.color": ".9", "axes.edgecolor": ".3"})

# Defined palette (for consistency with line plots if needed)
COLORS = {
    'G1': '#D32F2F', # Red
    'G2': '#00796B', # Teal
    'G3': '#FBC02D', # Gold
    'G4': '#303F9F', # Indigo
}

# --- Data Paths ---
DATA_PATHS = {
    'Qwen-0.5B (EXP-14)': r"experiments/EXP-14_UniversalSignature_2025-12-03/data/exp14_metrics_full.csv",
    'Qwen-1.5B (EXP-16B)': r"experiments/EXP-16B_Qwen15B_2025-12-08/data/exp16b_metrics_clean.csv",
    'Pythia-70m (EXP-16)': r"experiments/EXP-16_Pythia70m_2025-12-07/data/exp16_metrics_salvaged.csv" 
}

OUTPUT_DIR = r"research_history/figures/layerwise_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(name, path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping {name}.")
        return None
    
    print(f"Loading {name} from {path}...")
    df = pd.read_csv(path)
    df['Model'] = name
    
    # Derive Group if missing
    if 'group' not in df.columns and 'condition' in df.columns and 'correct' in df.columns:
        print(f"Deriving 'group' for {name}...")
        def get_group(row):
            cond = row['condition'].lower() # 'cot' or 'direct'
            corr = bool(row['correct']) # True or False
            if 'direct' in cond:
                return 'G2' if corr else 'G1'
            elif 'cot' in cond:
                return 'G4' if corr else 'G3'
            return None
        df['group'] = df.apply(get_group, axis=1)
    
    return df

def plot_heatmap(df, metric, model_name):
    """
    Plots a Heatmap: X=Layer, Y=Group, Color=Metric Value (Mean)
    """
    # 1. Pivot Data: Index=Group, Columns=Layer, Values=Metric
    pivot = df.groupby(['group', 'layer'])[metric].mean().unstack()
    
    if pivot.empty: return

    # Reorder Rows: G1, G3, G2, G4 (Fail -> Success)
    order = ['G1', 'G3', 'G2', 'G4']
    pivot = pivot.reindex(order)
    
    if pivot.isnull().all().all(): return

    plt.figure(figsize=(10, 5))
    
    # Diverging colormap? Or sequential? Depends on metric.
    # Radius/Dim usually > 0. Sequential (Viridis or Magma) is good.
    # Drift can be negative? No.
    # Cosine is -1 to 1.
    
    cmap = 'viridis'
    if 'cosine' in metric: cmap = 'coolwarm'
    
    sns.heatmap(pivot, cmap=cmap, annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': metric})
    
    plt.title(f"{model_name}: {metric} by Layer", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Layer Index", fontsize=12, fontweight='bold')
    plt.ylabel("Group", fontsize=12, fontweight='bold')
    plt.yticks(rotation=0)
    
    fname = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{metric}_heatmap.png"
    save_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def main():
    print("Generating Layerwise Heatmaps...")
    
    metrics = ['radius_of_gyration', 'effective_dim', 'cosine_sim_to_final']
    
    for name, path in DATA_PATHS.items():
        df = load_data(name, path)
        if df is None: continue
        
        available_metrics = [m for m in metrics if m in df.columns]
        
        for m in available_metrics:
            plot_heatmap(df, m, name)

if __name__ == "__main__":
    main()
