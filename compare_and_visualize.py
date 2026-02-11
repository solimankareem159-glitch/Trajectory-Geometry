
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Style Configuration for Publication ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid", {"grid.color": ".85", "axes.edgecolor": ".3"})

# Defined palette
COLORS = {
    'G1': '#D32F2F', # Red (Fail)
    'G2': '#00796B', # Teal (Direct Success)
    'G3': '#FBC02D', # Gold (CoT Fail)
    'G4': '#303F9F', # Indigo (CoT Success)
}

# --- Data Paths ---
DATA_PATHS = {
    'Qwen-0.5B (EXP-14)': r"experiments/EXP-14_UniversalSignature_2025-12-03/data/exp14_metrics_full.csv",
    'Qwen-1.5B (EXP-16B)': r"experiments/EXP-16B_Qwen15B_2025-12-08/data/exp16b_metrics_clean.csv",
    'Pythia-70m (EXP-16)': r"experiments/EXP-16_Pythia70m_2025-12-07/data/exp16_metrics_salvaged.csv"  # Using salvaged/clean version if possible
}

OUTPUT_DIR = r"research_history/figures/layerwise_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(name, path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping {name}.")
        return None
    
    print(f"Loading {name} from {path}...")
    df = pd.read_csv(path)
    df['Model'] = name
    
    # Derive Group if missing (e.g., Pythia)
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

def plot_layerwise_metric(df, metric, model_name, truncated=False):
    """
    Plots a metric vs Layer Index, Hue by Group.
    """
    plt.figure(figsize=(8, 5))
    
    # Filter for main groups
    subset = df[df['group'].isin(COLORS.keys())]
    
    sns.lineplot(data=subset, x='layer', y=metric, hue='group', palette=COLORS, linewidth=2.5, errorbar='ci')
    
    title = f"{model_name}: {metric}"
    if truncated:
        title += " (Truncated 32-tok)"
    else:
        title += " (Full Context)"
        
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Layer Index", fontsize=12, fontweight='bold')
    plt.ylabel(metric, fontsize=12, fontweight='bold')
    plt.legend(title='Group', loc='best')
    
    # Normalize X-axis to 0-1 for cross-model comparison? No, layer index is fine for now.
    
    fname = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{metric}.png"
    save_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def main():
    print("Starting Comparative Analysis...")
    
    # Metrics to visualize
    metrics = ['radius_of_gyration', 'effective_dim', 'drift_from_embed', 'cosine_sim_to_final']
    
    for name, path in DATA_PATHS.items():
        df = load_data(name, path)
        if df is None: continue
        
        # Check if metric exists
        available_metrics = [m for m in metrics if m in df.columns]
        
        for m in available_metrics:
            is_truncated = "Pythia" in name # Pythia is known truncated
            plot_layerwise_metric(df, m, name, truncated=is_truncated)
            
    print("Comparison complete.")

if __name__ == "__main__":
    main()
