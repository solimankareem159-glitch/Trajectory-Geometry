
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP18B_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-18B_ScalingGeometry_2026-02-13")
RESULTS_DIR = os.path.join(EXP18B_DIR, "results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = [
    {"name": "pythia_70m", "label": "Pythia 70M", "color": "blue"},
    {"name": "qwen_0_5b", "label": "Qwen 0.5B", "color": "orange"},
    {"name": "qwen_1_5b", "label": "Qwen 1.5B", "color": "green"}
]

def load_all_data():
    dfs = []
    print("Loading data...")
    for m in MODELS:
        path = os.path.join(RESULTS_DIR, f"{m['name']}_metrics_57.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['Model'] = m['label']
                df['ModelOrder'] = MODELS.index(m)
                
                # Normalize Layer Depth (0.0 to 1.0)
                if 'n_layers' in df.columns:
                    df['RelativeDepth'] = df['layer'] / (df['n_layers'] - 1)
                else:
                    max_l = df['layer'].max()
                    df['RelativeDepth'] = df['layer'] / max_l if max_l > 0 else 0
                
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {m['name']}: {e}")
        else:
            print(f"Warning: Missing data for {m['name']} at {path}")
    
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def plot_metric_scaling(df, metric, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Filter for Correct (G4/G2 combined or just CoT?)
    # Let's use Correct=True to compare capabilities
    subset = df[df['correct'] == True]
    
    if subset.empty:
        print(f"No correct data for plotting {metric}")
        return

    try:
        sns.lineplot(data=subset, x='RelativeDepth', y=metric, hue='Model', 
                     palette={m['label']: m['color'] for m in MODELS if m['label'] in df['Model'].unique()})
        
        plt.title(f"{title} (Correct Trajectories)")
        plt.xlabel("Relative Layer Depth")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot {output_path}")
    except Exception as e:
        print(f"Error plotting {metric}: {e}")

def main():
    print("Starting Cross-Model Analysis...")
    df = load_all_data()
    if df.empty:
        print("No data found.")
        return

    # Metrics to compare
    metrics_of_interest = [
        ('mean_attractor_dist', 'Attractor Distance'),
        ('info_gain_proxy', 'Information Gain'),
        ('radius_of_gyration', 'Radius of Gyration'),
        ('trajectory_curvature', 'Trajectory Curvature (Exp 15)'),
        ('commitment_sharpness', 'Commitment Sharpness (Exp 15)'),
        ('next_layer_corr', 'Next Layer Correlation'),
        ('effective_dimension', 'Effective Dimension')
    ]
    
    report_lines = ["# Cross-Model Scaling Analysis", ""]

    for m_col, m_name in metrics_of_interest:
        if m_col not in df.columns: 
            print(f"Metric {m_col} not found in dataframe columns.")
            continue
        
        print(f"Analyzing {m_name}...")
        fname = f"scaling_{m_col}.png"
        plot_metric_scaling(df, m_col, m_name, fname)
        
        report_lines.append(f"## {m_name}")
        report_lines.append(f"![{m_name}](../figures/{fname})")
        
        # Calculate Peak Value per model
        report_lines.append("| Model | Peak Value | Mean Value |")
        report_lines.append("|---|---|---|")
        for mod in MODELS:
            name = mod['label']
            sub = df[(df['Model'] == name) & (df['correct'] == True)]
            if not sub.empty:
                peak = sub[m_col].max()
                mean = sub[m_col].mean()
                report_lines.append(f"| {name} | {peak:.4f} | {mean:.4f} |")
        report_lines.append("")

    with open(os.path.join(REPORTS_DIR, "cross_model_scaling.md"), "w") as f:
        f.write("\n".join(report_lines))
    print("Done.")

if __name__ == "__main__":
    main()
