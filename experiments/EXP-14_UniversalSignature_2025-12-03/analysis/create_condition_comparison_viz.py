"""
Create comprehensive visualization showing clear distinction between
the four conditions (G1, G2, G3, G4) across different layers using multiple metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuration
DATA_DIR = r"..\data"
METRICS_FILE = "exp14_metrics.csv"
OUTPUT_DIR = r"..\figures"
OUTPUT_FILE = "condition_comparison_by_layer.png"

# Key metrics that should show clear separation
KEY_METRICS = [
    'speed',
    'dir_consistency', 
    'stabilization',
    'tortuosity',
    'effective_dim',
    'msd_exponent'
]

# Readable labels
METRIC_LABELS = {
    'speed': 'Speed',
    'dir_consistency': 'Directional Consistency',
    'stabilization': 'Stabilization Rate',
    'tortuosity': 'Tortuosity',
    'effective_dim': 'Effective Dimension',
    'msd_exponent': 'MSD Exponent'
}

GROUP_LABELS = {
    'G1': 'Direct Failure',
    'G2': 'Direct Success',
    'G3': 'CoT Failure',
    'G4': 'CoT Success'
}

# Color scheme for groups
COLORS = {
    'G1': '#e74c3c',  # Red - Direct Failure
    'G2': '#3498db',  # Blue - Direct Success
    'G3': '#f39c12',  # Orange - CoT Failure
    'G4': '#2ecc71'   # Green - CoT Success
}

# Focus on key layers
KEY_LAYERS = [0, 5, 10, 15, 20, 24]


def load_data():
    """Load metrics data."""
    print(f"Loading data from {METRICS_FILE}...")
    df = pd.read_csv(os.path.join(DATA_DIR, METRICS_FILE))
    print(f"Loaded {len(df)} rows")
    print(f"Unique layers: {sorted(df['layer'].unique())}")
    print(f"Unique groups: {sorted(df['group'].unique())}")
    return df


def create_visualization(df):
    """Create multi-panel visualization."""
    
    # Filter to key layers only
    df_filtered = df[df['layer'].isin(KEY_LAYERS)].copy()
    
    # Set up the figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(KEY_METRICS):
        ax = axes[idx]
        
        # Prepare data for this metric across layers
        plot_data = []
        for layer in KEY_LAYERS:
            layer_data = df_filtered[df_filtered['layer'] == layer]
            for group in ['G1', 'G2', 'G3', 'G4']:
                group_data = layer_data[layer_data['group'] == group]
                values = group_data[metric].dropna().values
                
                # Add data points
                for val in values:
                    plot_data.append({
                        'Layer': layer,
                        'Group': group,
                        'Value': val
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create line plot with error bands
        for group in ['G1', 'G2', 'G3', 'G4']:
            group_data = plot_df[plot_df['Group'] == group]
            
            # Calculate mean and std for each layer
            stats = group_data.groupby('Layer')['Value'].agg(['mean', 'std', 'sem'])
            
            # Plot mean line
            ax.plot(stats.index, stats['mean'], 
                   marker='o', 
                   linewidth=2.5,
                   markersize=8,
                   label=GROUP_LABELS[group],
                   color=COLORS[group],
                   alpha=0.9)
            
            # Add confidence interval
            ax.fill_between(stats.index,
                           stats['mean'] - 1.96 * stats['sem'],
                           stats['mean'] + 1.96 * stats['sem'],
                           alpha=0.15,
                           color=COLORS[group])
        
        # Styling
        ax.set_xlabel('Layer', fontsize=11, fontweight='bold')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11, fontweight='bold')
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(KEY_LAYERS)
        
        # Legend only on first plot
        if idx == 0:
            ax.legend(loc='best', framealpha=0.9, fontsize=9)
    
    # Overall title
    fig.suptitle('Trajectory Metrics Across Layers: Condition Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    
    return output_path


def print_summary_statistics(df):
    """Print summary statistics showing separation between groups."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: Group Separation Across Layers")
    print("="*80)
    
    for layer in KEY_LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_data = df[df['layer'] == layer]
        
        for metric in KEY_METRICS:
            print(f"\n  {METRIC_LABELS[metric]}:")
            for group in ['G1', 'G2', 'G3', 'G4']:
                group_data = layer_data[layer_data['group'] == group]
                values = group_data[metric].dropna()
                if len(values) > 0:
                    print(f"    {GROUP_LABELS[group]:20s}: mean={values.mean():.4f}, std={values.std():.4f}, n={len(values)}")


def main():
    print("="*80)
    print("CREATING CONDITION COMPARISON VISUALIZATION")
    print("="*80)
    print(f"PID: {os.getpid()}", flush=True)
    
    # Load data
    df = load_data()
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create visualization
    output_path = create_visualization(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nVisualization saved to:")
    print(f"  {os.path.abspath(output_path)}")
    

if __name__ == "__main__":
    main()
