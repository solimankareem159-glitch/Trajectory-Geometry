
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set high-quality aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("bright")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'savefig.dpi': 300
})

# Paths
ROOT_DIR = r"c:\Dev\Projects\Trajectory Geometry"
DATA_PATH = os.path.join(ROOT_DIR, "experiments", "EXP-18_ConsolidatedMetricSuite_2026-02-13", "results", "exp18_metrics_full.csv")
FIGURES_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-18_ConsolidatedMetricSuite_2026-02-13", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Group Mapping for readability
GROUP_LABELS = {
    'G1': 'G1: Direct Failure',
    'G2': 'G2: Direct Success',
    'G3': 'G3: CoT Failure',
    'G4': 'G4: CoT Success'
}

GROUP_PALETTE = {
    'G1: Direct Failure': '#E69F00', # Orange
    'G2: Direct Success': '#0072B2', # Blue
    'G3: CoT Failure': '#D55E00',    # Dark Red
    'G4: CoT Success': '#009E73'     # Green
}

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Rename groups for legend
    df['Condition_Group'] = df['group'].map(GROUP_LABELS)
    
    # Drop rows with NaN in key metrics for the specific plot
    return df

def plot_explore_commit(df):
    """Figure 1: The Explore-Commit Transition
    Visualizes Effective Dimension (PR) vs Layers.
    Takeaway: Success is marked by a deliberate exploration phase followed by a dimensionality collapse (Commit).
    """
    print("Plotting Figure 1: Explore-Commit...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(
        data=df, x='layer', y='effective_dimension', 
        hue='Condition_Group', style='Condition_Group',
        palette=GROUP_PALETTE, markers=True, dashes=False,
        linewidth=2, ax=ax
    )
    
    # Annotate Phase Transition
    ax.axvspan(0, 8, color='gray', alpha=0.1, label='Input Parsing')
    ax.axvspan(8, 18, color='yellow', alpha=0.05, label='Deliberation/Explore')
    ax.axvspan(18, 28, color='green', alpha=0.05, label='Commit Phase')
    
    plt.title("Signature: The 'Explore-Commit' Phase Transition", pad=20)
    plt.ylabel("Representational Participation Ratio (Dim)")
    plt.xlabel("Layer Index")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add Descriptive Text
    plt.annotate("Dimensionality Collapse (Commitment)", xy=(22, 10), xytext=(15, 50),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_explore_commit_transition.png"))
    plt.close()

def plot_attractor_force(df):
    """Figure 2: The Logic of Convergence
    Visualizes Attractor Distance across layers.
    Takeaway: Failure is not just being 'far' from truth, but diverging away from it in the middle layers.
    """
    print("Plotting Figure 2: Attractor Force...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(
        data=df, x='layer', y='mean_attractor_dist', 
        hue='Condition_Group', palette=GROUP_PALETTE,
        linewidth=2.5, ax=ax
    )
    
    plt.title("Signature: Attractor Pull vs. Divergence", pad=20)
    plt.ylabel("Distance to Ground Truth Vector")
    plt.xlabel("Layer Index")
    plt.yscale('log') # Log scale often shows divergence better
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.annotate("Terminal Convergence (Success)", xy=(26, 1e-1), xytext=(15, 1e1),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_attractor_force.png"))
    plt.close()

def plot_path_efficiency_story(df):
    """Figure 3: Wandering vs Focused Search
    Visualizes Tortuosity and stabilization.
    Takeaway: High tortuosity in early layers is 'good exploration', but high tortuosity in late layers is 'hallucination/confusion'.
    """
    print("Plotting Figure 3: Path Efficiency...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Path Efficiency (Tortuosity)
    sns.lineplot(
        data=df, x='layer', y='tortuosity', 
        hue='Condition_Group', palette=GROUP_PALETTE,
        ax=ax1, legend=False
    )
    ax1.set_title("Geometric Wandering (Tortuosity)")
    ax1.set_ylabel("Tortuosity index")
    
    # Stabilization Rate
    sns.lineplot(
        data=df, x='layer', y='stabilization_rate', 
        hue='Condition_Group', palette=GROUP_PALETTE,
        ax=ax2
    )
    ax2.set_title("State Stabilization")
    ax2.set_ylabel("Stabilization Rate")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_wandering_vs_focused.png"))
    plt.close()

def plot_phase_space_story(df):
    """Figure 4: Phase Space Dynamics (Dim vs Dist)
    Takeaway: Successful trajectories move 'Down and Left' (lowering both dim and dist).
    """
    print("Plotting Figure 4: Phase Space...")
    plt.figure(figsize=(10, 8))
    
    # We aggregate by Group and Layer for a cleaner flow plot
    avg_df = df.groupby(['Condition_Group', 'layer'])[['effective_dimension', 'mean_attractor_dist']].mean().reset_index()
    
    for group in avg_df['Condition_Group'].unique():
        sub = avg_df[avg_df['Condition_Group'] == group]
        plt.plot(sub['effective_dimension'], sub['mean_attractor_dist'], 
                 label=group, color=GROUP_PALETTE[group], marker='o', alpha=0.6)
        
        # Draw arrow for direction
        plt.arrow(sub['effective_dimension'].iloc[-2], sub['mean_attractor_dist'].iloc[-2],
                  sub['effective_dimension'].iloc[-1] - sub['effective_dimension'].iloc[-2],
                  sub['mean_attractor_dist'].iloc[-1] - sub['mean_attractor_dist'].iloc[-2],
                  color=GROUP_PALETTE[group], head_width=0.5, alpha=0.8)

    plt.title("Trajectory Phase Space: Dimensionality vs. Error", pad=20)
    plt.xlabel("Effective Dimension (Participation Ratio)")
    plt.ylabel("Distance to Attractor")
    plt.yscale('log')
    plt.legend()
    
    plt.annotate("Ideal 'Success' Path", xy=(10, 1e-1), xytext=(50, 1e2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_phase_space_dynamics.png"))
    plt.close()

def main():
    df = load_data()
    if df.empty:
        print("Data is empty!")
        return
    
    plot_explore_commit(df)
    plot_attractor_force(df)
    plot_path_efficiency_story(df)
    plot_phase_space_story(df)
    
    print(f"Narrative figures generated in {FIGURES_DIR}")

if __name__ == "__main__":
    main()
