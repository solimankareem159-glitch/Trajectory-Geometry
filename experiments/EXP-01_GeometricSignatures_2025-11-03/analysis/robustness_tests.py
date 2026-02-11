import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import os

INPUT_FILE = "warps.parquet"
REPORT_FILE = "ROBUSTNESS_REPORT.md"
FIGURE_DIR = "figures"

def ensure_dir(d):
    try:
        os.makedirs(d)
    except FileExistsError:
        pass

def run_robustness_tests():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        return

    print("Loading warps...")
    df = pd.read_parquet(INPUT_FILE)
    ensure_dir(FIGURE_DIR)
    
    state_defs = df['state_def'].unique()
    
    report_lines = []
    report_lines.append("# Robustness Analysis Report")
    report_lines.append("")
    report_lines.append("Analyzing the distribution of warp vectors across paraphrases and topics.")
    report_lines.append("We compare **Within-Operator** variance (variance of vectors belonging to the same operator) vs **Between-Operator** variance.")
    report_lines.append("")
    
    for s_def in state_defs:
        print(f"Analyzing State Definition: {s_def}")
        report_lines.append(f"## State Definition: {s_def}")
        
        subset = df[df['state_def'] == s_def].copy()
        
        # Vectors
        X = np.stack(subset['warp_vector'].values)
        labels = subset['operator_name'].values
        unique_labels = np.unique(labels)
        
        # 1. Variance ratio using sum of squares
        # Global mean
        global_mean = np.mean(X, axis=0)
        ss_total = np.sum((X - global_mean)**2)
        
        ss_within = 0
        ss_between = 0
        
        for label in unique_labels:
            mask = labels == label
            group_X = X[mask]
            group_mean = np.mean(group_X, axis=0)
            
            # Within: dist from group mean
            ss_within += np.sum((group_X - group_mean)**2)
            
            # Between: weighted dist of group mean from global mean
            n_group = len(group_X)
            ss_between += n_group * np.sum((group_mean - global_mean)**2)
            
        # Variance = SS / degrees of freedom
        # N = total samples, K = number of groups
        N = len(X)
        K = len(unique_labels)
        
        var_within = ss_within / (N - K)
        var_between = ss_between / (K - 1)
        
        f_ratio = var_between / var_within
        
        report_lines.append("### Variance Analysis (ANOVA-like)")
        report_lines.append(f"- **Within-Operator Variance:** {var_within:.4f}")
        report_lines.append(f"- **Between-Operator Variance:** {var_between:.4f}")
        report_lines.append(f"- **Ratio (Between/Within):** {f_ratio:.2f}")
        
        if var_within < var_between:
            report_lines.append("> ✅ **PASS:** Within-operator variance is lower than between-operator variance.")
        else:
            report_lines.append("> ❌ **FAIL:** Within-operator variance is higher or equal.")
            
        # 2. Pairwise Distance Distribution Analysis
        # Sample pairwise distances
        # Calculating ALL pairwise distances might be heavy if N is huge, but here N=500, so 500x500 = 250k floats, easy.
        
        dists = euclidean_distances(X, X)
        
        within_dists = []
        between_dists = []
        
        # Iterate upper triangle
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    between_dists.append(d)
                    
        # Plotting distributions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(within_dists, fill=True, label='Within-Operator', clip=(0, None))
        sns.kdeplot(between_dists, fill=True, label='Between-Operator', clip=(0, None))
        plt.title(f"Distribution of Pairwise Distances - State {s_def}")
        plt.xlabel("Euclidean Distance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = f"{FIGURE_DIR}/robustness_dists_{s_def}.png"
        plt.savefig(plot_path)
        plt.close()
        
        report_lines.append("\n### Distance Distributions")
        report_lines.append(f"![Distribution Plot]({plot_path})")
        
        # Stats on distances
        mean_within = np.mean(within_dists)
        mean_between = np.mean(between_dists)
        report_lines.append(f"- Mean Distance (Within): {mean_within:.4f}")
        report_lines.append(f"- Mean Distance (Between): {mean_between:.4f}")
        
        report_lines.append("\n---\n")
        
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Robustness analysis complete. Report written to {REPORT_FILE}")

if __name__ == "__main__":
    run_robustness_tests()
