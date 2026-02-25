import pandas as pd
import os
import argparse

def generate_markdown_report(ssd_root):
    data_dir = os.path.join(ssd_root, "data")
    
    # 1. Load Accuracy Information
    model_keys = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d not in ['pilot']]
    
    report = "# Experiment 19: Robustness Replication Report\n\n"
    report += "## 1. Accuracy Summary\n\n"
    
    for m_key in model_keys:
        meta_path = os.path.join(data_dir, m_key, "metadata.csv")
        if not os.path.exists(meta_path): continue
        
        df = pd.read_csv(meta_path)
        acc_table = df.groupby(['bin', 'condition'])['correct'].mean().unstack()
        
        report += f"### {m_key}\n\n"
        report += acc_table.to_markdown() + "\n\n"
        
        overall = df.groupby('condition')['correct'].mean()
        report += f"**Overall Accuracy:** Direct: {overall.get('direct', 0):.1%}, CoT: {overall.get('cot', 0):.1%}\n\n"

    # 2. Load Statistical Highlights
    stat_path = os.path.join(data_dir, "statistical_comparisons.csv")
    if os.path.exists(stat_path):
        stats_df = pd.read_csv(stat_path)
        report += "## 2. Geometric Signatures (G4 vs G1 Effect Sizes)\n\n"
        report += "Top 5 metrics by absolute Cohen's d per model:\n\n"
        
        for m_key in stats_df['model'].unique():
            m_stats = stats_df[stats_df['model'] == m_key]
            # Get top absolute cohen_d
            top = m_stats.iloc[m_stats['cohen_d'].abs().argsort()[-10:][::-1]]
            
            report += f"### {m_key} Top Predictors\n\n"
            report += top[['layer', 'metric', 'cohen_d', 'p_val']].to_markdown() + "\n\n"
            
    # 3. Discussion
    report += "## 3. Key Findings\n\n"
    report += "- **Replication Confirmation:** [To be completed based on results]\n"
    report += "- **Physical Trajectory Clarity:** The inclusion of full trajectories (200 tokens) confirms [drift/stability patterns].\n"
    
    out_path = os.path.join(ssd_root, "reports", "Experiment_19_Final_Report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(report)
    
    print(f"Report generated at {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", type=str, default="experiments/EXP-19_Robustness_2026-02-14")
    args = parser.parse_args()
    generate_markdown_report(args.ssd_root)
