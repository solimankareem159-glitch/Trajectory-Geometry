import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
import argparse
import itertools

# PID Printing
import os as os_sys
print(f"PID: {os_sys.getpid()}", flush=True)

def run_anova(df, metric):
    try:
        # factors: condition (direct/cot), correct (True/False)
        # We use C() to ensure they are categorical
        formula = f'Q("{metric}") ~ C(condition) * C(correct)'
        model = ols(formula, data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        
        ss_total = aov_table['sum_sq'].sum()
        eta_sq_regime = aov_table.loc['C(condition)', 'sum_sq'] / ss_total
        eta_sq_quality = aov_table.loc['C(correct)', 'sum_sq'] / ss_total
        eta_sq_interaction = aov_table.loc['C(condition):C(correct)', 'sum_sq'] / ss_total
        
        return {
            'eta_sq_regime': eta_sq_regime,
            'eta_sq_quality': eta_sq_quality,
            'eta_sq_interaction': eta_sq_interaction,
            'f_regime': aov_table.loc['C(condition)', 'F'],
            'p_regime': aov_table.loc['C(condition)', 'PR(>F)'],
            'f_quality': aov_table.loc['C(correct)', 'F'],
            'p_quality': aov_table.loc['C(correct)', 'PR(>F)'],
            'f_interaction': aov_table.loc['C(condition):C(correct)', 'F'],
            'p_interaction': aov_table.loc['C(condition):C(correct)', 'PR(>F)']
        }
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", type=str, default="experiments/EXP-19_Robustness_2026-02-14")
    args = parser.parse_args()
    
    data_dir = os.path.join(args.ssd_root, "data")
    output_dir = os.path.join(data_dir, "analysis_19b")
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(data_dir, "all_metrics.csv")
    pairwise_path = os.path.join(data_dir, "exhaustive_pairwise.csv")
    
    if not os.path.exists(metrics_path) or not os.path.exists(pairwise_path):
        print("Required CSV files not found.")
        return
        
    metrics_all = pd.read_csv(metrics_path)
    pairwise_all = pd.read_csv(pairwise_path)
    
    model_keys = ['qwen05b', 'qwen15b']
    
    # Key metrics for depth plots
    key_metrics = [
        'clean_radius_of_gyration', 'clean_effective_dimension', 
        'clean_dir_consistency', 'clean_tortuosity', 
        'clean_speed', 'full_time_to_commit', 
        'full_commitment_sharpness', 'full_cos_slope_to_final'
    ]
    
    all_anova_results = []
    
    for m_key in model_keys:
        print(f"\nProcessing model: {m_key}")
        # Load metadata
        meta_path = os.path.join(data_dir, m_key, "metadata.csv")
        if not os.path.exists(meta_path):
            continue
        m_meta = pd.read_csv(meta_path)
        m_metrics = metrics_all[metrics_all['model'] == m_key]
        
        # Merge to get Correctness
        merged = pd.merge(m_metrics, m_meta[['problem_id', 'condition', 'correct', 'group']], 
                         on=['problem_id', 'condition'])
        
        # --- Analysis 1: 2-Way ANOVA ---
        print("  Running Analysis 1: 2-Way ANOVA...")
        layers = sorted(merged['layer'].unique())
        metric_cols = [c for c in merged.columns if 'clean_' in c or 'full_' in c]
        
        m_res = []
        for layer in layers:
            l_df = merged[merged['layer'] == layer]
            # Ensure we have all groups for ANOVA
            if len(l_df['group'].unique()) < 4:
                continue
                
            for col in metric_cols:
                res = run_anova(l_df, col)
                if res:
                    res.update({'model': m_key, 'layer': layer, 'metric': col})
                    m_res.append(res)
        
        all_anova_results.extend(m_res)
        
        # --- Analysis 2: Layer Localization ---
        print("  Running Analysis 2: Layer Localization plots...")
        m_pairwise = pairwise_all[pairwise_all['model'] == m_key]
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle(f"Effect Localization: {m_key}", fontsize=16)
        
        for idx, metric in enumerate(key_metrics):
            ax = axes[idx // 2, idx % 2]
            metric_data = m_pairwise[m_pairwise['metric'] == metric]
            
            # G3 vs G1 (Regime)
            regime = metric_data[((metric_data['group_a'] == 'G1') & (metric_data['group_b'] == 'G3')) |
                                 ((metric_data['group_a'] == 'G3') & (metric_data['group_b'] == 'G1'))]
            # G4 vs G3 (CoT Quality)
            cot_qual = metric_data[((metric_data['group_a'] == 'G3') & (metric_data['group_b'] == 'G4')) |
                                   ((metric_data['group_a'] == 'G4') & (metric_data['group_b'] == 'G3'))]
            # G2 vs G1 (Direct Quality)
            dir_qual = metric_data[((metric_data['group_a'] == 'G1') & (metric_data['group_b'] == 'G2')) |
                                   ((metric_data['group_a'] == 'G2') & (metric_data['group_b'] == 'G1'))]
            
            ax.plot(regime['layer'], regime['cohen_d'].abs(), 'k--', label='G3 vs G1 (Regime)')
            ax.plot(cot_qual['layer'], cot_qual['cohen_d'].abs(), 'r-', label='G4 vs G3 (CoT Quality)')
            ax.plot(dir_qual['layer'], dir_qual['cohen_d'].abs(), 'b:', label='G2 vs G1 (Direct Quality)')
            
            ax.set_title(metric)
            ax.set_xlabel("Layer")
            ax.set_ylabel("|Cohen's d|")
            if idx == 0:
                ax.legend()
                
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f"effect_localization_{m_key}.png"), dpi=300)
        plt.close()
        
        # --- Analysis 3: G3 Position Analysis ---
        print("  Running Analysis 3: G3 Position Analysis...")
        pos_results = []
        for layer in layers:
            l_pairwise = m_pairwise[m_pairwise['layer'] == layer]
            for metric in metric_cols:
                # Need mean values: mean_G3, mean_G1, mean_G4
                # We can get these from exhaustive_pairwise
                # combinations: (G1, G2), (G1, G3), (G1, G4), (G2, G3), (G2, G4), (G3, G4)
                # G3 vs G1 comparison has mean_a (G1) and mean_b (G3)
                # G4 vs G1 comparison has mean_a (G1) and mean_b (G4)
                
                c_31 = l_pairwise[((l_pairwise['group_a'] == 'G1') & (l_pairwise['group_b'] == 'G3')) & (l_pairwise['metric'] == metric)]
                c_41 = l_pairwise[((l_pairwise['group_a'] == 'G1') & (l_pairwise['group_b'] == 'G4')) & (l_pairwise['metric'] == metric)]
                
                if not c_31.empty and not c_41.empty:
                    m1 = c_31.iloc[0]['mean_a']
                    m3 = c_31.iloc[0]['mean_b']
                    m4 = c_41.iloc[0]['mean_b']
                    
                    denom = (m4 - m1)
                    if abs(denom) > 1e-9:
                        pos_index = (m3 - m1) / denom
                        pos_results.append({'layer': layer, 'metric': metric, 'pos_index': pos_index})
        
        pos_df = pd.DataFrame(pos_results)
        if not pos_df.empty:
            pivot_pos = pos_df.pivot(index='metric', columns='layer', values='pos_index')
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_pos, center=0.5, cmap='RdBu_r')
            plt.title(f"G3 Position Index Heatmap ({m_key})\n0=Failure-like, 1=Success-like")
            plt.savefig(os.path.join(output_dir, f"position_heatmap_{m_key}.png"), dpi=300)
            plt.close()
            
            plt.figure(figsize=(8, 6))
            sns.histplot(pos_df['pos_index'], bins=30, kde=True)
            plt.title(f"Distribution of G3 Position Indices ({m_key})")
            plt.savefig(os.path.join(output_dir, f"position_hist_{m_key}.png"), dpi=300)
            plt.close()
            
        # --- Analysis 4: Interaction Signatures ---
        print("  Running Analysis 4: Interaction Signatures...")
        # (Already handled partially by ANOVA, but checking direction flips)
        interactions = []
        for layer in layers:
            l_pairwise = m_pairwise[m_pairwise['layer'] == layer]
            for metric in metric_cols:
                d_dir = l_pairwise[((l_pairwise['group_a'] == 'G1') & (l_pairwise['group_b'] == 'G2')) & (l_pairwise['metric'] == metric)]
                d_cot = l_pairwise[((l_pairwise['group_a'] == 'G3') & (l_pairwise['group_b'] == 'G4')) & (l_pairwise['metric'] == metric)]
                
                if not d_dir.empty and not d_cot.empty:
                    cd_dir = d_dir.iloc[0]['cohen_d']
                    cd_cot = d_cot.iloc[0]['cohen_d']
                    p_dir = d_dir.iloc[0]['p_val']
                    p_cot = d_cot.iloc[0]['p_val']
                    
                    if np.sign(cd_dir) != np.sign(cd_cot) and p_dir < 0.05 and p_cot < 0.05:
                        interactions.append({
                            'layer': layer, 'metric': metric, 
                            'd_direct': cd_dir, 'd_cot': cd_cot,
                            'p_direct': p_dir, 'p_cot': p_cot
                        })
        inter_df = pd.DataFrame(interactions)
        inter_df.to_csv(os.path.join(output_dir, f"interaction_signatures_{m_key}.csv"), index=False)
        
        # --- Analysis 5: Predictive Power (qwen05b only) ---
        if m_key == 'qwen05b':
            print("  Running Analysis 5: Within-Regime Predictive Power...")
            # Pick top 5 within-regime metrics (by avg abs cohen_d in G4 vs G3)
            cot_q_effs = m_pairwise[((m_pairwise['group_a'] == 'G3') & (m_pairwise['group_b'] == 'G4'))].groupby('metric')['cohen_d'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
            top5_metrics = cot_q_effs.head(5).index.tolist()
            print(f"    Top 5 predictive metrics: {top5_metrics}")
            
            auc_results = []
            for layer in layers:
                l_df = merged[merged['layer'] == layer].dropna(subset=top5_metrics)
                
                # CoT only
                cot_df = l_df[l_df['condition'] == 'cot']
                # Direct only
                dir_df = l_df[l_df['condition'] == 'direct']
                
                def get_auc(sub_df, features):
                    if len(sub_df['correct'].unique()) < 2: return np.nan
                    X = sub_df[features]
                    y = sub_df['correct'].astype(int)
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    aucs = []
                    for train_idx, val_idx in kf.split(X):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        if len(y_val.unique()) < 2: continue
                        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                        aucs.append(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
                    return np.mean(aucs) if aucs else np.nan

                auc_cot = get_auc(cot_df, top5_metrics)
                auc_dir = get_auc(dir_df, top5_metrics)
                
                # Regime only (predict correct from condition)
                # But regime is constant in cot_df/dir_df. 
                # Use total l_df for comparison
                l_df['regime_bin'] = (l_df['condition'] == 'cot').astype(int)
                auc_regime_only = get_auc(l_df, ['regime_bin'])
                auc_geom_only = get_auc(l_df, top5_metrics)
                auc_full = get_auc(l_df, top5_metrics + ['regime_bin'])
                
                auc_results.append({
                    'layer': layer,
                    'auc_cot_isolated': auc_cot,
                    'auc_dir_isolated': auc_dir,
                    'auc_regime_only': auc_regime_only,
                    'auc_geom_only': auc_geom_only,
                    'auc_full_model': auc_full
                })
            
            auc_df = pd.DataFrame(auc_results)
            plt.figure(figsize=(10, 6))
            plt.plot(auc_df['layer'], auc_df['auc_regime_only'], 'k--', label='Regime Only')
            plt.plot(auc_df['layer'], auc_df['auc_geom_only'], 'b-', label='Geometry Only')
            plt.plot(auc_df['layer'], auc_df['auc_full_model'], 'r-', label='Full Model (Regime + Geom)')
            plt.axhline(0.5, color='gray', linestyle=':')
            plt.title("Correctness Prediction AUC (Qwen-0.5B)")
            plt.xlabel("Layer")
            plt.ylabel("5-Fold CV AUC")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "predictive_power_auc_qwen05b.png"), dpi=300)
            plt.close()
            auc_df.to_csv(os.path.join(output_dir, "auc_results_qwen05b.csv"), index=False)

    # Finalize Analysis 1 Summary
    anova_df = pd.DataFrame(all_anova_results)
    anova_df.to_csv(os.path.join(output_dir, "anova_decomposition.csv"), index=False)
    
    # Summary Table: Avg Eta-Squared for top 10 metrics
    summary_data = []
    for m_key in model_keys:
        m_anova = anova_df[anova_df['model'] == m_key]
        avg_shares = m_anova.groupby('metric')[['eta_sq_regime', 'eta_sq_quality', 'eta_sq_interaction']].mean()
        avg_shares['total_explained'] = avg_shares.sum(axis=1)
        top10 = avg_shares.sort_values('total_explained', ascending=False).head(10)
        
        # Plot stacked bar
        top10[['eta_sq_regime', 'eta_sq_quality', 'eta_sq_interaction']].plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f"Variance Decomposition: Top 10 Metrics ({m_key})")
        plt.ylabel("Eta-Squared (Prop. Variance)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"variance_decomposition_{m_key}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
