"""
Script 05: Analysis D - Response Length Anomaly
===============================================
Correlates response length with geometric metrics and fits predictive models.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import EXP15_DATA_DIR, EXP15_FIGURES_DIR

def main():
    print("="*60)
    print("05_analysis_D.py: Response Length Anomaly")
    print("="*60)
    
    input_file = os.path.join(EXP15_DATA_DIR, "unified_metrics.csv")
    if not os.path.exists(input_file):
        print("Error: unified_metrics.csv not found.")
        return
        
    df = pd.read_csv(input_file)
    
    # Filter for CoT (Direct is too short/uniform usually?)
    # Or compare both?
    # Focus on CoT (G3/G4) where length variance is high.
    cot_df = df[df['condition'] == 'cot'].copy()
    
    # Select Layer 16
    layer_df = cot_df[cot_df['layer'] == 16].copy()
    
    if len(layer_df) == 0:
        print("No layer 16 data.")
        return
        
    print(f"Analyzing {len(layer_df)} CoT trajectories (Layer 16)")
    
    # 1. Correlations
    metrics = ['radius_of_gyration', 'effective_dim', 'msd_exponent', 'cos_to_running_mean']
    metrics = [m for m in metrics if m in layer_df.columns]
    
    correlations = []
    for m in metrics:
        valid = layer_df[[m, 'response_length_chars']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[m], valid['response_length_chars'])
            correlations.append({'metric': m, 'r': r, 'p': p})
            
    corr_df = pd.DataFrame(correlations)
    print("Correlations with Length:")
    print(corr_df)
    corr_df.to_csv(os.path.join(EXP15_DATA_DIR, "analysis_D_correlations.csv"), index=False)
    
    # 2. Predictive Models (Failure Prediction)
    # Target: G3 (Success=0, Failure=1) -> Wait, G3 is Failure? 
    # G4 is Success. Let's predict Success (G4=1, G3=0)
    
    layer_df['is_success'] = (layer_df['group'] == 'G4').astype(int)
    y = layer_df['is_success'].values
    
    models = {
        "Length Only": ['response_length_chars'],
        "Geometry Only": metrics,
        "Combined": ['response_length_chars'] + metrics
    }
    
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    for name, feats in models.items():
        X = layer_df[feats].dropna()
        # Align y using loc on the original dataframe column
        y_aligned = layer_df.loc[X.index, 'is_success'].values
        X = scaler.fit_transform(X)
        
        clf = LogisticRegression()
        scores = cross_val_score(clf, X, y_aligned, cv=skf, scoring='roc_auc')
        
        results.append({
            'model': name,
            'auc_mean': scores.mean(),
            'auc_std': scores.std()
        })
        
    res_df = pd.DataFrame(results)
    print("\nPredictive Power (AUC):")
    print(res_df)
    res_df.to_csv(os.path.join(EXP15_DATA_DIR, "analysis_D_prediction.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res_df, x='model', y='auc_mean', color='skyblue')
    plt.errorbar(x=range(len(res_df)), y=res_df['auc_mean'], yerr=res_df['auc_std'], fmt='none', c='black')
    plt.ylim(0.5, 1.0)
    plt.title("Predicting Success: Geometry vs Length")
    plt.savefig(os.path.join(EXP15_FIGURES_DIR, "analysis_D_prediction_auc.png"))
    plt.close()

if __name__ == "__main__":
    from scipy import stats
    main()
