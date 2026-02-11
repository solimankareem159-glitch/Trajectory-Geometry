"""
Script 02: Analysis A - Difficulty x Geometry
=============================================
Derives difficulty bands (magnitude, sign) and recomputes metric contrasts.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import EXP15_DATA_DIR, EXP15_FIGURES_DIR

def get_magnitude(truth_str):
    try:
        # Remove non-numeric chars except . and -
        clean = ''.join(c for c in str(truth_str) if c.isdigit() or c in '.-')
        val = float(clean)
        return abs(val)
    except:
        return np.nan

def get_sign(truth_str):
    try:
        clean = ''.join(c for c in str(truth_str) if c.isdigit() or c in '.-')
        val = float(clean)
        return 'positive' if val >= 0 else 'negative'
    except:
        return 'unknown'

def cohens_d(x, y):
    if len(x) < 2 or len(y) < 2: return np.nan
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def main():
    print("="*60)
    print("02_analysis_A.py: Difficulty x Geometry")
    print("="*60)
    
    input_file = os.path.join(EXP15_DATA_DIR, "unified_metrics.csv")
    if not os.path.exists(input_file):
        print(f"Error: Unified file {input_file} not found. Run script 01 first.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # 1. Derive Difficulty Bands
    print("Deriving difficulty bands...")
    
    # We need to extract truth values. 
    # The 'truth' column is a string like "123" or "-45".
    
    df['truth_mag'] = df['truth'].apply(get_magnitude)
    df['truth_sign'] = df['truth'].apply(get_sign)
    
    # Magnitude quartiles
    # Unique problems only for quantiles?
    # Or just use all rows? using all rows is fine 
    # but strictly we should compute quantiles over unique problems.
    # Actually, magnitude is derived from 'truth', which is constant per problem.
    # So we can just dropna and qcut.
    
    # Use qcut on the non-nan valid magnitudes
    valid_mags = df['truth_mag'].dropna()
    if len(valid_mags) > 0:
        # Quartiles: Q1, Q2, Q3, Q4
        df['mag_bin'] = pd.qcut(df['truth_mag'], 4, labels=['Small', 'Medium', 'Large', 'Extra Large'])
    else:
        df['mag_bin'] = 'Unknown'
        
    print(f"Magnitude distribution:\n{df['mag_bin'].value_counts()}")
    
    # 2. Stratified Analysis
    # We want to check if metric contrasts (e.g. G4 vs G2) vary by difficulty
    
    # Focused Metrics (Top Discriminators from Exp 14)
    # radius_of_gyration, effective_dim, msd_exponent, cos_to_late_window
    
    key_metrics = ['radius_of_gyration', 'effective_dim', 'msd_exponent', 'cos_to_late_window']
    # Filter for columns that actually exist
    key_metrics = [m for m in key_metrics if m in df.columns]
    
    results = []
    
    # We focus on layers 15-24 where effects are strongest? 
    # Or just iterate all layers? All layers is better for heatmaps.
    layers = sorted(df['layer'].unique())
    # Remove -1 (cross layer) for this
    layers = [l for l in layers if l >= 0]
    
    # Comparison: G4 (CoT Success) vs G2 (Direct Success)
    # "Does CoT help MORE on hard problems?"
    
    print("Computing stratified contrasts (G4 vs G2)...")
    
    for mag_bin in ['Small', 'Medium', 'Large', 'Extra Large']:
        sub_df = df[df['mag_bin'] == mag_bin]
        
        for layer in layers:
            g4_data = sub_df[(sub_df['group'] == 'G4') & (sub_df['layer'] == layer)]
            g2_data = sub_df[(sub_df['group'] == 'G2') & (sub_df['layer'] == layer)]
            
            for metric in key_metrics:
                vals_g4 = g4_data[metric].dropna()
                vals_g2 = g2_data[metric].dropna()
                
                d = cohens_d(vals_g4, vals_g2)
                
                results.append({
                    'mag_bin': mag_bin,
                    'layer': layer,
                    'metric': metric,
                    'cohens_d': d,
                    'n_g4': len(vals_g4),
                    'n_g2': len(vals_g2)
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(EXP15_DATA_DIR, "analysis_A_difficulty_results.csv"), index=False)
    
    # 3. Figures
    print("Generating figures...")
    
    # Heatmaps for each metric: X=Layer, Y=Magnitude Bin
    for metric in key_metrics:
        pivot = res_df[res_df['metric'] == metric].pivot(index='mag_bin', columns='layer', values='cohens_d')
        # Reorder index to valid size order
        pivot = pivot.reindex(['Small', 'Medium', 'Large', 'Extra Large'])
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt=".2f")
        plt.title(f"G4 vs G2 Effect Size by Difficulty: {metric}")
        plt.tight_layout()
        plt.savefig(os.path.join(EXP15_FIGURES_DIR, f"heatmap_difficulty_{metric}.png"))
        plt.close()

if __name__ == "__main__":
    main()
