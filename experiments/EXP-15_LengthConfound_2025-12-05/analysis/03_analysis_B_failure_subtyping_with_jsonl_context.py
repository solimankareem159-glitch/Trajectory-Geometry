"""
Script 03: Analysis B - Failure Subtyping
=========================================
Clusters CoT failures (G3) to identify subtypes of reasoning errors.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import EXP15_DATA_DIR, EXP15_FIGURES_DIR

def main():
    print("="*60)
    print("03_analysis_B.py: Failure Subtyping")
    print("="*60)
    
    input_file = os.path.join(EXP15_DATA_DIR, "unified_metrics.csv")
    if not os.path.exists(input_file):
        print("Error: unified_metrics.csv not found.")
        return
        
    df = pd.read_csv(input_file)
    
    # Filter for G3 (CoT Failure) at Representative Layer (e.g. 13 or 16)
    # Layer 16 showed high discriminative power in Exp 14.
    target_layer = 16
    print(f"Focusing on CoT Failures (G3) at Layer {target_layer}...")
    
    g3_df = df[(df['group'] == 'G3') & (df['layer'] == target_layer)].copy()
    
    if len(g3_df) < 10:
        print("Not enough G3 samples for clustering.")
        return
        
    print(f"G3 Samples: {len(g3_df)}")
    
    # Select features for clustering
    # We want geometric features that capture dynamics
    features = [
        'radius_of_gyration', 'effective_dim', 'msd_exponent', 
        'cos_to_late_window', 'tortuosity', 'speed', 'recurrence_rate'
    ]
    features = [f for f in features if f in g3_df.columns]
    
    X = g3_df[features].dropna()
    print(f"Clustering on {len(features)} features: {features}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal k (2-5)
    best_k = 2
    best_score = -1
    
    results = {}
    
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        results[k] = score
        print(f"k={k}: Silhouette={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Selected k={best_k}")
    
    # Final Fit
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    g3_df.loc[X.index, 'cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analysis of Clusters
    # 1. Centroids (unscaled)
    centroids = g3_df.groupby('cluster')[features].mean()
    print("\nCluster Centroids:")
    print(centroids)
    centroids.to_csv(os.path.join(EXP15_DATA_DIR, "analysis_B_centroids.csv"))
    
    # 2. Composition (Context enrichment)
    # Check response length, magnitude, etc.
    # We might need to handle nans if some rows dropped
    g3_df['truth_mag'] = g3_df['truth'].apply(lambda x: float(''.join(c for c in str(x) if c.isdigit() or c in '.-')) if pd.notnull(x) else np.nan)
    
    composition = g3_df.groupby('cluster')[['response_length_chars', 'truth_mag']].mean()
    print("\nCluster Composition:")
    print(composition)
    composition.to_csv(os.path.join(EXP15_DATA_DIR, "analysis_B_composition.csv"))
    
    # 3. Visualization (PCA)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=g3_df.loc[X.index, 'cluster'], palette='viridis', s=100)
    plt.title(f"CoT Failure Subtypes (Layer {target_layer}, k={best_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(os.path.join(EXP15_FIGURES_DIR, "analysis_B_clusters_pca.png"))
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
