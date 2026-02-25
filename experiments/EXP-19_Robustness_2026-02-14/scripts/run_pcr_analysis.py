"""
Probability Cloud Regression (PCR) - Denoised Feature Prediction
================================================================

This script implements the CloudRegressor described in PCR_PROJECT.md and
tests it on the EXP-15 Length Anomaly prediction task (distinguishing CoT success
from CoT failure based on geometric features).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

# Set up paths relative to experiment 19 or 15
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXP15_DATA_DIR = os.path.join(PROJECT_ROOT, "EXP-15_LengthConfound_2025-12-05", "data")
EXP19_REPORTS_DIR = os.path.join(PROJECT_ROOT, "EXP-19_Robustness_2026-02-14", "reports")


class CloudRegressor:
    """
    Fits a line y = mx + c through probability clouds by minimizing the
    weighted Mahalanobis distance from each point's cloud center to the line.
    """
    def __init__(self):
        self.m = None
        self.c = None
        self.inferred_x = None
        self.inferred_y = None
        
    def _objective(self, params, x_obs, y_obs, sig_x, sig_y):
        m, c = params
        
        # Calculate the closest point on the line y=mx+c for each (x_obs, y_obs)
        # considering the different standard deviations (Mahalanobis distance)
        # 
        # The perpendicular point (x_hat, y_hat) matching Mahalanobis is found by minimizing:
        # d^2 = (x - x_obs)^2 / sig_x^2 + (y - y_obs)^2 / sig_y^2 subject to y = mx + c
        
        w_x = 1.0 / (sig_x ** 2)
        w_y = 1.0 / (sig_y ** 2)
        
        # Solving for the minimum of the distance squared:
        x_hat = (w_x * x_obs + w_y * m * (y_obs - c)) / (w_x + w_y * (m ** 2))
        y_hat = m * x_hat + c
        
        dist_sq = w_x * (x_obs - x_hat) ** 2 + w_y * (y_obs - y_hat) ** 2
        
        return np.sum(dist_sq)

    def fit(self, X: np.ndarray, Y: np.ndarray, sig_x: np.ndarray, sig_y: np.ndarray) -> 'CloudRegressor':
        """
        Fits the probability cloud regression line.
        """
        # Ensure minimum variance to avoid division by zero
        sig_x = np.maximum(sig_x, 1e-6)
        sig_y = np.maximum(sig_y, 1e-6)
        
        # Initial guess from standard OLS
        A = np.vstack([X, np.ones(len(X))]).T
        m_ols, c_ols = np.linalg.lstsq(A, Y, rcond=None)[0]
        
        # Optimize
        res = minimize(self._objective, [m_ols, c_ols], args=(X, Y, sig_x, sig_y))
        
        self.m, self.c = res.x
        
        # Compute inferred true positions
        w_x = 1.0 / (sig_x ** 2)
        w_y = 1.0 / (sig_y ** 2)
        self.inferred_x = (w_x * X + w_y * self.m * (Y - self.c)) / (w_x + w_y * (self.m ** 2))
        self.inferred_y = self.m * self.inferred_x + self.c
        
        return self
        
    def get_inferred_true_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the retroactively inferred X and Y coordinates on the line."""
        return self.inferred_x, self.inferred_y


def run_analysis():
    print("="*60)
    print("Probability Cloud Regression (PCR) - Denoised Feature Prediction")
    print("="*60)
    
    # 1. Load Data
    metrics_file = os.path.join(EXP15_DATA_DIR, "unified_metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found.")
        print("Please verify the data path.")
        return
        
    df = pd.read_csv(metrics_file)
    
    # Focus on CoT trajectories (G3/G4) where we want to predict Success vs Failure
    cot_df = df[df['condition'] == 'cot'].copy()
    
    # We want features at Layer 16 (often highly predictive in EXP-15)
    layer_df = cot_df[cot_df['layer'] == 16].copy()
    
    if len(layer_df) == 0:
        print("No layer 16 data found.")
        return
        
    print(f"Loaded {len(layer_df)} CoT trajectories.")
    
    # Prepare Data for Global Variance Calculation
    # We will compute the standard deviation of each metric across all available layers for each trajectory.
    features = ['radius_of_gyration', 'effective_dim', 'msd_exponent', 'cos_to_running_mean']
    features = [f for f in features if f in layer_df.columns]
    
    print(f"Computing global variances for {len(features)} features...")
    
    # Compute standard deviation for each trajectory (problem_id + condition) across all layers
    traj_variances = {}
    for (pid, cond), group in cot_df.groupby(['problem_id', 'condition']):
        # If we have only 1 layer, std is 0.
        std_vals = group[features].std().fillna(0).to_dict()
        traj_variances[(pid, cond)] = std_vals
        
    # Map these variances back to our layer_df
    for f in features:
        layer_df[f'{f}_std'] = layer_df.apply(lambda row: traj_variances[(row['problem_id'], row['condition'])].get(f, 0.0), axis=1)

    # Convert response length to log scale for regression (often better behaved)
    layer_df['log_length'] = np.log1p(layer_df['response_length_chars'])
    # Estimate Y variance as heuristic (e.g., small constant since length is strictly observable, but let's give it mild noise)
    # or alternatively, regress against 'logit_gap' which might have layer-variance. Let's stick to length as an anchor.
    sig_y = np.ones(len(layer_df)) * layer_df['log_length'].std() * 0.1 
    
    # Apply Cloud Regression to Denoise Features
    print("Applying CloudRegressor feature denoising...")
    denoised_features = pd.DataFrame(index=layer_df.index)
    
    for f in features:
        X_obs = layer_df[f].values
        Y_obs = layer_df['log_length'].values
        sig_x = layer_df[f'{f}_std'].values
        
        # Replace 0 variance with a small number to avoid division by zero errors inside the regressor 
        sig_x = np.maximum(sig_x, 1e-4)
        
        cr = CloudRegressor()
        cr.fit(X_obs, Y_obs, sig_x, sig_y)
        
        x_hat, _ = cr.get_inferred_true_positions()
        denoised_features[f'{f}_clean'] = x_hat
        
    # Target: G4 (Success) == 1, G3 (Failure) == 0
    y = (layer_df['group'] == 'G4').astype(int).values
    
    # 3. Logistic Regression Evaluation
    print("Evaluating Predictive Power (5-Fold Stratified CV)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    models = {
        "Baseline (Noisy OLS Features)": features,
        "PCR Denoised Features": [f'{f}_clean' for f in features]
    }
    
    results = []
    
    # Combine original and denoised into one working DF to ensure row alignment
    working_df = pd.concat([layer_df[features], denoised_features], axis=1)
    
    for name, feats in models.items():
        X = working_df[feats].values
        
        # Remove any potential NaNs generated
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        if len(X_valid) == 0:
            print(f"Warning: No valid data for {name}")
            continue
            
        X_scaled = scaler.fit_transform(X_valid)
        
        clf = LogisticRegression()
        scores = cross_val_score(clf, X_scaled, y_valid, cv=skf, scoring='roc_auc')
        
        results.append({
            'Model': name,
            'AUC Mean': scores.mean(),
            'AUC Std': scores.std()
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- Results ---")
    print(res_df.to_string(index=False))
    
    os.makedirs(EXP19_REPORTS_DIR, exist_ok=True)
    out_file = os.path.join(EXP19_REPORTS_DIR, "pcr_feature_denoising_results.csv")
    res_df.to_csv(out_file, index=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    run_analysis()
