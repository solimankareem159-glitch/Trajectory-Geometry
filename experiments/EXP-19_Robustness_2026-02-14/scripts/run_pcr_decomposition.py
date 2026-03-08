"""
EXP-19 / EXP-19B: PCR-Corrected Decomposition Analysis
=======================================================

This script implements the full Experiment 19B decomposition using
Probability Cloud Regression (PCR) denoising to correct for
attenuation bias in geometric feature measurements.

PROBLEM ADDRESSED:
  Prior logistic regression AUC (~0.78) and Cohen's d values are
  systematically biased downward because geometric features computed
  from finite token windows contain within-trajectory measurement noise.
  OLS/logistic regression treats observed values as ground truth,
  compressing decision boundaries toward random chance (attenuation bias).
  This script corrects this by denoising features with CloudRegressor
  before all statistical analyses.

PCR METHODOLOGY:
  For each metric, per-trajectory sigma is estimated as the std deviation
  of that metric across all layers within the trajectory. This provides a
  heterogeneous per-point uncertainty estimate. CloudRegressor then fits
  the Mahalanobis-distance-minimizing line and infers the de-noised
  "true" position for each observation.

ANALYSES (mirrors original run_decomposition_analysis.py but with PCR):
  1. Two-Way ANOVA on denoised metrics (compare eta-sq to raw)
  2. Layer localization: Cohen's d on denoised features
  3. G3 Position Index on denoised group means
  4. Interaction Signatures: sign-flip detection on denoised Cohen's d
  5. PCR-AUC vs raw-AUC comparison (logistic regression)

OUTPUTS:
  data/analysis_19b/pcr_denoised_metrics_{model}.csv
  data/analysis_19b/pcr_anova_decomposition.csv
  data/analysis_19b/pcr_auc_comparison_qwen05b.csv
  data/analysis_19b/pcr_interaction_signatures_{model}.csv
  data/analysis_19b/pcr_*  (plots as 300-DPI PNGs)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

import os; print(f"PID: {os.getpid()}", flush=True)

# ─────────────────────────────────────────────
# CloudRegressor (PCR core — from run_pcr_analysis.py)
# ─────────────────────────────────────────────

class CloudRegressor:
    """
    Fits y = mx + c through probability clouds by minimising the
    weighted Mahalanobis distance from each cloud centre to the line.
    Implements errors-in-variables (Deming-style) regression with
    per-point heterogeneous uncertainty.
    """
    def __init__(self):
        self.m = None
        self.c = None
        self.inferred_x = None
        self.inferred_y = None

    def _objective(self, params: np.ndarray,
                   x_obs: np.ndarray, y_obs: np.ndarray,
                   sig_x: np.ndarray, sig_y: np.ndarray) -> float:
        m, c = params
        w_x = 1.0 / (sig_x ** 2)
        w_y = 1.0 / (sig_y ** 2)
        x_hat = (w_x * x_obs + w_y * m * (y_obs - c)) / (w_x + w_y * (m ** 2))
        y_hat = m * x_hat + c
        dist_sq = w_x * (x_obs - x_hat) ** 2 + w_y * (y_obs - y_hat) ** 2
        return float(np.sum(dist_sq))

    def fit(self, X: np.ndarray, Y: np.ndarray,
            sig_x: np.ndarray, sig_y: np.ndarray) -> "CloudRegressor":
        sig_x = np.maximum(sig_x, 1e-6)
        sig_y = np.maximum(sig_y, 1e-6)
        A = np.vstack([X, np.ones(len(X))]).T
        m0, c0 = np.linalg.lstsq(A, Y, rcond=None)[0]
        res = minimize(self._objective, [m0, c0],
                       args=(X, Y, sig_x, sig_y),
                       method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})
        self.m, self.c = res.x
        w_x = 1.0 / (sig_x ** 2)
        w_y = 1.0 / (sig_y ** 2)
        self.inferred_x = (w_x * X + w_y * self.m * (Y - self.c)) / (w_x + w_y * (self.m ** 2))
        self.inferred_y = self.m * self.inferred_x + self.c
        return self

    def get_denoised_x(self) -> np.ndarray:
        """Return the denoised X positions on the fitted line."""
        return self.inferred_x


# ─────────────────────────────────────────────
# PCR Feature Denoising
# ─────────────────────────────────────────────

METRIC_COLS = [
    "clean_phase_count", "full_speed", "full_turn_angle", "full_tortuosity",
    "full_dir_consistency", "full_stabilisation", "full_radius_of_gyration",
    "full_effective_dimension", "full_cos_slope_to_final", "full_time_to_commit",
    "full_commitment_sharpness", "full_phase_count", "clean_interlayer_alignment",
    "full_interlayer_alignment", "clean_speed", "clean_turn_angle",
    "clean_tortuosity", "clean_dir_consistency", "clean_stabilisation",
    "clean_radius_of_gyration", "clean_effective_dimension", "clean_cos_slope_to_final",
    "clean_time_to_commit", "clean_commitment_sharpness",
]

KEY_METRICS = [
    "clean_radius_of_gyration", "clean_effective_dimension",
    "clean_dir_consistency", "clean_tortuosity",
    "clean_speed", "full_time_to_commit",
    "full_commitment_sharpness", "full_cos_slope_to_final",
]


def denoise_features_pcr(metrics_df: pd.DataFrame,
                          metric_cols: List[str]) -> pd.DataFrame:
    """
    Apply CloudRegressor denoising to all metric columns.

    Strategy: estimate per-trajectory sigma from across-layer std deviation.
    Anchor Y to the trajectory index (ordinal rank of problem_id within model),
    which provides a stable reference axis for the Mahalanobis projection.

    Returns a copy of metrics_df with additional *_pcr columns.
    """
    df = metrics_df.copy()
    available = [c for c in metric_cols if c in df.columns]

    # Compute per-trajectory sigma (noise estimate) for each metric
    # Group by problem_id + condition across all layers
    traj_sigma: Dict[tuple, Dict[str, float]] = {}
    for (pid, cond, model), grp in df.groupby(["problem_id", "condition", "model"]):
        key = (pid, cond, model)
        sigma_vals = grp[available].std(numeric_only=True).fillna(0).to_dict()
        traj_sigma[key] = sigma_vals

    # Map sigmas back to rows
    for col in available:
        df[f"{col}_sigma"] = df.apply(
            lambda r: traj_sigma.get((r["problem_id"], r["condition"], r["model"]), {}).get(col, 0.0),
            axis=1
        )

    # Create a stable anchor Y: normalised problem_id ordinal rank (0–1)
    # This is a monotone but arbitrary axis; the PCR inference only uses it
    # to project each point onto the denoised line — what we extract is the
    # denoised X (metric value), not Y.
    uid_map = {uid: i for i, uid in enumerate(sorted(df["problem_id"].unique()))}
    df["_anchor_y"] = df["problem_id"].map(uid_map).astype(float)
    y_max = df["_anchor_y"].max()
    if y_max > 0:
        df["_anchor_y"] /= y_max
    sig_y_global = np.ones(len(df)) * 0.05  # small, stable anchor uncertainty

    for col in available:
        X_obs = df[col].values.astype(float)
        sig_x = df[f"{col}_sigma"].values.astype(float)
        Y_obs = df["_anchor_y"].values

        # Skip if metric has very little variance (constant column)
        if np.nanstd(X_obs) < 1e-8:
            df[f"{col}_pcr"] = X_obs
            continue

        # Replace NaN with column mean for fitting, track mask
        nan_mask = np.isnan(X_obs)
        col_mean = np.nanmean(X_obs)
        X_filled = np.where(nan_mask, col_mean, X_obs)
        sig_x_filled = np.maximum(np.where(nan_mask, np.nanstd(X_obs), sig_x), 1e-4)

        try:
            cr = CloudRegressor()
            cr.fit(X_filled, Y_obs, sig_x_filled, sig_y_global)
            denoised = cr.get_denoised_x()
            denoised[nan_mask] = np.nan
            df[f"{col}_pcr"] = denoised
        except Exception as e:
            print(f"  PCR failed for {col}: {e}. Using raw values.")
            df[f"{col}_pcr"] = df[col]

    df.drop(columns=["_anchor_y"], inplace=True)
    return df


# ─────────────────────────────────────────────
# ANOVA (2-Way: Regime × Correctness)
# ─────────────────────────────────────────────

def run_anova(df: pd.DataFrame, metric: str) -> Optional[dict]:
    try:
        formula = f'Q("{metric}") ~ C(condition) * C(correct)'
        model = ols(formula, data=df.dropna(subset=[metric, "condition", "correct"])).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        ss_total = aov["sum_sq"].sum()
        if ss_total < 1e-12:
            return None
        return {
            "eta_sq_regime": aov.loc["C(condition)", "sum_sq"] / ss_total,
            "eta_sq_quality": aov.loc["C(correct)", "sum_sq"] / ss_total,
            "eta_sq_interaction": aov.loc["C(condition):C(correct)", "sum_sq"] / ss_total,
            "f_regime": aov.loc["C(condition)", "F"],
            "p_regime": aov.loc["C(condition)", "PR(>F)"],
            "f_quality": aov.loc["C(correct)", "F"],
            "p_quality": aov.loc["C(correct)", "PR(>F)"],
        }
    except Exception:
        return None


def cohens_d_from_groups(g1: np.ndarray, g2: np.ndarray) -> float:
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    pooled_std = np.sqrt(((len(g1) - 1) * np.var(g1, ddof=1) +
                          (len(g2) - 1) * np.var(g2, ddof=1)) /
                         (len(g1) + len(g2) - 2))
    if pooled_std < 1e-12:
        return np.nan
    return (np.mean(g1) - np.mean(g2)) / pooled_std


# ─────────────────────────────────────────────
# Main Analysis Pipeline
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", default="experiments/EXP-19_Robustness_2026-02-14")
    args = parser.parse_args()

    data_dir = os.path.join(args.ssd_root, "data")
    out_dir = os.path.join(data_dir, "analysis_19b")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("EXP-19/19B PCR-Corrected Decomposition Analysis")
    print("=" * 65)

    # ── Load base data ──────────────────────────────────────────
    metrics_all = pd.read_csv(os.path.join(data_dir, "all_metrics.csv"))
    pairwise_all = pd.read_csv(os.path.join(data_dir, "exhaustive_pairwise.csv"))

    model_keys = ["qwen05b", "qwen15b"]
    raw_auc_comparison = []
    all_pcr_anova = []

    for m_key in model_keys:
        print(f"\n{'-' * 50}")
        print(f"Model: {m_key}")
        print(f"{'-' * 50}")

        meta_path = os.path.join(data_dir, m_key, "metadata.csv")
        if not os.path.exists(meta_path):
            print(f"  No metadata.csv for {m_key}, skipping.")
            continue

        m_meta = pd.read_csv(meta_path)
        m_metrics = metrics_all[metrics_all["model"] == m_key].copy()

        # Merge correctness labels
        merged = pd.merge(
            m_metrics,
            m_meta[["problem_id", "condition", "correct", "group"]],
            on=["problem_id", "condition"],
            how="inner",
        )
        print(f"  Rows after merge: {len(merged)} | Groups: {sorted(merged['group'].unique())}")

        # ── PCR Denoising ────────────────────────────────────────
        print(f"  Applying PCR denoising to {len(METRIC_COLS)} metrics...")
        denoised = denoise_features_pcr(merged, METRIC_COLS)

        # Export denoised metrics
        pcr_path = os.path.join(out_dir, f"pcr_denoised_metrics_{m_key}.csv")
        denoised.to_csv(pcr_path, index=False)
        print(f"  Saved: {pcr_path}")

        layers = sorted(denoised["layer"].unique())
        pcr_metric_cols = [f"{c}_pcr" for c in METRIC_COLS if f"{c}_pcr" in denoised.columns]
        all_metric_cols = [c for c in METRIC_COLS if c in denoised.columns]

        # ── Analysis 1B: 2-Way ANOVA on PCR features ────────────
        print("  Analysis 1B: 2-Way ANOVA on PCR-denoised features...")
        for layer in layers:
            l_df = denoised[denoised["layer"] == layer]
            if len(l_df["group"].unique()) < 4:
                continue
            for col in pcr_metric_cols:
                res = run_anova(l_df, col)
                if res:
                    res.update({"model": m_key, "layer": layer,
                                "metric": col.replace("_pcr", ""), "denoised": True})
                    all_pcr_anova.append(res)
            for col in all_metric_cols:
                res = run_anova(l_df, col)
                if res:
                    res.update({"model": m_key, "layer": layer,
                                "metric": col, "denoised": False})
                    all_pcr_anova.append(res)

        # ── Analysis 2B: Layer Localization (PCR Cohen's d) ──────
        print("  Analysis 2B: Layer localization with PCR Cohen's d...")
        m_pairwise = pairwise_all[pairwise_all["model"] == m_key]
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle(f"PCR-Corrected Effect Localization: {m_key}", fontsize=15)

        for idx, base_metric in enumerate(KEY_METRICS):
            ax = axes[idx // 2, idx % 2]
            raw_col = base_metric
            pcr_col = f"{base_metric}_pcr"
            if raw_col not in denoised.columns or pcr_col not in denoised.columns:
                continue

            regime_d_raw, regime_d_pcr = [], []
            cot_qual_d_raw, cot_qual_d_pcr = [], []
            dir_qual_d_raw, dir_qual_d_pcr = [], []

            for layer in layers:
                l_df = denoised[denoised["layer"] == layer]
                g1 = l_df[l_df["group"] == "G1"][raw_col].values
                g2 = l_df[l_df["group"] == "G2"][raw_col].values
                g3 = l_df[l_df["group"] == "G3"][raw_col].values
                g4 = l_df[l_df["group"] == "G4"][raw_col].values
                g1p = l_df[l_df["group"] == "G1"][pcr_col].values
                g2p = l_df[l_df["group"] == "G2"][pcr_col].values
                g3p = l_df[l_df["group"] == "G3"][pcr_col].values
                g4p = l_df[l_df["group"] == "G4"][pcr_col].values

                regime_d_raw.append(abs(cohens_d_from_groups(g3, g1)))
                regime_d_pcr.append(abs(cohens_d_from_groups(g3p, g1p)))
                cot_qual_d_raw.append(abs(cohens_d_from_groups(g4, g3)))
                cot_qual_d_pcr.append(abs(cohens_d_from_groups(g4p, g3p)))
                dir_qual_d_raw.append(abs(cohens_d_from_groups(g2, g1)))
                dir_qual_d_pcr.append(abs(cohens_d_from_groups(g2p, g1p)))

            ax.plot(layers, regime_d_raw, "k--", alpha=0.4, label="Regime (Raw)")
            ax.plot(layers, regime_d_pcr, "k-", lw=2, label="Regime (PCR)")
            ax.plot(layers, cot_qual_d_raw, color="red", alpha=0.4, ls="--", label="CoT Quality (Raw)")
            ax.plot(layers, cot_qual_d_pcr, color="red", lw=2, label="CoT Quality (PCR)")
            ax.plot(layers, dir_qual_d_raw, color="blue", alpha=0.4, ls="--")
            ax.plot(layers, dir_qual_d_pcr, color="blue", lw=2, label="Dir Quality (PCR)")
            ax.set_title(base_metric, fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel("|Cohen's d|")
            if idx == 0:
                ax.legend(fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        loc_path = os.path.join(out_dir, f"pcr_effect_localization_{m_key}.png")
        plt.savefig(loc_path, dpi=300)
        plt.close()
        print(f"  Saved: {loc_path}")

        # ── Analysis 3B: G3 Position Index (PCR means) ──────────
        print("  Analysis 3B: G3 Position Index on PCR-denoised means...")
        pos_results = []
        for layer in layers:
            l_df = denoised[denoised["layer"] == layer]
            for col in pcr_metric_cols:
                base = col.replace("_pcr", "")
                g1_mean = l_df[l_df["group"] == "G1"][col].mean()
                g3_mean = l_df[l_df["group"] == "G3"][col].mean()
                g4_mean = l_df[l_df["group"] == "G4"][col].mean()
                denom = g4_mean - g1_mean
                if abs(denom) > 1e-9:
                    pos_idx = (g3_mean - g1_mean) / denom
                    pos_results.append({"layer": layer, "metric": base, "pos_index": pos_idx})

        pos_df = pd.DataFrame(pos_results)
        if not pos_df.empty:
            pivot = pos_df.pivot(index="metric", columns="layer", values="pos_index")
            plt.figure(figsize=(14, 10))
            sns.heatmap(pivot, center=0.5, cmap="RdBu_r", vmin=-0.5, vmax=1.5)
            plt.title(f"PCR G3 Position Index Heatmap ({m_key})\n0=Like Failure (G1), 1=Like Success (G4)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pcr_position_heatmap_{m_key}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(8, 5))
            sns.histplot(pos_df["pos_index"].dropna(), bins=40, kde=True)
            plt.axvline(0.5, color="red", ls="--", label="Midpoint (0.5)")
            plt.title(f"PCR G3 Position Index Distribution ({m_key})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pcr_position_hist_{m_key}.png"), dpi=300)
            plt.close()
            print(f"  G3 mean PCR position index: {pos_df['pos_index'].mean():.3f}")

        # ── Analysis 4B: Interaction Signatures (PCR) ───────────
        print("  Analysis 4B: Interaction Signatures on PCR Cohen's d...")
        interactions = []
        for layer in layers:
            l_df = denoised[denoised["layer"] == layer]
            for col in pcr_metric_cols:
                base = col.replace("_pcr", "")
                g1 = l_df[l_df["group"] == "G1"][col].values
                g2 = l_df[l_df["group"] == "G2"][col].values
                g3 = l_df[l_df["group"] == "G3"][col].values
                g4 = l_df[l_df["group"] == "G4"][col].values

                cd_dir = cohens_d_from_groups(g2, g1)
                cd_cot = cohens_d_from_groups(g4, g3)

                if np.isnan(cd_dir) or np.isnan(cd_cot):
                    continue
                # Simple magnitude threshold for sign-flip detection
                if np.sign(cd_dir) != np.sign(cd_cot) and abs(cd_dir) > 0.3 and abs(cd_cot) > 0.3:
                    interactions.append({
                        "layer": layer, "metric": base,
                        "d_direct_pcr": cd_dir, "d_cot_pcr": cd_cot,
                    })

        inter_df = pd.DataFrame(interactions)
        inter_path = os.path.join(out_dir, f"pcr_interaction_signatures_{m_key}.csv")
        inter_df.to_csv(inter_path, index=False)
        print(f"  PCR interaction signatures: {len(inter_df)} found → {inter_path}")

        # ── Analysis 5B: PCR-AUC vs Raw-AUC ─────────────────────
        if m_key == "qwen05b":
            print("  Analysis 5B: PCR-AUC vs Raw-AUC comparison (qwen05b)...")

            # Identify top 5 quality-predictive metrics by CoT CohenD magnitude
            cot_d_by_metric = {}
            for col in pcr_metric_cols:
                ds = []
                for layer in layers:
                    l_df = denoised[denoised["layer"] == layer]
                    g3 = l_df[l_df["group"] == "G3"][col].values
                    g4 = l_df[l_df["group"] == "G4"][col].values
                    d = abs(cohens_d_from_groups(g4, g3))
                    if not np.isnan(d):
                        ds.append(d)
                cot_d_by_metric[col] = np.mean(ds) if ds else 0.0

            top5_pcr = sorted(cot_d_by_metric, key=lambda k: cot_d_by_metric[k], reverse=True)[:5]
            top5_raw = [c.replace("_pcr", "") for c in top5_pcr]
            print(f"  Top 5 metrics (PCR): {top5_pcr}")

            auc_results = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scaler = StandardScaler()

            for layer in layers:
                l_df = denoised[denoised["layer"] == layer].copy()
                cot_df = l_df[l_df["condition"] == "cot"]

                if len(cot_df["correct"].unique()) < 2:
                    continue

                y = cot_df["correct"].astype(int).values

                def get_auc(df_sub, feature_cols):
                    avail = [f for f in feature_cols if f in df_sub.columns]
                    if len(avail) == 0:
                        return np.nan
                    X = df_sub[avail].values
                    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(df_sub["correct"].values)
                    X_v, y_v = X[valid], df_sub["correct"].astype(int).values[valid]
                    if len(np.unique(y_v)) < 2 or X_v.shape[0] < 10:
                        return np.nan
                    X_s = scaler.fit_transform(X_v)
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    try:
                        scores = cross_val_score(clf, X_s, y_v, cv=skf, scoring="roc_auc")
                        return scores.mean()
                    except Exception:
                        return np.nan

                auc_raw = get_auc(cot_df, top5_raw)
                auc_pcr = get_auc(cot_df, top5_pcr)

                auc_results.append({
                    "layer": layer,
                    "auc_raw": auc_raw,
                    "auc_pcr": auc_pcr,
                    "auc_gain": (auc_pcr - auc_raw) if not (np.isnan(auc_raw) or np.isnan(auc_pcr)) else np.nan,
                })

            auc_df = pd.DataFrame(auc_results)
            auc_path = os.path.join(out_dir, "pcr_auc_comparison_qwen05b.csv")
            auc_df.to_csv(auc_path, index=False)
            raw_auc_comparison = auc_results

            # Plot AUC comparison
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            axes[0].plot(auc_df["layer"], auc_df["auc_raw"], "k--", lw=2, label="Raw AUC (attenuated)")
            axes[0].plot(auc_df["layer"], auc_df["auc_pcr"], "r-", lw=2, label="PCR AUC (corrected)")
            axes[0].axhline(0.5, color="gray", ls=":", alpha=0.5, label="Random chance")
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("AUC (CoT only, 5-fold CV)")
            axes[0].set_title(f"Within-Regime Quality Prediction AUC: Raw vs PCR-Corrected\n(Qwen-0.5B)")
            axes[0].legend()
            axes[0].set_ylim(0.4, 1.0)

            valid = auc_df["auc_gain"].notna()
            axes[1].bar(auc_df.loc[valid, "layer"], auc_df.loc[valid, "auc_gain"],
                        color="steelblue", alpha=0.8)
            axes[1].axhline(0, color="black", lw=0.8)
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("ΔAUC (PCR − Raw)")
            axes[1].set_title("AUC Gain from PCR Denoising (attenuation correction)")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "pcr_auc_comparison.png"), dpi=300)
            plt.close()
            print(f"  Saved: {auc_path}")
            print(f"  Peak raw AUC:  {auc_df['auc_raw'].max():.4f}")
            print(f"  Peak PCR AUC:  {auc_df['auc_pcr'].max():.4f}")
            print(f"  Mean AUC gain: {auc_df['auc_gain'].mean():.4f}")

    # ── Save combined ANOVA results ──────────────────────────────
    anova_df = pd.DataFrame(all_pcr_anova)
    anova_path = os.path.join(out_dir, "pcr_anova_decomposition.csv")
    anova_df.to_csv(anova_path, index=False)
    print(f"\nSaved combined ANOVA → {anova_path}")

    # ── Plot ANOVA variance decomposition comparison ─────────────
    if not anova_df.empty:
        for m_key in model_keys:
            m_anova = anova_df[anova_df["model"] == m_key]
            if m_anova.empty:
                continue
            for denoised_flag, label in [(False, "Raw"), (True, "PCR")]:
                sub = m_anova[m_anova["denoised"] == denoised_flag]
                if sub.empty:
                    continue
                agg = sub.groupby("layer")[["eta_sq_regime", "eta_sq_quality", "eta_sq_interaction"]].mean()

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.stackplot(
                    agg.index,
                    agg["eta_sq_regime"],
                    agg["eta_sq_quality"],
                    agg["eta_sq_interaction"],
                    labels=["Regime η²", "Quality η²", "Interaction η²"],
                    colors=["#2196F3", "#F44336", "#FF9800"],
                    alpha=0.85,
                )
                ax.set_xlabel("Layer")
                ax.set_ylabel("Proportion of Variance Explained")
                ax.set_title(f"Variance Decomposition ({label}) — {m_key}")
                ax.legend(loc="upper right")
                ax.set_ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"pcr_variance_decomp_{label.lower()}_{m_key}.png"), dpi=300)
                plt.close()

    print("\n" + "=" * 65)
    print("PCR analysis complete. All outputs saved to:", out_dir)
    print("=" * 65)


if __name__ == "__main__":
    main()
