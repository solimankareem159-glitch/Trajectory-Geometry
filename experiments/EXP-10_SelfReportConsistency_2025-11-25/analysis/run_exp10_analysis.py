
import json
import os
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import random

# --- Configuration ---
DATA_DIR = r"experiments/Experiment 10/data"
DATA_FILENAME = "exp10_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 10/results"
REPORT_FILENAME = "exp10_report.md"
TARGET_LAYERS = [10, 11, 12, 13, 14, 15, 16] # Mid-layers

# --- Load Data ---
def load_data():
    path = os.path.join(DATA_DIR, DATA_FILENAME)
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data

def get_mid_layer_metric(rec, metric_name):
    # Average across mid-layers
    vals = []
    for l in TARGET_LAYERS:
        l_str = str(l)
        if l_str in rec["geometry"]:
             vals.append(rec["geometry"][l_str][metric_name])
    if not vals: 
        return 0
    return np.mean(vals)

def evaluate_prediction(features, labels):
    # features: (N, D), labels: (N,)
    # 5-fold CV Logistic Regression
    if len(np.unique(labels)) < 2:
        return {"auc": 0.5, "acc": 0.5, "f1": 0}
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = []
    probs = []
    true_labels = []
    
    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        clf = LogisticRegression(class_weight='balanced', solver='liblinear')
        clf.fit(X_train, y_train)
        
        preds.extend(clf.predict(X_test))
        probs.extend(clf.predict_proba(X_test)[:, 1])
        true_labels.extend(y_test)
        
    return {
        "auc": roc_auc_score(true_labels, probs),
        "acc": balanced_accuracy_score(true_labels, preds),
        "f1": f1_score(true_labels, preds, average='macro')
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    data = load_data()
    print(f"Loaded {len(data)} samples.")
    
    # Filter valid self-reports
    valid_data = [d for d in data if d["self_report_immediate"]["parsed"] is not None]
    print(f"Samples with valid self-reports: {len(valid_data)}")
    
    if len(valid_data) < 5:
        print("Not enough data.")
        return

    # --- Analysis 1: Correlations ---
    
    # Features to correlate
    # Self-Report: effort, certainty, exploration, smoothness
    # Geometry: speed, stabilization, dir_consistency, curvature_early
    
    correlations = []
    
    # Arrays
    sr_effort = np.array([d["self_report_immediate"]["parsed"]["effort"] for d in valid_data])
    sr_certainty = np.array([d["self_report_immediate"]["parsed"]["certainty"] for d in valid_data])
    sr_exploration = np.array([d["self_report_immediate"]["parsed"]["exploration"] for d in valid_data])
    sr_smoothness = np.array([d["self_report_immediate"]["parsed"]["smoothness"] for d in valid_data])
    
    # Perturbed Arrays (Use parsed if available, else skip or handle)
    # Check if perturbed parsing worked
    valid_data_p = [d for d in valid_data if d["self_report_perturbed"]["parsed"] is not None]
    has_perturb = len(valid_data_p) > 10
    
    if has_perturb:
        sr_p_effort = np.array([d["self_report_perturbed"]["parsed"]["effort"] for d in valid_data_p])
        sr_p_certainty = np.array([d["self_report_perturbed"]["parsed"]["certainty"] for d in valid_data_p])
        sr_p_exploration = np.array([d["self_report_perturbed"]["parsed"]["exploration"] for d in valid_data_p])
        sr_p_smoothness = np.array([d["self_report_perturbed"]["parsed"]["smoothness"] for d in valid_data_p])
    
    # Geometry (Mid-Layers)
    geo_speed = np.array([get_mid_layer_metric(d, "speed") for d in valid_data])
    geo_stab = np.array([get_mid_layer_metric(d, "stabilization") for d in valid_data])
    geo_dc = np.array([get_mid_layer_metric(d, "dir_consistency") for d in valid_data])
    geo_curv = np.array([get_mid_layer_metric(d, "curvature_early") for d in valid_data])
    
    if has_perturb:
         # Need aligned geometry for perturbed subset (which might be same)
         geo_p_speed = np.array([get_mid_layer_metric(d, "speed") for d in valid_data_p])
         geo_p_stab = np.array([get_mid_layer_metric(d, "stabilization") for d in valid_data_p])
         geo_p_dc = np.array([get_mid_layer_metric(d, "dir_consistency") for d in valid_data_p])
         geo_p_curv = np.array([get_mid_layer_metric(d, "curvature_early") for d in valid_data_p])

    # Define Pairs
    pairs = [
        ("Effort", "Speed", sr_effort, geo_speed, sr_p_effort if has_perturb else None, geo_p_speed if has_perturb else None),
        ("Certainty", "Stabilization", sr_certainty, geo_stab, sr_p_certainty if has_perturb else None, geo_p_stab if has_perturb else None),
        ("Exploration", "Neg_DirCons", sr_exploration, -geo_dc, sr_p_exploration if has_perturb else None, -geo_p_dc if has_perturb else None),
        ("Smoothness", "Neg_Curvature", sr_smoothness, -geo_curv, sr_p_smoothness if has_perturb else None, -geo_p_curv if has_perturb else None)
    ]
    
    report_lines = ["# Experiment 10: Self-Report Consistency Findings\n"]
    report_lines.append("## 1. Correlations with Trajectory Geometry\n")
    report_lines.append("| Self-Report | Metric | Pearson r (Imm) | p | Pearson r (Pert) | Retention | Result |")
    report_lines.append("|---|---|---|---|---|---|---|")
    
    pass_criteria = False
    
    for name_sr, name_geo, imm_sr, imm_geo, per_sr, per_geo in pairs:
        # Immediate
        r_imm, p_imm = stats.pearsonr(imm_sr, imm_geo)
        
        # Perturbed
        r_per, retention = 0, 0
        if has_perturb:
            # Check length/variance
            if len(per_sr) > 5 and np.std(per_sr) > 0 and np.std(per_geo) > 0:
                r_per, p_per = stats.pearsonr(per_sr, per_geo)
                retention = abs(r_per) / (abs(r_imm) + 1e-9)
            else:
                r_per = 0
                retention = 0
                
        sig = "**SIGNIFICANT**" if p_imm < 0.05 and abs(r_imm) >= 0.3 else ""
        if sig: pass_criteria = True
        
        report_lines.append(f"| {name_sr} | {name_geo} | {r_imm:.3f} | {p_imm:.3f} | {r_per:.3f} | {retention:.2f} | {sig} |")
        
    # --- Analysis 2: Success Prediction ---
    report_lines.append("\n## 2. Predicting Success from Self-Report\n")
    
    labels = np.array([1 if d["is_correct"] else 0 for d in valid_data])
    n_succ = np.sum(labels)
    report_lines.append(f"Success Rate: {n_succ}/{len(labels)}\n")
    
    # Features: [Effort, Certainty, Exploration, Smoothness]
    X_sr = np.vstack([sr_effort, sr_certainty, sr_exploration, sr_smoothness]).T
    
    # Baseline: Length
    lens = np.array([len(d["cot_response"]) for d in valid_data]).reshape(-1, 1)
    
    # Baseline: Random
    rand = np.random.rand(len(labels), 1)
    
    res_sr = evaluate_prediction(X_sr, labels)
    res_len = evaluate_prediction(lens, labels)
    res_rand = evaluate_prediction(rand, labels)
    
    report_lines.append("| Model | AUC | Balanced Acc | Macro F1 |")
    report_lines.append("|---|---|---|---|")
    report_lines.append(f"| Self-Report Ratings Only | {res_sr['auc']:.3f} | {res_sr['acc']:.3f} | {res_sr['f1']:.3f} |")
    report_lines.append(f"| Response Length Baseline | {res_len['auc']:.3f} | {res_len['acc']:.3f} | {res_len['f1']:.3f} |")
    report_lines.append(f"| Random Baseline | {res_rand['auc']:.3f} | {res_rand['acc']:.3f} | {res_rand['f1']:.3f} |")
    
    # --- Conclusion ---
    report_lines.append("\n## Conclusion\n")
    if pass_criteria:
        report_lines.append("PASS: Significant correlations found between self-reports and trajectory geometry.")
    else:
        report_lines.append("FAIL/INCONCLUSIVE: No strong correlations found.")
        
    # Save Report
    with open(os.path.join(OUTPUT_DIR, REPORT_FILENAME), 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Analysis Complete. Report saved to {REPORT_FILENAME}")

if __name__ == "__main__":
    main()
