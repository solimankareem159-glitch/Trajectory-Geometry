import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
DATA_FILE = "experiments/03_regime_invariants/data/regime_traces.npz"
FIGURE_DIR = "experiments/03_regime_invariants/results/figures_regime_invariants"
REPORT_FILE = "experiments/03_regime_invariants/results/REGIME_INVARIANTS_REPORT.md"

def ensure_dir(d):
    try:
        os.makedirs(d)
    except FileExistsError:
        pass

def main():
    ensure_dir(FIGURE_DIR)
    
    # 1. Load Data
    print(f"Loading data from {DATA_FILE}...")
    data = np.load(DATA_FILE, allow_pickle=True)
    H = data['H'] # [N, L, T, D]
    warp = data['warp'] # [N, L, T-1]
    operator_labels = data['operator_labels']
    topic_labels = data['topic_labels']
    
    N, L, T, D = H.shape
    print(f"Loaded {N} samples with {L} layers, {T} tokens, {D} dim.")
    
    # 2. Define Regimes
    # listening (0-7), thinking (8-15), speaking (16-24)
    # hs is [N, L, T, D]
    z_listen = np.mean(np.mean(H[:, 0:8, :, :], axis=1), axis=1) # [N, D]
    z_think = np.mean(np.mean(H[:, 8:16, :, :], axis=1), axis=1) # [N, D]
    z_speak = np.mean(np.mean(H[:, 16:, :, :], axis=1), axis=1) # [N, D]
    
    regimes = {
        "Listen": z_listen,
        "Think": z_think,
        "Speak": z_speak
    }
    
    # 3. Probing for Operators
    print("Training operator probes...")
    le = LabelEncoder()
    y = le.fit_transform(operator_labels)
    classes = le.classes_
    
    results = {}
    
    for name, X in regimes.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = {"acc": acc, "f1": f1}
        print(f"{name} Regime -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title(f"Confusion Matrix - {name} Regime")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/cm_{name.lower()}.png")
        plt.close()

    # 4. Intent Lock-in (Time-resolved Probing)
    print("Running time-resolved probing...")
    time_accs = {"Listen": [], "Think": [], "Speak": []}
    
    for t in range(5, T, 5): # Every 5 tokens
        for name, layers_range in [("Listen", (0, 8)), ("Think", (8, 16)), ("Speak", (16, 24))]:
            # Cumulative mean up to t
            Z_t = np.mean(np.mean(H[:, layers_range[0]:layers_range[1], :t, :], axis=1), axis=1)
            X_train, X_test, y_train, y_test = train_test_split(Z_t, y, test_size=0.2, random_state=42, stratify=y)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            time_accs[name].append(acc)
            
    plt.figure(figsize=(10, 6))
    x_axis = range(5, T, 5)
    for name, accs in time_accs.items():
        plt.plot(x_axis, accs, label=name, marker='o')
    plt.title("Operator Intent Lock-in over Time")
    plt.xlabel("Token Index")
    plt.ylabel("Probing Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGURE_DIR}/intent_lockin.png")
    plt.close()

    # 5. Topic Leakage
    print("Testing topic leakage...")
    y_topic = LabelEncoder().fit_transform(topic_labels)
    topic_leakage = {}
    for name, X in regimes.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y_topic, test_size=0.2, random_state=42, stratify=y_topic)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        topic_leakage[name] = accuracy_score(y_test, clf.predict(X_test))
        print(f"{name} Topic Prediction Accuracy: {topic_leakage[name]:.4f}")

    # 6. Coupling Maps
    print("Learning coupling maps...")
    # F: Listen -> Think
    map_f = LinearRegression()
    map_f.fit(z_listen, z_think)
    z_think_hat = map_f.predict(z_listen)
    r2_f = map_f.score(z_listen, z_think)
    
    # G: Think -> Speak
    map_g = LinearRegression()
    map_g.fit(z_think, z_speak)
    z_speak_hat = map_g.predict(z_think)
    r2_g = map_g.score(z_think, z_speak)
    
    print(f"Coupling Map F (Listen->Think) R2: {r2_f:.4f}")
    print(f"Coupling Map G (Think->Speak) R2: {r2_g:.4f}")
    
    # Decode from mapped
    _, X_test_mapped, _, y_test = train_test_split(z_think_hat, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    # Re-train on partial or full? user asked "accuracy of F(z_listen) compared to z_listen"
    # To be fair, train on mapped and test on mapped
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(z_think_hat, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train_m, y_train_m)
    acc_mapped = accuracy_score(y_test_m, clf.predict(X_test_m))
    print(f"Mapped Operator Accuracy (F(Listen)->Think): {acc_mapped:.4f}")

    # 7. Shared Invariant Space (CCA)
    print("Learning shared invariant space (CCA)...")
    cca = CCA(n_components=10)
    # Align Listen and Speak
    u_listen, u_speak = cca.fit_transform(z_listen, z_speak)
    
    # Probing in CCA space
    X_train, X_test, y_train, y_test = train_test_split(u_listen, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc_cca = accuracy_score(y_test, clf.predict(X_test))
    print(f"CCA Space Operator Accuracy (Listen View): {acc_cca:.4f}")

    # 8. Report Generation
    print("Saving report...")
    with open(REPORT_FILE, "w") as f:
        f.write("# Cross-Regime Operator Invariants Report\n\n")
        f.write("## 1. Probing Results\n")
        f.write("| Regime | Accuracy | Macro-F1 |\n")
        f.write("| :--- | :--- | :--- |\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['acc']:.4f} | {res['f1']:.4f} |\n")
            
        f.write("\n## 2. Intent Lock-in\n")
        f.write("![Intent Lock-in](figures_regime_invariants/intent_lockin.png)\n\n")
        
        f.write("## 3. Topic Leakage\n")
        for name, acc in topic_leakage.items():
            f.write(f"- {name} Regime Topic Accuracy: {acc:.4f}\n")
            
        f.write("\n## 4. Coupling Maps\n")
        f.write(f"- F (Listen -> Think) R²: {r2_f:.4f}\n")
        f.write(f"- G (Think -> Speak) R²: {r2_g:.4f}\n")
        f.write(f"- Mapped (F(Listen)) Operator Acc: {acc_mapped:.4f}\n")
        
        f.write("\n## 5. Invariant Space (CCA)\n")
        f.write(f"- CCA Alignment Accuracy: {acc_cca:.4f}\n")

    print("Analysis complete.")

if __name__ == "__main__":
    main()
