import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import glob

REPORT_FILE = "PREDICTOR_REPORT.md"
INPUT_PATTERN = "labeled_warps_*.parquet"

class GRUPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        # hidden: [1, batch, hidden_dim]
        return self.fc(hidden.squeeze(0))

def run_prediction_analysis():
    files = glob.glob(INPUT_PATTERN)
    if not files:
        print(f"No labeled warps found matching {INPUT_PATTERN}. Run cluster_warps.py first.")
        return

    report_lines = []
    report_lines.append("# Predictor Analysis Report")
    report_lines.append("")

    for input_file in files:
        # Extract state_def from filename
        # labeled_warps_{s_def}.parquet
        basename = os.path.basename(input_file)
        s_def = basename.replace("labeled_warps_", "").replace(".parquet", "")
        
        print(f"Analyzing State Definition: {s_def}")
        report_lines.append(f"## State Definition: {s_def}")
        
        df = pd.read_parquet(input_file)
        
        # We need to construct sequences.
        # Group by Topic (our "conversation" unit)
        topics = df['topic'].unique()
        
        # Check available labels
        if 'cluster_hdb' not in df.columns:
            print("No cluster labels found.")
            continue
            
        # Use K-Means labels if available (stable k=10), else HDBSCAN
        label_col = 'cluster_kmeans' if 'cluster_kmeans' in df.columns else 'cluster_hdb'
        print(f"Using labels from: {label_col}")
        
        # Filter out noise (-1) if using HDBSCAN
        valid_mask = df[label_col] != -1
        df_clean = df[valid_mask].copy()
        
        # Encode labels to 0..N-1
        le = LabelEncoder()
        df_clean['encoded_label'] = le.fit_transform(df_clean[label_col])
        vocab_size = len(le.classes_)
        
        sequences = []
        for topic in topics:
            topic_df = df_clean[df_clean['topic'] == topic].sort_values('turn_id')
            seq = topic_df['encoded_label'].values
            if len(seq) > 5:
                sequences.append(seq)
                
        # Train/Test Split (Leave-One-Topic-Out would be ideal, but let's just do last topic as test for simplicity or 80/20)
        # We have 5 topics. Let's use 4 for train, 1 for test.
        train_seqs = sequences[:-1]
        test_seqs = sequences[-1:]
        
        report_lines.append(f"Train Topics: {len(train_seqs)}, Test Topics: {len(test_seqs)}")
        
        # --- 1. Markov-1 Baseline ---
        # Count transitions P(next | curr)
        counts = np.zeros((vocab_size, vocab_size))
        for seq in train_seqs:
            for t in range(len(seq) - 1):
                curr, next_val = seq[t], seq[t+1]
                counts[curr, next_val] += 1
                
        # Normalize
        probs = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)
        
        # Predict on Test
        y_true = []
        y_pred = []
        
        for seq in test_seqs:
            for t in range(len(seq) - 1):
                curr, next_val = seq[t], seq[t+1]
                # Predict argmax
                pred_token = np.argmax(probs[curr])
                y_true.append(next_val)
                y_pred.append(pred_token)
                
        acc_markov = accuracy_score(y_true, y_pred)
        f1_markov = f1_score(y_true, y_pred, average='macro')
        
        report_lines.append("### Baseline (Markov-1)")
        report_lines.append(f"- Accuracy: {acc_markov:.3f}")
        report_lines.append(f"- Macro-F1: {f1_markov:.3f}")
        
        # --- 2. Sequence Model (GRU) ---
        # Prepare sliding windows N=5
        N = 5
        
        def create_dataset(seqs, n_steps):
            X, Y = [], []
            for seq in seqs:
                for i in range(len(seq) - n_steps):
                    X.append(seq[i : i+n_steps])
                    Y.append(seq[i+n_steps])
            return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)
            
        train_X, train_Y = create_dataset(train_seqs, N)
        test_X, test_Y = create_dataset(test_seqs, N)
        
        model = GRUPredictor(vocab_size, embedding_dim=16, hidden_dim=32, output_dim=vocab_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        epochs = 50 
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_X)
            loss = criterion(outputs, train_Y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            outputs = model(test_X)
            _, predicted = torch.max(outputs.data, 1)
            
        acc_gru = accuracy_score(test_Y.numpy(), predicted.numpy())
        f1_gru = f1_score(test_Y.numpy(), predicted.numpy(), average='macro')
        
        report_lines.append(f"### Sequence Model (GRU, N={N})")
        report_lines.append(f"- Accuracy: **{acc_gru:.3f}**")
        report_lines.append(f"- Macro-F1: **{f1_gru:.3f}**")
        
        # Hypothesis Check
        report_lines.append("\n**Hypothesis Check:**")
        if acc_gru > 0.15: # Random guess for 10 classes is 0.1
            report_lines.append("> ✅ H2 Supported: Predictability > Random Chance.")
        else:
             report_lines.append("> ❌ H2 At Risk: Low predictability.")
             
        if acc_gru > acc_markov:
             report_lines.append("> GRU outperformed Markov Baseline.")
        else:
             report_lines.append("> Markov Baseline outperformed GRU (simple structure).")

        report_lines.append("\n---\n")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Prediction analysis complete. Report written to {REPORT_FILE}")

if __name__ == "__main__":
    run_prediction_analysis()
