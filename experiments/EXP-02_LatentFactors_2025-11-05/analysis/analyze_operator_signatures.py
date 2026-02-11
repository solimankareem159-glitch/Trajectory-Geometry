import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import ruptures as rpt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
INPUT_FILE = "operator_prompts.json"
FIGURE_DIR = "figures_signatures"

def ensure_dir(d):
    try:
        os.makedirs(d)
    except FileExistsError:
        pass

def load_model():
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        output_hidden_states=True, 
        output_attentions=True,
        device_map="auto"
    )
    return model, tokenizer

def extract_internals(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    hidden_states = outputs.hidden_states
    # Stack: (num_layers, seq_len, hidden_dim)
    hs_all = torch.stack([h.squeeze(0).cpu() for h in hidden_states])
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    safe_tokens = [t.encode('ascii', 'replace').decode('ascii') for t in tokens]
    
    return hs_all, safe_tokens

def compute_warp_signal(hs_all):
    # Delta(t) = ||h(t) - h(t-1)|| for each layer
    # Shape: (num_layers, seq_len - 1, hidden_dim)
    deltas = hs_all[:, 1:, :] - hs_all[:, :-1, :]
    
    # Norms: (num_layers, seq_len - 1)
    warp_norms = torch.norm(deltas, p=2, dim=-1).float().numpy()
    
    # Aggregated Warp Signal: Mean across all layers
    # Shape: (seq_len - 1,)
    mean_warp = np.mean(warp_norms, axis=0)
    
    return mean_warp, warp_norms

def detect_change_points(signal, pen=3):
    # PELT search method
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=pen)
    return result

def main():
    ensure_dir(FIGURE_DIR)
    
    with open(INPUT_FILE, "r") as f:
        prompts = json.load(f)
        
    model, tokenizer = load_model()
    
    all_data = [] # Store stats for cross-prompt comparison
    
    for i, item in enumerate(prompts):
        op_name = item['operator']
        prompt = item['full_prompt']
        print(f"\nProcessing {op_name}...")
        
        hs_all, tokens = extract_internals(model, tokenizer, prompt)
        mean_warp, warp_norms = compute_warp_signal(hs_all)
        
        # Change point detection
        bkps = detect_change_points(mean_warp, pen=2)
        
        # 1. Plot Warp Curve + Segmentation
        plt.figure(figsize=(10, 4))
        plt.plot(mean_warp, label='Mean Warp Trace', color='blue')
        
        # Overlay change points
        for bkp in bkps[:-1]: # Last bkp is always end of signal
            plt.axvline(x=bkp-0.5, color='red', linestyle='--', alpha=0.7, label='Change Point' if bkp == bkps[0] else "")
            
        plt.title(f"Operator: {op_name} - Warp Trace & Segmentation")
        plt.xlabel("Token Position (Offset by 1)")
        plt.ylabel("Warp Magnitude")
        plt.legend()
        
        # X-ticks
        valid_len = len(mean_warp)
        if valid_len < 30:
            plt.xticks(range(valid_len), tokens[1:], rotation=45, ha='right', fontsize=8)
            
        plt.tight_layout()
        safe_op_name = op_name.replace(" ", "_").replace("'", "")
        plt.savefig(f"{FIGURE_DIR}/warp_segment_{safe_op_name}.png")
        plt.close()
        
        # Store data for aggregate analysis
        all_data.append({
            "operator": op_name,
            "mean_warp": mean_warp,
            "warp_norms": warp_norms, # (layers, tokens)
            "bkps": bkps,
            "tokens": tokens
        })

    # 2. Compare Signatures
    # We will compute a fixed-size signature for the *first span* (often the operator part)
    # Strategy: Average warp vector of the first N tokens or first segment
    
    signatures = []
    labels = []
    
    print("\nExtracting Signatures...")
    for item in all_data:
        # Heuristic: Take the warp trace of the first 10 tokens (or length if shorter)
        # This captures the "shape" of the initial warp
        cutoff = min(15, len(item['mean_warp']))
        sig = item['mean_warp'][:cutoff]
        
        # Pad if short (unlikely for 10)
        if len(sig) < 15:
            sig = np.pad(sig, (0, 15 - len(sig)), 'constant')
            
        signatures.append(sig)
        labels.append(item['operator'])

    signatures = np.array(signatures)
    
    # Projection
    if len(signatures) > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(signatures)
        
        plt.figure(figsize=(8, 8))
        for j, label in enumerate(labels):
            plt.scatter(coords[j, 0], coords[j, 1], label=label, s=100)
            plt.text(coords[j, 0]+0.02, coords[j, 1]+0.02, label, fontsize=9)
            
        plt.title("Operator Warp Signatures (PCA Projection of First 15 Tokens)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/span_projection.png")
        plt.close()
        
    print(f"Analysis complete. Figures saved in {FIGURE_DIR}/")

if __name__ == "__main__":
    main()
