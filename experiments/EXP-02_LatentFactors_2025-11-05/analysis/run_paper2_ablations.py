import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.decomposition import NMF

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
INPUT_FILE = "composite_prompts.json"
FIGURE_DIR = "figures_paper2"
METRICS_FILE = "ablation_metrics.json"

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

def get_signature_from_layer(model, tokenizer, prompt, layer_idx=-1, cutoff=15):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    hidden_states = outputs.hidden_states
    # hidden_states is tuple of (batch, seq, dim). len = num_layers+1
    
    # helper to get specific layer or mean
    if layer_idx == 'mean':
        hs_stack = torch.stack([h.squeeze(0).cpu() for h in hidden_states])
        deltas = hs_stack[:, 1:, :] - hs_stack[:, :-1, :]
        warp_norms = torch.norm(deltas, p=2, dim=-1).float().numpy()
        trace = np.mean(warp_norms, axis=0)
    else:
        # specific layer
        # handle negative index
        if layer_idx < 0:
            layer_idx = len(hidden_states) + layer_idx
            
        h = hidden_states[layer_idx].squeeze(0).cpu()
        delta = h[1:, :] - h[:-1, :]
        trace = torch.norm(delta, p=2, dim=-1).float().numpy()
        
    # truncate/pad
    sig = trace[:cutoff]
    if len(sig) < cutoff:
        sig = np.pad(sig, (0, cutoff - len(sig)), 'constant')
        
    return sig

def run_nmf_reconstruction(signatures, labels, types, n_components):
    single_indices = [i for i, t in enumerate(types) if t == 'single']
    composite_indices = [i for i, t in enumerate(types) if t == 'composite']
    
    X_single = signatures[single_indices]
    X_composite = signatures[composite_indices]
    
    if len(X_single) < n_components:
        return None, None
        
    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=2000)
    W_single = nmf.fit_transform(X_single)
    H = nmf.components_
    
    W_composite = nmf.transform(X_composite)
    X_recon = np.dot(W_composite, H)
    
    # Calculate Mean Squared Error
    mse = np.mean((X_composite - X_recon) ** 2)
    return mse, H

def main():
    ensure_dir(FIGURE_DIR)
    
    with open(INPUT_FILE, "r") as f:
        prompts = json.load(f)
        
    model, tokenizer = load_model()
    
    # Pre-compute signatures for 'mean' layer to speed up k-sweep
    print("Extracting mean-layer signatures...")
    mean_signatures = []
    labels = []
    types = []
    for item in prompts:
        sig = get_signature_from_layer(model, tokenizer, item['full_prompt'], layer_idx='mean')
        mean_signatures.append(sig)
        labels.append(item['operator'])
        types.append(item['type'])
    mean_signatures = np.array(mean_signatures)

    metrics = {"k_sweep": {}, "layer_sweep": {}}

    # 1. K-Factor Sweep (k=2 to 8)
    print("\nRunning K-Factor Sweep...")
    k_errors = []
    k_values = range(2, 9)
    
    for k in k_values:
        mse, _ = run_nmf_reconstruction(mean_signatures, labels, types, n_components=k)
        if mse is not None:
            k_errors.append(mse)
            metrics["k_sweep"][k] = float(mse)
            print(f"k={k}, MSE={mse:.4f}")
        else:
            k_errors.append(np.nan)

    plt.figure()
    plt.plot(k_values, k_errors, marker='o')
    plt.title("Reconstruction Error vs. Number of Latent Factors (k)")
    plt.xlabel("k (Number of Factors)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGURE_DIR}/ablation_k_error.png")
    plt.close()

    # 2. Layer Sensitivity Analysis (Early, Mid, Late)
    # We'll use fixed k=4
    print("\nRunning Layer Sensitivity Analysis...")
    layer_indices = [0, 5, 12, 18, 24] # specific layers for qwen 0.5B (24 layers total)
    layer_errors = []
    
    for l_idx in layer_indices:
        print(f"Processing Layer {l_idx}...")
        # Re-extract for this layer
        l_sigs = []
        for item in prompts:
            sig = get_signature_from_layer(model, tokenizer, item['full_prompt'], layer_idx=l_idx)
            l_sigs.append(sig)
        l_sigs = np.array(l_sigs)
        
        mse, _ = run_nmf_reconstruction(l_sigs, labels, types, n_components=4)
        layer_errors.append(mse)
        metrics["layer_sweep"][l_idx] = float(mse) if mse else None

    plt.figure()
    plt.bar([str(l) for l in layer_indices], layer_errors)
    plt.title("Reconstruction Error by Transformer Layer (k=4)")
    plt.xlabel("Layer Index")
    plt.ylabel("MSE")
    plt.savefig(f"{FIGURE_DIR}/layer_sensitivity.png")
    plt.close()
    
    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Ablations complete. Results in {FIGURE_DIR}/ and {METRICS_FILE}")

if __name__ == "__main__":
    main()
