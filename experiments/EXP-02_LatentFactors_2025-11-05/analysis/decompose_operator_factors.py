import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.decomposition import NMF

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
INPUT_FILE = "composite_prompts.json"
FIGURE_DIR = "figures_factors"

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

def get_signature(model, tokenizer, prompt, cutoff=15):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    hidden_states = outputs.hidden_states
    hs_all = torch.stack([h.squeeze(0).cpu() for h in hidden_states])
    
    # Warp Signal (Mean over layers)
    deltas = hs_all[:, 1:, :] - hs_all[:, :-1, :]
    warp_norms = torch.norm(deltas, p=2, dim=-1).float().numpy()
    mean_warp = np.mean(warp_norms, axis=0)
    
    # truncate/pad
    sig = mean_warp[:cutoff]
    if len(sig) < cutoff:
        sig = np.pad(sig, (0, cutoff - len(sig)), 'constant')
        
    return sig

def main():
    ensure_dir(FIGURE_DIR)
    
    with open(INPUT_FILE, "r") as f:
        prompts = json.load(f)
        
    model, tokenizer = load_model()
    
    signatures = []
    labels = []
    types = []
    
    print("\nExtracting Signatures...")
    for item in prompts:
        op_name = item['operator']
        prompt = item['full_prompt']
        p_type = item['type'] # single or composite
        
        sig = get_signature(model, tokenizer, prompt)
        signatures.append(sig)
        labels.append(op_name)
        types.append(p_type)
        
    signatures = np.array(signatures)
    
    # 1. NMF Decomposition
    # We fit NMF on the *Single* operators to find the "basis" of operator space
    
    single_indices = [i for i, t in enumerate(types) if t == 'single']
    composite_indices = [i for i, t in enumerate(types) if t == 'composite']
    
    X_single = signatures[single_indices]
    X_composite = signatures[composite_indices]
    
    # Let's assume there are ~3-4 latent geometric primitives
    n_components = 4
    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=2000)
    
    W_single = nmf.fit_transform(X_single)
    H = nmf.components_ # The latent factors (shape: n_components x 15)
    
    # Transform composite signatures using the BASIS learned from singles
    W_composite = nmf.transform(X_composite)
    
    # Reconstruct
    X_composite_reconstructed = np.dot(W_composite, H)
    
    # 2. Visualization
    
    # A. Latent Factors
    plt.figure(figsize=(10, 5))
    for i in range(n_components):
        plt.plot(H[i], label=f"Factor {i+1}")
    plt.title("Latent Geometric Factors (Learned from Single Operators)")
    plt.xlabel("Token Position")
    plt.ylabel("Warp Magnitude Contribution")
    plt.legend()
    plt.savefig(f"{FIGURE_DIR}/latent_factors.png")
    plt.close()
    
    # B. Activations Heatmap (Composite)
    # Shows which factors are active for each composite operator
    comp_labels = [labels[i] for i in composite_indices]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(W_composite, yticklabels=comp_labels, xticklabels=[f"F{i+1}" for i in range(n_components)], cmap="viridis", annot=True, fmt=".2f")
    plt.title("Latent Factor Activations for Composite Operators")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/composite_activations.png")
    plt.close()
    
    # C. Reconstruction Quality
    # Plot Original vs Reconstructed for a few
    for idx, (real, recon) in enumerate(zip(X_composite, X_composite_reconstructed)):
        name = comp_labels[idx]
        plt.figure(figsize=(10, 4))
        plt.plot(real, label='Original Warp Trace', color='blue')
        plt.plot(recon, label='NMF Reconstruction', color='red', linestyle='--')
        plt.title(f"Reconstruction: {name}")
        plt.legend()
        safe_name = name.replace(" ", "_").replace("+", "plus")
        plt.savefig(f"{FIGURE_DIR}/recon_{safe_name}.png")
        plt.close()
        
    print(f"Analysis complete. Figures saved in {FIGURE_DIR}/")

if __name__ == "__main__":
    main()
