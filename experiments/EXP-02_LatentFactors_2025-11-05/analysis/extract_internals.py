import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "Summarize the following text in one sentence: Artificial intelligence is transforming science, medicine, and creativity."
FIGURE_DIR = "figures_internals"

def ensure_dir(d):
    try:
        os.makedirs(d)
    except FileExistsError:
        pass

def main():
    print(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            output_hidden_states=True, 
            output_attentions=True,
            device_map="auto" # Will use CPU if GPU not available
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded.")
    ensure_dir(FIGURE_DIR)

    # 1. Inference
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    # 2. Extract Internals
    # hidden_states: tuple of (batch, seq_len, hidden_dim). Length = num_layers + 1 (embedding + layers)
    hidden_states = outputs.hidden_states 
    
    # Stack layers: (num_layers, seq_len, hidden_dim)
    # We strip batch dim since it's 1
    hs_all = torch.stack([h.squeeze(0).cpu() for h in hidden_states])
    
    num_layers, seq_len, hidden_dim = hs_all.shape
    print(f"Extracted Hidden States. Shape: {hs_all.shape} (Layers x Tokens x Dim)")
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    safe_tokens = [t.encode('ascii', 'replace').decode('ascii') for t in tokens]
    print(f"Tokens ({len(tokens)}): {safe_tokens}")

    # 3. Compute Warp Traces
    # Delta(t) = ||h(t) - h(t-1)||
    # We can compute this for all layers and tokens > 0
    
    # Shifted difference
    # hs_all[:, 1:, :] - hs_all[:, :-1, :]
    deltas = hs_all[:, 1:, :] - hs_all[:, :-1, :]
    
    # Norms: (num_layers, seq_len - 1)
    warp_norms = torch.norm(deltas, p=2, dim=-1)
    
    # Make numpy for plotting
    warp_norms_np = warp_norms.float().numpy()
    
    # 4. Visualization
    
    # A. Heatmap (Layers x Tokens)
    plt.figure(figsize=(12, 8))
    # X-axis labels: tokens (starting from 2nd token, since 1st has no delta)
    # We might need to truncate labels if too many
    xtick_labels = tokens[1:] 
    
    sns.heatmap(warp_norms_np, xticklabels=xtick_labels, cmap="viridis")
    plt.title(f"Warp Trace Heatmap: {MODEL_NAME}\n(Norm of Delta between consecutive tokens)")
    plt.xlabel("Token (transition from t-1)")
    plt.ylabel("Layer Index")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/warp_heatmap.png")
    plt.close()
    
    # B. Line Plot (Selected Layers)
    plt.figure(figsize=(12, 6))
    
    # visual layers: [0, ~L/4, ~L/2, ~3L/4, L-1]
    layer_indices = [
        0,
        num_layers // 4,
        num_layers // 2,
        3 * num_layers // 4,
        num_layers - 1
    ]
    # Filter unique and valid
    layer_indices = sorted(list(set([i for i in layer_indices if i < num_layers])))
    
    for l_idx in layer_indices:
        plt.plot(warp_norms_np[l_idx], label=f"Layer {l_idx}", marker='o', markersize=3, alpha=0.7)
        
    plt.title(f"Warp Trace Magnitudes per Layer: {MODEL_NAME}")
    plt.xlabel("Token Position")
    plt.ylabel("Warp Magnitude ||Delta||")
    plt.xticks(ticks=range(len(xtick_labels)), labels=xtick_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/warp_trace_lines.png")
    plt.close()
    
    print("Plots generated in figures_internals/")
    print("\n--- Summary ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Hidden States Shape: {hs_all.shape}")
    print(f"Warp Matrix Shape: {warp_norms_np.shape}")
    print("Future Work: Hook operator segmentation logic to align with these token positions.")

if __name__ == "__main__":
    main()
