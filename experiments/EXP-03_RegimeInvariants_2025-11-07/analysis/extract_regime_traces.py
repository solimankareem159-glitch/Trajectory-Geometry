import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
INPUT_FILE = "experiments/03_regime_invariants/data/regime_invariants_prompts.json"
OUTPUT_FILE = "experiments/03_regime_invariants/data/regime_traces.npz"

def load_model():
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        output_hidden_states=True, 
        device_map="auto"
    )
    return model, tokenizer

def main():
    with open(INPUT_FILE, "r") as f:
        prompts_data = json.load(f)
        
    model, tokenizer = load_model()
    
    all_hidden_states = []
    all_warps = []
    all_labels = []
    all_topics = []
    all_token_ids = []
    
    max_t = 0
    
    print(f"Starting extraction for {len(prompts_data)} samples...")
    for i, item in enumerate(prompts_data):
        prompt = item['prompt_text']
        label = item['operator_label']
        topic = item['topic_label']
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # outputs.hidden_states is a tuple of (L+1) tensors, each [1, T, D]
        # Stack into [L+1, T, D]
        hs = torch.stack([h.squeeze(0).cpu() for h in outputs.hidden_states]) # [L, T, D]
        
        L, T, D = hs.shape
        if T > max_t:
            max_t = T
            
        # Warp Energy: ||h_t - h_{t-1}|| per layer
        # deltas: [L, T-1, D]
        deltas = hs[:, 1:, :] - hs[:, :-1, :]
        warps = torch.norm(deltas, p=2, dim=-1) # [L, T-1]
        
        all_hidden_states.append(hs.float().numpy().astype(np.float16)) # compressed storage
        all_warps.append(warps.float().numpy().astype(np.float16))
        all_labels.append(label)
        all_topics.append(topic)
        all_token_ids.append(inputs['input_ids'].squeeze(0).cpu().numpy())
        
        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{len(prompts_data)}...")

    # Padding and packing
    N = len(prompts_data)
    L_count = all_hidden_states[0].shape[0]
    D_dim = all_hidden_states[0].shape[2]
    
    H_padded = np.zeros((N, L_count, max_t, D_dim), dtype=np.float16)
    warp_padded = np.zeros((N, L_count, max_t - 1), dtype=np.float16)
    tokens_padded = np.zeros((N, max_t), dtype=np.int32)
    
    for i in range(N):
        t_len = all_hidden_states[i].shape[1]
        H_padded[i, :, :t_len, :] = all_hidden_states[i]
        warp_padded[i, :, :t_len-1] = all_warps[i]
        tokens_padded[i, :t_len] = all_token_ids[i]
        
    print(f"Saving results to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        H=H_padded,
        warp=warp_padded,
        operator_labels=np.array(all_labels),
        topic_labels=np.array(all_topics),
        token_ids=tokens_padded
    )
    
    print("Done.")

if __name__ == "__main__":
    main()
