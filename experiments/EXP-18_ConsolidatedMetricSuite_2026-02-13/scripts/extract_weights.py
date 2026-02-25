
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = r"experiments/EXP-18_ConsolidatedMetricSuite_2026-02-13/Data"

def main():
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    
    # Extract matrices
    # Qwen uses lm_head for unembedding
    # wte for input embedding
    w_u = model.lm_head.weight.detach().cpu().numpy() # (V, D)
    w_e = model.model.embed_tokens.weight.detach().cpu().numpy() # (V, D)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, "unembedding.npy"), w_u)
    np.save(os.path.join(OUTPUT_DIR, "embedding.npy"), w_e)
    
    # Save tokenizer config if needed, but we mainly need it to encode tokens
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    
    print(f"Matrices saved to {OUTPUT_DIR}")
    print(f"Unembedding shape: {w_u.shape}")
    print(f"Embedding shape: {w_e.shape}")

if __name__ == "__main__":
    main()
