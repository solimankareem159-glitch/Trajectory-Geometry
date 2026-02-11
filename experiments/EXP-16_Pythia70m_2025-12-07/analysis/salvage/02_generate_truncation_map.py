import pandas as pd
import json
import os
from transformers import AutoTokenizer

def main():
    metadata_path = "experiments/Experiment 16/data/metadata_reparsed.csv"
    output_path = "experiments/Experiment 16/data/truncation_map.json"
    model_name = "Qwen/Qwen2.5-1.5B"
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows.")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    truncation_map = {}
    
    for _, row in df.iterrows():
        pid = row['problem_id']
        cond = row['condition']
        full_response = str(row['response'])
        boundary_char = int(row['truncation_boundary_char'])
        
        # We need to find which token index the boundary_char corresponds to
        # Qwen2.5 tokenizer uses byte-level BPE. 
        # The most reliable way is to encode the full text and find the token prefix that matches the clean text
        
        full_tokens = tokenizer.encode(full_response, add_special_tokens=False)
        clean_text = full_response[:boundary_char]
        clean_tokens = tokenizer.encode(clean_text, add_special_tokens=False)
        
        # The truncation index is the length of the clean tokens
        # We cap at 32 because the hidden states only saved 32 tokens
        trunc_idx = min(len(clean_tokens), 32)
        
        # We need to ensure we have at least a few tokens for metrics
        # (Though if the answer is too short, we might have to drop it anyway)
        
        truncation_map[f"{pid}_{cond}"] = trunc_idx
        
    with open(output_path, 'w') as f:
        json.dump(truncation_map, f, indent=4)
        
    print(f"Truncation map saved to {output_path}")

if __name__ == "__main__":
    main()
