import numpy as np
import json
import os
from tqdm import tqdm

def main():
    map_path = "experiments/Experiment 16/data/truncation_map.json"
    src_dir = "experiments/Experiment 16/data/hidden_states"
    dst_dir = "experiments/Experiment 16/data/hidden_states_clean"
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    with open(map_path, 'r') as f:
        trunc_map = json.load(f)
        
    print(f"Loaded truncation map for {len(trunc_map)} samples.")
    
    count = 0
    skipped = 0
    
    for filename in tqdm(os.listdir(src_dir)):
        if not filename.endswith(".npy"):
            continue
            
        name = filename.replace(".npy", "")
        if name not in trunc_map:
            skipped += 1
            continue
            
        trunc_idx = trunc_map[name]
        
        # Load hidden state
        src_path = os.path.join(src_dir, filename)
        hidden = np.load(src_path)
        
        # Shape: (layers, tokens, dim)
        # Truncate tokens (second dimension)
        
        # We enforce a floor of 4 tokens if possible, or skip if the true reasoning was shorter than that?
        # Actually, if the answer was found in 1 token, we still want to keep the prompt tokens?
        # Wait, the hidden states are generated sequences ONLY (it was output_hidden_states[input_len:]).
        # So it's just the generated tokens.
        
        if trunc_idx < 4:
            # If it's less than 4, we might not get good curvature/dim results.
            # But let's keep it if that's all the model did.
            # Actually, most CoT should be longer.
            pass
            
        clean_hidden = hidden[:, :trunc_idx, :]
        
        dst_path = os.path.join(dst_dir, filename)
        np.save(dst_path, clean_hidden)
        count += 1
        
    print(f"Truncated {count} files. Skipped {skipped} files.")

if __name__ == "__main__":
    main()
