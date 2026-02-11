
import os
import sys
import json
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Import device helper
sys.path.insert(0, os.path.dirname(__file__))
UTILS_PATH = "experiments/Experiment 16/scripts"
if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

from exp16_utils import get_generation_device, set_seed

def parse_numeric_answer(text):
    """Extract numeric answer stoping at typical runaway points."""
    stop_patterns = ["Q:", "Question:", "A:", "\n\n"]
    clean_text = text
    for pattern in stop_patterns:
        if pattern in clean_text:
            clean_text = clean_text.split(pattern)[0]
    
    numbers = re.findall(r'-?\d+', clean_text)
    if numbers:
        return int(numbers[-1])
    return None

def run_salvage():
    print(f"PID: {os.getpid()}", flush=True)
    print("="*60)
    print("Experiment 16 Salvage: Full Pythia-70m Run (300 Problems)")
    print("="*60)
    
    set_seed(42)
    device = get_generation_device()
    print(f"Using device: {device}")
    
    model_name = "EleutherAI/pythia-70m"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Dataset
    DATASET_PATH = "experiments/Experiment 9/data/exp9_dataset.jsonl"
    with open(DATASET_PATH, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Output directory
    output_dir = "experiments/Experiment 16/data/pythia_salvage"
    os.makedirs(output_dir, exist_ok=True)
    hs_dir = os.path.join(output_dir, "hidden_states")
    os.makedirs(hs_dir, exist_ok=True)
    
    results = []
    
    # Hooks for hidden states
    all_hidden_states = []
    def hook(module, input, output):
        # Pythia-70m hidden states are in output[0] if it's a transformer layer
        if isinstance(output, tuple):
            all_hidden_states.append(output[0].detach().cpu().numpy())
        else:
            all_hidden_states.append(output.detach().cpu().numpy())

    # Register hooks on embedding and all 6 transformer layers
    hooks = []
    hooks.append(model.gpt_neox.embed_in.register_forward_hook(hook))
    for layer in model.gpt_neox.layers:
        hooks.append(layer.register_forward_hook(hook))

    for prob in tqdm(dataset[:300]):
        id = prob['id']
        question = prob['question']
        truth = prob['truth']
        
        for condition in ['direct', 'cot']:
            if condition == 'direct':
                prompt = f"Answer the following question directly.\nQ: Calculate {question}\nA:"
                max_new = 32
            else:
                prompt = f"Q: Calculate {question}\nLet's think step by step before answering."
                max_new = 120 # Shorter for Pythia as it's more concise

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            all_hidden_states.clear()
            with torch.no_grad():
                # We want the hidden states of the tokens generated? 
                # Standard project methodology: extract states for the first 32 tokens of the response
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new, 
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
            
            # Extract response
            gen_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            res_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            parsed = parse_numeric_answer(res_text)
            correct = (parsed == truth)
            
            # Extraction of hidden states (methodology match: first 32 tokens of response)
            # outputs.hidden_states is a tuple of length (tokens_generated)
            # each element is a tuple of length (layers+1)
            
            num_gen = len(gen_ids)
            limit = min(num_gen, 32)
            
            if limit > 0:
                # Shape: (layers, tokens, dim)
                # Pythia-70m: 7 layers (0-6), variable tokens, 512 dim
                layers_data = []
                for l in range(7): # 0 (embed) + 6 (layers)
                    token_states = []
                    for t in range(limit):
                        # outputs.hidden_states[t][l] has shape (1, 1, 512)
                        token_states.append(outputs.hidden_states[t][l].cpu().numpy()[0, 0, :])
                    layers_data.append(token_states)
                
                hs_array = np.array(layers_data) # (7, limit, 512)
                np.save(os.path.join(hs_dir, f"prob_{id:03d}_{condition}.npy"), hs_array)

            results.append({
                'problem_id': id,
                'condition': condition,
                'response': res_text,
                'parsed': parsed,
                'correct': correct,
                'response_length_tokens': num_gen
            })

    # Save metadata
    with open(os.path.join(output_dir, "pythia_metadata.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV version
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "pythia_metadata.csv"), index=False)
    
    print("\nSalvage complete.")

if __name__ == "__main__":
    run_salvage()
