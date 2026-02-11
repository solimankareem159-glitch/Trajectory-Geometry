
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
    print("Experiment 16 Salvage v3 (Fixed): Full Pythia-70m Run (300 Problems)")
    print("Capturing 8 layers: embed_in, layers[0-5], final_layer_norm")
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
    output_dir = "experiments/Experiment 16/data/pythia_salvage_v3"
    os.makedirs(output_dir, exist_ok=True)
    hs_dir = os.path.join(output_dir, "hidden_states")
    os.makedirs(hs_dir, exist_ok=True)
    
    metadata_list = []
    
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
                max_new = 120

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            curr_ids = input_ids
            
            gen_ids = []
            layers_data = [[] for _ in range(8)]
            
            # Hooks
            step_states = []
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    val = output[0]
                else:
                    val = output
                # During manual loop, each step has input length = previous_total
                # But we only want the LAST token's hidden state.
                step_states.append(val[:, -1, :].detach().cpu().numpy())

            hooks = []
            hooks.append(model.gpt_neox.embed_in.register_forward_hook(hook_fn))
            for layer in model.gpt_neox.layers:
                hooks.append(layer.register_forward_hook(hook_fn))
            hooks.append(model.gpt_neox.final_layer_norm.register_forward_hook(hook_fn))

            with torch.no_grad():
                for i in range(max_new):
                    step_states.clear()
                    outputs = model(curr_ids)
                    
                    # step_states now has 8 hidden states for the last token
                    # Only collect for the first 32 tokens of the response
                    if i < 32:
                        for l in range(8):
                            # step_states[l] has shape (1, 512)
                            layers_data[l].append(step_states[l][0, :])
                    
                    # Next token
                    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                    gen_ids.append(next_token_id.item())
                    curr_ids = torch.cat([curr_ids, next_token_id.unsqueeze(0)], dim=1)
                    
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
            
            for h in hooks:
                h.remove()
                
            # Extract response
            res_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            parsed = parse_numeric_answer(res_text)
            correct = (parsed == truth)
            
            if len(gen_ids) > 0:
                # layers_data shape is (8, actual_tokens, 512)
                hs_array = np.array(layers_data)
                np.save(os.path.join(hs_dir, f"prob_{id:03d}_{condition}.npy"), hs_array)

            metadata_list.append({
                'problem_id': id,
                'condition': condition,
                'prompt': prompt,
                'truth': truth,
                'response': res_text,
                'parsed': parsed,
                'correct': correct,
                'response_length_tokens': len(gen_ids),
                'n_layers': 8 if len(gen_ids) > 0 else 0
            })

    # Save metadata
    with open(os.path.join(output_dir, "pythia_salvage_dataset.json"), 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\nSalvage v3 complete. Data saved to {output_dir}")

if __name__ == "__main__":
    run_salvage()
