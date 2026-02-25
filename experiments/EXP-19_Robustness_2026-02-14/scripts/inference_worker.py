import os
# Suppress Hugging Face symlink warnings and focus on speed
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Force DirectML to use high performance if available
os.environ["DML_VISIBLE_DEVICES"] = "0" 

import json
from huggingface_hub import login

# Support for HF_TOKEN from environment or .env
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    print("Warning: HF_TOKEN not found in environment. Rate limits may apply.")
import time
import re
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Import DirectML
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

class StopOnSequence(StoppingCriteria):
    """Stop generation when any stop sequence is detected in output."""
    def __init__(self, tokenizer, stop_strings, input_length):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.input_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        for stop in self.stop_strings:
            if stop.lower() in text.lower():
                return True
        return False

def truncate_direct(text):
    # Try to find the first number in the response
    text = text.strip()
    # Handle cases where model repeats the question or includes "Answer is X"
    match = re.search(r'(?:Answer|Result|is|:)\s*(-?\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'(-?\d+)', text)
    if match:
        return match.group(1)
    return text

def truncate_cot(text):
    # Look for the explicit ANSWER tag first
    match = re.search(r'ANSWER:\s*(-?\d+)', text, re.IGNORECASE)
    if match:
        # Keep everything until the end of the answer
        return text[:match.end()]
    
    # Look for common concluding indicators
    match = re.search(r'(?:the|final)\s*(?:answer|result)\s*(?:is|=|:)\s*(-?\d+)', text, re.IGNORECASE)
    if match:
        return text[:match.end()]

    # Fallback: find the last occurrence of a number before any potential restart
    for pattern in [r'\n\s*(?:Question|Calculate|Q:)', r'(\d)\1{5,}']:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break
    return text

def find_clean_token_boundary(generated_tokens, tokenizer, clean_text):
    for i in range(len(generated_tokens), 0, -1):
        decoded = tokenizer.decode(generated_tokens[:i], skip_special_tokens=True)
        if len(decoded.strip()) <= len(clean_text.strip()):
            return i
    return len(generated_tokens)

def parse_answer(text, condition):
    if condition == 'direct':
        clean = truncate_direct(text)
        nums = re.findall(r'-?\d+', clean)
        return int(nums[0]) if nums else None
    elif condition == 'cot':
        clean = truncate_cot(text)
        match = re.search(r'ANSWER:\s*(-?\d+)', clean, re.IGNORECASE)
        if match: return int(match.group(1))
        match = re.search(r'(?:answer|result)\s*(?:is|=|:)\s*(-?\d+)', clean, re.IGNORECASE)
        if match: return int(match.group(1))
        nums = re.findall(r'-?\d+', clean)
        return int(nums[-1]) if nums else None

def validate_response(response, clean, condition):
    issues = []
    if len(response) > len(clean) * 2 and len(response) > 50:
        issues.append('response_much_longer_than_clean')
    if condition == 'direct':
        reasoning_words = ['step', 'first', 'multiply', 'calculate', 'because', 'therefore']
        if any(w in response.lower() for w in reasoning_words):
            issues.append('direct_contains_reasoning')
    if re.search(r'(\d)\1{5,}', response):
        issues.append('digit_repetition')
    # Note: question_restart_after_answer is ignored for now as it is a common 
    # artifact of few-shot prompting and handled by truncation.
    # if re.search(r'(?:Question|Calculate|Q:)', response[len(clean):] if len(response) > len(clean) else ''):
    #     issues.append('question_restart_after_answer')
    return issues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--problems", type=str, required=True)
    parser.add_argument("--ssd_root", type=str, required=True)
    parser.add_argument("--hdd_root", type=str, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    print(f"PID: {os.getpid()}", flush=True)
    
    # Setup Device
    if HAS_DIRECTML:
        device = torch_directml.device()
        print(f"Using DirectML: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {device}")

    # Load Model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        output_hidden_states=True,
        trust_remote_code=True
    ).to(device)
    model.eval()

    # Load Problems
    with open(args.problems, 'r') as f:
        problems = json.load(f)

    # Hybrid Output Setup
    model_ssd_dir = os.path.join(args.ssd_root, "data", args.model_key)
    model_hdd_dir = os.path.join(args.hdd_root, "data", "hidden_states", args.model_key)
    os.makedirs(model_ssd_dir, exist_ok=True)
    os.makedirs(model_hdd_dir, exist_ok=True)

    metadata_path = os.path.join(model_ssd_dir, "metadata.csv")
    
    # Resume Logic
    processed_pids = set()
    if os.path.exists(metadata_path):
        existing_df = pd.read_csv(metadata_path)
        processed_pids = set(existing_df['problem_id'].tolist())
        # Filter for completed direct AND cot
        counts = existing_df.groupby('problem_id').size()
        processed_pids = set(counts[counts >= 2].index.tolist())
    
    print(f"Resuming: {len(processed_pids)} problems already processed.")

    for prob in tqdm(problems, desc=f"Inference {args.model_key}"):
        pid = prob['id']
        if pid in processed_pids:
            continue

        results = []
        for condition in ['direct', 'cot']:
            if condition == 'direct':
                prompt = (
                    "Calculate the following arithmetic problems. Answer with ONLY the numerical result.\n\n"
                    "Question: 3 + 4\nAnswer: 7\n\n"
                    "Question: 10 - 2\nAnswer: 8\n\n"
                    f"Question: {prob['question']}\nAnswer:"
                )
                config = {'max_new_tokens': 15, 'do_sample': False}
                stops = ["\n", "Question", "Calculate", "Let", "Step", "The ", "Because"]
            else:
                prompt = (
                    "Calculate the following arithmetic problems step by step. After your working, write \"ANSWER: \" followed by the number.\n\n"
                    "Question: 3 + 4\nWorking: 3 plus 4 is 7. ANSWER: 7\n\n"
                    "Question: 10 - 2\nWorking: 10 minus 2 is 8. ANSWER: 8\n\n"
                    f"Question: {prob['question']}\nWorking:"
                )
                config = {'max_new_tokens': 200, 'do_sample': False}
                stops = ["Question:", "Calculate:", "Problem:", "\n\nQ:", "\n\nCalculate", "Let me verify", "Alternatively", "Double check"]

            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            input_len = inputs.input_ids.shape[1]
            
            stopping_criteria = StoppingCriteriaList([StopOnSequence(tokenizer, stops, input_len)])

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **config,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )

            gen_tokens = outputs.sequences[0][input_len:]
            full_resp = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            clean_resp = truncate_direct(full_resp) if condition == 'direct' else truncate_cot(full_resp)
            
            n_clean = find_clean_token_boundary(gen_tokens, tokenizer, clean_resp)
            n_clean = max(n_clean, 1)

            # Hidden State Extraction
            n_layers = model.config.num_hidden_layers + 1
            # SAVE FULL TRAJECTORY: Extract all available hidden states generated
            n_gen_tokens = len(outputs.hidden_states)
            
            if n_gen_tokens >= 1:
                # OPTIMIZATION: Batch GPU-to-CPU transfers
                try:
                    # Collect and stack on GPU
                    token_layer_stacks = []
                    for t in range(n_gen_tokens):
                        # outputs.hidden_states[t] is a tuple of L layers
                        # Each layer is [1, 1, D] for the token generated at step t
                        layers_at_t = torch.stack([l[:, -1, :] for l in outputs.hidden_states[t]]).squeeze(1) # [L, D]
                        token_layer_stacks.append(layers_at_t)
                    
                    trajectory_gpu = torch.stack(token_layer_stacks, dim=1) # [L, T, D]
                    trajectory = trajectory_gpu.cpu().float().numpy()
                except Exception as e:
                    print(f"!!! Optimization failed: {e}. Falling back to slow method.")
                    hidden_stack = []
                    for layer_idx in range(n_layers):
                        layer_states = []
                        for t in range(n_gen_tokens):
                            state = outputs.hidden_states[t][layer_idx][0, -1, :].cpu().float().numpy()
                            layer_states.append(state)
                        hidden_stack.append(np.stack(layer_states))
                    trajectory = np.stack(hidden_stack)
                
                # Shape Validation (EXP-18B Lesson)
                if trajectory.shape[-1] != model.config.hidden_size:
                    print(f"!!! Error: Dimension mismatch for {args.model_key}: {trajectory.shape[-1]} != {model.config.hidden_size}")
                    sys.exit(1)

                # Save to HDD
                traj_filename = f"{pid:03d}_{condition}.npy"
                np.save(os.path.join(model_hdd_dir, traj_filename), trajectory.astype(np.float16))
            else:
                traj_filename = None

            parsed = parse_answer(clean_resp, condition)
            correct = (parsed == prob['truth']) if parsed is not None else False
            issues = validate_response(full_resp, clean_resp, condition)
            
            group = ('G2' if correct else 'G1') if condition == 'direct' else ('G4' if correct else 'G3')

            results.append({
                'problem_id': pid,
                'condition': condition,
                'group': group,
                'correct': correct,
                'parsed_answer': parsed,
                'truth': prob['truth'],
                'bin': prob['bin'],
                'op_type': prob['op_type'],
                'response': full_resp,
                'clean_response': clean_resp,
                'n_clean_tokens': n_clean,
                'n_gen_tokens': n_gen_tokens,
                'n_full_tokens': len(gen_tokens),
                'answer_token_idx': n_clean - 1, # 0-indexed boundary for the "clean" answer
                'filename': traj_filename,
                'has_trajectory': traj_filename is not None,
                'issues': "|".join(issues),
                'contaminated': len(issues) > 0
            })

        # Incremental Save to SSD
        df_new = pd.DataFrame(results)
        if not os.path.exists(metadata_path):
            df_new.to_csv(metadata_path, index=False)
        else:
            df_new.to_csv(metadata_path, mode='a', header=False, index=False)

    print(f"Finished processing {args.model_key}")

if __name__ == "__main__":
    import sys
    main()
