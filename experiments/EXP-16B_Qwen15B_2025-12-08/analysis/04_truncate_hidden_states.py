"""
Experiment 16B: Token-Level Hidden State Truncation
====================================================
This script performs surgical truncation of hidden states to separate:
1. CLEAN: Reasoning phase (up to answer completion)
2. HALLUCINATION: Post-answer phase (worksheet continuation)

Strategy:
- Load reparsed dataset to get ground truth answer endpoints
- Tokenize each response to map character positions to token indices
- Truncate hidden state tensors at the answer boundary
- Save both clean and hallucination portions separately
"""

import os
import json
import numpy as np
import re
from transformers import AutoTokenizer
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATA_DIR = r"experiments/Experiment 16B/data"
REPARSED_FILE = os.path.join(DATA_DIR, "exp16b_dataset_reparsed.jsonl")
HIDDEN_DIR = os.path.join(DATA_DIR, "hidden_states")
CLEAN_DIR = os.path.join(DATA_DIR, "hidden_states_clean")
HALLUCINATION_DIR = os.path.join(DATA_DIR, "hidden_states_hallucination")
TRUNCATION_LOG = os.path.join(DATA_DIR, "truncation_log.json")

print("=" * 70)
print("Experiment 16B: Token-Level Hidden State Truncation")
print("=" * 70)

# Create output directories
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(HALLUCINATION_DIR, exist_ok=True)

# Load tokenizer
print(f"\n[1/4] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load reparsed dataset
print(f"\n[2/4] Loading reparsed dataset...")
dataset = []
with open(REPARSED_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line))
print(f"Loaded {len(dataset)} samples")

# --- Helper Functions ---
def find_answer_endpoint(text, truth_value):
    """
    Find the character index where the answer ends.
    Returns the position AFTER the answer statement.
    """
    # Patterns that indicate answer completion
    # We look for patterns like:
    # - "The answer is 388."
    # - "answer: 388"
    # - "= 388"
    # - Final standalone number on its own line
    
    patterns = [
        rf'[Tt]he answer is {truth_value}\.',
        rf'[Aa]nswer:\s*{truth_value}',
        rf'=\s*{truth_value}\b',
        rf'{truth_value}\s*\.\s*$',  # Number at end with period
    ]
    
    latest_match_end = None
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if latest_match_end is None or match.end() > latest_match_end:
                latest_match_end = match.end()
    
    # If we found a match, return it
    if latest_match_end is not None:
        return latest_match_end
    
    # Fallback: Find the last occurrence of the truth value
    # and assume answer ends shortly after
    last_idx = text.rfind(str(truth_value))
    if last_idx != -1:
        # Find next sentence boundary (period, newline, or end)
        for i in range(last_idx, len(text)):
            if text[i] in '.\\n':
                return i + 1
        return len(text)  # If no boundary, use end of text
    
    # Ultimate fallback: No answer found, return full text
    return len(text)

def tokenize_and_find_boundary(text, char_boundary):
    """
    Tokenize text and find which token index corresponds to char_boundary.
    Returns token index.
    """
    # Tokenize the full text
    full_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = full_tokens['input_ids'][0].tolist()
    
    # Tokenize up to the boundary
    truncated_text = text[:char_boundary]
    truncated_tokens = tokenizer(truncated_text, return_tensors="pt", add_special_tokens=False)
    truncated_token_ids = truncated_tokens['input_ids'][0].tolist()
    
    return len(truncated_token_ids)

# --- Main Processing ---
print(f"\n[3/4] Processing hidden states...")
truncation_log = []
stats = {
    'direct_skipped': 0,
    'cot_truncated': 0,
    'cot_no_hallucination': 0,
    'cot_failed_truncation': 0
}

for sample in tqdm(dataset, desc="Truncating"):
    problem_id = sample['id']
    
    # === Process DIRECT (usually no hallucination, but handle if present) ===
    direct_path = os.path.join(HIDDEN_DIR, f"problem_{problem_id:03d}_direct.npy")
    
    if os.path.exists(direct_path):
        direct_hidden = np.load(direct_path)  # Shape: [28, n_tokens, 1536]
        
        # Direct responses are typically clean, so just copy them
        # But we'll still check for hallucination
        if sample['direct']['hallucination']:
            # Truncate direct as well
            direct_text = sample['direct']['response']
            char_boundary = find_answer_endpoint(direct_text, sample['truth'])
            token_boundary = tokenize_and_find_boundary(direct_text, char_boundary)
            
            clean_direct = direct_hidden[:, :token_boundary, :]
            np.save(os.path.join(CLEAN_DIR, f"problem_{problem_id:03d}_direct.npy"), clean_direct)
        else:
            # No hallucination, copy entire sequence
            np.save(os.path.join(CLEAN_DIR, f"problem_{problem_id:03d}_direct.npy"), direct_hidden)
    
    # === Process CoT (likely has hallucination) ===
    cot_path = os.path.join(HIDDEN_DIR, f"problem_{problem_id:03d}_cot.npy")
    
    if not os.path.exists(cot_path):
        continue
    
    cot_hidden = np.load(cot_path)  # Shape: [28, n_tokens, 1536]
    cot_text = sample['cot']['response']
    
    if sample['cot']['hallucination']:
        # Find answer endpoint in text
        char_boundary = find_answer_endpoint(cot_text, sample['truth'])
        
        # Map to token index
        try:
            token_boundary = tokenize_and_find_boundary(cot_text, char_boundary)
            
            # Truncate
            clean_cot = cot_hidden[:, :token_boundary, :]
            hallucination_cot = cot_hidden[:, token_boundary:, :]
            
            # Save both portions
            np.save(os.path.join(CLEAN_DIR, f"problem_{problem_id:03d}_cot.npy"), clean_cot)
            np.save(os.path.join(HALLUCINATION_DIR, f"problem_{problem_id:03d}_cot.npy"), hallucination_cot)
            
            # Log
            truncation_log.append({
                'id': problem_id,
                'char_boundary': char_boundary,
                'token_boundary': token_boundary,
                'total_tokens': cot_hidden.shape[1],
                'clean_tokens': token_boundary,
                'hallucination_tokens': cot_hidden.shape[1] - token_boundary,
                'hallucination_type': sample['cot']['hallucination_type']
            })
            
            stats['cot_truncated'] += 1
            
        except Exception as e:
            print(f"\nWARNING: Failed to truncate problem {problem_id}: {e}")
            # Fallback: Save full sequence as clean (conservative)
            np.save(os.path.join(CLEAN_DIR, f"problem_{problem_id:03d}_cot.npy"), cot_hidden)
            stats['cot_failed_truncation'] += 1
    else:
        # No hallucination detected, save entire sequence as clean
        np.save(os.path.join(CLEAN_DIR, f"problem_{problem_id:03d}_cot.npy"), cot_hidden)
        stats['cot_no_hallucination'] += 1

# Save truncation log
print(f"\n[4/4] Saving truncation log...")
with open(TRUNCATION_LOG, 'w', encoding='utf-8') as f:
    json.dump(truncation_log, f, indent=2)

# === Summary ===
print("\n" + "=" * 70)
print("TRUNCATION SUMMARY")
print("=" * 70)
print(f"CoT Sequences Truncated:        {stats['cot_truncated']}")
print(f"CoT Sequences (No Hallucination): {stats['cot_no_hallucination']}")
print(f"CoT Failed Truncation:          {stats['cot_failed_truncation']}")

if truncation_log:
    avg_clean = np.mean([log['clean_tokens'] for log in truncation_log])
    avg_hallu = np.mean([log['hallucination_tokens'] for log in truncation_log])
    print(f"\nAverage Clean Tokens:           {avg_clean:.1f}")
    print(f"Average Hallucination Tokens:   {avg_hallu:.1f}")

print(f"\nOutputs:")
print(f"  Clean States:        {CLEAN_DIR}/")
print(f"  Hallucination States: {HALLUCINATION_DIR}/")
print(f"  Truncation Log:      {TRUNCATION_LOG}")
print("=" * 70)
