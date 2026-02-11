"""
Experiment 16B: Single-Sample Inference with Hidden State Extraction
=====================================================================
Non-batched, exact replication of Experiment 9 methodology.
Monitors for hallucinations and extracts hidden states from all 28 layers.

Environment: DirectML venv (Python 3.12)
"""

import os
import sys
import json
import torch
import torch_directml
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

print(f"PID: {os.getpid()}", flush=True)

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = torch_directml.device()  # DirectML device

# Paths
EXP9_DATASET = r"experiments/Experiment 9/data/exp9_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 16B/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exp16b_dataset.jsonl")
HIDDEN_DIR = os.path.join(OUTPUT_DIR, "hidden_states")

# Experiment 9 EXACT parameters
DIRECT_MAX_TOKENS = 32
COT_MAX_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = None  # Greedy decoding (no temperature)

print("=" * 70)
print("Experiment 16B: Non-Batched Inference with Hallucination Monitoring")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Direct Max Tokens: {DIRECT_MAX_TOKENS}")
print(f"CoT Max Tokens: {COT_MAX_TOKENS}")
print(f"Do Sample: {DO_SAMPLE}")
print(f"Temperature: {TEMPERATURE}")
print("=" * 70)

# --- Load Questions from Experiment 9 ---
print(f"\n[1/4] Loading Experiment 9 questions...")
questions = []
with open(EXP9_DATASET, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        questions.append({
            'id': data['id'],
            'question': data['question'],
            'truth': data['truth']
        })
print(f"Loaded {len(questions)} questions from Experiment 9")

# --- Load Model ---
print(f"\n[2/4] Loading {MODEL_NAME} on DirectML...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded successfully")
print(f"Model has {model.config.num_hidden_layers} layers")

# --- Create Output Directories ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HIDDEN_DIR, exist_ok=True)

# --- Helper Functions ---
def parse_answer(text):
    """Extract numerical answer from response text"""
    # Find all numbers (including negative)
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        try:
            return int(numbers[-1])  # Last number in response
        except:
            pass
    return None

def detect_hallucination(response_text):
    """
    Detect runaway hallucination patterns.
    Returns (is_hallucinating, contamination_type)
    """
    hallucination_patterns = [
        (r'Question:', 'new_question'),
        (r'\nQ:', 'new_question_shorthand'),
        (r'1{5,}|0{7,}', 'large_repeated_digits'),  # e.g., 11111... or 0000000...
        (r'\d{10,}', 'very_large_number'),  # 10+ digit numbers (suspicious)
    ]
    
    for pattern, contamination_type in hallucination_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            return True, contamination_type
    
    return False, None

def extract_hidden_states_single(model, inputs, max_new_tokens):
    """
    Extract hidden states for a SINGLE sample during generation.
    Returns: (output_ids, all_hidden_states)
    all_hidden_states shape: [n_layers, n_generated_tokens, hidden_dim]
    """
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    # outputs.hidden_states is a tuple of tuples:
    # len(outputs.hidden_states) = num_generated_tokens
    # each element is a tuple of (num_layers + 1) tensors [batch=1, 1, hidden_dim]
    
    generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
    n_gen = len(generated_ids)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    
    # Stack hidden states: [n_layers, n_tokens, hidden_dim]
    stacked_states = []
    for layer_idx in range(n_layers):
        layer_states = []
        for token_idx in range(n_gen):
            # outputs.hidden_states[token_idx] is tuple of layer outputs
            # outputs.hidden_states[token_idx][layer_idx+1] gives hidden state (skip embedding layer)
            h = outputs.hidden_states[token_idx][layer_idx + 1][0, 0, :].cpu().numpy()
            layer_states.append(h)
        stacked_states.append(np.array(layer_states, dtype=np.float16))
    
    return generated_ids, np.array(stacked_states, dtype=np.float16)

# --- Main Inference Loop ---
print(f"\n[3/4] Running single-sample inference...")
print("Monitoring for hallucinations during generation...\n")

results = []
hallucination_log = []

for item in tqdm(questions, desc="Processing"):
    problem_id = item['id']
    q_str = item['question']
    truth = item['truth']
    
    # === DIRECT CONDITION ===
    prompt_direct = f"Answer the following question directly.\nQ: Calculate {q_str}\nA:"
    inputs_direct = tokenizer(prompt_direct, return_tensors="pt").to(DEVICE)
    
    gen_ids_direct, hidden_direct = extract_hidden_states_single(
        model, inputs_direct, DIRECT_MAX_TOKENS
    )
    
    response_direct = tokenizer.decode(gen_ids_direct, skip_special_tokens=True)
    parsed_direct = parse_answer(response_direct)
    correct_direct = (parsed_direct == truth)
    
    # Check for hallucination
    is_hallu_direct, hallu_type_direct = detect_hallucination(response_direct)
    if is_hallu_direct:
        hallucination_log.append({
            'id': problem_id,
            'condition': 'direct',
            'type': hallu_type_direct,
            'response': response_direct
        })
    
    # === CoT CONDITION ===
    prompt_cot = f"Q: Calculate {q_str}\nLet's think step by step before answering."
    inputs_cot = tokenizer(prompt_cot, return_tensors="pt").to(DEVICE)
    
    gen_ids_cot, hidden_cot = extract_hidden_states_single(
        model, inputs_cot, COT_MAX_TOKENS
    )
    
    response_cot = tokenizer.decode(gen_ids_cot, skip_special_tokens=True)
    parsed_cot = parse_answer(response_cot)
    correct_cot = (parsed_cot == truth)
    
    # Check for hallucination
    is_hallu_cot, hallu_type_cot = detect_hallucination(response_cot)
    if is_hallu_cot:
        hallucination_log.append({
            'id': problem_id,
            'condition': 'cot',
            'type': hallu_type_cot,
            'response': response_cot
        })
    
    # === Save Hidden States (28 layers × 2 conditions) ===
    filename_direct = f"problem_{problem_id:03d}_direct.npy"
    filename_cot = f"problem_{problem_id:03d}_cot.npy"
    
    np.save(os.path.join(HIDDEN_DIR, filename_direct), hidden_direct)
    np.save(os.path.join(HIDDEN_DIR, filename_cot), hidden_cot)
    
    # === Save Result (Experiment 9 Format) ===
    result = {
        "id": problem_id,
        "question": q_str,
        "truth": truth,
        "direct": {
            "prompt": prompt_direct,
            "response": response_direct,
            "parsed": parsed_direct,
            "correct": correct_direct,
            "hallucination": is_hallu_direct,
            "hallucination_type": hallu_type_direct if is_hallu_direct else None
        },
        "cot": {
            "prompt": prompt_cot,
            "response": response_cot,
            "parsed": parsed_cot,
            "correct": correct_cot,
            "hallucination": is_hallu_cot,
            "hallucination_type": hallu_type_cot if is_hallu_cot else None
        }
    }
    results.append(result)

# --- Save Results ---
print(f"\n[4/4] Saving results...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

# Save hallucination log
hallu_log_path = os.path.join(OUTPUT_DIR, "hallucination_log.json")
with open(hallu_log_path, 'w', encoding='utf-8') as f:
    json.dump(hallucination_log, f, indent=2, ensure_ascii=False)

# === Summary Statistics ===
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

direct_acc = sum(1 for r in results if r['direct']['correct']) / len(results) * 100
cot_acc = sum(1 for r in results if r['cot']['correct']) / len(results) * 100
direct_hallu = sum(1 for r in results if r['direct']['hallucination'])
cot_hallu = sum(1 for r in results if r['cot']['hallucination'])

print(f"Total Problems: {len(results)}")
print(f"\nDirect Accuracy: {direct_acc:.2f}%")
print(f"CoT Accuracy: {cot_acc:.2f}%")
print(f"\nDirect Hallucinations: {direct_hallu} ({direct_hallu/len(results)*100:.1f}%)")
print(f"CoT Hallucinations: {cot_hallu} ({cot_hallu/len(results)*100:.1f}%)")

if hallucination_log:
    print(f"\n⚠️  CONTAMINATION DETECTED: {len(hallucination_log)} instances")
    print(f"See detailed log: {hallu_log_path}")
    
    # Group by type
    hallu_types = {}
    for entry in hallucination_log:
        htype = entry['type']
        hallu_types[htype] = hallu_types.get(htype, 0) + 1
    print("\nHallucination Types:")
    for htype, count in sorted(hallu_types.items(), key=lambda x: -x[1]):
        print(f"  {htype}: {count}")
else:
    print(f"\n✓ NO HALLUCINATIONS DETECTED")

print(f"\nOutputs:")
print(f"  Dataset: {OUTPUT_FILE}")
print(f"  Hidden States: {HIDDEN_DIR}/ (602 .npy files)")
print(f"  Hallucination Log: {hallu_log_path}")
print("=" * 70)
