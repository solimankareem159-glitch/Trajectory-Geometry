"""
Experiment 16B: Truncation Validation
======================================
Inspect a sample of truncated hidden states to verify:
1. Clean sequences end at appropriate answer boundaries
2. Hallucination sequences start after answer completion
3. Token counts are sensible
"""

import os
import json
import numpy as np
import random

# Configuration
DATA_DIR = r"experiments/Experiment 16B/data"
REPARSED_FILE = os.path.join(DATA_DIR, "exp16b_dataset_reparsed.jsonl")
TRUNCATION_LOG = os.path.join(DATA_DIR, "truncation_log.json")
CLEAN_DIR = os.path.join(DATA_DIR, "hidden_states_clean")
HALLUCINATION_DIR = os.path.join(DATA_DIR, "hidden_states_hallucination")

# Load data
with open(REPARSED_FILE, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

with open(TRUNCATION_LOG, 'r', encoding='utf-8') as f:
    truncation_log = json.load(f)

print("=" * 70)
print("TRUNCATION VALIDATION")
print("=" * 70)

# Sample 5 random truncated sequences
sample_ids = random.sample([entry['id'] for entry in truncation_log], min(5, len(truncation_log)))

for problem_id in sample_ids:
    sample = dataset[problem_id]
    log_entry = next(e for e in truncation_log if e['id'] == problem_id)
    
    print(f"\n--- PROBLEM {problem_id} ---")
    print(f"Question: {sample['question']}")
    print(f"Truth: {sample['truth']}")
    print(f"Parsed: {sample['cot']['parsed']}")
    print(f"Correct: {sample['cot']['correct']}")
    
    # Show truncation point
    response = sample['cot']['response']
    char_boundary = log_entry['char_boundary']
    
    print(f"\n[CLEAN SECTION] ({log_entry['clean_tokens']} tokens):")
    print(response[:char_boundary])
    
    print(f"\n[HALLUCINATION SECTION] ({log_entry['hallucination_tokens']} tokens):")
    print(response[char_boundary:char_boundary+200] + "...")
    
    # Load and check shapes
    clean_path = os.path.join(CLEAN_DIR, f"problem_{problem_id:03d}_cot.npy")
    hallu_path = os.path.join(HALLUCINATION_DIR, f"problem_{problem_id:03d}_cot.npy")
    
    clean_shape = np.load(clean_path).shape
    hallu_shape = np.load(hallu_path).shape
    
    print(f"\nClean Hidden States Shape: {clean_shape}")
    print(f"Hallucination Hidden States Shape: {hallu_shape}")

print("\n" + "=" * 70)
print("Summary Statistics:")
print("=" * 70)

clean_lengths = []
hallu_lengths = []

for entry in truncation_log:
    clean_lengths.append(entry['clean_tokens'])
    hallu_lengths.append(entry['hallucination_tokens'])

print(f"Clean Token Lengths:")
print(f"  Mean: {np.mean(clean_lengths):.1f}")
print(f"  Median: {np.median(clean_lengths):.1f}")
print(f"  Min/Max: {np.min(clean_lengths)} / {np.max(clean_lengths)}")

print(f"\nHallucination Token Lengths:")
print(f"  Mean: {np.mean(hallu_lengths):.1f}")
print(f"  Median: {np.median(hallu_lengths):.1f}")
print(f"  Min/Max: {np.min(hallu_lengths)} / {np.max(hallu_lengths)}")

print(f"\nHallucination Type Distribution:")
hallu_types = {}
for entry in truncation_log:
    htype = entry['hallucination_type']
    hallu_types[htype] = hallu_types.get(htype, 0) + 1

for htype, count in sorted(hallu_types.items(), key=lambda x: -x[1]):
    print(f"  {htype}: {count}")

print("=" * 70)
