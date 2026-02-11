"""
Reparse Experiment 16B Dataset
==============================
The original parser failed because Qwen 1.5B continues generating (hallucinating)
new questions after the correct answer. The parser grabbed the last number (garbage).

This script:
1. Loads the generated JSONL.
2. Truncates response at "Question:" or "\nQ:".
3. Re-parses the answer.
4. Calculates TRUE accuracy.
"""

import json
import re
import os

INPUT_FILE = r"experiments/Experiment 16B/data/exp16b_dataset.jsonl"
OUTPUT_FILE = r"experiments/Experiment 16B/data/exp16b_dataset_reparsed.jsonl"

def robust_parse(text):
    """
    Parse answer but STOP processing if we hit multiple newlines 
    or a new Question pattern.
    """
    # 1. Truncate at hallucination boundaries
    stop_patterns = [
        r'\nQuestion:', 
        r'Question:', 
        r'\nQ:',
        r'\n\n\n'
    ]
    
    clean_text = text
    for p in stop_patterns:
        match = re.search(p, clean_text, re.IGNORECASE)
        if match:
            clean_text = clean_text[:match.start()]
            
    # 2. Extract number from clean text
    # Look for "answer is X" or "equals X" or "= X"
    # Use findall to get ALL matches, then take the LAST one
    # This prevents catching intermediate steps like "38 * 9 = 342" as the answer
    answer_matches = re.findall(r'(?:answer is|equals?|=)\s*(-?\d+)', clean_text, re.IGNORECASE)
    if answer_matches:
        return int(answer_matches[-1])
        
    # Fallback: last number in cleaned text
    numbers = re.findall(r'-?\d+', clean_text)
    if numbers:
        return int(numbers[-1])
        
    return None

print("Reparsing dataset...")
reparsed_data = []
cot_correct = 0
direct_correct = 0
total = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        total += 1
        
        # Reparse Direct
        d_parsed = robust_parse(data['direct']['response'])
        d_is_correct = (d_parsed == data['truth'])
        if d_is_correct: direct_correct += 1
        
        # Reparse CoT
        c_parsed = robust_parse(data['cot']['response'])
        c_is_correct = (c_parsed == data['truth'])
        if c_is_correct: cot_correct += 1
        
        # Update Data
        data['direct']['parsed'] = d_parsed
        data['direct']['correct'] = d_is_correct
        data['cot']['parsed'] = c_parsed
        data['cot']['correct'] = c_is_correct
        
        reparsed_data.append(data)

# Save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in reparsed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("="*60)
print(f"Total processed: {total}")
print(f"New Direct Accuracy: {direct_correct/total*100:.2f}%")
print(f"New CoT Accuracy:    {cot_correct/total*100:.2f}%")
print("="*60)
