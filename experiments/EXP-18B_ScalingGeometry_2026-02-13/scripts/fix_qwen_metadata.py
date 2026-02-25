
import json
import pandas as pd
import os

# Paths
ROOT_DIR = r"c:\Dev\Projects\Trajectory Geometry"
QWEN_15B_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-16B_Qwen15B_2025-12-08", "data")
JSONL_FILE = os.path.join(QWEN_15B_DIR, "exp16b_dataset_reparsed.jsonl")
CSV_FILE = os.path.join(QWEN_15B_DIR, "metadata_reparsed.csv")

print(f"Reading {JSONL_FILE}...")
data = []
with open(JSONL_FILE, 'r') as f:
    for line in f:
        item = json.loads(line)
        problem_id = item['id']
        truth = item['truth']
        
        # Add direct entry
        data.append({
            'problem_id': problem_id,
            'condition': 'direct',
            'ground_truth': str(truth),
            'correct': item['direct'].get('correct', False)
        })
        
        # Add CoT entry
        data.append({
            'problem_id': problem_id,
            'condition': 'cot',
            'ground_truth': str(truth),
            'correct': item['cot'].get('correct', False)
        })

df = pd.DataFrame(data)
df.to_csv(CSV_FILE, index=False)
print(f"Saved {len(df)} rows to {CSV_FILE}")
