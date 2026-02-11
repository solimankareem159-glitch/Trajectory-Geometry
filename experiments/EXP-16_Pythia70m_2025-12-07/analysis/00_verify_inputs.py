"""
Script 00: Verify Inputs
========================
Validates the exp9_dataset.jsonl file exists and has correct schema.
"""

import os
import sys
import json

def find_dataset():
    """Locate the exp9_dataset.jsonl file."""
    # Primary path
    primary = "experiments/Experiment 9/data/exp9_dataset.jsonl"
    if os.path.exists(primary):
        return primary
    
    # Search experiments/ directory
    print(f"Primary path not found: {primary}")
    print("Searching experiments/ directory...")
    
    for root, dirs, files in os.walk("experiments"):
        for file in files:
            if file == "exp9_dataset.jsonl":
                path = os.path.join(root, file)
                print(f"Found: {path}")
                return path
    
    return None

def validate_schema(dataset_path):
    """Validate dataset schema."""
    required_keys = {'id', 'question', 'truth', 'direct', 'cot'}
    
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        print("ERROR: Dataset is empty")
        return False
    
    print(f"Dataset contains {len(lines)} problems")
    
    # Check first 5 records
    for i, line in enumerate(lines[:5]):
        try:
            record = json.loads(line)
            
            # Check required keys
            missing = required_keys - set(record.keys())
            if missing:
                print(f"ERROR: Record {i} missing keys: {missing}")
                return False
            
            # Check nested structure
            if not isinstance(record['direct'], dict) or 'response' not in record['direct']:
                print(f"ERROR: Record {i} direct.response missing")
                return False
            
            if not isinstance(record['cot'], dict) or 'response' not in record['cot']:
                print(f"ERROR: Record {i} cot.response missing")
                return False
                
        except json.JSONDecodeError as e:
            print(f"ERROR: Line {i} is not valid JSON: {e}")
            return False
    
    # Print examples
    print("\nExample problems:")
    for i in range(min(3, len(lines))):
        rec = json.loads(lines[i])
        print(f"  {i}: {rec['question']} = {rec['truth']}")
    
    print("\n[OK] Schema validation passed")
    return True

def main():
    print("="*60)
    print("00_verify_inputs.py: Dataset Validation")
    print("="*60)
    
    dataset_path = find_dataset()
    
    if dataset_path is None:
        print("\nERROR: exp9_dataset.jsonl not found")
        print("Searched in:")
        print("  - experiments/Experiment 9/data/")
        print("  - experiments/** (recursive)")
        sys.exit(1)
    
    print(f"\nDataset found: {dataset_path}")
    
    if not validate_schema(dataset_path):
        sys.exit(1)
    
    # Save path for downstream scripts
    with open("experiments/Experiment 16/data/dataset_path.txt", 'w') as f:
        f.write(dataset_path)
    
    print(f"\nDataset path saved to: experiments/Experiment 16/data/dataset_path.txt")

if __name__ == "__main__":
    main()
