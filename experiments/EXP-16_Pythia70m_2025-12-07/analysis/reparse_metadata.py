
import pandas as pd
import re
import os

def robust_parse(text):
    if not isinstance(text, str): return None
    
    # 1. Truncate at hallucination boundaries
    # The Qwen base model tends to generate "Question: ..." or "Q: ..." or "\n\n" after the answer.
    stop_tokens = ["Question:", "\nQ:", "Problem:"]
    
    # Find the earliest stop token
    cutoff_idx = len(text)
    for stop in stop_tokens:
        idx = text.find(stop)
        if idx != -1 and idx < cutoff_idx:
            cutoff_idx = idx
            
    clean_text = text[:cutoff_idx]
    
    # 2. Look for explicit pattern "answer is X" in the clean text
    # This is safer than just "last number"
    match = re.search(r"(?:answer|result)\s+is\s+(-?\d+)", clean_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 3. Fallback: Last number in clean text
    numbers = re.findall(r'-?\d+', clean_text)
    if numbers:
        return int(numbers[-1])
        
    return None

def main():
    print("Reparsing metadata with robust logic...")
    input_path = "experiments/Experiment 16/data/metadata.csv"
    
    if not os.path.exists(input_path):
        print("Error: metadata.csv not found")
        return

    df = pd.read_csv(input_path)
    
    fixed_count = 0
    
    for idx, row in df.iterrows():
        new_parsed = robust_parse(row['response'])
        
        # Handle NaN/None
        if pd.isna(new_parsed):
            new_parsed = None
            
        old_correct = bool(row['correct'])
        new_correct = (new_parsed == row['truth'])
        
        if new_correct != old_correct:
            fixed_count += 1
            
        df.at[idx, 'parsed'] = new_parsed
        df.at[idx, 'correct'] = new_correct
        
    df.to_csv(input_path, index=False)
    print(f"Reparsed {len(df)} rows. Fixed correctness for {fixed_count} rows.")
    
    # Print new stats
    acc = df.groupby('condition')['correct'].mean()
    print(f"New Direct Acc: {acc.get('direct', 0):.2%}")
    print(f"New CoT Acc: {acc.get('cot', 0):.2%}")
    
    # Count G4
    g4 = len(df[(df['condition']=='cot') & (df['correct']==True)])
    print(f"Group G4 Size: {g4}")

if __name__ == "__main__": main()
