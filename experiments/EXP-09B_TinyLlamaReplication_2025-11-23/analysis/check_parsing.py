
import json
import re
import os

# Use relative path from this script's location
LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp9b_dataset.jsonl')

def improved_parse(text):
    # Try specific patterns first
    match = re.search(r"So, the (?:result|answer) is (-?\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback to last number *before* any "Q:" or "A:" or "Can you"
    stop_patterns = ["Q:", "A:", "Can you", "How do I", "I hope"]
    text_lower = text.lower()
    
    indices = []
    for p in stop_patterns:
        idx = text.find(p)
        if idx != -1:
            indices.append(idx)
            
    if indices:
        cutoff = min(indices)
        text = text[:cutoff]
        
    numbers = re.findall(r'-?\d+', text)
    if not numbers:
        return None
    return int(numbers[-1])

def main():
    print(f"Checking {LOG_FILE}...")
    
    try:
        with open(LOG_FILE, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("File not found!")
        return

    print(f"Loaded {len(data)} records.")
        
    potential = 0
    fixed = 0
    
    for rec in data:
        truth = rec["truth"]
        cot = rec["cot"]
        
        # We only care about CoT failures
        if cot["correct"]:
            continue

        # Check if truth is in response text
        if str(truth) in cot["response"]:
             # Check if improved parse finds it
             new_ans = improved_parse(cot["response"])
             if new_ans == truth:
                 print(f"[FIX] ID {rec['id']}: Truth={truth}, Old={cot['parsed']}, New={new_ans}")
                 fixed += 1
             else:
                 print(f"[POTENTIAL] ID {rec['id']}: Truth={truth} in text. Parse got {new_ans}. End of text: {repr(cot['response'][-50:])}")
                 potential += 1
                     
    print(f"\nFound {fixed} confirmed fixes.")
    print(f"Found {potential} other potential matches.")

if __name__ == "__main__":
    main()
