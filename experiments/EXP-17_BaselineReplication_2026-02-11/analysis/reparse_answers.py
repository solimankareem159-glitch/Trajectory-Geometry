"""
Re-parse exp17a_dataset.jsonl with a smarter answer extraction strategy.
The original parser used 'last number in text' which fails when the model
hallucinates additional questions after answering correctly.

New strategy:
1. Look for "the answer is X" patterns first
2. Look for "= X" patterns at the end of arithmetic chains  
3. Fall back to first number after the last "=" sign
4. Final fallback: last number (original behavior)

Also re-evaluates Direct responses with the same improved parser.
"""
import json, re, os

DATA_PATH = r"experiments/EXP-17_BaselineReplication_2026-02-11/data/exp17a_dataset.jsonl"
OUTPUT_PATH = r"experiments/EXP-17_BaselineReplication_2026-02-11/data/exp17a_dataset.jsonl"

def extract_answer_smart(text, truth=None):
    """Smart answer extraction that handles hallucination tails."""
    if not text or not text.strip():
        return None
    
    # Strategy 1: "the answer is X" or "answer is X" or "answer: X"
    m = re.search(r'(?:the\s+)?answer\s+is\s+(-?\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    
    # Strategy 2: "= X." or "= X\n" at sentence boundaries (the final calculation)
    # Find all "= number" patterns, take the last one BEFORE any hallucinated Q:
    # Split on hallucination markers first
    # Common hallucination patterns: "Q:", "Question:", new problem text
    clean_text = text
    # Cut at hallucination boundary
    for marker in ['\nQ:', '\nQuestion:', '\nProblem:', '\nA train', '\nA store', '\nA man', '\nA car', '\nIf a', '\nHow many', '\nWhat is the']:
        idx = clean_text.find(marker)
        if idx > 0:
            clean_text = clean_text[:idx]
            break
    
    # Now find "= X" in the clean text
    equals_matches = re.findall(r'=\s*(-?\d+)', clean_text)
    if equals_matches:
        return int(equals_matches[-1])
    
    # Strategy 3: Last number in cleaned text
    nums = re.findall(r'-?\d+', clean_text)
    if nums:
        return int(nums[-1])
    
    # Strategy 4: Absolute fallback - last number in full text
    nums = re.findall(r'-?\d+', text)
    if nums:
        return int(nums[-1])
    
    return None

# Load data
data = [json.loads(l) for l in open(DATA_PATH)]
print(f"Loaded {len(data)} samples")

# Re-parse all responses
old_direct_correct = 0
new_direct_correct = 0
old_cot_correct = 0
new_cot_correct = 0

for d in data:
    truth = d["truth"]
    
    # Re-parse direct
    old_direct_correct += 1 if d["direct"]["correct"] else 0
    new_parsed_d = extract_answer_smart(d["direct"]["response"], truth)
    d["direct"]["parsed"] = new_parsed_d
    d["direct"]["correct"] = (new_parsed_d == truth)
    new_direct_correct += 1 if d["direct"]["correct"] else 0
    
    # Re-parse cot
    old_cot_correct += 1 if d["cot"]["correct"] else 0
    new_parsed_c = extract_answer_smart(d["cot"]["response"], truth)
    d["cot"]["parsed"] = new_parsed_c
    d["cot"]["correct"] = (new_parsed_c == truth)
    new_cot_correct += 1 if d["cot"]["correct"] else 0

print(f"\nDirect: {old_direct_correct} -> {new_direct_correct} correct")
print(f"CoT:    {old_cot_correct} -> {new_cot_correct} correct")

# Group counts
g1 = sum(1 for d in data if not d["direct"]["correct"])
g2 = sum(1 for d in data if d["direct"]["correct"])
g3 = sum(1 for d in data if not d["cot"]["correct"])
g4 = sum(1 for d in data if d["cot"]["correct"])
print(f"\nNew group sizes: G1={g1}, G2={g2}, G3={g3}, G4={g4}")

# Show some examples of changed parses
print("\n=== CoT responses that changed from WRONG to CORRECT ===")
shown = 0
for d in data:
    if d["cot"]["correct"] and shown < 3:
        resp = d["cot"]["response"]
        if "\nQ:" in resp or "A train" in resp or "A store" in resp:
            shown += 1
            print(f"\n  Sample {d['id']}: Q={d['question']} Truth={d['truth']}")
            print(f"  Response (first 400): {repr(resp[:400])}")
            print(f"  New parsed: {d['cot']['parsed']}")

# Save updated data
with open(OUTPUT_PATH, 'w') as f:
    for d in data:
        f.write(json.dumps(d) + "\n")
print(f"\nSaved re-parsed data to {OUTPUT_PATH}")
