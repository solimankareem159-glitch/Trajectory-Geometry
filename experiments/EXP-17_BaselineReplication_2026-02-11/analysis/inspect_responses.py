"""Inspect direct responses to diagnose G2 absence."""
import json, os, statistics

print(f"PID: {os.getpid()}", flush=True)

data_path = r"experiments/EXP-17_BaselineReplication_2026-02-11/data/exp17a_dataset.jsonl"
data = [json.loads(l) for l in open(data_path)]

print("=== DIRECT RESPONSE ANALYSIS ===")
print(f"Total samples: {len(data)}")
d_correct = sum(1 for d in data if d["direct"]["correct"])
c_correct = sum(1 for d in data if d["cot"]["correct"])
print(f"Direct correct: {d_correct}")
print(f"CoT correct: {c_correct}")

# Response lengths (character count)
d_lens = [len(d["direct"]["response"]) for d in data]
c_lens = [len(d["cot"]["response"]) for d in data]
print(f"\nDirect response char lengths: mean={statistics.mean(d_lens):.0f}, min={min(d_lens)}, max={max(d_lens)}")
print(f"CoT response char lengths: mean={statistics.mean(c_lens):.0f}, min={min(c_lens)}, max={max(c_lens)}")

# Token length estimation (split by whitespace)
d_tok = [len(d["direct"]["response"].split()) for d in data]
c_tok = [len(d["cot"]["response"].split()) for d in data]
print(f"Direct approx word count: mean={statistics.mean(d_tok):.0f}, min={min(d_tok)}, max={max(d_tok)}")
print(f"CoT approx word count: mean={statistics.mean(c_tok):.0f}, min={min(c_tok)}, max={max(c_tok)}")

# Check for truncation: if ~all direct responses are exactly the same length, they were truncated
print(f"\nDirect response char length distribution (top 5):")
from collections import Counter
len_counts = Counter(d_lens)
for length, count in len_counts.most_common(5):
    print(f"  {length} chars: {count} samples")

# Categorize direct response patterns
patterns = {
    "starts_with_number": 0,
    "multi_step_reasoning": 0,
    "hallucination_continuation": 0,
    "contains_correct_answer": 0,
    "answer_near_end": 0,
}

for d in data:
    resp = d["direct"]["response"].strip()
    truth = d["truth"]
    
    # Check if response starts with the answer
    if resp and resp[0].isdigit():
        patterns["starts_with_number"] += 1
    
    # Check for multi-step reasoning markers
    if any(marker in resp for marker in ["Step", "step", "First", "Let", "Calculate", "*", "="]):
        patterns["multi_step_reasoning"] += 1
    
    # Check for hallucinated continuations
    if any(marker in resp for marker in ["\nQ:", "\nQuestion:", "\nProblem:", "A train", "A store"]):
        patterns["hallucination_continuation"] += 1
    
    # Check if correct answer appears ANYWHERE in the response
    if str(truth) in resp:
        patterns["contains_correct_answer"] += 1
    
    # Check if correct answer is near the end (within last 30 chars)
    if str(truth) in resp[-30:]:
        patterns["answer_near_end"] += 1

print("\n=== DIRECT RESPONSE PATTERN ANALYSIS ===")
for pattern, count in patterns.items():
    print(f"  {pattern}: {count}/{len(data)} ({100*count/len(data):.1f}%)")

# Show examples
print("\n=== SAMPLE DIRECT RESPONSES ===")
for i in [0, 1, 2, 5, 10, 50, 100, 200]:
    if i >= len(data): continue
    d = data[i]
    resp = d["direct"]["response"]
    print(f"\nSample {i}: Q={d['question']} Truth={d['truth']}")
    print(f"  Direct Response: {repr(resp[:300])}")
    print(f"  Parsed: {d['direct']['parsed']} Correct: {d['direct']['correct']}")

# Show examples where truth IS in the response but parsing failed
print("\n\n=== RESPONSES WHERE ANSWER EXISTS BUT PARSED WRONG ===")
shown = 0
for d in data:
    truth = d["truth"]
    resp = d["direct"]["response"]
    parsed = d["direct"]["parsed"]
    if str(truth) in resp and parsed != truth and shown < 5:
        shown += 1
        print(f"\n  Sample {d['id']}: Q={d['question']} Truth={truth}")
        print(f"  Response: {repr(resp[:300])}")
        print(f"  Parsed as: {parsed}")
