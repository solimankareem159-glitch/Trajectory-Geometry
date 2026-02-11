"""Quick inspection of exp17a_dataset.jsonl to check for hallucinations and parsing issues."""
import json, re, os

DATA_PATH = r"experiments/EXP-17_BaselineReplication_2026-02-11/data/exp17a_dataset.jsonl"

data = [json.loads(l) for l in open(DATA_PATH)]
print(f"Total samples: {len(data)}")

# Check first 5 direct responses
print("\n" + "="*60)
print("FIRST 5 DIRECT RESPONSES")
print("="*60)
for i in range(5):
    d = data[i]
    resp = d["direct"]["response"]
    print(f"\n--- Sample {d['id']} ---")
    print(f"  Q: {d['question']}  Truth: {d['truth']}")
    print(f"  Response: {repr(resp[:300])}")
    print(f"  Parsed: {d['direct']['parsed']}  Correct: {d['direct']['correct']}")

# Check first 5 CoT responses
print("\n" + "="*60)
print("FIRST 5 COT RESPONSES")
print("="*60)
for i in range(5):
    d = data[i]
    resp = d["cot"]["response"]
    print(f"\n--- Sample {d['id']} ---")
    print(f"  Q: {d['question']}  Truth: {d['truth']}")
    print(f"  Response: {repr(resp[:400])}")
    print(f"  Parsed: {d['cot']['parsed']}  Correct: {d['cot']['correct']}")

# How many direct responses contain the truth value?
print("\n" + "="*60)
print("PARSING ANALYSIS")
print("="*60)

truth_in_direct = 0
truth_in_cot = 0

for d in data:
    truth_str = str(d["truth"])
    if truth_str in d["direct"]["response"]:
        truth_in_direct += 1
    if truth_str in d["cot"]["response"]:
        truth_in_cot += 1

print(f"Truth string appears in Direct responses: {truth_in_direct}/300")
print(f"Truth string appears in CoT responses: {truth_in_cot}/300")

# Show 5 cases where truth is in direct response but parsed wrong
print("\n" + "="*60)
print("CLOSE MISSES: Truth in text but parsed wrong (Direct)")
print("="*60)
shown = 0
for d in data:
    truth_str = str(d["truth"])
    resp = d["direct"]["response"]
    if truth_str in resp and not d["direct"]["correct"]:
        shown += 1
        if shown <= 5:
            nums = re.findall(r'-?\d+', resp)
            print(f"\n  Sample {d['id']}: Q={d['question']} Truth={d['truth']}")
            print(f"  Response: {repr(resp[:300])}")
            print(f"  All nums: {nums}")
            print(f"  Last num (parsed): {nums[-1] if nums else None}")

print(f"\n  Total close misses: {shown}")

# Check response lengths
direct_lens = [len(d["direct"]["response"]) for d in data]
cot_lens = [len(d["cot"]["response"]) for d in data]
print(f"\nDirect response length: min={min(direct_lens)}, max={max(direct_lens)}, mean={sum(direct_lens)/len(direct_lens):.0f}")
print(f"CoT response length: min={min(cot_lens)}, max={max(cot_lens)}, mean={sum(cot_lens)/len(cot_lens):.0f}")

# Empty responses?
empty_d = sum(1 for d in data if len(d["direct"]["response"].strip()) == 0)
empty_c = sum(1 for d in data if len(d["cot"]["response"].strip()) == 0)
print(f"\nEmpty direct responses: {empty_d}")
print(f"Empty CoT responses: {empty_c}")
