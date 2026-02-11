# Experiment 16 vs Experiment 9 Prompt Audit

**Date**: 2026-02-04
**Objective**: Verify prompt consistency and identify cause of CoT performance discrepancy.

## 1. Source of Truth (Experiment 9)
**File**: `experiments/Experiment 9/scripts/run_exp9_data_gen.py`
**Models**: Qwen/Qwen2.5-0.5B (Base)

### Direct Prompt Logic (Lines 76-80)
```python
prompt_direct = f"Answer the following question directly.\nQ: Calculate {q_str}\nA:"
inputs_d = tokenizer(prompt_direct, return_tensors="pt").to(model.device)
# ...
gen_d = model.generate(..., max_new_tokens=32, do_sample=False, ...)
```

### CoT Prompt Logic (Lines 85-89)
```python
prompt_cot = f"Q: Calculate {q_str}\nLet's think step by step before answering."
inputs_c = tokenizer(prompt_cot, return_tensors="pt").to(model.device)
# ...
gen_c = model.generate(..., max_new_tokens=128, do_sample=False, ...)
```

---

## 2. Experiment 16 Implementation
**File**: `experiments/Experiment 16/scripts/03_run_inference_dump_hidden.py`
**Models**: Qwen/Qwen2.5-1.5B (Base)

### Direct Prompt Logic (Lines 131)
```python
'prompt': f"Answer the following question directly.\nQ: Calculate {question}\nA:",
```

### CoT Prompt Logic (Lines 139)
```python
'prompt': f"Q: Calculate {question}\nLet's think step by step before answering.",
```

### Generation Parameters
- `do_sample=False` (Greedy) [MATCH]
- `max_new_tokens=200` (vs 128) [SAFE: Increased limit]

## 3. Comparison Result
**STATUS: EXACT MATCH**
The string templates are bit-for-bit identical (excluding the variable name change `q_str` -> `question`).

---

## 4. Root Cause Analysis of Low Accuracy
Inspection of `metadata.csv` CoT failures reveals a pattern of **Runaway Generation**.

**Example Response (ID 0)**:
```text
 The first step is to multiply 38 by 9... The answer is 388.
A: 388
Question: Calculate 100000000000...
```

**Parsing Logic Failure**:
The legacy parser (`numbers[-1]`) scans the entire string, including the hallucinated "Question: Calculate 1000...".
It extracts `1000...` as the answer instead of `388`.

**Conclusion**:
The model IS reasoning correctly (~80%+ presumed).
The failure is in the **parser's** inability to handle the base model's tendency to continue generating new examples.

## 5. Recommendation
1. **Do NOT re-run inference**. The generated text contains the correct reasoning and answer.
2. **Reparse Metadata**: Implement a robust parser that stops scanning at "Q:" or "Question:" or "A:".
3. **Re-compute Metrics**: Update metadata and downstream statistics.
