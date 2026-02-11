# Experiment 9 (0.5B) vs Experiment 16B (1.5B): Divergence Analysis

## Executive Summary
You asked why Experiment 16B (1.5B parameters) did not produce an "identical" dataset to Experiment 9 (0.5B parameters) despite using **identical prompts, equations, and generation settings**.

**The Conclusion**: The "failure" of the 1.5B model to stop is actually a result of its **superior capability**. The 1.5B model is better at recognizing and continuing the implicit document structure (a math worksheet), whereas the 0.5B model treated the prompt more as a single-turn completion or lacked the coherence to continue the pattern.

## 1. Experimental Constants (Verified)
We verified the source code for both experiments. The following variables were **identical**:

| Variable | Exp 9 (0.5B) | Exp 16B (1.5B) | Match? |
| :--- | :--- | :--- | :--- |
| **Questions** | `(38 * 9) + 46`, etc. | `(38 * 9) + 46`, etc. | ✅ YES |
| **Direct Prompt** | `"Answer... directly.\nQ: ...\nA:"` | `"Answer... directly.\nQ: ...\nA:"` | ✅ YES |
| **CoT Prompt** | `"Q: ...\nLet's think step by step..."` | `"Q: ...\nLet's think step by step..."` | ✅ YES |
| **Decoding** | Greedy (`do_sample=False`) | Greedy (`do_sample=False`) | ✅ YES |
| **Max Tokens** | 32 (Dir), 128 (CoT) | 32 (Dir), 128 (CoT) | ✅ YES |

## 2. The Divergence

### Experiment 9 Behavior (0.5B)
- **Response**: `"...So, the answer is 388."` <EOS>
- **Behavior**: The model answers the question and stops generating.
- **Why**: The 0.5B model likely has a higher probability mass on the `<EOS>` token immediately after the answer, OR it loses the "thread" of the document structure.

### Experiment 16B Behavior (1.5B)
- **Response**: `"...The answer is 388.\nA: 388\nQuestion: Calculate 100..."`
- **Behavior**: The model answers the question, **self-corrects/formats it** (`A: 388`), and then **generates the next question**.
- **Why**: The 1.5B model has a stronger internal model of "what document am I writing?". It recognizes the pattern `Q: ... A: ...` (a common training data format like a worksheet or exam). It doesn't "know" you only want one answer; it thinks it's helping you write the whole exam.

## 3. Detailed Evidence

### A. The "Smart" Hallucination
The 1.5B model's hallucinations are not random garbage. Look at `id: 0`:
```text
The first step is to multiply... 38 * 9 = 342.
...
The answer is 388.
A: 388  <-- IMPLICIT FORMATTING (Model ADDED this!)
Question: Calculate 100... <-- PATTERN CONTINUATION
```
The model explicitly adds `A: 388`, which was *not* in the CoT prompt. This shows it is actively trying to conform to a Q&A format it has seen during pre-training.

### B. The "Shorthand" Hallucination
In `id: 1` and `id: 3`, the model hallucinates:
```text
Q: 100000000...
```
This suggests it is trying to generate a new question but getting stuck in a repetition loop (a common failure mode for base models when they don't have a clear "next token" to break a pattern).

## 4. Why this matters for "Exact Replication"
"Exact Replication" for LLMs is tricky because **model capability alters prompt adherence**.
- A **weaker model** (0.5B) follows the immediate instruction or stops because it's "done".
- A **stronger base model** (1.5B) interprets the *context* and continues the *pattern*.

## 5. Conclusion
The experiments **did** produce identical *reasoning content* (the CoT math is correct in both). The difference is entirely in **Stopping Behavior**.
- **Exp 9**: Stopped naturally.
- **Exp 16B**: Continued the document.

**Impact**: This confirms our decision to use the **Reparsing Strategy**. By mechanically truncating the Exp 16B output at the end of the answer, we artificially impose the stopping behavior that the 0.5B model exhibited naturally. This forces the datasets to be comparable in terms of *content*, removing the artifact of *capability*.
