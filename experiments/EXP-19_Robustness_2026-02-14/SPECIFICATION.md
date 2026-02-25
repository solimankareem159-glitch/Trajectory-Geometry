# Experiment 19: Multi-Architecture Replication with Full Metric Suite

## Executive Summary

Experiment 19 is the definitive validation study for Trajectory Geometry. It runs identical problems through three architectures (Qwen2.5-0.5B, Qwen2.5-1.5B, Pythia-410m), computes the full 57-metric suite at every layer, and performs seven statistical analyses including difficulty stratification, failure subtyping, cross-architecture transfer, and predictive modelling. The critical design challenge is preventing the post-answer hallucination contamination that plagued Experiments 16 and 16B.

This document specifies the complete pipeline in execution-ready detail for Gemini Flash running within Antigravity with DirectML acceleration.

---

## 1. Problem Generation (Phase 1)

### 1.1 Design Principles

Every problem is tested on all three architectures. This enables paired comparisons (same problem, different model) which are far more powerful than independent group comparisons. The problem set is stratified along two orthogonal dimensions: computational difficulty (digit count) and sign complexity (presence of negative operands).

### 1.2 Problem Structure

All problems use the same operation template family as Experiment 9:

```
Type A:  (a * b) + c
Type B:  (a * b) - c
Type C:  a * (b + c)
Type D:  a * (b - c)
```

Each problem records: operands (a, b, c), operation type (A/B/C/D), intermediate result (a*b or b+c or b-c), ground truth answer, digit count of answer, and sign of answer.

### 1.3 Difficulty Bins

```
Bin 1 "Small":     a in [2,9],    b in [2,9],    c in [1,9]
Bin 2 "Medium":    a in [10,49],  b in [2,19],   c in [10,49]
Bin 3 "Large":     a in [10,99],  b in [10,99],  c in [10,99]
Bin 4 "Negative":  a in [10,49],  b in [2,19],   c in [10,49]
                   with one operand randomly negated
```

Bin 4 is matched to Bin 2 in magnitude range so that the only systematic difference is the presence of a negative operand. This isolates the negative-number confound discovered in Experiment 15 (where CoT success dropped from 86% to 39% for negative answers).

### 1.4 Sample Size

100 problems per bin, 400 problems total. Each problem tested twice per model (Direct + CoT), yielding 800 trajectories per architecture and 2,400 trajectories in the combined dataset.

Target group sizes based on Experiment 9 base rates (Qwen 0.5B):
- G1 (Direct Fail): ~330 (82%)
- G2 (Direct Success): ~70 (18%)
- G3 (CoT Fail): ~100 (25%)
- G4 (CoT Success): ~300 (75%)

The Negative bin will skew G3/G4 toward more failures, which is intentional: it creates natural difficulty variation within the CoT condition.

### 1.5 Problem Validation Rules

Before accepting a problem into the set:
- Ground truth must be verifiable by integer arithmetic (no floating point)
- No duplicate problems (check question string identity)
- No problems where the answer equals an operand (avoids trivial retrieval)
- No problems where intermediate result equals zero (degenerate case)
- Balanced operation types within each bin (25 each of A, B, C, D)
- Within each bin, roughly equal positive and negative answers (except Bin 4 which skews negative)

### 1.6 Generation Code

```python
import random
import json

random.seed(42)

BINS = {
    'Small':    {'a': (2, 9),    'b': (2, 9),    'c': (1, 9),    'negate': False},
    'Medium':   {'a': (10, 49),  'b': (2, 19),   'c': (10, 49),  'negate': False},
    'Large':    {'a': (10, 99),  'b': (10, 99),  'c': (10, 99),  'negate': False},
    'Negative': {'a': (10, 49),  'b': (2, 19),   'c': (10, 49),  'negate': True},
}

OPS = {
    'A': lambda a,b,c: (f"({a} * {b}) + {c}", a*b + c, a*b),
    'B': lambda a,b,c: (f"({a} * {b}) - {c}", a*b - c, a*b),
    'C': lambda a,b,c: (f"{a} * ({b} + {c})", a*(b+c), b+c),
    'D': lambda a,b,c: (f"{a} * ({b} - {c})", a*(b-c), b-c),
}

def generate_problem(bin_name, bin_cfg, op_name, problem_id):
    while True:
        a = random.randint(*bin_cfg['a'])
        b = random.randint(*bin_cfg['b'])
        c = random.randint(*bin_cfg['c'])

        if bin_cfg['negate']:
            # Randomly negate exactly one operand
            idx = random.choice([0, 1, 2])
            if idx == 0: a = -a
            elif idx == 1: b = -b
            else: c = -c

        question, truth, intermediate = OPS[op_name](a, b, c)

        # Validation
        if intermediate == 0:
            continue
        if truth in (a, b, c):
            continue

        return {
            'id': problem_id,
            'bin': bin_name,
            'op_type': op_name,
            'a': a, 'b': b, 'c': c,
            'question': question,
            'truth': truth,
            'intermediate': intermediate,
            'answer_sign': 'positive' if truth >= 0 else 'negative',
            'answer_digits': len(str(abs(truth))),
        }

problems = []
pid = 0
for bin_name, bin_cfg in BINS.items():
    op_names = list(OPS.keys())
    for i in range(100):
        op = op_names[i % 4]  # Cycle through A, B, C, D
        prob = generate_problem(bin_name, bin_cfg, op, pid)
        problems.append(prob)
        pid += 1

# Verify
assert len(problems) == 400
for bin_name in BINS:
    bin_probs = [p for p in problems if p['bin'] == bin_name]
    assert len(bin_probs) == 100
    for op in 'ABCD':
        assert len([p for p in bin_probs if p['op_type'] == op]) == 25

with open('exp19_problems.json', 'w') as f:
    json.dump(problems, f, indent=2)
print(f"Generated {len(problems)} problems")
print(f"Answer sign distribution: {sum(1 for p in problems if p['answer_sign']=='positive')} pos, "
      f"{sum(1 for p in problems if p['answer_sign']=='negative')} neg")
```

---

## 2. Prompting and Anti-Contamination (Phase 2)

### 2.1 The Contamination Problem

In Experiments 16 and 16B, models produced the correct answer then continued generating irrelevant content. The specific failure modes observed:

(a) **Question looping**: After answering, the model generates "Question: Calculate..." and starts solving a new problem. The hidden states from this second problem contaminate metrics computed on the full trajectory.

(b) **Digit repetition**: The model outputs the answer then generates sequences like "1000000..." or "0000..." filling the token budget with repetitive states that distort R_g, D_eff, and speed.

(c) **CoT continuation**: After "the answer is 42", the model continues with "Let me verify..." or "Alternatively..." adding unnecessary reasoning tokens.

(d) **Direct overthinking**: Asked for just a number, the model starts explaining its reasoning anyway, making "Direct" trajectories look like CoT trajectories.

The solution operates at three levels: aggressive prompting, constrained generation, and post-hoc truncation. All three must be applied.

### 2.2 Prompt Templates

**Direct condition:**
```
Calculate: {question}
Answer with ONLY the numerical result. No explanation.
Answer:
```

The trailing "Answer:" provides a strong completion anchor. The model is primed to output a number immediately.

**CoT condition:**
```
Calculate: {question}
Show your working step by step.
After your final step, write "ANSWER: " followed by the number.
```

The explicit "ANSWER: " marker creates a parseable boundary. The instruction to write it after the final step discourages continuation.

### 2.3 Generation Parameters

**Direct:**
```python
DIRECT_CONFIG = {
    'max_new_tokens': 15,      # Enough for any integer, too short for explanation
    'do_sample': False,         # Greedy decoding, fully deterministic
    'temperature': 1.0,         # Irrelevant with do_sample=False but explicit
    'repetition_penalty': 1.0,  # No penalty (avoid distorting logits)
}
```

**CoT:**
```python
COT_CONFIG = {
    'max_new_tokens': 200,      # Generous for multi-step arithmetic
    'do_sample': False,
    'temperature': 1.0,
    'repetition_penalty': 1.0,
}
```

### 2.4 Stop Sequences (Applied During Generation)

Transformers `generate()` supports `stopping_criteria` but not all frameworks support string-level stop sequences natively. Implementation approach:

```python
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnSequence(StoppingCriteria):
    """Stop generation when any stop sequence is detected in output."""
    def __init__(self, tokenizer, stop_strings, input_length):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the generated portion
        generated = input_ids[0][self.input_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        for stop in self.stop_strings:
            if stop.lower() in text.lower():
                return True
        return False

STOP_STRINGS_DIRECT = [
    "\n",           # Newline after number
    "Question",     # Start of new question
    "Calculate",    # Restart
    "Let",          # "Let's think..."
    "Step",         # Starting to explain
    "The ",         # "The answer is..." (already gave number)
    "Because",      # Starting explanation
]

STOP_STRINGS_COT = [
    "Question:",    # New question loop
    "Calculate:",   # Restart loop
    "Problem:",     # New problem
    "\n\nQ:",       # Q&A format restart
    "\n\nCalculate",# Restart with spacing
    "Let me verify",# Unnecessary verification
    "Alternatively",# Second attempt
    "Double check", # Verification loop
]
```

### 2.5 Post-Generation Truncation

Even with stop sequences, edge cases slip through. The truncation pipeline runs after generation, before answer parsing and hidden state extraction.

```python
import re

def truncate_direct(text):
    """For Direct: keep only the first number."""
    text = text.strip()
    # Find the first complete number (possibly negative)
    match = re.match(r'\s*(-?\d+)', text)
    if match:
        return match.group(1)
    return text

def truncate_cot(text):
    """For CoT: keep everything up to and including ANSWER: number."""
    # Look for explicit answer marker
    match = re.search(r'ANSWER:\s*(-?\d+)', text, re.IGNORECASE)
    if match:
        return text[:match.end()]

    # Fallback: find "the answer is X" pattern
    match = re.search(r'(?:answer|result)\s*(?:is|=|:)\s*(-?\d+)', text, re.IGNORECASE)
    if match:
        return text[:match.end()]

    # Last resort: check for runaway patterns and truncate before them
    for pattern in [r'\n\s*(?:Question|Calculate|Q:)', r'(\d)\1{5,}']:
        match = re.search(pattern, text)
        if match:
            return text[:match.start()]

    return text

def find_clean_token_boundary(generated_tokens, tokenizer, clean_text):
    """Map clean text back to token positions for hidden state extraction."""
    for i in range(len(generated_tokens), 0, -1):
        decoded = tokenizer.decode(generated_tokens[:i], skip_special_tokens=True)
        if len(decoded.strip()) <= len(clean_text.strip()):
            return i
    return len(generated_tokens)
```

### 2.6 Answer Parsing

```python
def parse_answer(text, condition):
    """Extract numerical answer from model output."""
    if condition == 'direct':
        clean = truncate_direct(text)
        nums = re.findall(r'-?\d+', clean)
        return int(nums[0]) if nums else None

    elif condition == 'cot':
        clean = truncate_cot(text)
        # Priority 1: ANSWER: marker
        match = re.search(r'ANSWER:\s*(-?\d+)', clean, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Priority 2: "answer is" pattern
        match = re.search(r'(?:answer|result)\s*(?:is|=|:)\s*(-?\d+)', clean, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Priority 3: last number in text
        nums = re.findall(r'-?\d+', clean)
        return int(nums[-1]) if nums else None
```

### 2.7 Contamination Validation (Quality Gate)

After generating all responses for a model, run this validation before proceeding to metric computation:

```python
def validate_dataset(metadata_df):
    """Flag any trajectories that might be contaminated."""
    flags = []
    for _, row in metadata_df.iterrows():
        issues = []
        response = row['response']
        clean = row['clean_response']

        # Check 1: Response much longer than clean version
        if len(response) > len(clean) * 2 and len(response) > 50:
            issues.append('response_much_longer_than_clean')

        # Check 2: Direct response contains reasoning words
        if row['condition'] == 'direct':
            reasoning_words = ['step', 'first', 'multiply', 'calculate', 'because', 'therefore']
            if any(w in response.lower() for w in reasoning_words):
                issues.append('direct_contains_reasoning')

        # Check 3: Repetitive digit sequences
        if re.search(r'(\d)\1{5,}', response):
            issues.append('digit_repetition')

        # Check 4: Question restart pattern
        if re.search(r'(?:Question|Calculate|Q:)', response[len(clean):] if len(response) > len(clean) else ''):
            issues.append('question_restart_after_answer')

        # Check 5: Token count anomaly
        if row['condition'] == 'direct' and row['n_clean_tokens'] > 10:
            issues.append('direct_too_many_tokens')

        flags.append({
            'problem_id': row['problem_id'],
            'condition': row['condition'],
            'issues': issues,
            'contaminated': len(issues) > 0
        })

    flags_df = pd.DataFrame(flags)
    n_contaminated = flags_df['contaminated'].sum()
    print(f"Contamination check: {n_contaminated}/{len(flags_df)} flagged "
          f"({100*n_contaminated/len(flags_df):.1f}%)")

    if n_contaminated / len(flags_df) > 0.1:
        print("WARNING: >10% contamination rate. Review prompts and stop sequences.")

    return flags_df
```

Manual spot-check: visually inspect 20 random responses per model (10 Direct, 10 CoT) to confirm the truncation is working correctly.

---

## 3. Hidden State Extraction (Phase 3)

### 3.1 Architecture Specifications

| Property | Qwen2.5-0.5B | Qwen2.5-1.5B | Pythia-410m |
|:---------|:-------------|:-------------|:------------|
| Layers | 24 + embed | 28 + embed | 24 + embed |
| Hidden dim | 896 | 1536 | 1024 |
| Vocab size | 151,936 | 151,936 | 50,304 |
| HF name | Qwen/Qwen2.5-0.5B | Qwen/Qwen2.5-1.5B | EleutherAI/pythia-410m |

Qwen 0.5B is the baseline (Experiment 9 reference). Qwen 1.5B provides within-family scale comparison. Pythia-410m provides cross-architecture validation (GPT-NeoX family vs Qwen family).

Why Pythia-410m rather than 70m: the 70m model works (100% accuracy post-cleanup) but its 6-layer architecture provides too few layers for meaningful layer-wise analysis. The 410m model has 24 layers matching Qwen 0.5B exactly, enabling direct layer-to-layer comparison.

### 3.2 Model Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    """Load model with DirectML acceleration if available."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML: {device}")
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        output_hidden_states=True,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device
```

### 3.3 Extraction Pipeline

For each problem and condition:

1. Tokenize prompt, record `input_length`
2. Generate with `output_hidden_states=True, return_dict_in_generate=True`
3. Collect hidden states for generated tokens only (not prompt tokens)
4. Apply truncation to find the clean token boundary
5. Slice hidden states to clean boundary only
6. Validate minimum token count (>= 4 for meaningful geometry)
7. Save as numpy array: shape `[n_layers, n_clean_tokens, hidden_dim]`
8. Store as float16 to manage disk usage

```python
def extract_trajectory(model, tokenizer, device, prompt, config, stop_strings):
    """Generate response and extract clean hidden state trajectory."""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_len = inputs.input_ids.shape[1]

    stopping_criteria = StoppingCriteriaList([
        StopOnSequence(tokenizer, stop_strings, input_len)
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **config,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    # Decode full and clean responses
    gen_tokens = outputs.sequences[0][input_len:]
    full_response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Determine condition from config
    is_direct = config['max_new_tokens'] <= 15
    clean_response = truncate_direct(full_response) if is_direct else truncate_cot(full_response)

    # Find clean token boundary
    clean_token_count = find_clean_token_boundary(gen_tokens, tokenizer, clean_response)
    clean_token_count = max(clean_token_count, 1)  # At least 1 token

    # Extract hidden states: outputs.hidden_states is tuple of (step, layer, batch, seq, dim)
    # Each step gives hidden states for the newly generated token at all layers
    n_layers = model.config.num_hidden_layers + 1
    n_tokens = min(clean_token_count, len(outputs.hidden_states))

    if n_tokens < 4:
        return None, full_response, clean_response, n_tokens

    hidden_stack = []
    for layer_idx in range(n_layers):
        layer_states = []
        for t in range(n_tokens):
            # outputs.hidden_states[t] is a tuple of (n_layers) tensors
            # each tensor shape: [batch, seq_len, hidden_dim]
            # We want the last position (the newly generated token)
            state = outputs.hidden_states[t][layer_idx][0, -1, :].cpu().float().numpy()
            layer_states.append(state)
        hidden_stack.append(np.stack(layer_states))

    trajectory = np.stack(hidden_stack)  # [n_layers, n_tokens, hidden_dim]

    return trajectory, full_response, clean_response, n_tokens
```

### 3.4 File Organisation

```
exp19/
  problems.json                    # 400 problems
  qwen05b/
    metadata.csv                   # problem_id, condition, correct, response, n_tokens...
    hidden_states/
      000_direct.npy               # [25, T, 896]
      000_cot.npy
      ...
    contamination_flags.csv        # Validation results
  qwen15b/
    metadata.csv
    hidden_states/
    contamination_flags.csv
  pythia410m/
    metadata.csv
    hidden_states/
    contamination_flags.csv
  combined_metrics.csv             # All metrics, all models
  analysis/                        # All statistical outputs
```

### 3.5 Running the Inference Loop

Process one model at a time (VRAM constraint). For each model:

```python
def run_inference(model_name, problems, output_dir):
    model, tokenizer, device = load_model(model_name)
    os.makedirs(f"{output_dir}/hidden_states", exist_ok=True)

    metadata = []
    for prob in tqdm(problems, desc=f"Running {model_name}"):
        pid = prob['id']

        for condition in ['direct', 'cot']:
            if condition == 'direct':
                prompt = (f"Calculate: {prob['question']}\n"
                          f"Answer with ONLY the numerical result. No explanation.\n"
                          f"Answer:")
                config = DIRECT_CONFIG
                stops = STOP_STRINGS_DIRECT
            else:
                prompt = (f"Calculate: {prob['question']}\n"
                          f"Show your working step by step.\n"
                          f"After your final step, write \"ANSWER: \" followed by the number.")
                config = COT_CONFIG
                stops = STOP_STRINGS_COT

            trajectory, full_resp, clean_resp, n_tokens = extract_trajectory(
                model, tokenizer, device, prompt, config, stops
            )

            parsed = parse_answer(clean_resp, condition)
            correct = (parsed == prob['truth']) if parsed is not None else False

            if trajectory is not None:
                np.save(f"{output_dir}/hidden_states/{pid:03d}_{condition}.npy",
                        trajectory.astype(np.float16))

            group = None
            if condition == 'direct':
                group = 'G2' if correct else 'G1'
            else:
                group = 'G4' if correct else 'G3'

            metadata.append({
                'problem_id': pid,
                'condition': condition,
                'group': group,
                'correct': correct,
                'parsed_answer': parsed,
                'truth': prob['truth'],
                'bin': prob['bin'],
                'op_type': prob['op_type'],
                'answer_sign': prob['answer_sign'],
                'answer_digits': prob['answer_digits'],
                'response': full_resp,
                'clean_response': clean_resp,
                'n_tokens': n_tokens,
                'n_full_tokens': len(tokenizer.encode(full_resp, add_special_tokens=False)),
                'filename': f"{pid:03d}_{condition}.npy",
                'has_trajectory': trajectory is not None,
            })

    df = pd.DataFrame(metadata)
    df.to_csv(f"{output_dir}/metadata.csv", index=False)

    # Run contamination validation
    flags = validate_dataset(df)
    flags.to_csv(f"{output_dir}/contamination_flags.csv", index=False)

    # Unload model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return df
```

---

## 4. Metric Computation (Phase 4)

### 4.1 Full Metric Suite (57 Metrics)

The suite includes all 54 metrics from the Definitive Metric Suite document plus 3 additional metrics motivated by gaps identified during experiments 13-18.

**Family 1: Kinematic (13 metrics)**
K1 speed, K2 turn_angle, K3 tortuosity, K4 directional_consistency, K5 stabilization_rate, K6-K9 vel_autocorr_lag{1,2,4,8}, K10-K13 dir_autocorr_lag{1,2,4,8}

**Family 2: Volumetric (4 metrics)**
V1 radius_of_gyration, V2 effective_dimension, V3 gyration_anisotropy, V4 drift_to_spread

**Family 3: Convergence (6 metrics)**
C1 cosine_slope, C2 distance_slope, C3 early_late_ratio, C4 time_to_commit, C5 cosine_to_late_window, C6 cosine_to_running_mean

**Family 4: Diffusion and Spectral (3 metrics)**
D1 msd_exponent, D2 spectral_entropy, D3 psd_slope

**Family 5: Recurrence (5 metrics)**
R1 recurrence_rate, R2 determinism, R3 laminarity, R4 trapping_time, R5 diagonal_entropy

**Family 6: Cross-Layer (2 metrics)**
X1 interlayer_alignment, X2 depth_acceleration

**Family 7: Semantic Landmarks (6 metrics)**
S1 answer_logit_trajectory (summary: final logit value), S2 wrong_logit_trajectory, S3 logit_gap (mean over final quarter), S4 operand_proximity (mean proximity to each operand), S5 intermediate_proximity, S6 landmark_crossing_order (encoded as ordering tuple)

**Family 8: Attractor and Stability (5 metrics)**
A1 mean_attractor_distance, A2 cosine_to_success_direction, A3 discriminant_axis_projection (final value), A4 local_expansion_rate (mean), A5 point_of_no_return (token position, NaN if never crossed)

**Family 9: Embedding Stability (4 metrics)**
E1 logit_consistency_across_layers, E2 landmark_rank_stability, E3 landmark_pair_similarity, E4 embedding_unembedding_alignment

**Family 10: Information-Theoretic (3 metrics)**
I1 mean_step_surprisal, I2 trajectory_entropy_rate, I3 cumulative_info_gain (final value)

**Family 11: Inference-Time Actionable (4 metrics)**
N1 confidence_slope, N2 regime_classifier_signal, N3 convergence_monitor (mean late value), N4 anomaly_score (max z-score)

**Family 12: Phase Dynamics (3 NEW metrics)**
P1 commitment_sharpness, P2 phase_count, P3 fractal_dimension

These three additions address specific gaps:

P1 **Commitment Sharpness**: The magnitude of the maximum single-step drop in windowed R_g, capturing how abruptly the model transitions from exploration to execution. In Experiment 12 we observed that this transition varies from gradual to cliff-like, but we never quantified the sharpness itself.

$$\text{commit\_sharp} = \max_i \left(R_g^{(w)}[i] - R_g^{(w)}[i+1]\right) / R_g^{(w)}[0]$$

Normalised by initial R_g to make it comparable across difficulty levels.

P2 **Phase Count**: Number of distinct computational phases detected via change-point analysis on the speed time series. A phase boundary is declared when the local mean speed shifts by more than 1 standard deviation within a sliding window. Direct answers should show 1-2 phases (retrieve, output). CoT success should show 3-4 phases (parse, compute intermediate, compute final, output). Failures may show more phases (repeated failed attempts) or fewer (premature convergence).

P3 **Fractal Dimension**: Correlation dimension of the trajectory, computed via the Grassberger-Procaccia algorithm on the embedded trajectory. This characterises the geometric complexity at a finer scale than effective dimension (which uses singular value participation ratio). The fractal dimension captures self-similar structure in the trajectory, which may distinguish genuine multi-scale reasoning from simple expansion.

### 4.2 Layer Strategy

Not all 57 metrics need computation at every layer. This saves time and reduces multiple-comparison burden.

**All layers (0 to L-1):** Families 1-4 (Kinematic, Volumetric, Convergence, Diffusion/Spectral) = 26 metrics. These are the core trajectory descriptors and their layer profiles are themselves informative.

**Diagnostic layers only:** Families 5-12. Computed at three layer groups:
- Early: layers {1, 3, 5, 7} (regime detection)
- Middle: layers {10, 12, 14} (success prediction)
- Late: layers {L-5, L-3, L-1} (commitment tracking)

This yields a maximum of 26 * L + 31 * 10 = 26 * 25 + 310 = 960 metric-layer combinations per trajectory for Qwen 0.5B.

**Cross-layer metrics (X1, X2):** Computed once per trajectory (they aggregate across layers by definition).

**Embedding stability metrics (E1-E4):** Computed once per model, not per trajectory. These characterise the model's representation space rather than individual trajectories.

### 4.3 Reference Trajectory Construction

Attractor metrics (A1-A5) and anomaly scores (N4) require reference trajectories. These must be constructed per-regime and per-difficulty-bin to avoid confounding.

Reference sets:
- CoT Success reference: all G4 trajectories within the same difficulty bin
- Direct Success reference: all G2 trajectories within the same difficulty bin
- CoT Failure reference: all G3 trajectories (used to define the failure pole of the discriminant axis)

Leave-one-out procedure: when computing attractor metrics for trajectory $i$, the reference set excludes trajectory $i$. This prevents information leakage.

Minimum reference set size: 15 trajectories. If a cell has fewer than 15 successes (likely for Direct condition in harder bins), flag those attractor metrics as unreliable and exclude from statistical tests.

### 4.4 Metric Computation Code Structure

Due to Gemini Flash context limits, break the metric computation into separate functions that can be called independently:

```python
# File: compute_core_metrics.py (Families 1-4)
# File: compute_rqa_metrics.py (Family 5)
# File: compute_landmark_metrics.py (Family 7, requires W_U)
# File: compute_attractor_metrics.py (Family 8, requires reference trajectories)
# File: compute_embedding_metrics.py (Family 9, one-time per model)
# File: compute_info_metrics.py (Family 10)
# File: compute_inference_metrics.py (Family 11)
# File: compute_phase_metrics.py (Family 12)
# File: compute_cross_layer_metrics.py (Family 6)
```

Each file is self-contained and produces a CSV fragment. A final merge script combines them:

```python
# merge_metrics.py
# Reads all per-family CSVs, joins on (problem_id, condition, layer)
# Outputs: exp19_metrics_{model_name}.csv
```

---

## 5. Statistical Analyses (Phase 5)

### 5.1 Analysis A: Difficulty x Geometry Interaction

**Purpose:** Test whether geometric discrimination persists when controlling for difficulty, or whether it is merely a proxy for problem complexity.

**Method:**
1. Split all trajectories by difficulty bin (Small, Medium, Large, Negative)
2. Within each bin, compute G4 vs G1 Cohen's d for each metric at each layer
3. Test the interaction: does d scale with difficulty?
4. Report via heatmap: rows = metrics (sorted by peak d), columns = difficulty bins, cells = Cohen's d at peak layer

**Key predictions from Experiment 15:**
- R_g effect size should scale from d ~ 5 (Small) to d > 17 (Large)
- D_eff should show similar scaling
- cos_to_late_window should be relatively stable across difficulty (it measures trajectory shape, not scale)

**New analysis (beyond Exp 15):** Also run G3 vs G4 within-bin comparisons. This tests whether CoT failures are structurally different from CoT successes even when problem difficulty is held constant. If the effect persists within-bin, it cannot be attributed to difficulty alone.

**Additional test:** Within the Negative bin, compare problems with negative answers vs positive answers (some Negative bin problems may produce positive answers depending on operation type). This directly tests the negative-number confound hypothesis.

### 5.2 Analysis B: Failure and Success Subtyping via Clustering

**Purpose:** Discover natural subtypes of reasoning behaviour using unsupervised clustering.

**Method:**
For each group (G1, G2, G3, G4), cluster trajectories based on their geometric profiles. The clustering is layer-specific because different layers capture different aspects of computation.

**Layer selection for clustering:**
- **Layer 5** (early): Captures regime detection. Cluster on {speed, dir_consistency, R_g, D_eff, regime_classifier_signal}
- **Layer 12** (middle): Captures success prediction. Cluster on {R_g, D_eff, time_to_commit, cos_to_running_mean, msd_exponent, commitment_sharpness}
- **Layer L-2** (late): Captures commitment. Cluster on {cos_to_late_window, time_to_commit, convergence_monitor, logit_gap, attractor_distance}

**For each layer and group:**
1. Standardise metrics (z-score within group)
2. Run K-means with k = 2, 3, 4, 5
3. Select k by silhouette score
4. Characterise each cluster by its centroid values
5. Map clusters to problem metadata: bin, op_type, answer_sign, answer_digits

**Expected findings (replication of Exp 13):**
- G3 (CoT Failure) should split into Collapsed (low D_eff, low R_g) and Wandering (high D_eff, high R_g) subtypes
- G4 (CoT Success) may split into Explore-then-Commit (high early D_eff, low late D_eff) and Structured-Reasoning (moderate D_eff throughout)
- G1 (Direct Failure) is expected to be relatively homogeneous (most Direct failures look similar: the model just didn't know)

**New analysis:** After clustering, test whether cluster membership predicts problem characteristics. A logistic regression from cluster label to {bin, answer_sign, op_type} reveals whether failure subtypes correlate with task features. If Collapsed failures concentrate in Small/Medium bins and Wandering failures concentrate in Large/Negative bins, this supports a difficulty-dependent failure mode theory.

### 5.3 Analysis C: Token-Level Dynamics

**Purpose:** Track how geometry evolves across the generated sequence, identifying inflection points and phase transitions.

**Method:**
1. Compute R_g and speed in sliding 3-token windows (stride 1) across the trajectory
2. Compute the first derivative (rate of change) of windowed R_g
3. Identify inflection points: positions where the derivative changes sign
4. For CoT trajectories, align inflection points with the text to identify which tokens trigger geometric transitions

**Visualization:** Overlay windowed R_g time series for G3 vs G4 at a selected layer (Layer 12). The group-averaged curves should show a clear divergence point. Compute the mean divergence onset token position.

**New analysis:** Compute the cross-correlation between windowed R_g and windowed speed. If R_g leads speed (R_g drops before speed drops), the model commits geometrically before the execution phase begins. If speed leads R_g, execution begins while the model is still exploring. The lag of peak cross-correlation reveals the temporal relationship between exploration and execution.

### 5.4 Analysis D: Length Correlation and Predictive Modelling

**Purpose:** Separate geometric information from the length confound.

**Method:**
1. Compute Pearson correlation between each metric (at its peak discriminative layer) and response length in characters
2. Report r and p values for the top metrics
3. Train three logistic regression models (5-fold stratified cross-validation):
   - Length Only: features = {n_tokens}
   - Geometry Only: features = {R_g, D_eff, speed, cos_to_late_window, time_to_commit} at Layer 12
   - Combined: all features
4. Report AUC for each model

**Prediction from Exp 15:** Geometry Only should achieve AUC ~ 0.79, Length Only ~ 0.77, Combined ~ 0.80.

**New analysis (truncated trajectory version):** Repeat the prediction analysis using only the first 16 tokens of each trajectory. In this truncated window, length variation is minimal (all trajectories are exactly 16 tokens). If geometry still predicts success in truncated trajectories, the signal cannot be attributed to length. This is the strongest test of geometric information content.

**New analysis (difficulty-stratified prediction):** Train the classifier on Small + Medium problems only, test on Large + Negative problems. If the classifier transfers, the geometric signature of success is difficulty-invariant (a very strong claim for the paper).

### 5.5 Analysis E: Direct-Only Successes

**Purpose:** Investigate cases where Direct answers correctly but CoT fails on the same problem.

**Method:**
1. Identify problems where G2 (Direct Success) and G3 (CoT Fail) apply to the same problem_id
2. Extract their trajectories and compare geometric profiles
3. Test the "overthinking hypothesis": does CoT failure on these problems show artificial expansion (high D_eff, high R_g) compared to CoT successes on matched-difficulty problems?

**Expected finding (from Exp 15):** These cases cluster among problems with negative answers, and the CoT failure shows the "Wandering" subtype while the Direct answer shows efficient retrieval.

**New analysis:** For each Direct-Only success, compute the attractor distance relative to the CoT success reference. If these Direct successes are close to the CoT success attractor (despite being Direct), the model may be performing implicit reasoning in the Direct condition. If they are far from both the CoT and Direct success attractors, they represent a genuinely different computational strategy.

### 5.6 Analysis F: Cross-Architecture Comparison (NEW)

**Purpose:** Test which geometric signatures are universal across architectures and which are architecture-specific.

**Method:**

**F.1 Paired Effect Size Comparison:**
For each metric at each layer, compute Cohen's d (G4 vs G1) separately for each architecture. Plot the three d values side by side. Classify each metric as:
- Universal (d > 1.0 in all three architectures, same sign)
- Scale-dependent (d > 1.0 in 2/3, weaker in the third)
- Architecture-specific (significant in only one architecture)

**F.2 Goldilocks Hypothesis Test:**
Compute the mean R_g for each group in each architecture. Test whether success occupies an intermediate zone between collapse and wandering failure modes. Specifically:
- Qwen 0.5B: predict Success R_g > Failure R_g (expansion relative to collapse)
- Qwen 1.5B: predict Success R_g < Failure R_g (compression relative to wandering)
- Pythia-410m: this is the genuine prediction; the expected direction depends on the model's dominant failure mode, which we discover empirically

**F.3 Layer Alignment:**
Both Qwen 0.5B and Pythia-410m have 24 layers, enabling direct layer-to-layer comparison. Compute the correlation between the two architectures' layer profiles for each metric. A metric whose layer profile correlates at r > 0.8 across architectures is processing-stage-aligned, suggesting the underlying computation maps onto similar network depths.

For Qwen 1.5B (28 layers), normalise layer indices to [0, 1] range before comparison.

**F.4 Cross-Architecture Transfer:**
Train the success classifier on Qwen 0.5B (the cleanest dataset), test on Qwen 1.5B and Pythia-410m. Report AUC. If transfer AUC exceeds 0.65, geometric signatures are partially architecture-independent. If it exceeds 0.75, they are strongly transferable.

**F.5 Effect Size Heterogeneity:**
For each metric, compute Cochran's Q statistic across the three architectures to test whether the effect sizes are homogeneous. Metrics with significant Q (heterogeneous effects) are architecture-sensitive. Metrics with non-significant Q (homogeneous effects) are the strongest candidates for universal claims.

### 5.7 Analysis G: Regime x Architecture Interaction (NEW)

**Purpose:** Test whether the regime-dependent geometry principle (Exp 14 core finding) replicates across architectures.

**Method:**
1. For each architecture, compute Cohen's d for G4 vs G2 (CoT Success vs Direct Success) at Layer 12
2. Test whether the polarity of key metrics (R_g, D_eff, speed) remains consistent: CoT success should show higher expansion than Direct success in all architectures
3. Compute the "polarity matrix": for each metric, record whether CoT success is higher (+) or lower (-) than Direct success, separately for each architecture
4. Count the number of polarity agreements (all three same sign) vs disagreements

If the polarity matrix shows >80% agreement across architectures, regime-dependent geometry is a universal principle. If there are systematic polarity flips (especially for R_g between 0.5B and 1.5B as seen in Exp 16B), document these as the "Goldilocks reversal" and analyse which difficulty bins drive the flip.

---

## 6. Visualisation Plan (Phase 6)

### 6.1 Publication Figures

**Figure 1: Effect Size Heatmaps (3 panels)**
One heatmap per architecture. Rows = metrics (sorted by mean absolute d). Columns = layers. Colour = Cohen's d for G4 vs G1. Shared colour scale across panels.

**Figure 2: Difficulty Scaling**
Line plot. X-axis = difficulty bin. Y-axis = Cohen's d at peak layer. One line per metric (R_g, D_eff, speed, cos_to_late_window). Faceted by architecture.

**Figure 3: Failure Taxonomy**
PCA scatter of G3 trajectories at Layer 12, coloured by cluster assignment. Panels for each architecture. Annotated with cluster characteristics.

**Figure 4: Cross-Architecture Concordance**
Scatter plot: X = Cohen's d in Qwen 0.5B, Y = Cohen's d in Pythia-410m, for each metric at its peak layer. Points on the diagonal indicate perfect replication. Points far from the diagonal indicate architecture-specific effects.

**Figure 5: Token-Level Dynamics**
Group-averaged windowed R_g across token position at Layer 12. Four lines (G1-G4). Shaded regions = 95% CI. Shows the divergence point and commitment timing.

**Figure 6: Predictive Performance**
Bar chart: AUC for Length Only, Geometry Only, Combined. Grouped by architecture. Error bars from cross-validation.

**Figure 7: Goldilocks Structure**
Box plots of R_g by group (G1-G4), faceted by architecture. Annotated to show which direction "success" goes relative to "failure" in each architecture.

---

## 7. Execution Plan for Gemini Flash

### 7.1 Phase Sequencing

The pipeline is broken into independent prompts. Each prompt should be self-contained and produce a verifiable output file.

**Prompt 1: Problem Generation**
Input: The problem generation code from Section 1.6
Output: exp19_problems.json (validate: 400 problems, balanced bins and operation types)
Complexity: Low. Can run in a single prompt.

**Prompt 2A: Qwen 0.5B Inference**
Input: problems.json, inference code from Sections 2 and 3
Output: qwen05b/metadata.csv, qwen05b/hidden_states/*.npy, qwen05b/contamination_flags.csv
Complexity: High (400 problems * 2 conditions * model loading). Allow extended runtime.
Validation: check metadata.csv for group distribution, contamination rate.

**Prompt 2B: Qwen 1.5B Inference**
Same as 2A but with model_name = "Qwen/Qwen2.5-1.5B"

**Prompt 2C: Pythia-410m Inference**
Same as 2A but with model_name = "EleutherAI/pythia-410m"
Note: Pythia may need different prompt formatting. The template above should work but monitor for systematic parsing failures.

**Prompt 3A: Core Metric Computation (per model)**
Input: hidden_states/*.npy, metadata.csv
Output: metrics_core_{model}.csv (Families 1-4, all layers)
Run separately for each model.

**Prompt 3B: Advanced Metric Computation (per model)**
Input: hidden_states/*.npy, metadata.csv, model weights (for W_U access)
Output: metrics_advanced_{model}.csv (Families 5-12, diagnostic layers)
Requires model to be loaded for landmark/embedding metrics.

**Prompt 3C: Metric Merge**
Input: metrics_core_{model}.csv, metrics_advanced_{model}.csv
Output: exp19_metrics_{model}.csv (one row per problem-condition-layer)

**Prompt 4A: Analysis A (Difficulty x Geometry)**
Input: exp19_metrics_qwen05b.csv (repeat for each model)
Output: analysis_A_results.csv, difficulty heatmaps

**Prompt 4B: Analysis B (Clustering)**
Input: exp19_metrics_qwen05b.csv
Output: analysis_B_clusters.csv, analysis_B_centroids.csv, PCA scatter figures

**Prompt 4C: Analysis C (Token-Level Dynamics)**
Input: hidden_states/*.npy (raw trajectories needed for windowed computation)
Output: analysis_C_dynamics.csv, dynamics figures

**Prompt 4D: Analysis D (Prediction)**
Input: exp19_metrics_{model}.csv for each model
Output: analysis_D_prediction.csv, AUC figures

**Prompt 4E: Analysis E (Direct-Only Successes)**
Input: exp19_metrics_{model}.csv
Output: analysis_E_direct_only.csv

**Prompt 4F: Analysis F (Cross-Architecture)**
Input: All three metrics files
Output: analysis_F_cross_arch.csv, concordance plots, transfer AUC

**Prompt 4G: Analysis G (Regime x Architecture)**
Input: All three metrics files
Output: analysis_G_polarity.csv

**Prompt 5: Final Report Compilation**
Input: All analysis outputs
Output: Experiment_19_Report.md

### 7.2 Context Management for Gemini Flash

Each prompt should include:
1. The specific code to execute (not the entire pipeline)
2. The specific input files it needs
3. Clear output file specifications
4. Validation checks to run before considering the prompt complete

Avoid including the full metric definitions in every prompt. Instead, include only the relevant computation function.

For Prompts 2A-2C (inference), the model will be in memory. Keep the prompt focused on the inference loop and data saving. Do not ask it to also compute metrics in the same prompt.

For Prompts 3A-3B (metrics), if the metric suite is too large for a single prompt, split by family: one prompt for Families 1-4, one for Families 5-8, one for Families 9-12.

### 7.3 Error Recovery

Each phase produces persistent output files. If a prompt fails:
- Check the last successfully written file
- Resume from that point in the next prompt
- For inference (Prompt 2), include a "resume from problem_id" parameter that skips already-processed problems

---

## 8. Statistical Rigour Checklist

Every comparison reported in the paper must satisfy all of these:

- [ ] Cohen's d with 95% bootstrap confidence interval (10,000 resamples)
- [ ] Permutation test p-value (10,000 shuffles, report exact p)
- [ ] Sample sizes for both groups explicitly stated
- [ ] Multiple comparison correction: Benjamini-Hochberg FDR at q < 0.05 within each analysis
- [ ] Effect size interpretation: d < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, 0.8-1.2 large, >1.2 very large
- [ ] For cross-architecture claims: Cochran's Q test for effect size heterogeneity
- [ ] For prediction claims: stratified 5-fold cross-validation with AUC and 95% CI
- [ ] For clustering claims: silhouette score, adjusted Rand index against a priori groups, cluster stability via bootstrap resampling

---

## 9. Success Criteria

The experiment is considered successful if:

1. **Clean Data:** Contamination rate < 5% across all three architectures after the full anti-contamination pipeline.

2. **Core Replication (Qwen 0.5B):** Effect sizes for R_g, D_eff, and speed (G4 vs G1) match Experiment 9 within a factor of 2 (i.e., if Exp 9 showed d = 4.25, Exp 19 should show d > 2.0).

3. **Scale Replication (Qwen 1.5B):** At least 80% of metrics classified as "Universal" in Analysis F (d > 1.0, same sign as Qwen 0.5B).

4. **Cross-Architecture Replication (Pythia-410m):** At least 60% of metrics classified as "Universal" (lower threshold due to architectural differences).

5. **Regime-Dependence:** The polarity matrix in Analysis G shows >80% agreement, confirming regime-dependent geometry is not architecture-specific.

6. **Clustering Validity:** Failure subtypes (Collapsed vs Wandering) emerge with silhouette score > 0.3 and purity > 0.7 in at least 2/3 architectures.

7. **Prediction Transfer:** Classifier trained on Qwen 0.5B achieves AUC > 0.60 on Pythia-410m (chance = 0.50).

8. **Difficulty Scaling:** Cohen's d for R_g increases monotonically from Small to Large bins in at least 2/3 architectures.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| Pythia-410m gets 0% on harder problems | Medium | High | Run a 20-problem preflight per bin. If any bin shows 0%, cap difficulty or switch to Pythia-1b |
| VRAM overflow with 1.5B model + hidden states | Medium | Medium | Process in batches of 50 problems, saving hidden states incrementally |
| Gemini Flash loses context mid-inference | High | Medium | Break into 50-problem batches per prompt, with resume logic |
| Landmark metrics fail for multi-token numbers | Medium | Low | Use first sub-token as primary signal, report multi-token mean as supplement |
| Clustering unstable with small group sizes | Medium | Medium | Set minimum cluster size = 10, merge smaller clusters with nearest neighbour |
| DirectML incompatible with hidden state extraction | Low | High | Fall back to CPU with float32, accept slower runtime |

---

## 11. Deliverables

On completion, the experiment produces:

1. **exp19_problems.json** - 400 validated arithmetic problems
2. **Metadata CSVs** (3) - one per architecture with response text, parsed answers, group assignments
3. **Hidden state files** (~2,400 .npy files) - clean trajectories for all valid responses
4. **Metrics CSVs** (3) - one per architecture with 57 metrics at relevant layers
5. **Combined metrics CSV** - merged across architectures with architecture column
6. **Analysis outputs** (A through G) - statistical results, figures, and tables
7. **Experiment 19 Report** - comprehensive write-up suitable for paper supplementary material
8. **Figure set** - 7 publication-quality figures

---

## Appendix A: Prompt Template for Pythia

Pythia models may respond differently to instruction formatting. If the Qwen-style prompt produces poor results, use this simpler format:

**Direct:**
```
Q: What is {question}?
A:
```

**CoT:**
```
Q: What is {question}? Let's work through this step by step.
```

Run a 10-problem preflight with both prompt styles and select whichever produces cleaner responses (higher % of parseable numerical answers).

## Appendix B: Metric Dependency Graph

Some metrics depend on others:

```
Independent:        K1-K13, V1-V4, C1-C6, D1-D3, R1-R5, P1-P3
Requires W_U:       S1-S6, E1-E4, I3, N1
Requires reference: A1-A5, N4
Requires X-layer:   X1, X2
```

Compute in this order:
1. Independent metrics (Families 1-5, 12)
2. Cross-layer metrics (Family 6)
3. Reference-building (compute G4 centroids from independent metrics)
4. Attractor metrics (Family 8) and anomaly score (N4)
5. Semantic landmark metrics (Family 7) and embedding stability (Family 9) [requires model reload for W_U]
6. Information metrics (Family 10) and inference metrics (Family 11)
