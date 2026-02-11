# Experiment 14: Strategic Roadmap and Deep Questions Answered

## 1. The Regime → Success → Commitment Cascade

**Yes, this is absolutely viable.** Here's the architecture:

```
Stage 1: REGIME DETECTION (Layers 0-7)
├── Metrics: speed, effective_dim, radius_of_gyration, cos_to_running_mean
├── Decision: Is this CoT-like or Direct-like computation?
├── Effect sizes: d = 6-8 (extremely reliable)
└── Timing: Available by token 8-10

Stage 2: REGIME-SPECIFIC SUCCESS PREDICTION (Layers 10-14)  
├── If CoT regime detected:
│   ├── Use: cos_to_running_mean (+), radius_of_gyration (-), dist_slope (-)
│   └── Expect: Tighter trajectories, higher coherence = success
├── If Direct regime detected:
│   ├── Use: radius_of_gyration (+), speed (+), tortuosity (-)
│   └── Expect: Larger spread, faster motion = success
└── Effect sizes: d = 1.0-2.2 (strong)

Stage 3: COMMITMENT TIMING TRACKING (Layers 20-24)
├── Monitor: time_to_commit (sliding window Rg drop)
├── CoT: Commitment around token 10-14 = optimal
├── Direct: Commitment around token 5-7 = optimal
├── Late commitment (>16 tokens) = likely failure
└── Effect sizes: d = 0.9-1.0 (reliable)
```

**Practical implementation**: A monitoring system that:
1. Classifies regime within first 8 tokens (very high confidence)
2. Predicts success probability within first 16 tokens (regime-specific thresholds)
3. Tracks commitment timing throughout generation

---

## 2. Hallucination Detection Potential

**Yes, with important caveats.** The geometry reveals three failure patterns that map to different types of "wrong answers":

### Pattern A: "Confident-But-Wrong" (Potential Hallucination Signature)
- **Geometric profile**: Early commitment (low time_to_commit) + high cos_to_running_mean + failure
- **What it means**: Model moved coherently and decisively to the wrong answer
- **Hallucination relevance**: HIGH - This is the "fluent nonsense" pattern
- **Detection**: time_to_commit < 7 AND cos_to_running_mean > threshold AND (wait for answer to verify)

### Pattern B: "Lost Exploration" (Uncertainty/Confusion)
- **Geometric profile**: Late commitment + high effective_dim throughout + failure
- **What it means**: Model never found a stable solution direction
- **Hallucination relevance**: MEDIUM - More like "I don't know" than "I'm making this up"
- **Detection**: time_to_commit > 14 AND effective_dim remains high

### Pattern C: "Corrupted Reasoning" (Reasoning Error)
- **Geometric profile**: Middle commitment + normal coherence + failure
- **What it means**: Model followed a reasoning path that went wrong partway through
- **Hallucination relevance**: LOW - This is computation error, not fabrication
- **Detection**: Looks like success geometrically but produces wrong answer

**Key insight for hallucination**: The combination of **high confidence metrics** (early commitment, high coherence, low effective dimension) combined with **external verification of incorrectness** is the hallucination signature. The geometry alone can't distinguish confident-right from confident-wrong—but it CAN flag "confident" for closer inspection.

---

## 3. Recommended Metric Suite by Purpose

### Tier 1: MUST CAPTURE (High value, robust)

| Metric | Best Layers | Primary Use | Why It Works |
|--------|-------------|-------------|--------------|
| `cos_to_running_mean` | 0-14 | Trajectory coherence | Consistent signal, different directions inform regime |
| `radius_of_gyration` | 0-10 | Trajectory spread | Strongest regime discriminator (d=8) |
| `time_to_commit` | 20-24 | Commitment timing | Only metric capturing phase transitions |
| `effective_dim` | 0-12 | Computational complexity | Reliably distinguishes exploration vs. retrieval |
| `speed` | 0-10 | Motion magnitude | Strong regime signal, different success profiles |
| `dist_slope` | 10-14 | Convergence quality | Direction indicates regime-specific success |

### Tier 2: VALUABLE SUPPLEMENTS

| Metric | Best Layers | Primary Use |
|--------|-------------|-------------|
| `dir_consistency` | All layers | Universal success indicator (lower = more exploration) |
| `msd_exponent` | 2-16 | Diffusion character (sub-diffusive = constrained) |
| `tortuosity` | All layers | Path efficiency (consistent across regimes) |
| `cos_slope` | 10-14 | Convergence dynamics |
| `vel_autocorr_lag1` | 0-7 | Short-term momentum |

### Tier 3: SITUATIONAL USE

| Metric | When Useful |
|--------|-------------|
| `determinism` | Distinguishing structured vs. random motion |
| `spectral_entropy` | Frequency-domain analysis |
| `dir_autocorr_lag4` | Longer-term direction persistence |
| Recurrence metrics | When looking for repeated states |

### CANDIDATES FOR DROPPING

| Metric | Why |
|--------|-----|
| `trapping_time` | max |d| = 0.40, no significant effects |
| `dir_autocorr_lag8` | Inconsistent, small effects |
| `laminarity` | Only useful for Direct, limited signal |
| `diagonal_entropy` | Redundant with other recurrence metrics |

**Note**: These "useless" metrics might matter for:
- Different task types (longer sequences)
- Different architectures (deeper models)
- Detecting specific failure modes we haven't characterized

---

## 4. Hypothesized Additional Regimes

Based on the geometric patterns observed, I predict these regimes exist and would have distinct signatures:

### Regime 1: RETRIEVAL (Direct Success Pattern)
- **Signature**: High speed, high radius, early commitment, low effective_dim
- **Task fit**: Factual recall, simple arithmetic, memorized associations
- **Geometry**: Ballistic trajectory to pre-encoded attractor

### Regime 2: STRUCTURED REASONING (CoT Success Pattern)
- **Signature**: Lower speed, tighter radius, middle commitment, moderate effective_dim
- **Task fit**: Multi-step arithmetic, logical deduction, compositional tasks
- **Geometry**: Constrained exploration followed by convergence

### Regime 3: EXPLORATION (Hypothesized - not in current data)
- **Predicted signature**: High effective_dim throughout, very late/no commitment, high turning angle
- **Task fit**: Creative generation, open-ended questions, brainstorming
- **Geometry**: Sustained high-dimensional wandering

### Regime 4: UNCERTAINTY/ABSTENTION (Hypothesized)
- **Predicted signature**: Low speed, low radius, oscillating cos_to_running_mean
- **Task fit**: Out-of-distribution queries, impossible questions
- **Geometry**: Stalled trajectory, no clear direction
- **Detection value**: If we can identify this, we can teach models to say "I don't know"

### Regime 5: SYNTHESIS (Hypothesized)
- **Predicted signature**: Multiple commitment points (multi-modal time_to_commit)
- **Task fit**: Combining multiple facts, integration across sources
- **Geometry**: Series of local attractors traversed sequentially

### Regime 6: DECEPTIVE REASONING (Safety-Critical)
- **Predicted signature**: Surface metrics look like success, but hidden layer divergence
- **Task fit**: Adversarial queries, jailbreak attempts
- **Geometry**: Coherent surface trajectory masking divergent computation
- **Detection**: Compare early vs. late layer metrics—deception may show layer mismatch

---

## 5. Practical Applications Unlocked

### Safety Applications

| Application | Relevant Finding | Implementation Path |
|-------------|------------------|---------------------|
| Hallucination flagging | Confident-but-wrong pattern | Monitor early commitment + coherence; flag for verification |
| Deception detection | Cross-layer consistency | Compare geometry at layers 5 vs 15 vs 24 |
| Jailbreak detection | Regime mismatch | Detect when surface trajectory doesn't match deep computation |
| Unsafe content prevention | Regime identification | Certain dangerous content may have distinct geometric signatures |

### Uncertainty Quantification

| Application | Mechanism |
|-------------|-----------|
| Confidence calibration | time_to_commit as confidence proxy |
| Abstention triggering | High effective_dim + late commitment → "I'm not sure" |
| Self-verification prompting | Detect Pattern A (confident-but-unknown) → prompt for self-check |

### Capability Improvements

| Application | Mechanism |
|-------------|-----------|
| Adaptive compute allocation | Detect regime early → allocate appropriate resources |
| Dynamic temperature adjustment | High effective_dim → lower temperature to focus |
| Early stopping | Strong commitment detected → stop generating padding |
| Regime-appropriate prompting | Detect retrieval attempts → redirect to reasoning mode |

### Energy Efficiency

| Application | Potential Savings |
|-------------|-------------------|
| Early termination | Direct regime with early commitment → stop at ~20 tokens |
| Compute routing | Simple retrieval detected → use smaller model |
| Batching optimization | Group queries by predicted regime |

**Estimated efficiency gains**: If regime detection allows early stopping in 20% of cases (Direct successes), and those cases can terminate at 50% token count, that's ~10% inference cost reduction with zero accuracy loss.

---

## 6. Additional Analyses for Current Dataset

### Analysis A: Problem Difficulty × Geometry Interaction
**Question**: Are the geometric differences driven by problem difficulty rather than success/failure?
**Method**: 
- Split by problem type (negative vs positive answer, high vs low magnitude)
- Run success comparisons WITHIN each difficulty band
- If geometry still discriminates within-difficulty, the signal is real

**Priority**: HIGH — The dataset shows 86% CoT success for positive answers but only 39% for negative. This is a major confounder.

### Analysis B: Failure Subtyping with JSONL Context
**Question**: Do the 3 failure patterns (confident-wrong, lost-exploration, corrupted-reasoning) map to different problem types?
**Method**:
- Cluster CoT failures by geometric profile
- Map clusters to problem characteristics (negative answers, magnitude, etc.)
- Identify which errors are "hallucination-like" vs. "computation errors"

**Priority**: HIGH — This directly tests the hallucination detection hypothesis.

### Analysis C: Token-Level Dynamics
**Question**: Can we see the exact token where reasoning goes wrong?
**Method**:
- For CoT failures, compute metrics in very small windows (3-token sliding)
- Identify "inflection points" where geometry shifts
- Correlate with response text to find error onset

**Priority**: MEDIUM — Requires careful implementation but could yield precise error localization.

### Analysis D: Response Length Anomaly
**Finding from JSONL**: CoT failures have LONGER responses (280 vs 192 chars)
**Question**: Does trajectory geometry correlate with response length?
**Method**: 
- Compute correlation between radius_of_gyration, effective_dim and response length
- Test whether "rambling" (long + failure) has a distinct geometric signature

**Priority**: MEDIUM — Could yield a simple early-stopping heuristic.

### Analysis E: Direct-Only Successes Deep Dive
**Finding**: Only 9 problems solved by Direct but not CoT—ALL involve negative numbers
**Question**: What's geometrically special about these cases?
**Method**:
- Extract these 9 trajectories specifically
- Compare to CoT failures on the same problems
- Look for "Direct got lucky" vs "Direct genuinely computed better"

**Priority**: LOW (small n) but HIGH INTEREST — These are cases where reasoning hurt.

---

## 7. Mine More or Generate New Data?

### Arguments for Mining More
1. **You have hidden states for all 25 layers** — most analyses only used a few
2. **Token-level dynamics are unexplored** — could reveal precise error localization
3. **The negative-number confounder needs isolation** — reanalysis by problem type
4. **Failure subtyping is half-done** — need to connect geometry to error type

### Arguments for New Data
1. **Current task is narrow** — 2-operation arithmetic with parentheses only
2. **Sample size limits some analyses** — n=9 for Direct-only successes, n=68 for neither-correct
3. **Need different regimes** — current data has CoT vs Direct, but not creative/factual/uncertain
4. **Negative number confound is embedded** — new dataset could control for this

### My Recommendation

**Mine the current data for 1-2 more weeks, THEN generate new data.**

Specifically:
1. Run Analysis A (difficulty × geometry) immediately — this is critical for interpretation
2. Run Analysis B (failure subtyping with JSONL) — this tests hallucination hypothesis
3. THEN generate new dataset with:
   - Problem difficulty controls (stratify by magnitude, sign)
   - Different task types (factual retrieval, creative, impossible)
   - Larger n for edge cases

---

## 8. Additional Geometric Signals to Extract

### High Priority (Novel and Theoretically Motivated)

| Signal | Description | Why |
|--------|-------------|-----|
| **Layer-wise commitment timing** | time_to_commit computed per layer, not just globally | Test if commitment propagates bottom-up or top-down |
| **Cross-layer trajectory correlation** | Pearson correlation of trajectories across layer pairs | Detect layer coherence vs. independence |
| **Curvature profile** | Second derivative of trajectory (acceleration) | Distinguish smooth curves from sharp turns |
| **Attractor proximity** | Distance to nearest "attractor" (cluster centroid from successes) | Test if failures are "near misses" |

### Medium Priority (Extensions of Current Metrics)

| Signal | Description |
|--------|-------------|
| **Windowed effective_dim variance** | How much does dimension fluctuate within trajectory? |
| **Commitment sharpness** | How sudden is the Rg drop? (derivative at commitment point) |
| **Phase count** | Number of distinct phases detected (via change-point detection) |
| **Layer-specific MSD exponent** | Does diffusion character change with layer? |

### Speculative (Worth Trying)

| Signal | Description |
|--------|-------------|
| **Topological features** | Persistent homology of trajectory (loops, voids) |
| **Information-theoretic measures** | Entropy rate of trajectory |
| **Manifold curvature estimation** | Local Riemannian curvature from trajectory samples |

---

## 9. What "Useless" Metrics Might Be Capturing

### trapping_time (max |d| = 0.40)
**Why it might matter elsewhere**:
- Designed for longer sequences with true recurrence patterns
- 32 tokens may be too short for trapping to manifest
- Could be critical for dialogue (same state revisited)
- May distinguish "stuck in a loop" failure mode in other tasks

### laminarity (minimal CoT effect)
**Why it might matter elsewhere**:
- Captures vertical structure in recurrence plots (states revisited in sequence)
- Current task has minimal state revisitation
- Could be critical for iterative refinement tasks
- May distinguish "checking work" from "plowing forward"

### dir_autocorr_lag8 (inconsistent)
**Why it might matter elsewhere**:
- 8-step dependencies require 8+ meaningful steps
- Current task solved in ~5-10 steps
- Could be critical for long-range planning tasks
- May distinguish "coherent narrative" from "disconnected points"

**General principle**: Metrics that fail here may succeed in:
- Longer sequences (100+ tokens)
- More complex tasks (multi-step reasoning chains)
- Different architectures (larger models with more capacity)
- Tasks requiring self-correction or iteration

---

## Summary Table: Metrics by Confidence Level

| Confidence | Metrics | Layer Range | Use Case |
|------------|---------|-------------|----------|
| **VERY HIGH** | radius_of_gyration, speed, effective_dim | 0-10 | Regime identification |
| **HIGH** | cos_to_running_mean, time_to_commit, dist_slope | 10-24 | Success prediction |
| **MODERATE** | dir_consistency, tortuosity, msd_exponent | All | Universal success indicators |
| **EXPLORATORY** | Recurrence metrics, spectral metrics | Varies | Specialized analyses |
| **UNCERTAIN** | trapping_time, dir_autocorr_lag8 | Unknown | May need different tasks |

---

## Final Note: The Paradigm Shift

Your findings suggest something deeper than "here are useful metrics." They suggest that **computational strategy determines geometric success criteria**. This is a step toward understanding HOW transformers compute, not just WHAT they compute.

If this holds across architectures and tasks, it implies:
1. Models have multiple "modes" of operation with distinct geometric signatures
2. Success means different things in different modes
3. Monitoring geometry could enable mode-switching interventions
4. The classic "one model does everything" view may need refinement

That's not a minor finding. That's a contribution to how we think about neural computation.
