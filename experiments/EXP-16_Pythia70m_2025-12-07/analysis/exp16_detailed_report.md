# EXP-16 (Qwen 1.5B) Detailed Analysis

## Overview
This report analyzes the trajectory geometry of **Qwen 2.5 1.5B** on arithmetic reasoning.
It focuses on internal group differences and comparisons with **Qwen 2.5 0.5B (EXP-14)**.

## 1. Internal Group Pairwise Comparisons (EXP-16)

We compare four groups:
- **G1 (Direct Fail):** "Fast System 1" failure.
- **G2 (Direct Success):** "Fast System 1" success.
- **G3 (CoT Fail):** "Slow System 2" failure.
- **G4 (CoT Success):** "Slow System 2" success.

### Key Quantitative Findings (Layer 25)
| Metric | G1 (Dir Fail) | G2 (Dir Succ) | G3 (CoT Fail) | G4 (CoT Succ) |
|---|---|---|---|---|
| **Speed** | **283.3** (High) | 220.0 | **163.2** (Low) | 239.4 |
| **Radius** | **836.7** (High) | 500.4 | **310.4** (Low) | 676.1 |
| **Anisotropy** | **0.579** (Low) | 0.708 | **0.780** (High) | 0.608 |
| **Eff Dim** | 11.43 | 11.84 | **14.13** (High) | **9.95** (Low) |

### Interpretation of Profiles

#### G1: The "Flailing" Fails (Direct Mode)
- **Signature:** Highest Speed, Highest Radius, Lowest Anisotropy.
- **Diagnosis:** **Wandering / Scrambling.**
- The model traverses a huge volume of state space rapidly but without direction (Low Anisotropy). It is "lost" in high-dimensional space, likely hallucinating or outputting noise.
- **Contrast to 0.5B:** In 0.5B, Direct Fails often collapsed. In 1.5B, they explode.

#### G3: The "Tunnel Vision" Fails (CoT Mode)
- **Signature:** Lowest Speed, Lowest Radius, Highest Anisotropy.
- **Diagnosis:** **Rigid Collapse / Tunnel Vision.**
- The model is stuck (Low Speed, Low Radius) in a very specific subspace (High Anisotropy).
- **Paradox:** It has High Effective Dimension (14.1) despite High Anisotropy. This suggests a dominant direction (Aniso) but significant high-frequency noise or "jitters" in the orthogonal directions (High Dim tail).
- **Meaning:** "Stable but Wrong". The model convinced itself of a wrong path and stayed rigidly on it.

#### G4: The "Efficient" Success (CoT Mode)
- **Signature:** Moderate Speed, Moderate Radius, **Lowest Dimension (9.95)**.
- **Diagnosis:** **Efficient Reasoning.**
- It explores enough (Radius > G2, G3) to solve the problem but uses the fewest dimensions to do so. It "compresses" the problem effectively.

## 2. Cross-Model Comparison (EXP-14 0.5B vs EXP-16 1.5B)

### 1. Failure Mode Phase Transition
- **0.5B Failure:** Often characterized by **Collapse** (Low Energy/Radius). The model "gives up".
- **1.5B Failure (Direct):** Characterized by **Wandering** (High Energy/Radius). The model "hallucinates".
- **1.5B Failure (CoT):** Characterized by **Rigidity**. The model "fixates".

### 2. Dimensionality Scaling
- 0.5B G4 Eff Dim: ~5-6 (from EXP-14).
- 1.5B G4 Eff Dim: ~10 (from EXP-16).
- **Finding:** Dimensionality correlates with model size/complexity. Successful reasoning happens in a lower-dimensional manifold *relative to the model's capacity*, but higher absolute dimension than smaller models.

## 3. High Value Findings
1.  **"Wandering" identified in Direct Failures (G1):** confirmed by High Radius + Low Anisotropy.
2.  **"Stable but Wrong" identified in CoT Failures (G3):** confirmed by Low Radius + High Anisotropy.
3.  **Success is Low-Dimensional:** G4 has the lowest effective dimension of all groups in 1.5B. **Metric for Success:** Low Effective Dimension might be a strong predictor of correct reasoning in CoT.

## Conclusion
Qwen 1.5B shows distinct geometric signatures for different failure modes. Direct failures are "high-energy chaos", while CoT failures are "low-energy rigidity". Successful CoT is "efficient expansion".
