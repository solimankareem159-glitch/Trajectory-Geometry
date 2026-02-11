# Findings Catalogue: Trajectory Geometry

**Date:** 2026-02-10
**Status:** Validated / ready for synthesis
**Context:** Summary of 18 experiments (EXP-01 through EXP-16B) investigating the geometric properties of transformer residual streams during reasoning tasks.

---

## 1. Preamble

This document catalogues the empirical findings of the **Trajectory Geometry** research project. The goal was to determine if high-level cognitive operations (Reasoning vs. Retrieval, Success vs. Failure) leave measurable geometric signatures in the model's latent space.

### Methodology Summary
*   **Primary Architecture:** Qwen2.5-0.5B (Decoder-only Transformer).
*   **Validation Architectures:** Qwen2.5-1.5B, Pythia-70m.
*   **Task Domain:** Multi-step arithmetic reasoning (e.g., `(A * B) + C`).
*   **Technique:** Analysis of residual stream hidden states $h_t$ across all layers.
*   **Control:** All comparative analyses restricted to the **first 32 tokens** to control for length confounds.
*   **Metrics:** 36 geometric/dynamical indicators (see [Metrics Appendix](Metrics_Appendix_2026-02-10.md)).

---

## 2. Empirical Results (The Fact Layer)

The following results are statistically significant ($p < 0.001$, Permutation Test) and replicated across scale (0.5B $\to$ 1.5B).

### 2.1 Dimensional Collapse in Failure
Failed reasoning trajectories exhibit a catastrophic collapse in dimensionality compared to successful reasoning. The model's computation is confined to a much narrower subspace when it fails.

*   **Metric:** Effective Dimension ($D_{eff}$, PCA Participation Ratio).
*   **Result:**
    *   **Successful CoT (G4):** Mean $D_{eff} \approx 13.1$
    *   **Failed Direct (G1):** Mean $D_{eff} \approx 3.4$
*   **Effect Size:** Cohen's $d > 4.5$ (Massive).
*   **Visual Evidence:**
    ![Dimensional Collapse](figures/FigA_TheCollapse.png)
*   **Source:** EXP-11, EXP-12.

### 2.2 Regime-Relative Success Geometry
There is no single "good" geometry. The geometric signature of success *flips sign* depending on the prompting regime (Direct vs. CoT).

*   **Observation:** 10 of 14 key metrics show opposite correlations with success in the two regimes.
    *   **CoT Success:** Characterized by **High Expansion** ($R_g \uparrow$), **Lower Speed**, and **Low Cosine Similarity** to final state (Delayed Commitment).
    *   **Direct Success:** Characterized by **Low Expansion** ($R_g \downarrow$), **Higher Speed**, and **High Cosine Similarity** (Early Commitment).
*   **Implication:** A "correct" Direct trajectory looks remarkably like a "failed" CoT trajectory (collapsed/efficient).
*   **Source:** EXP-14.

### 2.3 Difficulty-Driven Expansion
The magnitude of geometric expansion matches problem difficulty. The model selectively "spends" geometric volume to resolve complexity.

*   **Metric:** Radius of Gyration ($R_g$) at Layer 4.
*   **Result:**
    *   **Small Problems:** Effect size (CoT vs Direct) is moderate ($d \approx 5.0$).
    *   **Extra Large Problems:** Effect size spikes to $d > 17.0$.
*   **Visual Evidence:**
    ![Difficulty Scaling](figures/FigC_DifficultyScaling.png)
*   **Source:** EXP-15 (Analysis A).

### 2.4 Failure Subtypes (The Taxonomy of Error)
Unsupervised clustering of failure cases (G3) reveals two distinct structural subtypes. Failures are not monolithic.

*   **Subtype A (Collapsed Failure):**
    *   Low $D_{eff}$, Low $R_g$.
    *   Visually indistinguishable from Direct answers.
    *   *Interpretation:* Premature optimization / "Give up" mode.
*   **Subtype B (Wandering Failure):**
    *   High $D_{eff}$, High $R_g$ (often higher than success).
    *   Low Stabilization, Low Convergence.
    *   *Interpretation:* Active confusion / "Getting lost" mode.
*   **Visual Evidence:**
    ![Failure Taxonomy](figures/FigD_FailureTaxonomy.png)
*   **Source:** EXP-13.

### 2.5 Commitment Timing
The "Phase Transition" from exploration to execution is measurable and distinct.

*   **Metric:** `time_to_commit` (Token index of maximum $R_g$ drop).
*   **Result:**
    *   **Direct Answers:** Commit at tokens 0–5.
    *   **CoT Answers:** Commit at tokens 11–20.
*   **Visual Evidence:**
    ![Commitment Curve](figures/FigE_CommitmentCurve.png)
*   **Layer Profile:** The commitment signal is strongest in **Late Layers (20–24)**.
*   **Source:** EXP-12, EXP-14.

### 2.6 Length is a Strong Predictor (Full Context Update)
In full-context trajectories, success is strongly correlated with response length.

*   **Prediction Task:** Logistic Regression (Success vs Failure).
*   **AUC Scores (EXP-15 Full Context):**
    *   Length only: **AUC 0.77**
    *   Geometry only: AUC 0.64
    *   Combined: AUC 0.74
*   **Conclusion:** Unlike in truncated settings, full-context success is characterized by longer, complete trajectories. Geometry provides a weaker, secondary signal (AUC 0.64) compared to the dominant length signal.

---

## 3. Interpretive Framework

These frameworks synthesize the empirical results into a coherent theory of transformer reasoning.

### 3.1 CoT as Dimensional Expansion
Reasoning is a **Dynamic State-Space Expansion** mechanism. Chain-of-Thought prompting works because it forces the model to "unfold" compressed representations into a high-dimensional manifold where they can be manipulated linearly.
*   *Success* = Sufficient expansion to resolve entropy.
*   *Failure* = Insufficient expansion (Collapse) or uncontrolled expansion (Wandering).

### 3.2 The "Explore $\to$ Commit" Phase Transition
Reasoning is a two-phase process:
1.  **Exploration Phase:** High dimensionality, high entropy, low cosine similarity to outcome. (Searching the solution space).
2.  **Commitment Phase:** Sharp collapse in dimensionality, rapid convergence to the answer token.
*   *Direct answers* skip Phase 1.
*   *CoT answers* spend 50-70% of steps in Phase 1.

### 3.3 Resource-Rational Reasoning
The model targets an optimal level of expansion for the problem entropy.
*   **Easy problems:** Expansion is wasteful (noise). The model stays "flat."
*   **Hard problems:** Expansion is necessary. The model "inflates" the latent space.
*   **Pathology:** "Overthinking" (Direct-Only Successes) occurs when the model expands on a problem that could have been solved flat.

---

## 4. Negative Results & Null Findings

*   **Static Signatures (EXP-01–07):** Attempts to find fixed "operator vectors" (e.g., a "Summarization Vector") in the embedding space failed ($AMI \approx 0$). Thought is a *trajectory*, not a *position*.
*   **Cue-Word Triggering (EXP-08'):** Injecting words like "Therefore" or "Wait" did not reliably trigger geometric phase transitions. Dynamics are emergent, not simply token-reactive.
*   **Cross-Model Triggering (EXP-09B):** TinyLlama-1.1B failed to replicate geometric signatures because it failed to *reason* (0% accuracy). Geometry requires a capability floor.
*   **Universal Success Signature (EXP-14):** Hypothesis H1 (that success always looks the same) was falsified. Success geometry is regime-dependent.

---

## 5. Methodological Notes

*   **Control Window (EXP-01 $\to$ EXP-13):** Metrics in these experiments are computed on the **first 32 tokens**.
*   **Full Trajectory Capture (EXP-14 & EXP-16B):** Experiment 14 (Universal Signature) and 16B (Qwen) now use **full trajectories** (up to 128 tokens) to capture late-stage dynamics. EXP-14 was remediated to fix a prior 32-token truncation.
*   **Full Trajectory Capture (EXP-16B):** Experiment 16B (Qwen 1.5B) is the **only** experiment to capture full trajectories (up to 128 tokens) before analytical truncation. This dataset is available for analyzing late-stage dynamics.
*   **Hallucination Truncation:** "Runaway" generations (repeating the question forever) were identified in EXP-16 and strictly truncated using a regex stop-sequence pipeline.
*   **Permutation Testing:** All group differences (G1 vs G4) were validated using 10,000-shuffle permutation tests to ensure $p < 0.001$.

---


---

## 7. Comparative Analysis (Cross-Architecture)

**Objective:** Validation of geometric signatures across model scales (Qwen 0.5B vs 1.5B vs Pythia).

### 7.1 The Relativity of Success
The geometric signature of "Success" is relative to the model's dominant failure mode.

*   **Qwen 0.5B (EXP-14):**
    *   **Failure Mode:** Collapse (Repetition/Looping). Low $R_g$, Low Dim.
    *   **Success Signature:** **Expansion** (Success $R_g \gg$ Failure $R_g$).
    *   *Result:* Success looks like "opening up" the space.

*   **Qwen 1.5B (EXP-16B) & Pythia (EXP-16):**
    *   **Failure Mode:** Wandering (Hallucination/Gibberish). High $R_g$, High Dim.
    *   **Success Signature:** **Compression/Focus** (Success $R_g <$ Failure $R_g$).
    *   *Result:* Success looks like "constraining" the space against entropy.

**Conclusion:** There is no universal "Success Direction" (Up or Down). Success is a "Goldilocks" zone of **Controlled Expansion**—distinct from the extremes of Collapse (0.5B) and Wandering (1.5B).

### 7.2 Layer-wise Dynamics
*   **0.5B:** Divergence begins at **Layer 12** (Mid) and grows.
*   **1.5B:** Divergence is essentially **constant** or early-onset. The larger model commits to a mode (Wandering vs Reasoning) earlier.

---

## 6. Figure Index (Conceptual)

*   **Figure A: The Collapse.** (Data: EXP-14). Boxplot of $D_{eff}$ showing massive separation between G4 (Success) and G1 (Fail). `research_history/figures/FigA_TheCollapse_Publication.png`
*   **Figure B: The Fork.** (Conceptual). Schematic showing how CoT success expands while Direct success compresses.
*   **Figure C: Difficulty Scaling.** (Data: EXP-15). Line plot of Cohen's $d$ ($R_g$) vs Problem Size (Small $\to$ XL). `research_history/figures/FigC_DifficultyScaling_Publication.png`
*   **Figure D: Failure Taxonomy.** (Data: EXP-14). PCA scatter plot of G3 failures showing distinctive "Collapsed" and "Wandering" clusters. `research_history/figures/FigD_FailureTaxonomy_Publication.png`
*   **Figure E: The Commitment Curve.** (Data: EXP-14). Time-series of Radius of Gyration for G4 vs G1, showing the delayed phase transition in successful reasoning. `research_history/figures/FigE_CommitmentCurve_Publication.png`

For full definitions of all metrics, see [Metrics_Appendix_2026-02-10.md](Metrics_Appendix_2026-02-10.md).
