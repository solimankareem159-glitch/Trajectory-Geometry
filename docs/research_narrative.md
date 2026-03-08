# Trajectory Geometry: Consolidated Research History

**Date:** 2026-03-08
**Experiment Range:** EXP-01 through EXP-19B (25 experiment entries)
**Author:** Kareem Soliman
**Companion Documents:** [Findings_Catalogue_2026-02-10.md](Findings_Catalogue_2026-02-10.md) | [Metrics_Appendix_2026-02-10.md](Metrics_Appendix_2026-02-10.md)

---

## Overview

This document is the auditable, chronological record of the Trajectory Geometry research program — a project that began with the question "does cognitive operation leave a detectable shape in a language model's representation?" and evolved, through repeated failure and methodological reinvention, into a series of findings about how transformers physically navigate representational space when they reason.

Over 25 experiments spanning approximately 17 months, the research progressed from API output embeddings and clustering (EXP-01) to Probability Cloud Regression and multi-architecture invariant signature identification (EXP-19B). Each experiment is recorded with its research question, motivation, method, results, interpretation, and an explicit bridge to the next experiment. Phase-level narrative sections explain the larger pivots. The document captures both what was found and why the search evolved the way it did — including the seven failed or invalid experiments that shaped the eventual approach.

The theoretical origin was **Dynamic Semantic Geometry (DSG)** — a framework from psychology proposing that cognitive operators enact measurable geometric transformations on representational manifolds. The DSG framework was largely wrong in its specific predictions but generated the right questions. This history records how those questions were answered.

---

## Research Phase Map

| Exp | Phase | Date | Verdict | One-Line Summary |
|-----|-------|------|---------|-----------------|
| 01 | I — Shapes | Nov 2025 | Invalid | API output embeddings fail; topic noise dominates operator signal |
| 02 | I — Shapes | Nov 2025 | Invalid | Scalar warp magnitude loses direction; all operators look identical |
| 03 | II — Invariants | Nov 2025 | Failed | Regime-mapping finds topic confound, not operator signal |
| 04 | II — Detour | Nov 2025 | Success | Forcing multi-pass thinking (OG-MPT) improves accuracy 53%→77% |
| 05 | II — Detour | Nov 2025 | Invalid | Safety OG-MPT collapses at 0.5B scale; capacity floor discovered |
| 06 | II — Wilderness | Nov 2025 | Weak | Stricter validation (LOPO/LOCO) still yields F1~0.30-0.48 |
| 07 | II — Ceiling | Nov 2025 | Valid/Insufficient | Static stability 0.924 at L16, but separability F1~0.30 only |
| 08 | III — Dynamics | Nov 2025 | Breakthrough | Differential geometry metrics identify 9 computational regimes |
| 09 | III — Dynamics | Nov 2025 | Breakthrough | Speed/Curvature separates G4 vs G1 at d>3.0; reasoning≠hallucination |
| 09B | III — Dynamics | Nov 2025 | Failure | TinyLlama <1% accuracy; capability floor required for geometry |
| 10 | III — Dynamics | Nov 2025 | Failure | Model self-reports r≈0.00 with geometry; introspection unreliable |
| 11 | IV — Failure | Nov 2025 | Success | Dimensional Collapse: G4 D_eff≈13.1 vs G1≈3.4, d>4.5 |
| 12 | IV — Failure | Nov 2025 | Success | Fractal density and two-phase Explore→Commit transition confirmed |
| 13 | IV — Failure | Dec 2025 | Success | Two failure subtypes; AUC 0.898/0.772; "Retrieve-vs-Compute" |
| 14 | V — Paradigm | Dec 2025 | Breakthrough | 10/14 metrics flip sign; no universal success; regime-relative geometry |
| 15 | V — Validation | Dec 2025 | Success | Difficulty scaling d≈5→17; geometry AUC 0.79 > length AUC 0.77 |
| 16 | V — Scale | Dec 2025 | Success | Scale stable 0.5B→1.5B; Goldilocks zone; hallucination pipeline fixed |
| 16B | V — Scale | Dec 2025 | Success | Pythia-70m 0%→100% accuracy post parser fix; architecture independence |
| 17 | VI — Robustness | Feb 2026 | Partial | 3B scale tested; hardware ceiling limits full replication |
| 18 | VI — Robustness | Feb 2026 | Success | 54 metrics across 12 families formalized; TrajectoryMetrics class |
| 18B | VI — Robustness | Feb 2026 | Partial/Invalid | Pythia-70m data corrupted; pipeline hard constraints established |
| 19 | VI — Robustness | Feb 2026 | Breakthrough | 19 invariant signatures; Success Attractor confirmed; 3 architectures |
| 19B | VI — Robustness | Mar 2026 | Breakthrough | PCR corrects attenuation bias; AUC +0.119 at L0; G3 Position Index |

---

## Phase I: The Intuition of Shapes (EXP-01–02)

The project grew from a theoretical framework in cognitive psychology. While developing PsychScope, a construct-based AI evaluation tool, the question arose: if language models instantiate cognitive operations — summarization, critique, elaboration — do those operations leave a measurable trace in the model's representational geometry? The framework proposed that different operators would produce characteristic "warp vectors" detectable in embedding space.

Two experiments tested this intuition using the tools most immediately available: API embeddings and open-weights hidden states. Both failed, but the failures were informative about what to look for instead.

---

### EXP-01: Geometric Signatures (Turn-Level)

**Date:** November 2025 | **Verdict:** Invalid

**Research Question:** Do different cognitive operators (Summarize, Critique, Elaborate) leave distinct geometric signatures in the embedding space of model outputs?

**Prior State:** The Dynamic Semantic Geometry framework predicted that operators produce characteristic "warp vectors" — the displacement between input and output embeddings. No prior measurement attempt had been made.

**Method:**
- **Model:** Gemini API (`text-embedding-004`)
- **Dataset:** 500 single-turn prompts (10 operators × 10 paraphrases × 5 topics)
- **Procedure:** Computed warp vectors $w_t = E(\text{response}_t) - E(\text{turn}_t)$; applied K-Means and HDBSCAN clustering; attempted GRU classification
- **Metrics:** AMI, Silhouette Score, Within/Between variance ratio

**Results:**
- K-Means AMI = **0.13** (barely above random); HDBSCAN found **0 clusters**
- Within-operator variance (0.296) **exceeded** between-operator variance (0.112)
- Two paraphrases of "Summarize" were more different from each other than "Summarize" was from "Critique"
- GRU predictor: 57–66% accuracy (vs 10% random), likely learned transition artifacts

**Interpretation — Invalid:** Output embeddings capture lexical and topic content, not the "cognitive act" of the operator. The warp vector conflates semantic content change with the operator transformation. No signal above noise.

**Bridge to EXP-02:** The failure of output embeddings motivated a move inside the model. If the signature exists at all, it must live in the *internal* hidden states before the collapse to text. EXP-02 accessed those states directly via an open-weights model.

---

### EXP-02: Latent Factors (Token-Level)

**Date:** November 2025 | **Verdict:** Invalid

**Research Question:** Do hidden state trajectories in open-weights model internals show operator-distinct warp traces, and are composite operators decomposable into single-operator components?

**Prior State:** EXP-01 showed API output embeddings were uninformative. The hypothesis was that the signal existed in hidden states before the final token projection.

**Method:**
- **Model:** Qwen2.5-0.5B (first use of open-weights model)
- **Metric:** Warp $W_t = ||h_t - h_{t-1}||_2$ (magnitude of consecutive hidden state change)
- **Procedure:** Ran single and composite operator prompts; extracted hidden states across all 24 layers; applied NMF to decompose warp traces; attempted reconstruction of composite operators
- **Analysis:** Layers 5–12 for "stable geometry"; Layer 24 as output

**Results:**
- **Universal signature:** All operators produced the same warp trace — a massive spike at $t=0$ followed by a flat line
- No distinct "oscillating" or "ramping" profiles for specific operators
- Linear reconstruction worked, but only because "spike+flat" reconstructs trivially from "spike+flat" basis vectors
- Middle layers (5–12): coherent geometry; final layer (24): chaotic vocabulary projection noise

**Interpretation — Invalid:** The **magnitude** metric discards all directional information. The $t=0$ spike is a universal transformer property (first embedding ingestion), drowning any operator-specific signal. The problem wasn't the model — it was the metric.

**Bridge to EXP-03:** The realization that direction matters more than magnitude motivated a search for regime-specific *directional* patterns. The model's "Thinking" tokens might move in different *directions* than "Speaking" tokens, even if the magnitudes looked the same. EXP-03 tested this by separating Listen, Think, and Speak phases.

---

## Phase II: The Invariant Trap and The Wilderness (EXP-03–07)

The next five experiments attempted to rescue the static operator hypothesis through increasingly rigorous measurement: regime separation, cross-validation schemes, adversarial datasets, and finally a large-scale definitive test. Two of the five were productive in unexpected ways (EXP-04 proved that forcing dynamics works; EXP-07 established the exact ceiling of static analysis). Three confirmed the dead end.

The critical intellectual shift happened slowly. The intuition was that the static vector exists but is hard to find. The data showed, repeatedly, that even when the technique worked technically, the practical signal was too weak to be useful. By EXP-07, the evidence was conclusive: *position* cannot explain *competence*.

---

### EXP-03: Regime Invariants

**Date:** November 2025 | **Verdict:** Failed

**Research Question:** Does a "Summarization vector" (or any operator vector) persist as the model transitions between Listen, Think, and Speak processing phases?

**Prior State:** EXP-02 showed magnitude metrics failed. The new hypothesis was that operators leave directional signatures that are invariant across processing regimes — a stable "abstract representation" of the operator.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 10 operators × 2 topics (AI, Climate) × 10 paraphrases; balanced 50/50 topic split per operator
- **Procedure:** Extracted hidden states per regime (Listen/Think/Speak); trained linear probes (Logistic Regression) within- and cross-regime; CCA for shared subspace detection

**Results:**
- **Probes achieved 100% accuracy** — but this was **topic confounding**: the balanced design made any topic detector a perfect operator detector
- Cross-regime coupling maps: $R^2 < 0.07$ (< 7% variance explained)
- CCA alignment accuracy: **27.5%** (marginally above random 10%)
- No evidence of a shared geometric subspace between Listen and Think regimes

**Interpretation — Failed:** "Thought" is not a rotated version of instruction. The transition from processing input to generating output involves a fundamentally non-linear transformation, not a linear rotation or scaling. The "Static Vector" theory of operators was effectively dead.

**Bridge to EXP-04/08:** This killed the static position approach. But before fully pivoting, two experiments tested whether the *dynamic* failure was just a measurement problem (EXP-04 explored whether forced dynamics worked; EXP-06/07 made last-ditch static attempts). EXP-04 was a productive detour: it showed that *controlling* the dynamic process improves outcomes, even if we can't *measure* the static position.

---

### EXP-04: Operator-Gated Multi-Pass Thinking (OG-MPT)

**Date:** November 2025 | **Verdict:** Success

**Research Question:** If we cannot *find* the operator in the vector space, can we *force* it? Does explicitly structuring the model's internal monologue (Plan→Calculate→Verify) produce measurable performance gains?

**Prior State:** Experiments 1–3 failed to find static signatures. The complementary hypothesis: if dynamics matter (as Exp 3 implied), then *controlling* the dynamic process should improve behavior.

**Method:**
- **Model:** Qwen2.5-0.5B (Baseline vs. Orchestrator)
- **Dataset:** 60 rigorous prompts (Math, Constraints, Safety), generated via ChatGPT metaprompting and adversarially filtered
- **Procedure:** Heuristic task detector; Orchestrator forced multi-turn ChatML conversation per task type (Reasoning: Plan→Calculate→Verify→Answer; Constraints: List→Draft→Check→Answer; Safety: Identify→Check→Refuse/Answer)
- **Evaluation:** Strict regex + programmatic checking

**Results:**
- Overall accuracy: Baseline **53.3%** → Orchestrator **76.6%** (+23.3%)
- Constraint task: **40%** → **85%** (+45%)
- Reasoning (Math/Logic): +25%

**Interpretation — Success:** Cognitive capability is not just raw weights; it is **control flow**. A small model can substantially outperform its single-pass baseline when the *structure* of its thinking is made explicit. This proved that dynamics matter for performance even if the static vector theory is wrong.

**Bridge to EXP-05/08:** EXP-04 confirmed that forcing a thought structure works. EXP-05 tested if this generalized to safety. When EXP-05 failed, the combined signal was clear: the dynamic approach works for structured problems but requires model capacity. Eventually, EXP-08 returned to *measuring* these dynamics rather than forcing them.

---

### EXP-05: Safety Resilience (OG-MPT Expansion)

**Date:** November 2025 | **Verdict:** Invalid

**Research Question:** Does multi-pass gating (Plan→Check→Refuse/Answer) make a small model more resistant to jailbreaks?

**Prior State:** EXP-04 showed OG-MPT works for math and constraints. The question was whether it generalizes to safety — a more complex judgment task requiring the model to assess intent, not just execute a plan.

**Method:**
- **Model:** Qwen2.5-0.5B (Baseline vs. Orchestrator)
- **Dataset:** 54 prompts (benign controls + adversarial attacks: "Fictional Sandbox", "Authority Override")
- **Procedure:** Safety OG-MPT: Identify Harm → Check Policy → Refuse/Answer
- **Baseline ASR:** 56.8% (high vulnerability)

**Results:**
- OG-MPT **collapsed** into infinite repetition loops (e.g., repeating "Gründe" 75 times) or echoed the system prompt
- Only 15 valid OG-MPT samples vs 54 baseline samples — statistically meaningless comparison
- A code error also misclassified benign controls, rendering the helpfulness metric invalid

**Interpretation — Invalid:** The 0.5B model lacked the instruction-following bandwidth to maintain state across 3 internal passes. Complex dynamic architectures require a minimum capability threshold (likely 7B+ for stable multi-pass safety reasoning).

**Bridge to EXP-06:** EXP-05's failure, combined with EXP-04's success, clarified boundaries. The gating approach works when the task structure is well-defined and the steps are computationally within the model's reach. The focus returned to *measurement* rather than intervention — specifically, to finding what static signal remains once better controls are applied.

---

### EXP-06: Pilot Metric Validation (The Wilderness)

**Date:** November 2025 | **Verdict:** Inconclusive / Weak

**Research Question:** If topic confounding is removed via Leave-One-Paraphrase-Out (LOPO) and Leave-One-Content-Out (LOCO) cross-validation, does any meaningful static operator signal remain?

**Prior State:** EXP-03 found strong-looking results that were entirely confounded by topic. EXP-06 tested whether the signal survived strict holdout schemes.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Design:** Pilot subset; 3 operator classes
- **LOPO:** Train on $N-1$ paraphrases, test on held-out paraphrase
- **LOCO:** Train on $N-1$ content types, test on held-out content
- **Control:** 10,000-shuffle permutation test for empirical chance baseline (F1 ~0.24)

**Results:**
- LOPO F1: **~0.30–0.48** (chance ~0.33)
- LOCO F1: **~0.45–0.50** (chance ~0.33)
- No single layer or metric strongly separated operators across all contexts
- Permutation baseline: F1 ~0.24 (confirming slight but real above-chance performance)

**Interpretation — Weak:** Signal exists marginally above random — but "distinct cognitive operators" should produce F1 > 0.8 to be practically useful. Extensive engineering of validation controls revealed a signal too weak to be the foundation of a measurement system.

**Bridge to EXP-07:** One definitive large-scale test was needed before abandoning the static approach entirely. If the signal at F1 ~0.30 was just due to small-N pilot noise, a dataset of N=2,000 should reveal the true ceiling. EXP-07 ran that test.

---

### EXP-07: Static Operator Geometry (The Ceiling)

**Date:** November 2025 | **Verdict:** Valid but Insufficient

**Research Question:** With N=2,000 samples, do operator centroids form stable, separable geometric structures in the model's hidden space?

**Prior State:** EXP-06 found weak signal. The hypothesis was that scale would reveal it. If the centroids are stable but just noisy with small N, a large dataset should close the gap.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** N=2,000 (10 operators × 20 content domains × 10 paraphrases)
- **Procedure:** "Speed Patch" directional delta extraction; centroid stability (mean pairwise cosine similarity across content domains); Nearest Centroid Classifier (NCC)

**Results:**
- **Layer 16 stability: 0.924** — operators do form stable abstract internal states
- NCC F1: **~0.30** (chance 0.10) — high stability but low separability
- Related operators (Critique vs Question) shared closer subspaces than unrelated pairs — semantic topology is present but fuzzy

**Interpretation — Valid but Insufficient:** Both hypotheses partially confirmed: operators *do* produce stable representations (H2, stability 0.924), but they are too overlapping to distinguish reliably (F1 0.30). This hit the exact ceiling of what static analysis can deliver. The "Summarize" direction exists — it's just too fuzzy for practical control.

**Bridge to EXP-08:** The decisive paradigm shift. EXP-07 proved position-based analysis is insufficient — not because the signal is absent, but because it is too blurry. The question became: if we measured *how* the state moves (speed, curvature, trajectory shape) rather than *where* it sits, would the signal be sharper? EXP-08 answered: yes, dramatically so.

---

## Phase III: The Pivot to Dynamics (EXP-08–10)

The shift from static to dynamic analysis was not a single decision but the accumulated weight of seven experiments that all pointed to the same ceiling. EXP-07 provided the final evidence. If static coordinates couldn't separate operators with d > ~0.3 F1, the measurement instrument was wrong — not the underlying reality.

The correct instrument turned out to be differential geometry: not *where* states are, but *how* they move. Speed, curvature, volume — physical properties of trajectory paths rather than positions in a fixed coordinate system.

EXP-08 established the measurement framework. EXP-09 applied it to success vs failure and found the most striking result in the entire program. EXP-09B and EXP-10 probed the boundaries of that finding.

---

### EXP-08: Trajectory Geometry

**Date:** November 2025 | **Verdict:** Foundational Success

**Research Question:** Can differential geometry metrics (Speed, Curvature, Radius of Gyration) applied to transformer hidden state sequences identify distinct computational "regimes"?

**Prior State:** EXP-07 established the ceiling of static analysis. The key insight from EXP-04 and EXP-07 combined: *forcing* dynamic structure improves performance, and static *positions* cannot distinguish operators. The natural next step: measure the *motion*.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Metrics defined:**
  - **Speed:** $||h_t - h_{t-1}||_2$ — Euclidean step size
  - **Curvature:** Cosine angle between entrance and exit vectors at each state
  - **Radius of Gyration ($R_g$):** Volume of the trajectory cloud
- **Procedure:** K-Means on *dynamic metrics* (not states) to find regimes; Layer 13 analysis
- **EXP-08' sub-experiment:** Cue-word injection test — measured metric change before/after "Wait", "Therefore", "Plan"

**Results:**
- K=9 clusters found with silhouette score **0.179** and **69% predictability** (Layer 13)
- Distinct modes of motion confirmed — the model switches between recognizable "computational modes"
- **EXP-08' result: FAILED** — cue-word injection did not reliably shift geometric regime ($p > 0.05$)

**Interpretation — Foundational Success:** The right measurement instrument was found. Dynamic trajectory metrics reveal computational structure that static position metrics completely missed. However, dynamics are emergent — they cannot be triggered by individual tokens.

**Bridge to EXP-09:** EXP-08 could identify *that* regimes exist but not which were "good" or "bad." EXP-09 applied these metrics to the most important question: does a *correct* thought look different from a *wrong* one?

---

### EXP-09: Geometry-Capability Correlation

**Date:** November 2025 | **Verdict:** Breakthrough

**Research Question:** Do the dynamic trajectory metrics from EXP-08 distinguish successful reasoning (G4: CoT Success) from failed direct retrieval (G1: Direct Failure)?

**Prior State:** EXP-08 had the metrics but no outcome labels. EXP-04 had outcome variation (Success vs Failure) but no trajectory metrics. EXP-09 combined the two.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 300 multi-step arithmetic problems (e.g., $(A \times B) + C$)
- **Groups:**
  - **G1 (Direct Failure):** Answers immediately and incorrectly
  - **G4 (CoT Success):** Uses chain-of-thought and answers correctly
- **Metrics:** Speed, Directional Consistency, Stabilization Rate
- **Control:** Window size equalized across groups to prevent length confound

**Results:**
- **Speed:** G4 is **3–4× faster** than G1, Cohen's $d > 3.0$ (Layer 24) — successful thought moves with more energy
- **Directional Consistency:** G1 ~0.5 (near-straight path); G4 ~0.05 (winding path); $d = 2.6$
- **Stabilization:** G4 stabilizes in final layers (convergence); G1 destabilizes (wandering noise)
- Effect sizes exceeded all conventional thresholds ($p < 0.001$, permutation-tested)

**Interpretation — Breakthrough:** *Hallucination is geometrically a straight line. Reasoning is a winding path.* The model literally traverses more representational volume when it reasons correctly. A "System 1" direct failure fires ballistically to a wrong answer. A "System 2" CoT success diffuses through semantic space, exploring before converging. This was the first result that could be used to detect reasoning quality from latent geometry alone — without reading the output.

**Bridge to EXP-09B/10:** Two boundary conditions needed testing. First: does this replicate on a different architecture? Second: does the model *know* its own geometry — can it introspect on these states?

---

### EXP-09B: Cross-Model Replication (TinyLlama)

**Date:** November 2025 | **Verdict:** Failure (Technical)

**Research Question:** Do the geometric signatures of EXP-09 replicate on TinyLlama-1.1B-Chat?

**Prior State:** EXP-09 found strong signatures in Qwen-0.5B. Cross-architecture replication would support the claim that these are universal computational properties.

**Method:**
- **Model:** TinyLlama-1.1B-Chat
- **Dataset:** 300 multi-step arithmetic problems (same as EXP-09)
- **Same metrics:** Speed, Curvature, Stabilization

**Results:**
- G4 (CoT Success) = **<1%** across 300 problems (confirmed by manual audit)
- No statistics possible with 1–2 success samples in 300
- The model could not generate coherent chains of thought

**Interpretation — Failure (Technical):** Geometry requires a capability floor. If the model cannot reason, there is no reasoning geometry to measure. This is not a failure of the geometric framework — it is a boundary condition: the framework applies to models with sufficient capacity to actually execute the task.

**Bridge to EXP-10:** Returned to Qwen-0.5B. The next question: if external observers (us) can detect geometry, can the model detect it *about itself*? EXP-10 tested whether verbal self-reports correlated with measured geometric states.

---

### EXP-10: Self-Report Consistency

**Date:** November 2025 | **Verdict:** Failure of Hypothesis

**Research Question:** Do the model's verbal self-reports of effort, certainty, exploration, and smoothness correlate with its objective geometric trajectory metrics?

**Prior State:** EXP-09 showed geometry tracks reasoning quality. If the model's internal geometric state is meaningfully structured, it might have implicit introspective access. This would enable the model to self-monitor and self-correct.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Protocol:**
  1. **Solve:** Generate chain-of-thought solution
  2. **Measure:** Compute objective Speed, Curvature, Stabilization
  3. **Introspect:** Prompt model to rate Effort, Certainty, Exploration, Smoothness (1–5 scales)
  4. **Perturb:** Re-ask ratings after minimal irrelevant input ("ACK") to test stability

**Results:**
- **Effort vs Speed:** $r \approx 0.00$
- **Certainty vs Stabilization:** $r \approx -0.01$
- **Exploration vs Directional Consistency:** Weak, noisy, non-significant
- Self-reports were **inconsistent**: perturbing with "ACK" often changed ratings significantly
- Reports were **performance-biased**: model rated itself high on Certainty regardless of internal state (RLHF training artifact)

**Interpretation — Failure of Hypothesis:** Introspection is a hallucination. The model's verbal self-monitoring is not a readout of its geometric state — it is another generation, driven by surface text patterns rather than latent dynamics. We cannot ask the model "Are you stuck?" We must *measure* whether it is stuck.

**Bridge to EXP-11:** With introspection ruled out as a monitoring channel, the focus shifted entirely to external geometric measurement. EXP-11 deepened the metric suite to characterize the *structure* of failure and success more precisely.

---

## Phase IV: The Richness of Failure (EXP-11–13)

The breakthrough of EXP-09 showed that geometry separates success from failure. The next three experiments asked a harder set of questions: *what kind of failure?* *What kind of success?* And *can we predict outcome purely from geometry?*

These experiments introduced the metric suite that defined the project's analytical vocabulary: Effective Dimension, Tortuosity, Fractal Dimension, and the failure subtype taxonomy. The key discovery was that failures are not monolithic — they have distinct geometric signatures that reveal different *mechanisms* of failure, not just different *degrees* of failure.

---

### EXP-11: Extended Geometric Suite

**Date:** November 2025 | **Verdict:** Success

**Research Question:** Can an extended metric suite (Effective Dimension, Tortuosity, Directional Autocorrelation) reveal additional structural properties of success vs failure trajectories?

**Prior State:** EXP-09 established Speed and Curvature as predictors. The hypothesis was that these were proxies for deeper topological properties — specifically, the dimensionality and path efficiency of the trajectory manifold.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 300 arithmetic problems (EXP-09 dataset)
- **New Metrics:**
  - **Effective Dimension ($D_{eff}$):** PCA participation ratio — how many principal components are needed to explain the trajectory variance
  - **Tortuosity ($\tau$):** End-to-end distance / total arc length
  - **Turning Angle:** Mean angular change between consecutive step deltas
  - **Directional Autocorrelation:** Whether one step direction predicts the next

**Results:**
- **Dimensional Collapse (G4 vs G1):** G4 $D_{eff} \approx 13.1$; G1 $D_{eff} \approx 3.4$; Cohen's $d > 4.5$
- **Critical nuance:** Even *failed* CoT (G3) maintained $D_{eff} \approx 13.9$ — CoT prompting forces high-dimensional engagement regardless of outcome
- **Tortuosity:** G4 $\tau \approx 0.04$ (extremely winding); G1 $\tau \approx 0.40$ (relatively straight)
- Turning Angle and Directional Consistency were highly correlated ($r > 0.9$) — redundant information

**Interpretation — Success:** The landmark discovery: **Dimensional Collapse in Failure**. Successful reasoning is not just faster or more curved — it is fundamentally higher-dimensional. A failing direct answer collapses into ~3 effective dimensions (a thin "line" through representational space). Successful CoT reasoning uses ~13 dimensions. The theoretical reframe: CoT is not "extra compute" but **dimensional expansion** — unfolding compressed representations into a space where they can be manipulated.

**Bridge to EXP-12:** EXP-11 established dimensionality as a key predictor. EXP-12 probed the *texture* of that high-dimensional space: is it fractally complex? Does it have temporal structure (a "beginning" and "end" phase)?

---

### EXP-12: Advanced Geometric Diagnostics

**Date:** November 2025 | **Verdict:** Success

**Research Question:** Does successful reasoning show higher fractal complexity and a measurable two-phase (Explore→Commit) temporal structure?

**Prior State:** EXP-11 showed that success is high-dimensional. The hypothesis was that successful trajectories are not just high-dimensional but *fractally dense* — they revisit and re-evaluate representational regions, rather than simply traversing high-dimensional space in a straight line.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Advanced Metrics:**
  - **Fractal Dimension ($D_f$):** Space-filling complexity of the trajectory
  - **Intrinsic Dimension (MLE):** Minimum variables to describe the manifold
  - **Convergence Slopes:** Rate at which tokens approach the final hidden state
  - **Early-Late Ratio:** Trajectory "energy" in first vs second half
  - **RQA (Recurrence Quantification):** Repeating patterns (determinism, laminarity)

**Results:**
- **Fractal Complexity:** G4 $D_f \approx 2.0$ vs G1 $D_f \approx 1.7$ — reasoning is fractally denser
- **Two-Phase Transition:** Success showed high-dimensional expansion (Exploration) followed by sharp contraction (Commitment); failure showed flat convergence or divergent wandering
- **Layer-wise Peak:** Geometric signal (G4 vs G1 delta) peaked in **Layers 10–16** — reasoning is "born" in the computational middle of the network
- **RQA finding:** Determinism/Laminarity metrics too noisy in 0.5B model for reliable use

**Interpretation — Success:** Reasoning is not just high-dimensional — it is *fractally dense*. The model iterates and re-evaluates in a way that fills the local representational volume. The two-phase profile (Explore→Commit) was confirmed as a structural feature of success, not an artifact. The middle layers (10–16) host the reasoning computation; early layers parse input, late layers output.

**Bridge to EXP-13:** The next question: are all failures the same? The failure cases had been treated as a single group. But the emerging picture — some failures are flat, some wander — suggested two mechanistically distinct failure modes. EXP-13 mined the data for these subtypes without collecting new data.

---

### EXP-13: Regime Mining and Failure Subtyping

**Date:** December 2025 | **Verdict:** Success

**Research Question:** Do CoT failures (G3) cluster into mechanistically distinct subtypes? And how accurately does trajectory geometry predict correctness?

**Prior State:** EXP-12 described what success looks like. The failure cases had been analyzed only as a single group. The emerging intuition was that not all failures were the same: some were "never engaged" and some were "actively lost."

**Method:**
- **Model:** Qwen2.5-0.5B
- **Strategy:** Deep mining of existing dataset (no new data collection)
- **Clustering:** K-Means ($k=3$) on 31-metric suite
- **Phase Transition Detection:** Sliding window "Dimension Drop" (early dim − late dim)
- **Predictive Modeling:** Logistic Regression on trajectory metrics (5-fold cross-validation)
- **Analysis Layer:** Layer 13

**Results:**
- **Two failure subtypes confirmed:**
  - **Subtype A — "The Broken Engine" (Collapsed Failure):** High tortuosity, low $D_{eff}$; model never engaged the reasoning regime; geometrically indistinguishable from a Direct answer
  - **Subtype B — "The Lost Wanderer" (Incoherent Failure):** High expansion, high dimensionality, but negative convergence; model "thought hard" but drifted away from the solution basin
- **Direct Success (G2) profile:** Higher speed, straighter paths, dramatically lower $D_{eff}$ than CoT Success — confirmed "Retrieve-and-Commit" signature
- **G4 Commitment:** Significant Dimension Drop in second half of trajectory — the geometric "Consensus" signature
- **Predictive Power:** AUC **0.898** for Direct answers, **0.772** for CoT; vs. prompt-type alone AUC ~0.63

**Interpretation — Success:** Geometry can now diagnose *mechanism* (Retrieval vs Reasoning) and *failure mode* (Collapse vs Confusion) without reading the output text. The commitment signature (Dimension Drop) was confirmed as a real structural feature of success. The failure taxonomy added diagnostic richness: a "Collapsed" failure needs regime engagement; a "Wandering" failure needs commitment.

**Bridge to EXP-14:** The 10-metric suite was proving reliable. The next question was whether a more comprehensive metric expansion would reveal a *universal* success signature — one that works regardless of prompting regime. EXP-14 tested this hypothesis directly, and its answer was the most important result in the project.

---

## Phase V: Paradigm Shift and Validation (EXP-14–16B)

The transition from Phase IV to Phase V was driven by a specific hypothesis: if geometry predicts success this well within one regime, perhaps there is a *universal* success signature that works across regimes. EXP-14's refutation of that hypothesis — finding instead that metrics flip sign — was the project's most important single result. It reframed every subsequent analysis.

EXP-15 stress-tested the difficulty dimension. EXP-16 and EXP-16B extended the findings to new scales and architectures, encountering and solving the hallucination contamination problem along the way.

---

### EXP-14: Comprehensive Metric Expansion

**Date:** December 2025 | **Verdict:** Breakthrough

**Research Question:** Does a 33-metric suite computed across all 28 layers reveal a universal geometric signature of success — one that holds regardless of prompting regime?

**Prior State:** EXP-13 established high predictive accuracy. The hypothesis (H1) was that success would look the same whether achieved via CoT or Direct answering. The project's first strong universal claim.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Full Depth:** 28 layers
- **New Metrics (33 total):**
  - `cos_to_running_mean` (Coherence)
  - `time_to_commit` (Phase timing — token of maximum $R_g$ drop)
  - `msd_exponent` (Diffusion character)
  - `interlayer_alignment` (Cross-layer synchrony)
  - `spectral_entropy` (Path complexity)
- **Full trajectory capture** (up to 128 tokens) replacing prior 32-token window

**Results:**
- **Paradigm shift:** **10 of 14 key metrics showed opposite effects** for CoT vs Direct success
  - CoT Success: Lower Speed, Smaller $R_g$, Higher Coherence (Focusing)
  - Direct Success: Higher Speed, Larger $R_g$, Lower Coherence (Ballistic Retrieval)
- **`time_to_commit`:** Direct Success commits at ~5 tokens; CoT Success at ~11 tokens; Failures commit late or not at all
- **Layer depth profiles:**
  - **Layers 0–7:** Regime detection (CoT vs Direct), Cohen's $d = 6$–$8$
  - **Layers 10–14:** Within-regime success prediction, $d = 1.0$–$2.2$
  - **Layers 20–24:** Commitment timing

**Interpretation — Breakthrough:** **There is no universal success signature.** H1 was falsified. Success geometry is regime-dependent — what looks "good" under CoT criteria looks like failure under Direct criteria. A "correct" Direct answer looks like a failed CoT trajectory. This fundamentally changes how monitoring must be designed: any success detector must first identify the active computational regime before assessing trajectory quality. The regime-specific monitoring architecture became the project's central methodological principle.

**Bridge to EXP-15:** With the regime-relativity principle established, the next challenge was the length confound — the objection that geometry is just a proxy for token count. EXP-15 directly tested this.

---

### EXP-15: Stress-Testing the Phase Transition

**Date:** December 2025 | **Verdict:** Success

**Research Question:** Does geometric expansion scale with problem difficulty (independent of trajectory length), and does geometry outpredict length in success classification?

**Prior State:** EXP-14 confirmed regime-relative geometry. The most natural skeptical objection: "CoT works because longer outputs contain more information; geometry is just a proxy for length." EXP-15 tested this directly.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 300 problems, stratified by difficulty (Small to Extra Large by operand size)
- **Ablation:** Geometric metrics vs response length in logistic regression (AUC comparison)
- **Anomaly Analysis:** "Direct-Only Successes" — cases where CoT failed but Direct answered correctly

**Results:**
- **Difficulty Scaling:** $R_g$ effect size (G4 vs G1) scaled monotonically with difficulty:
  - Small: $d \approx 5.0$ → Extra Large: $d > 17.0$ (Layer 4)
- **Geometry vs Length:**
  - Geometry only: AUC **0.79**
  - Length only: AUC **0.77**
  - Geometry subsumes the predictive value of length
- **"Overthinking" signature:** Direct-Only Successes showed artificial dimensionality expansion on problems the model had already memorized — CoT introduced noise by over-unfolding a memorized answer

**Interpretation — Success:** The model selectively allocates geometric expansion proportional to problem complexity — a **resource-rational** strategy. Geometry is not downstream of length; it is the *mechanism* that length reflects. Long trajectories predict success only insofar as they contain genuine dimensional complexity. The overthinking signature further confirmed this: expansion that isn't resolution-driven is noise.

**Bridge to EXP-16:** The next critical test: does this replicate on a larger model (1.5B parameters) and a different architecture (Pythia)? EXP-16 and 16B ran those replications.

---

### EXP-16 / EXP-16-Salvage / EXP-16B: Scale and Architecture Pivot

**Date:** December 2025 | **Verdict:** Success

**Research Questions:** (a) Do geometric signatures scale from 0.5B to 1.5B parameters? (b) Are these signatures architecture-independent (Qwen vs Pythia)?

**Prior State:** EXP-15 established the geometric framework on Qwen-0.5B. Architecture independence was a necessary condition for generalizability claims. Qwen-1.5B and Pythia-70m were the test cases.

**Method:**
- **Models:** Qwen2.5-1.5B (Exp 16), Pythia-70m (Exp 16-Salvage)
- **Dataset:** Multi-step arithmetic (same benchmark)
- **New pipeline requirement:** Boundary-aware hallucination truncation (both models exhibited "Runaway Hallucination" — generating new questions after answering)
- **Pythia salvage:** Re-ran with boundary-aware parser after discovering the original 0% accuracy was a parsing artifact

**Results:**
- **Qwen-1.5B (post-truncation):** Replicated all primary signatures — dimensional expansion, regime-relative geometry, scale-stable effect sizes
- **Pythia-70m salvage:** Initial **0% accuracy** → **100% accuracy** post-parser fix; geometric signatures confirmed for architecture independence at 70M parameters
- **Cross-architecture finding:** Success geometry is relative to the model's dominant failure mode:
  - Qwen-0.5B fails by *collapsing* (repetition/looping) → success looks like **expansion**
  - Qwen-1.5B / Pythia fail by *wandering* (hallucination/gibberish) → success looks like **compression**
  - **Goldilocks Zone:** Success is controlled expansion — between the extremes of collapse and wandering

**Interpretation — Success:** Scale stability confirmed from 70M to 1.5B. Architecture independence confirmed across LLaMA-style (Qwen) and GPT-style (Pythia) transformers. The Goldilocks Zone finding deepened the regime-relativity principle: "success" is not just regime-dependent but model-dependent, calibrated against the model's characteristic failure mode.

**Bridge to EXP-17:** With the core findings replicated across scales and architectures, the project shifted to validation robustness. EXP-17 extended the scale ladder to 3B parameters and tested the full 33-metric pipeline on the larger model.

---

## Phase VI: Robustness, Scaling, and Denoising (EXP-17–19B)

With the core geometric framework validated through EXP-16B, the final phase addressed four questions: (1) Does the scale ladder extend to 3B? (2) Can the metric suite be standardized for replication? (3) Are the quantitative findings robust across architectures and difficulty bins in a single large experiment? (4) Are the raw AUC estimates accurate, or are they biased by measurement noise?

EXP-17 encountered hardware limits at 3B. EXP-18 formalized the metric framework. EXP-18B stress-tested the pipeline and discovered data integrity issues. EXP-19 was the definitive multi-architecture robustness study. EXP-19B applied Probability Cloud Regression to correct attenuation bias — revealing that the signal was even stronger than previously measured.

---

### EXP-17: Baseline Replication & Multi-Mode Prompting

**Date:** February 2026 | **Verdict:** Partial

**Research Question:** Do the EXP-09 geometric signatures replicate when computed via the full 33-metric pipeline on Qwen2.5-3B? Does the scale ladder extend to 3B parameters?

**Prior State:** EXP-16B confirmed signatures up to 1.5B. The next natural scale step was 3B.

**Method:**
- **Model:** Qwen2.5-3B-Instruct
- **Phase 17A:** Direct replication of EXP-09 with 300 problems; full 33-metric pipeline
- **Phase 17B:** 8-mode multi-mode prompting — same content across different computational modes
- **Hardware:** AMD RX 5700 XT, 8GB VRAM via DirectML

**Results:**
- Regime-relative geometry confirmed on 3B scale
- 33-metric pipeline successfully computed on 3B model
- **Limitation:** Hardware ceiling at 8GB VRAM with DirectML began causing instability at 3B scale; full replication dataset was incomplete

**Interpretation — Partial:** The 3B model appears consistent with smaller Qwen models, but hardware constraints prevented statistically robust replication. The scale ladder extends to 3B in principle but is not conclusively validated.

**Bridge to EXP-18:** The hardware ceiling at 3B redirected focus to formalizing the metric framework for the validated 0.5B–1.5B range. EXP-18 consolidated the metric suite into a replication-ready standard.

---

### EXP-18: Consolidated Metric Suite

**Date:** February 2026 | **Verdict:** Success (Infrastructure)

**Research Question:** Can the full geometric metric suite be formalized into a standardized, replication-ready framework for cross-experiment and cross-model comparison?

**Prior State:** The metric suite had grown organically through EXP-01 to EXP-16B from 3 metrics to ~33, spread across multiple scripts with inconsistent implementations. This created barriers to replication and cross-experiment comparison.

**Method:**
- **Model:** Qwen2.5-0.5B (existing hidden states from EXP-09 and EXP-14 combined)
- **Scope:** 54 distinct metrics grouped into 12 conceptual families:
  - Kinematic, Volumetric, Convergence, Diffusion, Spectral, RQA
  - Cross-Layer, Landmark, Attractor, Embedding Stability, Information, Inference
- **Output:** `TrajectoryMetrics` class (`metric_suite.py`) for comprehensive geometric profiling

**Results:**
- 54 metrics formalized with consistent implementations across all families
- Combined dataset (EXP-09 + EXP-14 hidden states) successfully processed through the new pipeline
- Framework validated: metrics reproduce prior results with matching effect sizes

**Interpretation — Success (Infrastructure):** The project transitioned from exploratory data analysis to a standardized analytical framework. The 54-metric suite enables rigorous cross-experiment and cross-model comparison with consistent implementations. This was a necessary precondition for the large-scale EXP-19 replication study.

**Bridge to EXP-18B:** Before deploying the new suite on all three architectures, a multi-architecture stress test was needed. EXP-18B ran the suite on Pythia-70m, Qwen-0.5B, and Qwen-1.5B simultaneously — and discovered critical pipeline vulnerabilities.

---

### EXP-18B: Scaling Geometry (Pipeline Stress Test)

**Date:** February 2026 | **Verdict:** Partial / Invalid (Pythia)

**Research Question:** Does the 54-metric suite compute correctly across all three architectures simultaneously, and what attractor dynamics emerge at scale?

**Prior State:** EXP-18 formalized the metric suite. EXP-18B would apply it to all three validated architectures using existing hidden state data from EXP-14 (Qwen-0.5B), EXP-16 (Pythia-70m), and EXP-16B (Qwen-1.5B).

**Method:**
- **Models:** Pythia-70m, Qwen2.5-0.5B, Qwen2.5-1.5B
- **Reused hidden states** from prior experiments
- **Focus:** 57 metrics across 12 families; attractor dynamics, distance to success centroids

**Results:**
- **Pythia-70m data corruption discovered:** Hidden states loaded as 1,536-dim instead of expected 512-dim — completely invalidating all Pythia-70m results
- **Broadcasting crash:** `attractor_metrics` crashed when $T_{ref} = D$ (ambiguous numpy broadcasting)
- **Windows multiprocessing overhead:** Heavy `torch`/`transformers` imports at global scope in `spawn` mode caused memory pressure and UI freezing
- **No persistence:** Any crash wiped all progress — entire pipeline redesigned for atomic operations and incremental CSV appending

**Interpretation — Partial/Invalid:** EXP-18B functioned as an involuntary pipeline stress test. Pythia-70m results were invalidated by data corruption. But the experiment established hard constraints that made all subsequent work more robust: mandatory pre-flight tensor shape validation, explicit dimension handling for centroids, lazy imports for multiprocessing, process isolation to prevent memory leakage.

**Bridge to EXP-19:** EXP-18B's failures clarified exactly what the robust pipeline required. EXP-19 ran a clean, fresh data collection on three architectures — upgrading Pythia from 70m to 410m to match the 24-layer depth of Qwen-0.5B — using all the engineering lessons from EXP-18B.

---

### EXP-19: Robustness Replication

**Date:** February 2026 | **Verdict:** Breakthrough

**Research Question:** Do the core geometric signatures replicate across three disparate architectures (Qwen-0.5B, Qwen-1.5B, Pythia-410m) on a fresh 400-problem dataset with strict anti-contamination controls, and can a set of architecture-invariant signatures be identified?

**Prior State:** EXP-18B had identified pipeline vulnerabilities; EXP-18 had formalized the metric suite. EXP-19 was the definitive validation study — new data collection, upgraded architectures, strict guardrails.

**Method:**
- **Models:** Qwen2.5-0.5B, Qwen2.5-1.5B, Pythia-410m *(upgraded from 70m to match 24-layer depth)*
- **Dataset:** 400 problems, balanced across 4 difficulty bins (Small, Medium, Large, Negative)
- **Key Design Improvements:**
  - **Anti-contamination pipeline:** Multi-stage guardrails (prompt engineering, generation stop sequences, post-generation text truncation, boundary detection) eliminating "runaway hallucination"
  - **Few-shot calibration:** CoT-guided few-shot examples enabling non-zero accuracy for the 410m model
  - **Physical Trajectory Preservation:** 1,200 full 200-token trajectories stored on external HDD; semantic and post-answer geometry captured

**Results — Accuracy:**

| Model | UltraSmall | Small | Overall CoT |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-1.5B** | 100.0% | 100.0% | **95.0%** |
| **Qwen2.5-0.5B** | 100.0% | 50.0% | **45.0%** |
| **Pythia-410m** | 25.0% | 0.0% | **5.0%** |

**Results — Top Invariant Predictors (G4 vs G1, cross-architecture):**

| Metric | Avg Cohen's $d$ | Peak |
| :--- | :--- | :--- |
| `phase_count` | 31.66 | Pythia-410m L23: $d = 70.11$ |
| `radius_of_gyration` | 13.99 | Qwen-1.5B L20: $d = 11.14$ |
| `effective_dimension` | 12.01 | — |
| `commitment_sharpness` | 9.83 | — |
| `tortuosity` | 6.93 | — |
| `direction_consistency` | 6.45 | — |

- **19 architecture-invariant signatures** identified (|Cohen's $d$| > 2.0 across all three models)
- **"The Snap" phenomenon:** Sharp phase transition (Commitment Sharpness) at the moment the model locks onto the correct solution — architecturally universal
- **Physical trajectory persistence:** Geometric signals survive beyond semantic answer boundaries; post-answer drift correlates with preceding reasoning quality

**Interpretation — Breakthrough:** The core geometric signatures are **architecturally invariant** — they hold across both LLaMA-style (Qwen) and GPT-style (Pythia) transformers, spanning 410M to 1.5B parameters. The Success Attractor — a tight, reproducible geometric manifold that successful reasoning trajectories converge onto — is a real, measurable, and universal feature of transformer computation.

**Bridge to EXP-19B:** The raw AUC estimates from EXP-19 (within-CoT AUC ~0.78) might themselves be biased. Geometric metrics computed from short token sequences have high per-token variance, which compresses estimated AUC toward chance — a statistical phenomenon known as attenuation bias. EXP-19B applied Probability Cloud Regression to correct for this.

---

### EXP-19B: Probability Cloud Regression (PCR) Reanalysis

**Date:** March 2026 | **Verdict:** Breakthrough

**Research Question:** Are the AUC estimates from EXP-19 accurate, or are they deflated by measurement noise? Does PCR denoising reveal a stronger underlying signal — and can regime-quality variance be formally decomposed?

**Prior State:** EXP-19 established within-CoT AUC ~0.78 at Layer 16. However, short token sequences produce noisy geometric feature estimates. If noise attenuates the true signal, raw AUC underestimates the model's true predictability.

**Method:**
- **Data:** EXP-19 hidden states (Qwen-0.5B, Qwen-1.5B)
- **PCR Methodology:**
  1. **Uncertainty estimation:** Per-trajectory $\sigma$ estimated from the standard deviation of each metric across layers of that trajectory
  2. **Denoising (Mode B):** Features projected onto a "true" manifold via `CloudRegressor`, anchored to sample ID (leakage-free — not anchored to correctness labels)
  3. **Re-prediction:** Logistic regression re-run on denoised features
- **Variance Decomposition:** Two-way ANOVA (Regime × Correctness) on 20+ metrics across all layers
- **Position Index:** Measure of where G3 (CoT Failure) sits on the axis from G1 (Direct Failure, PI=0) to G4 (CoT Success, PI=1)

**Results — PCR AUC Recovery (Qwen-0.5B):**

| Layer | Raw AUC | PCR-Corrected AUC | Gain |
| :--- | :--- | :--- | :--- |
| 0 | 0.659 | 0.778 | **+0.119** |
| 5 | 0.700 | 0.779 | **+0.079** |
| 16 | 0.799 | 0.779 | −0.020 *(slight over-smooth)* |

**Results — Regime-Quality Decomposition:**
- **Main Effect (Regime):** ~80–85% of total geometric variance — Direct and CoT trajectories are physically separated in embedding space
- **Main Effect (Quality):** η² ≈ 0.10 — robust within-regime signal persisting after regime is controlled
- **Within-CoT AUC:** Qwen-0.5B **0.78** (L16); Qwen-1.5B **0.74** (L26); far above regime-only baseline of 0.50
- **Interaction signatures — sign flips:**
  | Metric | Layer | Direct ($d$) | CoT ($d$) |
  |--------|-------|------------|---------|
  | `full_time_to_commit` | 3 | +1.50 | −0.38 |
  | `clean_cos_slope_to_final` | 4 | −0.33 | +0.46 |

**Results — G3 Position Index:**
- **Aggregate PI ≈ 0.033** — CoT Failures geometrically almost identical to Direct Failures overall
- **Layer-resolved:** G3 PI ~1.0 in early layers — CoT failures successfully *enter* the CoT attractor but fail to *commit* to the Success Centroid

**Interpretation — Breakthrough:**
1. **Proto-attractors form at Layer 0.** Raw measurements obscure this due to high per-token variance; PCR reveals the signal is present immediately. True predictability is likely **>0.85** if noise were perfectly eliminated.
2. **Deep layers (L16+) already have high SNR** — PCR provides minimal marginal gain there, confirming that commitment geometry solidifies progressively through the network.
3. **G3 failure is a commitment failure, not a regime-entry failure.** CoT failures enter the CoT manifold correctly (early-layer PI ~1.0) but cannot converge onto the Success Centroid. This refines the failure taxonomy established in EXP-13.
4. **The Success Attractor is real and distinct** from the CoT Regime Attractor. Within-regime geometry predicts success with AUC 0.78 — far above the 0.50 regime-only baseline.

---

## Methodology Evolution Table

| Decision Point | What Failed | What Replaced It | Experiment |
|---------------|-------------|------------------|------------|
| API output embeddings | Topic and lexical noise dominated operator signal | Open-weights hidden state extraction | EXP-01→02 |
| Scalar magnitude metrics ($||v||$) | Direction discarded; all operators identical | Vector-valued trajectory analysis | EXP-02→08 |
| Static centroid analysis | Operators overlap (F1 ~0.30); position too fuzzy | Dynamic regime classification (Speed, Curvature) | EXP-07→08 |
| Cross-regime probe (balanced dataset) | Topic confounding produced false 100% accuracy | LOPO/LOCO cross-validation; then full dynamic pivot | EXP-03→06→07 |
| Universal success model | 10/14 metrics flip sign across regimes | Regime-conditional monitoring | EXP-14 |
| Fixed 32-token analysis window | Missed late-stage dynamics and commitment | Full trajectory capture with controlled truncation | EXP-14→16B |
| Naive output parsing | Hallucination contamination; 0% Pythia accuracy → parsing artifact | Boundary-aware truncation pipeline | EXP-16 salvage |
| Cross-model replication (TinyLlama-1.1B) | Model below capability floor (0% accuracy) | Architecture-matched capability verification | EXP-09B→16B→19 |
| Introspective self-monitoring | $r \approx 0.00$ with objective geometry | External geometric measurement only | EXP-10 |
| Exploratory metric growth | Inconsistent implementations, no standard | Formalized `TrajectoryMetrics` class (54 metrics) | EXP-18 |
| Multi-architecture using stale hidden states | Pythia-70m data corruption (1,536 vs 512 dim) | Pre-flight tensor shape validation; fresh collection | EXP-18B→19 |
| Cross-process global scope imports | Windows spawn OOM on heavy torch imports | Lazy imports; process isolation | EXP-18B→19 |
| Raw AUC estimates | Attenuation bias from short-sequence noise | Probability Cloud Regression (PCR) denoising | EXP-19B |

---

## Established Knowledge (EXP-01 through EXP-19B)

### Core Empirical Findings

1. **Dimensional Collapse in Failure** (EXP-11, 12): Failed direct reasoning collapses into ~3 effective dimensions. Successful CoT reasoning uses ~13. Cohen's $d > 4.5$. Replicated at scale.
2. **Regime-Relative Success Geometry** (EXP-14): 10 of 14 metrics flip sign between CoT and Direct success. No universal "good trajectory" exists; success is regime-dependent.
3. **Difficulty-Driven Geometric Expansion** (EXP-15): $R_g$ effect size scales from $d \approx 5.0$ (Small) to $d > 17.0$ (XL) — resource-rational allocation of geometric volume to problem entropy.
4. **Failure Subtypes** (EXP-13, 19B): Collapsed failures (never engaged reasoning) and Wandering failures (engaged but not committed). G3 failures enter the CoT manifold but fail to commit to the Success Centroid.
5. **Commitment Timing** (EXP-12, 14): Direct answers commit at ~5 tokens; CoT at ~11 tokens. Measurable phase transition (Dimension Drop / "The Snap") is architecturally universal.
6. **Architecture-Invariant Signatures** (EXP-19): 19 signatures hold across Qwen and Pythia families, 410M–1.5B. Phase Count ($d=31.66$), Radius of Gyration ($d=13.99$), Effective Dimension ($d=12.01$).
7. **Regime-Quality Decomposition** (EXP-19B): Regime explains ~80–85% of geometric variance; quality effect (η² ≈ 0.10) is robust within regime (within-CoT AUC 0.78).
8. **PCR-Corrected Signal Strength** (EXP-19B): True predictability likely >0.85; proto-attractors form as early as Layer 0 but are obscured by early-layer noise in raw measurements.

### Confirmed Null Results

- Static operator vectors are too fuzzy to be useful (F1 ~0.30 maximum)
- Cue-word triggering does not reliably shift geometric regimes (EXP-08')
- Model self-reports do not correlate with internal geometry ($r \approx 0.00$, EXP-10)
- Universal success signatures do not exist (EXP-14)
- Models below capability floor have no reasoning geometry to measure (EXP-09B)

### Open Questions as of EXP-19B

- Do these signatures generalize beyond arithmetic to multi-hop reasoning, ambiguous questions, or retrieval tasks?
- Do they persist at frontier scale (>10B parameters)?
- Can intervening on trajectory geometry (activation patching) causally redirect G3-bound trajectories toward the Success Attractor?
- Does the full PCR-corrected metric suite replicate on non-Qwen, non-Pythia architectures (Llama-3, Gemma)?

---

*For the empirical fact layer, see [Findings_Catalogue_2026-02-10.md](Findings_Catalogue_2026-02-10.md).*
*For complete metric definitions, see [Metrics_Appendix_2026-02-10.md](Metrics_Appendix_2026-02-10.md) and [Trajectory_Geometry_Definitive_Metric_Suite.md](Trajectory_Geometry_Definitive_Metric_Suite.md).*
*For PCR analysis data, see `experiments/EXP-19_Robustness_2026-02-14/data/analysis_19b/`.*
