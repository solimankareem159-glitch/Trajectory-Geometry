# Trajectory Geometry: The Shape of Reasoning in Transformers

**A Research Narrative**
**Author:** Kareem Soliman
**Date:** February 2026

---

## Overview

This document tells the story of a research project that began with a question about creativity and ended with a method for watching transformers think. Over the course of 17 experiments, the project evolved from searching for static "thought vectors" in language model representations to discovering that the *motion* of hidden states through representational space carries rich, measurable signatures of computational quality. The findings include dimensional collapse in failed reasoning, regime-dependent success geometry, failure subtyping, measurable commitment timing, and difficulty-driven expansion — all with effect sizes far exceeding conventional thresholds.

The path was not linear. It involved seven failed or invalidated experiments, a complete paradigm shift, several methodological dead ends, and at least two results that were nearly discarded before being salvaged through careful forensic re-analysis. This narrative records the decisions that shaped the project and the evidence that forced those decisions.

---

## Part I: The Theoretical Origin

The project grew from work in psychology. While developing **PsychScope**, a framework for construct-based AI evaluation, an attempt to operationalize creativity measurement led to a theoretical insight: the distance between any two concepts is not fixed. It depends on the observer's position in representational space. A computational biologist sees biology and computer science as neighbours; a university administrator sees them as different buildings. The geometry of meaning shifts with perspective.

This was formalized as **Dynamic Semantic Geometry (DSG)** — a framework built on perspective-dependent metric tensors, mobile centroids, and visibility envelopes. Five types of conversational operators were proposed (Lens, Bridge, Wedge, Prism, Anchor), each defined by its geometric effect on the representational manifold. The framework made a specific prediction: if operators correspond to real geometric transformations, their signatures should be detectable in the hidden states of language models.

The theoretical framework was rich, internally consistent, and almost entirely wrong in its original form. But it generated the right questions.

---

## Part II: The Search for Static Signatures (EXP-01 through EXP-07)

### Phase 1: "Does thought have a shape?" (EXP-01 to EXP-02)

The first experiment embedded conversational operator outputs using a closed API model (Gemini `text-embedding-004`) and attempted to cluster them by operator type.

**Result: Failure.** Clustering produced AMI of 0.13 (barely above random). Within-operator variance exceeded between-operator variance — two paraphrases of "Summarize" were more different from each other than "Summarize" was from "Critique." The signal was swamped by topic and lexical content.

Experiment 02 moved inside an open-weights model (Qwen2.5-0.5B), decomposing hidden states with NMF. All operators produced the same warp signature: a spike at token 0 followed by a flat line. The magnitude-based metric discarded all directional information, and universal transformer dynamics drowned out any operator-specific signal.

**Lesson:** Output embeddings and scalar magnitude metrics were the wrong instruments. The signal, if it existed, required looking inside the model and preserving directional information.

### Phase 2: "Is the shape universal?" (EXP-03 to EXP-07)

Experiment 03 tested whether a "Summarization vector" persists across processing phases (Listen, Think, Speak). Linear probes achieved 100% accuracy — which turned out to be an artifact of topic confounding in the balanced experimental design. Cross-regime coupling maps explained less than 7% of variance. CCA found no shared subspace.

**Lesson:** Thought is not a rotated version of instruction. The geometry undergoes non-linear transformation when the model shifts from processing input to generating output. The "static vector" theory was dead.

Experiments 04-07 represented what the research notes call "The Wilderness" — attempts to rescue the static paradigm through better models, stricter cross-validation (LOPO, LOCO), and larger datasets (N=2,000). Experiment 04, the Operator-Gated Multi-Pass Thinking experiment, was a productive detour: forcing the model through structured Plan-Calculate-Verify passes improved accuracy from 53% to 77%, proving that *dynamics matter* even if the static signatures couldn't be found. Experiment 05 (safety-focused OG-MPT) collapsed — the 0.5B model lacked the capacity for multi-pass safety reasoning.

Experiment 07 was the definitive ceiling test. With 2,000 samples, operator centroids achieved stability of 0.924 at Layer 16, confirming that the model *does* abstract operator instructions into stable internal states. But the nearest-centroid classifier achieved only F1 of 0.30. The states were stable but indistinguishable from each other.

**Verdict:** Static structure exists but is insufficient. You can find a "Summarize" direction, but it is too fuzzy for practical use. Seven experiments had converged on the same conclusion: the signal is not in *where* the state is, but in *how it moves*.

---

## Part III: The Pivot to Dynamics (EXP-08 to EXP-10)

### "It's not coordinates; it's the trajectory" (EXP-08)

Experiment 08 defined the first differential geometry metrics for transformer hidden states: Speed (Euclidean step size), Curvature (angular change between steps), and Radius of Gyration (trajectory cloud volume). Applied to diverse prompts, K-means clustering on these dynamic metrics identified 9 distinct computational regimes at Layer 13 with 69% predictability.

A sub-experiment (EXP-08') tested whether injecting cue words ("Wait", "Therefore") could trigger geometric transitions. It could not. Dynamics were emergent properties of the computation, not simple reactions to individual tokens.

**Significance:** This was the foundational success. The research now had the right measurement instrument.

### The Breakthrough (EXP-09)

Experiment 09 combined the dynamic metrics from EXP-08 with success/failure classification from 300 multi-step arithmetic problems under Direct and Chain-of-Thought prompting.

The effect sizes were not subtle. Successful CoT reasoning (G4) moved 3-4x faster than Direct failures (G1), with Cohen's d exceeding 3.0 for speed at Layer 24. Directional consistency separated the groups at d = 2.6. Failed trajectories were ballistic — straight lines to the wrong answer. Successful trajectories wound through representational space, exploring before converging.

**The insight:** Hallucination is geometrically a straight line. Reasoning is a winding path.

### The Cross-Model Attempt and Self-Report Failure (EXP-09B, EXP-10)

Experiment 09B attempted replication on TinyLlama-1.1B. It produced fewer than 1% correct CoT answers. No statistics were possible. This established an important boundary condition: **geometry requires a capability floor.** If the model cannot reason, there is no reasoning geometry to measure.

Experiment 10 tested whether the model could introspect on its own geometry — self-reporting effort, certainty, exploration, and smoothness after solving problems. The correlation between verbal self-reports and objective geometric metrics was zero (r ~ 0.00 across all dimensions). Self-reports were inconsistent, performance-biased, and easily perturbed by irrelevant context.

**Lesson:** The model cannot *tell* you its internal state. You must *measure* it. Introspection, in this model, is a hallucination about a hallucination.

---

## Part IV: The Richness of Failure (EXP-11 to EXP-13)

### Dimensional Collapse (EXP-11)

Experiment 11 extended the metric suite to include Effective Dimension (PCA participation ratio), Tortuosity, and Directional Autocorrelation.

The result was the project's first landmark finding: **Dimensional Collapse**. Successful CoT trajectories (G4) explored approximately 13 effective dimensions. Failed Direct trajectories (G1) collapsed to approximately 3. Cohen's d exceeded 4.5.

An important nuance: even *failed* CoT (G3) maintained high dimensionality (~13.9). The act of chain-of-thought prompting forces the model into high-dimensional space regardless of outcome — it engages the engine, even if the steering is wrong.

**Theoretical shift:** CoT is not "extra compute." It is **dimensional expansion** — unfolding compressed representations into a high-dimensional manifold where they can be manipulated.

### The Texture of Thought (EXP-12)

Experiment 12 introduced Fractal Dimension, Intrinsic Dimension (MLE), and Convergence diagnostics. Successful reasoning showed higher fractal complexity (D_f ~ 2.0 vs 1.7), confirming that reasoning is not just high-dimensional but *fractally dense* — it iterates and re-evaluates in a way that fills the local representational volume.

The experiment also identified the **middle-layer reasoning peak**: the geometric signature of success was strongest in Layers 10-16, not at the input or output. Reasoning happens in the computational middle of the network.

### Mining for Subtypes (EXP-13)

A key strategic decision shaped Experiment 13: rather than generating new data, the existing dataset was mined more deeply. This was driven partly by compute constraints and partly by the intuition that the signal was already present but not being examined at the right resolution.

Unsupervised clustering of failure trajectories (G3, CoT Failures) revealed two distinct failure subtypes:
- **Type A: Collapsed Failure** — High tortuosity, low effective dimension. The model never engaged the reasoning regime. Geometrically indistinguishable from a Direct answer.
- **Type B: Wandering Failure** — High expansion, high dimensionality, but no convergence. The model "thought hard" but drifted away from the solution.

The experiment also confirmed the **Retrieve-and-Commit** profile for Direct Success (G2): higher speed, straighter paths, and dramatically lower effective dimension than CoT Success. Direct success is a retrieval event; CoT success is a computational event.

Trajectory geometry predicted correctness with AUC of 0.898 for Direct answers and 0.772 for CoT — well beyond what prompt type alone could explain (AUC 0.63).

---

## Part V: The Paradigm Shift and Validation (EXP-14 to EXP-16B)

### "Good Geometry is Regime-Relative" (EXP-14)

Experiment 14 expanded the metric suite to 33 variables computed across all 25+ layers and produced what is probably the project's most important finding.

The hypothesis was that a universal success signature would emerge. **It did not.** Instead, 10 of 14 key metrics showed *opposite* direction effects for CoT success versus Direct success. At Layer 13:

| Metric | CoT Success | Direct Success |
|--------|-------------|----------------|
| Speed | Lower | Higher |
| Effective Dimension | Lower | Higher |
| Radius of Gyration | Lower | Higher |
| Cosine to Running Mean | Higher | Lower |

A trajectory that looks "successful" under CoT criteria looks like failure under Direct criteria. **There is no universal "good trajectory" detector.** What constitutes good geometry depends on which computational regime is active.

This experiment also introduced **time_to_commit** — the token position at which the Radius of Gyration drops most sharply, capturing the explore-to-commit phase transition. Direct successes committed at ~5 tokens. CoT successes committed at ~11 tokens. Failures committed late or not at all.

A three-layer diagnostic architecture emerged from the data:
- **Layers 0-7:** Regime detection (CoT vs Direct), d = 6-8
- **Layers 10-14:** Within-regime success prediction, d = 1.0-2.2
- **Layers 20-24:** Commitment timing

### Stress-Testing (EXP-15)

Experiment 15 addressed the length confound: the objection that CoT works because it produces more tokens, not because of any geometric property.

Problem difficulty was stratified by operand size. The effect size for Radius of Gyration (G4 vs G1) scaled monotonically with difficulty: d ~ 5.0 for "Small" problems, spiking to d > 17.0 for "Extra Large" problems. The model selectively allocates geometric expansion proportional to problem complexity.

Geometric metrics alone (AUC 0.79) outperformed response length alone (AUC 0.77) for predicting success. Geometry subsumes the predictive value of length.

Cases where CoT *hurt* performance showed an "Overthinking" signature: artificial dimensionality expansion on problems the model had already memorized, introducing noise and error.

### Cross-Scale Validation (EXP-16 and EXP-16B)

Replication on Qwen2.5-1.5B initially looked chaotic. The model exhibited "Runaway Hallucination" — generating correct answers followed by endless repetitions of new questions. This was identified as metric contamination, not a failure of the geometric framework.

After implementing a truncation pipeline that identified answer boundaries and cleaned the trajectories, the 1.5B model reproduced all primary geometric signatures with stable effect sizes.

A separate salvage effort recovered the Pythia-70m dataset. The initial 0% accuracy had been a parsing artifact — the same hallucination-after-answer pattern caused the parser to grab wrong values. With a boundary-aware parser, Pythia achieved 100% accuracy on the arithmetic tasks, and its geometric signatures confirmed architecture independence at the extreme small scale.

An important cross-architecture finding emerged: **the geometric signature of success is relative to the model's dominant failure mode.** Qwen-0.5B's dominant failure was collapse (repetition/looping), so success looked like expansion. Qwen-1.5B and Pythia's dominant failure was wandering (hallucination/gibberish), so success looked like compression. Success is a "Goldilocks zone" of controlled expansion — distinct from both extremes.

---

## Part VI: What the Evidence Shows

### Established Findings (Replicated, p < 0.001, permutation-tested)

1. **Dimensional Collapse in Failure:** Failed reasoning collapses into low-dimensional subspace (D_eff ~ 3 vs ~13, d > 4.5)
2. **Regime-Relative Success Geometry:** 10 of 14 metrics flip sign between CoT and Direct success
3. **Difficulty-Driven Expansion:** Geometric expansion scales with problem complexity (d up to 17.0)
4. **Failure Subtypes:** Two structurally distinct failure modes — Collapsed (premature optimization) and Wandering (uncontrolled exploration)
5. **Commitment Timing:** Measurable phase transition from exploration to execution (Direct ~5 tokens, CoT ~11 tokens)
6. **Geometry Predicts Correctness:** AUC 0.898 for Direct, 0.772 for CoT — beyond prompt type alone
7. **Scale Stability:** Effect sizes stable from 70M to 1.5B parameters

### Negative Results (Equally Important)

1. **Static operator vectors do not exist** as practically useful constructs (EXP-01 through EXP-07)
2. **API embeddings are insufficient** — internal hidden states are required (EXP-01)
3. **Cue-word triggering does not work** — dynamics are emergent, not token-reactive (EXP-08')
4. **Model self-reports do not correlate with internal geometry** — introspection is unreliable (EXP-10)
5. **Cross-model analysis requires a capability floor** — models that cannot reason have no reasoning geometry (EXP-09B)
6. **Universal success signatures do not exist** — geometry is regime-dependent (EXP-14)

### Open Questions

- Do these signatures generalize beyond arithmetic to other reasoning domains?
- Do they persist at frontier scale (>10B parameters)?
- Can intervening on trajectory geometry (via activation patching) causally change outcomes?
- Does the full metric suite replicate on non-Qwen architectures (Llama, Gemma)?

---

## Part VII: The Evolution of Method

The project's methodology evolved substantially, often driven by failures:

| Decision Point | What Failed | What Replaced It |
|---------------|-------------|-------------------|
| API embeddings | Topic/lexical noise dominated | Open-weights hidden state extraction |
| Scalar magnitude metrics | Directional information lost | Vector-valued trajectory analysis |
| Static centroid analysis | Operators overlap too heavily (F1 ~0.30) | Dynamic regime classification |
| Universal success model | Metrics flip sign across regimes | Regime-conditional monitoring |
| Fixed 32-token window | Missed late-stage dynamics | Full trajectory capture with controlled truncation |
| Cross-model replication (TinyLlama) | Capability floor not met | Architecture-matched capability verification |
| Naive output parsing | Hallucination contamination | Boundary-aware truncation pipeline |

The methodological arc can be summarized: from looking at *where* states are to watching *how* they move; from assuming universal structure to respecting regime-dependent dynamics; from trusting model output at face value to forensic verification of every data pipeline.

---

## Note on Methodology and Collaboration

This research was conducted by a psychology researcher working with AI tools. The theoretical framework, experimental design logic, hypothesis generation, result interpretation (including uncomfortable ones), and course corrections were human contributions. The mathematical formalization, code implementation, statistical computation, and experimental execution were performed in collaboration with AI systems — primarily Claude (Anthropic) for theoretical development and experimental design, Google Antigravity for running experiments, and ChatGPT for comparative analysis.

The experiments were run locally on personal hardware using open-weights models (Qwen2.5-0.5B/1.5B, Pythia-70m). All reported statistics use permutation testing (10,000 shuffles) at p < 0.001. The full experimental dataset, metric suite, and analysis pipeline are available in this repository.

This collaboration model — domain expertise directing AI-assisted implementation — is itself an experiment in how research can be conducted. The findings stand or fall on their statistical merits, regardless of who or what wrote the Python.

---

## References and Further Reading

- **Experiment Summaries:** See `/experiment_summaries/` for concise accounts of each experiment
- **Experimental Logs:** See `/research_history/Experimental_logs/` for detailed methodology
- **Metric Definitions:** See `Metrics_Appendix_2026-02-10.md` for all 36 metrics
- **Findings Catalogue:** See `Findings_Catalogue_2026-02-10.md` for the empirical fact layer
- **Literature Context:** See `Perplexity Deep Research.md` for positioning within the field
