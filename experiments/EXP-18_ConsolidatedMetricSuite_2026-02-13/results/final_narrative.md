# The Geometry of Correctness: A Coherent Story of Experiment 18

## The Core Thesis

Computation in transformer models is not merely about "getting the right answer token"; it is a **dynamic trajectory through representation space** that follows specific geometric laws. Experiment 18, utilizing 54 definitive metrics, confirms that success and failure are geometrically distinct regimes.

## 1. The Ballistic Signature of Failure (G1)

Failed "Direct Answer" trajectories (G1) are characterized by **high early directional consistency**. At Layer 0, G1 shows a massive separation from CoT success (Cohen's d = 6.9).
> **The Story:** The model has "made up its mind" before the computation even begins, drifting ballistically toward a pre-determined (incorrect) output region. It lacks the complex, winding exploration seen in successful reasoning.

## 2. The Coherence of Reasoning (G4 vs G2)

One of the most striking findings is the difference between "Direct Success" (G2) and "Chain-of-Thought Success" (G4).

- **Metrics:** `cosine_to_running_mean` and `dir_autocorr_lag2` show extreme separation (d = -7.5).
- **The Story:** CoT trajectories are significantly more **self-consistent and geometrically persistent**. Each step builds upon the "momentum" of the previous steps. Direct success, by contrast, looks more like a lucky "leap" that lacks the sustained internal structure of a reasoning chain.

## 3. The Bifurcation Point (G4 vs G3)

Where does reasoning go wrong?

- **Diagnostic Layers:** Layers 16-18 are the "decision zones."
- **Key Signal:** `distance_slope` and `cosine_slope` show that by Layer 16, successful trajectories are "falling into" the attractor basin of the correct answer, while failures are decelerating or drifting away.
- **The Story:** Failure in CoT (G3) isn't necessarily a random walk; it's a **failure to commit** to the attractor. The trajectory remains "tortuous" and distant from the success centroid even as it approaches the final tokens.

## 4. Summary: The "Geometric Sweet Spot"

The most discriminative information about a model's performance is found in **Layers 17 through 19**. Analysis in these layers can predict success with far higher confidence than surface-level output logits.

### Key Geometric Rules

1. **Exploration is mandatory**: Low directional consistency early (Layer 0) is a prerequisite for complex reasoning.
2. **Momentum is evidence**: High directional autocorrelation at mid-layers (Lag 2) confirms the model is following a structured computation path.
3. **Attractors are real**: Successful trajectories cluster tightly around a population-defined "success centroid" in the final layers.
