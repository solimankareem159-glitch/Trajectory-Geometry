# Experiment 10: Self-Report Consistency

**Date / Context:** October 2026 / Phase 3: The Pivot to Dynamics & Intervention

## Motivation & Prior Assumptions
*   **Context:** Experiment 09 proved that *correctness* has a geometric signature (Speed, Curvature).
*   **Hypothesis:** If these signatures represent "modes of thought" (e.g., exploratory vs. direct), the model might have introspective access to them.
*   **Question:** Can the model *feel* its own geometry? Does high curvature "feel" like exploration? Does high speed "feel" like effort?

## Experimental Setup
*   **Model:** Qwen2.5-0.5B
*   **Task:** Multi-step arithmetic (Same behaviorally as Exp 9).
*   **Protocol:**
    1.  **Solve:** Model generates a Chain-of-Thought solution.
    2.  **Measure:** We compute the objective geometric metrics (Speed, Curvature, Stabilization).
    3.  **Introspect:** We immediately prompt the model to rate its own process on 1–5 scales:
        *   *Effort* (Automatic vs Effortful)
        *   *Certainty* (Uncertain vs Certain)
        *   *Exploration* (Direct vs Exploratory)
        *   *Smoothness* (Stuck vs Flowing)
    4.  **Perturb:** We re-ask the ratings after a minimal irrelevant input ("ACK") to test stability.

## Results & Observations

### 1. The Decoupling of Introspection
We found **zero significant correlation** between the model's self-report and its physical trajectory.

*   **Effort vs Speed:** $r \approx 0.00$
    *   The model does not report higher effort when its state moves faster or covers more distance.
*   **Certainty vs Stabilization:** $r \approx -0.01$
    *   The model is just as "certain" in chaotic, diverging trajectories as in converging ones.
*   **Exploration vs Directional Consistency:** Weak/Noisy.
    *   We aimed to see if winding paths (low consistency) were rated as "Exploratory". They were not reliably identified as such.

### 2. The Illusion of Awareness
The self-reports were:
1.  **Inconsistent:** Perturbing the context with a simple "ACK" often changed the ratings significantly.
2.  **Performance-Biased:** The model tended to rate itself high on "Certainty" regardless of the internal state, likely a training artifact (RLHF preference for confident-sounding answers).

## Interpretation
*   **Result:** FAILURE (of the hypothesis).
*   **Key Insight:** **Introspection is a Hallucination.** The model's verbal "self-monitoring" is not a read-out of its internal geometric state; it is just another generation, likely driven by the text surface form rather than the latent dynamics.
*   **Implication:** We cannot ask the model "Are you stuck?". We must *measure* if it is stuck (using Geometry).

## Next Steps
*   Since the model cannot *tell* us its state, and we cannot relying on it to self-correct based on "feeling" stuck, we must build **external** monitors.
*   Proceed to **Experiment 11** (or 9C/10B variants): attempting to *force* a geometric change (Steering) rather than just observing it.
