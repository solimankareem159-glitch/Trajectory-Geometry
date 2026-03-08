# Experiment 10: Self-Report Consistency

**Phase:** 3 — The Pivot to Dynamics & Intervention
**Date:** November 2025
**Model:** Qwen2.5-0.5B
**Status:** Completed — **FAILURE (of hypothesis)**

## Connection to Prior Work

EXP-09 proved that *correctness* has a geometric signature (Speed, Curvature, Stabilization). If these signatures represent "modes of thought," the model might have introspective access to them.

## Research Question

**Can the model "feel" its own geometry?** Does high curvature "feel" like exploration? Does high speed "feel" like effort? Is model self-report a reliable readout of internal geometric state?

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

## Conclusions & Implications

**Verdict: FAILURE (of the hypothesis).** **Introspection is a Hallucination.** The model's verbal "self-monitoring" is not a readout of its internal geometric state; it is just another generation, driven by text surface form rather than latent dynamics. We cannot ask the model "Are you stuck?" — we must *measure* if it is stuck using geometry.

## Influence on Next Experiment

*   Since the model cannot report its own state, we must build **external** geometric monitors. This reinforced the value of the trajectory metrics as the only reliable window into reasoning quality.
*   **EXP-11** expanded the metric suite with deeper topological metrics (Effective Dimension, Tortuosity) to build a richer external diagnostic.
