# Trajectory Geometry: A Ledger of Contributions & Guiding Intuitions

This document serves as a historical record of the *Trajectory Geometry* research project, specifically isolating the user's (Researcher's) key intellectual contributions, intuitions, and strategic decisions that shaped the direction and success of the work.

While AI assistants (ChatGPT, Claude) executed code and performed statistical analyses, the **conceptual framework, experimental design logic, and critical course-corrections were driven by the Researcher.**

## I. Foundational Framework (The "Why" and "How")

### 1. The Geometry of Motion vs. Static Representation
*   **Researcher Intuition:** Early in the project, the Researcher rejected the standard approach of analyzing static hidden state snapshots. They proposed that the *shape* and *dynamics* of the trajectory itself—how the model moves through space—contains the signal for reasoning.
*   **Impact:** This shifted the entire project from simple classification (Projected PCA) to trajectory analysis (Curvature, Velocity, Tortuosity).
*   **Quote from Logs:** *"It's not about where it is, it's about how it moves."*

### 2. "Thinking" as a Physical Act (Latent Dynamics)
*   **Researcher Intuition:** The Researcher consistently framed the model's computation in physical terms—"momentum," "stabilization," "hesitation," and "commitment."
*   **Impact:** Led to the development of specific metrics:
    *   **Velocity/Acceleration:** To measure "hesitation" or "speed of thought."
    *   **Radius of Gyration:** To measure the "spread" of exploration.
    *   **Turning Angle:** To measure "changes of mind."

### 3. The "Arithmetic" Benchmark Choice
*   **Researcher Decision:** Despite the trend of using complex logical riddles (GSM8K), the Researcher insisted on simple arithmetic (e.g., `(A * B) + C`).
*   **Rationale:** To maximize statistical power with unambiguous binary outcomes (Correct/Incorrect) and minimize linguistic ambiguity.
*   **Result:** Allowed for high-n datasets (n=300+) and robust statistical significance testing ($p < 0.001$) that would have been impossible with messier tasks.

---

## II. Methodological Rigor & Turning Points

### 4. The "First 32 Tokens" Control (Experiment 9)
*   **Researcher Intervention:** The Researcher identified a critical confound: CoT responses are longer than Direct responses. Comparing them naively is invalid.
*   **Solution:** Imposed a strict "First 32 Tokens" analysis window.
*   **Impact:** This isolated the *early conceptualization phase* of the computation, ensuring that differences in geometry were due to *strategy*, not just length. This was the "make or break" decision for the validity of all subsequent comparisons.

### 5. "Regime Mining" vs. New Data (Experiment 13)
*   **Researcher Decision:** When compute costs rose, the Researcher pushed to *mine existing data* deeper rather than generating new datasets.
*   **Insight:** "The signal is arguably already there, we just aren't looking at it right."
*   **Discovery:** This led to **Failure Subtyping** (Experiment 13). We discovered that "Failure" isn't one thing—it's either "Collapse" (giving up) or "Wandering" (getting lost). This nuance only emerged because the Researcher insisted on digging deeper into the "failed" clusters.

### 6. The "Hallucination" Save (Experiment 16/16B)
*   **Critical Save:** In the Qwen-1.5B replication, results initially looked chaotic. The AI missed it, but the Researcher spotted a pattern: the model was generating endless "1000000..." sequences after the answer.
*   **Diagnosis:** The Researcher correctly identified this as *contamination*—the hidden states included post-answer hallucinations.
*   **Fix:** The Researcher directed the implemention of `truncate_at_stop`, cleaning the dataset.
*   **Result:** What looked like a failure to replicate turned into the strongest confirmation yet. The "universal signatures" only reappeared after this user-driven intervention.

---

## III. Key Theoretical Contributions

### 7. Regime-Dependence ("Good Geometry looks different")
*   **Concept:** The Researcher challenged the idea of a single "good" trajectory shape.
*   **Formulation:** "Success in CoT looks like exploration (high entropy); Success in Direct looks like efficiency (low entropy)."
*   **Validation:** Experiment 14 confirmed this **Opposite Geometry** effect. CoT success correlates with *expansion*, while Direct success correlates with *compression*.

### 8. The "Time-to-Commit" Inductive Bias
*   **Concept:** The Researcher hypothesized that the model "decides" to answer long before it outputs the token.
*   **Validation:** The `time_to_commit` metric (where the trajectory stops exploring and starts executing) became our strongest predictor of strategy. Direct answers commit at token 0-4; CoT answers commit at token 12-20.

### 9. "Dimensional Collapse" in Failure
*   **Concept:** The Researcher intuitively proposed that when a model fails, it "runs out of ideas" or "loops."
*   **Validation:** Experiment 11 metrics ($D_{eff}$, Chaos Game) proved that failed trajectories literally collapse into lower-dimensional subspaces compared to successful reasoning.

---

## IV. Summary of Role Division

| Domain | AI Role (Assistant) | Researcher Role (User) |
| :--- | :--- | :--- |
| **Execution** | Code generation, API calls, Debugging syntax | Reviewing outputs, flagging logic errors, defining constraints |
| **Data** | Parsing strings, computing metrics, running stats | **Defining the valid dataset** (e.g., "Exclude hallucinations", "Use first 32 tokens") |
| **Analysis** | Calculating p-values, generating plots | **Interpreting the signal** (e.g., "That outlier isn't noise, it's a subtype") |
| **Strategy** | Proposing "Next Steps" based on standard workflow | **steering the ship** (e.g., "Stop generation, mine the data", "Focus on geometry, not text") |

**Conclusion:** The *Trajectory Geometry* project is an AI-assisted realization of the User's original hypothesis: that the *shape of thought* is a measurable, physical property of high-dimensional neural systems.
