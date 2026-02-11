# Experiment 6 Criteria: Go/No-Go & Failure Modes

**Status:** ACTIVE  
**Date:** 2026-01-27

## Success Criteria (The "Go" Signal)

To proceed to Stage 2 (Sparse Autoencoder Analysis / Geometry Characterization), the following must be true:

1. **Decoding Signal**: Peak mid-layer decoding accuracy $> 70\%$ (for 10 classes) in the **Joint Hold-out** split. (Chance is 10%).
2. **Generalization Gap**: The drop from "Random Split" accuracy to "Joint Hold-out" accuracy is $< 20$ percentage points.
3. **Lexical Dominance**: The internal state probe outperforms the Instruction Text TF-IDF baseline by $> 15\%$.
4. **Control Validity**: Permutation and Content-Only controls remain within $5\%$ of random chance.

## Failure Modes & Interpretations

| Outcome | Interpretation | Action / Redesign |
| :--- | :--- | :--- |
| **Accuracy is high everywhere (~90%)** | Suspicious. Likely leakage or dataset artifacts. | **STOP.** Audit dataset. Check for unique tokens associated with specific operators. |
| **High Random Split, Low Cross-Paraphrase** | Model relies on shallow pattern matching of instruction keywords. No "concept" formed. | **STOP.** The hypothesis is unsupported for this model size/family. Report as negative result. |
| **High Cross-Paraphrase, Low Cross-Topic** | Operator representation is entangled with content. Not abstract. | **RETRY.** Try diff-from-neutral normalization or larger models (e.g., 70B if using 7B). |
| **Controls Decode High Accuracy** | Methodology flaw. Information is leaking via sequence length, punctuation, etc. | **STOP.** Fix preprocessing. Ensure padding/masking is not confounding. |
| **Flat Layer-wise Curve** | Information is evenly distributed, no "emergence" in specific layers. | **NOTE.** Weakens the "mid-layer" hypothesis but doesn't falsify "decodability". adjusting claim. |

## Next Steps (If "Go")

1. **Geometry Analysis**: PCA/t-SNE of the "Operator Space" in the peak layers.
2. **Vector Arithmetic**: Test if $Mean(Summarize) - Mean(Translate)$ generalizes to new content.
3. **Steering**: Inject the "Simplification" vector; does the model simplify the output without being told?
