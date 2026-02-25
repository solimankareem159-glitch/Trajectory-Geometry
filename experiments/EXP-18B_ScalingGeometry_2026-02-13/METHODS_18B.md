
# Experiment 18B: Scaling Geometry - Methods

## Overview

This experiment computes a unified suite of 57 geometric metrics across three transformer models of varying scales to investigate the emergence of "attractor dynamics" and "geometric signatures" in chain-of-thought reasoning.

## Models

1. **Pythia-70m**: Small-scale baseline (EleutherAI).
2. **Qwen2.5-0.5B**: Medium-scale efficient model (Alibaba Cloud).
3. **Qwen2.5-1.5B**: Large-scale efficient model (Alibaba Cloud).

## Metrics Suite (v18b)

### Family 1: Kinematic (13)

Quantifies the motion of the thought trajectory in the representation space.

- `speed`: Mean step size.
- `turn_angle`: Mean angle between consecutive steps.
- `tortuosity`: Path length / End-to-end distance.
- `directional_consistency`: Norm of mean direction vector.
- `stabilization_rate`: Slope of step sizes over time.
- `vel_autocorr_lag{1,2,4,8}`: Speed autocorrelation.
- `dir_autocorr_lag{1,2,4,8}`: Directional autocorrelation.

### Family 2: Volumetric (4)

Quantifies the volume occupied by the trajectory.

- `radius_of_gyration`: RMS distance from centroid.
- `effective_dimension`: Participation ratio of PCA eigenvalues.
- `gyration_anisotropy`: Entropy of singular values.
- `drift_to_spread`: Displacement / Gyration Radius.

### Family 3: Convergence (6)

Measures progression towards a final state.

- `cos_slope`: Rate of cosine similarity increase to final state.
- `dist_slope`: Rate of distance decrease to final state.
- `early_late_ratio`: Ratio of early speed to late speed.
- `time_to_commit`: Point of maximum radius reduction.
- `cos_to_late_window`: Mean alignment with late-stage representation.
- `cos_to_running_mean`: Alignment with the running average.

### Family 4: Diffusion (1)

- `msd_exponent`: Slope of Mean Squared Displacement (log-log).

### Family 5: Spectral (2)

- `spectral_entropy`: Entropy of step size power spectrum.
- `psd_slope`: 1/f slope of power spectrum.

### Family 6: Recurrence (5)

- `recurrence_rate`: Probability of revisiting a state (RQA).
- `determinism`: Percentage of recurrence points in diagonal lines.
- `laminarity`: Percentage of recurrence points in vertical lines.
- `trapping_time`: Average length of vertical lines.
- `diagonal_entropy`: Entropy of diagonal line lengths.

### Family 7: Landmark (6)

Proximity to semantic anchors.

- `final_correct_logit`: Logit of the correct answer at the final step.
- `final_wrong_logit`: Logit of the wrong answer.
- `logit_gap`: Correct - Wrong logit.
- `operand_{0,1,2}_prox`: Max cosine similarity to operand embeddings.
- `intermediate_prox`: Max cosine similarity to intermediate result (if applicable).
- `landmark_crossing_order_entropy`: Consistency of visiting operands.

### Family 8: Attractor (5)

- `mean_attractor_dist`: Mean distance to the centroid of successful trajectories (G4).
- `attractor_div_slope`: Rate of divergence from success centroid.
- `cos_to_success_dir`: Alignment with direction to success.
- `local_expansion_rate`: Rate of trajectory divergence (Lyapunov proxy).
- `pnr_token`: Point of No Return (distance > 2*mean).

### Family 9: Embedding Stability (4)

- `logit_consistency`: Stability of the correct logit.
- `landmark_pair_sim`: Similarity between operand and result embeddings.
- `embed_unembed_align`: Alignment between input/output matrices.
- `landmark_rank_stability`: (Omitted/Simplified).

### Family 10: Information (3)

- `step_surprisal`: 1 - cosine(v_t, v_{t-1}).
- `trajectory_entropy_rate`: Lempel-Ziv complexity of speed changes.
- `info_gain_proxy`: Magnitude of projection shift in unbedding space.

### Family 11: Inference (4)

- `confidence_slope`: Rate of logit gap increase.
- `convergence_monitor`: Alignment with EMA.
- `regime_signal`: Effective Dim / Speed.

### Family 12: Exp 15 Extras (3)

- `next_layer_corr`: Correlation of representations between layer L and L+1.
- `trajectory_curvature`: Mean angle between velocity and acceleration vectors.
- `commitment_sharpness`: Maximum derivative of `cos_to_late_window`.

## Experimental Conditions

- **Direct Prompting**: Standard QA format.
- **Chain-of-Thought (CoT)**: "Let's think step by step..." prompting.
- **Groups**:
  - G1: Direct Failure
  - G2: Direct Success
  - G3: CoT Failure
  - G4: CoT Success
