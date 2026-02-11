# Experiment 14: Comprehensive Interpretation of Trajectory Geometry Results

## Executive Summary

Experiment 14 expanded the metric suite from ~10 to 33 metrics and computed them across all 25 layers of Qwen2.5-0.5B. The results reveal a **fundamental insight**: success looks geometrically *different* depending on computational regime (CoT vs Direct). This is not a limitation—it's a discovery about how the model computes.

---

## Tier 1: Paradigm-Shifting Insights

### 🔥 1. "Good Geometry" is Regime-Dependent (Novelty: ★★★★★, Value: ★★★★★)

**The Core Finding**: At layer 13, 10 of 14 key metrics show **opposite** direction effects when comparing CoT-success to CoT-failure versus Direct-success to Direct-failure.

| Metric | CoT Success Higher? | Direct Success Higher? | Same Direction? |
|--------|---------------------|------------------------|-----------------|
| speed | ✗ Lower | ✓ Higher | **OPPOSITE** |
| effective_dim | ✗ Lower | ✓ Higher | **OPPOSITE** |
| radius_of_gyration | ✗ Lower | ✓ Higher | **OPPOSITE** |
| cos_to_running_mean | ✓ Higher | ✗ Lower | **OPPOSITE** |
| dist_slope | ✗ Lower | ✓ Higher | **OPPOSITE** |

**Interpretation**: 
- **CoT Success** = Tighter trajectories, lower speed, smaller radius—the model "zooms in" on a solution subspace
- **Direct Success** = Larger trajectories, higher speed, bigger radius—the model "retrieves" with confident ballistic motion

This means you **cannot build a single "success detector"** that works across regimes. The geometry of success is defined by the computational strategy, not by outcome alone.

**Why This Matters**: Previous work showed CoT vs Direct have different geometries. This shows that *what constitutes good geometry* is regime-relative. A trajectory that looks "successful" under CoT criteria would look like failure under Direct criteria.

---

### 🔥 2. The Commitment Timing Signature (Novelty: ★★★★★, Value: ★★★★★)

**The `time_to_commit` metric** measures where in the trajectory the radius of gyration drops most sharply—i.e., when the model "commits" to a solution direction.

| Comparison | Cohen's d | G1 Mean | G2 Mean |
|------------|-----------|---------|---------|
| CoT Success vs CoT Failure | -0.91 | 10.7 tokens | 16.0 tokens |
| Direct Success vs Direct Failure | -1.02 | 7.0 tokens | 13.8 tokens |
| CoT Success vs Direct Success | +2.97 | 13.8 tokens | 5.4 tokens |

**Interpretation**:
- **Failures commit LATER** in both regimes—they wander longer before collapsing
- **Direct successes commit EARLIEST** (~5 tokens)—immediate retrieval
- **CoT successes commit in the MIDDLE** (~11 tokens)—explore-then-commit

This is direct evidence for the **explore→commit phase transition** that DSG predicts. The timing of commitment is a readable signal that distinguishes not just success/failure but computational strategy.

**Practical Application**: Early commitment (< 7 tokens) predicts direct success. Late commitment (> 14 tokens) predicts failure. This could enable real-time intervention.

---

## Tier 2: High-Value Discriminative Findings

### 3. `cos_to_running_mean`: The Coherence Metric (Novelty: ★★★★☆, Value: ★★★★★)

This new metric measures how aligned each hidden state is with the cumulative running mean of all states so far.

**Performance**:
- CoT Success vs CoT Failure: d = +1.22 at Layer 14 (strongest single-metric effect)
- Direct Success vs Direct Failure: d = -1.97 at Layer 18

**Interpretation**:
- **CoT successes have HIGHER cos_to_running_mean** → Each step stays coherent with the emerging solution
- **Direct successes have LOWER cos_to_running_mean** → Model moves decisively away from starting state

This metric captures "trajectory coherence"—whether the model is building on its previous states (CoT) or executing a direct leap (Direct).

---

### 4. The Layer Depth Profile Divergence (Novelty: ★★★★☆, Value: ★★★★☆)

Effects are NOT uniform across layers. Three distinct profiles emerge:

| Metric Type | Strongest at | Interpretation |
|-------------|--------------|----------------|
| Regime-discriminating (speed, effective_dim) | Early layers (0-7) | Computational strategy established early |
| Success-discriminating (dist_slope, cos_slope) | Middle layers (10-14) | Solution quality determined mid-network |
| Convergence metrics (time_to_commit) | Late layers (20-24) | Final commitment happens late |

**Practical Implication**: 
- **For regime classification**: Monitor layers 0-7
- **For success prediction**: Monitor layers 10-14
- **For commitment detection**: Monitor layers 20-24

---

### 5. Cross-Layer Metrics Validate Hierarchical Processing (Novelty: ★★★★☆, Value: ★★★★☆)

| Metric | CoT Success vs Direct Success | Interpretation |
|--------|-------------------------------|----------------|
| interlayer_alignment_mean | d = +6.35 | CoT has more coherent cross-layer motion |
| depth_accel_speed | d = +4.33 | CoT accelerates more across depth |
| depth_accel_tortuosity | d = +1.18 | CoT has more varying path efficiency |

**Interpretation**: CoT doesn't just use more tokens—it uses the *layer hierarchy* differently. CoT trajectories show more coordination between layers, while Direct trajectories are more independent per-layer.

---

## Tier 3: Useful Discriminative Patterns

### 6. `msd_exponent`: Diffusion Character (Novelty: ★★★☆☆, Value: ★★★★☆)

The Mean Squared Displacement exponent captures whether motion is sub-diffusive (constrained), diffusive (random walk), or super-diffusive (ballistic).

- **CoT Success**: Higher exponent in middle layers → More constrained exploration
- **Direct Success**: Lower exponent → More random-walk-like

This confirms that CoT success involves *structured* exploration, not just more exploration.

---

### 7. `radius_of_gyration`: Trajectory Spread (Novelty: ★★★☆☆, Value: ★★★★☆)

Best discriminator for CoT success (d = -1.21 at Layer 8).

- **CoT Success**: LOWER Rg → Tighter trajectories in middle layers
- **Direct Success**: HIGHER Rg → Wider initial spread

Combined with time_to_commit, this suggests CoT success involves early containment followed by gradual refinement, while Direct success involves wide initial spread followed by rapid convergence.

---

### 8. Recurrence Quantification Metrics (Novelty: ★★★☆☆, Value: ★★★☆☆)

The recurrence plot metrics (determinism, laminarity, trapping_time) show moderate effects:

| Metric | CoT Effect | Direct Effect |
|--------|------------|---------------|
| determinism | d = -0.84 (success lower) | d = +1.02 (success higher) |
| laminarity | Not significant | d = -1.04 (success lower) |

**Interpretation**: CoT success has less deterministic recurrence (more novel exploration), while Direct success has more deterministic patterns (reliable retrieval).

---

## Tier 4: Metrics Requiring Caution

### 9. `early_late_ratio`: Regime-Confounded (Novelty: ★★☆☆☆, Value: ★★☆☆☆)

This metric shows massive regime effects (d > 7 for CoT vs Direct) but weak within-regime success discrimination. It's useful for regime identification but not success prediction.

---

### 10. `gyration_anisotropy`: Layer-Sensitive (Novelty: ★★☆☆☆, Value: ★★☆☆☆)

Effects vary wildly by layer, suggesting this metric captures noise or layer-specific artifacts rather than robust signal.

---

### 11. Velocity Autocorrelations at Long Lags (Novelty: ★★☆☆☆, Value: ★★☆☆☆)

`vel_autocorr_lag4` and `lag8` show inconsistent patterns. Short-term dynamics (lag1, lag2) are more reliable.

---

## Key Takeaways for Paper/Presentation

### The Headline Finding
**"Success has no universal geometric signature—computational regime determines what good trajectories look like."**

This is a genuinely novel insight. Prior work (including your Exp13) showed regime differences exist. This work shows they're not just different paths to success—they represent fundamentally different definitions of what "good" means geometrically.

### The Practical Finding
**"Commitment timing is readable and predictive."**

`time_to_commit` provides a single-number summary of where in generation the model "decided." Early commitment → Direct regime likely. Late commitment → Likely failure. Middle commitment → CoT success.

### The Methodological Finding
**"Different metrics work at different layers for different purposes."**

A single-layer, single-metric approach will miss most of the signal. Effective monitoring requires:
- Layer 0-7 metrics for regime detection
- Layer 10-14 metrics for success prediction
- Layer 20-24 metrics for commitment timing

---

## Recommended Next Steps

1. **Build a multi-layer monitoring system** that combines regime detection (early), success prediction (middle), and commitment timing (late)

2. **Test intervention based on commitment timing**: Can prompting the model to "think more" after detecting early commitment improve accuracy?

3. **Validate across architectures**: Do these patterns hold in larger models? Different families?

4. **Explore the "opposite direction" metrics**: The fact that speed/effective_dim move opposite ways for success in CoT vs Direct is a rich vein for understanding how transformers implement different computational strategies.

---

## Appendix: Full Metric Rankings

### For CoT Success vs CoT Failure (within-CoT discrimination)
| Rank | Metric | Best d | Layer |
|------|--------|--------|-------|
| 1 | cos_to_running_mean | +1.22 | 14 |
| 2 | radius_of_gyration | -1.21 | 8 |
| 3 | dist_slope | -1.02 | 12 |
| 4 | cos_slope | +1.02 | 13 |
| 5 | vel_autocorr_lag4 | +0.98 | 0 |
| 6 | msd_exponent | -0.95 | 2 |
| 7 | dir_autocorr_lag4 | +0.95 | 0 |
| 8 | cos_to_late_window | +0.93 | 4 |
| 9 | time_to_commit | -0.91 | 24 |
| 10 | spectral_entropy | +0.85 | 4 |

### For Direct Success vs Direct Failure (within-Direct discrimination)
| Rank | Metric | Best d | Layer |
|------|--------|--------|-------|
| 1 | radius_of_gyration | +2.16 | 17 |
| 2 | dir_consistency | -2.11 | 17 |
| 3 | speed | +2.10 | 17 |
| 4 | effective_dim | +2.09 | 0 |
| 5 | vel_autocorr_lag2 | -2.08 | 24 |
| 6 | tortuosity | -2.05 | 24 |
| 7 | cos_to_running_mean | -1.97 | 18 |
| 8 | vel_autocorr_lag1 | -1.97 | 19 |
| 9 | cos_slope | -1.96 | 17 |
| 10 | dist_slope | +1.95 | 24 |

### For Regime Discrimination (CoT Success vs Direct Success)
| Rank | Metric | Best d | Layer |
|------|--------|--------|-------|
| 1 | radius_of_gyration | +8.00 | 2 |
| 2 | speed | +7.76 | 7 |
| 3 | vel_autocorr_lag2 | +7.76 | 18 |
| 4 | msd_exponent | +7.64 | 16 |
| 5 | dir_autocorr_lag2 | +6.91 | 18 |
| 6 | cos_to_late_window | -6.67 | 24 |
| 7 | effective_dim | +6.50 | 1 |
| 8 | interlayer_alignment_mean | +6.35 | cross |
| 9 | cos_to_running_mean | -6.27 | 3 |
| 10 | vel_autocorr_lag1 | -5.09 | 7 |
