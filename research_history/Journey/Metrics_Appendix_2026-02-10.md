# Appendix: Geometric and Dynamical Metrics

**Date:** 2026-02-10
**Scope:** Definitions for all metrics used in Experiments 14 and 15.

## 1. Overview

We analyze the geometry of the residual stream trajectories. 
Let $H \in \mathbb{R}^{L \times T \times D}$ be the tensor of hidden states for a single input prompt, where:
*   $L$ is the number of layers (typically 24-28).
*   $T$ is the number of tokens in the sequence.
*   $D$ is the hidden dimension of the model.

We denote the sequence of hidden states at a specific layer $l$ as $h = [h_0, h_1, \dots, h_{T-1}]$, where each $h_t \in \mathbb{R}^D$.
The difference vector (step) is defined as $\delta_t = h_{t+1} - h_t$.

---

## 2. Kinematic Metrics (Motion)

These metrics treat the thought process as a particle moving through high-dimensional space.

### 2.1 Speed (Mean Velocity)
**Concept:** Measures the average magnitude of change in the residual stream per token. High speed implies active computation or state updates; low speed implies stability or passivity.
$$ \text{Speed} = \frac{1}{T-1} \sum_{t=0}^{T-2} \|h_{t+1} - h_t\|_2 $$

### 2.2 Turn Angle
**Concept:** Measures the curvature of the trajectory. A value of 0 radians implies a straight line (ballistic); $\pi$ implies oscillation (back-and-forth).
$$ \theta_t = \arccos \left( \frac{\delta_t \cdot \delta_{t+1}}{\|\delta_t\| \|\delta_{t+1}\|} \right) $$
$$ \text{Turn Angle} = \frac{1}{T-2} \sum_{t=0}^{T-3} \theta_t $$

### 2.3 Tortuosity
**Concept:** Ratio of the net displacement to the total path length. Measures path efficiency.
*   $\tau \approx 1$: Straight, efficient path (Ballistic/Retrieval).
*   $\tau \ll 1$: Winding, inefficient path (Exploratory/Reasoning).
$$ \tau = \frac{\|h_{T-1} - h_0\|}{\sum_{t=0}^{T-2} \|h_{t+1} - h_t\|} $$

### 2.4 Directional Consistency
**Concept:** The magnitude of the mean direction vector. Indicates if there is a dominant global drift direction.
$$ \text{DirConsistency} = \left\| \frac{1}{T-1} \sum_{t=0}^{T-2} \frac{\delta_t}{\|\delta_t\|} \right\|_2 $$

### 2.5 Stabilization Rate
**Concept:** The slope of the step magnitude over time. Negative slope = "settling down" (convergence); Positive slope = "acceleration" (divergence).
$$ \beta_{\text{stab}} \text{ from } \|\delta_t\| \sim \alpha + \beta_{\text{stab}} \cdot t $$

---

## 3. Volumetric Metrics (Space Usage)

These metrics quantify the shape and size of the manifold occupied by the trajectory.

### 3.1 Radius of Gyration ($R_g$)
**Concept:** The root-mean-square distance of points from the trajectory's centroid. Measures the "volume" of thought.
$$ h_{\mu} = \frac{1}{T} \sum_{t=0}^{T-1} h_t $$
$$ R_g = \sqrt{ \frac{1}{T} \sum_{t=0}^{T-1} \|h_t - h_{\mu}\|^2 } $$
*   **Significance:** Strongest predictor of reasoning effort. Hard problems induce high $R_g$.

### 3.2 Effective Dimension ($D_{eff}$)
**Concept:** The participation ratio of the PCA eigenvalues. Estimates the number of orthogonal directions substantially used by the trajectory.
Let $\lambda_i$ be the eigenvalues of the covariance matrix of centered $\delta_t$.
$$ D_{eff} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2} $$
*   **Low $D_{eff}$:** Trajectory lies in a flat subspace (Stereotypic/Repetitive).
*   **High $D_{eff}$:** Trajectory utilizes many dimensions (Rich/Complex).

### 3.3 Gyration Anisotropy
**Concept:** Measures how "stretched" the trajectory cloud is. 
*   0 = Spherical (Isotropic)
*   1 = Line-like (Anisotropic)
Calculated as 1 - (Normalized Spectral Entropy of the Singular Values of the centered trajectory).

### 3.4 Drift-to-Spread Ratio
**Concept:** Comparison of net displacement to volumetric spread. Distinguishes "traveling" from "diffusing."
$$ \text{Drift:Spread} = \frac{\|h_{T-1} - h_0\|}{R_g} $$

---

## 4. Convergence Metrics (Commitment)

These metrics characterize how the trajectory approaches its final state.

### 4.1 Cosine Slope to Final
**Concept:** How quickly the trajectory aligns with the final token's representation. A sharp positive slope indicates a clear "convergence phase."
$$ \text{Slope of } \cos(h_t, h_{T-1}) \text{ vs } t $$

### 4.2 Distance Slope to Final
**Concept:** The rate of Euclidean approach to the final state.
$$ \text{Slope of } \|h_t - h_{T-1}\| \text{ vs } t $$

### 4.3 Early-Late Energy Ratio
**Concept:** Ratio of average step size in the first half vs second half.
*   $>1$: Decelerating (Exploration $\to$ Refinement).
*   $<1$: Accelerating (Hesitation $\to$ Rush).

### 4.4 Cosine to Running Mean
**Concept:** A measure of "centering." How well the current state aligns with the average of all previous states.
*   Increases if the trajectory spirals in or holds a steady course.
*   Decreases if it breaks away into a new orthogonal subspace.

### 4.5 Time to Commit
**Concept:** The token index where the Radius of Gyration (calculated on sliding windows) shows its maximum drop. Represents the "Phase Transition" from Exploration to Execution.
*   **Computation:** Sliding window $w=6$, stride=2. Find index of $\max(R_g[i] - R_g[i+1])$.

---

## 5. Diffusion & Spectral Metrics

### 5.1 MSD Exponent ($\alpha$)
**Concept:** Characterizes the diffusion process via Mean Squared Displacement.
$$ \langle \|h_{t+\tau} - h_t\|^2 \rangle \propto \tau^\alpha $$
*   $\alpha < 1$: Sub-diffusive (Trapped/Constrained).
*   $\alpha = 1$: Brownian Diffusion (Random Walk).
*   $\alpha > 1$: Super-diffusive (Directed Motion).

### 5.2 PSD Slope
**Concept:** The slope of the Power Spectral Density of step magnitudes (log-log plot).
*   Characterizes the "color" of the noise/signal.
*   Steeper slope (negative): More low-frequency components (smooth trends).
*   Flatter slope: White noise (random fluctuations).

### 5.3 Spectral Entropy
**Concept:** Entropy of the Power Spectral Density (PSD) of the step magnitudes. Measures the unpredictability of the "energy profile."
Higher entropy = less periodic/structured speed variations.

---

## 6. Recurrence Quantification Analysis (RQA)

Based on the Recurrence Plot $R_{i,j} = \Theta(\epsilon - \|h_i - h_j\|)$.

### 6.1 Recurrence Rate (RR)
Density of recurrence points. How often the trajectory revisits the same region of state space.

### 6.2 Determinism (DET)
Percentage of recurrence points that form diagonal lines. High DET implies the system follows deterministic rules (repeating sequences) rather than chaotic wandering.

### 6.3 Laminarity (LAM)
Percentage of recurrence points that form vertical lines. Indicates "Trapping" (staying in the same state for multiple tokens).

### 6.4 Trapping Time (TT)
Average length of vertical lines. The mean duration the model gets "stuck" in a state.

---

## 7. Cross-Layer Metrics

Computed by comparing trajectories across layer depth $l$.

### 7.1 Interlayer Alignment
Mean cosine similarity between the update vectors $\delta_t^{(l)}$ and $\delta_t^{(l+1)}$ of adjacent layers. High alignment implies the layers are moving in lockstep; low alignment implies independent processing.
$$ \text{Align} = \text{mean}_t \left( \cos(\delta_t^{(l)}, \delta_t^{(l+1)}) \right) $$

### 7.2 Depth Acceleration (Speed)
The rate of change of Speed as a function of layer depth.
*   Positive: Higher layers move faster (computation intensifies).
*   Negative: Higher layers slow down (stabilization).

### 7.3 Depth Acceleration (Tortuosity)
The rate of change of Tortuosity as a function of layer depth.
*   Positive: Higher layers become more erratic/exploratory.
*   Negative: Higher layers straighten out (ballistic/retrieval).

---

## Summary Table: Key Diagnostic Indicators

| Metric | High Value Meaning | Low Value Meaning | Best For Detecting |
| :--- | :--- | :--- | :--- |
| **Radius of Gyration** | **Exploration / Effort** | Efficiency / Retrieval | **Reasoning Intensity** |
| **Effective Dim** | **Complex Processing** | Repetitive / Linear | **Collapse / Loops** |
| **Tortuosity** | **Direct / Ballistic** | Winding / Search | **Retrieval vs Reasoning** |
| **Time to Commit** | **Late Decision** | Early Decision | **Phase Transitions** |
| **Laminarity** | **Stuck / Trapped** | Flowing | **Hesitation / Loops** |
| **Interlayer Align** | **Coherent Stack** | Independent Layers | **Deep Synchronization** |
| **Stabilization Rate** | **Divergence / Accel** | Convergence / Braking | **State Settling** |
| **PSD Slope** | **Smooth Trend** | White Noise | **Signal "Color"** |
| **Dir Autocorr** | **Smooth Curvature** | Zig-Zag / Random | **Path Continuity** |
| **Cos Running Mean** | **Centering / Spiral** | Orthogonal Breakout | **Subspace Stability** |
