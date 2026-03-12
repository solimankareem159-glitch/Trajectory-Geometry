# Trajectory Geometry: Definitive Metric Suite

**Version:** 3.0 (Final)
**Date:** 2026-02-13
**Purpose:** Complete, replication-ready specification of all geometric, dynamical, semantic, and stability metrics for analysing transformer hidden state trajectories. Designed for use with any model from which layer-wise hidden states can be extracted.

---

## Notation and Prerequisites

Let $H \in \mathbb{R}^{L \times T \times D}$ be the tensor of hidden states for a single input prompt, where:

- $L$ = number of layers (e.g. 24-28 for Qwen2.5-0.5B)
- $T$ = number of tokens in the generated sequence
- $D$ = hidden dimension of the model

At layer $l$, the sequence of hidden states is $h = [h_0, h_1, \dots, h_{T-1}]$, each $h_t \in \mathbb{R}^D$.

The step vector (token-to-token displacement) is $\delta_t = h_{t+1} - h_t$.

Let $W_U \in \mathbb{R}^{V \times D}$ be the model's unembedding matrix (output projection / lm_head), where $V$ is vocabulary size. For a token with vocabulary index $k$, the unembedding vector is $w_k = W_U[k, :]$.

Let $W_E \in \mathbb{R}^{V \times D}$ be the input embedding matrix. For token $k$, the embedding vector is $e_k = W_E[k, :]$.

---

## Family 1: Kinematic Metrics (Motion)

These metrics treat the trajectory as a particle moving through $D$-dimensional space.

### 1.1 Speed

Average magnitude of representational change per token.

$$\text{speed} = \frac{1}{T-1} \sum_{t=0}^{T-2} \|\delta_t\|_2$$

**Interpretation:** High speed = active computation. Low speed = stability or passivity. Regime-dependent: CoT success shows lower speed than CoT failure; Direct success shows higher speed than Direct failure.

### 1.2 Turn Angle

Mean curvature of the trajectory path.

$$\theta_t = \arccos\left(\frac{\delta_t \cdot \delta_{t+1}}{\|\delta_t\| \|\delta_{t+1}\|}\right)$$

$$\text{turn\_angle} = \frac{1}{T-2} \sum_{t=0}^{T-3} \theta_t$$

**Interpretation:** Near 0 = ballistic/straight. Near $\pi$ = oscillating. Reasoning typically shows moderate curvature (exploration with direction changes).

### 1.3 Tortuosity

Ratio of net displacement to total path length, measuring path efficiency.

$$\text{tortuosity} = \frac{\|h_{T-1} - h_0\|}{\sum_{t=0}^{T-2} \|\delta_t\|_2}$$

**Interpretation:** Near 1 = straight, efficient (retrieval). Near 0 = winding, circuitous (search/exploration).

### 1.4 Directional Consistency

Magnitude of the mean normalised direction vector, measuring whether a dominant drift direction exists.

$$\text{dir\_consistency} = \left\| \frac{1}{T-1} \sum_{t=0}^{T-2} \frac{\delta_t}{\|\delta_t\|} \right\|_2$$

**Interpretation:** Near 1 = all steps aligned (ballistic). Near 0 = random walk. Failures tend toward high consistency (one-directional drift to wrong answer).

### 1.5 Stabilisation Rate

Slope of step magnitude over time via linear regression.

$$\|\delta_t\| \sim \alpha + \beta_{\text{stab}} \cdot t$$

**Interpretation:** $\beta < 0$ = settling down (convergence). $\beta > 0$ = accelerating (divergence). Successful trajectories typically stabilise in late layers.

### 1.6 Velocity Autocorrelation (Lags 1, 2, 4, 8)

Pearson correlation between step magnitude vectors at lag $\tau$.

$$\text{vel\_autocorr\_lag}\tau = \text{corr}\left([\|\delta_0\|, \dots, \|\delta_{T-2-\tau}\|],\; [\|\delta_\tau\|, \dots, \|\delta_{T-2}\|]\right)$$

**Interpretation:** High autocorrelation = momentum in computation intensity. Low/negative = erratic energy profile. Short lags (1-2) are most reliable.

### 1.7 Directional Autocorrelation (Lags 1, 2, 4, 8)

Cosine similarity between normalised step vectors at lag $\tau$, averaged over all valid pairs.

$$\text{dir\_autocorr\_lag}\tau = \frac{1}{T-2-\tau} \sum_{t=0}^{T-3-\tau} \cos\left(\hat{\delta}_t, \hat{\delta}_{t+\tau}\right)$$

where $\hat{\delta}_t = \delta_t / \|\delta_t\|$.

**Interpretation:** Measures persistence of movement direction over time. High = sustained heading. Low = frequent reorientation.

---

## Family 2: Volumetric Metrics (Space Usage)

Quantify the shape and extent of the manifold occupied by the trajectory.

### 2.1 Radius of Gyration ($R_g$)

Root-mean-square distance from trajectory centroid.

$$h_\mu = \frac{1}{T} \sum_{t=0}^{T-1} h_t$$

$$R_g = \sqrt{\frac{1}{T} \sum_{t=0}^{T-1} \|h_t - h_\mu\|^2}$$

**Interpretation:** Strongest single predictor of reasoning effort. Scales monotonically with problem difficulty (d = 4.8 to 18.1 across difficulty bins).

### 2.2 Effective Dimension ($D_{\text{eff}}$)

Participation ratio of PCA eigenvalues over the step vectors.

$$D_{\text{eff}} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

where $\lambda_i$ are eigenvalues of the covariance matrix of centred $\delta_t$.

**Interpretation:** Low = trajectory confined to a flat subspace (repetitive/stereotypic). High = trajectory uses many orthogonal directions (complex processing). Regime-dependent: low $D_{\text{eff}}$ aids CoT success but hinders Direct success.

### 2.3 Gyration Anisotropy

How elongated the trajectory cloud is.

$$\text{anisotropy} = 1 - \frac{H(\sigma)}{\log(n)}$$

where $\sigma_i$ are singular values of the centred trajectory matrix and $H(\sigma)$ is the Shannon entropy of the normalised singular value spectrum.

**Interpretation:** 0 = spherical (isotropic exploration). 1 = line-like (single-axis motion). Layer-sensitive; use with caution.

### 2.4 Drift-to-Spread Ratio

Comparison of net displacement to volumetric spread.

$$\text{drift\_to\_spread} = \frac{\|h_{T-1} - h_0\|}{R_g}$$

**Interpretation:** High = travelling (translating through space). Low = diffusing (exploring locally without net movement).

---

## Family 3: Convergence Metrics (Commitment)

Characterise how the trajectory approaches its final state and commits to an answer.

### 3.1 Cosine Slope to Final

Rate of alignment with the final token's representation.

$$\text{cos\_slope} = \text{slope of } \cos(h_t, h_{T-1}) \text{ vs } t$$

**Interpretation:** Positive = converging toward final answer. Negative = drifting away. Sharp positive slope marks the convergence phase.

### 3.2 Distance Slope to Final

Rate of Euclidean approach to the final state.

$$\text{dist\_slope} = \text{slope of } \|h_t - h_{T-1}\|_2 \text{ vs } t$$

**Interpretation:** Negative = approaching final state (convergence). Positive = growing distance from eventual output (divergence or late correction).

### 3.3 Early-Late Energy Ratio

Ratio of mean step size in first half versus second half.

$$\text{early\_late\_ratio} = \frac{\text{mean}(\|\delta_t\| \text{ for } t < T/2)}{\text{mean}(\|\delta_t\| \text{ for } t \geq T/2)}$$

**Interpretation:** > 1 = decelerating (explore then refine). < 1 = accelerating (hesitate then rush). Strong regime discriminator but weak within-regime success predictor.

### 3.4 Time to Commit

Token position of maximum drop in windowed radius of gyration.

Compute $R_g$ on sliding windows of width $w=6$, stride 2. Then:

$$t_c = \arg\max_i \left(R_g^{(w)}[i] - R_g^{(w)}[i+1]\right)$$

**Interpretation:** Early commit (token 5) = retrieval. Mid commit (token 11) = structured reasoning. Late/absent commit = failure. One of the most robust cross-regime predictors.

### 3.5 Cosine to Late Window

Mean cosine similarity of hidden states to the mean of the final window of states.

$$\text{cos\_to\_late\_window} = \frac{1}{T} \sum_{t=0}^{T-1} \cos\left(h_t, \bar{h}_{\text{late}}\right)$$

where $\bar{h}_{\text{late}} = \text{mean}(h_{T-w}, \dots, h_{T-1})$.

**Interpretation:** High = trajectory was always oriented toward its eventual destination. Low = significant reorientation occurred.

### 3.6 Cosine to Running Mean

Mean alignment of each step with the cumulative trajectory direction.

$$\text{cos\_to\_running\_mean} = \frac{1}{T-1} \sum_{t=1}^{T-1} \cos\left(\delta_t, \bar{h}_{0:t} - h_0\right)$$

where $\bar{h}_{0:t} = \frac{1}{t+1}\sum_{s=0}^{t} h_s$.

**Interpretation:** High = each step builds coherently on prior progress. Low = incoherent wandering. CoT success shows higher coherence despite more reorientation (d = 1.22).

---

## Family 4: Diffusion and Spectral Metrics

Characterise the stochastic and frequency-domain properties of the trajectory.

### 4.1 MSD Exponent ($\alpha$)

Fit to the mean squared displacement scaling relation.

$$\langle \|h_{t+\tau} - h_t\|^2 \rangle \propto \tau^\alpha$$

Compute via log-log linear regression of MSD against lag $\tau$.

**Interpretation:** $\alpha < 1$ = sub-diffusive (trapped/constrained). $\alpha = 1$ = Brownian (random walk). $\alpha > 1$ = super-diffusive (ballistic/directed).

### 4.2 Spectral Entropy

Shannon entropy of the normalised power spectral density (PSD) of the step magnitude series.

$$S = -\sum_f \hat{P}(f) \log \hat{P}(f)$$

where $\hat{P}(f) = P(f) / \sum P(f)$ is the normalised PSD.

**Interpretation:** High entropy = complex, aperiodic dynamics. Low entropy = regular, periodic patterns.

### 4.3 PSD Slope

Slope of the log-log power spectral density, characterising the colour of the trajectory's "noise."

$$\log P(f) \sim \beta_{\text{psd}} \cdot \log f + c$$

**Interpretation:** $\beta \approx 0$ = white noise (uncorrelated). $\beta \approx -1$ = pink noise ($1/f$, balanced structure). $\beta \approx -2$ = Brownian (integrated noise).

---

## Family 5: Recurrence Quantification Analysis (RQA)

Based on the binary recurrence matrix $R_{i,j} = \Theta(\epsilon - \|h_i - h_j\|)$, where $\epsilon$ is a threshold (typically set as a percentile of the pairwise distance distribution).

### 5.1 Recurrence Rate (RR)

Density of recurrence points. How often the trajectory revisits the same region.

$$\text{RR} = \frac{1}{T^2} \sum_{i,j} R_{i,j}$$

### 5.2 Determinism (DET)

Percentage of recurrence points forming diagonal lines (length $\geq l_{\min}$).

**Interpretation:** High = system follows repeating sequences (structured retrieval). Low = chaotic or novel exploration. Regime-dependent: CoT success has lower DET.

### 5.3 Laminarity (LAM)

Percentage of recurrence points forming vertical lines.

**Interpretation:** High = trajectory gets "trapped" in states for multiple tokens. Low = free-flowing trajectory.

### 5.4 Trapping Time (TT)

Mean length of vertical lines in the recurrence plot.

**Interpretation:** Mean duration the model stays stuck in a state. Weak discriminator in current data (d < 0.4) but may matter for longer sequences.

### 5.5 Diagonal Entropy

Shannon entropy of the diagonal line length distribution.

**Interpretation:** Measures complexity of the recurrence structure. Partially redundant with determinism.

---

## Family 6: Cross-Layer Metrics

Computed by comparing trajectories across layer depth.

### 6.1 Interlayer Alignment

Mean cosine similarity between update vectors of adjacent layers.

$$\text{interlayer\_align} = \text{mean}_t \left(\cos(\delta_t^{(l)}, \delta_t^{(l+1)})\right)$$

**Interpretation:** High = layers processing in lockstep (coordinated stack). Low = independent per-layer processing. CoT success shows dramatically higher alignment (d = 6.35).

### 6.2 Depth Acceleration

Rate of change of speed or tortuosity as a function of layer depth.

$$\text{depth\_accel} = \text{slope of speed}(l) \text{ vs } l$$

**Interpretation:** Positive = deeper layers more active. Negative = deeper layers stabilise.

---

## Family 7: Semantic Landmark Metrics (NEW)

These metrics define fixed reference points in the hidden state space using the model's own representations of task-relevant tokens, then track the trajectory's relationship to those landmarks. This family requires access to the model's unembedding matrix $W_U$ and knowledge of the correct answer, wrong answer, and operands for each problem.

### 7.1 Logit Lens Projection

Project each hidden state through the unembedding matrix to obtain the raw logit for any target token $k$ at each position and layer.

$$\ell_k(t, l) = h_t^{(l)} \cdot w_k$$

where $w_k = W_U[k, :]$ is the unembedding vector for token $k$.

**Note on multi-token numbers:** When the target number spans multiple sub-tokens (e.g. "388" tokenised as ["3", "88"]), compute the logit for the first sub-token as the primary signal. Optionally report the mean logit across all sub-tokens as a secondary measure.

**Interpretation:** This is what the model would predict if forced to output at this layer and token. Tracking $\ell_{\text{correct}}(t, l)$ across $t$ reveals when and how rapidly the model converges toward the right answer.

### 7.2 Answer Logit Trajectory

The time series of logit values for the correct answer token across the generated sequence.

$$\text{answer\_logit\_trajectory}(t) = \ell_{\text{correct}}(t, l)$$

Report as a per-layer time series. Summary statistics:

- **Peak logit position:** $t^* = \arg\max_t \ell_{\text{correct}}(t, l)$
- **Final logit value:** $\ell_{\text{correct}}(T-1, l)$
- **Logit convergence slope:** Linear slope of $\ell_{\text{correct}}(t, l)$ over last half of tokens

**Interpretation:** Successful trajectories should show monotonically increasing correct-answer logits in later layers. Failed trajectories show flat or decreasing correct-answer logits.

### 7.3 Wrong Answer Logit Trajectory

Same as 7.2 but for the token corresponding to the model's actual (incorrect) output.

$$\text{wrong\_logit\_trajectory}(t) = \ell_{\text{wrong}}(t, l)$$

**Interpretation:** For failures, this should show increasing logits as the model commits to the wrong answer. The crossover point where $\ell_{\text{wrong}}$ exceeds $\ell_{\text{correct}}$ marks the "decision error" in logit space.

### 7.4 Logit Gap (Correct vs Wrong)

Difference between correct and wrong answer logits at each position.

$$\text{logit\_gap}(t, l) = \ell_{\text{correct}}(t, l) - \ell_{\text{wrong}}(t, l)$$

**Summary scalar:** Mean logit gap over the final quarter of tokens.

**Interpretation:** Positive = model is closer to correct answer. Negative = committed to wrong answer. The sign-change point is the "point of no return" in output space.

### 7.5 Operand Proximity Series

Cosine similarity between the hidden state and each operand's unembedding vector over time.

$$\text{operand\_prox}_i(t, l) = \cos(h_t^{(l)}, w_{\text{op}_i})$$

For a problem $(a \times b) + c$, compute for $a$, $b$, and $c$ separately.

**Interpretation:** If the model is performing sequential computation, the trajectory should show transient proximity to operand representations in the order they are needed. Multiplication operands might show proximity peaks before the addition operand. Failures may show operands being "visited" in the wrong order or not at all.

### 7.6 Intermediate Result Proximity

For problems with identifiable intermediate results (e.g. $a \times b$ in $(a \times b) + c$), compute proximity to the intermediate answer's unembedding vector.

$$\text{intermediate\_prox}(t, l) = \cos(h_t^{(l)}, w_{\text{intermediate}})$$

**Interpretation:** Successful multi-step reasoning should show the trajectory approaching the intermediate result representation before moving toward the final answer. Absence of this intermediate proximity peak may indicate the model is attempting to compute the answer in a single step.

### 7.7 Landmark Crossing Order

Define a threshold $\theta$ for "visiting" a landmark (e.g. operand proximity exceeding the 90th percentile of its time series). Record the token positions at which each landmark is first visited.

$$t_{\text{visit}}^{(i)} = \min\{t : \text{operand\_prox}_i(t, l) > \theta\}$$

Report the ordering of $t_{\text{visit}}$ values across operands and intermediates.

**Interpretation:** Correct computation order should produce a visiting sequence that matches the arithmetic operation sequence. Disordered visiting may predict errors.

---

## Family 8: Attractor and Stability Metrics (NEW)

These metrics characterise the trajectory's relationship to empirically defined attractor basins and local dynamical stability. They require a reference set of successful trajectories for the same task type.

### 8.1 Distance to Success Centroid

At each token position $t$ and layer $l$, compute the mean hidden state across all successful trajectories for problems of comparable type, then measure distance to it.

$$\bar{h}_t^{(\text{success})} = \frac{1}{N_s} \sum_{i \in \text{success}} h_{t,i}^{(l)}$$

$$d_{\text{attractor}}(t, l) = \|h_t^{(l)} - \bar{h}_t^{(\text{success})}\|_2$$

**Summary scalars:**
- **Mean attractor distance:** $\bar{d} = \frac{1}{T}\sum_t d_{\text{attractor}}(t)$
- **Attractor divergence slope:** Linear slope of $d_{\text{attractor}}(t)$ over the last half of tokens
- **Max attractor distance:** $\max_t d_{\text{attractor}}(t)$

**Interpretation:** Successful trajectories cluster near the centroid (low $d_{\text{attractor}}$). Failed trajectories that deviate permanently show a monotonically increasing divergence profile. The divergence slope captures the "escape velocity" from the attractor.

### 8.2 Cosine to Success Direction

At each token, compute alignment between the current step and the direction from the current position toward the success centroid.

$$\text{cos\_to\_attractor}(t) = \cos\left(\delta_t,\; \bar{h}_t^{(\text{success})} - h_t\right)$$

**Summary scalar:** Mean over all tokens.

**Interpretation:** Positive = trajectory is stepping toward where successful trajectories go. Negative = stepping away. A trajectory that permanently deviates (as observed in the visualisation) should show sustained negative values after its bifurcation point.

### 8.3 Discriminant Axis Projection

Compute the direction that best separates successful and unsuccessful final hidden states (the LDA axis or simply the normalised centroid difference), then project each trajectory's hidden state onto it.

$$\hat{d}_{\text{disc}} = \frac{\bar{h}_{T-1}^{(\text{success})} - \bar{h}_{T-1}^{(\text{failure})}}{\|\bar{h}_{T-1}^{(\text{success})} - \bar{h}_{T-1}^{(\text{failure})}\|}$$

$$p_{\text{disc}}(t) = h_t^{(l)} \cdot \hat{d}_{\text{disc}}$$

**Summary scalars:**
- **Final discriminant value:** $p_{\text{disc}}(T-1)$
- **Discriminant convergence slope:** Slope of $p_{\text{disc}}(t)$ over last half

**Interpretation:** This 1D projection defines a "correctness axis." The time series shows how the trajectory moves between the success-like and failure-like regions of state space. Wrong trajectories drifting off screen correspond to monotonic movement along the failure direction on this axis.

### 8.4 Local Expansion Rate

Ratio of consecutive step magnitudes, capturing whether the trajectory is accelerating away from its current neighbourhood.

$$\rho(t) = \frac{\|\delta_{t+1}\|}{\|\delta_t\|}$$

**Summary scalars:**
- **Mean expansion rate:** $\bar{\rho}$
- **Max sustained expansion:** Longest consecutive run where $\rho(t) > 1$

**Interpretation:** Persistent $\rho > 1$ indicates the trajectory is accelerating away from wherever it was, characteristic of being outside an attractor basin. Persistent $\rho < 1$ indicates convergence. This is a computationally cheap proxy for Lyapunov exponents.

### 8.5 Point of No Return

The token position after which no trajectory in the reference set ever recovered from a given attractor distance.

For each layer, compute the empirical basin radius:

$$r_{\text{basin}}(t) = \max_{i \in \text{recovered}} d_{\text{attractor}}(t, l)_i$$

where "recovered" means the trajectory eventually converged to the correct answer despite being distant at token $t$.

The point of no return for a given trajectory is:

$$t_{\text{PNR}} = \min\{t : d_{\text{attractor}}(t) > r_{\text{basin}}(t)\}$$

**Interpretation:** Analogous to time-to-commit but defined by the relationship between the trajectory and the population rather than intrinsic geometry. If $t_{\text{PNR}}$ occurs early, the model has gone wrong early and cannot recover. Undefined (never crossed) for successful trajectories.

---

## Family 9: Embedding Stability Metrics (NEW)

These metrics test the foundational assumption that semantic landmarks (number representations) are fixed coordinates versus perspective-dependent representations that shift across layers. This directly tests the perspective-dependent semantic geometry hypothesis.

### 9.1 Logit Lens Consistency Across Layers

For a given target token $k$, compute the logit at each layer for the same token position, then measure how stable it is.

$$\text{logit\_consistency}(k, t) = \text{std}_l\left(\ell_k(t, l)\right) / |\text{mean}_l\left(\ell_k(t, l)\right)|$$

This is the coefficient of variation of the logit for token $k$ across all layers at position $t$.

**Interpretation:** Low CV = the model's "belief" about this token is stable across layers (fixed coordinate). High CV = the representation of this token shifts substantially depending on which layer you read from (perspective-dependent geometry).

### 9.2 Landmark Drift Across Layers

Measure how the direction toward a semantic landmark changes across layers by computing the cosine similarity between the landmark direction at adjacent layers.

$$\text{landmark\_drift}(k) = \frac{1}{L-1} \sum_{l=0}^{L-2} \left(1 - \cos\left(w_k^{(l)}, w_k^{(l+1)}\right)\right)$$

where $w_k^{(l)}$ is the direction from $h_t^{(l)}$ to the landmark at layer $l$. If using a single unembedding matrix, this reduces to measuring how $h_t^{(l)} \cdot w_k$ varies as a function of $l$.

A more direct formulation: compute the rank of the correct answer token among all vocabulary tokens at each layer (using logit lens), and measure rank stability.

$$\text{rank\_stability}(k) = \text{std}_l\left(\text{rank}_k^{(l)}\right)$$

**Interpretation:** If number representations are fixed coordinates, the correct answer's rank should be stable (or monotonically improving) across layers. If rank fluctuates wildly, it suggests each layer constructs its own perspective on the answer space, vindicating the perspective-dependent geometry hypothesis.

### 9.3 Representational Similarity of Landmark Pairs

For two numbers that are arithmetically related (e.g. operand and result), compute their cosine similarity in the model's representation at each layer.

$$\text{sim\_pair}(k_1, k_2, l) = \cos(w_{k_1}, w_{k_2})$$

Or in residual stream space:

$$\text{sim\_pair}(k_1, k_2, l) = \cos\left(h_t^{(l)} \cdot w_{k_1},\; h_t^{(l)} \cdot w_{k_2}\right)$$

**Interpretation:** If arithmetically related numbers (e.g. 38 and 342 = 38 x 9) are closer in unembedding space than unrelated numbers, the model has encoded arithmetic structure in its representations. If this similarity changes across layers, arithmetic relationships are constructed rather than stored.

### 9.4 Embedding-Unembedding Alignment

Cosine similarity between the input embedding and the output unembedding for the same token.

$$\text{embed\_unembed\_align}(k) = \cos(e_k, w_k)$$

**Interpretation:** High alignment means the model uses similar representations for recognising and producing a token. Low alignment means input and output spaces are geometrically distinct, which would imply trajectories must "translate" between coordinate systems during computation.

---

## Family 10: Information-Theoretic Metrics (NEW)

These metrics quantify the information content and surprise in the trajectory itself, treating successive hidden states as an information source.

### 10.1 Step Surprisal

How unexpected each step is relative to the recent trajectory history. Computed as the negative log probability of the observed step direction under a simple predictive model (e.g. the previous step direction).

$$\text{step\_surprisal}(t) = -\log P(\hat{\delta}_t | \hat{\delta}_{t-1}) \approx 1 - \cos(\hat{\delta}_t, \hat{\delta}_{t-1})$$

The cosine-based approximation treats the previous step as the "expected" direction and measures deviation from it.

**Summary scalars:**
- **Mean surprisal:** Average across all tokens
- **Surprisal spike count:** Number of tokens where surprisal exceeds 2 standard deviations above the mean
- **Late surprisal:** Mean surprisal in the final quarter (should be low for convergent trajectories)

**Interpretation:** High surprisal = the trajectory made an unexpected turn. A surprisal spike corresponds to a potential correction, insight moment, or error. Successful reasoning should show decreasing surprisal over time (increasing predictability as the model commits).

### 10.2 Trajectory Entropy Rate

Shannon entropy rate of the discretised trajectory, estimating how much new information each step contributes.

Discretise the step direction by projecting onto the top-$K$ principal components (from the trajectory's own PCA), then quantise into bins. Compute the conditional entropy:

$$\hat{h} = H(\delta_t | \delta_{t-1}, \delta_{t-2}, \dots, \delta_{t-m})$$

estimated via Lempel-Ziv complexity of the discretised sequence as a practical proxy.

**Interpretation:** High entropy rate = each step is genuinely novel (exploration). Low entropy rate = trajectory is repetitive or predictable (convergence/retrieval). LZ complexity provides a compression-based estimate suitable for short sequences.

### 10.3 Cumulative Information Gain

How much the trajectory's representation changes relative to its starting state, measured as the KL divergence between the softmax distributions induced by the logit lens at the starting and current positions.

$$\text{info\_gain}(t, l) = D_{\text{KL}}\left(\text{softmax}(h_t^{(l)} W_U^T) \;\|\; \text{softmax}(h_0^{(l)} W_U^T)\right)$$

**Summary scalar:** Final information gain $\text{info\_gain}(T-1, l)$.

**Interpretation:** Quantifies how much the model's output distribution has changed during generation. Large info gain = substantial computation has occurred. Small info gain = the model's "opinion" barely shifted. Can be compared across regime: CoT should show larger cumulative info gain than Direct.

---

## Family 11: Inference-Time Actionable Metrics (NEW)

These metrics are specifically designed to be computationally cheap enough for real-time monitoring during inference and to produce signals that could theoretically guide intervention (early stopping, regime switching, confidence estimation).

### 11.1 Confidence from Logit Gap Slope

Running estimate of how quickly the model is committing to its top prediction.

At each generated token $t$, compute the gap between the top-1 and top-2 logits from the logit lens at a diagnostic layer (recommended: layer $\lfloor 2L/3 \rfloor$).

$$\text{confidence\_slope}(t) = \text{slope of } (\ell_{\text{top1}} - \ell_{\text{top2}}) \text{ over window } [t-w, t]$$

**Actionable signal:** If $\text{confidence\_slope}$ is strongly positive, the model is committing. If flat or negative late in generation, flag as potential failure.

### 11.2 Regime Classifier Signal

Use the ratio of early-layer effective dimension to speed as a fast regime indicator.

$$\text{regime\_signal} = \frac{D_{\text{eff}}^{(\text{early})}}{\text{speed}^{(\text{early})}}$$

computed over the first 5-8 generated tokens at an early layer (0-7).

**Actionable signal:** High ratio = reasoning/exploration regime. Low ratio = retrieval regime. This can be computed within the first few tokens and used to dynamically select whether to continue with CoT or switch to direct answering.

### 11.3 Convergence Monitor

Running cosine similarity between the current hidden state and the exponential moving average of recent states.

$$\bar{h}_t = \alpha \cdot h_t + (1-\alpha) \cdot \bar{h}_{t-1}$$

$$\text{convergence\_monitor}(t) = \cos(h_t, \bar{h}_{t-1})$$

**Actionable signal:** Consistently high (>0.95) for several tokens indicates the model has converged and additional generation may be unnecessary. Consistently low indicates instability. Dropping suddenly after being high suggests the model is "second-guessing" a committed answer (potential error accumulation).

### 11.4 Anomaly Score

$z$-scored distance from the expected trajectory position at each token, relative to a population of reference trajectories.

$$z(t) = \frac{d_{\text{attractor}}(t) - \mu_d(t)}{\sigma_d(t)}$$

where $\mu_d(t)$ and $\sigma_d(t)$ are the mean and standard deviation of attractor distances at token $t$ across the reference population.

**Actionable signal:** $z > 3$ at any token indicates the current trajectory is a strong outlier and may require intervention. Suitable for safety monitoring, deception detection, or flagging out-of-distribution computation.

---

## Summary: Complete Metric Index

### Retained from Experiments 14-15 (33 metrics, 7 families)

| ID | Metric | Family | Compute Cost |
|:---|:-------|:-------|:-------------|
| K1 | speed | Kinematic | Low |
| K2 | turn_angle | Kinematic | Low |
| K3 | tortuosity | Kinematic | Low |
| K4 | dir_consistency | Kinematic | Low |
| K5 | stabilisation | Kinematic | Low |
| K6-K9 | vel_autocorr_lag{1,2,4,8} | Kinematic | Low |
| K10-K13 | dir_autocorr_lag{1,2,4,8} | Kinematic | Low |
| V1 | radius_of_gyration | Volumetric | Low |
| V2 | effective_dim | Volumetric | Medium |
| V3 | gyration_anisotropy | Volumetric | Medium |
| V4 | drift_to_spread | Volumetric | Low |
| C1 | cos_slope | Convergence | Low |
| C2 | dist_slope | Convergence | Low |
| C3 | early_late_ratio | Convergence | Low |
| C4 | time_to_commit | Convergence | Medium |
| C5 | cos_to_late_window | Convergence | Low |
| C6 | cos_to_running_mean | Convergence | Low |
| D1 | msd_exponent | Diffusion | Medium |
| D2 | spectral_entropy | Spectral | Medium |
| D3 | psd_slope | Spectral | Medium |
| R1 | recurrence_rate | RQA | High |
| R2 | determinism | RQA | High |
| R3 | laminarity | RQA | High |
| R4 | trapping_time | RQA | High |
| R5 | diagonal_entropy | RQA | High |
| X1 | interlayer_alignment | Cross-Layer | Medium |
| X2 | depth_acceleration | Cross-Layer | Low |

### New Metrics (Families 7-11)

| ID | Metric | Family | Compute Cost | Requires |
|:---|:-------|:-------|:-------------|:---------|
| S1 | answer_logit_trajectory | Landmark | Low | $W_U$, correct answer token |
| S2 | wrong_logit_trajectory | Landmark | Low | $W_U$, model output token |
| S3 | logit_gap | Landmark | Low | $W_U$, both answer tokens |
| S4 | operand_proximity | Landmark | Low | $W_U$, operand tokens |
| S5 | intermediate_proximity | Landmark | Low | $W_U$, intermediate result |
| S6 | landmark_crossing_order | Landmark | Low | S4, S5 |
| A1 | distance_to_success_centroid | Attractor | Medium | Reference trajectories |
| A2 | cos_to_success_direction | Attractor | Medium | Reference trajectories |
| A3 | discriminant_axis_projection | Attractor | Medium | Reference trajectories |
| A4 | local_expansion_rate | Attractor | Low | None |
| A5 | point_of_no_return | Attractor | Medium | Reference trajectories |
| E1 | logit_consistency_across_layers | Embedding Stability | Medium | $W_U$ |
| E2 | landmark_rank_stability | Embedding Stability | Medium | $W_U$ |
| E3 | landmark_pair_similarity | Embedding Stability | Low | $W_U$ |
| E4 | embedding_unembedding_alignment | Embedding Stability | Low | $W_E$, $W_U$ |
| I1 | step_surprisal | Information | Low | None |
| I2 | trajectory_entropy_rate | Information | Medium | None |
| I3 | cumulative_info_gain | Information | High | $W_U$ |
| N1 | confidence_slope | Inference | Low | $W_U$ |
| N2 | regime_classifier_signal | Inference | Low | None |
| N3 | convergence_monitor | Inference | Low | None |
| N4 | anomaly_score | Inference | Medium | Reference trajectories |

### Total Count

- Retained: 33 metrics across 7 families
- New: 21 metrics across 5 families
- **Grand total: 54 metrics across 12 families**

---

## Practical Notes for Implementation

**Tokenisation of multi-digit numbers.** Most tokenisers split numbers into sub-tokens. For Qwen2.5, check the tokeniser output for each number in the problem. Use the first sub-token for logit lens projections as the primary signal, and report results for all sub-tokens as supplementary data.

**Reference trajectory construction (Families 8, 11).** For attractor and anomaly metrics, construct reference sets per regime (CoT Success, Direct Success) and per difficulty bin. Using a single combined reference set would confound regime and success signals. A minimum of 30 reference trajectories per cell is recommended.

**Computational feasibility.** All Family 7 (Landmark) metrics and Family 11 (Inference) metrics can be computed in a single forward pass with minimal overhead beyond the hidden state extraction itself. Family 8 (Attractor) metrics require a pre-computed reference set but are then cheap per trajectory. Family 9 (Embedding Stability) metrics are computed once per model and do not vary per problem. Family 10 (Information) metrics range from cheap (surprisal) to expensive (info gain via full vocabulary softmax).

**Layer selection.** Not all metrics need to be computed at all layers. Recommended layer ranges based on Experiment 14 findings:

- Early layers (0-7): Regime discrimination (K1, K4, V1, V2, N2)
- Middle layers (10-14): Success prediction (C1, C2, C6, S1-S3, A3)
- Late layers (20-24): Commitment tracking (C4, S3, A1, N1, N3)
- All layers: Stability metrics (E1, E2), cross-layer metrics (X1, X2)

**Visualisation alignment.** Every time-series metric (those producing a value per token, e.g. S1, S2, S4, A1, I1) can be aligned token-by-token with the model's text output, enabling direct overlay of geometric signals onto readable text in visualisation tools.

---

## Predicted Signatures for Future Task Types

Based on the metric suite above, here are expected geometric signatures for tasks beyond arithmetic:

**Factual retrieval (e.g. "What is the capital of France?"):** Low $R_g$, high tortuosity, early commit, high S1 logit from token 1, ballistic trajectory toward known answer landmark.

**Ambiguous/contested questions:** High $D_{\text{eff}}$ throughout, late or absent commit, S3 logit gap oscillating near zero, high I1 surprisal sustained late, convergence monitor unstable.

**Multi-hop reasoning:** Multiple peaks in S5 (intermediate proximity), ordered landmark crossing, higher interlayer alignment, step surprisal spikes at each "hop" boundary.

**Creative generation:** High $D_{\text{eff}}$, high I2 entropy rate, low tortuosity (winding path), S1 answer logit undefined (no single correct answer), A1 attractor distance high throughout (no convergence toward stereotyped output).

**Deception/adversarial responses:** Predicted mismatch between early and late layer metric profiles. Surface convergence metrics (C1, C2) may look normal while A3 discriminant projection shows drift toward failure region. E1 logit consistency may be abnormally low (different layers "disagreeing" about the answer).
