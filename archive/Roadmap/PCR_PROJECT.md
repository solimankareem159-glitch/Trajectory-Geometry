# Probability Cloud Regression
## Recovering the Signal That Noise Stole

**Author:** Kareem Soliman  
**Project Type:** Statistical Methods Research & Tool Development  
**Platform:** Google Antigravity  
**Status:** Experimental — Active Development  
**Last Updated:** February 2026

---

## 1. The Problem in One Sentence

Every linear regression model you have ever used **systematically underestimates** the true relationship between variables, because it treats noisy measurements as if they were perfect truth.

This project builds and tests a method that fixes this by treating every data point not as a fact, but as the most likely outcome from a range of possibilities — and then fitting the line through that range.

---

## 2. Where This Idea Came From

### 2.1 The Core Intuition

In quantum physics, a particle doesn't have a single definite position until you measure it. Before measurement, it exists as a **wave function** — a spread of probabilities describing everywhere it *could* be. The act of measurement "collapses" this wave into a single observed value.

This project borrows that metaphor (and **only** the metaphor — this is not quantum physics) to ask a statistical question:

> **If we could figure out what *could have been* from what *was*, would we get better at figuring out what *will be*?**

The answer, it turns out, is yes. And the mathematics for doing it already exist — they're just dramatically underused.

### 2.2 The Statistical Reality Behind the Metaphor

When a researcher records that a student studied for "4 hours" and scored "78%", those numbers are not exact. The student might have studied for somewhere between 3.2 and 4.8 hours (they rounded, they weren't counting precisely, they were also on their phone for some of it). Their "true" score ability on that test might range from 72% to 84% depending on which questions happened to appear.

Standard regression (Ordinary Least Squares / OLS) ignores all of this. It takes 4.0 and 78 as hard coordinates and draws the best line through a cloud of such coordinates. When the input variable (X) contains noise — which it almost always does — OLS **mathematically guarantees** a biased result. It flattens the slope toward zero. This is called **attenuation bias**, and it means:

- Your model **underestimates** the true strength of relationships
- Your predictions are **less accurate** than they could be
- Your R² is **artificially deflated** — the model looks worse than the underlying signal warrants

This isn't a minor technicality. In datasets with moderate measurement error, OLS can underestimate the true slope by **20–40%**.

### 2.3 The Proposal

**Probability Cloud Regression (PCR)** — originally conceived under the name "Retroactive Wave Function Inference" (RWFI) — replaces every data point on a scatter plot with a **probability cloud**: a region of values where the true measurement could plausibly exist. The regression line is then fitted not through the observed points, but through the **densest corridors** of these overlapping clouds.

The "retroactive" component is this: once a candidate line is proposed, the method performs a backward pass, asking for each data point: *"Given this line, where within this point's probability cloud is the most likely true position?"* It then iteratively adjusts both the line and the inferred true positions until they converge.

The result is a regression that:
- Recovers the **true slope** rather than the attenuated one
- Produces **better out-of-sample predictions**
- Naturally handles **heterogeneous uncertainty** (some measurements more reliable than others)
- Can enforce **physical bounds** (e.g., reaction times can't be negative, percentages can't exceed 100)

---

## 3. What This Is (And What It Is Not)

### What It Is

- A **quantum-inspired metaphor** applied to classical statistics
- An accessible framework for **errors-in-variables regression** — a real and well-founded statistical approach
- A practical tool that makes uncertainty-aware regression **easy to use and visualise**
- A testable claim: this approach produces **measurably better predictions** than standard OLS

### What It Is Not

- Not quantum physics, quantum computing, or quantum anything beyond the metaphor
- Not a claim of entirely new mathematics — it builds on Total Least Squares (Golub & Van Loan, 1980), Deming Regression (Deming, 1943), and Bayesian errors-in-variables models (Fuller, 1987; Richardson & Gilks, 1993)
- Not a replacement for all regression — it is specifically valuable when input variables contain non-trivial measurement error

### Relationship to Existing Methods

| Established Method | What PCR Shares With It | What PCR Adds |
|---|---|---|
| **Ordinary Least Squares (OLS)** | The baseline this improves upon | Models uncertainty in X, not just Y |
| **Deming Regression** | Accounts for error in both X and Y | Per-point heterogeneous uncertainty; physical truncation bounds; visualisation framework |
| **Total Least Squares** | Minimises orthogonal distance to line | Weighted by point-specific confidence; bounded distributions |
| **Bayesian Errors-in-Variables** | Treats true values as latent variables with priors | The "probability cloud" framing; the gradual collapse parameter; emphasis on visual intuition |
| **Orthogonal Distance Regression (ODR)** | Minimises perpendicular distance weighted by error | Truncation at physical bounds; the retroactive inference visualisation |

**The contribution of this project is not claiming to invent new algebra.** It is:
1. Building an **integrated, testable implementation** that combines the best elements of the above
2. Providing **visual tools** that make the method intuitive to non-statisticians
3. Systematically **benchmarking** prediction accuracy against OLS across noise conditions
4. Introducing **truncated probability clouds** (bounded to physically possible values) as a practical enhancement

---

## 4. Technical Specification

### 4.1 The Data Point as a Probability Cloud

In standard regression, a data point is a coordinate: $(x_i, y_i)$.

In PCR, a data point is a **joint probability density**:

$$p_i(x, y) = \mathcal{N}_{T}(x \mid \mu_{x_i}, \sigma_{x_i}, a_{x_i}, b_{x_i}) \cdot \mathcal{N}_{T}(y \mid \mu_{y_i}, \sigma_{y_i}, a_{y_i}, b_{y_i})$$

Where:
- $\mu_{x_i}, \mu_{y_i}$ are the observed measurements (the cloud centres)
- $\sigma_{x_i}, \sigma_{y_i}$ are the estimated measurement uncertainties (the cloud spread)
- $a, b$ are optional truncation bounds (physical limits — e.g., $x \geq 0$ for reaction times)
- $\mathcal{N}_T$ denotes the **truncated normal distribution**

Visually: replace every dot on a scatter plot with a fuzzy ellipse. Dark at the centre (high probability), fading at the edges (low probability), cut off sharply at any physical boundary.

### 4.2 The Fitting Objective

We seek parameters $\beta = (m, c)$ for the line $y = mx + c$ that **maximise the joint likelihood** of the line intersecting the high-density regions of all probability clouds.

For each data point $i$, we find the point $(\hat{x}_i, \hat{y}_i)$ on the line that minimises the **weighted Mahalanobis distance** to the observed cloud centre:

$$d_i^2 = \frac{(x_i - \hat{x}_i)^2}{\sigma_{x_i}^2} + \frac{(y_i - \hat{y}_i)^2}{\sigma_{y_i}^2}$$

Subject to: $\hat{y}_i = m\hat{x}_i + c$ (the inferred true point must lie on the line)

And subject to: $a_{x_i} \leq \hat{x}_i \leq b_{x_i}$, $a_{y_i} \leq \hat{y}_i \leq b_{y_i}$ (truncation bounds)

The total objective is:

$$\min_{m, c} \sum_{i=1}^{n} d_i^2$$

This has a closed-form solution when all $\sigma$ values are equal (it reduces to standard TLS via SVD). When they differ per point, it requires iterative optimisation (EM-style alternation between estimating $\beta$ and estimating $\hat{x}_i, \hat{y}_i$).

### 4.3 The Retroactive Pass

After convergence, the model produces two outputs:

1. **The fitted line** $(m, c)$ — the recovered relationship
2. **The retroactively inferred true positions** $(\hat{x}_i, \hat{y}_i)$ — where the model believes each data point "really" was before noise corrupted it

These inferred positions are the "un-collapsed wave functions" — the model's best guess at the pre-noise signal for every observation.

### 4.4 Estimating the Uncertainty Bounds ($\sigma$)

This is the most critical practical decision. Three strategies, in order of preference:

**Strategy A — Known measurement precision (ideal)**  
If you know your instrument's precision (e.g., a scale accurate to ±0.5kg), use that directly as $\sigma$.

**Strategy B — Error variance ratio (practical)**  
Estimate $\lambda = \sigma_\varepsilon^2 / \sigma_\delta^2$ (the ratio of Y-error to X-error variance) from domain knowledge. This is what Deming regression uses and it only requires a ratio, not absolute values.

**Strategy C — Bayesian estimation (principled when you don't know)**  
Place priors on $\sigma_x$ and $\sigma_y$ and let the model learn them from data. This is more computationally expensive but avoids assuming you know the error structure.

**Strategy D — Heuristic: fraction of observed SD (fallback)**  
Use a fraction of the observed standard deviation of X as $\sigma_x$. This was the original RWFI proposal (±1 SD). We will **test multiple fractions** (0.25, 0.5, 0.75, 1.0 of observed SD) to determine sensitivity.

The experimental plan below tests all four strategies.

---

## 5. The Core Hypothesis

> **H1:** Probability Cloud Regression produces significantly lower out-of-sample mean squared error (MSE) than OLS when X contains measurement error, with the magnitude of improvement proportional to the noise level.

> **H2:** PCR recovers slope estimates closer to the true generative slope than OLS, with the gap widening as X-noise increases.

> **H3:** Truncating probability clouds at physical bounds improves predictions on bounded variables beyond what untruncated methods achieve.

---

## 6. Experimental Plan

### Phase 1: Synthetic Data Benchmark (Controlled Ground Truth)

**Purpose:** Establish that the method works under ideal conditions where we know the true answer.

**Setup:**
```
True relationship:     y = 2.5x + 5.0
Sample size:           n = 200
Y noise (σ_y):         Fixed at 1.0
X noise (σ_x):         Sweep: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
X range:               0 to 10
Repetitions:           100 simulations per noise level
```

**Methods compared:**
1. **OLS** — `statsmodels.OLS` (the standard baseline)
2. **Deming Regression** — `scipy.odr` with known error ratio
3. **PCR (known σ)** — Our method with the true σ_x, σ_y provided
4. **PCR (estimated σ)** — Our method with σ estimated as fractions of observed SD

**Metrics:**
- **Slope bias:** $|\hat{m} - 2.5|$ averaged over 100 runs
- **Intercept bias:** $|\hat{c} - 5.0|$ averaged over 100 runs
- **Out-of-sample MSE:** 5-fold cross-validation on held-out test points generated from the true process (not the noisy observations)
- **R² on true signal:** Correlation between predictions and the true $y = 2.5x + 5.0$ values

**Expected outcome:** OLS slope estimates degrade as σ_x increases. PCR and Deming stay close to 2.5. PCR with truncation adds marginal benefit when bounds are tight.

### Phase 2: Bounded Variable Test

**Purpose:** Test whether truncated probability clouds outperform unbounded methods on naturally bounded data.

**Setup:**
```
Scenario:              Predicting exam scores (0–100%) from study hours (0–20h)
True relationship:     score = 4.0 * hours + 20
Both variables bounded: hours ∈ [0, 20], score ∈ [0, 100]
Noise:                 σ_x = 2.0 hours, σ_y = 8.0 points
n = 300
```

**Methods compared:**
1. OLS
2. Deming (unbounded)
3. PCR (unbounded clouds)
4. PCR (truncated clouds at physical limits)

**Key metric:** Out-of-sample MSE near the boundaries (hours < 2 or > 18; scores < 15 or > 90), where truncation should matter most.

### Phase 3: Real Data Validation

**Purpose:** Demonstrate practical value on a dataset where measurement error is a documented problem.

**Candidate datasets (choose one based on accessibility):**

| Dataset | X variable (noisy) | Y variable | Why it's good |
|---|---|---|---|
| **NHANES Blood Pressure** | Self-reported physical activity | Systolic BP | Both variables have well-documented measurement error; freely available |
| **UCI Airfoil Self-Noise** | Angle of attack, chord length | Sound pressure level | Instrument precision is documented; clean regression task |
| **Framingham Heart Study (public)** | BMI, cholesterol | Cardiovascular risk score | Classic epidemiological dataset; measurement error in exposures is extensively studied |

**Protocol:**
1. Estimate σ_x from repeated-measures data or published reliability coefficients
2. Fit OLS, Deming, and PCR
3. Compare 5-fold cross-validated MSE
4. Report slope differences and practical significance

---

## 7. Visualisation Plan

The visualisation component is a critical deliverable. It must make the method **instantly intuitive** to someone who has never heard of errors-in-variables regression.

### Visualisation 1: "The Scatter Plot You've Been Lied To By"

**Side-by-side panels:**

| Left panel: What you see | Right panel: What's really there |
|---|---|
| Standard scatter plot with OLS line | Same data, but each point is a translucent ellipse (probability cloud). The OLS line clearly misses the dense corridors. The PCR line threads through them. |

**Interactive element:** Slider that controls σ_x from 0 (standard scatter plot) to 3.0 (very fuzzy clouds). As the slider moves right, watch the OLS line flatten while the PCR line holds steady.

### Visualisation 2: "The Retroactive Pass"

**Animation sequence:**
1. Show observed scatter plot with probability clouds
2. Draw the PCR fitted line
3. Animate arrows from each observed point to its retroactively inferred position on the line
4. Colour the arrows by confidence: short arrows (high-confidence points) in green, long arrows (low-confidence) in amber

**Purpose:** Makes the "retroactive inference" concept visceral — you can see the model "pulling" noisy points back toward where they likely came from.

### Visualisation 3: "The Slope Recovery Dashboard"

**Bar chart or line plot:**
- X-axis: noise level (σ_x)
- Y-axis: estimated slope
- Horizontal dashed line at true slope = 2.5
- Three lines: OLS (drooping), Deming (stable), PCR (stable, possibly better near bounds)

**Table alongside:** showing MSE, slope bias, and R² at each noise level.

### Visualisation 4: "The Prediction Showdown" (for real data)

**Scatter plot of predicted vs actual (out-of-sample):**
- Three panels: OLS predictions, Deming predictions, PCR predictions
- Perfect prediction = 45° line
- Points coloured by measurement uncertainty of the input

---

## 8. Implementation Architecture for Antigravity

### Project Structure

```
rwfi-experiment/
├── README.md                    ← This document
├── src/
│   ├── cloud_regressor.py       ← The PCR implementation
│   ├── benchmarks.py            ← Synthetic data experiments (Phase 1 & 2)
│   ├── real_data.py             ← Real data experiments (Phase 3)
│   └── utils.py                 ← Data generation, metrics, helpers
├── viz/
│   ├── cloud_scatter.py         ← Visualisation 1: probability cloud scatter
│   ├── retroactive_pass.py      ← Visualisation 2: retroactive inference animation
│   ├── slope_recovery.py        ← Visualisation 3: slope recovery dashboard
│   └── prediction_showdown.py   ← Visualisation 4: pred vs actual comparison
├── data/
│   └── (downloaded datasets)
├── results/
│   ├── figures/
│   └── tables/
├── tests/
│   └── test_cloud_regressor.py  ← Unit tests confirming correct behaviour
└── requirements.txt
```

### Agent Task Sequence

These prompts are designed for Antigravity's agentic mode. Execute them in order.

**Task 1 — Core Implementation**
> Build `src/cloud_regressor.py` containing a `CloudRegressor` class. It should accept arrays X, Y and per-point uncertainty estimates sigma_x, sigma_y. It should also accept optional truncation bounds (x_lower, x_upper, y_lower, y_upper). The fitting method should minimise the sum of weighted Mahalanobis distances subject to the constraint that inferred true points lie on the fitted line. Include methods: `fit(X, Y, sigma_x, sigma_y, bounds=None)`, `predict(X_new)`, `get_inferred_true_positions()`, and `summary()` which prints slope, intercept, and comparison metrics. Write unit tests that verify: (a) with zero noise, CloudRegressor matches OLS exactly, (b) with known noise, CloudRegressor recovers the true slope more accurately than OLS.

**Task 2 — Synthetic Benchmark**
> Build `src/benchmarks.py` that runs the Phase 1 experiment described in this README. Generate synthetic data with true slope=2.5, intercept=5.0, n=200, sigma_y=1.0, and sweep sigma_x across [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]. For each noise level, run 100 repetitions. Compare OLS (statsmodels), Deming (scipy.odr), and CloudRegressor. Output a results table with columns: noise_level, method, mean_slope, slope_bias, mean_mse, mean_r2. Also run the Phase 2 bounded variable test. Save results to `results/tables/`.

**Task 3 — Visualisations**
> Build the four visualisations described in Section 7 of this README. Use matplotlib for static figures and optionally plotly for interactive elements. The probability cloud scatter (Vis 1) should render each point as a translucent ellipse whose size is proportional to its sigma values. Save all figures to `results/figures/`.

**Task 4 — Real Data (after reviewing Phase 1/2 results)**
> Download [chosen dataset]. Estimate measurement uncertainties using [chosen method]. Run the Phase 3 protocol. Generate Visualisation 4. Write a summary of findings.

---

## 9. Success Criteria

This project succeeds if:

1. **CloudRegressor demonstrably recovers slopes closer to truth than OLS** across all tested noise levels (synthetic data)
2. **Out-of-sample MSE is lower for CloudRegressor than OLS** by a statistically significant margin (paired t-test across cross-validation folds)
3. **Truncated clouds provide measurable benefit** on bounded variable scenarios
4. **The visualisations make the method self-explanatory** — a reader should understand why this works without needing to read the maths

This project produces a **strong portfolio piece** if, additionally:

5. The method shows improvement on at least one real-world dataset
6. The code is clean, tested, and documented enough to share on GitHub
7. The results are written up in a format suitable for a blog post (Medium/LinkedIn) or short paper

---

## 10. Intellectual Lineage and Credit

The core statistical machinery draws from:

- **Deming, W.E. (1943).** *Statistical Adjustment of Data.* — The foundational treatment of regression with errors in both variables.
- **Fuller, W.A. (1987).** *Measurement Error Models.* — The comprehensive reference for errors-in-variables regression theory.
- **Golub, G.H. & Van Loan, C.F. (1980).** "An Analysis of the Total Least Squares Problem." *SIAM J. Numerical Analysis.* — The linear algebra foundation for orthogonal distance regression.
- **Van Huffel, S. & Vandewalle, J. (1991).** *The Total Least Squares Problem.* — Extended treatment of TLS methods.

The quantum-inspired framing — treating data points as collapsed wave functions and performing retroactive inference on the pre-collapse distribution — is an original contribution of this project. It serves as an **intuitive bridge** that makes the errors-in-variables paradigm accessible to practitioners who would otherwise default to OLS.

The specific additions this project explores beyond standard Deming/TLS regression are:

- **Per-point heterogeneous uncertainty weighting** (not new in theory; underused in practice)
- **Truncated probability clouds** at physical bounds (combining truncated distributions with orthogonal distance regression)
- **The retroactive inference visualisation** (showing inferred true positions as a pedagogical and diagnostic tool)

---

## Appendix A: Quick-Start for AI Agents

If you are an AI agent (Gemini, Claude, GPT, or other) reading this as a project specification:

**This project implements a regression method that accounts for measurement error in input variables.** The method is mathematically equivalent to weighted orthogonal distance regression with heteroscedastic errors and optional truncation bounds. Your tasks are defined in Section 8. Start with Task 1. The key Python dependencies are: `numpy`, `scipy`, `statsmodels`, `matplotlib`, `pandas`, and optionally `plotly`. Do not import or reference any quantum computing libraries — the quantum language in this document is purely metaphorical.

**The single most important test:** Generate data from y = 2.5x + 5.0 with Gaussian noise in both X and Y. If OLS gives a slope of ~2.0 (attenuated) and CloudRegressor gives a slope of ~2.5 (correct), the implementation is working.
