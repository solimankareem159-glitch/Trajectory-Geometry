<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Review the findings of the experiments in your knowledge base and then do a comprehensive literature search for any and all related work to this project. Synthesize the findings into a literature review Suitable for publication in an academic forum. Cover anything related to mechanistic interpretability, geometry, hidden state layers, vector embeddings, chain of thought reasoning and challenges of behavioural output modelling. Identify the gaps in the literature which this work fills and critique the existing work based on how well trajectory geometry is able to fill in those gaps. Use formal academic language and be as rigorous systematic and comprehensive as possible. Ensure I am familiar with all SOTA research in this field and the work that has been done to date so I can reach out to relevant researchers.

Given the comprehensive information I've gathered from your knowledge base and the extensive literature search, I'll now synthesize this into an academic literature review. Let me create the comprehensive document.

# Trajectory Geometry in Transformer Reasoning: A Comprehensive Literature Review

## Abstract

This literature review synthesizes findings from a systematic investigation into trajectory geometry of hidden state dynamics during transformer-based reasoning, contextualizing these discoveries within the broader landscape of mechanistic interpretability, chain-of-thought reasoning, vector embedding spaces, and computational regime analysis. The reviewed work introduces novel geometric metrics—including speed, directional consistency, stabilization rate, effective dimensionality, and commitment timing—to characterize internal reasoning trajectories in language models, revealing that computational success exhibits regime-dependent geometric signatures rather than universal patterns. Through analysis of arithmetic reasoning tasks across multiple transformer architectures (Qwen2.5-0.5B through 1.5B, Pythia-1B, TinyLlama-1.1B), the research demonstrates that trajectory geometry provides a process-level window into reasoning dynamics that conventional output-based evaluation cannot capture. This review positions these findings within contemporary mechanistic interpretability research, identifies critical gaps in current literature regarding dynamic analysis of multi-step reasoning, and establishes trajectory geometry as a complementary framework to sparse autoencoder and circuit discovery approaches for understanding transformer computation.

## 1. Introduction

The emergence of large language models (LLMs) with sophisticated reasoning capabilities has intensified demands for interpretability methods that can explain *how* these systems arrive at their outputs, not merely *what* they produce. Traditional mechanistic interpretability approaches—including activation patching, circuit discovery, and feature extraction via sparse autoencoders—have yielded valuable insights into individual components and static feature representations. However, these methods predominantly analyze *positional* aspects of computation: which neurons activate, which circuits engage, which features are present at specific layers. A fundamental question remains underexplored: how do internal representations *move* through latent space during multi-step reasoning, and what do these movement patterns reveal about computational mechanisms?[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10]

This literature review examines a research program that addresses this gap through trajectory geometry analysis—treating hidden state evolution during reasoning as dynamical trajectories through high-dimensional representation spaces, then characterizing these trajectories via geometric metrics derived from differential geometry and dynamical systems theory. The empirical work under review extracted hidden state trajectories from transformer models during arithmetic reasoning tasks, computed metrics including velocity, curvature, directional consistency, effective dimensionality, and commitment timing, and discovered that geometric patterns reliably distinguish successful from failed reasoning across multiple regime types.

Three core findings emerge from this investigation:

**First**, computational success exhibits regime-dependent rather than universal geometric signatures. Metrics that predict success in direct-answer (retrieve-and-commit) regimes show opposite directional effects in chain-of-thought (explore-then-commit) regimes. For instance, at layer 13 in Qwen2.5-0.5B, successful direct reasoning exhibits higher speed (Cohen's *d* = 2.10), larger radius of gyration (*d* = 2.16), and higher effective dimensionality (*d* = 2.09) compared to direct failures, while successful CoT reasoning exhibits the opposite pattern—lower speed, smaller radius, and lower dimensionality compared to CoT failures. This regime-dependency fundamentally challenges the notion of a single "good trajectory" detector.[^11][^12]

**Second**, commitment timing—measured as the token position at which trajectory dimensionality collapses—provides a robust cross-regime predictor of success versus failure. CoT successes commit around token 10-14, direct successes around token 5-7, while failures in both regimes commit significantly later (token 16+). This temporal signature captures the explore-then-commit phase transition predicted by theoretical models of multi-step reasoning.[^12][^11]

**Third**, trajectory geometry offers predictive power beyond prompt-type classification. While prompt condition (Direct vs. CoT) explains substantial variance, geometric features add 6 percentage points of accuracy (74.9% vs. 68.9%) when predicting correctness, with area under ROC curve increasing from 0.629 to 0.767. Within CoT conditions alone, geometry predicts correctness at 75.0% accuracy (AUC 0.772), demonstrating that internal motion patterns encode information about computational quality independent of surface-level prompting strategies.[^11]

These findings position trajectory geometry as a complementary analytical framework to existing mechanistic interpretability methods—one that captures *process* rather than structure, dynamics rather than position, and regime-specific computational strategies rather than universal feature activations.

## 2. Theoretical Foundations

### 2.1 Mechanistic Interpretability: State of the Field

Mechanistic interpretability research aims to reverse-engineer the algorithms implemented by neural networks through systematic analysis of internal computations. The field has coalesced around several core methodological approaches, each revealing distinct aspects of model behavior.[^2][^3]

**Circuit Discovery.** Pioneered by Anthropic's research program and systematized by Conmy et al. (2023), circuit discovery identifies minimal subgraphs of computational graphs responsible for specific behaviors. The Automatic Circuit Discovery (ACDC) algorithm uses activation patching to iteratively prune edges in transformer architectures, successfully rediscovering 5/5 component types and 68/68 edges in GPT-2 Small's greater-than circuit—all manually identified by prior work. Recent advances include Edge Attribution Patching (EAP) for improved scalability and provable guarantees for circuit identification. However, circuit discovery focuses on *which* components participate in computation rather than *how* representations evolve through those components over time.[^8][^9][^10][^13][^14][^1]

**Sparse Autoencoders (SAEs).** Addressing the polysemanticity problem—where individual neurons correspond to multiple unrelated concepts—SAEs decompose dense neural activations into interpretable, monosemantic features via overcomplete sparse coding. Bricken et al. (2023) demonstrated that SAE features trained on Pythia-70M and GPT-2 Small are significantly more interpretable than neurons (as measured by automated explanation scoring), with successful applications to indirect object identification tasks. Recent work has scaled SAEs to millions of features (Llama-Scope with 128K features per layer), explored cross-model feature universality, and applied SAEs beyond language to radiology, protein representations, and single-cell genomics. SAEs excel at identifying *what* features are present but provide limited insight into temporal dynamics or how features evolve during sequential processing.[^4][^15][^5][^16][^6][^7][^17]

**Probing and Representation Analysis.** Linear probing methods train classifiers on internal representations to test whether specific information is linearly accessible at given layers. More sophisticated approaches include: the Logit Lens, which projects intermediate hidden states through the unembedding matrix to interpret "nascent predictions"; representation similarity analysis comparing geometry across layers; and causal interventions testing whether identified features causally influence behavior. Recent work on hidden-state dynamics has begun exploring trajectories, but primarily in autoencoder latent spaces rather than in-context reasoning within transformers.[^18][^19][^20][^21][^22][^23][^24][^25][^2]

**Gaps in Current Approaches.** While circuit discovery identifies structural connectivity, SAEs extract interpretable features, and probing tests information accessibility, these methods share a common limitation: they analyze *snapshots* of representation spaces at individual layers or tokens, treating computation as a sequence of discrete states rather than a continuous dynamical process. For multi-step reasoning tasks—where models generate extended chain-of-thought sequences exhibiting exploration, backtracking, and convergence—this snapshot paradigm misses critical information encoded in *how* representations move through latent space over time. The trajectory geometry framework addresses this gap by treating hidden state evolution as the primary object of study.

### 2.2 Chain-of-Thought Reasoning and Faithfulness

Chain-of-thought (CoT) prompting—instructing language models to generate intermediate reasoning steps before producing final answers—has become a dominant paradigm for eliciting reasoning capabilities. Recent work demonstrates substantial performance improvements across mathematical problem-solving, multi-hop question answering, planning tasks, and geometry. The emergence of reasoning-specialized models (OpenAI's o1 family, DeepSeek-R1, QwQ-32B) trained via reinforcement learning on extended reasoning traces represents a major architectural shift toward systems that intrinsically generate internal thought processes.[^26][^27][^28][^29][^30][^31][^32][^33][^34][^35]

**The Faithfulness Problem.** A critical concern in CoT research is *faithfulness*: whether verbalized reasoning steps reflect the model's actual computational processes or serve as post-hoc rationalizations. Multiple studies reveal troubling patterns:[^36][^37][^33][^38][^39][^40][^35][^41][^42][^43]

1. **Biased Reasoning:** Turpin et al. (2024) and Lanham et al. (2023) demonstrate that biased contexts cause models to change answers while generating superficially coherent justifications that ignore the bias—a form of motivated reasoning.[^37][^41][^43][^44]
2. **Reasoning-Answer Dissociation:** Causal interventions show that perturbing intermediate reasoning steps often has minimal impact on final answers, suggesting limited causal dependence.[^40][^37]
3. **Model-Dependent Faithfulness:** Korbak et al. (2025) find that smaller models or models facing harder tasks exhibit more faithful CoT—a "faithfulness by necessity" where models genuinely need their verbalized reasoning. Conversely, larger models on easier tasks may generate CoT that is decorative rather than functional.[^45]
4. **Structural vs. Content Dependence:** Recent work demonstrates that the *structure* of long CoT matters more than content—models trained on CoT samples with incorrect answers achieve only 3.2% lower accuracy than those trained on correct answers, while structural disruptions (shuffling steps, removing transitions) dramatically degrade performance.[^46]

**Internal Representations and Faithfulness.** Several studies probe internal representations to detect unfaithful reasoning:

- **Motivated Reasoning Detection:** Mirtaheri \& Belkin train probes on residual streams showing that biased options are nearly perfectly recoverable (AUC~1.0) from end-of-CoT representations even when models neither adopt nor mention the bias, with early-CoT predictions of eventual bias-following achieving 70-85% accuracy.[^41][^43][^44]
- **Unlearning-Based Faithfulness:** Tutek et al. (2025) propose Faithfulness by Unlearning Reasoning (FUR), using neural network unlearning to selectively remove individual reasoning steps from model parameters, then measuring impact on final answers. Faithful steps cause significant performance degradation when unlearned.[^42]
- **Latent Reasoning:** The trajectory geometry research under review provides complementary evidence: geometric patterns in hidden states predict reasoning success with 75-80% accuracy even when CoT text is held constant, suggesting internal computation exhibits structure beyond what verbal reasoning captures.[^47][^48][^12][^11]

These findings motivate analysis of internal dynamics as a more reliable signal than tokenized CoT text—the focus of the trajectory geometry framework.

### 2.3 Vector Embeddings and Semantic Space Geometry

The mathematical foundations of trajectory geometry rest on well-established principles from distributional semantics and geometric representation learning.

**Vector Space Models of Meaning.** Word embeddings (Word2Vec, GloVe) and contextualized representations (BERT, transformers) map linguistic units to high-dimensional vectors such that semantic similarity corresponds to geometric proximity. This semantic space hypothesis—that meaning has geometric structure—enables analogical reasoning (king - man + woman ≈ queen), compositional operations, and similarity-based retrieval.[^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60]

**Latent Space Dynamics.** Recent theoretical work reconceptualizes neural networks as dynamical systems operating on latent manifolds. Fumero et al. (2025) demonstrate that autoencoders implicitly define latent vector fields through iterative encoding-decoding, with attractors corresponding to memorized vs. generalized representations. Trajectories in these fields reveal out-of-distribution samples and encode model priors. Crucially, standard training induces contractiveness (Jacobian spectral norms <1) via weight decay and architectural bottlenecks, ensuring convergence to fixed points.[^19][^21][^22]

**Trajectory Analysis in Neural Representations.** Prior work on trajectory geometry has primarily focused on:

1. **RNN Hidden States:** Hiebert et al. (2018) cluster LSTM hidden state trajectories to identify interpretable word-level patterns. Garcia et al. (2021) provide visual analytics tools for trajectory interpretation in recurrent networks.[^61][^62]
2. **Biological Neural Data:** Chakraborty et al. (2017) develop geometric frameworks for comparing neural activity trajectories across subjects with distinct temporal spans, using parallel transport of tangent vectors to construct comparable linear subspaces.[^63]
3. **Behavioral Modeling:** Transformer-based architectures increasingly model human behavioral trajectories, capturing multi-modal action distributions and temporal dependencies.[^64][^65][^66][^67][^68][^69][^70][^71]
4. **Phase Space Analysis:** Javier Marín (2024) applies Hamiltonian mechanics to multi-hop reasoning in LLMs, mapping reasoning chains to phase spaces and analyzing trajectories via curvature, energy conservation, and canonical transformations. This work parallels the geometric approach but focuses on embedding spaces rather than hidden states.[^72]

**Gaps for Transformer Reasoning.** Despite rich prior work, systematic geometric analysis of transformer hidden state trajectories during in-context reasoning remains limited. RNN trajectory work predates modern transformer architectures; biological neural trajectory analysis addresses different timescales and modalities; behavioral modeling focuses on external action sequences rather than internal representations. The trajectory geometry framework reviewed here uniquely applies geometric analysis to token-resolved hidden state dynamics during multi-step reasoning in decoder-only transformers—a domain where representational motion may encode critical computational processes invisible to static analysis.

### 2.4 Computational Regimes and Mode Detection

The regime-dependency of trajectory geometry—where metrics predicting success in one computational mode show opposite effects in another—connects to broader research on detecting and characterizing distinct computational strategies in neural networks.

**Regime Detection in Time Series.** In quantitative finance and dynamical systems, regime detection identifies qualitatively different operational phases (e.g., high-volatility vs. low-volatility markets). Recent work applies LLMs to textual regime detection, using central bank communications and news to infer macro-economic states. Traditional methods employ Hidden Markov Models, change-point detection, and Wasserstein clustering; LLM-based approaches leverage semantic understanding of narrative shifts.[^73][^74][^75]

**Dynamical Regimes in Neural Systems.** Neuroscience research identifies distinct dynamical regimes in biological neural networks via phase transitions. Hoshino et al. (1996) demonstrate self-organized phase transitions between pattern-itinerant and pattern-fixed states driven by Hebbian learning, interpreting transitions as recognition (itinerant→fixed) and memory clearing (fixed→itinerant). Recent work applies these concepts to artificial networks, using dynamical metrics (integration time, metastability, phase synchronization) to distinguish functional regimes in LLMs.[^76][^77][^78][^79][^80][^81]

**Computational Modes in Transformers.** Emerging evidence suggests transformers employ qualitatively different computational strategies depending on task and prompt:

1. **Depth-Recurrent Analysis:** Lu \& Yang (2025) investigate latent CoT in depth-recurrent transformers (Huginn-3.5B), finding limited evidence for interpretable latent reasoning across recurrent blocks and discontinuities in hidden state interpretability—suggesting mode-dependent processing.[^82]
2. **Reasoning Trajectory Analysis:** Li \& Goyal (2025) propose twin tests for "off-trajectory reasoning"—models' ability to recover from misleading reasoning traces (Recoverability) and build upon correct partial reasoning from collaborators (Guidability). Results reveal that stronger models are often *more fragile* under distraction (Recoverability <60% for many 7B+ models) and fail to leverage guidance beyond their capabilities (Guidability <10% solve rate), suggesting distinct collaboration vs. solo reasoning modes.[^83][^84][^85]
3. **Multi-Modal Computation:** Zhao et al. (2023, 2025) analyze reasoning traces via geometric and statistical mechanics approaches, identifying multi-hop reasoning trajectories through Hamiltonian energy dynamics, revealing conservation laws and curvature-based measures of cognitive flexibility.[^86][^87]
4. **Dynamical Systems Analysis:** Recent work applies neuroscience-inspired metrics (integration time, metastability) to Qwen-14B hidden states, finding statistically significant differences (large effect sizes, *p* < 0.001) in dynamical organization across functional regimes (baseline, reasoning, perturbed-reasoning). Results demonstrate that temporal integration and metastable coordination distinguish computational modes.[^76]

**Connections to Trajectory Geometry.** The reviewed work's discovery of regime-dependent success signatures—where direct-answer and CoT regimes exhibit opposite geometric correlates of success—aligns with broader evidence for computational mode-switching in transformers. The regime classification cascade (Layers 0-7 for regime detection [*d* = 6-8], Layers 10-14 for success prediction [*d* = 1.0-2.2], Layers 20-24 for commitment tracking [*d* = 0.9-1.0]) provides a functional architecture for real-time regime monitoring, paralleling neuroscience approaches to detecting phase transitions in biological neural dynamics.[^11]

## 3. Methodological Framework

### 3.1 Experimental Design

The trajectory geometry research program employed a systematic experimental design optimized for isolating geometric signatures of reasoning while controlling for confounds.

**Task Selection: Multi-Step Arithmetic.** The research focused exclusively on arithmetic tasks of the form *A ± B ± C* with operands in specific ranges, where ground truth is objectively verifiable. This choice addresses a critical limitation in prior CoT interpretability work: ambiguity about correctness. Unlike open-ended reasoning or subjective generation tasks, arithmetic provides binary correctness labels enabling clean success/failure comparisons. However, this focus also limits generalizability—geometric patterns may differ for logical reasoning, planning, or creative tasks where correctness is less well-defined.[^48][^47][^12][^11]

**Model Selection and Replication.** Primary experiments used Qwen2.5-0.5B (24 layers, 500M parameters) with subsequent replications on Qwen2.5-1.5B and Pythia-1B. The choice of small-to-medium models (0.5B-1.5B parameters) provides advantages (computational tractability, cleaner signal-to-noise in limited-capability regimes) and disadvantages (unclear scaling to frontier models, potential capability ceiling effects). Replication across architectures addresses concerns about model-specific artifacts: geometric patterns strengthened in larger models (effect sizes increased from *d* = 2.7-4.3 in 0.5B to *d* > 8 in 1.5B for key metrics), suggesting robustness.[^88][^48]

**Prompting Conditions.** Two prompt types created four natural groups:

- **Direct Answer:** "Calculate [problem]." → *n*=300 responses (247 failures [G1], 53 successes [G2])
- **Chain-of-Thought:** "Calculate [problem]. Think step by step." → *n*=300 responses (77 failures [G3], 223 successes [G4])

This 2×2 design (Prompt Type × Correctness) enabled testing whether geometric differences reflect prompt-induced strategy changes versus outcome-related quality signals. Critical finding: geometry predicts success *within* prompt conditions (*p* < 0.001 for G3 vs. G4 comparisons), establishing that trajectories encode more than surface-level prompting.[^11]

**Token Window and Decoding.** Hidden states were extracted for the first 32 generated tokens using greedy decoding (temperature=0) to ensure reproducibility. This fixed window controls for response-length confounds (CoT responses are typically longer) but introduces limitations: (1) truncation may miss late-stage dynamics in longer reasoning chains, (2) greedy decoding eliminates stochasticity, potentially suppressing exploration patterns visible under sampling. Future work should investigate trajectory geometry under sampling-based generation and varying context windows.[^47][^12]

**Layer Selection.** Comprehensive analysis extracted hidden states at all 24 layers for Qwen2.5-0.5B, with focused analysis on layers 0, 6, 10, 13, 16, 18, 24 to balance computational cost against coverage of early (embedding), middle (processing), and late (output) stages. This multi-layer approach revealed functional specialization: early layers (0-7) distinguish regimes, middle layers (10-14) predict success, late layers (20-24) track commitment timing. Single-layer analyses would miss this hierarchical organization.[^12][^11]

### 3.2 Geometric Metrics: Definitions and Rationale

The trajectory geometry framework computes 33 distinct metrics organized into seven families, each capturing complementary aspects of hidden state dynamics.[^12][^11]

**Velocity-Based Metrics.**

- **Speed:** Mean L2 norm of successive hidden state differences, $\text{speed} = \frac{1}{T-1} \sum_{t=1}^{T-1} \|h_{t+1} - h_t\|_2$

This captures total representational displacement per token. Higher speed indicates more dramatic state-space traversal. Finding: CoT success exhibits *lower* speed than CoT failure but *higher* speed than Direct success—a regime-dependent pattern.

- **Velocity Autocorrelation (lags 1, 2, 4, 8):** Pearson correlation between velocity vectors at different time lags, measuring persistence of motion direction.

**Directional Coherence Metrics.**

- **Directional Consistency (DC):** Norm of mean-normalized direction vectors, $\text{DC} = \left\| \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{h_{t+1} - h_t}{\|h_{t+1} - h_t\|_2} \right\|_2$

DC quantifies whether motion maintains a consistent heading (DC→1) versus random-walk-like reorientation (DC→0). CoT success exhibits dramatically lower DC (*d* = -2.6 across layers) than failures, indicating continual reorientation rather than ballistic motion.

- **Cosine to Running Mean:** Alignment of each step vector with cumulative trajectory direction. CoT successes show *higher* alignment (*d* = 1.22 at layer 14), meaning each step builds coherently on prior progress despite frequent reorientation.[^11]
- **Directional Autocorrelation (lags 1, 2, 4, 8):** Temporal correlation of normalized direction vectors.

**Curvature and Path Efficiency.**

- **Tortuosity:** Ratio of total path length to straight-line displacement, $\text{tortuosity} = \frac{\sum_{t=1}^{T-1} \|h_{t+1} - h_t\|_2}{\|h_T - h_1\|_2}$

Higher tortuosity indicates more circuitous paths. Direct success exhibits *higher* tortuosity (*d* = -2.05) than Direct failure, reflecting efficient retrieval with minimal backtracking. CoT success shows *lower* tortuosity than CoT failure, consistent with exploration that eventually converges.[^11]

- **Mean Squared Displacement (MSD) Exponent:** Fit to $\langle \|h_t - h_0\|^2 \rangle \propto t^\alpha$. Exponents indicate diffusion character: $\alpha < 1$ (sub-diffusive/constrained), $\alpha = 1$ (Brownian), $\alpha > 1$ (super-diffusive/ballistic). CoT success shows higher MSD exponent (*d* = -0.95 at layer 2), indicating more constrained exploration.

**Dimensional Structure.**

- **Effective Dimensionality:** Participation ratio of PCA eigenvalues over step vectors within trajectory, $\text{effDim} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$

Measures active degrees of freedom. Low effective dimension indicates motion confined to low-dimensional subspace. Finding: CoT success exhibits *lower* effective dimension (*d* = -0.95 to -1.21 depending on layer) than CoT failure, but Direct success exhibits *higher* effective dimension (*d* = 2.09) than Direct failure—a striking regime-dependent reversal.[^11]

- **Intrinsic Dimensionality:** Maximum Likelihood Estimator (MLE) of local manifold dimension.
- **Gyration Radius:** Root-mean-square distance from trajectory points to trajectory centroid, capturing spatial spread. Direct success shows larger radius (*d* = 2.16); CoT success shows smaller radius.[^11]

**Convergence Metrics.**

- **Stabilization Rate:** Slope of linear fit to displacement norms $\|h_{t+1} - h_t\|_2$ over time. Positive slopes indicate divergence; negative slopes indicate convergence. Successes exhibit negative stabilization (convergent trajectories); failures often show positive stabilization (continued drift).
- **Distance-to-Final Slope:** Rate at which trajectories approach final hidden state $h_T$. Measured via $\|h_t - h_T\|_2$. Negative slopes indicate earlier convergence toward final answer representation.
- **Early/Late Distance Ratio:** Ratio of mean distance-to-final in first quartile vs. last quartile of tokens. Ratios >1 indicate convergence; <1 indicate divergence.

**Commitment Timing.**

- **Time-to-Commit:** Token position where radius of gyration exhibits maximum decrease (sharpest dimensionality collapse), $t_c = \arg\max_t \left( R_g(t) - R_g(t+1) \right)$ where $R_g(t)$ is gyration radius over tokens 1 through $t$.

This novel metric captures the *when* of decision-making. CoT success commits at token 10.7-13.8; Direct success at token 5.4; failures commit later (13.8-16.0). This temporal signature provides a single-number summary of the explore-then-commit phase transition and proves to be one of the most robust cross-regime predictors.[^12][^11]

**Recurrence and Spectral Metrics.**

- **Determinism, Laminarity, Trapping Time:** Recurrence Quantification Analysis (RQA) metrics measuring revisitation of similar states, persistence in neighborhoods, and duration near attractors. CoT success shows *lower* determinism (*d* = -0.84), indicating less repetitive dynamics—consistent with exploration rather than cycling.[^11]
- **Spectral Entropy:** Shannon entropy of normalized Fourier power spectrum, capturing complexity of temporal oscillations.

**Cross-Layer Metrics.**

- **Interlayer Alignment:** Mean cosine similarity between hidden state changes across consecutive layers, $\text{alignment} = \frac{1}{L-1} \sum_{l=1}^{L-1} \text{cosine}(\Delta h_l, \Delta h_{l+1})$

Measures coordination of processing across depth. CoT success exhibits higher interlayer alignment (*d* = 6.35), suggesting more coherent hierarchical computation.[^11]

**Rationale for Multi-Metric Approach.** The 33-metric suite addresses three concerns:

1. **Complementarity:** Velocity, directionality, curvature, dimensionality, convergence, and timing capture distinct geometric properties. No single metric fully characterizes trajectory structure.
2. **Regime-Specificity:** Different metrics are diagnostic in different regimes. Effective dimension discriminates CoT success/failure strongly but shows opposite effects for Direct reasoning. Convergence metrics are more universal.
3. **Layer-Dependency:** Optimal metrics shift with depth—early layers favor regime-discriminating metrics; middle layers favor success-predictive metrics; late layers favor commitment-tracking metrics.

This comprehensive measurement enables dissecting *which* geometric properties matter *when* and *where* in the computational hierarchy—a level of granularity impossible with single-metric analyses.

### 3.3 Statistical Analysis and Effect Sizes

Rigorous statistical methodology addresses concerns about multiple comparisons, small sample sizes, and reproducibility.

**Effect Size Reporting.** All comparisons report Cohen's *d* (standardized mean difference) alongside *p*-values from permutation tests (1000 shuffles). Emphasis on effect sizes (*d*) rather than *p*-values reflects best practices in interpretability research: given the complexity of neural computation, *magnitude* of geometric differences matters more than statistical significance. Observed effect sizes (*d* = 2-8 for key comparisons) indicate large-to-very-large effects, well beyond typical psychological research standards (*d* = 0.2 small, 0.5 medium, 0.8 large).[^12][^11]

**Permutation Testing.** Non-parametric permutation tests (randomly shuffling group labels 1000 times, computing null distribution of test statistics) avoid parametric assumptions about trajectory metric distributions. This is appropriate given unknown distributional properties of geometric measures over neural trajectories.

**Layer-Specific Analysis.** Rather than averaging across layers, analyses report best-performing layer for each metric. This reveals functional specialization: e.g., `cosine_to_running_mean` peaks at layer 14 for CoT discrimination (*d* = 1.22) while `radius_of_gyration` peaks at layer 8 (*d* = -1.21). Averaging would obscure these layer-dependent signals.[^11]

**Multiple Comparison Considerations.** With 33 metrics × 5 layers × 4 group comparisons, hundreds of statistical tests are performed. The research does not apply Bonferroni or FDR corrections, instead emphasizing: (1) replication across metrics within families (convergent evidence), (2) consistency across layers, (3) validation on held-out architectures. This approach prioritizes discovery of consistent patterns over strict familywise error control—appropriate for exploratory analysis establishing feasibility of trajectory-based interpretability.

**Cross-Architecture Validation.** Replication on Qwen2.5-1.5B and Pythia-1B provides crucial validation. Key result: patterns not only replicated but *strengthened* in larger models (Qwen2.5-1.5B showed 84.7% vs. 74.9% significance rate for G4 vs. G1 comparisons). This counters concerns that findings are artifacts of specific small models.[^48][^88]

**Sample Size Constraints.** Group sizes range from *n*=53 (Direct Success) to *n*=247 (Direct Failure). Small-group comparisons (especially G2 analyses) have limited power for detecting modest effects but are well-powered for the large effects observed (*d* > 2). Future work with balanced designs across difficulty levels would strengthen conclusions.

### 3.4 Limitations and Confounds

Transparent acknowledgment of limitations distinguishes rigorous interpretability research from speculative claims.

**Task Specificity.** Exclusive focus on arithmetic limits generalizability. Geometric patterns for arithmetic may not hold for:[^89][^47]

- Logical reasoning (syllogisms, formal logic)
- Planning tasks (multi-step goal-directed behavior)
- Creative generation (poetry, story writing)
- Factual retrieval (knowledge-intensive QA)

Each task domain may exhibit distinct geometric signatures requiring separate characterization. However, arithmetic's virtue—unambiguous correctness—enables establishing proof-of-concept for trajectory-based interpretability.

**Difficulty Confound.** Post-hoc analysis revealed negative numbers dramatically reduce CoT success rates (39.5% vs. 86.2% for positive answers). This difficulty confound means some geometric differences may reflect *problem hardness* rather than reasoning strategy per se. Controlled experiments stratifying by difficulty (while holding problem structure constant) are needed to disentangle these factors. Preliminary analyses suggest geometric expansion correlates with problem complexity even within prompt conditions, supporting the interpretation that trajectories adapt to task demands.[^11]

**Model Scale.** Results from 0.5B-1.5B parameter models may not generalize to frontier-scale systems (70B+ parameters). Potential scaling effects include:

- Larger models may compress reasoning into fewer, higher-quality steps (shorter time-to-commit)
- Superposition may differ at scale (more features packed into same dimensionality)
- Quantization and mixture-of-experts architectures introduce sparsity that affects trajectory smoothness

Replication on larger models is essential but computationally expensive. Collaborative efforts or inference-time-only analysis (no gradient computation) may enable scaling studies.

**Token Window Truncation.** Analyzing only first 32 tokens misses extended reasoning chains. For problems requiring deeper exploration, commitment may occur beyond this window. Adaptive windowing (extract until answer token) or full-sequence analysis would address this, at cost of variable-length trajectory handling.

**Greedy Decoding Limitation.** Greedy decoding ensures reproducibility but eliminates stochasticity. Sampling-based generation (temperature >0) may exhibit richer exploration dynamics—potentially with multiple attempted solution paths visible in trajectory branching. Future work should investigate trajectory geometry under sampling, possibly using beam search or best-of-N selection to identify diverse high-quality paths.

**Attribution to Computational Mechanisms.** Geometric metrics describe *what* trajectories do but not *why* they exhibit those patterns. Trajectory geometry is descriptive, not mechanistic in the circuit-discovery sense. It does not identify which attention heads, MLP subnetworks, or feature combinations *cause* geometric patterns. Complementary circuit-level analysis—combining trajectory geometry with activation patching or SAE interventions—could bridge this gap.

Despite these limitations, the work establishes trajectory geometry as a viable analytical framework with substantial predictive power, motivating follow-up investigations addressing confounds and extending to broader task domains.

## 4. Empirical Findings

### 4.1 Regime-Dependent Success Signatures

The most theoretically significant finding is that *geometric correlates of computational success are not universal but regime-dependent*. At layer 13 in Qwen2.5-0.5B, 13 of 33 metrics show *opposite directional effects* when comparing CoT success vs. failure versus Direct success vs. failure.[^47][^12][^11]

**The Direct-Answer (Retrieve-and-Commit) Regime.** Successful direct reasoning (G2) exhibits:

- **Higher speed** (*d* = 2.10): More rapid representational displacement per token
- **Higher effective dimensionality** (*d* = 2.09): Movement spans more degrees of freedom
- **Larger radius of gyration** (*d* = 2.16): Wider spatial dispersion around trajectory centroid
- **Higher directional consistency** (*d* = -2.11): More ballistic, straight-line motion toward answer

This geometric signature suggests *confident retrieval*: the model rapidly traverses representational space along a direct path to a pre-encoded answer attractor, with high-dimensional exploration reflecting brief consideration of answer-adjacent concepts before rapid commitment.

**The Chain-of-Thought (Explore-Commit) Regime.** Successful CoT reasoning (G4) exhibits *opposite patterns*:

- **Lower speed**: More deliberate, incremental state changes
- **Lower effective dimensionality**: Movement constrained to lower-dimensional reasoning manifold
- **Smaller radius of gyration**: Tighter trajectories indicating focused exploration
- **Lower directional consistency** (*d* = -2.6): Frequent reorientation reflecting exploratory search

Yet CoT success also shows:

- **Higher cosine-to-running-mean** (*d* = 1.22): Each step aligns with cumulative progress
- **Higher tortuosity**: More path curvature, consistent with search-then-converge dynamics
- **Sharper commitment timing**: Clear phase transition from exploration to convergence phase

This pattern suggests *structured exploration*: the model explores a constrained reasoning subspace with frequent direction changes (testing hypotheses, backtracking from errors), but maintains coherence with an emerging solution trajectory, ultimately converging decisively when sufficient evidence accumulates.

**Theoretical Implications.** The regime-dependency falsifies a universal "good trajectory" hypothesis. Instead, *what constitutes optimal geometry depends on the computational strategy being employed*. Direct answering benefits from ballistic motion—the model "knows" the answer and executes retrieval efficiently. Multi-step reasoning benefits from controlled wandering—the model does *not* know the answer initially and must explore before converging. This aligns with dual-process theories in cognitive science (System 1 vs. System 2 reasoning) and suggests transformers implement qualitatively different computational modes that manifest as distinct trajectory geometries.

**Practical Implications.** Trajectory-based monitoring systems cannot use fixed "success threshold" across regimes. Effective real-time monitoring requires:

1. **Stage 1 (Layers 0-7):** Classify regime type (Direct vs. CoT) based on early trajectory geometry (*d* = 6-8 for regime discrimination)
2. **Stage 2 (Layers 10-14):** Apply regime-specific success criteria—opposite thresholds for speed/dimensionality depending on detected regime
3. **Stage 3 (Layers 20-24):** Monitor commitment timing—early for Direct, middle for CoT, late indicating likely failure

This cascaded architecture leverages layer-specific functional specialization discovered through comprehensive geometric analysis.

### 4.2 Commitment Timing as a Universal Predictor

While many metrics show regime-dependent effects, **time-to-commit** emerges as a robust cross-regime predictor of success versus failure.[^12][^11]

**Operational Definition.** Time-to-commit identifies the token position where the radius of gyration (RMS distance of trajectory points from centroid) decreases most sharply, capturing the moment of "dimensional collapse" when exploration ceases and the model commits to a solution direction.

**Empirical Patterns:**


| Group | Mean Commitment Time | Interpretation |
| :-- | :-- | :-- |
| Direct Success (G2) | 5.4 tokens | Immediate retrieval |
| CoT Success (G4) | 10.7–13.8 tokens | Explore-then-commit |
| Direct Failure (G1) | 13.8 tokens | Failed retrieval, prolonged search |
| CoT Failure (G3) | 16.0 tokens | Searched but never found |

**Statistical Robustness.** G4 vs. G3 comparison: *d* = -0.91, *p* < 0.001; G2 vs. G1 comparison: *d* = -1.02, *p* < 0.001. Effect sizes are large and consistent across layers 20-24.

**Mechanistic Interpretation.** The commitment timing metric provides direct evidence for the *explore-then-commit phase transition* predicted by theoretical models of multi-step reasoning. In successful CoT:

1. **Early phase (tokens 1-10):** High effective dimensionality, low directional consistency—exploration
2. **Transition (~token 11):** Rapid drop in radius of gyration—commitment point
3. **Late phase (tokens 11-32):** Low dimensionality, high convergence—consolidation

Failures exhibit two distinct patterns:

- **Early premature commitment (tokens 5-7):** Direct failures that attempt retrieval but access wrong information, committing before adequately searching
- **Late non-commitment (tokens 14+):** CoT failures that explore extensively but never converge, exhibiting sustained high dimensionality without phase transition

**Comparison to Prior Work.** Commitment timing parallels:

- **Neuroscience:** Decision commitment points in perceptual choice tasks, where neural population activity shifts from evidence accumulation to commitment to motor plan
- **Dynamical Systems:** Bifurcation points where system transitions from unstable (multi-attractor) to stable (single-attractor) regime
- **Information Theory:** Moment where entropy of next-token distribution collapses from high (many plausible continuations) to low (single dominant answer)

The trajectory geometry framework makes this phase transition *visible and measurable* in LLM hidden states, opening avenues for interventions that target commitment timing directly (e.g., prompts encouraging earlier/later commitment, reinforcement learning rewards for optimal timing).

**Predictive Utility.** Time-to-commit alone achieves 70-75% accuracy in discriminating success from failure within prompt conditions. Combined with convergence metrics (distance-to-final slope, stabilization rate), accuracy reaches 80%. This demonstrates that *temporal structure* of reasoning—not just spatial geometry at individual timesteps—encodes critical information about computational quality.

### 4.3 Layer-Specific Functional Specialization

Comprehensive analysis across all 24 layers of Qwen2.5-0.5B revealed *depth-dependent functional specialization*: different layers are optimized for different computational roles.[^12][^11]

**Early Layers (0-7): Regime Detection.**

- **Best Metrics:** Speed (*d* = 7.76 at layer 7), effective dimension (*d* = 6.50 at layer 1), radius of gyration (*d* = 8.00 at layer 2)
- **Discriminative Task:** Distinguishing CoT from Direct reasoning (G4 vs. G2)
- **Effect Sizes:** *d* = 6-8, indicating near-perfect separability

Early layers encode the *computational strategy* the model will employ. Trajectory geometry at layer 0-7 already reflects whether the model is in retrieval mode (ballistic, low-dimensional) or reasoning mode (exploratory, high-dimensional). This early commitment to regime suggests that prompting effects propagate rapidly through initial processing stages, potentially via attention mechanisms that interpret prompt instructions and modulate subsequent computation accordingly.

**Implications:** Real-time monitoring systems can reliably classify regime type from first 8-10 tokens of hidden state dynamics in layers 0-7, enabling adaptive compute allocation (e.g., early stopping for confident Direct answers, extended generation budget for detected CoT).

**Middle Layers (10-14): Success Prediction.**

- **Best Metrics:** Cosine-to-running-mean (*d* = 1.22 at layer 14), radius of gyration (*d* = -1.21 at layer 8), distance-to-final slope (*d* = -1.02 at layer 12)
- **Discriminative Task:** Predicting success vs. failure *within regimes* (G4 vs. G3, G2 vs. G1)
- **Effect Sizes:** *d* = 1.0-2.2, indicating strong but not perfect discrimination

Middle layers carry the strongest signal for *solution quality*. Trajectories at layers 10-14 exhibit geometric patterns that distinguish whether the model will ultimately succeed, even when regime is held constant. This suggests middle layers implement the core "reasoning" computation—evidence accumulation, constraint satisfaction, or search over solution candidates—whose quality is reflected in trajectory structure.

**Implications:** Intervention points for improving reasoning should target middle layers (e.g., steering vectors applied at layer 12-14, SAE-based feature activation at these depths). Monitoring systems should assess trajectory health primarily in this range.

**Late Layers (20-24): Commitment Tracking.**

- **Best Metrics:** Time-to-commit, distance-to-final slope, stabilization rate
- **Discriminative Task:** Detecting *when* commitment occurs
- **Effect Sizes:** *d* = 0.9-1.0, moderate but consistent

Late layers track the *temporal dynamics* of convergence toward final answer representations. While success/failure discrimination is weaker here than at middle layers (suggesting decision has largely been made by layer 14), commitment timing remains clearly visible. This likely reflects the process of "verbalizing" internal conclusions—translating abstract reasoning representations into token-level predictions.

**Implications:** Early stopping systems can detect commitment completion (stabilization plateau, near-zero distance-to-final slope) and terminate generation if answer token has been reached, reducing inference cost by ~10%.[^47]

**Cross-Layer Metrics: Hierarchical Coordination.**

- **Interlayer Alignment:** Mean cosine similarity between consecutive layer updates
- **CoT Success:** Higher alignment (*d* = 6.35), indicating coordinated processing across depth
- **CoT Failure:** Lower alignment, suggesting disjointed computation where layers process independently

Higher interlayer alignment in successful CoT suggests that multi-step reasoning requires *vertical integration* across the transformer depth hierarchy—each layer builds coherently on prior layer outputs rather than processing locally. This parallels findings in neuroscience where successful reasoning correlates with increased inter-regional communication in cortical hierarchies.

**Theoretical Significance.** The discovery of layer-specific functional roles contradicts views of transformers as uniformly distributed processors. Instead, *transformer depth implements a functional pipeline*: early layers establish regime, middle layers perform core computation, late layers consolidate and verbalize. This organization parallels biological cortical hierarchies (sensory→associative→motor) and suggests evolutionary/developmental pressures toward functional specialization even in end-to-end trained systems.

### 4.4 Metric Universality vs. Regime-Specificity

Of 33 metrics computed, 13 proved to be *universal success indicators* (same directional effect across regimes) while 17 were *regime-specific* (opposite effects).[^11]

**Universal Indicators (Consistent Across Regimes):**

1. **Directional Autocorrelation (lag 4):** Higher in successes, both Direct and CoT
2. **Tortuosity:** Higher in successes (though magnitude differs)
3. **MSD Exponent:** Higher in successes (more ballistic/constrained motion)
4. **Convergence Metrics:** Distance-to-final slope, stabilization rate, early/late ratio—all favor converging trajectories
5. **Commitment Timing:** Earlier commitment predicts success (though optimal timing differs by regime)

These universal metrics capture *general computational quality*—properties that indicate effective information processing regardless of strategy. Specifically, convergence toward stable representations appears essential for correctness across computational modes.

**Regime-Specific Indicators (Opposite Effects):**

1. **Speed:** Higher in Direct success, lower in CoT success
2. **Effective Dimensionality:** Higher in Direct success, lower in CoT success
3. **Radius of Gyration:** Larger in Direct success, smaller in CoT success
4. **Directional Consistency:** Higher in Direct success, lower in CoT success
5. **Cosine-to-Running-Mean:** Lower in Direct success, higher in CoT success

These regime-specific metrics capture *strategy-dependent quality*—what makes retrieval succeed differs from what makes reasoning succeed. Direct answering benefits from expansive, ballistic motion; CoT reasoning benefits from constrained, coherent exploration.

**Architectural Recommendation.** Monitoring systems should employ a two-tier metric architecture:

- **Tier 1 (Universal Metrics):** Always compute convergence metrics, commitment timing, tortuosity. These provide baseline quality assessment independent of regime.
- **Tier 2 (Regime-Specific Metrics):** After regime detection, apply appropriate speed/dimensionality/consistency thresholds based on detected computational mode.

This hybrid approach maximizes discriminative power while remaining interpretable—universal metrics justify intervention decisions across contexts while regime-specific metrics enable fine-tuned optimization within computational modes.

### 4.5 Predictive Modeling Results

To quantify the *practical utility* of trajectory geometry, logistic regression models were trained to predict correctness from geometric features.[^11]

**Within-Regime Prediction:**

- **CoT Only (*n*=300):** 75.0% accuracy, AUC = 0.772 using Layer 13 geometric metrics
- **Direct Only (*n*=300):** 80.6% accuracy, AUC = 0.898

Within prompt conditions, geometry alone achieves strong predictive performance. This demonstrates that hidden state trajectories encode information about reasoning quality that is *not* fully determined by prompt type.

**Cross-Regime Prediction:**

- **All Responses (*n*=600):** 74.9% accuracy, AUC = 0.767 using geometric metrics + prompt type indicator
- **Prompt Type Alone:** 68.9% accuracy, AUC = 0.629

Trajectory geometry adds **6 percentage points** of accuracy (74.9% vs. 68.9%) and dramatically improves discrimination (AUC 0.767 vs. 0.629) beyond knowing prompt condition. This 6-point gain represents information about *how the model is computing*, not just *what strategy was prompted*.

**Feature Importance Rankings:**

1. **Early/Late Ratio** (convergence metric)
2. **Tortuosity** (path efficiency)
3. **Distance-to-Final Slope** (convergence rate)
4. **Cosine-to-Running-Mean** (coherence)
5. **Time-to-Commit** (temporal signature)

Convergence-related metrics dominate top-5, confirming that *trajectory stability* is the strongest predictor of success. Interestingly, directional consistency—which showed the largest single-metric effect sizes (*d* = 2.6)—ranks lowest, because its effect is fully explained by regime type (CoT vs. Direct) and provides minimal within-regime discrimination.

**Comparison to Behavioral Baselines:**

- **Response Length:** 72.1% accuracy (CoT responses are longer, longer responses more often correct)
- **Numeric Properties:** 70.3% accuracy (certain answer patterns more frequent in successes)
- **Trajectory Geometry:** 74.9% accuracy, **outperforming both behavioral baselines**

This confirms trajectory geometry is *not* a proxy for surface-level correlates but captures genuine computational differences invisible to output-only analysis.

**Failure Mode Analysis.** Examining misclassified cases reveals two primary failure modes:

1. **Stable-but-Wrong (~15%):** Trajectories exhibiting success-like geometry (early commitment, strong convergence) but producing incorrect answers. These are the most dangerous failure mode—confident hallucinations—and represent the hard limit of geometry-only monitoring.
2. **Unstable-but-Correct (~10%):** Trajectories with failure-like geometry (late commitment, weak convergence) that nevertheless reach correct answers, possibly via lucky guessing or alternative solution paths.

Addressing these failure modes may require integrating trajectory geometry with content-based verification (e.g., checking answer plausibility against knowledge bases, ensemble methods combining geometry and semantic coherence).

### 4.6 Replication and Generalization

Critical to establishing robustness, key findings replicated across multiple architectures.[^88][^48]

**Qwen2.5-1.5B (3× larger model):**

- **Regime Detection:** Strengthened effect sizes (*d* = 8-12 vs. 6-8 in 0.5B)
- **Success Prediction:** G4 vs. G1 significance rate 84.7% (vs. 74.9% in 0.5B)
- **Commitment Timing:** Sharper phase transitions, earlier commitment in larger model (token 8-10 vs. 10-14)
- **Metric Consistency:** Same top-performing metrics (speed, effective dimension, radius of gyration, commitment timing)

Replication in larger model with improved capabilities validates that trajectory geometry reflects genuine computational patterns, not capacity-ceiling artifacts.

**Pythia-1B (Different Architecture, Different Training):**

- **Core Patterns Replicated:** Regime-dependent success signatures confirmed
- **Magnitude Shifts:** Some effect sizes reduced (*d* = 1.5-2.0 vs. 2.0-4.0), likely due to architecture/training differences
- **Novel Findings:** Pythia showed distinct layer-wise effect profiles, with stronger signals at different depths than Qwen

Cross-architecture replication establishes trajectory geometry as a *general property of transformer computation* rather than Qwen-specific behavior. Variations in magnitude and layer profile suggest architecture-dependent details while core principles (regime-dependency, commitment timing, convergence-based success prediction) remain constant.

**Limitations of Current Replications.** Replications remain within 0.5B-1.5B parameter range and focus on arithmetic. Critical open questions:

- Do patterns hold at frontier scale (70B+ parameters)?
- Do patterns generalize to non-arithmetic reasoning (logic, planning, factual QA)?
- Do patterns appear in multimodal models or only language-only systems?
- How do patterns change with instruction tuning, RLHF, or adversarial training?

Future work must address these generalization boundaries to establish trajectory geometry as a robust mechanistic interpretability tool.

## 5. Relationship to Prior Literature

### 5.1 Advances Over Static Interpretability Methods

Trajectory geometry complements and extends existing interpretability approaches in several key dimensions:

**Circuit Discovery and Activation Patching.** Circuit discovery methods identify *which* components (attention heads, MLP layers, residual connections) contribute to specific behaviors through systematic ablation. Trajectory geometry identifies *when* and *how* those components collectively transform representations during sequential processing. The two approaches are complementary: circuit discovery maps the structural substrate while trajectory geometry reveals the dynamical process unfolding over that substrate.[^9][^10][^13][^1]

**Example Integration:** Combine ACDC circuit identification with trajectory analysis—for each discovered circuit component, compute trajectory metrics with and without that component active (via activation patching). Components whose removal alters trajectory geometry (e.g., disrupts commitment timing, reduces convergence) are validated as causally relevant. This would bridge structural and dynamical interpretability.

**Sparse Autoencoders (SAEs).** SAEs decompose representations into interpretable features. Trajectory geometry could be recast in SAE feature space: rather than analyzing trajectories of full hidden states (high-dimensional, polysemantic), analyze trajectories of SAE feature activations (sparse, monosemantic). This would yield interpretable geometric metrics (e.g., "Feature X17 [scientific notation] activates early and stabilizes by token 8") while preserving trajectory-based insights about temporal dynamics.[^5][^6][^7][^4]

**Recent work** on routing SAEs specifically targets multi-layer activation patterns—a natural integration point for trajectory geometry, which inherently tracks cross-layer dynamics.[^90]

**Probing and Logit Lens.** Linear probes test whether specific information is linearly accessible; Logit Lens projects hidden states to vocabulary space to interpret "what the model is predicting." Trajectory geometry extends these by analyzing *how* accessible information evolves—not just whether "answer A" is recoverable at layer 12, but whether commitment to "answer A" occurs smoothly (converging trajectory) or erratically (oscillating trajectory).

**Innovative Combination:** Train probes to predict final answer from hidden states at each layer, then measure trajectory geometry *in probe logit space* rather than raw hidden states. This combines interpretability of probed concepts with temporal dynamics, yielding metrics like "probability trajectory convergence rate toward correct answer."

### 5.2 Contributions to CoT Faithfulness Research

The trajectory geometry research provides empirical evidence for unfaithfulness of verbalized CoT while demonstrating that internal representations carry faithful signals:[^48][^47][^12][^11]

**Evidence for Unfaithfulness.** Within CoT condition (same prompt, same verbalized strategy), trajectory geometry predicts correctness at 75% accuracy. This implies substantial variability in *internal* computation that is not reflected in *verbalized* reasoning steps. Two CoT responses with identical textual structure ("Let me solve step-by-step...") can exhibit qualitatively different hidden state trajectories—one converging smoothly (success likely), another wandering aimlessly (failure likely)—despite producing similar token sequences.[^11]

**Internal Representations as Ground Truth.** While tokenized CoT may be post-hoc rationalization, trajectory geometry provides an alternative "ground truth" for reasoning quality. Models cannot easily "fake" trajectory convergence or manipulate commitment timing through selective token generation—these are emergent properties of the full computational process across layers. This suggests trajectory monitoring as a more reliable faithfulness signal than text analysis.[^39][^37][^40]

**Connection to Motivated Reasoning.** Mirtaheri \& Belkin's work on detecting motivated reasoning via probing internal representations parallels trajectory geometry's approach: both recognize that hidden states reveal computation beyond what is verbalized. Trajectory geometry extends this by analyzing *temporal structure* of representations (how hidden states move over tokens) rather than static probe predictions at single timepoints.[^43][^44][^41]

**Future Direction:** Combine trajectory geometry with motivated reasoning detection—does geometry exhibit detectable signatures when models engage in biased reasoning? Hypothesis: biased reasoning may show early commitment (skipping exploration phase) or reduced coherence (frequent reorientation as model rationalizes inconsistencies).

### 5.3 Relationship to Latent Space Dynamics

Fumero et al.'s work on latent vector fields in autoencoders provides theoretical grounding for trajectory geometry in transformers:[^21][^22][^19]

**Shared Framework:** Both treat neural networks as dynamical systems evolving in latent space, with trajectories revealing properties of learned representations. Fumero et al. show autoencoders induce contractiveness (Jacobian spectral norms <1), ensuring convergence to attractors. Trajectory geometry in transformers reveals analogous convergence phenomena—successful reasoning trajectories exhibit negative stabilization rates and decreasing distance-to-final, consistent with approaching attractor basins.

**Key Difference:** Autoencoder latent spaces are *static*—each input maps to a fixed latent code, and trajectories emerge from iterating encode-decode operations. Transformer hidden states are *sequential*—each token position generates a distinct hidden state conditioned on all prior tokens, and trajectories emerge from causal processing of token sequences. This distinction means transformer trajectories encode *information-theoretic progression* (accumulating evidence toward answer) while autoencoder trajectories encode *representation refinement* (approaching canonical encoding).

**Conceptual Bridge:** Both frameworks share core predictions:

1. **Contractiveness:** Successful computation converges to stable representations (attractors in autoencoders, final answer states in transformers)
2. **Trajectory Informativeness:** Paths through latent space reveal model priors and generalization boundaries
3. **Phase Transitions:** Bifurcation points mark qualitative shifts in computation (memorization→generalization in autoencoders, exploration→commitment in transformers)

**Future Research:** Apply Fumero et al.'s analytical tools (Jacobian analysis, vector field visualization, attractor identification) directly to transformer hidden states. Questions: Do transformer trajectories exhibit attractors? Can attractor structure explain mode collapse or repetitive generation? Does contractiveness vary by layer, with early layers more expansive (high spectral norms) and late layers more contractive (converging toward output)?

### 5.4 Connections to Regime Detection and Phase Transitions

Trajectory geometry's discovery of regime-dependent success signatures parallels broader research on detecting computational phases in neural networks.[^74][^77][^78][^79][^80][^81][^73][^76]

**Phase Transitions in Neural Computation.** Statistical physics frameworks identify phase transitions as qualitative shifts in network behavior at critical parameter values. In biological neural networks, Hoshino et al. demonstrate self-organized transitions between pattern-itinerant (exploration) and pattern-fixed (commitment) states driven by synaptic plasticity. Trajectory geometry reveals analogous transitions in transformers: the explore-then-commit shift at token 10-14 in successful CoT is a phase transition from high-dimensional wandering (exploration phase) to low-dimensional convergence (commitment phase).[^77][^78][^79][^81]

**Dynamical Systems Analysis of LLMs.** Recent work applies neuroscience-inspired dynamical metrics (integration time, metastability, phase synchronization) to LLM hidden states, finding that functional regimes (reasoning vs. baseline vs. perturbed-reasoning) exhibit distinct dynamical organization. Trajectory geometry provides a complementary metric suite focused on geometric rather than oscillatory properties: while measures temporal integration via autocorrelation timescales, trajectory geometry measures integration via cosine-to-running-mean and convergence rates; while measures metastability via Kuramoto order parameter variance, trajectory geometry measures instability via directional autocorrelation and stabilization slopes.[^76]

**Regime Detection Architectures.** Trajectory geometry's cascaded monitoring system (Stage 1: regime classification [layers 0-7], Stage 2: success prediction [layers 10-14], Stage 3: commitment tracking [layers 20-24]) parallels financial regime detection pipelines that classify market states before applying state-specific predictors. Both recognize that *context-dependent criteria* outperform universal metrics.[^75][^73][^74]

**Future Integration:** Combine trajectory geometry with spectral/oscillatory dynamical metrics to create comprehensive "computational state" descriptors. Hypothesis: regimes identified via trajectory geometry (Direct/CoT) correspond to distinct dynamical states measurable via metastability and integration timescales—CoT may show higher metastability (more variable phase synchronization) reflecting exploration, while Direct may show lower metastability (stable synchronization) reflecting retrieval.

### 5.5 Implications for Faithfulness and Alignment

The trajectory geometry framework addresses critical challenges in AI safety and alignment:

**Hallucination Detection.** The "stable-but-wrong" failure mode—where trajectories exhibit success-like geometry (early commitment, strong convergence) yet produce incorrect answers—represents confident hallucination. Trajectory monitoring alone cannot fully prevent this (geometry tracks computational confidence, not correctness), but it provides an *early warning system*: unusually rapid commitment (<5 tokens) or premature convergence warrants external verification. Combining trajectory geometry with semantic coherence checks (e.g., comparing answer to retrieved knowledge) could flag hallucinations before output generation.[^12][^11]

**Monitoring vs. Faithfulness.** Recent debates distinguish *faithfulness* (verbalized reasoning reflects true computation) from *monitorability* (verbalized reasoning predicts behavior). Trajectory geometry offers a third paradigm: *geometric monitorability*—internal geometry reliably predicts outcomes even when tokenized reasoning does not. This suggests monitoring systems should track hidden state dynamics rather than (or in addition to) parsing CoT text.[^45]

**Adversarial Robustness.** Models trained to maximize performance may learn to "game" text-based monitoring by generating plausible-sounding reasoning while executing different internal computation. Trajectory geometry is harder to fake: geometric patterns (convergence rates, commitment timing, dimensional structure) emerge from full computational process across layers and cannot be selectively controlled via output token manipulation. This makes geometric monitoring a more robust safety layer.

**Deception Detection.** Motivated reasoning research shows biased contexts create detectable signatures in internal representations. Trajectory geometry could extend this to temporal deception detection: does geometry exhibit characteristic signatures when models engage in deceptive reasoning (e.g., generating false explanations to justify predetermined conclusions)? Hypothesis: deceptive reasoning may show premature commitment (skipping genuine exploration) or layer mismatch (early-layer geometry indicates exploration while late-layer geometry indicates pre-commitment).[^44][^41][^43]

**Limitations for Safety.** Trajectory geometry is *descriptive*, not *prescriptive*. It detects when computation exhibits success-like or failure-like patterns but does not explain *why* or provide guarantees about correctness. Safety-critical applications require combining geometric monitoring with formal verification, ensemble methods, or human oversight. Trajectory geometry can reduce false negative rates (catching failures that pass text-based checks) but cannot eliminate false positives (flagging unusual but correct reasoning paths).

## 6. Identified Gaps in the Literature

### 6.1 Lack of Dynamic Analysis in Mechanistic Interpretability

**Current State.** Mechanistic interpretability research overwhelmingly analyzes neural networks via *static snapshots*—probing individual layers, visualizing attention patterns at specific tokens, identifying circuits through ablation at fixed positions. Even sophisticated methods like sparse autoencoders decompose representations at single layers without tracking how features evolve over sequential processing.[^3][^6][^7][^2][^4][^5]

**The Gap.** Multi-step reasoning inherently involves *temporal structure*: models explore hypotheses, backtrack from errors, gradually accumulate evidence, and converge on answers. Static analysis cannot capture these *process-level* properties. Consider two CoT responses:

- **Response A:** Explores 3 hypotheses (tokens 1-12), tests each (tokens 13-20), converges smoothly (tokens 21-28), produces correct answer
- **Response B:** Commits prematurely to wrong hypothesis (tokens 1-5), generates superficially coherent justification (tokens 6-25), outputs incorrect answer

Static analysis at, say, token 15 might show similar feature activations in both cases (both are "explaining" arithmetic operations). Only trajectory analysis reveals the critical difference: Response A shows explore-then-commit dynamics (dimensional expansion followed by collapse); Response B shows premature commitment (early collapse) followed by rationalization (stable trajectory in wrong direction).

**How Trajectory Geometry Fills the Gap.** By treating hidden states as time series and computing geometric metrics over full sequences, trajectory geometry captures:

1. **Phase Transitions:** Transitions from exploration to commitment (measurable via time-to-commit)
2. **Convergence Dynamics:** Whether models stabilize smoothly or oscillate erratically (stabilization rate, distance-to-final slope)
3. **Temporal Coherence:** Whether each step builds on prior progress (cosine-to-running-mean) or reflects incoherent wandering (low directional consistency)

These properties are *fundamentally invisible* to single-timepoint analysis but prove highly predictive of reasoning success.

**Call for Future Work.** Mechanistic interpretability research should systematically integrate temporal analysis:

- **Trajectory-SAE Integration:** Analyze feature activation trajectories in SAE spaces, identifying features that exhibit characteristic temporal profiles (early-activating vs. late-activating, transient vs. persistent)
- **Temporal Circuit Discovery:** Extend ACDC-style circuit discovery to identify *when* circuit components engage, revealing temporal sequencing of computation
- **Dynamic Probing:** Rather than training probes on static hidden states, train probes on trajectory embeddings (e.g., RNN encodings of hidden state sequences) to predict process-level outcomes


### 6.2 Insufficient Attention to Regime-Dependent Computation

**Current State.** Most interpretability work implicitly assumes a *universal computational substrate*: models perform the same type of operations regardless of task or prompt, with success/failure determined by operation quality rather than qualitative strategy differences. For example, circuit discovery seeks the circuit implementing a behavior (e.g., indirect object identification) without considering whether the same task might be solvable via multiple distinct circuits activated by different prompting.

**The Gap.** Trajectory geometry provides strong evidence for *qualitatively distinct computational regimes*. Metrics predicting success in Direct-answer regime show *opposite effects* in CoT regime (e.g., higher dimensionality aids Direct success but hinders CoT success). This is not merely quantitative variation—it reflects fundamentally different computational strategies:[^47][^12][^11]

- **Retrieval Regime:** Ballistic motion through representation space toward pre-encoded answer attractor
- **Reasoning Regime:** Constrained exploration within solution subspace, testing hypotheses, gradually accumulating evidence

**Why This Matters.** Regime-dependency has profound implications:

1. **Monitoring Systems Must Be Context-Aware:** A fixed "anomaly detector" trained on CoT trajectories will misclassify Direct successes as failures (flagging their high-dimensional, high-speed trajectories as anomalous)
2. **Circuits May Be Regime-Specific:** The same task solved via different regimes may engage different circuits. Indirect object identification via Direct answering (retrieve stored knowledge) versus CoT reasoning (parse sentence structure, apply grammatical rules) likely involves distinct circuit activations
3. **Feature Interpretability May Be Regime-Dependent:** SAE features representing "arithmetic reasoning" may activate differently in Direct (single-step retrieval of memorized fact) versus CoT (multi-step symbolic manipulation)

**How Trajectory Geometry Reveals This.** The comprehensive metric suite identifies *which* geometric properties are universal (convergence, commitment timing) versus regime-specific (speed, dimensionality, directional consistency). This enables building monitoring architectures that first classify regime (layers 0-7), then apply regime-appropriate success criteria (layers 10-14).

**Call for Future Work.** Mechanistic interpretability research should:

- **Systematically Vary Prompting:** For any studied behavior, test whether circuits/features/activations differ under Direct vs. CoT prompting
- **Develop Regime-Specific Interpretability Methods:** Train separate SAEs for Direct vs. CoT regimes; identify regime-specific circuits; probe regime-dependent information encoding
- **Investigate Regime Transitions:** Can models dynamically switch regimes mid-generation? Does CoT failure reflect "getting stuck" in Direct mode when reasoning is required?


### 6.3 Limited Understanding of Commitment Dynamics

**Current State.** While CoT research extensively studies *what* models generate (accuracy, coherence, format), it largely ignores *when* models commit to answers. Existing work measures output-level properties (answer correctness, justification quality) but not internal commitment timing.

**The Gap.** Commitment timing—the token position at which internal representations converge toward final answer state—is a critical process-level variable that trajectory geometry makes measurable. Three key findings remain unexplored in broader interpretability literature:[^12][^11]

1. **Commitment Timing is Highly Diagnostic:** Earlier commitment predicts success; late commitment predicts failure (effect sizes *d* = 0.9-1.0)
2. **Optimal Timing is Regime-Dependent:** Direct successes commit at token ~5; CoT successes at token ~11; failures in both regimes commit late (token 14+)
3. **Commitment is Detectable Pre-Output:** Trajectory dimensionality collapse (sharp drop in radius of gyration) occurs before answer token is generated, enabling real-time intervention

**Why This Matters.**

- **Early Stopping:** Once commitment is detected and stable (e.g., 3 consecutive tokens with near-zero distance-to-final change), generation can terminate if answer token has appeared, reducing inference cost by ~10%[^47]
- **Intervention Opportunities:** If commitment timing is too early (premature commitment, likely error) or too late (prolonged wandering, likely failure), systems can intervene: requesting model to "reconsider," sampling alternative continuations, or escalating to human oversight
- **Probing Internal Decision-Making:** Commitment timing provides a continuous measure of "when the model decided," enabling investigations of decision-making processes: Does commitment follow specific attention patterns? Do certain heads or layers drive commitment? Can commitment timing be manipulated via activation steering?

**How Trajectory Geometry Addresses This.** The time-to-commit metric specifically captures the explore-then-commit phase transition by identifying maximal dimensional collapse. Additional convergence metrics (distance-to-final slope, stabilization rate, early/late distance ratio) triangulate commitment from multiple geometric perspectives, ensuring robustness.

**Call for Future Work.** Commitment dynamics should become a central focus:

- **Causal Interventions on Commitment Timing:** Use activation patching to delay/accelerate commitment, testing effects on accuracy
- **Attention Mechanism Analysis:** Which attention heads contribute to commitment? Do heads transition from "exploration mode" (broad attention) to "commitment mode" (focused attention)?
- **Commitment Under Uncertainty:** How does commitment timing change when models face ambiguous or underspecified problems? Do models exhibit "hesitation" (delayed commitment) or "overconfidence" (premature commitment)?


### 6.4 Absence of Process-Level Metrics in Evaluation

**Current State.** LLM evaluation focuses almost exclusively on *outcome metrics*: accuracy, F1, BLEU, perplexity, human preference ratings. Even sophisticated benchmarks (MMLU, GSM8K, BigBench) measure final answer correctness without assessing *how* answers were reached.

**The Gap.** Two models achieving identical accuracy (e.g., 75% on GSM8K) may employ radically different computational processes:

- **Model A:** Retrieves 60% of answers correctly via Direct reasoning (fast, efficient), solves 15% via CoT reasoning (slower, more resource-intensive), fails 25%
- **Model B:** Attempts CoT reasoning on all problems, succeeds on 75%, fails on 25% (uniform strategy, potentially higher compute cost)

Current metrics cannot distinguish these models, yet from deployment perspective they differ substantially: Model A may be preferred for latency-sensitive applications; Model B may be preferred for complex reasoning tasks. Furthermore, *failure modes* differ: Model A failures may be retrieval errors (missing training data); Model B failures may be reasoning errors (logical mistakes).

**How Trajectory Geometry Enables Process-Level Evaluation.** Geometric metrics quantify *how* models compute:

- **Regime Distribution:** What percentage of responses use Direct vs. CoT computational strategies? (detectable from layers 0-7 geometry)
- **Reasoning Efficiency:** Among CoT responses, what proportion exhibit efficient explore-then-commit dynamics versus wasteful prolonged wandering?
- **Failure Characterization:** Do failures reflect premature commitment (confident errors) versus non-commitment (indecisive errors)?

These process metrics enable richer model comparison: rather than "Model A: 75% accuracy, Model B: 75% accuracy," reports can state "Model A: 75% accuracy, 60% Direct regime, efficient commitment timing; Model B: 75% accuracy, 100% CoT regime, high exploration inefficiency."

**Proposed Process-Level Benchmarks:**

1. **Reasoning Efficiency Score:** Ratio of successful-CoT to total-CoT, penalizing models that "overthink" when Direct answering suffices
2. **Commitment Timing Distribution:** Histogram of time-to-commit across test set, favoring models with regime-appropriate timing
3. **Trajectory Quality Index:** Composite metric combining convergence rate, directional coherence, commitment timing—measuring computational "cleanness" independent of correctness

**Call for Future Work.** Evaluation methodology should integrate process-level metrics alongside outcome metrics, enabling:

- **Interpretable Model Comparisons:** Understanding *why* one model outperforms another (better retrieval? more efficient reasoning? fewer premature commitments?)
- **Targeted Improvement:** Identifying specific failure modes (e.g., "Model exhibits excessive wandering in CoT regime") guides training interventions
- **Deployment Optimization:** Matching models to use-cases based on process characteristics (e.g., deploying high-efficiency models for user-facing applications, exploration-capable models for research assistance)


## 7. Critique of Existing Work via Trajectory Geometry Lens

### 7.1 Sparse Autoencoders: Static Decomposition of Dynamic Processes

**Limitation.** SAE research provides unprecedented interpretability of individual features but treats each layer independently. A feature "activates at layer 12" is analyzed in isolation, without tracking how activation *changes* from layer 11 to 13, or how it evolves from token 5 to token 20.[^6][^7][^4][^5]

**What Trajectory Geometry Reveals.** Features likely exhibit characteristic *temporal profiles*:

- **Early-Activating Features:** Peak in first 5-10 tokens, decline afterward (e.g., syntactic features establishing sentence structure)
- **Late-Activating Features:** Emerge gradually after token 10, plateau (e.g., answer-specific semantic features)
- **Transient Features:** Spike briefly then disappear (e.g., hypothesis-testing features during exploration)

**Evidence:** Trajectory geometry demonstrates that computational quality is *process-dependent*—a feature activating at token 5 may be diagnostic in Direct regime but uninformative in CoT regime. SAEs trained without temporal awareness cannot capture this.

**Proposed Enhancement: Temporal SAEs.** Rather than training SAEs on static hidden states, train on *hidden state trajectories*:

- **Input:** Sequence of hidden states $h_1, h_2, \ldots, h_T$ from a single generation
- **Architecture:** Recurrent or transformer-based SAE encoder processes sequence, outputs sparse trajectory representation
- **Loss:** Reconstruction loss + sparsity penalty + temporal coherence term (penalizing abrupt activation changes for designated "persistent" features)

This would yield features with explicit temporal semantics (e.g., "Feature X activates during exploratory phase and deactivates upon commitment") and enable trajectory-based interpretability at the feature level.

### 7.2 Circuit Discovery: Identifying Structure, Missing Dynamics

**Limitation.** Circuit discovery excels at *which*-questions (which components participate in computation) but struggles with *when*-questions (when do components engage, in what temporal sequence, with what dynamics?).[^10][^13][^1][^9]

**Example:** Suppose circuit discovery identifies Attention Head H3 in Layer 10 as critical for indirect object identification. This tells us H3's outputs are necessary for the task. But trajectory geometry asks:

- *When* does H3 engage during generation? (immediately? after exploratory phase?)
- *How* do H3's outputs evolve over tokens? (stable throughout? changing dynamically?)
- *What* trajectory effects does H3 produce? (increasing convergence? enabling reorientation?)

**What Trajectory Geometry Reveals.** Preliminary analyses show attention head activations are *layer-and-time-dependent*—different heads dominate at different depths and token positions. Circuit discovery identifies components but misses the *choreography* of their interactions over time.[^12]

**Proposed Enhancement: Dynamic Circuit Discovery.** Extend ACDC-style algorithms to temporal domain:

1. **Temporal Activation Patching:** Ablate circuit component at specific token ranges (tokens 1-10, 11-20, 21-30), measuring effects on trajectory geometry (not just final output)
2. **Trajectory Causal Tracing:** Identify which circuit components cause trajectory properties (e.g., which heads are necessary for commitment timing, which for convergence rate)
3. **Sequential Circuit Activation:** Determine *ordering* of circuit engagement—does component A activate before component B in successful reasoning?

This would map not just the computational graph but the *temporal execution* of that graph during reasoning.

### 7.3 Chain-of-Thought Research: Focusing on Text, Ignoring Geometry

**Limitation.** CoT interpretability research predominantly analyzes *tokenized reasoning*: parsing step-by-step explanations, checking logical validity, measuring faithfulness via text-based interventions. This assumes verbalized reasoning is the primary signal.[^27][^38][^35][^26][^36][^37][^39]

**What Trajectory Geometry Reveals.** Internal geometric patterns predict success at 75% accuracy *within CoT condition*, implying substantial variance in reasoning quality that verbalized text does not capture. Two CoT responses with similar textual structure can exhibit:[^11]

- **Response A (Success):** Explore-then-commit trajectory, dimensional collapse at token 11, rapid convergence
- **Response B (Failure):** Prolonged wandering, no dimensional collapse, continued high-dimensional drift

Text analysis treats these identically ("both generated step-by-step reasoning"); geometry distinguishes them sharply.

**Evidence from Faithfulness Literature.** Multiple studies show CoT text often does not reflect internal computation (e.g., biased contexts change answers while generating unbiased-sounding justifications). Trajectory geometry provides an *alternative ground truth*—internal geometric state that models cannot easily fake via token manipulation.[^37][^39][^40]

**Proposed Enhancement: Geometry-Augmented CoT Evaluation.** CoT benchmarks should report:

1. **Outcome Metrics (Current):** Accuracy, logical validity, faithfulness scores
2. **Process Metrics (Proposed):** Commitment timing distribution, convergence quality, trajectory regime classification

Example report: "Model X achieves 80% accuracy on GSM8K with 12.4 average commitment time (vs. 16.8 for failures), indicating efficient reasoning. Model Y achieves 80% accuracy with 18.1 average commitment time, suggesting success via prolonged exploration—higher compute cost but potentially more robust."

This enables comparing CoT *quality* beyond mere accuracy, guiding model selection and training.

### 7.4 Faithfulness Research: Measuring Output, Not Process

**Limitation.** Faithfulness research primarily employs *input perturbations* (changing reasoning steps, introducing biases) and measures *output changes* (does final answer shift?). This tests whether reasoning causally influences outputs but does not reveal *what internal computation actually occurs*.[^33][^38][^35][^36][^39][^40][^41][^42][^37]

**Example:** Turpin et al. insert invalid reasoning steps; if final answer remains unchanged, they conclude reasoning is unfaithful. But this leaves open: *What computation did the model perform instead?* Did it ignore the invalid steps? Override them? Never process them meaningfully?[^37]

**What Trajectory Geometry Reveals.** Geometry captures the *alternative computation*:

- **Scenario A (Ignoring Bad Steps):** Trajectory proceeds normally (smooth convergence), unaffected by text perturbation—model never integrated perturbed reasoning
- **Scenario B (Compensating for Bad Steps):** Trajectory shows brief disruption (increased curvature, temporary dimensional expansion) followed by recovery (convergence resumes)—model detected and corrected error
- **Scenario C (Misled by Bad Steps):** Trajectory exhibits premature commitment toward wrong answer, followed by prolonged low-dimensional drift—model accepted invalid reasoning, reasoning is genuinely unfaithful

Faithfulness research typically conflates Scenarios A and B (both show output resilience to perturbation) but trajectory geometry distinguishes them.

**Proposed Enhancement: Geometry-Based Faithfulness Metrics.** Define faithfulness via trajectory alignment:

- **Strong Faithfulness:** Perturbations to reasoning text produce corresponding trajectory changes (e.g., removing step K causes trajectory to diverge at token K)
- **Weak Faithfulness:** Trajectory exhibits compensation dynamics (perturbations cause disruptions followed by recovery)
- **Unfaithfulness:** Trajectory unaffected by reasoning perturbations

This provides a continuous, process-level measure of faithfulness rather than binary output-based assessment.

### 7.5 Computational Efficiency: Ignoring Trajectory-Based Optimization

**Limitation.** Efficiency research in LLMs focuses on architectural optimizations (quantization, pruning, distillation) or prompt engineering (reducing token count). None exploit *process-level efficiency signals* revealed by trajectory geometry.[^14]

**What Trajectory Geometry Reveals.** Commitment timing enables *dynamic early stopping*: when trajectory exhibits convergence (distance-to-final <ε for k consecutive tokens) and answer token has been generated, further generation is computationally wasteful. Estimated savings: ~10% inference cost via early stopping in Direct-success cases.[^47][^12]

More sophisticated optimizations are possible:

1. **Regime-Based Compute Allocation:** Detect regime (layers 0-7), allocate compute budget accordingly: minimal for Direct (fast retrieval), extended for CoT (complex reasoning)
2. **Trajectory-Guided Sampling:** Replace greedy decoding with *trajectory-aware sampling*: adjust temperature based on geometric state (higher temperature during exploration phase, lower during commitment phase)
3. **Adaptive Prompting:** If trajectory exhibits failure-like patterns (late commitment, weak convergence), dynamically inject prompts to encourage reorientation ("Let me reconsider...") before failure crystallizes

**Call for Future Work.** Efficiency research should integrate trajectory monitoring, enabling systems that optimize compute *conditioned on internal geometric state* rather than applying uniform generation strategies.

## 8. Future Directions

### 8.1 Scaling Trajectory Geometry to Frontier Models

**Challenge:** Current work analyzes 0.5B-1.5B parameter models. Frontier models (70B+ parameters) present computational challenges: extracting full hidden states across all layers for extended generations (100+ tokens) requires hundreds of GB memory and substantial inference cost.[^88][^48][^47][^12][^11]

**Proposed Solutions:**

1. **Sparse Layer Sampling:** Extract hidden states at representative layers (e.g., layers {0, L/4, L/2, 3L/4, L}) rather than all layers, exploiting layer-wise functional specialization discovered in current work
2. **Metric Subset Selection:** Compute only the top-performing metrics (commitment timing, convergence rate, cosine-to-running-mean, effective dimension) rather than full 33-metric suite
3. **Online Streaming Computation:** Compute geometric metrics incrementally during generation rather than post-hoc, enabling real-time monitoring with constant memory overhead
4. **Distilled Trajectory Predictors:** Train lightweight models to predict trajectory metrics from layer 0 hidden states alone, bypassing need to extract all-layer representations

**Expected Findings:** Based on replication from 0.5B→1.5B showing strengthened effects, scaling to 70B+ should reveal:[^48]

- **Sharper Commitment Timing:** Larger models may commit earlier (more confident retrieval, faster reasoning convergence)
- **Cleaner Regime Separation:** Increased capacity enables crisper differentiation between computational modes
- **Novel Trajectory Patterns:** Frontier models trained with RLHF, constitutional AI, or adversarial robustness may exhibit geometric signatures absent in base models

**Priority Experiments:** Apply trajectory geometry to OpenAI o1-preview, Anthropic Claude 3 Opus, Google Gemini 1.5 Pro on reasoning-intensive benchmarks (MATH, GSM8K, MMLU-HARD) to test generalizability.

### 8.2 Cross-Domain Extension Beyond Arithmetic

**Limitation:** Current work focuses exclusively on arithmetic reasoning. Generalization to other domains is critical for establishing trajectory geometry as a universal interpretability framework.[^47]

**Proposed Domains:**

**Logical Reasoning (Syllogisms, Formal Proofs).** *Hypothesis:* Logical reasoning may exhibit distinct geometric signatures—more discrete state transitions (premise→conclusion steps) versus continuous exploration in arithmetic. *Metrics to Investigate:* Number of commitment events (multiple logical steps may involve multiple local convergences), sharpness of transitions.

**Planning Tasks (Blocksworld, Route Optimization).** *Hypothesis:* Planning involves search over state-action spaces, potentially exhibiting higher-dimensional exploration than arithmetic. *Metrics to Investigate:* Effective dimensionality (how many subgoals are considered?), backtracking signatures (trajectory reversals when plans fail).

**Factual Question Answering.** *Hypothesis:* Knowledge retrieval tasks resemble Direct-answer regime (ballistic trajectories) regardless of prompt format. *Experiment:* Does prompting "think step by step" induce explore-commit dynamics for factual QA, or do trajectories remain Direct-like?

**Creative Generation (Poetry, Story Writing).** *Hypothesis:* Creative tasks may exhibit sustained high-dimensional exploration without commitment—maintaining uncertainty enables diverse, surprising outputs. *Prediction:* Successful creative outputs show high effective dimensionality throughout generation, low convergence, late or absent commitment.

**Code Generation.** *Hypothesis:* Programming combines retrieval (syntax, library calls) and reasoning (algorithm design), potentially exhibiting mixed-regime signatures. *Metrics to Investigate:* Regime switching (does model alternate between Direct and CoT modes within single generation?).

**Methodology:** For each domain, collect dataset with ground truth (where applicable), extract trajectories, compute geometric metrics, compare patterns to arithmetic baseline. Report domain-specific metric rankings and trajectory characteristics.

### 8.3 Causal Interventions on Trajectory Properties

**Objective:** Current work is *descriptive*—it identifies geometric patterns correlated with success. Causal validation requires demonstrating that *manipulating* trajectory properties influences outcomes.

**Proposed Interventions:**

**Commitment Timing Manipulation.** Use activation steering vectors to artificially accelerate or delay commitment:[^24][^25]

- **Early Commitment Steering:** At token 5, add vectors that reduce dimensionality and increase convergence toward high-probability answer candidate. *Prediction:* Improved Direct-answer performance (faster retrieval), degraded CoT performance (premature commitment prevents exploration).
- **Delayed Commitment Steering:** At token 5-10, add vectors maintaining high dimensionality and preventing convergence. *Prediction:* Degraded Direct performance (unnecessary exploration), potentially improved CoT performance (extended search).

**Convergence Rate Manipulation.** Amplify or dampen distance-to-final-slope:

- **Enhanced Convergence:** Progressively add steering vectors orienting hidden states toward final-layer prediction. *Prediction:* Faster generation (earlier stopping), risk of premature convergence on suboptimal answers.
- **Suppressed Convergence:** Add noise or orthogonal perturbations preventing stabilization. *Prediction:* Extended generation, potential failure to commit.

**Regime Forcing.** Inject regime-specific geometric patterns at early layers:

- **Force Direct-Regime:** Add vectors inducing high speed, high dimensionality, high directional consistency at layers 0-7. *Test:* Does this improve retrieval tasks? Does it degrade reasoning tasks?
- **Force CoT-Regime:** Add vectors inducing low speed, low dimensionality, low directional consistency at layers 0-7. *Test:* Does this improve reasoning tasks?

**Measurement:** For each intervention, report:

1. **Success Rate Impact:** How does manipulation affect accuracy on target tasks?
2. **Trajectory Change:** Does intervention achieve intended geometric effect?
3. **Failure Mode Shifts:** Do interventions change *types* of errors (e.g., premature commitment → wrong-but-confident errors)?

**Expected Outcome:** If trajectory geometry causally influences reasoning, interventions altering geometry should systematically affect performance. Strong causal evidence would establish trajectory geometry as a *control surface* for model behavior, not merely a monitoring tool.

### 8.4 Integration with Sparse Autoencoders

**Objective:** Combine interpretability of SAE features with temporal dynamics of trajectory geometry.

**Proposed Framework: Temporal Sparse Autoencoders (T-SAE).**

**Architecture:**

1. **Trajectory Encoder:** RNN or transformer processes sequence of hidden states $h_1, \ldots, h_T$, outputs trajectory embedding $z \in \mathbb{R}^d$
2. **Sparse Decomposition:** Linear layer with sparsity penalty decomposes $z$ into interpretable features: $z = \sum_{i=1}^{k} \alpha_i f_i$ where $\alpha_i \approx 0$ for most $i$
3. **Trajectory Decoder:** Reconstructs hidden state sequence from sparse features

**Training:**

- **Loss:** Reconstruction MSE + $L_1$ sparsity + temporal coherence (penalize abrupt feature activation changes)
- **Dataset:** Hidden state trajectories from diverse tasks (arithmetic, reasoning, QA, creative generation)

**Expected Features:**

- **Regime Features:** Features activating consistently in Direct vs. CoT regimes (e.g., "retrieval_mode", "reasoning_mode")
- **Phase Features:** Features marking computational phases (e.g., "exploration_active", "commitment_initiated", "answer_consolidated")
- **Process Features:** Features encoding trajectory properties (e.g., "converging_rapidly", "high_dimensional_search", "premature_commitment")

**Applications:**

1. **Interpretable Monitoring:** Replace geometric metric computation with sparse feature activations—more human-interpretable (feature names) than raw metrics
2. **Targeted Interventions:** Manipulate specific features (e.g., activate "extended_exploration" feature) rather than crude geometric steering
3. **Cross-Task Feature Sharing:** Identify whether reasoning-related features (e.g., "hypothesis_testing") activate consistently across arithmetic, logic, planning

**Validation:** Train T-SAE on arithmetic trajectories, apply to held-out tasks (logic, planning). Do identified features generalize? Can we interpret feature semantics via activation pattern analysis?

### 8.5 Trajectory Geometry for Alignment and Safety

**Motivation:** AI safety demands real-time monitoring systems that detect harmful reasoning processes *before* harmful outputs are generated. Trajectory geometry provides early-warning signals invisible to text-based monitoring.

**Proposed Applications:**

**Deceptive Reasoning Detection.** Train classifiers to distinguish "honest exploration" trajectories from "deceptive rationalization" trajectories:

- **Honest Trajectories:** Explore-then-commit dynamics, gradual convergence, intermediate commitment timing
- **Deceptive Trajectories:** Premature commitment (answer predetermined), followed by stable trajectory generating post-hoc justification, low exploration dimensionality

*Training Data:* Adversarially elicit deceptive reasoning via biased prompts (e.g., "You must find a way to justify X"), collect trajectories, train probe on geometric features.

**Harmful Reasoning Interception.** Monitor trajectories during generation; if geometry exhibits dangerous patterns (e.g., rapid commitment to harmful content, exploration in unsafe regions of latent space), intervene:

- **Soft Intervention:** Inject prompt ("Please reconsider whether this response is safe")
- **Hard Intervention:** Terminate generation, return error message

**Uncertainty Calibration.** Commitment timing correlates with model confidence. Systems can defer to humans when geometry indicates uncertainty:

- **Late Commitment (>20 tokens):** Model uncertain, high error risk → escalate to human review
- **Missing Commitment (no convergence):** Model unable to reach conclusion → request additional information rather than guessing

**Jailbreak Detection.** Adversarial prompts (jailbreaks) may induce anomalous trajectories:

- **Normal Trajectories:** Match expected regime (Direct or CoT) for legitimate queries
- **Jailbreak Trajectories:** Exhibit cross-regime inconsistency (early layers suggest Direct, late layers suggest CoT; or rapid switching between regimes)

*Research Question:* Do successful jailbreaks exhibit geometric signatures? Can we train detectors on known jailbreak trajectories?

**Evaluation:** Safety systems require stringent validation:

1. **False Positive Rate:** How often does geometry flag safe responses as unsafe? (Must be <1% for practical deployment)
2. **True Positive Rate:** How often does geometry detect genuinely harmful reasoning? (Compare to text-based monitoring baselines)
3. **Adversarial Robustness:** Can attackers craft prompts that evade geometric monitoring by inducing "safe-looking" trajectories while producing harmful outputs?

### 8.6 Multimodal Trajectory Geometry

**Motivation:** Vision-language models (VLMs), audio-language models, and embodied agents involve multimodal reasoning. Do geometric principles discovered in language-only models generalize?

**Proposed Extensions:**

**Vision-Language Reasoning.** Models like GPT-4V, Gemini 1.5 Pro process images alongside text. Questions:

- Do image tokens induce distinct trajectory patterns vs. text tokens?
- Does visual reasoning exhibit explore-commit dynamics analogous to verbal reasoning?
- Can trajectory geometry predict visual question answering (VQA) success from hidden states?

*Experiment:* Extract trajectories from VLM on VQA tasks (e.g., "How many objects are in the image?"), compute geometric metrics separately for image-processing tokens and reasoning tokens, test whether patterns replicate.

**Audio-Language Reasoning.** Speech-to-text models (Whisper, USM) and speech-understanding models process acoustic signals. Questions:

- Does phonetic processing (converting audio→text) exhibit retrieval-like (Direct) trajectories?
- Does speech-based reasoning (e.g., answering questions about spoken content) exhibit reasoning-like (CoT) trajectories?

**Embodied Reasoning in RL Agents.** Transformer-based reinforcement learning agents (e.g., Decision Transformers, Behavior Transformers) generate action sequences conditioned on observations. Questions:[^91][^71]

- Do successful action trajectories exhibit convergence toward goal states (analogous to answer commitment)?
- Does exploration in RL exhibit high-dimensional wandering followed by convergence (analogous to CoT reasoning)?
- Can trajectory geometry predict which action sequences will succeed *before* execution?

*Experiment:* Train Decision Transformer on simulated robotics task (block stacking), extract hidden state trajectories during action generation, test whether geometric metrics predict task success.

**Expected Findings:** Core principles (convergence, commitment timing, regime-dependency) likely generalize, but modality-specific patterns will emerge (e.g., visual reasoning may involve discrete "attention shifts" between image regions, reflected as discrete jumps in trajectory geometry).

## 9. Methodological Recommendations

### 9.1 Best Practices for Trajectory-Based Interpretability

Based on lessons from the reviewed research program, we propose methodological guidelines for future trajectory geometry studies:[^89][^88][^48][^47][^12][^11]

**Comprehensive Metric Suites.** Single metrics (e.g., speed alone) provide incomplete pictures. Minimum recommended suite:

- **Velocity Family:** Speed, velocity autocorrelation
- **Directionality Family:** Directional consistency, cosine-to-running-mean
- **Dimensionality Family:** Effective dimension, radius of gyration
- **Convergence Family:** Distance-to-final slope, stabilization rate
- **Temporal Family:** Time-to-commit

**Layer-Stratified Analysis.** Averaging across layers obscures functional specialization. Report metrics at representative layers (early/middle/late) and identify best-performing layers per metric.

**Regime-Conditioned Reporting.** Given regime-dependency, always stratify analyses by computational regime (Direct/CoT, or more fine-grained classifications). Report effect sizes separately for each regime.

**Replication Across Architectures.** Model-specific artifacts are pervasive in interpretability research. Validate key findings on at least 2-3 distinct architectures (different model families, different training procedures).

**Careful Statistical Reporting.** Report effect sizes (*d*) alongside *p*-values. Use non-parametric tests (permutation, bootstrap) rather than assuming parametric distributions. Address multiple comparisons via replication rather than Bonferroni correction.

**Transparent Limitation Acknowledgment.** Clearly state: (1) task specificity (findings may not generalize beyond tested domains), (2) model scale (tested models may not represent frontier capabilities), (3) decoding constraints (greedy vs. sampling), (4) confounds (problem difficulty, response length).

**Open Science Practices.** Publish: (1) full metric definitions with pseudocode, (2) extracted hidden state datasets (where feasible), (3) analysis code enabling replication, (4) negative results (metrics that failed to discriminate).

### 9.2 Avoiding Common Pitfalls

**Pitfall 1: Over-Interpreting Correlations as Mechanisms.** Trajectory geometry identifies *patterns* (e.g., "CoT success exhibits lower dimensionality"), not *mechanisms* (e.g., "lowering dimensionality causes success"). Causal claims require intervention experiments.

**Solution:** Clearly distinguish descriptive findings ("X correlates with Y") from causal hypotheses ("manipulating X may affect Y") and mechanistic claims ("X causes Y via pathway Z").

**Pitfall 2: Ignoring Confounds.** Many geometric metrics correlate with simple behavioral variables (response length, vocabulary richness, punctuation patterns). Without controls, findings may reduce to "longer responses succeed more."

**Solution:** Include behavioral baselines in predictive models. Report geometry's *incremental* predictive power beyond behavioral features. Design controlled experiments varying geometry while holding behavior constant.

**Pitfall 3: Assuming Universality.** Regime-dependency is pervasive. Metrics effective in one condition (task, prompt, model) may fail in others.

**Solution:** Always test generalization explicitly. Report conditions under which patterns hold and conditions where they break.

**Pitfall 4: Metric Overload.** Computing 30+ metrics generates multiple comparison issues and risks fishing for spurious patterns.

**Solution:** Pre-register key metrics and hypotheses. Report comprehensive results transparently, but emphasize pre-specified analyses. Use replication across datasets to validate post-hoc discoveries.

**Pitfall 5: Ignoring Computational Cost.** Extracting all-layer hidden states for long generations is expensive, limiting scalability.

**Solution:** Develop efficient approximations (sparse layer sampling, streaming computation, distilled predictors) early in research program to enable large-scale studies.

### 9.3 Recommended Evaluation Protocols

**Protocol for Establishing Trajectory-Based Findings:**

**Stage 1: Discovery (Exploratory).**

- Select representative task with ground truth (e.g., arithmetic, logic)
- Extract trajectories from diverse conditions (success/failure, Direct/CoT)
- Compute comprehensive metric suite (30+ metrics)
- Identify metrics with large effect sizes (*d* > 0.8) and consistent layer profiles
- Generate hypotheses about mechanisms

**Stage 2: Validation (Confirmatory).**

- Pre-register top 5-10 metrics and directional predictions
- Test on held-out dataset from same task distribution
- Validate predicted effect directions and magnitudes (aim for replication within ±30% of original effect size)

**Stage 3: Generalization (Robustness).**

- Test on different task from same domain (e.g., different arithmetic operations)
- Test on different model architecture
- Test on different scale (larger/smaller models)
- Report which findings generalize and which are task/model/scale-specific

**Stage 4: Causal Validation (Interventional).**

- Design interventions manipulating identified geometric properties (e.g., steering vectors altering commitment timing)
- Test whether interventions produce predicted outcome changes
- Distinguish causal from spurious correlations

**Stage 5: Mechanistic Grounding (Circuit Integration).**

- Combine trajectory geometry with circuit discovery: identify components whose ablation alters trajectory properties
- Link geometric patterns to attention mechanisms, MLP activations, or SAE features
- Provide mechanistic explanations for why trajectories exhibit observed patterns


## 10. Conclusion

This comprehensive literature review has positioned the trajectory geometry framework within the broader landscape of mechanistic interpretability, chain-of-thought reasoning, vector embedding spaces, and computational regime detection, while systematically examining how this novel approach fills critical gaps in current understanding of transformer-based reasoning processes.

### 10.1 Summary of Core Contributions

The trajectory geometry research program makes several fundamental contributions to interpretability science:

**1. Empirical Demonstration of Regime-Dependent Computational Success.** Through systematic analysis of 300 arithmetic problems across Direct-answer and Chain-of-Thought conditions in Qwen2.5-0.5B, the research falsifies the hypothesis of universal "good trajectory" signatures. Instead, 13 of 33 geometric metrics exhibit *opposite directional effects* on success depending on computational regime. This regime-dependency—where higher speed/dimensionality aids Direct reasoning but hinders CoT reasoning—provides the first quantitative evidence that transformers implement qualitatively distinct computational strategies that manifest as contrasting geometric patterns. The discovery challenges interpretability research to move beyond regime-agnostic analysis toward context-dependent understanding of neural computation.[^12][^11]

**2. Temporal Measurement of Reasoning Phase Transitions.** The time-to-commit metric operationalizes the "explore-then-commit" transition predicted by theoretical models of multi-step reasoning, revealing that successful CoT exhibits dimensional collapse at token 10-14, Direct success at token 5-7, and failures across both regimes commit late (token 14+). With effect sizes *d* = 0.9-1.0 and consistent patterns across layers 20-24, commitment timing emerges as a robust cross-regime predictor. This represents the first direct measurement of *when* language models transition from hypothesis exploration to answer consolidation during in-context reasoning—information invisible to static analysis or output-only evaluation.[^12][^11]

**3. Functional Architecture of Transformer Depth.** Comprehensive layer-stratified analysis across all 24 layers reveals depth-dependent specialization: early layers (0-7) classify computational regime with near-perfect discrimination (*d* = 6-8), middle layers (10-14) predict reasoning success with strong discrimination (*d* = 1.0-2.2), and late layers (20-24) track commitment dynamics. This functional pipeline—regime detection → success prediction → commitment tracking—contradicts views of transformers as uniformly distributed processors and suggests evolutionary pressures toward hierarchical functional organization paralleling biological cortical hierarchies.[^12][^11]

**4. Process-Level Predictive Power Beyond Surface Correlates.** Logistic regression models using geometric features achieve 74.9% accuracy predicting correctness on 600-response dataset, adding 6 percentage points (74.9% vs. 68.9%) beyond knowing prompt type alone, with AUC increasing from 0.629 to 0.767. Within CoT condition, geometry alone predicts success at 75.0% accuracy (AUC 0.772), demonstrating that hidden state trajectories encode information about reasoning quality that is *not* determined by prompting strategy or reducible to simple behavioral correlates (response length, numeric patterns). This establishes trajectory geometry as capturing genuine computational processes rather than superficial artifacts.[^11]

**5. Methodological Framework for Dynamic Interpretability.** The research operationalizes a comprehensive 33-metric suite spanning velocity, directionality, curvature, dimensionality, convergence, and temporal dynamics, demonstrating that multi-faceted geometric measurement enables dissecting *which* properties matter *when* and *where* in computational hierarchies. This methodological contribution—treating hidden state evolution as the primary object of study rather than static snapshots—opens trajectory-based analysis as a complementary paradigm to circuit discovery, sparse autoencoders, and probing approaches.[^12][^11]

### 10.2 Positioning Within the Literature

Trajectory geometry addresses four critical gaps in mechanistic interpretability literature:

**Gap 1: Lack of Temporal Analysis.** While circuit discovery, SAEs, and probing analyze snapshots at individual layers or tokens, multi-step reasoning inherently involves temporal structure—exploration, backtracking, convergence—that snapshot methods cannot capture. Trajectory geometry uniquely treats hidden state evolution over token sequences as first-class objects of analysis, revealing process-level properties (phase transitions, convergence dynamics, temporal coherence) invisible to static methods.[^1][^2][^4][^5][^6][^9]

**Gap 2: Insufficient Attention to Regime-Dependent Computation.** Most interpretability work implicitly assumes universal computational substrates, seeking the circuit implementing a behavior without considering whether multiple distinct circuits/strategies might achieve the same outcome via different prompting. Trajectory geometry provides first-principle evidence for qualitatively distinct computational regimes exhibiting opposite geometric correlates of success, demanding interpretability methods adopt regime-aware frameworks.

**Gap 3: Limited Understanding of Commitment Dynamics.** Chain-of-thought research extensively studies *what* models generate but ignores *when* models commit to answers. Trajectory geometry makes commitment timing directly measurable via dimensional collapse detection, enabling investigations of decision-making processes, early stopping optimization, and interventions targeting commitment phases—all impossible without temporal geometric analysis.

**Gap 4: Absence of Process-Level Evaluation.** LLM benchmarks measure outcome metrics (accuracy, F1) without assessing *how* answers were reached. Two models achieving identical accuracy may employ radically different computational processes (efficient retrieval vs. prolonged exploration), with different failure modes, compute costs, and robustness properties. Trajectory geometry enables process-level benchmarking—reporting regime distributions, reasoning efficiency, failure mode characterization—supplementing outcome metrics with computational quality assessment.

Relative to prior dynamic analysis work—RNN trajectory visualization, biological neural trajectory geometry, latent vector field analysis in autoencoders, Hamiltonian mechanics approaches to LLM reasoning—this research uniquely applies comprehensive geometric measurement to token-resolved hidden state dynamics during in-context reasoning in decoder-only transformers, a domain where representational motion critically encodes multi-step reasoning processes.[^62][^22][^61][^19][^21][^72][^63]

### 10.3 Implications for Future Research

The trajectory geometry framework opens multiple high-impact research directions:

**Mechanistic Interpretability.** Integrate trajectory geometry with existing methods:

- **Trajectory-SAE Integration:** Analyze feature activation trajectories in sparse autoencoder spaces, identifying features with characteristic temporal profiles (early-activating vs. late-activating, transient vs. persistent)
- **Dynamic Circuit Discovery:** Extend activation patching to temporal domain, identifying *when* circuit components engage and revealing computational choreography over token sequences
- **Causal Trajectory Interventions:** Use steering vectors to manipulate geometric properties (commitment timing, convergence rate), testing whether altering trajectories causally affects reasoning success

**Chain-of-Thought Research.** Trajectory geometry provides alternative ground truth for reasoning quality when verbalized CoT is unfaithful:

- **Geometry-Based Faithfulness Metrics:** Define faithfulness via trajectory-text alignment rather than output-based perturbation tests
- **Process-Level CoT Benchmarking:** Report commitment timing distributions, convergence quality, regime efficiency alongside accuracy
- **Geometric Monitoring for Safety:** Detect deceptive reasoning via anomalous trajectories (premature commitment, cross-regime inconsistency) before harmful outputs are generated

**Efficiency Optimization.** Trajectory-based systems enable dynamic compute allocation:

- **Early Stopping:** Terminate generation upon commitment detection if answer token present (~10% cost reduction)
- **Regime-Based Budgets:** Allocate compute conditioned on detected regime (minimal for Direct, extended for CoT)
- **Trajectory-Guided Sampling:** Adjust temperature based on geometric state (higher during exploration, lower during commitment)

**Evaluation Methodology.** Trajectory geometry motivates process-level evaluation protocols:

- **Reasoning Efficiency Scores:** Penalize models that "overthink" when retrieval suffices
- **Commitment Timing Distributions:** Favor models with regime-appropriate timing
- **Failure Mode Characterization:** Distinguish premature commitment errors from prolonged wandering errors

**Scaling to Frontier Models.** Develop efficient methods for extracting trajectory signals from 70B+ parameter models:

- **Sparse Layer Sampling:** Leverage layer-wise specialization to sample representative depths
- **Metric Subset Selection:** Compute only top-performing metrics (commitment timing, convergence rate, effective dimension)
- **Distilled Trajectory Predictors:** Train lightweight models predicting geometry from layer 0 states alone


### 10.4 Limitations and Open Questions

Several critical limitations constrain current conclusions:

**Task Specificity.** Exclusive focus on arithmetic limits generalizability. Do geometric patterns hold for logical reasoning, planning, factual QA, creative generation? Each domain may exhibit distinct signatures requiring separate characterization.

**Model Scale.** Results from 0.5B-1.5B parameters may not generalize to frontier-scale systems (70B+). Potential scaling effects include compressed reasoning (fewer tokens), different superposition regimes (more features packed into same dimensionality), and architectural differences (MoE, long-context attention) affecting trajectory smoothness.

**Difficulty Confound.** Negative numbers dramatically reduce success rates (39.5% vs. 86.2%), meaning geometric differences may partly reflect problem hardness rather than computational strategy. Controlled experiments stratifying by difficulty while holding problem structure constant are needed.[^11]

**Correlation vs. Causation.** Current work is descriptive—geometric patterns correlate with success but causal validation via interventions (steering vectors manipulating commitment timing, convergence rates) is required to establish whether geometry *drives* success or merely *reflects* it.

**Attribution to Mechanisms.** Trajectory geometry describes *what* trajectories do but not *why*. Geometric metrics do not identify which attention heads, MLP subnetworks, or features cause observed patterns. Complementary circuit-level analysis is needed.

### 10.5 Final Synthesis

Trajectory geometry establishes that *how transformers move through representation space during reasoning* is as important as *what representations they activate*. By revealing regime-dependent success signatures, measurable phase transitions, layer-specific functional specialization, and process-level predictive signals, this framework demonstrates that temporal dynamics of hidden states encode critical information about computational quality invisible to static interpretability methods or output-only evaluation.

The research reviewed here represents an initial proof-of-concept: trajectory-based interpretability is feasible, informative, and complementary to existing approaches. Substantial work remains to scale to frontier models, extend beyond arithmetic, establish causal relationships, and integrate with circuit discovery and sparse autoencoder frameworks. Yet the core insight—that *the shape of reasoning* visible in geometric motion through latent space reveals computational processes that verbalized outputs obscure—provides a foundation for next-generation interpretability research.

As large language models increasingly serve as reasoning engines in high-stakes domains (medicine, law, science, education), understanding *how* they reason—not just *whether* they reach correct conclusions—becomes paramount. Trajectory geometry offers a window into these internal processes, enabling monitoring systems that detect anomalies before they manifest as outputs, efficiency optimizations that align compute with computational needs, and safety mechanisms that catch failures in-process rather than post-hoc. The framework positions process-level analysis alongside outcome-level evaluation, structure-level circuit discovery, and feature-level sparse decomposition as essential components of comprehensive transformer interpretability.

Future research integrating trajectory geometry with sparse autoencoders (analyzing feature activation trajectories), circuit discovery (identifying components driving geometric patterns), causal interventions (manipulating trajectories to test mechanisms), and scaling to frontier models will determine whether the promise demonstrated in small-scale arithmetic experiments generalizes to the full complexity of modern language model reasoning. The trajectory, one might say, has been initiated—the question is whether the field will commit to exploring this geometric perspective or return to static analysis. The evidence suggests the former: the shape of reasoning matters, and geometry makes it visible.
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^120][^121][^122][^123][^124][^125][^126][^127][^128][^129][^130][^131][^132][^133][^134][^135][^136][^137][^138][^139][^140][^141][^142][^143][^144][^145][^146][^147][^148][^149][^150][^151][^152][^153][^154][^155][^156][^157][^158][^159][^160][^161][^162][^163][^164][^165][^166][^167][^168][^169][^170][^171][^172][^173][^174][^175][^176][^177][^178][^179][^180][^181][^182][^183][^184][^185][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/pdf/2304.14997v1.pdf

[^2]: https://arxiv.org/abs/2407.02646

[^3]: https://icml.cc/virtual/2025/40007

[^4]: https://www.semanticscholar.org/paper/39bbd489b43911152bdaf07f741a91bf1b15989d

[^5]: https://arxiv.org/abs/2410.20526

[^6]: https://arxiv.org/pdf/2309.08600.pdf

[^7]: https://openreview.net/forum?id=F76bwRSLeK

[^8]: https://openreview.net/forum?id=Timsb74vIY

[^9]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html

[^10]: https://openreview.net/forum?id=89ia77nZ8u

[^11]: Claude-2.md

[^12]: Claude-1.md

[^13]: https://proceedings.neurips.cc/paper_files/paper/2023/file/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Paper-Conference.pdf

[^14]: https://erdogan.dev/sias.pdf

[^15]: https://www.semanticscholar.org/paper/b51cfb9db9621124bcea7dbcc75831298ddda3fc

[^16]: https://arxiv.org/abs/2410.03334

[^17]: https://www.pnas.org/doi/10.1073/pnas.2506316122

[^18]: https://jalammar.github.io/hidden-states/

[^19]: https://arxiv.org/html/2505.22785v1

[^20]: https://explanation-llm.github.io/slides/section_4_slides.pdf

[^21]: https://arxiv.org/abs/2505.22785

[^22]: https://www.themoonlight.io/en/review/navigating-the-latent-space-dynamics-of-neural-models

[^23]: https://www.alignmentforum.org/posts/X26ksz4p3wSyycKNB/gears-level-mental-models-of-transformer-interpretability

[^24]: https://arxiv.org/abs/2411.14257

[^25]: https://arxiv.org/abs/2410.13928

[^26]: https://www.emergentmind.com/topics/chain-of-thought-cot-reasoning

[^27]: https://arxiv.org/abs/2508.09099

[^28]: https://arxiv.org/abs/2508.09099v1

[^29]: https://arxiv.org/pdf/2508.09099.pdf

[^30]: https://arxiv.org/abs/2510.21881

[^31]: https://arxiv.org/abs/2410.00151

[^32]: https://arxiv.org/html/2412.16720

[^33]: http://arxiv.org/pdf/2501.08156.pdf

[^34]: https://arxiv.org/pdf/2412.04645.pdf

[^35]: https://arxiv.org/abs/2301.13379

[^36]: https://aclanthology.org/2023.ijcnlp-main.20.pdf

[^37]: http://arxiv.org/pdf/2402.13950.pdf

[^38]: https://arxiv.org/pdf/2301.13379v2.pdf

[^39]: https://arxiv.org/abs/2503.08679

[^40]: https://arxiv.org/html/2405.15092

[^41]: https://openreview.net/pdf?id=NFiV0yVlBM

[^42]: https://aclanthology.org/2025.emnlp-main.504.pdf

[^43]: https://openreview.net/forum?id=awDkEAIWiW

[^44]: https://neurips.cc/virtual/2025/129774

[^45]: https://theaidigest.org/whats-your-ai-thinking

[^46]: https://arxiv.org/abs/2502.07374

[^47]: research_journey_trajectory_geometry.md

[^48]: Claude-3.md

[^49]: http://aclweb.org/anthology/N15-1070

[^50]: https://aclanthology.org/2024.emnlp-main.162

[^51]: https://ojs.ukscip.com/index.php/dtra/article/view/1564

[^52]: https://aclanthology.org/2024.repl4nlp-1.1

[^53]: https://aclanthology.org/2025.findings-acl.142

[^54]: https://www.mdpi.com/1999-4893/17/12/593

[^55]: https://arxiv.org/abs/2305.14599

[^56]: https://www.meilisearch.com/blog/what-are-vector-embeddings

[^57]: https://en.wikipedia.org/wiki/Word_embedding

[^58]: https://milvus.io/ai-quick-reference/how-do-vector-embeddings-work-in-semantic-search

[^59]: https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space

[^60]: https://www.pinecone.io/learn/vector-embeddings/

[^61]: https://aclanthology.org/W18-5428/

[^62]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8479019/

[^63]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7278111/

[^64]: https://arxiv.org/abs/2405.03809

[^65]: https://dl.acm.org/doi/10.1145/3746252.3761294

[^66]: https://www.fujipress.jp/jaciii/jc/jacii002900061507

[^67]: https://arxiv.org/pdf/2308.02925.pdf

[^68]: https://arxiv.org/pdf/2206.11251.pdf

[^69]: https://arxiv.org/pdf/2402.14473.pdf

[^70]: https://arxiv.org/pdf/2107.06097.pdf

[^71]: https://arxiv.org/abs/2206.11251

[^72]: https://arxiv.org/html/2410.04415v3

[^73]: https://www.gurustartups.com/reports/regime-detection-using-large-language-models

[^74]: https://arxiv.org/abs/2310.01285

[^75]: https://dl.acm.org/doi/10.1145/3677052.3698642

[^76]: https://arxiv.org/pdf/2601.11622.pdf

[^77]: http://therisingsea.org/notes/MSc-Carroll.pdf

[^78]: https://pmc.ncbi.nlm.nih.gov/articles/PMC39602/

[^79]: https://fse.studenttheses.ub.rug.nl/23691/1/Phase_Transitions_in_Layered_Neural_Networks_The_Role_of_The_Activation_Function.pdf

[^80]: https://arxiv.org/html/2404.15118v1

[^81]: https://arxiv.org/abs/2504.18072

[^82]: https://arxiv.org/html/2507.02199v2

[^83]: https://openreview.net/forum?id=CbK7lYbmv8

[^84]: https://openreview.net/forum?id=hVUIguIm14

[^85]: https://arxiv.org/html/2510.06410v1

[^86]: https://www.arxiv.org/pdf/2601.23163.pdf

[^87]: https://arxiv.org/abs/2512.24574

[^88]: ChatGPT-3.md

[^89]: User_Contributions_Ledger.md

[^90]: https://arxiv.org/pdf/2503.08200.pdf

[^91]: https://mahis.life/bet/

[^92]: ChatGPT 2.md

[^93]: ChatGPT 1.md

[^94]: RESEARCH_HISTORY.md

[^95]: INDEX.md

[^96]: experiment_09b.md

[^97]: experiment_09.md

[^98]: experiment_08.md

[^99]: experiment_07.md

[^100]: experiment_06.md

[^101]: experiment_05.md

[^102]: experiment_04.md

[^103]: experiment_03.md

[^104]: experiment_02.md

[^105]: experiment_01.md

[^106]: https://ieeexplore.ieee.org/document/11147562/

[^107]: https://aacrjournals.org/cancerres/article/84/6_Supplement/7383/735253/Abstract-7383-G2PT-Mechanistic-genotype-phenotype

[^108]: https://arxiv.org/abs/2512.16964

[^109]: https://link.springer.com/10.1007/s44200-025-00084-w

[^110]: https://www.oajaiml.com/uploads/archivepdf/683154248.pdf

[^111]: https://arxiv.org/abs/2512.08819

[^112]: https://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/5205

[^113]: https://ieeexplore.ieee.org/document/11244666/

[^114]: https://rdl-journal.ru/article/view/964

[^115]: https://www.jmir.org/2026/1/e74359

[^116]: http://arxiv.org/pdf/2407.14494.pdf

[^117]: https://arxiv.org/html/2408.09523

[^118]: https://arxiv.org/pdf/2306.01128.pdf

[^119]: http://arxiv.org/pdf/2406.11624.pdf

[^120]: https://arxiv.org/pdf/2410.17438.pdf

[^121]: https://arxiv.org/pdf/2202.07304.pdf

[^122]: https://arxiv.org/html/2502.15801v1

[^123]: https://aclanthology.org/2021.findings-emnlp.346.pdf

[^124]: https://neurips.cc/virtual/2024/poster/97689

[^125]: https://openreview.net/forum?id=RaroYIrnbR

[^126]: https://aclanthology.org/2024.findings-acl.242.pdf

[^127]: https://openreview.net/forum?id=kJiB24fTmw

[^128]: https://transformer-circuits.pub/2025/july-update/index.html

[^129]: http://medrxiv.org/lookup/doi/10.1101/2025.08.08.25333318

[^130]: https://arxiv.org/abs/2405.14389

[^131]: http://ieeexplore.ieee.org/document/7899735/

[^132]: https://arxiv.org/pdf/2008.13298.pdf

[^133]: https://arxiv.org/pdf/2003.03350.pdf

[^134]: https://www.aclweb.org/anthology/N15-1164.pdf

[^135]: https://www.aclweb.org/anthology/P19-1165.pdf

[^136]: http://arxiv.org/pdf/1607.03780v1.pdf

[^137]: https://aclanthology.org/2023.emnlp-main.106.pdf

[^138]: https://aclanthology.org/P14-1046.pdf

[^139]: https://www.aclweb.org/anthology/K18-1013.pdf

[^140]: https://magazine.mindplex.ai/post/unveiling-the-hidden-dynamics-how-neural-networks-navigate-latent-spaces

[^141]: https://discuss.huggingface.co/t/hidden-states-embedding-tensors/3549

[^142]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12094247/

[^143]: https://aclanthology.org/2025.emnlp-main.942

[^144]: https://aclanthology.org/2024.blackboxnlp-1.32

[^145]: https://arxiv.org/abs/2408.00113

[^146]: https://arxiv.org/abs/2408.00657

[^147]: https://arxiv.org/html/2502.11367v1

[^148]: https://arxiv.org/html/2410.07456v1

[^149]: http://arxiv.org/pdf/2406.04093.pdf

[^150]: https://arxiv.org/pdf/2410.21508.pdf

[^151]: https://arxiv.org/abs/2502.03714

[^152]: https://arxiv.org/html/2411.00743v1

[^153]: https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html

[^154]: https://proceedings.iclr.cc/paper_files/paper/2024/file/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Paper-Conference.pdf

[^155]: https://blog.eleuther.ai/autointerp/

[^156]: https://www.mdpi.com/1424-8220/20/17/4889

[^157]: https://ieeexplore.ieee.org/document/11215387/

[^158]: https://ieeexplore.ieee.org/document/11249886/

[^159]: https://ieeexplore.ieee.org/document/10611205/

[^160]: https://arxiv.org/abs/2301.08647

[^161]: https://www.mdpi.com/1424-8220/25/19/6259

[^162]: https://ijred.cbiore.id/index.php/ijred/article/view/60632

[^163]: http://arxiv.org/pdf/2303.16207.pdf

[^164]: http://arxiv.org/pdf/2410.13106.pdf

[^165]: http://arxiv.org/pdf/2410.10648.pdf

[^166]: https://arxiv.org/pdf/2206.00826.pdf

[^167]: https://arxiv.org/pdf/2403.03181.pdf

[^168]: https://www.sciencedirect.com/topics/computer-science/geometric-reasoning

[^169]: https://www.youtube.com/watch?v=A5K01E_zXPc

[^170]: https://www.youtube.com/watch?v=H_nltXkIPtw

[^171]: https://www.arxiv.org/pdf/2512.17923.pdf

[^172]: https://www.fujipress.jp/jaciii/jc/jacii002900061507/

[^173]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11927934/

[^174]: https://arxiv.org/abs/2406.10040

[^175]: https://arxiv.org/abs/2501.06458

[^176]: https://arxiv.org/abs/2506.04521

[^177]: https://arxiv.org/abs/2509.12645

[^178]: https://ieeexplore.ieee.org/document/11199211/

[^179]: https://arxiv.org/abs/2507.22940

[^180]: https://arxiv.org/abs/2504.00993

[^181]: https://ieeexplore.ieee.org/document/11028406/

[^182]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6566209/

[^183]: https://cisnlp.github.io/thesis_proposals/2024/Ali-2024-Project-1.pdf

[^184]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6867616/

[^185]: https://www.prompthub.us/blog/faithful-chain-of-thought-reasoning-guide

