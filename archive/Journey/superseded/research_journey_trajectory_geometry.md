# The Shape of Reasoning: A Research Journey from Perspective-Dependent Meaning to Trajectory Geometry

---

## Part I: The Seed — What If Semantic Distance Isn't Fixed?

### The Observation That Started Everything

Consider two concepts: **Biology** and **Computer Science**. How far apart are they?

If you're a university administrator organising departments, they're distant — different buildings, different methodologies, entirely separate domains. But if you're a computational biologist, these concepts are neighbours. Both deal with information storage, error correction, parallel processing, and signal transmission. The distance between them hasn't changed because the concepts moved. It changed because *you* adopted a different perspective.

This observation — that the distance between any two ideas depends on where you're standing when you measure it — became the foundation of everything that followed.

Traditional AI systems treat semantic distance as a fixed property: the relationship between "dog" and "cat" has one number, computed once, stored forever. Word embeddings, sentence transformers, the entire architecture of modern NLP rests on this assumption. But human cognition doesn't work this way. We experience meaning as fluid, context-dependent, and shaped by the stance we bring to it. A comedian and a physicist look at the same pair of concepts and perceive entirely different relationships — not because one is wrong, but because the geometry of meaning shifts with the observer.

The question became: could this be formalised mathematically? And if so, what would it unlock?

---

### Perspective-Dependent Semantic Distance

The core proposal was deceptively simple: **semantic distance is a function of the observer's cognitive stance**. The geometry of conceptual space is not fixed; it is shaped by which dimensions of meaning are foregrounded, which are suppressed, and how they are weighted.

This was formalised through a perspective-dependent metric tensor:

$$g^{(p)} = P_p + \lambda R_p$$

Where $P_p$ is a projection operator representing the observer's perspective, $R_p$ captures the residual (background) structure, and $\lambda$ controls the balance between focused and diffuse attention.

Under this framework, adopting a perspective is mathematically equivalent to projecting concepts onto a subspace defined by that perspective. When we apply a projection operator $P_p$ to a concept vector $v$:

$$v' = P_p v = \frac{\langle v, v_p \rangle}{\|v_p\|^2} v_p$$

This retains only the component aligned with the perspective direction, discarding everything else. The distance between any two concepts becomes:

$$d_p(u, v) = \sqrt{(u - v)^T g^{(p)} (u - v)}$$

The same pair of concepts yields different distances under different perspectives — not as noise, but as a fundamental feature of how meaning works.

Two geometric operations were initially distinguished:

- **Intelligence**: Navigating efficiently *within* a fixed semantic geometry — finding the shortest path, retrieving the relevant connection, optimising under existing structure.
- **Creativity**: Transforming the geometry itself — adopting a perspective that makes previously distant concepts suddenly adjacent, or revealing structure that was invisible from any prior vantage point.

A critical test was proposed: the **Hysteresis Criterion**. If two concepts are brought close together by a particular perspective, and they *remain* close even after that perspective is removed, then a genuine creative transformation has occurred — the manifold itself has been permanently restructured. If the proximity vanishes when the perspective shifts, it was navigation, not creation.

---

## Part II: From Filters to Positions — The Centroid Model

### The Limitation of Projection

The projection framework was elegant, but it had a fundamental problem: projection *destroys* information. When you project onto a subspace, the rejected dimensions are gone. This doesn't match the phenomenology of insight — the feeling of "it was there all along, I just couldn't see it." If projection truly eliminates dimensions, there's nothing left to discover.

This drove a major reconceptualisation. The question shifted from "what filter are you applying?" to "where are you standing?"

### Perspective as Position

The revised framework proposed that perspective is fundamentally **positional**: the observer occupies a location within the representational manifold, and meaning emerges from the angular and metric relationships visible from that position.

Several new constructs formalised this intuition:

**The Mobile Centroid** ($C$): Rather than a filter applied to static concepts, the observer has a coordinate position in the full manifold. This position evolves over time — through learning, through conversation, through deliberate reframing. A physicist's centroid sits deep in the physics region of the manifold. A poet's centroid occupies a different region entirely. The same concepts look different from each position, not because of filtering, but because of geometry.

**The Visibility Envelope** ($\mathcal{V}(C)$): From any position, the observer has an anisotropic region of accessibility — a shape that extends further in some dimensions than others. A physicist's visibility extends far along physics dimensions but barely reaches Renaissance poetry. This isn't a binary in/out — it's a continuous gradient of accessibility, with the shape determined by expertise, attention, and current cognitive state.

**Semantic Parallax**: Just as viewing a nearby object from different angles produces parallax — apparent displacement relative to the background — viewing concepts from different positions in the manifold produces apparent shifts in their relationships. Two concepts that appear identical from one position may appear utterly distinct from another. This isn't error. It's a geometric consequence of the observer's position.

**Dimensional Occlusion**: Objects exist in their full dimensionality regardless of who's looking. But from any given position, only certain dimensions are visible. A concept like "democracy" has political dimensions, historical dimensions, philosophical dimensions, emotional dimensions. From a political scientist's position, the political dimensions dominate. From a philosopher's position, entirely different structure becomes visible. The other dimensions aren't gone — they're occluded by the observer's position.

This explained the phenomenology of insight: "It was there all along, I just couldn't see it." The information was never destroyed. The observer simply moved to a position where previously occluded dimensions became visible.

### Recovering the Original Framework

A key theoretical result: the original projection model was recovered as a limiting case. When the visibility envelope is infinitely elongated in one direction and contracted in all others, the centroid model reduces to projection. The parameter $\lambda$ from the original metric tensor emerged naturally as the ratio of background to foreground visibility in the envelope structure — no longer a free parameter, but a derived quantity.

### Reclassifying Cognitive Operations

The centroid model demanded a new taxonomy. The original intelligence/creativity binary was too clean. Instead, cognitive operations were classified by their geometric effect:

- **Position Translation**: Moving the centroid to a new location in the manifold (empathy, perspective-taking)
- **Visibility Expansion**: Enlarging the envelope to encompass more of the space (learning, broadening)
- **Dimensional Revelation**: Making previously occluded dimensions visible (insight, reframing)
- **Manifold Extension**: Adding entirely new structure to the space (genuine novelty, creative production)
- **Topological Restructuring**: Permanently altering the connectivity of the manifold (paradigm shifts, deep learning)

Importantly, persistence became orthogonal to operator type — any of these operations could be transient or permanent, independent of their geometric character.

---

## Part III: The Taxonomy of Conversational Geometry

### Operators as Manifold Manipulations

If meaning has geometry, then conversations are trajectories through that geometry. And if cognitive operations transform the geometry, then specific conversational moves should correspond to specific geometric transformations.

This led to a taxonomy of five fundamental operator types, each characterised by their effect on the metric tensor:

**Type I — The Lens (Projection)**: "Focus on...", "In the context of...", "Assuming X..." The operator projects the manifold onto a lower-dimensional subspace. Dimensionality decreases. Noise is reduced. Specificity increases. Saying "analyse this code from a security perspective" projects the representation onto the vulnerability axis, suppressing dimensions related to elegance, efficiency, or readability.

**Type II — The Bridge (Metric Contraction)**: "How is X like Y?", "Make an analogy...", "Synthesise these ideas." The operator folds the manifold so that distance between two previously disparate clusters collapses. DNA and computer code become neighbours. Gravity and trampolines share a subspace. The metric tensor is locally deformed to bring distant regions into contact.

**Type III — The Wedge (Orthogonalisation)**: "Don't confuse X with Y", "What is the difference?", "Critique this." The operator forces cosine similarity between target vectors toward zero, stripping shared components. "Intelligence" and "consciousness" get separated, their shared "cognition" variance removed to isolate what's unique to each.

**Type IV — The Prism (Decomposition)**: "Break this down", "Step by step", "What are the components?" The operator takes a dense, compressed representation and rotates the basis to reveal constituent features. "Democracy" decomposes into voting, rights, equality, representation. Effective dimensionality increases. Previously dormant sparse features activate.

**Type V — The Anchor (Hysteresis Induction)**: "Remember this definition...", "From now on...", "Adopt this persona." The operator attempts to make a temporary geometric shift permanent — invariant to future perspective changes. This is topological remodelling: not just moving through the space, but restructuring it.

### Conversational Chains

Just as chess has openings, conversational geometry has chains — recurring sequences of operators that achieve specific cognitive goals:

The **Insight Chain** (Prism → Rotation → Bridge → Anchor): Break the concept apart, rotate to a familiar domain, bridge the gap, lock the connection. "What are the components of quantum entanglement? Think of it like two coins always landing on opposite sides. See how information is correlated, not transmitted? Remember: entanglement is correlation, not communication."

The **De-escalation Chain** (Lens → Wedge → Projection): Objectify the emotion ("I notice you're frustrated"), separate intent from harm, realign on shared values.

The **Socratic Chain** (Lens → Perturbation → Wait → Bridge): Identify the assumption, introduce a minor contradiction, allow self-correction, bridge only if self-correction fails.

These weren't just theoretical constructs — they were falsifiable predictions about the geometric signatures that should appear in model internals during different conversational moves.

---

## Part IV: The Leap to Empirical Work

### The Resolution Hypothesis

The theoretical framework made a specific, testable prediction: if conversational operators correspond to real geometric transformations in representational space, their signatures should be detectable in the hidden states of language models. But the resolution of measurement matters. Turn-level embeddings compress token-level dynamics. Embedding-level measurement compresses layer-level computation. The signal might be there but smeared.

This became the **Resolution Hypothesis**: operator signatures would sharpen dramatically with higher-resolution measurement — moving from turn-level embeddings to token-resolved hidden states across transformer layers.

### First Experiments: Operator Warp Vectors

The first experimental series tested whether conversational operators leave measurable traces in the behaviour of a language model. Using 500 operator turns (10 operators × 10 paraphrases × 5 diverse topics), warp vectors were computed as the difference between successive state embeddings.

Three findings emerged in productive tension:

**Operators exist as a structural phenomenon.** K-means clustering showed weak but consistent alignment with operator type (Adjusted Mutual Information ≈ 0.13-0.14), while topic explained essentially zero variance (AMI ≈ 0). Something operator-like was there, and it wasn't reducible to content.

**Operator sequences are predictable.** A GRU sequence model achieved 66% accuracy on 10-way operator classification — far above the 10% chance baseline — with a macro-F1 of 0.56 indicating balanced performance across operator classes. The model generalised across topics: trained on four, tested on the held-out fifth.

**But operators aren't pointwise separable.** Within-operator variance exceeded between-operator variance. "Summarise the physics concept" and "Summarise the relationship advice" landed in different regions of embedding space because the semantic endpoints differed, even though both were performing the same operation.

This conjunction — **predictability without separability** — was the key finding. It was analogous to motion blur: any single frame of a dance is ambiguous, but the trajectory is unmistakable. The slow shutter speed of turn-level embeddings was smearing the operator signal across content displacement. The underlying structure was there, but it needed higher resolution to resolve.

---

## Part V: Inside the Machine — Hidden State Analysis with Google Antigravity

### Moving from Theory to Measurement

The theoretical framework had been developed in collaboration with AI systems. But testing it required running actual experiments on actual neural networks — extracting hidden states, computing metrics, analysing geometric structure. This is where Google's Antigravity became essential.

Antigravity provided the computational infrastructure to run the experiments that the theory demanded: extracting full-precision hidden states from transformer models, computing trajectory metrics across all layers, and performing the statistical analyses needed to evaluate geometric predictions.

The approach was deliberate: design experiments based on theoretical predictions, run them through Antigravity, and let the evidence guide understanding. Not seeking confirmation — seeking the shape of what's actually there.

### Experiment Series: Operator Signatures in Hidden States

The first Antigravity experiments examined Qwen2.5-0.5B, a 500M parameter transformer with 24 layers. Ten conversational operators were applied to fixed base content, with hidden states extracted at every layer and token position.

The **warp signal** — the Euclidean norm of token-to-token displacement in representation space — became the primary probe. It measures how much the model's internal state "moves" from one position to the next.

Three principal findings confirmed and extended the Resolution Hypothesis:

**Layer Localisation**: Operator signatures were cleanest in middle transformer layers (5-12), with reconstruction MSE of 2.8-4.4. At the final output layer, geometry catastrophically degraded (MSE > 1800), where representation was dominated by vocabulary projection. The "cognitive geometry" of operators lived in intermediate processing stages — not at the input, not at the output, but in the computational middle where abstract reasoning happens.

**Distinct Temporal Profiles**: Different operators exhibited characteristic velocity curves. Summarisation induced a sharp initial spike followed by stability — compression happening fast, then maintaining. Poetic transformation produced sustained oscillation — the model continuously exploring aesthetic alternatives. Critique showed gradual ramping — evaluation building incrementally.

**Linear Compositionality**: Composite operators (e.g., "summarise and criticise") could be modelled as linear superpositions of individual operator geometries, with the dominant trend captured at MSE ≈ 14.8 using four factors. The approximation had limits — composite reconstruction error was roughly 3× higher than single-operator reconstruction — but the basic principle held: operators combine additively in geometric space.

### The Pivot to Arithmetic Reasoning

With operator signatures confirmed in hidden states, the research pivoted to a more controlled domain: multi-step arithmetic. This offered something conversational operators couldn't — unambiguous ground truth. A model either gets $(38 × 9) + 46 = 388$ or it doesn't. This allowed trajectory geometry to be studied against objective correctness, not just operator classification.

300 multi-step arithmetic problems were run through Qwen2.5-0.5B under two conditions: Direct-answer prompting ("What is the answer?") and Chain-of-Thought prompting ("Think step by step"). This created four natural groups:

| Group | Condition | Outcome | n |
|-------|-----------|---------|---|
| G1 | Direct | Failure | 247 |
| G2 | Direct | Success | 53 |
| G3 | CoT | Failure | 77 |
| G4 | CoT | Success | 223 |

Hidden states were extracted across all 25 layers for the first 32 generated tokens of each response. What followed was a series of experiments — numbered 9 through 15 — each building on the last, each guided by what the evidence revealed rather than what the theory hoped for.

---

## Part VI: What the Trajectories Revealed

### Experiment 9-12: Initial Trajectory Metrics

The first pass computed approximately ten metrics per trajectory: speed, directional consistency, curvature, stabilisation rate, tortuosity, effective dimension, fractal dimension, and convergence diagnostics.

The effect sizes were enormous. Cohen's d of 4.25 for speed at Layer 24. Directional consistency effects of d ≈ −2.6 across all layers. Effective dimension d up to 5.66. These weren't marginal statistical effects — they were massive, consistent, layer-dependent geometric signatures that reliably distinguished successful reasoning from failure.

Four a priori predicted regimes were confirmed by clustering:

- **Retrieve-and-Commit** (G2: Direct Success): Nearly ballistic trajectories. High directional consistency, low effective dimension, rapid convergence. The model "knows" the answer and moves straight to it.
- **Explore-then-Commit** (G4: CoT Success): High-dimensional exploration followed by dimensional collapse. The model searches, expands into multiple possibilities, then converges.
- **Failed Exploration** (subset of G3: CoT Failure): Sustained high dimensionality throughout — the model keeps searching but never commits. No convergence phase.
- **Stable-but-Wrong** (subset of G3: CoT Failure): Clean-looking trajectories that converge confidently to the wrong answer. Geometrically the most dangerous — they *look* like success.

The retrieve-and-commit predictions were tested explicitly, with all four directional hypotheses confirmed: G2 showed higher directional consistency (d = 2.19), higher tortuosity (d = 1.92), lower effective dimension (d = −4.98), and faster convergence (d = −1.95) compared to G4. Two geometrically distinct computational regimes were both producing correct answers, activated by different prompting strategies.

### Experiment 13: Mining for Deeper Structure

Rather than generating new data, Experiment 13 applied additional analyses to the existing hidden states: failure subtyping via clustering, phase transition detection via sliding-window dimensionality tracking, and predictive modelling to determine how much geometry tells us beyond just knowing the prompt type.

The key result: trajectory geometry predicted correctness at 75-80% accuracy beyond what prompt type alone could explain. Convergence metrics were the strongest predictors. The geometric signal was real, not just a proxy for "CoT is better."

A process-level finding emerged: G4 (CoT Success) showed a dimension-drop signature — high effective dimension early, followed by collapse — in 77.6% of trajectories, versus only 58.4% for G3 (CoT Failure). The explore→commit phase transition wasn't just a theoretical construct. It was observable, measurable, and predictive.

### Experiment 14: The Paradigm Shift

Experiment 14 expanded the metric suite from approximately 10 to 33 metrics and computed them across all 25 layers of Qwen2.5-0.5B. The new metrics included dispersion and cloud geometry measures, mean squared displacement scaling (diffusion exponents), recurrence quantification, spectral analysis, and a novel metric: **time_to_commit** — the token position at which the radius of gyration drops most sharply, capturing when the model "decides."

The results revealed the study's most fundamental discovery.

**"Good geometry" is regime-dependent.** At layer 13, the majority of key metrics showed *opposite* direction effects when comparing what makes CoT succeed versus what makes Direct succeed:

| Metric | CoT Success | Direct Success |
|--------|-------------|----------------|
| Speed | Lower | Higher |
| Effective dimension | Lower | Higher |
| Radius of gyration | Lower | Higher |
| Cosine to running mean | Higher | Lower |
| Distance slope | Lower | Higher |

These weren't subtle differences — they were d > 0.7 effects pointing in opposite directions. A trajectory that looks "successful" under CoT criteria would look like failure under Direct criteria. There is no universal "good trajectory" detector. What constitutes good geometry is defined by the computational strategy being employed.

Of the 33 metrics, 13 proved to be universal success indicators (pointing in the same direction regardless of regime), while 17 were regime-specific (pointing in opposite directions). The universal indicators — including directional consistency, tortuosity, and several convergence metrics — capture something about general computational quality. The regime-specific indicators capture the fundamentally different computational strategies the model employs.

**Commitment timing became a primary signal.** The time_to_commit metric revealed a clean, interpretable pattern:

| Group | Mean Commitment Time | Interpretation |
|-------|---------------------|----------------|
| Direct Success (G2) | ~5.4 tokens | Immediate retrieval |
| CoT Success (G4) | ~10.7-13.8 tokens | Explore, then commit |
| Direct Failure (G1) | ~13.8 tokens | Couldn't retrieve, kept searching |
| CoT Failure (G3) | ~16.0 tokens | Searched but never found |

Failures commit late, in both regimes. Direct successes commit earliest. CoT successes commit in the middle — enough exploration to find the answer, then decisive convergence. This is direct evidence for the explore→commit phase transition, and it's readable in real-time from hidden state dynamics.

**Different layers serve different diagnostic purposes.** The evidence revealed a functional architecture within the transformer's processing:

- **Layers 0-7**: Regime detection. These early layers distinguish CoT-like from Direct-like computation with effect sizes of d = 6-8. The computational strategy is legible almost immediately.
- **Layers 10-14**: Success prediction. These middle layers carry the strongest within-regime success discrimination, with regime-specific metrics like cosine-to-running-mean and radius of gyration providing d = 1.0-2.2 effects.
- **Layers 20-24**: Commitment timing. The late layers track when and how the model commits to its answer direction.

### Experiment 15: Difficulty and Failure Subtyping

The final experiment in the current series stratified results by problem difficulty and characterised failure subtypes in detail.

A major confounder was discovered: problems with negative answers dramatically reduced CoT success rates (39.5% vs 86.2% for positive answers). This wasn't a property of geometry — it was a property of the task. Negative numbers appear to be genuinely harder for the model, and this difficulty confound was embedded throughout the dataset. Acknowledging this didn't invalidate the geometric findings, but it constrained their interpretation and demanded future experiments with controlled difficulty stratification.

---

## Part VII: The Architecture of Discovery

### What the Evidence Showed — and What It Challenged

The experimental journey was not a confirmation of the original theory. It was an evolution guided by evidence.

The theory predicted perspective-dependent geometry. The experiments found regime-dependent geometry. These aren't the same thing, but they're related — and the empirical version is arguably more useful. The theory imagined a smooth manifold of meaning where observers occupy positions. The experiments found discrete computational regimes where success looks geometrically opposite depending on which regime is active.

The theory predicted operators as smooth geometric transformations. The experiments found operators as distributed processes — not pointwise separable but trajectory-predictable, with signatures that localise to specific transformer layers and unfold with characteristic temporal profiles.

The theory predicted a single "good trajectory" signature that would emerge with sufficient measurement resolution. The experiments falsified this — demonstrating that good geometry is regime-relative, and that the same metric can predict success in one regime and failure in another.

In each case, the evidence was richer and more surprising than the prediction. The willingness to let data reshape understanding — rather than force-fitting results to theory — is what allowed these discoveries to emerge.

### The Cascade: Regime → Success → Commitment

The experiments revealed a three-stage monitoring architecture that could be built from these findings:

**Stage 1 — Regime Detection** (Layers 0-7, first ~8 tokens): Classify the computational strategy. Is this retrieval or reasoning? Effect sizes of d = 6-8 make this extremely reliable.

**Stage 2 — Regime-Specific Success Prediction** (Layers 10-14, first ~16 tokens): Apply the appropriate success criteria for the detected regime. For CoT: tighter trajectories and higher coherence predict success. For Direct: larger spread and faster motion predict success. Effect sizes d = 1.0-2.2.

**Stage 3 — Commitment Tracking** (Layers 20-24, throughout generation): Monitor when and how decisively the model commits. Late commitment signals likely failure. Appropriate commitment timing — early for retrieval, middle for reasoning — signals likely success. Effect sizes d = 0.9-1.0.

### Implications

If these findings generalise across architectures and tasks, they point toward several practical applications:

**Safety**: Hallucination detection through geometric monitoring — flagging trajectories that show "confident-but-wrong" signatures (clean convergence to incorrect answers). Deception detection through cross-layer consistency checking. Jailbreak detection through regime-mismatch identification.

**Efficiency**: Early termination when the model has committed with high confidence in the retrieve-and-commit regime. Adaptive compute allocation based on detected regime. Estimated ~10% inference cost reduction from early stopping in Direct-success cases alone.

**Capability**: Intervention based on commitment timing — detecting failed exploration early enough to redirect. Dynamic temperature adjustment based on geometric state. Regime-appropriate prompting strategies.

**Understanding**: Perhaps most fundamentally, trajectory geometry offers a window into *how* transformers compute, not just *what* they compute. The discovery that the same model uses geometrically opposite strategies depending on prompting suggests multiple computational modes with distinct signatures — a finding that deepens understanding of what these systems actually do when they "think."

---

## Part VIII: Honest Accounting

### What Was Contributed

The conceptual framework — perspective-dependent meaning, the centroid model, the operator taxonomy, the resolution hypothesis, the experimental designs — emerged from a psychology background applied to geometric intuitions about how meaning works. The experimental logic, the interpretation of results (including the uncomfortable ones), and the connections between domains (psychology, differential geometry, interpretability, dynamical systems) were the primary intellectual contributions.

### What Required Collaboration

The mathematical formalisation, the code implementation, the experimental execution, and the statistical computation were performed in collaboration with AI systems — primarily Claude for theoretical development and experimental design, and Google's Antigravity for running the actual experiments on Qwen2.5-0.5B.

### What Remains Uncertain

The findings are from a single model architecture (Qwen2.5-0.5B), a single task domain (arithmetic), and relatively modest sample sizes. The geometric signatures need cross-architecture replication — planned experiments on Phi-2 will test whether these patterns are universal properties of transformer computation or specific to one model family.

The theoretical framework began grander than the evidence currently supports. The experiments didn't validate the full centroid model or the complete operator taxonomy. What they validated was something more specific and arguably more valuable: that transformer hidden states carry rich geometric structure that is regime-dependent, layer-specific, temporally dynamic, and predictive of correctness.

The gap between theory and evidence is where the next research lives.

### What Comes Next

Cross-architecture replication is the immediate priority — testing whether the geometric signatures discovered in Qwen2.5-0.5B appear in fundamentally different model families. Beyond that: different task domains, longer sequences, creative and uncertain reasoning where ground truth is harder to define.

The long arc points from proof-of-concept through framework integration to practical monitoring systems. But each step must be earned through evidence, not assumed through theory.

The shape of reasoning is geometric. The geometry is regime-dependent. The regimes are readable. Everything else is still a question.
