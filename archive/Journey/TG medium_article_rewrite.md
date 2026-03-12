# What Does "Thinking" Look Like Inside a Language Model? A Psychologist Measured It.

## When a transformer reasons correctly, it literally moves differently through its own mind. And the shape of that movement tells a story that cognitive science has been telling for decades.

I started this project because I kept reading interpretability papers that described what models *know* but never what they *do*. As a psychologist, that gap bothered me in a way I could articulate but couldn't initially prove. In clinical assessment, we learned decades ago that a correct answer tells you almost nothing. What matters is the *process* that generated it. Two patients can both say "four" when you ask them what two plus two is, but the cognitive paths they took to get there can be radically different, and those paths are what tell you something diagnostically useful. I wanted to know if the same principle applied inside a neural network.

It does. And what I found was more interesting than I expected.

## The assumption worth questioning

There is a quiet assumption baked into most AI interpretability research: that understanding a model means understanding what it *represents*. Find the right neuron. Identify the right feature. Map the concept to a vector. It is an approach built on the idea that knowledge lives in fixed locations, that somewhere inside GPT or Claude or LLaMA, there is a "truth detector" or a "reasoning circuit" sitting at a specific address, waiting to be found.

This is not a strawman. Anthropic's [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) work extracted millions of interpretable features from Claude 3 Sonnet using sparse autoencoders, including features for concepts as specific as the Golden Gate Bridge. It is genuinely impressive work and represents the state of the art in what you might call *structural* interpretability, asking what a model has encoded in its weights. The [mechanistic interpretability](https://www.neelnanda.io/mechanistic-interpretability/glossary) community has built powerful tools for answering this question.

But I keep returning to what I think of as the Dictionary Trap. Having a million interpretable features is like having a dictionary with a million entries. You know what the words mean individually. But a dictionary does not tell you how sentences work. It does not tell you why one sequence of words produces a poem and another produces nonsense. The dynamic relationships between features, the way representations evolve through time and across layers during actual computation, that is where the action is. And almost nobody is measuring it.

I spent six months trying to find fixed addresses of competence inside a small transformer model. I failed. What I found instead was that the model's competence is not a *place*. It is a *path*. The difference between a correct answer and a hallucination is not about *where* the model's internal state ends up. It is about *how it moves to get there*.

I call this approach **Trajectory Geometry**: measuring the shape, speed, and texture of a model's reasoning process as it unfolds across layers and tokens. And the results were not subtle.

## What I actually did

Before diving into findings, some context on the experimental design, because the specifics matter.

This work emerged from 16 experiments conducted over the course of a year, using small open-source transformer models (primarily [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), with replication on Qwen2.5-1.5B and Pythia-70m). The task domain is multi-step arithmetic, deliberately simple problems like `(47 × 3) + 12`, chosen specifically because they produce unambiguous binary outcomes. Either the model gets 153 or it does not. No judgment calls, no fuzzy evaluation rubrics. This is important because it eliminates the interpretive noise that plagues most reasoning benchmarks.

I tested two prompting strategies across 300 problems. **Direct prompting** asks the model to answer immediately: "What is (47 × 3) + 12? Answer with just the number." **Chain-of-Thought** ([Wei et al., 2022](https://arxiv.org/abs/2201.11903)) asks it to reason explicitly: "What is (47 × 3) + 12? Think step by step."

This creates four natural groups. G1 is Direct Fail: the model answers directly and gets it wrong. G2 is Direct Success: answers directly and gets it right. G3 is CoT Fail: thinks step by step but still gets it wrong. G4 is CoT Success: thinks step by step and gets it right.

Every comparison that follows is drawn from the model's *internal hidden states*, not its text output. I extracted residual stream activations across all 25 layers for the **full generation trajectory** of every response (not truncated to a fixed window), then computed 33 geometric and dynamical metrics on these trajectories. All reported effects are statistically significant (p < 0.001 via permutation testing, 10,000 shuffles, N = 300).

For those unfamiliar with the effect size metric I use throughout: Cohen's d measures the standardised difference between two groups. In psychology, d = 0.2 is "small," d = 0.5 is "medium," and d = 0.8 is considered "large." I am routinely measuring effects of d = 4.0 or higher. To be transparent about what that means: the distributions barely overlap. These are not subtle signals requiring sophisticated statistics to detect. They are visible to the naked eye.

## Finding 1: Failure is flat

The single most striking result across all 16 experiments is what I call **dimensional collapse**.

When the model reasons its way to a correct answer using Chain-of-Thought, its internal trajectory occupies a rich, high-dimensional subspace. I measured this using *effective dimension*, essentially how many independent directions of variation the model's hidden states explore as they process the problem. Think of it as a rough count of how many conceptual axes the model is actively using during computation.

When the model fails on a direct answer, that subspace collapses.

The numbers are not close. Successful CoT trajectories (G4) explore approximately 19 to 23 effective dimensions depending on the layer, while failed Direct trajectories (G1) are confined to roughly 2 to 4 dimensions. The effect size at Layer 14 is Cohen's d = 4.69.

**Figure 1** shows this clearly. The separation between the red violin (G1, Direct Fail) and the blue violin (G4, CoT Success) is near-total. But notice something else: the yellow violin (G3, CoT Fail) is actually *wider* than the blue one. I will come back to that, because it changes the story considerably.

![Figure 1: Dimensional Collapse at Layer 14](Trajectory%20Geometry/figures/FigA_TheCollapse_Publication.png)
*Violin plot showing the distribution of effective dimension for all four groups at Layer 14. The separation between Direct Fail (G1, red) and CoT Success (G4, blue) is extreme (d = 4.69). Note that CoT Fail (G3, yellow) actually shows wider expansion than CoT Success.*

What does this mean in plain language? Imagine navigating a building. A successful CoT reasoner is exploring the full three-dimensional layout, turning corners, going up and down stairs, covering real ground. A failed Direct model is walking along a single corridor. Same building, radically different experience of it. The model's computation during failure is genuinely confined to a narrow corridor of its representational capacity. It is pattern-matching, not computing.

Anyone who has worked with clinical populations will recognise something familiar here. Cognitive rigidity, the restriction of thought to narrow, well-worn pathways, is one of the most reliable markers of dysfunction across the entire clinical literature. Obsessive-compulsive presentations, certain forms of depression, traumatic stress responses: all of them share this signature of a mind that has collapsed its exploratory range. I am not making a clinical analogy to suggest that language models are experiencing psychological distress. I am pointing out that the geometric signature of reduced computational capacity looks structurally similar whether you are measuring it in a human brain or a transformer's hidden states.

This connects to a deeper principle in neuroscience. Recent work on [neural manifolds](https://pmc.ncbi.nlm.nih.gov/articles/PMC11058347/) has shown that biological neural populations represent information on low-dimensional surfaces embedded in high-dimensional state spaces, and that the *geometry* of these manifolds determines computational capacity. A 2025 study in [Science Advances](https://www.science.org/doi/10.1126/sciadv.adv0431) demonstrated that macaque V2 neurons solve classification problems by expanding from a 3-dimensional sensory manifold to a 7-dimensional perceptual manifold through geometric "twist" operations. The parallel to what I am seeing in transformer hidden states is hard to ignore: successful computation requires dimensional expansion, and failure correlates with dimensional confinement.

## Finding 2: There is no universal "success detector"

This is where the story gets genuinely interesting, and where the work diverges from the standard interpretability playbook.

After discovering dimensional collapse, you might expect a clean narrative: successful reasoning equals high dimensionality, failure equals low dimensionality. Build a classifier, ship it, done.

That is not what I found.

When I expanded the analysis to include all four groups, and especially when I compared *Direct successes* (G2) against *CoT successes* (G4), the signatures inverted. **Direct success looks geometrically like CoT failure.** Successful direct answers are low-dimensional, fast, and efficient. The model retrieves the answer in a compressed, ballistic trajectory, the opposite of the expansive, winding path that characterises successful Chain-of-Thought.

Out of 12 core metrics tested at middle layers, **9 flip sign** depending on which prompting strategy the model is using. A metric that predicts success under CoT actively predicts *failure* under Direct answering, and vice versa. Speed, effective dimension, radius of gyration, directional consistency, tortuosity, spectral entropy: all of them reverse their relationship to correctness when you switch regimes.

I call this **regime-relative success geometry**, and it may be the most important finding in the entire project. It means there is no single geometric "truth signature" that works across all computational strategies. The model has fundamentally different *modes* of arriving at correct answers, and what "good geometry" looks like depends entirely on which mode is active.

Coming from psychology, I could not help but think of Kahneman's dual-process theory ([Kahneman, 2011](https://us.macmillan.com/books/9780374533557/thinkingfastandslow)). System 1 is fast, automatic, and low-effort, pattern recognition that produces answers without deliberation. System 2 is slow, effortful, and deliberate, step-by-step reasoning that consumes cognitive resources. The parallel is striking. Direct success (G2) looks like System 1 cognition: compressed, efficient, ballistic. CoT success (G4) looks like System 2: expansive, exploratory, computationally expensive.

But here is the part that genuinely surprised me. In cognitive psychology, the dual-process framework has always been somewhat metaphorical. We infer the existence of two systems from behavioural data, response times, error patterns, susceptibility to cognitive load. Nobody has directly *seen* System 1 and System 2 as geometric objects.

In a transformer, you can.

The hidden-state trajectories of Direct success and CoT success are not just behaviourally different. They are *geometrically distinguishable at every layer of the network*. They occupy different regions of representational space, move at different speeds, show different curvature profiles, and exhibit different dimensional signatures. If dual-process theory describes something real about cognitive architecture, and not just a useful metaphor, then perhaps what I am measuring is its geometric substrate.

This is not just a cute analogy. A recent review in [Nature Reviews Psychology](https://www.nature.com/articles/s44159-025-00506-1) (Brady et al., 2025) examined LLM reasoning through the dual-process lens and concluded that LLMs mimic both System-1-like and System-2-like responses through different prompting methods. My work adds something new to that conversation: the first direct geometric evidence that these two modes produce fundamentally different *internal* computational structures, not just different output behaviours.

And the implication for AI safety is immediate. If you are trying to build a hallucination detector or a reasoning monitor, you cannot build a universal one. You need to know what computational regime the model is operating in *first*, and then apply regime-appropriate geometric criteria. A single threshold on any metric will misclassify systematically.

## Finding 3: The Goldilocks problem, or how CoT fails differently than you think

Here is the finding that the full-trajectory analysis revealed and that earlier truncated measurements completely missed.

The standard story about Chain-of-Thought failure is that it fails for the same reasons Direct prompting fails, just with extra tokens. The model does not know the answer, so adding "think step by step" just generates confident-sounding nonsense.

That is wrong. CoT failure looks nothing like Direct failure. It looks like an exaggerated version of CoT *success*.

Across virtually every layer from 0 to 23, G3 (CoT Fail) consistently shows *higher* effective dimensionality than G4 (CoT Success). The effect is small to medium (d ≈ 0.2 to 0.45 per layer) but remarkably consistent, appearing at 24 of 25 layers. CoT failures are not collapsing. They are *over-expanding*. They wander through an even higher-dimensional space than successful CoT reasoning, but they never converge on a solution.

This only became visible with full-trajectory analysis. When I previously truncated measurements to 32 tokens, both G3 and G4 looked similar because at token 32, both groups are still in their expansion phase. The full trajectory captures the extended wandering that distinguishes failed CoT from successful CoT, the part where G4 starts converging while G3 keeps exploring.

This completes what I think of as a Goldilocks story with three failure modes. Direct failures collapse: too little exploration, the trajectory is confined to a narrow corridor and never engages the problem's complexity. CoT successes explore and converge: just the right amount of expansion, followed by decisive commitment to a solution direction. CoT failures over-expand: too much exploration, the trajectory wanders through high-dimensional space without the convergent phase that would focus it toward a correct answer.

**Figure 3** shows this taxonomy directly. When I ran PCA on the geometric profiles of G3 (CoT Fail) trajectories alone, two distinct failure subtypes emerged. Type A failures show *mode collapse*, similar to Direct failures, where the CoT machinery simply does not engage. Type B failures show *wandering*, the trajectory explores extensively but never finds its way to convergence. The colour gradient (radius of gyration) reveals that these subtypes are geometrically distinct, occupying separate regions of the failure space.

![Figure 3: Taxonomy of Failure](Trajectory%20Geometry/figures/FigD_FailureTaxonomy_Publication.png)
*PCA decomposition of CoT Failure (G3) trajectories reveals two distinct failure subtypes. Type A (left cluster, purple) shows mode collapse. Type B (right cluster, yellow-green) shows extensive wandering without convergence. PC1 (99.8% variance) captures expansion/dimensionality.*

This has a precedent in cognitive psychology that I find difficult to dismiss as coincidence. In clinical assessment, two of the most common patterns of incorrect responses are *perseveration* (getting stuck on a single strategy and repeating it) and *confabulation* (generating elaborate but unfounded reasoning). These map almost directly onto what I am seeing geometrically. Type A failures (collapsed) resemble perseveration. Type B failures (wandering) resemble confabulation. The model is either stuck or lost, and these two failure modes produce distinct geometric signatures.

## Finding 4: The moment of commitment

Perhaps the most tangible finding for practitioners. I can detect the precise moment the model *commits* to an answer direction.

I tracked the point at which the radius of gyration (how spread out the trajectory is) drops most sharply. This commitment point shows dramatically different timing between groups:

For Direct responses (G1 and G2), commitment happens almost immediately, typically within the first 3 tokens, regardless of whether the answer is correct. This is consistent with System 1 processing: the model recognises the problem pattern, fires a ballistic response, and commits before any real computation occurs. The difference between Direct success and Direct failure is not *when* the model commits but *what* it commits to.

For CoT responses, the timing diverges. Successful CoT (G4) shows a median commitment point around token 51 with a bimodal distribution, suggesting two sub-strategies within CoT success. Failed CoT (G3) commits later still, around token 53 in median but with a much longer tail (mean = 62), consistent with the over-expansion finding: the model keeps searching longer because it never finds a satisfying convergent state.

**Figure 4** shows this contrast starkly. The red distribution (G1, Direct Fail) clusters tightly near zero. The blue distribution (G4, CoT Success) shows two distinct peaks, one around token 35 and another around token 55. The arrow marks the point below which all Direct commitments fall. The separation between fast commitment and deliberate commitment is not gradual. It is a phase transition.

![Figure 4: The Phase Transition: Time to Commitment](Trajectory%20Geometry/figures/FigE_CommitmentCurve_Publication.png)
*Density plots of commitment timing (token index of maximum radius-of-gyration drop) for G4 (CoT Success, blue) and G1 (Direct Fail, red). The bimodal structure in G4 suggests two distinct CoT sub-strategies. Direct responses commit almost immediately regardless of correctness.*

The bimodal structure in the G4 distribution is something I had not anticipated and that I think deserves further investigation. It suggests that successful Chain-of-Thought reasoning may involve at least two distinct computational sub-strategies: one that commits relatively early (around token 35) and one that requires extended deliberation (around token 55). Whether these correspond to different problem types, different solution methods, or some other structural variable is an open question. But the fact that commitment timing is *measurable* from internal geometry alone, without reading any of the model's text output, is what makes this finding practically useful.

## Finding 5: Harder problems produce bigger geometry

One concern you might reasonably raise is whether these geometric signatures are just proxies for problem difficulty. Maybe harder problems simply produce more tokens, and more tokens mechanically inflate metrics like radius of gyration or effective dimension.

The data says otherwise, and forcefully. When I binned problems by the magnitude of the correct answer (a proxy for computational complexity, since larger answers require more arithmetic steps), the geometric separation between success and failure groups *scaled with difficulty*:

**Figure 2** shows the effect size for radius of gyration (G4 vs G1) across four difficulty bins. For the simplest problems (2-3 digit answers), the effect size is d = 4.8, already enormous. For the hardest problems (8+ digit answers), it reaches d = 18.1.

![Figure 2: Difficulty-Driven Expansion](Trajectory%20Geometry/figures/FigC_DifficultyScaling_Publication.png)
*Effect size (Cohen's d) for radius of gyration comparing CoT Success (G4) vs Direct Fail (G1) across four difficulty bins. The geometric separation scales monotonically with problem complexity, from d = 4.8 for simple problems to d = 18.1 for complex ones.*

An effect size of 18.1 is not something I have ever encountered in the psychology literature. For reference, the largest effect sizes in cognitive psychology typically fall in the range of d = 2 to 3. What this tells me is that the model's internal geometry is not just correlated with difficulty; it is *calibrated* to it. Harder problems produce proportionally larger trajectories, and the separation between success and failure scales accordingly. This is exactly what you would expect if the geometry reflects genuine computational work rather than a superficial artefact.

And this scaling held across model sizes. I replicated the core findings on Qwen2.5-1.5B (three times the parameters) and Pythia-70m (a completely different architecture with 14 times fewer parameters). The absolute values change, but the qualitative patterns persist. Dimensional collapse, regime-relativity, and difficulty scaling all survived cross-architecture validation at these scales.

## Finding 6: Geometry predicts correctness (better than token length)

A practical question: can you actually predict whether a model will get the right answer from its trajectory shape? Yes. Using logistic regression with geometric features, I achieved an AUC of 0.898 for predicting correctness on Direct prompting, without reading a single token of output.

For context, response length alone (the simplest baseline you could imagine) achieves AUC of 0.645. Geometry alone achieves 0.898. Combining geometry with length does not improve over geometry alone, meaning the geometric metrics subsume the predictive value of length entirely. The shape of the trajectory contains strictly more information about correctness than the length of the response.

This has an immediate practical implication. If you can compute trajectory geometry in real time (which is computationally feasible for the metrics I use), you can build a monitoring system that flags likely failures *before* the model finishes generating. Not by reading its output. Not by asking it to self-assess (I tested that too, and the model's verbal self-reports do not correlate with its actual geometric state). By measuring the *shape* of its thought.

## What this means (and what I think it might mean)

The findings above are empirical. Here is where I want to be honest about the distinction between what I have demonstrated and what I believe these demonstrations point toward.

**What I have demonstrated:** In small transformer models solving arithmetic, internal hidden-state trajectories have measurable geometric properties that distinguish success from failure, differentiate computational regimes, scale with problem difficulty, and predict outcomes better than token length. These properties are not universal but regime-dependent, meaning any monitoring or interpretability tool must be regime-aware to work correctly.

**What I believe this points toward:** A fundamental reorientation of how we think about interpretability. The dominant paradigm in mechanistic interpretability, exemplified by brilliant work like Anthropic's sparse autoencoder research, is essentially lexicographic. It asks: "What are the basic units of representation?" My work suggests a complementary paradigm that is essentially *dynamical*. It asks: "How do these representations move through time?"

To use a language analogy: mechanistic interpretability is building the dictionary. Trajectory Geometry is studying the grammar. Both are necessary. Neither is sufficient alone. And right now, the field is heavily invested in dictionary-building while grammar remains almost entirely unexplored.

The dual-process parallel extends further than I initially expected. [Li et al., 2025](https://arxiv.org/abs/2502.17419) surveyed the landscape of reasoning LLMs and explicitly framed the field's trajectory as a shift from System-1-like to System-2-like architectures. [Bellini-Leite (2024)](https://journals.sagepub.com/doi/10.1177/10597123231206604) applied the Predicting and Reflecting Framework from dual-process theory to suggest that Chain-of-Thought and Tree-of-Thought prompting function as Type 2 reasoning strategies. My contribution to this conversation is not theoretical. It is geometric. I can *show* you the difference between System 1 and System 2 inside the model's hidden states. The trajectories look different. They live in different parts of the space. They commit at different times. And critically, they fail in completely different ways.

If this generalises beyond arithmetic (and I genuinely do not know if it does yet), the implications for AI safety are significant. Consider deception detection: a model that is "lying" (generating text it internally represents as false) might show a geometric signature distinct from a model that is genuinely uncertain or one that is confabulating. Right now, we detect deception by probing for truth-related features in the residual stream, a structural approach. A dynamical approach would ask: does the *trajectory* of a deceptive response look different from an honest one? My intuition says yes, but that remains to be tested.

Or consider capability prediction: if you can characterise the geometric repertoire of a model (what shapes of trajectories it can produce), you might be able to predict what it can and cannot do without exhaustive benchmarking. Dimensional collapse at a certain layer might indicate a fundamental computational bottleneck that no amount of prompting can overcome.

These are speculations. I flag them as such. But they are grounded in empirical observations that I did not expect to find when I started.

## What I have not proven (yet)

Intellectual honesty demands naming the limitations clearly, and I have several significant ones.

**Single task domain.** All of this was measured on arithmetic. I do not know if the same signatures appear in creative writing, factual recall, logical deduction, or code generation. I expect the regime-relativity principle to generalise, because it reflects something structural about how transformers process information rather than something specific to mathematics. But that is a hypothesis, not a finding.

**Small models only.** My primary model has 500 million parameters. I have replicated on 1.5 billion (Qwen) and 70 million (Pythia), showing scale stability and architecture independence at small scale. But I have not tested on frontier models with hundreds of billions of parameters. The geometric signatures could wash out at scale, or they could become even more pronounced. I genuinely do not know.

**Correlation, not causation.** I have shown that geometry *predicts* success and that it contains information beyond token count. I have not shown that *changing* the geometry changes the outcome. That is the causal intervention question, and it is the obvious next step. Someone with access to activation patching infrastructure could test this relatively quickly.

**No non-Qwen full replication with the complete metric suite.** Pythia confirmed the basic signatures on a different architecture but at a much smaller scale. A full replication on Gemma, LLaMA, or Mistral using all 33 metrics remains future work, and it is the single most important validation this line of research needs.

## Where this goes next (and who I need)

I want to be direct about something. I am a psychologist, not a machine learning engineer. I built this research programme by combining theoretical intuitions from cognitive science with AI tools that helped me implement experiments I could not have coded entirely on my own. The findings are real, the statistics are rigorous, and the experimental controls are solid. But I am one person working independently, and this work needs to be stress-tested by people with different expertise than mine.

The immediate next step is **cross-architecture replication**. If dimensional collapse and regime-relative geometry show up in LLaMA or Gemma the same way they appear in Qwen, this stops being an interesting observation about one model family and starts being a fundamental property of how transformers compute. I have the experimental framework and the metric suite ready. What I need is access to models and compute at larger scales than I can currently manage.

Beyond replication, the findings that most interest me are the potential for **geometric monitoring** (real-time trajectory analysis for detecting reasoning failures before they manifest in output) and **causal validation** (whether intervening on trajectory geometry via activation patching actually changes outcomes). Both require infrastructure and expertise that I do not currently have.

I am actively looking for collaborators. Mathematicians who can formalise the geometry. Mechanistic interpretability researchers who can connect trajectory signatures to specific circuits. ML engineers who can scale these experiments to frontier models. Cognitive scientists who can help strengthen or critique the dual-process parallels. I am open to co-authorship, and I need an [ArXiv](https://arxiv.org/) endorsement in cs.LG, cs.AI, or cs.CL before I can submit a preprint. If you have endorsement capability and think this work merits formal publication, I would genuinely appreciate the help.

The full experimental dataset, all 16 experiment logs, the complete metric suite code, and the statistical analysis pipeline will be available on GitHub [link forthcoming]. I welcome replication attempts. I welcome criticism. If someone can show me that these findings are artefactual, I want to know. And if they are real, I think they matter enough to warrant attention from people who can take them further than I can alone.

The shape of thought is measurable. I have measured it. And what it reveals is that the distinction between genuine reasoning and sophisticated pattern-matching is not just philosophical. It is geometric, it is quantifiable, and it is hiding in plain sight in every transformer ever trained.

---

**References**

1. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

2. Kojima, T., et al. (2022). Large Language Models are Zero-Shot Reasoners. [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)

3. Kahneman, D. (2011). *Thinking, Fast and Slow.* Farrar, Straus and Giroux. [Publisher link](https://us.macmillan.com/books/9780374533557/thinkingfastandslow)

4. Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. [Transformer Circuits](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

5. Jazayeri, M. & Ostojic, S. (2023). A unifying perspective on neural manifolds and circuits for cognition. *Nature Reviews Neuroscience, 24*(6), 363-377. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11058347/)

6. Chen, Y., et al. (2025). From sensory to perceptual manifolds: The twist of neural geometry. *Science Advances.* [DOI](https://www.science.org/doi/10.1126/sciadv.adv0431)

7. Brady, W., et al. (2025). Dual-process theory and decision-making in large language models. *Nature Reviews Psychology.* [DOI](https://www.nature.com/articles/s44159-025-00506-1)

8. Bellini-Leite, S. C. (2024). Dual Process Theory for Large Language Models. *Minds and Machines.* [SAGE](https://journals.sagepub.com/doi/10.1177/10597123231206604)

9. Li, Z., et al. (2025). From System 1 to System 2: A Survey of Reasoning Large Language Models. [arXiv:2502.17419](https://arxiv.org/abs/2502.17419)

10. Cohen, U., et al. (2020). Separability and geometry of object manifolds in deep neural networks. *Nature Communications.* [DOI](https://www.nature.com/articles/s41467-020-14578-5)

---

**About the author**

Kareem is a psychology researcher whose work sits at the intersection of cognitive science and AI interpretability. His framework, Trajectory Geometry, applies principles from clinical assessment and dynamical systems to understand how language models reason. He welcomes collaboration from researchers across disciplines and can be reached at [contact info]. Code and data: [GitHub link forthcoming].

*This article presents findings from an independent research project. All statistical claims are backed by permutation testing (10,000 shuffles) at p < 0.001, with effect sizes reported as Cohen's d. Sample size is N = 300 problems across all analyses unless otherwise noted. The author welcomes replication, critique, and collaboration.*
