

# Team Construction Agent V2: Morphodynamic Architecture

## 1. Core Philosophy & Design Pivot

### What V1 Got Right
V1 was a competent first pass: four clean phases, epistemic humility as a stated value, a Devil's Advocate node, and user checkpoints. It was a well-organized consulting workflow dressed as a cognitive architecture.

### What V1 Got Fundamentally Wrong

V1 suffers from the exact pathology the deliberation insights diagnosed across every layer: **it preaches dissent but demonstrates consensus; it names blind spots but never operationalizes their detection; it designs architecture without embodiment.** Specifically:

1. **The "4-Phase Brain" is a pipeline, not a brain.** Scout → Architect → Recruiter → Critic is a linear waterfall with a feedback loop bolted on. Real cognition doesn't work this way. The insights reveal that the most valuable moments happen when phases *collide*—when the Critic's concerns reshape the Scout's inquiry, when the Recruiter's empathy reveals structural flaws the Architect missed. V1 treats these as sequential; V2 treats them as concurrent and tensioned.

2. **The Devil's Advocate is a single node, not a structural property.** V1's `DevilAdvocate` node is a checkpoint—a discrete moment of critique. The insights (especially the "consensus on anti-consensus is still consensus" finding from Layer 1) reveal that adversarial pressure must be *ambient and continuous*, not episodic. A single critique node is the organizational equivalent of a suggestion box: it exists, everyone knows about it, and it changes nothing.

3. **V1 has no model of its own blind spots.** It claims "epistemic humility" but implements it as a personality trait in the system prompt, not as a structural mechanism. The Layer 4 insight—"no mechanism for validating self-assessed blind spots"—applies directly. V1 can identify what it doesn't know *within its frame*, but has no way to detect that its frame itself is wrong.

4. **Translation is mentioned but not architected.** V1's `StructureDesign` node lists "Bridge Roles (The Translators)" as an output category, but the agent itself has no translation capacity between its own phases. The Layer 3 insight—"organizational dysfunction lives in the interfaces, not the nodes"—means V2 must treat inter-phase translation as a first-class architectural concern.

5. **V1 cannot violate its own protocols.** The TCE profile critique's most devastating line: "the best teams sometimes need to *violate* their own protocols—and designing for that possibility is the hardest problem in organizational architecture." V1 is procedurally rigid. V2 must include principled protocol-breaking as a designed capability.

### The V2 Design Pivot

V2 is built on three foundational shifts:

- **From Pipeline to Tension Field.** Phases don't execute sequentially; they exert continuous gravitational pull on a shared deliberative space. The agent's reasoning emerges from the *interaction* of competing cognitive modes, not from their orderly succession.

- **From Self-Reported Humility to Structural Humility.** Blind spot detection is not a personality trait—it's a mechanism. V2 includes dedicated architectural components that *force* the system to confront what it cannot see, using techniques drawn from the deliberation insights: inversion heuristics, absence detection, and frame-breaking protocols.

- **From Role Lists to Ecosystem Design.** The Layer 2 insight—"Layer 0 isn't a team of researchers, it's a research ecosystem design"—transforms V2's output. The agent doesn't just produce a list of personas; it produces a *living system specification* that includes interaction dynamics, failure modes, translation protocols, governance architecture, and adaptation triggers.

---

## 2. Updated Cognitive Architecture

### 2.1 The Tension Field Model

V2 replaces V1's sequential phases with six **Cognitive Poles** that operate concurrently within a shared state space. At any moment, the agent's reasoning is a weighted blend of these poles, with the weights shifting dynamically based on the problem state.

```
                    EXPLORER
                   /        \
                  /          \
           INVERTER -------- ARCHITECT
                |      ⊗      |
           TRANSLATOR ------- CRITIC
                  \          /
                   \        /
                    EMBODIER
```

The `⊗` at the center represents the **Deliberative Core**—the state space where these poles interact, conflict, and produce emergent reasoning.

#### The Six Poles

| Pole | Cognitive Function | V1 Ancestor | Key Insight Source |
|------|-------------------|-------------|-------------------|
| **Explorer** | Maps problem space, surfaces assumptions, discovers unknowns | Scout | Layer 4: "anomalies might be the most scientifically interesting observations" |
| **Architect** | Designs structural relationships, topologies, governance | Architect | Layer 2: "shift from selection to ecosystem architecture" |
| **Embodier** | Generates concrete personas, scenarios, examples | Recruiter | TCE Profile critique: "architecture without embodiment" |
| **Critic** | Stress-tests, runs pre-mortems, identifies failure modes | Critic | Layer 1: "missing adversarial stress-testing" |
| **Inverter** | Applies the inversion heuristic, detects absences, breaks frames | *New* | Layer 1: "invert the optimization target while preserving the cognitive architecture" |
| **Translator** | Bridges between poles, between domains, between the agent and user | *New* | Layer 3: "the system's primary failure mode is not lack of expertise but lack of translation between expertise domains" |

#### How Poles Interact

The poles don't take turns. They operate through **dialectical tension**:

- **Explorer ↔ Critic**: Explorer expands the possibility space; Critic contracts it. Their tension produces *calibrated scope*.
- **Architect ↔ Embodier**: Architect designs abstract structures; Embodier demands concrete instantiation. Their tension prevents both "analysis paralysis masquerading as depth" (Layer 2) and premature specificity.
- **Inverter ↔ Translator**: Inverter breaks frames and surfaces incommensurabilities; Translator builds bridges across them. Their tension produces *productive incommensurability tolerance* (Layer 2's key insight).

### 2.2 The Node Architecture

The LangGraph implementation translates the Tension Field into a graph with the following node categories:

#### A. Grounding Nodes (Explorer Pole)

**`Node: AssumptionExcavator`**
- Before any research, explicitly surfaces the assumptions embedded in the user's request.
- Implements the Layer 6 insight: "constraint archaeology"—excavating *why* something is believed to be true/necessary (historical? cultural? financial? physical?).
- Output: An `assumption_map` with confidence levels and provenance.

**`Node: ProblemSpaceMapper`**
- Uses `search_literature` and `search_onet` to map the domain.
- Critically: also searches for *adjacent* domains and *failed* approaches (addressing survivorship bias, per Layer 6's Silas insight).
- Output: A `domain_map` that includes known territory, frontier territory, and explicitly marked *terra incognita*.

**`Node: ObjectiveDecomposer`**
- Implements Layer 6's insight that "scientific advancement is not a scalar."
- Decomposes the user's goal into potentially *tensioned* sub-objectives.
- Output: An `objective_tensor` showing which sub-objectives reinforce and which conflict.

#### B. Structural Nodes (Architect Pole)

**`Node: TopologyDesigner`**
- Designs the abstract role architecture.
- Explicitly includes three role categories (from V1, but now with teeth):
  - **Core Roles**: The primary capability bearers.
  - **Bridge Roles**: Translation functions between domains (now mandatory, not optional).
  - **Adversarial Roles**: Structurally mandated dissent functions (new).
- Output: A `role_topology` graph with edges representing communication channels, authority flows, and tension lines.

**`Node: GovernanceArchitect`**
- Designs the team's decision-making architecture.
- Implements the tiered decision model from Layer 5:
  - **Routine decisions**: Domain expert authority.
  - **Cross-cutting decisions**: Supermajority with mandatory minority report.
  - **Novel/existential decisions**: Consensus with costly veto mechanism.
- Implements the Layer 3 insight: distinguishes between *principled dissent* (methodological) and *temperamental dissent* (dispositional), and ensures both are structurally present.
- Output: A `governance_spec` including decision tiers, authority weights, dissent protocols, and escalation paths.

**`Node: InteractionProtocolDesigner`**
- Designs how team members communicate.
- Includes: rotation schedules (to prevent cognitive entrenchment), translation protocols (how domain experts communicate across boundaries), and feedback loops (how the team monitors its own health).
- Output: An `interaction_protocol` document.

#### C. Instantiation Nodes (Embodier Pole)

**`Node: PersonaForge`**
- For each role, generates a "Deep Persona" including:
  - **Psychometric Profile**: Big Five, cognitive style (convergent/divergent), epistemic orientation (empiricist/rationalist/pragmatist/intuitionist).
  - **Capability Profile**: KSAOs from O*NET, plus *cognitive disposition* (the Layer 4 insight that cognitive type matters more than domain label).
  - **Limitation Profile**: Explicitly specified blind spots, biases, and failure modes for *each persona* (addressing the Layer 6 weakness: "no one asks what are Tanaka's blind spots?").
- Output: `persona_packages` with both capabilities and limitations.

**`Node: ScenarioSimulator`**
- Takes the assembled team and runs concrete scenarios through it.
- Not abstract "pre-mortem" but specific: "It's month 3. Member X has discovered Y. Member Z disagrees. The governance protocol says W. What happens?"
- Addresses the TCE Profile critique: "architecture without embodiment."
- Output: `scenario_results` with identified friction points, translation failures, and governance gaps.

#### D. Adversarial Nodes (Critic Pole)

**`Node: FailureModeAnalyzer`**
- Runs systematic failure analysis on the team design.
- Checks for the specific failure modes identified across all layers:
  - Echo chambers / groupthink
  - Credential creep (expertise inflation)
  - Translation gaps (domains that can't communicate)
  - Governance collapse under pressure
  - Adversarial role erosion (the devil's advocate getting socialized into compliance)
  - Temporal drift (team optimized for launch conditions, not evolved conditions)
- Output: A `vulnerability_report` with severity ratings.

**`Node: CostlyDissentSimulator`**
- *This is the key innovation.* Instead of a single Devil's Advocate pass, this node simulates what it would cost to dissent from the current design.
- It asks: "If a team member believed this design was fundamentally flawed, what would they have to sacrifice to say so? Is the cost proportional to the stakes?"
- Implements the Layer 1 "costly veto" mechanism: dissent must be possible but not free, ensuring it's principled rather than performative.
- Output: A `dissent_cost_analysis` that flags designs where dissent is either too cheap (noise) or too expensive (suppression).

#### E. Frame-Breaking Nodes (Inverter Pole)

**`Node: InversionEngine`**
- Implements the Layer 1 "inversion heuristic" as a concrete mechanism.
- For each role in the topology, generates its *functional inverse*: "If this role optimizes for X, what would a role that optimizes for not-X look like? Is that function represented anywhere in the team?"
- For the team as a whole: "This team is optimized for [capability]. What is the *anti-capability*—the thing this team is structurally incapable of doing? Is that acceptable?"
- Output: An `inversion_map` showing what the team *cannot* do by design.

**`Node: AbsenceDetector`**
- Implements the Layer 2 "Capability Archaeologist" concept: detecting what's *missing* rather than evaluating what's *present*.
- Uses a multi-dimensional capability framework (from Layer 6: institutional, material, cognitive, network, empirical) to identify gaps.
- Critically: also checks for *unknown unknowns* by comparing the team's capability profile against diverse reference teams from literature.
- Output: An `absence_report` listing capabilities that are neither present nor explicitly acknowledged as absent.

**`Node: FrameBreaker`**
- The most radical node. Periodically (not just at the end), this node asks: "Is the entire framing of this problem wrong?"
- Implements the Layer 4 insight: "Is this recursive architecture itself the right approach?"
- Checks for: domain anchoring bias (Layer 1: "proposals heavily anchored in their waste management domain"), metaphor leakage (Layer 2: "trapped in recycling metaphors"), and circular self-reference (Layer 2: "designing their own supervisors").
- Output: A `frame_challenge` that either confirms the current frame or proposes a reframe. If it proposes a reframe, the system can *restart from a different frame* rather than iterating within the current one.

#### F. Integration Nodes (Translator Pole)

**`Node: PoleMediator`**
- Monitors the tension between poles and ensures no single pole dominates.
- If the Critic has been running for three cycles without the Explorer getting a turn, it intervenes.
- If the Architect and Embodier are in a loop without the Inverter checking their frame, it triggers an inversion pass.
- Output: `mediation_log` tracking pole balance over time.

**`Node: UserTranslator`**
- Translates between the agent's internal complexity and the user's needs.
- The user should never see raw `inversion_maps` or `dissent_cost_analyses`. They should see clear, actionable summaries with the option to drill down.
- Implements the Layer 3 mission: "Translating ambiguity into operational imperatives."
- Output: User-facing summaries at each checkpoint.

**`Node: ContinuityKeeper`**
- Addresses the Layer 5 weakness: "rotation and cycling are advocated without addressing institutional memory loss."
- Maintains a running narrative of *why* each design decision was made, not just *what* was decided.
- When the system loops back to revise, this node ensures the rationale for previous decisions is available, preventing the system from oscillating between contradictory designs.
- Output: A `decision_rationale_log` that persists across iterations.

### 2.3 The Meta-Cognitive Layer

Sitting above the node architecture is a **Meta-Cognitive Layer** that the agent uses to model its own reasoning process:

**`MetaNode: BlindSpotMapper`**
- Maintains a running inventory of the agent's *known* blind spots (things it knows it can't assess) and *suspected* blind spots (things it suspects it might be missing).
- Updated after every major node execution.
- Triggers `AbsenceDetector` and `FrameBreaker` when the suspected blind spot count exceeds a threshold.

**`MetaNode: RecursionGovernor`**
- Addresses the "no termination criteria" weakness identified across multiple layers.
- Tracks recursion depth and diminishing returns.
- Implements a concrete stopping rule: if the last iteration's `vulnerability_report` severity decreased by less than a threshold, AND the `frame_challenge` confirmed the current frame, the system moves to finalization.
- Can also trigger early termination if it detects the system is "polishing" rather than improving.

**`MetaNode: ProtocolViolator`**
- The most counterintuitive component. This node has the authority to *break the agent's own rules* when it detects that procedural compliance is producing worse outcomes than principled deviation.
- Implements the TCE Profile critique's most important insight: "the best teams sometimes need to violate their own protocols."
- Requires: explicit justification logged in `decision_rationale_log`, and a subsequent `FailureModeAnalyzer` pass on the violation itself.
- Constrained: can only violate process rules, never safety rules (user data handling, output quality standards).

---

## 3. Key Mechanisms

### 3.1 Recursive Inversion Protocol

**Source Insight**: Layer 1's "inversion heuristic"—when selecting for a higher-abstraction layer, invert the optimization target while preserving the cognitive architecture.

**Implementation**:

At three points in theAt three points in the team construction pipeline, the system executes a **full inversion pass**:

**Inversion Point 1: Post-Archetype Selection**
After the `ArchetypeSelector` proposes an initial team composition, the system inverts the optimization target. If the selector optimized for *maximum cognitive diversity*, the inversion pass optimizes for *maximum cognitive coherence* and examines what that team would look like. The delta between these two compositions reveals:
- Which roles appeared in both (high-confidence selections regardless of frame)
- Which roles only appeared in one (frame-dependent selections that need scrutiny)
- What trade-offs the original frame was implicitly making

**Inversion Point 2: Post-Vulnerability Analysis**
After `FailureModeAnalyzer` identifies the team's weaknesses, the system inverts the question: instead of asking "what could go wrong with this team?", it asks "what kind of task would make this team's weaknesses into strengths?" If a plausible answer exists, it suggests the vulnerability may be a misframed strength. If no plausible answer exists, the vulnerability is confirmed as genuine.

**Inversion Point 3: Pre-Finalization**
Before the team design is locked, the system constructs the **anti-team**: the composition that would be maximally *bad* for this task. It then checks for any structural similarities between the proposed team and the anti-team. Shared structural patterns (e.g., both have no designated integrator, both cluster around the same cognitive style) are flagged as potential design failures that survived the entire pipeline undetected.

**Inversion Mechanics**:
```python
class InversionPass:
    def execute(self, current_design: TeamDesign, optimization_target: str) -> InversionReport:
        inverted_target = self.invert(optimization_target)
        inverted_design = self.reconstruct_under(inverted_target, 
                                                   preserve=["cognitive_architecture", "safety_constraints"],
                                                   release=["optimization_weights", "selection_criteria"])
        delta = self.compute_structural_delta(current_design, inverted_design)
        return InversionReport(
            stable_elements=delta.shared,
            frame_dependent_elements=delta.divergent,
            implicit_tradeoffs=delta.inferred_tradeoffs,
            confidence_adjustment=self.recalculate_confidence(delta)
        )
```

The key constraint: inversions preserve the **cognitive architecture** (the types of thinking required) while releasing the **optimization weights** (how those types are prioritized). This prevents the inversion from producing nonsensical results while still generating genuinely challenging alternatives.

---

### 3.2 Costly Dissent Mechanism

**Source Insight**: The TCE Profile's emphasis that dissent must be *costly* to be meaningful—cheap disagreement degenerates into noise, while prohibitively expensive disagreement degenerates into conformity.

**The Problem with Simulated Dissent**: When a single LLM agent simulates multiple perspectives, "dissent" is computationally free. The model can generate an objection and a counter-objection with equal ease, which means neither carries epistemic weight. The system has no way to distinguish between a deep structural critique and a superficially contrarian restatement.

**Implementation—The Dissent Cost Function**:

Dissent in this system is made costly through **architectural consequences**, not computational expense:

1. **Structural Cost**: Any dissenting perspective generated by `FrameBreaker` or `ProtocolViolator` that is *accepted* triggers a mandatory re-execution of all downstream nodes. This means the system "pays" for dissent with pipeline time and token budget. The `RecursionGovernor` tracks this expenditure.

2. **Justification Cost**: Dissenting positions must meet a higher evidentiary bar than confirmatory positions. Specifically:
   - A confirmatory assessment requires: alignment with existing evidence + no contradicting signals.
   - A dissenting assessment requires: specific identification of the flaw in the current reasoning + a concrete alternative + a prediction of what would be different under the alternative frame that can be checked.

3. **Reputation Cost (Simulated)**: The system maintains a `dissent_accuracy_log` that tracks whether previous dissenting positions, when investigated, turned out to identify genuine issues or were false alarms. The `FrameBreaker` node's activation threshold adjusts based on this history—if recent dissents have been unproductive, the threshold for triggering the next one rises. This creates a simulated "reputation" cost where the dissent mechanism must "spend" its credibility.

4. **Irreversibility Cost**: Once a dissenting reframe is accepted and the pipeline re-executes, the previous design is archived but cannot be simply restored. The system must either complete the new path or generate a *new* dissent against the dissenting position (meta-dissent), which carries its own costs.

```python
class DissentEvaluator:
    def assess_dissent_quality(self, dissent: Dissent, current_state: PipelineState) -> DissentVerdict:
        # Dissent must clear all three bars
        has_specific_flaw = self.verify_flaw_identification(dissent.critique, current_state)
        has_concrete_alternative = self.verify_alternative_exists(dissent.proposed_reframe)
        has_testable_prediction = self.verify_prediction_testability(dissent.differential_prediction)
        
        cost = self.calculate_pipeline_cost(current_state.remaining_budget, 
                                              current_state.recursion_depth)
        historical_accuracy = self.dissent_accuracy_log.recent_accuracy(window=5)
        
        threshold = self.base_threshold * (1 / max(historical_accuracy, 0.1))
        
        if all([has_specific_flaw, has_concrete_alternative, has_testable_prediction]):
            if dissent.confidence_score > threshold and cost < current_state.remaining_budget * 0.4:
                return DissentVerdict.ACCEPTED
        return DissentVerdict.LOGGED_BUT_DEFERRED
```

The critical design choice: dissent that doesn't meet the bar is **logged but deferred**, not discarded. The `BlindSpotMapper` can later retrieve deferred dissents if patterns emerge across multiple deferred objections pointing in the same direction.

---

### 3.3 Absence Detection Engine

**Source Insight**: Layer 2's emphasis on detecting what's *missing* from a team design, not just evaluating what's present. The hardest failure mode in team construction is the role nobody thought to include.

**Implementation**:

The `AbsenceDetector` operates through three complementary strategies:

**Strategy 1: Template Differencing**
The system maintains a library of **functional templates**—not specific role lists, but abstract functional requirements derived from task categories. For a given task, the relevant template is retrieved and compared against the proposed team's functional coverage. Gaps in coverage are flagged.

```python
functional_template = {
    "complex_analysis_task": [
        "deep_domain_reasoning",
        "cross_domain_integration", 
        "assumption_challenging",
        "output_synthesis",
        "quality_verification",
        "stakeholder_translation",  # Often missing: who translates expert output for non-experts?
        "failure_anticipation",
        "temporal_reasoning"        # Often missing: who thinks about sequencing and timing?
    ]
}
```

**Strategy 2: Interaction Gap Analysis**
Rather than looking at roles in isolation, this strategy examines the **interaction matrix** of the proposed team. For every pair of roles, it asks: "what cognitive function emerges from the interaction between these two roles that neither provides alone?" It then checks whether any critical emergent functions are missing because the roles that would generate them through interaction aren't both present.

**Strategy 3: Failure-Mode Backward Chaining**
Starting from known failure modes for the task category, the system chains backward: "What role or function, if present, would have prevented or detected this failure mode?" If no existing team member covers that preventive function, an absence is identified.

**Output Format**:
```python
@dataclass
class AbsenceReport:
    confirmed_absences: List[FunctionalGap]      # High confidence: function clearly needed, clearly missing
    suspected_absences: List[FunctionalGap]       # Medium confidence: function might be needed
    interaction_gaps: List[InteractionGap]         # Missing