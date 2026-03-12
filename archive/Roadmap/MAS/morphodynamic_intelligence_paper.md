# Morphodynamic Intelligence: Adaptive Organisational Topology as a Coordination Mechanism in Multi-Agent Systems

**Kareem Soliman**  
Independent Researcher, Sydney, Australia  
*Correspondence: [email]*

---

## Abstract

Multi-agent systems (MAS) increasingly employ large language model (LLM) agents in coordinated architectures, yet the organisational structures governing inter-agent communication and authority remain overwhelmingly static. This paper introduces Morphodynamic Intelligence (MI), a framework in which multi-agent systems dynamically restructure their organisational topology in response to environmental signals, drawing on established principles from organisational psychology and contingency theory. We present experimental evidence demonstrating that MI-enabled systems autonomously transition between organisational structures (hub-spoke, mesh, hierarchical) based on real-time assessment of environmental volatility, ambiguity, conflict, pressure, and drift. We contextualise these findings within a broader experimental programme evaluating six distinct organisational psychology-informed MAS architectures, all of which demonstrate measurable improvements over naive agent chaining baselines on complex cognitive tasks. Results indicate that the application of organisational design principles to multi-agent coordination represents an underexplored but productive direction for improving collective intelligence in artificial systems, with implications for both multi-agent system engineering and organisational science methodology.

**Keywords**: multi-agent systems, organisational psychology, morphodynamic intelligence, adaptive topology, collective intelligence, large language models, contingency theory

---

## 1. Introduction

The rapid proliferation of large language model (LLM) agents has produced a corresponding surge in multi-agent system architectures designed to solve complex tasks through coordinated agent interaction. Contemporary approaches include serial agent chains (Wu et al., 2023), debate-based architectures (Du et al., 2023), hierarchical delegation systems (Hong et al., 2023), and mixture-of-agents approaches (Wang et al., 2024). Despite this architectural diversity, the organisational structures governing these systems share a critical limitation: they are static. Once deployed, the communication topology, authority relationships, and role assignments remain fixed regardless of how the task environment evolves during execution.

This limitation stands in sharp contrast to the adaptive behaviour observed in effective human organisations. Decades of organisational psychology research have established that no single organisational structure optimises performance across all conditions (Burns & Stalker, 1961; Lawrence & Lorsch, 1967; Galbraith, 1973). Mechanistic structures with clear hierarchies and standardised procedures excel in stable, predictable environments, while organic structures with distributed authority and lateral communication channels prove superior under conditions of uncertainty and rapid change (Burns & Stalker, 1961). Contingency theory, one of the most empirically robust frameworks in organisational science, holds that organisational effectiveness is a function of the fit between internal structure and external environment (Donaldson, 2001).

The present paper proposes that this well-established principle transfers directly to multi-agent systems. We introduce Morphodynamic Intelligence (MI), a framework in which multi-agent systems monitor environmental conditions and autonomously restructure their organisational topology to maintain optimal structure-environment fit. The framework formalises organisational structure as a continuously optimisable variable defined by four parameters: centralisation, hierarchy, specialisation density, and agent autonomy. Environmental conditions are characterised through five signal dimensions: volatility, ambiguity, conflict, pressure, and drift. Structural transitions are governed by explicit mapping functions linking environmental states to optimal topological configurations, with hysteresis mechanisms preventing oscillatory restructuring.

We present proof-of-concept experimental evidence demonstrating that MI-enabled systems successfully execute autonomous structural transitions in response to environmental perturbations, and situate these findings within a broader experimental programme demonstrating that organisational psychology principles consistently improve multi-agent performance across diverse task domains.

### 1.1 Contributions

This paper makes four primary contributions. First, we introduce a formal parameterisation of multi-agent organisational topology that enables continuous structural adaptation. Second, we present experimental evidence of autonomous topology switching in an LLM-based multi-agent system responding to environmental signals. Third, we provide comparative evidence from six organisational psychology-informed MAS architectures demonstrating consistent improvements over naive baselines. Fourth, we articulate a research programme connecting organisational psychology and multi-agent system design through bidirectional knowledge transfer.

---

## 2. Related Work

### 2.1 Multi-Agent System Architectures

Contemporary multi-agent LLM systems employ several coordination paradigms. AutoGen (Wu et al., 2023) and similar frameworks implement conversational agent pipelines where agents interact through structured dialogue. MetaGPT (Hong et al., 2023) assigns software engineering roles to agents within a fixed hierarchical structure. CrewAI (Moura, 2024) provides role-based agent coordination with predefined task delegation patterns. Multi-agent debate approaches (Du et al., 2023; Liang et al., 2023) employ adversarial or collaborative discussion between agents to improve reasoning quality.

These systems represent significant advances in multi-agent coordination, yet they share a common structural assumption: the organisational architecture is designed once and remains invariant during execution. This contrasts with the adaptive organisations studied in management science, where structural flexibility is identified as a key determinant of long-term effectiveness (Volberda, 1996).

### 2.2 Adaptive Multi-Agent Systems

Research on adaptive multi-agent systems has explored several related but distinct dimensions of adaptation. Reinforcement learning approaches enable agents to adapt their individual policies in response to environmental feedback (Lowe et al., 2017; Yu et al., 2022). Communication topology learning allows agents to discover effective communication patterns through gradient-based optimisation (Jiang & Lu, 2018; Das et al., 2019). Role assignment mechanisms enable dynamic allocation of functional roles within fixed structural constraints (Wang et al., 2020). Attention-based architectures allow agents to selectively weight communications from different partners (Iqbal & Sha, 2019).

However, these approaches adapt within fixed structural parameters. An agent may learn which partners to attend to, but the fundamental organisational topology (hierarchical versus flat, centralised versus distributed, specialist versus generalist) remains predetermined. The present work extends adaptation to the organisational structure itself.

### 2.3 Organisational Psychology and Contingency Theory

Contingency theory emerged from empirical studies demonstrating that organisational effectiveness depends on alignment between internal structure and external environment. Burns and Stalker (1961) identified mechanistic and organic organisational forms as differentially suited to stable and turbulent environments respectively. Lawrence and Lorsch (1967) demonstrated that effective organisations match their internal differentiation and integration mechanisms to the complexity and uncertainty of their task environments. Galbraith's (1973) information processing theory formalised the relationship between environmental uncertainty and organisational information processing capacity requirements.

Subsequent research has identified specific structural parameters that mediate environment-performance relationships. Centralisation (the degree to which decision authority concentrates at upper hierarchical levels) proves beneficial under time pressure but detrimental to innovation (Sine et al., 2006). Formalisation (standardisation of procedures and communication channels) enhances efficiency in routine tasks but constrains adaptive responses to novel situations (Adler & Borys, 1996). Span of control (the number of subordinates reporting to each superior) affects both coordination costs and agent autonomy (Meier & Bohte, 2000).

Mintzberg's (1979) structural taxonomy provides a comprehensive classification of organisational forms including simple structure, machine bureaucracy, professional bureaucracy, divisionalised form, and adhocracy, each suited to different combinations of environmental complexity, dynamism, and hostility. The present work draws on this taxonomy to define the structural space through which multi-agent systems can navigate.

### 2.4 Organisational Psychology in Artificial Systems

Applications of organisational psychology to artificial systems remain sparse. Park et al. (2023) demonstrated emergent social behaviours in simulated agent societies but did not manipulate organisational structure as an independent variable. Qian et al. (2023) employed software engineering role structures in ChatDev but used fixed organisational designs. Recent work on "society of minds" architectures (Zhuge et al., 2023) explores multi-agent collaboration but without systematic reference to organisational theory.

The present work differs from these approaches by treating organisational psychology not as a metaphor for agent interaction but as a source of empirically validated design principles with direct computational applicability.

---

## 3. Theoretical Framework

### 3.1 Structural Parameterisation

We define multi-agent organisational topology through a four-dimensional parameter space $\mathcal{T} = (c, h, d, a)$ where:

- $c \in [0, 1]$: **Centralisation** — the degree to which decision authority concentrates in a single coordinator agent. At $c = 0$, all agents have equal decision weight; at $c = 1$, a single hub agent controls all task allocation and synthesis.

- $h \in [0, 1]$: **Hierarchy** — the depth of authority layers normalised by the number of agents. At $h = 0$, the structure is fully flat; at $h = 1$, the structure forms a strict chain of command.

- $d \in [0, 1]$: **Specialisation density** — the proportion of agents assigned to domain-specific roles versus generalist coordination roles. Higher values indicate greater functional differentiation.

- $a \in [0, 1]$: **Agent autonomy** — the degree to which individual agents can independently pursue sub-goals without coordinator approval. At $a = 0$, all actions require explicit authorisation; at $a = 1$, agents operate with full independence.

This parameterisation defines a continuous structural space within which standard organisational topologies occupy characteristic regions:

| Topology | $c$ | $h$ | $d$ | $a$ | Organisational Analog |
|---|---|---|---|---|---|
| Hub-Spoke | 0.8-1.0 | 0.7-1.0 | 0.3-0.5 | 0.2-0.4 | Simple structure |
| Mesh/Network | 0.1-0.3 | 0.0-0.2 | 0.3-0.5 | 0.7-1.0 | Adhocracy |
| Parallel Pods | 0.3-0.5 | 0.3-0.5 | 0.7-0.9 | 0.5-0.7 | Divisionalised form |
| Command Hierarchy | 0.9-1.0 | 0.9-1.0 | 0.5-0.7 | 0.1-0.3 | Machine bureaucracy |
| Matrix | 0.4-0.6 | 0.4-0.6 | 0.7-0.9 | 0.4-0.6 | Professional bureaucracy |

### 3.2 Environmental Signal Space

We characterise the task environment through a five-dimensional signal vector $\mathcal{E} = (v, m, k, p, \delta)$ where:

- $v \in [0, 1]$: **Volatility** — the rate of change in task requirements or environmental conditions during execution.
- $m \in [0, 1]$: **Ambiguity** — the degree of uncertainty regarding what constitutes a correct or adequate solution.
- $k \in [0, 1]$: **Conflict** — the degree to which sub-goals or agent assessments are contradictory.
- $p \in [0, 1]$: **Pressure** — time constraints and urgency of response requirements.
- $\delta \in [0, 1]$: **Drift** — the rate at which the problem definition itself shifts during execution.

### 3.3 Structure-Environment Mapping

The central thesis of morphodynamic intelligence is that a mapping function $f: \mathcal{E} \rightarrow \mathcal{T}$ exists such that for any environmental state $\mathcal{E}_t$, there is an optimal (or satisficing) structural configuration $\mathcal{T}^* = f(\mathcal{E}_t)$ that maximises collective performance. This function encodes the core insights of contingency theory:

- High volatility ($v > 0.7$) and high ambiguity ($m > 0.7$) favour distributed structures with high autonomy (mesh/network topology).
- High pressure ($p > 0.7$) and high conflict ($k > 0.7$) favour centralised structures with clear authority (hub-spoke or command hierarchy).
- High specialisation demands ($d_{task} > 0.7$) favour parallel pod structures with domain-specific clusters.
- Mixed signals favour matrix structures balancing multiple coordination demands.

### 3.4 Transition Dynamics and Hysteresis

Structural transitions impose coordination costs: agents must renegotiate roles, communication channels must be reconfigured, and in-progress work must be redistributed. To prevent pathological oscillation between structures in response to noise, the framework incorporates hysteresis mechanisms. A transition from topology $\mathcal{T}_i$ to $\mathcal{T}_j$ is triggered only when the expected performance improvement exceeds a threshold $\theta$ that accounts for transition costs:

$$\Delta P_{expected}(\mathcal{T}_j | \mathcal{E}_t) - \Delta P_{expected}(\mathcal{T}_i | \mathcal{E}_t) > \theta(\mathcal{T}_i, \mathcal{T}_j)$$

where $\theta(\mathcal{T}_i, \mathcal{T}_j)$ reflects the structural distance between topologies and the estimated disruption cost of transition. Transitions between structurally similar topologies (e.g., increasing centralisation within a hub-spoke configuration) incur lower costs than transitions between fundamentally different organisational forms (e.g., shifting from command hierarchy to mesh network).

### 3.5 Structural Operators

We define a set of structural operators that modify $\mathcal{T}$ in response to environmental signals:

| Operator | Effect on $\mathcal{T}$ | Trigger Conditions |
|---|---|---|
| **Centralise** | $c\uparrow$, $h\uparrow$, $a\downarrow$ | Crisis, decision paralysis, high pressure |
| **Distribute** | $c\downarrow$, $h\downarrow$, $a\uparrow$ | Innovation needed, exploration phase |
| **Specialise** | $d\uparrow$, role-specific channels | Task complexity exceeds generalist capacity |
| **Integrate** | $d\downarrow$, cross-role channels | Silo formation, integration failure |
| **Flatten** | $h\downarrow$ | Speed needed, hierarchy bottleneck |
| **Deepen** | $h\uparrow$ | Scale increased, span too wide |

---

## 4. Experimental Evidence

### 4.1 Morphodynamic Intelligence Proof-of-Concept

#### 4.1.1 Experimental Design

We implemented a morphodynamic multi-agent system using LLM-based agents (Gemini Pro 1.5) with the following architecture. A Meta-Organisational Agent (MOA) continuously monitors environmental signals and evaluates structure-environment fit. Three task agents (synthesis agent, evidence retriever, and coordinator) execute domain work. The MOA maintains authority to restructure the topology by modifying centralisation, hierarchy, specialisation density, and autonomy parameters, issuing restructuring directives with explicit reasoning when environmental signals indicate suboptimal fit.

The system was deployed on a marketing strategy task for a consumer product (budget electric toy plane targeting students). Environmental perturbations were introduced during execution to test adaptive restructuring capabilities. Initial environmental signals were set to moderate levels (volatility: 0.4, ambiguity: 0.5, conflict: 0.2, pressure: 0.3, drift: 0.3) with a perturbation at mid-execution shifting volatility to 0.6 and pressure to 0.5.

#### 4.1.2 Results

The system executed five turns of collaborative work with two autonomous structural transitions:

**Transition 1 (Turn 2)**: Hub-spoke → Mesh. The MOA assessed that high ambiguity and volatility in the initial task environment exceeded the capacity of centralised coordination, reasoning: "The current hub-spoke structure may not be able to effectively handle the high levels of ambiguity and volatility in the environment. Transitioning to a mesh topology will increase collaboration and responsiveness, allowing the organization to better adapt to changes."

**Transition 2 (Turn 4)**: Mesh → Hub-spoke. Following environmental perturbation increasing volatility to 0.6 and pressure to 0.5, the MOA assessed that the distributed mesh structure provided insufficient stability for the shifted conditions, reasoning: "Given the current volatility (0.6) and pressure (0.5) signals, the organization is facing a dynamic environment that may require more centralized decision-making typical of a hub-spoke model."

The final structural configuration ($c = 0.8$, $h = 1.0$, $d = 0.4$, $a = 0.4$) reflected a return to centralised coordination appropriate for the elevated pressure conditions. Task completion was achieved within the five-turn window with the system correctly adapting its coordination strategy to environmental conditions.

#### 4.1.3 Analysis

Several observations emerge from this proof-of-concept. First, the system demonstrated autonomous structural reasoning: the MOA generated coherent justifications for transitions that align with contingency theory predictions. High ambiguity favoured distributed structures; elevated pressure favoured centralisation. Second, the system exhibited appropriate transition frequency. Two transitions across five turns represents selective restructuring rather than oscillatory behaviour, suggesting the implicit cost-benefit assessment functioned effectively. Third, the transitions reflected genuine structural changes in agent coordination: the shift from hub-spoke to mesh altered communication patterns and decision authority, while the return to hub-spoke re-established centralised synthesis.

### 4.2 Contextual Evidence: Organisational Psychology-Informed Architectures

The morphodynamic intelligence experiment is contextualised within a broader experimental programme evaluating six MAS architectures, each implementing a distinct principle from organisational psychology. We summarise results from the most directly comparable experiment here; full results for all six architectures are presented in the supplementary materials.

#### 4.2.1 Adaptive Workload Distribution Grid (AWDG)

The AWDG implements shared mental model theory (Cannon-Bowers et al., 1993) through parallel specialist agents with shared context and structured synthesis phases. Evaluated on a product launch risk assessment task (AutoFlow AI enterprise platform), the AWDG was compared against a naive linear agent chain using identical model tiers (Gemini Pro 1.5) and identical input data.

The AWDG achieved a composite score of 88.4/100 versus 67.5/100 for the baseline, representing a 31% improvement. Performance gains were concentrated in expert-judged dimensions associated with organisational coordination: cascading risk awareness (+5.5 points), coordination quality (+5.0 points), actionability (+2.5 points), and novelty (+2.5 points). Both architectures achieved equivalent scores on structural metrics (schema compliance, categorisation), confirming that organisational design improvements manifest specifically in higher-order cognitive tasks requiring cross-domain integration.

A subsequent v2 iteration incorporating stake-aware triage and dynamic confidence thresholds achieved further improvement, with the escalation architecture scoring 27/30 on objective metrics versus 20/30 for both the baseline and v1 MAS, while simultaneously achieving variable (1-3 call) resource efficiency compared to the fixed costs of earlier architectures.

#### 4.2.2 Additional Architectures

Five additional architectures demonstrated consistent patterns. A Collective Intelligence Amplifier implementing the Nominal Group Technique achieved 41% novelty rates in creative ideation tasks. A Crisis Containment Task Force implementing groupthink prevention through dedicated process auditing successfully detected and corrected cognitive fixation during simulated crisis response. An Expert-Escalation Engine implementing contingency-based expertise matching achieved 0.92 confidence scores through appropriate task-complexity routing. A Probabilistic Forecasting Ensemble implementing epistemic role separation produced calibrated probability distributions with explicit bias correction (-5% to -10% adjustment). A Policy Simulation Sandbox implementing multi-stakeholder analysis detected 10 distinct policy exploitation loopholes and identified significant stakeholder alignment conflicts (score: 0.65).

These results collectively demonstrate that organisational psychology principles transfer systematically to multi-agent system design, producing improvements across diverse task domains. The morphodynamic intelligence framework extends this principle from static organisational design to dynamic structural adaptation.

---

## 5. Discussion

### 5.1 Theoretical Implications

The experimental results support three theoretical propositions. First, contingency theory's core insight (structure-environment fit determines performance) applies to artificial multi-agent systems. The morphodynamic system's autonomous transitions followed contingency theory predictions, favouring distributed structures under ambiguity and centralised structures under pressure. Second, the broader experimental programme demonstrates that organisational psychology principles are not mere metaphors when applied to computational systems but functional design principles producing measurable performance improvements. Third, the morphodynamic framework suggests that the most significant advances in multi-agent coordination may come not from improving individual agent capabilities but from optimising the organisational structures within which agents operate.

### 5.2 Comparison to Existing Approaches

Current multi-agent frameworks treat organisational structure as a design-time decision. MetaGPT assigns fixed software engineering roles; AutoGen defines static conversation patterns; CrewAI establishes predetermined task delegation hierarchies. Morphodynamic intelligence differs fundamentally in treating structure as a runtime variable optimised continuously during execution.

The closest existing work involves communication topology learning (Jiang & Lu, 2018), where agents learn optimal communication patterns through gradient-based optimisation. However, this approach operates at the level of pairwise communication weights rather than organisational topology, and adapts through slow gradient updates rather than discrete structural transitions triggered by environmental assessment. Dynamic role assignment mechanisms (Wang et al., 2020) adapt role allocation but within fixed structural constraints. Morphodynamic intelligence adapts the structural constraints themselves.

### 5.3 Computational Organisational Psychology

The broader experimental programme suggests a productive bidirectional relationship between organisational psychology and multi-agent system design. In one direction, organisational psychology provides empirically validated design principles for multi-agent coordination, as demonstrated across the six experimental architectures. In the other direction, multi-agent systems provide controlled experimental environments for testing organisational hypotheses with precision, speed, and scale impossible in human field studies.

Consider the methodological constraints of traditional organisational research. Testing whether flat versus hierarchical structures better support innovation requires identifying existing organisations with different structures (introducing selection bias), controlling for numerous confounds (likely incompletely), and relying on imperfect outcome measures over extended timescales (Ilgen et al., 2005). Computational experiments can randomly assign identical agent populations to different structures, systematically vary environmental characteristics, precisely measure performance outcomes, and replicate across thousands of trials. What requires decades of field research can be conducted computationally in hours.

This bidirectional relationship suggests the possibility of an accelerating research cycle in which multi-agent experiments generate organisational hypotheses, human validation studies test their applicability to biological organisations, and validated principles are re-incorporated into improved multi-agent designs.

### 5.4 Limitations

Several limitations constrain interpretation of the present findings. The morphodynamic intelligence experiment employed a single task domain with a small agent team (three task agents plus meta-organisational agent). Generalisation across task domains, agent populations, and environmental conditions requires further investigation. The environmental signal dimensions (volatility, ambiguity, conflict, pressure, drift) were assessed by the meta-organisational agent using its own judgment rather than through objective measurement, introducing potential assessment bias. The structure-environment mapping was implemented through prompted reasoning rather than learned mappings, meaning the system relied on the LLM's implicit understanding of contingency theory rather than on empirically optimised transition functions.

The broader experimental programme, while spanning six architectures and multiple task domains, employed a single model family (Gemini Pro 1.5) and did not systematically vary model capability as an independent variable. Whether organisational design improvements interact with model capability (i.e., whether they are more or less beneficial for stronger versus weaker models) remains an open question. Additionally, the evaluation metrics, while combining automated and expert-judged dimensions, did not include blind evaluation by independent organisational psychology researchers familiar with the relevant theoretical frameworks.

### 5.5 Future Directions

Several research directions emerge from this work. First, systematic benchmark evaluation across standard multi-agent tasks would establish whether morphodynamic adaptation produces consistent improvements over fixed-structure alternatives across diverse domains. Second, learning the structure-environment mapping function from experience, rather than relying on prompted contingency reasoning, could produce empirically optimised transition policies. Third, scaling to larger agent populations (10-100+ agents) would test whether the structural parameterisation remains tractable and whether additional structural dimensions become necessary. Fourth, human-agent comparative studies could test whether MI-enabled systems exhibit the same structural adaptation patterns as effective human organisations facing equivalent environmental conditions, providing validation of the bidirectional research programme.

A particularly promising direction involves integration with psychometric evaluation frameworks capable of assessing individual agent characteristics (cognitive style, risk tolerance, communication preferences) and using these assessments to inform role assignment during structural transitions. This would extend morphodynamic intelligence from purely structural adaptation to structure-personnel fit optimisation, paralleling the person-environment fit tradition in organisational psychology (Kristof-Brown et al., 2005).

---

## 6. Conclusion

This paper has introduced Morphodynamic Intelligence, a framework for adaptive organisational topology in multi-agent systems grounded in contingency theory from organisational psychology. Proof-of-concept experiments demonstrated that LLM-based multi-agent systems can autonomously restructure their organisational topology in response to environmental signals, executing transitions that align with contingency theory predictions. Broader experimental evidence from six organisational psychology-informed architectures demonstrated consistent improvements over naive baselines across diverse task domains, with a 31% composite improvement in the most rigorously controlled comparison.

These findings suggest that the organisational structures governing multi-agent coordination represent a significant and underexplored dimension of system design. Just as decades of organisational psychology research have demonstrated that structure-environment fit is a primary determinant of human organisational effectiveness, the present results indicate that this principle extends to artificial multi-agent systems. The implication is that advancing multi-agent capabilities requires not only improving individual agent intelligence but also developing more sophisticated approaches to the organisational contexts within which that intelligence operates.

Traditional organisations treat structure as relatively stable, requiring deliberate intervention spanning months or years to restructure. Morphodynamic intelligence creates the first framework where organisational structure itself becomes a continuously optimised variable responding in real-time to environmental conditions. This represents a qualitative shift in how we conceptualise multi-agent coordination, from designing organisations to designing organisms — systems that adapt their very structure to maintain effectiveness as conditions change.

---

## References

Adler, P. S., & Borys, B. (1996). Two types of bureaucracy: Enabling and coercive. *Administrative Science Quarterly*, 41(1), 61-89.

Burns, T., & Stalker, G. M. (1961). *The Management of Innovation*. Tavistock Publications.

Cannon-Bowers, J. A., Salas, E., & Converse, S. (1993). Shared mental models in expert team decision making. In N. J. Castellan (Ed.), *Individual and Group Decision Making* (pp. 221-246). Lawrence Erlbaum.

Das, A., Gerber, T., Levine, S., & Abbeel, P. (2019). TarMAC: Targeted multi-agent communication. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 1538-1546).

Donaldson, L. (2001). *The Contingency Theory of Organizations*. Sage Publications.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*.

Galbraith, J. R. (1973). *Designing Complex Organizations*. Addison-Wesley.

Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., ... & Wu, Y. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *arXiv preprint arXiv:2308.00352*.

Ilgen, D. R., Hollenbeck, J. R., Johnson, M., & Jundt, D. (2005). Teams in organizations: From input-process-output models to IMOI models. *Annual Review of Psychology*, 56, 517-543.

Iqbal, S., & Sha, F. (2019). Actor-attention-critic for multi-agent reinforcement learning. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 2961-2970).

Janis, I. L. (1972). *Victims of Groupthink*. Houghton Mifflin.

Jiang, J., & Lu, Z. (2018). Learning attentional communication for multi-agent cooperation. In *Advances in Neural Information Processing Systems*, 31.

Kristof-Brown, A. L., Zimmerman, R. D., & Johnson, E. C. (2005). Consequences of individuals' fit at work: A meta-analysis of person-job, person-organization, person-group, and person-supervisor fit. *Personnel Psychology*, 58(2), 281-342.

Lawrence, P. R., & Lorsch, J. W. (1967). *Organization and Environment*. Harvard Business School Press.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., ... & Shi, S. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *arXiv preprint arXiv:2305.19118*.

Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In *Advances in Neural Information Processing Systems*, 30.

Meier, K. J., & Bohte, J. (2000). Ode to Luther Gulick: Span of control and organizational performance. *Administration & Society*, 32(2), 115-137.

Mintzberg, H. (1979). *The Structuring of Organizations*. Prentice-Hall.

Moura, J. (2024). CrewAI: Framework for orchestrating role-playing autonomous AI agents. https://github.com/joaomdmoura/crewAI.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*.

Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., ... & Sun, M. (2023). Communicative agents for software development. *arXiv preprint arXiv:2307.07924*.

Sine, W. D., Mitsuhashi, H., & Kirsch, D. A. (2006). Revisiting Burns and Stalker: Formal structure and new venture performance in emerging economic sectors. *Academy of Management Journal*, 49(1), 121-132.

Volberda, H. W. (1996). Toward the flexible form: How to remain vital in hypercompetitive environments. *Organization Science*, 7(4), 359-374.

Wang, T., Liao, R., Ba, J., & Fidler, S. (2020). NerveNet: Learning structured policy with graph neural networks. In *International Conference on Learning Representations*.

Wang, J., Wang, Z., Xu, S., & Wang, S. (2024). Mixture-of-agents enhances large language model capabilities. *arXiv preprint arXiv:2406.04692*.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., ... & Wang, C. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.

Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Baez, A., & Fei-Fei, L. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. In *Advances in Neural Information Processing Systems*, 35.

Zhuge, M., Liu, H., Faccio, F., Ashley, D. R., Csordás, R., Gober, A., ... & Schmidhuber, J. (2023). Mindstorms in natural language-based societies of mind. *arXiv preprint arXiv:2305.17066*.

---

## Supplementary Materials

### A. Full Experimental Results for All Six Architectures

Comprehensive experimental data, scoring rubrics, raw outputs, and analysis for all six organisational psychology-informed MAS architectures are provided in the supplementary compilation document.

### B. Implementation Details

The morphodynamic intelligence proof-of-concept was implemented using the Anthropic and Google AI APIs with Gemini Pro 1.5 as the base model for all agents. The Meta-Organisational Agent received environmental signal vectors and current topology parameters as structured input, with instructions to evaluate structure-environment fit and recommend transitions with explicit reasoning. Task agents received topology-specific coordination instructions that varied based on current structural parameters (e.g., communication channel restrictions under hub-spoke topology, expanded peer-to-peer channels under mesh topology). Full implementation code is available at [repository URL].

### C. Scoring Methodology for AWDG Evaluation

The AWDG evaluation employed a 100-point composite score combining five automated objective metrics (50 points) and five expert-judged rubric metrics (50 points). Each metric was scored on a 10-point scale with explicit anchoring criteria to ensure consistent evaluation. Objective metrics were computed algorithmically from structured outputs. Rubric metrics were evaluated by the first author against pre-specified anchors. Inter-rater reliability was not assessed in this preliminary evaluation; future work will employ independent evaluators blind to experimental condition.
