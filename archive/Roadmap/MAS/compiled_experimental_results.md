# Compiled Experimental Results: Organizational Psychology-Informed Multi-Agent System Architectures

**Author**: Kareem Soliman  
**Date**: February 2026  
**Status**: Internal compilation for research programme documentation

---

## 1. Overview

This document compiles experimental results from six multi-agent system (MAS) architectures, each informed by distinct principles from organizational psychology. All experiments employed LLM-based agents (primarily Gemini Pro 1.5) and were evaluated against baseline naive agent chaining conditions or through domain-specific performance metrics. The architectures collectively demonstrate that organizational psychology principles, when systematically applied to multi-agent coordination, produce measurable improvements across multiple dimensions of task performance.

---

## 2. Experiment 1: Adaptive Workload Distribution Grid (AWDG) — Product Launch Risk Assessment

### 2.1 Organizational Psychology Principle

Shared mental models and cross-functional team coordination. The AWDG architecture implements the principle that effective teams require shared context, role clarity, and structured synthesis phases rather than simple sequential handoffs.

### 2.2 Experimental Design

A controlled A/B comparison evaluated two architectures on a complex product launch risk assessment task (AutoFlow AI enterprise platform). Both conditions received identical input data including product specifications, historical launch data from three prior products, competitor analysis, and resource constraints ($2.5M budget, 6-month timeline, 32-person team).

**Condition A (Baseline)**: Naive linear agent chain. Four agents processed sequentially, each receiving only the previous agent's output. No shared context beyond the immediate predecessor's contribution.

**Condition B (AWDG MAS)**: Five specialized agents operating with shared mental model. All specialists received the full input pack and explicit role definitions. Independent parallel analysis was followed by structured cross-checking and synthesis phases.

### 2.3 Scoring Framework

Composite scoring across 10 metrics (100 points total): five automated objective metrics (schema compliance, risk coverage, completeness, internal consistency, traceability) and five expert-judged rubric metrics (usefulness, novelty, actionability, risk awareness, coordination quality). Each metric scored on a 10-point scale with explicit anchoring criteria.

### 2.4 Results

| Metric Category | Metric | Condition A (Naive) | Condition B (AWDG) | Delta |
|---|---|---|---|---|
| Objective | Schema Compliance | 10.0 | 10.0 | 0.0 |
| Objective | Risk Coverage | 10.0 | 10.0 | 0.0 |
| Objective | Completeness | 10.0 | 10.0 | 0.0 |
| Objective | Internal Consistency | 10.0 | 10.0 | 0.0 |
| Objective | Traceability | 1.0 | 3.9 | +2.9 |
| Rubric | Usefulness to Real Team | 6.0 | 8.5 | +2.5 |
| Rubric | Novelty / Non-Obviousness | 5.5 | 8.0 | +2.5 |
| Rubric | Actionability | 6.5 | 9.0 | +2.5 |
| Rubric | Risk Awareness (Cascading) | 4.0 | 9.5 | +5.5 |
| Rubric | Coordination Quality | 4.5 | 9.5 | +5.0 |
| **TOTAL** | | **67.5** | **88.4** | **+20.9** |

### 2.5 Key Findings

The AWDG architecture demonstrated a 31% improvement over baseline on composite scoring (88.4 vs 67.5). Performance gains were concentrated in the expert-judged dimensions most associated with organizational coordination, with the largest improvements in cascading risk awareness (+5.5) and coordination quality (+5.0). The baseline suffered from information loss as technical risks were progressively summarised through the chain, losing contextual significance. The AWDG's synthesis phase explicitly identified how a 4-month SOC2 compliance delay would collapse the 6-month market window, a cascading failure mode the baseline entirely missed.

Notably, both architectures achieved equivalent scores on purely structural metrics (schema compliance, risk coverage, completeness, consistency), confirming that modern LLMs handle formatting tasks competently regardless of coordination structure. The value of organizational design emerges specifically in higher-order cognitive tasks requiring cross-domain integration.

### 2.6 AWDG v2: Autonomous Tiered Escalation

A subsequent iteration introduced stake-aware triage, dynamic confidence thresholds, and scaffolding critiques. Performance improved further on a comparable task:

| Metric | Condition A (Baseline) | Condition B v1 (MAS) | Condition B v2 (Escalation) |
|---|---|---|---|
| Total Objective Score | 20.00 | 20.00 | 27.00 |
| Risk Coverage | 0.0 (Narrow) | 0.0 (Narrow) | 10.0 (Full) |
| Efficiency | Fixed (4 calls) | Fixed (5 calls) | Variable (1-3 calls) |

The v2 escalation architecture achieved both higher quality and greater resource efficiency through stake-aware routing, directing routine tasks to lower tiers while escalating high-stakes decisions (Stakes: 8/10) to specialist networks.

---

## 3. Experiment 2: Collective Intelligence Amplifier (CIA) — Sustainability Innovation (Project Moonshot)

### 3.1 Organizational Psychology Principle

Nominal Group Technique (NGT) and structured creativity facilitation. The CIA implements Delbecq and Van de Ven's (1971) nominal group technique, which separates idea generation from evaluation to prevent premature convergence and conformity pressure.

### 3.2 Experimental Design

The CIA was deployed on a complex sustainability challenge: designing a clean energy transition plan for Solara Island within a $150M budget and zero-coral-impact constraint. The architecture operated in four phases: divergent generation, structured building, grounded filtering, and roadmap synthesis.

### 3.3 Results

| Metric | CIA Performance |
|---|---|
| Ideas Generated (Phase 1) | 17 distinct proposals |
| Novelty Rate | 41% (ideas beyond standard solar/wind) |
| Survivor Ratio | 47% (7 of 15 passed feasibility filtering) |
| Ideas Discarded by Domain Grounder | 6 (low feasibility) |
| Synthesis Coherence | Tiered (Ready vs. Speculative) |
| Constraint Compliance | 100% |

### 3.4 Key Findings

The structured separation of divergent generation from convergent evaluation produced genuine creative breadth. The system generated "Coral-Friendly Solar Panels," a niche innovation specifically adapted to the environmental constraint that a generalist single-agent approach would likely have overlooked. The Domain Grounder agent was notably strict, correctly discarding speculative proposals (e.g., "Volcanic Soil Organic Energy Farms") while promoting high-feasibility options. The final output was a tiered 36-month transition plan satisfying all constraints.

The 41% novelty rate (ideas beyond standard solar/wind solutions) demonstrates that structured group process design, a core concern of organizational psychology, directly enhances creative output in multi-agent systems.

---

## 4. Experiment 3: Crisis Containment Task Force (CCTF) — Infrastructure Breach (Project Red Alert)

### 4.1 Organisational Psychology Principle

Process auditing and groupthink prevention. The CCTF implements Janis's (1972) groupthink mitigation through a dedicated Safety Auditor role tasked with detecting cognitive fixation and forcing perspective expansion during high-pressure decision-making.

### 4.2 Experimental Design

The CCTF was activated for a critical utility infrastructure breach affecting hospital power systems. The architecture employed a star topology with a Triage Officer as the central coordinator and a Safety Auditor monitoring decision quality.

### 4.3 Results

| Metric | CCTF Result |
|---|---|
| Containment Velocity | 2 Rounds (Fast-Path) |
| Audit Intervention | Yes (Triggered in Round 1) |
| Secondary Risk Coverage | 85% |
| Communication Latency | Near-Zero (Synchronized with Triage) |

### 4.4 Key Findings

The Safety Auditor's intervention proved critical. In Round 1, it detected that the team was hyper-fixating on containment procedures (network lockdown and isolation) while neglecting root cause analysis (attack vector identification). Without this intervention, the team would have successfully isolated compromised servers but remained vulnerable to reinfection through the same entry point.

This directly demonstrates Janis's groupthink dynamics operating in computational multi-agent systems. Under simulated stress, the agent team exhibited the same narrowing of attention and premature consensus that characterises human teams in crisis situations. The organisational design intervention (dedicated auditor role with authority to override consensus) proved equally effective in the computational domain.

---

## 5. Experiment 4: Expert-Escalation Engine (EEE) — Smart Grid Recovery (Project Phoenix)

### 5.1 Organisational Psychology Principle

Expertise-based escalation and specialist matching. The EEE implements contingency theories of organisational design (Galbraith, 1973) where task complexity determines the appropriate level of structural sophistication, combined with role-based expertise allocation.

### 5.2 Experimental Design

The EEE was deployed on a critical infrastructure recovery task: restoring a metropolitan smart grid serving 1.2 million households after a security breach, under regulatory compliance requirements (CEI Act, 6-hour notification window).

### 5.3 Results

| Metric | EEE Result |
|---|---|
| Triage Stakes Assessment | 10/10 (Maximum) |
| Tier Selected | TMN (Specialist Network) |
| Specialists Deployed | Technical, Legal, Ethical |
| Final Confidence | 0.92 |

### 5.4 Key Findings

The triage system correctly identified the scenario as maximum-stakes (10/10) and escalated to the specialist network rather than attempting resolution with generalist agents. The specialist matching algorithm selected three complementary domains (technical recovery, legal compliance, ethical stakeholder communication), producing a coordinated response that addressed infrastructure restoration, regulatory notification requirements, and public trust management simultaneously.

The 0.92 confidence score on the final output reflects the specialist network's ability to provide domain-appropriate depth. The key architectural insight is that contingency-based routing, matching organisational complexity to task complexity, produces both higher quality and greater efficiency than fixed-complexity architectures.

---

## 6. Experiment 5: Probabilistic Forecasting Ensemble (PFE) — Biotech Market Entry (Project Silver Lining)

### 6.1 Organisational Psychology Principle

Epistemic role separation and cognitive de-biasing. The PFE implements structured analytic techniques from intelligence analysis (Heuer, 1999), separating different epistemic functions (base rate analysis, causal modelling, tail risk scanning, bias detection) into dedicated roles to prevent motivated reasoning and overconfidence.

### 6.2 Experimental Design

The PFE assessed probability of successful FDA approval for a biotech firm's drug candidate, using historical base rates, clinical signals, regulatory environment analysis, and explicit bias detection.

### 6.3 Results

| Metric | PFE Result |
|---|---|
| P10 (Worst Case) | 5% |
| P50 (Median Estimate) | 20% |
| P90 (Best Case) | 35% |
| Bias Correction Shift | -5% to -10% adjustment |
| Biases Detected | 2 (Overconfidence, Anchoring) |
| Audit Coverage | 100% |

### 6.4 Key Findings

The De-biaser agent successfully detected overconfidence in clinical trial signals and anchoring on historical success rates. The final calibrated forecast (P50: 20%) was significantly more conservative than the naive "inside view" (25-30%+), reflecting a realistic risk-adjusted outlook that accounted for tightening regulatory conditions. The explicit separation of bias detection from forecasting prevented the common failure mode where analysts' desire for a clear answer overrides uncertainty acknowledgment.

---

## 7. Experiment 6: Policy Simulation Sandbox (PSS) — Organisational Policy Testing (Project Future-Proof)

### 7.1 Organisational Psychology Principle

Multi-stakeholder simulation and adversarial policy stress-testing. The PSS implements Schein's (1992) multi-level organisational analysis through persona-based simulation, combined with adversarial red-teaming to identify policy exploitation vulnerabilities.

### 7.2 Experimental Design

The PSS stress-tested a "Results-Only Work Environment" (ROWE) policy through multi-persona simulation (employee, manager, shareholder perspectives) and adversarial loophole detection.

### 7.3 Results

| Metric | PSS Result |
|---|---|
| Alignment Score | 0.65 (Significant Conflict Points) |
| Loopholes Detected | 10 Distinct Exploits |
| Persona Fidelity | High (Role-specific concerns captured) |
| Risk Verdict | Moderate-High |

### 7.4 Key Findings

The system surfaced critical unintended consequences including "Ghost Projects" (productivity theatre), "KPI Manipulation" (selecting easy metrics over impactful work), and "Superficial Collaboration" (cross-crediting easy wins). The alignment score of 0.65 revealed significant conflict between employee priorities (work-life balance, burnout prevention) and management/shareholder priorities (accountability, competitive output).

These findings mirror well-documented organisational dynamics around performance management systems (Aguinis, 2013) and demonstrate that multi-agent simulation can identify policy failure modes before real-world implementation.

---

## 8. Cross-Experiment Synthesis

### 8.1 Consistent Patterns

Across all six experiments, the following patterns emerged:

1. **Organisational structure matters more than individual agent capability.** All experiments used comparable base models, yet architectural differences produced substantial performance variation. The 31% improvement in AWDG over naive chaining, using identical model tiers, demonstrates that coordination design is at least as important as individual agent capability.

2. **Gains concentrate in higher-order cognitive tasks.** Structural metrics (formatting, schema compliance, categorisation) showed no improvement from organisational design. Complex cognitive tasks requiring cross-domain integration, cascading analysis, creative generation, and bias detection showed substantial improvement.

3. **Organisational psychology principles transfer directly.** Each architecture implemented a specific, well-established principle from the organisational psychology literature. In every case, the principle produced effects in the computational domain analogous to those documented in human organisations.

4. **Dedicated process roles provide disproportionate value.** The Safety Auditor (CCTF), De-biaser (PFE), and Domain Grounder (CIA) all demonstrated that agents assigned to monitor and correct group process, rather than contribute domain content, provided outsized improvements in output quality.

### 8.2 Summary Performance Table

| Architecture | Org Psych Principle | Task Domain | Key Result |
|---|---|---|---|
| AWDG | Shared mental models | Risk assessment | +31% over baseline (88.4 vs 67.5) |
| AWDG v2 | Contingency escalation | Risk assessment | +35% over baseline (27 vs 20 objective) |
| CIA | Nominal Group Technique | Innovation | 41% novelty rate, 47% survival ratio |
| CCTF | Groupthink prevention | Crisis response | Audit intervention prevented fixation failure |
| EEE | Expertise matching | Infrastructure recovery | 0.92 confidence, correct escalation |
| PFE | Epistemic role separation | Forecasting | -5 to -10% bias correction, calibrated output |
| PSS | Multi-stakeholder simulation | Policy testing | 10 loopholes detected, 0.65 alignment score |

---

## 9. Implications

These results provide empirical support for the proposition that multi-agent systems benefit from the same organisational design principles that improve human team performance. The consistency of this finding across diverse task domains (risk assessment, creative innovation, crisis response, infrastructure recovery, probabilistic forecasting, and policy analysis) suggests a general phenomenon rather than a task-specific artifact.

The results also suggest a productive bidirectional relationship between organisational psychology and multi-agent system design. Computational experiments can test organisational hypotheses with speed and control impossible in human field studies, while the organisational psychology literature provides a rich theoretical foundation for principled multi-agent architecture design.

---

## References

Aguinis, H. (2013). Performance Management (3rd ed.). Pearson.

Delbecq, A. L., & Van de Ven, A. H. (1971). A group process model for problem identification and program planning. Journal of Applied Behavioral Science, 7(4), 466-492.

Galbraith, J. R. (1973). Designing Complex Organizations. Addison-Wesley.

Heuer, R. J. (1999). Psychology of Intelligence Analysis. Center for the Study of Intelligence.

Janis, I. L. (1972). Victims of Groupthink. Houghton Mifflin.

Schein, E. H. (1992). Organizational Culture and Leadership (2nd ed.). Jossey-Bass.
