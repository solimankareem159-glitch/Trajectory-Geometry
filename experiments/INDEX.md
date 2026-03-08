# Experiment Index

This directory contains all experimental work from the Trajectory Geometry research project.

## All Experiments

| ID | Folder | Focus | Date | Model | Verdict |
|----|--------|-------|------|-------|---------|
| 01 | [EXP-01_GeometricSignatures_2025-11-03](./EXP-01_GeometricSignatures_2025-11-03) | Geometric Signatures | 2025-11-03 | Gemini API | Invalid |
| 02 | [EXP-02_LatentFactors_2025-11-05](./EXP-02_LatentFactors_2025-11-05) | Latent Factors | 2025-11-05 | Qwen-0.5B | Invalid |
| 03 | [EXP-03_RegimeInvariants_2025-11-07](./EXP-03_RegimeInvariants_2025-11-07) | Regime Invariants | 2025-11-07 | Qwen-0.5B | Failed |
| 04 | [EXP-04_OGMPT_2025-11-10](./EXP-04_OGMPT_2025-11-10) | OG-MPT | 2025-11-10 | Qwen-0.5B | Success |
| 05 | [EXP-05_SafetyOGMPT_2025-11-12](./EXP-05_SafetyOGMPT_2025-11-12) | Safety OG-MPT | 2025-11-12 | Qwen-0.5B | Failed |
| 06 | [EXP-06_DataGeneration_2025-11-15](./EXP-06_DataGeneration_2025-11-15) | Pilot Metric Validation | 2025-11-15 | Qwen-0.5B | Inconclusive |
| 07 | [EXP-07_StaticGeometry_2025-11-18](./EXP-07_StaticGeometry_2025-11-18) | Static Geometry | 2025-11-18 | Qwen-0.5B | Valid/Insufficient |
| 08 | [EXP-08_TrajectoryGeometry_2025-11-20](./EXP-08_TrajectoryGeometry_2025-11-20) | Trajectory Geometry | 2025-11-20 | Qwen-0.5B | **Breakthrough** |
| 09 | [EXP-09_GeometryCapability_2025-11-22](./EXP-09_GeometryCapability_2025-11-22) | Geometry & Capability | 2025-11-22 | Qwen-0.5B | **Breakthrough** |
| 09B | [EXP-09B_TinyLlamaReplication_2025-11-23](./EXP-09B_TinyLlamaReplication_2025-11-23) | TinyLlama Replication | 2025-11-23 | TinyLlama-1.1B | Failed |
| 10 | [EXP-10_SelfReportConsistency_2025-11-25](./EXP-10_SelfReportConsistency_2025-11-25) | Self-Report Consistency | 2025-11-25 | Qwen-0.5B | Failed |
| 11 | [EXP-11_ExtendedGeometricSuite_2025-11-27](./EXP-11_ExtendedGeometricSuite_2025-11-27) | Extended Geometric Suite | 2025-11-27 | Qwen-0.5B | Success |
| 12 | [EXP-12_AdvancedDiagnostics_2025-11-29](./EXP-12_AdvancedDiagnostics_2025-11-29) | Advanced Diagnostics | 2025-11-29 | Qwen-0.5B | Success |
| 13 | [EXP-13_FailureSubtyping_2025-12-01](./EXP-13_FailureSubtyping_2025-12-01) | Failure Subtyping | 2025-12-01 | Qwen-0.5B | Success |
| 14 | [EXP-14_UniversalSignature_2025-12-03](./EXP-14_UniversalSignature_2025-12-03) | Universal Signature | 2025-12-03 | Qwen-0.5B | **Breakthrough** |
| 15 | [EXP-15_LengthConfound_2025-12-05](./EXP-15_LengthConfound_2025-12-05) | Length Confound | 2025-12-05 | Qwen-0.5B | Success |
| 16 | [EXP-16_Pythia70m_2025-12-07](./EXP-16_Pythia70m_2025-12-07) | Pythia-70m | 2025-12-07 | Pythia-70m | Success |
| 16B | [EXP-16B_Qwen15B_2025-12-08](./EXP-16B_Qwen15B_2025-12-08) | Qwen2.5-1.5B | 2025-12-08 | Qwen2.5-1.5B | Success |
| 17 | [EXP-17_BaselineReplication_2026-02-11](./EXP-17_BaselineReplication_2026-02-11) | Baseline Replication (3B) | 2026-02-11 | Qwen2.5-3B | Success |
| 18 | [EXP-18_ConsolidatedMetricSuite_2026-02-13](./EXP-18_ConsolidatedMetricSuite_2026-02-13) | Consolidated Metric Suite | 2026-02-13 | Qwen2.5-0.5B | Success |
| 18B | [EXP-18B_ScalingGeometry_2026-02-13](./EXP-18B_ScalingGeometry_2026-02-13) | Scaling Geometry | 2026-02-13 | Multi-model | Partial |
| 19 | [EXP-19_Robustness_2026-02-14](./EXP-19_Robustness_2026-02-14) | Robustness Replication | 2026-02-14 | Multi-arch | **Confirmed** |

## Folder Structure

Each experiment follows this standardized structure:

```
EXP-<NN>_<ShortTitle>_<YYYY-MM-DD>/
├── data/          # Raw and processed outputs (CSVs, metrics)
├── analysis/      # Python scripts for extraction, computation, and visualization
├── figures/       # Generated plots (optional)
├── reports/       # Extended analysis reports (optional)
└── metadata.yaml  # Experiment configuration (optional)
```

## Quick Navigation

- **Per-experiment summaries:** See [`/experiment_summaries/`](../experiment_summaries/)
- **Research narrative:** See [`/docs/research_narrative.md`](../docs/research_narrative.md)
- **Main README:** See [project root](../README.md)

## Research Phases

| Phase | Experiments | Focus |
|-------|-------------|-------|
| I — Shapes | 01–02 | Static operator signatures in output/hidden states |
| II — Invariants | 03–07 | Cross-regime mapping, gating interventions, probing |
| III — Dynamics | 08–10 | Differential geometry metrics, capability correlation |
| IV — Failure Analysis | 11–13 | Dimensional collapse, fractal complexity, subtyping |
| V — Paradigm & Scale | 14–16B | Regime-relative geometry, length confound, cross-model |
| VI — Robustness | 17–19 | Scale replication, metric consolidation, multi-architecture validation |
