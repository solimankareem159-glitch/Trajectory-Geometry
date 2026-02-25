# Experiment Index

This directory contains all experimental work from the DSG research project, organized in standardized format.

## All Experiments

| ID | Folder | Focus | Date | Model |
|----|--------|-------|------|-------|
| 01 | [EXP-01_GeometricSignatures_2025-11-03](./EXP-01_GeometricSignatures_2025-11-03) | Geometric Signatures | 2025-11-03 | TBD |
| 02 | [EXP-02_LatentFactors_2025-11-05](./EXP-02_LatentFactors_2025-11-05) | Latent Factors | 2025-11-05 | TBD |
| 03 | [EXP-03_RegimeInvariants_2025-11-07](./EXP-03_RegimeInvariants_2025-11-07) | Regime Invariants | 2025-11-07 | TBD |
| 04 | [EXP-04_OGMPT_2025-11-10](./EXP-04_OGMPT_2025-11-10) | OG-MPT | 2025-11-10 | TBD |
| 05 | [EXP-05_SafetyOGMPT_2025-11-12](./EXP-05_SafetyOGMPT_2025-11-12) | Safety OG-MPT | 2025-11-12 | TBD |
| 06 | [EXP-06_DataGeneration_2025-11-15](./EXP-06_DataGeneration_2025-11-15) | Data Generation | 2025-11-15 | TBD |
| 07 | [EXP-07_StaticGeometry_2025-11-18](./EXP-07_StaticGeometry_2025-11-18) | Static Geometry | 2025-11-18 | TBD |
| 08 | [EXP-08_TrajectoryGeometry_2025-11-20](./EXP-08_TrajectoryGeometry_2025-11-20) | Trajectory Geometry | 2025-11-20 | TBD |
| 09 | [EXP-09_GeometryCapability_2025-11-22](./EXP-09_GeometryCapability_2025-11-22) | Geometry & Capability | 2025-11-22 | Qwen-0.5B |
| 09B | [EXP-09B_TinyLlamaReplication_2025-11-23](./EXP-09B_TinyLlamaReplication_2025-11-23) | TinyLlama Replication | 2025-11-23 | TinyLlama-1.1B |
| 10 | [EXP-10_SelfReportConsistency_2025-11-25](./EXP-10_SelfReportConsistency_2025-11-25) | Self-Report Consistency | 2025-11-25 | TBD |
| 11 | [EXP-11_ExtendedGeometricSuite_2025-11-27](./EXP-11_ExtendedGeometricSuite_2025-11-27) | Extended Geometric Suite | 2025-11-27 | Qwen-0.5B |
| 12 | [EXP-12_AdvancedDiagnostics_2025-11-29](./EXP-12_AdvancedDiagnostics_2025-11-29) | Advanced Diagnostics | 2025-11-29 | Qwen-0.5B |
| 13 | [EXP-13_FailureSubtyping_2025-12-01](./EXP-13_FailureSubtyping_2025-12-01) | Failure Subtyping | 2025-12-01 | Qwen-0.5B |
| 14 | [EXP-14_UniversalSignature_2025-12-03](./EXP-14_UniversalSignature_2025-12-03) | Universal Signature | 2025-12-03 | Qwen-0.5B |
| 15 | [EXP-15_LengthConfound_2025-12-05](./EXP-15_LengthConfound_2025-12-05) | Length Confound | 2025-12-05 | Qwen-0.5B |
| 16 | [EXP-16_Pythia70m_2025-12-07](./EXP-16_Pythia70m_2025-12-07) | Pythia-70m | 2025-12-07 | Pythia-70m |
| 16B | [EXP-16B_Qwen15B_2025-12-08](./EXP-16B_Qwen15B_2025-12-08) | Qwen2.5-1.5B | 2025-12-08 | Qwen2.5-1.5B |
| 17 | [EXP-17_BaselineReplication_2026-02-11](./EXP-17_BaselineReplication_2026-02-11) | Baseline Replication | 2026-02-11 | TBD |
| 18 | [EXP-18_ConsolidatedMetricSuite_2026-02-13](./EXP-18_ConsolidatedMetricSuite_2026-02-13) | Consolidated Metric Suite | 2026-02-13 | Qwen2.5-0.5B |
| 19 | [EXP-19_Robustness_2026-02-14](./EXP-19_Robustness_2026-02-14) | Robustness Replication | 2026-02-14 | Multi-Arch |

## Folder Structure

Each experiment follows this standardized structure:

```
EXP-<NN>_<ShortTitle>_<YYYY-MM-DD>/
├── data/          # Raw and processed outputs
├── analysis/      # Scripts, notebooks, summaries
├── summary.md     # Concise experiment summary (optional)
└── metadata.md    # Full experimental metadata (optional)
```

## Quick Navigation

- **View all summaries:** See `/experiment_summaries/`
- **Research timeline:** See `/research_history/`
- **Main README:** See project root

## Research Phases

- **Phase 1 (01-05):** Static operator signatures exploration
- **Phase 2 (06-08):** Trajectory geometry and dynamic patterns
- **Phase 3 (09-13):** Scaling metrics and failure analysis
- **Phase 4 (14-16B):** Universal signatures and cross-model validation
