# EXP-17: Baseline Replication & Multi-Mode Prompting

**Date:** 2026-02-11
**Model:** Qwen2.5-3B-Instruct
**Task:** Multi-step arithmetic (same as EXP-09)
**Dataset:** 300 problems, operands A∈[10,50], B∈[2,20], C∈[10,100]

## Phases
- **17A:** Replication of EXP-09 with full 33-metric pipeline
- **17B:** 8-mode multi-mode prompting (same content, different computational modes)

## Context
- Extends Qwen family scale ladder: 0.5B (EXP-14) → 1.5B (EXP-16B) → **3B (EXP-17)**
- Tests regime-relative geometry and multi-mode computational signatures
- Hardware: AMD RX 5700 XT (8GB), DirectML
