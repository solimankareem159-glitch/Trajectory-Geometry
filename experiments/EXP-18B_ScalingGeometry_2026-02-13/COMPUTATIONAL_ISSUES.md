# Computational Issues Report: EXP-18B Scaling Geometry

This document outlines the critical computational challenges, data integrity failures, and implementation bugs encountered during the execution of Experiment 18B. These issues must be addressed as hard constraints for future experiments.

## 1. Data Integrity: The "Pythia 70M" Identity Crisis

**Severity:** Critical (Blocker)
**Description:**
The directory identifying itself as Pythia 70M data (`EXP-16_Pythia70m_2025-12-07/data/hidden_states_clean`) contained hidden states with dimension **1536** (consistent with Qwen 1.5B) instead of the expected **512**.
**Impact:**

- caused persistent `ValueError: matmul mismatch` errors in worker processes.
- Wasted significant debug time chasing phantom code bugs when the underlying data was corrupted.
- Invalidated all Pythia 70M results for this experiment.
**Constraint for Future Experiments:**
- **Pre-Flight Validation:** All input data must pass a `shape_check` before processing begins. Assert `h.shape[-1] == model.config.hidden_size`.
- **Source Verification:** Do not trust directory names. Verify metadata or config files within the data directory.

## 2. Metric Computation: Broadcasting Ambiguity

**Severity:** High (Crash)
**Description:**
The `attractor_metrics` function failed when handling centroids. It did not distinguish between a single *point* centroid (shape `(D,)`) and a *trajectory* centroid (shape `(T, D)`).

- When subtracting a point `(D,)` from a trajectory `(T, D)`, numpy broadcasting worked fine.
- When subtracting a trajectory `(T_ref, D)` from a trajectory `(T, D)` where `T != T_ref`, it caused a crash or incorrect broadcasting if `T_ref == D`.
**Impact:**
- Crashed worker processes silently or overly noisily.
**Constraint for Future Experiments:**
- **Explicit Dimensions:** Centroids must be explicitly typed or flagged as `static` vs `dynamic`.
- **Robust Broadcasting:** Use `np.newaxis` explicitly or validate shapes before element-wise operations.

## 3. Multiprocessing on Windows

**Severity:** Medium (Performance/Stability)
**Description:**
Windows uses `spawn` start method for multiprocessing, which re-imports the main script in every worker.

- **Resource Heaviness:** Importing `torch` and `transformers` in the global scope caused massive overhead and memory usage for *each* of the 4-16 workers.
- **init_worker Failure:** Passing large objects (like full `W_U` matrices) to workers via `initializer` relies on pickling, which is slow and memory-intensive.
**Impact:**
- High memory pressure, slow startup times.
- Difficulty keeping "global" state (like the `TrajectoryMetrics` object) consistent.
**Constraint for Future Experiments:**
- **Lazy Imports:** Heavy libraries (`torch`, `transformers`) must be imported *inside* the `main()` function or task function, not globally.
- **Vector Injection:** Pass only the *necessary vectors* (truth, target) to the worker function, as strictly typed numpy arrays. Do not pass entire model matrices.
- **Worker Cap:** Limit `max_workers` to `min(4, cpu_count - 2)` on Windows to prevent UI freeze and OOM.

## 4. Resume & Persistence

**Severity:** Medium (Operational)
**Description:**
Initial scripts attempted to process all files in one go and save at the end.

- **Data Loss:** Any crash (like the broadcasting error) resulted in losing *all* progress for that model.
- **Wasted Compute:** Restarting required re-calculating thousands of expensive metrics.
**Implementation Fix:**
- **Incremental Saving:** Append results to CSV after *each* task completion.
- **Check-and-Skip:** Read the existing CSV at startup to identify and skip already-processed file IDs.
**Constraint for Future Experiments:**
- **Atomic Operations:** All long-running sweeps MUST implement incremental saving (append mode) and resume logic by default.

## 5. Model State Leakage

**Severity:** Low (Risk)
**Description:**
When processing multiple models in a loop, there was a risk of `transformers` or `torch` retaining VRAM or global state.
**Constraint for Future Experiments:**

- **Process Isolation:** Ideally, launch a separate *OS process* (e.g., via `subprocess.run`) for each model configuration rather than looping within one python script. This guarantees a clean memory slate.
