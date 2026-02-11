# Experiment 16B: Exact Replication of Experiment 9

## Objective
Replicate Experiment 9's methodology using Qwen2.5-1.5B instead of Qwen2.5-0.5B, with **exact** matching of:
- Questions (all 301 from Experiment 9 dataset)
- Prompts (word-for-word identical)
- Generation parameters (32/128 max_tokens, greedy decoding)
- Output format (identical JSONL structure)

## Key Differences from Experiment 16
1. **No batching** - single-sample processing to avoid padding artifacts
2. **DirectML venv** (Python 3.12) for proper GPU acceleration
3. **Hallucination monitoring** - detects and logs contamination patterns
4. **All 28 layers** - extracts hidden states from complete model depth

## Running the Experiment

### Prerequisites
- DirectML virtual environment activated (`.venv-directml`)
- Experiment 9 dataset available at `experiments/Experiment 9/data/exp9_dataset.jsonl`

### Execution
```powershell
# Activate your virtual environment first
# e.g.: & ".venv-directml\Scripts\Activate.ps1"

# Run inference (single-sample, no batching)
python "experiments/EXP-16B_Qwen15B_2025-12-08/scripts/01_run_inference_single_sample.py"
```

### Expected Runtime
- ~30-45 minutes on DirectML (AMD GPU)
- ~3-4 hours on CPU (Python 3.13 fallback)

## Outputs

1. **Dataset** (`data/exp16b_dataset.jsonl`):
   - Identical format to Experiment 9
   - Includes hallucination flags per response
   
2. **Hidden States** (`data/hidden_states/*.npy`):
   - 602 files (301 problems × 2 conditions)
   - Format: `problem_XXX_direct.npy` and `problem_XXX_cot.npy`
   - Shape: `[28 layers, n_tokens, 1536 hidden_dim]`

3. **Hallucination Log** (`data/hallucination_log.json`):
   - Detailed list of detected contamination
   - Types: `new_question`, `large_repeated_digits`, `very_large_number`

## Validation Checklist

After running, verify:
- [ ] Direct accuracy similar to Experiment 9 (~10-20%)
- [ ] CoT accuracy similar to Experiment 9 (~70-85%)
- [ ] **No hallucinations detected** (or minimal, <5%)
- [ ] All 602 .npy files exist in `hidden_states/`
- [ ] JSONL file has 301 lines

## Next Steps

If validation passes:
1. Compute all 31 metrics (from Experiment 14)
2. Run statistical tests
3. Generate figures
4. Compile final report

If contamination detected:
- Review `hallucination_log.json`
- Consider CPU fallback (Python 3.13) for cleaner generation
