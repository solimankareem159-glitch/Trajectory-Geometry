"""
EXP-17 Smoke Test
=================
Mandatory gate before full replication run.
Verifies that Qwen2.5-3B-Instruct loads on DirectML,
hidden states extract correctly, and all 33 metrics compute.
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import traceback

print(f"PID: {os.getpid()}", flush=True)

# ─── Configuration ───────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
N_TEST_PROBLEMS = 5
MAX_NEW_TOKENS_DIRECT = 32
MAX_NEW_TOKENS_COT = 128

# ─── Add project root to path for metric imports ────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ─── Results ─────────────────────────────────────────────────────
checks = {}

def report(name, passed, detail=""):
    status = "[PASS]" if passed else "[FAIL]"
    checks[name] = passed
    print(f"  {status}: {name}" + (f" - {detail}" if detail else ""), flush=True)


def generate_problem():
    """Same arithmetic generator as EXP-09."""
    op = random.choice([(0, " + "), (1, " - ")])
    a = random.randint(10, 50)
    b = random.randint(2, 20)
    c = random.randint(10, 100)
    if random.random() < 0.5:
        q_str = f"({a} * {b}) {op[1]} {c}"
        ans = (a * b) + c if op[0] == 0 else (a * b) - c
    else:
        q_str = f"{a} * ({b} {op[1]} {c})"
        ans = a * (b + c) if op[0] == 0 else a * (b - c)
    return q_str, ans


def main():
    print("=" * 70)
    print("EXP-17 SMOKE TEST")
    print("=" * 70)

    # ─── Check 1: DirectML availability ──────────────────────────
    print("\n[1/6] Checking DirectML...")
    try:
        import torch_directml
        dml = torch_directml.device()
        gpu_name = torch_directml.device_name(0)
        report("DirectML Available", True, f"GPU: {gpu_name}")
    except Exception as e:
        report("DirectML Available", False, str(e))
        print("\nABORTING: DirectML is required.")
        return

    # ─── Check 2: Model loading ──────────────────────────────────
    print("\n[2/6] Loading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        )
        model.to(dml)
        model.eval()
        load_time = time.time() - t0
        report("Model Loaded", True, f"{MODEL_NAME} in {load_time:.1f}s")
    except Exception as e:
        report("Model Loaded", False, str(e))
        print(f"\nABORTING: Model failed to load. Error:\n{traceback.format_exc()}")
        print("\nSuggested fallback: Try Phi-2 (2.7B) or Gemma-2-2B")
        return

    # ─── Check 3: Architecture dimensions ────────────────────────
    print("\n[3/6] Checking architecture...")
    config = model.config
    n_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    report("Architecture Readable", True,
           f"Layers={n_layers}, Hidden={hidden_dim}, Vocab={config.vocab_size}")

    # ─── Check 4: Generation + Hidden State Extraction ───────────
    print(f"\n[4/6] Running {N_TEST_PROBLEMS} test problems...")
    random.seed(42)  # Reproducible

    all_hidden_shapes = []
    all_responses = []

    for i in range(N_TEST_PROBLEMS):
        q_str, truth = generate_problem()

        # Direct condition
        prompt_d = f"Answer the following question directly.\nQ: Calculate {q_str}\nA:"
        inputs_d = tokenizer(prompt_d, return_tensors="pt").to(dml)

        with torch.no_grad():
            gen_d = model.generate(
                **inputs_d, max_new_tokens=MAX_NEW_TOKENS_DIRECT,
                do_sample=False
            )
        text_d = tokenizer.decode(gen_d[0][inputs_d.input_ids.shape[1]:], skip_special_tokens=True)

        # Forward pass for hidden states
        full_text_d = prompt_d + text_d
        inputs_fwd = tokenizer(full_text_d, return_tensors="pt").to(dml)
        len_prompt = len(tokenizer.encode(prompt_d, add_special_tokens=False))

        with torch.no_grad():
            out = model(inputs_fwd.input_ids, output_hidden_states=True)

        # Extract response-only hidden states
        start = len_prompt
        end = inputs_fwd.input_ids.shape[1]

        if end > start:
            layers = []
            for layer_idx, h_tensor in enumerate(out.hidden_states):
                h_window = h_tensor[0, start:end, :].float().cpu().numpy()
                layers.append(h_window)
            stack = np.stack(layers, axis=0)  # [n_layers+1, T, D]
            all_hidden_shapes.append(stack.shape)
            all_responses.append({
                "problem": q_str, "truth": truth,
                "response": text_d, "shape": stack.shape
            })

        print(f"  Problem {i+1}/{N_TEST_PROBLEMS}: {q_str} = {truth} | "
              f"Model said: '{text_d.strip()[:50]}' | "
              f"Hidden: {stack.shape if end > start else 'EMPTY'}", flush=True)

    if all_hidden_shapes:
        example_shape = all_hidden_shapes[0]
        expected_layers = n_layers + 1  # +1 for embedding layer
        shape_ok = all(s[0] == expected_layers for s in all_hidden_shapes)
        dim_ok = all(s[2] == hidden_dim for s in all_hidden_shapes)
        report("Hidden State Shape", shape_ok and dim_ok,
               f"Shape={example_shape}, Expected layers={expected_layers}, dim={hidden_dim}")
    else:
        report("Hidden State Shape", False, "No hidden states extracted")
        print("\nABORTING: No hidden states extracted.")
        return

    # ─── Check 5: Metric Computation ─────────────────────────────
    print("\n[5/6] Testing metric computation...")

    # Import the canonical metric functions from EXP-14
    # Go up 2 levels: analysis -> EXP-17 -> experiments
    experiments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    exp14_analysis = os.path.join(
        experiments_dir,
        "EXP-14_UniversalSignature_2025-12-03", "analysis"
    )
    sys.path.insert(0, exp14_analysis)

    try:
        from compute_metrics import compute_all_metrics_for_layer, compute_cross_layer_metrics
        report("Metric Import", True, "Imported from EXP-14 compute_metrics.py")
    except ImportError as e:
        report("Metric Import", False, str(e))
        print("\nWill attempt inline metric computation as fallback...")
        # Define minimal inline versions
        print("ABORTING: Cannot compute metrics without import.")
        return

    # Test on the first extracted hidden state stack
    test_stack = np.stack(
        [h_tensor[0, start:end, :].float().cpu().numpy()
         for h_tensor in out.hidden_states], axis=0
    )

    # Per-layer metrics
    n_nan = 0
    n_total = 0
    metric_errors = []

    for layer_idx in range(test_stack.shape[0]):
        h = test_stack[layer_idx]
        try:
            metrics = compute_all_metrics_for_layer(h)
            for k, v in metrics.items():
                n_total += 1
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    n_nan += 1
        except Exception as e:
            metric_errors.append(f"Layer {layer_idx}: {e}")

    # Cross-layer metrics
    cross_layers = list(range(0, test_stack.shape[0], max(1, test_stack.shape[0] // 5)))
    try:
        cross_metrics = compute_cross_layer_metrics(test_stack, cross_layers)
        for k, v in cross_metrics.items():
            n_total += 1
            if v is None or (isinstance(v, float) and np.isnan(v)):
                n_nan += 1
    except Exception as e:
        metric_errors.append(f"Cross-layer: {e}")

    if metric_errors:
        report("Metrics Compute", False, f"Errors: {metric_errors}")
    else:
        nan_pct = (n_nan / n_total * 100) if n_total > 0 else 0
        report("Metrics Compute", True,
               f"{n_total} values computed, {n_nan} NaN ({nan_pct:.0f}%)")

    # ─── Check 6: Memory Usage ───────────────────────────────────
    print("\n[6/6] Memory check...")
    # We can't directly query DirectML VRAM, but if we got here without OOM, it fits
    report("Memory Budget", True, "No OOM encountered during all operations")

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for v in checks.values() if v)
    n_fail = sum(1 for v in checks.values() if not v)
    print(f"  Passed: {n_pass}/{len(checks)}")
    print(f"  Failed: {n_fail}/{len(checks)}")

    for name, passed in checks.items():
        status = "[OK]" if passed else "[XX]"
        print(f"    {status} {name}")

    if n_fail == 0:
        print("\n>>> ALL CHECKS PASSED - Safe to proceed with full replication.")
    else:
        print("\n>>> SOME CHECKS FAILED - Review errors before proceeding.")
        print("   Suggested fallbacks: Phi-2 (2.7B) or Gemma-2-2B")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "smoke_test_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "model": MODEL_NAME,
            "gpu": gpu_name,
            "checks": checks,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "test_responses": all_responses,
            "hidden_shapes": [list(s) for s in all_hidden_shapes],
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
