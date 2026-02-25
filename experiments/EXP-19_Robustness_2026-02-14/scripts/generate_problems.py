import random
import json
import os
import pandas as pd

# Set seeds for absolute reproducibility
random.seed(42)

BINS = {
    'UltraSmall': {'a': (1, 5),    'b': (1, 5),    'c': (1, 5),    'negate': False},
    'Small':      {'a': (2, 9),    'b': (2, 9),    'c': (1, 9),    'negate': False},
    'Medium':     {'a': (10, 49),  'b': (2, 19),   'c': (10, 49),  'negate': False},
    'Large':      {'a': (10, 99),  'b': (10, 99),  'c': (10, 99),  'negate': False},
    'Negative':   {'a': (10, 49),  'b': (2, 19),   'c': (10, 49),  'negate': True},
}

OPS = {
    'A': lambda a,b,c: (f"({a} * {b}) + {c}", a*b + c, a*b),
    'B': lambda a,b,c: (f"({a} * {b}) - {c}", a*b - c, a*b),
    'C': lambda a,b,c: (f"{a} * ({b} + {c})", a*(b+c), b+c),
    'D': lambda a,b,c: (f"{a} * ({b} - {c})", a*(b-c), b-c),
}

def generate_problem(bin_name, bin_cfg, op_name, problem_id):
    while True:
        a = random.randint(*bin_cfg['a'])
        b = random.randint(*bin_cfg['b'])
        c = random.randint(*bin_cfg['c'])

        if bin_cfg['negate']:
            # Randomly negate exactly one operand
            idx = random.choice([0, 1, 2])
            if idx == 0: a = -a
            elif idx == 1: b = -b
            else: c = -c

        question, truth, intermediate = OPS[op_name](a, b, c)

        # Validation rules:
        # 1. Intermediate result must not be zero
        if intermediate == 0:
            continue
        # 2. Answer must not equal an operand (avoids trivial retrieval)
        if truth in (a, b, c):
            continue
        # 3. Intermediate must not equal an operand
        if intermediate in (a, b, c):
             continue

        return {
            'id': problem_id,
            'bin': bin_name,
            'op_type': op_name,
            'a': a, 'b': b, 'c': c,
            'question': question,
            'truth': truth,
            'intermediate': intermediate,
            'answer_sign': 'positive' if truth >= 0 else 'negative',
            'answer_digits': len(str(abs(truth))),
        }

def main():
    problems = []
    pid = 0
    # Process Bins
    for bin_name, bin_cfg in BINS.items():
        op_names = list(OPS.keys())
        # Generate 80 per bin to keep total around 400
        for i in range(80):
            op = op_names[i % 4]  # Cycle through A, B, C, D
            prob = generate_problem(bin_name, bin_cfg, op, pid)
            problems.append(prob)
            pid += 1

    # Final validation
    assert len(problems) == 400
    for bin_name in BINS:
        bin_probs = [p for p in problems if p['bin'] == bin_name]
        assert len(bin_probs) == 80
        for op in 'ABCD':
            assert len([p for p in bin_probs if p['op_type'] == op]) == 20

    # Save outputs
    output_dir = "experiments/EXP-19_Robustness_2026-02-14/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON and CSV
    with open(f"{output_dir}/problems.json", 'w') as f:
        json.dump(problems, f, indent=2)
    
    df = pd.DataFrame(problems)
    df.to_csv(f"{output_dir}/problems.csv", index=False)
    
    print(f"Generated {len(problems)} problems.")
    print(f"Answer sign distribution: {sum(1 for p in problems if p['answer_sign']=='positive')} positive, "
          f"{sum(1 for p in problems if p['answer_sign']=='negative')} negative.")
    print(f"Problem set saved to {output_dir}/problems.json and {output_dir}/problems.csv")

if __name__ == "__main__":
    import os
    print(f"PID: {os.getpid()}", flush=True)
    main()
