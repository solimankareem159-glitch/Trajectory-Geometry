import csv
import os

def check_metadata(path):
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    
    print(f"\nScanning {path}...")
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        count = 0
        qwen_halls = 0
        potential_pythia = 0
        
        for row in reader:
            count += 1
            res = row.get('response', '')
            # Qwen hallmark: Runaway repetitions or specific patterns
            if "10000000000" in res or "Question: Calculate" in res or "A:" in res:
                qwen_halls += 1
            else:
                # Potential Pythia: Usually shorter, more direct or garbage
                potential_pythia += 1
                if potential_pythia < 5:
                    print(f"  [ID {row['problem_id']} {row['condition']}] Potential non-Qwen: {res[:50]}...")

        print(f"  Total rows: {count}")
        print(f"  Qwen-style halls detected: {qwen_halls}")
        print(f"  Potential alternative rows: {potential_pythia}")

def main():
    paths = [
        "experiments/Experiment 16/data/metadata.csv",
        "experiments/Experiment 16/data/checkpoints/metadata_checkpoint_1.csv",
        "experiments/Experiment 16/data/checkpoints/metadata_checkpoint_2.csv",
        "experiments/Experiment 16/data/checkpoints/metadata_checkpoint_3.csv",
    ]
    for p in paths:
        check_metadata(p)

if __name__ == "__main__":
    main()
