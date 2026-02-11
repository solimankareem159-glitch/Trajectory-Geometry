
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    print("="*60)
    print("06_make_figures.py: Figure Generation (All Pairs)")
    print("="*60)
    
    os.makedirs("experiments/Experiment 16/figures", exist_ok=True)
    
    # Load data
    try:
        df_comp = pd.read_csv("experiments/Experiment 16/data/exp16_comparisons.csv")
        metadata = pd.read_csv("experiments/Experiment 16/data/metadata.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 1. Heatmaps for all comparisons
    # Get unique pairs
    pairs = df_comp[['group1', 'group2']].drop_duplicates().values
    print(f"Found {len(pairs)} unique comparisons to plot.")
    
    for g1, g2 in pairs:
        subset = df_comp[(df_comp['group1'] == g1) & (df_comp['group2'] == g2)]
        if len(subset) == 0: continue
        
        # pivot: index=metric, columns=layer, values=cohens_d
        try:
            pivot = subset.pivot(index='metric', columns='layer', values='cohens_d')
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt=".2f", cbar_kws={'label': "Cohen's d"})
            
            # Title
            desc = subset['comparison'].iloc[0] if 'comparison' in subset.columns else f"{g1} vs {g2}"
            plt.title(f"{desc}\n(Effect Size Cohen's d)")
            plt.xlabel("Layer")
            plt.ylabel("Metric")
            plt.tight_layout()
            
            filename = f"experiments/Experiment 16/figures/heatmap_{g1}_vs_{g2}.png"
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"[OK] Saved {os.path.basename(filename)}")
        except Exception as e:
            print(f"Error plotting {g1} vs {g2}: {e}")

    # 2. Accuracy Plot
    try:
        acc = metadata.groupby('condition')['correct'].mean()
        plt.figure(figsize=(8, 6))
        bars = acc.plot(kind='bar', color=['skyblue', 'orange'], edgecolor='black')
        plt.title("Model Accuracy by Condition (Qwen2.5-1.5B)")
        plt.ylabel("Accuracy")
        plt.xlabel("Condition")
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=0)
        
        # Add values on top
        for p in bars.patches:
            bars.annotate(f"{p.get_height():.2f}", 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', 
                          xytext=(0, 9), 
                          textcoords='offset points')
                          
        plt.tight_layout()
        plt.savefig("experiments/Experiment 16/figures/accuracy_plot.png", dpi=150)
        plt.close()
        print("[OK] Saved accuracy_plot.png")
    except Exception as e:
        print(f"Error plotting accuracy: {e}")
        
    print("\n[OK] Figure generation complete.")

if __name__ == "__main__":
    main()
