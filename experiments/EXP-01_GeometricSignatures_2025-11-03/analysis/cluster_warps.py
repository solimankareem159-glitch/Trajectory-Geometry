import pandas as pd
import numpy as np
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

INPUT_FILE = "warps.parquet"
REPORT_FILE = "CLUSTER_REPORT.md"
FIGURE_DIR = "figures"

def ensure_dir(d):
    try:
        os.makedirs(d)
    except FileExistsError:
        pass

def run_clustering_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"{INPUT_FILE} not found.")
        return

    print("Loading warps...")
    df = pd.read_parquet(INPUT_FILE)
    ensure_dir(FIGURE_DIR)
    
    state_defs = df['state_def'].unique()
    
    report_lines = []
    report_lines.append("# Clustering Analysis Report")
    report_lines.append("")
    
    for s_def in state_defs:
        print(f"Analyzing State Definition: {s_def}")
        report_lines.append(f"## State Definition: {s_def}")
        
        subset = df[df['state_def'] == s_def].copy()
        X = np.stack(subset['warp_vector'].values)
        
        labels_true_op = subset['operator_name'].values
        labels_true_topic = subset['topic'].values
        
        # --- HDBSCAN ---
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        try:
            labels_hdb = clusterer.fit_predict(X)
            # -1 are noise
            n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
            
            if n_clusters_hdb > 1:
                sil_hdb = silhouette_score(X, labels_hdb)
            else:
                sil_hdb = -1.0
                
            ami_op_hdb = adjusted_mutual_info_score(labels_true_op, labels_hdb)
            ami_topic_hdb = adjusted_mutual_info_score(labels_true_topic, labels_hdb)
            
            report_lines.append("### HDBSCAN")
            report_lines.append(f"- Clusters Found: {n_clusters_hdb}")
            report_lines.append(f"- Silhouette Score: {sil_hdb:.3f}")
            report_lines.append(f"- AMI (Operator): **{ami_op_hdb:.3f}**")
            report_lines.append(f"- AMI (Topic): {ami_topic_hdb:.3f}")
            
        except Exception as e:
            report_lines.append(f"HDBSCAN Failed: {e}")
            labels_hdb = None

        # --- K-Means (10 seeds) ---
        # Target k=10 (since we have 10 operators)
        k = 10
        print(f"Running K-Means (k={k}) stability test...")
        
        amis_op = []
        amis_topic = []
        sils = []
        
        best_kmeans_labels = None
        best_sil = -1
        
        for seed in range(10):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels_km = kmeans.fit_predict(X)
            
            sil = silhouette_score(X, labels_km)
            sils.append(sil)
            
            amis_op.append(adjusted_mutual_info_score(labels_true_op, labels_km))
            amis_topic.append(adjusted_mutual_info_score(labels_true_topic, labels_km))
            
            if sil > best_sil:
                best_sil = sil
                best_kmeans_labels = labels_km
        
        avg_sil = np.mean(sils)
        std_sil = np.std(sils)
        avg_ami_op = np.mean(amis_op)
        avg_ami_topic = np.mean(amis_topic)
        
        report_lines.append(f"### K-Means (k={k}, 10 seeds)")
        report_lines.append(f"- Avg Silhouette: {avg_sil:.3f} (±{std_sil:.3f})")
        report_lines.append(f"- Avg AMI (Operator): **{avg_ami_op:.3f}**")
        report_lines.append(f"- Avg AMI (Topic): {avg_ami_topic:.3f}")
        
        # Hypothesis Check
        report_lines.append("\n**Hypothesis Check:**")
        if avg_ami_op > avg_ami_topic + 0.1:
            report_lines.append("> ✅ H1 & H3 Supported: Clusters align better with Operators than Topics.")
        else:
            report_lines.append("> ❌ H1/H3 At Risk: Content/Topic signal may be dominating.")
            
        # --- Visualization (TSNE) ---
        print("Generating TSNE plot...")
        try:
            tsne = TSNE(n_components=2, random_state=42)
            X_embedded = tsne.fit_transform(X)
            
            # Plot colored by Operator
            plt.figure(figsize=(10, 8))
            unique_ops = np.unique(labels_true_op)
            for op in unique_ops:
                mask = labels_true_op == op
                plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=op, alpha=0.6, s=30)
            plt.title(f"TSNE - State {s_def} - By Operator")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{FIGURE_DIR}/tsne_{s_def}_operator.png")
            plt.close()
            
            # Plot colored by Topic
            plt.figure(figsize=(10, 8))
            unique_topics = np.unique(labels_true_topic)
            for top in unique_topics:
                mask = labels_true_topic == top
                plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=top, alpha=0.6, s=30)
            plt.title(f"TSNE - State {s_def} - By Topic")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{FIGURE_DIR}/tsne_{s_def}_topic.png")
            plt.close()
            
            # Plot colored by HDBSCAN Cluster
            if labels_hdb is not None:
                plt.figure(figsize=(10, 8))
                plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_hdb, cmap='tab20', alpha=0.6, s=30)
                plt.title(f"TSNE - State {s_def} - HDBSCAN Clusters")
                plt.colorbar(label='Cluster ID')
                plt.tight_layout()
                plt.savefig(f"{FIGURE_DIR}/tsne_{s_def}_hdbscan.png")
                plt.close()

                # Save labeled data for Predictor
                # We add the labels to the subset dataframe
                subset_labeled = subset.copy()
                subset_labeled['cluster_hdb'] = labels_hdb
                
                # Use KMeans labels (best seed) as well as they might be more stable for prediction if HDBSCAN has too much noise
                if best_kmeans_labels is not None:
                    subset_labeled['cluster_kmeans'] = best_kmeans_labels
                
                # Save
                labeled_output = f"labeled_warps_{s_def}.parquet"
                subset_labeled.to_parquet(labeled_output)
                print(f"Saved labeled warps to {labeled_output}")

            report_lines.append("\n**Visualizations:**")
            report_lines.append(f"![TSNE Operator]({FIGURE_DIR}/tsne_{s_def}_operator.png)")
            report_lines.append(f"![TSNE Topic]({FIGURE_DIR}/tsne_{s_def}_topic.png)")
            
        except Exception as e:
            print(f"Plotting failed: {e}")
            
        report_lines.append("\n---\n")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Analysis complete. Report written to {REPORT_FILE}")

if __name__ == "__main__":
    run_clustering_analysis()
