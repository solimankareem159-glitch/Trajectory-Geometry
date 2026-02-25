import os
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform
import multiprocessing as mp
from tqdm import tqdm
import argparse

# --- METRIC FUNCTIONS ---

def compute_kinematic_metrics(h):
    """Family 1: Kinematic Metrics"""
    T, D = h.shape
    if T < 2: return {}
    
    # Steps
    delta = np.diff(h, axis=0) # [T-1, D]
    step_mags = np.linalg.norm(delta, axis=1)
    
    # 1.1 Speed
    speed = np.mean(step_mags)
    
    # 1.2 Turn Angle
    # theta = arccos( (d_t . d_t+1) / (|d_t| |d_t+1|) )
    turn_angles = []
    for i in range(len(delta) - 1):
        v1 = delta[i]
        v2 = delta[i+1]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-9 and norm2 > 1e-9:
            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
            turn_angles.append(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    turn_angle = np.mean(turn_angles) if turn_angles else 0.0
    
    # 1.3 Tortuosity
    net_disp = np.linalg.norm(h[-1] - h[0])
    total_path = np.sum(step_mags)
    tortuosity = net_disp / total_path if total_path > 1e-9 else 1.0
    
    # 1.4 Directional Consistency
    norm_delta = delta / (step_mags[:, None] + 1e-9)
    mean_dir = np.mean(norm_delta, axis=0)
    dir_consistency = np.linalg.norm(mean_dir)
    
    # 1.5 Stabilisation Rate
    if len(step_mags) > 2:
        slope, _, _, _, _ = linregress(np.arange(len(step_mags)), step_mags)
        stabilisation = slope
    else:
        stabilisation = 0.0
        
    return {
        'speed': speed,
        'turn_angle': turn_angle,
        'tortuosity': tortuosity,
        'dir_consistency': dir_consistency,
        'stabilisation': stabilisation
    }

def compute_volumetric_metrics(h):
    """Family 2: Volumetric Metrics"""
    T, D = h.shape
    if T < 2: return {}
    
    # 2.1 Radius of Gyration
    centroid = np.mean(h, axis=0)
    rg = np.sqrt(np.mean(np.sum((h - centroid)**2, axis=1)))
    
    # 2.2 Effective Dimension
    # Participation ratio of PCA eigenvalues of delta
    delta = np.diff(h, axis=0)
    if len(delta) > 2:
        centered_delta = delta - np.mean(delta, axis=0)
        # Use SVD for stability instead of covariance
        _, s, _ = np.linalg.svd(centered_delta, full_matrices=False)
        lambdas = s**2 / (len(delta) - 1)
        eff_dim = (np.sum(lambdas)**2) / np.sum(lambdas**2) if np.sum(lambdas**2) > 1e-9 else 1.0
    else:
        eff_dim = 1.0
        
    return {
        'radius_of_gyration': rg,
        'effective_dimension': eff_dim
    }

def compute_convergence_metrics(h):
    """Family 3: Convergence Metrics"""
    T, D = h.shape
    if T < 4: return {}
    
    # 3.1 Cosine Slope to Final
    final = h[-1]
    cosines = []
    for t in range(T-1):
        norm_t = np.linalg.norm(h[t])
        norm_f = np.linalg.norm(final)
        if norm_t > 1e-9 and norm_f > 1e-9:
            cosines.append(np.dot(h[t], final) / (norm_t * norm_f))
    
    if len(cosines) > 2:
        cos_slope, _, _, _, _ = linregress(np.arange(len(cosines)), cosines)
    else:
        cos_slope = 0.0
        
    # 3.4 Time to Commit (Token where Rg drop is max)
    # Using windowed Rg
    window_size = min(6, T // 2)
    drops = []
    if T > window_size:
        window_rgs = []
        for i in range(T - window_size + 1):
            sub = h[i : i + window_size]
            sub_centroid = np.mean(sub, axis=0)
            window_rgs.append(np.sqrt(np.mean(np.sum((sub - sub_centroid)**2, axis=1))))
        
        drops = np.diff(window_rgs)
        if len(drops) > 0:
            time_to_commit = np.argmin(drops)
            commitment_sharpness = -np.min(drops) # Drop is negative, so negate it
        else:
            time_to_commit = 0
            commitment_sharpness = 0.0
    else:
        time_to_commit = 0
        commitment_sharpness = 0.0
        
    return {
        'cos_slope_to_final': cos_slope,
        'time_to_commit': time_to_commit,
        'commitment_sharpness': commitment_sharpness
    }

def compute_phase_metrics(h):
    """Detect phase transitions using Effective Dimension shifts."""
    T, D = h.shape
    if T < 10: return {'phase_count': 1}
    
    # Compute windowed effective dimension
    window_size = 6
    eff_dims = []
    for i in range(T - window_size + 1):
        sub = h[i : i + window_size]
        centered = sub - np.mean(sub, axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        lambdas = s**2 / (window_size - 1)
        ed = (np.sum(lambdas)**2) / np.sum(lambdas**2) if np.sum(lambdas**2) > 1e-9 else 1.0
        eff_dims.append(ed)
    
    # Phase Count: Number of major shifts in effective dimension
    # Using a simple threshold for mean-crossing
    eff_dims = np.array(eff_dims)
    avg_ed = np.mean(eff_dims)
    crossings = np.where(np.diff(np.sign(eff_dims - avg_ed)))[0]
    phase_count = len(crossings) + 1
    
    return {
        'phase_count': phase_count
    }

def compute_diffusion_metrics(h):
    """Family 4: Diffusion Metrics"""
    T, D = h.shape
    if T < 10: return {} # Need enough points for log-log fit
    
    # 4.1 MSD Exponent (alpha)
    # MSD(tau) = < |h(t+tau) - h(t)|^2 >
    lags = np.arange(1, min(20, T // 2))
    msds = []
    for tau in lags:
        diffs = h[tau:] - h[:-tau]
        msds.append(np.mean(np.sum(diffs**2, axis=1)))
    
    # Fit log(MSD) ~ alpha * log(tau)
    log_lags = np.log(lags)
    log_msds = np.log(np.array(msds) + 1e-9)
    alpha, _, _, _, _ = linregress(log_lags, log_msds)
    
    return {
        'msd_exponent': alpha,
        'fractal_dimension': 2.0 / alpha if alpha > 0 else 0.0 # Approximation
    }

def compute_cross_layer_metrics(data_slice):
    """Family 6: Cross-Layer Metrics"""
    # data_slice: [L, T, D]
    L, T, D = data_slice.shape
    if L < 2 or T < 2: return {}
    
    # Update vectors in each layer: [L, T-1, D]
    deltas = np.diff(data_slice, axis=1)
    
    alignments = []
    for l in range(L - 1):
        # Cosine similarity between deltas of layer l and l+1
        d1 = deltas[l]   # [T-1, D]
        d2 = deltas[l+1] # [T-1, D]
        
        norms1 = np.linalg.norm(d1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(d2, axis=1, keepdims=True)
        
        # Avoid div by zero
        n1 = np.clip(norms1, 1e-9, None)
        n2 = np.clip(norms2, 1e-9, None)
        
        cos_sims = np.sum((d1 / n1) * (d2 / n2), axis=1)
        alignments.append(np.mean(cos_sims))
        
    return {
        'interlayer_alignment': np.mean(alignments)
    }

# --- PROCESSOR ---

def process_single_trajectory(args_tuple):
    model_key, pid, condition, hdd_root, ssd_root, metadata_row = args_tuple
    
    # Load .npy
    traj_path = os.path.join(hdd_root, "data", "hidden_states", model_key, metadata_row['filename'])
    if not os.path.exists(traj_path):
        return None
    
    try:
        data = np.load(traj_path).astype(np.float32) # [L, T, D]
    except Exception as e:
        print(f"Error loading {traj_path}: {e}")
        return None
    
    L, T, D = data.shape
    ans_idx = int(metadata_row['answer_token_idx'])
    
    traj_results = []
    
    # 1. CLEAN Trajectory Metrics
    m_clean_cross = compute_cross_layer_metrics(data[:, :ans_idx+1, :])
    
    # 2. FULL Trajectory Metrics
    m_full_cross = compute_cross_layer_metrics(data)
    
    for l in range(L):
        # We compute for TWO versions: CLEAN and FULL
        # Clean: 0 to ans_idx
        # Full: 0 to T-1
        
        h_clean = data[l, :ans_idx+1, :]
        h_full = data[l, :, :]
        
        m_clean = {}
        m_clean.update(compute_kinematic_metrics(h_clean))
        m_clean.update(compute_volumetric_metrics(h_clean))
        m_clean.update(compute_convergence_metrics(h_clean))
        m_clean.update(compute_diffusion_metrics(h_clean))
        m_clean.update(compute_phase_metrics(h_clean))
        
        m_full = {}
        m_full.update(compute_kinematic_metrics(h_full))
        m_full.update(compute_volumetric_metrics(h_full))
        m_full.update(compute_convergence_metrics(h_full))
        m_full.update(compute_diffusion_metrics(h_full))
        m_full.update(compute_phase_metrics(h_full))
        
        layer_res = {
            'problem_id': pid,
            'condition': condition,
            'layer': l,
            'model': model_key
        }
        # Flatten and prefix
        for k, v in m_clean.items(): layer_res[f'clean_{k}'] = v
        for k, v in m_full.items(): layer_res[f'full_{k}'] = v
        
        # Add cross-layer metrics (replicated for all layers for ease of table join)
        layer_res['clean_interlayer_alignment'] = m_clean_cross.get('interlayer_alignment', 0.0)
        layer_res['full_interlayer_alignment'] = m_full_cross.get('interlayer_alignment', 0.0)
        
        traj_results.append(layer_res)
        
    return traj_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", type=str, default="experiments/EXP-19_Robustness_2026-02-14")
    parser.add_argument("--hdd_root", type=str, default=r"D:\Dev\Projects Cold Storage\Trajectory Geometry\EXP-19_Robustness_2026-02-14")
    parser.add_argument("--n_workers", type=int, default=os.cpu_count() - 1)
    args = parser.parse_args()
    
    data_dir = os.path.join(args.ssd_root, "data")
    model_keys = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    output_path = os.path.join(data_dir, "all_metrics.csv")
    
    all_tasks = []
    for m_key in model_keys:
        meta_path = os.path.join(data_dir, m_key, "metadata.csv")
        if not os.path.exists(meta_path): continue
        
        df = pd.read_csv(meta_path)
        # Only process if has trajectory
        df = df[df['has_trajectory'] == True]
        
        for _, row in df.iterrows():
            all_tasks.append((m_key, row['problem_id'], row['condition'], args.hdd_root, args.ssd_root, row.to_dict()))
            
    print(f"Starting metric computation for {len(all_tasks)} trajectories using {args.n_workers} workers...")
    
    results_list = []
    with mp.Pool(args.n_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_single_trajectory, all_tasks), total=len(all_tasks)):
            if res:
                results_list.extend(res)
    
    if results_list:
        final_df = pd.DataFrame(results_list)
        final_df.to_csv(output_path, index=False)
        print(f"Saved {len(final_df)} layer-metrics to {output_path}")
    else:
        print("No metrics computed.")

if __name__ == "__main__":
    main()
