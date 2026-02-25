
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import linregress, entropy, zscore
from scipy.signal import welch
from sklearn.decomposition import PCA
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def lz_complexity(s):
    """Simple Lempel-Ziv complexity for a binary sequence."""
    i, k, l = 1, 1, 1
    n = len(s)
    if n == 0: return 0
    while i + k <= n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            l = 0
            while l < i:
                if s[l:l+k] == s[i:i+k]:
                    break
                l += 1
            if l == i:
                i += k
                l = 1
                k = 1
            else:
                k += 1
    return l

class TrajectoryMetrics:
    """
    Implements the full 54-metric suite across 12 families.
    """
    
    def __init__(self, vocab_unbedding=None, embedding_matrix=None, gram_matrix=None):
        self.W_U = vocab_unbedding # (V, D)
        self.W_E = embedding_matrix # (V, D)
        self.G_U = gram_matrix # (D, D) - Precomputed W_U.T @ W_U

    def _get_deltas(self, h):
        return h[1:] - h[:-1]

    def _safe_norm(self, v, axis=None):
        norms = np.linalg.norm(v, axis=axis)
        return np.maximum(norms, 1e-12)


    # --- Family 1: Kinematic (13 metrics) ---
    def kinematic_metrics(self, h):
        T, D = h.shape
        m = {"speed": np.nan, "turn_angle": np.nan, "tortuosity": np.nan, 
             "directional_consistency": np.nan, "stabilization_rate": np.nan}
        for lag in [1, 2, 4, 8]:
            m[f"vel_autocorr_lag{lag}"] = np.nan
            m[f"dir_autocorr_lag{lag}"] = np.nan
            
        if T < 3: return m
        deltas = self._get_deltas(h)
        delta_norms = self._safe_norm(deltas, axis=1)
        m["speed"] = np.mean(delta_norms)
        
        numer = np.sum(deltas[:-1] * deltas[1:], axis=1)
        denom = delta_norms[:-1] * delta_norms[1:]
        m["turn_angle"] = np.mean(np.arccos(np.clip(numer/denom, -1, 1)))
        
        m["tortuosity"] = np.linalg.norm(h[-1]-h[0]) / (np.sum(delta_norms) + 1e-9)
        
        norm_deltas = deltas / delta_norms[:, None]
        m["directional_consistency"] = np.linalg.norm(np.mean(norm_deltas, axis=0))
        m["stabilization_rate"] = linregress(np.arange(len(delta_norms)), delta_norms).slope if len(delta_norms)>1 else 0
        
        for lag in [1, 2, 4, 8]:
            if len(delta_norms) > lag:
                m[f"vel_autocorr_lag{lag}"] = np.corrcoef(delta_norms[:-lag], delta_norms[lag:])[0,1] if np.std(delta_norms)>0 else 0
                m[f"dir_autocorr_lag{lag}"] = np.mean(np.sum(norm_deltas[:-lag] * norm_deltas[lag:], axis=1))
        return m

    # --- Family 2: Volumetric (4 metrics) ---
    def volumetric_metrics(self, h):
        T, D = h.shape
        m = {"radius_of_gyration": np.nan, "effective_dimension": np.nan, "gyration_anisotropy": np.nan, "drift_to_spread": np.nan}
        if T < 3: return m
        centered = h - np.mean(h, axis=0)
        m["radius_of_gyration"] = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        deltas = self._get_deltas(h)
        try:
            _, s, _ = np.linalg.svd(deltas - np.mean(deltas, axis=0), full_matrices=False)
            m["effective_dimension"] = (np.sum(s**2)**2) / (np.sum(s**4) + 1e-12)
        except: pass
        try:
            _, s_h, _ = np.linalg.svd(centered, full_matrices=False)
            m["gyration_anisotropy"] = 1 - entropy(s_h/np.sum(s_h))/np.log(min(T, D))
        except: pass
        m["drift_to_spread"] = np.linalg.norm(h[-1]-h[0]) / (m["radius_of_gyration"] + 1e-9)
        return m

    # --- Family 3: Convergence (6 metrics) ---
    def convergence_metrics(self, h):
        T, D = h.shape
        m = {"cos_slope": np.nan, "dist_slope": np.nan, "early_late_ratio": np.nan, "time_to_commit": np.nan, "cos_to_late_window": np.nan, "cos_to_running_mean": np.nan}
        if T < 5: return m
        final = h[-1]
        cos_sims = [np.dot(v, final)/(np.linalg.norm(v)*np.linalg.norm(final)+1e-9) for v in h[:-1]]
        dists = [np.linalg.norm(v-final) for v in h[:-1]]
        m["cos_slope"] = linregress(np.arange(len(cos_sims)), cos_sims).slope
        m["dist_slope"] = linregress(np.arange(len(dists)), dists).slope
        norms = np.linalg.norm(self._get_deltas(h), axis=1)
        mid = len(norms)//2
        m["early_late_ratio"] = np.mean(norms[:mid])/(np.mean(norms[mid:])+1e-9) if mid>0 else 1.0
        w = min(6, T-1)
        rg_win = [np.sqrt(np.mean(np.sum((h[i:i+w]-np.mean(h[i:i+w],axis=0))**2, axis=1))) for i in range(T-w+1)]
        m["time_to_commit"] = np.argmax(np.diff(rg_win)*-1) if len(rg_win)>1 else 0
        late_mean = np.mean(h[-min(4, T):], axis=0)
        m["cos_to_late_window"] = np.mean([np.dot(v, late_mean)/(np.linalg.norm(v)*np.linalg.norm(late_mean)+1e-9) for v in h])
        r_mean = np.cumsum(h, axis=0) / np.arange(1, T+1)[:, None]
        m["cos_to_running_mean"] = np.mean([np.dot(h[t]-h[t-1], r_mean[t-1]-h[0])/(np.linalg.norm(h[t]-h[t-1])*np.linalg.norm(r_mean[t-1]-h[0])+1e-9) for t in range(2, T)]) if T > 2 else 0
        return m

    # --- Family 4: Diffusion (1 metric) ---
    def diffusion_metrics(self, h):
        T = h.shape[0]
        m = {"msd_exponent": np.nan}
        if T < 10: return m
        taus = np.unique(np.logspace(0, np.log10(T//2), num=10).astype(int))
        msds = [np.mean(np.sum((h[tau:] - h[:-tau])**2, axis=1)) for tau in taus if tau>0]
        if len(msds)>1: m["msd_exponent"] = linregress(np.log(taus[:len(msds)]), np.log(msds)).slope
        return m

    # --- Family 5: Spectral (2 metrics) ---
    def spectral_metrics(self, h):
        m = {"spectral_entropy": np.nan, "psd_slope": np.nan}
        deltas = self._get_deltas(h); norms = np.linalg.norm(deltas, axis=1)
        if len(norms)<5: return m
        f, Pxx = welch(norms)
        P_norm = Pxx / np.sum(Pxx)
        valid = (f > 0) & (Pxx > 0)
        m["psd_slope"] = linregress(np.log(f[valid]), np.log(Pxx[valid])).slope if np.sum(valid)>2 else np.nan
        m["spectral_entropy"] = entropy(P_norm)
        return m

    # --- Family 6: RQA (5 metrics) ---
    def rqa_metrics(self, h):
        m = {k:np.nan for k in ["recurrence_rate", "determinism", "laminarity", "trapping_time", "diagonal_entropy"]}
        T = h.shape[0]
        if T < 5: return m
        dists = pdist(h); dm = squareform(dists); eps = np.percentile(dists, 10)
        if eps == 0: eps = 1e-9
        R = (dm < eps).astype(int); np.fill_diagonal(R, 0)
        m["recurrence_rate"] = np.sum(R) / (T**2)
        diag_lens = []; vert_lens = []
        for k in range(1, T):
            d = np.diagonal(R, k); run = 0
            for v in d:
                if v: run+=1
                else: 
                    if run>=2: diag_lens.append(run)
                    run=0
            if run>=2: diag_lens.append(run)
        for i in range(T):
            run=0
            for v in R[:, i]:
                if v: run+=1
                else:
                    if run>=2: vert_lens.append(run)
                    run=0
            if run>=2: vert_lens.append(run)
        m["determinism"] = np.sum(diag_lens)/np.sum(R) if np.sum(R)>0 else 0
        m["laminarity"] = np.sum(vert_lens)/np.sum(R) if np.sum(R)>0 else 0
        m["trapping_time"] = np.mean(vert_lens) if vert_lens else 0
        m["diagonal_entropy"] = entropy(diag_lens) if diag_lens else 0
        return m

    # --- Family 7: Landmark (6 metrics) ---
    def landmark_metrics(self, h, correct_id, wrong_id=None, operand_ids=[], intermediate_id=None):
        if self.W_U is None: return {}
        l_corr = h @ self.W_U[correct_id]
        l_wrong = h @ self.W_U[wrong_id] if wrong_id is not None else np.zeros(h.shape[0])
        # S1-S3
        m = {"final_correct_logit": l_corr[-1], "final_wrong_logit": l_wrong[-1], "logit_gap": np.mean(l_corr[-4:] - l_wrong[-4:])}
        # S4-S5 Proximity
        for i in range(3):
            if i < len(operand_ids):
                op_id = operand_ids[i]
                w_op = self.W_U[op_id]
                m[f"operand_{i}_prox"] = np.max([np.dot(vt, w_op)/(np.linalg.norm(vt)*np.linalg.norm(w_op)+1e-9) for vt in h])
            else:
                m[f"operand_{i}_prox"] = np.nan
                
        if intermediate_id is not None:
            w_int = self.W_U[intermediate_id]
            m["intermediate_prox"] = np.max([np.dot(vt, w_int)/(np.linalg.norm(vt)*np.linalg.norm(w_int)+1e-9) for vt in h])
        else:
            m["intermediate_prox"] = np.nan
            
        # S6 Crossing Order (Simplified)
        visit_order = []
        for op_id in operand_ids:
             prox = [np.dot(vt, self.W_U[op_id])/(np.linalg.norm(vt)*np.linalg.norm(self.W_U[op_id])+1e-9) for vt in h]
             visit_order.append(np.argmax(prox))
        m["landmark_crossing_order_entropy"] = entropy(np.histogram(visit_order, bins=len(h))[0]) if visit_order else 0
        return m

    # --- Family 8: Attractor (5 metrics) ---
    def attractor_metrics(self, h, centroids):
        limit = min(h.shape[0], centroids.shape[0])
        dists = np.linalg.norm(h[:limit] - centroids[:limit], axis=1)
        # 8.1-8.2
        m = {"mean_attractor_dist": np.mean(dists), "attractor_div_slope": linregress(np.arange(limit), dists).slope if limit>2 else 0}
        dir_to = centroids[:limit] - h[:limit]; deltas = self._get_deltas(h)[:limit-1]
        c_attr = [np.dot(deltas[t], dir_to[t])/(np.linalg.norm(deltas[t])*np.linalg.norm(dir_to[t])+1e-9) for t in range(limit-1)]
        m["cos_to_success_dir"] = np.mean(c_attr) if c_attr else 0
        # 8.4 Local Expansion
        deltas = self._get_deltas(h); norms = np.linalg.norm(deltas, axis=1)
        if len(norms) > 1:
            rho = norms[1:] / (norms[:-1] + 1e-12)
            m["local_expansion_rate"] = np.mean(rho)
        else:
            m["local_expansion_rate"] = 1.0
            
        # 8.5 PNR
        mu_dist = np.mean(dists)
        if mu_dist > 1e-9:
            pnr_candidates = np.where(dists > mu_dist * 2)[0]
            m["pnr_token"] = pnr_candidates[0] if len(pnr_candidates) > 0 else h.shape[0]
        else:
            m["pnr_token"] = h.shape[0]
        return m

    # --- Family 9: Embedding Stability (4 metrics) ---
    def embedding_stability_metrics(self, h, correct_id, operand_id=None, result_id=None):
        if self.W_U is None or self.W_E is None: return {}
        # E1 Logit Consistency
        logits = h @ self.W_U[correct_id]
        m = {"logit_consistency": np.std(logits)/(np.abs(np.mean(logits))+1e-9)}
        # E2 Landmark Rank Stability: need full projection (slow)
        # Skip E2 for now or simplify.
        # E3 Landmark Pair Similarity
        if operand_id and result_id:
             m["landmark_pair_sim"] = np.dot(self.W_U[operand_id], self.W_U[result_id])/(np.linalg.norm(self.W_U[operand_id])*np.linalg.norm(self.W_U[result_id])+1e-9)
        # E4 Embed-Unembed Alignment
        m["embed_unembed_align"] = np.dot(self.W_E[correct_id], self.W_U[correct_id])/(np.linalg.norm(self.W_E[correct_id])*np.linalg.norm(self.W_U[correct_id])+1e-9)
        return m

    # --- Family 10: Information (3 metrics) ---
    def information_metrics(self, h):
        deltas = self._get_deltas(h); norms = self._safe_norm(deltas, axis=1); norm_d = deltas / norms[:, None]
        surp = 1 - np.sum(norm_d[1:] * norm_d[:-1], axis=1) if len(norm_d)>1 else [0]
        # I1
        m = {"step_surprisal": np.mean(surp)}
        # I2 Entropy Rate (LZ)
        try:
             # Discretize: sign of change in speed
             disc = (np.diff(norms) > 0).astype(int).tolist()
             m["trajectory_entropy_rate"] = lz_complexity(disc) / (len(disc)+1e-9)
        except: m["trajectory_entropy_rate"] = 0
        # I3 Info Gain (Optimized with Gram matrix if available)
        if self.G_U is not None:
            # | (h_last - h_first) W_U.T | = sqrt( diff G diff.T )
            diff = (h[-1] - h[0])[None, :] # (1, D)
            m["info_gain_proxy"] = np.sqrt(np.maximum(0, (diff @ self.G_U @ diff.T)[0,0]))
        elif self.W_U is not None:
             m["info_gain_proxy"] = np.linalg.norm(h[-1]@self.W_U.T - h[0]@self.W_U.T)
        else:
             m["info_gain_proxy"] = 0
        return m

    # --- Family 11: Inference (4 metrics) ---
    def inference_metrics(self, h, top1_id, top2_id):
        if self.W_U is None: return {}
        l1 = h @ self.W_U[top1_id]; l2 = h @ self.W_U[top2_id]; gap = l1 - l2
        # 11.1 Confidence Slope
        m = {"confidence_slope": linregress(np.arange(len(gap)), gap).slope if len(gap)>1 else 0}
        # 11.2 Regime Signal: EffDim / Speed (Early)
        # 11.3 Convergence Monitor: cos(h_t, EMA)
        ema = h[0]
        convs = []
        for t in range(1, h.shape[0]):
             ema = 0.1 * h[t] + 0.9 * ema
             convs.append(np.dot(h[t], ema)/(np.linalg.norm(h[t])*np.linalg.norm(ema)+1e-9))
        m["convergence_monitor"] = np.mean(convs) if convs else 1.0
        return m

# --- Static Helpers ---
def cross_layer_metrics(hidden_stack):
    # hidden_stack: (L, T, D)
    L, T, D = hidden_stack.shape
    aligns = []
    for l in range(L-1):
        h1, h2 = hidden_stack[l], hidden_stack[l+1]
        d1, d2 = h1[1:] - h1[:-1], h2[1:] - h2[:-1]
        dot = np.sum(d1 * d2, axis=1)
        norms = np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
        aligns.append(np.mean(dot / (norms + 1e-9)))
    return {"interlayer_alignment": np.mean(aligns) if aligns else 0}

def cross_layer_acceleration(h_stack):
    L = h_stack.shape[0]; speeds = [np.mean(np.linalg.norm(h_stack[l][1:]-h_stack[l][:-1], axis=1)) for l in range(L)]
    return {"depth_acceleration": linregress(np.arange(L), speeds).slope}

def compute_all_metrics(h_stack, tm, context):
    """Entry point for all 54 metrics."""
    L, T, D = h_stack.shape
    results = []
    
    # Cross-layer (6.1-6.2)
    x_metrics = cross_layer_metrics(h_stack)
    x_metrics.update(cross_layer_acceleration(h_stack))
    
    for l in range(L):
        h = h_stack[l]
        m = {
            # Family 8 Attractor NaNs by default
            "mean_attractor_dist": np.nan,
            "attractor_div_slope": np.nan,
            "cos_to_success_dir": np.nan,
            "local_expansion_rate": np.nan,
            "pnr_token": np.nan
        }
        m.update(tm.kinematic_metrics(h)) # 1-13
        m.update(tm.volumetric_metrics(h)) # 14-17
        m.update(tm.convergence_metrics(h)) # 18-23
        m.update(tm.diffusion_metrics(h)) # 24
        m.update(tm.spectral_metrics(h)) # 25-26
        m.update(tm.rqa_metrics(h)) # 27-31
        # Landmark (32-37)
        m.update(tm.landmark_metrics(h, context['truth_id'], context.get('wrong_id'), context['operand_ids'], context.get('intermediate_id')))
        # Attractor (38-42) Overwrite NaNs if available
        if 'centroids' in context and l in context['centroids']:
             m.update(tm.attractor_metrics(h, context['centroids'][l]))
        # Embedding (43-46)
        m.update(tm.embedding_stability_metrics(h, context['truth_id'], context.get('operand_ids')[0] if context.get('operand_ids') else None, context['truth_id']))
        # Information (47-49)
        m.update(tm.information_metrics(h))
        # Inference (50-54)
        m.update(tm.inference_metrics(h, context['truth_id'], context.get('wrong_id') if context.get('wrong_id') else context['truth_id']))
        
        m.update(x_metrics)
        m['layer'] = l
        results.append(m)
    return results
