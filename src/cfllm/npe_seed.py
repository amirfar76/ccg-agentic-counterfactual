# cfllm/npe_seed.py
# Full NPE for discrete seed inference: q_phi(seed | features(trace), action)

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# sbi imports
from sbi.inference import prepare_for_sbi, SNPE
from sbi.utils import BoxUniform

# ----------------------------------------------------------------------
# Utilities: featureization of a time series trace (factual)
# ----------------------------------------------------------------------

def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std <= 1e-12, 1.0, std)
    return (x - mean) / std

def summarize_trace(
    time_s: np.ndarray,
    thr_mbps: np.ndarray,
    dly_ms: np.ndarray,
    tmin: float = 0.0,
    tmax: Optional[float] = None,
    downsample: int = 1,
) -> np.ndarray:
    """
    Produce a compact, informative feature vector from the factual time series.
    We concatenate:
      - raw downsampled thr (first N)
      - raw downsampled delay (first N)
      - simple statistics: mean/std/max/min for both channels
      - frequency-domain energy in 3 coarse bands for both channels
    """
    mask = (time_s >= tmin) & ((time_s <= tmax) if tmax is not None else True)
    thr = thr_mbps[mask]
    dly = dly_ms[mask]
    if downsample > 1:
        thr = thr[::downsample]
        dly = dly[::downsample]

    # choose a fixed length window by truncating or zero-padding
    L = 64
    def _fit_len(x):
        x = x[:L]
        if len(x) < L:
            x = np.pad(x, (0, L - len(x)), mode="constant")
        return x

    thrL = _fit_len(thr)
    dlyL = _fit_len(dly)

    # stats
    def _stats(x):
        return np.array([x.mean(), x.std(), x.max(initial=0.0), x.min(initial=0.0)], dtype=np.float32)

    # coarse spectral energy: low/mid/high thirds of FFT magnitude (skip DC)
    def _spec_feats(x):
        X = np.fft.rfft(x - x.mean())
        mag = np.abs(X)[1:]  # drop DC
        if len(mag) == 0:
            return np.zeros(3, dtype=np.float32)
        thirds = np.array_split(mag, 3)
        return np.array([seg.mean() for seg in thirds], dtype=np.float32)

    feats = np.concatenate([
        thrL.astype(np.float32),
        dlyL.astype(np.float32),
        _stats(thrL), _stats(dlyL),
        _spec_feats(thrL), _spec_feats(dlyL),
    ], axis=0)
    return feats  # shape: 64 + 64 + 4 + 4 + 3 + 3 = 142

def action_to_vec(action: Dict) -> np.ndarray:
    """
    Map an action dict to a fixed, normalized vector.
    Keys expected: numUes, scheduler, trafficMbps, duration, (bandwidthMHz optional)
    """
    num_ues = float(action.get("numUes", action.get("num_ues", 5)))
    traffic = float(action.get("trafficMbps", action.get("traffic_mbps", 8.0)))
    duration = float(action.get("duration", action.get("duration_s", 8.0)))
    bw = float(action.get("bandwidthMHz", action.get("bandwidth_mhz", 10.0)))
    sched = str(action.get("scheduler", "rr")).lower()
    sched_vec = {
        "rr": [1, 0, 0],
        "pf": [0, 1, 0],
        "mt": [0, 0, 1],  # tolerate 'mt' if present in some runs
    }.get(sched, [1, 0, 0])

    # crude normalization
    v = np.array([num_ues / 20.0, traffic / 50.0, duration / 20.0, bw / 50.0] + sched_vec, dtype=np.float32)
    return v  # length 4 + 3 = 7

# ----------------------------------------------------------------------
# Dataset I/O (training-set) format
# ----------------------------------------------------------------------
# We assume you have a directory with JSON lines where each line contains:
# {
#   "seed": 1554486822,
#   "action": { ... },
#   "time_s": [...], "thr_total_mbps": [...], "avg_delay_ms": [...]
# }
# created from the hi-fi runs (or your existing data pipeline).
# ----------------------------------------------------------------------

def load_training_traces(jsonl_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    seeds: List[int] = []
    X: List[np.ndarray] = []
    A: List[np.ndarray] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            s = int(rec["seed"])
            a = rec["action"]
            t = np.array(rec["time_s"], dtype=np.float32)
            thr = np.array(rec["thr_total_mbps"], dtype=np.float32)
            dly = np.array(rec["avg_delay_ms"], dtype=np.float32)

            x = summarize_trace(t, thr, dly, tmin=1.0, tmax=None, downsample=1)
            v = action_to_vec(a)

            seeds.append(s)
            X.append(x)
            A.append(v)

    X = np.stack(X, axis=0)           # [N, Dx]
    A = np.stack(A, axis=0)           # [N, Da]
    seeds_list = seeds
    return X, A, seeds_list

# ----------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------

class NPESeedEstimator:
    """
    Discrete-seed NPE using sbi. We treat the seed as a categorical parameter
    by embedding it into a continuous index space [0, S-1] and learning a
    classifier-like posterior over the discrete catalog of seeds.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._posterior = None
        self._cat_values = None            # tensor of seed indices (0..S-1)
        self._catalog_seeds: List[int] = []  # actual seed values aligned with indices
        self._x_mean = None
        self._x_std = None
        self._a_mean = None
        self._a_std = None

    # ---------------- Training ---------------- #

    def fit(
        self,
        X: np.ndarray,            # [N, Dx] trace features
        A: np.ndarray,            # [N, Da] action features
        seeds: List[int],         # [N] raw seed values
        hidden_features: int = 128,
        num_atoms: int = 10,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 123,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Make catalog mapping
        uniq_seeds = sorted(list({int(s) for s in seeds}))
        seed_to_idx = {s: i for i, s in enumerate(uniq_seeds)}
        cat_idx = np.array([seed_to_idx[int(s)] for s in seeds], dtype=np.int64)

        # Standardize inputs
        x_mean, x_std = X.mean(0), X.std(0)
        a_mean, a_std = A.mean(0), A.std(0)
        Xz = _standardize(X, x_mean, x_std).astype(np.float32)
        Az = _standardize(A, a_mean, a_std).astype(np.float32)

        # Concatenate (x, a) -> single observation vector
        O = np.concatenate([Xz, Az], axis=1).astype(np.float32)
        O_t = torch.from_numpy(O).to(self.device)
        S_t = torch.from_numpy(cat_idx).to(self.device)

        # Define a "box" prior over the discrete index space [0, S-1] (relaxed):
        # We'll let sbi treat it as continuous, but we only evaluate the posterior on discrete atoms.
        low, high = torch.tensor([0.0], device=self.device), torch.tensor([float(len(uniq_seeds) - 1)], device=self.device)
        prior = BoxUniform(low, high)  # 1-D pseudo-prior over index

        # Simulator-free usage: directly use (parameters, observations) pairs
        inference = SNPE(prior=prior, density_estimator="maf", device=self.device)
        density_estimator = inference.append_simulations(
            torch.tensor(cat_idx, dtype=torch.float32, device=self.device).unsqueeze(1),  # shape [N,1]
            O_t
        ).train(training_batch_size=batch_size, learning_rate=lr, stop_after_epochs=epochs)

        posterior = inference.build_posterior(density_estimator)

        # Save state
        self._posterior = posterior
        self._cat_values = torch.arange(len(uniq_seeds), device=self.device, dtype=torch.float32).unsqueeze(1)  # [S,1]
        self._catalog_seeds = uniq_seeds
        self._x_mean, self._x_std = x_mean, x_std
        self._a_mean, self._a_std = a_mean, a_std

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "posterior": self._posterior.net.state_dict(),  # density net weights
            "catalog_seeds": self._catalog_seeds,
            "x_mean": self._x_mean, "x_std": self._x_std,
            "a_mean": self._a_mean, "a_std": self._a_std,
        }, path)

    def load(self, path: str):
        chk = torch.load(path, map_location=self.device)
        self._catalog_seeds = list(map(int, chk["catalog_seeds"]))
        self._x_mean, self._x_std = chk["x_mean"], chk["x_std"]
        self._a_mean, self._a_std = chk["a_mean"], chk["a_std"]

        # Rebuild an identical architecture to load weights.
        low, high = torch.tensor([0.0], device=self.device), torch.tensor([float(len(self._catalog_seeds)-1)], device=self.device)
        prior = BoxUniform(low, high)
        inference = SNPE(prior=prior, density_estimator="maf", device=self.device)
        # Minimal fake append to instantiate net with correct shapes
        dummy_o = torch.zeros((1, 142 + 7), device=self.device)  # 142 trace feats + 7 action feats
        _ = inference.append_simulations(torch.zeros((1, 1), device=self.device), dummy_o)
        density_estimator = inference._build_neural_net()
        density_estimator.load_state_dict(chk["posterior"])
        self._posterior = inference.build_posterior(density_estimator)
        self._cat_values = torch.arange(len(self._catalog_seeds), device=self.device, dtype=torch.float32).unsqueeze(1)

    # ---------------- Inference ---------------- #

    @torch.no_grad()
    def infer_seed(
        self,
        time_s: np.ndarray,
        thr_mbps: np.ndarray,
        dly_ms: np.ndarray,
        action: Dict,
        tmin: float = 1.0,
        tmax: Optional[float] = None,
        downsample: int = 1,
        topk: int = 1,
        temperature: float = 0.7,     # posterior tempering (<1 sharpen, >1 soften)
    ) -> Tuple[int, List[Tuple[int, float]]]:
        """
        Returns (best_seed, [(seed, score) ... topk]) where score is normalized posterior weight.
        """
        assert self._posterior is not None, "Call fit() or load() first."
        x = summarize_trace(time_s, thr_mbps, dly_ms, tmin=tmin, tmax=tmax, downsample=downsample)
        a = action_to_vec(action)
        xz = _standardize(x, self._x_mean, self._x_std)
        az = _standardize(a, self._a_mean, self._a_std)
        obs = torch.from_numpy(np.concatenate([xz, az], axis=0)).float().to(self.device).unsqueeze(0)  # [1,D]

        # Evaluate log-prob at all catalog atoms
        logp = self._posterior.log_prob(self._cat_values, x=obs)  # [S]
        lp = (logp / max(temperature, 1e-6)).squeeze(1)  # temper
        w = torch.softmax(lp, dim=0).detach().cpu().numpy()  # [S]

        # rank seeds
        idx_sorted = np.argsort(-w)
        top_idx = idx_sorted[:topk]
        best_seed = self._catalog_seeds[int(top_idx[0])]
        top_list = [(self._catalog_seeds[int(i)], float(w[int(i)])) for i in top_idx]
        return best_seed, top_list
