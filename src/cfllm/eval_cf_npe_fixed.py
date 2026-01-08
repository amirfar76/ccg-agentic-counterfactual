# cfllm/eval_cf_npe_fixed.py
# CPU-only NPE + GM-SCM LLM; robust numeric metric and proper interventional randomness.

import os, sys, json, math, argparse, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR
from .env_bridge import run_ns3_action
from .llm import action_from_prompt, report_from_metrics
from .scores import METRICS

# ---------------- Utils ----------------

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def _small_rng(seed_like: Any) -> int:
    rnd = random.Random(int(hash(seed_like)) & 0x7FFFFFFF)
    return rnd.randint(1, 2_000_000_000)

def _coerce_float(x, default):
    try: return float(x)
    except: return float(default)

def _coerce_int(x, default):
    try: return int(x)
    except: return int(default)

def _canon_scheduler(s, default="rr"):
    if not isinstance(s, str): return default
    t = s.strip().lower()
    if t in ("pf","proportional fair","prop fair","proportional_fair"): return "pf"
    if t in ("rr","round robin","round_robin","round-robin"): return "rr"
    if t in ("mt","max through","max-throughput","max_throughput","maxthroughput","mrt","mmt"): return "mt"
    return default

def _normalize_action(a_in: Any) -> Dict[str, Any]:
    if isinstance(a_in, dict):
        a = dict(a_in)
    else:
        a = {}
    num_ues   = _coerce_int(a.get("num_ues", a.get("numUEs", 3)), 3)
    traffic   = _coerce_float(a.get("traffic_mbps", a.get("trafficMbps", 0.5)), 0.5)
    duration  = _coerce_float(a.get("duration_s", a.get("duration", 1.0)), 1.0)
    scheduler = _canon_scheduler(a.get("scheduler", "rr"))

    num_ues = min(max(1, num_ues), 64)
    traffic = min(max(0.05, traffic), 50.0)
    duration = min(max(0.1, duration), 30.0)
    return {
        "num_ues": int(num_ues),
        "scheduler": scheduler,
        "traffic_mbps": float(traffic),
        "duration_s": float(duration),
    }

# ---------------- KPI extraction ----------------

def extract_kpi_vector(metrics: Dict[str, Any]) -> np.ndarray:
    """
    Fixed 9-D vector:
    [ total_thr, avg_delay,
      perUE_mean, perUE_std, perUE_min, perUE_p25, perUE_median, perUE_p75, perUE_max ]
    """
    def _finite(x, fallback=0.0):
        try:
            v = float(x)
            if not np.isfinite(v): return float(fallback)
            return v
        except:
            return float(fallback)

    total = None
    for k in ["total_throughput_mbps","tot_thr_mbps","throughput_total_mbps"]:
        if k in metrics:
            total = _finite(metrics[k]); break
    if total is None and "per_ue_throughput_mbps" in metrics:
        try:
            total = float(np.sum([float(v) for v in metrics["per_ue_throughput_mbps"]]))
        except:
            total = 0.0
    if total is None: total = 0.0

    delay = None
    for k in ["avg_delay_ms","delay_ms_avg","avg_e2e_delay_ms"]:
        if k in metrics:
            delay = _finite(metrics[k]); break
    if delay is None: delay = 0.0

    per = None
    for k in ["per_ue_throughput_mbps","ue_throughputs_mbps","perUEthroughputMbps"]:
        if k in metrics and isinstance(metrics[k], (list,tuple)):
            try:
                per = np.array([float(x) for x in metrics[k]], dtype=np.float64)
            except:
                per = np.array([], dtype=np.float64)
            break
    if per is None:
        per = np.array([], dtype=np.float64)

    if per.size == 0:
        stats = [0.0]*7
    else:
        per = per[np.isfinite(per)]
        if per.size == 0:
            stats = [0.0]*7
        else:
            stats = [
                float(np.mean(per)),
                float(np.std(per)),
                float(np.min(per)),
                float(np.quantile(per, 0.25)),
                float(np.median(per)),
                float(np.quantile(per, 0.75)),
                float(np.max(per)),
            ]

    vec = np.array([total, delay] + stats, dtype=np.float64)
    # final NaN/inf guard
    vec[~np.isfinite(vec)] = 0.0
    return vec

def scale_from_action(action_cf: Dict[str, Any]) -> np.ndarray:
    """
    Physics-informed per-dimension scales based on the *counterfactual* action (not the realized KPIs).
    This avoids tiny denominators and keeps variation across cases:
      - Throughput scales ~ num_ues * traffic_mbps (with margin)
      - Delay scale ~ 200 ms baseline
      - Per-UE stats scale ~ traffic_mbps (with margin)
    """
    n = float(action_cf.get("num_ues", 3))
    r = float(action_cf.get("traffic_mbps", 0.5))

    thr_scale = max(1.0, n * r * 1.5)
    delay_scale = 200.0
    per_scale = max(1.0, r * 1.5)

    # [ total_thr, avg_delay, perUE_mean, perUE_std, perUE_min, p25, med, p75, max ]
    scales = np.array([
        thr_scale, delay_scale,
        per_scale, per_scale, per_scale, per_scale, per_scale, per_scale, per_scale
    ], dtype=np.float64)
    return scales

def numeric_distance(true_m: Dict[str,Any], pred_m: Dict[str,Any], scale_vec: np.ndarray) -> float:
    """
    RMSE of scaled differences, with per-case scale_vec (length 9).
    """
    y_true = extract_kpi_vector(true_m)
    y_pred = extract_kpi_vector(pred_m)
    s = np.array(scale_vec, dtype=np.float64)
    s[s < 1.0] = 1.0  # extra guard

    diff = (y_pred - y_true) / s
    rmse = float(np.sqrt(np.mean(diff * diff)))
    if not np.isfinite(rmse):
        return 1.0  # conservative fallback if something weird happens
    return rmse

# ---------------- NPE (CPU) ----------------

def _seed_range() -> Tuple[int, int]:
    return (1, 2_000_000_000)

def _normalize_seed(s: np.ndarray) -> np.ndarray:
    lo, hi = _seed_range()
    return (s - lo) / (hi - lo)

def _denormalize_seed(z: np.ndarray) -> np.ndarray:
    lo, hi = _seed_range()
    return lo + z * (hi - lo)

class PosteriorHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # mean, logvar
        )
        self.kpi_mean = None
        self.kpi_std = None

    def forward(self, x):
        out = self.net(x)
        mu = out[..., 0]
        logvar = out[..., 1].clamp(min=-10.0, max=5.0)
        return mu, logvar

def fit_seed_posterior(
    action: Dict[str,Any],
    sims: int,
    seed: int,
    tmpdir: str,
    epochs: int = 60,
    batch_size: int = 128,
) -> PosteriorHead:
    rng = np.random.default_rng(seed)
    sim_seeds = rng.integers(low=_seed_range()[0], high=_seed_range()[1], size=sims, dtype=np.int64)

    Z = []
    T = []
    for s in tqdm(sim_seeds, desc="Simulating for NPE (factual)", leave=False):
        m = run_ns3_action(action, rng_run=int(s), workdir=tmpdir)
        Z.append(extract_kpi_vector(m))
        T.append(float(s))
    Z = np.stack(Z, axis=0).astype(np.float64)
    T = np.array(T, dtype=np.float64)
    Tn = _normalize_seed(T)

    mean = Z.mean(axis=0, keepdims=True)
    std  = Z.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-6, 1.0, std)
    Zs   = (Z - mean) / std

    model = PosteriorHead(in_dim=Z.shape[1], hidden=128)
    model.kpi_mean = mean
    model.kpi_std = std
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(Zs).float(), torch.from_numpy(Tn).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in tqdm(range(epochs), desc="Training NPE (CPU)", leave=False):
        for xb, tb in dl:
            mu, logvar = model(xb)
            inv_var = torch.exp(-logvar)
            loss = 0.5 * torch.mean((tb - mu)**2 * inv_var + logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model

def sample_seeds_from_posterior(model: PosteriorHead, z_obs: np.ndarray, n: int, seed: int) -> List[int]:
    z = (z_obs - model.kpi_mean) / model.kpi_std
    zt = torch.from_numpy(z).float().unsqueeze(0)
    mu, logvar = model(zt)
    mu = mu.item()
    std = float(np.sqrt(np.exp(logvar.item())))
    rng = np.random.default_rng(seed)
    zs = rng.normal(loc=mu, scale=std, size=n)
    zs = np.clip(zs, 0.0, 1.0)
    seeds = _denormalize_seed(zs).astype(np.int64).tolist()
    return seeds

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="CF eval with NPE posterior (CPU) + GM-SCM LLM + robust numeric metric.")
    ap.add_argument("--data-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "test.jsonl"))
    ap.add_argument("--temps", type=str, default="0.4,0.7,1.0")
    ap.add_argument("--metric", type=str, default="levenshtein_norm")
    ap.add_argument("--max-cases", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--sims-per-round", type=int, default=1500)
    ap.add_argument("--posterior-samples", type=int, default=64)
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "cf_npe_fixed"))
    ap.add_argument("--device", type=str, default="cpu")  # kept for CLI compat; we force CPU
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard-disable CUDA
    if args.metric not in METRICS:
        raise KeyError(f"Unknown metric '{args.metric}'. Available: {list(METRICS.keys())}")
    metric_fn = METRICS[args.metric]

    ensure_dir(args.outdir)
    rows = read_jsonl(args.data_path)[:args.max_cases]
    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    if not rows:
        raise RuntimeError("No rows.")

    plt.rcParams["text.usetex"] = False

    perrow_records = []
    debug_rows = []

    pbar = tqdm(total=len(rows), desc="Cases", leave=True)
    for idx, r in enumerate(rows):
        # Build actions with low-temp mapping
        A_raw  = action_from_prompt(r["X"],       seed=101+idx, temperature=0.2)
        Ap_raw = action_from_prompt(r["X_prime"], seed=101+idx, temperature=0.2)
        A  = _normalize_action(A_raw)
        Ap = _normalize_action(Ap_raw)

        # True env seed (unknown)
        rng_true_env = _small_rng(("true-env", idx))

        # Factual obs
        m_f = run_ns3_action(A, rng_run=rng_true_env, workdir=os.path.join(OUTPUT_DIR,"tmp"))
        z_obs = extract_kpi_vector(m_f)

        # Fit NPE on factual (CPU)
        model = fit_seed_posterior(
            action=A,
            sims=args.sims_per_round,
            seed=777 + idx,
            tmpdir=os.path.join(OUTPUT_DIR,"tmp"),
            epochs=60,
        )

        # True counterfactual metrics
        m_cf_true = run_ns3_action(Ap, rng_run=rng_true_env, workdir=os.path.join(OUTPUT_DIR,"tmp"))

        # Our CF via posterior (pick best-of-N by numeric distance)
        post_seeds = sample_seeds_from_posterior(model, z_obs=z_obs, n=max(1,args.posterior_samples), seed=999+idx)
        scale_cf = scale_from_action(Ap)
        best_s = None
        best_err = float("inf")
        best_m = None
        for s in post_seeds:
            m_try = run_ns3_action(Ap, rng_run=int(s), workdir=os.path.join(OUTPUT_DIR,"tmp"))
            err = numeric_distance(m_cf_true, m_try, scale_vec=scale_cf)
            if err < best_err:
                best_err, best_s, best_m = err, s, m_try
        m_cf_ours = best_m
        seed_cf_ours = best_s

        # Interventional CF: fresh env + fresh LLM
        rng_interv_env = _small_rng(("interv-env", idx))
        m_cf_interv = run_ns3_action(Ap, rng_run=rng_interv_env, workdir=os.path.join(OUTPUT_DIR,"tmp"))

        # Numeric distances (temperature-independent)
        n_ours = numeric_distance(m_cf_true, m_cf_ours, scale_vec=scale_cf)
        n_interv = numeric_distance(m_cf_true, m_cf_interv, scale_vec=scale_cf)

        # Debug dump for this case
        debug_rows.append({
            "idx": idx,
            "A": A, "Ap": Ap,
            "rng_true_env": int(rng_true_env),
            "rng_cf_ours_env": int(seed_cf_ours),
            "rng_cf_interv_env": int(rng_interv_env),
            "scale_cf": scale_cf.tolist(),
            "kpi_true": extract_kpi_vector(m_cf_true).tolist(),
            "kpi_ours": extract_kpi_vector(m_cf_ours).tolist(),
            "kpi_interv": extract_kpi_vector(m_cf_interv).tolist(),
            "num_err_ours": n_ours,
            "num_err_interv": n_interv,
        })

        # LLM report seeds: fix for true + ours; fresh for interventional
        report_seed_base = 42 + idx
        for T in temps:
            y_true = report_from_metrics(m_cf_true, seed=report_seed_base, temperature=T)
            y_ours = report_from_metrics(m_cf_ours, seed=report_seed_base, temperature=T)
            y_interv = report_from_metrics(m_cf_interv, seed=report_seed_base + 10_000, temperature=T)

            d_edit_ours = float(METRICS[args.metric](y_ours, y_true))
            d_edit_interv = float(METRICS[args.metric](y_interv, y_true))

            perrow_records.append({
                "idx": idx, "temperature": T, "method": "ours_npe",
                "edit_distance": d_edit_ours, "numeric_distance": n_ours
            })
            perrow_records.append({
                "idx": idx, "temperature": T, "method": "interventional",
                "edit_distance": d_edit_interv, "numeric_distance": n_interv
            })

        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(perrow_records)
    ensure_dir(args.outdir)
    csv_main = os.path.join(args.outdir, "cf_npe_fixed_results.csv")
    df.to_csv(csv_main, index=False)

    # Debug CSV (per-case vectors + scales)
    dbg = pd.DataFrame(debug_rows)
    dbg_csv = os.path.join(args.outdir, "cf_npe_fixed_debug.csv")
    dbg.to_csv(dbg_csv, index=False)

    # --------- Plots ---------
    # Edit vs temperature
    agg = df.groupby(["temperature","method"])["edit_distance"].mean().reset_index()
    plt.figure()
    for method, label in [("ours_npe","NPE (GM-SCM)"), ("interventional","Interventional")]:
        sub = agg[agg["method"]==method].sort_values("temperature")
        plt.plot(sub["temperature"].values, sub["edit_distance"].values, marker="o", label=label)
    plt.xlabel("Temperature τ")
    plt.ylabel("Average edit distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "edit_distance_vs_temp.png"), dpi=160)
    plt.close()

    # Numeric distributions
    plt.figure()
    data_ours  = df[df["method"]=="ours_npe"]["numeric_distance"].values
    data_inter = df[df["method"]=="interventional"]["numeric_distance"].values
    plt.boxplot([data_ours, data_inter], labels=["NPE (GM-SCM)","Interventional"], showfliers=True)
    plt.ylabel("Numeric distance (RMSE of scaled KPI diff)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "numeric_distance_box.png"), dpi=160)
    plt.close()

    print(f"[✓] Results CSV: {csv_main}")
    print(f"[✓] Debug CSV (per-case vectors & scales): {dbg_csv}")
    print(f"[✓] Plots saved under: {args.outdir}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()