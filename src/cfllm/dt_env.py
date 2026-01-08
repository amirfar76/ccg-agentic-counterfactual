# cfllm/dt_env.py
# Parametric "digital twin" environment and parameter estimation.
# No ns-3 changes required. If later you expose flags in ns-3, you can route them.

import math, json, hashlib, random
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

# ---------- KPI distance (same spirit as your eval_cf) ----------

def _env_signature(metrics: Dict[str,Any]) -> Tuple[float, float, List[float]]:
    total = None
    for k in ["total_throughput_mbps","tot_thr_mbps","throughput_total_mbps"]:
        if k in metrics:
            total = float(metrics[k]); break
    if total is None and "per_ue_throughput_mbps" in metrics:
        total = float(np.sum(metrics["per_ue_throughput_mbps"]))
    if total is None: total = 0.0

    delay = None
    for k in ["avg_delay_ms","delay_ms_avg","avg_e2e_delay_ms"]:
        if k in metrics:
            delay = float(metrics[k]); break
    if delay is None: delay = 0.0

    per_ue = None
    for k in ["per_ue_throughput_mbps","ue_throughputs_mbps","perUEthroughputMbps"]:
        if k in metrics and isinstance(metrics[k], (list,tuple)):
            per_ue = [float(x) for x in metrics[k]]; break
    if per_ue is None: per_ue = []
    return total, delay, sorted(per_ue)

def _rel(x: float, y: float) -> float:
    denom = max(1e-9, abs(x) + abs(y))
    return abs(x - y) / denom

def env_metric_distance(m_pred: Dict[str,Any], m_obs: Dict[str,Any]) -> float:
    t1, d1, u1 = _env_signature(m_pred)
    t2, d2, u2 = _env_signature(m_obs)
    a = _rel(t1, t2)
    b = _rel(d1, d2)
    n = max(len(u1), len(u2))
    if n == 0:
        c = 0.0
    else:
        uu1 = (u1 + [0.0]*n)[:n]
        uu2 = (u2 + [0.0]*n)[:n]
        denom = max(1e-6, sum(uu1) + sum(uu2))
        c = sum(abs(p - q) for p,q in zip(uu1, uu2)) / denom
    return a + b + c

# ---------- Digital twin models (simple / medium) ----------

def _sched_eff(s: str) -> float:
    s = (s or "rr").lower()
    if s.startswith("pf"): return 1.00
    if s.startswith("mt"): return 1.05
    return 0.95  # rr default

def _rng(seed: int) -> random.Random:
    return random.Random(int(seed) & 0x7FFFFFFF)

def twin_simple(action: Dict[str,Any], params: Dict[str,Any], seed: int) -> Dict[str,Any]:
    """
    Very simple capacity + utilization twin with wireless-meaningful knobs.
    params:
      pl_exp in [2,4] (path-loss exponent proxy)
      shadow_sigma_db in [0,12] (lognormal shadowing spread)
      nakagami_m in [0.5, 3] (fading severity proxy)
    """
    num_ues   = int(action.get("num_ues", 3))
    traffic   = float(action.get("traffic_mbps", 0.5))
    duration  = float(action.get("duration_s", 1.0))
    sched     = str(action.get("scheduler", "rr"))

    pl_exp = float(params.get("pl_exp", 2.5))
    shadow_sigma_db = float(params.get("shadow_sigma_db", 4.0))
    nakagami_m = float(params.get("nakagami_m", 1.5))

    # Base "effective capacity" in Mbps (rough heuristic)
    sched_mult = _sched_eff(sched)
    eff_fading = nakagami_m / (1.0 + nakagami_m)            # in (0,1)
    eff_path   = math.exp(-0.18 * max(0.0, pl_exp - 2.0))   # ↓ with PL exponent
    eff_shadow = 1.0 / math.sqrt(1.0 + shadow_sigma_db/8.0) # ↓ with shadow spread

    base_cap = 25.0 * sched_mult * eff_fading * eff_path * eff_shadow
    offered  = num_ues * traffic

    # Tiny deterministic wiggle so it's not flat across seeds/cases
    rnd = _rng(hash((seed, num_ues, traffic, sched)))
    base_cap *= (1.0 + 0.03 * math.sin(0.37 * num_ues + 0.51 * traffic + 0.11))

    total = min(base_cap, offered)
    util  = offered / max(1e-6, base_cap)

    # Delay proxy: queuing blowup past utilization 1
    delay_ms = 6.0 + 140.0 * max(0.0, util - 1.0)

    # Per-UE split (scheduler-dependent dispersion)
    # RR ~ even; PF ~ near-even; MT ~ skewed
    shares = np.ones(num_ues, dtype=float)
    if sched.startswith("mt"):
        # Zipf-like tilt
        ranks = np.arange(1, num_ues+1)
        shares = 1.0 / ranks
    elif sched.startswith("pf"):
        shares = np.ones(num_ues)
    else:  # rr
        shares = np.ones(num_ues)

    shares /= shares.sum()
    per_ue = (total * shares).tolist()

    return {
        "total_throughput_mbps": float(total),
        "avg_delay_ms": float(delay_ms),
        "per_ue_throughput_mbps": [float(x) for x in per_ue],
        "duration_s": float(duration),
        "model": "twin_simple",
        "params": dict(pl_exp=pl_exp, shadow_sigma_db=shadow_sigma_db, nakagami_m=nakagami_m),
    }

def twin_medium(action: Dict[str,Any], params: Dict[str,Any], seed: int) -> Dict[str,Any]:
    """
    Medium twin adds mobility (Doppler proxy) and shadowing dynamics.
    params: pl_exp, shadow_sigma_db, nakagami_m, ue_speed_mps, doppler_mult
    """
    base = twin_simple(action, params, seed)
    num_ues   = int(action.get("num_ues", 3))
    traffic   = float(action.get("traffic_mbps", 0.5))
    sched     = str(action.get("scheduler", "rr"))

    ue_speed = float(params.get("ue_speed_mps", 1.0))    # 0..3 m/s
    doppler_mult = float(params.get("doppler_mult", 1.0))# 0.5..2.0

    # Mobility reduces coherent combining → reduce capacity slightly; increase dispersion
    mobility_penalty = math.exp(-0.05 * ue_speed * doppler_mult)
    total = float(base["total_throughput_mbps"]) * mobility_penalty

    # Slightly higher delay penalty under mobility
    util = (num_ues * traffic) / max(1e-6, total / max(1e-6, mobility_penalty))
    delay_ms = 6.5 + 160.0 * max(0.0, util - 1.0)

    # Per-UE: more spread if RR, small spread if PF, strong spread if MT (keep style)
    per_ue = base["per_ue_throughput_mbps"]
    rng = np.random.default_rng(abs(hash((seed, num_ues, traffic, sched))) % (2**32))
    if sched.startswith("pf"):
        jit = rng.normal(0, 0.03, size=len(per_ue))
    elif sched.startswith("mt"):
        jit = rng.normal(0, 0.10, size=len(per_ue))
    else:
        jit = rng.normal(0, 0.05, size=len(per_ue))
    per_ue = np.maximum(0.0, (np.array(per_ue) * (1.0 + jit)))
    s = per_ue.sum()
    if s > 0: per_ue *= (total / s)

    base.update({
        "total_throughput_mbps": float(total),
        "avg_delay_ms": float(delay_ms),
        "per_ue_throughput_mbps": [float(x) for x in per_ue],
        "model": "twin_medium",
        "params": dict(**base["params"],
                       ue_speed_mps=ue_speed,
                       doppler_mult=doppler_mult),
    })
    return base

# ---------- Estimator (random search; replace with BO/CMA-ES later) ----------

_TWIN_SPECS = {
    "simple": {
        "fn": twin_simple,
        "search_space": {
            "pl_exp": (2.0, 4.0),
            "shadow_sigma_db": (0.0, 12.0),
            "nakagami_m": (0.5, 3.0),
        },
    },
    "medium": {
        "fn": twin_medium,
        "search_space": {
            "pl_exp": (2.0, 4.0),
            "shadow_sigma_db": (0.0, 12.0),
            "nakagami_m": (0.5, 3.0),
            "ue_speed_mps": (0.0, 3.0),
            "doppler_mult": (0.5, 2.0),
        },
    },
}

def _sample_params(space: Dict[str,Tuple[float,float]], rng: random.Random) -> Dict[str,float]:
    out = {}
    for k,(lo,hi) in space.items():
        out[k] = float(lo + (hi-lo) * rng.random())
    return out

def estimate_twin_params(
    fidelity: str,
    action_factual: Dict[str,Any],
    metrics_factual_complex: Dict[str,Any],
    budget: int,
    seed: int
) -> Tuple[Dict[str,Any], float]:
    """
    Fit twin params on factual case by minimizing KPI distance to 'complex' ns-3 factual metrics.
    Returns (best_params, best_score)
    """
    spec = _TWIN_SPECS[fidelity]
    space = spec["search_space"]
    twin_fn = spec["fn"]

    rnd = random.Random(int(seed) & 0x7FFFFFFF)

    best_p = None
    best_s = float("inf")
    for _ in range(max(1, budget)):
        p = _sample_params(space, rnd)
        m_pred = twin_fn(action_factual, p, seed=seed)
        s = env_metric_distance(m_pred, metrics_factual_complex)
        if s < best_s:
            best_s = s
            best_p = p
    if best_p is None:
        best_p = _sample_params(space, rnd)
    return best_p, float(best_s)

