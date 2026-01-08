# Evaluate CF generation using black-box fitted digital twins (low & high fidelity),
# plus an interventional baseline. Fitting is done against factual KPIs only, and
# then applied to counterfactual KPIs under X'.
#
# Truth:     real env (rng_true) + same LLM randomness across methods (GM-SCM proxy).
# Low Twin:  base twin env (rng_twin != rng_true) + low-fidelity knob fitting + same LLM randomness.
# High Twin: base twin env (rng_twin != rng_true) + high-fidelity knob fitting (more knobs + local refine)
#            + same LLM randomness.
# Interv.:   base twin env with fresh rng + fresh LLM randomness (worst-case baseline).
#
# Outputs:
#   - CSV of per-case distances (text and numeric)
#   - Summary CSV (by temperature & method)
#   - PDF plots:
#       1) Average TEXT distance vs temperature (line plot)
#       2) NUMERIC distance vs temperature (line plot)
#       3) Boxplot of NUMERIC distances by method (all temps pooled)
#       4) Boxplot of TEXT distances by method (all temps pooled)
#
# Numeric distance = mean over coordinates of bounded relative squared error in [0,1]:
#   err_i = min( ((x_i - y_i)^2) / (max(|x_i|, |y_i|, eps)^2), 1.0 )
#   ND(x,y) = average_i(err_i)
#
# Requirements: uses only cfllm.env_bridge.run_ns3_action, cfllm.llm.*, cfllm.config, cfllm.scores.METRICS

import os, re, json, ast, argparse, hashlib, random, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR
from .env_bridge import run_ns3_action
from .llm import action_from_prompt, report_from_metrics
from .scores import METRICS


# ------------------------------- utils -------------------------------

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _small_rng(seed: int) -> int:
    # ns-3 expects a reasonable positive run id; bound into [1, 2e9]
    rnd = random.Random(int(seed) & 0x7FFFFFFF)
    return rnd.randint(1, 2_000_000_000)

def _strip_code_fences(txt: str) -> str:
    m = re.search(r"```+(.+?)```+", txt, flags=re.S | re.M)
    if m:
        return m.group(1)
    return txt

def _extract_first_json_blob(txt: str) -> Optional[str]:
    start = txt.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(txt)):
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
            if depth == 0:
                return txt[start:i+1]
    return None

def _canon_scheduler(s: Any, default: str = "rr") -> str:
    if not isinstance(s, str): return default
    t = s.strip().lower()
    if t in ("pf","proportional fair","prop fair","proportional_fair"): return "pf"
    if t in ("rr","round robin","round_robin","round-robin"): return "rr"
    if t in ("mt","max throughput","max-throughput","max_throughput","maxthroughput","mrt","mmt"): return "mt"
    return default

def _coerce_float(x: Any, default: float) -> float:
    try: return float(x)
    except Exception: return float(default)

def _coerce_int(x: Any, default: int) -> int:
    try: return int(x)
    except Exception: return int(default)

def _parse_action_from_string(txt: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fences(txt)

    blob = _extract_first_json_blob(s)
    if blob:
        try:
            return json.loads(blob)
        except Exception:
            try:
                fixed = blob.replace("'", '"')
                fixed = re.sub(r'("numUEs")', '"num_ues"', fixed)
                fixed = re.sub(r'("trafficMbps")', '"traffic_mbps"', fixed)
                fixed = re.sub(r'("duration")', '"duration_s"', fixed)
                return json.loads(fixed)
            except Exception:
                pass

    try:
        obj = ast.literal_eval(s.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    cand = {}
    m = re.search(r"(num[_\s]*u(e)?s|users)\s*[:=]\s*([0-9]+)", s, flags=re.I)
    if m: cand["num_ues"] = int(m.group(3))
    m = re.search(r"(traffic[_\s]*mbps|rate[_\s]*mbps|traffic|rate)\s*[:=]\s*([0-9.]+)", s, flags=re.I)
    if m: cand["traffic_mbps"] = float(m.group(2))
    m = re.search(r"(duration|duration[_\s]*s|time|seconds)\s*[:=]\s*([0-9.]+)", s, flags=re.I)
    if m: cand["duration_s"] = float(m.group(2))
    m = re.search(r"(scheduler|sched)\s*[:=]\s*([A-Za-z\- _]+)", s, flags=re.I)
    if m: cand["scheduler"] = _canon_scheduler(m.group(2))

    return cand or None

def _normalize_action(a_in: Any) -> Dict[str, Any]:
    if isinstance(a_in, str):
        parsed = _parse_action_from_string(a_in)
        if parsed is None: parsed = {}
        a = parsed
    elif isinstance(a_in, dict):
        a = dict(a_in)
    else:
        a = {}

    lower_map = {k.lower(): k for k in a.keys()}
    def _get(*names, default=None):
        for n in names:
            if n in a: return a[n]
            if n.lower() in lower_map: return a[lower_map[n.lower()]]
        return default

    num_ues   = _coerce_int(_get("num_ues","numUEs","users","ues", default=3), 3)
    traffic   = _coerce_float(_get("traffic_mbps","trafficMbps","traffic","rate", default=0.5), 0.5)
    duration  = _coerce_float(_get("duration_s","duration","time","seconds", default=1.0), 1.0)
    scheduler = _canon_scheduler(_get("scheduler","sched", default="rr"))

    num_ues  = min(max(1, num_ues), 64)
    duration = min(max(0.1, duration), 30.0)
    traffic  = min(max(0.01, traffic), 50.0)

    return {
        "num_ues": int(num_ues),
        "scheduler": scheduler,
        "traffic_mbps": float(traffic),
        "duration_s": float(duration),
    }

def _ns3_cache_key(action: Dict[str,Any], rng_run: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(action, sort_keys=True).encode("utf-8"))
    h.update(str(rng_run).encode("utf-8"))
    return h.hexdigest()[:16]

def _safe_run_ns3_action(action: Dict[str, Any], rng_run: int, tmpdir: str, cache_dir: Optional[str]) -> Dict[str, Any]:
    ensure_dir(tmpdir)
    if cache_dir:
        ensure_dir(cache_dir)
        key = _ns3_cache_key(action, rng_run)
        cp = os.path.join(cache_dir, f"ns3_{key}.json")
        if os.path.exists(cp):
            with open(cp, "r") as f:
                return json.load(f)
    # run (may raise)
    m = run_ns3_action(action, rng_run=int(rng_run), workdir=tmpdir)

    if cache_dir:
        key = _ns3_cache_key(action, rng_run)
        cp = os.path.join(cache_dir, f"ns3_{key}.json")
        with open(cp, "w") as f:
            json.dump(m, f, indent=2)
    return m

# ----------------------- metrics extraction & distance -----------------------

def _env_signature(metrics: Dict[str,Any]) -> Tuple[float, float, List[float]]:
    total = None
    for k in ["total_throughput_mbps","tot_thr_mbps","throughput_total_mbps"]:
        if k in metrics:
            total = float(metrics[k]); break
    if total is None and "per_ue_throughput_mbps" in metrics:
        try:
            total = float(np.sum(metrics["per_ue_throughput_mbps"]))
        except Exception:
            total = None
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
    if per_ue is None:
        per_ue = []

    return total, delay, sorted(per_ue)

def _numeric_vector(metrics: Dict[str,Any], max_ues: int = None) -> np.ndarray:
    t, d, u = _env_signature(metrics)
    u = list(u)
    if max_ues is not None:
        u = (u + [0.0]*max_ues)[:max_ues]
    return np.array([t, d] + u, dtype=float)

def numeric_distance_bounded_mse(m_pred: Dict[str,Any], m_true: Dict[str,Any]) -> float:
    # build vectors with same length (pad by max UEs found in either)
    _, _, u1 = _env_signature(m_pred)
    _, _, u2 = _env_signature(m_true)
    n = max(len(u1), len(u2))
    v1 = _numeric_vector(m_pred, max_ues=n)
    v2 = _numeric_vector(m_true, max_ues=n)
    eps = 1e-8
    denom = np.maximum(np.maximum(np.abs(v1), np.abs(v2)), eps)
    errs = ((v1 - v2) / denom)**2
    errs = np.clip(errs, 0.0, 1.0)  # bound each coord to [0,1]
    return float(np.mean(errs))

# ----------------------- digital twin: knobs & transforms -----------------------

# We do NOT change ns-3 command-line (to keep this robust). Instead,
# we model a "twin layer" that perturbs KPIs deterministically given knobs θ
# and a twin-seed (for un-inferred noise), applied on top of base ns-3 KPIs.

def _rng_from_key(key: str, base_seed: int) -> random.Random:
    h = hashlib.sha256()
    h.update(key.encode("utf-8"))
    h.update(str(base_seed).encode("utf-8"))
    return random.Random(int.from_bytes(h.digest()[:8], "big") & 0x7FFFFFFF)

def _apply_twin_low(theta: Dict[str,float], base_metrics: Dict[str,Any], twin_seed_key: str, base_seed: int) -> Dict[str,Any]:
    """
    Low-fidelity: θ = {shadow_sigma_db, doppler_hz, sched_jitter}
    Effect (simple, monotone, deterministic + tiny pseudo-noise):
      - total_throughput ↓ with shadow_sigma_db and doppler_hz
      - per-UE redistributed with sched_jitter
      - delay ↑ slightly with shadow_sigma_db and doppler_hz
    """
    rng = _rng_from_key(twin_seed_key + "|low", base_seed)
    t, d, u = _env_signature(base_metrics)
    u = list(u)

    shadow = max(0.0, float(theta.get("shadow_sigma_db", 4.0)))  # [0, ~8]
    doppler = max(0.0, float(theta.get("doppler_hz", 3.0)))      # [0, ~10]
    jitter = max(0.0, float(theta.get("sched_jitter", 0.1)))     # [0, ~0.5]

    # throughput scale (decrease)
    scale_t = math.exp(-0.03*shadow) * (1.0 - 0.02*min(doppler, 10.0))
    scale_t = max(0.5, min(1.05, scale_t))

    # delay scale (increase)
    scale_d = 1.0 + 0.02*shadow + 0.01*min(doppler, 10.0)
    scale_d = max(0.7, min(2.0, scale_d))

    # jitter redistributes per-UE, conserving sum
    if len(u) > 0:
        tot = max(1e-9, sum(u))
        noise = [rng.uniform(-jitter, jitter) for _ in u]
        # softplus-ish perturb, then renormalize to keep sum
        nu = [max(0.0, ui * (1.0 + ni)) for ui, ni in zip(u, noise)]
        su = sum(nu) or 1e-9
        nu = [x * tot / su for x in nu]
    else:
        nu = u

    out = dict(base_metrics)
    # write back (favor total as sum(nu))
    out["per_ue_throughput_mbps"] = nu
    out["total_throughput_mbps"] = float(sum(nu)) * scale_t
    out["avg_delay_ms"] = float(d) * scale_d
    return out

def _apply_twin_high(theta: Dict[str,float], base_metrics: Dict[str,Any], twin_seed_key: str, base_seed: int) -> Dict[str,Any]:
    """
    High-fidelity: θ = low knobs + extras:
        pathloss_bias_db, ue_bias_1..k (first K UEs), fading_corr, delay_bias
    Effects:
      - total throughput impacted by pathloss_bias_db and fading_corr
      - per-UE offsets via ue_bias_i (conserve sum overall)
      - delay scaled and biased
    """
    rng = _rng_from_key(twin_seed_key + "|high", base_seed)
    t, d, u = _env_signature(base_metrics)
    u = list(u)

    # inherit low knobs as baseline
    shadow = max(0.0, float(theta.get("shadow_sigma_db", 4.0)))
    doppler = max(0.0, float(theta.get("doppler_hz", 3.0)))
    jitter = max(0.0, float(theta.get("sched_jitter", 0.1)))

    pathloss_bias = float(theta.get("pathloss_bias_db", 0.0))  # [-6, +6]
    fading_corr = max(0.0, min(1.0, float(theta.get("fading_corr", 0.3))))
    delay_bias = float(theta.get("delay_bias_ms", 0.0))        # [-20, +20]

    # per-UE biases: ue_bias_1 .. ue_bias_K (we'll honor up to len(u))
    ue_biases = [float(theta.get(f"ue_bias_{i+1}", 0.0)) for i in range(len(u))]
    # ---- start from low twin ----
    interim = _apply_twin_low(
        {"shadow_sigma_db": shadow, "doppler_hz": doppler, "sched_jitter": jitter},
        base_metrics, twin_seed_key, base_seed
    )
    tt, dd, uu = _env_signature(interim)
    uu = list(uu)

    # apply per-UE biases
    if len(uu) > 0:
        tot = max(1e-9, sum(uu))
        biased = [max(0.0, ui*(1.0 + b)) for ui, b in zip(uu, ue_biases)]
        sb = sum(biased) or 1e-9
        biased = [x * tot / sb for x in biased]
    else:
        biased = uu

    # pathloss_bias_db: lower throughput when positive (worse pathloss)
    scale_pl = math.exp(-0.02*max(-8.0, min(8.0, pathloss_bias)))
    # fading_corr: slightly changes effective throughput (more corr => less multiuser diversity)
    scale_fc = 1.0 - 0.05*fading_corr
    t_out = float(sum(biased)) * scale_pl * scale_fc

    # delay: scaled via correlation and pathloss + additive bias
    d_scale = 1.0 + 0.01*abs(pathloss_bias) + 0.03*fading_corr
    d_out = float(dd) * d_scale + float(delay_bias)

    out = dict(interim)
    out["per_ue_throughput_mbps"] = biased
    out["total_throughput_mbps"] = t_out
    out["avg_delay_ms"] = d_out
    return out

# loss to fit twin knobs against observed factual KPIs
def _kpi_loss(pred: Dict[str,Any], obs: Dict[str,Any]) -> float:
    # numeric bounded MSE on vectors
    return numeric_distance_bounded_mse(pred, obs)

def _sample_theta_low(rnd: random.Random) -> Dict[str,float]:
    return {
        "shadow_sigma_db": rnd.uniform(0.0, 8.0),
        "doppler_hz": rnd.uniform(0.0, 10.0),
        "sched_jitter": rnd.uniform(0.0, 0.5),
    }

def _sample_theta_high(rnd: random.Random, n_ues: int) -> Dict[str,float]:
    th = _sample_theta_low(rnd)
    th.update({
        "pathloss_bias_db": rnd.uniform(-6.0, 6.0),
        "fading_corr": rnd.uniform(0.0, 1.0),
        "delay_bias_ms": rnd.uniform(-20.0, 20.0),
    })
    # per-UE biases
    for i in range(n_ues):
        th[f"ue_bias_{i+1}"] = rnd.uniform(-0.5, 0.5)
    return th

def _refine_theta_coord_descent(theta0: Dict[str,float],
                                apply_fn,
                                base_metrics: Dict[str,Any],
                                obs_metrics: Dict[str,Any],
                                twin_seed_key: str,
                                base_seed: int,
                                step: float,
                                iters: int) -> Dict[str,float]:
    rnd = _rng_from_key("refine|" + twin_seed_key, base_seed)
    theta = dict(theta0)
    best_pred = apply_fn(theta, base_metrics, twin_seed_key, base_seed)
    best_loss = _kpi_loss(best_pred, obs_metrics)

    keys = list(theta.keys())
    for _ in range(iters):
        improved = False
        rnd.shuffle(keys)
        for k in keys:
            cur = theta[k]
            for delta in (+step, -step, +step/2, -step/2):
                theta[k] = cur + delta
                pred = apply_fn(theta, base_metrics, twin_seed_key, base_seed)
                loss = _kpi_loss(pred, obs_metrics)
                if loss + 1e-9 < best_loss:
                    best_loss = loss
                    best_pred = pred
                    improved = True
                    cur = theta[k]
            theta[k] = cur  # keep best
        if not improved:
            break
    return theta

# ----------------------- main experiment -----------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate CF with fitted digital twins (low/high) + interventional baseline.")
    ap.add_argument("--data-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "test.jsonl"))
    ap.add_argument("--temps", type=str, default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--metric", type=str, default="levenshtein_norm", help="Text distance metric key from METRICS.")
    ap.add_argument("--max-cases", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)  # kept for CLI compatibility; we run serial for robustness
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "cf_twins_eval"))
    ap.add_argument("--cache", action="store_true", help="Cache ns-3 metrics by (action, rng_run).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--drop-temps", type=str, default="", help="Comma list of temperatures to drop from plots.")
    ap.add_argument("--use-tex", action="store_true", help="Use LaTeX text rendering in plots.")
    # budgets
    ap.add_argument("--low-budget", type=int, default=32, help="Random search trials for low-fid.")
    ap.add_argument("--high-budget", type=int, default=96, help="Random search trials for high-fid (includes refine).")
    ap.add_argument("--high-refine-iters", type=int, default=20, help="Coord-descent refinement iterations.")
    ap.add_argument("--high-refine-step", type=float, default=0.1, help="Coord-descent initial step.")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    cache_dir = os.path.join(args.outdir, "ns3_cache") if args.cache else None

    # plotting setup (avoid unicode)
    if args.use_tex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
    else:
        plt.rcParams["text.usetex"] = False

    # distances
    if args.metric not in METRICS:
        raise KeyError(f"Unknown text metric '{args.metric}'. Available: {list(METRICS.keys())}")
    text_metric = METRICS[args.metric]

    # data
    rows = read_jsonl(args.data_path)
    if args.max_cases is not None:
        rows = rows[:args.max_cases]
    if not rows:
        raise RuntimeError(f"No rows in {args.data_path}")

    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    drop = set([float(x.strip()) for x in args.drop_temps.split(",") if x.strip()])
    temps = [t for t in temps if t not in drop]
    temps = temps or [0.7]

    # per-case loop
    records = []
    pbar = tqdm(total=len(rows), desc="Cases", leave=True)

    for idx, r in enumerate(rows):
        # build actions from X and X'
        X  = r["X"]
        Xp = r["X_prime"]
        A_raw  = action_from_prompt(X,  seed=args.seed, temperature=0.2)  # stable mapping
        Ap_raw = action_from_prompt(Xp, seed=args.seed, temperature=0.2)
        A  = _normalize_action(A_raw)
        Ap = _normalize_action(Ap_raw)

        # true env run id (unknown to twins)
        rng_true_env = _small_rng(hash((idx, "env_true", args.seed)))
        # twin env base seed (different from true; represents unknown, un-inferred noise)
        rng_twin_env = _small_rng(hash((idx, "env_twin", args.seed)))
        # interventional uses yet another fresh seed
        rng_interv_env = _small_rng(hash((idx, "env_interv", args.seed)))

        # factual metrics
        m_f_true = _safe_run_ns3_action(A, rng_run=rng_true_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)
        # cf truth with same env seed
        m_cf_true = _safe_run_ns3_action(Ap, rng_run=rng_true_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # twin base metrics (we always run the twin with its own seed)
        m_f_base_twin  = _safe_run_ns3_action(A,  rng_run=rng_twin_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)
        m_cf_base_twin = _safe_run_ns3_action(Ap, rng_run=rng_twin_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # ---------------- LOW-FID twin fit ----------------
        rnd_low = random.Random((args.seed ^ 0xA5A5) + 31*idx)
        best_low_theta = None
        best_low_loss  = float("inf")
        # search around number of UEs (for per-UE bias vector length in high fid later)
        _, _, u_base = _env_signature(m_f_base_twin)
        n_ues = len(u_base)

        for _ in range(args.low_budget):
            th = _sample_theta_low(rnd_low)
            pred = _apply_twin_low(th, m_f_base_twin, twin_seed_key=f"low|{idx}|fit", base_seed=args.seed)
            loss = _kpi_loss(pred, m_f_true)
            if loss < best_low_loss:
                best_low_loss = loss
                best_low_theta = th

        # low-fid CF estimate
        m_cf_low = _apply_twin_low(best_low_theta, m_cf_base_twin, twin_seed_key=f"low|{idx}|cf", base_seed=args.seed)

        # ---------------- HIGH-FID twin fit ----------------
        rnd_high = random.Random((args.seed ^ 0x5A5A) + 97*idx)
        # warm start from expanded low-theta
        init_high = _sample_theta_high(rnd_high, n_ues)  # random init
        # replace overlapping fields with low best (warm start)
        for k in ("shadow_sigma_db","doppler_hz","sched_jitter"):
            init_high[k] = best_low_theta[k]

        best_high_theta = init_high
        best_high_pred  = _apply_twin_high(best_high_theta, m_f_base_twin, twin_seed_key=f"high|{idx}|fit", base_seed=args.seed)
        best_high_loss  = _kpi_loss(best_high_pred, m_f_true)

        # random exploration (global)
        trials = max(0, args.high_budget - args.high_refine_iters)
        for _ in range(trials):
            th = _sample_theta_high(rnd_high, n_ues)
            # tie to low-theta a bit (optional): small blend
            for k in ("shadow_sigma_db","doppler_hz","sched_jitter"):
                th[k] = 0.7*best_low_theta[k] + 0.3*th[k]
            pred = _apply_twin_high(th, m_f_base_twin, twin_seed_key=f"high|{idx}|fit", base_seed=args.seed)
            loss = _kpi_loss(pred, m_f_true)
            if loss < best_high_loss:
                best_high_loss, best_high_theta, best_high_pred = loss, th, pred

        # local coordinate refinement
        best_high_theta = _refine_theta_coord_descent(
            best_high_theta, _apply_twin_high, m_f_base_twin, m_f_true,
            twin_seed_key=f"high|{idx}|fit", base_seed=args.seed,
            step=args.high_refine_step, iters=args.high_refine_iters
        )
        best_high_pred = _apply_twin_high(best_high_theta, m_f_base_twin, twin_seed_key=f"high|{idx}|fit", base_seed=args.seed)
        best_high_loss = _kpi_loss(best_high_pred, m_f_true)

        # high-fid CF estimate
        m_cf_high = _apply_twin_high(best_high_theta, m_cf_base_twin, twin_seed_key=f"high|{idx}|cf", base_seed=args.seed)

        # ---------------- Interventional baseline (fresh env + fresh LLM) ----------------
        m_cf_interv = _safe_run_ns3_action(Ap, rng_run=rng_interv_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # ---------------- Reports & distances per temperature ----------------
        for T in temps:
            # "GM-SCM-like": same LLM randomness for truth & twins
            y_true  = report_from_metrics(m_cf_true,  seed=args.seed,        temperature=T)
            y_low   = report_from_metrics(m_cf_low,   seed=args.seed,        temperature=T)
            y_high  = report_from_metrics(m_cf_high,  seed=args.seed,        temperature=T)
            # interventional: fresh LLM rng
            y_inter = report_from_metrics(m_cf_interv, seed=args.seed + 999, temperature=T)

            d_txt_low   = float(text_metric(y_low,  y_true))
            d_txt_high  = float(text_metric(y_high, y_true))
            d_txt_inter = float(text_metric(y_inter,y_true))

            d_num_low   = numeric_distance_bounded_mse(m_cf_low,   m_cf_true)
            d_num_high  = numeric_distance_bounded_mse(m_cf_high,  m_cf_true)
            d_num_inter = numeric_distance_bounded_mse(m_cf_interv, m_cf_true)

            records += [
                {"idx": idx, "temperature": T, "method": "GM-SCM (Low Twin)",  "text_dist": d_txt_low,  "numeric_dist": d_num_low},
                {"idx": idx, "temperature": T, "method": "GM-SCM (High Twin)", "text_dist": d_txt_high, "numeric_dist": d_num_high},
                {"idx": idx, "temperature": T, "method": "Interventional",      "text_dist": d_txt_inter,"numeric_dist": d_num_inter},
            ]
        pbar.update(1)

    pbar.close()

    # ---------------- Save CSVs ----------------
    df = pd.DataFrame(records)
    ensure_dir(args.outdir)
    csv_per = os.path.join(args.outdir, "twins_per_case.csv")
    df.to_csv(csv_per, index=False)

    # Summaries (by temperature & method)
    summ = df.groupby(["temperature","method"]).agg(
        avg_text=("text_dist","mean"),
        avg_numeric=("numeric_dist","mean"),
        n=("text_dist","count")
    ).reset_index()
    csv_sum = os.path.join(args.outdir, "twins_summary.csv")
    summ.to_csv(csv_sum, index=False)

    # ---------------- Plots (PDFs) ----------------
    # 1) TEXT average vs temperature
    plt.figure()
    for method in ["GM-SCM (Low Twin)","GM-SCM (High Twin)","Interventional"]:
        sub = summ[summ["method"]==method].sort_values("temperature")
        if not len(sub): continue
        plt.plot(sub["temperature"].values, sub["avg_text"].values, marker="o", label=method)
    plt.xlabel(r"Temperature $\tau$" if args.use_tex else "Temperature tau")
    plt.ylabel("Average text distance to truth")
    plt.legend()
    plt.tight_layout()
    fig1 = os.path.join(args.outdir, "text_distance_vs_temp.pdf")
    plt.savefig(fig1)
    plt.close()

    # 2) NUMERIC average vs temperature
    plt.figure()
    for method in ["GM-SCM (Low Twin)","GM-SCM (High Twin)","Interventional"]:
        sub = summ[summ["method"]==method].sort_values("temperature")
        if not len(sub): continue
        plt.plot(sub["temperature"].values, sub["avg_numeric"].values, marker="o", label=method)
    plt.xlabel(r"Temperature $\tau$" if args.use_tex else "Temperature tau")
    plt.ylabel("Average numeric distance to truth (bounded MSE)")
    plt.legend()
    plt.tight_layout()
    fig2 = os.path.join(args.outdir, "numeric_distance_vs_temp.pdf")
    plt.savefig(fig2)
    plt.close()

    # 3) NUMERIC boxplot pooled over temps
    plt.figure()
    data_num = [df[df["method"]==m]["numeric_dist"].values
                for m in ["GM-SCM (Low Twin)","GM-SCM (High Twin)","Interventional"]]
    plt.boxplot(data_num, tick_labels=["GM-SCM (Low Twin)","GM-SCM (High Twin)","Interventional"], showfliers=False)
    plt.ylabel("Numeric distance to truth (bounded MSE)")
    plt.tight_layout()
    fig3 = os.path.join(args.outdir, "numeric_distance_boxplot.pdf")
    plt.savefig(fig3)
    plt.close()

    # 4) TEXT boxplot pooled over temps
    plt.figure()
    data_txt = [df[df["method"]==m]["text_dist"].values
                for m in ["GM-SCM (Low Twin)","GM-SCM (High Twin)","Interventional"]]
    plt.boxplot(data_txt, tick_labels=["GM-SCM (Low Twin)","GM-SCM (High Twin)","Interventional"], showfliers=False)
    plt.ylabel("Text distance to truth")
    plt.tight_layout()
    fig4 = os.path.join(args.outdir, "text_distance_boxplot.pdf")
    plt.savefig(fig4)
    plt.close()

    print(f"[✓] Per-case CSV:  {csv_per}")
    print(f"[✓] Summary CSV:   {csv_sum}")
    print(f"[✓] Figure (text avg vs T):     {fig1}")
    print(f"[✓] Figure (numeric avg vs T):  {fig2}")
    print(f"[✓] Figure (numeric box):       {fig3}")
    print(f"[✓] Figure (text box):          {fig4}")


if __name__ == "__main__":
    main()
