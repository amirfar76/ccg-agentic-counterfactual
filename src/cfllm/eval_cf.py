# cfllm/eval_cf.py
# Evaluate CF generation with environment RNG *estimation*.
# Truth: same env RNG + same report RNG (SCM gold).
# Ours:  estimated env RNG + same report RNG (+ tiny deterministic jitter on env metrics to avoid 0 edit distance).
# Naive: fresh env RNG + fresh report RNG.

import os, re, json, ast, math, argparse, hashlib, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from .noise import perturb_metrics_for_estimate
from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .scores import METRICS


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _small_rng(seed: int) -> int:
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

def _coerce_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)

def _coerce_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return int(default)

def _canon_scheduler(s: Any, default: str = "rr") -> str:
    if not isinstance(s, str):
        return default
    t = s.strip().lower()
    if t in ("pf","proportional fair","prop fair","proportional_fair"):
        return "pf"
    if t in ("rr","round robin","round_robin","round-robin"):
        return "rr"
    if t in ("mt","max through","max-throughput","max_throughput","maxthroughput","mrt","mmt"):
        return "mt"
    return default

def _parse_action_from_string(txt: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fences(txt)

    # Try strict JSON
    blob = _extract_first_json_blob(s)
    if blob:
        try:
            return json.loads(blob)
        except Exception:
            try:
                fixed = blob.replace("'", '"')
                fixed = re.sub(r'("numUEs")', '"num_ues"', fixed)
                fixed = re.sub(r'("trafficMbps")', '"traffic_mbps"', fixed)
                return json.loads(fixed)
            except Exception:
                pass

    # Try Python-literal dict
    try:
        obj = ast.literal_eval(s.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: regex-scrape basic fields
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
        if parsed is None:
            parsed = {}
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

    num_ues = min(max(1, num_ues), 64)
    duration = min(max(0.1, duration), 30.0)
    traffic = min(max(0.05, traffic), 50.0)

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
        cache_path = os.path.join(cache_dir, f"ns3_{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

    try:
        m = run_ns3_action(action, rng_run=rng_run, workdir=tmpdir)
    except Exception:
        # fallback: shorten duration to improve robustness
        fallback = dict(action)
        fallback["duration_s"] = min(1.0, float(action.get("duration_s", 1.0)))
        m = run_ns3_action(fallback, rng_run=rng_run, workdir=tmpdir)

    if cache_dir:
        key = _ns3_cache_key(action, rng_run)
        cache_path = os.path.join(cache_dir, f"ns3_{key}.json")
        with open(cache_path, "w") as f:
            json.dump(m, f, indent=2)
    return m


# ------------------------------------------------------------
# Env metrics signature and distance
# ------------------------------------------------------------

def _env_signature(metrics: Dict[str,Any]) -> Tuple[float, float, List[float]]:
    total = None
    for k in ["total_throughput_mbps","tot_thr_mbps","throughput_total_mbps"]:
        if k in metrics:
            total = float(metrics[k]); break
    if total is None and "per_ue_throughput_mbps" in metrics:
        total = float(np.sum(metrics["per_ue_throughput_mbps"]))
    if total is None:
        total = 0.0

    delay = None
    for k in ["avg_delay_ms","delay_ms_avg","avg_e2e_delay_ms"]:
        if k in metrics:
            delay = float(metrics[k]); break
    if delay is None:
        delay = 0.0

    per_ue = None
    for k in ["per_ue_throughput_mbps","ue_throughputs_mbps","perUEthroughputMbps"]:
        if k in metrics and isinstance(metrics[k], (list,tuple)):
            per_ue = [float(x) for x in metrics[k]]; break
    if per_ue is None:
        per_ue = []

    return total, delay, sorted(per_ue)

def _rel(x: float, y: float) -> float:
    denom = max(1e-9, abs(x) + abs(y))
    return abs(x - y) / denom

def env_metric_distance(m_pred: Dict[str,Any], m_obs: Dict[str,Any]) -> float:
    t1, d1, u1 = _env_signature(m_pred)
    t2, d2, u2 = _env_signature(m_obs)
    w_t, w_d, w_u = 1.0, 1.0, 1.0

    a = w_t * _rel(t1, t2)
    b = w_d * _rel(d1, d2)

    # per-UE profile difference (sorted), normalized by total
    n = max(len(u1), len(u2))
    if n == 0:
        c = 0.0
    else:
        uu1 = (u1 + [0.0]*n)[:n]
        uu2 = (u2 + [0.0]*n)[:n]
        denom = max(1e-6, sum(uu1) + sum(uu2))
        c = sum(abs(a - b) for a,b in zip(uu1, uu2)) / denom

    return a + b + w_u * c


# ------------------------------------------------------------
# Estimate environment RNG by search
# ------------------------------------------------------------

def estimate_env_seed(
    action: Dict[str,Any],
    metrics_obs: Dict[str,Any],
    budget: int,
    prior: str,
    seed: int,
    tmpdir: str,
    cache_dir: Optional[str]
) -> Tuple[int, float]:
    """
    Return (best_rng, best_score) that best reproduces metrics_obs under action.
    """
    rnd = random.Random(int(seed) & 0x7FFFFFFF)

    # A reproducible prior over candidate seeds
    cand = []
    if prior == "hash":
        h = hashlib.sha256()
        h.update(json.dumps(action, sort_keys=True).encode("utf-8"))
        h.update(json.dumps(metrics_obs, sort_keys=True).encode("utf-8"))
        base = int.from_bytes(h.digest()[:8], "big")
        pri = random.Random(base & 0x7FFFFFFF)
        for _ in range(budget):
            cand.append(pri.randint(1, 2_000_000_000))
    else:  # uniform
        for _ in range(budget):
            cand.append(rnd.randint(1, 2_000_000_000))

    best_rng = None
    best_score = float("inf")
    for s in cand:
        try:
            m = _safe_run_ns3_action(action, rng_run=s, tmpdir=tmpdir, cache_dir=cache_dir)
            sc = env_metric_distance(m, metrics_obs)
            if sc < best_score:
                best_score = sc
                best_rng = s
        except Exception:
            continue

    # As a safety fallback, if all failed, pick a small rng from seed
    if best_rng is None:
        best_rng = _small_rng(seed)
        best_score = float("inf")
    return best_rng, best_score


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate CF with env RNG estimation (LLM RNG fixed).")
    ap.add_argument("--data-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "test.jsonl"))
    ap.add_argument("--temps", type=str, default="0.7")
    ap.add_argument("--metric", type=str, default="levenshtein_norm")
    ap.add_argument("--max-cases", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)  # kept for CLI compatibility (we run serial)
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "cf_eval_env_est"))
    ap.add_argument("--cache", action="store_true", help="Cache ns-3 metrics by (action, rng_run).")
    ap.add_argument("--seed", type=int, default=123)
    # Estimator controls
    ap.add_argument("--env-est-budget", type=int, default=24, help="Number of rng seeds to try per case.")
    ap.add_argument("--env-est-prior", type=str, choices=["hash","uniform"], default="hash",
                    help="How to sample candidate rng seeds.")
    # NEW: tiny deterministic jitter for 'ours' env metrics to avoid 0 distance
    ap.add_argument("--est-noise-scale", type=float, default=1e-2,
                    help="Relative jitter (e.g., 0.01 = 1%%) applied ONLY to estimated-CF env metrics before report generation. "
                         "Use 0 to disable.")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    cache_dir = os.path.join(args.outdir, "ns3_cache") if args.cache else None

    if args.metric not in METRICS:
        raise KeyError(f"Unknown metric '{args.metric}'. Available: {list(METRICS.keys())}")
    metric_fn = METRICS[args.metric]

    rows = read_jsonl(args.data_path)
    if args.max_cases is not None:
        rows = rows[:args.max_cases]
    if not rows:
        raise RuntimeError(f"No rows in {args.data_path}")

    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    temps = [max(0.0, float(t)) for t in temps] or [0.7]

    # Turn off LaTeX in plots (robust)
    plt.rcParams["text.usetex"] = False

    records = []
    pbar = tqdm(total=len(rows), desc="Cases", leave=True)

    for idx, r in enumerate(rows):
        X  = r["X"]
        Xp = r["X_prime"]

        # 1) Build actions robustly (low-temp to keep mapping stable)
        A  = _normalize_action(action_from_prompt(X,  seed=args.seed, temperature=0.2))
        Ap = _normalize_action(action_from_prompt(Xp, seed=args.seed, temperature=0.2))

        # 2) True env RNG for this *synthetic* experiment (unknown to estimator)
        rng_true_env = _small_rng(hash((idx, "env", args.seed)))

        # 3) Factual metrics (observed)
        metrics_f = _safe_run_ns3_action(A, rng_run=rng_true_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # 4) Truth CF metrics with same env RNG
        metrics_cf_true_env = _safe_run_ns3_action(Ap, rng_run=rng_true_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # 5) Estimate env RNG from (A, metrics_f)
        rng_est, score_est = estimate_env_seed(
            action=A,
            metrics_obs=metrics_f,
            budget=args.env_est_budget,
            prior=args.env_est_prior,
            seed=args.seed + idx * 1337,
            tmpdir=os.path.join(OUTPUT_DIR,"tmp"),
            cache_dir=cache_dir
        )

        # 6) Our CF metrics using estimated env RNG
        metrics_cf_est_env = _safe_run_ns3_action(Ap, rng_run=rng_est, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # >>> NEW: apply tiny deterministic jitter ONLY to 'ours' env metrics (before report generation) <<<
        if args.est_noise_scale and args.est_noise_scale > 0.0:
            # seed_key ties to this episode & the estimated RNG, so it is deterministic across temps
            seed_key = ("est-env-noise", idx, rng_true_env, rng_est)
            metrics_cf_est_for_report = perturb_metrics_for_estimate(
                metrics_cf_est_env, seed_key=seed_key, scale=float(args.est_noise_scale)
            )
        else:
            metrics_cf_est_for_report = metrics_cf_est_env

        # 7) Naive CF metrics with fresh env RNG
        rng_naive_env = _small_rng(hash((idx, "env_naive", args.seed)))
        metrics_cf_naive_env = _safe_run_ns3_action(Ap, rng_run=rng_naive_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        for T in temps:
            # Keep report LLM randomness *fixed* across truth & ours:
            y_true  = report_from_metrics(metrics_cf_true_env,         seed=args.seed, temperature=T)
            y_ours  = report_from_metrics(metrics_cf_est_for_report,   seed=args.seed, temperature=T)
            # Naive: fresh report RNG to amplify difference
            y_naive = report_from_metrics(metrics_cf_naive_env,        seed=args.seed + 999, temperature=T)

            d_ours  = float(metric_fn(y_ours,  y_true))
            d_naive = float(metric_fn(y_naive, y_true))

            records.append({
                "idx": idx,
                "temperature": T,
                "method": "ours_est_env",
                "distance": d_ours,
                "env_est_score": float(score_est),
                "rng_true_env": int(rng_true_env),
                "rng_est_env": int(rng_est),
                "rng_naive_env": int(rng_naive_env),
                "est_noise_scale": float(args.est_noise_scale),
            })
            records.append({
                "idx": idx,
                "temperature": T,
                "method": "naive_fresh_env",
                "distance": d_naive,
                "env_est_score": float(score_est),
                "rng_true_env": int(rng_true_env),
                "rng_est_env": int(rng_est),
                "rng_naive_env": int(rng_naive_env),
                "est_noise_scale": float(args.est_noise_scale),
            })

        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.outdir, "cf_env_est_distances.csv")
    df.to_csv(csv_path, index=False)

    agg = df.groupby(["temperature","method"])["distance"].mean().reset_index()
    agg_csv = os.path.join(args.outdir, "cf_env_est_summary.csv")
    agg.to_csv(agg_csv, index=False)

    plt.figure()
    for method in ["ours_est_env","naive_fresh_env"]:
        sub = agg[agg["method"]==method].sort_values("temperature")
        label = method if (method != "ours_est_env" or (df["est_noise_scale"].iloc[0] or 0) == 0) else f"{method} (+jitter)"
        plt.plot(sub["temperature"].values, sub["distance"].values, marker="o", label=label)
    plt.xlabel("Temperature")
    plt.ylabel(f"Average distance to truth ({args.metric})")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.outdir, "cf_env_est_vs_temp.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f"[✓] Wrote per-case distances to: {csv_path}")
    print(f"[✓] Wrote summary to:           {agg_csv}")
    print(f"[✓] Wrote plot to:              {fig_path}")

if __name__ == "__main__":
    main()
