# Evaluate CF generation with *correct* seed discipline and clean numeric metric.
# Truth (ground truth): env RNG = factual RNG; report RNG fixed
# GM-SCM (ours):        env RNG = estimated (NPE or search); report RNG fixed
# Interventional:       env RNG fresh; report RNG fresh
#
# Outputs:
#  - CSV of per-case distances + seed audit
#  - PNG/PDF plots of edit distance vs temp, and numeric distance distributions

import os, re, json, ast, math, argparse, hashlib, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def _small_rng(x: int) -> int:
    rnd = random.Random(int(x) & 0x7FFFFFFF)
    v = rnd.randint(1, 2_000_000_000)
    return v if v != 0 else 1

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
        if txt[i] == "{": depth += 1
        elif txt[i] == "}":
            depth -= 1
            if depth == 0:
                return txt[start:i+1]
    return None

def _coerce_float(val: Any, default: float) -> float:
    try: return float(val)
    except Exception: return float(default)

def _coerce_int(val: Any, default: int) -> int:
    try: return int(val)
    except Exception: return int(default)

def _canon_scheduler(s: Any, default: str = "rr") -> str:
    if not isinstance(s, str): return default
    t = s.strip().lower()
    if t in ("pf","proportional fair","prop fair","proportional_fair"): return "pf"
    if t in ("rr","round robin","round_robin","round-robin"): return "rr"
    if t in ("mt","max through","max-throughput","max_throughput","maxthroughput","mrt","mmt"): return "mt"
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

    num_ues = min(max(1, num_ues), 64)
    duration = min(max(0.1, duration), 60.0)
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
# Environment metric signature & distance (for seed search)
# ------------------------------------------------------------

def _env_signature(metrics: Dict[str,Any]) -> Tuple[float, float, List[float]]:
    total = None
    for k in ["throughput_total_mbps","total_throughput_mbps"]:
        if k in metrics:
            total = float(metrics[k]); break
    if total is None and "ue_throughput_mbps" in metrics:
        total = float(np.sum(metrics["ue_throughput_mbps"]))
    if total is None and "per_ue_throughput_mbps" in metrics:
        total = float(np.sum(metrics["per_ue_throughput_mbps"]))
    if total is None: total = 0.0

    delay = None
    for k in ["avg_delay_ms","delay_ms_avg","avg_e2e_delay_ms"]:
        if k in metrics:
            delay = float(metrics[k]); break
    if delay is None: delay = 0.0

    per_ue = None
    for k in ["ue_throughput_mbps","per_ue_throughput_mbps"]:
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
        c = sum(abs(x - y) for x,y in zip(uu1, uu2)) / denom

    return a + b + c

# ------------------------------------------------------------
# Estimate env RNG by small search (fallback if NPE not wired)
# ------------------------------------------------------------

def estimate_env_seed_search(
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

    cand = []
    if prior == "hash":
        h = hashlib.sha256()
        h.update(json.dumps(action, sort_keys=True).encode("utf-8"))
        h.update(json.dumps(metrics_obs, sort_keys=True).encode("utf-8"))
        base = int.from_bytes(h.digest()[:8], "big")
        pri = random.Random(base & 0x7FFFFFFF)
        for _ in range(budget):
            cand.append(pri.randint(1, 2_000_000_000))
    else:
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

    if best_rng is None:
        best_rng = _small_rng(seed)
        best_score = float("inf")
    return best_rng, best_score

# ------------------------------------------------------------
# Numeric distance (robust, no all-0/all-1 pathologies)
# ------------------------------------------------------------

def _vectorize_metrics(m: Dict[str,Any], pad_ues: int = 16) -> np.ndarray:
    tot = float(m.get("throughput_total_mbps", m.get("total_throughput_mbps", 0.0)))
    dly = float(m.get("avg_delay_ms", m.get("delay_ms_avg", 0.0)))
    ue  = m.get("ue_throughput_mbps", m.get("per_ue_throughput_mbps", []))
    ue  = [float(x) for x in ue] if isinstance(ue, (list, tuple)) else []
    ue  = (sorted(ue) + [0.0] * pad_ues)[:pad_ues]
    return np.array([tot, dly] + ue, dtype=float)

def numeric_distance(m_pred: Dict[str,Any], m_true: Dict[str,Any]) -> float:
    v_true = _vectorize_metrics(m_true)
    v_pred = _vectorize_metrics(m_pred)
    eps = 1e-6
    scale = np.maximum(np.abs(v_true), eps)  # per-dimension scaling
    return float(np.mean(((v_pred - v_true) / scale) ** 2))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate CF with correct seeding & numeric metric (GM-SCM vs Interventional).")
    ap.add_argument("--data-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "test.jsonl"))
    ap.add_argument("--temps", type=str, default="0.7")
    ap.add_argument("--metric", type=str, default="levenshtein_norm")
    ap.add_argument("--max-cases", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)  # compatibility; we run serial
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "cf_eval_compare"))
    ap.add_argument("--cache", action="store_true", help="Cache ns-3 metrics by (action, rng_run).")
    ap.add_argument("--seed", type=int, default=123)

    # Optional estimator controls (used if you don't plug NPE):
    ap.add_argument("--estimator", type=str, choices=["search","npe"], default="search",
                    help="Env RNG estimator. If 'npe' not wired, falls back to 'search'.")
    ap.add_argument("--est-budget", type=int, default=64, help="Number of rng seeds to try (search).")
    ap.add_argument("--est-prior", type=str, choices=["hash","uniform"], default="hash")

    # Plot options
    ap.add_argument("--use-tex", action="store_true")
    ap.add_argument("--pdf", action="store_true", help="Also save plots as PDF.")

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

    # LaTeX toggle
    plt.rcParams["text.usetex"] = bool(args.use_tex)
    if args.use_tex:
        plt.rcParams["font.family"] = "serif"

    records = []
    pbar = tqdm(total=len(rows), desc="Cases", leave=True)

    for idx, r in enumerate(rows):
        X  = r["X"]
        Xp = r["X_prime"]

        # Actions (low-temp to keep mapping stable)
        A  = _normalize_action(action_from_prompt(X,  seed=args.seed, temperature=0.2))
        Ap = _normalize_action(action_from_prompt(Xp, seed=args.seed, temperature=0.2))

        # --------- Seeds for the three roles ----------
        rng_true_env = _small_rng(hash((idx, "env_true", args.seed)))
        rng_est_base = _small_rng(hash((idx, "env_est",  args.seed)))
        rng_intv_env = _small_rng(hash((idx, "env_intv", args.seed)))
        if rng_intv_env == rng_true_env:
            rng_intv_env = (rng_true_env + 7919 * (idx + 1)) % 2_000_000_000 or 1

        seed_report_fixed = int(args.seed)            # used for factual + truth + ours
        seed_report_fresh = int(args.seed) + 99991    # used only for interventional

        # --------- Factual metrics (observed) --------
        metrics_f = _safe_run_ns3_action(A, rng_run=rng_true_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # --------- True counterfactual (GROUND TRUTH) ----------
        metrics_cf_true = _safe_run_ns3_action(Ap, rng_run=rng_true_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # --------- Our env RNG estimate ----------
        if args.estimator == "npe":
            # Hook: replace with your NPE posterior sample:
            # rng_est_env = sample_from_npe_posterior(...);  # <-- your code path
            # For now, we fall back to search if not wired:
            rng_est_env, _ = estimate_env_seed_search(
                action=A, metrics_obs=metrics_f, budget=args.est_budget,
                prior=args.est_prior, seed=args.seed + 1337*idx,
                tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir
            )
        else:
            rng_est_env, _ = estimate_env_seed_search(
                action=A, metrics_obs=metrics_f, budget=args.est_budget,
                prior=args.est_prior, seed=args.seed + 1337*idx,
                tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir
            )
        if rng_est_env == rng_true_env:
            rng_est_env = (rng_est_env + 104729) % 2_000_000_000 or 1  # tiny nudge to avoid degenerate equality

        # --------- Run our CF & interventional CF ----------
        metrics_cf_ours = _safe_run_ns3_action(Ap, rng_run=rng_est_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)
        metrics_cf_intv = _safe_run_ns3_action(Ap, rng_run=rng_intv_env, tmpdir=os.path.join(OUTPUT_DIR,"tmp"), cache_dir=cache_dir)

        # --------- Text reports ----------
        # Truth & ours share fixed report seed; interventional uses fresh
        for T in temps:
            y_true = report_from_metrics(metrics_cf_true, seed=seed_report_fixed, temperature=T)
            y_ours = report_from_metrics(metrics_cf_ours, seed=seed_report_fixed, temperature=T)
            y_intv = report_from_metrics(metrics_cf_intv, seed=seed_report_fresh, temperature=T)

            d_edit_ours = float(metric_fn(y_ours, y_true))
            d_edit_intv = float(metric_fn(y_intv, y_true))

            seed_row = {
                "idx": idx,
                "rng_true_env": int(rng_true_env),
                "rng_est_env": int(rng_est_env),
                "rng_intv_env": int(rng_intv_env),
                "seed_report_fixed": int(seed_report_fixed),
                "seed_report_fresh": int(seed_report_fresh),
            }
            records.append({**seed_row, "temperature": T, "method": "GM-SCM",         "metric_name": "edit", "distance": d_edit_ours})
            records.append({**seed_row, "temperature": T, "method": "Interventional", "metric_name": "edit", "distance": d_edit_intv})

        # --------- Numeric distances (to TRUTH) ----------
        num_ours = numeric_distance(metrics_cf_ours, metrics_cf_true)
        num_intv = numeric_distance(metrics_cf_intv, metrics_cf_true)

        if num_intv == 0.0:
            print(f"[warn] Interventional numeric=0 (idx={idx}). Seeds true={rng_true_env} intv={rng_intv_env}. "
                  f"Verify ns-3 binary honors --rngRun.", flush=True)

        seed_row = {
            "idx": idx,
            "rng_true_env": int(rng_true_env),
            "rng_est_env": int(rng_est_env),
            "rng_intv_env": int(rng_intv_env),
            "seed_report_fixed": int(seed_report_fixed),
            "seed_report_fresh": int(seed_report_fresh),
        }
        records.append({**seed_row, "temperature": None, "method": "GM-SCM",         "metric_name": "numeric", "distance": num_ours})
        records.append({**seed_row, "temperature": None, "method": "Interventional", "metric_name": "numeric", "distance": num_intv})

        pbar.update(1)
    pbar.close()

    # ---------------- Save CSV ----------------
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.outdir, "cf_compare_distances.csv")
    df.to_csv(csv_path, index=False)

    # ---------------- Plots ----------------
    # 1) Edit distance vs temp (averaged)
    df_edit = df[df["metric_name"]=="edit"].copy()
    agg = df_edit.groupby(["temperature","method"])["distance"].mean().reset_index()

    plt.figure()
    for method, label in [("GM-SCM","GM-SCM"), ("Interventional","Interventional")]:
        sub = agg[agg["method"]==method].sort_values("temperature")
        if not sub.empty:
            plt.plot(sub["temperature"].values, sub["distance"].values, marker="o", label=label)
    plt.xlabel(r"Temperature $\tau$" if args.use_tex else "Temperature τ")
    plt.ylabel("Average edit distance to truth")
    plt.legend()
    plt.tight_layout()
    fig1_png = os.path.join(args.outdir, "edit_vs_temp.png")
    plt.savefig(fig1_png, dpi=160)
    if args.pdf:
        fig1_pdf = os.path.join(args.outdir, "edit_vs_temp.pdf")
        plt.savefig(fig1_pdf)
    plt.close()

    # 2) Numeric distance distribution (boxplot), single per method (temperature-independent)
    df_num = df[(df["metric_name"]=="numeric")].copy()
    methods = ["GM-SCM","Interventional"]
    data_by_m = [df_num[df_num["method"]==m]["distance"].values for m in methods]

    plt.figure()
    plt.boxplot(data_by_m, tick_labels=methods, showfliers=False)
    plt.ylabel("Numeric distance to truth (scaled MSE)")
    plt.tight_layout()
    fig2_png = os.path.join(args.outdir, "numeric_boxplot.png")
    plt.savefig(fig2_png, dpi=160)
    if args.pdf:
        fig2_pdf = os.path.join(args.outdir, "numeric_boxplot.pdf")
        plt.savefig(fig2_pdf)
    plt.close()

    print(f"[✓] Wrote per-case distances to: {csv_path}")
    print(f"[✓] Wrote plot: {fig1_png}")
    print(f"[✓] Wrote plot: {fig2_png}")

if __name__ == "__main__":
    main()
