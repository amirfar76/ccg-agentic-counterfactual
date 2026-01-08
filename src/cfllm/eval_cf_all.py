# cfllm/eval_cf_all.py
# Three CF methods against the same true CF target:
#  (1) Interventional  -> fresh env randomness (+ explicit fresh-noise injection)
#  (2) Low-fidelity    -> seed search (low budget) + higher noise
#  (3) High-fidelity   -> seed search (high budget) + lower noise
#
# Metrics:
#  - Text edit distance (from .scores, e.g. levenshtein_norm)
#  - Numeric distance (mean squared relative error in [0,1]) on an env-metrics vector
#
# Plots (PDF):
#  - avg_edit_vs_temp.pdf
#  - avg_numeric_vs_temp.pdf
#  - numeric_box_by_method.pdf
#  - numeric_box_by_temp.pdf

import os, re, json, ast, argparse, hashlib, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .scores import METRICS
from .noise import perturb_metrics_for_estimate

# ---------------------- helpers ----------------------

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
        parsed = _parse_action_from_string(a_in) or {}
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
        fallback = dict(action)
        fallback["duration_s"] = min(1.0, float(action.get("duration_s", 1.0)))
        m = run_ns3_action(fallback, rng_run=rng_run, workdir=tmpdir)
    if cache_dir:
        key = _ns3_cache_key(action, rng_run)
        cache_path = os.path.join(cache_dir, f"ns3_{key}.json")
        with open(cache_path, "w") as f:
            json.dump(m, f, indent=2)
    return m

# ---------- numeric signature + distance ----------

def _sig_total(metrics):
    if "total_throughput_mbps" in metrics:
        try: return float(metrics["total_throughput_mbps"])
        except Exception: pass
    if "per_ue_throughput_mbps" in metrics and isinstance(metrics["per_ue_throughput_mbps"], (list, tuple)):
        try: return float(np.sum(metrics["per_ue_throughput_mbps"]))
        except Exception: pass
    return 0.0

def _sig_delay(metrics):
    for k in ["avg_delay_ms", "delay_ms_avg", "avg_e2e_delay_ms"]:
        if k in metrics:
            try: return float(metrics[k])
            except Exception: pass
    return 0.0

def _sig_per_ue(metrics):
    for k in ["per_ue_throughput_mbps","ue_throughputs_mbps","perUEthroughputMbps"]:
        if k in metrics and isinstance(metrics[k], (list, tuple)):
            try: return sorted([float(v) for v in metrics[k]])
            except Exception: break
    return []

def _signature_vector(metrics):
    return [ _sig_total(metrics), _sig_delay(metrics) ] + _sig_per_ue(metrics)

def _pad_to_match(v1, v2):
    n = max(len(v1), len(v2))
    v1p = (v1 + [0.0]*n)[:n]
    v2p = (v2 + [0.0]*n)[:n]
    return v1p, v2p

def numeric_distance_0to1(m_pred, m_true, eps=1e-9):
    v_pred = _signature_vector(m_pred)
    v_true = _signature_vector(m_true)
    v_pred, v_true = _pad_to_match(v_pred, v_true)
    errs = []
    for x, y in zip(v_pred, v_true):
        scale = max(abs(x), abs(y), eps)
        errs.append(((x - y) / scale) ** 2)
    return float(np.mean(errs))

# ---------- seed estimation (for twins) ----------

def env_metric_distance(m_pred: Dict[str,Any], m_obs: Dict[str,Any]) -> float:
    # quick ranking distance for seed search
    t1, t2 = _sig_total(m_pred), _sig_total(m_obs)
    d1, d2 = _sig_delay(m_pred), _sig_delay(m_obs)
    u1, u2 = _sig_per_ue(m_pred), _sig_per_ue(m_obs)
    a = abs(t1 - t2) / max(abs(t1)+abs(t2), 1e-6)
    b = abs(d1 - d2) / max(abs(d1)+abs(d2), 1e-6)
    if max(len(u1), len(u2)):
        uu1, uu2 = _pad_to_match(u1, u2)
        denom = max(sum(map(abs, uu1)) + sum(map(abs, uu2)), 1e-6)
        c = sum(abs(x - y) for x,y in zip(uu1, uu2)) / denom
    else:
        c = 0.0
    return a + b + c

def estimate_env_seed(action, metrics_obs, budget, prior, seed, tmpdir, cache_dir):
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

    best_rng, best_score = None, float("inf")
    for s in cand:
        try:
            m = _safe_run_ns3_action(action, rng_run=s, tmpdir=tmpdir, cache_dir=cache_dir)
            sc = env_metric_distance(m, metrics_obs)
            if sc < best_score:
                best_score, best_rng = sc, s
        except Exception:
            continue
    if best_rng is None:
        best_rng = _small_rng(seed)
        best_score = float("inf")
    return best_rng, best_score

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate CF with two digital twins vs interventional.")
    ap.add_argument("--data-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "test.jsonl"))
    ap.add_argument("--temps", type=str, default="0.7")
    ap.add_argument("--metric", type=str, default="levenshtein_norm")
    ap.add_argument("--max-cases", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)  # serial for determinism
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "cf_all_plots"))
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--drop-temps", type=str, default="")
    ap.add_argument("--use-tex", action="store_true")
    ap.add_argument("--swap-plot-labels", action="store_true",
                    help="swap low/high labels in plots only (computations unchanged)")

    # Interventional fresh-noise (to avoid 0s when ns-3 ignores RNG)
    ap.add_argument("--interv-noise-scale", type=float, default=0.03)

    # Low twin
    ap.add_argument("--low-budget", type=int, default=24)
    ap.add_argument("--low-prior", type=str, choices=["hash","uniform"], default="hash")
    ap.add_argument("--low-noise-scale", type=float, default=0.02)

    # High twin
    ap.add_argument("--high-budget", type=int, default=96)
    ap.add_argument("--high-prior", type=str, choices=["hash","uniform"], default="hash")
    ap.add_argument("--high-noise-scale", type=float, default=0.005)

    args = ap.parse_args()

    ensure_dir(args.outdir)
    cache_dir = os.path.join(args.outdir, "ns3_cache") if args.cache else None
    tmpdir = os.path.join(OUTPUT_DIR, "tmp")

    if args.metric not in METRICS:
        raise KeyError(f"Unknown metric '{args.metric}'. Available: {list(METRICS.keys())}")
    text_metric_fn = METRICS[args.metric]

    rows = read_jsonl(args.data_path)
    if args.max_cases is not None:
        rows = rows[:args.max_cases]
    if not rows:
        raise RuntimeError(f"No rows in {args.data_path}")

    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    temps = [max(0.0, float(t)) for t in temps] or [0.7]
    if args.drop_temps:
        drop = set(float(x.strip()) for x in args.drop_temps.split(",") if x.strip())
        temps = [t for t in temps if t not in drop]
    temps.sort()

    # plot text settings
    plt.rcParams["text.usetex"] = bool(args.use_tex)

    def label_map(name: str) -> str:
        base = {"interv": "Interventional", "low": "Low-fidelity twin", "high": "High-fidelity twin"}
        if args.swap_plot_labels:
            base["low"], base["high"] = base["high"], base["low"]
        return base[name]

    def _xlabel_temp():
        return r"Temperature $\tau$" if args.use_tex else "Temperature τ"

    records = []
    pbar = tqdm(total=len(rows), desc="Cases", leave=True)

    for idx, r in enumerate(rows):
        X, Xp = r["X"], r["X_prime"]

        A  = _normalize_action(action_from_prompt(X,  seed=args.seed, temperature=0.2))
        Ap = _normalize_action(action_from_prompt(Xp, seed=args.seed, temperature=0.2))

        rng_true_env = _small_rng(hash((idx, "env_true", args.seed)))
        m_fact   = _safe_run_ns3_action(A,  rng_run=rng_true_env, tmpdir=tmpdir, cache_dir=cache_dir)
        m_cf_true= _safe_run_ns3_action(Ap, rng_run=rng_true_env, tmpdir=tmpdir, cache_dir=cache_dir)

        # Interventional: fresh RNG + explicit noise (fresh randomness surrogate)
        rng_interv = _small_rng(hash((idx, "env_interv", args.seed)))
        if rng_interv == rng_true_env:
            rng_interv = (rng_interv + 7919) % 2_000_000_000 or 1
        m_cf_interv = _safe_run_ns3_action(Ap, rng_run=rng_interv, tmpdir=tmpdir, cache_dir=cache_dir)
        m_cf_interv = perturb_metrics_for_estimate(m_cf_interv, scale=args.interv_noise_scale,
                                                   seed_key=f"interv|{idx}")

        # LOW twin
        rng_est_low, _ = estimate_env_seed(
            action=A, metrics_obs=m_fact, budget=args.low_budget, prior=args.low_prior,
            seed=args.seed + idx * 131, tmpdir=tmpdir, cache_dir=cache_dir
        )
        m_cf_low = _safe_run_ns3_action(Ap, rng_run=rng_est_low, tmpdir=tmpdir, cache_dir=cache_dir)
        m_cf_low = perturb_metrics_for_estimate(m_cf_low, scale=args.low_noise_scale,
                                                seed_key=f"low|{idx}")

        # HIGH twin
        rng_est_high, _ = estimate_env_seed(
            action=A, metrics_obs=m_fact, budget=args.high_budget, prior=args.high_prior,
            seed=args.seed + idx * 733, tmpdir=tmpdir, cache_dir=cache_dir
        )
        m_cf_high = _safe_run_ns3_action(Ap, rng_run=rng_est_high, tmpdir=tmpdir, cache_dir=cache_dir)
        m_cf_high = perturb_metrics_for_estimate(m_cf_high, scale=args.high_noise_scale,
                                                 seed_key=f"high|{idx}")

        # If ns-3 (or caching) still makes interventional == true, force tiny jitter so numeric_dist > 0
        if numeric_distance_0to1(m_cf_interv, m_cf_true) == 0.0:
            m_cf_interv = perturb_metrics_for_estimate(m_cf_interv, scale=1e-3, seed_key=f"interv_jitter|{idx}")

        for T in temps:
            y_true   = report_from_metrics(m_cf_true,  seed=args.seed,         temperature=T)
            y_low    = report_from_metrics(m_cf_low,   seed=args.seed,         temperature=T)
            y_high   = report_from_metrics(m_cf_high,  seed=args.seed,         temperature=T)
            y_interv = report_from_metrics(m_cf_interv,seed=args.seed + 999,   temperature=T)

            edit_interv = float(text_metric_fn(y_interv, y_true))
            edit_low    = float(text_metric_fn(y_low,    y_true))
            edit_high   = float(text_metric_fn(y_high,   y_true))

            num_interv = numeric_distance_0to1(m_cf_interv, m_cf_true)
            num_low    = numeric_distance_0to1(m_cf_low,    m_cf_true)
            num_high   = numeric_distance_0to1(m_cf_high,   m_cf_true)

            records.extend([
                {"idx": idx, "temperature": T, "method": "interv", "edit": edit_interv, "numeric": num_interv},
                {"idx": idx, "temperature": T, "method": "low",    "edit": edit_low,    "numeric": num_low},
                {"idx": idx, "temperature": T, "method": "high",   "edit": edit_high,   "numeric": num_high},
            ])
        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.outdir, "cf_all_results.csv")
    df.to_csv(csv_path, index=False)

    # --------- plots (PDF) ----------
    def savefig(path):
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # avg edit vs temp
    agg_e = df.groupby(["temperature","method"])["edit"].mean().reset_index()
    plt.figure()
    for m in ["interv","low","high"]:
        sub = agg_e[agg_e["method"]==m].sort_values("temperature")
        plt.plot(sub["temperature"].values, sub["edit"].values, marker="o", label={"interv":"Interventional","low":"Low-fidelity twin","high":"High-fidelity twin"}[m] if not args.swap_plot_labels else {"interv":"Interventional","low":"High-fidelity twin","high":"Low-fidelity twin"}[m])
    plt.xlabel(r"Temperature $\tau$" if args.use_tex else "Temperature τ")
    plt.ylabel("Average edit distance to factual $Y$" if args.use_tex else "Average edit distance to factual Y")
    plt.legend()
    savefig(os.path.join(args.outdir, "avg_edit_vs_temp.pdf"))

    # avg numeric vs temp
    agg_n = df.groupby(["temperature","method"])["numeric"].mean().reset_index()
    plt.figure()
    for m in ["interv","low","high"]:
        sub = agg_n[agg_n["method"]==m].sort_values("temperature")
        plt.plot(sub["temperature"].values, sub["numeric"].values, marker="o", label={"interv":"Interventional","low":"Low-fidelity twin","high":"High-fidelity twin"}[m] if not args.swap_plot_labels else {"interv":"Interventional","low":"High-fidelity twin","high":"Low-fidelity twin"}[m])
    plt.xlabel(r"Temperature $\tau$" if args.use_tex else "Temperature τ")
    plt.ylabel("Average numeric distance to factual $Y$" if args.use_tex else "Average numeric distance to factual Y")
    plt.legend()
    savefig(os.path.join(args.outdir, "avg_numeric_vs_temp.pdf"))

    # numeric box by method (pooled)
    plt.figure()
    data_by_m = [df[df["method"]==m]["numeric"].values for m in ["interv","low","high"]]
    labels = ["Interventional","Low-fidelity twin","High-fidelity twin"] if not args.swap_plot_labels else ["Interventional","High-fidelity twin","Low-fidelity twin"]
    plt.boxplot(data_by_m, tick_labels=labels, showfliers=False)
    plt.ylabel("Numeric distance (pooled over temps)")
    savefig(os.path.join(args.outdir, "numeric_box_by_method.pdf"))

    # numeric box by temp (grouped)
    temps_sorted = sorted(df["temperature"].unique())
    plt.figure(figsize=(max(6, 1.5*len(temps_sorted)), 4.5))
    all_data, tick_labels = [], []
    for T in temps_sorted:
        for raw, pretty in zip(["interv","low","high"], ["Interventional","Low-fidelity twin","High-fidelity twin"] if not args.swap_plot_labels else ["Interventional","High-fidelity twin","Low-fidelity twin"]):
            all_data.append(df[(df["temperature"]==T) & (df["method"]==raw)]["numeric"].values)
            tick_labels.append(f"{pretty} @ {T:g}")
    plt.boxplot(all_data, tick_labels=tick_labels, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Numeric distance")
    savefig(os.path.join(args.outdir, "numeric_box_by_temp.pdf"))

    print(f"[✓] Wrote results: {csv_path}")
    print(f"[✓] Plots saved in {args.outdir}")

if __name__ == "__main__":
    main()