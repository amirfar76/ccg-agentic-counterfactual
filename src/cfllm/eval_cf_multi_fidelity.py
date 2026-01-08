# cfllm/eval_cf_multi_fidelity.py
# CF evaluation with multiple environment fidelities.
# Truth: complex ns-3 (your existing run_ns3_action).
# Digital twins (simple/medium): parametric surrogates calibrated on factual, then used at CF.

import os, json, argparse, hashlib, random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .scores import METRICS
from .dt_env import estimate_twin_params, twin_simple, twin_medium, env_metric_distance

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    rows=[]
    with open(path,"r") as f:
        for l in f:
            if l.strip(): rows.append(json.loads(l))
    return rows

def _small_rng(seed: int) -> int:
    rnd = random.Random(int(seed) & 0x7FFFFFFF)
    return rnd.randint(1, 2_000_000_000)

def _normalize_action(a_in: Any) -> Dict[str,Any]:
    # same as your eval_cf version, but minimal:
    if isinstance(a_in, dict):
        a = dict(a_in)
    else:
        # fall back to prompting helper
        a = {}
    num_ues   = int(a.get("num_ues", 3))
    traffic   = float(a.get("traffic_mbps", 0.5))
    duration  = float(a.get("duration_s", 1.0))
    sched     = str(a.get("scheduler", "rr")).lower()
    if sched.startswith("proportional"): sched="pf"
    if sched not in ("rr","pf","mt"): sched="rr"
    return {
        "num_ues": max(1, min(64, num_ues)),
        "traffic_mbps": max(0.05, min(50.0, traffic)),
        "duration_s": max(0.1, min(30.0, duration)),
        "scheduler": sched,
    }

def _safe_run_ns3_action(action: Dict[str,Any], rng_run: int, tmpdir: str, cache_dir: str) -> Dict[str,Any]:
    ensure_dir(tmpdir)
    ensure_dir(cache_dir)
    key = hashlib.sha256((json.dumps(action, sort_keys=True)+f"|{rng_run}").encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"ns3_{key}.json")
    if os.path.exists(path):
        return json.load(open(path))
    m = run_ns3_action(action, rng_run=rng_run, workdir=tmpdir)
    json.dump(m, open(path,"w"), indent=2)
    return m

def main():
    ap = argparse.ArgumentParser(description="CF eval with multiple env fidelities (truth=ns-3 complex).")
    ap.add_argument("--data-path", type=str, default=os.path.join(OUTPUT_DIR,"data","test.jsonl"))
    ap.add_argument("--temps", type=str, default="0.4,0.7,1.0")
    ap.add_argument("--metric", type=str, default="levenshtein_norm")
    ap.add_argument("--fidelities", type=str, default="simple,medium")
    ap.add_argument("--max-cases", type=int, default=5)
    ap.add_argument("--budget", type=int, default=48, help="Twin param search budget per case")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR,"cf_multi_fidelity"))
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    cache_dir = os.path.join(args.outdir, "ns3_cache") if args.cache else os.path.join(args.outdir, "ns3_cache_tmp")
    ensure_dir(cache_dir)

    if args.metric not in METRICS:
        raise KeyError(f"Unknown metric '{args.metric}'. Available: {list(METRICS.keys())}")
    text_metric = METRICS[args.metric]

    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    rows = read_jsonl(args.data_path)[:args.max_cases]
    if not rows: raise RuntimeError("No rows to evaluate.")

    plt.rcParams["text.usetex"] = False  # keep robust

    records = []
    pbar = tqdm(total=len(rows), desc="Cases")

    for idx, r in enumerate(rows):
        X  = r["X"];  Xp = r["X_prime"]
        # Stable action mapping (low temp)
        A  = _normalize_action(action_from_prompt(X,  seed=args.seed, temperature=0.2))
        Ap = _normalize_action(action_from_prompt(Xp, seed=args.seed, temperature=0.2))

        # Truth env RNG for this episode
        rng_true = _small_rng(hash((idx, "env", args.seed)))

        # Complex (real world) factual & counterfactual KPIs
        metrics_f_complex  = _safe_run_ns3_action(A,  rng_true, os.path.join(OUTPUT_DIR,"tmp"), cache_dir)
        metrics_cf_complex = _safe_run_ns3_action(Ap, rng_true, os.path.join(OUTPUT_DIR,"tmp"), cache_dir)

        # Generate *true* CF report at each temp
        ytrue_by_T = {}
        for T in temps:
            ytrue_by_T[T] = report_from_metrics(metrics_cf_complex, seed=args.seed, temperature=T)

        # Digital twins to evaluate
        fidelities = [f.strip() for f in args.fidelities.split(",") if f.strip()]
        for fid in fidelities:
            # Fit twin params on factual
            params_hat, fit_score = estimate_twin_params(
                fidelity=fid,
                action_factual=A,
                metrics_factual_complex=metrics_f_complex,
                budget=args.budget,
                seed=args.seed + idx*7919
            )
            # Predict CF KPIs under X'
            if fid == "simple":
                metrics_cf_twin = twin_simple(Ap, params_hat, seed=args.seed + 17)
            elif fid == "medium":
                metrics_cf_twin = twin_medium(Ap, params_hat, seed=args.seed + 17)
            else:
                raise ValueError(f"Unknown fidelity '{fid}'")

            # Also provide a naive baseline twin with default params (no fitting)
            default_params = {"pl_exp": 3.0, "shadow_sigma_db": 6.0, "nakagami_m": 1.0}
            if fid == "medium":
                default_params.update({"ue_speed_mps": 1.0, "doppler_mult": 1.0})
            if fid == "simple":
                metrics_cf_twin_naive = twin_simple(Ap, default_params, seed=args.seed + 99)
            else:
                metrics_cf_twin_naive = twin_medium(Ap, default_params, seed=args.seed + 99)

            for T in temps:
                y_true  = ytrue_by_T[T]
                y_ours  = report_from_metrics(metrics_cf_twin,       seed=args.seed, temperature=T)
                y_naive = report_from_metrics(metrics_cf_twin_naive, seed=args.seed, temperature=T)

                d_ours  = float(text_metric(y_ours,  y_true))
                d_naive = float(text_metric(y_naive, y_true))

                records.append({
                    "idx": idx, "fidelity": fid, "temperature": T,
                    "method": "GM-SCM (twin est.)", "distance": d_ours,
                    "fit_score_kpi": float(fit_score),
                })
                records.append({
                    "idx": idx, "fidelity": fid, "temperature": T,
                    "method": "Interventional (twin default)", "distance": d_naive,
                    "fit_score_kpi": float(fit_score),
                })

        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.outdir, "multi_fidelity_distances.csv")
    df.to_csv(csv_path, index=False)

    # Plot: for each fidelity, average distance vs temperature
    for fid in df["fidelity"].unique():
        sub = df[df["fidelity"]==fid]
        agg = sub.groupby(["temperature","method"])["distance"].mean().reset_index()
        plt.figure()
        for method in agg["method"].unique():
            aa = agg[agg["method"]==method].sort_values("temperature")
            plt.plot(aa["temperature"].values, aa["distance"].values, marker="o", label=method)
        plt.xlabel("Temperature")
        plt.ylabel(f"Average distance to truth ({args.metric})")
        plt.title(f"Digital twin fidelity: {fid}")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(args.outdir, f"dist_vs_temp_{fid}.png")
        plt.savefig(fig_path, dpi=160)
        plt.close()

    print(f"[✓] Wrote per-case distances: {csv_path}")
    print(f"[✓] Wrote figures under:     {args.outdir}/dist_vs_temp_*.png")

if __name__ == "__main__":
    main()

