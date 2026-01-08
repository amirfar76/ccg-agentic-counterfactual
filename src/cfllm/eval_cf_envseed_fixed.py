# cfllm/eval_cf_envseed_fixed.py
# Evaluate CF generation with:
# - GM-SCM: fixed LLM randomness, estimated env RNG (via budgeted search)
# - Interventional: fresh LLM randomness + fresh env RNG
#
# Numeric distance is computed on raw ns-3 KPIs using elementwise squared relative error
# with safe normalization. This avoids degenerate 0 or 1 walls and reflects true KPI mismatch.
#
# Requirements:
# - env var CFCTWIN_NS3_BIN must point to your ns-3 binary
#   e.g., export CFCTWIN_NS3_BIN="$HOME/Desktop/NS/ns-3-dev/build/scratch/ns3-dev-ran-sim-optimized"

import os, re, json, ast, math, argparse, hashlib, random, subprocess, tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---- If your project already defines these, keep imports as-is ----
from .llm import action_from_prompt, report_from_metrics
from .scores import METRICS

# ------------------------------------------------------------------------------------
# ns-3 BRIDGE (direct binary call; independent of old env_bridge)
# ------------------------------------------------------------------------------------

def _ns3_bin() -> str:
    path = os.environ.get("CFCTWIN_NS3_BIN", "").strip()
    if not path:
        raise RuntimeError(
            "CFCTWIN_NS3_BIN is not set. Export it to your ran-sim binary, e.g.\n"
            '  export CFCTWIN_NS3_BIN="$HOME/Desktop/NS/ns-3-dev/build/scratch/ns3-dev-ran-sim-optimized"\n'
        )
    if not os.path.exists(path):
        raise RuntimeError(f"CFCTWIN_NS3_BIN points to a non-existent file: {path}")
    return path

def run_ns3_action(
    action: Dict[str, Any],
    rng_run: int,
    workdir: str,
    bandwidth_mhz: float = 10.0
) -> Dict[str, Any]:
    """Run the ns-3 LTE sim and return metrics JSON."""
    os.makedirs(workdir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=workdir, prefix="ns3_metrics_", suffix=".json")
    tmp.close()
    out_path = tmp.name

    num_ues   = int(action.get("num_ues", 3))
    scheduler = str(action.get("scheduler", "rr"))
    traffic   = float(action.get("traffic_mbps", 0.5))
    duration  = float(action.get("duration_s", 1.0))

    cmd = [
        _ns3_bin(),
        f"--numUes={num_ues}",
        f"--scheduler={scheduler}",
        f"--trafficMbps={traffic}",
        f"--duration={duration}",
        f"--rngRun={int(rng_run)}",
        f"--bandwidthMHz={float(bandwidth_mhz)}",
        f"--output={out_path}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"NS-3 run failed (code {e.returncode}).\nCMD: {' '.join(cmd)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e

    with open(out_path, "r") as f:
        m = json.load(f)
    return m

# ------------------------------------------------------------------------------------
# I/O + Action parsing (copied/kept robust)
# ------------------------------------------------------------------------------------

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
        parsed = _parse_action_from_string(a_in)
        a = parsed if parsed is not None else {}
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
    traffic  = min(max(0.05, traffic), 50.0)
    return {
        "num_ues": int(num_ues),
        "scheduler": scheduler,
        "traffic_mbps": float(traffic),
        "duration_s": float(duration),
    }

# ------------------------------------------------------------------------------------
# Numeric distance built on raw ns-3 metrics (robust; no degenerate 0/1 walls)
# ------------------------------------------------------------------------------------

def _numeric_vector(m: Dict[str, Any], pad_ues: int = 16) -> np.ndarray:
    """Return a fixed-length numeric vector for KPI comparison."""
    # names come from your ran-sim.cc JSON
    tot = float(m.get("throughput_total_mbps", m.get("total_throughput_mbps", 0.0)))
    dly = float(m.get("avg_delay_ms", 0.0))
    ue  = m.get("ue_throughput_mbps") or m.get("per_ue_throughput_mbps") or []
    ue  = [float(x) for x in ue]
    ue  = sorted(ue)
    if len(ue) < pad_ues:
        ue = ue + [0.0] * (pad_ues - len(ue))
    else:
        ue = ue[:pad_ues]
    return np.array([tot, dly] + ue, dtype=np.float64)

def numeric_distance(m_pred: Dict[str, Any], m_true: Dict[str, Any]) -> float:
    """Mean of squared *relative* error with safe denominator."""
    x = _numeric_vector(m_pred)
    y = _numeric_vector(m_true)
    denom = (np.abs(x) + np.abs(y) + 1e-6)**2
    se = (x - y)**2 / denom
    return float(np.mean(se))

# ------------------------------------------------------------------------------------
# Env RNG estimation (budgeted search over candidates)
# ------------------------------------------------------------------------------------

def _env_metric_distance(m_pred: Dict[str,Any], m_obs: Dict[str,Any]) -> float:
    # Use the same numeric distance for seed match scoring (works well and stable)
    return numeric_distance(m_pred, m_obs)

def estimate_env_seed(
    action: Dict[str,Any],
    metrics_obs: Dict[str,Any],
    budget: int,
    prior: str,
    seed: int,
    tmpdir: str
) -> Tuple[int, float]:
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
            m = run_ns3_action(action, rng_run=s, workdir=tmpdir)
            sc = _env_metric_distance(m, metrics_obs)
            if sc < best_score:
                best_score = sc
                best_rng = s
        except Exception:
            continue
    if best_rng is None:
        best_rng = _small_rng(seed)
        best_score = float("inf")
    return best_rng, best_score

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate CF with fixed-LLM + estimated env RNG vs Interventional.")
    ap.add_argument("--data-path", type=str, required=True)
    ap.add_argument("--temps", type=str, default="0.4,0.7,1.0")
    ap.add_argument("--metric", type=str, default="levenshtein_norm")
    ap.add_argument("--max-cases", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)  # unused; for CLI compatibility
    ap.add_argument("--outdir", type=str, default="outputs/cf_envseed_fixed")
    ap.add_argument("--seed", type=int, default=123)

    # estimator
    ap.add_argument("--env-est-budget", type=int, default=48)
    ap.add_argument("--env-est-prior", type=str, choices=["hash","uniform"], default="hash")

    # plotting
    ap.add_argument("--pdf", action="store_true", help="Save plots as PDF in addition to PNG.")
    ap.add_argument("--use-tex", action="store_true", help="Use LaTeX fonts if available.")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # edit-distance scorer
    if args.metric not in METRICS:
        raise KeyError(f"Unknown metric '{args.metric}'. Available: {list(METRICS.keys())}")
    metric_fn = METRICS[args.metric]

    # Matplotlib config
    if args.use_tex:
        try:
            plt.rcParams["text.usetex"] = True
            plt.rcParams["font.family"] = "serif"
        except Exception:
            plt.rcParams["text.usetex"] = False
    else:
        plt.rcParams["text.usetex"] = False

    rows = read_jsonl(args.data_path)
    if args.max_cases is not None:
        rows = rows[:args.max_cases]
    if not rows:
        raise RuntimeError(f"No rows in {args.data_path}")

    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
    temps = [max(0.0, float(t)) for t in temps] or [0.7]

    records = []
    pbar = tqdm(total=len(rows), desc="Cases", leave=True)

    for idx, r in enumerate(rows):
        X  = r["X"]
        Xp = r["X_prime"]

        # Actions: keep low temp for stability; same seed here for both methods (action mapping)
        A  = _normalize_action(action_from_prompt(X,  seed=args.seed + idx, temperature=0.2))
        Ap = _normalize_action(action_from_prompt(Xp, seed=args.seed + idx, temperature=0.2))

        # True env RNG (unknown to estimator)
        rng_true_env = _small_rng(hash((idx, "env", args.seed)))

        # Factual metrics (observed)
        metrics_f = run_ns3_action(A, rng_run=rng_true_env, workdir=os.path.join(args.outdir, "tmp"))

        # Truth CF metrics (same env RNG)
        metrics_cf_true = run_ns3_action(Ap, rng_run=rng_true_env, workdir=os.path.join(args.outdir, "tmp"))

        # ---- OUR (GM-SCM): estimate env RNG from (A, metrics_f); keep LLM randomness fixed ----
        rng_est, score_est = estimate_env_seed(
            action=A,
            metrics_obs=metrics_f,
            budget=args.env_est_budget,
            prior=args.env_est_prior,
            seed=args.seed + 1337*idx,
            tmpdir=os.path.join(args.outdir, "tmp")
        )
        metrics_cf_ours = run_ns3_action(Ap, rng_run=rng_est, workdir=os.path.join(args.outdir, "tmp"))

        # ---- INTERVENTIONAL: fresh env RNG + fresh LLM randomness ----
        rng_intv_env = _small_rng(hash((idx, "env_intv", args.seed)))
        metrics_cf_intv = run_ns3_action(Ap, rng_run=rng_intv_env, workdir=os.path.join(args.outdir, "tmp"))

        # ----- Numeric distances (do not depend on τ); store once per case/method -----
        num_ours = numeric_distance(metrics_cf_ours, metrics_cf_true)
        num_intv = numeric_distance(metrics_cf_intv, metrics_cf_true)
        records.append({
            "idx": idx, "temperature": None, "method": "GM-SCM", "metric_name": "numeric", "distance": num_ours
        })
        records.append({
            "idx": idx, "temperature": None, "method": "Interventional", "metric_name": "numeric", "distance": num_intv
        })

        # ----- Text (edit) distances (depend on τ) -----
        for T in temps:
            # True & ours share the *same report LLM seed* (fixed randomness)
            y_true = report_from_metrics(metrics_cf_true, seed=args.seed, temperature=T)
            y_ours = report_from_metrics(metrics_cf_ours, seed=args.seed, temperature=T)

            # Interventional uses a *fresh* report seed
            y_intv = report_from_metrics(metrics_cf_intv, seed=args.seed + 9999, temperature=T)

            d_ours = float(metric_fn(y_ours, y_true))
            d_intv = float(metric_fn(y_intv, y_true))

            records.append({
                "idx": idx, "temperature": T, "method": "GM-SCM", "metric_name": args.metric, "distance": d_ours
            })
            records.append({
                "idx": idx, "temperature": T, "method": "Interventional", "metric_name": args.metric, "distance": d_intv
            })

        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.outdir, "cf_envseed_fixed_distances.csv")
    df.to_csv(csv_path, index=False)

    # ---------------- PLOTS ----------------

    # 1) Average edit distance vs temperature
    df_edit = df[df["metric_name"] == args.metric].copy()
    agg = df_edit.groupby(["temperature","method"])["distance"].mean().reset_index()
    plt.figure()
    for method in ["GM-SCM","Interventional"]:
        sub = agg[agg["method"]==method].sort_values("temperature")
        plt.plot(sub["temperature"].values, sub["distance"].values, marker="o", label=method)
    plt.xlabel(r"Temperature $\tau$" if args.use_tex else "Temperature (tau)")
    plt.ylabel(f"Average distance to true report ({args.metric})")
    plt.legend()
    plt.tight_layout()
    fig1_png = os.path.join(args.outdir, "edit_distance_vs_temp.png")
    plt.savefig(fig1_png, dpi=160)
    if args.pdf:
        fig1_pdf = os.path.join(args.outdir, "edit_distance_vs_temp.pdf")
        plt.savefig(fig1_pdf)
    plt.close()

    # 2) Numeric distance distribution (boxplot) per method
    df_num = df[df["metric_name"] == "numeric"].copy()
    data_gm = df_num[df_num["method"]=="GM-SCM"]["distance"].values
    data_iv = df_num[df_num["method"]=="Interventional"]["distance"].values
    plt.figure()
    plt.boxplot([data_gm, data_iv], labels=["GM-SCM","Interventional"], showfliers=False)
    plt.ylabel("Numeric distance (mean squared relative error)")
    plt.tight_layout()
    fig2_png = os.path.join(args.outdir, "numeric_distance_boxplot.png")
    plt.savefig(fig2_png, dpi=160)
    if args.pdf:
        fig2_pdf = os.path.join(args.outdir, "numeric_distance_boxplot.pdf")
        plt.savefig(fig2_pdf)
    plt.close()

    print(f"[✓] Wrote distances CSV to:   {csv_path}")
    print(f"[✓] Wrote edit plot to:       {fig1_png}")
    print(f"[✓] Wrote numeric boxplot to: {fig2_png}")

if __name__ == "__main__":
    main()
