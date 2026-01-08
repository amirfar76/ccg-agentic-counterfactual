# cfllm/eval_cf_npe.py
# NPE-based posterior over (latent) env seed proxy with CPU/MPS hardening for sbi/torch.

import os, re, json, ast, argparse, hashlib, random, math
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# Force CPU by default: hide CUDA, allow MPS fallback (harmless on non-Mac).
# ------------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .scores import METRICS

# We import sbi AFTER setting CUDA_VISIBLE_DEVICES, and immediately patch its device resolver.
try:
    import sbi.utils.torchutils as _sbi_tu
except Exception:
    _sbi_tu = None
from sbi.inference import SNPE
from sbi.utils import BoxUniform

# ------------------------------------------------------------------------------
# Device helpers & SBI device monkey patch
# ------------------------------------------------------------------------------
def _safe_device_choice(requested: str = "cpu") -> torch.device:
    req = (requested or "cpu").lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "mps":
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    # Never allow CUDA here; environment likely CPU-only.
    return torch.device("cpu")

def _patch_sbi_force_cpu():
    """Monkey-patch sbi's device resolver to never poke CUDA on CPU-only installs."""
    if _sbi_tu is None:
        return
    def _force_cpu_device(_dev=None):
        # Return a real torch.device so BoxUniform(..., device=...) is happy.
        return torch.device("cpu")
    _sbi_tu.process_device = _force_cpu_device

_patch_sbi_force_cpu()

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
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

# ------------------------------------------------------------------------------
# KPI vectorization and numeric distance
# ------------------------------------------------------------------------------
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

def kpi_vector(metrics: Dict[str,Any], action: Dict[str,Any], pad_ues:int=8) -> np.ndarray:
    """Return a normalized vector in [0,1] from metrics/action, per-UE padded to pad_ues."""
    tot, delay_ms, per = _env_signature(metrics)
    num_ues = max(1, int(action.get("num_ues", 3)))
    traffic = max(1e-6, float(action.get("traffic_mbps", 0.5)))

    # Normalizations to [0,1]
    # total throughput normalized by num_ues * traffic (clip to [0,1.5] then /1.5)
    tot_norm = np.clip(tot / (num_ues * traffic + 1e-9), 0.0, 1.5) / 1.5
    # delay: scale with soft cap (1000 ms)
    delay_norm = np.clip(delay_ms / 1000.0, 0.0, 1.0)
    # per-UE: each normalized by traffic, same cap as total then averaged footprint
    per = (per + [0.0]*pad_ues)[:pad_ues]
    per_norm = [np.clip(x / (traffic + 1e-9), 0.0, 1.5) / 1.5 for x in per]

    return np.array([tot_norm, delay_norm] + per_norm, dtype=np.float32)

def numeric_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error in [0,1] since inputs are normalized to [0,1]."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))

# ------------------------------------------------------------------------------
# NPE posterior over a latent seed-proxy u in [0,1]
# ------------------------------------------------------------------------------
def _u_to_seed(u: float) -> int:
    # Map a continuous u in [0,1] to an integer seed in [1, 2e9]
    u = float(np.clip(u, 0.0, 1.0))
    return 1 + int(round(u * (2_000_000_000 - 1)))

def fit_posterior_for_A(
    action: Dict[str,Any],
    z_obs: np.ndarray,
    rounds: int,
    sims_per_round: int,
    density_estimator: str,
    device: torch.device,
) -> Any:
    """
    Build SNPE posterior over u in [0,1] such that seed = f(u). Simulator: ns-3 at action with seed.
    """
    # 1-D latent u \in [0,1]
    theta_low  = torch.tensor([0.0], dtype=torch.float32, device=device)
    theta_high = torch.tensor([1.0], dtype=torch.float32, device=device)
    prior = BoxUniform(low=theta_low, high=theta_high, device=device)

    inference = SNPE(prior=prior, density_estimator=density_estimator, device=device)

    # Observations vector as torch
    x_obs = torch.as_tensor(z_obs, dtype=torch.float32, device=device).unsqueeze(0)

    # Simulations
    for r in range(max(1, rounds)):
        # Sample u ~ Uniform[0,1], simulate
        u_batch = np.random.rand(sims_per_round).astype(np.float32)
        seeds = [_u_to_seed(float(u)) for u in u_batch]
        x_list = []
        for s in tqdm(seeds, desc=f"Simulating (round {r+1}/{rounds or 1})", leave=False):
            m = run_ns3_action(action, rng_run=int(s), workdir=os.path.join(OUTPUT_DIR,"tmp"))
            x_vec = kpi_vector(m, action)  # normalized KPI vector
            x_list.append(x_vec)

        theta_batch = torch.as_tensor(u_batch.reshape(-1,1), dtype=torch.float32, device=device)
        x_batch     = torch.as_tensor(np.stack(x_list, axis=0), dtype=torch.float32, device=device)

        inference.append_simulations(theta_batch, x_batch)

    posterior = inference.train().build_posterior()
    return posterior, x_obs

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NPE-based CF evaluation with CPU/MPS safe device handling.")
    ap.add_argument("--data-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "test.jsonl"))
    ap.add_argument("--temps", type=str, default="0.4,0.7,1.0")
    ap.add_argument("--metric", type=str, default="levenshtein_norm")
    ap.add_argument("--max-cases", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--sims-per-round", type=int, default=1500)
    ap.add_argument("--posterior-samples", type=int, default=64)
    ap.add_argument("--density-estimator", type=str, default="nsf", choices=["nsf","maf","mdn"])
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "cf_npe"))
    ap.add_argument("--device", type=str, default="cpu", help="cpu or mps")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = _safe_device_choice(args.device)
    print(f"[eval_cf_npe] Using device: {device}")
    # (Sanity prints)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    try:
        print("torch.version.cuda:", torch.version.cuda)
    except Exception:
        print("torch.version.cuda: <none>")

    ensure_dir(args.outdir)

    if args.metric not in METRICS:
        raise KeyError(f"Unknown metric '{args.metric}'. Available: {list(METRICS.keys())}")
    metric_fn = METRICS[args.metric]

    rows = read_jsonl(args.data_path)
    rows = rows[:args.max_cases] if args.max_cases else rows
    if not rows:
        raise RuntimeError(f"No rows in {args.data_path}")

    temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]

    plt.rcParams["text.usetex"] = False  # keep robust

    records = []
    pbar = tqdm(total=len(rows), desc="Cases", leave=True)

    for idx, r in enumerate(rows):
        # Build actions (low-temp to keep mapping stable)
        A_raw  = action_from_prompt(r["X"],       seed=args.seed+idx, temperature=0.2)
        Ap_raw = action_from_prompt(r["X_prime"], seed=args.seed+idx, temperature=0.2)
        A  = _normalize_action(A_raw)
        Ap = _normalize_action(Ap_raw)

        # True env seed for this factual/CF pair
        rng_true_env = _small_rng(hash((idx, "env_true", args.seed)))

        # Factual & true CF metrics
        m_f       = run_ns3_action(A,  rng_run=rng_true_env, workdir=os.path.join(OUTPUT_DIR,"tmp"))
        m_cf_true = run_ns3_action(Ap, rng_run=rng_true_env, workdir=os.path.join(OUTPUT_DIR,"tmp"))

        # Observation vector from factual
        z_obs = kpi_vector(m_f, A)

        # Fit posterior p(u | z_obs) via NPE (CPU/MPS)
        posterior, x_obs = fit_posterior_for_A(
            action=A,
            z_obs=z_obs,
            rounds=args.rounds,
            sims_per_round=args.sims_per_round,
            density_estimator=args.density_estimator,
            device=device
        )

        # Our method: sample u ~ posterior, map to seed, run CF under Ap, average KPIs
        num_samp = int(args.posterior_samples)
        with torch.no_grad():
            # Sample on device but we only need the numpy values
            thetas = posterior.sample((num_samp,), x=x_obs, show_progress_bars=False).cpu().numpy().reshape(-1)
        seeds_est = [_u_to_seed(float(np.clip(u,0.0,1.0))) for u in thetas]

        m_cf_est_list = []
        for s_est in seeds_est:
            m_est = run_ns3_action(Ap, rng_run=int(s_est), workdir=os.path.join(OUTPUT_DIR,"tmp"))
            m_cf_est_list.append(m_est)
        # Choose the *median* KPI vector across samples for robustness
        v_stack = np.stack([kpi_vector(m, Ap) for m in m_cf_est_list], axis=0)
        v_med = np.median(v_stack, axis=0)
        # find sample closest to median in L2
        sel = int(np.argmin(np.sum((v_stack - v_med[None,:])**2, axis=1)))
        m_cf_est = m_cf_est_list[sel]

        # Interventional baseline: fresh env seed + fresh report LLM randomness
        rng_naive_env = _small_rng(hash((idx, "env_naive", args.seed)))
        m_cf_naive = run_ns3_action(Ap, rng_run=rng_naive_env, workdir=os.path.join(OUTPUT_DIR,"tmp"))

        # Prepare text reports & distances
        # Keep report LLM randomness fixed for True & Ours; change for Interventional
        for T in temps:
            seed_report_base = args.seed + 777  # fixed across methods except naive uses +999
            y_true  = report_from_metrics(m_cf_true, seed=seed_report_base,       temperature=T)
            y_ours  = report_from_metrics(m_cf_est,  seed=seed_report_base,       temperature=T)
            y_naive = report_from_metrics(m_cf_naive,seed=seed_report_base + 999, temperature=T)

            d_edit_ours  = float(metric_fn(y_ours,  y_true))
            d_edit_naive = float(metric_fn(y_naive, y_true))

            # Numeric distances (independent of T by design of KPI vector)
            v_true  = kpi_vector(m_cf_true,  Ap)
            v_ours  = kpi_vector(m_cf_est,   Ap)
            v_naive = kpi_vector(m_cf_naive, Ap)
            d_num_ours  = numeric_distance(v_ours,  v_true)
            d_num_naive = numeric_distance(v_naive, v_true)

            records.append({
                "idx": idx, "temperature": T, "method": "ours_npe",
                "edit_distance": d_edit_ours, "numeric_distance": d_num_ours
            })
            records.append({
                "idx": idx, "temperature": T, "method": "interventional",
                "edit_distance": d_edit_naive, "numeric_distance": d_num_naive
            })

        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records)
    ensure_dir(args.outdir)
    csv_path = os.path.join(args.outdir, "cf_npe_distances.csv")
    df.to_csv(csv_path, index=False)

    # Plot 1: Edit distance vs Temperature (ours vs interventional)
    agg = df.groupby(["temperature","method"])["edit_distance"].mean().reset_index()
    plt.figure()
    for method, name in [("ours_npe","GM-SCM (NPE env)"), ("interventional","Interventional")]:
        sub = agg[agg["method"]==method].sort_values("temperature")
        plt.plot(sub["temperature"].values, sub["edit_distance"].values, marker="o", label=name)
    plt.xlabel("Temperature")
    plt.ylabel(f"Average edit distance ({args.metric})")
    plt.legend()
    plt.tight_layout()
    fig1 = os.path.join(args.outdir, "edit_distance_vs_temp.png")
    plt.savefig(fig1, dpi=160)
    plt.close()

    # Plot 2: Numeric distance distributions (boxplots) per method (aggregate over temps)
    plt.figure()
    data_by_method = [df[df["method"]==m]["numeric_distance"].values for m in ["ours_npe","interventional"]]
    plt.boxplot(data_by_method, tick_labels=["GM-SCM (NPE env)", "Interventional"], showfliers=False)
    plt.ylabel("Numeric distance (MSE in [0,1])")
    plt.tight_layout()
    fig2 = os.path.join(args.outdir, "numeric_distance_boxplot.png")
    plt.savefig(fig2, dpi=160)
    plt.close()

    print(f"[✓] Wrote distances CSV: {csv_path}")
    print(f"[✓] Wrote edit-distance plot: {fig1}")
    print(f"[✓] Wrote numeric-distance plot: {fig2}")

if __name__ == "__main__":
    main()