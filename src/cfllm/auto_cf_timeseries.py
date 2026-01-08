# cfllm/auto_cf_timeseries.py
# Full automation: factual trace -> NPE inference -> ns-3 runs -> plots
#
# - Auto-detects ns-3 flags (like plot_cf_timeseries.py you used).
# - Uses an NPE model if available (sbi/pyknos); otherwise falls back to a
#   simple nearest-neighbor-on-candidates baseline so the script always runs.
# - Produces sidecar CSVs and two PDFs: throughput and delay vs time.
#
# Requirements:
#   export CFCTWIN_NS3_BIN=".../ns3-dev-ran-sim-optimized"
#
# Example:
#   python -m cfllm.auto_cf_timeseries \
#     --action-json '{"num_ues":5,"scheduler":"rr","traffic_mbps":8,"duration_s":8,"bandwidth_mhz":10}' \
#     --rng-factual 1554486822 \
#     --ts-dt 0.2 \
#     --tmin 1.0 \
#     --outdir outputs/ts_plots_auto \
#     --use_tex \
#     --npe-model outputs/npe/model.pt
#
# Notes:
# - If --npe-model is omitted or cannot be loaded, a robust fallback is used
#   (candidate-scan + nearest neighbor on simple features) so you can test the
#   wiring end-to-end immediately.

import os
import re
import json
import math
import random
import argparse
import subprocess
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ------------------------- plotting helpers -------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _select_window(df: pd.DataFrame, tmin: Optional[float], tmax: Optional[float]) -> pd.DataFrame:
    if tmin is not None:
        df = df[df["time_s"] >= tmin]
    if tmax is not None:
        df = df[df["time_s"] <= tmax]
    return df.reset_index(drop=True)

def _plot_series(df_true: pd.DataFrame, df_cg: pd.DataFrame, df_ig: pd.DataFrame,
                 col: str, ylabel: str, out_pdf: str) -> None:
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(df_true["time_s"], df_true[col], lw=1.2, label="True CF")
    plt.plot(df_cg["time_s"],   df_cg[col],   lw=1.2, label="CG")
    plt.plot(df_ig["time_s"],   df_ig[col],   lw=1.2, label="IG")
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"[✓] Wrote {out_pdf}")


# ------------------------- ns-3 invocation (auto-flag) -------------------------

def _ns3_bin() -> str:
    b = os.environ.get("CFCTWIN_NS3_BIN")
    if not b or not os.path.isfile(b):
        raise RuntimeError(
            "Set CFCTWIN_NS3_BIN to your built binary, e.g.\n"
            '  export CFCTWIN_NS3_BIN="$HOME/Desktop/NS/ns-3-dev/build/scratch/ns3-dev-ran-sim-optimized"'
        )
    return b

def _ns3_help(bin_path: str) -> str:
    p = subprocess.run([bin_path, "--PrintHelp"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout

def _flag_is_supported(help_text: str, flag: str) -> bool:
    pat = re.compile(rf"^\s*--{re.escape(flag)}\b", flags=re.MULTILINE)
    return bool(pat.search(help_text))

def _normalize_action(a_in: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "numUes": int(a_in.get("num_ues", a_in.get("numUes", 5))),
        "scheduler": str(a_in.get("scheduler", "rr")),
        "trafficMbps": float(a_in.get("traffic_mbps", a_in.get("trafficMbps", 8.0))),
        "duration": float(a_in.get("duration_s", a_in.get("duration", 8.0))),
        "bandwidthMHz": float(a_in.get("bandwidth_mhz", a_in.get("bandwidthMHz", 10.0))),
    }
    return out

def _build_ns3_args(bin_path: str, action: Dict[str, Any], rng_run: int, ts_dt: float,
                    metrics_path: str, csv_path: str) -> List[str]:
    help_txt = _ns3_help(bin_path)
    args = [bin_path]

    # Core flags
    if _flag_is_supported(help_txt, "numUes"):
        args.append(f'--numUes={action["numUes"]}')
    if _flag_is_supported(help_txt, "scheduler"):
        args.append(f'--scheduler={action["scheduler"]}')
    if _flag_is_supported(help_txt, "trafficMbps"):
        args.append(f'--trafficMbps={action["trafficMbps"]}')
    if _flag_is_supported(help_txt, "duration"):
        args.append(f'--duration={action["duration"]}')
    if _flag_is_supported(help_txt, "rngRun"):
        args.append(f'--rngRun={rng_run}')
    if _flag_is_supported(help_txt, "output"):
        args.append(f'--output={metrics_path}')

    # Optional bandwidth
    if _flag_is_supported(help_txt, "bandwidthMHz"):
        args.append(f'--bandwidthMHz={action["bandwidthMHz"]}')

    # Timeseries flags (your binary lists these)
    if _flag_is_supported(help_txt, "tsCsv"):
        args.append(f'--tsCsv={csv_path}')
    if _flag_is_supported(help_txt, "tsDt"):
        args.append(f'--tsDt={ts_dt}')

    return args

def _run_ns3_once(action: Dict[str, Any], rng_run: int, ts_dt: float,
                  workdir: str, tag: str) -> Tuple[pd.DataFrame, str, str]:
    bin_path = _ns3_bin()
    metrics_path = os.path.join(workdir, f"metrics_{tag}.json")
    csv_path     = os.path.join(workdir, f"timeseries_{tag}.csv")

    args = _build_ns3_args(bin_path, action, rng_run, ts_dt, metrics_path, csv_path)
    print("[ns3] " + " ".join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ns-3 exited with code {proc.returncode}\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}"
        )

    if not os.path.exists(csv_path):
        raise RuntimeError(f"Timeseries CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"time_s", "thr_total_mbps", "avg_delay_ms"}.issubset(df.columns):
        raise RuntimeError(f"Unexpected CSV columns in {csv_path}: {list(df.columns)}")

    return df, metrics_path, csv_path


# ------------------------- NPE adapter (inference) -------------------------

def _featurize_trace(csv_path: str) -> pd.Series:
    """
    Turn a trace into a simple, deterministic feature vector.
    This is used by both the NPE adapter and the fallback heuristic.
    """
    df = pd.read_csv(csv_path)
    # Robust summary features (change later to your exact stats if desired)
    s = pd.Series({
        "n": len(df),
        "thr_mean": df["thr_total_mbps"].mean(),
        "thr_std": df["thr_total_mbps"].std(ddof=0),
        "thr_p10": df["thr_total_mbps"].quantile(0.10),
        "thr_p50": df["thr_total_mbps"].quantile(0.50),
        "thr_p90": df["thr_total_mbps"].quantile(0.90),
        "dly_mean": df["avg_delay_ms"].mean(),
        "dly_std": df["avg_delay_ms"].std(ddof=0),
        "dly_p50": df["avg_delay_ms"].quantile(0.50),
        "dly_p90": df["avg_delay_ms"].quantile(0.90),
    })
    return s.fillna(0.0)

def _try_load_npe(model_path: Optional[str]):
    """
    Best-effort load of an sbi/pyknos model. If unavailable, returns None.
    We only need a callable that maps (action, features) -> posterior over RNG
    or noise that we can hash into an RNG.
    """
    if not model_path:
        return None
    try:
        import torch  # noqa
        # Example: user stores a TorchScript or pickled object with .infer(dict) -> dict
        # We’ll assume a very simple interface to stay flexible.
        obj = torch.load(model_path, map_location="cpu")
        return obj
    except Exception as e:
        print(f"[!] Could not load NPE model '{model_path}': {e}")
        return None

def _npe_infer_rng(npe_obj, action: Dict[str, Any], factual_features: pd.Series) -> int:
    """
    Minimal adapter: call user model if it follows a simple protocol:
      rng = npe_obj.infer_rng(action_dict, features_dict) -> int
    If it returns a noise vector 'z', we hash it into an ns-3 rngRun.
    """
    # Common patterns, try them in order:
    try:
        if hasattr(npe_obj, "infer_rng"):
            out = npe_obj.infer_rng(action, factual_features.to_dict())
            if isinstance(out, int):
                return max(1, int(out) & 0x7FFFFFFF)
    except Exception:
        pass

    try:
        if hasattr(npe_obj, "infer_noise"):
            z = npe_obj.infer_noise(action, factual_features.to_dict())
            # Hash z -> rng
            import hashlib
            b = repr(z).encode("utf-8")
            h = int.from_bytes(hashlib.sha256(b).digest()[:8], "little")
            return max(1, h & 0x7FFFFFFF)
    except Exception:
        pass

    # TorchScript-style callable?
    try:
        out = npe_obj({"action": action, "features": factual_features.to_dict()})
        if isinstance(out, dict) and "rng" in out:
            return max(1, int(out["rng"]) & 0x7FFFFFFF)
    except Exception:
        pass

    raise RuntimeError("NPE model loaded but no compatible infer interface was found.")

def _fallback_infer_rng(action: Dict[str, Any], factual_csv: str,
                        ts_dt: float, workdir: str,
                        candidate_count: int = 64, seed: int = 12345) -> int:
    """
    No NPE available? Do a quick&dirty candidate scan:
    - Sample a small set of RNG runs
    - Run short sims with the same action and ts_dt
    - Pick the RNG whose features are closest to factual features
    This is just to keep the pipeline fully automated for you;
    replace with your trained NPE when ready.
    """
    print("[i] Using fallback RNG inference (candidate scan).")
    random.seed(seed)
    fact_f = _featurize_trace(factual_csv)

    # Limit duration for speed if the binary allows, otherwise just reuse action['duration']
    # We'll respect your action duration to avoid changing dynamics.
    cands: List[int] = [random.randint(1, 2**31 - 2) for _ in range(candidate_count)]
    best_rng, best_dist = None, float("inf")

    for i, r in enumerate(cands):
        df_c, _, _ = _run_ns3_once(action, r, ts_dt, workdir, f"cand_{i}")
        cand_f = _featurize_trace(df_c.to_csv(os.path.join(workdir, f"cand_{i}.csv"), index=False) or os.path.join(workdir, f"cand_{i}.csv"))
        # simple L2 on a few stable stats
        keys = ["thr_mean", "thr_std", "thr_p50", "thr_p90", "dly_mean", "dly_p90"]
        d = 0.0
        for k in keys:
            d += (float(fact_f[k]) - float(cand_f[k]))**2
        d = math.sqrt(d)
        if d < best_dist:
            best_rng, best_dist = r, d

    print(f"[i] Fallback inferred rng={best_rng} (dist={best_dist:.4g})")
    return int(best_rng) if best_rng is not None else random.randint(1, 2**31 - 2)


# ------------------------- pipeline -------------------------

def main():
    ap = argparse.ArgumentParser(description="Automated CF time-series: factual -> NPE -> CG/IG -> plots")
    ap.add_argument("--action-json", type=str, help="Inline JSON for action")
    ap.add_argument("--action-file", type=str, help="Path to JSON for action")

    ap.add_argument("--rng-factual", type=int, required=True, help="RNG for factual/true CF")
    ap.add_argument("--rng-intv", type=int, default=None, help="RNG for IG (fresh). If omitted, random is chosen.")
    ap.add_argument("--ts-dt", type=float, default=0.2, help="Sampling step for ns-3 --tsDt")

    ap.add_argument("--tmin", type=float, default=None, help="Plot window start (s)")
    ap.add_argument("--tmax", type=float, default=None, help="Plot window end (s)")

    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--use_tex", action="store_true")

    # NPE
    ap.add_argument("--npe-model", type=str, default=None, help="Path to trained NPE model (optional).")

    # Repro for fallback
    ap.add_argument("--fallback-candidates", type=int, default=64, help="Fallback candidate count if no NPE.")
    ap.add_argument("--fallback-seed", type=int, default=12345, help="Fallback random seed.")

    args = ap.parse_args()

    if args.use_tex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    _ensure_dir(args.outdir)
    tmpdir = os.path.join(args.outdir, "tmp_runs")
    _ensure_dir(tmpdir)

    # Load & normalize action
    if args.action_json:
        action_in = json.loads(args.action_json)
    elif args.action_file:
        with open(args.action_file, "r") as f:
            action_in = json.load(f)
    else:
        raise ValueError("Provide --action-json or --action-file.")
    action = _normalize_action(action_in)
    print("[action] " + json.dumps(action, indent=2))

    # 1) Run factual to get the true counterfactual env trace (same env RNG as factual)
    df_factual, m_factual, csv_factual = _run_ns3_once(action, args.rng_factual, args.ts_dt, tmpdir, "factual")

    # 2) Infer rng for CG using NPE (or fallback)
    npe_obj = _try_load_npe(args.npe_model)
    if npe_obj is not None:
        print(f"[i] NPE model loaded: {args.npe_model}")
        features = _featurize_trace(csv_factual)
        try:
            rng_cg = _npe_infer_rng(npe_obj, action, features)
        except Exception as e:
            print(f"[!] NPE inference failed ({e}); falling back to candidate scan.")
            rng_cg = _fallback_infer_rng(action, csv_factual, args.ts_dt, tmpdir,
                                         candidate_count=args.fallback_candidates,
                                         seed=args.fallback_seed)
    else:
        rng_cg = _fallback_infer_rng(action, csv_factual, args.ts_dt, tmpdir,
                                     candidate_count=args.fallback_candidates,
                                     seed=args.fallback_seed)

    # 3) IG rng (fresh)
    rng_ig = args.rng_intv if args.rng_intv is not None else random.randint(1, 2**31 - 2)

    # 4) Run three scenarios for plotting (True CF uses same rng as factual)
    df_true, _, _ = _run_ns3_once(action, args.rng_factual, args.ts_dt, tmpdir, "true")
    df_cg,   _, _ = _run_ns3_once(action, rng_cg,           args.ts_dt, tmpdir, "cg")
    df_ig,   _, _ = _run_ns3_once(action, rng_ig,           args.ts_dt, tmpdir, "ig")

    # 5) Optional cropping to visualize the interesting region
    df_true = _select_window(df_true, args.tmin, args.tmax)
    df_cg   = _select_window(df_cg,   args.tmin, args.tmax)
    df_ig   = _select_window(df_ig,   args.tmin, args.tmax)

    # 6) Save series + seeds we used
    df_true.to_csv(os.path.join(args.outdir, "true_timeseries.csv"), index=False)
    df_cg.to_csv(  os.path.join(args.outdir, "cg_timeseries.csv"),   index=False)
    df_ig.to_csv(  os.path.join(args.outdir, "ig_timeseries.csv"),   index=False)
    with open(os.path.join(args.outdir, "seeds_used.json"), "w") as f:
        json.dump({"rng_true": args.rng_factual, "rng_cg": int(rng_cg), "rng_ig": int(rng_ig)}, f, indent=2)
    print(f"[✓] Seeds: true={args.rng_factual}, cg={rng_cg}, ig={rng_ig}")

    # 7) Plot raw per-interval signals
    _plot_series(df_true, df_cg, df_ig, "thr_total_mbps",
                 "throughput (Mbps) per interval",
                 os.path.join(args.outdir, "ts_throughput.pdf"))
    _plot_series(df_true, df_cg, df_ig, "avg_delay_ms",
                 "avg delay (ms) per interval",
                 os.path.join(args.outdir, "ts_delay.pdf"))

if __name__ == "__main__":
    main()
