# cfllm/plot_cf_timeseries.py
# Runs ns-3 for True/CG/IG, writes CSVs, plots, and can infer CG seed via NPE
# with a robust feature adapter (pads/truncates to the model's expected input dim).

import os
import json
import argparse
import tempfile
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from cfllm.env_bridge import run_ns3_action  # must accept timeseries_csv, ts_dt

# ----------------- small utils -----------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _normalize_action(a_raw: Dict[str, Any]) -> Dict[str, Any]:
    a = dict(a_raw or {})
    mapping = {
        "num_ues": "numUes",
        "scheduler": "scheduler",
        "traffic_mbps": "trafficMbps",
        "duration_s": "duration",
        "bandwidth_mhz": "bandwidthMHz",
    }
    out: Dict[str, Any] = {}
    for k_in, k_out in mapping.items():
        if k_in in a:
            out[k_out] = a[k_in]
    for k in ["numUes", "scheduler", "trafficMbps", "duration", "bandwidthMHz"]:
        if k in a:
            out[k] = a[k]
    if "numUes" in out: out["numUes"] = int(out["numUes"])
    if "trafficMbps" in out: out["trafficMbps"] = float(out["trafficMbps"])
    if "duration" in out: out["duration"] = float(out["duration"])
    if "bandwidthMHz" in out: out["bandwidthMHz"] = float(out["bandwidthMHz"])
    if "scheduler" in out: out["scheduler"] = str(out["scheduler"])
    return out

def _read_timeseries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["time_s", "thr_total_mbps", "avg_delay_ms"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns {missing}. Found {df.columns.tolist()}")
    return df

def _select_time_window(df: pd.DataFrame, tmin: Optional[float], tmax: Optional[float]) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    if tmin is not None: m &= (df["time_s"] >= float(tmin))
    if tmax is not None: m &= (df["time_s"] <= float(tmax))
    return df.loc[m].reset_index(drop=True)

# ----------------- NPE inference (robust) -----------------

def _summ_feats_vec(x: np.ndarray, lags=(1,2,3,4)) -> np.ndarray:
    """Return a compact summary for one series: [mean, std, acf(lag=1..4)] -> 6 floats."""
    x = np.asarray(x, dtype=float)
    feats = [float(np.mean(x)), float(np.std(x) + 1e-9)]
    for lag in lags:
        if len(x) > lag:
            xc = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            if np.isnan(xc): xc = 0.0
        else:
            xc = 0.0
        feats.append(float(xc))
    return np.array(feats, dtype=np.float32)

def _assemble_feats_12(df: pd.DataFrame) -> np.ndarray:
    """Default 12-dim feature: 6 for throughput + 6 for delay."""
    thr = df["thr_total_mbps"].to_numpy()
    dly = df["avg_delay_ms"].to_numpy()
    f_thr = _summ_feats_vec(thr)  # 6
    f_dly = _summ_feats_vec(dly)  # 6
    return np.concatenate([f_thr, f_dly], axis=0).astype(np.float32)  # (12,)

def _get_expected_in_dim(model) -> Optional[int]:
    """Try to introspect the first Linear layer input dim for both TorchScript and nn.Module."""
    try:
        # TorchScript/nn.Module with Sequential at .net
        first = model.net[0]
        w = first.weight
        # w shape [out_features, in_features]
        return int(w.shape[1])
    except Exception:
        pass
    # Fallback: try common attributes
    for name in ["fc1", "linear1", "lin1", "in", "input"]:
        try:
            w = getattr(model, name).weight
            return int(w.shape[1])
        except Exception:
            continue
    return None

def _pad_or_trunc(feats: np.ndarray, expected: int) -> np.ndarray:
    d = feats.shape[0]
    if d == expected:
        return feats
    if d < expected:
        return np.pad(feats, (0, expected - d))
    return feats[:expected]

def _infer_seed_with_npe(factual_csv: str, model_path: str, fallback_seed: int = 1) -> int:
    """Robust seed inference: build 12-dim features, then pad/truncate to the model's expected dim."""
    try:
        import torch
    except Exception:
        print("[WARN] torch not available; using fallback seed.")
        return fallback_seed

    if not (factual_csv and os.path.isfile(factual_csv)):
        print("[WARN] factual CSV not found; using fallback seed.")
        return fallback_seed
    if not (model_path and os.path.isfile(model_path)):
        print(f"[WARN] NPE model not found at {model_path}. Using fallback seed={fallback_seed}.")
        return fallback_seed

    df = _read_timeseries(factual_csv)
    feats12 = _assemble_feats_12(df)  # (12,)

    # Load TorchScript or pickled module
    try:
        M = torch.jit.load(model_path, map_location="cpu")
        M.eval()
    except Exception:
        M = torch.load(model_path, map_location="cpu")
        if hasattr(M, "eval"): M.eval()

    expected = _get_expected_in_dim(M)
    if expected is None:
        # Last resort: try to forward once; if it fails with matmul mismatch, assume 14
        expected = 14

    feats = _pad_or_trunc(feats12, expected).astype(np.float32)
    x = torch.from_numpy(feats).unsqueeze(0)  # [1, F]

    # Try forward. If it still shape-errors, try to recover by catching and padding to the
    # rhs dimension parsed from the error message.
    try:
        with torch.no_grad():
            y = M(x)
    except RuntimeError as e:
        # Try to parse “(1x12 and 14x128)” to recover expected input dim=14
        import re
        msg = str(e)
        found = re.findall(r'and\s+(\d+)x', msg)
        if found:
            maybe_in = int(found[0])
            feats = _pad_or_trunc(feats12, maybe_in).astype(np.float32)
            x = torch.from_numpy(feats).unsqueeze(0)
            with torch.no_grad():
                y = M(x)
        else:
            # Give up gracefully
            print("[WARN] NPE forward failed; using fallback seed.")
            return fallback_seed

    # Interpret output
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = y.squeeze()
    if getattr(y, "ndim", 0) >= 1 and y.numel() > 1:
        seed_pred = int(torch.argmax(y).item())
    else:
        seed_pred = int(round(float(y.item())))
    if seed_pred <= 0:
        seed_pred = fallback_seed
    print(f"[infer] CG seed -> {seed_pred}")
    return seed_pred

# ----------------- ns-3 runner -----------------

def _run_ns3_once(action: Dict[str, Any],
                  rng_run: int,
                  ts_dt: float,
                  tmpdir: str,
                  label: str,
                  outdir: str) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    csv_path = os.path.join(outdir, f"{label}_timeseries.csv")
    metrics_tmp = os.path.join(tmpdir, f"metrics_{label}.json")
    try:
        metrics = run_ns3_action(
            action,
            rng_run=rng_run,
            workdir=tmpdir,
            output_json=metrics_tmp,
            timeseries_csv=csv_path,
            ts_dt=float(ts_dt),
        )
    except TypeError:
        raise RuntimeError(
            "Your env_bridge.run_ns3_action() doesn't accept 'timeseries_csv'/'ts_dt'. "
            "Please update env_bridge to forward --tsCsv and --tsDt to the ns-3 binary."
        )
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"Timeseries CSV not found: {csv_path}")
    df = _read_timeseries(csv_path)
    return df, metrics, csv_path

# ----------------- plotting -----------------

def _plot_three(df_true: pd.DataFrame,
                df_cg: pd.DataFrame,
                df_ig: pd.DataFrame,
                out_pdf: str,
                use_tex: bool = False,
                tmin: Optional[float] = None,
                tmax: Optional[float] = None):
    if use_tex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    df_true = _select_time_window(df_true, tmin, tmax)
    df_cg   = _select_time_window(df_cg,   tmin, tmax)
    df_ig   = _select_time_window(df_ig,   tmin, tmax)

    fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True)
    ax[0].plot(df_true["time_s"], df_true["thr_total_mbps"], label="True CF")
    ax[0].plot(df_cg["time_s"],   df_cg["thr_total_mbps"],   label="CG (ours)")
    ax[0].plot(df_ig["time_s"],   df_ig["thr_total_mbps"],   label="IG")
    ax[0].set_ylabel("Throughput (Mbps)")
    ax[0].legend(loc="best", fontsize=9)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df_true["time_s"], df_true["avg_delay_ms"])
    ax[1].plot(df_cg["time_s"],   df_cg["avg_delay_ms"])
    ax[1].plot(df_ig["time_s"],   df_ig["avg_delay_ms"])
    ax[1].set_ylabel("Avg delay (ms)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True, alpha=0.3)

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(out_pdf)
    plt.close()
    print(f"[✓] Wrote {out_pdf}")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Run ns-3 and plot True/CG/IG time series; write CSVs.")
    ap.add_argument("--action-json", type=str, required=True,
                    help="JSON dict with num_ues, scheduler, traffic_mbps, duration_s (bandwidth optional).")
    ap.add_argument("--rng-true", type=int, required=True)
    ap.add_argument("--rng-est",  type=str, required=True,
                    help="CG seed integer OR 'infer' to estimate with NPE.")
    ap.add_argument("--rng-intv", type=int, required=True,
                    help="IG seed integer.")
    ap.add_argument("--ts-dt", type=float, default=0.2, help="Sampling step (s) for ns-3 timeseries.")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--use_tex", action="store_true")

    # NPE inference (optional when --rng-est infer)
    ap.add_argument("--factual-csv", type=str, default="", help="Factual run CSV for NPE features.")
    ap.add_argument("--npe-model", type=str, default="", help="TorchScript (or pickled) model file for seed inference.")

    # optional plotting window
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)

    args = ap.parse_args()
    _ensure_dir(args.outdir)

    try:
        a_raw = json.loads(args.action_json)
    except Exception as e:
        raise SystemExit(f"Invalid --action-json: {e}")
    action = _normalize_action(a_raw)
    print("[action] " + json.dumps(action, indent=2))

    if args.use_tex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    # Resolve CG seed
    if str(args.rng_est).lower() == "infer":
        cg_seed = _infer_seed_with_npe(args.factual_csv, args.npe_model, fallback_seed=1)
    else:
        try:
            cg_seed = int(args.rng_est)
        except Exception:
            cg_seed = 1
            print(f"[WARN] could not parse --rng-est={args.rng_est}; using {cg_seed}")

    with tempfile.TemporaryDirectory(prefix="cfctwin_ns3_") as tmpdir:
        df_true, m_true, csv_true = _run_ns3_once(action, args.rng_true, args.ts_dt, tmpdir, "true", args.outdir)
        df_cg,   m_cg,   csv_cg   = _run_ns3_once(action, cg_seed,      args.ts_dt, tmpdir, "cg",   args.outdir)
        df_ig,   m_ig,   csv_ig   = _run_ns3_once(action, args.rng_intv, args.ts_dt, tmpdir, "ig",   args.outdir)

    pdf_path = os.path.join(args.outdir, "timeseries_true_cg_ig.pdf")
    _plot_three(df_true, df_cg, df_ig, pdf_path, use_tex=args.use_tex, tmin=args.tmin, tmax=args.tmax)

    manifest = {
        "action": action,
        "rng_true": int(args.rng_true),
        "rng_cg": int(cg_seed),
        "rng_ig": int(args.rng_intv),
        "csv_true": os.path.join(args.outdir, "true_timeseries.csv"),
        "csv_cg":   os.path.join(args.outdir, "cg_timeseries.csv"),
        "csv_ig":   os.path.join(args.outdir, "ig_timeseries.csv"),
        "plot_pdf": pdf_path,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "ts_dt": float(args.ts_dt),
    }
    with open(os.path.join(args.outdir, "timeseries_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[✓] Wrote {os.path.join(args.outdir, 'timeseries_manifest.json')}")

if __name__ == "__main__":
    main()