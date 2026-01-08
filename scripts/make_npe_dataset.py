#!/usr/bin/env python3
# scripts/make_npe_dataset.py
# Build a dataset (X, y) by running ns-3 at many rngRun seeds and extracting
# features used by NPE seed inference.

from __future__ import annotations
import os, sys, json, argparse, tempfile
import numpy as np
import pandas as pd

# --- Make sure we can import the local package cfllm regardless of CWD ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfllm.env_bridge import run_ns3_action  # uses your existing runner

def read_ts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize names just in case
    rename = {
        "time": "time_s",
        "time_sec": "time_s",
        "thr_mbps": "thr_total_mbps",
        "throughput_mbps": "thr_total_mbps",
        "delay_ms": "avg_delay_ms",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    keep = [c for c in ["time_s", "thr_total_mbps", "avg_delay_ms"] if c in df.columns]
    if not keep:
        raise RuntimeError(f"{csv_path} missing required columns")
    df = df[keep].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if "time_s" in df.columns:
        df = df.sort_values("time_s").reset_index(drop=True)
    else:
        df["time_s"] = np.arange(len(df), dtype=float)
    return df

def features_from_df(df: pd.DataFrame) -> np.ndarray:
    feats = []
    # throughput features (10 total: mean, std, max, min, 6 FFT magnitudes)
    if "thr_total_mbps" in df.columns:
        x = df["thr_total_mbps"].values.astype(np.float32)
        feats += [float(np.mean(x)), float(np.std(x)+1e-9), float(np.max(x)), float(np.min(x))]
        fft = np.fft.rfft(x - x.mean())
        mags = np.abs(fft)
        k = min(6, len(mags))
        feats += [float(m) for m in mags[:k]]
        if k < 6:
            feats += [0.0]*(6-k)
    else:
        feats += [0.0]*10
    # delay features (4 total: mean, std, max, min)
    if "avg_delay_ms" in df.columns:
        y = df["avg_delay_ms"].values.astype(np.float32)
        feats += [float(np.mean(y)), float(np.std(y)+1e-9), float(np.max(y)), float(np.min(y))]
    else:
        feats += [0.0]*4
    return np.array(feats, dtype=np.float32)  # shape (14,)

def normalize_action_json(s: str) -> dict:
    raw = json.loads(s)
    return {
        "numUes": int(raw.get("num_ues", raw.get("numUes", 5))),
        "scheduler": str(raw.get("scheduler", "rr")),
        "trafficMbps": float(raw.get("traffic_mbps", raw.get("trafficMbps", 8))),
        "duration": float(raw.get("duration_s", raw.get("duration", 8))),
    }

def main():
    ap = argparse.ArgumentParser("Generate (X,y) dataset for seed NPE")
    ap.add_argument("--action-json", required=True, type=str)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end",   type=int, default=200)  # inclusive
    ap.add_argument("--ts-dt", type=float, default=0.2)
    ap.add_argument("--outdir", required=True, type=str)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    action = normalize_action_json(args.action_json)

    seeds = list(range(args.seed_start, args.seed_end+1))
    X, y = [], []
    csv_dir = os.path.join(args.outdir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    for s in seeds:
        out_csv = os.path.join(csv_dir, f"seed_{s}.csv")
        with tempfile.TemporaryDirectory(prefix="npe_ns3_") as tmp:
            _metrics, _csv_path = run_ns3_action(
                action=action,
                rng_run=int(s),
                workdir=tmp,
                ts_csv=out_csv,
                ts_dt=float(args.ts_dt),
            )
        df = read_ts(out_csv)
        X.append(features_from_df(df))
        y.append(s)

    X = np.stack(X, axis=0)  # (N, 14)
    y = np.array(y, dtype=np.int64)

    np.savez(os.path.join(args.outdir, "dataset_seed_np.npz"), X=X, y=y)
    pd.DataFrame({"seed": y}).to_csv(os.path.join(args.outdir, "seed_index.csv"), index=False)

    print(f"[✓] Wrote {args.outdir}/dataset_seed_np.npz (X:{X.shape}, y:{y.shape})")

if __name__ == "__main__":
    main()