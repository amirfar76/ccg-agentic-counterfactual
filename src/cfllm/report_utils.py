# cfllm/report_utils.py
import math
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

# -------------------------------
# Basic IO
# -------------------------------
def read_ts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time_s", "thr_total_mbps", "avg_delay_ms"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.sort_values("time_s").reset_index(drop=True)
    return df

def select_window(df: pd.DataFrame, tmin: float=None, tmax: float=None) -> pd.DataFrame:
    if tmin is not None:
        df = df[df["time_s"] >= tmin]
    if tmax is not None:
        df = df[df["time_s"] <= tmax]
    return df.reset_index(drop=True)

# -------------------------------
# Simple signal features
# -------------------------------
def _time_above_threshold(df: pd.DataFrame, col: str, thr: float) -> float:
    """Returns fraction of time samples with value > thr."""
    if len(df) == 0:
        return 0.0
    return float((df[col].values > thr).mean())

def _num_peaks(x: np.ndarray, min_prominence: float=0.0) -> int:
    """Very light peak count: sign changes in first difference + small prominence."""
    if len(x) < 3:
        return 0
    dx = np.diff(x)
    signs = np.sign(dx)
    # peak when slope goes + to -
    peaks = (signs[:-1] > 0) & (signs[1:] < 0)
    idxs = np.where(peaks)[0] + 1
    if min_prominence <= 0:
        return int(len(idxs))
    # simple prominence: peak - avg(min of neighbors)
    cnt = 0
    for i in idxs:
        left = x[max(0, i-1)]
        right = x[min(len(x)-1, i+1)]
        prom = x[i] - 0.5*(left+right)
        if prom >= min_prominence:
            cnt += 1
    return cnt

def summarize_series(df: pd.DataFrame,
                     thr_thr_mbps: float = 5.0,
                     thr_dly_ms: float = 20.0) -> Dict[str, float]:
    thr = df["thr_total_mbps"].values.astype(float)
    dly = df["avg_delay_ms"].values.astype(float)
    out = {}
    # Throughput stats
    out["thr_mean"]   = float(np.mean(thr)) if len(thr) else 0.0
    out["thr_p95"]    = float(np.percentile(thr, 95)) if len(thr) else 0.0
    out["thr_max"]    = float(np.max(thr)) if len(thr) else 0.0
    out["thr_tabove"] = _time_above_threshold(df, "thr_total_mbps", thr_thr_mbps)

    # Delay stats
    out["dly_mean"]   = float(np.mean(dly)) if len(dly) else 0.0
    out["dly_p95"]    = float(np.percentile(dly, 95)) if len(dly) else 0.0
    out["dly_max"]    = float(np.max(dly)) if len(dly) else 0.0
    out["dly_tabove"] = _time_above_threshold(df, "avg_delay_ms", thr_dly_ms)

    # Burstiness quick features
    out["thr_peaks"]  = _num_peaks(thr, min_prominence=0.0)
    out["dly_peaks"]  = _num_peaks(dly, min_prominence=0.0)

    return out

def render_report(df: pd.DataFrame,
                  label: str,
                  thr_thr_mbps: float = 5.0,
                  thr_dly_ms: float = 20.0) -> str:
    s = summarize_series(df, thr_thr_mbps, thr_dly_ms)
    lines = []
    lines.append(f"## Report: {label}")
    lines.append("")
    lines.append(f"- **Window:** {df['time_s'].min():.2f}s to {df['time_s'].max():.2f}s "
                 f"({len(df)} samples)")
    lines.append("- **Throughput (Mbps):** "
                 f"mean {s['thr_mean']:.2f}, p95 {s['thr_p95']:.2f}, max {s['thr_max']:.2f}, "
                 f"time > {thr_thr_mbps} Mbps: {100*s['thr_tabove']:.1f}%")
    lines.append("- **Delay (ms):** "
                 f"mean {s['dly_mean']:.2f}, p95 {s['dly_p95']:.2f}, max {s['dly_max']:.2f}, "
                 f"time > {thr_dly_ms} ms: {100*s['dly_tabove']:.1f}%")
    lines.append("- **Burstiness:** "
                 f"thr-peaks {int(s['thr_peaks'])}, dly-peaks {int(s['dly_peaks'])}")
    lines.append("")
    return "\n".join(lines)

# -------------------------------
# Offline “judge” metrics
# -------------------------------
def mae(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(np.mean(np.abs(a - b)))

def auc(arr: np.ndarray, dt: float) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.sum(arr) * dt)

def auc_diff(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    return float(abs(auc(a, dt) - auc(b, dt)))

def xcorr_peak(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Normalized cross-correlation peak within +-max_lag samples."""
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    A = (a - a.mean()); B = (b - b.mean())
    sa = np.linalg.norm(A); sb = np.linalg.norm(B)
    if sa == 0 or sb == 0:
        return 0.0
    corr = np.correlate(A, B, mode="full") / (sa * sb)
    mid = len(corr)//2
    lo  = max(0, mid - max_lag)
    hi  = min(len(corr), mid + max_lag + 1)
    return float(np.max(corr[lo:hi]))

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Simple O(N^2) DTW; OK for our short windows."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = abs(ai - b[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[n, m])

def jaccard_above(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    """Jaccard similarity of {indices where value>thr}."""
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    A = (a > thr)
    B = (b > thr)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter) / float(union) if union > 0 else 0.0
