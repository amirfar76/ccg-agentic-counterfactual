# -*- coding: utf-8 -*-
"""
Evaluate CG vs IG against the True counterfactual using time-series.
- Loads three CSVs with columns: time_s, thr_total_mbps, avg_delay_ms
- Aligns on a common grid (interpolates CG/IG to True grid if needed)
- Applies an optional time window [tmin, tmax]
- Plots: time-series (thr & delay), absolute-errors over time, metric bars
- Computes metrics: RMSE, MAE, corr(0), best-lag RMSE, Jaccard(>thr),
  time-above-thr error, first-passage-time error, Wasserstein-1 (1D approx)
- Writes CSV + Markdown summary + PDFs

Usage:
  python -m cfllm.eval_timeseries \
    --true outputs/ts_plots/true_timeseries.csv \
    --cg   outputs/ts_plots/cg_timeseries.csv \
    --ig   outputs/ts_plots/ig_timeseries.csv \
    --outdir outputs/ts_eval \
    --tmin 1.0 \
    --max-lag-s 1.0 \
    --use_tex
"""

import os
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------------- I/O helpers --------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"time_s", "thr_total_mbps", "avg_delay_ms"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    # enforce numeric
    for c in ["time_s", "thr_total_mbps", "avg_delay_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    return df

def _window(df: pd.DataFrame, tmin: Optional[float], tmax: Optional[float]) -> pd.DataFrame:
    if tmin is not None:
        df = df[df["time_s"] >= tmin]
    if tmax is not None:
        df = df[df["time_s"] <= tmax]
    return df.reset_index(drop=True)

def _interp_to(times_ref: np.ndarray, times_src: np.ndarray, values_src: np.ndarray) -> np.ndarray:
    """Linear interpolate values_src@times_src onto times_ref."""
    if len(times_src) < 2:
        # degenerate; just pad with the only value or zeros
        return np.interp(times_ref, [times_ref[0], times_ref[-1]], [values_src[0] if len(values_src) else 0.0, values_src[0] if len(values_src) else 0.0])
    return np.interp(times_ref, times_src, values_src)

def _align_on_true_grid(df_true: pd.DataFrame, df_hat: pd.DataFrame) -> pd.DataFrame:
    """Return hat aligned to true's time grid via linear interpolation."""
    t_ref = df_true["time_s"].to_numpy()
    out = pd.DataFrame({"time_s": t_ref})
    for col in ["thr_total_mbps", "avg_delay_ms"]:
        out[col] = _interp_to(t_ref, df_hat["time_s"].to_numpy(), df_hat[col].to_numpy())
    return out

# -------------------------- metrics --------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def corr0(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def best_lag_rmse(x_true: np.ndarray, x_hat: np.ndarray, max_lag_samples: int) -> Tuple[float, int]:
    best_e, best_tau = 1e18, 0
    L = max_lag_samples
    for tau in range(-L, L + 1):
        if tau >= 0:
            a, b = x_true[tau:], x_hat[: len(x_true) - tau]
        else:
            a, b = x_true[: len(x_true) + tau], x_hat[-tau:]
        if len(a) < max(5, L // 2):
            continue
        e = rmse(a, b)
        if e < best_e:
            best_e, best_tau = e, tau
    return float(best_e), int(best_tau)

def jaccard_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0

def first_passage_time(x: np.ndarray, t: np.ndarray, thr: float) -> Optional[float]:
    above = x >= thr
    idx = np.argmax(above)
    if above[idx]:
        return float(t[idx])
    return None

def wasserstein1_sorted(x: np.ndarray, y: np.ndarray) -> float:
    """
    1D Wasserstein-1 for equal-weight empirical distributions:
    mean absolute difference between sorted samples.
    If lengths differ, we interpolate onto common quantiles.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    if nx == ny:
        xs = np.sort(x)
        ys = np.sort(y)
        return float(np.mean(np.abs(xs - ys)))
    # resample both to common grid of quantiles
    q = np.linspace(0.0, 1.0, num=min(nx, ny), endpoint=True)
    xs = np.quantile(x, q)
    ys = np.quantile(y, q)
    return float(np.mean(np.abs(xs - ys)))

# -------------------------- plotting --------------------------

def _maybe_use_tex(use_tex: bool):
    if use_tex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["axes.unicode_minus"] = False
        # keep preamble minimal; avoid unicode in strings
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

def _safe_tight():
    try:
        plt.tight_layout()
    except Exception:
        pass

def plot_timeseries(df_true, df_cg, df_ig, outdir, title_suffix=""):
    # Throughput
    plt.figure(figsize=(6.6, 3.8))
    plt.plot(df_true.time_s, df_true.thr_total_mbps, lw=1.4, label="True CF")
    plt.plot(df_cg.time_s,   df_cg.thr_total_mbps,   lw=1.1, label="CG")
    plt.plot(df_ig.time_s,   df_ig.thr_total_mbps,   lw=1.1, label="IG")
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    if title_suffix:
        plt.title(f"Throughput {title_suffix}")
    plt.legend(frameon=False)
    _safe_tight()
    p = os.path.join(outdir, "timeseries_throughput.pdf")
    plt.savefig(p)
    plt.close()

    # Delay
    plt.figure(figsize=(6.6, 3.8))
    plt.plot(df_true.time_s, df_true.avg_delay_ms, lw=1.4, label="True CF")
    plt.plot(df_cg.time_s,   df_cg.avg_delay_ms,   lw=1.1, label="CG")
    plt.plot(df_ig.time_s,   df_ig.avg_delay_ms,   lw=1.1, label="IG")
    plt.xlabel("Time (s)")
    plt.ylabel("Average delay (ms)")
    if title_suffix:
        plt.title(f"Delay {title_suffix}")
    plt.legend(frameon=False)
    _safe_tight()
    p = os.path.join(outdir, "timeseries_delay.pdf")
    plt.savefig(p)
    plt.close()

def plot_abs_error(df_true, df_cg, df_ig, outdir, title_suffix=""):
    # |hat - true| over time (throughput)
    e_cg_thr = np.abs(df_cg.thr_total_mbps.to_numpy() - df_true.thr_total_mbps.to_numpy())
    e_ig_thr = np.abs(df_ig.thr_total_mbps.to_numpy() - df_true.thr_total_mbps.to_numpy())
    plt.figure(figsize=(6.6, 3.3))
    plt.plot(df_true.time_s, e_cg_thr, lw=1.1, label="|CG − True|")
    plt.plot(df_true.time_s, e_ig_thr, lw=1.1, label="|IG − True|")
    plt.xlabel("Time (s)")
    plt.ylabel("Abs error (Mbps)")
    if title_suffix:
        plt.title(f"Throughput abs error {title_suffix}")
    plt.legend(frameon=False)
    _safe_tight()
    p = os.path.join(outdir, "abs_error_throughput.pdf")
    plt.savefig(p)
    plt.close()

    # |hat - true| over time (delay)
    e_cg_d = np.abs(df_cg.avg_delay_ms.to_numpy() - df_true.avg_delay_ms.to_numpy())
    e_ig_d = np.abs(df_ig.avg_delay_ms.to_numpy() - df_true.avg_delay_ms.to_numpy())
    plt.figure(figsize=(6.6, 3.3))
    plt.plot(df_true.time_s, e_cg_d, lw=1.1, label="|CG − True|")
    plt.plot(df_true.time_s, e_ig_d, lw=1.1, label="|IG − True|")
    plt.xlabel("Time (s)")
    plt.ylabel("Abs error (ms)")
    if title_suffix:
        plt.title(f"Delay abs error {title_suffix}")
    plt.legend(frameon=False)
    _safe_tight()
    p = os.path.join(outdir, "abs_error_delay.pdf")
    plt.savefig(p)
    plt.close()

def plot_metric_bars(metric_rows: List[Dict], outdir: str, title: str):
    """
    metric_rows: list of dicts with keys:
      'metric', 'signal' in {'thr','delay'}, 'cg', 'ig', 'better' ('lower'|'higher')
    Produces separate bar plots for each metric.
    """
    # group by metric name
    by_metric: Dict[str, List[Dict]] = {}
    for r in metric_rows:
        by_metric.setdefault(r["metric"], []).append(r)

    for mname, rows in by_metric.items():
        signals = ["thr", "delay"]
        labels = {"thr":"Throughput", "delay":"Delay"}
        cg_vals, ig_vals, xticks = [], [], []
        for sig in signals:
            r = [x for x in rows if x["signal"] == sig]
            if r:
                rr = r[0]
                cg_vals.append(rr["cg"])
                ig_vals.append(rr["ig"])
                xticks.append(labels[sig])
        if not cg_vals:
            continue

        x = np.arange(len(cg_vals))
        w = 0.35
        plt.figure(figsize=(6.0, 3.8))
        plt.bar(x - w/2, cg_vals, width=w, label="CG")
        plt.bar(x + w/2, ig_vals, width=w, label="IG")
        plt.xticks(x, xticks)
        plt.ylabel(mname)
        plt.title(title + " — " + mname)
        plt.legend(frameon=False)
        _safe_tight()
        p = os.path.join(outdir, f"bars_{mname.replace(' ','_').replace('/','_')}.pdf")
        plt.savefig(p)
        plt.close()

# -------------------------- evaluate one metric set --------------------------

def eval_against_true(t: np.ndarray, x_true: np.ndarray, x_cg: np.ndarray, x_ig: np.ndarray,
                      ts_dt: float, max_lag_s: float) -> Dict[str, Dict[str, float]]:
    """
    Returns nested dict:
      metric_name -> {'cg': val, 'ig': val, 'better': 'lower' or 'higher'}
    """
    out: Dict[str, Dict[str, float]] = {}
    L = max(1, int(round(max_lag_s / max(ts_dt, 1e-9))))

    # Errors
    out["RMSE"] = {"cg": rmse(x_true, x_cg), "ig": rmse(x_true, x_ig), "better": "lower"}
    out["MAE"]  = {"cg": mae(x_true, x_cg),  "ig": mae(x_true, x_ig),  "better": "lower"}

    # Correlations
    out["Corr@0"] = {"cg": corr0(x_true, x_cg), "ig": corr0(x_true, x_ig), "better": "higher"}

    # Best-lag RMSE
    br_cg, _ = best_lag_rmse(x_true, x_cg, L)
    br_ig, _ = best_lag_rmse(x_true, x_ig, L)
    out["Best-lag RMSE"] = {"cg": br_cg, "ig": br_ig, "better": "lower"}

    # Threshold-based (use 75th pct of True)
    thr = float(np.quantile(x_true, 0.75))
    m_t = x_true >= thr
    m_c = x_cg  >= thr
    m_i = x_ig  >= thr
    out["Jaccard(>thr)"] = {"cg": jaccard_bool(m_t, m_c), "ig": jaccard_bool(m_t, m_i), "better": "higher"}

    tab_t = float(m_t.mean())
    tab_c = float(m_c.mean())
    tab_i = float(m_i.mean())
    out["Time-above-thr abs err"] = {"cg": abs(tab_c - tab_t), "ig": abs(tab_i - tab_t), "better": "lower"}

    # First passage time
    fpt_t = first_passage_time(x_true, t, thr)
    fpt_c = first_passage_time(x_cg,   t, thr)
    fpt_i = first_passage_time(x_ig,   t, thr)
    def fpt_err(fpt_hat):
        if fpt_t is None and fpt_hat is None:
            return 0.0
        if (fpt_t is None) ^ (fpt_hat is None):
            return float("inf")
        return abs(fpt_hat - fpt_t)
    out["First-passage-time err (s)"] = {"cg": fpt_err(fpt_c), "ig": fpt_err(fpt_i), "better": "lower"}

    # Distributional
    out["Wasserstein-1 (value)"] = {
        "cg": wasserstein1_sorted(x_true, x_cg),
        "ig": wasserstein1_sorted(x_true, x_ig),
        "better": "lower",
    }

    return out

# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate CG vs IG time-series against True CF.")
    ap.add_argument("--true", required=True, help="CSV for True CF")
    ap.add_argument("--cg",   required=True, help="CSV for CG series")
    ap.add_argument("--ig",   required=True, help="CSV for IG series")
    ap.add_argument("--outdir", required=True, help="Where to save outputs")
    ap.add_argument("--tmin", type=float, default=None, help="Crop start time (s)")
    ap.add_argument("--tmax", type=float, default=None, help="Crop end time (s)")
    ap.add_argument("--max-lag-s", type=float, default=1.0, help="Best-lag search half-window (s)")
    ap.add_argument("--use_tex", action="store_true", help="Use LaTeX text rendering")
    args = ap.parse_args()

    _ensure_dir(args.outdir)
    _maybe_use_tex(args.use_tex)

    df_t = _load_csv(args.true)
    df_cg = _load_csv(args.cg)
    df_ig = _load_csv(args.ig)

    # Align to True's grid (after windowing we will re-align again to be safe)
    # Window
    df_t  = _window(df_t,  args.tmin, args.tmax)
    df_cg = _window(df_cg, args.tmin, args.tmax)
    df_ig = _window(df_ig, args.tmin, args.tmax)

    # Re-align CG/IG onto True grid (linear interpolation)
    df_cg = _align_on_true_grid(df_t, df_cg)
    df_ig = _align_on_true_grid(df_t, df_ig)

    # Timeseries plots (nice to look at the exact region you care about)
    title_suffix = ""
    if args.tmin is not None or args.tmax is not None:
        lo = f"{args.tmin:g}" if args.tmin is not None else ""
        hi = f"{args.tmax:g}" if args.tmax is not None else ""
        title_suffix = f"[t∈({lo},{hi})]".replace("∈","in")  # TeX-safe-ish
    plot_timeseries(df_t, df_cg, df_ig, args.outdir, title_suffix)
    plot_abs_error(df_t, df_cg, df_ig, args.outdir, title_suffix)

    # dt estimate from True grid
    if len(df_t) >= 2:
        ts_dt = float(np.median(np.diff(df_t.time_s.to_numpy())))
    else:
        ts_dt = 0.0

    # Evaluate for both signals
    metrics_rows: List[Dict] = []
    for col, signal_name in [("thr_total_mbps", "thr"), ("avg_delay_ms", "delay")]:
        mt = eval_against_true(
            t=df_t.time_s.to_numpy(),
            x_true=df_t[col].to_numpy(),
            x_cg=df_cg[col].to_numpy(),
            x_ig=df_ig[col].to_numpy(),
            ts_dt=ts_dt,
            max_lag_s=args.max_lag_s,
        )
        # flatten for CSV + bar plots
        for m, v in mt.items():
            metrics_rows.append({
                "metric": m,
                "signal": signal_name,
                "cg": v["cg"],
                "ig": v["ig"],
                "better": v["better"],
            })

    # Save metrics table (CSV)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(args.outdir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Bar plots (one per metric)
    plot_metric_bars(metrics_rows, args.outdir, title="CG vs IG vs True")

    # Simple win/tie/lose scoreboard (CG vs IG)
    def _win_score(row):
        better = row["better"]
        cg = row["cg"]; ig = row["ig"]
        if np.isinf(cg) and np.isinf(ig):
            return 0.0
        if better == "lower":
            if cg < ig: return 1.0
            if cg > ig: return 0.0
            return 0.5
        else:
            if cg > ig: return 1.0
            if cg < ig: return 0.0
            return 0.5

    metrics_df["cg_win"] = metrics_df.apply(_win_score, axis=1)
    total = float(len(metrics_df))
    wins = float((metrics_df["cg_win"] == 1.0).sum())
    ties = float((metrics_df["cg_win"] == 0.5).sum())
    loses = float((metrics_df["cg_win"] == 0.0).sum())

    # Markdown summary
    md_path = os.path.join(args.outdir, "metrics_summary.md")
    with open(md_path, "w") as f:
        f.write("# Time-series evaluation (CG vs IG vs True)\n\n")
        if args.tmin is not None or args.tmax is not None:
            f.write(f"- Window: t in [{args.tmin if args.tmin is not None else '-inf'}, "
                    f"{args.tmax if args.tmax is not None else '+inf'}] s\n")
        f.write(f"- Best-lag search ±{args.max_lag_s:g} s\n")
        f.write("\n## Scoreboard (pairwise CG vs IG)\n")
        f.write(f"- Wins: **{int(wins)}** / {int(total)}\n")
        f.write(f"- Ties: **{int(ties)}** / {int(total)}\n")
        f.write(f"- Losses: **{int(loses)}** / {int(total)}\n\n")
        f.write("## Per-metric table\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n")

    print(f"[✓] Wrote {metrics_csv}")
    print(f"[✓] Wrote {md_path}")
    print(f"[✓] Wrote plots to {args.outdir}")
    print(f"[CG scoreboard] wins={int(wins)} ties={int(ties)} losses={int(loses)} / total={int(total)}")
    # Quick hint which series wins more often
    if wins + 0.5 * ties > loses + 0.5 * ties:
        print("Result: CG beats IG overall on these metrics.")
    elif wins + 0.5 * ties < loses + 0.5 * ties:
        print("Result: IG beats CG overall on these metrics.")
    else:
        print("Result: CG and IG are tied overall.")

if __name__ == "__main__":
    main()
