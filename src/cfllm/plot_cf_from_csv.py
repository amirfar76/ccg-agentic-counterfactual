# cfllm/plot_cf_from_csv.py
# Re-plot results *only from the saved CSV* (no re-simulation).
# Outputs:
#   - edit_vs_temp.pdf  (Average edit distance vs temperature)
#   - numeric_violin.pdf (Numeric distance distribution: violin + jitter)
#   - numeric_ecdf.pdf   (Optional ECDF across methods)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _jitter(x, scale=0.04, size=None, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    return x + rng.normal(0, scale, size=size)

def plot_edit_vs_temp(df, outdir, use_tex=False, drop_temps=None):
    df_edit = df[df["metric_name"] == "edit"].copy()
    if drop_temps:
        df_edit = df_edit[~df_edit["temperature"].isin(drop_temps)]

    agg = df_edit.groupby(["temperature", "method"])["distance"].mean().reset_index()

    plt.rcParams["text.usetex"] = bool(use_tex)
    if use_tex:
        plt.rcParams["font.family"] = "serif"

    plt.figure()
    for method, label in [("GM-SCM","GM-SCM"), ("Interventional","Interventional")]:
        sub = agg[agg["method"]==method].sort_values("temperature")
        if not sub.empty:
            plt.plot(sub["temperature"].values, sub["distance"].values, marker="o", label=label)

    plt.xlabel(r"Temperature $\tau$" if use_tex else "Temperature τ")
    plt.ylabel("Average edit distance to truth")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "edit_vs_temp.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")

def plot_numeric_violin(df, outdir, use_tex=False):
    df_num = df[(df["metric_name"]=="numeric")].copy()

    plt.rcParams["text.usetex"] = bool(use_tex)
    if use_tex:
        plt.rcParams["font.family"] = "serif"

    # Prepare data per method
    methods = ["GM-SCM","Interventional"]
    data = [df_num[df_num["method"]==m]["distance"].dropna().values for m in methods]

    # Violin plot
    fig, ax = plt.subplots()
    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(methods)+1),
        showmeans=False,
        showmedians=True,
        widths=0.7
    )
    # Make violins nicer
    for pc in parts['bodies']:
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
    for k in ('cbars','cmins','cmaxes','cmedians'):
        if k in parts:
            parts[k].set_linewidth(1.2)

    # Jittered scatter on top (so you see the distribution)
    rng = np.random.RandomState(0)
    for i, vals in enumerate(data, start=1):
        x = np.full_like(vals, i, dtype=float)
        ax.scatter(_jitter(x, scale=0.05, size=len(vals), rng=rng), vals, s=10, alpha=0.6, linewidths=0)

    ax.set_xticks(np.arange(1, len(methods)+1))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Numeric distance to truth (scaled MSE)")
    ax.set_title("Numeric distance distribution")
    plt.tight_layout()
    path = os.path.join(outdir, "numeric_violin.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")

def plot_numeric_ecdf(df, outdir, use_tex=False):
    df_num = df[(df["metric_name"]=="numeric")].copy()

    plt.rcParams["text.usetex"] = bool(use_tex)
    if use_tex:
        plt.rcParams["font.family"] = "serif"

    methods = ["GM-SCM","Interventional"]
    fig, ax = plt.subplots()
    for m in methods:
        vals = np.sort(df_num[df_num["method"]==m]["distance"].dropna().values)
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals)+1) / len(vals)
        ax.step(vals, y, where='post', label=m)

    ax.set_xlabel("Numeric distance to truth (scaled MSE)")
    ax.set_ylabel("ECDF")
    ax.set_title("Numeric distance ECDF")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "numeric_ecdf.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")

def main():
    ap = argparse.ArgumentParser(description="Re-plot CF results from saved CSV (no re-simulation).")
    ap.add_argument("--csv", type=str, required=True, help="Path to cf_compare_distances.csv")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for PDFs")
    ap.add_argument("--use-tex", action="store_true", help="Use LaTeX rendering")
    ap.add_argument("--drop-temps", type=str, default="", help="Comma-separated temps to drop (e.g. 0.01)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.csv)

    # Parse drop-temps list
    drop = []
    if args.drop_temps.strip():
        drop = [float(x.strip()) for x in args.drop_temps.split(",") if x.strip()]

    # Plots
    plot_edit_vs_temp(df, args.outdir, use_tex=args.use_tex, drop_temps=drop)
    plot_numeric_violin(df, args.outdir, use_tex=args.use_tex)
    plot_numeric_ecdf(df, args.outdir, use_tex=args.use_tex)

if __name__ == "__main__":
    main()
