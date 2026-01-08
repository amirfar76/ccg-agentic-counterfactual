# cfllm/replot_clm.py
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Re-plot CLM test figures from saved CSV (no re-sampling).")
    ap.add_argument("--results", required=True, help="Path to outputs/test_clm/results.csv")
    ap.add_argument("--summary", default=None, help="Path to outputs/test_clm/summary.json (for eps0 line)")
    ap.add_argument("--outdir", default=None, help="Output directory for PDFs (default: same folder as CSV)/figs_pdf")
    ap.add_argument("--use-tex", action="store_true", help="Use LaTeX for text rendering (requires a LaTeX install).")
    ap.add_argument("--font-size", type=int, default=10, help="Base font size")
    ap.add_argument("--figsize", default="4,3", help="Width,Height in inches (e.g. 4,3)")
    args = ap.parse_args()

    # Fonts & LaTeX
    if args.use_tex:
        rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "axes.unicode_minus": False,
            "font.size": args.font_size,
        })
    else:
        rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": args.font_size,
        })

    # IO
    df = pd.read_csv(args.results)
    if args.summary:
        try:
            eps0 = json.load(open(args.summary)).get("eps0", None)
        except Exception:
            eps0 = None
    else:
        eps0 = None

    if args.outdir is None:
        base = os.path.dirname(os.path.abspath(args.results))
        outdir = os.path.join(base, "figs_pdf")
    else:
        outdir = args.outdir
    ensure_dir(outdir)

    # Figure size
    try:
        w, h = [float(x) for x in args.figsize.split(",")]
    except Exception:
        w, h = 4.0, 3.0

    # 1) Histogram of min distance to truth
    vals = df["min_dist_to_truth"].values
    finite = vals[np.isfinite(vals)]
    if finite.size:
        plt.figure(figsize=(w, h))
        bins = min(20, max(6, int(np.sqrt(finite.size))))
        plt.hist(finite, bins=bins)
        if eps0 is not None and np.isfinite(eps0):
            plt.axvline(eps0, linestyle="--", linewidth=1)
        plt.xlabel(r"$\min_{y\in C} d\bigl(y,\, Y'_{\mathrm{true}}\bigr)$")
        plt.ylabel("Count")
        # no title
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "min_dist_hist.pdf"), bbox_inches="tight")
        plt.close()

        # 2) ECDF of min distance to truth
        xs = np.sort(finite)
        ys = np.arange(1, xs.size + 1) / xs.size
        plt.figure(figsize=(w, h))
        plt.plot(xs, ys, drawstyle="steps-post")
        if eps0 is not None and np.isfinite(eps0):
            plt.axvline(eps0, linestyle="--", linewidth=1)
        plt.xlabel(r"$\min_{y\in C} d\bigl(y,\, Y'_{\mathrm{true}}\bigr)$")
        plt.ylabel("ECDF")
        # no title
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "min_dist_ecdf.pdf"), bbox_inches="tight")
        plt.close()

    # 3) Histogram of set sizes
    if "set_size" in df.columns and df["set_size"].notna().any():
        data = df["set_size"].values
        plt.figure(figsize=(w, h))
        # bins as integer edges
        max_sz = int(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 0
        edges = np.arange(0, max_sz + 2)
        plt.hist(data, bins=edges, align="left", rwidth=0.9)
        plt.xlabel(r"$|C|$")
        plt.ylabel("Count")
        # no title
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "set_sizes_hist.pdf"), bbox_inches="tight")
        plt.close()

    # 4) Histogram of samples tried
    if "samples_tried" in df.columns and df["samples_tried"].notna().any():
        data = df["samples_tried"].values
        plt.figure(figsize=(w, h))
        bins = min(20, max(6, int(np.sqrt(len(data)))))
        plt.hist(data, bins=bins)
        plt.xlabel("Samples tried")
        plt.ylabel("Count")
        # no title
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "samples_tried_hist.pdf"), bbox_inches="tight")
        plt.close()

    print(f"[✓] Saved PDFs under: {outdir}")

if __name__ == "__main__":
    main()

