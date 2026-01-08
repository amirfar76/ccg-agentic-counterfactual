# cfllm/plot_cf_pairs.py
# Plots CG (GM-SCM) vs IG (Interventional) from a saved CSV (no recompute).
# Numeric metric is temperature-agnostic. Distances > 1 are dropped for numeric.
# TeX-safe labeling (no raw unicode).

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------- Method label mapping ---------
METHOD_MAP = {
    "GM-SCM": "CG",
    "Interventional": "IG",
}

# --------- TeX-safe helpers (no unicode) ---------

_ASCII_TO_TEX: Dict[str, str] = {
    "->": r"$\to$",
    "<-": r"$\leftarrow$",
    "=>": r"$\Rightarrow$",
    "<=": r"$\leq$",
    ">=": r"$\geq$",
}
_UNICODE_TO_TEX: Dict[str, str] = {
    "⇒": r"$\Rightarrow$",
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "τ": r"$\tau$",
    "–": "-",
    "—": "-",
    "’": "'",
    "“": "``",
    "”": "''",
    "•": r"$\bullet$",
}

def _to_tex(s: str) -> str:
    if not isinstance(s, str):
        return s
    for k, v in _UNICODE_TO_TEX.items():
        s = s.replace(k, v)
    for k, v in _ASCII_TO_TEX.items():
        s = s.replace(k, v)
    # strip any non-ascii that slipped through
    s = s.encode("ascii", "ignore").decode("ascii")
    return s

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _safe_tight():
    try:
        plt.tight_layout()
    except Exception:
        pass

# --------- Load & clean ---------

REQUIRED_COLS = {
    "idx",
    "rng_true_env",
    "rng_est_env",
    "rng_intv_env",
    "seed_report_fixed",
    "seed_report_fresh",
    "temperature",
    "method",
    "metric_name",
    "distance",
}

def _load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # Types
    for c in ["idx", "rng_true_env", "rng_est_env", "rng_intv_env", "seed_report_fixed", "seed_report_fresh"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["method"] = df["method"].astype(str).str.strip()
    df["metric_name"] = df["metric_name"].astype(str).str.strip()

    # Map method names
    df["method"] = df["method"].replace(METHOD_MAP)

    # Drop rows without essential fields
    df = df.dropna(subset=["idx", "method", "metric_name", "distance"])

    return df

# --------- Pairing utilities ---------

def _paired_edit_by_temp(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["metric_name"] == "edit") & (~df["temperature"].isna())].copy()
    if sub.empty:
        return sub
    piv = sub.pivot_table(index=["idx", "temperature"],
                          columns="method",
                          values="distance",
                          aggfunc="first")
    for m in ["CG", "IG"]:
        if m not in piv.columns:
            piv[m] = np.nan
    piv = piv.reset_index().dropna(subset=["CG", "IG"])
    piv["diff"] = piv["IG"] - piv["CG"]
    return piv.sort_values(["temperature", "idx"]).reset_index(drop=True)

def _paired_numeric_overall(df: pd.DataFrame, drop_over_1: bool = True) -> pd.DataFrame:
    sub = df[df["metric_name"] == "numeric"].copy()
    if drop_over_1:
        sub = sub[sub["distance"] <= 1.0]
    sub = sub[sub["distance"].notna()]
    if sub.empty:
        return sub
    piv = sub.pivot_table(index=["idx"],
                          columns="method",
                          values="distance",
                          aggfunc="first")
    for m in ["CG", "IG"]:
        if m not in piv.columns:
            piv[m] = np.nan
    piv = piv.reset_index().dropna(subset=["CG", "IG"])
    piv["diff"] = piv["IG"] - piv["CG"]
    return piv.sort_values(["idx"]).reset_index(drop=True)

# --------- Plotters ---------

def plot_edit_paired_by_temp(df: pd.DataFrame, outdir: str):
    p = _paired_edit_by_temp(df)
    if p.empty:
        return
    temps = sorted(p["temperature"].unique())
    plt.figure(figsize=(6.3, 4.2))
    for T in temps:
        sub = p[p["temperature"] == T]
        for _, r in sub.iterrows():
            plt.plot([0, 1], [r["CG"], r["IG"]], lw=0.8, alpha=0.35, color="#888")
        plt.scatter(np.zeros(len(sub)), sub["CG"].values, s=18)
        plt.scatter(np.ones(len(sub)), sub["IG"].values, s=18)

    plt.xticks([0, 1], [_to_tex("CG"), _to_tex("IG")])
    plt.ylabel(_to_tex("Edit distance to true CF"))
    _safe_tight()
    path = os.path.join(outdir, "edit_paired_by_temp.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")

def plot_edit_diff_by_temp(df: pd.DataFrame, outdir: str):
    p = _paired_edit_by_temp(df)
    if p.empty:
        return
    temps = sorted(p["temperature"].unique())
    data = [p[p["temperature"] == T]["diff"].values for T in temps]

    plt.figure(figsize=(6.3, 4.2))
    plt.boxplot(data, showfliers=False)
    plt.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    plt.xticks(range(1, len(temps) + 1), [f"{T:g}" for T in temps])
    plt.xlabel(_to_tex("Temperature $\\tau$"))
    plt.ylabel(_to_tex(r"$\Delta$ edit (IG $-$ CG)"))
    _safe_tight()
    path = os.path.join(outdir, "edit_diff_by_temp.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")

def plot_numeric_paired_overall(df: pd.DataFrame, outdir: str):
    p = _paired_numeric_overall(df, drop_over_1=True)
    if p.empty:
        return
    y0 = p["CG"].values
    y1 = p["IG"].values

    plt.figure(figsize=(6.3, 4.2))
    for a, b in zip(y0, y1):
        plt.plot([0, 1], [a, b], lw=0.8, alpha=0.35, color="#888")
    plt.scatter(np.zeros(len(p)), y0, s=18, label=_to_tex("CG"))
    plt.scatter(np.ones(len(p)), y1, s=18, label=_to_tex("IG"))
    plt.xticks([0, 1], [_to_tex("CG"), _to_tex("IG")])
    plt.ylabel(_to_tex("Numeric distance to true CF"))
    _safe_tight()
    path = os.path.join(outdir, "numeric_paired_overall.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")

    # symlog plot if needed
    ymin, ymax = float(np.nanmin([y0.min(), y1.min()])), float(np.nanmax([y0.max(), y1.max()]))
    if ymax > 0 and (ymax / max(ymin, 1e-12) > 1e3):
        plt.figure(figsize=(6.3, 4.2))
        for a, b in zip(y0, y1):
            plt.plot([0, 1], [a, b], lw=0.8, alpha=0.35, color="#888")
        plt.scatter(np.zeros(len(p)), y0, s=18)
        plt.scatter(np.ones(len(p)), y1, s=18)
        plt.yscale("symlog", linthresh=1e-8)
        plt.xticks([0, 1], [_to_tex("CG"), _to_tex("IG")])
        plt.ylabel(_to_tex("Numeric distance (symlog)"))
        _safe_tight()
        path2 = os.path.join(outdir, "numeric_paired_overall_symlog.pdf")
        plt.savefig(path2)
        plt.close()
        print(f"[✓] Wrote {path2}")

def plot_numeric_diff_overall(df: pd.DataFrame, outdir: str):
    p = _paired_numeric_overall(df, drop_over_1=True)
    if p.empty:
        return
    vals = p["diff"].values

    plt.figure(figsize=(6.3, 4.2))
    parts = plt.violinplot(vals, showextrema=False, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_alpha(0.4)
    plt.boxplot(vals, widths=0.18, positions=[1], showfliers=False)
    plt.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    plt.xticks([1], [_to_tex("All cases")])
    plt.ylabel(_to_tex(r"$\Delta$ numeric (IG $-$ CG)"))
    _safe_tight()
    path = os.path.join(outdir, "numeric_diff_overall.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")

# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(description="Paired plots for CG vs IG (from CSV only).")
    ap.add_argument("--csv", type=str, required=True, help="cf_compare_distances.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Where to write PDFs")
    ap.add_argument("--use-tex", action="store_true", help="Use LaTeX text rendering")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    if args.use_tex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    df = _load_df(args.csv)

    # Edit plots (by temperature)
    plot_edit_paired_by_temp(df, args.outdir)
    plot_edit_diff_by_temp(df, args.outdir)

    # Numeric plots (temperature-agnostic)
    plot_numeric_paired_overall(df, args.outdir)
    plot_numeric_diff_overall(df, args.outdir)


if __name__ == "__main__":
    main()