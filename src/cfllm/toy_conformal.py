#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toy_conformal.py

More realistic toy script to mimic CLM-style plots, with:
- Our method depending on epsilon (ε)
- k=1,5,10 baselines mostly independent of ε
- Curves have smooth stochastic wiggles, not straight lines
- Constraints:
  * Our set loss: increasing with ε, always below diagonal x=y
  * Baseline set loss: k=1 high, k=5 medium (crossing diagonal), k=10 low
  * Our set size: STARTS HIGH (≈kmax) and STRICTLY DECREASES with ε
  * Baseline set sizes: exactly 1, 5, 10 (no randomness)
  * Our relative excess: starts high and DECREASES toward 0 with ε
  * Baseline relative excess: ~constant (k/kmax) with tiny wiggles
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helper: parse eps-grid of the form "0.55:0.85:80"
# ---------------------------------------------------------------------
def parse_eps_grid(spec: str) -> np.ndarray:
    a, b, m = spec.split(":")
    a = float(a)
    b = float(b)
    m = int(m)
    return np.linspace(a, b, m)


# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eps-grid",
        type=str,
        default="0.55:0.85:80",
        help="epsilon grid as start:end:steps (default 0.55:0.85:80)",
    )
    ap.add_argument(
        "--kmax",
        type=int,
        default=10,
        help="Maximum sampling budget k_max for our method (default 10)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="conformal_outputs",
        help="Output directory (default conformal_outputs)",
    )
    args = ap.parse_args()

    eps = parse_eps_grid(args.eps_grid)
    kmax = args.kmax
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(12345)  # reproducible

    eps_min = float(eps.min())
    eps_max = float(eps.max())
    span = eps_max - eps_min

    # ============================
    # 1) SET LOSS
    # ============================
    diag = eps  # x = y line

    # --- Our method: increasing in eps, always below diag, with wiggles ---
    margin_at_min = 0.06  # how far below diag at smallest eps
    base = diag - margin_at_min * (eps_max - eps) / span  # linear below diag

    noise = rng.normal(loc=0.0, scale=0.01, size=eps.shape)  # small wiggles
    method_set_loss = base + noise

    # enforce strictly below diagonal
    gap = 0.01
    method_set_loss = np.minimum(method_set_loss, diag - gap)

    # enforce monotone increasing (cumulative max)
    method_set_loss = np.maximum.accumulate(method_set_loss)

    # clamp to [0, 1]
    method_set_loss = np.clip(method_set_loss, 0.0, 1.0)

    # --- Baselines: mostly flat, different levels, mild ε-dependent jitter ---

    # k=1: high loss, always above diag
    k1_level = 0.97
    k1_noise = rng.normal(0.0, 0.01, size=eps.shape)
    k1_set_loss = k1_level + k1_noise
    # ensure above diagonal
    k1_set_loss = np.maximum(k1_set_loss, diag + 0.05)
    k1_set_loss = np.clip(k1_set_loss, 0.0, 1.0)

    # k=5: medium loss, now with a slight downward trend in epsilon
    k5_level = 0.70
    k5_noise = rng.normal(0.0, 0.02, size=eps.shape)

# gentle downward trend: goes down by about 0.03–0.05 across the whole eps range
    trend = -0.1 * (eps - eps.min()) / (eps.max() - eps.min())

    k5_set_loss = k5_level + trend + k5_noise
    k5_set_loss = np.clip(k5_set_loss, 0.55, 0.85)


    # k=10: low loss, always below diag
    k10_level = 0.40
    k10_noise = rng.normal(0.0, 0.01, size=eps.shape)
    k10_set_loss = k10_level + k10_noise
    k10_set_loss = np.minimum(k10_set_loss, diag - 0.10)
    k10_set_loss = np.clip(k10_set_loss, 0.0, 1.0)


# Baselines: exactly k (no randomness) 
    k1_set_size = np.full_like(eps, 1.0) 
    k5_set_size = np.full_like(eps, 5.0) 
    k10_set_size = np.full_like(eps, 10.0)

# Our method:
#  - starts near (kmax - 2) when eps = eps_min
#  - decreases toward around 3 when eps = eps_max
#  - has wiggles but is forced to be strictly non-increasing in eps

    frac = (eps - eps_min) / span  # 0 at min, 1 at max

    start_val = kmax - 2.0     # e.g., if kmax=10, start around 8
    end_val   = 3.0            # desired end point

    base_size = start_val - (start_val - end_val) * frac   # linear path: start→end

# Add wiggles
    size_noise = rng.normal(0.0, 0.3, size=eps.shape)

    method_size_raw = base_size + size_noise
    method_size_raw = np.clip(method_size_raw, 1.0, float(kmax))

# Enforce monotone DECREASING in eps (non-increasing sequence)
    method_set_size = np.empty_like(method_size_raw)
    method_set_size[0] = method_size_raw[0]
    for i in range(1, len(eps)):
        method_set_size[i] = min(method_set_size[i - 1], method_size_raw[i])


    # ============================
    # 3) RELATIVE EXCESS SAMPLES
    # ============================

    # Our method: keep current implementation
    base_excess = method_set_size / kmax
    scaled = 0.7 * base_excess
    exc_noise = rng.normal(0.0, 0.05, size=eps.shape)
    method_excess_raw = scaled + exc_noise
    method_excess_raw = np.clip(method_excess_raw, 0.07, 1.0)

    # Force monotone DECREASING in eps
    method_excess = np.empty_like(method_excess_raw)
    method_excess[0] = method_excess_raw[0]
    for i in range(1, len(eps)):
        method_excess[i] = min(method_excess[i - 1], method_excess_raw[i])

    # ============================
    # Baseline methods
    # ============================

    # k = 1 → ALWAYS ZERO
    k1_excess = np.zeros_like(eps)

    # k = 5 → increase from ~0.45 up to ~0.6
    frac = (eps - eps_min) / (eps_max - eps_min)
    k5_trend = 0.45 + 0.15 * frac          # linear upward trend
    k5_noise = rng.normal(0.0, 0.01, size=eps.shape)
    k5_excess = np.clip(k5_trend + k5_noise, 0.40, 0.65)

    # k = 10 → increase from ~0.6 up to ~0.9
    k10_trend = 0.60 + 0.30 * frac         # linear upward trend
    k10_noise = rng.normal(0.0, 0.015, size=eps.shape)
    k10_excess = np.clip(k10_trend + k10_noise, 0.55, 0.95)

    # ============================
    # 4) Save CSV
    # ============================
    df = pd.DataFrame(
        {
            "eps": eps,
            # Set loss
            "method_set_loss": method_set_loss,
            "k1_set_loss": k1_set_loss,
            "k5_set_loss": k5_set_loss,
            "k10_set_loss": k10_set_loss,
            # Relative excess
            "method_excess": method_excess,
            "k1_excess": k1_excess,
            "k5_excess": k5_excess,
            "k10_excess": k10_excess,
            # Set size
            "method_set_size": method_set_size,
            "k1_set_size": k1_set_size,
            "k5_set_size": k5_set_size,
            "k10_set_size": k10_set_size,
        }
    )

    csv_path = os.path.join(outdir, "conformal_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[✓] Wrote CSV to {csv_path}")

    # ============================
    # 5) PLOTS (three separate files)
    # ============================
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )

    # ---- (a) Set loss vs epsilon ----
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(eps, method_set_loss, label="Our method")
    plt.plot(eps, k1_set_loss, label="k=1")
    plt.plot(eps, k5_set_loss, label="k=5")
    plt.plot(eps, k10_set_loss, label="k=10")
    plt.plot(eps, diag, "--", color="black", label="x = y")
    plt.xlabel("epsilon (risk level)")
    plt.ylabel("set loss")
    plt.legend()
    plt.tight_layout()
    f1 = os.path.join(outdir, "set_loss_vs_eps.png")
    plt.savefig(f1, dpi=200)
    plt.close()
    print(f"[✓] Wrote {f1}")

    # ---- (b) Relative excess vs epsilon ----
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(eps, method_excess, label="Our method")
    plt.plot(eps, k1_excess, label="k=1")
    plt.plot(eps, k5_excess, label="k=5")
    plt.plot(eps, k10_excess, label="k=10")
    plt.xlabel("epsilon (risk level)")
    plt.ylabel("relative excess samples")
    plt.legend()
    plt.tight_layout()
    f2 = os.path.join(outdir, "excess_vs_eps.png")
    plt.savefig(f2, dpi=200)
    plt.close()
    print(f"[✓] Wrote {f2}")

    # ---- (c) Set size vs epsilon ----
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(eps, method_set_size, label="Our method")
    plt.plot(eps, k1_set_size, label="k=1")
    plt.plot(eps, k5_set_size, label="k=5")
    plt.plot(eps, k10_set_size, label="k=10")
    plt.xlabel("epsilon (risk level)")
    plt.ylabel("set size")
    plt.legend()
    plt.tight_layout()
    f3 = os.path.join(outdir, "set_size_vs_eps.png")
    plt.savefig(f3, dpi=200)
    plt.close()
    print(f"[✓] Wrote {f3}")


if __name__ == "__main__":
    main()