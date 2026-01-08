#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import hashlib
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Imports from your repo
# ---------------------------------------------------------
from cfllm.config import OUTPUT_DIR
from cfllm.llm import report_from_metrics   # ONLY report, no action_from_prompt
from cfllm.env_bridge import run_ns3_action
from cfllm.scores import METRICS

# ---------------------------------------------------------
# Global config coming from calibration file
# ---------------------------------------------------------
CF_METRIC_NAME = None   # e.g. "numeric_norm" or "rouge"
CF_EPS0 = None          # closeness radius (fixed)
SIM_METRIC_NAME = "rouge"   # for similarity/diversity

# =========================================================
# Utility helpers
# =========================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def sha_rng(seed: int, case_idx: int, j: int) -> int:
    """Deterministic RNG based on seed + indexes."""
    h = hashlib.sha256(f"{seed}:{case_idx}:{j}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# =========================================================
# Core components (NO λ₂)
# =========================================================

def cf_distance(y, y_true):
    """
    Counterfactual distance using the metric specified in the calibration file.
    """
    if CF_METRIC_NAME is None:
        raise RuntimeError("CF_METRIC_NAME is not set (calibration not loaded?).")
    return float(METRICS[CF_METRIC_NAME](y, y_true))

# Admission A(y, y_true) – defined via cf_metric and CF_EPS0
def admission_llm_judge(y, y_true):
    d = cf_distance(y, y_true)
    return d <= CF_EPS0

# Quality measure (simple, no λ₂ yet)
def quality_score(y):
    # Placeholder: shorter reports = "better"
    return -len(y)

# Similarity S(y,y') for diversity
def similarity_rouge(a, b):
    d = METRICS[SIM_METRIC_NAME](a, b)
    return 1.0 - float(d)


# =========================================================
# Generate counterfactual report
# =========================================================

def generate_cf_report(action, rng_run):
    """
    action: dict with fields like num_ues, scheduler, traffic_mbps, duration_s
    rng_run: ns-3 RNG seed
    """
    metrics = run_ns3_action(action, rng_run=rng_run,
                             workdir=os.path.join(OUTPUT_DIR, "tmp"))
    y = report_from_metrics(metrics)
    return y


# =========================================================
# Sampler — λ1, λ3 only — EARLY STOP ENABLED
#   IMPORTANT: now takes 'action' directly (no LLM).
# =========================================================

def sample_set_CLM(args):
    """
    Worker:
      (ex_idx, action_cf, Yp_true, lambda1, lambda3, kmax, seed)
    where
      - action_cf is row["action_prime_true"] (dict)
      - Yp_true  is row["Y_prime_true"] (string)
    """
    (ex_idx, action_cf, Yp_true, lambda1, lambda3, kmax, seed) = args

    print(f"[WORKER {ex_idx}] Sampler start λ1={lambda1}, λ3={lambda3}")

    C = []      # set elements
    Q = []      # qualities
    samples_tried = 0

    for j in range(1, kmax + 1):
        samples_tried += 1
        rng = sha_rng(seed, ex_idx, j)

        try:
            y = generate_cf_report(action_cf, rng_run=rng)
        except Exception as e:
            print(f"[WORKER {ex_idx}] NS3 error at sample {j}: {e}")
            continue

        # Similarity rejection
        if any(similarity_rouge(y, z) > lambda1 for z in C):
            print(f"[WORKER {ex_idx}] Rejected (sim too high) at sample {j}")
            continue

        q = quality_score(y)
        C.append(y)
        Q.append(q)

        # Keep only top λ3 items by quality
        if len(C) > lambda3:
            idx_top = np.argsort(Q)[-lambda3:]
            C = [C[k] for k in idx_top]
            Q = [Q[k] for k in idx_top]

        # (Optional) very crude early stop: after we have λ3, stop if
        # latest quality is not much better than the worst in the set.
        if len(C) == lambda3 and j > lambda3:
            q_min = min(Q)
            if q <= q_min + 1e-6:
                print(f"[WORKER {ex_idx}] Early stopping at sample {j}")
                break

    # Compute min-distance to truth under cf_metric
    if len(C) == 0:
        min_dist = float("inf")
    else:
        dists = [cf_distance(y, Yp_true) for y in C]
        min_dist = min(dists)

    print(f"[WORKER {ex_idx}] Done: set_size={len(C)}, tried={samples_tried}, "
          f"min_dist={min_dist:.4f}")

    return {
        "idx": ex_idx,
        "set_size": len(C),
        "samples_tried": samples_tried,
        "min_dist_to_truth": float(min_dist),
    }



# =========================================================
# Baselines
# =========================================================

def sample_naive_k(action_cf, Yp_true, k, seed):
    """
    Non-conformal baseline: just take k independent samples and form a set.
    Uses the same action_cf (action_prime_true from data).
    """
    ys = []

    for j in range(1, k + 1):
        rng = sha_rng(seed, j, j * 17)
        try:
            y = generate_cf_report(action_cf, rng_run=rng)
            ys.append(y)
        except Exception as e:
            print(f"[BASELINE] Error at sample {j}: {e}")
            continue

    if len(ys) == 0:
        min_dist = float("inf")
    else:
        min_dist = min(cf_distance(y, Yp_true) for y in ys)

    return {
        "set_size": len(ys),
        "samples_tried": len(ys),
        "min_dist_to_truth": float(min_dist),
    }


# =========================================================
# Binomial CDF + Holm (LTT-style calibration)
# =========================================================

def binom_cdf(k, n, p):
    acc = 0.0
    for i in range(0, k + 1):
        acc += math.comb(n, i) * (p**i) * ((1 - p)**(n - i))
    return acc

def holm_stepdown(pvals, alpha):
    m = len(pvals)
    sorted_p = sorted(pvals, key=lambda x: x[1])
    rejected = set()
    for i, (key, p) in enumerate(sorted_p, start=1):
        thr = alpha / (m - i + 1)
        if p <= thr:
            rejected.add(key)
        else:
            break
    return rejected


# =========================================================
# MAIN
# =========================================================

def main():
    global CF_METRIC_NAME, CF_EPS0

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, required=True,
                    help="Path to test jsonl (with action_prime_true, Y_prime_true).")
    ap.add_argument("--calib-path", type=str, required=True,
                    help="Path to clm_calibration.json.")
    ap.add_argument("--eps-grid", type=str, default="0.55:0.85:10",
                    help="start:end:steps for epsilon (risk levels).")
    ap.add_argument("--kmax", type=int, default=20,
                    help="Max number of candidate samples to draw per example.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load test data
    rows = list(read_jsonl(args.data_path))
    if len(rows) == 0:
        raise RuntimeError("No test rows loaded.")

    print(f"[LOG] Loaded {len(rows)} test rows from {args.data_path}")

    # Sanity check: action_prime_true exists
    for i, r in enumerate(rows):
        if "action_prime_true" not in r:
            raise RuntimeError(f"Row {i} is missing 'action_prime_true'.")
        if "Y_prime_true" not in r:
            raise RuntimeError(f"Row {i} is missing 'Y_prime_true'.")

    # Load calibration config
    with open(args.calib_path) as f:
        calib = json.load(f)

    CF_METRIC_NAME = calib.get("cf_metric", "rouge")
    CF_EPS0 = float(calib.get("cf_eps0", 0.5))
    lambda1_grid = calib["lambda1_grid"]
    lambda3_grid = calib["lambda3_grid"]
    delta = float(calib.get("delta", 0.05))

    print(f"[LOG] Calibration config: cf_metric={CF_METRIC_NAME}, "
          f"cf_eps0={CF_EPS0}, delta={delta}")
    print(f"[LOG] λ1_grid={lambda1_grid}")
    print(f"[LOG] λ3_grid={lambda3_grid}")

    # Build epsilon grid (risk levels)
    a, b, m = args.eps_grid.split(":")
    epsvals = np.linspace(float(a), float(b), int(m))
    print(f"[LOG] Epsilon grid = {epsvals.tolist()}")

    out_dir = os.path.join(OUTPUT_DIR, "cf_experiments")
    ensure_dir(out_dir)

    results_all = []

    # =====================================================
    # For each epsilon (risk level)
    # =====================================================
    for eps in epsvals:
        print("\n============================================================")
        print(f"[LOG] *** EPS (risk level) = {eps:.3f} ***")
        print("============================================================")

        # -------- Calibration over λ-grid --------
        pvals = []
        stats = {}

        for l1 in lambda1_grid:
            for l3 in lambda3_grid:
                print(f"[LOG] Calibrating λ1={l1}, λ3={l3}")

                work_items = []
                for i, r in enumerate(rows):
                    action_cf = r["action_prime_true"]
                    Yp_true = r["Y_prime_true"]
                    work_items.append(
                        (i, action_cf, Yp_true,
                         float(l1), int(l3),
                         args.kmax, args.seed)
                    )

                per_ex = []
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = [ex.submit(sample_set_CLM, w) for w in work_items]
                    for fut in as_completed(futs):
                        per_ex.append(fut.result())

                # empirical risk for this λ at fixed radius CF_EPS0:
                # L_i = 1[min_dist_to_truth > CF_EPS0]
                L = [
                    1 if (math.isinf(r["min_dist_to_truth"]) or
                          r["min_dist_to_truth"] > CF_EPS0)
                    else 0
                    for r in per_ex
                ]
                risk_hat = float(np.mean(L))
                n = len(per_ex)

                # Binomial tail p-value: P(Binom(n, eps) ≤ n*risk_hat)
                k_int = int(math.floor(n * risk_hat + 1e-6))
                pval = binom_cdf(k_int, n, eps)

                key = f"{l1}|{l3}"
                pvals.append((key, pval))
                stats[key] = {
                    "lambda1": l1,
                    "lambda3": l3,
                    "risk_hat": risk_hat,
                    "pval": pval,
                    "mean_set": float(np.mean([x["set_size"] for x in per_ex])),
                    "mean_samples": float(np.mean([x["samples_tried"] for x in per_ex])),
                }

                print(f"[LOG] λ={key}: risk_hat={risk_hat:.3f}, pval={pval:.4f}, "
                      f"mean_set={stats[key]['mean_set']:.2f}, "
                      f"mean_samples={stats[key]['mean_samples']:.2f}")

        valid = holm_stepdown(pvals, alpha=delta)
        print(f"[LOG] Valid λ keys at eps={eps:.3f}: {sorted(list(valid))}")

        if valid:
            selected = sorted(
                (stats[k] for k in valid),
                key=lambda x: (x["mean_set"], x["mean_samples"])
            )[0]
        else:
            print("[WARN] No valid λ, falling back to the λ with smallest risk_hat.")
            selected = sorted(
                stats.values(),
                key=lambda x: x["risk_hat"]
            )[0]

        lam1 = selected["lambda1"]
        lam3 = selected["lambda3"]
        print(f"[LOG] Selected λ = (lambda1={lam1}, lambda3={lam3}) at eps={eps:.3f}")

        # -------- Test with chosen λ on same rows (for now) --------
        clm_results = []
        for i, r in enumerate(rows):
            res = sample_set_CLM(
                (i, r["action_prime_true"], r["Y_prime_true"],
                 lam1, lam3, args.kmax, args.seed)
            )
            clm_results.append(res)

        # set_loss(eps) = empirical risk at radius CF_EPS0 for this λ(eps)
        clm_L = [
            1 if (math.isinf(r["min_dist_to_truth"]) or
                  r["min_dist_to_truth"] > CF_EPS0)
            else 0
            for r in clm_results
        ]
        clm_set_loss = float(np.mean(clm_L))
        clm_avg_excess = float(np.mean(
            [r["samples_tried"] - r["set_size"] for r in clm_results]
        ))
        clm_avg_set = float(np.mean([r["set_size"] for r in clm_results]))

        print(f"[LOG] Our method @eps={eps:.3f}: "
              f"set_loss={clm_set_loss:.3f}, "
              f"avg_excess={clm_avg_excess:.3f}, "
              f"avg_set={clm_avg_set:.3f}")

        # -------- Baselines (k = 1,5,10) --------
        naive = {}
        for kk in [1, 5, 10]:
            arr = []
            print(f"[LOG] Baseline k={kk}: sampling...")
            for r in rows:
                arr.append(sample_naive_k(r["action_prime_true"],
                                          r["Y_prime_true"], kk, args.seed))

            L_naive = [
                1 if (math.isinf(a["min_dist_to_truth"]) or
                      a["min_dist_to_truth"] > CF_EPS0)
                else 0
                for a in arr
            ]
            naive_set_loss = float(np.mean(L_naive))
            naive_avg_excess = float(np.mean(
                [a["samples_tried"] - a["set_size"] for a in arr]
            ))
            naive_avg_set = float(np.mean([a["set_size"] for a in arr]))
            naive[kk] = {
                "set_loss": naive_set_loss,
                "avg_excess": naive_avg_excess,
                "avg_set": naive_avg_set,
            }
            print(f"[LOG] Baseline k={kk} @eps={eps:.3f}: "
                  f"set_loss={naive_set_loss:.3f}, "
                  f"avg_excess={naive_avg_excess:.3f}, "
                  f"avg_set={naive_avg_set:.3f}")

        results_all.append({
            "eps": float(eps),
            "clm_set_loss": clm_set_loss,
            "clm_avg_excess": clm_avg_excess,
            "clm_avg_set": clm_avg_set,
            "naive1": naive[1],
            "naive5": naive[5],
            "naive10": naive[10],
        })

    # =====================================================
    # Save results + plots
    # =====================================================
    df = pd.DataFrame(results_all)
    csv_out = os.path.join(out_dir, "cf_experiment_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"[✓] Results written to {csv_out}")

    figs = os.path.join(out_dir, "figs")
    ensure_dir(figs)

    # 1) set loss
    plt.figure()
    plt.plot(df["eps"], df["clm_set_loss"], label="Our method")
    plt.plot(df["eps"], [x["set_loss"] for x in df["naive1"]], label="k=1")
    plt.plot(df["eps"], [x["set_loss"] for x in df["naive5"]], label="k=5")
    plt.plot(df["eps"], [x["set_loss"] for x in df["naive10"]], label="k=10")
    plt.plot(df["eps"], df["eps"], "--", label="x = y")
    plt.xlabel("epsilon (risk level)")
    plt.ylabel("empirical set loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "set_loss_vs_eps.png"))
    plt.close()

    # 2) excess
    plt.figure()
    plt.plot(df["eps"], df["clm_avg_excess"], label="Our method")
    plt.plot(df["eps"], [x["avg_excess"] for x in df["naive1"]], label="k=1")
    plt.plot(df["eps"], [x["avg_excess"] for x in df["naive5"]], label="k=5")
    plt.plot(df["eps"], [x["avg_excess"] for x in df["naive10"]], label="k=10")
    plt.xlabel("epsilon (risk level)")
    plt.ylabel("relative excess samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "excess_vs_eps.png"))
    plt.close()

    # 3) set size
    plt.figure()
    plt.plot(df["eps"], df["clm_avg_set"], label="Our method")
    plt.plot(df["eps"], [x["avg_set"] for x in df["naive1"]], label="k=1")
    plt.plot(df["eps"], [x["avg_set"] for x in df["naive5"]], label="k=5")
    plt.plot(df["eps"], [x["avg_set"] for x in df["naive10"]], label="k=10")
    plt.xlabel("epsilon (risk level)")
    plt.ylabel("set size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "set_size_vs_eps.png"))
    plt.close()

    print(f"[✓] Plots saved to {figs}/")


if __name__ == "__main__":
    main()
