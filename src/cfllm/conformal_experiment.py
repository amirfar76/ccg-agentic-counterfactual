"""
Conformal language-model style experiment for CFLLM.

This script:
- loads the calibration and test JSONL files produced by `data_gen_calib.py`;
- for each target risk level epsilon, calibrates a CLM-style sampling configuration
  using Learn-Then-Test with a Binomial tail p-value and Bonferroni FWER control;
- evaluates the calibrated configuration on a held-out test set;
- compares against non-conformal baselines that always sample a fixed number of
  candidates (k = 1, 5, 10);
- produces a 3-panel matplotlib figure with:
    (i)   set loss vs epsilon,
    (ii)  relative excess samples vs epsilon,
    (iii) set size vs epsilon.

The implementation is deliberately self-contained and only depends on:
  - numpy
  - matplotlib
  - `scores.distance_rouge_l` for a simple text similarity
It does *not* call any external LLM APIs; instead it uses a simple heuristic quality
function based on ROUGE-L similarity to the ground-truth report.  This is meant to
mirror the CLM setup while remaining runnable without API keys.  You can plug in
your own LLM-based admission and quality functions by modifying `admission_fn`
and `quality_fn` below.

Run for example as

  python -m cfllm.conformal_experiment \\
     --n-calib 200 --n-test 200 \\
     --alpha 0.1 --metric rouge --seed 123

Assuming you have already created the corresponding calibration/test files with

  python -m cfllm.data_gen_calib --n-calib 200 --n-test 200 --alpha 0.1 --metric rouge --seed 123
"""
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR
from .scores import distance_rouge_l

# ---------------------------------------------------------------------------
# Utility: load calibration / test JSONL in the same format as data_gen_calib
# ---------------------------------------------------------------------------

def _build_paths(n_calib: int, n_test: int, alpha: float, metric: str, seed: int) -> Tuple[str, str, str]:
    """Return the calib / test / meta paths used in `data_gen_calib.py`.

    In the current repo version, `data_gen_calib.py` saves files to
    OUTPUT_DIR / "data" / {calib.jsonl, test.jsonl, meta.json}, independent
    of the specific values of (n_calib, n_test, alpha, metric, seed).
    We keep the arguments for interface compatibility, but only use the
    shared directory structure here.
    """
    data_dir = os.path.join(OUTPUT_DIR, "data")
    calib_path = os.path.join(data_dir, "calib.jsonl")
    test_path  = os.path.join(data_dir, "test.jsonl")
    meta_path  = os.path.join(data_dir, "meta.json")
    return calib_path, test_path, meta_path


def _load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

# ---------------------------------------------------------------------------
# CLM-style sampling on top of *pre-existing* candidate generations.
# To keep this script runnable without calling an LLM, we synthesize a
# small candidate pool for each example directly from the ground-truth
# counterfactual report by simple perturbations.  On your side, you can
# replace `generate_candidates` with calls to your actual LLM.
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    text: str
    quality: float


def text_similarity(a: str, b: str) -> float:
    """1 - distance_rouge_l: a crude similarity in [0, 1]."""
    return 1.0 - float(distance_rouge_l(a, b))


def admission_fn(y_true: str, y_hat: str, tau: float = 0.7) -> int:
    """
    Admission function A.

    Conceptually this should be implemented via LLM-as-a-judge comparing
    `y_hat` to `y_true`.  As a stand-in that keeps the code runnable, we
    use ROUGE-L similarity and threshold it at `tau`.
    """
    sim = text_similarity(y_true, y_hat)
    return int(sim >= tau)


def quality_fn(y_true: str, y_hat: str) -> float:
    """
    Quality function Q(x, y).

    In the CLM paper this is based on (length-normalised) log-likelihood.
    Here we again use ROUGE-L similarity to the ground-truth counterfactual
    as a proxy, so that higher similarity = higher "quality".
    """
    return text_similarity(y_true, y_hat)


def generate_candidates(y_true: str, kmax: int, rng: random.Random) -> List[Candidate]:
    """
    Generate a pool of kmax synthetic candidate reports for a given ground-truth
    report y_true.  This is a *placeholder* for real LLM sampling.

    We simply:
      - with some probability, return y_true itself;
      - otherwise return y_true with a short prefix/suffix truncated,
        which lowers the similarity score.
    """
    words = y_true.split()
    n = len(words)
    candidates: List[Candidate] = []

    for _ in range(kmax):
        if n <= 5 or rng.random() < 0.2:
            # Exact report
            cand_text = y_true
        else:
            # Randomly drop a prefix/suffix segment
            drop_prefix = rng.randint(0, max(0, n // 4))
            drop_suffix = rng.randint(0, max(0, n // 4))
            start = drop_prefix
            end = max(start + 1, n - drop_suffix)
            cand_text = " ".join(words[start:end])
        q = quality_fn(y_true, cand_text)
        candidates.append(Candidate(text=cand_text, quality=q))

    return candidates

# ---------------------------------------------------------------------------
# CLM-style sampling with rejection (Algorithm 1 in Quach et al.)
# Our configuration is lambda = (lambda1, lambda2, lambda3) where
#   - lambda1: max allowed similarity between two samples in the set (diversity)
#   - lambda2: minimum quality for a sample to enter the set
#   - lambda3: stopping threshold on F(C) = max_y in C Q(y)         (MAX rule)
# ---------------------------------------------------------------------------

@dataclass
class LambdaConfig:
    lam_sim: float   # lambda1
    lam_qual: float  # lambda2
    lam_stop: float  # lambda3


@dataclass
class SamplingStats:
    loss: int              # 1 if no admissible candidate in set, else 0
    num_samples: int       # total number of draws (<= kmax)
    oracle_index: int      # smallest j with admissible candidate among the pool (1-based, or kmax+1 if none)
    set_size: int          # |C_lambda|


def clm_sample(
    y_true: str,
    candidates: List[Candidate],
    lam: LambdaConfig,
    kmax: int,
) -> SamplingStats:
    """
    Run CLM-style sampling with rejection on a *fixed* ordered candidate list.

    We treat the candidate list as y_1, ..., y_kmax (already sampled).
    """
    C: List[Candidate] = []
    used_indices: List[int] = []
    max_quality_in_set = -1e9

    # Oracle index: first candidate in the *pool* that is admissible
    oracle_index = kmax + 1
    for j, cand in enumerate(candidates, start=1):
        if oracle_index == kmax + 1:
            if admission_fn(y_true, cand.text):
                oracle_index = j

    # Now perform CLM sampling following Algorithm 1 but using the pre-sampled pool
    for j, cand in enumerate(candidates, start=1):
        # Diversity rejection: reject if too similar to any element already in C
        too_similar = False
        for existing in C:
            if text_similarity(existing.text, cand.text) > lam.lam_sim:
                too_similar = True
                break
        if too_similar:
            continue

        # Quality rejection
        if cand.quality < lam.lam_qual:
            continue

        # Accept into set
        C.append(cand)
        used_indices.append(j)
        if cand.quality > max_quality_in_set:
            max_quality_in_set = cand.quality

        # Stopping rule: F(C) = max quality >= lambda3
        if max_quality_in_set >= lam.lam_stop:
            break

        if len(used_indices) >= kmax:
            break

    # If we never accepted anything, max_quality_in_set stays very low;
    # in that case we simply stop after scanning kmax candidates.
    num_samples = used_indices[-1] if used_indices else min(kmax, len(candidates))

    # Loss: did we miss all admissible candidates?
    loss = 1
    for c in C:
        if admission_fn(y_true, c.text):
            loss = 0
            break

    return SamplingStats(
        loss=loss,
        num_samples=num_samples,
        oracle_index=oracle_index,
        set_size=len(C),
    )

# ---------------------------------------------------------------------------
# Binomial tail p-value and simple Bonferroni FWER control
# ---------------------------------------------------------------------------

def binom_tail_pvalue(n: int, eps: float, r_hat: float) -> float:
    """
    p = P(Binom(n, eps) <= n * r_hat).
    We compute this exactly using the binomial PMF.
    """
    k = int(math.floor(n * r_hat))
    p = 0.0
    for j in range(0, k + 1):
        p += math.comb(n, j) * (eps ** j) * ((1.0 - eps) ** (n - j))
    return p


def calibrate_lambda_grid(
    y_trues: List[str],
    kmax: int,
    eps: float,
    delta: float,
    rng: random.Random,
    lam_grid: List[LambdaConfig],
) -> LambdaConfig:
    """
    Perform LTT over a finite grid of lambda configurations with simple
    Bonferroni FWER control at level `delta`.
    """
    n = len(y_trues)
    assert n > 0

    # Pre-generate candidate pools for all calibration points
    all_pools: List[List[Candidate]] = [
        generate_candidates(y_true, kmax, rng) for y_true in y_trues
    ]

    m = len(lam_grid)
    alpha_per_hyp = delta / m

    valid_configs: List[Tuple[LambdaConfig, float, float, float]] = []  # (lambda, risk_hat, avg_set_size, avg_rel_excess)

    for lam in lam_grid:
        losses = []
        set_sizes = []
        rel_excesses = []

        for y_true, pool in zip(y_trues, all_pools):
            stats = clm_sample(y_true, pool, lam, kmax)
            losses.append(stats.loss)
            set_sizes.append(stats.set_size)

            if stats.oracle_index <= kmax:
                excess = max(0, stats.num_samples - stats.oracle_index)
                rel_excess = excess / float(stats.oracle_index)
            else:
                # No admissible candidate in the pool at all; by convention we
                # treat the relative excess as zero.
                rel_excesses.append(0.0)
                continue
            rel_excesses.append(rel_excess)

        R_hat = float(np.mean(losses))
        pval = binom_tail_pvalue(n, eps, R_hat)

        if pval <= alpha_per_hyp:
            avg_size = float(np.mean(set_sizes))
            avg_rel_excess = float(np.mean(rel_excesses))
            valid_configs.append((lam, R_hat, avg_size, avg_rel_excess))

    if not valid_configs:
        # If nothing is valid, fall back to a very conservative configuration:
        # no diversity filter, no quality threshold, and very high stop threshold.
        return LambdaConfig(lam_sim=1.0, lam_qual=-1.0, lam_stop=1.0 + 1e-6)

    # Choose the config that minimises a weighted combination of set size and relative excess
    best = None
    best_score = None
    rho1 = 0.5
    rho2 = 0.5
    for lam, _, avg_size, avg_rel_excess in valid_configs:
        score = rho1 * avg_size + rho2 * avg_rel_excess
        if best_score is None or score < best_score:
            best_score = score
            best = lam

    assert best is not None
    return best

# ---------------------------------------------------------------------------
# Non-conformal baselines: fixed-k sampling (no calibration)
# ---------------------------------------------------------------------------

def fixed_k_sampling(
    y_true: str,
    candidates: List[Candidate],
    k: int,
) -> SamplingStats:
    """
    Simple non-conformal baseline:
      - take the first k candidates from the pool,
      - do not apply any diversity/quality/stopping rule.
    """
    k_eff = min(k, len(candidates))
    C = candidates[:k_eff]

    # Loss
    loss = 1
    for c in C:
        if admission_fn(y_true, c.text):
            loss = 0
            break

    # Oracle index
    oracle_index = len(candidates) + 1
    for j, cand in enumerate(candidates, start=1):
        if admission_fn(y_true, cand.text):
            oracle_index = j
            break

    return SamplingStats(
        loss=loss,
        num_samples=k_eff,
        oracle_index=oracle_index,
        set_size=len(C),
    )

# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------

def run_experiment(
    n_calib: int,
    n_test: int,
    alpha: float,
    metric: str,
    seed: int,
    kmax: int,
    n_eps: int,
    eps_min: float,
    eps_max: float,
    out_dir: str,
) -> str:
    """
    Run the conformal vs fixed-k comparison and save the resulting figure.

    Returns
    -------
    fig_path : str
        Path to the saved PNG figure.
    """
    rng = random.Random(seed)

    calib_path, test_path, meta_path = _build_paths(n_calib, n_test, alpha, metric, seed)
    if not os.path.exists(calib_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find calibration/test files.\n"
            f"Expected:\n  {calib_path}\n  {test_path}\n"
            f"Please generate them first with `python -m cfllm.data_gen_calib`."
        )

    calib_data = _load_jsonl(calib_path)
    test_data = _load_jsonl(test_path)

    # Extract ground-truth counterfactual reports
    calib_y_true = [d["Y_prime_true"] for d in calib_data]
    test_y_true  = [d["Y_prime_true"] for d in test_data]

    # Epsilon grid
    epsilons = np.linspace(eps_min, eps_max, n_eps)

    # Containers for results
    methods = ["conformal", "fixed_k1", "fixed_k5", "fixed_k10"]
    loss_curves    = {m: [] for m in methods}
    excess_curves  = {m: [] for m in methods}
    size_curves    = {m: [] for m in methods}

    delta = 0.05  # FWER level

    # Pre-generate candidate pools for *test* points (to reuse across epsilons)
    test_pools = [generate_candidates(y_true, kmax, rng) for y_true in test_y_true]

    # Fixed-k baselines do not depend on epsilon, so we can pre-compute them once
    baseline_stats = {m: [] for m in ["fixed_k1", "fixed_k5", "fixed_k10"]}
    k_map = {"fixed_k1": 1, "fixed_k5": 5, "fixed_k10": 10}
    for method, k in k_map.items():
        losses = []
        rel_excesses = []
        sizes = []
        for y_true, pool in zip(test_y_true, test_pools):
            stats = fixed_k_sampling(y_true, pool, k=min(k, kmax))
            losses.append(stats.loss)
            sizes.append(stats.set_size)

            if stats.oracle_index <= len(pool) and stats.oracle_index <= stats.num_samples:
                excess = max(0, stats.num_samples - stats.oracle_index)
                rel_excess = excess / float(max(1, stats.oracle_index))
            else:
                rel_excess = 0.0
            rel_excesses.append(rel_excess)
        baseline_stats[method] = (
            float(np.mean(losses)),
            float(np.mean(rel_excesses)),
            float(np.mean(sizes)),
        )

    # Fill baseline curves (horizontal lines w.r.t epsilon)
    for eps in epsilons:
        for method in ["fixed_k1", "fixed_k5", "fixed_k10"]:
            l, ex, sz = baseline_stats[method]
            loss_curves[method].append(l)
            excess_curves[method].append(ex)
            size_curves[method].append(sz)

    # Lambda grid for conformal method
    sim_grid  = [0.6, 0.8, 0.9]              # diversity thresholds
    qual_grid = [0.0, 0.3, 0.5, 0.7]         # min quality
    stop_grid = [0.5, 0.7, 0.8, 0.9, 0.95]   # stopping MAX(Q) threshold

    lam_grid: List[LambdaConfig] = [
        LambdaConfig(lam_sim=s, lam_qual=q, lam_stop=t)
        for s in sim_grid
        for q in qual_grid
        for t in stop_grid
    ]

    # For each epsilon, calibrate lambda on the calibration set and evaluate on the test set
    for eps in epsilons:
        print(f"Calibrating for epsilon = {eps:.3f} ...")
        lam_star = calibrate_lambda_grid(
            y_trues=calib_y_true,
            kmax=kmax,
            eps=float(eps),
            delta=delta,
            rng=rng,
            lam_grid=lam_grid,
        )

        # Evaluate on test set
        losses = []
        rel_excesses = []
        sizes = []
        for y_true, pool in zip(test_y_true, test_pools):
            stats = clm_sample(y_true, pool, lam_star, kmax)
            losses.append(stats.loss)
            sizes.append(stats.set_size)
            if stats.oracle_index <= kmax and stats.oracle_index <= stats.num_samples:
                excess = max(0, stats.num_samples - stats.oracle_index)
                rel_excess = excess / float(max(1, stats.oracle_index))
            else:
                rel_excess = 0.0
            rel_excesses.append(rel_excess)

        loss_curves["conformal"].append(float(np.mean(losses)))
        excess_curves["conformal"].append(float(np.mean(rel_excesses)))
        size_curves["conformal"].append(float(np.mean(sizes)))

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Set loss
    ax = axes[0]
    for m, label in [
        ("conformal", "Conformal (CLM-style)"),
        ("fixed_k1", "Fixed k = 1"),
        ("fixed_k5", "Fixed k = 5"),
        ("fixed_k10", "Fixed k = 10"),
    ]:
        ax.plot(epsilons, loss_curves[m], marker="o", label=label)
    ax.plot(epsilons, epsilons, linestyle="--", label="y = epsilon")
    ax.set_xlabel(r"Target risk $\epsilon$")
    ax.set_ylabel("Set loss")
    ax.set_title("Set loss vs risk level")
    ax.legend()

    # 2) Relative excess samples
    ax = axes[1]
    for m, label in [
        ("conformal", "Conformal (CLM-style)"),
        ("fixed_k1", "Fixed k = 1"),
        ("fixed_k5", "Fixed k = 5"),
        ("fixed_k10", "Fixed k = 10"),
    ]:
        ax.plot(epsilons, excess_curves[m], marker="o", label=label)
    ax.set_xlabel(r"Target risk $\epsilon$")
    ax.set_ylabel("Relative excess samples")
    ax.set_title("Relative excess vs risk level")
    ax.legend()

    # 3) Set size
    ax = axes[2]
    for m, label in [
        ("conformal", "Conformal (CLM-style)"),
        ("fixed_k1", "Fixed k = 1"),
        ("fixed_k5", "Fixed k = 5"),
        ("fixed_k10", "Fixed k = 10"),
    ]:
        ax.plot(epsilons, size_curves[m], marker="o", label=label)
    ax.set_xlabel(r"Target risk $\epsilon$")
    ax.set_ylabel("Set size")
    ax.set_title("Set size vs risk level")
    ax.legend()

    fig.tight_layout()

    fig_path = os.path.join(out_dir, f"conformal_experiment_ncalib{n_calib}_ntest{n_test}_seed{seed}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return fig_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLM-style conformal experiment for CFLLM.")
    parser.add_argument("--n-calib", type=int, required=True, help="Number of calibration examples (must match data_gen_calib).")
    parser.add_argument("--n-test", type=int, required=True, help="Number of test examples (must match data_gen_calib).")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha used when generating the data (for file naming).")
    parser.add_argument("--metric", type=str, default="rouge", help="Metric string used in data_gen_calib (for file naming).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--kmax", type=int, default=10, help="Maximum number of samples per example.")
    parser.add_argument("--n-eps", type=int, default=8, help="Number of epsilon values in the grid.")
    parser.add_argument("--eps-min", type=float, default=0.05, help="Minimum epsilon.")
    parser.add_argument("--eps-max", type=float, default=0.5, help="Maximum epsilon.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for the figure (defaults to OUTPUT_DIR).")

    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(OUTPUT_DIR, "conformal_plots")

    fig_path = run_experiment(
        n_calib=args.n_calib,
        n_test=args.n_test,
        alpha=args.alpha,
        metric=args.metric,
        seed=args.seed,
        kmax=args.kmax,
        n_eps=args.n_eps,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        out_dir=out_dir,
    )
    print(f"Saved conformal experiment figure to: {fig_path}")


if __name__ == "__main__":
    main()

