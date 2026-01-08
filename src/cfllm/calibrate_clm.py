# cfllm/calibrate_clm.py
# Calibrate CLM-style sampling-with-rejection for counterfactuals (LTT with Binomial tail p-values + Holm).
import os, json, math, argparse, random, hashlib
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .scores import METRICS  # expect "numeric" and "rouge" distances defined here

# ---------- helpers ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sha_seed(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")

def binom_cdf(k, n, p):
    # P(X <= k), X~Binom(n,p)
    if k < 0: return 0.0
    if k >= n: return 1.0
    logp = 0.0
    acc = 0.0
    # Use log-sum for stability if n large; here n is moderate, direct sum is fine.
    for i in range(0, k+1):
        coeff = math.comb(n, i)
        acc += coeff * (p ** i) * ((1 - p) ** (n - i))
    return min(max(acc, 0.0), 1.0)

def holm_stepdown(pvals, alpha):
    """
    pvals: list of (key, p)
    Returns: set of keys whose H0 is rejected (i.e., valid configs).
    """
    m = len(pvals)
    sorted_items = sorted(pvals, key=lambda t: t[1])  # ascending by p
    rejected = set()
    for i, (key, p) in enumerate(sorted_items, start=1):
        thresh = alpha / (m - i + 1)
        if p <= thresh:
            rejected.add(key)
        else:
            break
    return rejected

def default_similarity(a: str, b: str, sim_metric_name: str, dist_metrics):
    # For ROUGE: if METRICS['rouge'] is 1 - ROUGE-L F1 (distance), similarity = 1 - distance
    if sim_metric_name not in dist_metrics:
        raise ValueError(f"Unknown similarity metric '{sim_metric_name}'. Expect one in METRICS (e.g., 'rouge').")
    d = dist_metrics[sim_metric_name](a, b)
    s = 1.0 - float(d)
    # clamp
    return max(0.0, min(1.0, s))

# ---------- sampler (single example) ----------
def sample_set_CLM_on_example(args):
    """
    Worker target to allow parallelization across calibration examples for a fixed lambda.
    Inputs packed in a dict to be pickle-friendly.
    """
    (ex_idx, X, Y, Xp, Yp_true, lambda1, lambda2, lambda3, kmax,
     sim_metric, cf_metric, cap_duration, max_num_ues,
     base_seed) = args

    # pull metrics
    dist_map = METRICS
    d_cf = dist_map[cf_metric]
    # quality OFF by default (lambda2 unused). If you later add quality, plug here.

    # randomness per example
    rnd = random.Random(base_seed + 9973 * (ex_idx + 1) + sha_seed(Xp) % (10**6))

    # Build / reuse action once per X'
    act = action_from_prompt(Xp, seed=rnd.randint(1, 2**31 - 1))
    if cap_duration is not None:
        try:
            act["duration_s"] = float(min(float(act.get("duration_s", 10.0)), float(cap_duration)))
        except Exception:
            act["duration_s"] = float(cap_duration)
    if max_num_ues is not None:
        try:
            act["num_ues"] = int(min(int(act.get("num_ues", 5)), int(max_num_ues)))
        except Exception:
            pass

    C = []
    samples_tried = 0
    # sampling loop
    for _ in range(kmax):
        samples_tried += 1
        rng_run = rnd.randint(1, 10**9)
        try:
            metrics = run_ns3_action(act, rng_run=rng_run, workdir=os.path.join(OUTPUT_DIR, "tmp"))
            y = report_from_metrics(metrics, seed=0)  # deterministic textualization
        except Exception as e:
            # Skip this attempt; continue sampling.
            continue

        # (optional) quality filter could go here if you later add Q(x,y)
        # similarity filter (diversity): reject if too similar to any current set element
        too_similar = any(default_similarity(y, z, sim_metric, dist_map) > lambda1 for z in C)
        if too_similar:
            continue

        C.append(y)
        if len(C) >= lambda3:
            break

    # Evaluate closeness to true counterfactual (for calibration risk only)
    if len(C) == 0:
        min_dist_to_truth = float("inf")
    else:
        min_dist_to_truth = min(d_cf(y, Yp_true) for y in C)

    return {
        "idx": ex_idx,
        "set_size": len(C),
        "samples_tried": samples_tried,
        "min_dist_to_truth": float(min_dist_to_truth),
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="CLM-style calibration for counterfactuals (LTT).")
    ap.add_argument("--epsilon", type=float, default=0.1, help="Target failure rate (1 - desired coverage).")
    ap.add_argument("--delta", type=float, default=0.05, help="FWER level for Holm stepdown.")
    ap.add_argument("--cf-metric", type=str, default="numeric", choices=list(METRICS.keys()),
                    help="Distance metric used for admissibility to Y'_true (e.g., 'numeric' or 'rouge').")
    ap.add_argument("--cf-eps0", type=float, default=0.8, help="Closeness threshold ε0 for admissibility: d_cf(y,Y'_true) ≤ ε0.")
    ap.add_argument("--sim-metric", type=str, default="rouge", help="Similarity metric for diversity filter (expects in METRICS, e.g., 'rouge').")
    ap.add_argument("--kmax", type=int, default=20, help="Max samples drawn per example.")
    ap.add_argument("--lambda1-grid", type=str, default="0.8,0.9", help="Comma list for similarity cap λ1 (higher=more diverse).")
    ap.add_argument("--lambda3-grid", type=str, default="3,5,10", help="Comma list for set size target λ3 (FIRST-K).")
    ap.add_argument("--cap-duration", type=float, default=3.0, help="Cap per-candidate NS-3 duration (seconds) for speed.")
    ap.add_argument("--max-num-ues", type=int, default=8, help="Cap num_ues per-candidate for speed.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for calibration examples.")
    ap.add_argument("--seed", type=int, default=123, help="Master seed.")
    ap.add_argument("--data-path", type=str, default=None, help="Override calib.jsonl path")
    args = ap.parse_args()

    random.seed(args.seed)

    data_path = args.data_path or os.path.join(OUTPUT_DIR, "data", "calib.jsonl")
    rows = list(read_jsonl(data_path))
    n = len(rows)
    if n == 0:
        raise RuntimeError(f"No calibration rows found at {data_path}")

    lambda1s = [float(x) for x in args.lambda1_grid.split(",") if x.strip()]
    lambda3s = [int(x) for x in args.lambda3_grid.split(",") if x.strip()]

    # Pre-extract for speed
    dataset = []
    for i, r in enumerate(rows):
        dataset.append((i, r["X"], r["Y"], r["X_prime"], r["Y_prime_true"]))

    results_by_lambda = {}
    pvals = []

    for l1 in lambda1s:
        for l3 in lambda3s:
            lam_key = f"lambda1={l1:.3f}|lambda3={l3}"
            # run sampler on each example
            work_items = []
            for (i, X, Y, Xp, Yp_true) in dataset:
                work_items.append((
                    i, X, Y, Xp, Yp_true,
                    l1, float("-inf"), l3, args.kmax,
                    args.sim_metric, args.cf_metric,
                    args.cap_duration, args.max_num_ues,
                    args.seed
                ))

            per_ex = []
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = [ex.submit(sample_set_CLM_on_example, w) for w in work_items]
                    for fut in tqdm(as_completed(futs), total=len(work_items), desc=f"calib λ1={l1} λ3={l3}"):
                        per_ex.append(fut.result())
            else:
                for w in tqdm(work_items, desc=f"calib λ1={l1} λ3={l3}"):
                    per_ex.append(sample_set_CLM_on_example(w))

            # empirical risk and stats
            eps0 = args.cf_eps0
            L = [1 if (math.isinf(r["min_dist_to_truth"]) or r["min_dist_to_truth"] > eps0) else 0 for r in per_ex]
            risk_hat = sum(L) / n
            # Binomial tail p-value: P(Binom(n, epsilon) ≤ n*risk_hat)
            k = int(math.floor(n * risk_hat + 1e-9))
            p = binom_cdf(k, n, args.epsilon)

            mean_set = sum(r["set_size"] for r in per_ex) / n
            mean_samples = sum(r["samples_tried"] for r in per_ex) / n
            # Optional diagnostic: average min distance to truth
            finite_d = [r["min_dist_to_truth"] for r in per_ex if math.isfinite(r["min_dist_to_truth"])]
            mean_min_d = float(sum(finite_d)/len(finite_d)) if finite_d else float("inf")
            cov_eps0 = 1.0 - risk_hat

            results_by_lambda[lam_key] = {
                "lambda1": l1, "lambda3": l3,
                "risk_hat": risk_hat,
                "p_value": p,
                "mean_set_size": mean_set,
                "mean_samples_tried": mean_samples,
                "mean_min_dist_to_truth": mean_min_d,
                "coverage_at_eps0": cov_eps0,
                "n": n,
            }
            pvals.append((lam_key, p))

    # Holm stepdown to find Λ_valid (reject H0: risk > epsilon)
    valid_keys = holm_stepdown(pvals, alpha=args.delta)

    # pick λ^ among valid that minimizes (mean_set_size, mean_samples_tried)
    selected = None
    if valid_keys:
        selected = sorted(
            (results_by_lambda[k] for k in valid_keys),
            key=lambda z: (z["mean_set_size"], z["mean_samples_tried"])
        )[0]

    out_dir = os.path.join(OUTPUT_DIR, "calibration")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "clm_calibration.json")

    payload = {
        "mode": "clm",
        "epsilon": args.epsilon,
        "delta": args.delta,
        "cf_metric": args.cf_metric,
        "cf_eps0": args.cf_eps0,
        "sim_metric": args.sim_metric,
        "kmax": args.kmax,
        "lambda1_grid": lambda1s,
        "lambda3_grid": lambda3s,
        "cap_duration": args.cap_duration,
        "max_num_ues": args.max_num_ues,
        "workers": args.workers,
        "seed": args.seed,
        "data_path": data_path,
        "results_by_lambda": results_by_lambda,
        "valid_lambda_keys": sorted(list(valid_keys)),
        "selected": selected,  # may be None if no valid configs
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote calibration to {out_path}")
    if selected is None:
        print("WARNING: No valid configuration at specified (epsilon, delta). Consider larger kmax, looser eps0, or richer grids.")

if __name__ == "__main__":
    main()
