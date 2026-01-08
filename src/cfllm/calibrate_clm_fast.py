# cfllm/calibrate_clm_fast.py
import os, json, math, argparse, random, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .scores import METRICS  # provides distances, e.g., "numeric", "rouge"

# ----------------- utils -----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def sha_seed(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")

def default_similarity(a: str, b: str, sim_metric_name: str):
    d = METRICS[sim_metric_name](a, b)  # distance in [0,1] if rouge
    s = 1.0 - float(d)
    return max(0.0, min(1.0, s))

def binom_cdf(k, n, p):
    # P(X <= k) for X~Binom(n,p)
    if k < 0: return 0.0
    if k >= n: return 1.0
    acc = 0.0
    for i in range(0, k+1):
        acc += math.comb(n, i) * (p**i) * ((1-p)**(n-i))
    return min(max(acc, 0.0), 1.0)

def holm_stepdown(pvals, alpha):
    """
    pvals = list[(key, pvalue)], return set of keys rejected (valid configs).
    """
    m = len(pvals)
    items = sorted(pvals, key=lambda t: t[1])
    rejected = set()
    for i, (key, p) in enumerate(items, start=1):
        thr = alpha / (m - i + 1)
        if p <= thr:
            rejected.add(key)
        else:
            break
    return rejected

# ----------------- PRESAMPLING -----------------
def presample_one(ex_idx, X, Y, Xp, kmax, cap_duration, max_num_ues, base_seed):
    rnd = random.Random(base_seed + 6211*(ex_idx+1) + sha_seed(Xp) % (10**6))
    # Fix action per X'
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

    cands = []
    tried = 0
    for _ in range(kmax):
        tried += 1
        rng_run = rnd.randint(1, 10**9)
        try:
            metrics = run_ns3_action(act, rng_run=rng_run, workdir=os.path.join(OUTPUT_DIR, "tmp"))
            y = report_from_metrics(metrics, seed=0)  # deterministic textualization
            cands.append({"rng_run": rng_run, "report": y})
        except Exception:
            # skip failed attempt
            continue
    return {"idx": ex_idx, "X_prime": Xp, "num_candidates": len(cands), "candidates": cands}

def build_set_offline(candidates, lambda1, lambda3, sim_metric):
    """
    candidates: [{"rng_run":..., "report":...}, ...] in the order they were drawn
    Apply diversity (similarity threshold) and FIRST-K stopping offline.
    """
    C = []
    kept = []
    for c in candidates:
        y = c["report"]
        if any(default_similarity(y, z, sim_metric) > lambda1 for z in C):
            continue
        C.append(y)
        kept.append(c)
        if len(C) >= lambda3:
            break
    return C, kept  # kept preserves rng order/info

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="FAST CLM calibration with presampling reuse")
    ap.add_argument("--epsilon", type=float, default=0.1, help="Target failure rate (1 - desired success).")
    ap.add_argument("--delta", type=float, default=0.05, help="FWER level for Holm stepdown.")
    ap.add_argument("--cf-metric", type=str, default="numeric", choices=list(METRICS.keys()),
                    help="Distance metric to Y'_true for admissibility.")
    ap.add_argument("--cf-eps0", type=float, default=0.8, help="Admissibility threshold ε0.")
    ap.add_argument("--sim-metric", type=str, default="rouge", help="Similarity metric name (expects in METRICS).")
    ap.add_argument("--kmax", type=int, default=20, help="Max presampled candidates per example.")
    ap.add_argument("--lambda1-grid", type=str, default="0.85,0.9", help="Comma list for λ1 (similarity cap).")
    ap.add_argument("--lambda3-grid", type=str, default="5,10", help="Comma list for λ3 (FIRST-K stop).")
    ap.add_argument("--cap-duration", type=float, default=2.0, help="Cap per-candidate ns-3 duration (sec).")
    ap.add_argument("--max-num-ues", type=int, default=6, help="Cap num_ues per candidate.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for presampling.")
    ap.add_argument("--seed", type=int, default=123, help="Master seed.")
    ap.add_argument("--data-path", type=str, default=None, help="Override calib.jsonl path.")
    ap.add_argument("--presample-path", type=str, default=None, help="Where to save/load presamples JSONL.")
    ap.add_argument("--reuse-presamples", action="store_true", help="Load presamples if available; skip ns-3 runs.")
    args = ap.parse_args()

    random.seed(args.seed)
    data_path = args.data_path or os.path.join(OUTPUT_DIR, "data", "calib.jsonl")
    rows = list(read_jsonl(data_path))
    n = len(rows)
    if n == 0:
        raise RuntimeError(f"No calibration rows at {data_path}")

    presample_path = args.presample_path or os.path.join(OUTPUT_DIR, "calibration", "presamples.jsonl")
    ensure_dir(os.path.dirname(presample_path))

    # ---------- presample once ----------
    if args.reuse_presamples and os.path.exists(presample_path):
        print(f"[*] Reusing presamples from {presample_path}")
        presamples = list(read_jsonl(presample_path))
        if len(presamples) != n:
            print("[!] Presample count != dataset size; ignoring cache and re-running presamples.")
            presamples = None
    else:
        presamples = None

    if presamples is None:
        print("[*] Presampling candidates (one pass over calibration set)...")
        tasks = []
        for i, r in enumerate(rows):
            tasks.append((i, r["X"], r["Y"], r["X_prime"]))

        out_rows = [None]*n
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(presample_one, i, X, Y, Xp,
                                  args.kmax, args.cap_duration, args.max_num_ues, args.seed)
                        for (i, X, Y, Xp) in tasks]
                for fut in tqdm(as_completed(futs), total=len(tasks), desc="presampling"):
                    r = fut.result()
                    out_rows[r["idx"]] = r
        else:
            for i, X, Y, Xp in tqdm(tasks, desc="presampling"):
                r = presample_one(i, X, Y, Xp, args.kmax, args.cap_duration, args.max_num_ues, args.seed)
                out_rows[i] = r

        write_jsonl(presample_path, out_rows)
        print(f"[*] Wrote presamples to {presample_path}")
        presamples = out_rows

    # ---------- evaluate λ grid offline ----------
    lambda1s = [float(x) for x in args.lambda1_grid.split(",") if x.strip()]
    lambda3s = [int(x) for x in args.lambda3_grid.split(",") if x.strip()]
    d_cf = METRICS[args.cf_metric]
    eps0 = float(args.cf_eps0)

    results_by_lambda = {}
    pvals = []

    for l1 in lambda1s:
        for l3 in lambda3s:
            lam_key = f"lambda1={l1:.3f}|lambda3={l3}"
            L = []  # 1 if set fails to include an ε0-close candidate
            set_sizes = []
            kept_counts = []
            min_dists = []

            for i, r in enumerate(rows):
                cands = presamples[i]["candidates"]
                C, kept = build_set_offline(cands, l1, l3, args.sim_metric)
                set_sizes.append(len(C))
                kept_counts.append(len(kept))

                # compute min distance to Y'_true (for calibration risk; truth available)
                Yp_true = r["Y_prime_true"]
                if len(C) == 0:
                    min_d = float("inf")
                else:
                    min_d = min(float(d_cf(y, Yp_true)) for y in C)
                min_dists.append(min_d)
                L.append(1 if (math.isinf(min_d) or min_d > eps0) else 0)

            n_fail = sum(L)
            risk_hat = n_fail / n
            k = int(math.floor(n * risk_hat + 1e-9))
            p = binom_cdf(k, n, args.epsilon)

            finite_ds = [x for x in min_dists if math.isfinite(x)]
            mean_min_d = float(sum(finite_ds)/len(finite_ds)) if finite_ds else float("inf")

            results_by_lambda[lam_key] = {
                "lambda1": l1, "lambda3": l3,
                "risk_hat": risk_hat,
                "p_value": p,
                "coverage_at_eps0": 1.0 - risk_hat,
                "mean_set_size": float(sum(set_sizes)/n),
                "mean_samples_tried": float(sum(kept_counts)/n),  # proxies presampled attempts actually kept
                "mean_min_dist_to_truth": mean_min_d,
                "n": n,
            }
            pvals.append((lam_key, p))

    valid = holm_stepdown(pvals, alpha=args.delta)
    selected = None
    if valid:
        selected = sorted(
            (results_by_lambda[k] for k in valid),
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
        "cf_eps0": eps0,
        "sim_metric": args.sim_metric,
        "kmax": args.kmax,
        "lambda1_grid": lambda1s,
        "lambda3_grid": lambda3s,
        "cap_duration": args.cap_duration,
        "max_num_ues": args.max_num_ues,
        "workers": args.workers,
        "seed": args.seed,
        "data_path": data_path,
        "presample_path": presample_path,
        "results_by_lambda": results_by_lambda,
        "valid_lambda_keys": sorted(list(valid)),
        "selected": selected
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[✓] Wrote calibration to {out_path}")
    if selected is None:
        print("[!] No valid λ at the current (epsilon, delta). Try larger kmax, looser cf-eps0, or broader grids.")

if __name__ == "__main__":
    main()
