# cfllm/retest_from_presamples.py
import os, json, argparse, math
from collections import defaultdict

from cfllm.config import OUTPUT_DIR
from cfllm.scores import METRICS

TEST_DIR = os.path.join(OUTPUT_DIR, "test_clm")
DATA_TEST = os.path.join(OUTPUT_DIR, "data", "test.jsonl")
PRESAMPLES = os.path.join(TEST_DIR, "presamples.jsonl")
CAL_PATH = os.path.join(OUTPUT_DIR, "calibration", "clm_calibration.json")

def _load_presamples():
    rows = [json.loads(l) for l in open(PRESAMPLES)]
    return {r["idx"]: r["candidates"] for r in rows}

def _similarity(sim_metric_name: str, a: str, b: str) -> float:
    # similarity = 1 - distance (assumes distance scaled to [0,1])
    d = float(METRICS[sim_metric_name](a, b))
    d = min(max(d, 0.0), 1.0)
    return 1.0 - d

def _build_set_from_candidates(cands, lambda1: float, lambda3: int, sim_metric: str):
    C = []
    tried = 0
    for cand in cands:
        yk = cand.get("report") or cand.get("text") or ""
        if not yk.strip():
            continue
        tried += 1
        if C:
            smax = max(_similarity(sim_metric, yk, ci.get("report") or ci.get("text") or "") for ci in C)
            if smax > lambda1:
                continue
        C.append({"report": yk})
        if len(C) >= lambda3:
            break
    return C, tried

def _ecdf(vals):
    xs = sorted(vals)
    ys = [ (i+1)/len(xs) for i in range(len(xs)) ]
    return xs, ys

def main():
    ap = argparse.ArgumentParser(description="Instant test from presamples; rebuild sets & evaluate.")
    ap.add_argument("--dump-sets", action="store_true", help="Write per-case sets under outputs/test_clm/sets/")
    ap.add_argument("--cache-sets", action="store_true", help="Cache sets under sets_cached/<lambda_key>/")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # Load calibration (selected lambda and metrics)
    cal = json.load(open(CAL_PATH))
    sel = cal.get("selected")
    if sel is None:
        raise RuntimeError("No selected λ in clm_calibration.json. Run calibrate_clm_fast first.")
    lambda1 = float(sel["lambda1"])
    lambda3 = int(sel["lambda3"])
    sim_metric = cal.get("sim_metric", "rouge")
    cf_metric = cal.get("cf_metric", "rouge")
    eps0 = float(cal.get("cf_eps0", 0.5))
    epsilon = float(cal.get("epsilon", 0.1))

    # Load data & presamples
    test_rows = [json.loads(l) for l in open(DATA_TEST)]
    pres = _load_presamples()

    # Optional set output dirs
    os.makedirs(TEST_DIR, exist_ok=True)
    if args.dump-sets:
        sets_out = os.path.join(TEST_DIR, "sets"); os.makedirs(sets_out, exist_ok=True)
    else:
        sets_out = None

    # Evaluate
    d_cf = METRICS[cf_metric]
    min_dists = []
    set_sizes = []
    samples_tried = []
    misses = 0

    for i, rec in enumerate(test_rows):
        ytrue = rec["Y_prime_true"]
        cands = pres.get(i, [])
        C, tried = _build_set_from_candidates(cands, lambda1, lambda3, sim_metric)

        if sets_out:
            with open(os.path.join(sets_out, f"case_{i}.json"), "w") as f:
                json.dump({"idx": i, "kept": C}, f, indent=2)

        set_sizes.append(len(C))
        samples_tried.append(tried)

        if not C:
            min_dists.append(float("inf"))
            misses += 1
            continue

        ds = [ float(d_cf(ci["report"], ytrue)) for ci in C ]
        m = min(ds)
        min_dists.append(m)
        if not (m <= eps0): misses += 1

    cov = 1.0 - misses / max(1, len(test_rows))

    # Save summary
    summary = {
        "n": len(test_rows),
        "epsilon": epsilon,
        "cf_metric": cf_metric,
        "cf_eps0": eps0,
        "sim_metric": sim_metric,
        "lambda1": lambda1,
        "lambda3": lambda3,
        "coverage_empirical": cov,
        "avg_set_size": sum(set_sizes)/len(set_sizes) if set_sizes else 0.0,
        "avg_samples_tried": sum(samples_tried)/len(samples_tried) if samples_tried else 0.0,
        "median_min_dist": float("nan") if not min_dists else float(sorted([x for x in min_dists if x==x])[len(min_dists)//2]),
    }
    with open(os.path.join(TEST_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Plots (optional if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        figs_dir = os.path.join(TEST_DIR, "figs"); os.makedirs(figs_dir, exist_ok=True)
        # Histogram of min distances
        finite = [v for v in min_dists if math.isfinite(v)]
        plt.figure()
        plt.hist(finite, bins=20)
        plt.axvline(eps0, linestyle="--")
        plt.title("Min distance to Y'_true")
        plt.xlabel(f"d_cf ({cf_metric})"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir, "min_dist_hist.png")); plt.close()
        # ECDF
        xs, ys = _ecdf(finite) if finite else ([], [])
        plt.figure()
        if xs:
            plt.step(xs, ys, where="post")
            plt.axvline(eps0, linestyle="--")
            plt.axhline(1.0 - epsilon, linestyle="--")
        plt.title("ECDF of min distances")
        plt.xlabel(f"d_cf ({cf_metric})"); plt.ylabel("F(x)")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir, "min_dist_ecdf.png")); plt.close()
        # Set sizes
        plt.figure()
        plt.hist(set_sizes, bins=range(0, max(set_sizes or [1]) + 2))
        plt.title("Set sizes"); plt.xlabel("|C|"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir, "set_sizes_hist.png")); plt.close()
        # Samples tried
        plt.figure()
        plt.hist(samples_tried, bins=range(0, max(samples_tried or [1]) + 2))
        plt.title("Samples tried per case"); plt.xlabel("tried"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir, "samples_tried_hist.png")); plt.close()
        print(f"[✓] Wrote plots under {figs_dir}")
    except Exception as e:
        print(f"[i] Skipped plots (matplotlib not available?): {e}")

if __name__ == "__main__":
    main()
