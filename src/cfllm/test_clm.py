# cfllm/test_clm.py
# Test CLM-style calibrated sampler with presampling/caching for fast re-runs.
# Saves per-case sets & full artifacts, plus CSV, summary, and plots.

import os, json, math, argparse, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .scores import METRICS

# ------------------------ utils ------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sha_rng(seed: int, case_idx: int, j: int) -> int:
    """Deterministic rng_run from (seed, case_idx, j)."""
    h = hashlib.sha256(f"{seed}:{case_idx}:{j}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")  # 32-bit

def similarity(sim_metric_name: str, a: str, b: str) -> float:
    # Treat metrics as distances in [0,1]; convert to similarity
    d = float(METRICS[sim_metric_name](a, b))
    d = min(max(d, 0.0), 1.0)
    return 1.0 - d

# ------------------------ core building blocks ------------------------

def build_action_from_Xp(Xp: str, cap_duration, max_num_ues, seed: int):
    act = action_from_prompt(Xp, seed=seed)
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
    return act

def draw_candidates_for_case(case_idx, Xp, kmax, seed, cap_duration, max_num_ues):
    """Run ns-3 (with its own disk-cache) to get kmax candidates for this case. Returns list of dicts."""
    act = build_action_from_Xp(Xp, cap_duration, max_num_ues, seed=seed + 13 * (case_idx + 1))
    cands = []
    for j in range(1, kmax + 1):
        rng_run = sha_rng(seed, case_idx, j)
        try:
            metrics = run_ns3_action(act, rng_run=rng_run, workdir=os.path.join(OUTPUT_DIR, "tmp"))
            y = report_from_metrics(metrics, seed=0)
            cands.append({
                "rng_run": rng_run,
                "report": y,
                "meta": {
                    "duration_s": act.get("duration_s"),
                    "num_ues": act.get("num_ues"),
                    "scheduler": act.get("scheduler"),
                    "traffic_mbps": act.get("traffic_mbps"),
                }
            })
        except Exception:
            # miss or failure -> skip; we still return what we have
            continue
    return cands

def build_set_from_candidates(cands, lam, sim_metric):
    """Apply (lambda1, lambda3) to candidate texts."""
    l1 = float(lam["lambda1"]); l3 = int(lam["lambda3"])
    C, kept_runs = [], []
    tried = 0
    for cand in cands:
        yk = (cand.get("report") or cand.get("text") or "").strip()
        if not yk:
            continue
        tried += 1
        if C:
            smax = max(similarity(sim_metric, yk, cj) for cj in C)
            if smax > l1:
                continue
        C.append(yk); kept_runs.append(cand.get("rng_run"))
        if len(C) >= l3:
            break
    return C, kept_runs, tried

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Test CLM-style calibrated sampler; fast presample & reuse.")
    ap.add_argument("--calib-path", type=str, default=None, help="Path to clm_calibration.json")
    ap.add_argument("--data-path", type=str, default=None, help="Override test.jsonl path")
    ap.add_argument("--dump-sets", action="store_true", help="Save per-case artifacts (X,Y,X',Y'_true,C, rngs).")
    ap.add_argument("--cache-sets", action="store_true", help="Cache sets under sets_cached/<lambda_key>/ for reuse.")
    ap.add_argument("--presample-only", action="store_true", help="Just presample kmax candidates per case and exit.")
    ap.add_argument("--reuse-presamples", action="store_true", help="Build sets from saved presamples (no ns-3).")
    ap.add_argument("--save-presamples", action="store_true", help="When sampling online, also write presamples.jsonl.")
    ap.add_argument("--kmax", type=int, default=None, help="Override calibration kmax for this run.")
    ap.add_argument("--limit", type=int, default=None, help="Run only the first N test cases.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    ap.add_argument("--seed", type=int, default=123, help="Master seed.")
    args = ap.parse_args()

    calib_path = args.calib_path or os.path.join(OUTPUT_DIR, "calibration", "clm_calibration.json")
    with open(calib_path, "r") as f:
        calib = json.load(f)
    if not calib.get("selected"):
        raise RuntimeError("Calibration has no selected λ (Λ_valid empty). Relax eps0/epsilon, enlarge kmax, or expand grids.")
    lam = calib["selected"]
    kmax_calib = calib["kmax"]
    kmax = int(args.kmax or kmax_calib)
    cf_metric = calib["cf_metric"]
    eps0 = float(calib["cf_eps0"])
    sim_metric = calib["sim_metric"]
    cap_duration = calib.get("cap_duration")
    max_num_ues = calib.get("max_num_ues")

    # input/output layout
    test_path = args.data_path or os.path.join(OUTPUT_DIR, "data", "test.jsonl")
    rows = list(read_jsonl(test_path))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError(f"No test rows at {test_path} (after limit filter).")

    out_dir = os.path.join(OUTPUT_DIR, "test_clm")
    ensure_dir(out_dir)
    sets_dir = os.path.join(out_dir, "sets")
    if args.dump_sets:
        ensure_dir(sets_dir)

    # presample storage
    presamples_path = os.path.join(out_dir, "presamples.jsonl")
    presample_meta = os.path.join(out_dir, "presample_meta.json")

    # optional sets cache
    lambda_key = f"l1={lam['lambda1']}_l3={lam['lambda3']}_sim={sim_metric}_cf={cf_metric}_eps0={eps0}"
    sets_cache_root = os.path.join(out_dir, "sets_cached", lambda_key)  # per-λ̂ cache
    if args.cache_sets:
        ensure_dir(sets_cache_root)

    # ---------------- Presample path ----------------
    if args.presample_only:
        print(f"[*] Presampling kmax={kmax} candidates for {len(rows)} cases...")
        # parallelize per-case candidate drawing
        results = [None] * len(rows)
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(draw_candidates_for_case, i, rows[i]["X_prime"], kmax, args.seed, cap_duration, max_num_ues): i
                    for i in range(len(rows))
                }
                for fut in tqdm(as_completed(futs), total=len(rows), desc="presample"):
                    i = futs[fut]; results[i] = fut.result()
        else:
            for i in tqdm(range(len(rows)), desc="presample"):
                results[i] = draw_candidates_for_case(i, rows[i]["X_prime"], kmax, args.seed, cap_duration, max_num_ues)

        with open(presamples_path, "w") as f:
            for i, cands in enumerate(results):
                rec = {
                    "idx": i,
                    "X": rows[i]["X"],
                    "Y": rows[i]["Y"],
                    "X_prime": rows[i]["X_prime"],
                    "Y_prime_true": rows[i].get("Y_prime_true"),
                    "candidates": cands or [],
                }
                f.write(json.dumps(rec) + "\n")
        with open(presample_meta, "w") as f:
            json.dump({"kmax": kmax, "seed": args.seed, "n": len(rows)}, f, indent=2)
        print(f"[✓] Wrote {presamples_path} and {presample_meta}")
        return

    # ---------------- Load or create candidates ----------------
    have_presamples = os.path.isfile(presamples_path) and os.path.isfile(presample_meta)
    if args.reuse_presamples:
        if not have_presamples:
            raise RuntimeError("No presamples found. Run with --presample-only (once) or without --reuse-presamples.")
        pres_rows = [json.loads(l) for l in read_jsonl(presamples_path)]
        # Align by idx (assume same order/limit for simplicity)
        candidates_by_idx = {r["idx"]: r["candidates"] for r in pres_rows}
    else:
        # Online sampling (still benefits from ns-3 disk cache). Optionally write presamples for future runs.
        print(f"[*] Sampling online (kmax={kmax}) for {len(rows)} cases...")
        results = [None] * len(rows)
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(draw_candidates_for_case, i, rows[i]["X_prime"], kmax, args.seed, cap_duration, max_num_ues): i
                    for i in range(len(rows))
                }
                for fut in tqdm(as_completed(futs), total=len(rows), desc="sample"):
                    i = futs[fut]; results[i] = fut.result()
        else:
            for i in tqdm(range(len(rows)), desc="sample"):
                results[i] = draw_candidates_for_case(i, rows[i]["X_prime"], kmax, args.seed, cap_duration, max_num_ues)
        candidates_by_idx = {i: results[i] for i in range(len(rows))}
        if args.save_presamples:
            with open(presamples_path, "w") as f:
                for i in range(len(rows)):
                    rec = {
                        "idx": i,
                        "X": rows[i]["X"],
                        "Y": rows[i]["Y"],
                        "X_prime": rows[i]["X_prime"],
                        "Y_prime_true": rows[i].get("Y_prime_true"),
                        "candidates": candidates_by_idx[i] or [],
                    }
                    f.write(json.dumps(rec) + "\n")
            with open(presample_meta, "w") as f:
                json.dump({"kmax": kmax, "seed": args.seed, "n": len(rows)}, f, indent=2)
            print(f"[✓] Saved presamples to {presamples_path}")

    # ---------------- Build sets & evaluate ----------------
    dist_cf = METRICS[cf_metric]
    results_rows = []
    min_dists = []
    set_sizes = []
    samples_tried_list = []

    for i, r in enumerate(tqdm(rows, desc="build+eval")):
        cands = candidates_by_idx.get(i, [])
        C, kept_runs, tried = build_set_from_candidates(cands, lam, sim_metric)

        # cache set if requested
        if args.cache_sets:
            p = os.path.join(sets_cache_root, f"case_{i}.json")
            with open(p, "w") as f:
                json.dump({"idx": i, "kept": C, "rng_runs": kept_runs}, f, indent=2)

        # dump full per-case artifact if requested
        if args.dump_sets:
            payload = {
                "idx": i,
                "lambda": lam,
                "cf_metric": cf_metric,
                "eps0": eps0,
                "X": r["X"],
                "Y": r["Y"],
                "X_prime": r["X_prime"],
                "Y_prime_true": r.get("Y_prime_true"),
                "set": C,
                "rng_runs": kept_runs,
                "candidates": cands,  # include all candidates we saw
            }
            with open(os.path.join(sets_dir, f"cfset_case_{i}.json"), "w") as g:
                json.dump(payload, g, indent=2)

        # evaluate against truth
        set_size = len(C)
        samples_tried_list.append(tried)
        set_sizes.append(set_size)

        ytrue = r.get("Y_prime_true")
        if ytrue and set_size:
            dists = [float(dist_cf(y, ytrue)) for y in C]
            m = min(dists)
        elif ytrue and not set_size:
            m = float("inf")
        else:
            m = float("nan")
        min_dists.append(m)
        covered = int((not math.isinf(m)) and (m <= eps0))

        results_rows.append({
            "idx": i,
            "set_size": set_size,
            "samples_tried": tried,
            "min_dist_to_truth": m,
            "covered_eps0": covered
        })

    # ---------------- Write CSV & summary ----------------
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "results.csv")
    pd.DataFrame(results_rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(results_rows)
    n = len(df)
    finite_vals = df["min_dist_to_truth"].replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
    coverage = float(df["covered_eps0"].mean()) if n else float("nan")
    avg_set = float(df["set_size"].mean()) if n else float("nan")
    avg_samples = float(df["samples_tried"].mean()) if n else float("nan")
    median_min_dist = float(np.median(finite_vals)) if finite_vals.size else float("nan")
    mean_min_dist = float(np.mean(finite_vals)) if finite_vals.size else float("nan")

    summary = {
        "mode": "clm",
        "selected_lambda": lam,
        "kmax": kmax,
        "cf_metric": cf_metric,
        "eps0": eps0,
        "sim_metric": sim_metric,
        "cap_duration": cap_duration,
        "max_num_ues": max_num_ues,
        "n_cases": n,
        "coverage_empirical": coverage,
        "avg_set_size": avg_set,
        "avg_samples_tried": avg_samples,
        "median_min_dist_to_truth": median_min_dist,
        "mean_min_dist_to_truth": mean_min_dist,
        "csv_path": csv_path
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:")
    print(json.dumps(summary, indent=2))

    # ---------------- Plots ----------------
    figs_dir = os.path.join(out_dir, "figs")
    ensure_dir(figs_dir)

    if len(finite_vals):
        # 1) Histogram
        plt.figure()
        plt.hist(finite_vals, bins=min(12, max(4, int(np.sqrt(len(finite_vals))))))
        plt.axvline(eps0, linestyle="--")
        plt.xlabel("min_{y ∈ C} d_cf(y, Y'_true)")
        plt.ylabel("Count")
        plt.title("Best candidate distance to truth")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "min_dist_hist.png"))
        plt.close()

        # 2) ECDF
        xs = np.sort(finite_vals)
        ys = np.arange(1, xs.size + 1) / xs.size
        plt.figure()
        plt.step(xs, ys, where="post")
        plt.axvline(eps0, linestyle="--")
        # draw target coverage line: 1 - epsilon (from calib file)
        epsilon = float(calib.get("epsilon", 0.1))
        plt.axhline(1.0 - epsilon, linestyle="--")
        plt.xlabel("min_{y ∈ C} d_cf(y, Y'_true)")
        plt.ylabel("ECDF")
        plt.title("ECDF: distance vs target")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "min_dist_ecdf.png"))
        plt.close()

    # 3) Set sizes
    plt.figure()
    plt.hist(df["set_size"].values, bins=range(0, int(df["set_size"].max() or 0) + 2))
    plt.xlabel("|C|")
    plt.ylabel("Count")
    plt.title("Set sizes")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "set_sizes_hist.png"))
    plt.close()

    # 4) Samples tried
    plt.figure()
    plt.hist(df["samples_tried"].values, bins=min(12, max(4, int(np.sqrt(n)))))
    plt.xlabel("samples_tried")
    plt.ylabel("Count")
    plt.title("Sampling effort per case")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "samples_tried_hist.png"))
    plt.close()

    print(f"Saved CSV to {csv_path} and figures under {figs_dir}/")

if __name__ == "__main__":
    main()
