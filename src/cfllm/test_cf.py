import argparse
import csv
import json
import os
import random
from tqdm import tqdm
from typing import Dict, Any
from tqdm import trange

from .config import OUTPUT_DIR, DEFAULT_MAX_SAMPLES, DEFAULT_METRIC
from .scores import METRICS
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action

def _load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "test.jsonl"))
    ap.add_argument("--calibration", type=str, default=os.path.join(OUTPUT_DIR, "calibration", "calibration.json"))
    ap.add_argument("--metric", type=str, default=DEFAULT_METRIC, choices=["rouge","numeric"])
    ap.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.calibration, "r") as f:
        calib = json.load(f)
    tau = float(calib["threshold"])
    metric_name = args.metric
    dist = METRICS[metric_name]

    results_csv = os.path.join(OUTPUT_DIR, "test", "results.csv")
    summary_json = os.path.join(OUTPUT_DIR, "test", "summary.json")

    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    n = 0
    covered = 0
    accepted = 0
    total_samples = 0
    total_error_if_accepted = 0.0

    with open(results_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["idx","covered_truth","accepted","samples_used","dist_Y_Yp_true","dist_Yp_accepted_to_true"])

        rows = list(_load_jsonl(args.test_path))
        for idx, rec in enumerate(tqdm(rows, desc="test", unit="case")):
            n += 1
            X = rec["X"]
            Xp = rec["X_prime"]
            Y = rec["Y"]
            Yp_true = rec["Y_prime_true"]

            # coverage of truth
            d_truth = dist(Y, Yp_true)
            is_covered = d_truth <= tau
            if is_covered:
                covered += 1

            # Now try to find an accepted sample by sampling exogenous noise for X'
            found = False
            chosen_report = None
            samples_used = 0
            for sidx in range(args.max_samples):
                samples_used += 1
                seed_try = random.randint(1, 2**31-1)
                rng_run_try = random.randint(1, 10**6)
                act = action_from_prompt(Xp, seed=seed_try)
                act["duration_s"] = min(act.get("duration_s", 10.0), 3.0)   # 3s cap
                act["num_ues"]    = min(int(act.get("num_ues", 5)), 8)      # optional cap
                metrics = run_ns3_action(act, rng_run=rng_run_try, workdir=os.path.join(OUTPUT_DIR, "tmp"))
                rep = report_from_metrics(metrics, seed=seed_try)
                if dist(Y, rep) <= tau:
                    found = True
                    chosen_report = rep
                    break

            total_samples += samples_used
            if found:
                accepted += 1
                total_error_if_accepted += dist(chosen_report, Yp_true)
                w.writerow([idx, int(is_covered), 1, samples_used, f"{d_truth:.6f}", f"{dist(chosen_report, Yp_true):.6f}"])
            else:
                w.writerow([idx, int(is_covered), 0, samples_used, f"{d_truth:.6f}", ""])

    summary = {
        "n": n,
        "tau": tau,
        "metric": metric_name,
        "coverage_truth": covered / max(1, n),
        "accept_rate": accepted / max(1, n),
        "avg_samples": total_samples / max(1, n),
        "avg_error_if_accepted": (total_error_if_accepted / accepted) if accepted > 0 else None
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote results to {results_csv}\nSummary:\n{json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()
