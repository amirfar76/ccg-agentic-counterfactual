import argparse
import json
import os
from typing import List
import numpy as np

from .config import OUTPUT_DIR, DEFAULT_ALPHA, DEFAULT_METRIC
from .scores import METRICS

def _load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def conformal_threshold(scores: List[float], alpha: float) -> float:
    n = len(scores)
    if n == 0:
        raise ValueError("No scores to calibrate on.")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    s_sorted = np.sort(np.array(scores))
    return float(s_sorted[k - 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--metric", type=str, default=DEFAULT_METRIC, choices=list(METRICS.keys()))
    ap.add_argument("--calib-path", type=str, default=os.path.join(OUTPUT_DIR, "data", "calib.jsonl"))
    args = ap.parse_args()

    dist = METRICS[args.metric]
    scores = []
    for rec in _load_jsonl(args.calib_path):
        y = rec["Y"]
        yp = rec["Y_prime_true"]
        s = dist(y, yp)
        scores.append(s)
    tau = conformal_threshold(scores, args.alpha)

    out = {
        "alpha": args.alpha,
        "metric": args.metric,
        "n": len(scores),
        "threshold": tau,
    }
    out_path = os.path.join(OUTPUT_DIR, "calibration", "calibration.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved calibration to {out_path}:\n{json.dumps(out, indent=2)}")

if __name__ == "__main__":
    main()
