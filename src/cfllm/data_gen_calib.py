import argparse
import json
import os
import random
from datetime import datetime
from tqdm import trange
from typing import Dict, Any
import numpy as np

from .config import OUTPUT_DIR, DEFAULT_ALPHA, DEFAULT_METRIC
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action
from .edits import make_base_prompt, edit_prompt
from .scm import ExogenousNoise

def one_pipeline(prompt: str, noise: ExogenousNoise) -> Dict[str, Any]:
    act = action_from_prompt(prompt, seed=noise.ollama_seed)
    metrics = run_ns3_action(act, rng_run=noise.rng_run, workdir=os.path.join(OUTPUT_DIR, "tmp"))
    report = report_from_metrics(metrics, seed=noise.ollama_seed)
    return {"action": act, "metrics": metrics, "report": report}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--metric", type=str, default=DEFAULT_METRIC, choices=["rouge","numeric"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = os.path.join(OUTPUT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    calib_path = os.path.join(data_dir, "calib.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")
    meta_path = os.path.join(data_dir, "meta.json")

    def sample_base():
        num_ues = random.randint(3, 12)
        scheduler = random.choice(["rr","pf","mt"])
        traffic = round(random.uniform(0.5, 5.0), 2)
        duration = round(random.uniform(5.0, 12.0), 1)
        return num_ues, scheduler, traffic, duration

    # write meta
    meta = {
        "created": datetime.utcnow().isoformat() + "Z",
        "alpha": args.alpha,
        "metric": args.metric,
        "notes": "Calibration and test data with shared exogenous noise per (X, X')."
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # calibration
    with open(calib_path, "w") as fcal:
        for _ in trange(args.n_calib, desc="calib"):
            num_ues, scheduler, traffic, duration = sample_base()
            X = make_base_prompt(num_ues, scheduler, traffic, duration)
            Xp = edit_prompt(X)
            # shared exogenous noise
            noise = ExogenousNoise(
                ollama_seed=random.randint(1, 2**31-1),
                rng_run=random.randint(1, 10**6)
            )
            out_X = one_pipeline(X, noise)
            out_Xp_true = one_pipeline(Xp, noise)  # true counterfactual under same noise
            rec = {
                "X": X,
                "X_prime": Xp,
                "noise": noise.__dict__,
                "Y": out_X["report"],
                "Y_prime_true": out_Xp_true["report"],
                "Z": out_X["metrics"],
                "Z_prime_true": out_Xp_true["metrics"],
                "action": out_X["action"],
                "action_prime_true": out_Xp_true["action"],
            }
            fcal.write(json.dumps(rec) + "\n")

    # test set
    with open(test_path, "w") as ftest:
        for _ in trange(args.n_test, desc="test"):
            num_ues, scheduler, traffic, duration = sample_base()
            X = make_base_prompt(num_ues, scheduler, traffic, duration)
            Xp = edit_prompt(X)
            noise = ExogenousNoise(
                ollama_seed=random.randint(1, 2**31-1),
                rng_run=random.randint(1, 10**6)
            )
            out_X = one_pipeline(X, noise)
            out_Xp_true = one_pipeline(Xp, noise)  # true counterfactual, hidden at test time
            rec = {
                "X": X,
                "X_prime": Xp,
                "noise": noise.__dict__,
                "Y": out_X["report"],
                "Y_prime_true": out_Xp_true["report"],
                "Z": out_X["metrics"],
                "Z_prime_true": out_Xp_true["metrics"],
                "action": out_X["action"],
                "action_prime_true": out_Xp_true["action"],
            }
            ftest.write(json.dumps(rec) + "\n")

    print(f"Wrote\n  {calib_path}\n  {test_path}\n  {meta_path}")

if __name__ == "__main__":
    main()
