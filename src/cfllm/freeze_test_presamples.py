# cfllm/freeze_test_presamples.py
import os, json, argparse, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from cfllm.config import OUTPUT_DIR
from cfllm.llm import action_from_prompt, report_from_metrics
from cfllm.env_bridge import run_ns3_action

TEST_DIR = os.path.join(OUTPUT_DIR, "test_clm")
DATA_TEST = os.path.join(OUTPUT_DIR, "data", "test.jsonl")
PRESAMPLES = os.path.join(TEST_DIR, "presamples.jsonl")
PRESAMPLE_META = os.path.join(TEST_DIR, "presample_meta.json")

def _rng_seq(seed_base: int, i: int, kmax: int):
    # deterministic rng_run per (seed_base, i, j)
    for j in range(1, kmax + 1):
        h = hashlib.sha256(f"{seed_base}-{i}-{j}".encode()).digest()
        yield int.from_bytes(h[:4], "big")

def _one_case(i, rec, kmax, seed):
    Xp = rec["X_prime"]
    # Use FAST_ACTION if set; this avoids Ollama calls
    act = action_from_prompt(Xp, seed=seed + i)
    # keep sim short & stable if caller used short caps during calib/test
    candidates = []
    for rng_run in _rng_seq(seed, i, kmax):
        metrics = run_ns3_action(act, rng_run=rng_run)
        y = report_from_metrics(metrics, seed=0)
        candidates.append({
            "rng_run": rng_run,
            "report": y,
            "meta": {
                "duration_s": act.get("duration_s"),
                "num_ues": act.get("num_ues"),
                "scheduler": act.get("scheduler"),
                "traffic_mbps": act.get("traffic_mbps"),
            }
        })
    return {"idx": i, "num_candidates": len(candidates), "candidates": candidates}

def main():
    ap = argparse.ArgumentParser(description="Freeze presampled candidates for the test set.")
    ap.add_argument("--kmax", type=int, required=True, help="Candidates per test case.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(TEST_DIR, exist_ok=True)
    test_rows = [json.loads(l) for l in open(DATA_TEST)]
    print(f"[*] Freezing presamples for {len(test_rows)} test cases; kmax={args.kmax}")

    rows = [None] * len(test_rows)
    with ThreadPoolExecutor(max_workers=args.workers or 1) as ex:
        futs = {ex.submit(_one_case, i, test_rows[i], args.kmax, args.seed): i for i in range(len(test_rows))}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            rows[i] = fut.result()
            done += 1
            if done % 5 == 0 or done == len(test_rows):
                print(f"  - {done}/{len(test_rows)}")

    with open(PRESAMPLES, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with open(PRESAMPLE_META, "w") as f:
        json.dump({"kmax": args.kmax, "seed": args.seed, "n": len(test_rows)}, f, indent=2)

    print(f"[✓] Wrote {PRESAMPLES} and {PRESAMPLE_META}")
    print("    Future tests can now use:  python -m cfllm.retest_from_presamples --dump-sets")
if __name__ == "__main__":
    main()
