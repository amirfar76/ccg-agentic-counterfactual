# cfllm/eval_counterfactual.py
# Evaluate counterfactual generation *without* conformalization.
# Compares GM-SCM (shared randomness) vs Naive (fresh randomness) across temperatures.
# Outputs per-case CSV, summary JSON, and a PDF plot of avg normalized edit vs temperature.
#
# Added: tqdm progress bar + existing --max-cases support for quick runs.

import os
import json
import math
import argparse
import hashlib
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm  # <-- progress bar

from .config import OUTPUT_DIR
from .llm import action_from_prompt, report_from_metrics
from .env_bridge import run_ns3_action

# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def sha_seed(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")

def clamp_action_fields(act: dict, cap_duration: float | None, max_num_ues: int | None) -> dict:
    act = dict(act)
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

def norm_levenshtein(a: str, b: str) -> float:
    """
    Normalized Levenshtein edit distance in [0,1].
    If both empty -> 0.0; if one empty -> 1.0; else edit / max_len.
    """
    if a is None: a = ""
    if b is None: b = ""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0 if (la + lb) > 0 else 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,       # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost # substitution
            )
        prev = cur
    edit = prev[lb]
    denom = max(la, lb)
    return float(edit) / float(denom)

def extract_noise_fields(r: dict) -> dict:
    """
    Pull abduced/shared randomness seeds (for GM-SCM) if present in dataset row.
    Supports multiple historical field names.
    """
    n = r.get("noise", {}) or {}
    seeds = {
        "ollama_seed": n.get("ollama_seed") or r.get("ollama_seed"),
        "report_seed": n.get("report_seed") or r.get("report_seed"),
        "ns3_rng_run": n.get("rng_run") or n.get("ns3_seed") or r.get("rng_run") or r.get("ns3_seed"),
    }
    return seeds

def cached_path(outdir: str, idx: int, mode: str, temp: float) -> str:
    return os.path.join(outdir, "cache", f"case_{idx:06d}_{mode}_t{temp:.3f}.json")

# ----------------------------
# Single-case evaluation
# ----------------------------

def eval_one_case(idx: int,
                  row: dict,
                  temperature: float,
                  mode: str,                      # "gm" or "naive"
                  base_seed: int,
                  cap_duration: float | None,
                  max_num_ues: int | None,
                  cache_dir: str,
                  reuse_cache: bool) -> dict:
    """
    Returns a dict with:
      idx, mode, temperature, evaluated, norm_edit, error (str), est_text (for cache)
    """
    ensure_dir(cache_dir)
    cbase = os.path.dirname(cache_dir)
    cpath = cached_path(cbase, idx, mode, temperature)
    if reuse_cache and os.path.exists(cpath):
        try:
            return json.loads(Path(cpath).read_text())
        except Exception:
            pass  # compute anew

    X   = row.get("X", "")
    Xp  = row.get("X_prime", "")
    Yp_true = row.get("Y_prime_true", None)
    if Yp_true is None:
        Yp_true = ""

    seeds = extract_noise_fields(row)
    rnd = random.Random(base_seed + 104729 * (idx + 1) + int(temperature * 1000))

    if mode == "gm":
        ollama_seed = seeds.get("ollama_seed")
        report_seed = seeds.get("report_seed", ollama_seed)
        ns3_seed    = seeds.get("ns3_rng_run")

        if ollama_seed is None:
            ollama_seed = sha_seed(row.get("Y", ""))
        if report_seed is None:
            report_seed = ollama_seed
        if ns3_seed is None:
            ns3_seed = (sha_seed(X) ^ sha_seed(Xp)) % (10**9)

    elif mode == "naive":
        ollama_seed = rnd.randint(1, 2**31 - 1)
        report_seed = rnd.randint(1, 2**31 - 1)
        ns3_seed    = rnd.randint(1, 10**9)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Build action for X'
    try:
        act = action_from_prompt(
            Xp,
            seed=ollama_seed,
            temperature=temperature,
            use_ollama=True,           # force LLM (ignore FAST path)
        )
        act = clamp_action_fields(act, cap_duration, max_num_ues)
    except Exception as e:
        payload = {
            "idx": idx, "mode": mode, "temperature": float(temperature),
            "evaluated": 0, "norm_edit": float("nan"),
            "error": f"action_from_prompt failed: {e}",
            "est_text": ""
        }
        Path(cpath).write_text(json.dumps(payload, indent=2))
        return payload

    # ns-3
    try:
        metrics = run_ns3_action(act, rng_run=int(ns3_seed), workdir=os.path.join(OUTPUT_DIR, "tmp"))
    except Exception as e:
        payload = {
            "idx": idx, "mode": mode, "temperature": float(temperature),
            "evaluated": 0, "norm_edit": float("nan"),
            "error": f"ns3 failed: {e}",
            "est_text": ""
        }
        Path(cpath).write_text(json.dumps(payload, indent=2))
        return payload

    # Summarize metrics
    try:
        est_text = report_from_metrics(
            metrics,
            seed=report_seed,
            temperature=temperature,
            use_ollama=True,           # force LLM (ignore FAST path)
        )
    except Exception as e:
        payload = {
            "idx": idx, "mode": mode, "temperature": float(temperature),
            "evaluated": 0, "norm_edit": float("nan"),
            "error": f"report_from_metrics failed: {e}",
            "est_text": ""
        }
        Path(cpath).write_text(json.dumps(payload, indent=2))
        return payload

    d = norm_levenshtein(est_text, Yp_true)
    payload = {
        "idx": idx,
        "mode": mode,
        "temperature": float(temperature),
        "evaluated": 1,
        "norm_edit": float(d),
        "error": "",
        "est_text": est_text
    }
    Path(cpath).write_text(json.dumps(payload, indent=2))
    return payload

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate CF generation (GM-SCM vs Naive) vs temperature, without conformalization.")
    ap.add_argument("--data-path", type=str, default=None, help="Path to test.jsonl (must include Y_prime_true).")
    ap.add_argument("--temps", type=str, default="0.0,0.5,1.0", help="Comma-separated temperatures.")
    ap.add_argument("--max-cases", type=int, default=None, help="Limit number of test cases (for quick runs).")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    ap.add_argument("--outdir", type=str, default=os.path.join(OUTPUT_DIR, "cf_eval"), help="Output directory.")
    ap.add_argument("--reuse-cache", action="store_true", help="Reuse cached per-case results.")
    ap.add_argument("--use-tex", action="store_true", help="Use LaTeX fonts in plots.")
    ap.add_argument("--font-size", type=int, default=11, help="Base font size.")
    ap.add_argument("--figsize", type=str, default="4,3", help="Figure size 'W,H' in inches.")
    ap.add_argument("--cap-duration", type=float, default=0.5, help="Cap sim duration (seconds) for speed.")
    ap.add_argument("--max-num-ues", type=int, default=3, help="Cap number of UEs for speed.")
    ap.add_argument("--seed", type=int, default=123, help="Master seed for Naive randomness.")
    args = ap.parse_args()

    # Fonts / LaTeX setup
    if args.use_tex:
        rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "axes.unicode_minus": False,
            "font.size": args.font_size,
        })
    else:
        rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": args.font_size,
        })

    outdir = args.outdir
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "cache"))
    figs_dir = os.path.join(outdir, "figs")
    ensure_dir(figs_dir)

    test_path = args.data_path or os.path.join(OUTPUT_DIR, "data", "test.jsonl")
    rows = list(read_jsonl(test_path))
    if len(rows) == 0:
        raise RuntimeError(f"No test rows at {test_path}")

    # Subset if requested
    if args.max_cases is not None:
        rows = rows[: int(args.max_cases)]

    # Parse temperatures
    temps = []
    for t in args.temps.split(","):
        t = t.strip()
        if not t:
            continue
        temps.append(float(t))
    if not temps:
        temps = [0.0]

    # Build job list: each idx × each temp × mode in {"gm","naive"}
    jobs = []
    for i, r in enumerate(rows):
        for temp in temps:
            for mode in ("gm", "naive"):
                jobs.append((i, r, float(temp), mode))

    # Evaluate with progress bar
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = []
            for (i, r, temp, mode) in jobs:
                futs.append(ex.submit(
                    eval_one_case, i, r, temp, mode, args.seed,
                    args.cap_duration, args.max_num_ues,
                    os.path.join(outdir, "cache"),
                    args.reuse_cache
                ))
            with tqdm(total=len(futs), desc="Evaluating (parallel)", leave=True) as pbar:
                for fut in as_completed(futs):
                    results.append(fut.result())
                    pbar.update(1)
    else:
        for (i, r, temp, mode) in tqdm(jobs, desc="Evaluating", leave=True):
            results.append(eval_one_case(
                i, r, temp, mode, args.seed,
                args.cap_duration, args.max_num_ues,
                os.path.join(outdir, "cache"),
                args.reuse_cache
            ))

    # Save per-case CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(outdir, "cf_eval_results.csv")
    df.to_csv(csv_path, index=False)

    # Summary aggregation
    summary_df = (
        df.groupby(["mode", "temperature"])
          .agg(
              n_cases=("idx", "count"),
              n_evaluated=("evaluated", "sum"),
              avg_norm_edit=("norm_edit", "mean"),
          )
          .reset_index()
          .sort_values(["mode", "temperature"])
    )

    summary_payload = {
        "n_rows": int(len(df)),
        "by_temp_mode": summary_df.to_dict(orient="records"),
        "csv": csv_path
    }
    summary_json_path = os.path.join(outdir, "cf_eval_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    # Diagnostics
    print("\n[Diagnostics] n_evaluated by mode & temperature:")
    for _, row in summary_df.iterrows():
        print(f"  mode={row['mode']:5s}  temp={row['temperature']:>4.2f}  "
              f"n_evaluated={int(row['n_evaluated'])}  avg_norm_edit={row['avg_norm_edit']}")

    # Plot: avg normalized edit vs temperature (GM vs Naive)
    try:
        w, h = [float(x) for x in args.figsize.split(",")]
    except Exception:
        w, h = 4.0, 3.0

    plt.figure(figsize=(w, h))
    drew_any_curve = False
    for mode, label in [("gm", "GM-SCM (shared randomness)"),
                        ("naive", "Naive (fresh randomness)")]:
        sub = summary_df[summary_df["mode"] == mode].sort_values("temperature")
        if len(sub):
            plt.plot(sub["temperature"], sub["avg_norm_edit"], marker="o", label=label)
            drew_any_curve = True
        else:
            print(f"[Warning] No points for mode={mode}; curve omitted.")

    plt.xlabel("Temperature")
    plt.ylabel("Average normalized edit distance")
    if drew_any_curve:
        plt.legend(frameon=False)
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, "avg_norm_edit_vs_temp.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print(f"\n[✓] Per-case CSV:  {csv_path}")
    print(f"[✓] Summary JSON: {summary_json_path}")
    print(f"[✓] Figure:       {fig_path}")

if __name__ == "__main__":
    main()
