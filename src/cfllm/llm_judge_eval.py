# cfllm/llm_judge_eval.py
# Single- and multi-case evaluation of CG vs IG reports using an LLM judge.

import os
import json
import argparse
import random
import time
from typing import Optional, Tuple, List, Dict

import requests
import pandas as pd
import matplotlib.pyplot as plt


# -------------- Ollama client ----------------

def ollama_generate(model: str, prompt: str, seed: Optional[int] = None,
                    temperature: float = 0.7, top_p: float = 0.9,
                    num_ctx: int = 4096, base_url: str = "http://localhost:11434") -> str:
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p, "num_ctx": num_ctx}
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# -------------- Data utils ----------------

def read_timeseries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"time_s", "thr_total_mbps", "avg_delay_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    return df


def summarize_series_for_llm(df: pd.DataFrame, max_points: int = 120) -> str:
    if len(df) > max_points:
        step = max(1, len(df) // max_points)
        df = df.iloc[::step].reset_index(drop=True)
    df2 = df.copy()
    df2["time_s"] = df2["time_s"].round(2)
    df2["thr_total_mbps"] = df2["thr_total_mbps"].round(3)
    df2["avg_delay_ms"] = df2["avg_delay_ms"].round(3)
    lines = ["time_s,thr_mbps,delay_ms"]
    lines += [f"{r.time_s},{r.thr_total_mbps},{r.avg_delay_ms}" for r in df2.itertuples(index=False)]
    return "\n".join(lines)


# -------------- Prompts ----------------

REPORT_SYSTEM = (
    "You analyze mobile network experiment time series. "
    "Given throughput (Mbps) and average UE delay (ms) over time, write a short, factual report: "
    "1) overall trend, 2) notable events (peaks, dips, congestion), 3) any threshold breaches. "
    "Be concise (~5-8 sentences). Do not fabricate numbers; summarize from the data."
)
REPORT_USER_TEMPLATE = """You are given a time series from an experiment:

{series_block}

Write the report now.
"""
JUDGE_SYSTEM = (
    "You are a careful evaluator. You will be given a true reference report and two candidate reports. "
    "Decide which candidate is closer in content and conclusions to the reference. "
    "Only output a single token: 'CG' or 'IG'. If they are tied, output 'TIE'. No extra text."
)
JUDGE_USER_TEMPLATE = """[REFERENCE REPORT]
{true_report}

[CANDIDATE A: CG]
{cg_report}

[CANDIDATE B: IG]
{ig_report}

Which candidate is closer to the reference? Reply with exactly one of: CG, IG, TIE
"""


def build_chat_prompt(system: str, user: str) -> str:
    sep = "\n\n---\n\n"
    return f"<<SYSTEM>>\n{system}\n<</SYSTEM>>{sep}{user}".strip()


# -------------- Core functions ----------------

def make_report(model: str, df: pd.DataFrame, seed: int, temperature: float, top_p: float, base_url: str) -> str:
    series_block = summarize_series_for_llm(df)
    prompt = build_chat_prompt(REPORT_SYSTEM, REPORT_USER_TEMPLATE.format(series_block=series_block))
    return ollama_generate(model, prompt, seed=seed, temperature=temperature, top_p=top_p, base_url=base_url)


def judge_reports(model: str, true_report: str, cg_report: str, ig_report: str,
                  seed: int, base_url: str) -> str:
    prompt = build_chat_prompt(JUDGE_SYSTEM, JUDGE_USER_TEMPLATE.format(
        true_report=true_report, cg_report=cg_report, ig_report=ig_report
    ))
    verdict = ollama_generate(model, prompt, seed=seed, temperature=0.0, top_p=1.0, base_url=base_url).strip().upper()
    if "CG" in verdict and "IG" in verdict:
        return "TIE"
    if verdict not in {"CG", "IG", "TIE"}:
        if "CG" in verdict: return "CG"
        if "IG" in verdict: return "IG"
        return "TIE"
    return verdict


def plot_wins_bar(outdir: str, cg_wins: int, ig_wins: int, ties: int):
    os.makedirs(outdir, exist_ok=True)
    labels = ["CG", "IG", "TIE"]
    vals = [cg_wins, ig_wins, ties]
    plt.figure(figsize=(4.2, 3.2))
    plt.bar(labels, vals)
    plt.title("LLM Judge Results")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(outdir, "judge_wins_bar.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[✓] Wrote {path}")


def save_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# -------------- Main loop ----------------

def process_case(case_dir: str, args) -> Dict:
    print(f"[INFO] Processing case: {case_dir}")
    true_csv = os.path.join(case_dir, "true_timeseries.csv")
    cg_csv = os.path.join(case_dir, "cg_timeseries.csv")
    ig_csv = os.path.join(case_dir, "ig_timeseries.csv")

    df_true = read_timeseries(true_csv)
    df_cg = read_timeseries(cg_csv)
    df_ig = read_timeseries(ig_csv)

    seed_true = int(args.llm_seed_true)
    seed_cg = seed_true if args.llm_seed_cg is None else int(args.llm_seed_cg)
    seed_ig = int(args.llm_seed_ig) if args.llm_seed_ig else random.randint(1, 2**31 - 1)

    true_report = make_report(args.model, df_true, seed_true, args.llm_temp, args.llm_top_p, args.ollama_url)
    cg_report = make_report(args.model, df_cg, seed_cg, args.llm_temp, args.llm_top_p, args.ollama_url)
    ig_report = make_report(args.model, df_ig, seed_ig, args.llm_temp, args.llm_top_p, args.ollama_url)

    case_id = os.path.basename(case_dir.rstrip("/"))
    save_text(os.path.join(args.outdir, f"{case_id}_true.txt"), true_report)
    save_text(os.path.join(args.outdir, f"{case_id}_cg.txt"), cg_report)
    save_text(os.path.join(args.outdir, f"{case_id}_ig.txt"), ig_report)

    verdict = judge_reports(args.model, true_report, cg_report, ig_report, seed=args.judge_seed, base_url=args.ollama_url)
    print(f"[verdict for {case_id}] {verdict}")

    return {
        "case": case_id,
        "verdict": verdict,
        "seed_true": seed_true,
        "seed_cg": seed_cg,
        "seed_ig": seed_ig,
    }


def main():
    ap = argparse.ArgumentParser()
    # single case mode
    ap.add_argument("--true", help="CSV true")
    ap.add_argument("--cg", help="CSV CG")
    ap.add_argument("--ig", help="CSV IG")
    # batch mode
    ap.add_argument("--cases-root", help="Folder with multiple case directories")
    # common
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="llama3:latest")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--llm-seed-true", type=int, default=4242)
    ap.add_argument("--llm-seed-cg", type=int, default=None)
    ap.add_argument("--llm-seed-ig", type=int, default=None)
    ap.add_argument("--llm-temp", type=float, default=0.2)
    ap.add_argument("--llm-top-p", type=float, default=0.9)
    ap.add_argument("--judge-seed", type=int, default=999)
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results = []

    # Batch mode
    if args.cases_root:
        cases = sorted([os.path.join(args.cases_root, d)
                        for d in os.listdir(args.cases_root)
                        if os.path.isdir(os.path.join(args.cases_root, d))])
        print(f"[INFO] Found {len(cases)} cases")
        for c in cases:
            results.append(process_case(c, args))

    # Single mode
    elif args.true and args.cg and args.ig:
        tmp_case = {
            "case": "single",
            "verdict": process_case(
                case_dir=os.path.dirname(args.true),
                args=args
            )
        }
        results.append(tmp_case)

    else:
        raise ValueError("Either provide --true/--cg/--ig OR --cases-root")

    # Aggregate results
    cg_wins = sum(1 for r in results if r["verdict"] == "CG")
    ig_wins = sum(1 for r in results if r["verdict"] == "IG")
    ties = sum(1 for r in results if r["verdict"] == "TIE")

    plot_wins_bar(args.outdir, cg_wins, ig_wins, ties)

    # Save results table
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.outdir, "results.csv"), index=False)
    with open(os.path.join(args.outdir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[RESULTS] CG={cg_wins}, IG={ig_wins}, TIE={ties}")
    print(f"[✓] Results saved to {args.outdir}")


if __name__ == "__main__":
    main()