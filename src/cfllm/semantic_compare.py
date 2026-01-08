# cfllm/semantic_compare.py
"""
Compare IG vs CG against the TRUE counterfactual in two ways:
  1) Analytic reports derived from time series (always available).
  2) LLM-as-judge (optional; requires OPENAI_API_KEY). Falls back to a deterministic judge.

Outputs:
  - <outdir>/report_true.md
  - <outdir>/report_cg.md
  - <outdir>/report_ig.md
  - <outdir>/judge_decision.json
  - <outdir>/summary.md  (side-by-side snippets + decision)
"""
import os
import json
import argparse
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from .report_utils import (
    read_ts, select_window, render_report,
    mae, auc_diff, xcorr_peak, dtw_distance, jaccard_above
)

# ----- optional OpenAI (guarded import)
def _maybe_openai():
    try:
        import openai  # type: ignore
        return openai
    except Exception:
        return None

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _dt_from_time(df: pd.DataFrame) -> float:
    t = df["time_s"].values
    if len(t) < 2:
        return 1.0
    dt = np.diff(t).mean()
    return float(dt) if dt > 0 else 1.0

def _offline_score(true: pd.DataFrame, cand: pd.DataFrame,
                   thr_thr_mbps: float, max_lag_s: float) -> Dict[str, float]:
    # Align to same times (assumes same sampling grid from your ns-3 export)
    y_true = true["thr_total_mbps"].values.astype(float)
    y_cand = cand["thr_total_mbps"].values.astype(float)
    dt = _dt_from_time(true)
    max_lag = max(1, int(round(max_lag_s / max(dt, 1e-6))))

    s = {}
    s["mae"]       = mae(y_true, y_cand)
    s["auc_diff"]  = auc_diff(y_true, y_cand, dt)
    s["xcorr_pk"]  = xcorr_peak(y_true, y_cand, max_lag=max_lag)
    s["dtw"]       = dtw_distance(y_true, y_cand)
    s["jaccard"]   = jaccard_above(y_true, y_cand, thr_thr_mbps)
    return s

def _offline_judge(true: pd.DataFrame, cg: pd.DataFrame, ig: pd.DataFrame,
                   thr_thr_mbps: float, max_lag_s: float) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Returns ('CG'|'IG'|'TIE', metrics)."""
    m_cg = _offline_score(true, cg, thr_thr_mbps, max_lag_s)
    m_ig = _offline_score(true, ig, thr_thr_mbps, max_lag_s)

    # Win “votes” across metrics: lower is better except xcorr, jaccard (higher better).
    votes_cg = 0
    votes_ig = 0
    # MAE
    votes_cg += (m_cg["mae"] < m_ig["mae"])
    votes_ig += (m_ig["mae"] < m_cg["mae"])
    # AUC diff
    votes_cg += (m_cg["auc_diff"] < m_ig["auc_diff"])
    votes_ig += (m_ig["auc_diff"] < m_cg["auc_diff"])
    # DTW
    votes_cg += (m_cg["dtw"] < m_ig["dtw"])
    votes_ig += (m_ig["dtw"] < m_cg["dtw"])
    # XCorr peak (higher better)
    votes_cg += (m_cg["xcorr_pk"] > m_ig["xcorr_pk"])
    votes_ig += (m_ig["xcorr_pk"] > m_cg["xcorr_pk"])
    # Jaccard (higher better)
    votes_cg += (m_cg["jaccard"] > m_ig["jaccard"])
    votes_ig += (m_ig["jaccard"] > m_cg["jaccard"])

    if votes_cg > votes_ig:
        return "CG", {"CG": m_cg, "IG": m_ig}
    elif votes_ig > votes_cg:
        return "IG", {"CG": m_cg, "IG": m_ig}
    else:
        return "TIE", {"CG": m_cg, "IG": m_ig}

def _make_judge_prompt(true_md: str, cg_md: str, ig_md: str) -> str:
    return (
        "You are a strict evaluator. You will be given three short analytical reports:\n"
        "1) a REFERENCE report describing the true counterfactual time series; and\n"
        "2) two CANDIDATE reports (CG and IG) describing alternative time series.\n\n"
        "Task: Decide which candidate (CG or IG) is semantically closer to the REFERENCE report.\n"
        "Judge on magnitude, temporal patterns (rises/dips), burstiness, and time-over-threshold behavior.\n"
        "Respond ONLY in JSON with fields: {\"winner\": \"CG\"|\"IG\"|\"TIE\", \"reason\": \"<one sentence>\"}.\n\n"
        "=== REFERENCE ===\n"
        f"{true_md}\n\n"
        "=== CANDIDATE: CG ===\n"
        f"{cg_md}\n\n"
        "=== CANDIDATE: IG ===\n"
        f"{ig_md}\n\n"
        "JSON:"
    )

def _llm_judge(true_md: str, cg_md: str, ig_md: str, model: str="gpt-4o-mini") -> Optional[Dict]:
    openai = _maybe_openai()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if openai is None or not api_key:
        return None
    openai.api_key = api_key

    prompt = _make_judge_prompt(true_md, cg_md, ig_md)
    try:
        # Compatible with openai>=1.0 “responses” API; if your env uses legacy,
        # replace with ChatCompletion create.
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You evaluate reports and must return strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        # normalize
        w = str(data.get("winner", "")).upper()
        if w not in {"CG", "IG", "TIE"}:
            w = "TIE"
        return {"winner": w, "reason": data.get("reason", "")}
    except Exception as e:
        return None

def main():
    ap = argparse.ArgumentParser(description="Compare CG vs IG to TRUE using analytic reports + (optional) LLM judge.")
    ap.add_argument("--true", required=True, help="CSV for true counterfactual TS")
    ap.add_argument("--cg",   required=True, help="CSV for CG TS")
    ap.add_argument("--ig",   required=True, help="CSV for IG TS")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--thr-thr-mbps", type=float, default=5.0, help="Throughput threshold for thresholded metrics")
    ap.add_argument("--thr-dly-ms",   type=float, default=20.0, help="Delay threshold (ms)")
    ap.add_argument("--max-lag-s", type=float, default=1.0, help="XCorr window in seconds")
    ap.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="LLM judge model (if OPENAI_API_KEY set)")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    # Load
    df_true = select_window(read_ts(args.true), args.tmin, args.tmax)
    df_cg   = select_window(read_ts(args.cg),   args.tmin, args.tmax)
    df_ig   = select_window(read_ts(args.ig),   args.tmin, args.tmax)

    # Reports (deterministic, no API needed)
    rep_true = render_report(df_true, "True Counterfactual", args.thr_thr_mbps, args.thr_dly_ms)
    rep_cg   = render_report(df_cg,   "CG (ours)",          args.thr_thr_mbps, args.thr_dly_ms)
    rep_ig   = render_report(df_ig,   "IG (interventional)",args.thr_thr_mbps, args.thr_dly_ms)

    with open(os.path.join(args.outdir, "report_true.md"), "w") as f: f.write(rep_true + "\n")
    with open(os.path.join(args.outdir, "report_cg.md"),   "w") as f: f.write(rep_cg   + "\n")
    with open(os.path.join(args.outdir, "report_ig.md"),   "w") as f: f.write(rep_ig   + "\n")

    # LLM judge if possible; otherwise offline judge
    llm_result = _llm_judge(rep_true, rep_cg, rep_ig, model=args.judge_model)
    if llm_result is not None:
        decision = {"mode": "llm", **llm_result}
    else:
        winner, metrics = _offline_judge(df_true, df_cg, df_ig, args.thr_thr_mbps, args.max_lag_s)
        decision = {"mode": "offline", "winner": winner, "reason": "Metric majority vote", "metrics": metrics}

    with open(os.path.join(args.outdir, "judge_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)

    # Compact human-friendly summary (for paper appendix / sanity)
    summary = []
    summary.append("# Semantic comparison (TRUE vs CG vs IG)")
    summary.append("")
    summary.append(f"- Judge mode: **{decision['mode']}**")
    summary.append(f"- Winner: **{decision['winner']}**")
    if "reason" in decision and decision["reason"]:
        summary.append(f"- Reason: {decision['reason']}")
    summary.append("")
    summary.append("## True report (excerpt)")
    summary.append("\n".join(rep_true.splitlines()[:8]) + "\n")
    summary.append("## CG report (excerpt)")
    summary.append("\n".join(rep_cg.splitlines()[:8]) + "\n")
    summary.append("## IG report (excerpt)")
    summary.append("\n".join(rep_ig.splitlines()[:8]) + "\n")

    # If offline, add the metric table
    if decision["mode"] == "offline" and "metrics" in decision:
        md = []
        md.append("\n## Offline judge metrics (throughput series)")
        md.append("")
        md.append("| Metric | CG | IG | Better |")
        md.append("|---|---:|---:|:--:|")
        M = decision["metrics"]
        # lower-better
        def lb_row(name, key):
            cg = M["CG"][key]; ig = M["IG"][key]
            better = "CG" if cg < ig else ("IG" if ig < cg else "Tie")
            md.append(f"| {name} | {cg:.4g} | {ig:.4g} | {better} |")
        # higher-better
        def hb_row(name, key):
            cg = M["CG"][key]; ig = M["IG"][key]
            better = "CG" if cg > ig else ("IG" if ig > cg else "Tie")
            md.append(f"| {name} | {cg:.4g} | {ig:.4g} | {better} |")

        lb_row("MAE (↓)", "mae")
        lb_row("AUC diff (↓)", "auc_diff")
        lb_row("DTW (↓)", "dtw")
        hb_row("XCorr peak (↑)", "xcorr_pk")
        hb_row("Jaccard@thr (↑)", "jaccard")
        summary.append("\n".join(md))

    with open(os.path.join(args.outdir, "summary.md"), "w") as f:
        f.write("\n".join(summary) + "\n")

    print(f"[✓] Wrote {os.path.join(args.outdir, 'report_true.md')}")
    print(f"[✓] Wrote {os.path.join(args.outdir, 'report_cg.md')}")
    print(f"[✓] Wrote {os.path.join(args.outdir, 'report_ig.md')}")
    print(f"[✓] Wrote {os.path.join(args.outdir, 'judge_decision.json')}")
    print(f"[✓] Wrote {os.path.join(args.outdir, 'summary.md')}")
    print(f"Winner: {decision['winner']} (mode={decision['mode']})")

if __name__ == "__main__":
    main()
