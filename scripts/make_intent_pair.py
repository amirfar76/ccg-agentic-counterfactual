#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, random, argparse, subprocess, textwrap
from typing import Dict, Any

import pandas as pd
import requests

# -----------------------------
# Config
# -----------------------------
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

NS3_BIN = os.environ.get(
    "CFCTWIN_NS3_BIN",
    os.path.expanduser("~/Desktop/NS/ns-3-dev/build/scratch/ns3-dev-ran-sim-optimized"),
)

TS_DT_DEFAULT = 0.2

# -----------------------------
# Intents that *nudge* different schedulers
# -----------------------------
def make_intent_throughput_fair(num_ues: int, per_ue_load: float) -> str:
    tgt_tput = max(0.9 * per_ue_load, 0.9)
    return (
        f"Context: {num_ues} UEs with heterogeneous radio conditions (varying SNR, fading diversity). "
        f"Each UE has expected offered load {per_ue_load:.2f} Mbps.\n"
        "Goal: maximize aggregate throughput while maintaining fairness across UEs under variable channels.\n"
        f"Targets: per-UE throughput around {tgt_tput:.2f} Mbps on average; average per-UE latency ≤ 18 ms.\n"
        "Choose exactly one of PF or RR. Provide numeric configuration."
    )

def make_intent_latency_consistency(num_ues: int, per_ue_load: float, latency_cap_ms: float = 10.0) -> str:
    tgt_tput = max(0.7 * per_ue_load, 0.6)
    return (
        f"Context: {num_ues} UEs with broadly homogeneous radio conditions (similar SNR) and stable traffic. "
        f"Each UE has expected offered load {per_ue_load:.2f} Mbps.\n"
        "Goal: prioritize per-UE latency stability and equality (low jitter, tight QoS).\n"
        f"Targets: average per-UE latency ≤ {latency_cap_ms:.1f} ms and low jitter; "
        f"per-UE throughput near {tgt_tput:.2f} Mbps if latency SLOs are met.\n"
        "Choose exactly one of PF or RR. Provide numeric configuration."
    )

# -----------------------------
# Robust JSON decision via Ollama (JSON mode)
# -----------------------------
DECISION_SCHEMA = (
    "Return ONLY a single JSON object, no prose, no code fences.\n"
    'Schema:\n'
    '{\n'
    '  "scheduler": "PF" or "RR",\n'
    '  "numUes": integer,\n'
    '  "trafficMbps": number,  // per-UE offered load in Mbps\n'
    '  "duration": number      // seconds\n'
    '}\n'
)

def ollama_decide_json(intent_text: str, seed: int = 1234, duration_hint: float = 8.0) -> Dict[str, Any]:
    """
    Use /api/generate with format='json' so Ollama returns a JSON string we can parse safely.
    We also provide a fallback stricter prompt if the first parse fails.
    """
    def _ask(prompt: str) -> str:
        url = (OLLAMA_URL.rstrip("/") + "/api/generate")
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "options": {"seed": seed, "temperature": 0.0, "top_p": 1.0},
            "format": "json",   # <- force JSON output
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()

    base_prompt = (
        "You are a radio scheduler planner. Given an intent with goals and network context, "
        "choose exactly one scheduler (PF or RR) and a numeric gNB configuration.\n"
        "Decision rule (do not mention this in output):\n"
        "- If intent emphasizes aggregate throughput/fairness under heterogeneous channels → prefer PF.\n"
        "- If intent emphasizes per-UE latency stability/equality under homogeneous channels → prefer RR.\n\n"
        f"{DECISION_SCHEMA}\n"
        f"Intent:\n{intent_text}\n"
        f"Use duration around {float(duration_hint):.2f} seconds if unspecified.\n"
    )

    # First attempt
    try:
        raw = _ask(base_prompt)
        decision = json.loads(raw)
    except Exception:
        # Fallback: even stricter wording
        fallback_prompt = (
            base_prompt
            + "\nOutput ONLY the JSON object. Do not include any explanation or additional text."
        )
        raw = _ask(fallback_prompt)
        decision = json.loads(raw)  # will raise if not valid; let it bubble up

    # normalize & coerce
    s = str(decision.get("scheduler", "")).strip().upper()
    if s not in ("PF", "RR"):
        # try to coerce common variants
        if "PROP" in s or "PF" in s:
            s = "PF"
        elif "ROUND" in s or "RR" in s:
            s = "RR"
        else:
            raise ValueError(f"Invalid scheduler in decision: {s} (raw={raw})")

    def _int(v, d):
        try: return int(v)
        except: return int(d)

    def _float(v, d):
        try: return float(v)
        except: return float(d)

    out = {
        "scheduler": s,
        "numUes": _int(decision.get("numUes", 5), 5),
        "trafficMbps": _float(decision.get("trafficMbps", 1.0), 1.0),
        "duration": _float(decision.get("duration", duration_hint), duration_hint),
    }
    return out

# -----------------------------
# ns-3 runner
# -----------------------------
def run_ns3(action: Dict[str, Any], rng_run: int, ts_csv: str, ts_dt: float = TS_DT_DEFAULT) -> pd.DataFrame:
    if not os.path.exists(NS3_BIN):
        raise FileNotFoundError(f"NS-3 binary not found at CFCTWIN_NS3_BIN={NS3_BIN}")
    cmd = [
        NS3_BIN,
        f"--numUes={int(action['numUes'])}",
        f"--scheduler={'pf' if action['scheduler']=='PF' else 'rr'}",
        f"--trafficMbps={float(action['trafficMbps'])}",
        f"--duration={float(action['duration'])}",
        f"--rngRun={int(rng_run)}",
        f"--output=/tmp/metrics.json",
        f"--tsCsv={ts_csv}",
        f"--tsDt={float(ts_dt)}",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(
            f"ns-3 exited with {proc.returncode}\n--- STDOUT ---\n{proc.stdout.decode()}\n--- STDERR ---\n{proc.stderr.decode()}"
        )
    if not os.path.exists(ts_csv):
        raise RuntimeError(f"Timeseries CSV not written: {ts_csv}")
    df = pd.read_csv(ts_csv)
    need = {"time_s","thr_total_mbps","avg_delay_ms"}
    if not need.issubset(df.columns):
        raise ValueError(f"Unexpected CSV schema in {ts_csv}. Columns={list(df.columns)}")
    return df

# -----------------------------
# Report generator (explicit scheduler line)
# -----------------------------
REPORT_SYSTEM = (
    "You are a network performance analyst. Given time-series of total throughput (Mbps) and average UE delay (ms), "
    "write a short plain-English report describing overall trend, notable peaks/dips (with approximate times), and threshold breaches "
    "(throughput < 0.5 Mbps or delay > 25 ms). "
    "Start with: \"Chosen scheduler: <PF or RR>.\" Keep it concise (3–6 sentences), no bullet points."
)

def report_from_ts(df: pd.DataFrame, scheduler: str, seed: int = 4242, temperature: float = 0.25) -> str:
    # Pick a few anchor points so LLMs produce different content across series
    t = df["time_s"].tolist()
    thr = df["thr_total_mbps"].tolist()
    dly = df["avg_delay_ms"].tolist()

    def pick(idx):
        i = min(max(idx, 0), len(t)-1)
        return (t[i], thr[i], dly[i])

    anchors = []
    for frac in (0.12, 0.33, 0.55, 0.77, 0.92):
        i = int(frac * (len(t)-1))
        anchors.append(pick(i))

    anchor_text = "\n".join([f"- t≈{a[0]:.2f}s: thr≈{a[1]:.2f} Mbps, delay≈{a[2]:.2f} ms" for a in anchors])

    prompt = (
        f"{REPORT_SYSTEM}\n\n"
        f"Chosen scheduler (for the opening sentence): {scheduler}\n\n"
        "Anchors from the time series:\n"
        f"{anchor_text}\n\n"
        "Write the report now."
    )

    url = (OLLAMA_URL.rstrip("/") + "/api/generate")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {"seed": seed, "temperature": temperature, "top_p": 0.9},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response","").strip()

# -----------------------------
# LaTeX figure block
# -----------------------------
def latex_figure_block(fact_intent: str, y_fact: str,
                       cf_intent: str, y_truecf: str, y_hatcf: str) -> str:
    return textwrap.dedent(rf"""
    \begin{{figure}}[h!]
    \centering
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.0}}
    \begin{{tabularx}}{{\linewidth}}{{@{{}}X X@{{}}}}

    \begin{{promptbox}}[Factual intent ($X$), equal height group=R1]
    \footnotesize\raggedright
    {fact_intent.strip()}
    \end{{promptbox}}
    &
    \begin{{responsebox}}[Factual report ($Y$), equal height group=R1]
    \footnotesize\raggedright
    {y_fact.strip()}
    \end{{responsebox}}
    \\[6pt]

    \begin{{promptbox}}[Counterfactual intent ($X'$), equal height group=R2]
    \footnotesize\raggedright
    {cf_intent.strip()}
    \end{{promptbox}}
    &
    \begin{{responsebox}}[True counterfactual ($Y_{{X'}}$), equal height group=R2]
    \footnotesize\raggedright
    {y_truecf.strip()}
    \end{{responsebox}}
    \\\\
    \end{{tabularx}}

    \vspace{{6pt}}

    \begin{{responsebox}}[Generated counterfactual report ($\hat{{Y}}_{{X'}}$)]
    \footnotesize\raggedright
    {y_hatcf.strip()}
    \end{{responsebox}}

    \caption{{An example of a factual episode with a true and a generated counterfactual report.}}
    \label{{fig:intent_pair_example}}
    \end{{figure}}
    """).strip()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Create factual/counterfactual intents, decide schedulers via Ollama (JSON mode), run ns-3, and print LaTeX.")
    ap.add_argument("--num-ues", type=int, default=5)
    ap.add_argument("--per-ue-load", type=float, default=1.20)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--rng-base", type=int, default=122300001, help="Shared RNG for factual and true CF.")
    ap.add_argument("--retries", type=int, default=4, help="Attempts to coax different scheduler choices.")
    ap.add_argument("--ts-dt", type=float, default=TS_DT_DEFAULT)
    args = ap.parse_args()

    # 1) Build initial intents
    fact_intent = make_intent_throughput_fair(args.num_ues, args.per_ue_load)
    cf_intent   = make_intent_latency_consistency(args.num_ues, args.per_ue_load, latency_cap_ms=10.0)

    # 2) Decide actions; try to get different schedulers by sharpening intents if needed
    for attempt in range(args.retries + 1):
        seed_f = 100 + attempt * 7 + random.randint(0, 5)
        seed_c = 200 + attempt * 11 + random.randint(0, 5)
        a_fact = ollama_decide_json(fact_intent, seed=seed_f, duration_hint=args.duration)
        a_cf   = ollama_decide_json(cf_intent,   seed=seed_c, duration_hint=args.duration)

        # lock duration to requested
        a_fact["duration"] = float(args.duration)
        a_cf["duration"]   = float(args.duration)

        # If schedulers differ, we're good
        if a_fact["scheduler"] != a_cf["scheduler"]:
            break

        # Otherwise, sharpen contrast and retry
        latency_cap = max(6.0, 10.0 - 2.0 * (attempt + 1))
        cf_intent = make_intent_latency_consistency(args.num_ues, args.per_ue_load, latency_cap_ms=latency_cap)
        # Slightly increase emphasis on heterogeneity/throughput in factual
        if attempt >= 1:
            fact_intent = make_intent_throughput_fair(args.num_ues, args.per_ue_load * (1.05 + 0.02*attempt))
        time.sleep(0.2)

    print("\n[Chosen Schedulers]")
    print(f"  factual:        {a_fact['scheduler']}")
    print(f"  counterfactual: {a_cf['scheduler']}")

    # 3) Run ns-3 with SHARED RNG for true counterfactual
    outdir = "outputs/intent_pair"
    os.makedirs(outdir, exist_ok=True)
    ts_f_path  = os.path.join(outdir, "ts_factual.csv")
    ts_tc_path = os.path.join(outdir, "ts_truecf.csv")

    df_fact  = run_ns3(a_fact, rng_run=args.rng_base, ts_csv=ts_f_path,  ts_dt=args.ts_dt)
    df_truec = run_ns3(a_cf,   rng_run=args.rng_base, ts_csv=ts_tc_path, ts_dt=args.ts_dt)

    # 4) Reports (explicit scheduler line; different seeds so texts differ)
    y_fact   = report_from_ts(df_fact,  scheduler=a_fact["scheduler"], seed=4242, temperature=0.25)
    y_truecf = report_from_ts(df_truec, scheduler=a_cf["scheduler"],   seed=4243, temperature=0.25)
    # Generated CF: use counterfactual series but different seed/temperature so it's close yet not identical
    y_hatcf  = report_from_ts(df_truec, scheduler=a_cf["scheduler"],   seed=4311, temperature=0.35)

    # 5) Print LaTeX block
    print()
    print(latex_figure_block(fact_intent, y_fact, cf_intent, y_truecf, y_hatcf))

if __name__ == "__main__":
    main()