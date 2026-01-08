#!/usr/bin/env python3
import os, re, json, argparse, subprocess, textwrap, time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import csv
import math
import requests
from requests.exceptions import ReadTimeout, ConnectionError as ReqConnError

# ---- Config ----
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
NS3_BIN = os.environ.get(
    "CFCTWIN_NS3_BIN",
    os.path.expanduser("~/Desktop/NS/ns-3-dev/build/scratch/ns3-dev-ran-sim-optimized"),
)

OUT_ROOT = "outputs/intent_case"
os.makedirs(OUT_ROOT, exist_ok=True)

# RNG policy
RNG_FACTUAL = 122300001
RNG_TRUE_CF = RNG_FACTUAL  # keep same randomness sheet across factual/true-CF

# --------------------- Data classes ---------------------
@dataclass
class Intent:
    num_ues: int
    per_ue_load_mbps: float
    tgt_thr_mbps_per_ue: float
    tgt_lat_ms_per_ue: float
    text: str

@dataclass
class Action:
    numUes: int
    scheduler: str  # "RR" or "PF"
    trafficMbps: float  # per-UE offered load
    duration: float

# --------------------- Ollama helpers ---------------------
def ping_ollama(url: str, timeout: float) -> None:
    try:
        r = requests.get(f"{url}/api/tags", timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Ollama is not responding at {url}. Start it with:\n"
            f"  ollama serve\n"
            f"and ensure the model exists:\n"
            f"  ollama pull {OLLAMA_MODEL}\n"
            f"Original error: {repr(e)}"
        )

def ollama_generate(prompt: str, seed: int, temperature: float, top_p: float,
                    url: str, model: str, timeout: float,
                    max_retries: int = 4, backoff: float = 1.5) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
        },
        "stream": False,
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except (ReadTimeout, ReqConnError, requests.HTTPError) as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(backoff ** attempt)
    raise RuntimeError(
        f"Ollama generate failed after {max_retries} attempts. "
        f"Last error: {repr(last_err)}"
    )

def llm_choose_scheduler_from_intent(intent_text: str, seed: int,
                                     url: str, model: str, timeout: float) -> str:
    """LLM picks PF or RR; we enforce only PF/RR output."""
    sys_prompt = (
        "You are selecting a 5G MAC scheduler for a single-cell gNB. "
        "You must output exactly one token: PF or RR.\n"
        "- PF (Proportional Fair): favors higher throughput when moderate latency is acceptable.\n"
        "- RR (Round Robin): prioritizes fairness and more stable latency when tight latency targets matter.\n"
        "Return only PF or RR. No explanation."
    )
    user_prompt = f"Intent:\n{intent_text}\n\nChoose the scheduler (PF or RR) now:"
    full = f"{sys_prompt}\n\n{user_prompt}"
    out = ollama_generate(full, seed=seed, temperature=0.0, top_p=0.9,
                          url=url, model=model, timeout=timeout)
    m = re.search(r"\b(PF|RR)\b", out)
    if not m:
        # heuristic fallback: tight latency => RR, else PF
        m_lat = re.search(r"latenc(?:y|ies)\s*of\s*([0-9]+(?:\.[0-9]+)?)\s*ms", intent_text, re.I)
        if m_lat and float(m_lat.group(1)) <= 10.0:
            return "RR"
        return "PF"
    return m.group(1)

# --------------------- Intent parsing / construction ---------------------
def parse_intent_text(intent: str) -> Intent:
    m = re.search(r"have\s+(\d+)\s+UEs", intent, re.I)
    num_ues = int(m.group(1)) if m else 5

    m = re.search(r"expected\s+traffic\s+load\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*Mbps", intent, re.I)
    load = float(m.group(1)) if m else 1.0

    m = re.search(r"throughputs?\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*Mbps", intent, re.I)
    tgt_thr = float(m.group(1)) if m else load

    m = re.search(r"latenc(?:y|ies)\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*ms", intent, re.I)
    tgt_lat = float(m.group(1)) if m else 12.0

    return Intent(
        num_ues=num_ues,
        per_ue_load_mbps=load,
        tgt_thr_mbps_per_ue=tgt_thr,
        tgt_lat_ms_per_ue=tgt_lat,
        text=intent.strip()
    )

def make_counterfactual_intent(fact: Intent) -> Intent:
    """Nudge targets: +10% throughput, -20% latency (min 5 ms)."""
    cf_thr = round(fact.tgt_thr_mbps_per_ue * 1.10, 2)
    cf_lat = max(5.0, round(fact.tgt_lat_ms_per_ue * 0.80, 2))
    cf_text = (
        f"Given that we have {fact.num_ues} UEs and that each UE has an expected traffic load of "
        f"{fact.per_ue_load_mbps:.2f} Mbps, choose a scheduling algorithm and target "
        f"per-UE throughputs of {cf_thr:.2f} Mbps and per-UE latencies of {cf_lat:.2f} ms."
    )
    return Intent(
        num_ues=fact.num_ues,
        per_ue_load_mbps=fact.per_ue_load_mbps,
        tgt_thr_mbps_per_ue=cf_thr,
        tgt_lat_ms_per_ue=cf_lat,
        text=cf_text
    )

# --------------------- ns-3 bridge ---------------------
def run_ns3(action: Action, rng_run: int, ts_csv: str, metrics_json: str) -> None:
    os.makedirs(os.path.dirname(ts_csv), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_json), exist_ok=True)

    cmd = [
        NS3_BIN,
        f"--numUes={action.numUes}",
        f"--scheduler={action.scheduler.lower()}",
        f"--trafficMbps={action.trafficMbps}",
        f"--duration={action.duration}",
        f"--rngRun={rng_run}",
        f"--tsCsv={ts_csv}",
        f"--tsDt=0.2",
        f"--output={metrics_json}",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ns-3 failed (code {proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

# --------------------- CSV stats for prompting ---------------------
def summarize_timeseries(ts_csv: str) -> Dict[str, Any]:
    times, thr, dly = [], [], []
    with open(ts_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            thr.append(float(row["thr_total_mbps"]))
            dly.append(float(row["avg_delay_ms"]))
    def argmax(xs): return max(range(len(xs)), key=lambda i: xs[i]) if xs else 0
    def argmin(xs): return min(range(len(xs)), key=lambda i: xs[i]) if xs else 0
    out = {}
    if times:
        out.update({
            "n": len(times),
            "t0": times[0],
            "t1": times[-1],
            "thr_mean": sum(thr)/len(thr),
            "thr_max": max(thr), "thr_tmax": times[argmax(thr)],
            "dly_mean": sum(dly)/len(dly),
            "dly_max": max(dly), "dly_tmax": times[argmax(dly)],
            "thr_breach": any(x > 20.0 for x in thr),
            "dly_breach": any(x > 30.0 for x in dly),
        })
    return out

# --------------------- Reporting ---------------------
def make_report_from_timeseries(ts_csv: str, meta: Dict[str, Any], seed: int, temperature: float,
                                url: str, model: str, timeout: float,
                                max_rows: int = 120) -> str:
    """
    Build a grounded prompt:
    - include scheduler, UE count, load, duration (so LLM can state the chosen scheduler)
    - include simple stats (means, peaks) computed from CSV
    - include first rows of CSV (kept short) to anchor details
    """
    # compute stats
    stats = summarize_timeseries(ts_csv)

    # small CSV head (helps latency & avoids timeouts)
    lines = []
    with open(ts_csv, "r") as f:
        for i, line in enumerate(f):
            lines.append(line)
            if i >= max_rows:
                break
    head = "".join(lines)

    # build prompt
    prompt = textwrap.dedent(f"""
    You are the reporting agent for a 5G experiment.
    Configuration (for this run):
      - Scheduler: {meta.get("scheduler")}
      - Number of UEs: {meta.get("numUes")}
      - Per-UE offered load (Mbps): {meta.get("trafficMbps")}
      - Duration (s): {meta.get("duration")}

    CSV columns: time_s, thr_total_mbps, avg_delay_ms

    Grounding summary from the data:
      - time range: {stats.get("t0","?")} to {stats.get("t1","?")} (n={stats.get("n","?")} samples)
      - mean throughput: {stats.get("thr_mean","?"):.2f} Mbps; max {stats.get("thr_max","?"):.2f} Mbps at t={stats.get("thr_tmax","?")} s
      - mean delay: {stats.get("dly_mean","?"):.2f} ms; max {stats.get("dly_max","?"):.2f} ms at t={stats.get("dly_tmax","?")} s
      - threshold breaches: throughput>20 Mbps? {stats.get("thr_breach", False)}; delay>30 ms? {stats.get("dly_breach", False)}

    First rows of the timeseries:
    {head}

    Write a concise 3–5 sentence report that:
      1) **Begins** with the exact phrase: "Scheduler: {meta.get("scheduler")}".
      2) Summarizes overall trend and notable peaks/dips, referencing approximate times.
      3) States whether thresholds were breached.
      4) Avoids boilerplate; write in your own words based on the numbers above.
    """).strip()

    return ollama_generate(prompt, seed=seed, temperature=temperature, top_p=0.9,
                           url=url, model=model, timeout=timeout)

# --------------------- LaTeX figure ---------------------
def latex_block(fact_intent: str, cf_intent: str, y_fact: str, y_truecf: str, y_cg: str) -> str:
    return textwrap.dedent(f"""
    % Auto-generated figure block
    \\begin{{figure}}[h!]
    \\centering
    \\small
    \\setlength{{\\tabcolsep}}{{6pt}}
    \\renewcommand{{\\arraystretch}}{{1.0}}
    \\begin{{tabularx}}{{\\linewidth}}{{@{{}}X X@{{}}}}

    \\begin{{promptbox}}[Factual prompt ($X$), equal height group=R1]
    \\footnotesize\\raggedright
    {fact_intent}
    \\end{{promptbox}}
    &
    \\begin{{responsebox}}[Factual report ($Y$), equal height group=R1]
    \\footnotesize\\raggedright
    {y_fact}
    \\end{{responsebox}}
    \\\\[6pt]

    \\begin{{promptbox}}[Counterfactual prompt ($X'$), equal height group=R2]
    \\footnotesize\\raggedright
    {cf_intent}
    \\end{{promptbox}}
    &
    \\begin{{responsebox}}[True counterfactual ($Y_{{X'}}$), equal height group=R2]
    \\footnotesize\\raggedright
    {y_truecf}
    \\end{{responsebox}}
    \\\\
    \\end{{tabularx}}

    \\vspace{{6pt}}

    \\begin{{responsebox}}[Generated counterfactual report ($\\hat{{Y}}_{{X'}}$)]
    \\footnotesize\\raggedright
    {y_cg}
    \\end{{responsebox}}

    \\caption{{An example factual episode with true and generated counterfactual reports.}}
    \\label{{fig:example_prompt_1}}
    \\end{{figure}}
    """).strip()

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", type=str, required=True, help="Factual intent text")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--ollama-url", type=str, default=None)
    ap.add_argument("--ollama-timeout", type=float, default=180.0)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    url = args.ollama_url or OLLAMA_URL
    model = args.model or OLLAMA_MODEL
    timeout = float(args.ollama_timeout)

    # Parse intents
    fact = parse_intent_text(args.intent)
    cf = make_counterfactual_intent(fact)

    # Choose schedulers via LLM (or fallback heuristic)
    if not args.no_llm:
        ping_ollama(url, timeout=min(10.0, timeout))
        sched_fact = llm_choose_scheduler_from_intent(fact.text, seed=101, url=url, model=model, timeout=timeout)
        sched_cf   = llm_choose_scheduler_from_intent(cf.text,   seed=102, url=url, model=model, timeout=timeout)
    else:
        sched_fact = "RR" if fact.tgt_lat_ms_per_ue <= 10.0 else "PF"
        sched_cf   = "RR" if cf.tgt_lat_ms_per_ue   <= 10.0 else "PF"

    # Actions
    act_fact = Action(numUes=fact.num_ues, scheduler=sched_fact, trafficMbps=fact.per_ue_load_mbps, duration=args.duration)
    act_cf   = Action(numUes=cf.num_ues,   scheduler=sched_cf,   trafficMbps=cf.per_ue_load_mbps,   duration=args.duration)

    # Paths
    fact_ts = os.path.join(OUT_ROOT, "fact_timeseries.csv")
    fact_metrics = os.path.join(OUT_ROOT, "fact_metrics.json")
    truecf_ts = os.path.join(OUT_ROOT, "true_cf_timeseries.csv")
    truecf_metrics = os.path.join(OUT_ROOT, "true_cf_metrics.json")

    # Run simulations (same RNG sheet, different config -> different series)
    run_ns3(act_fact, rng_run=RNG_FACTUAL, ts_csv=fact_ts, metrics_json=fact_metrics)
    run_ns3(act_cf,   rng_run=RNG_TRUE_CF, ts_csv=truecf_ts, metrics_json=truecf_metrics)

    # Reports (now explicitly include scheduler + stats)
    if not args.no_llm:
        meta_fact = {"scheduler": act_fact.scheduler, "numUes": act_fact.numUes,
                     "trafficMbps": act_fact.trafficMbps, "duration": act_fact.duration}
        meta_cf   = {"scheduler": act_cf.scheduler,   "numUes": act_cf.numUes,
                     "trafficMbps": act_cf.trafficMbps, "duration": act_cf.duration}

        # Factual report
        y_fact   = make_report_from_timeseries(fact_ts,   meta=meta_fact, seed=4242, temperature=0.2,
                                               url=url, model=model, timeout=timeout)

        # True counterfactual report (deterministic-ish)
        y_truecf = make_report_from_timeseries(truecf_ts, meta=meta_cf,   seed=4242, temperature=0.1,
                                               url=url, model=model, timeout=timeout)

        # Generated counterfactual report: same CF series but different seed/temp
        # (close in content, not identical wording)
        y_cg     = make_report_from_timeseries(truecf_ts, meta=meta_cf,   seed=4343, temperature=0.5,
                                               url=url, model=model, timeout=timeout)
    else:
        y_fact = "(LLM disabled)"
        y_truecf = "(LLM disabled)"
        y_cg = "(LLM disabled)"

    # Save manifest
    manifest = {
        "factual_intent": fact.text,
        "counterfactual_intent": cf.text,
        "scheduler_factual": act_fact.scheduler,
        "scheduler_counterfactual": act_cf.scheduler,
        "action_factual": act_fact.__dict__,
        "action_counterfactual": act_cf.__dict__,
        "rng_factual": RNG_FACTUAL,
        "rng_true_cf": RNG_TRUE_CF,
        "fact_timeseries_csv": fact_ts,
        "true_cf_timeseries_csv": truecf_ts,
        "factual_report": y_fact,
        "true_cf_report": y_truecf,
        "generated_cf_report": y_cg,
    }
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Print LaTeX block
    print(latex_block(fact.text, cf.text, y_fact, y_truecf, y_cg))

if __name__ == "__main__":
    main()