# cfllm/llm.py
# Tiny shim that routes to GMSCM wrappers (or other engines in the future).

import os
import json
from typing import Dict, Any, Optional, Tuple

from .gmscm import gmscm_action_pair, gmscm_report_pair

# Defaults via env (optionally override per call)
_ENGINE = os.environ.get("CFLLM_ENGINE", "gmscm")
_GMSCM_MODEL = os.environ.get("CFLLM_GMSCM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# System prompts
ACTION_SYSTEM = (
    "You are an LTE simulation planner. Given a user requirement, produce a compact JSON with keys:\n"
    "  num_ues (int), scheduler (pf|rr|mt), traffic_mbps (per-UE, float), duration_s (float).\n"
    "Only output valid JSON. No commentary."
)
REPORT_SYSTEM = (
    "You are a technical writer. Given KPIs and metadata, write a short, factual summary for a lab notebook. "
    "Avoid hype; include totals, per-UE throughput and average delay if available."
)

def _stringify_metrics(metrics: Dict[str, Any]) -> str:
    # Canonical text the report model sees; stable across seeds.
    return json.dumps(metrics, sort_keys=True, indent=2)

# --------------------------------------------------------------------
# Public API used across the repo
# --------------------------------------------------------------------

def action_from_prompt(user: str, seed: int = 0, temperature: float = 0.3) -> Any:
    """
    Returns LLM text (often JSON-like). Callers are expected to normalize downstream.
    """
    if _ENGINE == "gmscm":
        a_text, _, _ = gmscm_action_pair(
            user, user,
            model_name=_GMSCM_MODEL,
            seed=seed,
            temperature=temperature,
            top_p=0.95,
            top_k=None,
            max_new_tokens=160,
        )
        # Try best-effort JSON parse; OK if it fails (downstream normalizer can handle strings)
        try:
            return json.loads(a_text)
        except Exception:
            return a_text
    else:
        raise RuntimeError(f"Unsupported CFLLM_ENGINE: '{_ENGINE}'")

def report_from_metrics(metrics: Dict[str, Any], seed: int = 0, temperature: float = 0.7) -> str:
    """
    Turns KPI dict -> short natural-language report using the selected engine.
    Seeding ensures (same seed, same prompt) => same text (our 'same-noise' path).
    """
    facts = _stringify_metrics(metrics)
    if _ENGINE == "gmscm":
        y, _, _ = gmscm_report_pair(
            REPORT_SYSTEM,
            facts, facts,
            model_name=_GMSCM_MODEL,
            seed=seed,
            temperature=temperature,
            top_p=0.95,
            top_k=None,
            max_new_tokens=220,
        )
        return y
    else:
        raise RuntimeError(f"Unsupported CFLLM_ENGINE: '{_ENGINE}'")
