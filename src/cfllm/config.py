"""Project configuration.

This module centralizes environment variables and default paths.

Design goals:
- No hard-coded user-specific paths.
- Defaults work when running from the repo root.
- All overrides happen via environment variables.

Key env vars:
- NS3_ROOT: path to your ns-3 checkout (used by patch/build script).
- CFCTWIN_NS3_BIN: path to the built ran-sim binary (used by Python runners).
- OUTPUT_DIR: where to write results (default: <repo>/outputs).
- OLLAMA_HOST, OLLAMA_MODEL: LLM serving settings.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root (…/src/cfllm/config.py -> parents[2] is repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(REPO_ROOT / "outputs"))).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "data").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "calibration").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "test").mkdir(parents=True, exist_ok=True)

# ns-3
NS3_ROOT = os.getenv("NS3_ROOT", "").rstrip("/")  # used mainly by scripts/patch_build_ns3.sh
NS3_BIN = os.getenv("CFCTWIN_NS3_BIN", "").strip()  # path to built binary for Python calls

# LLM serving
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# Defaults for common CLI flags
DEFAULT_ALPHA = float(os.getenv("ALPHA", "0.1"))
DEFAULT_METRIC = os.getenv("METRIC", "rouge")
DEFAULT_MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "50"))
