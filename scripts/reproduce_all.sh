#!/usr/bin/env bash
set -euo pipefail

# One-shot reproduction helper.
# Assumes: python deps installed, .env configured, ns-3 binary built.

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

echo "[*] Using CFCTWIN_NS3_BIN=${CFCTWIN_NS3_BIN:-<unset>}"

# 1) Generate any required datasets (if your workflow uses them)
# python -m cfllm.data_gen_calib --help

# 2) Train abduction/NPE model (if required)
# python -m cfllm.train_npe_seed --help

# 3) Run KPI fidelity evaluation (CG vs IG vs SIG)
python -m cfllm.eval_cf_compare --help

# 4) Run conformal experiments
python -m cfllm.run_conformal_cf_experiments --help

echo "[✓] Done (edit this script to match your exact paper experiments)."
