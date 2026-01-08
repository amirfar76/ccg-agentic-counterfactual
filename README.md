# Conformal Counterfactual Generation (CCG) for LLM-Based Autonomous Control

This repo contains an end-to-end research pipeline for **counterfactual generation** in
closed-loop **LLM agent ↔ environment** systems, with a wireless control case study using **ns-3**.

It supports three counterfactual estimators used in the paper:
- **IG**: re-execute the full agent+real environment under the edited prompt.
- **SIG**: re-execute the agent with a **digital twin** simulator.
- **CG**: **abduct** environment noise from a factual episode and replay the LLM decoding
  with **shared exogenous noise** (Gumbel-Max), then simulate the counterfactual environment.

On top, it includes **conformal calibration (CCG)** to produce *sets* of counterfactual reports with
reliability guarantees (test-time scaling + accept/stop rules).

> Notes
> - The code uses **Ollama** for local LLM serving by default, but the LLM wrapper is isolated.
> - ns-3 must be installed separately.

---

## Repository layout

- `src/cfllm/` – Python package (LLM wrappers, ns-3 bridge, CG/IG/SIG, NPE/abduction, conformal)
- `ns3/` – ns-3 scenario (`ran-sim.cc`)
- `scripts/` – helper scripts (build/patch ns-3, reproduction helpers)
- `outputs/` – generated artifacts (**git-ignored**)
- `models/` – trained posterior/NPE artifacts (**git-ignored**)
- `docs/` – additional notes

---

## Quickstart

### 0) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env (NS3_ROOT, CFCTWIN_NS3_BIN, optionally OLLAMA_*)
```

### 1) Patch/build the ns-3 scenario

```bash
source .env
chmod +x scripts/patch_build_ns3.sh
./scripts/patch_build_ns3.sh
```

### 2) Sanity run ns-3 (no Python)

```bash
"$CFCTWIN_NS3_BIN" --numUes=5 --scheduler=rr --trafficMbps=2.0 --duration=5.0 --rngRun=1 --output=/tmp/metrics.json
cat /tmp/metrics.json
```

### 3) Run the main experiments (entrypoints)

The package contains several entrypoints used during development. The most relevant ones are:

- **KPI fidelity / CG vs IG vs SIG**: `python -m cfllm.eval_cf_compare`
- **Abduction/NPE-based counterfactual eval**: `python -m cfllm.eval_cf_npe`
- **Conformal set generation experiments**: `python -m cfllm.run_conformal_cf_experiments`

Run `--help` on any module for the available flags.

---

## Configuration

Settings are read from environment variables (see `.env.example`) and `src/cfllm/config.py`.
Key variables:
- `NS3_ROOT` – ns-3 checkout (for patch/build)
- `CFCTWIN_NS3_BIN` – built `ran-sim` binary path
- `OUTPUT_DIR` – where to write generated outputs
- `OLLAMA_HOST`, `OLLAMA_MODEL` – LLM serving

---

## License
MIT (see `LICENSE`).
