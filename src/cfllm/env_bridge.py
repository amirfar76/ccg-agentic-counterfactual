# cfllm/env_bridge.py
import os
import json
import subprocess
import tempfile
from typing import Optional

# cfllm/env_bridge.py (excerpt)
def run_ns3_action(action: dict, rng_run: int, workdir: str,
                   output_json: Optional[str] = None,
                   timeseries_csv: Optional[str] = None,
                   ts_dt: Optional[float] = None) -> dict:
    """
    Build argv like:
      ... --rngRun=SEED --output=metrics.json --tsCsv=PATH --tsDt=DT
    """
    import subprocess, shlex, json, os, tempfile

    bin_path = os.environ["CFCTWIN_NS3_BIN"]  # already set by you
    argv = [
        bin_path,
        f"--numUes={int(action['numUes'])}",
        f"--scheduler={action['scheduler']}",
        f"--trafficMbps={float(action['trafficMbps'])}",
        f"--duration={float(action['duration'])}",
        f"--rngRun={int(rng_run)}",
    ]
    if output_json:
        argv.append(f"--output={output_json}")
    if timeseries_csv:
        argv.append(f"--tsCsv={timeseries_csv}")
    if ts_dt is not None and ts_dt > 0:
        argv.append(f"--tsDt={float(ts_dt)}")

    env = os.environ.copy()
    # Optionally add extra flags via CFCTWIN_NS3_EXTRA (space-separated)
    extra = env.get("CFCTWIN_NS3_EXTRA", "").strip()
    if extra:
        argv.extend(shlex.split(extra))

    print("[ns3] Running:", " ".join(argv))
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"NS-3 run failed (code {proc.returncode}):\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")

    # read metrics if created
    metrics = {}
    if output_json and os.path.isfile(output_json):
        with open(output_json, "r") as f:
            metrics = json.load(f)
    return metrics