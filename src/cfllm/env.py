import json
import os
import subprocess
import tempfile
import shutil

def run_ns3_env(
    rng_run: int,
    num_ues: int,
    scheduler: str,
    traffic_mbps: float,
    duration_s: float,
    bandwidth_mhz: float,
) -> dict:
    """
    Run the ns-3 RAN simulator with a given RNG seed and scenario parameters.
    Returns parsed JSON metrics.
    """

    ns3_bin = os.environ.get("CFCTWIN_NS3_BIN")
    if not ns3_bin or not os.path.exists(os.path.expanduser(ns3_bin)):
        raise RuntimeError(
            "CFCTWIN_NS3_BIN is not set or does not point to an existing binary.\n"
            "Set it first, e.g.:\n"
            '  export CFCTWIN_NS3_BIN="$HOME/Desktop/NS/ns-3-dev/build/scratch/ns3-dev-ran-sim-optimized"'
        )

    out_dir = tempfile.mkdtemp(prefix="ns3_run_")
    out_json = os.path.join(out_dir, "metrics.json")

    cmd = [
        os.path.expanduser(ns3_bin),
        f"--rngRun={int(rng_run)}",
        f"--numUes={int(num_ues)}",
        f"--scheduler={scheduler}",
        f"--trafficMbps={float(traffic_mbps)}",
        f"--duration={float(duration_s)}",
        f"--bandwidthMHz={float(bandwidth_mhz)}",
        f"--output={out_json}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["LC_ALL"] = "C"

    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(out_json, "r") as f:
            data = json.load(f)
        return data
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
