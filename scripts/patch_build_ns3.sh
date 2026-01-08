#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NS3_ROOT:-}" ]]; then
  echo "NS3_ROOT is not set. Source your .env or export NS3_ROOT first."
  exit 1
fi

echo "[*] Using NS3_ROOT=${NS3_ROOT}"

# Copy the scenario into scratch
mkdir -p "${NS3_ROOT}/scratch"
cp -f "$(dirname "$0")/../ns3/ran-sim.cc" "${NS3_ROOT}/scratch/ran-sim.cc"
echo "[*] Copied ran-sim.cc into ${NS3_ROOT}/scratch/"

# Try building by asking ns3 runner to build 'ran-sim' target, fallback to waf
if [[ -x "${NS3_ROOT}/ns3" ]]; then
  echo "[*] Building via ns3 runner"
  pushd "${NS3_ROOT}" >/dev/null
  ./ns3 build scratch/ran-sim
  popd >/dev/null
elif [[ -x "${NS3_ROOT}/waf" ]]; then
  echo "[*] Building via waf"
  pushd "${NS3_ROOT}" >/dev/null
  ./waf build
  popd >/dev/null
else
  echo "[-] Could not find ns3 or waf in ${NS3_ROOT}"
  exit 1
fi

echo "[+] Build attempt complete."
