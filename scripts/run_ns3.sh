#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NS3_ROOT:-}" ]]; then
  echo "NS3_ROOT is not set"
  exit 1
fi

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 NUM_UES SCHEDULER TRAFFIC_MBPS DURATION RNG_RUN OUTPUT_JSON"
  exit 1
fi

NUM_UES="$1"
SCHEDULER="$2"
TRAFFIC="$3"
DURATION="$4"
RNG="$5"
OUT="$6"

if [[ -x "${NS3_ROOT}/ns3" ]]; then
  "${NS3_ROOT}/ns3" run ran-sim -- --numUes="${NUM_UES}" --scheduler="${SCHEDULER}"         --trafficMbps="${TRAFFIC}" --duration="${DURATION}" --rngRun="${RNG}" --output="${OUT}"
else
  "${NS3_ROOT}/waf" --run "ran-sim --numUes=${NUM_UES} --scheduler=${SCHEDULER} --trafficMbps=${TRAFFIC} --duration=${DURATION} --rngRun=${RNG} --output=${OUT}"
fi

echo "Wrote ${OUT}"
