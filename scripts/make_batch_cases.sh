#!/usr/bin/env bash
set -e

CASES=20  # number of test cases
OUTROOT="outputs/cases"
ACTION_JSON='{"num_ues":5,"scheduler":"rr","traffic_mbps":8,"duration_s":8}'
NPE_MODEL="models/npe_seed.pt"

mkdir -p "${OUTROOT}"

for i in $(seq 1 $CASES); do
    CASE_DIR="${OUTROOT}/case_$(printf "%03d" $i)"
    mkdir -p "${CASE_DIR}"

    # pick a random base seed for true run
    BASE_SEED=$((RANDOM * 10000 + i))
    INTV_SEED=$((RANDOM * 10000 + 100000 + i))

    echo "[INFO] Generating case $i (true_seed=$BASE_SEED)"

    # Run plot_cf_timeseries to produce all three time series
    python -m cfllm.plot_cf_timeseries \
      --action-json "${ACTION_JSON}" \
      --rng-true "${BASE_SEED}" \
      --rng-est infer \
      --factual-csv "${CASE_DIR}/factual.csv" \
      --npe-model "${NPE_MODEL}" \
      --rng-intv "${INTV_SEED}" \
      --ts-dt 0.2 \
      --tmin 1.0 \
      --outdir "${CASE_DIR}" \
      --use_tex

    # Move files to standard names for the judge
    mv "${CASE_DIR}/true_timeseries.csv" "${CASE_DIR}/true_timeseries.csv"
    mv "${CASE_DIR}/cg_timeseries.csv"   "${CASE_DIR}/cg_timeseries.csv"
    mv "${CASE_DIR}/ig_timeseries.csv"   "${CASE_DIR}/ig_timeseries.csv"
done
