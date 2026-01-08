# cfllm/plots/plot_cf_from_csv.py
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

LEGEND_MAP_DEFAULT = {
    "ours_est_env": "GM-SCM",
    "naive_fresh_env": "Interventional",
    "low_twin": "Low-fidelity twin",
    "medium_twin": "Medium-fidelity twin",
}

def main():
    ap = argparse.ArgumentParser(description="Post-process CF eval CSVs into publication-quality PDFs (no recompute).")
    ap.add_argument("--runs", type=str, required=True,
                    help="Comma-separated list of run folders that contain cf_env_est_distances.csv")
    ap.add_argument("--out", type=str, required=True, help="Output PDF path")
    ap.add_argument("--drop-temps", type=str, default="0.01",
                    help="Comma-separated temperatures to remove (exact string match), e.g. '0.01'")
    ap.add_argument("--legend-map", type=str, default="",
                    help="Optional mapping 'raw1=Pretty 1,raw2=Pretty 2' to override labels")
    args = ap.parse_args()

    # LaTeX font look
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    # Load and stack
    dfs = []
    for run in [s.strip() for s in args.runs.split(",") if s.strip()]:
        csv_path = os.path.join(run, "cf_env_est_distances.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        df["__run__"] = os.path.basename(os.path.normpath(run))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Filter out temperatures
    drops = [t.strip() for t in args.drop_temps.split(",") if t.strip()]
    if "temperature" in df.columns and drops:
        df = df[~df["temperature"].astype(str).isin(drops)].copy()

    # Build legend map
    legend_map = dict(LEGEND_MAP_DEFAULT)
    if args.legend_map:
        for pair in args.legend_map.split(","):
            if "=" in pair:
                raw, pretty = pair.split("=", 1)
                legend_map[raw.strip()] = pretty.strip()

    # If multiple runs contain different "method" names (e.g., twin runs), keep them all
    # Group by (temperature, method) and average distance
    agg = df.groupby(["temperature", "method"], as_index=False)["distance"].mean()
    # Sort by temperature numeric
    agg = agg.sort_values("temperature")

    # Plot
    plt.figure()
    for method, sub in agg.groupby("method"):
        label = legend_map.get(method, method)
        plt.plot(sub["temperature"].values, sub["distance"].values, marker="o", label=label)

    plt.xlabel(r"Temperature $\tau$")
    plt.ylabel(r"Average distance to the factual report $Y$")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    plt.close()
    print(f"[✓] Wrote {args.out}")

if __name__ == "__main__":
    main()

