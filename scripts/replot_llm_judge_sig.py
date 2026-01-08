#!/usr/bin/env python3
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    in_csv  = sys.argv[1] if len(sys.argv) > 1 else "outputs/llm_judge_batch/results.csv"
    outbase = sys.argv[2] if len(sys.argv) > 2 else "Figures/llm_judge_bar_fraction"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)

    col = None
    for c in ("verdict","winner"):
        if c in df.columns:
            col = c
            break
    if col is None:
        raise KeyError(f"Expected a 'verdict' or 'winner' column. Found: {list(df.columns)}")

    winners = (
        df[col].astype(str).str.strip().str.upper()
          .replace({
              "C.G.":"CG", "I.G.":"IG", "S.I.G.":"SIG",
              "COUNTERFACTUAL GENERATION":"CG",
              "INTERVENTIONAL GENERATION":"IG",
              "SIMULATOR INTERVENTIONAL GENERATION":"SIG",
          })
    )

    counts = winners.value_counts()
    total = max(int(counts.sum()), 1)

    for k in ("CG","IG","SIG","TIE"):
        if k not in counts:
            counts.loc[k] = 0

    order = ["CG","IG","SIG"] + (["TIE"] if counts.get("TIE",0) > 0 else [])
    counts = counts[order]
    fracs  = counts / total

    # LaTeX fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    fracs.plot(kind="bar", ax=ax)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Method")
    ax.set_ylabel("Fraction of wins")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    # annotate percentages
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{100*h:.0f}\\%",
                        (p.get_x()+p.get_width()/2., h),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0,3), textcoords="offset points")

    os.makedirs(os.path.dirname(outbase), exist_ok=True)
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    fig.savefig(outbase + ".png", dpi=300, bbox_inches="tight")

    print(f"[✓] Saved {outbase}.pdf")
    print(f"[✓] Saved {outbase}.png")
    print("\nFractions:\n", fracs.to_string())

if __name__ == "__main__":
    main()
