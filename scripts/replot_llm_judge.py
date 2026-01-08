#!/usr/bin/env python3
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # path argument or default
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/llm_judge_batch/results.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    out_dir = os.path.dirname(path)
    base = "llm_judge_bar_fraction"
    out_pdf = os.path.join(out_dir, f"{base}.pdf")
    out_png = os.path.join(out_dir, f"{base}.png")

    # read and normalize
    df = pd.read_csv(path)
    if "verdict" not in df.columns:
        raise KeyError(f"'verdict' column not in CSV. Columns: {list(df.columns)}")

    winners = (
        df["verdict"].astype(str).str.strip().str.upper()
        .replace({"C.G.":"CG", "I.G.":"IG"})
    )

    counts = winners.value_counts()
    # ensure both bars present
    for k in ("CG", "IG", "TIE"):
        if k not in counts:
            counts.loc[k] = 0
    counts = counts[["CG","IG","TIE"]]

    # drop Tie if empty
    if counts.loc["TIE"] == 0:
        counts = counts.drop(index="TIE")

    # convert to fractions
    fractions = counts / counts.sum()

    # LaTeX fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax = fractions.plot(kind="bar", ax=ax, color=["#4472C4", "#ED7D31", "#A5A5A5"])

    ax.set_xlabel("Method")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[✓] Saved {out_pdf}")
    print(f"[✓] Saved {out_png}")
    print("\nFractions:\n", fractions.to_string())

if __name__ == "__main__":
    main()