"""Charts for the Melaka 2026 simulation (reads data/processed, writes reports/*.png).

Colours: reference categorical slots 1-3 (validated), light surface.
  PACT (BN+PN) = blue, PH = red, Bersatu/Putra = aqua.
  Red vs aqua is in the colour-blind warning band, so every segment carries a text label.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("data/processed")
REP = Path("reports")
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL = {"PACT": "#2a78d6", "PH": "#e34948", "BSTU": "#1baf7a"}
LAB = {"PACT": "BN + PN pact", "PH": "Pakatan Harapan", "BSTU": "Bersatu / Putra"}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
                     "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)


def seat_totals():
    tot = pd.read_csv(OUT / "seat_totals_sims_2026.csv")
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURF)
    style(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    bins = np.arange(0, 29) - 0.5
    for k in ("PACT", "PH"):
        share = np.bincount(tot[k], minlength=29)[:28] / len(tot) * 100
        x = np.arange(28)
        ax.bar(x, share, width=0.8, color=COL[k], alpha=0.9, edgecolor=SURF, linewidth=1.2)
        m = tot[k].mean()
        ax.text(m, share.max() * 1.07, f"{LAB[k]}\nmean {m:.1f}", ha="center", va="bottom",
                color=INK, fontsize=10, fontweight="bold")
    ax.axvline(14.5, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(14.7, ax.get_ylim()[1] * 0.9, "majority\n(15 seats)", color=INK2, fontsize=9, va="top")
    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.set_xlabel("Seats won (out of 28)")
    ax.set_ylabel("% of 20,000 simulations")
    ax.set_title("How many seats each side wins across simulations",
                 loc="left", fontsize=13, fontweight="bold", pad=14)
    fig.text(0.01, 0.01, "Scenario model, not a forecast. Transfer rate and Bersama share are assumptions.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(REP / "seat_totals.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def seat_probs():
    p = pd.read_csv(OUT / "seat_probabilities_2026.csv")
    p["rest"] = p[["BSTU", "BERSAMA"]].sum(axis=1)
    p = p.sort_values(["PACT", "PH"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=SURF)
    style(ax)
    y = np.arange(len(p))
    left = np.zeros(len(p))
    small = [[] for _ in range(len(p))]  # per-row labels for segments too narrow to hold text
    for k, col, short in (("PACT", "PACT", "Pact"), ("PH", "PH", "PH"), ("rest", "BSTU", "Other")):
        v = p[k].values * 100
        ax.barh(y, v, left=left, height=0.72, color=COL[col], edgecolor=SURF, linewidth=1.5)
        for yi, (l, w) in enumerate(zip(left, v)):
            if w >= 12:  # wide enough: label inside the segment
                ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                        color="#ffffff" if col != "BSTU" else INK, fontweight="bold")
            elif w > 0.05:  # narrow: goes in the right-hand column instead
                small[yi].append(f"{short} {'<1' if w < 1 else f'{w:.0f}'}%")
        left += v
    # right-hand column: percentages for narrow segments, so every non-zero share is readable
    for yi, labels in enumerate(small):
        if labels:
            ax.text(101.5, yi, "  ".join(labels), ha="left", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(s)}. {n.title()}" for s, n in zip(p["seat"], p["name"])], fontsize=9, color=INK2)
    ax.set_xlim(0, 100)  # the right-hand label column draws outside this range
    ax.set_xlabel("Win probability (%)")
    ax.set_title("Win probability by seat", loc="left", fontsize=13, fontweight="bold", pad=34)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("PACT", "PH", "BSTU")]
    ax.legend(handles, [LAB["PACT"], LAB["PH"], "Bersatu / Putra / Bersama"], ncol=3, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)  # leave room for the label column
    fig.savefig(REP / "seat_probabilities.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    REP.mkdir(exist_ok=True)
    seat_totals()
    seat_probs()
    print("wrote", sorted(x.name for x in REP.glob("*.png")))


if __name__ == "__main__":
    sys.exit(main())
