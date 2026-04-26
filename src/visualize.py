"""
visualize.py – bar charts and radar plots from aggregated MOS scores.
"""
from pathlib import Path
import math

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "gtts":     "#f472b6",
    "coqui":    "#38bdf8",
    "speecht5": "#4ade80",
}

# ── bar chart ─────────────────────────────────────────────────────────────────
def plot_bar(agg: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.patch.set_facecolor("#0f1923")

    for ax, lang in zip(axes, ["ar", "en"]):
        ax.set_facecolor("#0f1923")
        sub = agg[agg["lang"] == lang].copy()
        if sub.empty:
            ax.set_title(f"{lang.upper()} — no data", color="#94a3b8")
            continue

        x = np.arange(len(sub))
        width = 0.35
        colors = [PALETTE.get(s, "#888") for s in sub["system"]]

        bars_c = ax.bar(x - width/2, sub["clarity"],  width, label="Clarity",
                        color=[c + "cc" for c in colors])
        bars_n = ax.bar(x + width/2, sub["naturalness"], width, label="Naturalness",
                        color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["system"], color="#94a3b8", fontsize=11)
        ax.set_ylim(0, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(colors="#64748b")
        ax.set_title(f"{'Arabic' if lang=='ar' else 'English'}", color="#e2e8f0",
                     fontsize=14, pad=10)
        ax.set_ylabel("MOS Score (1–5)", color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        ax.yaxis.label.set_color("#94a3b8")
        ax.grid(axis="y", color="#1e293b", linewidth=0.8)

        for bar in list(bars_c) + list(bars_n):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f"{h:.1f}",
                    ha="center", va="bottom", color="#e2e8f0", fontsize=9)

    handles = [
        mpatches.Patch(color="#aaaaaa55", label="Clarity"),
        mpatches.Patch(color="#aaaaaa", label="Naturalness"),
    ]
    fig.legend(handles=handles, loc="upper right", framealpha=0.2,
               labelcolor="#e2e8f0", facecolor="#1e293b")
    fig.suptitle("MOS Score Comparison — Clarity & Naturalness", color="#e2e8f0",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "bar_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1923")
    plt.close(fig)
    print(f"  [✓] bar chart saved → {out}")
    return out


# ── radar chart ───────────────────────────────────────────────────────────────
def plot_radar(agg: pd.DataFrame) -> Path:
    dims = ["clarity", "naturalness", "arabic_accuracy"]
    dim_labels = ["Clarity", "Naturalness", "Arabic\nAccuracy"]
    N = len(dims)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#64748b", size=8)
    ax.set_ylim(0, 5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, color="#94a3b8", size=11)
    ax.spines["polar"].set_color("#1e293b")
    ax.grid(color="#1e293b", linewidth=0.8)

    for system, grp in agg.groupby("system"):
        row = grp.iloc[0]
        vals = [row.get(d, 0) or 0 for d in dims]
        vals += vals[:1]
        color = PALETTE.get(system, "#888")
        ax.plot(angles, vals, color=color, linewidth=2, label=system)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
              facecolor="#1e293b", labelcolor="#e2e8f0", framealpha=0.8)
    ax.set_title("Score Profile per System", color="#e2e8f0", size=14, pad=20)
    out = PLOTS_DIR / "radar_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1923")
    plt.close(fig)
    print(f"  [✓] radar chart saved → {out}")
    return out


def generate_all_plots(results_csv: str = "scores/results.csv") -> None:
    df = pd.read_csv(results_csv)
    agg = df.groupby(["system", "lang"])[["clarity","naturalness","arabic_accuracy"]].mean().round(2)
    agg = agg.reset_index()
    plot_bar(agg)
    plot_radar(agg)


if __name__ == "__main__":
    generate_all_plots()
