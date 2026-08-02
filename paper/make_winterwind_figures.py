"""WinterWind (Chopin Etude Op.25 No.11, 314 steps/episode) N-hand runs.

Diagnostic figure: is the existing WinterWind data usable as a morphology
ladder? Encoding — colour = morphology (3/4/5 hands), linestyle = variant
(Prototype solid, StaticPartition dashed). Algorithm and budget differ per
run and are stated in the legend, because that heterogeneity is the point.

Periodic-eval CSVs carry one row per eval episode; eval_freq = 10,000 env
steps (verified: 500 evals = 5M, 801 = 8M, 1001 = 10M), so
x = eval_index * 10_000.
"""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LOGS = "/home/accelerator/SafeRoboPianist/examples/logs"
OUT = "/home/accelerator/SafeRoboPianist/paper/figures"
os.makedirs(OUT, exist_ok=True)
EVAL_FREQ = 10_000

C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 7.5,
})

# (dir, label, colour, linestyle, complete?)
# Restricted to the four configurations under consideration for the
# morphology ladder. The two extra 4-hand *Prototype* runs (SAC 5M F1=0.417,
# TQC 10M F1=0.425) are omitted on purpose: they share (morphology, variant)
# with nothing else here, so under the colour=morphology / linestyle=variant
# contract they would be indistinguishable from each other.
RUNS = [
    ("sac_3hand_winter_wind_2", "3-hand Prototype · SAC 5M", C3, "-", True),
    ("sac_3hand_winter_wind_1", "3-hand Prototype · SAC (stopped 1.9M)", C3, "-", False),
    ("tqc_4hand_winter_wind_static_partition_1", "4-hand StaticPartition · TQC 5M", C4, "--", True),
    ("sac_4hand_winter_wind_static_partition_8M_1", "4-hand StaticPartition · SAC (stopped 1.6M of 8M)", C4, "--", False),
    ("tqc_5hand_winter_wind_static_partition_8M_1", "5-hand StaticPartition · TQC 8M", C5, "--", True),
]


def periodic(dirname, col):
    """Largest eval CSV in the dir = the periodic-eval log (the 11-row
    sibling is the final multi-episode eval)."""
    cs = glob.glob(f"{LOGS}/{dirname}/eval_episode_metrics_*.csv")
    if not cs:
        return None, None
    f = max(cs, key=lambda p: sum(1 for _ in open(p)))
    rows = list(csv.reader(open(f)))
    h = [x.strip() for x in rows[0]]
    i = h.index(col)
    y = [float(r[i]) for r in rows[1:] if r[i].strip() != ""]
    if len(y) < 20:                       # final-eval-only file, no curve
        return None, None
    x = np.arange(1, len(y) + 1) * EVAL_FREQ / 1e6
    return x, np.array(y)


def smooth(y, w=9):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def panel(ax, col, ylabel):
    for d, lab, c, ls, complete in RUNS:
        x, y = periodic(d, col)
        if x is None:
            continue
        ys = smooth(y)
        a_raw, a_line = (0.16, 1.0) if complete else (0.10, 0.45)
        ax.plot(x, y, color=c, lw=0.5, alpha=a_raw, zorder=2)
        ax.plot(x[len(x) - len(ys):], ys, color=c, ls=ls, lw=1.7,
                alpha=a_line, label=lab, zorder=3)
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
panel(axes[0], "ep_f1", "note F1  (periodic eval)")
panel(axes[1], "ep_return", "episode return")
axes[0].set_ylim(0, 0.55)

# 3-hand StaticPartition has never been trained on this piece — say so on
# the figure rather than letting its absence read as "not plotted".
axes[0].text(0.03, 0.965, "3-hand StaticPartition: no run exists",
             transform=axes[0].transAxes, fontsize=7.5, color=INK2,
             va="top")

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.20),
           handlelength=2.0, columnspacing=1.6)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figW_winterwind_nhand.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/figW_winterwind_nhand.[pdf|png]")
