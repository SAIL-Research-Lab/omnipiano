"""PicturesGreatKiev (Mussorgsky, The Great Gate of Kiev; 720 steps/episode)
N-hand PPO baselines.

Unlike the WinterWind set, these are homogeneous: one algorithm (PPO), 2-3
seeds per configuration. Curves are therefore the across-seed mean with a
min-max band, not single runs.

Encoding — colour = morphology (3 / 4 hands), linestyle = variant
(Prototype solid, StaticPartition dashed). Budget differs by morphology
(3-hand 5M, 4-hand 8M) and is stated in the legend.

eval_freq = 50,000 env steps (verified: 100 evals = 5M, 160 evals = 8M).
"""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = "/home/accelerator/SafeRoboPianist/examples/logs"
OUT = "/home/accelerator/SafeRoboPianist/paper/figures"
os.makedirs(OUT, exist_ok=True)
EVAL_FREQ = 50_000

C3, C4 = "#2a78d6", "#eb6834"                    # validated categorical slots
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

# (glob for the seed dirs, label, colour, linestyle)
GROUPS = [
    ("ppo_3hand_proto_gamma08_seed*_1",
     "3-hand Prototype · PPO 5M", C3, "-"),
    ("ppo_sb3_baseline_picturesgreatkiev_threehand_staticpartition_seed*_1",
     "3-hand StaticPartition · PPO 5M", C3, "--"),
    ("ppo_sb3_baseline_picturesgreatkiev_fourhandprototype_seed*_1",
     "4-hand Prototype · PPO 8M", C4, "-"),
    ("ppo_sb3_baseline_picturesgreatkiev_fourhand_staticpartition_seed*_1",
     "4-hand StaticPartition · PPO 8M", C4, "--"),
]


def series(pattern, col):
    """Per-seed curves for one configuration, truncated to the shortest."""
    out = []
    for d in sorted(glob.glob(f"{LOGS}/{pattern}")):
        cs = glob.glob(f"{d}/eval_episode_metrics_*.csv")
        if not cs:
            continue
        f = max(cs, key=lambda p: sum(1 for _ in open(p)))
        rows = list(csv.reader(open(f)))
        h = [x.strip() for x in rows[0]]
        i = h.index(col)
        y = [float(r[i]) for r in rows[1:] if r[i].strip() != ""]
        if len(y) >= 20:
            out.append(np.array(y))
    if not out:
        return None, None, None, None, 0
    n = min(len(y) for y in out)
    Y = np.stack([y[:n] for y in out])
    x = np.arange(1, n + 1) * EVAL_FREQ / 1e6
    return x, Y.mean(0), Y.min(0), Y.max(0), len(out)


def smooth(y, w=5):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def panel(ax, col, ylabel):
    for pat, lab, c, ls in GROUPS:
        x, m, lo, hi, k = series(pat, col)
        if x is None:
            continue
        ms, los, his = smooth(m), smooth(lo), smooth(hi)
        xs = x[len(x) - len(ms):]
        ax.fill_between(xs, los, his, color=c, alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, ms, color=c, ls=ls, lw=1.8, zorder=3,
                label=f"{lab}  (n={k})")
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 8.1)


fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
panel(axes[0], "ep_f1", "note F1  (periodic eval)")
panel(axes[1], "ep_return", "episode return")
axes[0].text(0.03, 0.965, "band = min–max across seeds",
             transform=axes[0].transAxes, fontsize=7.5, color=INK2, va="top")
axes[0].text(0.03, 0.895, "5-hand: not registered on this piece",
             transform=axes[0].transAxes, fontsize=7.5, color=INK2, va="top")

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16),
           handlelength=2.0, columnspacing=1.6)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figK_kiev_nhand.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/figK_kiev_nhand.[pdf|png]")
