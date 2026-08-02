"""Paper figures for the OmniPiano robust leg.

Encoding contract (kept identical across every figure):
  colour  = noise distribution   Gaussian #2a78d6 / Uniform #eb6834 / Shift #1baf7a
  facet   = perturbation channel
  grey    = the clean reference (a reference, never a series)
Palette validated with dataviz/scripts/validate_palette.js (light, --pairs all):
all checks PASS; the aqua contrast WARN is relieved by direct value labels.
"""
import csv, glob, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

FINAL = "/home/accelerator/SafeRoboPianist/paper/data/final_eval10"
LOGS = "/home/accelerator/SafeRoboPianist/examples/logs"
OUT = "/home/accelerator/SafeRoboPianist/paper/figures"
os.makedirs(OUT, exist_ok=True)

C_GAUSS, C_UNIF, C_SHIFT = "#2a78d6", "#eb6834", "#1baf7a"
DIST_C = {"Gaussian": C_GAUSS, "Uniform": C_UNIF, "Shift": C_SHIFT}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 8,
})

RUNS = {  # key -> (label, channel, distribution)
    "clean": ("Clean", "—", "—"),
    "a_gauss_p15": ("A-Gauss", "Action", "Gaussian"),
    "a_uniform_p15": ("A-Uniform", "Action", "Uniform"),
    "a_shift_p15": ("A-Shift", "Action", "Shift"),
    "o_gauss_p15": ("O-Gauss", "Observation", "Gaussian"),
    "o_uniform_p15": ("O-Uniform", "Observation", "Uniform"),
    "o_shift_p15": ("O-Shift", "Observation", "Shift"),
    "r_gauss_p50": ("R-Gauss", "Reward", "Gaussian"),
    "r_uniform_p50": ("R-Uniform", "Reward", "Uniform"),
    "r_shift_p50": ("R-Shift", "Reward", "Shift"),
    "ao_gauss_p15": ("AO-Gauss", "Action+Obs", "Gaussian"),
    "ar_uniform_a15_r50": ("AR-Uniform", "Action+Reward", "Uniform"),
    "or_shift_on15_rn50": ("OR-Shift", "Obs+Reward", "Shift"),
}


def _cols(path, *names):
    rows = list(csv.reader(open(path)))
    h = [x.strip() for x in rows[0]]
    return {n: [float(r[h.index(n)]) for r in rows[1:] if r[h.index(n)].strip() != ""]
            for n in names}


def final_stats():
    """mean/std over the 10 deterministic matched-eval episodes."""
    out = {}
    for k in RUNS:
        f = max(glob.glob(f"{FINAL}/{k}/**/*.csv", recursive=True),
                key=lambda p: sum(1 for _ in open(p)))
        c = _cols(f, "ep_f1", "ep_precision", "ep_recall")
        out[k] = dict(f1=st.mean(c["ep_f1"]), sd=st.pstdev(c["ep_f1"]),
                      prec=st.mean(c["ep_precision"]), rec=st.mean(c["ep_recall"]))
    return out


def train_curve(k, every=50_000):
    """Periodic-eval F1 over training; x = eval index * eval_freq."""
    f = max(glob.glob(f"{LOGS}/ppo_sb3_baseline_clairdelune_{k}_seed0_1/eval_episode_metrics_*.csv"),
            key=lambda p: sum(1 for _ in open(p)))
    y = _cols(f, "ep_f1")["ep_f1"]
    x = np.arange(1, len(y) + 1) * every / 1e6
    return x, np.array(y)


def smooth(y, w=5):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


S = final_stats()
CLEAN = S["clean"]["f1"]


# ===========================================================================
# Fig 1 — main results: F1 by channel x distribution
# ===========================================================================
def fig1():
    chans = ["Action", "Observation", "Reward"]
    dists = ["Gaussian", "Uniform", "Shift"]
    keys = {(RUNS[k][1], RUNS[k][2]): k for k in RUNS if RUNS[k][1] in chans}

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    w, gap = 0.24, 0.06
    for j, d in enumerate(dists):
        xs, ys, es = [], [], []
        for i, ch in enumerate(chans):
            k = keys[(ch, d)]
            xs.append(i + (j - 1) * (w + gap))
            ys.append(S[k]["f1"]); es.append(S[k]["sd"])
        b = ax.bar(xs, ys, w, yerr=es, capsize=2.5, color=DIST_C[d], label=d,
                   error_kw=dict(ecolor=INK2, lw=0.8), zorder=3)
        for x, y, e in zip(xs, ys, es):      # direct labels = contrast relief
            ax.text(x, y + e + 0.022, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)

    ax.axhline(CLEAN, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(2.52, CLEAN + 0.012, f"clean {CLEAN:.2f}", fontsize=7.5,
            color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels([f"{c}\n(σ/level {'0.50' if c=='Reward' else '0.15'})" for c in chans])
    ax.set_ylabel("note F1  (matched eval)")
    ax.set_ylim(0, 0.79)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.005, 1.02), ncol=3,
              columnspacing=1.2, handlelength=1.1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1_main_f1.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Fig 2 — learning dynamics: eval F1 over training, faceted by channel
# ===========================================================================
def fig2():
    chans = [("Action", ["a_gauss_p15", "a_uniform_p15", "a_shift_p15"]),
             ("Observation", ["o_gauss_p15", "o_uniform_p15", "o_shift_p15"]),
             ("Reward", ["r_gauss_p50", "r_uniform_p50", "r_shift_p50"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
    xc, yc = train_curve("clean")
    for ax, (ch, ks) in zip(axes, chans):
        ax.plot(xc[len(xc) - len(smooth(yc)):], smooth(yc), color=MUTED, lw=1.1,
                ls=(0, (4, 3)), zorder=2)
        for k in ks:
            d = RUNS[k][2]
            x, y = train_curve(k)
            ys = smooth(y)
            ax.plot(x, y, color=DIST_C[d], lw=0.6, alpha=0.22, zorder=3)
            ax.plot(x[len(x) - len(ys):], ys, color=DIST_C[d], lw=1.6,
                    label=d, zorder=4)
        ax.set_title(ch, fontsize=9, color=INK, pad=4)
        ax.set_xlabel("env steps (M)")
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 5.05)
    axes[0].set_ylabel("note F1")
    axes[0].set_ylim(0, 0.78)
    axes[0].text(0.15, CLEAN + 0.015, "clean", fontsize=7.5, color=INK2)
    axes[-1].legend(loc="lower right", handlelength=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig2_learning_curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Fig 3 — compound perturbations vs their single-channel components
# ===========================================================================
def fig3():
    # NOTE: the OR-Shift panel is deliberately absent until its correct-direction
    # controls finish training. OR-Shift-ON15-RN50 perturbs obs by -0.15 and
    # reward by -0.50, so its components are O-Shift-N15 / R-Shift-N50 — NOT the
    # canonical +0.15 / +0.50 tasks. Plotting those would compare against the
    # wrong sign. Re-add as:
    #   ("Observation + Reward", "Shift", "or_shift_on15_rn50",
    #    "o_shift_n15", "r_shift_n50"),
    # once both runs land (and add them to RUNS).
    groups = [
        ("Action + Observation", "Gaussian", "ao_gauss_p15", "a_gauss_p15", "o_gauss_p15"),
        ("Action + Reward", "Uniform", "ar_uniform_a15_r50", "a_uniform_p15", "r_uniform_p50"),
    ]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2, 2.6), sharey=True)
    for ax, (title, dist, kc, k1, k2) in zip(axes, groups):
        col = DIST_C[dist]
        vals = [S[k1]["f1"], S[k2]["f1"], S[kc]["f1"]]
        errs = [S[k1]["sd"], S[k2]["sd"], S[kc]["sd"]]
        labs = [RUNS[k1][0], RUNS[k2][0], RUNS[kc][0] + "\n(compound)"]
        cols = [col, col, col]
        alphas = [0.42, 0.42, 1.0]
        for i, (v, e, c, a) in enumerate(zip(vals, errs, cols, alphas)):
            ax.bar(i, v, 0.62, yerr=e, capsize=2.5, color=c, alpha=a,
                   error_kw=dict(ecolor=INK2, lw=0.8), zorder=3)
            ax.text(i, v + e + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
        ax.axhline(CLEAN, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7.5)
        ax.set_title(f"{title}  ·  {dist}", fontsize=8.5, color=INK, pad=4)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    axes[0].set_ylabel("note F1")
    axes[0].set_ylim(0, 0.79)
    axes[0].text(-0.42, CLEAN + 0.015, "clean", fontsize=7.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MUTED, alpha=0.42),
               plt.Rectangle((0, 0), 1, 1, fc=MUTED, alpha=1.0)]
    axes[-1].legend(handles, ["single channel", "both channels"],
                    loc="upper right", ncol=1, handlelength=1.1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig3_compound.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Fig 4 — conservatism: precision holds, recall collapses
# ===========================================================================
def fig4():
    order = ["clean", "a_shift_p15", "o_uniform_p15", "r_shift_p50", "o_gauss_p15",
             "o_shift_p15", "a_uniform_p15", "r_uniform_p50", "a_gauss_p15", "r_gauss_p50"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        d = RUNS[k][2]
        col = MUTED if d == "—" else DIST_C[d]
        p, r = S[k]["prec"], S[k]["rec"]
        ax.plot([r, p], [y, y], color=col, lw=1.6, alpha=0.55, zorder=3,
                solid_capstyle="round")
        ax.scatter([r], [y], s=42, facecolor="white", edgecolor=col, lw=1.6, zorder=4)
        ax.scatter([p], [y], s=42, color=col, zorder=4)
        ax.text(r - 0.022, y, f"{r:.2f}", ha="right", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([RUNS[k][0] for k in reversed(order)], fontsize=8)
    ax.set_xlabel("value  ·  matched eval")
    ax.set_xlim(0, 1.06)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor="white",
                      markeredgecolor=INK2, markersize=7, label="recall"),
               Line2D([], [], marker="o", ls="", color=INK2, markersize=7,
                      label="precision")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, -0.03),
              ncol=2, handlelength=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig4_conservatism.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


for fn in (fig1, fig2, fig3, fig4):
    fn(); print("ok", fn.__name__)
print("→", OUT)
