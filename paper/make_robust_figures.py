"""Paper figures for the OmniPiano robust leg.

METRIC PRIORITY (decided 2026-08-28): **episode return is the benchmark's
primary metric; note F1 is secondary but always reported.** Every figure here
therefore carries both -- reward on the top row, F1 below -- rather than F1
alone as the first version did.

Encoding contract (identical across every figure):
  colour     = noise distribution   Gaussian #2a78d6 / Uniform #eb6834 / Shift #1baf7a
  facet      = perturbation channel
  row        = metric               reward (primary) above, F1 below
  linestyle  = measurement          solid = periodic deterministic eval,
                                    dashed = training rollout
  grey       = the clean reference (a reference, never a series)
Palette validated with dataviz/scripts/validate_palette.js (light, --pairs all):
all checks PASS; the aqua contrast WARN is relieved by direct value labels.

Which reward: ``ep_return`` -- the NOISED return the agent actually received.
``ep_return_true`` (the clean-reward diagnostic) is deliberately not plotted:
in deployment nobody can observe the unperturbed reward, so the curve a reader
should judge robustness by is the one the agent lived with.

Bars start at zero in both rows. Reward spans ~1480-2140, so a truncated axis
would exaggerate the differences; at full scale a 20% drop still reads clearly.
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
    # Negative-shift single-channel controls: these ARE the components of
    # OR-Shift-ON15-RN50 (obs -0.15, reward -0.50). The canonical
    # O-Shift-P15 / R-Shift-P50 use the opposite sign and are not.
    "o_shift_n15": ("O-Shift-N15", "Observation", "Shift"),
    "r_shift_n50": ("R-Shift-N50", "Reward", "Shift"),
}


def _cols(path, *names):
    rows = list(csv.reader(open(path)))
    h = [x.strip() for x in rows[0]]
    return {n: [float(r[h.index(n)]) for r in rows[1:] if r[h.index(n)].strip() != ""]
            for n in names}


def final_stats():
    """mean/std over the 10 deterministic matched-eval episodes, both metrics."""
    out = {}
    for k in RUNS:
        f = max(glob.glob(f"{FINAL}/{k}/**/*.csv", recursive=True),
                key=lambda p: sum(1 for _ in open(p)))
        c = _cols(f, "ep_f1", "ep_precision", "ep_recall", "ep_return")
        out[k] = dict(
            f1=st.mean(c["ep_f1"]), f1_sd=st.pstdev(c["ep_f1"]),
            ret=st.mean(c["ep_return"]), ret_sd=st.pstdev(c["ep_return"]),
            prec=st.mean(c["ep_precision"]), rec=st.mean(c["ep_recall"]))
    return out


def eval_curve(k, col, every=50_000):
    """Periodic deterministic eval over training; x = eval index * eval_freq."""
    f = max(glob.glob(f"{LOGS}/ppo_sb3_baseline_clairdelune_{k}_seed0_1/"
                      "eval_episode_metrics_*.csv"),
            key=lambda p: sum(1 for _ in open(p)))
    y = _cols(f, col)[col]
    return np.arange(1, len(y) + 1) * every / 1e6, np.array(y)


def rollout_curve(k):
    """Training-rollout reward from SB3's native progress.csv.

    This is the behaviour policy's own experience, not a held-out evaluation,
    and it is the only curve that exists at every training step. There is no
    rollout F1 counterpart -- neither SB3 nor OmniSafe computes F1 during
    collection -- which is why the F1 row below carries eval curves only.
    """
    p = f"{LOGS}/ppo_sb3_baseline_clairdelune_{k}_seed0_1/progress.csv"
    if not os.path.exists(p):
        return None, None
    rows = list(csv.DictReader(open(p)))
    xs, ys = [], []
    for r in rows:
        t, v = r.get("time/total_timesteps", ""), r.get("rollout/ep_rew_mean", "")
        if t.strip() and v.strip():
            try:
                xs.append(float(t) / 1e6); ys.append(float(v))
            except ValueError:
                pass
    return np.array(xs), np.array(ys)


def smooth(y, w=5):
    return y if len(y) < w else np.convolve(y, np.ones(w) / w, mode="valid")


# fig1/3/4 read FINAL (final-policy 10-episode eval); fig2 reads examples/logs
# only. FINAL is not in the repo -- regenerate it with the robust_eval_sweep.py
# command recorded in paper/robust_experiments_index.md. Gated so the module
# imports and fig2 still draws when it is absent.
HAVE_FINAL = os.path.isdir(FINAL) and bool(glob.glob(f"{FINAL}/*/"))
S = final_stats() if HAVE_FINAL else {}
CLEAN_F1, CLEAN_RET = (S["clean"]["f1"], S["clean"]["ret"]) if HAVE_FINAL else (None, None)

CHANS = ["Action", "Observation", "Reward"]
DISTS = ["Gaussian", "Uniform", "Shift"]

# The canonical v1 matrix cell for each (channel, distribution), listed
# EXPLICITLY. Deriving it from RUNS with a dict comprehension silently picked
# the wrong run: o_shift_n15 / r_shift_n50 carry the same (channel, dist) pair
# as o_shift_p15 / r_shift_p50 and, coming later in RUNS, overwrote them. The
# N-suffix runs are negative-shift DIAGNOSTIC CONTROLS -- they belong to fig3
# as the components of OR-Shift, never to the main matrix. (Bug found and
# fixed 2026-08-28; every earlier fig1 showed the controls in those two cells.)
MATRIX = {
    ("Action", "Gaussian"): "a_gauss_p15",
    ("Action", "Uniform"): "a_uniform_p15",
    ("Action", "Shift"): "a_shift_p15",
    ("Observation", "Gaussian"): "o_gauss_p15",
    ("Observation", "Uniform"): "o_uniform_p15",
    ("Observation", "Shift"): "o_shift_p15",
    ("Reward", "Gaussian"): "r_gauss_p50",
    ("Reward", "Uniform"): "r_uniform_p50",
    ("Reward", "Shift"): "r_shift_p50",
}


# ===========================================================================
# Fig 1 — main results: reward AND F1 by channel x distribution
# ===========================================================================
def fig1():
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    w, gap = 0.24, 0.06

    for row, (ax, metric, sdk, clean_v, ylab, ymax, fmt) in enumerate((
        (axes[0], "ret", "ret_sd", CLEAN_RET, "episode return  (matched eval)", 2450, "%.0f"),
        (axes[1], "f1", "f1_sd", CLEAN_F1, "note F1  (matched eval)", 0.82, "%.2f"),
    )):
        for j, d in enumerate(DISTS):
            xs, ys, es = [], [], []
            for i, ch in enumerate(CHANS):
                k = MATRIX[(ch, d)]
                xs.append(i + (j - 1) * (w + gap))
                ys.append(S[k][metric]); es.append(S[k][sdk])
            ax.bar(xs, ys, w, yerr=es, capsize=2.5, color=DIST_C[d],
                   label=d if row == 0 else None,
                   error_kw=dict(ecolor=INK2, lw=0.8), zorder=3)
            for x, y, e in zip(xs, ys, es):     # direct labels = contrast relief
                ax.text(x, y + e + ymax * 0.018, fmt % y, ha="center",
                        va="bottom", fontsize=7.5, color=INK)
        ax.axhline(clean_v, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(2.52, clean_v + ymax * 0.012, "clean " + (fmt % clean_v),
                fontsize=7.5, color=INK2, ha="right", va="bottom")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    axes[1].set_xticks(range(len(CHANS)))
    axes[1].set_xticklabels(
        [f"{c}\n(σ/level {'0.50' if c == 'Reward' else '0.15'})" for c in CHANS])
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.005, 1.02), ncol=3,
                   columnspacing=1.2, handlelength=1.1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1_main_reward_f1.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Fig 2 — learning dynamics. Row 1 reward (training rollout + periodic eval),
#         row 2 F1 (periodic eval only -- no rollout F1 exists).
# ===========================================================================
def fig2():
    chans = [("Action", ["a_gauss_p15", "a_uniform_p15", "a_shift_p15"]),
             ("Observation", ["o_gauss_p15", "o_uniform_p15", "o_shift_p15"]),
             ("Reward", ["r_gauss_p50", "r_uniform_p50", "r_shift_p50"])]
    # sharey="row": the three channels must be directly comparable within a
    # metric. Without it each facet auto-scales and a 20% reward drop in one
    # panel looks the same size as a 5% drop in another.
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), sharex=True, sharey="row")

    # ---- row 0: reward ----
    xc, yc = eval_curve("clean", "ep_return")
    xrc, yrc = rollout_curve("clean")
    for ax, (ch, ks) in zip(axes[0], chans):
        ax.plot(xc[len(xc) - len(smooth(yc)):], smooth(yc), color=MUTED,
                lw=1.1, ls=(0, (4, 3)), zorder=2)
        for k in ks:
            d = RUNS[k][2]
            x, y = eval_curve(k, "ep_return")
            ys = smooth(y)
            ax.plot(x, y, color=DIST_C[d], lw=0.6, alpha=0.20, zorder=3)
            ax.plot(x[len(x) - len(ys):], ys, color=DIST_C[d], lw=1.6,
                    label=d, zorder=4)
            xr, yr = rollout_curve(k)
            if xr is not None and len(xr):
                yrs = smooth(yr, 9)
                ax.plot(xr[len(xr) - len(yrs):], yrs, color=DIST_C[d], lw=1.0,
                        ls=(0, (2, 2)), alpha=0.75, zorder=3)
        ax.set_title(ch, fontsize=9, color=INK, pad=4)
    axes[0][0].set_ylabel("episode return")

    # ---- row 1: F1 ----
    xc, yc = eval_curve("clean", "ep_f1")
    for ax, (ch, ks) in zip(axes[1], chans):
        ax.plot(xc[len(xc) - len(smooth(yc)):], smooth(yc), color=MUTED,
                lw=1.1, ls=(0, (4, 3)), zorder=2)
        for k in ks:
            d = RUNS[k][2]
            x, y = eval_curve(k, "ep_f1")
            ys = smooth(y)
            ax.plot(x, y, color=DIST_C[d], lw=0.6, alpha=0.20, zorder=3)
            ax.plot(x[len(x) - len(ys):], ys, color=DIST_C[d], lw=1.6, zorder=4)
        ax.set_xlabel("env steps (M)")
    axes[1][0].set_ylabel("note F1")
    axes[1][0].set_ylim(0, 0.78)

    for row in axes:
        for ax in row:
            ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(0, 5.05)

    h, l = axes[0][0].get_legend_handles_labels()
    h = ([Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 3)))] + h
         + [Line2D([], [], color=INK2, lw=1.0, ls=(0, (2, 2)))])
    l = ["clean (no perturbation)"] + l + ["training rollout (reward row only)"]
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07),
               handlelength=1.8, columnspacing=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig2_learning_curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Fig 3 — compound perturbations vs their single-channel components
# ===========================================================================
def fig3():
    # Each compound is paired with its CORRECT-direction components. The shift
    # compound uses obs -0.15 / reward -0.50, so its components are the N-suffix
    # controls, not the canonical positive-shift runs.
    groups = [
        ("AO-Gauss", "ao_gauss_p15", ["a_gauss_p15", "o_gauss_p15"], "Gaussian"),
        ("AR-Uniform", "ar_uniform_a15_r50", ["a_uniform_p15", "r_uniform_p50"], "Uniform"),
        ("OR-Shift", "or_shift_on15_rn50", ["o_shift_n15", "r_shift_n50"], "Shift"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    for ax, metric, sdk, clean_v, ylab, ymax, fmt in (
        (axes[0], "ret", "ret_sd", CLEAN_RET, "episode return  (matched eval)", 2450, "%.0f"),
        (axes[1], "f1", "f1_sd", CLEAN_F1, "note F1  (matched eval)", 0.82, "%.2f"),
    ):
        pos = 0.0
        ticks, labels = [], []
        for name, comp, parts, dist in groups:
            for p in parts:
                ax.bar(pos, S[p][metric], 0.55, yerr=S[p][sdk], capsize=2.5,
                       color=DIST_C[dist], alpha=0.42,
                       error_kw=dict(ecolor=INK2, lw=0.8), zorder=3)
                ax.text(pos, S[p][metric] + ymax * 0.018, fmt % S[p][metric],
                        ha="center", va="bottom", fontsize=7, color=INK2)
                ticks.append(pos); labels.append(RUNS[p][0]); pos += 0.72
            ax.bar(pos, S[comp][metric], 0.55, yerr=S[comp][sdk], capsize=2.5,
                   color=DIST_C[dist], error_kw=dict(ecolor=INK2, lw=0.8), zorder=3)
            ax.text(pos, S[comp][metric] + ymax * 0.018, fmt % S[comp][metric],
                    ha="center", va="bottom", fontsize=7.5, color=INK,
                    fontweight="bold")
            ticks.append(pos); labels.append(name + "\n(compound)"); pos += 1.25
        ax.axhline(clean_v, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.set_ylabel(ylab); ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5)
    axes[0].text(0.005, 0.97, "pale = single-channel component   solid = compound",
                 transform=axes[0].transAxes, fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig3_compound.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Fig 4 — conservatism: precision holds while recall collapses.
#         F1-specific by construction, so this one stays single-metric.
# ===========================================================================
def fig4():
    order = list(MATRIX.values())      # canonical matrix only, no N-controls
    order.sort(key=lambda k: S[k]["rec"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, k in enumerate(order):
        d = RUNS[k][2]
        ax.plot([S[k]["rec"], S[k]["prec"]], [i, i], color=GRID, lw=1.6, zorder=2)
        ax.scatter(S[k]["rec"], i, s=42, color=DIST_C[d], zorder=4)
        ax.scatter(S[k]["prec"], i, s=42, facecolor="white",
                   edgecolor=DIST_C[d], linewidth=1.6, zorder=4)
        ax.text(S[k]["rec"] - 0.02, i, f"{S[k]['rec']:.2f}", ha="right",
                va="center", fontsize=7, color=INK2)
        ax.text(S[k]["prec"] + 0.02, i, f"{S[k]['prec']:.2f}", ha="left",
                va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([RUNS[k][0] for k in order], fontsize=8)
    ax.set_xlabel("value  (matched eval)")
    ax.set_xlim(-0.06, 1.12)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="", color=INK2, ms=7),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=7)]
    ax.legend(h, ["recall", "precision"], loc="lower right", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig4_conservatism.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig2()
    written = ["fig2_learning_curves"]
    if HAVE_FINAL:
        fig1(); fig3(); fig4()
        written = ["fig1_main_reward_f1"] + written + ["fig3_compound",
                                                       "fig4_conservatism"]
    print("wrote " + " / ".join(written) + "  (.pdf + .png) -> " + OUT)
    if not HAVE_FINAL:
        print(f"skipped fig1/fig3/fig4: no final-eval data under {FINAL} "
              "(see paper/robust_experiments_index.md for the command that "
              "regenerates it)")
        raise SystemExit(0)
    print()
    print("%-22s %10s %8s | %8s %7s" % ("task", "return", "±sd", "F1", "±sd"))
    for k in RUNS:
        print("%-22s %10.1f %8.2f | %8.4f %7.4f" %
              (RUNS[k][0], S[k]["ret"], S[k]["ret_sd"], S[k]["f1"], S[k]["f1_sd"]))
