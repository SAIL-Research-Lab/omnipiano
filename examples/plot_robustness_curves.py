"""plot_robustness_curves.py — robustness curves from a sweep CSV (Phase 1, §5.3).

Reads the combined CSV written by ``examples/robust_eval_sweep.py``
(``robust_sweep_<env>.csv``) and draws, per (env_id, ckpt) series, the
robustness curves:
  - F1 vs eval_noise_scale        (clean headline metric, noise-immune)
  - ep_return vs eval_noise_scale (received/noised return for reward tasks)
  - ep_return_true overlaid       (denoised return; = ep_return for A/O tasks)

x-axis = eval_noise_scale (deployment-time robustness: same policy, rising eval
noise; decisions 10/11 make this uniform across action/obs/reward channels).

Multiple sweep CSVs (e.g. one per algo) can be passed; each contributes its own
series so a single figure compares algorithms on the same task.

The CSV parsing + aggregation core (`load_series`, `aggregate`) is pure and
unit-tested; matplotlib is imported lazily so the module loads without a
display backend.

Example:
    python examples/plot_robustness_curves.py \
        --csv runs/ppo/robust_sweep_OmniPiano-ClairDeLune-A-Gauss-P05-v0.csv \
              runs/sac/robust_sweep_OmniPiano-ClairDeLune-A-Gauss-P05-v0.csv \
        --out robustness_a_gauss_p05.png
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

# Metrics plotted vs eval_noise_scale.
_F1_COL = "ep_f1"
_RETURN_COL = "ep_return"
_RETURN_TRUE_COL = "ep_return_true"
_SCALE_COL = "eval_noise_scale"


def load_series(csv_paths):
    """Read one or more sweep CSVs → {series_label: [row_dict, ...]}.

    series_label = "<env_id> · <ckpt>" (falls back to the CSV basename if those
    columns are absent). Row dicts keep string values; numeric parsing happens
    in `aggregate`.
    """
    series = defaultdict(list)
    for path in csv_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                env_id = row.get("env_id", base)
                ckpt = row.get("ckpt", "")
                label = f"{env_id} · {ckpt}" if ckpt else env_id
                series[label].append(row)
    return dict(series)


def aggregate(rows, metric_col):
    """Group rows by eval_noise_scale → (scales, means, stds) sorted by scale.

    Rows missing the metric or scale (empty string / absent) are skipped.
    """
    by_scale = defaultdict(list)
    for row in rows:
        s_raw, m_raw = row.get(_SCALE_COL, ""), row.get(metric_col, "")
        if s_raw == "" or m_raw == "":
            continue
        by_scale[float(s_raw)].append(float(m_raw))
    scales = sorted(by_scale)
    means = np.array([np.mean(by_scale[s]) for s in scales])
    stds = np.array([np.std(by_scale[s]) for s in scales])
    return np.array(scales), means, stds


def _plot(series, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_f1, ax_ret) = plt.subplots(1, 2, figsize=(12, 5))
    for label, rows in sorted(series.items()):
        s, m, sd = aggregate(rows, _F1_COL)
        if len(s):
            ax_f1.errorbar(s, m, yerr=sd, marker="o", capsize=3, label=label)
        s, m, sd = aggregate(rows, _RETURN_COL)
        if len(s):
            line = ax_ret.errorbar(s, m, yerr=sd, marker="o", capsize=3,
                                   label=f"{label} (received)")
        st, mt, _ = aggregate(rows, _RETURN_TRUE_COL)
        if len(st):
            ax_ret.plot(st, mt, marker="x", linestyle="--",
                        color=line[0].get_color(), label=f"{label} (true)")

    ax_f1.set(xlabel="eval_noise_scale", ylabel="F1", title="F1 robustness")
    ax_ret.set(xlabel="eval_noise_scale", ylabel="ep_return",
               title="Return robustness (received vs true)")
    for ax in (ax_f1, ax_ret):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot_robustness_curves] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, nargs="+",
                   help="One or more robust_sweep_*.csv files (one per algo/run).")
    p.add_argument("--out", default="robustness_curves.png",
                   help="Output figure path (.png).")
    args = p.parse_args()

    series = load_series(args.csv)
    if not series:
        p.error("no rows found in the provided CSV(s).")
    _plot(series, args.out)


if __name__ == "__main__":
    main()
