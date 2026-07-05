"""Phase 1 · unit gate — plot_robustness_curves.py CSV/aggregation core (§5.3).

Tests the pure data path (`load_series`, `aggregate`); matplotlib rendering is
not exercised (lazy Agg import, no assertions on pixels).
"""
import csv
import importlib.util
import os

import numpy as np
import pytest

_PLOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "plot_robustness_curves.py")
_spec = importlib.util.spec_from_file_location("plot_robustness_curves", _PLOT_PATH)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_load_series_groups_by_env_and_ckpt(tmp_path):
    p = tmp_path / "s.csv"
    hdr = ["env_id", "ckpt", "eval_noise_scale", "ep_f1"]
    _write(p, hdr, [
        ["EnvA", "best", "0.0", "0.9"],
        ["EnvA", "best", "1.0", "0.7"],
        ["EnvB", "best", "0.0", "0.8"],
    ])
    series = plot.load_series([str(p)])
    assert set(series) == {"EnvA · best", "EnvB · best"}
    assert len(series["EnvA · best"]) == 2


def test_aggregate_means_and_sort(tmp_path):
    rows = [
        {"eval_noise_scale": "1.0", "ep_f1": "0.6"},
        {"eval_noise_scale": "0.0", "ep_f1": "0.8"},
        {"eval_noise_scale": "1.0", "ep_f1": "0.8"},  # same scale → averaged
    ]
    scales, means, stds = plot.aggregate(rows, "ep_f1")
    assert list(scales) == [0.0, 1.0]           # sorted
    assert means[0] == pytest.approx(0.8)
    assert means[1] == pytest.approx(0.7)       # mean(0.6, 0.8)
    assert stds[1] == pytest.approx(0.1)


def test_aggregate_skips_missing(tmp_path):
    rows = [
        {"eval_noise_scale": "0.0", "ep_f1": "0.8"},
        {"eval_noise_scale": "1.0", "ep_f1": ""},     # missing metric → skip
        {"eval_noise_scale": "", "ep_f1": "0.5"},     # missing scale → skip
    ]
    scales, means, _ = plot.aggregate(rows, "ep_f1")
    assert list(scales) == [0.0]
    assert means[0] == pytest.approx(0.8)


def test_load_series_falls_back_to_basename(tmp_path):
    p = tmp_path / "robust_sweep_EnvX.csv"
    _write(p, ["eval_noise_scale", "ep_f1"], [["0.0", "0.9"]])
    series = plot.load_series([str(p)])
    assert "robust_sweep_EnvX" in series
