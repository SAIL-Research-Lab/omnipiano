"""Phase 1 · smoke gate — examples/robust_eval_sweep.py (§5.2).

Exercises the sweep harness end-to-end WITHOUT a trained checkpoint, using a
dummy zero-action policy:
  - `_run_one_scale` builds the eval env via make(mode="eval",
    eval_noise_scale=scale), runs episodes, and its
    SafeRecordEpisodeStatistics writes a schema-locked per-scale CSV whose
    `eval_noise_scale` column equals the requested scale and which carries the
    S6 robust columns;
  - `_concat_csvs` merges per-scale CSVs and prepends env_id + ckpt columns.

Model loading (`_load_model`) is NOT exercised here (needs a real SB3 zip);
that path is trivial dispatch to PPO/SAC/TQC.load.
"""
import csv
import importlib.util
import os

import numpy as np
import pytest

from omnipiano.envs.registration import make

# Load the example script as a module (examples/ is not a package).
_SWEEP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "robust_eval_sweep.py")
_spec = importlib.util.spec_from_file_location("robust_eval_sweep", _SWEEP_PATH)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

_ENV = "OmniPiano-ClairDeLune-A-Gauss-P05-v0"
_S6_COLS = ["eval_noise_scale", "ep_return_true",
            "ep_noise_action_l2", "ep_noise_obs_l2", "ep_noise_reward"]


class _DummyPolicy:
    """Stand-in for an SB3 model: predict → fixed zero action."""
    def __init__(self, act_dim):
        self._act = np.zeros(act_dim, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self._act, None


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _act_dim():
    probe = make(_ENV, mode="eval")
    try:
        return probe.action_space.shape[0]
    finally:
        probe.close()


def test_run_one_scale_writes_schema_locked_csv(tmp_path):
    model = _DummyPolicy(_act_dim())
    scale_dir = str(tmp_path / "scale_2")
    csv_path = sweep._run_one_scale(
        model, _ENV, scale=2.0, scale_dir=scale_dir,
        num_eval_eps=1, seed_base=10_000)
    header, rows = _read_csv(csv_path)
    # S6 robust columns present
    for col in _S6_COLS:
        assert col in header, f"missing {col} in {header}"
    assert len(rows) == 1
    # eval_noise_scale column equals the requested scale
    idx = header.index("eval_noise_scale")
    assert float(rows[0][idx]) == pytest.approx(2.0)


def test_scale_zero_is_clean_noise_cols(tmp_path):
    model = _DummyPolicy(_act_dim())
    csv_path = sweep._run_one_scale(
        model, _ENV, scale=0.0, scale_dir=str(tmp_path / "scale_0"),
        num_eval_eps=1, seed_base=10_000)
    header, rows = _read_csv(csv_path)
    a_idx = header.index("ep_noise_action_l2")
    s_idx = header.index("eval_noise_scale")
    assert float(rows[0][s_idx]) == 0.0
    # scale=0 → action noise scaled to 0 → zero accumulated action-noise L2
    assert float(rows[0][a_idx]) == pytest.approx(0.0, abs=1e-9)


def test_concat_prepends_env_and_ckpt(tmp_path):
    # Fabricate two per-scale CSVs with identical schema.
    hdr = ["eval_noise_scale", "ep_return", "ep_f1"]
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    for p, sc in ((p1, "0.0"), (p2, "2.0")):
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerow([sc, "1.5", "0.8"])
    combined = str(tmp_path / "sweep.csv")
    sweep._concat_csvs([str(p1), str(p2)], combined, _ENV, "best_model")
    header, rows = _read_csv(combined)
    assert header == ["env_id", "ckpt"] + hdr
    assert len(rows) == 2
    assert all(r[0] == _ENV and r[1] == "best_model" for r in rows)


def test_concat_detects_schema_drift(tmp_path):
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    with open(p1, "w", newline="") as f:
        csv.writer(f).writerow(["eval_noise_scale", "ep_return"])
    with open(p2, "w", newline="") as f:
        csv.writer(f).writerow(["eval_noise_scale", "ep_f1"])  # drift
    with pytest.raises(RuntimeError, match="schema drift"):
        sweep._concat_csvs([str(p1), str(p2)], str(tmp_path / "o.csv"),
                           _ENV, "ckpt")
