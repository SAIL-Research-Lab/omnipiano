# Minimal validation — 2026-09-19

These are software integration checks, **not formal benchmark results**.

## Power Fraction addition — manifest v2

The current catalogue has 12 semantic/setting combinations, 48 extension
environments, 58 unique registrations, and 96 optional extension screen runs.
Main/hand/budget groups remain 120/60/63 runs. All 243 original cells were compared
before/after the change: task IDs, algorithm/seed/steps, budgets and power references
are identical. Total-power Event/Excess formulas are unchanged.

- Local and full Linux unit suite: **41 passed**. Cases include threshold equality,
  selected-hand denominators, weighted fractions, zero/all overload, 2–5 hands,
  and `[6, 0]` where total-power Event is zero but per-hand Fraction is 0.5.
- Four added full piano environments passed 20 zero-action and 20 random-action
  steps each, with a reset between action modes: **160 steps**, no training.
- Tested through `SafetySemanticCMDP`: the returned cost tensor matches the
  existing SafetyWrapper total and the measured violating-hand fraction.
- All observed values are finite, lie in [0,1], and are multiples of `1/K`
  (within float32 tolerance). The per-hand reference is 4 for each hand count.

| Hands | Candidate episode budget | Fraction values observed in this short test |
|---:|---:|---|
| 2 | 19.95 | 0, 1 |
| 3 | 28.15 | 0, 2/3, 1 |
| 4 | 36 | 0, 1/2, 1 |
| 5 | 36 | 0, 1/5, 4/5, 1 |

These observations verify the cost channel, not task difficulty or learnability.
Five-algorithm training tests below are historical v1 evidence; they were not
rerun for this additive cost change. Existing v1 batches must retain their frozen
code version, because adding code changes the provenance fingerprint.

## Original v1 validation (commit 303e68b)

The following counts and fingerprint describe v1, before Power Fraction was added.

## Scope

- Tested against main `8a1545aba66a6baf03c5c905a8bf5bee80a2fe2a`.
- Only new files under `omnipiano/safety/`; original `constraints.py` and
  `__init__.py` unchanged. Main's config, factory, wrappers and examples unchanged.
- Full simulator checks ran on a separate Linux checkout, not the previous live
  experiment directory. Source code and MIDI provenance are recorded by the runner.
- No 2M extension or 5M formal batch was launched; no W&B data was written.

## Results

| Check | Result |
|---|---|
| New unit suite, real Linux package imports | 34 passed |
| Local unit harness (avoids unavailable Windows audio library) | 34 passed |
| Real piano registration/reset/step | 54 unique environments passed |
| Environment smoke coverage | Each environment: reset + 2 zero-action steps, reset + 2 random-action steps; 216 steps total |
| Cost catalogue | All 11 semantic/setting combinations at 2/3/4/5 hands |
| Main and ablations | All main, nested-hand and budget variants included in environment smoke |
| Plot CLI | All 3 groups × replay/rollout; 30 PNG/PDF files generated from explicitly synthetic fixtures |

The 54 unique registrations combine the 44 extension candidates, the retained
main choices, four nested-hand tasks, and budget variants; overlapping task
definitions are registered once. Run groups remain independent even when their
environment configuration is identical.

## Algorithm integration

Verified runtime: Python 3.10.21, OmniSafe 0.5.0, Torch 2.13.0, NumPy 2.2.6,
MuJoCo 3.12.0, dm-control 1.0.45, Gymnasium 0.28.1. These are the observed test
versions, not a claim that every other dependency combination is supported.

Each algorithm ran one 2,000-step epoch on the first two-hand ForElise range/fraction
task, training seed 1, episode budget 19.95, CUDA device 0 (RTX 4070 Ti). One
deterministic full-episode evaluation was performed at each of checkpoints 0 and
2,000. Every run produced finite reward/cost/F1, an OmniSafe progress log, checkpoint
files, evaluation NPZ/CSV, and status `complete`.

| Algorithm | Training + replay integration | Wall time (seconds, smoke only) |
|---|---|---:|
| PPO | passed | 23.74 |
| PPOLag | passed | 23.79 |
| OnCRPO | passed | 24.16 |
| CPO | passed | 24.15 |
| CUP | passed | 26.21 |

Code fingerprint recorded by these runs (Python files under `omnipiano/`, normalized
line endings): `cf9f659bb89424f7e0dcceaf62e44b4f669b81bb3b93690e37a60a939260ee8b`.
This is not a performance ranking or a reliable 5M wall-time prediction. Formal
runs use 20K-step epochs and 10 evaluation episodes per checkpoint, not the smoke
schedule. Repeated identical deterministic episodes are not additional training seeds.

## Limitations and resolved issues

- Main's BenchmarkEnvConfig does not declare `energy_penalty_coef`. Directly
  supplying it failed during the first smoke attempt. The final implementation
  uses the safety-local dataclass subclass and main's existing forwarding path.
- An initial PPO smoke transport session stopped due to local Windows console
  encoding. It was not counted as successful; the five successful runs used a
  fresh isolated output directory with captured UTF-8 logs.
- EGL driver warnings did not prevent physics or CUDA-network tests. Rendering
  and video export were not validated. Windows still needs its external FluidSynth
  runtime for full piano imports; that system dependency is not changed here.
- Threshold feasibility, initialization transient cost, learning stability,
  multi-seed performance and paper conclusions require the planned experiments.
