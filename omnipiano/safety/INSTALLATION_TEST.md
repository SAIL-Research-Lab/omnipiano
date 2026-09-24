# Isolated installation test — 2026-09-24

The [Quick Start](QUICKSTART.md) was exercised in a newly created Python 3.10
Conda environment on Linux with an **NVIDIA GeForce RTX 4070 Ti**. Existing
training environments were not modified. Source baseline: safety commit
`5ca71cf65013a7c7f52664ba319bf26cc006a89c`, plus this installation change.
No newer `main` commits were merged for this test.

## Final tested runtime

| Package | Version |
|---|---|
| OmniSafe | 0.5.0 |
| PyTorch | 2.13.0, CUDA 13 build |
| NumPy / SciPy | 1.26.4 / 1.11.4 |
| pandas | 2.0.3 |
| MuJoCo / dm-control | 3.12.0 / 1.0.45 |
| Gymnasium / Shimmy | 0.28.1 / 0.2.1 |
| Stable-Baselines3 / PettingZoo | 2.0.0 / 1.24.3 |
| Safety-Gymnasium | 0.4.1 |
| MoviePy | 1.0.3 |

Final tests used `MUJOCO_GL=egl` and one thread each for Torch/OMP/BLAS.

## Results

| Check | Result |
|---|---|
| OmniSafe/PyTorch imports | Passed |
| Built-in Twinkle debug environment reset, before adding PIG | Passed |
| CUDA matrix multiplication on RTX 4070 Ti | Passed; expected value 32.0 |
| Safety unit tests, including native NumPy broadcasting regression | **42 passed** |
| PIG availability | 150 preprocessed pieces detected |
| Real safety environment checks | **58/58 passed**, 232 control steps |
| Coverage | 2/3/4/5 hands; all 12 semantic/setting combinations; main and ablation tasks |
| Main/hands/budget plan generation | Passed; no formal training launched |
| Matplotlib PNG generation | Passed; synthetic rendering check, not experiment data |

Each real-environment check performed a reset and two zero-action steps, then
a reset and two random-action steps. Observations, rewards and nonnegative costs
were checked for finite values.

| Algorithm | Device | Training steps | Checkpoints evaluated | Result |
|---|---|---:|---|---|
| PPO | cuda:0 | 2,000 | 0 and 2,000 | Complete |
| PPOLag | cuda:0 | 2,000 | 0 and 2,000 | Complete |
| OnCRPO | cuda:0 | 2,000 | 0 and 2,000 | Complete |
| CPO | cuda:0 | 2,000 | 0 and 2,000 | Complete |
| CUP | cuda:0 | 2,000 | 0 and 2,000 | Complete |

These used the first main task (two-hand ForElise, joint-range Fraction,
episode budget 19.95, seed 1). Each checkpoint received one complete deterministic
evaluation episode. All five runs produced checkpoints, progress logs, evaluation
CSV/NPZ files and `status: complete`. Reward/cost/F1 arrays had shape `(2, 1)` and
finite values; costs were nonnegative and F1 lay within [0, 1].

Tested Python-source fingerprint, using `safety.runtime.code_hash()`:
`75c79a22eebe82363f058f27ca257e6bc787005c0ebd1774d2cb6116165bfe38`.

## Installation issues found and handled

1. Pin NumPy 1.26.4 for the pandas 2.0.3 runtime required by OmniSafe 0.5.0.
2. Pin SciPy 1.11.4 to avoid an import-time conflict with the old main NumPy shim.
3. Restore native `numpy.array` under NumPy 1.x in `safety/__init__.py` when the
   active replacement is that exact main shim. Otherwise dm_env action-space
   construction fails on `subok`. The guard leaves NumPy 2.x and unrelated
   replacements untouched. No file outside `omnipiano/safety/` was edited.
4. Include PettingZoo, PyAudio and termcolor, which are imported by this branch
   but not all declared in its root package metadata.
5. Use EGL for the tested GPU profile. OSMesa passed physical stepping but
   triggered `free(): invalid pointer` during native Triton import when starting
   training; that attempt is **not** counted as passing. EGL completed all five
   algorithm tests. With `MUJOCO_GL=disable`, optimizer construction and a real
   environment reset/step also passed; a full CPU training run was not tested.

## Explicit limitations

- `pip check` returns the one documented upstream metadata conflict:
  Safety-Gymnasium 0.4.1 declares MuJoCo 2.3.0, while this environment uses 3.12.0.
  No other missing/conflicting dependencies were reported. This is runtime-tested,
  **not a resolver-clean dependency set**. See the installation exception in the
  Quick Start; no dependency metadata or third-party algorithm source was patched.
- The server's `nvidia-smi` reported a driver/library mismatch, and EGL emitted
  initialization warnings. Actual CUDA operations, network training and numerical
  checkpoint replay passed. Drivers were not changed and the server was not
  rebooted. MuJoCo image/video rendering was not validated; the Matplotlib check
  above is separate from MuJoCo rendering.
- Existing preprocessed PIG files were copied into the isolated checkout and
  checked. The preprocessing command's CLI was checked, but the raw dataset was
  not downloaded or reprocessed in this test. No PIG files are committed.
- No 5M runs, new algorithm integrations, reward/cost changes or formal protocol
  changes are included. This branch still uses 10 evaluation episodes per formal
  checkpoint and one for smoke tests; separate eval-1 handoff packages were not
  merged. Short installation tests are not evidence of learning performance.
- The profile is for this safety branch and tested GPU/runtime, not a claim of
  compatibility with every current main revision, older GPU or future package
  release. Preserve environment and source versions when comparing results.
