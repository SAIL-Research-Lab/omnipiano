# Additional-algorithm integration test — 2026-09-24

Historical safety-branch algorithm record. See
[MAIN_INTEGRATION.md](MAIN_INTEGRATION.md) for the separate merged-main checks.

Source baseline: safety commit `0968b10`, plus this algorithm-interface change.
All changed files are under `omnipiano/safety/`. No newer main commits were
merged and no root dependency, environment, wrapper or task files were edited.

Tests used the separate Linux Python 3.10 environment documented in
[INSTALLATION_TEST.md](INSTALLATION_TEST.md): OmniSafe 0.5.0, PyTorch 2.13.0,
NumPy 1.26.4, MuJoCo 3.12.0, dm-control 1.0.45, and an **RTX 4070 Ti**.
Training device was `cuda:0`, with `MUJOCO_GL=egl` and one Torch/OMP/BLAS thread.
The installation's documented Safety-Gymnasium/MuJoCo metadata conflict remains;
this interface adds no dependencies and does not claim to resolve that conflict.

## Final validation results

| Check | Result |
|---|---|
| Safety unit tests and all 32 algorithms' native configuration-key coverage | **78 passed** |
| Algorithm list equals installed OmniSafe's on-policy + off-policy registry | **32/32 names matched** |
| Original matrix planning | **120 main / 60 hands / 63 budget / 96 extension cells**, unchanged |
| Custom selection of all 32 algorithms on one task/seed | Plan generation passed; **not 32 training tests** |
| Custom evaluation plots | Reward/cost/F1 PNG and PDF generated |
| Custom training plots | Reward/cost PNG and PDF generated |

The representative runs below all used main task 0: **2-hand ForElise,
joint-range Fraction, episode budget 19.95, seed 1**. Every evaluated checkpoint
used one deterministic full-song episode of 399 steps.

| Algorithm | Why selected | Training steps | Evaluated checkpoints (training steps) | Result |
|---|---|---:|---|---|
| PPO | Existing baseline regression | 4,000 | 0 / 2,000 / 4,000 | Complete |
| FOCOPS | Additional on-policy Lagrangian method | 4,000 | 0 / 2,000 / 4,000 | Complete |
| IPO | Direct algorithm-level budget routing | 4,000 | 0 / 2,000 / 4,000 | Complete |
| CPPOPID | PID multiplier method | 4,000 | 0 / 2,000 / 4,000 | Complete |
| SACLag | Off-policy buffer, actor/critic and multiplier updates | 4,000 | 0 / 2,000 / 4,000 | Complete |
| PPOSaute | Augmented safety-state policy and checkpoint replay | 4,000 | 0 / 2,000 / 4,000 | Complete |
| PPOSimmerPID | Adaptive training budget and fixed-target replay, including a second epoch | 4,000 | 0 / 2,000 / 4,000 | Complete |
| SACLag | Native 2K epochs with checkpoint frequency 10 | 20,000 | 0 / 20,000 | Complete |

The 4K runs used the documented smoke overrides. For SACLag, starting learning
after 256 steps and setting warm-up epochs to zero allowed actor/critic updates
and a second-epoch multiplier update. Its logged mean multiplier changed from
approximately **0.001 to 0.01104**. The separate 20K run retained native learning
warm-up settings and tested the formal checkpoint interval, not multiplier
convergence. It produced ten training log rows and two evaluated checkpoints.

All eight runs had `status: complete`. Initial and final actor state dictionaries
differed in every run. All reward/cost/F1 arrays were finite, costs nonnegative,
and F1 in [0, 1]. The 4K runs' evaluation arrays had shape `(3, 1)` and the 20K
run's `(2, 1)`. Plot legends correctly showed one seed and no uncertainty band.

Tested Python-source fingerprint (`safety.runtime.code_hash()`):

```text
6f0666a768f8114ac5a5f1e8a1b5f728a13ecc106b7d70d588ee0338da4de60c
```

## Reproduce a representative check

After following [QUICKSTART.md](QUICKSTART.md), including the PIG preprocessing:

```bash
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m pytest omnipiano/safety/tests -q

python -m omnipiano.safety.run --group main --task-index 0 \
  --algorithms PPO FOCOPS IPO CPPOPID SACLag PPOSaute PPOSimmerPID \
  --seeds 1 --steps 4000 --smoke-test --device cuda:0 \
  --out algorithm_smoke --execute

python -m omnipiano.safety.compare --group main --runs algorithm_smoke \
  --out algorithm_smoke_figures --smoke-only
```

These are integration tests, **not 5M benchmark results, safety guarantees, or
evidence of good learning performance**. Six of the 27 additional algorithms
received training tests. The remaining 21 were checked for registry membership
and configuration-key compatibility, but were not individually trained here.
This turn does not revalidate all tasks, all seeds, CPU training, offline methods,
model-based methods, video rendering, or long-run numerical stability.

See [ALGORITHMS.md](ALGORITHMS.md) for usage, parameter differences, full-song
evaluation conventions and the Simmer checkpoint-controller limitation.
