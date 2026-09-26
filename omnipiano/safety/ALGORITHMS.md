# Using additional OmniSafe algorithms

Use the separate environment in [QUICKSTART.md](QUICKSTART.md). This interface
calls **OmniSafe 0.5.0's existing implementations**; it does not implement new
optimizers or require another package. Run commands from the repository root.

The available scope is **32 online, model-free algorithms: 23 on-policy + 9
off-policy**, including the original PPO, PPOLag, OnCRPO, CPO and CUP. Thus there
are **27 additional choices**, not 34 total. Seven of the 32 are unconstrained
RL baselines, not constraint-enforcing algorithms. Offline and model-based
OmniSafe methods are outside this interface.

## 1. Choose algorithms and tasks

```bash
python -m omnipiano.safety.run --list-algorithms

# Plan only: first main environment, FOCOPS, three seeds, 5M training steps.
python -m omnipiano.safety.run --group main --task-index 0 \
  --algorithms FOCOPS --seeds 1 2 3 --device cuda:0 \
  --eval-episodes 1 --out results_focops

# Add --execute to actually train and evaluate.
python -m omnipiano.safety.run --group main --task-index 0 \
  --algorithms FOCOPS --seeds 1 2 3 --device cuda:0 \
  --eval-episodes 1 --out results_focops --execute

# Compare several algorithms on one task, including existing baselines.
python -m omnipiano.safety.run --group main --task-index 0 \
  --algorithms PPO PPOLag FOCOPS CPPOPID SACLag --seeds 1 2 3 \
  --device cuda:0 --eval-episodes 1 --out results_comparison --execute
```

`--task-index` selects **environments**, starting from zero; it is not the old
`--index` (which selects a task/algorithm/seed cell in the fixed paper matrix).
Omit `--task-index` to select all tasks in the chosen group. Multiple indices,
algorithm names and seeds are accepted. The selected cells execute sequentially;
independent commands can select different cells/devices/output directories.

| Group | Task indices | Default steps per run | Default seeds |
|---|---|---:|---|
| `main` | 0–7: the original eight tasks | 5,000,000 | 1, 2, 3 |
| `hands` | 0–3: 2, 3, 4, 5 hands respectively | 5,000,000 | 1, 2, 3 |
| `budget` | 0–4: budgets 1.44, 4.32, 14.4, 43.2, 144 | 5,000,000 | 1, 2, 3 |
| `extensions` | 0–47: the existing semantic/setting catalogue | 2,000,000 | 0 |

The printed plan and manifest contain the exact environment ID, song, cost,
budget, algorithm, seed and configuration. With `--algorithms`, the selection is
a Cartesian product of tasks × algorithms × seeds. In particular, selecting PPO
with all five budget tasks creates five unconstrained reference runs; selecting
only `--task-index 2` gives the default-budget reference. The original fixed
budget matrix still uses its single default-budget PPO reference.

## 2. Available names

| Family / budget handling | Names |
|---|---|
| Unconstrained on-policy | `PolicyGradient`, `NaturalPG`, `TRPO`, `PPO` |
| On-policy Lagrangian / PID | `PPOLag`, `TRPOLag`, `RCPO`, `PDO`, `FOCOPS`, `CUP`, `CPPOPID`, `TRPOPID` |
| Direct `algo_cfgs.cost_limit` | `CPO`, `PCPO`, `OnCRPO`, `IPO`, `P3O`, `PPOEarlyTerminated`, `TRPOEarlyTerminated` |
| Safety-state augmentation | `PPOSaute`, `TRPOSaute`, `PPOSimmerPID`, `TRPOSimmerPID` |
| Unconstrained off-policy | `DDPG`, `TD3`, `SAC` |
| Off-policy Lagrangian / PID | `DDPGLag`, `TD3Lag`, `SACLag`, `DDPGPID`, `TD3PID`, `SACPID` |

Use the exact names above: for example, `OnCRPO`, not `OnCPRO`, and `PPOLag`,
not `PPO-lag`. Availability means the configuration and environment interface
are connected; it does **not** mean every algorithm has been empirically validated
on every OmniPiano task.

## 3. Short test and parameters

```bash
# Two checkpoints (initial and after 2K steps), one evaluation episode each.
python -m omnipiano.safety.run --group main --task-index 0 \
  --algorithms SACLag --seeds 1 --device cuda:0 --smoke-test \
  --out sac_smoke --execute

# A formal-protocol run with a chosen step count and replay-buffer capacity.
python -m omnipiano.safety.run --group main --task-index 0 \
  --algorithms SACLag --seeds 1 --steps 1000000 --buffer-size 100000 \
  --device cuda:0 --eval-episodes 1 --out sac_1m --execute
```

Shared overrides preserve the benchmark's reward discount `gamma=0.8`, raw
reward/cost scales, observation normalization, and one environment per process.
On-policy cost discount is also 0.8; off-policy methods use `gamma` for their
cost critic and have no separate `cost_gamma` configuration field. Other
algorithm-specific network/optimizer parameters use OmniSafe's native defaults.
This is a common environment/protocol, **not equal compute or identically tuned
hyperparameters across algorithms**.

- On-policy: 20K training steps per epoch, checkpoint every epoch.
- Off-policy: native 2K steps per epoch, checkpoint every 10 epochs. Native
  internal evaluation uses one episode each epoch and is separate from the
  full-song reward/cost/F1 checkpoint replay. These evaluation steps are not
  added to the training-step budget.
- Formal checkpoint spacing is therefore 20K for both families, including the
  initial model. `--steps` must be a positive multiple of 20K.
- Off-policy replay capacity defaults to **100,000**, reduced from OmniSafe's
  1,000,000 to bound RAM use with large piano observations. `--buffer-size`
  changes it; it applies only to off-policy methods. High-dimensional 5-hand
  buffers can still consume substantial RAM.
- Smoke tests use 2K-step epochs/checkpoints. Off-policy smoke tests use a 2K
  buffer by default, start learning after 256 steps, and set the constrained
  method's `warmup_epochs` to 0. Native Lag/PID multiplier updates begin at
  epoch index 1, so use `--smoke-test --steps 4000` to exercise that update too.
  The 2K test exercises actor/critic updates, but its
  configuration must not be used as a formal benchmark result.
- `--eval-episodes 1` explicitly selects one full song per checkpoint. Without
  it, custom formal runs retain 10; smoke runs use 1. Training seeds and
  evaluation episode count are different quantities.

Without `--algorithms`, all original five-algorithm matrices and their defaults
remain unchanged. Custom options require `--algorithms` to avoid silently
changing those paper experiments. A new output directory is appropriate after
code/dependency changes. Checkpoint replay can be retried after training, but
these commands do not resume interrupted optimizer/replay-buffer state.

## 4. Evaluation conventions

All checkpoints are evaluated deterministically on the **original full song**,
reporting unmodified task reward, raw accumulated cost and terminal F1. The same
evaluation seeds (`training_seed + 10000 + episode_index`) are used across
checkpoints. Safety budget is read from the task, not left at OmniSafe's default.

For Saute/Simmer, the policy needs one extra safety-state coordinate. The replay
code appends it after base-observation normalization, resets it to 1, and updates
`z_next = (z - cost / B) / saute_gamma`, where
`B = target_budget * (1 - saute_gamma**H) / ((1 - saute_gamma) * H)`.
`H` is the song's catalogue horizon; `saute_gamma` retains the native 0.999 and
is distinct from the reward discount 0.8. Positive budgets are required.

Simmer trains with its native adaptive-budget controller. OmniSafe checkpoints
do not save that controller's state; evaluation therefore uses the **fixed target
upper budget**, not a claimed restoration of the training curriculum. Initial
training budget and upper budget are both set to the task budget. This convention
is recorded as `full-song-fixed-target-budget-v1` in custom manifests.

EarlyTerminated methods keep their native early-termination training adapter.
Full-song evaluation intentionally does not terminate when cost exceeds budget,
so every algorithm receives the same full-song measurement. Their shortened
training rollouts must not be interpreted as full-song evaluation scores.

## 5. Results and plots

Each completed run writes `cell.json`, `overrides.json`, native `logs/config.json`
and `progress.csv`, checkpoints, `eval_episodes.csv`, `evaluations.npz`, and
`status.json`. The native config records the merged defaults and overrides.

```bash
# Per-task reward/cost/F1 panels, means and sample SD across available seeds.
python -m omnipiano.safety.compare --group main --runs results_comparison \
  --out figures_comparison

# Per-task training reward/cost panels.
python -m omnipiano.safety.compare --group main --runs results_comparison \
  --out figures_comparison --source rollout

# Installation-test curves must be explicitly selected and are labelled SMOKE.
python -m omnipiano.safety.compare --group main --runs sac_smoke \
  --out smoke_figures --smoke-only
```

Only completed custom runs are plotted. Legends show actual training-seed
counts; one seed has no uncertainty shading. Missing runs are not filled or
extrapolated. Keep different tuning configurations in separate output directories.
The original `safety.plot` remains the strict paper-matrix plotter and excludes
custom runs; it does not silently mix them with the original experiments.

## Implementation map

| File | Responsibility |
|---|---|
| `algorithms.py` | Names, algorithm families, budget routing categories, evaluation budget conversion |
| `runtime.py` | Protocol overrides layered on native OmniSafe defaults |
| `run.py` | CLI selection and reproducible run manifests |
| `cmdp.py` | Existing observation/action/reward/cost environment bridge; unchanged |
| `train.py` | Invoke native `omnisafe.Agent`, save status, then replay checkpoints |
| `evaluate.py` | Full-song reward/cost/F1 replay, including safety-state augmentation |
| `compare.py` | Flexible custom-run figures; separate from the fixed paper figures |

No changes outside `omnipiano/safety/` are needed.
