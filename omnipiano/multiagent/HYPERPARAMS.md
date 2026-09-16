# Multi-agent baseline hyperparameters

The machine-readable source of truth is `marl_train_config.json`; this file
adds the rationale. The trainer snapshots that JSON and the final CLI-resolved
values into every run's `run_config.json`, so an artifact is self-describing.

`marl_train_config_2026-09-15.json` is the frozen Phase-1 archive of those
experiment settings. It is byte-for-byte identical to the source of truth at
commit `fd42702d7049434ba6848d4f7421293197afdd6f` (SHA-256
`3075152813bf578f10602b567caa7793aec32e3063b91a1603f3273c1adde863`) and is
not read implicitly by the trainer.

`python -m omnipiano.multiagent.train --list-algos` prints the live registry.
Canonical defaults require no long parameter list:

    python -m omnipiano.multiagent.train --algo mappo --seed 0

Use `--config path/to/variant.json` for an ablation. Explicit CLI flags remain
available for one-off overrides. Resolution order is shared JSON < selected
`algorithm_overrides` < `smoke_test_overrides` < explicit CLI.

## 1. Who owns what

| Layer | Owner | Must be identical across algorithms? |
|---|---|---|
| Protocol (steps, seeds, gamma, eval cadence) | `marl_train_config.json`, checked against `BenchmarkProtocolConfig` | **Yes** |
| PPO and network hyperparameters | `marl_train_config.json` | **Yes** (IPPO vs MAPPO is a single-factor ablation) |
| Algorithm identity (critic input, global state) | `algos/<name>.py` | No — this IS the independent variable |
| Compute and W&B defaults | `marl_train_config.json` | No (may be overridden by the launcher and must be recorded) |

## 2. Algorithm registry

| algo | actor input | critic input | env emits `s` | note |
|---|---|---|---|---|
| `ippo` | `o_i` | `o_i` | no | de Witt et al. 2020 |
| `mappo` | `o_i` | `s` (agent-specific) | yes | Yu et al. 2022 |
| `mappo-own-critic` | `o_i` | `o_i` | yes (unread) | validation only: must reproduce `ippo` bit-for-bit |
| `ippo-rllib-module` | `o_i` | `o_i` | no | debug anchor; different architecture, do NOT mix numbers |

Actor and critic share no trunk, and critics are **not** parameter-shared
across agents. Both choices are deliberate: they prevent critic gradients from
altering the actor and keep ownership identical across algorithms. MAPPO's
first critic layer necessarily has more parameters because `s` is wider than
`o_i`; `ctde_modules` in every `run_config.json` records that unavoidable
input-dimensionality difference explicitly.

## 3. Protocol

| Parameter | Value | Why |
|---|---|---|
| `total_env_steps` | 5,000,000 | **ENVIRONMENT** steps, not agent steps. With 2 agents a summed agent-step counter would end the run at 2.5M physical interactions. Enforced by `training.runtime.extract_env_steps`. |
| `gamma` | 0.8 | Task property, not an algorithm knob — same value for SAC/PPO/TQC. Effective horizon ~5 control steps. |
| `eval_freq_env_steps` | 50,000 | Protocol cadence, giving 100 evaluation points over a 5M run. |
| `num_eval_eps` | 1 | Protocol default. Cross-seed aggregation supplies the main uncertainty estimate; each evaluation episode is deterministically reseeded. |
| `seed` | 0 / 1 / 2 | Paper needs all three. One seed is a validation run, not a result. |
| `eval_seed` | `seed + 10000`, episode `i` uses `+ i*10000` | Deterministic and non-colliding across training seeds. |

## 4. PPO learner

The PPO values and implementation are shared by IPPO and MAPPO so the critic
observation remains the controlled algorithmic difference.

| Parameter | Ours | Rationale |
|---|---|---|
| train batch / minibatch | `4000 / 4000` | One full-batch minibatch per epoch, matching official MAPPO's `num_mini_batch=1`; avoids repeatedly fitting tiny correlated slices. |
| epochs | `5` | Conservative end of MAPPO's recommended 5–15 range; the previous 10 remains a later ablation, not the recovery default. |
| actor / critic LR | `3e-4 / 3e-4` | Separate Adam optimizers but matched rates, so optimizer state is no longer shared without introducing an LR confound. |
| Adam epsilon | `1e-5` | Official MAPPO setting (instead of PyTorch Adam's `1e-8`). |
| GAE λ | `0.95` | Standard PPO/MAPPO value; `gamma=0.8` remains task-specific and unchanged. |
| policy clip ε | `0.2` | PPO/MAPPO clipping value. |
| KL loss | off | Official MAPPO is clip-only. RLlib's adaptive KL penalty would make updates additionally conservative and introduces hidden coefficient dynamics. |
| entropy coefficient | `0.0` | Kept at the task-proven recovery value. Official MAPPO's generic default is `0.01`, but that is a future `0 / 1e-4 / 3e-4` sweep rather than bundled into this recovery run. |
| value loss coefficient | `1.0` | Official MAPPO configuration. |
| actor / critic grad clip | `10 / 10`, separate global norms | Official MAPPO clips the two independently; RLlib now sees two named optimizers and therefore does the same. |

### ValueNorm and value clipping

`RunningValueNorm(beta=0.99999, epsilon=1e-5)` is enabled. The critic network
predicts normalized values; GAE and explained variance receive denormalized
values. Statistics are RLModule buffers, so checkpoints and learner→runner
weight synchronization include them.

RLlib 2.55.1's stock `vf_clip_param` is **not PPO value-prediction clipping**:
it caps squared error (`clamp((V-target)^2, 0, limit)`), which gives every
sample above the cap zero critic gradient. Therefore neither the previous
`10` nor `1000` has the desired semantics. The custom learner instead uses:

    V_clip = V_old + clamp(V_new - V_old, -0.2, +0.2)
    L_v = max(0.5 (target_norm - V_new)^2,
              0.5 (target_norm - V_clip)^2)

`V_old` and `V_new` are normalized critic outputs; return targets are normalized
with the running statistics. W&B records `value_clip_fraction`,
`value_norm_mean`, `value_norm_std`, both optimizer gradient norms, and:

    EV = 1 − Var[R̂ − V_denormalized] / Var[R̂]

`EV → 1` means a good critic; `EV ≈ 0` means approximately constant prediction;
`EV < 0` is worse than the constant baseline.

## 5. Network

`MLP 256-256`, `tanh`, feature `LayerNorm` on the separately sliced actor and
critic inputs, orthogonal init with gain `sqrt(2)` on hidden layers,
**0.01 on the policy output** (Engstrom et al. 2020: keeps the initial action
distribution near-isotropic so exploration is not collapsed at step 0), `1.0`
on the value head. Gaussian policy with a **state-independent** `log_std`
parameter, clamped to `[-5, 2]`, matching MAPPO's continuous-control reference.

Actor and critic remain completely separate MLPs. This newer architecture is
kept: the official implementation also uses separate actor/critic networks,
orthogonal initialization, zero biases, input feature normalization, a small
policy-head gain, and state-independent action standard deviation. Changing it
back to RLlib's default module would confound the IPPO/MAPPO comparison.

## 6. Global state (MAPPO only, agent-specific / AS variant)

Concatenated in a fixed spatial left→right order (permutation-consistent):

    every hand's joints_pos | FULL 88-key piano state | global sustain
    | FULL goal tensor (all lookahead frames) | [prev joint action, prev reward
    if the OAR wrapper is present] | one-hot agent id

Under a **shared** team reward every agent has the same return, so an
environment-provided-only (EP) critic is already unbiased; the one-hot merely
lets a per-agent critic specialize. The AS/EP ablation is a one-line change
(drop the trailing `num_agents` entries).

The observation is `[global_state | own]`, and the `own` block is
**bit-identical** to the IPPO observation — locked by
`global_state_test.test_own_block_matches_ippo_observation_bitwise`.

## 7. Compute

| Parameter | Value | Note |
|---|---|---|
| `num_learners` | 1 (hard-pinned) | `total_train_batch_size` scales with it. |
| `num_gpus_per_learner` | 1.0 | Choose the device with `CUDA_VISIBLE_DEVICES`; also set `MUJOCO_EGL_DEVICE_ID` to the same physical id. |
| `num_workers` | 9 | Keep the newer experiment's execution layout; sampling is CPU-bound (MuJoCo). |
| `ray_num_cpus` | 11 (`num_workers + 2`) | Prevent concurrent Ray jobs from fighting over the whole node. |
| `checkpoint_freq` | 500,000 | 10 recovery points; a crash costs at most 500k steps. |

## 8. Reading a run

| Metric | Meaning |
|---|---|
| `eval/musical_f1` | **The headline number.** Deterministic, from `MidiEvaluationWrapper`. |
| `eval/team_return_mean` | **One copy** of the shared team return. |
| `train/rllib_agent_sum_return_mean` | RLlib **sums** simultaneously-acting agents ⇒ ≈ N × the team return. Never report this as the team return. |
| `learner/<agent>/vf_explained_var` | Critic health. Check this first. |
| `learner/<agent>/value_clip_fraction` | Fraction of critic predictions whose update exceeds the ±0.2 value-delta clip. |
| `learner/<agent>/value_norm_mean`, `value_norm_std` | Running scale of unnormalized return targets. |
| `learner/<agent>/gradients_actor_global_norm`, `gradients_critic_global_norm` | Pre-clip gradient norms for the independent optimizers. |
| `time/env_steps_per_second` | Throughput; `time/eta_hours_to_target` is the projected finish. |
