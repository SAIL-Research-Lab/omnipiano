# Multi-agent baseline hyperparameters

Single source of truth for the OmniPiano MARL baselines. Every value here is
also written into each run's `run_config.json → effective_config`, so an
artifact is always self-describing; this file only adds the *rationale*.

`python -m omnipiano.multiagent.train --list-algos` prints the live registry.

## 1. Who owns what

| Layer | Owner | Must be identical across algorithms? |
|---|---|---|
| Protocol (steps, seeds, gamma, eval cadence) | `BenchmarkProtocolConfig` | **Yes** |
| PPO hyperparameters | `train.py` CLI defaults | **Yes** (IPPO vs MAPPO is a single-factor ablation) |
| Algorithm identity (critic input, global state) | `algos/<name>.py` | No — this IS the independent variable |
| Compute (workers, GPUs, CPU cap) | CLI / launch script | No (must not change results) |

## 2. Algorithm registry

| algo | actor input | critic input | env emits `s` | note |
|---|---|---|---|---|
| `ippo` | `o_i` | `o_i` | no | de Witt et al. 2020 |
| `mappo` | `o_i` | `s` (agent-specific) | yes | Yu et al. 2022 |
| `mappo-own-critic` | `o_i` | `o_i` | yes (unread) | validation only: must reproduce `ippo` bit-for-bit |
| `ippo-rllib-module` | `o_i` | `o_i` | no | debug anchor; different architecture, do NOT mix numbers |

Actor and critic share no trunk, and critics are **not** parameter-shared
across agents. Both choices are deliberate: they keep the per-agent parameter
count identical between IPPO and MAPPO, so "parameters per critic" cannot
become a second, uncontrolled variable.

## 3. Protocol

| Parameter | Value | Why |
|---|---|---|
| `total_env_steps` | 5,000,000 | **ENVIRONMENT** steps, not agent steps. With 2 agents a summed agent-step counter would end the run at 2.5M physical interactions. Enforced by `_ippo_common.extract_env_steps`. |
| `gamma` | 0.8 | Task property, not an algorithm knob — same value for SAC/PPO/TQC. Effective horizon ~5 control steps. |
| `eval_freq_env_steps` | 50,000 | Protocol cadence, giving 100 evaluation points over a 5M run. |
| `num_eval_eps` | 1 | Protocol default. Cross-seed aggregation supplies the main uncertainty estimate; each evaluation episode is deterministically reseeded. |
| `seed` | 0 / 1 / 2 | Paper needs all three. One seed is a validation run, not a result. |
| `eval_seed` | `seed + 10000`, episode `i` uses `+ i*10000` | Deterministic and non-colliding across training seeds. |

## 4. PPO — five corrected RLlib defaults

The PPO values below are shared by IPPO and MAPPO so the critic observation is
the controlled algorithmic difference.

| Parameter | RLlib default | Ours | Why |
|---|---|---|---|
| `vf_clip_param` | **10.0** | **1000.0** | Widens RLlib's value-loss clipping margin for this reward scale. Critic health must still be verified from `vf_explained_var` during the pilot. |
| `lambda_` (GAE λ) | 1.0 | 0.95 | Standard PPO/MAPPO GAE setting, shared by both baselines. |
| `clip_param` (ε) | 0.3 | 0.2 | PPO paper and MAPPO's ablation both favour 0.2. |
| `use_kl_loss` | True | False | RLlib adds an adaptive KL penalty *on top of* clipping. MAPPO's reference implementation uses clipping only; removing it also removes a hidden adaptive coefficient from the IPPO/MAPPO comparison. |
| `grad_clip` | None | 10.0 (`global_norm`) | MAPPO uses `max_grad_norm=10`. |

Other PPO values: `lr=3e-4`, `train_batch_size=4000` (env steps per iteration),
`minibatch_size=256`, `num_epochs=10` (MAPPO recommends 5–15),
`entropy_coeff=0.0`, `vf_loss_coeff=1.0`.

**Value normalization is deliberately NOT implemented.** It is MAPPO's factor
#1, but a correct implementation needs a custom `PPOTorchLearner`. Instead the
trainer logs `learner/<agent>/vf_explained_var` to W&B:

    EV = 1 − Var[R̂ − V] / Var[R̂]

`EV → 1` = perfect critic; `EV ≈ 0` = the critic outputs a constant; `EV < 0` =
worse than a constant. **If a pilot run shows EV flat near zero after the
`vf_clip_param` fix, PopArt becomes necessary.** Measure before adding
complexity.

## 5. Network

`MLP 256-256`, `tanh`, orthogonal init with gain `sqrt(2)` on hidden layers,
**0.01 on the policy output** (Engstrom et al. 2020: keeps the initial action
distribution near-isotropic so exploration is not collapsed at step 0), `1.0`
on the value head. Gaussian policy with a **state-independent** `log_std`
parameter, clamped to `[-5, 2]`, matching MAPPO's continuous-control reference.

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
| `num_workers` | 4 | Env runners. Sampling is CPU-bound (MuJoCo). |
| `ray_num_cpus` | `num_workers + 2` | **Required** when several jobs share a node, or the Ray instances fight over cores. |
| `checkpoint_freq` | 500,000 | 10 recovery points; a crash costs at most 500k steps. |

## 8. Reading a run

| Metric | Meaning |
|---|---|
| `eval/musical_f1` | **The headline number.** Deterministic, from `MidiEvaluationWrapper`. |
| `eval/team_return_mean` | **One copy** of the shared team return. |
| `train/rllib_agent_sum_return_mean` | RLlib **sums** simultaneously-acting agents ⇒ ≈ N × the team return. Never report this as the team return. |
| `learner/<agent>/vf_explained_var` | Critic health. Check this first. |
| `time/env_steps_per_second` | Throughput; `time/eta_hours_to_target` is the projected finish. |
