# OmniPiano Framework Architecture Guide

`OmniPiano` is a safety and robustness benchmark built on top of a vendored copy
of `robopianist`. The benchmark keeps three concerns decoupled:

1. Task reward comes from the underlying RoboPianist task.
2. Safety cost is computed separately and exposed through `info`.
3. Robustness perturbations are injected by wrappers without changing task logic.

This separation is intentional: as a benchmark, `OmniPiano` should return reward
and cost, not hard-code how an algorithm trades them off.

## Selected Demos

<p align="center">
  <img src="demos/Right_Hand_Only/preview.gif" alt="OmniPiano-Twinkle-RightHandOnly-v0 demo" width="420"/>
  <img src="demos/Collision_Safe/preview.gif" alt="OmniPiano-Twinkle-CollisionSafe-v0 demo" width="420"/>
</p>
<p align="center">
  <strong>Left:</strong> <code>OmniPiano-Twinkle-RightHandOnly-v0</code>
  &nbsp;&nbsp;&nbsp;
  <strong>Right:</strong> <code>OmniPiano-Twinkle-CollisionSafe-v0</code>
</p>
<p align="center">
  Animated previews are loaded from the repository <code>demos/</code> folder.
</p>

## Current Package Layout

The previous README described an older structure (`configs.py`,
`safe_piano_env.py`, `tasks/registry.py`, `safe_piano_task.py`, etc.). The
current codebase is organized as follows:

```text
OmniPiano/
├── __init__.py
├── README.md
├── configs/
│   └── __init__.py
├── envs/
│   ├── __init__.py
│   ├── registration.py
│   └── robopianist/
├── safety/
│   └── constraints.py
├── tasks/
│   ├── __init__.py
│   └── omni_piano_task.py
├── utils/
│   ├── env_unwrap.py
│   ├── info_keys.py
│   ├── iteration_summary_callback.py
│   └── logger_wrapper.py
└── wrappers/
    ├── __init__.py
    ├── metrics_wrapper.py
    ├── robust_wrapper.py
    └── safety_wrapper.py
```

There is also a repository-level example entry point at `examples/run_template.py`.

## High-Level Execution Flow

The end-to-end environment construction pipeline is:

1. `import OmniPiano`
   `OmniPiano/__init__.py` exposes `make` and `register`, and imports
   `OmniPiano.envs` for registration side effects.
2. Registered tasks are declared in `OmniPiano/envs/__init__.py`
   Each task is added with `register(id=..., base_env_name=..., ...)`.
3. `OmniPiano.envs.registration.make(...)` resolves configs
   If `env_name` is registered, `make(...)` loads its default `TaskSpec` and
   merges any caller overrides.
4. `suite.load_with_task(..., task_cls=OmniPianoTask, ...)` builds the base task
   This is where the vendored `robopianist` environment is instantiated.
5. dm_env-level wrappers are applied
   `MidiEvaluationWrapper` is always applied; `PianoSoundVideoWrapper` is added
   only when `record_dir` is provided.
6. The dm_env is converted to Gymnasium with Shimmy
   `DmControlCompatibilityV0` is used, with a temporary patch so the wrapped
   dm_env chain is still recognized as a `dm_control` environment.
7. Gym-level wrappers are applied in a fixed order
   `MetricsWrapper` -> `SafetyWrapper` -> `RobustWrapper`
8. Eval-only episode CSV logging is added when requested
   `SafeRecordEpisodeStatistics` writes eval episode metrics to CSV.

## Module Responsibilities

### `OmniPiano/__init__.py`

- Public package entry point.
- Ensures the vendored `robopianist` under `OmniPiano/envs/` is found first on
  `sys.path`.
- Re-exports `make` and `register`.

### `OmniPiano/configs/__init__.py`

- Defines the benchmark configuration dataclasses.
- `SafetyConfig`: list of instantiated safety constraints.
- `RobustConfig`: perturbation parameters such as `action_noise_std` and
  `obs_noise_std`.
- `TaskVariantConfig`: low-level task switches such as
  `left_hand_immobile` / `right_hand_immobile`.
- `LoggingConfig` and `EvalProtocolConfig`: shared logging/evaluation settings.

### `OmniPiano/envs/registration.py`

- Implements the `register()` + `make()` pattern.
- `register(...)` stores a `TaskSpec` in the benchmark registry.
- `make(...)` is the single environment factory used by examples and training
  code.
- This file is the assembly point that connects registry defaults, the
  underlying RoboPianist task, dm_env wrappers, Gymnasium conversion, and
  benchmark wrappers.

### `OmniPiano/envs/__init__.py`

- Contains the actual benchmark task declarations.
- Importing this module triggers the `register(...)` calls.
- Current tasks include:
  - Right-hand wrist limit
  - Right-hand only
  - Collision-safe
  - Power-constrained
  - Action-robust
  - Observation-robust

### `OmniPiano/tasks/omni_piano_task.py`

- Defines `OmniPianoTask`, which subclasses
  `robopianist.suite.tasks.piano_with_shadow_hands.PianoWithShadowHands`.
- Intercepts task construction after the default MJCF tree is built.
- Applies task variants such as freezing one hand by:
  - clamping joint ranges,
  - clamping actuator control ranges,
  - relocating the frozen hand away from the piano.
- This is where structural task variants belong, because they modify the
  underlying physical task rather than just post-process signals.

### `OmniPiano/safety/constraints.py`

- Defines the modular safety constraint interface (`BaseConstraint`) and concrete
  implementations.
- Example constraints currently implemented:
  - joint magnitude limits,
  - hand collision cost,
  - total actuator power cost.
- Constraints compute per-step cost independently from reward and write their own
  metric keys into `info`.

### `OmniPiano/wrappers/metrics_wrapper.py`

- Extracts task-side metrics after each environment step.
- Reads raw reward terms from the underlying composer task.
- Reads musical metrics from `MidiEvaluationWrapper`.
- Standardizes all exported keys through `OmniPiano/utils/info_keys.py`.
- This wrapper is applied to all tasks because benchmark logging should stay
  consistent across safety-only, robustness-only, and mixed tasks.

### `OmniPiano/wrappers/safety_wrapper.py`

- Computes benchmark safety cost from the configured constraints.
- Aggregates per-step cost and episode cost.
- Writes safety signals into `info`.
- Does **not** modify the environment reward.

### `OmniPiano/wrappers/robust_wrapper.py`

- Injects perturbations at the Gymnasium layer.
- Action noise is added before forwarding the action to the wrapped environment.
- Observation noise is added after receiving the observation from the wrapped
  environment.
- Logs perturbation magnitude for analysis.

### `OmniPiano/utils/info_keys.py`

- Defines the canonical `info` keys used across wrappers, callbacks, and loggers.
- Separates step-level keys from episode-level keys.
- Ensures all logging code depends on one naming source of truth.

### `OmniPiano/utils/env_unwrap.py`

- Centralizes wrapper unwrapping logic.
- Used when benchmark wrappers need to recover the underlying dm_env/composer
  environment safely from a Gymnasium wrapper stack.

### `OmniPiano/utils/logger_wrapper.py`

- Provides the eval-time episode CSV logger.
- Records episode metrics without coupling logging logic to a specific RL
  algorithm.

### `OmniPiano/utils/iteration_summary_callback.py`

- Stable-Baselines3 callback used by the example training script.
- Aggregates completed training episodes per iteration and writes
  `train_iteration_summary.csv`.

## Example Entry Point

The standard runnable example is `examples/run_template.py`, not
`OmniPiano/examples/run_template.py`.

It demonstrates how to:

1. create a registered environment with `make(...)`,
2. train a PPO policy,
3. save the final model,
4. run deterministic evaluation episodes,
5. record evaluation videos and CSV metrics.

## Notes on Installation

`setup.py` only defines the Python package and Python dependencies. It does not
create a conda environment.

For a fresh machine, installation should be thought of as three layers:

1. system dependencies such as `fluidsynth`, `portaudio`, and `ffmpeg`,
2. a Python environment such as a conda env,
3. `pip install -e .` to install `OmniPiano`.

If you want, this README can be extended further with a concrete installation
section once the repository-level environment/bootstrap files are finalized.