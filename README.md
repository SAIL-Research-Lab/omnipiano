# OmniPiano Framework Architecture Guide

`OmniPiano` is a safety and robustness benchmark built on top of DeepMind's `robopianist`.

The framework employs a "sandwich" architecture design:
1. **Top Layer (Registry & Examples)**: Provides a unified, extremely simple interface for users to instantiate pre-defined benchmark tasks via a Task Registry.
2. **Middle Layer (Gymnasium Wrappers)**: Handles signal-level noise injection, safety cost calculation, and metric extraction through decoupled wrappers.
3. **Bottom Layer (Task & mjcf)**: Intercepts the underlying physics compilation process to enable dynamic modification of the MuJoCo XML tree.

## Selected Demos

<p align="center">
  <strong><code>OmniPiano-Twinkle-RightHandOnly-v0</code></strong>
</p>
<p align="center">
  <img src="demos/Right_Hand_Only/preview.gif" alt="OmniPiano-Twinkle-RightHandOnly-v0 demo" width="520"/>
</p>
<p align="center">
  <strong><code>OmniPiano-Twinkle-CollisionSafe-v0</code></strong>
</p>
<p align="center">
  <img src="demos/Collision_Safe/preview.gif" alt="OmniPiano-Twinkle-CollisionSafe-v0 demo" width="520"/>
</p>
<p align="center">
  Animated previews are loaded from the repository <code>demos/</code> folder.
</p>

## Quick Start

### Installation

OmniPiano is supported on Linux and can be installed with Python >= 3.10.
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[Miniforge](https://github.com/conda-forge/miniforge) to manage your Python
environment.

> **Note**: Unlike the original RoboPianist, OmniPiano bundles the Shadow Hand
> model files and the default soundfont (`TimGM6mb.sf2`) directly in the
> repository. You do **not** need to run `git submodule` or `install_deps.sh`.

**Step 1 — Install system dependencies**

```bash
# Linux (Ubuntu / Debian)
sudo apt-get update
sudo apt-get install -y build-essential fluidsynth libfluidsynth-dev portaudio19-dev ffmpeg

# macOS
brew install portaudio fluid-synth ffmpeg
```

**Step 2 — Create a conda environment**

```bash
conda create -n pianist python=3.10 -y
conda activate pianist
```

**Step 3 — Clone and install**

```bash
git clone https://github.com/<your-org>/omnipiano.git
cd omnipiano
pip install -e .
```

**Step 4 — Verify installation**

```bash
python -c "from OmniPiano import make; env = make('OmniPiano-Twinkle-CollisionSafe-v0'); print('OK')"
```

**Step 5 (Optional) — Preprocess the PIG dataset**

The built-in tasks use short debug pieces (e.g., Twinkle Twinkle Little Star).
To unlock 150 additional pieces from the
[PIG dataset](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/),
download `PianoFingeringDataset_v1.2.zip`, extract it, then run:

```bash
robopianist preprocess --dataset-dir /PATH/TO/PianoFingeringDataset_v1.2
robopianist --check-pig-exists   # should print "PIG dataset is ready to use!"
```

**Step 6 (Optional) — Download a higher-quality soundfont**

The default soundfont (`TimGM6mb.sf2`) is bundled for basic audio synthesis.
For higher-quality piano sound in recorded videos:

```bash
robopianist soundfont --download
```

### Available Tasks

| Task ID | Category | Description |
|---------|----------|-------------|
| `OmniPiano-Twinkle-RightHandWristLimit-v0` | Safety | Right wrist pitch magnitude limit |
| `OmniPiano-Twinkle-RightHandOnly-v0` | Variant | Left hand frozen, right hand solo |
| `OmniPiano-Twinkle-CollisionSafe-v0` | Safety | Binary hand-hand collision cost |
| `OmniPiano-Twinkle-PowerConstrained-v0` | Safety | Dense actuator power cost |
| `OmniPiano-Twinkle-ActionRobust-v0` | Robustness | Gaussian noise on actions |
| `OmniPiano-Twinkle-ObservationRobust-v0` | Robustness | Gaussian noise on observations |
| `OmniPiano-TwinkleRousseau-CollisionForce-v0` | Safety | Continuous contact force cost |

### Minimal Example

```python
from OmniPiano import make

env = make("OmniPiano-Twinkle-CollisionSafe-v0")
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    cost = info["safety/cost_total"]        # per-step safety cost
    ep_cost = info["safety/ep_cost_total"]  # cumulative episode cost

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training with SB3 (PPO)

A full training + evaluation template is provided at `examples/run_template.py`:

```bash
cd examples
python run_template.py
```

This will:
1. Train a PPO policy on the selected task with parallel environments
2. Save the trained model to `examples/logs/<run>/final_model.zip`
3. Run deterministic evaluation episodes with video recording
4. Export training curves to `train_iteration_summary.csv` and evaluation metrics to `eval_episode_log.csv`

To switch tasks, edit the `env_name` variable in `run_template.py`:

```python
# Choose one:
env_name = "OmniPiano-Twinkle-RightHandWristLimit-v0"
env_name = "OmniPiano-Twinkle-CollisionSafe-v0"
env_name = "OmniPiano-Twinkle-PowerConstrained-v0"
# ... etc.
```

For custom SafeRL algorithms, use the `make()` function directly — the environment
follows the standard Gymnasium API and exposes `reward`, `cost`, and task metrics
through `info`. See `task_description.md` and `design_rationale.md` for details.

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

### 1. Package Entry Point
* **`OmniPiano/__init__.py`**
  * **Role**: The public package entry point.
  * **Responsibilities**: Ensures the vendored `robopianist` under `OmniPiano/envs/` is found first on `sys.path`, re-exports `make` and `register`, and triggers task registration through the import of `OmniPiano.envs`.

### 2. Core Configuration & Registry
* **`OmniPiano/configs/__init__.py`**
  * **Role**: Defines the benchmark configuration dataclasses.
  * **Content**:
    1. `SafetyConfig`: list of instantiated safety constraints.
    2. `RobustConfig`: perturbation parameters such as `action_noise_std` and `obs_noise_std`.
    3. `TaskVariantConfig`: low-level task switches such as `left_hand_immobile` / `right_hand_immobile`.
    4. `LoggingConfig`: shared logging configuration such as log directory and split.
    5. `EvalProtocolConfig`: shared evaluation protocol settings such as seeds and number of evaluation episodes.
* **`OmniPiano/envs/__init__.py`**
  * **Role**: The official repository of benchmark task declarations.
  * **Content**: Registers all benchmark tasks through `register(...)`. Each task maps a human-readable benchmark ID to its `TaskSpec`, which bundles the base environment name plus the default safety, robustness, and task-variant configurations. Current tasks include right-hand wrist limit, right-hand only, collision-safe, power-constrained, action-robust, and observation-robust.

### 3. Environment Assembly Factory (`envs/`)
* **`OmniPiano/envs/registration.py`**
  * **Role**: The entry point factory for environment creation.
  * **Responsibilities**:
    1. Implements the `register()` + `make()` pattern.
    2. Resolves a registered task name into its `TaskSpec` and default configs, while still allowing caller-provided overrides.
    3. Calls the underlying `load_with_task(...)` to instantiate `OmniPianoTask` on top of the vendored RoboPianist task.
    4. Adds dm_env-level wrappers such as `MidiEvaluationWrapper` and optional `PianoSoundVideoWrapper`.
    5. Converts the dm_env into a Gymnasium environment via Shimmy.
    6. Applies the benchmark wrapper stack in a fixed order: `MetricsWrapper` -> `SafetyWrapper` -> `RobustWrapper`.
    7. Optionally adds eval-time CSV episode logging when `log_split == "eval"`.

### 4. Low-Level Physics Interceptor (`tasks/`)
* **`OmniPiano/tasks/omni_piano_task.py`**
  * **Role**: The low-level task customization layer that reaches into the MuJoCo/MJCF build process.
  * **Responsibilities**: Subclasses `robopianist.suite.tasks.piano_with_shadow_hands.PianoWithShadowHands`, intercepts task construction after the default MJCF tree is built, and applies structural task variants such as freezing one hand by clamping joint ranges, clamping actuator control ranges, and relocating the frozen hand away from the piano. This is the correct layer for physical task modifications because it changes the underlying MJCF/composer task itself rather than only post-processing observations or actions.

### 5. Safety Constraint Definitions (`safety/`)
* **`OmniPiano/safety/constraints.py`**
  * **Role**: The mathematical logic library defining what is "unsafe".
  * **Responsibilities**: Defines the modular safety constraint interface (`BaseConstraint`) and concrete implementations such as joint magnitude limits, hand collision cost, and total actuator power cost. Each constraint computes its per-step cost independently from reward, writes its own metric key into `info`, and can be extended without modifying the core environment factory or wrapper pipeline.

### 6. Modular Wrappers (`wrappers/`)
These wrappers follow the standard Gymnasium API and handle the data flow between the agent and the underlying physics environment.

* **`OmniPiano/wrappers/metrics_wrapper.py`**
  * **Role**: Responsible for extracting task-side metrics.
  * **Responsibilities**: Reads raw reward terms from the underlying composer task, reads musical metrics from `MidiEvaluationWrapper`, standardizes exported keys through `OmniPiano/utils/info_keys.py`, and aggregates episode-level task metrics. This wrapper is applied to all tasks to keep benchmark logging consistent across safety-only, robustness-only, and mixed tasks.
* **`OmniPiano/wrappers/safety_wrapper.py`**
  * **Role**: Responsible for executing safety constraints and exposing costs.
  * **Responsibilities**: Iterates through all configured safety constraints, aggregates per-step and per-episode safety cost, writes safety signals into `info`, and intentionally does **not** modify the environment reward. This preserves the benchmark principle that reward-cost trade-offs should be handled by the RL algorithm rather than hard-coded inside the environment.
* **`OmniPiano/wrappers/robust_wrapper.py`**
  * **Role**: Responsible for signal-level robustness perturbation.
  * **Responsibilities**: Injects action noise before forwarding actions to the wrapped environment, injects observation noise after receiving observations, and logs perturbation magnitude for analysis. This keeps robustness perturbations decoupled from both the base task definition and the safety-cost computation logic.

### 7. Metrics & Logging Utilities (`utils/`)
* **`OmniPiano/utils/info_keys.py`**
  * **Role**: The unified data dictionary for the framework.
  * **Responsibilities**: Defines the canonical step-level and episode-level `info` keys used across wrappers, callbacks, and loggers. This avoids hard-coded string duplication and keeps metric naming consistent across training, evaluation, CSV export, and future algorithm integrations.
* **`OmniPiano/utils/env_unwrap.py`**
  * **Role**: The shared unwrapping utility layer.
  * **Responsibilities**: Centralizes wrapper-unwrapping logic so benchmark wrappers can safely recover the underlying dm_env/composer environment from a Gymnasium wrapper stack. This makes the metrics and safety code less fragile when the wrapper chain changes.
* **`OmniPiano/utils/logger_wrapper.py`**
  * **Role**: The evaluation-phase episode logger.
  * **Responsibilities**: Records episode metrics to CSV without coupling logging to a specific RL algorithm. This is mainly used for evaluation-time episode logging.
* **`OmniPiano/utils/iteration_summary_callback.py`**
  * **Role**: The training-phase iteration logger used by the example script.
  * **Responsibilities**: Aggregates completed training episodes per iteration and writes `train_iteration_summary.csv`. This is mainly used for training-time summary logging in the Stable-Baselines3 example pipeline.

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