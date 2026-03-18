# SafeRoboPianist Framework Architecture Guide

`SafeRoboPianist` is a modular, extensible Reinforcement Learning benchmark framework built on top of DeepMind's `robopianist`. It is designed to provide an evaluation platform for **Safety** and **Robustness** in high-dimensional continuous control tasks.

The framework employs a "sandwich" architecture design:
1. **Top Layer (Registry & Examples)**: Provides a unified, extremely simple interface for users to instantiate pre-defined benchmark tasks via a Task Registry.
2. **Middle Layer (Gymnasium Wrappers)**: Handles signal-level noise injection, safety cost calculation, and metric extraction through decoupled wrappers.
3. **Bottom Layer (Task & mjcf)**: Intercepts the underlying physics compilation process to enable dynamic modification of the MuJoCo XML tree.

---

## Directory Structure and File Descriptions

```text
safe_robopianist/
├── configs.py
├── envs/
│   ├── __init__.py
│   └── safe_piano_env.py
├── examples/
│   ├── __init__.py
│   ├── iteration_summary_callback.py
│   └── run_template.py
├── logging/
│   └── logger_wrapper.py
├── metrics/
│   └── info_keys.py
├── safety/
│   └── constraints.py
├── tasks/
│   ├── registry.py
│   └── safe_piano_task.py
└── wrappers/
    ├── __init__.py
    ├── metrics_wrapper.py
    ├── robust_wrapper.py
    └── safety_wrapper.py
```

### 1. Core Configuration & Registry
* **`configs.py`**
  * **Role**: The "control panel" data structures of the framework. Uses Python `dataclass` to define all configuration classes.
  * **Content**: Contains `SafetyConfig` (list of safety constraints), `RobustConfig` (noise parameters), `TaskVariantConfig` (underlying physics variant switches), and `LoggingConfig`.
* **`tasks/registry.py`**
  * **Role**: The official repository of benchmark tasks.
  * **Content**: Defines `REGISTERED_TASKS`, a dictionary mapping task names (e.g., `"SafeRoboPianist-debug-Twinkle-WristLimit-v0"`) to their specific `TaskSpec` (which bundles the base environment name and all necessary configs). This allows users to load complex tasks with a single string.

### 2. Environment Assembly Factory (`envs/`)
* **`envs/safe_piano_env.py`**
  * **Role**: The entry point factory for environment creation (the `make` function).
  * **Responsibilities**:
    1. Looks up the requested `env_name` in the Registry to fetch predefined configs, falling back to defaults if not found.
    2. Calls the underlying `load_with_task` to instantiate the custom `SafePianoTask`.
    3. Handles compatibility conversion from `dm_env` to `gymnasium` (via Shimmy).
    4. Nests the various Wrappers in a strict order (`Metrics` -> `Safety` -> `Robust`).

### 3. Low-Level Physics Interceptor (`tasks/`)
* **`tasks/safe_piano_task.py`**
  * **Role**: The "scalpel" reaching deep into the physics engine.
  * **Responsibilities**: Inherits from the native `PianoWithShadowHands`. It intercepts and dynamically modifies the XML tree (`mjcf_model`) in memory *before* the MuJoCo physics engine is formally initialized. For example, by reading `TaskVariantConfig`, it can dynamically remove the left hand's actuator nodes to implement a "left hand immobile" task, or implement Domain Randomization in `before_step`.

### 4. Safety Constraint Definitions (`safety/`)
* **`safety/constraints.py`**
  * **Role**: The mathematical logic library defining what is "unsafe" (Strategy Pattern).
  * **Responsibilities**: Contains an abstract base class `BaseConstraint` and concrete constraint implementations (e.g., `JointMagnitudeConstraint`). Each constraint class is responsible for calculating the safety cost for the current step and specifying the key name for this cost in the log dictionary. This makes the extension of safety rules completely independent of the core environment code.

### 5. Modular Wrappers (`wrappers/`)
These wrappers follow the standard Gymnasium API and handle the data flow between the Agent and the underlying physics environment.

* **`wrappers/robust_wrapper.py`**
  * **Role**: Responsible for signal-level robustness testing.
  * **Responsibilities**: Injects action noise before the Agent's action is passed to the physics environment, and injects observation noise before the physics environment's observation is passed to the Agent.
* **`wrappers/safety_wrapper.py`**
  * **Role**: Responsible for executing safety constraints and exposing costs.
  * **Responsibilities**: Iterates through all constraint instances in the `SafetyConfig`, calculates the total safety cost, and injects the violation information and costs into the `info` dictionary. **Crucially, as a benchmark, it does NOT modify the raw reward**, leaving the reward-cost trade-off to the Safe RL algorithm.
* **`wrappers/metrics_wrapper.py`**
  * **Role**: Responsible for extracting pure task metrics.
  * **Responsibilities**: Extracts raw musical evaluation metrics (like F1 score) and individual sub-reward terms (like Energy, Fingering) from the underlying environment, saving them into the `info` dictionary using standardized key names.

### 6. Metrics & Logging System (`metrics/` & `logging/`)
* **`metrics/info_keys.py`**
  * **Role**: The unified "data dictionary" for the entire framework.
  * **Responsibilities**: Defines string constants for `InfoKeys` (step-level data) and `EpisodeInfoKeys` (episode-level data). It enforces standardized key names in the `info` dictionary (e.g., `safety/cost_total`, `task/f1`), eliminating the risk of spelling errors from hardcoded strings and ensuring perfect alignment of logs across all algorithms.
* **`logging/logger_wrapper.py`**
  * **Role**: The logger used for the evaluation phase (Eval).
  * **Responsibilities**: As a Gymnasium Wrapper, it automatically extracts standardized metrics from the `info` dictionary at the end of each Episode and appends them to a CSV file, achieving evaluation logging that is completely decoupled from the RL algorithm.

### 7. Running Examples (`examples/`)
* **`examples/run_template.py`**
  * **Role**: The standard launch template provided to users.
  * **Responsibilities**: Demonstrates how to use the `make` function with a registered task name, how to integrate with Stable Baselines 3 (PPO) for training, and how to conduct evaluation testing. It showcases the extremely clean user API achieved by the Registry pattern.
* **`examples/iteration_summary_callback.py`**
  * **Role**: The log aggregator used for the training phase (Train).
  * **Responsibilities**: A Stable Baselines 3 Callback. At the end of each PPO Iteration (Rollout), it collects the metrics of all completed Episodes, calculates their averages, and writes them to `train_iteration_summary.csv`, which is used for plotting smooth training curves.