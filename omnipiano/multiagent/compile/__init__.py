"""Pure configuration boundary for OmniPiano MARL experiments.

The compiler stays import-light; ``environment`` imports MuJoCo lazily.
"""

from omnipiano.multiagent.compile.compiler import (
    DEFAULT_TRAIN_CONFIG_PATH,
    compile_experiment,
    compile_task,
    resolve_train_config_path,
)
from omnipiano.multiagent.compile.environment import (
    make_parallel,
    make_parallel_from_task,
    prepare_task,
    resolve_registered_task,
)
from omnipiano.multiagent.compile.schema import (
    CONFIG_FIELDS,
    ROBUST_OBSERVATION_FIELDS,
    TASK_CONFIG_SCHEMA_VERSION,
    TRAIN_CONFIG_SCHEMA_VERSION,
    ExperimentRequest,
    ResolvedAgent,
    ResolvedExperiment,
    ResolvedHand,
    ResolvedTask,
)

__all__ = [
    "CONFIG_FIELDS",
    "ROBUST_OBSERVATION_FIELDS",
    "DEFAULT_TRAIN_CONFIG_PATH",
    "ExperimentRequest",
    "ResolvedAgent",
    "ResolvedExperiment",
    "ResolvedHand",
    "ResolvedTask",
    "TASK_CONFIG_SCHEMA_VERSION",
    "TRAIN_CONFIG_SCHEMA_VERSION",
    "compile_experiment",
    "compile_task",
    "make_parallel",
    "make_parallel_from_task",
    "prepare_task",
    "resolve_registered_task",
    "resolve_train_config_path",
]
