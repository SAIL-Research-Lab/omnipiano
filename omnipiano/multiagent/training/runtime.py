"""Shared, testable runtime infrastructure for OmniPiano RLlib MARL baselines.

This module deliberately has no import-time dependency on Ray, PettingZoo, or
the OmniPiano package.  Keeping the metric aggregation and scheduling logic
lightweight makes it possible to unit-test the scientific bookkeeping without
starting MuJoCo or a Ray cluster.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


RLLIB_ENV_NAME = "omnipiano_ippo"
LEGACY_RLLIB_ENV_NAME = "omnipiano_4hand_ma"

# ``__common__`` is RLlib's reserved key for information that belongs to an
# environment rather than one agent.  ``_global_`` was used by the original
# OmniPiano MA implementation and remains readable for old wrappers/checkpoints.
GLOBAL_INFO_KEYS = ("__common__", "_global_")
REQUIRED_MUSICAL_METRICS = (
    "episode_task/musical_f1",
    "episode_task/musical_precision",
    "episode_task/musical_recall",
    "episode_task/sustain_f1",
)
COORDINATION_RATE_METRICS = (
    "episode_coordination/common_area_success_rate",
    "episode_coordination/common_area_duplicate_press_rate",
    "episode_coordination/inter_agent_collision_step_rate",
)
CHECKPOINT_METADATA_MARKERS = (
    # Current Ray Checkpointable and legacy RLlib metadata formats.
    "metadata.json",
    "rllib_checkpoint.json",
)
ALGORITHM_STATE_MARKERS = (
    # These distinguish an Algorithm checkpoint from an arbitrary RLlib
    # Checkpointable (for example, an RLModule also has metadata.json).
    "algorithm_state.pkl",
    "algorithm_state.msgpack",
    # Older RLlib msgpack spelling.
    "algorithm_state.msgpck",
)


def normalize_rllib_infos(infos: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate OmniPiano's legacy global-info key to RLlib's reserved key."""
    normalized = dict(infos)
    legacy = normalized.pop("_global_", None)
    if legacy is not None:
        if "__common__" in normalized:
            raise RuntimeError(
                "infos contain both '_global_' and '__common__'; global metrics "
                "would be ambiguous"
            )
        normalized["__common__"] = legacy
    return normalized


def wrap_parallel_env_for_rllib(parallel_env: Any) -> Any:
    """Build RLlib's PettingZoo adapter with compliant common-info handling."""
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    class _OmniPianoParallelPettingZooEnv(ParallelPettingZooEnv):
        def step(self, action_dict: Mapping[str, Any]) -> Any:
            observations, rewards, terminated, truncated, infos = super().step(
                action_dict
            )
            return (
                observations,
                rewards,
                terminated,
                truncated,
                normalize_rllib_infos(infos),
            )

    return _OmniPianoParallelPettingZooEnv(parallel_env)


def _json_default(value: Any) -> Any:
    """Convert common scientific-Python values to JSON scalars/lists."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one human-readable JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append and flush one machine-readable progress record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            + "\n"
        )
        f.flush()


def extract_env_steps(result: Mapping[str, Any]) -> int:
    """Read RLlib's lifetime *environment-step* counter, or fail loudly.

    Agent steps are not interchangeable with environment steps in this
    two-agent benchmark.  In particular, a counter that sums both agents would
    terminate the nominal 5M budget at roughly 2.5M physical interactions.
    """
    value = result.get("num_env_steps_sampled_lifetime")
    if value is None:
        env_runners = result.get("env_runners")
        if isinstance(env_runners, Mapping):
            value = env_runners.get("num_env_steps_sampled_lifetime")
    if value is None:
        raise RuntimeError(
            "RLlib result has no num_env_steps_sampled_lifetime counter; "
            "refusing to substitute an agent-step or ambiguous legacy counter"
        )
    steps = int(value)
    if steps < 0:
        raise RuntimeError(f"RLlib reported a negative env-step count: {steps}")
    return steps


def crossed_eval_targets(
    current_env_steps: int,
    next_eval_step: int,
    eval_freq_env_steps: int,
) -> Tuple[List[int], int]:
    """Return protocol eval thresholds crossed by the latest train iteration."""
    if eval_freq_env_steps <= 0:
        raise ValueError("eval_freq_env_steps must be positive")
    if next_eval_step <= 0:
        raise ValueError("next_eval_step must be positive")
    crossed: List[int] = []
    while current_env_steps >= next_eval_step:
        crossed.append(next_eval_step)
        next_eval_step += eval_freq_env_steps
    return crossed, next_eval_step


def next_periodic_target(
    current_env_steps: int,
    frequency_env_steps: int,
) -> int:
    """Return the first periodic threshold strictly after the current step.

    Resumed native runs restore a lifetime environment-step counter. Their
    evaluation and checkpoint schedules must skip thresholds already reached
    by the source checkpoint instead of replaying the schedule from step zero.
    """
    if current_env_steps < 0:
        raise ValueError("current_env_steps must be non-negative")
    if frequency_env_steps <= 0:
        raise ValueError("frequency_env_steps must be positive")
    return (current_env_steps // frequency_env_steps + 1) * frequency_env_steps


def _numeric_metrics(info: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in info.items():
        if not isinstance(key, str) or not key.startswith("episode_"):
            continue
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            number = float(value)
            if math.isfinite(number):
                metrics[key] = number
    return metrics


def extract_terminal_metrics(infos: Mapping[str, Any]) -> Dict[str, float]:
    """Extract and validate the authoritative terminal musical metrics."""
    global_info: Optional[Mapping[str, Any]] = None
    for key in GLOBAL_INFO_KEYS:
        candidate = infos.get(key)
        if isinstance(candidate, Mapping):
            global_info = candidate
            break

    # Compatibility with adapters that duplicate global episode metrics into
    # each agent's info instead of using a reserved common-info key.
    if global_info is None:
        for candidate in infos.values():
            if isinstance(candidate, Mapping) and all(
                key in candidate for key in REQUIRED_MUSICAL_METRICS
            ):
                global_info = candidate
                break

    if global_info is None:
        raise RuntimeError(
            "terminal infos contain no '__common__' (or legacy '_global_') "
            "musical metrics"
        )

    metrics = _numeric_metrics(global_info)
    missing = [key for key in REQUIRED_MUSICAL_METRICS if key not in metrics]
    if missing:
        raise RuntimeError(
            "terminal musical metrics are incomplete; missing " + ", ".join(missing)
        )
    bounded_metrics = REQUIRED_MUSICAL_METRICS + tuple(
        key for key in COORDINATION_RATE_METRICS if key in metrics
    )
    invalid = {
        key: metrics[key]
        for key in bounded_metrics
        if not 0.0 <= metrics[key] <= 1.0
    }
    if invalid:
        raise RuntimeError(f"terminal rate metrics must be in [0, 1]: {invalid}")
    return metrics


def shared_team_reward(rewards: Mapping[str, Any]) -> float:
    """Validate shared reward and return exactly one team-reward scalar."""
    if not rewards:
        raise RuntimeError("environment returned an empty reward mapping")
    values = [float(value) for value in rewards.values()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"non-finite per-agent reward(s): {values}")
    reference = values[0]
    if not all(math.isclose(value, reference, rel_tol=1e-7, abs_tol=1e-7)
               for value in values[1:]):
        raise RuntimeError(
            "cooperative MARL evaluation requires reward_mode='shared', but "
            "agents received "
            f"different rewards: {dict(rewards)}"
        )
    return reference


def _to_numpy_action(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


class DeterministicActionComputer:
    """Compute deterministic per-agent actions without a zero-action fallback.

    New-stack RLlib checkpoints expose RLModules.  Old-stack checkpoints expose
    Policies.  Backend selection happens once at construction; a failure during
    actual inference is never retried through a different backend and is always
    surfaced to the caller.
    """

    def __init__(self, algorithm: Any, agent_ids: Sequence[str]) -> None:
        self._handles: Dict[str, Tuple[str, Any]] = {}
        errors: Dict[str, List[str]] = {}
        for agent_id in agent_ids:
            handle: Optional[Any] = None
            agent_errors: List[str] = []

            get_module = getattr(algorithm, "get_module", None)
            if callable(get_module):
                try:
                    handle = get_module(agent_id)
                except Exception as exc:  # API-stack probe; not inference.
                    agent_errors.append(f"get_module: {type(exc).__name__}: {exc}")
            if handle is not None:
                self._handles[agent_id] = ("rlmodule", handle)
                continue

            get_policy = getattr(algorithm, "get_policy", None)
            if callable(get_policy):
                try:
                    handle = get_policy(agent_id)
                except Exception as exc:  # API-stack probe; not inference.
                    agent_errors.append(f"get_policy: {type(exc).__name__}: {exc}")
            if handle is not None:
                self._handles[agent_id] = ("policy", handle)
                continue

            errors[agent_id] = agent_errors or ["algorithm exposes neither backend"]

        if errors:
            detail = "; ".join(
                f"{agent}: {' | '.join(messages)}"
                for agent, messages in sorted(errors.items())
            )
            raise RuntimeError(f"could not resolve MARL inference backend: {detail}")

    @property
    def backends(self) -> Dict[str, str]:
        return {agent: backend for agent, (backend, _) in self._handles.items()}

    @staticmethod
    def _module_action(module: Any, observation: Any) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - RLlib torch requires it.
            raise RuntimeError("torch is required for RLModule inference") from exc

        obs_np = np.asarray(observation, dtype=np.float32)
        device = None
        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            try:
                device = next(parameters()).device
            except (StopIteration, TypeError):
                pass
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.inference_mode():
            output = module.forward_inference({"obs": obs_tensor.unsqueeze(0)})
        if not isinstance(output, Mapping):
            raise RuntimeError(
                f"RLModule.forward_inference returned {type(output).__name__}, expected mapping"
            )

        if "actions" in output:
            action = output["actions"]
        elif "action_dist_inputs" in output:
            get_dist_cls = getattr(module, "get_inference_action_dist_cls", None)
            if not callable(get_dist_cls):
                raise RuntimeError(
                    "RLModule returned action_dist_inputs but exposes no "
                    "get_inference_action_dist_cls()"
                )
            dist_cls = get_dist_cls()
            if dist_cls is None:
                raise RuntimeError("RLModule inference distribution class is None")
            distribution = dist_cls.from_logits(output["action_dist_inputs"])
            action = distribution.to_deterministic().sample()
        else:
            raise RuntimeError(
                "RLModule inference returned neither 'actions' nor 'action_dist_inputs'"
            )
        return _to_numpy_action(action)

    @staticmethod
    def _policy_action(policy: Any, observation: Any) -> np.ndarray:
        result = policy.compute_single_action(observation, explore=False)
        action = result[0] if isinstance(result, tuple) else result
        return _to_numpy_action(action)

    def __call__(self, agent_id: str, observation: Any, action_space: Any) -> np.ndarray:
        if agent_id not in self._handles:
            raise RuntimeError(f"no inference backend registered for agent {agent_id!r}")
        backend, handle = self._handles[agent_id]
        try:
            if backend == "rlmodule":
                action = self._module_action(handle, observation)
            else:
                action = self._policy_action(handle, observation)
        except Exception as exc:
            raise RuntimeError(
                f"deterministic {backend} inference failed for agent {agent_id!r}: {exc}"
            ) from exc

        expected_shape = tuple(action_space.shape)
        if action.ndim == len(expected_shape) + 1 and action.shape[0] == 1:
            action = action[0]
        if tuple(action.shape) != expected_shape:
            raise RuntimeError(
                f"agent {agent_id!r} action shape {tuple(action.shape)} != {expected_shape}"
            )
        if not np.isfinite(action).all():
            raise RuntimeError(f"agent {agent_id!r} produced a non-finite action")

        # CanonicalSpecWrapper clips to the same Box.  Applying the clip here
        # mirrors environment-runner behavior for direct RLModule inference.
        action = np.clip(action, action_space.low, action_space.high).astype(
            action_space.dtype, copy=False
        )
        if not action_space.contains(action):
            raise RuntimeError(f"agent {agent_id!r} action is outside its action space")
        return action


def summarize_episodes(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not episodes:
        raise ValueError("cannot summarize zero evaluation episodes")
    team_returns = [float(ep["team_return"]) for ep in episodes]
    lengths = [float(ep["episode_length"]) for ep in episodes]
    summary: Dict[str, float] = {
        "team_return_mean": float(np.mean(team_returns)),
        "team_return_std": float(np.std(team_returns)),
        # Compatibility aliases for the single-agent eval-summary schema.
        "return_mean": float(np.mean(team_returns)),
        "return_std": float(np.std(team_returns)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
    }
    metric_keys = sorted(
        {key for episode in episodes for key in episode.get("metrics", {})}
    )
    for key in metric_keys:
        values = [
            float(episode["metrics"][key])
            for episode in episodes
            if key in episode.get("metrics", {})
        ]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def evaluate_marl(
    algorithm: Any,
    env_id: str,
    *,
    eval_seed: int,
    num_episodes: int,
    record_dir: Optional[str] = None,
    record_resolution: Tuple[int, int] = (480, 640),
    camera_id: str = "piano/back",
    max_episode_steps: int = 1_000_000,
    env_factory: Optional[Callable[..., Any]] = None,
    action_computer: Optional[Callable[[str, Any, Any], np.ndarray]] = None,
    include_global_state: bool = False,
    inter_agent_collision_penalty_coef: float = 0.0,
    task: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run deterministic PettingZoo episodes and collect authoritative metrics."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if len(record_resolution) != 2 or any(int(v) <= 0 for v in record_resolution):
        raise ValueError("record_resolution must contain two positive integers")
    if not str(camera_id).strip():
        raise ValueError("camera_id must be non-empty")
    if (not math.isfinite(inter_agent_collision_penalty_coef)
            or inter_agent_collision_penalty_coef < 0.0):
        raise ValueError(
            "inter_agent_collision_penalty_coef must be finite and non-negative"
        )
    if env_factory is None:
        from omnipiano.multiagent.compile.environment import (
            make_parallel_from_task,
            resolve_registered_task,
        )
        if task is None:
            task = resolve_registered_task(env_id).to_dict()
        env_factory = make_parallel_from_task

    env_kwargs: Dict[str, Any] = {
        "seed": int(eval_seed),
        "flatten_obs": True,
        "inter_agent_collision_penalty_coef": float(
            inter_agent_collision_penalty_coef
        ),
    }
    if include_global_state:
        # MAPPO checkpoints were trained on [global_state | own]; the actor
        # ignores the global block but the observation SHAPE must still match.
        env_kwargs["include_global_state"] = True
    if record_dir is not None:
        env_kwargs.update(
            record_dir=record_dir,
            record_every=1,
            record_resolution=(int(record_resolution[0]), int(record_resolution[1])),
            camera_id=str(camera_id),
        )
    env = env_factory(task if task is not None else env_id, **env_kwargs)

    episodes: List[Dict[str, Any]] = []
    joint_action_computer = None
    try:
        agent_ids = list(env.possible_agents)
        if action_computer is None:
            candidate = getattr(algorithm, "compute_joint_actions", None)
            if callable(candidate):
                joint_action_computer = candidate
            else:
                action_computer = DeterministicActionComputer(algorithm, agent_ids)

        for episode_index in range(num_episodes):
            episode_seed = int(eval_seed + episode_index * 10_000)
            observations, _ = env.reset(seed=episode_seed)
            per_agent_returns = {agent: 0.0 for agent in agent_ids}
            team_return = 0.0
            episode_length = 0
            terminal_infos: Mapping[str, Any] = {}

            while env.agents:
                acting_agents = list(env.agents)
                actions: Dict[str, np.ndarray] = {}
                if joint_action_computer is not None:
                    try:
                        actions = joint_action_computer(
                            observations,
                            {agent: env.action_space(agent) for agent in acting_agents},
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "MARL evaluation inference failed at "
                            f"episode={episode_index}, step={episode_length}, "
                            f"joint_backend=native: {exc}"
                        ) from exc
                    if set(actions) != set(acting_agents):
                        raise RuntimeError(
                            "joint inference action keys do not match active agents: "
                            f"actions={sorted(actions)}, active={sorted(acting_agents)}"
                        )
                    for agent, action in actions.items():
                        action = np.asarray(action, dtype=np.float32)
                        space = env.action_space(agent)
                        if tuple(action.shape) != tuple(space.shape):
                            raise RuntimeError(
                                f"agent {agent!r} action shape {tuple(action.shape)} "
                                f"!= {tuple(space.shape)}"
                            )
                        if not np.isfinite(action).all() or not space.contains(action):
                            raise RuntimeError(
                                f"agent {agent!r} produced an invalid joint action"
                            )
                        actions[agent] = action
                else:
                    for agent in acting_agents:
                        try:
                            actions[agent] = action_computer(
                                agent, observations[agent], env.action_space(agent)
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                "MARL evaluation inference failed at "
                                f"episode={episode_index}, step={episode_length}, "
                                f"agent={agent!r}: {exc}"
                            ) from exc

                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
                del terminations, truncations  # env.agents is the PettingZoo done source.
                if set(rewards) != set(acting_agents):
                    raise RuntimeError(
                        "reward keys do not match agents that acted: "
                        f"acted={acting_agents}, rewards={sorted(rewards)}"
                    )
                team_return += shared_team_reward(rewards)
                for agent, reward in rewards.items():
                    per_agent_returns[agent] += float(reward)
                episode_length += 1
                terminal_infos = infos
                if episode_length >= max_episode_steps and env.agents:
                    raise RuntimeError(
                        "evaluation exceeded max_episode_steps="
                        f"{max_episode_steps} without terminating"
                    )

            if episode_length == 0:
                raise RuntimeError("evaluation episode ended without an environment step")
            metrics = extract_terminal_metrics(terminal_infos)
            episodes.append(
                {
                    "episode_index": int(episode_index),
                    "episode_seed": episode_seed,
                    "episode_length": int(episode_length),
                    "team_return": float(team_return),
                    "per_agent_returns": per_agent_returns,
                    "metrics": metrics,
                }
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            active_exception = sys.exc_info()[0] is not None
            try:
                close()
            except Exception:
                # Preserve the inference/metric exception the caller needs to
                # diagnose. A cleanup error is raised only on an otherwise
                # successful evaluation.
                if not active_exception:
                    raise

    result: Dict[str, Any] = {
        "env_id": env_id,
        "include_global_state": bool(include_global_state),
        "eval_seed": int(eval_seed),
        "num_eval_eps": int(num_episodes),
        "num_eval_episodes": int(num_episodes),
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
    }
    backends = getattr(action_computer, "backends", None)
    if backends is not None:
        result["inference_backends"] = dict(backends)
    elif joint_action_computer is not None:
        result["inference_backends"] = {
            agent: "native_joint" for agent in agent_ids
        }
    return result


def _checkpoint_result_path(result: Any) -> str:
    if isinstance(result, (str, Path)):
        return str(result)
    path = getattr(result, "path", None)
    if path is not None:
        return str(path)
    checkpoint = getattr(result, "checkpoint", None)
    checkpoint_path = getattr(checkpoint, "path", None)
    if checkpoint_path is not None:
        return str(checkpoint_path)
    raise RuntimeError(
        f"could not resolve checkpoint path from {type(result).__name__}"
    )


def save_algorithm_checkpoint(
    algorithm: Any,
    checkpoint_dir: Path,
    *,
    native_replay_transitions: Optional[int] = None,
) -> str:
    """Save an Algorithm checkpoint using the current or legacy RLlib API.

    ``native_replay_transitions`` is intentionally explicit.  ``None`` keeps
    the historical/native full-state behaviour, ``0`` writes a lightweight
    policy checkpoint, and a positive value retains only the newest replay
    tail for recovery.  RLlib checkpoints never receive this native-only
    argument.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    save_to_path = getattr(algorithm, "save_to_path", None)
    if callable(save_to_path):
        if native_replay_transitions is None:
            result = save_to_path(str(checkpoint_dir))
        else:
            result = save_to_path(
                str(checkpoint_dir),
                replay_transitions=int(native_replay_transitions),
            )
    else:
        save = getattr(algorithm, "save", None)
        if not callable(save):
            raise RuntimeError("RLlib Algorithm exposes no checkpoint save method")
        result = save(str(checkpoint_dir.parent))
    path = _checkpoint_result_path(result)
    try:
        validated = _validate_checkpoint_path(Path(path))
    except FileNotFoundError as exc:
        raise RuntimeError(f"RLlib reported an invalid checkpoint: {path}") from exc
    return str(validated)


def checkpoint_reference(checkpoint_path: Path, run_dir: Path) -> str:
    """Return a portable run-relative reference when the checkpoint is local."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    try:
        return str(checkpoint.relative_to(run_dir))
    except ValueError:
        return str(checkpoint)


def _validate_checkpoint_path(path: Path) -> Path:
    """Validate a complete RLlib or OmniPiano-native checkpoint."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    if path.is_file():
        # Ray's ancient, pre-directory format used files named checkpoint-N.
        if re.fullmatch(r"checkpoint-\d+", path.name):
            return path
        raise FileNotFoundError(
            f"not a recognized legacy RLlib Algorithm checkpoint file: {path}"
        )
    native_state = path / "state.pt"
    native_marker = path / "native_checkpoint.json"
    if native_state.is_file() and native_marker.is_file():
        try:
            marker = json.loads(native_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(
                f"invalid native checkpoint metadata: {native_marker}"
            ) from exc
        if marker.get("format") != "omnipiano-native-v1":
            raise FileNotFoundError(
                f"unsupported native checkpoint format in {native_marker}"
            )
        return path
    state_markers = [
        name for name in ALGORITHM_STATE_MARKERS if (path / name).is_file()
    ]
    if not state_markers:
        raise FileNotFoundError(
            "checkpoint directory has no Algorithm state file "
            f"{ALGORITHM_STATE_MARKERS}: {path}"
        )
    metadata_markers = [
        name for name in CHECKPOINT_METADATA_MARKERS if (path / name).is_file()
    ]
    if not metadata_markers:
        raise FileNotFoundError(
            "checkpoint directory has no supported metadata file "
            f"{CHECKPOINT_METADATA_MARKERS}: {path}"
        )
    return path
