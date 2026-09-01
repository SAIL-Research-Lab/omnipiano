"""Algorithm-independent multi-agent evaluation and scorecard output.

RLlib, CleanRL, or a hand-written decentralized controller can provide a
callable mapping ``(agent_id, observation)`` to one action.  A monolithic
baseline can instead provide one centralized callable over the complete
agent-observation mapping.  Episode execution and benchmark accounting live
here so controller families cannot drift in metrics or double-count the
shared team reward.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from omnipiano.benchmark.aggregation import aggregate_seed_values
from omnipiano.benchmark.metrics import (
    EpisodeTrace,
    attribution_window_weights,
    extract_note_events,
    match_note_events,
)
from omnipiano.utils.json_utils import strict_json_dumps


PolicyFn = Callable[[str, np.ndarray], np.ndarray]
CentralizedPolicyFn = Callable[
    [Mapping[str, np.ndarray]], Mapping[str, np.ndarray]
]


def _finite_or_nan(value: Any) -> float:
    """Convert numpy/Python numeric values to JSON-friendly floats."""

    return float(value)


def _aggregate_episode_values(values: Sequence[float]) -> Dict[str, float]:
    """Aggregate repeated eval episodes without calling them training seeds."""

    result = aggregate_seed_values(values)
    result["num_episodes"] = result.pop("num_seeds")
    result["num_successful_episodes"] = result.pop("num_successful_seeds")
    return result


def summarize_episode_scorecards(
    scorecards: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate all common scalar fields from one deterministic eval call."""

    if not scorecards:
        raise ValueError("scorecards must contain at least one episode")
    metric_maps = [card["metrics"] for card in scorecards]
    common_metrics = set.intersection(*(set(metrics) for metrics in metric_maps))
    aggregate = {
        "team_return": _aggregate_episode_values(
            [_finite_or_nan(card["team_return"]) for card in scorecards]
        ),
        "episode_steps": _aggregate_episode_values(
            [_finite_or_nan(card["episode_steps"]) for card in scorecards]
        ),
        "metrics": {
            name: _aggregate_episode_values(
                [_finite_or_nan(metrics[name]) for metrics in metric_maps]
            )
            for name in sorted(common_metrics)
        },
    }
    return {
        "num_episodes": len(scorecards),
        "episode_seeds": [int(card["seed"]) for card in scorecards],
        "aggregate": aggregate,
    }


def write_scorecards(
    output_dir: str | Path,
    scorecards: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Dict[str, str]:
    """Persist auditable per-episode JSONL plus a compact aggregate JSON."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    episodes_path = destination / "evaluation_episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as stream:
        for card in scorecards:
            stream.write(strict_json_dumps(card, sort_keys=True) + "\n")
    summary_path = destination / "evaluation_summary.json"
    summary_path.write_text(
        strict_json_dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"episodes": str(episodes_path), "summary": str(summary_path)}


def export_episode_trace(
    trace: EpisodeTrace,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Write the raw arrays and contact-attribution audit table.

    The NPZ is the machine-auditable source.  The CSV intentionally contains
    ordinary, human-readable rows that can be spot-checked against video at
    ``onset_seconds``.
    """

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    npz_path = destination / "episode_trace.npz"
    np.savez_compressed(
        npz_path,
        target_keys=trace.target_keys,
        actual_keys=trace.actual_keys,
        target_sustain=trace.target_sustain,
        actual_sustain=trace.actual_sustain,
        key_contact_force=trace.key_contact_force,
        hand_power=trace.hand_power,
        hand_collision_force=trace.hand_collision_force,
        control_timestep=np.asarray(trace.control_timestep),
        hand_names=np.asarray(trace.hand_names),
    )

    audit_path = destination / "contact_attribution_audit.csv"
    events = extract_note_events(trace.actual_keys)
    force = (
        None
        if trace.key_contact_force is None
        else np.asarray(trace.key_contact_force, dtype=np.float64)
    )
    window_weights = attribution_window_weights(trace.control_timestep, 0.05)
    fieldnames = [
        "actual_event_index",
        "pitch_key_index",
        "midi_pitch",
        "note_name",
        "onset_frame",
        "onset_seconds",
        "video_frame",
        "video_seconds",
        "offset_frame",
        "attribution_window_end_frame",
        "matched_target_event_index",
        "target_onset_frame",
        "onset_error_ms",
        "is_matched_correct_event",
        "primary_hand",
        "primary_agent",
        "contacting_hands",
        "per_hand_force_sample_sum_n",
        "per_hand_force_impulse_ns",
        "per_hand_peak_force_n",
        "per_hand_peak_force_frame",
        "attributed",
    ]
    target_events = extract_note_events(trace.target_keys)
    matches = match_note_events(
        target_events,
        events,
        control_timestep=trace.control_timestep,
        onset_tolerance_seconds=0.05,
    )
    actual_to_target = {
        match.predicted_index: match.target_index for match in matches
    }
    note_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    def midi_note_name(midi_pitch: int) -> str:
        return f"{note_names[midi_pitch % 12]}{midi_pitch // 12 - 1}"

    with audit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for event_index, event in enumerate(events):
            per_hand = np.zeros(len(trace.hand_names), dtype=np.float64)
            per_hand_peak = np.zeros(len(trace.hand_names), dtype=np.float64)
            per_hand_peak_frame = np.full(len(trace.hand_names), -1, dtype=np.int64)
            stop = event.onset_frame
            if force is not None and trace.hand_names:
                stop = min(
                    force.shape[0], event.onset_frame + len(window_weights)
                )
                force_window = force[event.onset_frame:stop, :, event.pitch]
                per_hand = force_window.sum(axis=0)
                per_hand_impulse = (
                    force_window
                    * window_weights[: force_window.shape[0], None]
                ).sum(axis=0)
                if force_window.shape[0]:
                    per_hand_peak = force_window.max(axis=0)
                    per_hand_peak_frame = (
                        event.onset_frame + force_window.argmax(axis=0)
                    )
            else:
                per_hand_impulse = np.zeros(
                    len(trace.hand_names), dtype=np.float64
                )
            touched = [
                trace.hand_names[index]
                for index in np.flatnonzero(per_hand_impulse > 0.0)
            ]
            primary_hand = ""
            primary_agent = ""
            if touched:
                primary_hand = trace.hand_names[
                    int(np.argmax(per_hand_impulse))
                ]
                primary_agent = trace.hand_to_agent[primary_hand]
            target_index = actual_to_target.get(event_index)
            target_event = (
                target_events[target_index] if target_index is not None else None
            )
            midi_pitch = event.pitch + 21
            writer.writerow({
                "actual_event_index": event_index,
                "pitch_key_index": event.pitch,
                "midi_pitch": midi_pitch,
                "note_name": midi_note_name(midi_pitch),
                "onset_frame": event.onset_frame,
                "onset_seconds": event.onset_frame * trace.control_timestep,
                # DmControlVideoWrapper records one reset frame before the
                # first trace step, hence trace frame j is MP4 frame j+1.
                "video_frame": event.onset_frame + 1,
                "video_seconds": (event.onset_frame + 1) * trace.control_timestep,
                "offset_frame": event.offset_frame,
                "attribution_window_end_frame": stop - 1,
                "matched_target_event_index": (
                    target_index if target_index is not None else ""
                ),
                "target_onset_frame": (
                    target_event.onset_frame if target_event is not None else ""
                ),
                "onset_error_ms": (
                    abs(event.onset_frame - target_event.onset_frame)
                    * trace.control_timestep
                    * 1000.0
                    if target_event is not None
                    else ""
                ),
                "is_matched_correct_event": target_event is not None,
                "primary_hand": primary_hand,
                "primary_agent": primary_agent,
                "contacting_hands": "|".join(touched),
                "per_hand_force_sample_sum_n": strict_json_dumps({
                    hand: float(per_hand[index])
                    for index, hand in enumerate(trace.hand_names)
                }, sort_keys=True),
                "per_hand_force_impulse_ns": strict_json_dumps({
                    hand: float(per_hand_impulse[index])
                    for index, hand in enumerate(trace.hand_names)
                }, sort_keys=True),
                "per_hand_peak_force_n": strict_json_dumps({
                    hand: float(per_hand_peak[index])
                    for index, hand in enumerate(trace.hand_names)
                }, sort_keys=True),
                "per_hand_peak_force_frame": strict_json_dumps({
                    hand: int(per_hand_peak_frame[index])
                    for index, hand in enumerate(trace.hand_names)
                }, sort_keys=True),
                "attributed": bool(touched),
            })

    metadata_path = destination / "episode_trace_metadata.json"
    metadata_path.write_text(
        strict_json_dumps({
            "control_timestep": trace.control_timestep,
            "hand_names": list(trace.hand_names),
            "hand_to_agent": dict(trace.hand_to_agent),
            "hand_key_ranges": {
                hand: list(key_range)
                for hand, key_range in trace.hand_key_ranges.items()
            },
            "contact_attribution_window_seconds": 0.05,
            "contact_force_threshold_n": 0.0,
            "video_frames_per_second": 1.0 / trace.control_timestep,
            "video_reset_frame_offset": 1,
            "video_frame_rule": "video_frame = trace_frame + 1",
            "num_actual_note_events": len(events),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "trace": str(npz_path),
        "audit": str(audit_path),
        "metadata": str(metadata_path),
    }


def evaluate_policy(
    env_id: str,
    policy_fn: Optional[PolicyFn],
    *,
    centralized_policy_fn: Optional[CentralizedPolicyFn] = None,
    episode_seeds: Sequence[int],
    output_dir: Optional[str | Path] = None,
    record_video: bool = False,
    record_sound: bool = False,
    export_trace: bool = False,
    export_actions: bool = False,
    camera_id: str = "piano/back",
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a policy and optionally persist scorecards/video/audit data.

    Only episode zero is recorded and exported.  Rebuilding the environment
    per episode guarantees that a requested seed is actually applied and
    prevents recording wrappers from carrying counters across evaluations.
    """

    if not episode_seeds:
        raise ValueError("episode_seeds must contain at least one seed")
    if (policy_fn is None) == (centralized_policy_fn is None):
        raise ValueError(
            "provide exactly one of policy_fn or centralized_policy_fn"
        )
    from omnipiano.multiagent import make_parallel

    destination = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else None
    )
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    scorecards = []
    audit_files: Dict[str, str] = {}
    action_files: Dict[str, str] = {}
    for episode_index, seed in enumerate(episode_seeds):
        should_record = bool(record_video and episode_index == 0)
        kwargs: Dict[str, Any] = {
            "seed": int(seed),
            "flatten_obs": True,
        }
        if should_record:
            if destination is None:
                raise ValueError("record_video=True requires output_dir")
            kwargs.update({
                "record_dir": str(destination / "video"),
                "record_every": 1,
                "camera_id": camera_id,
                "record_sound": record_sound,
            })
        env = make_parallel(env_id, **kwargs)
        try:
            observations, _ = env.reset(seed=int(seed))
            agent_ids = list(env.possible_agents)
            received_returns = {agent: 0.0 for agent in agent_ids}
            team_return = 0.0
            episode_steps = 0
            final_infos: Mapping[str, Any] = {}
            episode_actions = {agent: [] for agent in agent_ids}
            while env.agents:
                if centralized_policy_fn is not None:
                    raw_actions = centralized_policy_fn(observations)
                    if set(raw_actions) != set(env.agents):
                        raise ValueError(
                            "centralized policy action keys do not match live "
                            f"agents: {sorted(raw_actions)} != {sorted(env.agents)}"
                        )
                else:
                    raw_actions = {
                        agent: policy_fn(agent, observations[agent])
                        for agent in env.agents
                    }
                actions = {}
                for agent in env.agents:
                    action = np.asarray(
                        raw_actions[agent], dtype=np.float32
                    ).reshape(-1)
                    expected_shape = env.action_space(agent).shape
                    if action.shape != expected_shape:
                        raise ValueError(
                            f"policy action for {agent!r} has shape {action.shape}; "
                            f"expected {expected_shape}"
                        )
                    actions[agent] = np.clip(action, -1.0, 1.0)
                    if episode_index == 0 and export_actions:
                        episode_actions[agent].append(actions[agent].copy())
                observations, rewards, _, _, final_infos = env.step(actions)
                if rewards:
                    # Shared reward is broadcast.  It is a team signal and is
                    # therefore counted once, never summed across agents.
                    step_team_reward = float(rewards[agent_ids[0]])
                    if any(
                        not np.isclose(float(reward), step_team_reward)
                        for reward in rewards.values()
                    ):
                        raise RuntimeError(
                            "reward_mode='shared' produced non-identical "
                            f"agent rewards: {rewards}"
                        )
                    team_return += step_team_reward
                    for agent, reward in rewards.items():
                        received_returns[agent] += float(reward)
                episode_steps += 1

            metrics = dict(final_infos.get("_global_", {}))
            if not metrics:
                raise RuntimeError(
                    "episode ended without benchmark metrics in infos['_global_']"
                )
            card = {
                "env_id": env_id,
                "episode_index": episode_index,
                "seed": int(seed),
                "episode_steps": episode_steps,
                "team_return": team_return,
                "per_agent_received_return": received_returns,
                "metrics": metrics,
            }
            if context:
                card.update(context)
            scorecards.append(card)
            if episode_index == 0 and destination is not None and export_trace:
                audit_files = export_episode_trace(
                    env.get_last_episode_trace(), destination / "audit"
                )
            if episode_index == 0 and destination is not None and export_actions:
                actions_dir = destination / "audit"
                actions_dir.mkdir(parents=True, exist_ok=True)
                actions_path = actions_dir / "policy_actions.npz"
                np.savez_compressed(
                    actions_path,
                    **{
                        f"actions__{agent}": np.asarray(
                            episode_actions[agent], dtype=np.float32
                        )
                        for agent in agent_ids
                    },
                )
                actions_metadata_path = actions_dir / "policy_actions_metadata.json"
                actions_metadata_path.write_text(
                    strict_json_dumps({
                        "env_id": env_id,
                        "seed": int(seed),
                        "agent_order": agent_ids,
                        "num_steps": episode_steps,
                        "action_shapes": {
                            agent: list(env.action_space(agent).shape)
                            for agent in agent_ids
                        },
                    }, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                action_files = {
                    "actions": str(actions_path),
                    "actions_metadata": str(actions_metadata_path),
                }
        finally:
            env.close()

    summary = summarize_episode_scorecards(scorecards)
    summary.update({
        "env_id": env_id,
        "reward_accounting": "shared team reward counted once",
    })
    if context:
        summary.update(context)
    files: Dict[str, str] = {}
    if destination is not None:
        files = write_scorecards(destination, scorecards, summary)
        files.update(audit_files)
        files.update(action_files)
    return {"episodes": scorecards, "summary": summary, "files": files}
