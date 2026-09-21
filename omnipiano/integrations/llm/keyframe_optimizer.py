"""Let an LLM write and refine an open-loop piano controller."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from robopianist.models.hands import shadow_hand as shadow_hand_model
from robopianist.models.hands import shadow_hand_constants

from omnipiano.utils.env_unwrap import get_composer_env_from_gym
from omnipiano.utils.info_keys import EpisodeInfoKeys

from .policy import ClaudeAgent, OpenAIClient, OpenAIResponseClient
from .policy_config import KeyframeOptimizerConfig


_FINGER_NAMES = [
    "right_thumb", "right_index", "right_middle", "right_ring", "right_little",
    "left_thumb", "left_index", "left_middle", "left_ring", "left_little",
]
_MAX_ROBOT_REFERENCE_CALLS = 5
_MAX_FK_CALLS = 6
_MAX_TRAJECTORY_CALLS = 8
_TOOL_CALL_LIMITS = {
    "get_robot_reference": _MAX_ROBOT_REFERENCE_CALLS,
    "fk_query": _MAX_FK_CALLS,
    "get_practice_trajectory": _MAX_TRAJECTORY_CALLS,
}


class KeyframeOptimizer:
    """RoboDojo-style program synthesis followed by simulation practice."""

    def __init__(self, config: KeyframeOptimizerConfig):
        self.env, self.action_space = config.env, config.action_space
        self.path = Path(config.path)
        self.program_path = self.path.with_name(
            self.path.stem.replace("keyframes_", "play_") + ".py"
        )
        self.num_optimization_steps = config.num_optimization_steps
        self.target_f1, self.practice_seed = config.target_f1, config.practice_seed
        self.resume = getattr(config, "resume", False)
        self.trajectories = {}
        model_name = config.model.rsplit("/", 1)[-1].lower()
        if model_name.startswith("claude"):
            client_class = ClaudeAgent
        elif model_name == "gpt-6-astra":
            client_class = OpenAIResponseClient
        else:
            client_class = OpenAIClient
        self.client = client_class(
            config.model, base_url=config.base_url, api_key=config.api_key,
            thinking=config.thinking,
            temperature=config.temperature, max_tokens=config.max_tokens,
        )

    def optimize(self):
        self.env.reset(seed=self.practice_seed)
        physics = get_composer_env_from_gym(self.env).physics
        self._fk_state = physics.get_state().copy()
        self._fk_ctrl = physics.data.ctrl.copy()
        context = self._context()
        self._robot_context = context
        self._fk_joint_size = len(context["fk_joint_names"])
        self._fk_joint_low = np.asarray(context["fk_joint_low"], dtype=float)
        self._fk_joint_high = np.asarray(context["fk_joint_high"], dtype=float)
        if self.resume:
            messages, best, history = self._load_resume(context)
        else:
            messages = [
                {"role": "system", "content": "You synthesize safe robot piano controllers. Use tools when useful."},
                {"role": "user", "content": self._prompt(context)},
            ]
            self._save_conversation(messages)
            best, history = None, []
        for attempt in range(len(history) + 1, self.num_optimization_steps + 1):
            if best is not None and best[0]["f1"] >= self.target_f1:
                break
            usage_before = self.client.token_usage.copy()
            source = self._unpracticed_source(messages)
            if source is None:
                _, source, actions = self._request_and_run(messages, context)
            else:
                actions = self._run_program(source, context)
            metrics, trajectory, key_geometry = self._practice(actions, attempt)
            metrics["token_usage"] = {
                name: self.client.token_usage[name] - usage_before[name]
                for name in self.client.token_usage
            }
            history.append(metrics)
            trajectory_path = self._save_trajectory(attempt, metrics, key_geometry, trajectory)
            error_ranges = self._error_ranges(trajectory)
            print(
                f"practice {attempt}/{self.num_optimization_steps}: "
                f"f1={metrics['f1']:.4f}, precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"tokens={metrics['token_usage']['total_tokens']}"
            )
            if best is None or metrics["f1"] > best[0]["f1"]:
                best = metrics, source, actions, attempt
            self._save(context, history, *best)
            if metrics["f1"] >= self.target_f1:
                break
            self._append_message(messages, {"role": "user", "content": self._feedback(
                attempt, metrics, best[3], best[0]["f1"], trajectory_path.name, error_ranges,
            )})
        if best is None:
            raise ValueError("no completed practice attempt to freeze")
        self._save(context, history, *best)
        print(f"best controller -> {self.program_path}")
        print(f"frozen keyframes -> {self.path}")
        print(f"token usage -> {self.client.token_usage}")

    def _load_resume(self, context):
        conversation_path = self.path.with_name(self.path.stem + "_conversation.json")
        conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
        messages = conversation["messages"]
        # if conversation["model"] != self.client.model or len(messages) < 2:
        #     raise ValueError("saved conversation model or messages do not match this run")
        original_prompt = messages[1].get("content", "")
        try:
            original_context = json.loads(original_prompt.split("**Context:**\n", 1)[1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ValueError("saved conversation has no usable environment context") from error
        for key in ("environment_id", "num_steps", "action_size", "action_names",
                    "control_timestep", "score", "hand_partitions"):
            if original_context.get(key) != context.get(key):
                raise ValueError(f"saved conversation environment differs at {key}")
        self.client.token_usage.update(conversation["token_usage"])

        history, best = [], None
        if self.path.is_file():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            history = saved["practice_attempts"]
            if [item["attempt"] for item in history] != list(range(1, len(history) + 1)):
                raise ValueError("saved practice attempts are not consecutive")
            summary_path = self.path.with_name("optimization_summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["practice_seed"] != self.practice_seed:
                raise ValueError("saved practice seed differs from --eval-seed")
            if summary["practice_attempts"] != history:
                raise ValueError("saved practice summary differs from keyframes")
            for item in history:
                attempt = item["attempt"]
                trajectory_path = self.path.with_name(f"attempt_{attempt:03d}_trajectory.json")
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                if trajectory["attempt"] != attempt or trajectory["metrics"] != item:
                    raise ValueError(f"saved trajectory {attempt} differs from practice history")
                self.trajectories[attempt] = trajectory["steps"]
            best_attempt = saved["best_attempt"]
            if best_attempt not in range(1, len(history) + 1):
                raise ValueError("invalid saved best attempt")
            if saved["practice_metrics"] != history[best_attempt - 1]:
                raise ValueError("saved best metrics differ from practice history")
            source = self.program_path.read_text(encoding="utf-8")
            actions = np.asarray(saved["actions"], dtype=np.float32)
            if actions.shape != (context["num_steps"], context["action_size"]):
                raise ValueError("saved best actions have the wrong shape")
            best = (saved["practice_metrics"], source, actions, best_attempt)

        # A practice may have been saved before its feedback was appended.
        if history and history[-1]["f1"] < self.target_f1:
            last_attempt = history[-1]["attempt"]
            feedback_written = any(
                message.get("role") == "user"
                and message.get("content", "").startswith(
                    f"Revise the preceding play.py. Practice completed: {{\"attempt\":{last_attempt},"
                )
                for message in messages
            )
            if not feedback_written:
                trajectory = self.trajectories[last_attempt]
                self._append_message(messages, {"role": "user", "content": self._feedback(
                    last_attempt, history[-1], best[3], best[0]["f1"],
                    f"attempt_{last_attempt:03d}_trajectory.json", self._error_ranges(trajectory),
                )})
        print(f"resuming {self.path.parent}: {len(history)} completed practices, "
              f"{len(messages)} conversation messages")
        return messages, best, history

    @classmethod
    def _unpracticed_source(cls, messages):
        last_feedback = max(
            (index for index, message in enumerate(messages)
             if message.get("role") == "user" and message.get("content", "").startswith(
                 "Revise the preceding play.py. Practice completed:")),
            default=1,
        )
        if not messages or messages[-1].get("role") != "assistant":
            return None
        if len(messages) - 1 <= last_feedback:
            return None
        content = messages[-1].get("content") or ""
        try:
            source = cls._python_block(content)
            cls._validate_program(source)
            return source
        except (SyntaxError, ValueError):
            return None

    def _context(self):
        composer = get_composer_env_from_gym(self.env)
        task, physics = composer.task, composer.physics
        count = len(task._notes)
        buffer_time = float(getattr(task, "_initial_buffer_time", 0.0))
        score = []
        for note in task._midi.seq.notes:
            onset, part = float(note.start_time) + buffer_time, int(note.part)
            if onset < count * task.control_timestep:
                score.append({
                    "key": int(note.pitch) - 21,
                    "onset": onset, "offset": float(note.end_time) + buffer_time,
                    "finger": part if 0 <= part <= 9 else None,
                })
        pedals, previous = [], None
        for step, sustain in enumerate(task._sustains[:count]):
            sustain = bool(sustain)
            if sustain != previous:
                pedals.append({"time": step * task.control_timestep, "down": sustain})
                previous = sustain
        keys = physics.bind(task.piano.key_geoms)
        spec = task.action_spec(physics)
        action_metadata = self._action_metadata(task)
        hand_specs = task.hand_specs
        hand_partitions = [{
            "hand": hand.name,
            "key_range": list(hand_spec.key_range) if hand_spec.key_range is not None else None,
            "y_range": (list(hand_spec.resolved_y_range)
                        if hand_spec.resolved_y_range is not None else None),
        } for hand, hand_spec in zip(task.hands, hand_specs)]
        fk_joints = [joint for hand in task.hands for joint in hand.joints]
        fk_joint_ranges = self._joint_ranges(physics, fk_joints)
        return {
            "environment_id": getattr(getattr(self.env, "spec", None), "id", None),
            "control_timestep": float(task.control_timestep), "num_steps": count,
            "hands": [h.name for h in task.hands],
            "hand_count": len(task.hands),
            "static_partition": any(partition["y_range"] is not None for partition in hand_partitions),
            "hand_partitions": hand_partitions,
            "action_size": int(self.action_space.shape[0]),
            "action_names": [a.full_identifier for h in task.hands for a in h.actuators]
            + ["sustain"],
            "action_low": np.asarray(spec.minimum).tolist(),
            "action_high": np.asarray(spec.maximum).tolist(),
            "action_metadata": action_metadata,
            "initial_joint_angles": {
                h.name: physics.bind(h.joints).qpos.tolist() for h in task.hands
            },
            "fk_joint_names": [joint.full_identifier for joint in fk_joints],
            "fk_joint_low": fk_joint_ranges[:, 0].tolist(),
            "fk_joint_high": fk_joint_ranges[:, 1].tolist(),
            "fingertip_names": {
                h.name: [s.name for s in h.fingertip_sites] for h in task.hands
            },
            "initial_fingertip_positions": {
                h.name: physics.bind(h.fingertip_sites).xpos.tolist() for h in task.hands
            },
            "keys": [{
                "key": i, "position": p.tolist(), "size": s.tolist(),
                "black": bool(task.piano.is_key_black(i)),
            } for i, (p, s) in enumerate(zip(keys.xpos, keys.size))],
            "finger_names": _FINGER_NAMES,
            "score": score, "pedal": pedals,
        }

    @staticmethod
    def _joint_ranges(physics, joints):
        return np.asarray(physics.bind(joints).range, dtype=float).reshape(-1, 2)

    @staticmethod
    def _action_metadata(task):
        metadata = []
        for hand in task.hands:
            joints = {joint.name: joint for joint in hand.joints}
            for actuator in hand.actuators:
                joint = getattr(actuator, "joint", None)
                if isinstance(joint, str):
                    joint = joints.get(joint)
                if joint is None:
                    target = getattr(actuator, "tendon", None)
                    target_name = getattr(target, "full_identifier", target)
                    joint_type = "fixed_tendon"
                else:
                    target_name = joint.full_identifier
                    joint_type = joint.type or "hinge"
                metadata.append({
                    "target": target_name,
                    "joint_type": joint_type,
                })
        metadata.append({
            "target": "piano sustain pedal",
            "joint_type": "pedal",
        })
        return metadata

    def _prompt(self, context):
        return (
            "**Task:**\n"
            "- Control the Shadow Hand to press the correct piano keys at their scheduled times.\n"
            "- Avoid both wrong and missed key presses; maximize episode task F1.\n\n"
            "- Respect environment_id, hand_count, static_partition, and hand_partitions in Context.\n\n"
            "**Score:**\n"
            "- key is a zero-based piano-key index (0-87); pitch = key + 21.\n"
            "- onset and offset are seconds from episode start; a note is active when onset <= t < offset.\n"
            "- finger indexes finger_names; use finger_names[finger] for the annotated name and "
            "its right_/left_ prefix for the annotated hand. fingertip_names are the actual "
            "available robot sites and may differ.\n\n"
            "**Rules:**\n"
            "- action_names, fk_joint_names, and tool return values are the only authoritative source "
            "for actuator and joint names. Never construct names by adding a hand prefix, rh_/lh_, "
            "or a suffix; look up action indices using complete identifiers.\n"
            "- action_metadata and action_names are aligned by index.\n"
            "- Read JSON context from sys.stdin and print only a JSON array shaped "
            f"[{context['num_steps']}, {context['action_size']}].\n"
            "- Each row is physical actuator targets in action_names order, within action_low/action_high, "
            f"for one {context['control_timestep']}-second step; do not normalize them.\n"
            "- Use only json, sys, math, and numpy. Return the complete play.py program in one Python code block.\n"
            "- After practice, make a substantive, diagnostics-driven change to controller logic or parameters; "
            "do not reproduce a previous program unchanged.\n\n"
            "**Tools:**\n"
            f"- get_robot_reference provides exact naming, actuator mappings, joint groups, and forearm semantics; use it at most {_MAX_ROBOT_REFERENCE_CALLS} times.\n"
            "- fk_query verifies complete poses. fk_query returns joint positions, not an action row: match direct "
            "joint actuators by action_metadata target name and tune tendon actuators separately.\n"
            "- fk_query accepts 1-8 complete "
            f"{len(context['fk_joint_names'])}-D joint-qpos vectors in fk_joint_names order and returns "
            "actual joint angles plus all fingertip world coordinates.\n"
            f"- You may call fk_query at most {_MAX_FK_CALLS} times while producing one program. Batch up "
            "to 8 carefully chosen poses into each call instead of querying poses one at a time.\n"
            "- FK describes one complete pose: it does not execute actions or simulate a key press, and "
            "coordinate differences between poses are not additive offsets.\n"
            f"- get_practice_trajectory is available only after a practice attempt completes; it reads any requested "
            f"range from that attempt, where error:true steps identify errors. Use it at most {_MAX_TRAJECTORY_CALLS} times.\n\n"
            "**Context:**\n"
            + json.dumps(context, separators=(",", ":"))
        )

    def _tool_response(self, call):
        if not isinstance(call, dict):
            call = call.model_dump(exclude_none=True, mode="json")
        name = call["function"]["name"]
        try:
            result = self._tool_result(name, json.loads(call["function"]["arguments"]))
        except (json.JSONDecodeError, ValueError) as error:
            result = {"error": str(error)}
        return {
            "role": "tool", "tool_call_id": call["id"],
            "content": json.dumps(result, separators=(",", ":")),
        }

    def _request_program(self, messages):
        latest_feedback = max(
            (index for index, item in enumerate(messages)
             if item.get("role") == "user" and item.get("content", "").startswith(
                 "Revise the preceding play.py. Practice completed:")),
            default=1,
        )
        if self._unpracticed_source(messages) is not None:
            latest_feedback = len(messages) - 1
        current_attempt = messages[latest_feedback + 1:]
        tool_calls = {name: 0 for name in _TOOL_CALL_LIMITS}
        for item in current_attempt:
            for call in item.get("tool_calls") or ():
                name = call["function"]["name"]
                if name in tool_calls:
                    tool_calls[name] += 1
        all_tools_notice_sent = any(
            item.get("role") == "user" and item.get("content", "").startswith(
                "All available tool budgets are exhausted")
            for item in current_attempt
        )
        # A crash may happen after an assistant tool call but before every
        # result has been written. Finish those calls before asking the model.
        answered = {item.get("tool_call_id") for item in current_attempt
                    if item.get("role") == "tool"}
        for item in current_attempt:
            for call in item.get("tool_calls") or ():
                if call["id"] not in answered:
                    self._append_message(messages, self._tool_response(call))
                    answered.add(call["id"])
        while True:
            message = self.client.chat(messages, tools=self._tools(
                include_reference=tool_calls["get_robot_reference"] < _MAX_ROBOT_REFERENCE_CALLS,
                include_fk=tool_calls["fk_query"] < _MAX_FK_CALLS,
                include_trajectory=(bool(self.trajectories)
                                    and tool_calls["get_practice_trajectory"] < _MAX_TRAJECTORY_CALLS),
            ))
            assistant = message.model_dump(exclude_none=True, mode="json")
            self._append_message(messages, assistant)
            calls = message.tool_calls or []
            if calls:
                for call in calls:
                    name = call.function.name
                    try:
                        limit = _TOOL_CALL_LIMITS.get(name)
                        if limit is None:
                            raise ValueError(f"unknown tool {name}")
                        tool_calls[name] += 1
                        if tool_calls[name] > limit:
                            raise ValueError(
                                f"{name} limit of {limit} calls reached; "
                                "finish the program using existing results"
                            )
                        result = self._tool_result(name, json.loads(call.function.arguments))
                    except (json.JSONDecodeError, ValueError) as error:
                        result = {"error": str(error)}
                    self._append_message(messages, {
                        "role": "tool", "tool_call_id": call.id,
                        "content": json.dumps(result, separators=(",", ":")),
                    })
                all_tools_exhausted = (
                    tool_calls["get_robot_reference"] >= _MAX_ROBOT_REFERENCE_CALLS
                    and tool_calls["fk_query"] >= _MAX_FK_CALLS
                    and (not self.trajectories
                         or tool_calls["get_practice_trajectory"] >= _MAX_TRAJECTORY_CALLS)
                )
                if all_tools_exhausted and not all_tools_notice_sent:
                    self._append_message(messages, {
                        "role": "user",
                        "content": (
                            "All available tool budgets are exhausted and the tools have been removed. "
                            "Use existing results and immediately return the complete play.py Python code block; "
                            "do not make further tool calls."
                        ),
                    })
                    all_tools_notice_sent = True
                continue
            if not message.content:
                self._append_message(messages, {
                    "role": "user", "content": "Return a complete play.py Python code block.",
                })
                continue
            try:
                source = self._python_block(message.content)
                self._validate_program(source)
                return message.content, source
            except (SyntaxError, ValueError) as error:
                self._append_message(messages, {
                    "role": "user",
                    "content": f"Invalid play.py: {error}. Return a corrected complete program.",
                })

    def _request_and_run(self, messages, context):
        for retry in range(2):
            response = ""
            try:
                response, source = self._request_program(messages)
                return response, source, self._run_program(source, context)
            except (
                json.JSONDecodeError,
                subprocess.SubprocessError,
                RuntimeError,
                ValueError,
            ) as error:
                if retry:
                    raise
                self._append_message(messages, {
                    "role": "user",
                    "content":
                        f"play.py failed: {error}. Return a corrected complete program.",
                })

    @staticmethod
    def _tools(*, include_reference=True, include_fk=True, include_trajectory=True):
        tools = []
        if include_reference:
            tools.append({
            "type": "function",
            "function": {
                "name": "get_robot_reference",
                "description": "Return exact local Shadow Hand reference data for the current environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": "Reference section to return.",
                            "enum": ["overview", "naming", "actuators", "joints", "forearm"],
                        },
                    },
                    "required": ["section"],
                    "additionalProperties": False,
                },
            },
            })
        if include_trajectory:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_practice_trajectory",
                    "description": "Return an inclusive step range from a completed practice attempt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "attempt": {
                                "type": "integer",
                                "description": "Completed attempt number.",
                            },
                            "start_step": {
                                "type": "integer",
                                "description": "First step to return, inclusive.",
                                "minimum": 0,
                            },
                            "end_step": {
                                "type": "integer",
                                "description": "Last step to return, inclusive.",
                                "minimum": 0,
                            },
                        },
                        "required": ["attempt", "start_step", "end_step"],
                        "additionalProperties": False,
                    },
                },
            })
        if include_fk:
            tools.append({
                "type": "function",
                "function": {
                    "name": "fk_query",
                    "description": "Compute forward kinematics for one to eight complete joint-qpos vectors.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "joint_positions": {
                                "type": "array",
                                "description": "One to eight complete joint-qpos vectors in fk_joint_names order.",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {
                                    "type": "array",
                                    "description": "One complete finite joint-qpos vector.",
                                    "items": {"type": "number"},
                                },
                            },
                        },
                        "required": ["joint_positions"],
                        "additionalProperties": False,
                    },
                },
            })
        return tools

    def _tool_result(self, name, arguments):
        if name == "get_robot_reference":
            return self._robot_reference(arguments.get("section"))
        if name == "get_practice_trajectory":
            attempt, start, end = (arguments.get(key) for key in ("attempt", "start_step", "end_step"))
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in (attempt, start, end)):
                raise ValueError("attempt, start_step and end_step must be integers")
            trajectory = self.trajectories.get(attempt)
            if trajectory is None:
                raise ValueError(f"attempt {attempt} is not complete")
            if start < 0 or end < start or end >= len(trajectory):
                raise ValueError(f"invalid step range [{start}, {end}]")
            return trajectory[start:end + 1]
        if name == "fk_query":
            queries = arguments.get("joint_positions")
            if not isinstance(queries, list) or not 1 <= len(queries) <= 8:
                raise ValueError("joint_positions must contain 1 to 8 vectors")
            low, high = self._fk_joint_low, self._fk_joint_high
            for query in queries:
                if not isinstance(query, list) or len(query) != self._fk_joint_size:
                    raise ValueError(f"each joint-qpos vector must contain {self._fk_joint_size} values")
                if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                       for value in query):
                    raise ValueError("joint-qpos values must be finite numbers")
                if np.any(np.asarray(query) < low) or np.any(np.asarray(query) > high):
                    raise ValueError("joint-qpos values must be within fk_joint_low and fk_joint_high")
            return self._forward_kinematics(queries)
        raise ValueError(f"unknown tool {name}")

    def _robot_reference(self, section):
        context = self._robot_context
        if section == "overview":
            return {
                "robot_model": "Shadow Hand",
                "environment_id": context.get("environment_id"),
                "hands": context["hands"],
                "hand_count": context.get("hand_count", len(context["hands"])),
                "static_partition": context.get("static_partition", False),
                "hand_partitions": context.get("hand_partitions", []),
                "fingertip_names": context["fingertip_names"],
                "fingertip_body_order": list(shadow_hand_constants.FINGERTIP_BODIES),
                "note": "Joint axes are hand-local; use fk_query for world-coordinate effects.",
            }
        if section == "naming":
            return {
                "action_names": context["action_names"],
                "fk_joint_names": context["fk_joint_names"],
                "note": (
                    "Use these exact identifiers. Forearm names do not contain rh_/lh_ after the slash; "
                    "XML hand actuator names do."
                ),
            }
        if section == "actuators":
            return [{
                "index": index, "name": name,
                "low": context["action_low"][index],
                "high": context["action_high"][index],
                **context["action_metadata"][index],
            } for index, name in enumerate(context["action_names"])]
        if section == "joints":
            return {
                "groups": {name: list(joints) for name, joints in shadow_hand_constants.JOINT_GROUP.items()},
                "joint_positions": [{
                    "index": index, "name": name,
                    "low": context["fk_joint_low"][index],
                    "high": context["fk_joint_high"][index],
                } for index, name in enumerate(context["fk_joint_names"])],
            }
        if section == "forearm":
            active = {}
            for index, metadata in enumerate(context["action_metadata"]):
                name = str(metadata["target"]).rsplit("/", 1)[-1]
                if name in shadow_hand_model._FOREARM_DOFS:
                    dof = shadow_hand_model._FOREARM_DOFS[name]
                    active[context["action_names"][index]] = {
                        "joint_type": dof.joint_type,
                        "hand_local_axis": list(dof.axis),
                        "low": context["action_low"][index],
                        "high": context["action_high"][index],
                        "left_hand_axis_reflected": dof.reflect,
                    }
            return {
                "active_forearm_actuators": active,
                "note": "Axes are hand-local; use fk_query to measure their world-coordinate effects.",
            }
        raise ValueError("section must be overview, naming, actuators, joints, or forearm")

    def _forward_kinematics(self, queries):
        composer = get_composer_env_from_gym(self.env)
        task, physics = composer.task, composer.physics
        live_state, live_ctrl = physics.get_state().copy(), physics.data.ctrl.copy()
        results = []
        try:
            for query in queries:
                self._restore_physics(physics, self._fk_state, self._fk_ctrl)
                qpos = np.asarray(query, dtype=np.float64)
                if qpos.shape != (self._fk_joint_size,):
                    raise ValueError(f"FK joint-qpos has shape {qpos.shape}")
                if not np.isfinite(qpos).all():
                    raise ValueError("FK joint-qpos values must be finite")
                offset = 0
                for hand in task.hands:
                    size = len(hand.joints)
                    physics.bind(hand.joints).qpos = qpos[offset : offset + size]
                    offset += size
                physics.forward()
                results.append({
                    "fingertip_positions": {
                        h.name: physics.bind(h.fingertip_sites).xpos.tolist()
                        for h in task.hands
                    },
                    "joint_angles": {
                        h.name: physics.bind(h.joints).qpos.tolist()
                        for h in task.hands
                    },
                })
        finally:
            self._restore_physics(physics, live_state, live_ctrl)
        return results

    @staticmethod
    def _restore_physics(physics, state, ctrl):
        with physics.reset_context():
            physics.set_state(state)
            physics.data.ctrl[:] = ctrl

    def _run_program(self, source, context):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        candidate = self.path.with_name("candidate_play.py")
        candidate.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-I", str(candidate.resolve())],
            cwd=self.path.parent, text=True, capture_output=True, timeout=30,
            input=json.dumps(context),
        )
        if result.returncode:
            raise ValueError(result.stderr.strip() or f"exit code {result.returncode}")
        actions = np.asarray(json.loads(result.stdout), dtype=np.float32)
        expected = (context["num_steps"], context["action_size"])
        if actions.shape != expected:
            raise ValueError(f"play.py returned {actions.shape}; expected {expected}")
        physical_low = np.asarray(context["action_low"], dtype=np.float32)
        physical_high = np.asarray(context["action_high"], dtype=np.float32)
        actions = np.clip(actions, physical_low, physical_high)
        actions = self._rescale_actions(
            actions, physical_low, physical_high,
            self.action_space.low, self.action_space.high,
        )
        return np.clip(actions, self.action_space.low, self.action_space.high)

    @staticmethod
    def _rescale_actions(actions, source_low, source_high, target_low, target_high):
        return target_low + (
            (actions - source_low) * (target_high - target_low)
            / (source_high - source_low)
        )

    @staticmethod
    def _named_nonzero_actions(names, actions):
        return {
            name: round(float(value), 6)
            for name, value in zip(names, actions)
            if abs(value) > 1e-6
        }

    def _practice(self, actions, attempt):
        self.env.reset(seed=self.practice_seed)
        composer = get_composer_env_from_gym(self.env)
        task, physics = composer.task, composer.physics
        task._disable_colorization = False
        spec = task.action_spec(physics)
        action_names = [
            actuator.full_identifier
            for hand in task.hands for actuator in hand.actuators
        ] + ["sustain"]
        keys = physics.bind(task.piano.key_geoms)
        key_geometry = [{
            "key": i, "position": position.tolist(), "size": size.tolist(),
        } for i, (position, size) in enumerate(zip(keys.xpos.copy(), keys.size.copy()))]
        trajectory = []
        done, info = False, {}
        while not done:
            step = int(task._t_idx)
            notes = task._notes[step]
            target = sorted(int(note.key) for note in notes)
            action = actions[min(step, len(actions) - 1)]
            physical_action = self._rescale_actions(
                action, self.action_space.low, self.action_space.high,
                spec.minimum, spec.maximum,
            )
            _, _, terminated, truncated, info = self.env.step(action)
            played = sorted(np.flatnonzero(task.piano.activation).astype(int).tolist())
            error = target != played
            trajectory.append({
                "step": step,
                "time": float(physics.data.time),
                "required_keys": target,
                "played_keys": played,
                "error": error,
                "missed_keys": sorted(set(target) - set(played)),
                "wrong_keys": sorted(set(played) - set(target)),
                "physical_action_targets": self._named_nonzero_actions(
                    action_names, physical_action
                ),
                "fingertips": self._fingertip_positions(physics, task.hands),
            })
            done = bool(terminated or truncated)
        metrics = {
            "attempt": attempt,
            "f1": float(info[EpisodeInfoKeys.EPISODE_TASK_F1]),
            "precision": float(info[EpisodeInfoKeys.EPISODE_TASK_KEY_PRECISION]),
            "recall": float(info[EpisodeInfoKeys.EPISODE_TASK_KEY_RECALL]),
        }
        return metrics, trajectory, key_geometry

    def _feedback(self, attempt, metrics, best_attempt, best_f1, trajectory, error_ranges):
        return (
            "Revise the preceding play.py. Practice completed: " + json.dumps({
                "attempt": attempt, "metrics": metrics, "best_attempt": best_attempt,
                "best_f1": best_f1, "trajectory": trajectory, "error_ranges": error_ranges,
            }, separators=(",", ":")) + ". error:true marks each bad step; inspect the "
            "trajectory tool around those ranges before revising. The best program is already in "
            "this conversation; do not request or repeat it. Make a substantive, diagnostics-driven "
            "change to the controller logic or parameters."
        )

    @staticmethod
    def _fingertip_positions(physics, hands):
        return {
            hand.name: {
                site.name: position.tolist()
                for site, position in zip(
                    hand.fingertip_sites, physics.bind(hand.fingertip_sites).xpos
                )
            }
            for hand in hands
        }

    @staticmethod
    def _error_ranges(trajectory):
        ranges = []
        for sample in trajectory:
            if sample["error"]:
                if ranges and sample["step"] == ranges[-1][1] + 1:
                    ranges[-1][1] = sample["step"]
                else:
                    ranges.append([sample["step"], sample["step"]])
        return ranges

    def _save_trajectory(self, attempt, metrics, key_geometry, trajectory):
        self.trajectories[attempt] = trajectory
        path = self.path.with_name(f"attempt_{attempt:03d}_trajectory.json")
        path.write_text(json.dumps({"attempt": attempt, "metrics": metrics,
            "key_geometry": key_geometry, "steps": trajectory}, indent=2), encoding="utf-8")
        return path

    def _save(self, context, history, metrics, source, actions, best_attempt):
        self.program_path.write_text(source, encoding="utf-8")
        self.path.write_text(json.dumps({
            "control_timestep": context["control_timestep"],
            "best_attempt": best_attempt, "practice_attempts": history,
            "practice_metrics": metrics, "program": self.program_path.name,
            "actions": actions.tolist(),
        }, indent=2), encoding="utf-8")
        self.path.with_name("optimization_summary.json").write_text(json.dumps({
            "model": self.client.model,
            "practice_seed": self.practice_seed,
            "best_attempt": best_attempt,
            "best_metrics": metrics,
            "practice_attempts": history,
            "token_usage": self.client.token_usage,
        }, indent=2, sort_keys=True), encoding="utf-8")

    def _append_message(self, messages, message):
        messages.append(message)
        self._save_conversation(messages)

    def _save_conversation(self, messages):
        path = self.path.with_name(self.path.stem + "_conversation.json")
        temporary = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({
            "model": self.client.model,
            "thinking": self.client.thinking,
            "token_usage": self.client.token_usage,
            "messages": messages,
        }, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _python_block(response):
        if "```python" not in response:
            raise ValueError("LLM response does not contain a Python code block")
        return response.split("```python", 1)[1].split("```", 1)[0].strip() + "\n"

    @staticmethod
    def _validate_program(source):
        allowed = {"json", "sys", "math", "numpy"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import) and any(
                a.name.split(".")[0] not in allowed for a in node.names
            ):
                raise ValueError("play.py may import only json, sys, math and numpy")
            if isinstance(node, ast.ImportFrom) and (
                node.module is None or node.module.split(".")[0] not in allowed
            ):
                raise ValueError("play.py may import only json, sys, math and numpy")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (
                node.func.id in {"open", "exec", "eval", "compile", "__import__"}
            ):
                raise ValueError(f"play.py may not call {node.func.id}")
