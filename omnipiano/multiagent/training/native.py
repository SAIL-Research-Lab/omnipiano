"""Native backend adapter for the existing training lifecycle.

Checkpoint resume restores learning state but resets environment episodes.
Load only checkpoints produced by trusted code: torch.load(weights_only=False).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from omnipiano.multiagent.algos._native import make_model
from omnipiano.multiagent.training.native_io import (
    Codec, EnvPool, Replay, add_gae,
)
from omnipiano.multiagent.training.runtime import write_json, evaluate_marl


FORMAT = "omnipiano-native-v1"
MODEL_FIELDS = (
    "algo", "seed", "gamma", "train_batch_size", "minibatch_size", "num_epochs",
    "lr", "critic_lr", "adam_epsilon", "gae_lambda", "clip_param",
    "vf_clip_param", "vf_loss_coeff", "entropy_coeff", "grad_clip",
    "grad_clip_by", "hidden_sizes_parsed", "activation",
    "hidden_orthogonal_gain", "policy_output_gain", "value_output_gain",
    "initial_log_std", "log_std_min", "log_std_max", "input_layer_norm",
    "value_norm", "value_norm_beta", "value_norm_epsilon",
    "value_norm_variance_floor", "num_workers",
)


def native_options(args):
    options = {}
    if args.algo == "mat":
        options = {
            "embed_dim": 128, "heads": 4, "blocks": 2,
            "std_parameter_init": 1.0,
        }
    elif args.algo in ("facmac", "masac"):
        options = {
            "replay_capacity": 50_000,
            "batch_size": 256,
            "learning_starts": 10_000,
            "collect_steps": 24,
            "updates_per_env_step": 0.25,
            "tau": 0.005,
            "max_replay_gib": 4.0,
        }
        if args.algo == "facmac":
            options.update(
                noise_std=0.1, action_l2=0.001,
                mixer_embed=64, monotonic=True)
        else:
            options.update(
                alpha_init=0.1, alpha_lr=0.0003,
                target_entropy_scale=1.0,
                variant="cooperative_joint_entropy_v1")
    supplied = getattr(args, "_native_options", {})
    if not isinstance(supplied, dict) or set(supplied) - set(options):
        raise ValueError(f"Unknown native options for {args.algo}: {supplied}")
    options.update(supplied)

    if args.grad_clip_by != "global_norm":
        raise ValueError("Native backend currently supports global_norm clipping only")
    if args.use_kl_loss:
        raise ValueError("Native backend does not implement PPO KL penalties")
    if args.num_cpus_per_env_runner != 1:
        raise ValueError("Native workers currently use one CPU-thread setting")
    if args.ray_num_cpus is not None:
        raise ValueError("For native jobs set compute.ray_num_cpus=null")
    if args.algo == "mat" and args.critic_lr != args.lr:
        raise ValueError("MAT has a joint optimizer; critic_lr must equal lr")
    if args.algo == "masac" and options["variant"] != "cooperative_joint_entropy_v1":
        raise ValueError("Unsupported MASAC definition")

    integers = (
        "embed_dim", "heads", "blocks", "replay_capacity", "batch_size",
        "learning_starts", "collect_steps", "mixer_embed",
    )
    for name in integers:
        if name in options:
            value = options[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
    for name, value in options.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite native option: {name}")
    for name in ("updates_per_env_step", "max_replay_gib", "alpha_init",
                 "alpha_lr", "target_entropy_scale"):
        if name in options and options[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("noise_std", "action_l2"):
        if name in options and options[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if "tau" in options and not 0 < options["tau"] <= 1:
        raise ValueError("tau must be in (0,1]")
    if "monotonic" in options and type(options["monotonic"]) is not bool:
        raise ValueError("monotonic must be boolean")
    if args.algo == "mat" and options["embed_dim"] % options["heads"]:
        raise ValueError("MAT embed_dim must be divisible by heads")
    if "replay_capacity" in options:
        if args.smoke_test:
            # Explicitly make the smoke run exercise off-policy updates.
            options["replay_capacity"] = min(options["replay_capacity"], 2048)
            options["batch_size"] = min(options["batch_size"], 128)
            options["learning_starts"] = min(options["learning_starts"], 128)
        if options["batch_size"] > options["replay_capacity"]:
            raise ValueError("Replay batch exceeds capacity")
        if options["collect_steps"] > options["replay_capacity"]:
            raise ValueError("Collection size exceeds capacity")
    return options


def identity(args, meta, options):
    data = {
        "format": FORMAT, "task": args._resolved_task,
        "parameters": {key: getattr(args, key) for key in MODEL_FIELDS},
        "collision_penalty": args.inter_agent_collision_penalty_coef,
        "meta": meta, "native_options": options,
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "state.pt"
    marker = json.loads((path.parent / "native_checkpoint.json").read_text())
    if marker.get("format") != FORMAT:
        raise ValueError("Not a supported native checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != FORMAT:
        raise ValueError("Native checkpoint payload format mismatch")
    return state


class NativePolicy:
    def tensor(self, batch):
        return {
            key: torch.as_tensor(value, device=self.device)
            for key, value in batch.items()
        }

    def compute_joint_actions(self, observations, action_spaces):
        if set(action_spaces) != set(self.codec.agents):
            raise ValueError("Evaluation agent set differs from training")
        for agent, dim in zip(self.codec.agents, self.codec.action_dims):
            space = action_spaces[agent]
            if space.shape != (dim,) or not np.all(space.low == -1) \
                    or not np.all(space.high == 1):
                raise ValueError("Evaluation action space changed")
        with torch.inference_mode():
            batch = self.tensor(self.codec.pack([observations]))
            actions = self.model.act(batch, deterministic=True)["a"].cpu().numpy()
        return self.codec.split(actions)[0]


class NativeAlgorithm(NativePolicy):
    def __init__(self, args, meta):
        self.args, self.meta = args, meta
        self.options = native_options(args)
        self.codec = Codec(meta)
        self.device = torch.device(
            "cuda:0" if args.num_gpus_per_learner > 0 else "cpu")
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        self.model = make_model(args, meta, self.options, self.device)
        self.env_steps, self.iterations, self.gradient_updates = 0, 0, 0
        self.update_credit, self.resume_count = 0.0, 0
        self.pool, self.replay = None, None
        if not self.model.on_policy:
            self.replay = Replay(
                self.codec, self.options["replay_capacity"],
                self.options["max_replay_gib"])

        checkpoint = None
        if args.resume_native:
            checkpoint = load_checkpoint(args.resume_native)
            if checkpoint["identity"] != identity(args, meta, self.options):
                raise ValueError("Resume task, hyperparameters, spaces or workers changed")
            self.model.load_state_dict(checkpoint["model"])
            self.model.restore_optimizers(checkpoint["optimizers"])
            for key in ("env_steps", "iterations", "gradient_updates",
                        "update_credit", "resume_count"):
                setattr(self, key, checkpoint[key])
            self.resume_count += 1
            if self.env_steps >= args.total_steps:
                raise ValueError("Checkpoint already reached the requested budget")
            if self.replay is not None:
                self.replay.restore(checkpoint["replay"])
        try:
            self.pool = EnvPool(
                args._resolved_task,
                args.seed + self.resume_count * 1_000_000,
                args.inter_agent_collision_penalty_coef,
                args.num_workers, args.sample_timeout_s)
            if self.pool.meta != meta:
                raise ValueError("Probe and worker environment layouts disagree")
            if checkpoint is not None:
                self.pool.episode = [
                    value + 1 for value in checkpoint["collector_episodes"]]
                restore_rng(checkpoint["rng"])
        except BaseException:
            self.stop()
            raise

    def effective_config(self):
        return {
            "backend": "native",
            "backend_format": FORMAT,
            "ray_cluster_started": False,
            "algo": self.args.algo,
            "parameters": {
                key: getattr(self.args, key) for key in MODEL_FIELDS
            },
            "native_options": self.options,
            "agent_order": self.meta["agents"],
            "layout": self.meta,
            "include_global_state": True,
            "count_steps_by": "env_steps",
            "reward_mode": "shared",
            "inter_agent_collision_penalty_coef":
                self.args.inter_agent_collision_penalty_coef,
            "eval_freq_env_steps": self.args.eval_freq,
            "checkpoint_freq_env_steps": self.args.checkpoint_freq,
            "native_resume_source": self.args.resume_native,
            "resume_semantics": "learner/replay/RNG restored; environment episodes reset",
            "on_policy": self.model.on_policy,
            "off_policy_ignores": (
                [] if self.model.on_policy else [
                    "PPO clipping", "GAE", "PPO epochs", "ValueNorm",
                    "PPO entropy_coeff",
                ]
            ),
            "mat_specifics": (
                {
                    "observation_adapters": "independent per-agent LayerNorm/Linear",
                    "agent_order": "fixed compiled order",
                    "ratio": "per-valid-action-dimension",
                    "entropy_reduction": "sum valid dimensions, mean agents/rows",
                    "std": "0.5*sigmoid(parameter), floor 1e-6",
                    "joint_actor_value_optimizer": True,
                } if self.args.algo == "mat" else None
            ),
        }

    def collect(self, count):
        chunks, collected = [], 0
        while collected < count:
            k = min(self.pool.n, count - collected)
            current_step = self.env_steps + collected
            warmup = (
                not self.model.on_policy
                and current_step < self.options["learning_starts"])
            if warmup:
                k = min(k, self.options["learning_starts"] - current_step)
            before = self.codec.pack(self.pool.observations[:k])
            keys = self.pool.keys(k)
            with torch.no_grad():
                output = self.model.act(self.tensor(before))
                output = {key: value.cpu().numpy() for key, value in output.items()}
            if warmup:
                output["a"] = np.random.uniform(
                    -1, 1, (k, self.codec.action_dim)).astype(np.float32)
            elif self.args.algo == "facmac":
                noise = np.random.normal(
                    0, self.options["noise_std"], output["a"].shape)
                output["a"] = np.clip(output["a"] + noise, -1, 1).astype(np.float32)

            results = self.pool.step(self.codec.split(output["a"]))
            following = self.codec.pack([row["obs"] for row in results])
            chunk = {
                **before, "a": output["a"], "no": following["o"],
                "ns": following["s"], "keys": keys,
                "r": np.asarray([row["r"] for row in results], np.float32),
                "term": np.asarray([row["term"] for row in results], np.float32),
                "trunc": np.asarray([row["trunc"] for row in results], np.float32),
            }
            if self.model.on_policy:
                with torch.no_grad():
                    _, nv = self.model.values(self.tensor(following))
                chunk.update(
                    lp=output["lp"], vn=output["vn"], v=output["v"],
                    nv=nv.cpu().numpy())
            chunks.append(chunk)
            collected += k

        batch = {
            key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
            for key in chunks[0]
        }
        if len({tuple(row) for row in batch["keys"]}) != len(batch["keys"]):
            raise RuntimeError("Duplicate joint rollout row keys")
        if self.model.on_policy:
            add_gae(batch, self.args.gamma, self.args.gae_lambda, self.pool.n)
        return batch

    def train(self):
        limit = (
            self.args.train_batch_size if self.model.on_policy
            else self.options["collect_steps"])
        count = min(limit, self.args.total_steps - self.env_steps)
        if count <= 0:
            raise RuntimeError("Native train called after budget completion")
        previous = self.env_steps
        batch = self.collect(count)
        self.env_steps += count
        stats = {}
        if self.model.on_policy:
            stats = self.model.learn(self.tensor(batch))
            self.gradient_updates += 1  # rollout-update calls, not optimizer steps
        else:
            self.replay.add(batch)
            eligible = (
                max(0, self.env_steps - self.options["learning_starts"])
                - max(0, previous - self.options["learning_starts"])
            )
            if self.replay.size >= self.options["batch_size"]:
                self.update_credit += eligible * self.options["updates_per_env_step"]
                updates = int(self.update_credit)
                self.update_credit -= updates
                for _ in range(updates):
                    stats = self.model.learn(self.tensor(
                        self.replay.sample(self.options["batch_size"])))
                    self.gradient_updates += 1
            stats["replay_size"] = self.replay.size
            stats["gradient_updates"] = self.gradient_updates
        self.iterations += 1
        return {
            "num_env_steps_sampled_lifetime": self.env_steps,
            "learners": {"__all_modules__": stats},
            "env_runners": {},
        }

    def save_to_path(self, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        state = {
            "format": FORMAT,
            "identity": identity(self.args, self.meta, self.options),
            "meta": self.meta,
            "model": self.model.state_dict(),
            "optimizers": self.model.optimizer_state(),
            "rng": rng_state(),
            "replay": self.replay.state() if self.replay is not None else None,
            "collector_episodes": self.pool.episode,
            **{key: getattr(self, key) for key in (
                "env_steps", "iterations", "gradient_updates",
                "update_credit", "resume_count",
            )},
        }
        temporary = directory / "state.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(directory / "state.pt")
        write_json(directory / "native_checkpoint.json", {
            "format": FORMAT, "algo": self.args.algo,
            "env_steps": self.env_steps,
            "contains_replay": self.replay is not None,
            "exact_environment_resume": False,
        })
        return str(directory)

    def stop(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None


class NativeConfig:
    def __init__(self, args, meta):
        self.args, self.meta = args, meta

    def build_algo(self):
        return NativeAlgorithm(self.args, self.meta)


def build_native_config(args, spec, agents, obs_spaces, act_spaces, layouts):
    native_options(args)  # Validate before creating workers.
    widths = [
        layouts[a]["global_state"][1] - layouts[a]["global_state"][0]
        for a in agents
    ]
    if len(set(widths)) != 1 or widths[0] <= len(agents):
        raise ValueError("Native backend requires an AS global-state observation")
    for agent in agents:
        if not np.all(act_spaces[agent].low == -1) \
                or not np.all(act_spaces[agent].high == 1):
            raise ValueError("Noncanonical action bounds")
    meta = {
        "agents": list(agents),
        "obs_dims": [int(obs_spaces[a].shape[0]) for a in agents],
        "action_dims": [int(act_spaces[a].shape[0]) for a in agents],
        "own_slices": [list(layouts[a]["own"]) for a in agents],
        "state_slices": [list(layouts[a]["global_state"]) for a in agents],
        "state_dim": widths[0] - len(agents),
    }
    return NativeConfig(args, meta), spec.resolve_learner_class()


def evaluate_saved(args):
    state = load_checkpoint(args.eval_native)
    options = native_options(args)
    if state["identity"] != identity(args, state["meta"], options):
        raise ValueError("Evaluation config does not match checkpoint identity")
    policy = NativePolicy()
    policy.device = torch.device(
        "cuda:0" if args.num_gpus_per_learner > 0 else "cpu")
    policy.codec = Codec(state["meta"])
    policy.model = make_model(args, state["meta"], options, policy.device)
    policy.model.load_state_dict(state["model"])
    result = evaluate_marl(
        policy, args.env_id, task=args._resolved_task,
        eval_seed=args.seed + args.eval_seed_offset,
        num_episodes=args.num_eval_eps, include_global_state=True,
        inter_agent_collision_penalty_coef=args.inter_agent_collision_penalty_coef,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0