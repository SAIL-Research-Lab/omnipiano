"""Shared protocol, provenance and native OmniSafe configurations."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import uuid

from omnipiano.safety.suite import SONGS
from omnipiano.safety.algorithms import (
    SUPPORTED_ALGORITHMS, OFF_POLICY, LAGRANGIAN, DIRECT_BUDGET, AUGMENTED,
)


def code_hash():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "omnipiano").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def versions():
    result = {"python": sys.version.split()[0]}
    for name in ("omnisafe", "torch", "numpy", "mujoco", "dm-control", "gymnasium"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def midi_hash(item):
    from robopianist import music
    name = SONGS[item.song][0].removeprefix("RoboPianist-repertoire-150-").removesuffix("-v0")
    path = music._PIG_NAME_TO_FILE.get(name)
    if path is None:
        raise FileNotFoundError(f"Install/preprocess PIG MIDI for {name}")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def algorithm_config(algorithm, budget, steps=5_000_000, seed=1, device="cpu", *,
                     smoke=False, max_ep_len=None, buffer_size=None):
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(algorithm)
    checkpoint_interval = 2_000 if smoke else 20_000
    epoch = 2_000 if smoke or algorithm in OFF_POLICY else 20_000
    if type(steps) is not int or steps < checkpoint_interval or steps % checkpoint_interval:
        raise ValueError(f"Training steps must be a positive multiple of {checkpoint_interval}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    import math
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("Invalid cost budget")
    cfg = {"seed": seed,
           "train_cfgs": {"total_steps": steps, "vector_env_nums": 1, "parallel": 1,
                          "torch_threads": 1, "device": device},
           "algo_cfgs": {"steps_per_epoch": epoch, "gamma": 0.8, "cost_gamma": 0.8,
                         "obs_normalize": True, "reward_normalize": False, "cost_normalize": False},
           "logger_cfgs": {"use_wandb": False, "use_tensorboard": True,
                           "save_model_freq": checkpoint_interval // epoch}}
    if algorithm in LAGRANGIAN:
        cfg["lagrange_cfgs"] = {"cost_limit": budget}
    elif algorithm in DIRECT_BUDGET:
        cfg["algo_cfgs"]["cost_limit"] = budget
        if algorithm in ("OnCRPO", "CPO"):
            cfg["algo_cfgs"]["use_cost"] = True
    if algorithm in AUGMENTED:
        if budget <= 0 or type(max_ep_len) is not int or max_ep_len <= 0:
            raise ValueError("Saute/Simmer require a positive budget and max_ep_len")
        cfg["algo_cfgs"].update(safety_budget=budget, max_ep_len=max_ep_len)
        if "Simmer" in algorithm:
            cfg["algo_cfgs"]["upper_budget"] = budget
    if algorithm in OFF_POLICY:
        cfg["algo_cfgs"].pop("cost_gamma")  # off-policy critics use gamma for both
        size = buffer_size if buffer_size is not None else (2_000 if smoke else 100_000)
        if type(size) is not int or size < 256:
            raise ValueError("buffer_size must be an integer >= 256")
        cfg["algo_cfgs"]["size"] = size
        cfg["train_cfgs"]["eval_episodes"] = 1
        if smoke:
            cfg["algo_cfgs"]["start_learning_steps"] = 256
            if algorithm not in ("DDPG", "TD3", "SAC"):
                cfg["algo_cfgs"]["warmup_epochs"] = 0
    elif buffer_size is not None:
        raise ValueError("buffer_size is only meaningful for off-policy algorithms")
    return cfg
