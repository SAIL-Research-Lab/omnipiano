"""Shared protocol, provenance and five native OmniSafe baseline configurations."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import uuid

from omnipiano.safety.suite import ALGORITHMS, SONGS


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


def algorithm_config(algorithm, budget, steps=5_000_000, seed=1, device="cpu", *, smoke=False):
    if algorithm not in ALGORITHMS:
        raise ValueError(algorithm)
    epoch = 2_000 if smoke else 20_000
    if type(steps) is not int or steps < epoch or steps % epoch:
        raise ValueError(f"Training steps must be a positive multiple of {epoch}")
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
           "logger_cfgs": {"use_wandb": False, "use_tensorboard": True, "save_model_freq": 1}}
    if algorithm in ("PPOLag", "CUP"):
        cfg["lagrange_cfgs"] = {"cost_limit": budget}
    elif algorithm in ("OnCRPO", "CPO"):
        cfg["algo_cfgs"].update(cost_limit=budget, use_cost=True)
    return cfg
