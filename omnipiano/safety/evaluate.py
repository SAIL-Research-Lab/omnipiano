"""Deterministic post-training checkpoint replay, distinct from training rollouts."""
import csv
import json
from pathlib import Path
import re

import numpy as np

from omnipiano.safety.algorithms import evaluation_budget


def replay(save_dir, cell, output, episodes=10):
    import torch
    import omnisafe
    from omnipiano.safety import cmdp  # register before Evaluator loads the env
    del cmdp
    save_dir, output = Path(save_dir), Path(output)
    config = json.loads((save_dir / "config.json").read_text())
    if config["env_id"] != cell["env_id"] or config["seed"] != cell["seed"]:
        raise ValueError("Checkpoint task/seed does not match manifest")
    interval = cell["steps_per_epoch"]
    if config["algo_cfgs"]["steps_per_epoch"] != interval:
        raise ValueError("Checkpoint step interval mismatch")
    checkpoints = []
    for path in (save_dir / "torch_save").glob("epoch-*.pt"):
        match = re.fullmatch(r"epoch-(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match[1]), path.name))
    checkpoints.sort()
    frequency = cell.get("save_model_freq", 1)
    if config["logger_cfgs"]["save_model_freq"] != frequency:
        raise ValueError("Checkpoint save frequency mismatch")
    expected = list(range(0, cell["steps"] // interval + 1, frequency))
    if [n for n, _ in checkpoints] != expected:
        raise ValueError("Missing/unexpected checkpoints; refusing a partial learning curve")
    rows, returns, costs, f1s, lengths = [], [], [], [], []
    safety_budget = evaluation_budget(cell["algorithm"], config["algo_cfgs"])
    for epoch, filename in checkpoints:
        evaluator = omnisafe.Evaluator()
        try:
            evaluator.load_saved(save_dir=str(save_dir), model_name=filename)
            env, actor = evaluator._env, evaluator._actor
            batch = []
            for ep in range(episodes):
                seed = cell["seed"] + 10_000 + ep
                obs, _ = env.reset(seed=seed)
                safety_state = 1.0
                ret = cost = 0.0
                for length in range(1, 100_001):
                    with torch.no_grad():
                        actor_obs = obs if safety_budget is None else torch.cat(
                            (obs, obs.new_full((*obs.shape[:-1], 1), safety_state)), dim=-1)
                        action = actor.predict(actor_obs, deterministic=True)
                    obs, rew, c, done, trunc, info = env.step(action)
                    ret += float(rew.item())
                    cost += float(c.item())
                    if safety_budget is not None:
                        safety_state = (safety_state - float(c.item()) / safety_budget) / config["algo_cfgs"]["saute_gamma"]
                    if bool(done.item() or trunc.item()):
                        terminal = info.get("final_info", info)
                        if isinstance(terminal, (list, tuple, np.ndarray)):
                            terminal = terminal[0]
                        f1 = float(terminal["episode_task/f1"])
                        if not np.isfinite([ret, cost, f1]).all() or not 0 <= f1 <= 1:
                            raise ValueError("Invalid evaluation metrics")
                        row = dict(env_steps=epoch * interval, eval_seed=seed, episode=ep,
                                   reward=ret, cost=cost, f1=f1, length=length)
                        rows.append(row)
                        batch.append(row)
                        break
                else:
                    raise RuntimeError("Episode did not terminate")
            returns.append([r["reward"] for r in batch])
            costs.append([r["cost"] for r in batch])
            f1s.append([r["f1"] for r in batch])
            lengths.append([r["length"] for r in batch])
        finally:
            if getattr(evaluator, "_env", None) is not None:
                evaluator._env.close()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "eval_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(output / "evaluations.npz", timesteps=np.asarray(expected) * interval,
             results=returns, ep_costs=costs, ep_f1=f1s, ep_lengths=lengths)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    cell = json.loads((args.run_dir / "cell.json").read_text())
    from omnipiano.safety.runtime import code_hash, versions, midi_hash
    from omnipiano.safety.suite import task_from_dict
    if cell["code_hash"] != code_hash() or cell["runtime"] != versions() or cell["midi_hash"] != midi_hash(task_from_dict(cell["task"])):
        raise ValueError("Code/dependencies/MIDI differ from the training manifest")
    state = json.loads((args.run_dir / "status.json").read_text())
    replay(state["save_dir"], cell, args.run_dir, episodes=cell["eval_episodes"])
