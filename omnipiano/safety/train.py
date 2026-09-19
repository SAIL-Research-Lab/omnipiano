"""One manifested run; used by safety.run, with replay-only retry after training."""
import argparse
import json
import os
from pathlib import Path

from omnipiano.safety.runtime import algorithm_config, atomic_json, code_hash, midi_hash, versions
from omnipiano.safety.suite import task_from_dict


def run(directory):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import omnisafe
    from omnipiano.safety import cmdp
    from omnipiano.safety.evaluate import replay
    del cmdp
    torch.set_num_threads(1)
    directory = Path(directory).resolve()
    cell = json.loads((directory / "cell.json").read_text())
    if versions()["omnisafe"] != "0.5.0":
        raise ValueError("This protocol requires OmniSafe 0.5.0")
    if (cell["code_hash"] != code_hash() or cell["runtime"] != versions()
            or cell["midi_hash"] != midi_hash(task_from_dict(cell["task"]))):
        raise ValueError("Run provenance changed; create a new manifest")
    status_path = directory / "status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"status": "new"}
    if state["status"] == "complete":
        if not (directory / "evaluations.npz").is_file():
            raise ValueError("Complete status but missing evaluation artifact")
        return
    if state["status"] not in ("new", "trained"):
        raise RuntimeError("Interrupted/failed training: inspect this run and use a fresh output directory; no optimizer resume")
    lock = directory / "run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if state["status"] == "new":
            if (directory / "logs").exists():
                raise RuntimeError("Unexpected existing logs; refusing overwrite")
            cfg = algorithm_config(cell["algorithm"], cell["task"]["budget"], cell["steps"],
                                   cell["seed"], cell["device"], smoke=cell["smoke"])
            cfg["logger_cfgs"]["log_dir"] = str(directory / "logs")
            atomic_json(directory / "overrides.json", cfg)
            atomic_json(status_path, {"status": "training"})
            try:
                agent = omnisafe.Agent(cell["algorithm"], cell["env_id"], custom_cfgs=cfg)
                agent.learn()
                configs = list((directory / "logs").rglob("config.json"))
                if len(configs) != 1:
                    raise RuntimeError("Expected exactly one OmniSafe run directory")
                state = {"status": "trained", "save_dir": str(configs[0].parent)}
                atomic_json(status_path, state)
            except Exception as error:
                atomic_json(status_path, {"status": "failed", "stage": "training", "error": str(error)})
                raise
        try:
            replay(state["save_dir"], cell, directory, episodes=cell["eval_episodes"])
        except Exception as error:
            atomic_json(status_path, {**state, "status": "trained", "replay_error": str(error)})
            raise
        atomic_json(status_path, {**state, "status": "complete"})
    finally:
        lock.unlink()  # this process created this exact lock with exclusive open


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run(parser.parse_args().run_dir)
