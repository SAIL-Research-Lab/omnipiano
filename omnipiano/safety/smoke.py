"""Short real-physics checks for every task; never launches training."""
import argparse
import json
from pathlib import Path

import numpy as np

import omnipiano
from omnipiano.safety.suite import register_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("safety_smoke.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    rows = []
    for env_id, item in register_all().items():
        env = omnipiano.make(env_id, seed=0)
        total_steps = 0
        try:
            for mode in ("zero", "random"):
                obs, _ = env.reset(seed=0)
                assert np.isfinite(obs).all()
                env.action_space.seed(0)
                for _ in range(args.steps):
                    action = np.zeros(env.action_space.shape, dtype=np.float32) if mode == "zero" else env.action_space.sample()
                    obs, reward, done, trunc, info = env.step(action)
                    cost = info["step_safety/cost_total"]
                    assert np.isfinite(obs).all() and np.isfinite([cost, reward]).all() and cost >= 0
                    total_steps += 1
                    if done or trunc:
                        env.reset()
            rows.append(dict(env_id=env_id, hands=item.hands, semantic=item.cost.semantic,
                             setting=item.cost.setting, steps=total_steps, status="passed"))
            print(json.dumps(rows[-1]), flush=True)
        finally:
            env.close()
    from omnipiano.safety.runtime import atomic_json, code_hash, versions
    atomic_json(args.out, {"code_hash": code_hash(), "runtime": versions(), "tasks": rows})


if __name__ == "__main__":
    main()
