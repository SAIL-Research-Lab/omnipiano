"""Load a trained MAPPO checkpoint and render ONE eval episode as video.

Usage:
    MUJOCO_GL=egl python -m omnipiano.multiagent._eval_mappo \
        --checkpoint-dir /tmp/ma_4hand_winter_wind_ckpt \
        --record-dir /tmp/ma_4hand_winter_wind_video \
        --env-id OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0

Writes a single mp4 (+mp3 sound if ffmpeg available) to --record-dir for
the trained policy playing the registered env. Reports final F1 / reward
summary.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Eval a trained MAPPO checkpoint + record video.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--env-id", default="OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    record_dir = Path(args.record_dir).expanduser().resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] checkpoint = {ckpt_dir}")
    print(f"[info] record_dir = {record_dir}")
    print(f"[info] env_id     = {args.env_id}")

    # If a checkpoint_path.txt sidecar exists, use that exact path.
    sidecar = ckpt_dir / "checkpoint_path.txt"
    if sidecar.exists():
        actual_ckpt = sidecar.read_text().strip()
        print(f"[info] resolved checkpoint via sidecar → {actual_ckpt}")
    else:
        actual_ckpt = str(ckpt_dir)

    try:
        import numpy as np
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from ray.tune.registry import register_env

        from omnipiano.multiagent import make_parallel
    except ImportError as e:
        print(f"[error] missing dependency: {e}", file=sys.stderr)
        return 1

    # Register the env (needed for Algorithm.from_checkpoint to know its env_creator).
    # Use the same creator as training, with flatten_obs=True.
    def _eval_env_creator(env_config):
        return ParallelPettingZooEnv(
            make_parallel(
                env_config.get("env_id", args.env_id),
                seed=env_config.get("seed", args.seed),
                flatten_obs=True,
                record_dir=env_config.get("record_dir"),
                metrics_capture_physics=True,
            )
        )

    register_env("omnipiano_4hand_ma", _eval_env_creator)

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    try:
        print("[info] Loading checkpoint ...")
        algo = Algorithm.from_checkpoint(actual_ckpt)
        print("[info] Checkpoint loaded.")

        # Build eval env directly (we want a clean instance with record_dir set;
        # the algorithm's internal eval_env is configured for training, no recording).
        eval_env = make_parallel(
            args.env_id,
            seed=args.seed,
            flatten_obs=True,
            record_dir=str(record_dir),
            record_every=1,
            metrics_capture_physics=True,
        )

        # Resolve per-agent policy module via Algorithm's modules.
        # In modern RLlib (2.5+) policies are accessed via algo.get_module(policy_id).
        agent_ids = list(eval_env.possible_agents)
        print(f"[info] agents = {agent_ids}")

        def _act(agent_id: str, obs):
            # Use the per-agent policy (registered under the agent's name).
            try:
                module = algo.get_module(agent_id)
            except Exception:
                # Fallback for older RLlib API
                module = algo.get_policy(agent_id)
            # New API: compute_single_action via Policy; module.forward_inference for RLModule
            try:
                import torch
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                out = module.forward_inference({"obs": obs_t})
                actions = out["action_dist_inputs"] if "action_dist_inputs" in out else out.get("actions")
                # PPO outputs logits / mean; for continuous, sample mean directly.
                if actions is None:
                    return np.zeros(eval_env.action_space(agent_id).shape, dtype=np.float32)
                act_np = actions[0].detach().cpu().numpy()
                # For continuous PPO, action_dist_inputs is [mean, log_std] concatenated.
                # Take just the mean half.
                act_dim = eval_env.action_space(agent_id).shape[0]
                if act_np.shape[0] == 2 * act_dim:
                    act_np = act_np[:act_dim]
                return np.clip(act_np, -1.0, 1.0).astype(np.float32)
            except Exception as e:
                print(f"[warn] policy inference failed for {agent_id}: {e}; using zero action.")
                return np.zeros(eval_env.action_space(agent_id).shape, dtype=np.float32)

        obs, infos = eval_env.reset(seed=args.seed)
        total_reward_per_agent = {a: 0.0 for a in agent_ids}
        n_steps = 0
        while eval_env.agents:
            actions = {a: _act(a, obs[a]) for a in eval_env.agents}
            obs, rewards, terms, truncs, infos = eval_env.step(actions)
            for a, r in rewards.items():
                total_reward_per_agent[a] += float(r)
            n_steps += 1

        eval_env.close()
        print()
        print(f"[done] episode steps = {n_steps}")
        print(f"[done] total reward (per agent, shared mode = identical):")
        for a, r in total_reward_per_agent.items():
            print(f"          {a}: {r:.2f}")

        # List output video files.
        videos = sorted(record_dir.glob("*.mp4"))
        if videos:
            print(f"[done] {len(videos)} video file(s) at {record_dir}:")
            for v in videos:
                size_mb = v.stat().st_size / 1024 / 1024
                print(f"          {v.name}  ({size_mb:.1f} MB)")
        else:
            print(f"[warn] no mp4 in {record_dir} — check MUJOCO_GL / ffmpeg.")

    finally:
        ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
