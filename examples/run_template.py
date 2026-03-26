"""
Template for running OmniPiano with any RL algorithm.

This example uses Stable Baselines 3 (PPO) as a demonstration, but the logging
is completely algorithm-agnostic and handled by the environment wrapper.
"""
import os

os.environ["MUJOCO_GL"] = "egl"

from OmniPiano import make
from OmniPiano.utils.iteration_summary_callback import TrainIterationSummaryCallback
from OmniPiano.utils.info_keys import EpisodeInfoKeys
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.env_util import make_vec_env


def main():
    # 1. Define logging directory
    base_experiment_name = "ppo_run_template"
    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_root, exist_ok=True)
    latest_run_id = get_latest_run_id(logs_root, base_experiment_name)
    next_run_id = latest_run_id + 1
    experiment_name = f"{base_experiment_name}_{next_run_id}"
    log_dir = os.path.join(logs_root, experiment_name)

    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    n_envs = 3

    # 2. Create the environment using the registered task name
    # env_name = "OmniPiano-Twinkle-RightHandWristLimit-v0"
    # env_name = "OmniPiano-Twinkle-RightHandOnly-v0"
    env_name = "OmniPiano-Twinkle-CollisionSafe-v0"
    print(f"Creating Safe/Robust environment: {env_name}")

    def env_creator():
        return make(
            env_name,
            log_dir=log_dir,
            log_split="train",
        )

    vec_env = make_vec_env(env_creator, n_envs=n_envs)

    # 3. Initialize RL Algorithm
    print("Initializing RL Algorithm...")
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        tensorboard_log=tensorboard_dir,
    )

    # 4. Train
    print("Starting training...")
    iteration_summary_callback = TrainIterationSummaryCallback(log_dir=log_dir)
    # model.learn(total_timesteps=30720, callback=iteration_summary_callback)
    model.learn(total_timesteps=15360, callback=iteration_summary_callback)
    print(f"Training completed. Check the CSV logs in: {log_dir}")

    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)
    print(f"Model saved to: {model_path}.zip")

    # 5. Evaluate and record video
    print("Testing environment step and recording video...")
    video_dir = os.path.join(log_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    eval_env = make(
        env_name,
        log_dir=log_dir,
        log_split="eval",
        record_dir=video_dir,
    )

    for ep_idx in range(1):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(
            f"Eval Episode {ep_idx + 1} | Reward: {total_reward:.3f} "
            f"| Safety Cost: {info.get(EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL, 0.0):.3f}"
        )

    eval_env.close()
    print(f"Training and evaluation logs are saved in: {log_dir}")
    print(f"Evaluation videos are saved in: {video_dir}")

if __name__ == "__main__":
    main()
