"""
Template for running SafeRoboPianist with any RL algorithm.

This example uses Stable Baselines 3 (PPO) as a demonstration, but the logging
is completely algorithm-agnostic and handled by the environment wrapper.
"""
import os
import sys

# Add the parent directory to sys.path to allow imports from safe_robopianist
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.safe_piano_env import make
from iteration_summary_callback import TrainIterationSummaryCallback
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.env_util import make_vec_env

from safe_robopianist.configs import RobustConfig, SafetyConfig, TaskVariantConfig
from safe_robopianist.safety.constraints import JointMagnitudeConstraint

def main():
    # 1. Define configurations
    robust_config = RobustConfig(
        action_noise_std=0.05,
        obs_noise_std=0.01
    )
    
    safety_config = SafetyConfig(
        constraints=[
            # Limit Right Hand Wrist Pitch (WRJ1, index 1) magnitude to 0.5
            JointMagnitudeConstraint(index=1, max_magnitude=0.5, penalty_coef=5.0),
            # Limit Left Hand Wrist Pitch (WRJ1, index 23) magnitude to 0.5
            JointMagnitudeConstraint(index=23, max_magnitude=0.5, penalty_coef=5.0)
        ]
    )
    
    task_config = TaskVariantConfig(
        left_hand_immobile=False  # Set to True to test the XML modification!
    )
    
    # 2. Define logging directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_experiment_name = "ppo_run_template"
    logs_root = os.path.join(project_root, "logs")
    os.makedirs(logs_root, exist_ok=True)
    latest_run_id = get_latest_run_id(logs_root, base_experiment_name)
    next_run_id = latest_run_id + 1
    experiment_name = f"{base_experiment_name}_{next_run_id}"
    log_dir = os.path.join(logs_root, experiment_name)

    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    n_envs = 3
    
    # 3. Create the environment
    # By passing `log_dir`, the environment will automatically wrap itself with 
    # SafeRecordEpisodeStatistics and log metrics to CSV.
    env_name = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    print(f"Creating Safe/Robust environment: {env_name}")
    
    def env_creator():
        return make(
            env_name, 
            robust_config=robust_config, 
            safety_config=safety_config,
            task_config=task_config,
            log_dir=log_dir,
            log_split="train",
        )
        
    # Vectorize the environment
    vec_env = make_vec_env(env_creator, n_envs=n_envs)
    
    # 4. Initialize your RL Algorithm (e.g., PPO)
    print("Initializing RL Algorithm...")
    model = PPO(
        "MultiInputPolicy", 
        vec_env, 
        verbose=1,
        n_steps=512,
        batch_size=64,
        tensorboard_log=tensorboard_dir  # Keep TensorBoard files under log_dir.
    )
    
    # 5. Train
    # Notice we don't need any custom callbacks for CSV logging!
    print("Starting training...")
    iteration_summary_callback = TrainIterationSummaryCallback(log_dir=log_dir)
    model.learn(total_timesteps=8192, callback=iteration_summary_callback)
    print(f"Training completed. Check the CSV logs in: {log_dir}")
    
    # 6. Test the trained model
    print("Testing environment step...")
    eval_env = make(
        env_name,
        robust_config=robust_config,
        safety_config=safety_config,
        task_config=task_config,
        log_dir=log_dir,
        log_split="eval",
    )
    for ep_idx in range(3):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        from safe_robopianist.metrics.info_keys import EpisodeInfoKeys
        print(
            f"Eval Episode {ep_idx + 1} | Reward: {total_reward:.3f} "
            f"| Safety Cost: {info.get(EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL, 0.0):.3f}"
        )

    print(f"Training and evaluation logs are saved in: {log_dir}")

if __name__ == "__main__":
    main()