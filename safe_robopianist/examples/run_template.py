"""
Template for running SafeRoboPianist with any RL algorithm.

This example uses Stable Baselines 3 (PPO) as a demonstration, but the logging
is completely algorithm-agnostic and handled by the environment wrapper.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
#force Headless Rendering

# Add the parent directory to sys.path to allow imports from safe_robopianist
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# Also add the grand-parent directory so 'safe_robopianist' is recognized as a package
grandparent_dir = os.path.dirname(project_root)
sys.path.append(grandparent_dir)
# Add the robopianist directory to sys.path to fix the import error
sys.path.append(os.path.join(grandparent_dir, "robopianist"))

from envs.safe_piano_env import make
from iteration_summary_callback import TrainIterationSummaryCallback
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.env_util import make_vec_env

from safe_robopianist.configs import RobustConfig, SafetyConfig, TaskVariantConfig
from safe_robopianist.safety.constraints import JointMagnitudeConstraint

def main():
    # 1. Define logging directory
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
    
    # 2. Create the environment
    # We now use the registered task name! The registry handles all configs.
    # env_name = "SafeRoboPianist-debug-Twinkle-WristLimit-v0"
    env_name = "SafeRoboPianist-Twinkle-RightHandWristLimit-v0"
    print(f"Creating Safe/Robust environment: {env_name}")
    
    def env_creator():
        return make(
            env_name, 
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
    
    # The `callback` parameter allows us to inject custom logic into the SB3 training loop.
    # SB3 will automatically call specific methods on the callback object at different stages:
    # - `_on_training_start()`: Called once before the first step.
    # - `_on_step()`: Called after every single environment step.
    # - `_on_rollout_end()`: Called when the algorithm finishes collecting a batch of data (an iteration).
    # This mechanism lets us passively monitor and log data without modifying the core PPO algorithm.
    model.learn(total_timesteps=30720, callback=iteration_summary_callback)
    print(f"Training completed. Check the CSV logs in: {log_dir}")
    
    # Save the trained model
    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)
    print(f"Model saved to: {model_path}.zip")
    
    # 6. Test the trained model and record video
    print("Testing environment step and recording video...")
    video_dir = os.path.join(log_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    eval_env = make(
        env_name,
        log_dir=log_dir,
        log_split="eval",
        record_dir=video_dir,  # This will trigger PianoSoundVideoWrapper for audio+video
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
        
    eval_env.close()

    print(f"Training and evaluation logs are saved in: {log_dir}")
    print(f"Evaluation videos are saved in: {video_dir}")

if __name__ == "__main__":
    main()