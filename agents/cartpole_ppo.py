from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import os
import tempfile
import imageio
import json
import datetime
import gymnasium
import torch
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainingCallback(BaseCallback):
    """Custom callback for monitoring PPO training progress."""

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = str(log_dir / 'best_model')
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timestamps = []

    def _get_episode_statistics(self):
        if len(self.model.ep_info_buffer) == 0:
            return 0.0, 0.0

        rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
        lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]

        mean_reward = np.mean(rewards)
        mean_length = np.mean(lengths)

        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(mean_length)
        self.episode_timestamps.append(self.n_calls)

        return mean_reward, mean_length

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            mean_reward, mean_length = self._get_episode_statistics()

            print("\n" + "="*50)
            print(f"Training Progress at Step {self.n_calls}")
            print(f"Mean episode reward: {mean_reward:.2f}")
            print(f"Mean episode length: {mean_length:.2f}")
            print(f"Number of episodes: {len(self.model.ep_info_buffer)}")
            print(f"Best mean reward: {self.best_mean_reward:.2f}")
            print("="*50 + "\n")

            if mean_reward > self.best_mean_reward:
                print(
                    f"New best mean reward: {mean_reward:.2f} -> Saving model")
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)

        return True

    def get_training_history(self):
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'timestamps': self.episode_timestamps
        }


def train_ppo_agent(env: VecEnv, total_timesteps):
    """Train a PPO agent using stable-baselines3."""

    # Create PPO model with optimized hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,          # Slightly lower than A2C for stability
        n_steps=2048,                # Larger batch size than A2C
        batch_size=64,               # Mini-batch size for optimization
        n_epochs=10,                 # Number of optimization epochs
        gamma=0.99,                  # Discount factor
        gae_lambda=0.95,             # GAE parameter
        clip_range=0.2,              # PPO clipping parameter
        clip_range_vf=None,          # No value function clipping
        ent_coef=0.01,              # Entropy coefficient
        vf_coef=0.5,                # Value function coefficient
        max_grad_norm=0.5,          # Gradient clipping
        target_kl=0.01,             # Target KL divergence
        normalize_advantage=True,
        verbose=1,
        device="cpu"                 # Use CPU for non-CNN policies
    )

    # Setup callback for monitoring
    log_dir = Path('./training_logs')
    log_dir.mkdir(exist_ok=True)

    callback = TrainingCallback(
        check_freq=10000,
        log_dir=log_dir
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    return model


def evaluate_agent(env: VecEnv, max_steps, n_eval_episodes, policy: PPO):
    """Evaluate the agent for n_eval_episodes episodes without computing gradients."""
    episode_rewards = []

    for _ in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.predict(state)
            new_state, reward, terminated, info = env.step(action)
            total_rewards_ep += reward

            if terminated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env: VecEnv, policy: PPO, out_directory, fps=30):
    """Generate a replay video of the agent."""
    images = []
    done = False
    state = env.reset()
    img = env.render()
    images.append(img)
    print("Recording video...")
    while not done:
        action, _ = policy.predict(state)
        state, reward, done, info = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img)
                    for i, img in enumerate(images)], fps=fps)


def record_agent_training(model: PPO, hyperparameters, eval_env, video_fps=30):
    """Evaluate, generate a video, and save the model locally."""
    local_directory = Path('./agent_eval_data')
    local_directory.mkdir(exist_ok=True)

    model.save(local_directory / "model")

    with open(local_directory / "hyperparameters.json", "w") as outfile:
        json.dump(hyperparameters, outfile)

    mean_reward, std_reward = evaluate_agent(
        eval_env,
        hyperparameters["max_t"],
        hyperparameters["n_evaluation_episodes"],
        model
    )

    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        "env_id": hyperparameters["env_id"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
        "eval_datetime": eval_form_datetime,
    }

    video_path = local_directory / "replay.mp4"
    record_video(eval_env, model, video_path, video_fps)

    return local_directory


def start_cartpole_training():
    """Start training the PPO agent on CartPole environment."""
    env_id = "CartPole-v1"

    # Create vectorized environment for training
    vec_env = make_vec_env(
        lambda: gymnasium.make(env_id, render_mode="rgb_array"),
        n_envs=4  # Using 4 environments for parallel training
    )

    # Create evaluation environment
    vec_eval_env = make_vec_env(
        lambda: gymnasium.make(env_id, render_mode="rgb_array"),
        n_envs=1
    )

    # Print environment information
    env = gymnasium.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print("Sample observation", env.observation_space.sample())
    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Action Space Sample", env.action_space.sample())

    # Train the PPO agent
    model = train_ppo_agent(vec_env, total_timesteps=1000)

    # Save the final model
    model.save("cartpole_ppo")

    # Record training results
    hyperparameters = {
        "env_id": env_id,
        "max_t": 1000,
        "n_evaluation_episodes": 10
    }
    out_dir = record_agent_training(model, hyperparameters, vec_eval_env)

    print(f"Training complete. Model saved to {out_dir}")


if __name__ == "__main__":
    start_cartpole_training()
