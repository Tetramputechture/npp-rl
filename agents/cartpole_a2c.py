from stable_baselines3 import A2C
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
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from collections import deque

import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainingCallback(BaseCallback):
    """Custom callback for monitoring A2C training progress.

    This callback tracks episode rewards and lengths, calculating running averages
    to monitor training progress. It saves the model when performance improves
    and provides regular updates during training.

    The callback processes the episode information buffer, which contains
    dictionaries of episode statistics, to extract meaningful metrics about
    the agent's learning progress.
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = str(log_dir / 'best_model')
        self.best_mean_reward = -np.inf

        # Keep track of episode rewards for plotting
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timestamps = []

    def _get_episode_statistics(self):
        """Calculate statistics from the episode info buffer.

        The episode info buffer contains dictionaries with 'r' (reward)
        and 'l' (length) keys for each completed episode.

        Returns:
            tuple: (mean_reward, mean_length) - Average reward and length
                  of recent episodes
        """
        # Extract rewards and lengths from episode info buffer
        if len(self.model.ep_info_buffer) == 0:
            return 0.0, 0.0

        rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
        lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]

        # Calculate means
        mean_reward = np.mean(rewards)
        mean_length = np.mean(lengths)

        # Store for later analysis
        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(mean_length)
        self.episode_timestamps.append(self.n_calls)

        return mean_reward, mean_length

    def _on_step(self):
        """Called after every step during training.

        This method:
        1. Checks if it's time to evaluate (based on check_freq)
        2. Calculates current performance metrics
        3. Saves the model if performance has improved
        4. Prints progress information

        Returns:
            bool: Whether to continue training
        """
        # Check if it's time to evaluate
        if self.n_calls % self.check_freq == 0:
            # Calculate current performance metrics
            mean_reward, mean_length = self._get_episode_statistics()

            # Print detailed training progress
            print("\n" + "="*50)
            print(f"Training Progress at Step {self.n_calls}")
            print(f"Mean episode reward: {mean_reward:.2f}")
            print(f"Mean episode length: {mean_length:.2f}")
            print(f"Number of episodes: {len(self.model.ep_info_buffer)}")
            print(f"Best mean reward: {self.best_mean_reward:.2f}")
            print("="*50 + "\n")

            # Save best model
            if mean_reward > self.best_mean_reward:
                print(
                    f"New best mean reward: {mean_reward:.2f} -> Saving model")
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)

        return True

    def get_training_history(self):
        """Get the training history for plotting or analysis.

        Returns:
            dict: Training history containing rewards, lengths, and timestamps
        """
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'timestamps': self.episode_timestamps
        }


def setup_training_env(env: VecEnv):
    """Prepare environment for training with proper monitoring."""
    # Create logging directory
    log_dir = Path('./training_logs')
    log_dir.mkdir(exist_ok=True)

    # Wrap environment with Monitor for logging
    env = Monitor(env.unwrapped, str(log_dir))

    return env, log_dir


def train_a2c_agent(env, total_timesteps=1000000):
    """Train an A2C agent using stable-baselines3.

    Args:
        env: The N++ environment
        total_timesteps: Number of steps to train for

    Returns:
        trained_model: The trained A2C model
    """
    # Prepare environment
    # env, log_dir = setup_training_env(env)

    # Create A2C model
    model = A2C(
        policy="MlpPolicy",  # Uses MLP policy network
        env=env,
        learning_rate=7e-4,
        n_steps=5,          # Number of steps before updating
        gamma=0.99,         # Discount factor
        gae_lambda=0.95,    # GAE parameter
        ent_coef=0.01,      # Entropy coefficient for exploration
        vf_coef=0.5,        # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        rms_prop_eps=1e-5,  # RMSprop epsilon
        use_rms_prop=True,  # Use RMSprop optimizer
        normalize_advantage=True,
        verbose=1,
        device="cpu"  # Non-CNN policies are faster on CPU
    )

    # Setup callback for monitoring
    log_dir = Path('./training_logs')
    log_dir.mkdir(exist_ok=True)

    callback = TrainingCallback(
        check_freq=10000,    # Check every 10000 steps
        log_dir=log_dir
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    return model


def evaluate_agent(env: VecEnv, max_steps, n_eval_episodes, policy: A2C):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
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


def record_video(env: VecEnv, policy: A2C, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    state = env.reset()
    img = env.render()
    images.append(img)
    print("Recording video...")
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.predict(state)
        # We directly put next_state = state for recording logic
        state, reward, done, info = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img)
                    for i, img in enumerate(images)], fps=fps)


def record_agent_training(model: A2C,
                          hyperparameters,
                          eval_env,
                          video_fps=30
                          ):
    """
    Evaluate, Generate a video and save the model locally
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It saves the model locally and replay video locally

    :param model: the pytorch model we want to save
    :param hyperparameters: training hyperparameters
    :param eval_env: evaluation environment
    :param video_fps: how many frame per seconds to record our video replay
    """
    # save to the directory agent_eval_data
    local_directory = Path('./agent_eval_data')

    # Step 2: Save the model
    model.save(local_directory / "model")

    # Step 3: Save the hyperparameters to JSON
    with open(local_directory / "hyperparameters.json", "w") as outfile:
        json.dump(hyperparameters, outfile)

    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(eval_env,
                                             hyperparameters["max_t"],
                                             hyperparameters["n_evaluation_episodes"],
                                             model)
    # Get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        "env_id": hyperparameters["env_id"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
        "eval_datetime": eval_form_datetime,
    }

    # Step 6: Record a video
    video_path = local_directory / "replay.mp4"
    record_video(eval_env, model, video_path, video_fps)

    return local_directory

# Now, lets test our cartpole training with a 'start_cartpole_training' function


def start_cartpole_training():
    # Train the agent
    env_id = "CartPole-v1"
    # Create the env
    env = gymnasium.make(env_id, render_mode="rgb_array")

    vec_env = make_vec_env(lambda: gymnasium.make(
        env_id, render_mode="rgb_array"), n_envs=4)

    # Create the evaluation env
    eval_env = gymnasium.make(env_id)

    vec_eval_env = make_vec_env(lambda: gymnasium.make(
        env_id, render_mode="rgb_array"), n_envs=1)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    # Get a random observation
    print("Sample observation", env.observation_space.sample())
    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    # Take a random action
    print("Action Space Sample", env.action_space.sample())

    model = train_a2c_agent(vec_env, total_timesteps=250000)

    model.save("cartpole_a2c")

    # Record the training
    hyperparameters = {
        "env_id": env_id,
        "max_t": 1000,
        "n_evaluation_episodes": 10
    }
    out_dir = record_agent_training(model, hyperparameters, vec_eval_env)

    print(f"Training complete. Model saved to {out_dir}")


start_cartpole_training()
