from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import torch
from torch import nn
import numpy as np
from pathlib import Path
import json
import datetime
import imageio
from environments.nplusplus import NPlusPlus

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


class CustomLevelCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for N++ environment.

    Architecture designed to process observations from our N++ environment where:
    Input: Single tensor of shape (84, 84, frame_stack + 8) containing:
        - First frame_stack channels: Stacked grayscale frames
        - Last 8 channels: Numerical features broadcast to 84x84 spatial dimensions

    The network separates and processes visual and numerical data through appropriate 
    pathways before combining them into a final feature representation.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]  # frame_stack + 8
        self.frame_stack = n_input_channels - 8  # Subtract feature channels

        # Visual processing network for the stacked frames
        self.cnn = nn.Sequential(
            # Reshape and process frame stack
            # Input: (batch, 84, 84, frame_stack)
            # Convert to: (batch, frame_stack, 84, 84)

            # Layer 1: 84x84 -> 20x20
            nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Layer 2: 20x20 -> 9x9
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 3: 9x9 -> 7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten()
        )

        # Calculate CNN output dimension
        # 64 channels * 7 * 7 = 3136
        self.cnn_output_dim = 64 * 7 * 7

        # Network for processing numerical features
        # Input: Flattened 84x84x8 numerical features
        self.numerical_net = nn.Sequential(
            # First reduce spatial dimensions with a small CNN
            nn.Conv2d(8, 16, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 20x20 -> 9x9
            nn.ReLU(),
            nn.Flatten(),
            # Process resulting features
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU()
        )

        # Combined network to merge visual and numerical features
        self.combined_net = nn.Sequential(
            nn.Linear(self.cnn_output_dim + 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Process the combined observation tensor.

        Args:
            observations: Tensor of shape (batch_size, 84, 84, frame_stack + 8)
                        Contains stacked frames and numerical features

        Returns:
            torch.Tensor: Feature vector of shape (batch_size, 512)
        """
        # Split observation into visual and numerical components
        # Move channel dimension to proper position for CNNs
        visual = observations[..., :self.frame_stack]  # Get stacked frames
        visual = visual.permute(0, 3, 1, 2)  # (batch, frame_stack, 84, 84)

        # Get numerical features
        numerical = observations[..., self.frame_stack:]
        numerical = numerical.permute(0, 3, 1, 2)  # (batch, 8, 84, 84)

        # Process visual path
        visual_features = self.cnn(visual)

        # Process numerical path
        numerical_features = self.numerical_net(numerical)

        # Combine features
        combined = torch.cat([visual_features, numerical_features], dim=1)

        # Generate final feature vector
        return self.combined_net(combined)


def setup_training_env(env):
    """Prepare environment for training with proper monitoring."""
    # Create logging directory
    log_dir = Path('./training_logs')
    log_dir.mkdir(exist_ok=True)

    # Wrap environment with Monitor for logging
    env = Monitor(env, str(log_dir))

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
    env, log_dir = setup_training_env(env)

    policy_kwargs = dict(
        features_extractor_class=CustomLevelCNN,
        features_extractor_kwargs=dict(features_dim=512)
    )

    # Create A2C model with custom policy network parameters
    model = A2C(
        policy="CnnPolicy",  # Uses CNN for visual input
        policy_kwargs=policy_kwargs,  # Custom policy network
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
        verbose=1
    )

    # Setup callback for monitoring
    callback = TrainingCallback(
        check_freq=50,    # Check every 50 steps
        log_dir=log_dir
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    return model


def evaluate_agent(env, policy, n_episodes):
    """Evaluate the agent without computing gradients."""
    scores = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Use evaluation mode to disable gradient tracking
            action, _ = policy.act(state, evaluation=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        scores.append(episode_reward)

    return np.mean(scores), np.std(scores)


def record_agent_training(env, model,
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
    torch.save(model, local_directory / "model.pt")

    # Step 3: Save the hyperparameters to JSON
    with open(local_directory / "hyperparameters.json", "w") as outfile:
        json.dump(hyperparameters, outfile)

    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(
        env, model, hyperparameters['n_episodes'])
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

    # Write a JSON file
    with open(local_directory / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    model_card = f"""
# **A2C** Agent playing **N++**
This is a trained model of a **A2C** agent playing **N++** .
"""

    readme_path = local_directory / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Step 6: Record a video
    video_path = local_directory / "replay.mp4"
    record_video(env, model, video_path, video_fps)

    return local_directory


def record_video(env, model, video_path, num_episodes=1):
    """Record a video of the trained agent playing."""
    images = []
    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = False

        while not done:
            images.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    # Save video
    imageio.mimsave(video_path, [np.array(img) for img in images], fps=30)


def start_training(game_value_fetcher, game_controller):
    """Initialize environment and start training process."""
    try:
        # Create environment
        env = NPlusPlus(game_value_fetcher, game_controller)
        env.reset()

        print("Starting A2C training...")
        model = train_a2c_agent(env, total_timesteps=1000000)

        # Save final model
        model.save("npp_a2c_final")

        # Record gameplay video
        video_path = Path('./training_logs/gameplay.mp4')
        record_video(env, model, video_path)

        print("Training completed successfully!")
        return model

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
