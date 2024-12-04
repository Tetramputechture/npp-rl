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
from environments.level_environment import NPlusPlus

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainingCallback(BaseCallback):
    """Custom callback for monitoring A2C training progress.

    This callback tracks episode rewards and lengths, providing regular updates
    during training and saving the best model when performance improves.
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = str(log_dir / 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self):
        # Check if it's time to evaluate
        if self.n_calls % self.check_freq == 0:
            # Get current rewards
            x, y = self.model.ep_info_buffer.get_mean_rewards()
            mean_reward = np.mean(y)

            # Print training progress
            print(f"Steps: {self.n_calls}")
            print(f"Mean reward: {mean_reward:.2f}")
            print(f"Episodes: {len(self.model.ep_info_buffer)}")
            print("-" * 40)

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)

        return True


class CustomLevelCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for N++ environment.

    Architecture designed for processing 4 frames of 84x84 grayscale images
    along with 8 channels of numerical data per frame. The network uses
    a CNN backbone similar to successful DQN architectures but modified
    to handle multiple frames and additional numerical data.

    Input format:
    - Visual: 4 frames of 84x84 grayscale images
    - Numerical: 32 values (8 channels × 4 frames)
    Output: 512-dimensional feature vector
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # Visual processing network
        # Input: (batch, 4, 84, 84)
        self.cnn = nn.Sequential(
            # Layer 1: 84x84 -> 20x20
            # (84 - 8)/4 + 1 = 20
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Layer 2: 20x20 -> 9x9
            # (20 - 4)/2 + 1 = 9
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 3: 9x9 -> 7x7
            # (9 - 3)/1 + 1 = 7
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten()
        )

        # Calculate CNN output dimension
        # Output will be 64 channels * 7 * 7 = 3136
        self.cnn_output_dim = 64 * 7 * 7

        # Network for processing numerical data
        # Input: 32 values (8 channels × 4 frames)
        self.numerical_net = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Combined network to merge visual and numerical features
        self.combined_net = nn.Sequential(
            # 3136 (CNN) + 256 (numerical) = 3392 input features
            nn.Linear(self.cnn_output_dim + 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),  # Final output: 512 dimensions
            nn.ReLU()
        )

    def forward(self, observations):
        """Forward pass through the network.

        Args:
            observations (dict): Contains:
                'visual': Tensor of shape (batch_size, 4, 84, 84)
                'numerical': Tensor of shape (batch_size, 32)

        Returns:
            torch.Tensor: Feature vector of shape (batch_size, 512)
        """
        # Process visual input (4 frames)
        visual_features = self.cnn(observations['visual'])

        # Process numerical data (32 values)
        numerical_features = self.numerical_net(observations['numerical'])

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
        check_freq=1000,    # Check every 1000 steps
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
