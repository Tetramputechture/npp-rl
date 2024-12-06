from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from game.game_controller import GameController
import torch
from torch import nn
import numpy as np
from pathlib import Path
import json
import datetime
import imageio
from environments.nplusplus import NPlusPlus
from agents.ppo_training_callback import PPOTrainingCallback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_NUMERICAL_FEATURES = 11


class CustomLevelCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for N++ environment.

    Architecture designed to process observations from our N++ environment where:
    Input: Single tensor of shape (84, 84, frame_stack + 9) containing:
        - First frame_stack channels: Stacked grayscale frames
        - Last 9 channels: Numerical features broadcast to 84x84 spatial dimensions

    The network separates and processes visual and numerical data through appropriate 
    pathways before combining them into a final feature representation.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]
        self.frame_stack = n_input_channels - NUM_NUMERICAL_FEATURES

        # Visual processing network
        self.cnn = nn.Sequential(
            # Layer 1: 84x84 -> 20x20
            nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Layer 2: 20x20 -> 9x9
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Layer 3: 9x9 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Added residual connection
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Flatten()
        )

        # Calculate CNN output dimension: 128 channels * 7 * 7 = 6272
        self.cnn_output_dim = 128 * 7 * 7

        # Numerical feature processing
        self.numerical_net = nn.Sequential(
            nn.Conv2d(NUM_NUMERICAL_FEATURES, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(self.cnn_output_dim + 512, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Split and process visual input
        visual = observations[..., :self.frame_stack]
        visual = visual.permute(0, 3, 1, 2)

        # Process numerical features
        numerical = observations[..., self.frame_stack:]
        numerical = numerical.permute(0, 3, 1, 2)

        # Extract features through respective pathways
        visual_features = self.cnn(visual)
        numerical_features = self.numerical_net(numerical)

        # Combine and process features
        combined = torch.cat([visual_features, numerical_features], dim=1)
        return self.combined_net(combined)


def setup_training_env(env):
    """Prepare environment for training with proper monitoring."""
    # Create logging directory
    log_dir = Path('./training_logs/ppo_training_log')
    log_dir.mkdir(exist_ok=True)

    # Wrap environment with Monitor for logging
    env = Monitor(env, str(log_dir))

    return env, log_dir


def create_ppo_agent(env: NPlusPlus) -> PPO:
    """
    Creates a PPO agent with optimized hyperparameters for the N++ environment.

    The hyperparameters are specifically tuned for:
    - Complex visual inputs (frame stacks + numerical features)
    - Long episode horizons
    - Sparse but shaped rewards
    - Precise movement requirements

    Args:
        env: The N++ environment instance
        total_timesteps: Total number of timesteps to train for

    Returns:
        PPO: Configured PPO model instance
    """

    # Learning rate schedule: Start higher for faster initial learning,
    # then decay to allow fine-tuning of policies
    learning_rate = get_linear_fn(
        start=3e-4,  # Higher initial rate for faster early learning
        end=5e-5,    # Lower final rate for stability
        end_fraction=0.85  # Decay over most of training
    )

    # Entropy coefficient schedule: Start high for exploration,
    # then decay to encourage exploitation of learned policies
    ent_coef = get_linear_fn(
        start=0.01,  # Initial entropy for exploration
        end=0.001,   # Final entropy for exploitation
        end_fraction=0.7  # Decay more quickly than learning rate
    )

    seed = 42

    # Configure policy network architecture
    policy_kwargs = dict(
        # Features extractor configuration
        features_extractor_class=CustomLevelCNN,
        features_extractor_kwargs=dict(features_dim=512),

        # Separate network architectures for policy and value functions
        # Larger networks to handle complex state space
        net_arch=dict(
            pi=[512, 256, 128],  # Policy network
            vf=[512, 256, 128]   # Value network
        ),

        # Use ReLU for faster training
        activation_fn=nn.ReLU
    )

    # Setup exploration noise
    n_actions = env.action_space.n

    # Unused for now, but can be added for exploration
    # Note, the PPO constructor in stable-baselines3 does not support action noise,
    # so we will have to add this manually if needed via a wrapper around the PPO model
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),  # Moderate noise magnitude
        theta=0.15,  # Rate of mean reversion
        dt=1/60  # Match game's frame rate
    )

    # Initialize PPO with optimized parameters
    model = PPO(
        policy="CnnPolicy",
        env=env,

        # Learning parameters
        learning_rate=learning_rate,
        n_steps=2048,        # Reduced for more frequent updates
        batch_size=64,       # Smaller batches for more stable gradients
        n_epochs=10,         # More epochs per update for better learning

        # GAE and discount parameters
        gamma=0.99,          # Standard discount factor
        gae_lambda=0.95,     # Slightly higher for better advantage estimation

        # PPO-specific parameters
        clip_range=0.2,      # Standard PPO clipping
        clip_range_vf=0.2,   # Match policy clipping
        # Entropy coefficient for exploration. Note: we alter this using our PPOTrainingCallback
        # We set this to the max value to encourage exploration, but our callback will adjust it
        # to maintain a target KL divergence
        ent_coef=0.02,
        vf_coef=0.8,        # Increased value function importance

        # Training stability parameters
        max_grad_norm=0.5,   # Prevent explosive gradients
        target_kl=0.015,     # Conservative policy updates

        # Additional settings
        policy_kwargs=policy_kwargs,
        # action_noise=action_noise,
        normalize_advantage=True,  # Important for stable training
        verbose=1,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        seed=seed
    )

    return model


def train_ppo_agent(env: NPlusPlus, log_dir, total_timesteps=1000000) -> PPO:
    """
    Trains the PPO agent

    Args:
        env: The N++ environment instance
        log_dir: Directory for saving logs and models
        total_timesteps: Total training timesteps
    """
    # Create and set up the model
    model = create_ppo_agent(env)

    # Configure callback for monitoring and saving
    callback = PPOTrainingCallback(
        check_freq=50,
        log_dir=log_dir,
        min_ent_coef=0.005,
        max_ent_coef=0.02
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    return model


def evaluate_agent(env: NPlusPlus, policy: PPO, n_episodes):
    """Evaluate the agent without computing gradients."""
    scores = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Use evaluation mode to disable gradient tracking
            action, _ = policy.predict(state)
            # convert action from 0d tensor to int
            action = action.item()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            done = terminated or truncated

        scores.append(episode_reward)

    return np.mean(scores), np.std(scores)


def record_video(env: NPlusPlus, policy: PPO, video_path, num_episodes=1):
    """Record a video of the trained agent playing."""
    images = []
    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False

        while not done:
            images.append(env.render())
            action, _ = policy.predict(state, deterministic=True)
            action = action.item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    # Save video
    imageio.mimsave(video_path, [np.array(img) for img in images], fps=30)


def record_agent_training(env: NPlusPlus, model: PPO,
                          hyperparameters):
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
    # save to the directory agent_eval_data_ppo
    local_directory = Path('./agent_eval_data/ppo')
    local_directory.mkdir(exist_ok=True)

    # Step 2: Save the model
    model.save(local_directory / "model")

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

    print("Evaluation results:", evaluate_data)

    print("Recording video...")

    # Step 6: Record videos for 5 episodes
    video_path = local_directory / "replay.mp4"
    record_video(env, model, video_path, num_episodes=5)

    return local_directory


def start_training(game_value_fetcher, game_controller: GameController):
    """Initialize environment and start training process. Assumes the player is already in the game
    and playing the level, as this method will attempt to press the reset key so training can start
    from a fresh level."""

    try:
        env = NPlusPlus(game_value_fetcher, game_controller)
        game_controller.press_reset_key()

        s_size = env.observation_space.shape[0]
        a_size = env.action_space.n

        print("_____OBSERVATION SPACE_____ \n")
        print("The State Space is: ", s_size)
        print("\n _____ACTION SPACE_____ \n")
        print("The Action Space is: ", a_size)

        print("Starting PPO training...")
        log_dir = Path('./training_logs/ppo_training_log')
        model = train_ppo_agent(env, log_dir, total_timesteps=25000)

        # Save final model
        print("Training completed. Saving model...")
        model.save("npp_ppo")

        # Record gameplay video
        # First, press the reset key to start a new episode
        print("Resetting environment to record video...")
        game_controller.press_reset_key()

        print("Recording video...")
        hyperparameters = {
            "env_id": "N++",
            "n_episodes": 5,
            "n_evaluation_episodes": 5
        }
        record_agent_training(env, model, hyperparameters)

        print("Training completed successfully!")
        return model

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
