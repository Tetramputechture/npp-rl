from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor
import torch
from torch import nn
import numpy as np
from pathlib import Path
import json
import datetime
import imageio
from npp_rl.environments.nplusplus import NPlusPlus
from npp_rl.agents.ppo_training_callback import PPOTrainingCallback
from npp_rl.game.game_controller import GameController
from npp_rl.agents.npp_feature_extractor import NppFeatureExtractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_training_env(env):
    """Prepare environment for training with proper monitoring."""
    # Create logging directory
    log_dir = Path('./training_logs/ppo_training_log')
    log_dir.mkdir(exist_ok=True)

    # Wrap environment with Monitor for logging
    env = Monitor(env, str(log_dir))

    return env, log_dir


def create_ppo_agent(env: NPlusPlus, n_steps: int) -> PPO:
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

    learning_rate = get_linear_fn(
        start=3e-4,    # Higher initial rate
        end=5e-5,      # Lower final rate for fine-tuning
        end_fraction=0.8  # Longer learning period
    )

    policy_kwargs = dict(
        features_extractor_class=NppFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=512,
        ),
        net_arch=dict(
            pi=[512, 512, 256, 128],  # Policy network
            vf=[512, 512, 256, 128]   # Matched value network
        ),
        # Layer normalization helps with varying episode lengths
        normalize_images=True,
        activation_fn=nn.ReLU,
        # net_arch_kwargs is not a valid argument for PPO
    )

    batch_size = n_steps // 16

    model = PPO(
        policy="CnnPolicy",
        env=env,

        # Learning parameters
        learning_rate=learning_rate,
        n_steps=n_steps,              # Shorter trajectories for better exploration
        batch_size=batch_size,        # Smaller batches for better generalization
        n_epochs=10,           # More epochs per update

        # Modified GAE parameters
        gamma=0.99,            # Higher discount for better long-term planning
        gae_lambda=0.95,       # Lower lambda for more emphasis on immediate rewards

        # PPO-specific parameters
        clip_range=0.2,        # More aggressive clipping
        clip_range_vf=0.2,     # Matched value function clipping
        ent_coef=0.025,        # Higher entropy for better exploration
        vf_coef=0.7,          # Lower value coefficient

        # Stability parameters
        max_grad_norm=0.7,     # Higher grad norm for faster learning
        target_kl=0.02,        # Higher KL target for more aggressive updates

        # Action noise is not a valid argument for PPO

        # Additional settings
        policy_kwargs=policy_kwargs,
        normalize_advantage=True,
        verbose=1,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        seed=42
    )

    return model


def train_ppo_agent(env: NPlusPlus, log_dir, game_controller: GameController, n_steps=2048, total_timesteps=1000000) -> PPO:
    """
    Trains the PPO agent

    Args:
        env: The N++ environment instance
        log_dir: Directory for saving logs and models
        total_timesteps: Total training timesteps
    """
    # Create and set up the model
    model = create_ppo_agent(env, n_steps)

    # Configure callback for monitoring and saving
    callback = PPOTrainingCallback(
        check_freq=50,
        log_dir=log_dir,
        game_controller=game_controller,
        n_steps=n_steps,
        min_ent_coef=0.005,
        max_ent_coef=0.03
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
        model = train_ppo_agent(
            env, log_dir, game_controller, total_timesteps=25000)

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
