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
import subprocess
import threading
from npp_rl.environments.nplusplus import NPlusPlus
from npp_rl.agents.ppo_training_callback import PPOTrainingCallback
from npp_rl.game.game_controller import GameController
from npp_rl.agents.npp_feature_extractor import NPPFeatureExtractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_training_env(env):
    """Prepare environment for training with monitoring."""
    # Create logging directory with timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_dir = Path(
        f'./training_logs/ppo_training_log/training_session-{timestamp}')
    log_dir.mkdir(exist_ok=True, parents=True)

    # Wrap environment with Monitor for logging
    env = Monitor(env, str(log_dir))

    return env, log_dir


def start_tensorboard(logdir):
    """Start Tensorboard in a separate thread."""
    def run_tensorboard():
        subprocess.Popen(['tensorboard', '--logdir',
                         str(logdir), '--port', '6006'])

    tensorboard_thread = threading.Thread(target=run_tensorboard)
    tensorboard_thread.daemon = True
    tensorboard_thread.start()
    print("Tensorboard started. View at http://localhost:6006")


def create_ppo_agent(env: NPlusPlus, n_steps: int, tensorboard_log: str) -> PPO:
    """
    Creates a PPO agent with optimized hyperparameters for the N++ environment.

    Args:
        env: The N++ environment instance
        n_steps: Number of steps to run for each environment per update
        tensorboard_log: Directory for Tensorboard logs

    Returns:
        PPO: Configured PPO model instance
    """

    learning_rate = get_linear_fn(
        start=2.5e-4,
        end=5e-5,
        end_fraction=0.8
    )

    policy_kwargs = dict(
        features_extractor_class=NPPFeatureExtractor,
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        ),
        normalize_images=True,
        activation_fn=nn.ReLU,
    )

    batch_size = min(512, n_steps // 4)

    model = PPO(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        normalize_advantage=True,
        verbose=1,
        # Uncomment to use tensorboard
        # tensorboard_log=tensorboard_log,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        seed=42
    )

    return model


def train_ppo_agent(env: NPlusPlus, log_dir, game_controller: GameController, n_steps=2048, total_timesteps=1000000) -> PPO:
    """
    Trains the PPO agent with Tensorboard integration

    Args:
        env: The N++ environment instance
        log_dir: Directory for saving logs and models
        game_controller: Game controller instance
        n_steps: Number of steps per update
        total_timesteps: Total training timesteps
    """
    # Set up Tensorboard log directory
    tensorboard_log = log_dir / "tensorboard"
    tensorboard_log.mkdir(exist_ok=True)

    # Start Tensorboard
    # Uncomment to use tensorboard
    # start_tensorboard(tensorboard_log)

    # Create and set up the model
    model = create_ppo_agent(env, n_steps, str(tensorboard_log))

    # Configure callback for monitoring and saving
    callback = PPOTrainingCallback(
        check_freq=50,
        log_dir=log_dir,
        game_controller=game_controller,
        n_steps=n_steps,
        min_ent_coef=0.0005,
        max_ent_coef=0.01
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
        wrapped_env, log_dir = setup_training_env(env)
        game_controller.press_reset_key()

        s_size = env.observation_space.shape[0]
        a_size = env.action_space.n

        print("_____OBSERVATION SPACE_____ \n")
        print("The State Space is: ", s_size)
        print("\n _____ACTION SPACE_____ \n")
        print("The Action Space is: ", a_size)

        print("Starting PPO training...")
        model = train_ppo_agent(
            wrapped_env, log_dir, game_controller, total_timesteps=100000)

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
