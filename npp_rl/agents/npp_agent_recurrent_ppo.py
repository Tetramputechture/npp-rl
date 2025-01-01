from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecCheckNan, VecNormalize
import torch
import numpy as np
from pathlib import Path
import json
import datetime
import imageio
import subprocess
import threading
from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from stable_baselines3.common.callbacks import EvalCallback
from .hyperparameters.recurrent_ppo_hyperparameters import HYPERPARAMETERS
from npp_rl.agents.npp_feature_extractor_impala import NPPFeatureExtractorImpala

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_training_env(vec_env):
    """Prepare environment for training with monitoring."""
    # Create logging directory with timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_dir = Path(
        f'./training_logs/ppo_training_log/training_session-{timestamp}')
    log_dir.mkdir(exist_ok=True, parents=True)

    # Wrap environment with Monitor for logging
    env = VecMonitor(vec_env, str(log_dir))

    env = VecCheckNan(env, raise_exception=True)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

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


def create_ppo_agent(env: BasicLevelNoGold, tensorboard_log: str) -> RecurrentPPO:
    """
    Creates a PPO agent with optimized hyperparameters for the N++ environment.
    Memory-optimized version with smaller network architecture.

    Args:
        env: The N++ environment instance
        tensorboard_log: Directory for Tensorboard logs

    Returns:
        PPO: Configured PPO model instance
    """

    learning_rate = get_linear_fn(
        start=3e-4,
        end=5e-5,
        end_fraction=0.85
    )

    policy_kwargs = {
        "features_extractor_class": NPPFeatureExtractorImpala,
        "features_extractor_kwargs": {
            "features_dim": 512
        }
    }

    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=learning_rate,
        **HYPERPARAMETERS,
        tensorboard_log=tensorboard_log,
        device=device,
        seed=42
    )

    return model


def train_ppo_agent(env: BasicLevelNoGold, log_dir, total_timesteps=1e7, load_model_path=None) -> RecurrentPPO:
    """
    Trains the PPO agent with Tensorboard integration

    Args:
        env: The N++ environment instance
        log_dir: Directory for saving logs and models
        n_steps: Number of steps per update
        total_timesteps: Total training timesteps
        load_model_path: Optional path to a previously saved model to continue training from
    """
    # Set up Tensorboard log directory
    tensorboard_log = log_dir / "tensorboard"
    tensorboard_log.mkdir(exist_ok=True)

    # Start Tensorboard
    # Uncomment to use tensorboard
    # start_tensorboard(tensorboard_log)

    # Create or load the model
    if load_model_path is not None and Path(load_model_path).exists():
        print(f"Loading pre-trained model from {load_model_path}")
        model = RecurrentPPO.load(load_model_path, env=env)
        # Update the learning rate schedule
        model.learning_rate = get_linear_fn(
            start=2.5e-4,
            end=5e-5,
            end_fraction=0.8
        )
    else:
        print("Creating new model")
        model = create_ppo_agent(env, str(tensorboard_log))

    # Configure callback for monitoring and saving,
    # save best model in new directory saved_models/
    if not Path('./saved_models').exists():
        Path('./saved_models').mkdir(exist_ok=True)
    callback = EvalCallback(env, n_eval_episodes=5,
                            eval_freq=10000, deterministic=True, verbose=1, log_path=log_dir, best_model_save_path='./saved_models')

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        # Only reset if starting from scratch
        reset_num_timesteps=load_model_path is None
    )

    return model


def evaluate_agent(env: BasicLevelNoGold, policy: RecurrentPPO, n_episodes):
    """Evaluate the agent without computing gradients."""
    scores = []
    num_envs = 1

    for _ in range(n_episodes):
        state, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)
        episode_reward = 0
        done = False

        while not done:
            action, lstm_states = policy.predict(
                state, state=lstm_states, episode_start=episode_starts, deterministic=True)
            # convert action from 0d tensor to int
            action = action.item()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            done = terminated or truncated

        scores.append(episode_reward)

    return np.mean(scores), np.std(scores)


def record_video(env: BasicLevelNoGold, policy: RecurrentPPO, video_path, num_episodes=1):
    """Record a video of the trained agent playing."""
    images = []
    num_envs = 1

    for _ in range(num_episodes):
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)
        state = env.reset()[0]
        done = False

        while not done:
            images.append(env.render())
            action, lstm_states = policy.predict(
                state, state=lstm_states, episode_start=episode_starts, deterministic=True)
            action = action.item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    # Save video
    imageio.mimsave(video_path, [np.array(img) for img in images], fps=30)


def record_agent_training(env: BasicLevelNoGold, model: RecurrentPPO,
                          hyperparameters):
    """
    Evaluate, Generate a video and save the model locally
    This method does the complete pipeline:
    - It evaluates the model
    - It generates a replay video of the agent
    - It saves the model locally and replay video locally

    :param model: the pytorch model we want to save
    :param hyperparameters: training hyperparameters
    :param eval_env: evaluation environment
    :param video_fps: how many frame per seconds to record our video replay
    """
    # save to the directory agent_eval_data_ppo
    local_directory = Path('./agent_eval_data/recurrent_ppo_sim')
    local_directory.mkdir(exist_ok=True)

    # Step 2: Save the model
    model.save(local_directory / "model")

    # Step 3: Save the hyperparameters to JSON
    with open(local_directory / "hyperparameters.json", "w") as outfile:
        json.dump(hyperparameters, outfile)

    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(
        env, model, 5)
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
    env.reset()
    video_path = local_directory / "replay.mp4"
    record_video(env, model, video_path, num_episodes=3)

    return local_directory


def start_training(load_model_path=None, render_mode='rgb_array', n_envs=1):
    """Initialize environment and start training process. Assumes the player is already in the game
    and playing the level, as this method will attempt to press the reset key so training can start
    from a fresh level.

    Args:
        load_model_path: Optional path to a previously saved model to continue training from
    """

    try:
        env = BasicLevelNoGold(render_mode=render_mode)
        # check if the environment is valid
        check_env(env)

        # Print our observation and action spaces
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space}")

        if render_mode == 'human':
            print('Rendering in human mode with 1 environment')
            vec_env = make_vec_env(lambda: BasicLevelNoGold(render_mode='human'), n_envs=1,
                                   vec_env_cls=DummyVecEnv)
        else:
            print('Rendering in rgb_array mode with 4 environments')
            vec_env = make_vec_env(lambda: BasicLevelNoGold(render_mode='rgb_array'), n_envs=n_envs,
                                   vec_env_cls=SubprocVecEnv)
        wrapped_env, log_dir = setup_training_env(vec_env)

        print("Starting PPO training...")
        model = train_ppo_agent(
            wrapped_env, log_dir, total_timesteps=1e7, load_model_path=load_model_path)

        # Save final model
        print("Training completed. Saving model...")
        model.save("npp_recurrent_ppo_sim")

        # Record gameplay video
        # First, press the reset key to start a new episode
        print("Resetting environment to record video...")
        env.reset()

        print("Recording video...")
        hyperparameters = {
            "env_id": "N++",
            "n_evaluation_episodes": 3
        }
        record_agent_training(env, model, hyperparameters)

        print("Training completed successfully!")
        return model

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
