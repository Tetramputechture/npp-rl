from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecCheckNan, VecNormalize
import torch
from torch import nn
import numpy as np
from pathlib import Path
import json
import datetime
import imageio
import subprocess
import threading
from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from npp_rl.agents.hyperparameters.ppo_hyperparameters import HYPERPARAMETERS, NET_ARCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_N_ENVS = 64
TRAIN_EVAL_FREQ = max(10000 // TRAIN_N_ENVS, 1)


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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)

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


def create_ppo_agent(env: BasicLevelNoGold, tensorboard_log: str) -> PPO:
    """
    Creates a PPO agent with optimized hyperparameters for the N++ environment.
    Memory-optimized version with smaller network architecture.

    Args:
        env: The N++ environment instance
        n_steps: Number of steps to run for each environment per update
        tensorboard_log: Directory for Tensorboard logs

    Returns:
        PPO: Configured PPO model instance
    """

    learning_rate = get_linear_fn(
        start=0.00047032591206943436,
        end=5e-6,
        end_fraction=0.85
    )

    policy_kwargs = dict(
        # features_extractor_class=NPPFeatureExtractor,
        net_arch=dict(
            pi=NET_ARCH_SIZE,
            vf=NET_ARCH_SIZE
        ),
        normalize_images=True,
        activation_fn=nn.ReLU,
    )

    # Use hyperparameters from config file
    model = PPO(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=learning_rate,
        n_steps=HYPERPARAMETERS["n_steps"],
        batch_size=HYPERPARAMETERS["batch_size"],
        n_epochs=HYPERPARAMETERS["n_epochs"],
        gamma=HYPERPARAMETERS["gamma"],
        gae_lambda=HYPERPARAMETERS["gae_lambda"],
        clip_range=HYPERPARAMETERS["clip_range"],
        clip_range_vf=HYPERPARAMETERS["clip_range_vf"],
        ent_coef=HYPERPARAMETERS["ent_coef"],
        vf_coef=HYPERPARAMETERS["vf_coef"],
        max_grad_norm=HYPERPARAMETERS["max_grad_norm"],
        normalize_advantage=HYPERPARAMETERS["normalize_advantage"],
        verbose=HYPERPARAMETERS["verbose"],
        tensorboard_log=tensorboard_log,
        device=device,
        seed=42
    )

    return model


def train_ppo_agent(env: BasicLevelNoGold, log_dir, total_timesteps=1000000, load_model_path=None) -> PPO:
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
        model = PPO.load(load_model_path, env=env)
        # Update the learning rate schedule
        model.learning_rate = get_linear_fn(
            start=0.000117,
            end=0.000001,
            end_fraction=0.8
        )
    else:
        print("Creating new model")
        model = create_ppo_agent(env, str(tensorboard_log))

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=30, min_evals=50, verbose=1)

    # Configure callback for monitoring and saving
    callback = EvalCallback(
        eval_env=env,
        eval_freq=TRAIN_EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
        log_path=str(log_dir / "eval"),
        best_model_save_path=str(log_dir / "best_model"),
        callback_after_eval=stop_callback
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        # Only reset if starting from scratch
        reset_num_timesteps=load_model_path is None
    )

    return model


def evaluate_agent(env: BasicLevelNoGold, policy: PPO, n_episodes):
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


def record_video(env: BasicLevelNoGold, policy: PPO, video_path, num_episodes=1):
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


def record_agent_training(env: BasicLevelNoGold, model: PPO,
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
    local_directory = Path('./agent_eval_data/ppo_sim')
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
    env.reset()
    video_path = local_directory / "replay.mp4"
    record_video(env, model, video_path, num_episodes=3)

    return local_directory


def start_training(load_model_path=None, render_mode='rgb_array'):
    """Initialize environment and start training process.

    Args:
        load_model_path: Optional path to a previously saved model to continue training from
        render_mode: 'human' or 'rgb_array'
    """

    try:
        env = BasicLevelNoGold(render_mode=render_mode,
                               enable_frame_stack=False)
        check_env(env)

        if render_mode == 'human':
            print('Rendering in human mode with 1 environment')
            vec_env = make_vec_env(lambda: BasicLevelNoGold(render_mode='human', enable_frame_stack=False), n_envs=1,
                                   vec_env_cls=DummyVecEnv)
        else:
            print(
                f'Rendering in rgb_array mode with {TRAIN_N_ENVS} environments')
            vec_env = make_vec_env(lambda: BasicLevelNoGold(render_mode='rgb_array', enable_frame_stack=False), n_envs=TRAIN_N_ENVS,
                                   vec_env_cls=SubprocVecEnv)

        wrapped_env, log_dir = setup_training_env(vec_env)

        print("Starting PPO training...")
        model = train_ppo_agent(
            wrapped_env, log_dir, total_timesteps=1e7, load_model_path=load_model_path)

        # Save final model
        print("Training completed. Saving model...")
        # Create a new directory for the model
        model_dir = Path('./agent_eval_data/ppo_sim')
        model_dir.mkdir(exist_ok=True)
        model.save(model_dir / "model")

        # Record gameplay video
        # First, press the reset key to start a new episode
        print("Resetting environment to record video...")
        env.reset()

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
