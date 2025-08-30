from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
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
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from npp_rl.agents.hyperparameters.ppo_hyperparameters import HYPERPARAMETERS, NET_ARCH_SIZE
from npp_rl.feature_extractors import FeatureExtractor
from npp_rl.environments.vectorization_wrapper import make_vectorizable_env
from npp_rl.optimization.h100_optimization import enable_h100_optimizations, get_recommended_batch_size, H100OptimizedTraining
from npp_rl.callbacks import create_pbrs_callbacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_EVAL_FREQ_BASE = 10000


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


def create_ppo_agent(env, tensorboard_log: str, n_envs: int) -> PPO:
    """
    Creates a PPO agent for the N++ environment.

    Args:
        env: The N++ environment instance
        tensorboard_log: Directory for Tensorboard logs
        n_envs: Number of parallel environments

    Returns:
        PPO: Configured PPO model instance
    """

    learning_rate = get_linear_fn(
        start=3e-4,  # Higher starting LR for larger networks
        end=1e-6,    # Lower end LR for fine-tuning
        end_fraction=0.9
    )

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=NET_ARCH_SIZE,
            vf=NET_ARCH_SIZE
        ),
        normalize_images=False,  # We normalize in the feature extractor
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


def train_ppo_agent(env: BasicLevelNoGold, log_dir, n_envs: int, total_timesteps=1000000, load_model_path=None) -> PPO:
    """
    Trains the PPO agent with Tensorboard integration

    Args:
        env: The N++ environment instance
        log_dir: Directory for saving logs and models
        n_envs: Number of parallel environments
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
        model = create_ppo_agent(env, str(tensorboard_log), n_envs)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=30, min_evals=50, verbose=1)

    # Configure callback for monitoring and saving
    # Adjust eval_freq based on the number of environments
    eval_freq = max(TRAIN_EVAL_FREQ_BASE // n_envs, 1)
    eval_callback = EvalCallback(
        eval_env=env,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
        log_path=str(log_dir / "eval"),
        best_model_save_path=str(log_dir / "best_model"),
        callback_after_eval=stop_callback
    )
    
    # Create PBRS logging callbacks
    pbrs_callbacks = create_pbrs_callbacks(verbose=1)
    
    # Combine all callbacks
    callback = CallbackList([eval_callback] + pbrs_callbacks)

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


def start_training(load_model_path=None, render_mode='rgb_array', num_envs=64, env_kwargs=None, enable_h100_optimization=True):
    """Initialize environment and start training process.

    Args:
        load_model_path: Optional path to a previously saved model to continue training from
        render_mode: 'human' or 'rgb_array'
        num_envs: Number of parallel environments to use for training.
        env_kwargs: Dictionary of environment configuration parameters
        enable_h100_optimization: Whether to enable H100/GPU optimizations (TF32, memory management)
    """
    
    # Enable H100/GPU optimizations at the start of training
    optimization_status = None
    if enable_h100_optimization:
        optimization_status = enable_h100_optimizations(
            enable_tf32=True,
            enable_memory_optimization=True,
            log_optimizations=True
        )
        
        # Adjust batch size based on GPU capabilities if using CUDA
        if optimization_status['cuda_available']:
            recommended_batch_size = get_recommended_batch_size(
                device_name=optimization_status['device_name'],
                base_batch_size=HYPERPARAMETERS.get('batch_size', 2048)
            )
            print(f"Recommended batch size for {optimization_status['device_name']}: {recommended_batch_size}")
    
    # Default environment configuration with Phase 1 enhancements
    if env_kwargs is None:
        env_kwargs = {
            'enable_frame_stack': True,
            'observation_profile': 'rich',
            'enable_pbrs': True,
            'pbrs_weights': {
                'objective_weight': 1.0,
                'hazard_weight': 0.5, 
                'impact_weight': 0.3,
                'exploration_weight': 0.2
            },
            'pbrs_gamma': 0.99
        }
    
    print(f"Environment configuration: {env_kwargs}")

    try:
        # Test environment creation and compliance
        test_env_kwargs = env_kwargs.copy()
        test_env_kwargs['render_mode'] = render_mode
        env = make_vectorizable_env(test_env_kwargs)
        check_env(env)
        env.close()
        print("✅ Environment passed Gymnasium compliance check")

        if render_mode == 'human':
            print('Rendering in human mode with 1 environment')
            def make_env():
                kwargs = env_kwargs.copy()
                kwargs['render_mode'] = 'human'
                return make_vectorizable_env(kwargs)
            vec_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
        else:
            print(f'Rendering in rgb_array mode with {num_envs} environments using SubprocVecEnv')
            def make_env():
                kwargs = env_kwargs.copy()
                kwargs['render_mode'] = 'rgb_array'
                return make_vectorizable_env(kwargs)
            vec_env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)

        wrapped_env, log_dir = setup_training_env(vec_env)

        print("Starting PPO training...")
        
        # Use H100 optimization context if enabled
        if enable_h100_optimization and optimization_status and optimization_status['cuda_available']:
            with H100OptimizedTraining(enable_tf32=True, enable_memory_optimization=True, log_memory_usage=True):
                model = train_ppo_agent(
                    wrapped_env, log_dir, n_envs=num_envs, total_timesteps=1e7, load_model_path=load_model_path)
        else:
            model = train_ppo_agent(
                wrapped_env, log_dir, n_envs=num_envs, total_timesteps=1e7, load_model_path=load_model_path)

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
