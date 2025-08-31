"""
Training Script for N++ RL Agent

This script implements state-of-the-art improvements for training an RL agent
to play N++ based on recent research in procedural environments (e.g., ProcGen benchmarks),
large-scale RL (e.g., OpenAI Five, IMPALA), and exploration strategies (e.g., ICM, Go-Explore).

Usage:
    python training.py --num_envs 64 --total_timesteps 10000000
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines3.common.logger import configure

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from npp_rl.agents.hyperparameters.ppo_hyperparameters import HYPERPARAMETERS, NET_ARCH_SIZE
from npp_rl.feature_extractors import FeatureExtractor
from npp_rl.agents.adaptive_exploration import AdaptiveExplorationManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoggingCallback(BaseCallback):
    """Callback for logging training metrics and exploration statistics."""
    
    def __init__(self, exploration_manager: AdaptiveExplorationManager, log_freq: int = 1000):
        super().__init__()
        self.exploration_manager = exploration_manager
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log exploration statistics periodically
        if self.n_calls % self.log_freq == 0:
            stats = self.exploration_manager.get_exploration_stats()
            for key, value in stats.items():
                self.logger.record(f"exploration/{key}", value)
                
        return True
    
    def _on_rollout_end(self) -> None:
        # Log episode statistics
        if hasattr(self.training_env, 'get_attr'):
            try:
                episode_rewards = self.training_env.get_attr('episode_rewards')
                episode_lengths = self.training_env.get_attr('episode_lengths')
                
                if episode_rewards and episode_rewards[0]:
                    avg_reward = np.mean([r for env_rewards in episode_rewards for r in env_rewards])
                    self.logger.record("rollout/avg_episode_reward", avg_reward)
                    
                if episode_lengths and episode_lengths[0]:
                    avg_length = np.mean([length for env_lengths in episode_lengths for length in env_lengths])
                    self.logger.record("rollout/avg_episode_length", avg_length)
                    
            except Exception:
                # Gracefully handle any logging errors
                pass


def create_ppo_agent(env, tensorboard_log: str, n_envs: int, 
                              exploration_manager: AdaptiveExplorationManager = None) -> PPO:
    """
    Creates a PPO agent
    
    Args:
        env: The N++ environment instance
        tensorboard_log: Directory for Tensorboard logs
        n_envs: Number of parallel environments
        exploration_manager: Adaptive exploration manager
        
    Returns:
        PPO: PPO model instance
    """
    
    learning_rate = get_linear_fn(
        start=3e-4,  # Higher starting LR for larger networks
        end=1e-6,    # Lower end LR for fine-tuning
        end_fraction=0.9
    )
    
    features_extractor_class = FeatureExtractor
    print("🚀 Using 3D Feature Extractor with temporal modeling")

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=NET_ARCH_SIZE,
            vf=NET_ARCH_SIZE
        ),
        normalize_images=False,  # We normalize in the feature extractor
        activation_fn=torch.nn.ReLU,
    )
    
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
    
    # Initialize curiosity module if exploration manager is provided
    # (Utilizing ICM for intrinsic motivation and novelty for broader search)
    if exploration_manager is not None:
        exploration_manager.initialize_curiosity_module(feature_dim=512, action_dim=5)
    
    return model


def setup_training_env(num_envs: int, render_mode: str = 'rgb_array'):
    """Set up training environment with monitoring."""
    
    # Create timestamp for logging
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_dir = Path(f'./training_logs/ppo_training/session-{timestamp}')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create environment factory
    def make_env():
        return BasicLevelNoGold(
            render_mode=render_mode,
            enable_frame_stack=True,  # Enable 12-frame stacking
            enable_animation=False,
            enable_logging=False,
            enable_debug_overlay=False,
            enable_short_episode_truncation=False
        )
    
    # Create vectorized environment
    if render_mode == 'human':
        print('🎮 Rendering in human mode with 1 environment')
        vec_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
    else:
        print(f'🚀 Creating {num_envs} parallel environments for training')
        vec_env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)
    
    # Wrap with monitoring and normalization
    env = VecMonitor(vec_env, str(log_dir))
    env = VecCheckNan(env, raise_exception=True)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)
    
    return env, log_dir


def train_agent(num_envs: int = 64,
                         total_timesteps: int = 10_000_000,
                         load_model_path: str = None,
                         render_mode: str = 'rgb_array',
                         enable_exploration: bool = True):
    """
    Train the N++ agent with all improvements.
    
    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        load_model_path: Path to load existing model
        render_mode: Rendering mode ('rgb_array' or 'human')
        enable_exploration: Whether to enable adaptive exploration
    """
    
    print("🚀 Starting N++ Agent Training")
    print("📊 Configuration:")
    print(f"   - Environments: {num_envs}")
    print(f"   - Total timesteps: {total_timesteps:,}")
    print("   - Temporal frames: 12")
    print(f"   - Network architecture: {NET_ARCH_SIZE}")
    print(f"   - Adaptive exploration: {enable_exploration}")
    print(f"   - Device: {device}")
    
    try:
        # Set up environment
        env, log_dir = setup_training_env(num_envs, render_mode)
        
        # Set up tensorboard logging
        tensorboard_log = log_dir / "tensorboard"
        tensorboard_log.mkdir(exist_ok=True)
        
        # Initialize adaptive exploration manager
        exploration_manager = None
        if enable_exploration:
            exploration_manager = AdaptiveExplorationManager(
                curiosity_weight=0.1,
                novelty_weight=0.05,
                progress_window=100
            )
            print("🔍 Adaptive exploration enabled")
        
        # Create or load model
        if load_model_path and Path(load_model_path).exists():
            print(f"📂 Loading pre-trained model from {load_model_path}")
            model = PPO.load(load_model_path, env=env)
            # Update learning rate for continued training
            model.learning_rate = get_linear_fn(start=1e-4, end=1e-6, end_fraction=0.8)
        else:
            print("🆕 Creating new model")
            model = create_ppo_agent(
                env, str(tensorboard_log), num_envs, exploration_manager
            )
        
        # Set up callbacks
        callbacks = []
        
        if exploration_manager:
            callbacks.append(LoggingCallback(exploration_manager, log_freq=1000))
        
        # Early stopping callback
        # (Common practice to prevent overfitting and save computational resources)
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=30, 
            min_evals=50, 
            verbose=1
        )
        
        # Evaluation callback
        eval_freq = max(10000 // num_envs, 1)
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
        callbacks.append(eval_callback)
        
        # Configure logger
        model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
        
        print("🎯 Starting training...")
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=load_model_path is None
        )
        
        # Save final model
        final_model_path = log_dir / "final_model"
        model.save(final_model_path)
        print(f"💾 Final model saved to {final_model_path}")
        
        # Save training configuration
        config = {
            "num_envs": num_envs,
            "total_timesteps": total_timesteps,
            "enable_exploration": enable_exploration,
            "hyperparameters": HYPERPARAMETERS,
            "net_arch": NET_ARCH_SIZE,
            "device": str(device),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(log_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Print exploration statistics if available
        if exploration_manager:
            stats = exploration_manager.get_exploration_stats()
            print("\n🔍 Final Exploration Statistics:")
            for key, value in stats.items():
                print(f"   - {key}: {value:.4f}")
        
        print("✅ Training completed successfully!")
        print(f"📁 Logs saved to: {log_dir}")
        
        return model, log_dir
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main training function with command line arguments."""
    parser = argparse.ArgumentParser(description="N++ RL Agent Training")
    
    parser.add_argument("--num_envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000,
                        help="Total training timesteps")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load existing model")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        choices=["rgb_array", "human"],
                        help="Rendering mode")
    parser.add_argument("--disable_exploration", action="store_true",
                        help="Disable adaptive exploration")
    
    args = parser.parse_args()
    
    enable_exploration = not args.disable_exploration
    
    # Start training
    model, log_dir = train_agent(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        load_model_path=args.load_model,
        render_mode=args.render_mode,
        enable_exploration=enable_exploration
    )
    
    if model is not None:
        print(f"\n🎉 Training completed! Model and logs saved to: {log_dir}")
    else:
        print("\n💥 Training failed!")


if __name__ == "__main__":
    main() 