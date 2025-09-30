"""
Training Script for N++ RL Agent

This script implements HGT-based multimodal architecture
for training an RL agent to play N++. It uses the HGTMultimodalExtractor
with Heterogeneous Graph Transformers, type-specific attention, and multimodal fusion.

Key Features:
- HGT-based multimodal feature extraction (PRIMARY)
- Heterogeneous Graph Transformers with type-specific attention
- Specialized processing for different node/edge types
- Advanced multimodal fusion with cross-modal attention
- Adaptive exploration with ICM and novelty detection
- Comprehensive logging and evaluation

Usage:
    python -m npp_rl.agents.training --num_envs 64 --total_timesteps 10000000 --extractor_type hgt
"""

import argparse
import torch
from pathlib import Path
import json
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    DummyVecEnv,
    VecCheckNan,
)
from stable_baselines3.common.logger import configure

from nclone.gym_environment.npp_environment import (
    NppEnvironment,
)
from npp_rl.agents.hyperparameters.ppo_hyperparameters import (
    HYPERPARAMETERS,
    NET_ARCH_SIZE,
)
from npp_rl.feature_extractors import HGTMultimodalExtractor
from npp_rl.agents.adaptive_exploration import AdaptiveExplorationManager
from npp_rl.environments import create_reachability_aware_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HierarchicalLoggingCallback(BaseCallback):
    """Callback for logging hierarchical training metrics."""

    def __init__(
        self, exploration_manager: AdaptiveExplorationManager, log_freq: int = 1000
    ):
        super().__init__()
        self.exploration_manager = exploration_manager
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log exploration statistics
        if self.n_calls % self.log_freq == 0:
            stats = self.exploration_manager.get_exploration_stats()
            for key, value in stats.items():
                self.logger.record(f"exploration/{key}", value)

        # Log hierarchical extractor auxiliary losses if available
        if hasattr(self.model.policy.features_extractor, "get_auxiliary_losses"):
            aux_losses = self.model.policy.features_extractor.get_auxiliary_losses()
            for loss_name, loss_value in aux_losses.items():
                if isinstance(loss_value, torch.Tensor):
                    self.logger.record(f"hierarchical/{loss_name}", loss_value.item())

        return True


def create_environment(render_mode: str = "rgb_array", **kwargs):
    """Create environment with hierarchical graph observations and reachability features."""

    def _init():
        # Create base environment
        base_env = NppEnvironment(
            render_mode=render_mode,
            enable_animation=False,
            enable_logging=False,
            enable_debug_overlay=False,
            **kwargs,
        )

        env = create_reachability_aware_env(
            base_env=base_env,
            cache_ttl_ms=100.0,  # 100ms cache for real-time performance
            performance_target="fast",  # Use Tier-1 algorithms
            enable_monitoring=True,
            debug=False,
        )

        return env

    return _init


def train_agent(
    num_envs: int = 64,
    total_timesteps: int = 10_000_000,
    load_model: str = None,
    render_mode: str = "rgb_array",
    disable_exploration: bool = False,
    save_freq: int = 100_000,
    eval_freq: int = 50_000,
    log_interval: int = 10,
):
    """
    Train the multimodal agent with HGT or hierarchical architecture and reachability features.

    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        load_model: Path to existing model to resume training
        render_mode: Rendering mode ('rgb_array' for headless, 'human' for visual)
        disable_exploration: Whether to disable adaptive exploration
        save_freq: Frequency of model saves
        eval_freq: Frequency of evaluation
        log_interval: Logging interval
        extractor_type: Type of feature extractor ('hgt' or 'hierarchical')
    """

    # Force single environment for human rendering
    if render_mode == "human":
        num_envs = 1
        print("Human rendering mode detected. Setting num_envs=1.")

    print(f"Training hierarchical agent with {num_envs} environments")
    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps:,}")

    # Create timestamped log directory
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_dir = Path(f"./training_logs/ppo_training/session-{timestamp}")
    log_dir.mkdir(exist_ok=True, parents=True)

    # Create environment factory
    def make_env():
        return create_environment(render_mode=render_mode)

    # Create vectorized environment
    if num_envs == 1:
        env = DummyVecEnv([make_env])
    else:
        env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Add monitoring and NaN checking
    env = VecMonitor(env, str(log_dir / "monitor"))
    env = VecCheckNan(env, raise_exception=True)

    # Initialize adaptive exploration manager
    exploration_manager = None
    if not disable_exploration:
        exploration_manager = AdaptiveExplorationManager(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
        print("Adaptive exploration enabled (ICM + Novelty)")
    else:
        print("Adaptive exploration disabled")

    print("Using HGT-based multimodal extractor with reachability")
    extractor_kwargs = {
        "features_dim": 512,
        "hgt_hidden_dim": 256,
        "hgt_num_layers": 3,
        "hgt_output_dim": 256,
        "use_cross_modal_attention": True,
        "use_spatial_attention": True,
        "reachability_dim": 8,  # 8-dimensional reachability features
    }

    policy_kwargs = {
        "features_extractor_class": HGTMultimodalExtractor,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": [dict(pi=NET_ARCH_SIZE, vf=NET_ARCH_SIZE)],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,  # We handle normalization in the extractor
    }

    # Create or load model
    if load_model:
        print(f"Loading model from: {load_model}")
        model = PPO.load(
            load_model, env=env, device=device, policy_kwargs=policy_kwargs
        )
        # Update hyperparameters for continued training
        for key, value in HYPERPARAMETERS.items():
            if hasattr(model, key):
                setattr(model, key, value)
    else:
        # Create new model with hyperparameters
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            device=device,
            policy_kwargs=policy_kwargs,
            **HYPERPARAMETERS,
        )

    # Configure learning rate schedule
    model.learning_rate = get_linear_fn(3e-4, 1e-6, 0.8)

    # Setup logging
    logger = configure(str(log_dir / "tensorboard"), ["tensorboard", "csv"])
    model.set_logger(logger)

    # Save training configuration
    config = {
        "num_envs": num_envs,
        "total_timesteps": total_timesteps,
        "hyperparameters": HYPERPARAMETERS,
        "policy_kwargs": {k: str(v) for k, v in policy_kwargs.items()},
        "device": str(device),
        "exploration_enabled": not disable_exploration,
        "timestamp": timestamp,
    }

    with open(log_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup callbacks
    callbacks = []

    # Evaluation callback
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=eval_freq // num_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    callbacks.append(eval_callback)

    # Stop training callback
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=5, verbose=1
    )
    callbacks.append(stop_callback)

    # Logging callback
    if exploration_manager:
        logging_callback = HierarchicalLoggingCallback(
            exploration_manager=exploration_manager, log_freq=1000
        )
        callbacks.append(logging_callback)

    print("Starting training...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Monitor tensorboard with: tensorboard --logdir {log_dir / 'tensorboard'}")

    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            progress_bar=True,
        )

        # Save final model
        final_model_path = log_dir / "final_model"
        model.save(str(final_model_path))
        print(f"Training completed! Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current model
        interrupt_model_path = log_dir / "interrupted_model"
        model.save(str(interrupt_model_path))
        print(f"Model saved to: {interrupt_model_path}")

    finally:
        env.close()
        if "eval_env" in locals():
            eval_env.close()


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="N++ RL Agent Training")

    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments (default: 64)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps (default: 10M)",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path to existing model to resume training",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "human"],
        help="Rendering mode (default: rgb_array)",
    )
    parser.add_argument(
        "--disable_exploration",
        action="store_true",
        help="Disable adaptive exploration",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100_000,
        help="Model save frequency (default: 100k)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=50_000,
        help="Evaluation frequency (default: 50k)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval (default: 10)"
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        default="hgt",
        choices=["hgt", "hierarchical"],
        help="Feature extractor type: 'hgt' (recommended) or 'hierarchical' (default: hgt)",
    )
    parser.add_argument(
        "--disable_reachability",
        action="store_true",
        help="Disable reachability feature integration (default: enabled)",
    )

    args = parser.parse_args()

    # Train the agent
    train_agent(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        load_model=args.load_model,
        render_mode=args.render_mode,
        disable_exploration=args.disable_exploration,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        log_interval=args.log_interval,
        extractor_type=args.extractor_type,
    )


if __name__ == "__main__":
    main()
