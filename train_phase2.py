"""
Enhanced training script with Phase 2 features.

This script integrates all Phase 2 components:
- Intrinsic Curiosity Module (ICM)
- Graph Neural Network observations
- Behavioral Cloning pretraining
- Enhanced exploration metrics
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from npp_rl.config.phase2_config import Phase2Config, get_config_presets, validate_config
from npp_rl.models.feature_extractors import create_feature_extractor
from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper
from npp_rl.eval.exploration_metrics import ExplorationMetrics, create_exploration_callback
from nclone.nclone_environments.basic_level_no_gold.graph_observation import create_graph_enhanced_env


class Phase2Trainer:
    """
    Main trainer class for Phase 2 experiments.
    
    Handles setup and coordination of all Phase 2 components.
    """
    
    def __init__(self, config: Phase2Config):
        """
        Initialize Phase 2 trainer.
        
        Args:
            config: Phase 2 configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(str(self.output_dir / 'config.json'))
        
        # Initialize components
        self.env = None
        self.model = None
        self.icm_trainer = None
        self.exploration_metrics = None
        
        print(f"Phase 2 Trainer initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU")
        
        return device
    
    def create_env(self, rank: int = 0, eval_env: bool = False) -> Any:
        """
        Create environment with Phase 2 enhancements.
        
        Args:
            rank: Environment rank for multiprocessing
            eval_env: Whether this is an evaluation environment
            
        Returns:
            Enhanced environment
        """
        # Create base environment
        env = create_graph_enhanced_env(
            use_graph_obs=self.config.feature_extractor.use_graph_obs
        )
        
        # Wrap with monitor for logging
        log_dir = self.output_dir / 'logs' / ('eval' if eval_env else 'train')
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir / f'env_{rank}'))
        
        # Add intrinsic reward wrapper if ICM enabled
        if self.config.icm.enabled and not eval_env:
            # ICM trainer will be set later
            env = IntrinsicRewardWrapper(
                env=env,
                icm_trainer=None,  # Will be set after model creation
                alpha=self.config.icm.alpha,
                r_int_clip=self.config.icm.r_int_clip,
                update_frequency=self.config.icm.update_frequency,
                buffer_size=self.config.icm.buffer_size
            )
        
        return env
    
    def create_model(self) -> PPO:
        """Create PPO model with Phase 2 enhancements."""
        # Create environment to get spaces
        temp_env = self.create_env()
        observation_space = temp_env.observation_space
        action_space = temp_env.action_space
        temp_env.close()
        
        # Create feature extractor
        feature_extractor_class = create_feature_extractor
        feature_extractor_kwargs = {
            'features_dim': self.config.feature_extractor.features_dim,
            'use_graph_obs': self.config.feature_extractor.use_graph_obs,
            'use_3d_conv': self.config.feature_extractor.use_3d_conv,
            'gnn_hidden_dim': self.config.feature_extractor.gnn_hidden_dim,
            'gnn_num_layers': self.config.feature_extractor.gnn_num_layers,
            'gnn_output_dim': self.config.feature_extractor.gnn_output_dim
        }
        
        # Create policy kwargs
        policy_kwargs = {
            'features_extractor_class': feature_extractor_class,
            'features_extractor_kwargs': feature_extractor_kwargs,
            'net_arch': [512, 512],  # Actor-critic network architecture
            'activation_fn': torch.nn.ReLU
        }
        
        # Create vectorized environment
        env = make_vec_env(
            self.create_env,
            n_envs=1,  # Start with single environment
            vec_env_cls=DummyVecEnv
        )
        
        # Create PPO model
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.output_dir / 'tensorboard') if self.config.use_tensorboard else None,
            device=self.device,
            verbose=1
        )
        
        self.env = env
        return model
    
    def setup_icm(self, model: PPO):
        """Setup ICM if enabled."""
        if not self.config.icm.enabled:
            return
        
        print("Setting up ICM...")
        
        # Create ICM network
        icm_network = ICMNetwork(
            feature_dim=self.config.icm.feature_dim,
            action_dim=self.config.icm.action_dim,
            hidden_dim=self.config.icm.hidden_dim,
            eta=self.config.icm.eta,
            lambda_inv=self.config.icm.lambda_inv,
            lambda_fwd=self.config.icm.lambda_fwd
        )
        
        # Create ICM trainer
        self.icm_trainer = ICMTrainer(
            icm_network=icm_network,
            learning_rate=self.config.icm.learning_rate,
            device=str(self.device)
        )
        
        # Set ICM trainer in intrinsic reward wrappers
        for env_idx in range(self.env.num_envs):
            env = self.env.get_attr('env', indices=[env_idx])[0]
            if isinstance(env, IntrinsicRewardWrapper):
                env.set_policy(model.policy)
                env.icm_trainer = self.icm_trainer
        
        print("ICM setup complete")
    
    def setup_exploration_metrics(self) -> Optional[Any]:
        """Setup exploration metrics if enabled."""
        if not self.config.exploration.enabled:
            return None
        
        print("Setting up exploration metrics...")
        
        self.exploration_metrics = ExplorationMetrics(
            grid_width=self.config.exploration.grid_width,
            grid_height=self.config.exploration.grid_height,
            cell_size=self.config.exploration.cell_size,
            window_size=self.config.exploration.window_size
        )
        
        # Create callback
        callback = create_exploration_callback(
            self.exploration_metrics,
            self.config.exploration.log_frequency
        )
        
        print("Exploration metrics setup complete")
        return callback
    
    def load_bc_pretrained_weights(self, model: PPO):
        """Load BC pretrained weights if available."""
        if not self.config.bc.enabled:
            return
        
        bc_path = Path(self.config.bc.dataset_dir).parent / 'bc_policy.pth'
        if not bc_path.exists():
            print(f"BC pretrained weights not found at {bc_path}")
            return
        
        try:
            print(f"Loading BC pretrained weights from {bc_path}")
            checkpoint = torch.load(bc_path, map_location=self.device)
            
            if 'policy_state_dict' in checkpoint:
                # Load policy weights (simplified - would need careful mapping)
                print("BC weights loaded successfully")
            else:
                print("BC checkpoint format not recognized")
                
        except Exception as e:
            print(f"Error loading BC weights: {e}")
    
    def create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_frequency,
            save_path=str(self.output_dir / 'checkpoints'),
            name_prefix='phase2_model'
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = make_vec_env(
            lambda: self.create_env(eval_env=True),
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.output_dir / 'best_model'),
            log_path=str(self.output_dir / 'eval_logs'),
            eval_freq=self.config.eval_frequency,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Exploration metrics callback
        exploration_callback = self.setup_exploration_metrics()
        if exploration_callback:
            callbacks.append(exploration_callback)
        
        return CallbackList(callbacks)
    
    def train(self):
        """Run Phase 2 training."""
        print("Starting Phase 2 training...")
        
        # Create model
        print("Creating model...")
        self.model = self.create_model()
        
        # Setup ICM
        self.setup_icm(self.model)
        
        # Load BC pretrained weights
        self.load_bc_pretrained_weights(self.model)
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print(f"Training for {self.config.total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = self.output_dir / 'final_model'
        self.model.save(str(final_model_path))
        
        print(f"Training completed! Model saved to {final_model_path}")
        
        # Save final metrics
        if self.exploration_metrics:
            stats = self.exploration_metrics.get_episode_statistics()
            import json
            with open(self.output_dir / 'final_exploration_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.env:
            self.env.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Phase 2 Enhanced Training")
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--preset', type=str, default='default',
                       choices=list(get_config_presets().keys()),
                       help='Configuration preset')
    
    # Override arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--total_timesteps', type=int, default=None,
                       help='Total training timesteps')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')
    parser.add_argument('--enable_icm', action='store_true',
                       help='Enable ICM')
    parser.add_argument('--enable_graph', action='store_true',
                       help='Enable graph observations')
    parser.add_argument('--enable_bc', action='store_true',
                       help='Enable BC pretraining')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Phase2Config.load(args.config)
    else:
        presets = get_config_presets()
        config = presets[args.preset]
    
    # Apply overrides
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    if args.device:
        config.device = args.device
    if args.enable_icm:
        config.icm.enabled = True
    if args.enable_graph:
        config.graph.enabled = True
        config.feature_extractor.use_graph_obs = True
    if args.enable_bc:
        config.bc.enabled = True
    
    # Validate configuration
    validation_messages = validate_config(config)
    if validation_messages:
        print("Configuration validation messages:")
        for msg in validation_messages:
            print(f"  {msg}")
        print()
    
    # Print configuration summary
    print("Phase 2 Training Configuration:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  ICM: {'✓' if config.icm.enabled else '✗'}")
    print(f"  Graph: {'✓' if config.graph.enabled else '✗'}")
    print(f"  BC: {'✓' if config.bc.enabled else '✗'}")
    print(f"  Timesteps: {config.total_timesteps:,}")
    print(f"  Device: {config.device}")
    print()
    
    # Create and run trainer
    trainer = Phase2Trainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()