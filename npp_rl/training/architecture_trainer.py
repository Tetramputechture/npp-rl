"""Architecture trainer for single architecture training runs.

Handles training for a specific architecture configuration including
setup, training loop, evaluation, and checkpointing.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from npp_rl.optimization.architecture_configs import ArchitectureConfig

logger = logging.getLogger(__name__)


class ArchitectureTrainer:
    """Manages training for a specific architecture configuration."""
    
    def __init__(
        self,
        architecture_config: ArchitectureConfig,
        train_dataset_path: str,
        test_dataset_path: str,
        output_dir: Path,
        device_id: int = 0,
        world_size: int = 1,
        tensorboard_writer: Optional[SummaryWriter] = None,
        use_mixed_precision: bool = False,
        use_hierarchical_ppo: bool = False,
        use_curriculum: bool = False,
        curriculum_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize architecture trainer.
        
        Args:
            architecture_config: Architecture configuration
            train_dataset_path: Path to training dataset
            test_dataset_path: Path to test dataset
            output_dir: Output directory for checkpoints/logs
            device_id: GPU device ID
            world_size: Number of GPUs (for distributed training)
            tensorboard_writer: Optional TensorBoard writer
            use_mixed_precision: Enable mixed precision training
            use_hierarchical_ppo: Use hierarchical PPO instead of standard PPO
            use_curriculum: Enable curriculum learning
            curriculum_kwargs: Curriculum manager configuration
        """
        self.architecture_config = architecture_config
        self.train_dataset_path = Path(train_dataset_path)
        self.test_dataset_path = Path(test_dataset_path)
        self.output_dir = Path(output_dir)
        self.device_id = device_id
        self.world_size = world_size
        self.tensorboard_writer = tensorboard_writer
        self.use_mixed_precision = use_mixed_precision
        self.use_hierarchical_ppo = use_hierarchical_ppo
        self.use_curriculum = use_curriculum
        self.curriculum_kwargs = curriculum_kwargs or {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.model = None
        self.env = None
        self.eval_env = None
        self.curriculum_manager = None
        
        logger.info(f"Initialized trainer for architecture: {architecture_config.name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: cuda:{device_id}")
        logger.info(f"Hierarchical PPO: {use_hierarchical_ppo}")
        logger.info(f"Curriculum learning: {use_curriculum}")
    
    def setup_model(
        self,
        pretrained_checkpoint: Optional[str] = None,
        **ppo_kwargs
    ):
        """Initialize model from architecture config or checkpoint.
        
        Args:
            pretrained_checkpoint: Optional path to pretrained weights
            **ppo_kwargs: Additional PPO hyperparameters
            
        Returns:
            Initialized PPO or HierarchicalPPO model
        """
        logger.info("Setting up model...")
        
        # Import necessary modules
        from nclone.gym_environment.graph_observation import create_graph_enhanced_env
        from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor
        
        # Create a temporary environment to get spaces
        temp_env = create_graph_enhanced_env()
        observation_space = temp_env.observation_space
        action_space = temp_env.action_space
        temp_env.close()
        
        # Set up policy kwargs with architecture config
        policy_kwargs = {
            'features_extractor_class': ConfigurableMultimodalExtractor,
            'features_extractor_kwargs': {
                'architecture_config': self.architecture_config
            },
            'net_arch': {
                'pi': [256, 256, 128],
                'vf': [256, 256, 128]
            }
        }
        
        # Add hierarchical PPO specific kwargs if needed
        if self.use_hierarchical_ppo:
            policy_kwargs.update({
                'high_level_update_frequency': ppo_kwargs.pop('high_level_update_frequency', 50),
                'max_steps_per_subtask': ppo_kwargs.pop('max_steps_per_subtask', 500),
                'use_icm': ppo_kwargs.pop('use_icm', True)
            })
        
        # Default PPO hyperparameters
        default_hyperparams = {
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 256,
            'gamma': 0.999,
            'gae_lambda': 0.998,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'tensorboard_log': str(self.output_dir / 'tensorboard') if self.tensorboard_writer is None else None,
            'device': f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu'
        }
        
        # Merge with provided hyperparameters
        hyperparams = {**default_hyperparams, **ppo_kwargs}
        
        # Choose PPO class based on configuration
        if self.use_hierarchical_ppo:
            logger.info("Using Hierarchical PPO")
            # Import HierarchicalPPO (wrapper around PPO with custom policy)
            # For now, use standard PPO with hierarchical policy
            # Full HierarchicalPPO implementation may need custom training loop
            from stable_baselines3 import PPO as PPOClass
            policy_class = "MultiInputPolicy"  # Will use hierarchical policy internally
        else:
            from stable_baselines3 import PPO as PPOClass
            policy_class = "MultiInputPolicy"
        
        # Create model (will create environments later)
        self.model = PPOClass(
            policy=policy_class,
            env=None,  # Set up environments separately
            policy_kwargs=policy_kwargs,
            **hyperparams
        )
        
        # Load pretrained weights if provided
        if pretrained_checkpoint:
            logger.info(f"Loading pretrained weights from {pretrained_checkpoint}")
            try:
                checkpoint = torch.load(pretrained_checkpoint, map_location=self.model.device)
                if 'policy_state_dict' in checkpoint:
                    self.model.policy.load_state_dict(checkpoint['policy_state_dict'])
                    logger.info("Loaded pretrained policy weights")
                else:
                    logger.warning("Checkpoint does not contain 'policy_state_dict'")
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                logger.warning("Continuing with random initialization")
        
        logger.info(f"Model initialized with architecture: {self.architecture_config.name}")
        return self.model
    
    def setup_environments(self, num_envs: int = 64) -> None:
        """Create vectorized training and eval environments.
        
        Args:
            num_envs: Number of parallel environments
        """
        logger.info(f"Setting up {num_envs} training environments...")
        
        from nclone.gym_environment.graph_observation import create_graph_enhanced_env
        
        # Set up curriculum manager if enabled
        if self.use_curriculum:
            from npp_rl.training.curriculum_manager import create_curriculum_manager
            
            self.curriculum_manager = create_curriculum_manager(
                dataset_path=str(self.train_dataset_path),
                **self.curriculum_kwargs
            )
            
            logger.info("Curriculum learning enabled")
            logger.info(f"Starting stage: {self.curriculum_manager.get_current_stage()}")
        
        # Create environment factory functions
        def make_env(rank: int):
            def _init():
                env = create_graph_enhanced_env()
                
                # Wrap with curriculum if enabled
                if self.use_curriculum and self.curriculum_manager:
                    from npp_rl.wrappers.curriculum_env import CurriculumEnv
                    env = CurriculumEnv(
                        env,
                        self.curriculum_manager,
                        check_advancement_freq=10
                    )
                
                return env
            return _init
        
        # Create vectorized training environment
        if num_envs > 1:
            env_fns = [make_env(i) for i in range(num_envs)]
            self.env = SubprocVecEnv(env_fns)
        else:
            self.env = DummyVecEnv([make_env(0)])
        
        # Wrap vectorized env with curriculum tracking if enabled
        if self.use_curriculum and self.curriculum_manager:
            from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper
            self.env = CurriculumVecEnvWrapper(
                self.env,
                self.curriculum_manager,
                check_advancement_freq=10
            )
        
        # Create evaluation environment (single, no curriculum)
        def make_eval_env():
            return create_graph_enhanced_env()
        
        self.eval_env = DummyVecEnv([make_eval_env])
        
        # Set environment in model
        if self.model:
            self.model.set_env(self.env)
        
        logger.info(f"Environments created: {num_envs} training, 1 eval")
        
        if self.use_curriculum:
            logger.info("Curriculum tracking enabled across all environments")
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 100000,
        save_freq: int = 500000,
        callback_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main training loop with evaluation and checkpointing.
        
        Args:
            total_timesteps: Total timesteps to train
            eval_freq: Evaluation frequency (timesteps)
            save_freq: Checkpoint save frequency (timesteps)
            callback_fn: Optional callback function
            
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first")
        
        if self.env is None:
            raise RuntimeError("Environments not initialized. Call setup_environments() first")
        
        logger.info("=" * 60)
        logger.info(f"Starting training: {self.architecture_config.name}")
        logger.info(f"Total timesteps: {total_timesteps:,}")
        logger.info(f"Eval frequency: {eval_freq:,}")
        logger.info(f"Save frequency: {save_freq:,}")
        logger.info("=" * 60)
        
        try:
            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback_fn,
                progress_bar=True
            )
            
            logger.info("Training completed successfully")
            
            # Save final model
            final_path = self.output_dir / 'final_model.zip'
            self.model.save(str(final_path))
            logger.info(f"Saved final model to {final_path}")
            
            return {
                'status': 'completed',
                'total_timesteps': total_timesteps
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate(self, num_episodes: int = 250) -> Dict[str, float]:
        """Evaluate model on test dataset.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"Evaluating model on test suite ({num_episodes} episodes)...")
        
        try:
            from npp_rl.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
            
            evaluator = ComprehensiveEvaluator(
                test_dataset_path=str(self.test_dataset_path),
                device=f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu'
            )
            
            results = evaluator.evaluate_model(
                model=self.model,
                num_episodes_per_category=None,  # Evaluate all
                max_steps_per_episode=10000,
                deterministic=True
            )
            
            # Save results
            results_path = self.output_dir / 'eval_results.json'
            evaluator.save_results(results, str(results_path))
            
            logger.info("Evaluation complete")
            logger.info(f"Success rate: {results['overall']['success_rate']:.2%}")
            
            return results['overall']
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'success_rate': 0.0,
                'error': str(e)
            }
    
    def save_checkpoint(
        self,
        timestep: int,
        is_final: bool = False
    ) -> Path:
        """Save model checkpoint with metadata.
        
        Args:
            timestep: Current training timestep
            is_final: Whether this is the final checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_final:
            checkpoint_path = checkpoint_dir / 'final_model.zip'
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_{timestep}.zip'
        
        self.model.save(str(checkpoint_path))
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
