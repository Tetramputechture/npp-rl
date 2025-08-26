"""
Behavioral Cloning pretraining script.

This script provides a command-line interface for training policies using
supervised learning on human replay data to provide good initialization
for subsequent reinforcement learning training.
"""

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from npp_rl.data.bc_dataset import create_bc_dataloader, create_mock_replay_data
from npp_rl.training.bc_trainer import BCTrainer
from nclone.nclone_environments.basic_level_no_gold.graph_observation import create_graph_enhanced_env


def parse_arguments():
    """Parse command line arguments for behavioral cloning training."""
    parser = argparse.ArgumentParser(description="Behavioral Cloning Pretraining")
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default='datasets/shards',
                       help='Directory containing replay data')
    parser.add_argument('--create_mock_data', action='store_true',
                       help='Create mock data for testing')
    
    # Model arguments
    parser.add_argument('--policy', type=str, default='npp', choices=['npp', 'simple'],
                       help='Policy architecture')
    parser.add_argument('--use_graph_obs', action='store_true',
                       help='Use graph observations')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy regularization coefficient')
    parser.add_argument('--freeze_backbone_steps', type=int, default=0,
                       help='Steps to freeze backbone for')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='pretrained',
                       help='Output directory for checkpoints')
    parser.add_argument('--run_name', type=str, default='bc_pretrain',
                       help='Run name for logging')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to train on')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Maximum episodes to load')
    
    return parser.parse_args()


def setup_data(args):
    """
    Set up data for training.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (train_dataloader, observation_space, action_space)
    """
    # Create mock data if requested
    if args.create_mock_data:
        print("Creating mock replay data...")
        create_mock_replay_data(
            output_dir=args.dataset_dir,
            num_episodes=100,
            episode_length=200
        )
    
    # Check if dataset exists
    if not Path(args.dataset_dir).exists():
        print(f"Dataset directory {args.dataset_dir} does not exist.")
        print("Use --create_mock_data to create test data.")
        return None, None, None
    
    # Create environment to get observation/action spaces
    env = create_graph_enhanced_env(use_graph_obs=args.use_graph_obs)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    
    print(f"Observation space: {observation_space}")
    print(f"Action space: {action_space}")
    
    # Create dataloaders
    print("Loading dataset...")
    try:
        train_dataloader = create_bc_dataloader(
            data_dir=args.dataset_dir,
            observation_space=observation_space.spaces,
            action_space=action_space,
            batch_size=args.batch_size,
            max_episodes=args.max_episodes,
            use_graph_obs=args.use_graph_obs
        )
        
        # Print dataset statistics
        stats = train_dataloader.dataset.get_statistics()
        print(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        
        return train_dataloader, observation_space, action_space
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None


def run_training(args, train_dataloader, observation_space, action_space):
    """
    Run the behavioral cloning training loop.
    
    Args:
        args: Parsed command line arguments
        train_dataloader: Training data loader
        observation_space: Environment observation space
        action_space: Environment action space
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = BCTrainer(
        observation_space=observation_space,
        action_space=action_space,
        policy_class=args.policy,
        use_graph_obs=args.use_graph_obs,
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef,
        freeze_backbone_steps=args.freeze_backbone_steps,
        device=args.device
    )
    
    # Setup logging
    log_dir = output_dir / 'logs' / args.run_name
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_dataloader, writer)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.3f}")
        
        # Save checkpoint
        is_best = train_metrics['accuracy'] > trainer.best_accuracy
        if is_best:
            trainer.best_accuracy = train_metrics['accuracy']
        
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
        trainer.save_checkpoint(str(checkpoint_path), is_best=is_best)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy'], epoch)
    
    # Export final model for use with Stable-Baselines3
    export_path = output_dir / 'bc_policy.pth'
    trainer.export_for_sb3(str(export_path))
    
    # Save training configuration and results
    config = {
        'args': vars(args),
        'final_accuracy': trainer.best_accuracy,
        'total_steps': trainer.step
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    writer.close()
    print(f"\nTraining completed! Best accuracy: {trainer.best_accuracy:.3f}")
    print(f"Models saved to: {output_dir}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set up data and environment
    train_dataloader, observation_space, action_space = setup_data(args)
    
    if train_dataloader is None:
        return
    
    # Run training
    run_training(args, train_dataloader, observation_space, action_space)


if __name__ == "__main__":
    main()