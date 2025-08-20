"""
Behavioral Cloning pretraining script.

This script trains a policy using supervised learning on human replay data
to provide a good initialization for subsequent RL training.
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium.spaces import Discrete, Box, Dict as SpacesDict

from npp_rl.data.bc_dataset import create_bc_dataloader, create_mock_replay_data
from npp_rl.models.feature_extractors import create_feature_extractor
from nclone.nclone_environments.basic_level_no_gold.graph_observation import create_graph_enhanced_env


class BCTrainer:
    """
    Behavioral Cloning trainer for policy pretraining.
    """
    
    def __init__(
        self,
        observation_space: SpacesDict,
        action_space: Discrete,
        policy_class: str = 'npp',
        use_graph_obs: bool = False,
        learning_rate: float = 3e-4,
        entropy_coef: float = 0.01,
        freeze_backbone_steps: int = 0,
        device: str = 'auto'
    ):
        """
        Initialize BC trainer.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            policy_class: Policy architecture ('npp' or 'default')
            use_graph_obs: Whether to use graph observations
            learning_rate: Learning rate
            entropy_coef: Entropy regularization coefficient
            freeze_backbone_steps: Steps to freeze backbone for
            device: Device to train on
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_graph_obs = use_graph_obs
        self.entropy_coef = entropy_coef
        self.freeze_backbone_steps = freeze_backbone_steps
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create policy
        self.policy = self._create_policy(policy_class)
        self.policy.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-6
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_accuracy = 0.0
        
        # Metrics
        self.train_metrics = {
            'loss': [],
            'accuracy': [],
            'entropy': []
        }
    
    def _create_policy(self, policy_class: str) -> nn.Module:
        """Create policy network."""
        if policy_class == 'npp':
            # Create custom policy with multimodal feature extractor
            features_extractor = create_feature_extractor(
                observation_space=self.observation_space,
                features_dim=512,
                use_graph_obs=self.use_graph_obs
            )
            
            # Create policy network
            policy = nn.Sequential(
                features_extractor,
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.action_space.n)
            )
            
        else:
            # Simple MLP policy for testing
            # Flatten observation space for simple policy
            obs_dim = sum(
                np.prod(space.shape) for space in self.observation_space.spaces.values()
                if not (space.dtype == np.int32 and len(space.shape) == 2)  # Skip edge_index
            )
            
            policy = nn.Sequential(
                nn.Flatten(),
                nn.Linear(obs_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.action_space.n)
            )
        
        return policy
    
    def train_epoch(
        self,
        dataloader,
        writer: Optional[SummaryWriter] = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_entropy = 0.0
        num_batches = 0
        
        # Freeze backbone if specified
        if self.step < self.freeze_backbone_steps:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (observations, actions) in enumerate(pbar):
            # Move to device
            observations = {
                key: value.to(self.device) for key, value in observations.items()
            }
            actions = actions.to(self.device)
            
            # Forward pass
            try:
                logits = self.policy(observations)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                print(f"Observation shapes: {[(k, v.shape) for k, v in observations.items()]}")
                continue
            
            # Compute loss
            loss = nn.functional.cross_entropy(logits, actions)
            
            # Add entropy regularization
            if self.entropy_coef > 0:
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                loss = loss - self.entropy_coef * entropy
                epoch_entropy += entropy.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == actions).float().mean()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1
            self.step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy.item():.3f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to tensorboard
            if writer and self.step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), self.step)
                writer.add_scalar('train/accuracy', accuracy.item(), self.step)
                writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.step)
                if self.entropy_coef > 0:
                    writer.add_scalar('train/entropy', entropy.item(), self.step)
        
        # Compute epoch averages
        metrics = {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches,
            'entropy': epoch_entropy / num_batches if self.entropy_coef > 0 else 0.0
        }
        
        self.epoch += 1
        return metrics
    
    def validate(
        self,
        dataloader,
        writer: Optional[SummaryWriter] = None
    ) -> Dict[str, float]:
        """Validate the model."""
        self.policy.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for observations, actions in tqdm(dataloader, desc="Validation"):
                # Move to device
                observations = {
                    key: value.to(self.device) for key, value in observations.items()
                }
                actions = actions.to(self.device)
                
                # Forward pass
                try:
                    logits = self.policy(observations)
                except Exception as e:
                    print(f"Error in validation forward pass: {e}")
                    continue
                
                # Compute loss and accuracy
                loss = nn.functional.cross_entropy(logits, actions)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == actions).float().mean()
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
                num_batches += 1
        
        metrics = {
            'loss': val_loss / num_batches if num_batches > 0 else float('inf'),
            'accuracy': val_accuracy / num_batches if num_batches > 0 else 0.0
        }
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('val/loss', metrics['loss'], self.epoch)
            writer.add_scalar('val/accuracy', metrics['accuracy'], self.epoch)
        
        return metrics
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        if hasattr(self.policy[0], 'player_encoder'):
            for param in self.policy[0].player_encoder.parameters():
                param.requires_grad = False
        if hasattr(self.policy[0], 'global_encoder'):
            for param in self.policy[0].global_encoder.parameters():
                param.requires_grad = False
        if hasattr(self.policy[0], 'graph_encoder'):
            for param in self.policy[0].graph_encoder.parameters():
                param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.policy.parameters():
            param.requires_grad = True
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'train_metrics': self.train_metrics
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = str(Path(path).parent / 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_metrics = checkpoint['train_metrics']
    
    def export_for_sb3(self, path: str):
        """Export policy in SB3-compatible format."""
        # Create a dummy PPO model to get the right structure
        dummy_env = create_graph_enhanced_env(use_graph_obs=self.use_graph_obs)
        
        # Create PPO model with matching architecture
        ppo_model = PPO(
            policy="MultiInputPolicy",
            env=dummy_env,
            policy_kwargs={
                'features_extractor_class': type(self.policy[0]) if hasattr(self.policy, '__getitem__') else None,
                'features_extractor_kwargs': {'features_dim': 512, 'use_graph_obs': self.use_graph_obs}
            }
        )
        
        # Copy weights (this is a simplified version - full implementation would need careful mapping)
        try:
            # Save just the policy state dict for now
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'use_graph_obs': self.use_graph_obs
            }, path)
            
            print(f"Exported BC policy to {path}")
            
        except Exception as e:
            print(f"Error exporting policy: {e}")
            # Fallback: save raw policy
            torch.save(self.policy.state_dict(), path)
        
        dummy_env.close()


def main():
    """Main training function."""
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        return
    
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
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
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
        
        # Train
        train_metrics = trainer.train_epoch(train_dataloader, writer)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.3f}")
        
        # Save checkpoint
        is_best = train_metrics['accuracy'] > trainer.best_accuracy
        if is_best:
            trainer.best_accuracy = train_metrics['accuracy']
        
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
        trainer.save_checkpoint(str(checkpoint_path), is_best=is_best)
        
        # Log epoch metrics
        writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy'], epoch)
    
    # Export final model
    export_path = output_dir / 'bc_policy.pth'
    trainer.export_for_sb3(str(export_path))
    
    # Save training config
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


if __name__ == "__main__":
    main()