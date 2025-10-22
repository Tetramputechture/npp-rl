"""Behavioral Cloning Trainer.

Trains policy networks using behavioral cloning from expert demonstrations
(replay data). Can be used standalone or integrated into pretraining pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from npp_rl.training.architecture_configs import ArchitectureConfig
from npp_rl.training.bc_dataset import BCReplayDataset
from npp_rl.training.policy_utils import (
    create_observation_space_from_config,
    create_policy_network,
    save_policy_checkpoint,
    log_model_info,
)

logger = logging.getLogger(__name__)


class BCTrainer:
    """Behavioral Cloning trainer for policy pretraining.
    
    Features:
    - Trains policy networks from expert demonstrations
    - Supports validation split and early stopping
    - TensorBoard logging integration
    - Checkpoint saving in RL-compatible format
    - Action distribution metrics
    """
    
    def __init__(
        self,
        architecture_config: ArchitectureConfig,
        dataset: BCReplayDataset,
        output_dir: str,
        device: str = "auto",
        validation_split: float = 0.1,
        tensorboard_writer: Optional[SummaryWriter] = None,
    ):
        """Initialize BC trainer.
        
        Args:
            architecture_config: Architecture configuration
            dataset: BC replay dataset
            output_dir: Output directory for checkpoints and logs
            device: Device to train on ('auto', 'cpu', 'cuda')
            validation_split: Fraction of data to use for validation
            tensorboard_writer: Optional TensorBoard writer
        """
        self.architecture_config = architecture_config
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Split dataset into train/validation
        self.train_dataset, self.val_dataset = self._split_dataset(validation_split)
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        # Create observation and action spaces
        from gymnasium import spaces
        obs_space = create_observation_space_from_config(architecture_config)
        action_space = spaces.Discrete(6)  # N++ has 6 actions
        
        # Create policy network
        self.policy = create_policy_network(
            observation_space=obs_space,
            action_space=action_space,
            architecture_config=architecture_config,
            features_dim=512,
            net_arch=[256, 256],
        )
        
        self.policy = self.policy.to(self.device)
        
        # Log model info
        log_model_info(self.policy, "BC Policy")
        
        # TensorBoard writer
        self.tensorboard_writer = tensorboard_writer
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None
        self.global_step = 0
    
    def _split_dataset(
        self, validation_split: float
    ) -> Tuple[BCReplayDataset, BCReplayDataset]:
        """Split dataset into training and validation sets.
        
        Args:
            validation_split: Fraction for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        total_size = len(self.dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        
        return train_dataset, val_dataset
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[Dict, torch.Tensor]:
        """Prepare batch for training by moving to device.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Tuple of (observations, actions) on device
        """
        observations, actions = batch
        
        # Move observations to device
        obs_device = {}
        for key, value in observations.items():
            obs_device[key] = value.to(self.device)
        
        # Move actions to device
        actions_device = actions.to(self.device)
        
        return obs_device, actions_device
    
    def compute_bc_loss(
        self, observations: Dict, actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute behavioral cloning loss.
        
        BC loss is the negative log-likelihood of expert actions under the policy.
        
        Args:
            observations: Batch of observations
            actions: Batch of expert actions
            
        Returns:
            Dictionary containing loss and metrics
        """
        # Forward pass through policy
        logits = self.policy(observations)
        
        # Compute cross-entropy loss (negative log-likelihood)
        loss = F.cross_entropy(logits, actions)
        
        # Compute accuracy
        with torch.no_grad():
            pred_actions = torch.argmax(logits, dim=1)
            accuracy = (pred_actions == actions).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
        }
    
    def train_epoch(
        self, epoch: int, dataloader: DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            observations, actions = self._prepare_batch(batch)
            
            # Compute loss
            metrics = self.compute_bc_loss(observations, actions)
            loss = metrics['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_accuracy += metrics['accuracy'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': metrics['accuracy'].item(),
            })
            
            # TensorBoard logging
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar(
                    'bc_train/loss', loss.item(), self.global_step
                )
                self.tensorboard_writer.add_scalar(
                    'bc_train/accuracy', metrics['accuracy'].item(), self.global_step
                )
            
            self.global_step += 1
        
        # Compute epoch averages
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on validation set.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.policy.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            observations, actions = self._prepare_batch(batch)
            
            # Compute loss
            metrics = self.compute_bc_loss(observations, actions)
            
            # Update metrics
            val_loss += metrics['loss'].item()
            val_accuracy += metrics['accuracy'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': metrics['loss'].item(),
                'acc': metrics['accuracy'].item(),
            })
        
        # Compute averages
        avg_loss = val_loss / num_batches
        avg_accuracy = val_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
        }
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        num_workers: int = 4,
        save_frequency: int = 5,
        early_stopping_patience: int = 5,
    ) -> str:
        """Run full training loop.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            num_workers: Number of data loading workers
            save_frequency: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Path to best checkpoint
        """
        logger.info("=" * 60)
        logger.info("Starting Behavioral Cloning Training")
        logger.info(f"Architecture: {self.architecture_config.name}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        logger.info("=" * 60)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        
        # Learning rate scheduler
        # Note: verbose parameter removed in PyTorch 2.0+
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        # Training loop
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch, train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Log results
            logger.info(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s):")
            logger.info(
                f"  Train - Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics['accuracy']:.4f}"
            )
            logger.info(
                f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f}"
            )
            
            # TensorBoard logging
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar(
                    'bc_epoch/train_loss', train_metrics['loss'], epoch
                )
                self.tensorboard_writer.add_scalar(
                    'bc_epoch/train_accuracy', train_metrics['accuracy'], epoch
                )
                self.tensorboard_writer.add_scalar(
                    'bc_epoch/val_loss', val_metrics['loss'], epoch
                )
                self.tensorboard_writer.add_scalar(
                    'bc_epoch/val_accuracy', val_metrics['accuracy'], epoch
                )
                self.tensorboard_writer.add_scalar(
                    'bc_epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch
                )
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Save checkpoint
            if epoch % save_frequency == 0:
                checkpoint_path = self.output_dir / f"bc_checkpoint_epoch_{epoch}.pth"
                save_policy_checkpoint(
                    self.policy,
                    str(checkpoint_path),
                    epoch=epoch,
                    metrics=val_metrics,
                    architecture_config=self.architecture_config,
                )
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_checkpoint_path = self.output_dir / "bc_best.pth"
                
                save_policy_checkpoint(
                    self.policy,
                    str(self.best_checkpoint_path),
                    epoch=epoch,
                    metrics=val_metrics,
                    architecture_config=self.architecture_config,
                )
                
                logger.info(f"  ✓ New best model! Val loss: {val_metrics['loss']:.4f}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"\nEarly stopping after {epoch} epochs "
                    f"(no improvement for {early_stopping_patience} epochs)"
                )
                break
        
        total_time = time.time() - start_time
        
        # Save final checkpoint
        final_checkpoint = self.output_dir / "bc_checkpoint.pth"
        save_policy_checkpoint(
            self.policy,
            str(final_checkpoint),
            epoch=epoch,
            metrics=val_metrics,
            architecture_config=self.architecture_config,
        )
        
        logger.info("=" * 60)
        logger.info("BC Training Complete!")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best checkpoint: {self.best_checkpoint_path}")
        logger.info(f"Final checkpoint: {final_checkpoint}")
        logger.info("=" * 60)
        
        return str(self.best_checkpoint_path)
