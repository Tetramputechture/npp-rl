"""Behavioral Cloning Trainer.

Trains policy networks using behavioral cloning from expert demonstrations
(replay data). Can be used standalone or integrated into pretraining pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

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
        frame_stack_config: Optional[Dict] = None,
    ):
        """Initialize BC trainer.
        
        Args:
            architecture_config: Architecture configuration
            dataset: BC replay dataset
            output_dir: Output directory for checkpoints and logs
            device: Device to train on ('auto', 'cpu', 'cuda')
            validation_split: Fraction of data to use for validation
            tensorboard_writer: Optional TensorBoard writer
            frame_stack_config: Frame stacking configuration dict
        """
        self.architecture_config = architecture_config
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_stack_config = frame_stack_config or {}
        
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
        
        # Compute accuracy and other metrics
        with torch.no_grad():
            pred_actions = torch.argmax(logits, dim=1)
            accuracy = (pred_actions == actions).float().mean()
            
            # Compute per-action accuracy for detailed analysis
            action_accuracies = {}
            for action_idx in range(6):  # N++ has 6 actions
                mask = actions == action_idx
                if mask.sum() > 0:
                    action_acc = (pred_actions[mask] == actions[mask]).float().mean()
                    action_accuracies[f'action_{action_idx}_acc'] = action_acc
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            **action_accuracies,
        }
    
    def _log_sample_images(
        self, dataloader: DataLoader, epoch: int, max_samples: int = 10
    ) -> None:
        """Log sample images from player_frame and global_view to TensorBoard.
        
        Args:
            dataloader: Data loader to sample from
            epoch: Current epoch number
            max_samples: Maximum number of samples to log
        """
        if self.tensorboard_writer is None:
            return
        
        self.policy.eval()
        samples_logged = 0
        
        with torch.no_grad():
            for batch in dataloader:
                observations, actions = self._prepare_batch(batch)
                batch_size = actions.shape[0]
                
                # Log player_frame images if available
                if 'player_frame' in observations:
                    frames = observations['player_frame'][:max_samples]
                    # Normalize to [0, 1] for visualization
                    frames_norm = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)
                    
                    # Create grid of images
                    for i in range(min(len(frames), max_samples - samples_logged)):
                        img = frames_norm[i]
                        # Handle different channel formats (H, W, C) or (C, H, W)
                        if img.shape[-1] == 1:  # (H, W, 1)
                            img = img.squeeze(-1).unsqueeze(0)  # (1, H, W)
                        elif img.shape[0] != 1 and img.shape[0] != 3:  # (H, W, C)
                            img = img.permute(2, 0, 1)  # (C, H, W)
                        
                        self.tensorboard_writer.add_image(
                            f'bc_samples/player_frame_{samples_logged}',
                            img,
                            epoch
                        )
                
                # Log global_view images if available
                if 'global_view' in observations:
                    global_views = observations['global_view'][:max_samples]
                    # Normalize to [0, 1] for visualization
                    views_norm = (global_views - global_views.min()) / (global_views.max() - global_views.min() + 1e-8)
                    
                    for i in range(min(len(global_views), max_samples - samples_logged)):
                        img = views_norm[i]
                        # Handle different channel formats
                        if img.shape[-1] == 1:  # (H, W, 1)
                            img = img.squeeze(-1).unsqueeze(0)  # (1, H, W)
                        elif img.shape[0] != 1 and img.shape[0] != 3:  # (H, W, C)
                            img = img.permute(2, 0, 1)  # (C, H, W)
                        
                        self.tensorboard_writer.add_image(
                            f'bc_samples/global_view_{samples_logged}',
                            img,
                            epoch
                        )
                
                samples_logged += batch_size
                if samples_logged >= max_samples:
                    break
        
        self.policy.train()
    
    def _log_action_distributions(
        self, dataloader: DataLoader, epoch: int, dataset_name: str = "train"
    ) -> None:
        """Log action distribution histograms to TensorBoard.
        
        Args:
            dataloader: Data loader to compute distributions from
            epoch: Current epoch number
            dataset_name: Name of the dataset (train/val)
        """
        if self.tensorboard_writer is None:
            return
        
        self.policy.eval()
        expert_actions = []
        predicted_actions = []
        
        with torch.no_grad():
            for batch in dataloader:
                observations, actions = self._prepare_batch(batch)
                
                # Get predictions
                logits = self.policy(observations)
                pred_actions = torch.argmax(logits, dim=1)
                
                expert_actions.append(actions.cpu().numpy())
                predicted_actions.append(pred_actions.cpu().numpy())
        
        expert_actions = np.concatenate(expert_actions)
        predicted_actions = np.concatenate(predicted_actions)
        
        # Log histograms
        self.tensorboard_writer.add_histogram(
            f'bc_actions/{dataset_name}_expert',
            expert_actions,
            epoch
        )
        self.tensorboard_writer.add_histogram(
            f'bc_actions/{dataset_name}_predicted',
            predicted_actions,
            epoch
        )
        
        # Create action distribution comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        action_names = ['NOOP', 'LEFT', 'RIGHT', 'JUMP', 'LEFT+JUMP', 'RIGHT+JUMP']
        expert_counts = np.bincount(expert_actions, minlength=6)
        pred_counts = np.bincount(predicted_actions, minlength=6)
        
        ax1.bar(action_names, expert_counts)
        ax1.set_title('Expert Action Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(action_names, pred_counts)
        ax2.set_title('Predicted Action Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Convert plot to image and log to tensorboard
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        
        self.tensorboard_writer.add_image(
            f'bc_distributions/{dataset_name}_comparison',
            img_array,
            epoch
        )
        
        plt.close(fig)
        self.policy.train()
    
    def _log_observation_statistics(
        self, dataloader: DataLoader, epoch: int
    ) -> None:
        """Log observation statistics to TensorBoard.
        
        Args:
            dataloader: Data loader to compute statistics from
            epoch: Current epoch number
        """
        if self.tensorboard_writer is None:
            return
        
        # Collect observations
        obs_dict = {}
        for batch in dataloader:
            observations, _ = self._prepare_batch(batch)
            for key, value in observations.items():
                if key not in obs_dict:
                    obs_dict[key] = []
                obs_dict[key].append(value.cpu())
            break  # Just use first batch for efficiency
        
        # Log statistics for each observation component
        for key, values in obs_dict.items():
            values_tensor = torch.cat(values, dim=0)
            values_flat = values_tensor.flatten()
            
            # Log histogram
            self.tensorboard_writer.add_histogram(
                f'bc_observations/{key}',
                values_flat,
                epoch
            )
            
            # Log statistics
            self.tensorboard_writer.add_scalar(
                f'bc_obs_stats/{key}_mean',
                values_flat.mean().item(),
                epoch
            )
            self.tensorboard_writer.add_scalar(
                f'bc_obs_stats/{key}_std',
                values_flat.std().item(),
                epoch
            )
    
    def _log_per_action_metrics(
        self, metrics: Dict[str, float], epoch: int, prefix: str = "train"
    ) -> None:
        """Log per-action accuracy metrics to TensorBoard.
        
        Args:
            metrics: Dictionary containing per-action metrics
            epoch: Current epoch number
            prefix: Prefix for metric names (train/val)
        """
        if self.tensorboard_writer is None:
            return
        
        # Log per-action accuracies
        action_names = ['NOOP', 'LEFT', 'RIGHT', 'JUMP', 'LEFT+JUMP', 'RIGHT+JUMP']
        for action_idx in range(6):
            key = f'action_{action_idx}_acc'
            if key in metrics:
                self.tensorboard_writer.add_scalar(
                    f'bc_per_action/{prefix}_{action_names[action_idx]}',
                    metrics[key],
                    epoch
                )
    
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
        
        # Log dataset statistics to TensorBoard
        if self.tensorboard_writer is not None:
            # Log dataset sizes
            self.tensorboard_writer.add_scalar(
                'bc_dataset/train_size',
                len(self.train_dataset),
                0
            )
            self.tensorboard_writer.add_scalar(
                'bc_dataset/val_size',
                len(self.val_dataset),
                0
            )
            
            # Log action distribution from the full dataset if available
            if hasattr(self.train_dataset, 'dataset') and hasattr(self.train_dataset.dataset, 'get_action_distribution'):
                action_dist = self.train_dataset.dataset.get_action_distribution()
                total_samples = sum(action_dist.values())
                action_names = ['NOOP', 'LEFT', 'RIGHT', 'JUMP', 'LEFT+JUMP', 'RIGHT+JUMP']
                
                for action_idx, name in enumerate(action_names):
                    if action_idx in action_dist:
                        count = action_dist[action_idx]
                        percentage = (count / total_samples) * 100
                        self.tensorboard_writer.add_scalar(
                            f'bc_dataset/action_{name}_percentage',
                            percentage,
                            0
                        )
            
            # Log normalization statistics if available
            if hasattr(self.train_dataset, 'dataset') and hasattr(self.train_dataset.dataset, 'normalizer'):
                normalizer = self.train_dataset.dataset.normalizer
                if normalizer.stats:
                    for key, stats in normalizer.stats.items():
                        mean = stats['mean']
                        std = stats['std']
                        # Log mean values (average across all dimensions)
                        if isinstance(mean, np.ndarray):
                            self.tensorboard_writer.add_scalar(
                                f'bc_normalization/{key}_mean_avg',
                                float(mean.mean()),
                                0
                            )
                            self.tensorboard_writer.add_scalar(
                                f'bc_normalization/{key}_std_avg',
                                float(std.mean()),
                                0
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
                # Basic metrics
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
                
                # Per-action metrics
                self._log_per_action_metrics(train_metrics, epoch, prefix="train")
                self._log_per_action_metrics(val_metrics, epoch, prefix="val")
                
                # Log additional visualizations every 5 epochs or on first epoch
                if epoch == 1 or epoch % 5 == 0:
                    logger.info("  Logging visualizations to TensorBoard...")
                    
                    # Sample images
                    self._log_sample_images(val_loader, epoch, max_samples=10)
                    
                    # Action distributions
                    self._log_action_distributions(train_loader, epoch, dataset_name="train")
                    self._log_action_distributions(val_loader, epoch, dataset_name="val")
                    
                    # Observation statistics
                    self._log_observation_statistics(train_loader, epoch)
            
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
                    frame_stack_config=self.frame_stack_config,
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
                    frame_stack_config=self.frame_stack_config,
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
            frame_stack_config=self.frame_stack_config,
        )
        
        logger.info("=" * 60)
        logger.info("BC Training Complete!")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best checkpoint: {self.best_checkpoint_path}")
        logger.info(f"Final checkpoint: {final_checkpoint}")
        logger.info("=" * 60)
        
        return str(self.best_checkpoint_path)
