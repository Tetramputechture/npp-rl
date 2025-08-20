"""
Intrinsic Curiosity Module (ICM) implementation for exploration.

This module implements the ICM from "Curiosity-driven Exploration by Self-supervised Prediction"
(Pathak et al., 2017). It provides intrinsic rewards based on prediction error to encourage
exploration in sparse reward environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ICMNetwork(nn.Module):
    """
    Intrinsic Curiosity Module network with inverse and forward models.
    
    The ICM consists of:
    1. Inverse model: predicts action from state features φ(s_t) and φ(s_{t+1})
    2. Forward model: predicts next state features φ(s_{t+1}) from φ(s_t) and action
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        eta: float = 0.01,
        lambda_inv: float = 0.1,
        lambda_fwd: float = 0.9
    ):
        """
        Initialize ICM network.
        
        Args:
            feature_dim: Dimension of feature representations φ(s)
            action_dim: Number of discrete actions (6 for N++)
            hidden_dim: Hidden layer dimension for ICM networks
            eta: Scaling factor for intrinsic reward
            lambda_inv: Weight for inverse model loss
            lambda_fwd: Weight for forward model loss
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.eta = eta
        self.lambda_inv = lambda_inv
        self.lambda_fwd = lambda_fwd
        
        # Inverse model: φ(s_t), φ(s_{t+1}) -> action distribution
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Forward model: φ(s_t), action -> φ(s_{t+1})
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ICM networks.
        
        Args:
            features_current: Current state features φ(s_t) [batch_size, feature_dim]
            features_next: Next state features φ(s_{t+1}) [batch_size, feature_dim]
            actions: Actions taken [batch_size] (discrete action indices)
            
        Returns:
            Dictionary containing:
                - predicted_actions: Inverse model predictions [batch_size, action_dim]
                - predicted_features: Forward model predictions [batch_size, feature_dim]
                - intrinsic_reward: Intrinsic rewards [batch_size]
                - inverse_loss: Inverse model loss
                - forward_loss: Forward model loss
        """
        batch_size = features_current.shape[0]
        
        # Inverse model: predict action from state features
        inverse_input = torch.cat([features_current, features_next], dim=1)
        predicted_actions = self.inverse_model(inverse_input)
        
        # Forward model: predict next state features
        actions_onehot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        forward_input = torch.cat([features_current, actions_onehot], dim=1)
        predicted_features = self.forward_model(forward_input)
        
        # Compute losses
        inverse_loss = F.cross_entropy(predicted_actions, actions.long())
        forward_loss = F.mse_loss(predicted_features, features_next.detach())
        
        # Compute intrinsic reward as prediction error
        prediction_error = torch.norm(predicted_features - features_next.detach(), dim=1, p=2)
        intrinsic_reward = self.eta * 0.5 * prediction_error.pow(2)
        
        return {
            'predicted_actions': predicted_actions,
            'predicted_features': predicted_features,
            'intrinsic_reward': intrinsic_reward,
            'inverse_loss': inverse_loss,
            'forward_loss': forward_loss
        }
    
    def compute_intrinsic_reward(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intrinsic reward without computing gradients.
        
        Args:
            features_current: Current state features
            features_next: Next state features  
            actions: Actions taken
            
        Returns:
            Intrinsic rewards [batch_size]
        """
        with torch.no_grad():
            actions_onehot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
            forward_input = torch.cat([features_current, actions_onehot], dim=1)
            predicted_features = self.forward_model(forward_input)
            
            prediction_error = torch.norm(predicted_features - features_next, dim=1, p=2)
            intrinsic_reward = self.eta * 0.5 * prediction_error.pow(2)
            
        return intrinsic_reward


class ICMTrainer:
    """
    Trainer for ICM that handles optimization and logging.
    """
    
    def __init__(
        self,
        icm_network: ICMNetwork,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize ICM trainer.
        
        Args:
            icm_network: ICM network to train
            learning_rate: Learning rate for ICM optimizer
            device: Device to run on
        """
        self.icm_network = icm_network.to(device)
        self.optimizer = torch.optim.Adam(self.icm_network.parameters(), lr=learning_rate)
        self.device = device
        
        # Logging
        self.train_stats = {
            'inverse_loss': [],
            'forward_loss': [],
            'total_loss': [],
            'mean_intrinsic_reward': []
        }
    
    def update(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update ICM networks with a batch of experience.
        
        Args:
            features_current: Current state features
            features_next: Next state features
            actions: Actions taken
            
        Returns:
            Dictionary of training statistics
        """
        # Move to device
        features_current = features_current.to(self.device)
        features_next = features_next.to(self.device)
        actions = actions.to(self.device)
        
        # Forward pass
        icm_output = self.icm_network(features_current, features_next, actions)
        
        # Compute total loss
        total_loss = (
            self.icm_network.lambda_inv * icm_output['inverse_loss'] +
            self.icm_network.lambda_fwd * icm_output['forward_loss']
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Log statistics
        stats = {
            'inverse_loss': icm_output['inverse_loss'].item(),
            'forward_loss': icm_output['forward_loss'].item(),
            'total_loss': total_loss.item(),
            'mean_intrinsic_reward': icm_output['intrinsic_reward'].mean().item()
        }
        
        for key, value in stats.items():
            self.train_stats[key].append(value)
            
        return stats
    
    def get_intrinsic_reward(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor
    ) -> np.ndarray:
        """
        Get intrinsic rewards for a batch of transitions.
        
        Args:
            features_current: Current state features
            features_next: Next state features
            actions: Actions taken
            
        Returns:
            Intrinsic rewards as numpy array
        """
        features_current = features_current.to(self.device)
        features_next = features_next.to(self.device)
        actions = actions.to(self.device)
        
        intrinsic_reward = self.icm_network.compute_intrinsic_reward(
            features_current, features_next, actions
        )
        
        return intrinsic_reward.cpu().numpy()
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Get recent training statistics.
        
        Args:
            window: Number of recent updates to average over
            
        Returns:
            Dictionary of averaged statistics
        """
        stats = {}
        for key, values in self.train_stats.items():
            if len(values) > 0:
                recent_values = values[-window:]
                stats[key] = np.mean(recent_values)
            else:
                stats[key] = 0.0
                
        return stats