"""
Adaptive Exploration Strategies for N++ RL Agent

This module implements advanced exploration techniques based on recent research
in procedural environments and reinforcement learning.

Key features:
- Curiosity-driven exploration (Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", 2017)
- Adaptive exploration scaling (Inspired by Go-Explore, Ecoffet et al., 2019, and general principles of exploration scheduling)
- Novelty detection (Count-based methods, e.g., Bellemare et al., "Unifying Count-Based Exploration and Intrinsic Motivation", 2016)
- Hierarchical exploration (Conceptual, not fully implemented here but structure allows for future extension)
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional


class CuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for exploration bonus.
    
    Based on "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017).
    Provides intrinsic rewards based on prediction error.
    """
    
    def __init__(self, feature_dim: int = 512, action_dim: int = 5):
        super().__init__()
        
        # Feature encoder (shared)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Forward model: predicts next state features from current features + action
        self.forward_model = nn.Sequential(
            nn.Linear(64 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Inverse model: predicts action from current and next state features
        self.inverse_model = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, state_features: torch.Tensor, next_state_features: torch.Tensor, 
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute curiosity-based intrinsic reward.
        
        Returns:
            intrinsic_reward: Curiosity-based reward
            forward_loss: Forward model loss
            inverse_loss: Inverse model loss
        """
        # Encode features
        phi_state = self.feature_encoder(state_features)
        phi_next_state = self.feature_encoder(next_state_features)
        
        # Forward model prediction
        action_onehot = torch.zeros(actions.size(0), 5, device=actions.device)
        action_onehot.scatter_(1, actions.long().unsqueeze(1), 1)
        
        forward_input = torch.cat([phi_state, action_onehot], dim=1)
        predicted_next_features = self.forward_model(forward_input)
        
        # Inverse model prediction
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        predicted_actions = self.inverse_model(inverse_input)
        
        # Compute losses
        forward_loss = self.mse_loss(predicted_next_features, phi_next_state.detach())
        inverse_loss = self.ce_loss(predicted_actions, actions.long())
        
        # Intrinsic reward is the forward prediction error
        intrinsic_reward = torch.norm(predicted_next_features - phi_next_state.detach(), dim=1)
        
        return intrinsic_reward, forward_loss, inverse_loss


class NoveltyDetector:
    """
    Novelty detection for exploration bonus.
    
    Uses a simple count-based approach with state discretization.
    (Inspired by count-based exploration methods, e.g., Strehl & Littman, 2008; Bellemare et al., 2016)
    """
    
    def __init__(self, grid_size: int = 32, decay_factor: float = 0.99):
        self.grid_size = grid_size
        self.decay_factor = decay_factor
        self.visit_counts = defaultdict(float)
        self.total_visits = 0
        
    def get_novelty_bonus(self, x: float, y: float) -> float:
        """Get novelty bonus based on visit count."""
        # Discretize position
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)
        cell = (grid_x, grid_y)
        
        # Get visit count
        count = self.visit_counts[cell]
        
        # Update visit count
        self.visit_counts[cell] += 1
        self.total_visits += 1
        
        # Novelty bonus inversely proportional to visit count
        novelty_bonus = 1.0 / (1.0 + count)
        
        return novelty_bonus
    
    def decay_counts(self):
        """Decay visit counts to forget old information."""
        for cell in self.visit_counts:
            self.visit_counts[cell] *= self.decay_factor


class AdaptiveExplorationManager:
    """
    Manages adaptive exploration strategies for the N++ agent.
    
    Combines multiple exploration techniques:
    - Curiosity-driven exploration (Pathak et al., 2017)
    - Novelty detection (Count-based methods)
    - Adaptive exploration scaling (Inspired by concepts from Go-Explore, Ecoffet et al., 2019)
    - Progress-based exploration adjustment
    """
    
    def __init__(self, 
                 curiosity_weight: float = 0.1,
                 novelty_weight: float = 0.05,
                 progress_window: int = 100):
        
        self.curiosity_weight = curiosity_weight
        self.novelty_weight = novelty_weight
        self.progress_window = progress_window
        
        # Initialize components
        self.novelty_detector = NoveltyDetector()
        self.curiosity_module = None  # Will be initialized when we have feature dimensions
        
        # Progress tracking
        self.recent_rewards = deque(maxlen=progress_window)
        self.recent_completion_times = deque(maxlen=progress_window)
        self.exploration_scale = 1.0
        
        # Statistics
        self.total_intrinsic_reward = 0.0
        self.episode_count = 0
        
    def initialize_curiosity_module(self, feature_dim: int, action_dim: int = 5):
        """Initialize curiosity module with known dimensions."""
        self.curiosity_module = CuriosityModule(feature_dim, action_dim)
        
    def compute_exploration_bonus(self, 
                                  state_features: Optional[torch.Tensor],
                                  next_state_features: Optional[torch.Tensor],
                                  action: int,
                                  player_x: float,
                                  player_y: float) -> float:
        """
        Compute total exploration bonus combining multiple strategies.
        
        Args:
            state_features: Current state features (for curiosity)
            next_state_features: Next state features (for curiosity)
            action: Action taken
            player_x: Player x position
            player_y: Player y position
            
        Returns:
            Total exploration bonus
        """
        total_bonus = 0.0
        
        # Novelty bonus
        novelty_bonus = self.novelty_detector.get_novelty_bonus(player_x, player_y)
        total_bonus += self.novelty_weight * novelty_bonus * self.exploration_scale
        
        # Curiosity bonus (if available)
        if (self.curiosity_module is not None and 
            state_features is not None and 
            next_state_features is not None):
            
            with torch.no_grad():
                action_tensor = torch.tensor([action], dtype=torch.float32)
                intrinsic_reward, _, _ = self.curiosity_module(
                    state_features.unsqueeze(0),
                    next_state_features.unsqueeze(0),
                    action_tensor
                )
                curiosity_bonus = intrinsic_reward.item()
                total_bonus += self.curiosity_weight * curiosity_bonus * self.exploration_scale
                
                self.total_intrinsic_reward += curiosity_bonus
        
        return total_bonus
    
    def update_progress(self, episode_reward: float, completion_time: Optional[int] = None):
        """Update progress tracking and adjust exploration scale."""
        self.recent_rewards.append(episode_reward)
        if completion_time is not None:
            self.recent_completion_times.append(completion_time)
        
        self.episode_count += 1
        
        # Adjust exploration scale based on progress
        if len(self.recent_rewards) >= self.progress_window:
            # If performance is improving, reduce exploration
            # If performance is stagnating, increase exploration
            recent_avg = np.mean(list(self.recent_rewards)[-50:])
            older_avg = np.mean(list(self.recent_rewards)[-100:-50])
            
            if recent_avg > older_avg * 1.1:  # Significant improvement
                self.exploration_scale *= 0.95  # Reduce exploration
            elif recent_avg < older_avg * 0.9:  # Performance degradation
                self.exploration_scale *= 1.05  # Increase exploration
                
            # Clamp exploration scale
            self.exploration_scale = np.clip(self.exploration_scale, 0.1, 2.0)
    
    def get_exploration_stats(self) -> Dict[str, float]:
        """Get exploration statistics for logging."""
        stats = {
            'exploration_scale': self.exploration_scale,
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'avg_recent_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'episode_count': self.episode_count,
        }
        
        if self.recent_completion_times:
            stats['avg_completion_time'] = np.mean(self.recent_completion_times)
            
        return stats
    
    def reset_episode(self):
        """Reset episode-specific tracking."""
        # Decay novelty counts periodically
        if self.episode_count % 100 == 0:
            self.novelty_detector.decay_counts()
    
    def train_curiosity_module(self, 
                               state_features: torch.Tensor,
                               next_state_features: torch.Tensor,
                               actions: torch.Tensor,
                               optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train the curiosity module.
        
        Returns:
            Dictionary of training losses
        """
        if self.curiosity_module is None:
            return {}
            
        optimizer.zero_grad()
        
        intrinsic_reward, forward_loss, inverse_loss = self.curiosity_module(
            state_features, next_state_features, actions
        )
        
        # Total curiosity loss
        total_loss = forward_loss + inverse_loss
        total_loss.backward()
        optimizer.step()
        
        return {
            'curiosity_forward_loss': forward_loss.item(),
            'curiosity_inverse_loss': inverse_loss.item(),
            'curiosity_total_loss': total_loss.item(),
            'avg_intrinsic_reward': intrinsic_reward.mean().item(),
        } 