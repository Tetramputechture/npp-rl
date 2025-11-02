"""
High-Level Policy for Hierarchical RL

This module implements the high-level policy that selects subtasks based on
reachability analysis and game state. The high-level policy makes strategic
decisions about which subtask to pursue next.

Architecture:
- Input: 8D reachability features + switch states + ninja position + time
- Output: 4 discrete subtask selections
- Updates: Every 50-100 steps (configurable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class Subtask(Enum):
    """Enumeration of available subtasks for hierarchical control."""
    NAVIGATE_TO_EXIT_SWITCH = 0
    NAVIGATE_TO_LOCKED_DOOR_SWITCH = 1
    NAVIGATE_TO_EXIT_DOOR = 2
    EXPLORE_FOR_SWITCHES = 3


class HighLevelPolicy(nn.Module):
    """
    High-level policy for strategic subtask selection.
    
    This policy uses reachability features and game state to make strategic
    decisions about which subtask the low-level policy should execute.
    
    Input Features:
    - 8D reachability features (primary decision input)
      [0] Ninja-centric reachability score
      [1] Objective distance metric
      [2] Exit switch accessibility
      [3] Exit door accessibility
      [4] Hazard proximity
      [5] Level connectivity score
      [6] Exploration frontier size
      [7] Path complexity metric
    - Switch state vector (variable size based on level)
    - Ninja position (2D normalized)
    - Time remaining in episode (1D)
    
    Output:
    - Logits over 4 subtask options
    """
    
    def __init__(
        self,
        reachability_dim: int = 8,
        max_switches: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize high-level policy network.
        
        Args:
            reachability_dim: Dimension of reachability features
            max_switches: Maximum number of switches to track
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.reachability_dim = reachability_dim
        self.max_switches = max_switches
        self.hidden_dim = hidden_dim
        
        # Input dimension calculation:
        # reachability (8) + switch states (max_switches) + position (2) + time (1)
        self.input_dim = reachability_dim + max_switches + 2 + 1
        
        # Build network layers
        layers = []
        prev_dim = self.input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Output layer for subtask selection
        self.subtask_head = nn.Linear(hidden_dim, 4)
        
        # Attention mechanism for reachability features
        self.reachability_attention = nn.Sequential(
            nn.Linear(reachability_dim, reachability_dim),
            nn.Tanh(),
            nn.Linear(reachability_dim, reachability_dim),
            nn.Softmax(dim=-1),
        )
    
    def _normalize_switch_states(self, switch_states: torch.Tensor) -> torch.Tensor:
        """
        Normalize switch states tensor to expected dimension.
        
        Args:
            switch_states: Variable-length switch states tensor
            
        Returns:
            Switch states tensor with shape [..., max_switches]
        """
        batch_size = switch_states.shape[0]
        
        if switch_states.shape[-1] < self.max_switches:
            # Pad with zeros if fewer switches
            padding = torch.zeros(
                batch_size, 
                self.max_switches - switch_states.shape[-1],
                device=switch_states.device
            )
            return torch.cat([switch_states, padding], dim=-1)
        elif switch_states.shape[-1] > self.max_switches:
            # Truncate if more switches
            return switch_states[:, :self.max_switches]
        return switch_states
        
    def forward(
        self,
        reachability_features: torch.Tensor,
        switch_states: torch.Tensor,
        ninja_position: torch.Tensor,
        time_remaining: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute subtask logits.
        
        Args:
            reachability_features: [batch_size, 8] reachability features
            switch_states: [batch_size, max_switches] switch activation states
            ninja_position: [batch_size, 2] normalized ninja position
            time_remaining: [batch_size, 1] normalized time remaining
            
        Returns:
            Subtask logits [batch_size, 4]
        """
        # Apply attention to reachability features to focus on important aspects
        attention_weights = self.reachability_attention(reachability_features)
        attended_features = reachability_features * attention_weights
        
        # Normalize switch states dimension
        switch_states = self._normalize_switch_states(switch_states)
        
        # Concatenate all input features
        combined_features = torch.cat([
            attended_features,
            switch_states,
            ninja_position,
            time_remaining,
        ], dim=-1)
        
        # Forward through network
        hidden = self.feature_net(combined_features)
        subtask_logits = self.subtask_head(hidden)
        
        return subtask_logits
    
    def select_subtask(
        self,
        reachability_features: torch.Tensor,
        switch_states: torch.Tensor,
        ninja_position: torch.Tensor,
        time_remaining: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select a subtask given current state.
        
        Args:
            reachability_features: Reachability features
            switch_states: Switch activation states
            ninja_position: Normalized ninja position
            time_remaining: Normalized time remaining
            deterministic: If True, select argmax; otherwise sample
            
        Returns:
            Tuple of (selected_subtask, log_probability)
        """
        logits = self.forward(
            reachability_features, 
            switch_states, 
            ninja_position, 
            time_remaining
        )
        
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            subtask = torch.argmax(probs, dim=-1)
        else:
            subtask = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        log_prob = F.log_softmax(logits, dim=-1)
        selected_log_prob = log_prob.gather(-1, subtask.unsqueeze(-1)).squeeze(-1)
        
        return subtask, selected_log_prob
    
    def heuristic_subtask_selection(
        self,
        reachability_features: np.ndarray,
        switch_states: Dict[str, bool],
    ) -> Subtask:
        """
        Heuristic-based subtask selection (based on completion planner logic).
        
        This method provides a rule-based baseline for subtask selection that
        can be used for initialization or comparison with learned policy.
        
        Args:
            reachability_features: 8D reachability feature array
            switch_states: Dictionary of switch states
            
        Returns:
            Selected subtask
        """
        # Extract relevant features
        exit_switch_reachable = reachability_features[2] > 0.5
        exit_door_reachable = reachability_features[3] > 0.5
        
        # Check if exit switch is activated
        exit_switch_activated = any(
            'exit' in key.lower() and value 
            for key, value in switch_states.items()
        )
        
        # Apply decision logic from task specification
        if not exit_switch_activated:
            if exit_switch_reachable:
                return Subtask.NAVIGATE_TO_EXIT_SWITCH
            else:
                return Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH
        else:
            if exit_door_reachable:
                return Subtask.NAVIGATE_TO_EXIT_DOOR
            else:
                return Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH


class SubtaskTransitionManager:
    """
    Manages transitions between subtasks with timeout and success detection.
    
    This class tracks subtask execution and determines when to transition
    based on completion, timeout, or failure conditions.
    """
    
    def __init__(
        self,
        max_steps_per_subtask: int = 500,
        min_steps_between_switches: int = 50,
    ):
        """
        Initialize subtask transition manager.
        
        Args:
            max_steps_per_subtask: Maximum steps before forced transition
            min_steps_between_switches: Minimum cooldown between switches
        """
        self.max_steps_per_subtask = max_steps_per_subtask
        self.min_steps_between_switches = min_steps_between_switches
        
        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_step_count = 0
        self.steps_since_last_switch = 0
        self.last_switch_states = {}
        
    def should_update_subtask(self) -> bool:
        """
        Check if high-level policy should reselect subtask.
        
        Returns:
            True if subtask should be reconsidered
        """
        # Force update if max steps reached
        if self.subtask_step_count >= self.max_steps_per_subtask:
            return True
        
        # Allow update if minimum cooldown passed
        if self.steps_since_last_switch >= self.min_steps_between_switches:
            return True
        
        return False
    
    def detect_subtask_completion(
        self,
        current_subtask: Subtask,
        switch_states: Dict[str, bool],
        level_complete: bool,
    ) -> bool:
        """
        Detect if current subtask has been completed.
        
        Args:
            current_subtask: Current active subtask
            switch_states: Current switch states
            level_complete: Whether level is complete
            
        Returns:
            True if subtask completed
        """
        # Check for switch state changes
        if switch_states != self.last_switch_states:
            if current_subtask in [
                Subtask.NAVIGATE_TO_EXIT_SWITCH,
                Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH
            ]:
                return True
        
        # Check for level completion
        if level_complete and current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            return True
        
        return False
    
    def update_subtask(self, new_subtask: Subtask):
        """Update to new subtask and reset counters."""
        if new_subtask != self.current_subtask:
            self.current_subtask = new_subtask
            self.subtask_step_count = 0
            self.steps_since_last_switch = 0
        
    def step(self):
        """Increment step counters."""
        self.subtask_step_count += 1
        self.steps_since_last_switch += 1
    
    def update_switch_states(self, switch_states: Dict[str, bool]):
        """Update tracked switch states."""
        self.last_switch_states = switch_states.copy()
    
    def reset(self):
        """Reset for new episode."""
        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_step_count = 0
        self.steps_since_last_switch = 0
        self.last_switch_states = {}
