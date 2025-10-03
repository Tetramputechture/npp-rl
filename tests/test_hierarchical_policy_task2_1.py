"""
Unit tests for Phase 2 Task 2.1: Two-Level Policy Architecture

Tests the high-level policy, low-level policy, and hierarchical integration
to ensure they work correctly and meet the acceptance criteria.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from npp_rl.hrl.high_level_policy import (
    HighLevelPolicy,
    Subtask,
    SubtaskTransitionManager,
)
from npp_rl.hrl.subtask_policies import (
    LowLevelPolicy,
    SubtaskEmbedding,
    SubtaskContextEncoder,
    ICMIntegration,
    SubtaskSpecificFeatures,
)
from npp_rl.models.hierarchical_policy import (
    HierarchicalPolicyNetwork,
    HierarchicalExperienceBuffer,
)


class MockFeatureExtractor(nn.Module):
    """Mock feature extractor for testing."""
    def __init__(self, features_dim=512):
        super().__init__()
        self.features_dim = features_dim
        self.linear = nn.Linear(10, features_dim)
    
    def forward(self, obs):
        if isinstance(obs, dict):
            obs = obs.get('observation', torch.randn(1, 10))
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != 10:
            obs = torch.randn(obs.shape[0], 10)
        return self.linear(obs)


class TestHighLevelPolicy:
    """Tests for high-level policy network."""
    
    def test_high_level_policy_initialization(self):
        """Test that high-level policy initializes correctly."""
        policy = HighLevelPolicy(
            reachability_dim=8,
            max_switches=5,
            hidden_dim=128,
        )
        
        assert policy.reachability_dim == 8
        assert policy.max_switches == 5
        assert policy.hidden_dim == 128
        assert policy.input_dim == 8 + 5 + 2 + 1  # reachability + switches + pos + time
    
    def test_high_level_policy_forward(self):
        """Test forward pass through high-level policy."""
        policy = HighLevelPolicy()
        batch_size = 4
        
        reachability_features = torch.randn(batch_size, 8)
        switch_states = torch.randint(0, 2, (batch_size, 5)).float()
        ninja_position = torch.randn(batch_size, 2)
        time_remaining = torch.rand(batch_size, 1)
        
        logits = policy(
            reachability_features,
            switch_states,
            ninja_position,
            time_remaining,
        )
        
        assert logits.shape == (batch_size, 4), "Should output 4 subtask logits"
    
    def test_high_level_policy_subtask_selection(self):
        """Test subtask selection with deterministic and stochastic modes."""
        policy = HighLevelPolicy()
        batch_size = 4
        
        reachability_features = torch.randn(batch_size, 8)
        switch_states = torch.randint(0, 2, (batch_size, 5)).float()
        ninja_position = torch.randn(batch_size, 2)
        time_remaining = torch.rand(batch_size, 1)
        
        # Deterministic selection
        subtask_det, log_prob_det = policy.select_subtask(
            reachability_features,
            switch_states,
            ninja_position,
            time_remaining,
            deterministic=True,
        )
        
        assert subtask_det.shape == (batch_size,)
        assert log_prob_det.shape == (batch_size,)
        assert torch.all((subtask_det >= 0) & (subtask_det < 4))
        
        # Stochastic selection
        subtask_sto, log_prob_sto = policy.select_subtask(
            reachability_features,
            switch_states,
            ninja_position,
            time_remaining,
            deterministic=False,
        )
        
        assert subtask_sto.shape == (batch_size,)
        assert log_prob_sto.shape == (batch_size,)
        assert torch.all((subtask_sto >= 0) & (subtask_sto < 4))
    
    def test_heuristic_subtask_selection(self):
        """Test heuristic-based subtask selection logic."""
        policy = HighLevelPolicy()
        
        # Test case 1: Exit switch not activated and reachable
        reachability = np.array([0.5, 0.3, 0.8, 0.2, 0.1, 0.6, 0.4, 0.5])
        switch_states = {'switch_1': False}
        subtask = policy.heuristic_subtask_selection(reachability, switch_states)
        assert subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH
        
        # Test case 2: Exit switch not activated and unreachable
        reachability = np.array([0.5, 0.3, 0.2, 0.1, 0.1, 0.6, 0.4, 0.5])
        switch_states = {'switch_1': False}
        subtask = policy.heuristic_subtask_selection(reachability, switch_states)
        assert subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH
        
        # Test case 3: Exit switch activated and exit reachable
        reachability = np.array([0.5, 0.3, 0.8, 0.8, 0.1, 0.6, 0.4, 0.5])
        switch_states = {'exit_switch': True}
        subtask = policy.heuristic_subtask_selection(reachability, switch_states)
        assert subtask == Subtask.NAVIGATE_TO_EXIT_DOOR
        
        # Test case 4: Exit switch activated but exit unreachable
        reachability = np.array([0.5, 0.3, 0.8, 0.2, 0.1, 0.6, 0.4, 0.5])
        switch_states = {'exit_switch': True}
        subtask = policy.heuristic_subtask_selection(reachability, switch_states)
        assert subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH


class TestSubtaskTransitionManager:
    """Tests for subtask transition management."""
    
    def test_transition_manager_initialization(self):
        """Test transition manager initialization."""
        manager = SubtaskTransitionManager(
            max_steps_per_subtask=500,
            min_steps_between_switches=50,
        )
        
        assert manager.max_steps_per_subtask == 500
        assert manager.min_steps_between_switches == 50
        assert manager.current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH
    
    def test_should_update_subtask(self):
        """Test subtask update conditions."""
        manager = SubtaskTransitionManager(
            max_steps_per_subtask=100,
            min_steps_between_switches=20,
        )
        
        # Initially should not update (cooldown)
        assert not manager.should_update_subtask()
        
        # After cooldown, should update
        for _ in range(20):
            manager.step()
        assert manager.should_update_subtask()
        
        # After max steps, forced update
        manager.subtask_step_count = 100
        assert manager.should_update_subtask()
    
    def test_subtask_completion_detection(self):
        """Test detection of subtask completion."""
        manager = SubtaskTransitionManager()
        
        # Test switch activation detection
        manager.last_switch_states = {'switch_1': False}
        new_switch_states = {'switch_1': True}
        
        completed = manager.detect_subtask_completion(
            Subtask.NAVIGATE_TO_EXIT_SWITCH,
            new_switch_states,
            level_complete=False,
        )
        assert completed, "Should detect switch activation"
        
        # Test level completion
        completed = manager.detect_subtask_completion(
            Subtask.NAVIGATE_TO_EXIT_DOOR,
            new_switch_states,
            level_complete=True,
        )
        assert completed, "Should detect level completion"


class TestLowLevelPolicy:
    """Tests for low-level policy network."""
    
    def test_subtask_embedding_initialization(self):
        """Test subtask embedding initialization."""
        embedding = SubtaskEmbedding(num_subtasks=4, embedding_dim=64)
        
        assert embedding.num_subtasks == 4
        assert embedding.embedding_dim == 64
    
    def test_subtask_embedding_forward(self):
        """Test subtask embedding forward pass."""
        embedding = SubtaskEmbedding(num_subtasks=4, embedding_dim=64)
        batch_size = 4
        
        subtask_indices = torch.randint(0, 4, (batch_size,))
        embed = embedding(subtask_indices)
        
        assert embed.shape == (batch_size, 64)
    
    def test_context_encoder(self):
        """Test subtask context encoder."""
        encoder = SubtaskContextEncoder(context_dim=32)
        batch_size = 4
        
        target_position = torch.randn(batch_size, 2)
        distance_to_target = torch.rand(batch_size, 1)
        mine_proximity = torch.rand(batch_size, 1) * 5
        time_in_subtask = torch.rand(batch_size, 1)
        
        context = encoder(
            target_position,
            distance_to_target,
            mine_proximity,
            time_in_subtask,
        )
        
        assert context.shape == (batch_size, 32)
    
    def test_low_level_policy_initialization(self):
        """Test low-level policy initialization."""
        policy = LowLevelPolicy(
            observation_dim=512,
            subtask_embedding_dim=64,
            context_dim=32,
            num_actions=6,
        )
        
        assert policy.observation_dim == 512
        assert policy.subtask_embedding_dim == 64
        assert policy.context_dim == 32
        assert policy.num_actions == 6
    
    def test_low_level_policy_forward(self):
        """Test forward pass through low-level policy."""
        policy = LowLevelPolicy()
        batch_size = 4
        
        observations = torch.randn(batch_size, 512)
        subtask_indices = torch.randint(0, 4, (batch_size,))
        target_position = torch.randn(batch_size, 2)
        distance_to_target = torch.rand(batch_size, 1)
        mine_proximity = torch.rand(batch_size, 1) * 5
        time_in_subtask = torch.rand(batch_size, 1)
        
        logits = policy(
            observations,
            subtask_indices,
            target_position,
            distance_to_target,
            mine_proximity,
            time_in_subtask,
        )
        
        assert logits.shape == (batch_size, 6), "Should output 6 action logits"
    
    def test_low_level_policy_action_selection(self):
        """Test action selection."""
        policy = LowLevelPolicy()
        batch_size = 4
        
        observations = torch.randn(batch_size, 512)
        subtask_indices = torch.randint(0, 4, (batch_size,))
        target_position = torch.randn(batch_size, 2)
        distance_to_target = torch.rand(batch_size, 1)
        mine_proximity = torch.rand(batch_size, 1) * 5
        time_in_subtask = torch.rand(batch_size, 1)
        
        # Test deterministic selection
        action, log_prob = policy.select_action(
            observations,
            subtask_indices,
            target_position,
            distance_to_target,
            mine_proximity,
            time_in_subtask,
            deterministic=True,
        )
        
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert torch.all((action >= 0) & (action < 6))


class TestICMIntegration:
    """Tests for ICM integration with subtasks."""
    
    def test_icm_integration_initialization(self):
        """Test ICM integration initialization."""
        icm = ICMIntegration(base_curiosity_weight=0.01)
        
        assert icm.base_curiosity_weight == 0.01
        assert len(icm.subtask_modulation) == 4
    
    def test_curiosity_modulation(self):
        """Test curiosity reward modulation."""
        icm = ICMIntegration(base_curiosity_weight=0.01)
        batch_size = 4
        
        base_curiosity = torch.rand(batch_size) * 0.1
        subtask_indices = torch.tensor([0, 1, 2, 3])  # One of each subtask
        mine_proximity = torch.rand(batch_size) * 5
        reachability_score = torch.rand(batch_size)
        
        modulated = icm.modulate_curiosity(
            base_curiosity,
            subtask_indices,
            mine_proximity,
            reachability_score,
        )
        
        assert modulated.shape == (batch_size,)
        assert torch.all(modulated >= 0), "Curiosity should be non-negative"
    
    def test_mine_proximity_reduction(self):
        """Test that curiosity is reduced near dangerous mines."""
        icm = ICMIntegration(base_curiosity_weight=1.0)
        
        base_curiosity = torch.ones(2) * 0.1
        subtask_indices = torch.zeros(2, dtype=torch.long)
        reachability_score = torch.ones(2)
        
        # Far from mine
        mine_proximity_far = torch.tensor([10.0, 10.0])
        modulated_far = icm.modulate_curiosity(
            base_curiosity,
            subtask_indices,
            mine_proximity_far,
            reachability_score,
        )
        
        # Near mine
        mine_proximity_near = torch.tensor([0.5, 0.5])
        modulated_near = icm.modulate_curiosity(
            base_curiosity,
            subtask_indices,
            mine_proximity_near,
            reachability_score,
        )
        
        assert torch.all(modulated_far > modulated_near), \
            "Curiosity should be lower near dangerous mines"


class TestHierarchicalPolicyNetwork:
    """Tests for complete hierarchical policy network."""
    
    def test_hierarchical_network_initialization(self):
        """Test hierarchical network initialization."""
        feature_extractor = MockFeatureExtractor()
        
        network = HierarchicalPolicyNetwork(
            features_extractor=feature_extractor,
            features_dim=512,
            high_level_update_frequency=50,
        )
        
        assert network.features_dim == 512
        assert network.high_level_update_frequency == 50
        assert isinstance(network.high_level_policy, HighLevelPolicy)
        assert isinstance(network.low_level_policy, LowLevelPolicy)
    
    def test_hierarchical_network_forward(self):
        """Test forward pass through hierarchical network."""
        feature_extractor = MockFeatureExtractor()
        network = HierarchicalPolicyNetwork(
            features_extractor=feature_extractor,
            features_dim=512,
        )
        
        batch_size = 4
        obs_dict = {
            'observation': torch.randn(batch_size, 10),
            'reachability_features': torch.randn(batch_size, 8),
            'switch_states': torch.randint(0, 2, (batch_size, 5)).float(),
            'ninja_position': torch.randn(batch_size, 2),
            'time_remaining': torch.rand(batch_size, 1),
        }
        
        actions, values, log_probs, info = network(obs_dict)
        
        assert actions.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert 'current_subtask' in info
        assert 'high_level_updated' in info
    
    def test_hierarchical_network_value_estimation(self):
        """Test value estimation."""
        feature_extractor = MockFeatureExtractor()
        network = HierarchicalPolicyNetwork(
            features_extractor=feature_extractor,
            features_dim=512,
        )
        
        batch_size = 4
        obs_dict = {
            'observation': torch.randn(batch_size, 10),
            'reachability_features': torch.randn(batch_size, 8),
            'switch_states': torch.randint(0, 2, (batch_size, 5)).float(),
            'ninja_position': torch.randn(batch_size, 2),
            'time_remaining': torch.rand(batch_size, 1),
        }
        
        values = network.get_value(obs_dict)
        
        assert values.shape == (batch_size,)
    
    def test_episode_reset(self):
        """Test episode reset functionality."""
        feature_extractor = MockFeatureExtractor()
        network = HierarchicalPolicyNetwork(
            features_extractor=feature_extractor,
            features_dim=512,
        )
        
        # Run a few steps
        batch_size = 2
        obs_dict = {
            'observation': torch.randn(batch_size, 10),
            'reachability_features': torch.randn(batch_size, 8),
            'switch_states': torch.randint(0, 2, (batch_size, 5)).float(),
            'ninja_position': torch.randn(batch_size, 2),
            'time_remaining': torch.rand(batch_size, 1),
        }
        
        for _ in range(10):
            network(obs_dict)
        
        # Reset
        network.reset_episode()
        
        assert network.step_count == 0
        assert network.current_subtask.item() == 0  # NAVIGATE_TO_EXIT_SWITCH


class TestHierarchicalExperienceBuffer:
    """Tests for hierarchical experience buffer."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = HierarchicalExperienceBuffer(
            buffer_size=2048,
            high_level_update_frequency=50,
        )
        
        assert buffer.buffer_size == 2048
        assert buffer.high_level_update_frequency == 50
        assert len(buffer.low_level_buffer['observations']) == 0
        assert len(buffer.high_level_buffer['observations']) == 0
    
    def test_low_level_experience_addition(self):
        """Test adding low-level experiences."""
        buffer = HierarchicalExperienceBuffer()
        
        obs = {'test': torch.randn(4, 10)}
        buffer.add_low_level(obs, 0, 1.0, 0.5, -0.1, 0, False)
        
        assert len(buffer.low_level_buffer['observations']) == 1
        assert buffer.low_level_buffer['actions'][0] == 0
        assert buffer.low_level_buffer['rewards'][0] == 1.0
    
    def test_high_level_experience_addition(self):
        """Test adding high-level experiences."""
        buffer = HierarchicalExperienceBuffer()
        
        # Add some low-level experiences first
        obs = {'test': torch.randn(4, 10)}
        for i in range(5):
            buffer.add_low_level(obs, i, 0.5, 0.3, -0.1, 0, False)
        
        # Add high-level experience
        buffer.add_high_level(obs, 1, 0.4, -0.2, False)
        
        assert len(buffer.high_level_buffer['observations']) == 1
        assert buffer.high_level_buffer['subtasks'][0] == 1
        assert buffer.current_subtask_reward == 0.0  # Reset after addition
    
    def test_buffer_clearing(self):
        """Test buffer clearing."""
        buffer = HierarchicalExperienceBuffer()
        
        obs = {'test': torch.randn(4, 10)}
        buffer.add_low_level(obs, 0, 1.0, 0.5, -0.1, 0, False)
        buffer.add_high_level(obs, 0, 0.4, -0.2, False)
        
        buffer.clear()
        
        assert len(buffer.low_level_buffer['observations']) == 0
        assert len(buffer.high_level_buffer['observations']) == 0
        assert buffer.step_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
