"""
Unit tests for mine-aware curiosity modulation.

Tests cover:
- MineAwareCuriosityModulator behavior
- Curiosity modulation based on mine proximity
- Batch modulation
- Exploration bias calculation
"""

import pytest
import numpy as np

from npp_rl.intrinsic.mine_aware_curiosity import (
    MineAwareCuriosityModulator,
    modulate_curiosity_for_mines,
)


class TestMineAwareCuriosityModulator:
    """Test cases for MineAwareCuriosityModulator."""
    
    def test_modulator_initialization(self):
        """Test basic initialization."""
        modulator = MineAwareCuriosityModulator()
        
        assert modulator.min_modulation == 0.1
        assert modulator.max_modulation == 1.0
        assert modulator.safe_boost == 1.2
    
    def test_modulate_curiosity_danger_zone(self):
        """Test curiosity modulation in danger zone."""
        modulator = MineAwareCuriosityModulator()
        
        base_curiosity = 1.0
        mine_proximity = 24.0  # Very close (< DANGER_THRESHOLD)
        
        modulated = modulator.modulate_curiosity(base_curiosity, mine_proximity)
        
        # Should be heavily reduced in danger zone
        assert modulated < 0.2 * base_curiosity
        assert modulated >= modulator.min_modulation * base_curiosity
    
    def test_modulate_curiosity_safe_zone(self):
        """Test curiosity modulation in safe zone."""
        modulator = MineAwareCuriosityModulator()
        
        base_curiosity = 1.0
        mine_proximity = 200.0  # Far away (> SAFE_THRESHOLD)
        
        modulated = modulator.modulate_curiosity(base_curiosity, mine_proximity)
        
        # Should be boosted in safe zone
        assert modulated >= base_curiosity
        assert modulated <= modulator.safe_boost * base_curiosity
    
    def test_modulate_curiosity_transition_zone(self):
        """Test curiosity modulation in transition zone."""
        modulator = MineAwareCuriosityModulator()
        
        base_curiosity = 1.0
        mine_proximity = 72.0  # Between DANGER and SAFE thresholds
        
        modulated = modulator.modulate_curiosity(base_curiosity, mine_proximity)
        
        # Should be moderately modulated in transition zone
        assert modulated > modulator.min_modulation * base_curiosity
        assert modulated < modulator.safe_boost * base_curiosity
    
    def test_modulate_curiosity_unsafe_path(self):
        """Test curiosity modulation with unsafe path."""
        modulator = MineAwareCuriosityModulator()
        
        base_curiosity = 1.0
        mine_proximity = 200.0  # Far but path is unsafe
        
        modulated = modulator.modulate_curiosity(
            base_curiosity, 
            mine_proximity, 
            is_path_safe=False
        )
        
        # Should be reduced even if mine is far when path is unsafe
        assert modulated < base_curiosity
    
    def test_modulate_curiosity_batch(self):
        """Test batch curiosity modulation."""
        modulator = MineAwareCuriosityModulator()
        
        base_curiosity = np.array([1.0, 1.0, 1.0])
        mine_proximities = np.array([24.0, 72.0, 200.0])  # danger, transition, safe
        
        modulated = modulator.modulate_curiosity_batch(base_curiosity, mine_proximities)
        
        assert modulated.shape == base_curiosity.shape
        
        # Danger zone should be most reduced
        assert modulated[0] < modulated[1] < modulated[2]
    
    def test_calculate_exploration_bias(self):
        """Test exploration bias calculation."""
        modulator = MineAwareCuriosityModulator()
        
        ninja_pos = (100.0, 100.0)
        target_pos = (200.0, 100.0)  # East of ninja
        
        # Mine between ninja and target
        mine_positions = [(150.0, 100.0)]
        mine_states = [0]  # Toggled (dangerous)
        
        bias = modulator.calculate_exploration_bias(
            ninja_pos,
            target_pos,
            mine_positions,
            mine_states
        )
        
        # Bias should be reduced due to mine in path
        assert 0.2 <= bias < 1.0
    
    def test_calculate_exploration_bias_safe_mine(self):
        """Test exploration bias with safe mine."""
        modulator = MineAwareCuriosityModulator()
        
        ninja_pos = (100.0, 100.0)
        target_pos = (200.0, 100.0)
        
        # Safe mine between ninja and target
        mine_positions = [(150.0, 100.0)]
        mine_states = [1]  # Untoggled (safe)
        
        bias = modulator.calculate_exploration_bias(
            ninja_pos,
            target_pos,
            mine_positions,
            mine_states
        )
        
        # Bias should not be reduced for safe mine
        assert bias == 1.0
    
    def test_get_safe_exploration_zones(self):
        """Test safe exploration zone generation."""
        modulator = MineAwareCuriosityModulator()
        
        ninja_pos = (100.0, 100.0)
        
        # Dangerous mine to the east
        mine_positions = [(200.0, 100.0)]
        mine_states = [0]  # Toggled
        
        safety_scores = modulator.get_safe_exploration_zones(
            ninja_pos,
            mine_positions,
            mine_states,
            grid_resolution=8
        )
        
        assert len(safety_scores) == 8
        assert all(0.0 <= score <= 1.0 for score in safety_scores)
        
        # The direction toward the mine should have lower safety
        # (East is at index 0 for angle 0)
        assert safety_scores[0] < max(safety_scores)
    
    def test_get_statistics(self):
        """Test statistics collection."""
        modulator = MineAwareCuriosityModulator()
        
        # Perform some modulations
        modulator.modulate_curiosity(1.0, 24.0)  # Danger zone
        modulator.modulate_curiosity(1.0, 200.0)  # Safe zone
        modulator.modulate_curiosity(1.0, 24.0)  # Danger zone
        
        stats = modulator.get_statistics()
        
        assert stats['total_modulations'] == 3
        assert stats['danger_zone_count'] == 2
        assert stats['safe_zone_count'] == 1
        assert 'average_modulation' in stats
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        modulator = MineAwareCuriosityModulator()
        
        # Perform modulations
        modulator.modulate_curiosity(1.0, 24.0)
        
        # Reset
        modulator.reset_statistics()
        
        stats = modulator.get_statistics()
        assert stats['total_modulations'] == 0


class TestSimpleCuriosityModulation:
    """Test cases for simple utility function."""
    
    def test_modulate_curiosity_for_mines_danger(self):
        """Test simple modulation in danger zone."""
        curiosity = 1.0
        mine_proximity = 24.0
        
        modulated = modulate_curiosity_for_mines(curiosity, mine_proximity)
        
        assert modulated == pytest.approx(0.1)
    
    def test_modulate_curiosity_for_mines_safe(self):
        """Test simple modulation in safe zone."""
        curiosity = 1.0
        mine_proximity = 200.0
        
        modulated = modulate_curiosity_for_mines(curiosity, mine_proximity)
        
        assert modulated == pytest.approx(1.0)
    
    def test_modulate_curiosity_for_mines_transition(self):
        """Test simple modulation in transition zone."""
        curiosity = 1.0
        mine_proximity = 72.0  # Midpoint between 48 and 96
        
        modulated = modulate_curiosity_for_mines(curiosity, mine_proximity)
        
        # Should be around 0.55 (0.1 + 0.9 * 0.5)
        assert 0.4 < modulated < 0.7
    
    def test_modulate_curiosity_for_mines_custom_thresholds(self):
        """Test simple modulation with custom thresholds."""
        curiosity = 1.0
        mine_proximity = 30.0
        
        modulated = modulate_curiosity_for_mines(
            curiosity,
            mine_proximity,
            danger_threshold=20.0,
            safe_threshold=40.0
        )
        
        # Should be in transition zone with custom thresholds
        assert 0.1 < modulated < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
