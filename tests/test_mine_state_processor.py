"""
Unit tests for mine state processing functionality.

Tests cover:
- MineState class behavior
- MineStateProcessor tracking and queries
- Mine proximity calculations
- Path safety checks
- Integration with observation processor
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Import from nclone
import sys
sys.path.insert(0, '/workspace/nclone')
from nclone.gym_environment.mine_state_processor import (
    MineState,
    MineStateProcessor,
)
from nclone.constants.physics_constants import (
    TOGGLE_MINE_RADIUS_TOGGLED,
    TOGGLE_MINE_RADIUS_UNTOGGLED,
    TOGGLE_MINE_RADIUS_TOGGLING,
)


class TestMineState:
    """Test cases for MineState class."""
    
    def test_mine_state_initialization(self):
        """Test basic mine state initialization."""
        mine = MineState(
            mine_id=1,
            position=(100.0, 200.0),
            state=MineState.UNTOGGLED
        )
        
        assert mine.mine_id == 1
        assert mine.position == (100.0, 200.0)
        assert mine.state == MineState.UNTOGGLED
        assert mine.is_safe
        assert not mine.is_dangerous
    
    def test_mine_state_radii(self):
        """Test that mine radii match physics constants."""
        # Untoggled mine
        mine_untoggled = MineState(1, (0, 0), MineState.UNTOGGLED)
        assert mine_untoggled.radius == TOGGLE_MINE_RADIUS_UNTOGGLED
        
        # Toggled mine
        mine_toggled = MineState(2, (0, 0), MineState.TOGGLED)
        assert mine_toggled.radius == TOGGLE_MINE_RADIUS_TOGGLED
        
        # Toggling mine
        mine_toggling = MineState(3, (0, 0), MineState.TOGGLING)
        assert mine_toggling.radius == TOGGLE_MINE_RADIUS_TOGGLING
    
    def test_mine_danger_status(self):
        """Test mine danger status based on state."""
        # Safe state
        mine_safe = MineState(1, (0, 0), MineState.UNTOGGLED)
        assert mine_safe.is_safe
        assert not mine_safe.is_dangerous
        
        # Dangerous states
        mine_toggled = MineState(2, (0, 0), MineState.TOGGLED)
        assert not mine_toggled.is_safe
        assert mine_toggled.is_dangerous
        
        mine_toggling = MineState(3, (0, 0), MineState.TOGGLING)
        assert not mine_toggling.is_safe
        assert mine_toggling.is_dangerous
    
    def test_mine_distance_calculation(self):
        """Test distance calculation to mine."""
        mine = MineState(1, (100.0, 100.0), MineState.UNTOGGLED)
        
        # Distance to self
        assert mine.distance_to((100.0, 100.0)) == pytest.approx(0.0)
        
        # Distance to nearby point (3-4-5 triangle)
        assert mine.distance_to((103.0, 104.0)) == pytest.approx(5.0, rel=1e-6)
        
        # Distance to far point
        assert mine.distance_to((200.0, 200.0)) == pytest.approx(141.42, rel=1e-2)
    
    def test_mine_path_blocking(self):
        """Test path blocking detection."""
        # Mine in the middle of a path
        mine = MineState(1, (100.0, 100.0), MineState.TOGGLED)
        
        # Path that goes through mine
        assert mine.is_blocking_path((50.0, 100.0), (150.0, 100.0))
        
        # Path that avoids mine
        assert not mine.is_blocking_path((50.0, 150.0), (150.0, 150.0))
        
        # Safe mine doesn't block
        safe_mine = MineState(2, (100.0, 100.0), MineState.UNTOGGLED)
        assert not safe_mine.is_blocking_path((50.0, 100.0), (150.0, 100.0))
    
    def test_mine_state_update(self):
        """Test mine state updates."""
        mine = MineState(1, (100.0, 100.0), MineState.UNTOGGLED)
        
        # Update to toggled
        mine.update_state(MineState.TOGGLED)
        assert mine.state == MineState.TOGGLED
        assert mine.is_dangerous
        
        # Time in state should reset
        assert mine.time_in_state >= 0.0
    
    def test_mine_to_vector(self):
        """Test conversion to feature vector."""
        mine = MineState(1, (100.0, 200.0), MineState.TOGGLED)
        ninja_pos = (150.0, 200.0)
        
        vector = mine.to_vector(ninja_pos)
        
        assert len(vector) == 7
        assert vector[0] == 100.0  # x position
        assert vector[1] == 200.0  # y position
        assert vector[2] == MineState.TOGGLED  # state
        assert vector[3] == TOGGLE_MINE_RADIUS_TOGGLED  # radius
        assert vector[5] == pytest.approx(50.0)  # distance


class TestMineStateProcessor:
    """Test cases for MineStateProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = MineStateProcessor()
        
        assert len(processor.mine_states) == 0
        assert processor.safety_radius == 2.0
    
    def test_update_mine_states(self):
        """Test updating mine states from entities."""
        processor = MineStateProcessor()
        
        # Create mock entities
        entity1 = Mock()
        entity1.entity_type = 1  # TOGGLE_MINE
        entity1.x = 100.0
        entity1.y = 200.0
        entity1.state = MineState.TOGGLED
        entity1.id = 'mine1'
        
        entity2 = Mock()
        entity2.entity_type = 1
        entity2.x = 300.0
        entity2.y = 400.0
        entity2.state = MineState.UNTOGGLED
        entity2.id = 'mine2'
        
        entities = [entity1, entity2]
        
        processor.update_mine_states(entities)
        
        assert len(processor.mine_states) == 2
        assert 'mine1' in processor.mine_states
        assert 'mine2' in processor.mine_states
    
    def test_get_dangerous_mines(self):
        """Test getting dangerous mines near ninja."""
        processor = MineStateProcessor()
        
        # Add mines manually
        processor.mine_states['mine1'] = MineState('mine1', (100.0, 100.0), MineState.TOGGLED)
        processor.mine_states['mine2'] = MineState('mine2', (200.0, 100.0), MineState.UNTOGGLED)
        processor.mine_states['mine3'] = MineState('mine3', (300.0, 100.0), MineState.TOGGLING)
        
        ninja_pos = (100.0, 100.0)
        
        # Get all dangerous mines
        dangerous = processor.get_dangerous_mines(ninja_pos)
        assert len(dangerous) == 2  # mine1 and mine3
        
        # Get within distance
        dangerous_close = processor.get_dangerous_mines(ninja_pos, max_distance=50.0)
        assert len(dangerous_close) == 1  # Only mine1
    
    def test_get_nearest_dangerous_mine(self):
        """Test finding nearest dangerous mine."""
        processor = MineStateProcessor()
        
        processor.mine_states['mine1'] = MineState('mine1', (100.0, 100.0), MineState.TOGGLED)
        processor.mine_states['mine2'] = MineState('mine2', (300.0, 100.0), MineState.TOGGLING)
        
        ninja_pos = (150.0, 100.0)
        
        nearest = processor.get_nearest_dangerous_mine(ninja_pos)
        assert nearest is not None
        assert nearest.mine_id == 'mine1'
    
    def test_get_mine_proximity_score(self):
        """Test mine proximity threat scoring."""
        processor = MineStateProcessor(safety_radius=2.0)
        
        # Add a dangerous mine
        mine_pos = (100.0, 100.0)
        processor.mine_states['mine1'] = MineState('mine1', mine_pos, MineState.TOGGLED)
        
        # Very close ninja - high score
        close_score = processor.get_mine_proximity_score((100.0, 100.0))
        assert close_score > 0.9
        
        # Moderate distance - medium score
        mid_score = processor.get_mine_proximity_score((105.0, 100.0))
        assert 0.0 < mid_score < 1.0
        
        # Far away - low score
        far_score = processor.get_mine_proximity_score((200.0, 100.0))
        assert far_score == 0.0
    
    def test_is_path_safe(self):
        """Test path safety checking."""
        processor = MineStateProcessor()
        
        # Add mine blocking a path
        processor.mine_states['mine1'] = MineState('mine1', (100.0, 100.0), MineState.TOGGLED)
        
        # Path through mine - unsafe
        assert not processor.is_path_safe((50.0, 100.0), (150.0, 100.0))
        
        # Path around mine - safe
        assert processor.is_path_safe((50.0, 150.0), (150.0, 150.0))
    
    def test_get_mines_blocking_path(self):
        """Test getting mines that block a path."""
        processor = MineStateProcessor()
        
        processor.mine_states['mine1'] = MineState('mine1', (100.0, 100.0), MineState.TOGGLED)
        processor.mine_states['mine2'] = MineState('mine2', (120.0, 100.0), MineState.TOGGLING)
        processor.mine_states['mine3'] = MineState('mine3', (100.0, 150.0), MineState.UNTOGGLED)
        
        # Get mines blocking horizontal path
        blocking = processor.get_mines_blocking_path((50.0, 100.0), (150.0, 100.0))
        assert len(blocking) == 2  # mine1 and mine2, not mine3 (safe)
    
    def test_get_mine_features(self):
        """Test getting mine feature array."""
        processor = MineStateProcessor()
        
        # Add some mines
        processor.mine_states['mine1'] = MineState('mine1', (100.0, 100.0), MineState.TOGGLED)
        processor.mine_states['mine2'] = MineState('mine2', (200.0, 100.0), MineState.TOGGLING)
        processor.mine_states['mine3'] = MineState('mine3', (300.0, 100.0), MineState.UNTOGGLED)
        
        ninja_pos = (150.0, 100.0)
        features = processor.get_mine_features(ninja_pos, max_mines=5)
        
        assert features.shape == (5, 7)  # max_mines x 7 features
        # First mine should be closest (mine1 at 50px distance)
        assert features[0, 5] == pytest.approx(50.0, abs=0.1)
    
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        processor = MineStateProcessor()
        
        processor.mine_states['mine1'] = MineState('mine1', (100.0, 100.0), MineState.TOGGLED)
        processor.mine_states['mine2'] = MineState('mine2', (200.0, 100.0), MineState.UNTOGGLED)
        processor.mine_states['mine3'] = MineState('mine3', (300.0, 100.0), MineState.TOGGLING)
        
        stats = processor.get_summary_stats()
        
        assert stats['total_mines'] == 3
        assert stats['dangerous_mines'] == 2
        assert stats['safe_mines'] == 1
    
    def test_processor_reset(self):
        """Test processor reset."""
        processor = MineStateProcessor()
        
        # Add some mines
        processor.mine_states['mine1'] = MineState('mine1', (100.0, 100.0), MineState.TOGGLED)
        
        # Reset
        processor.reset()
        
        assert len(processor.mine_states) == 0


class TestMineAwareIntegration:
    """Integration tests for mine awareness."""
    
    def test_mine_processor_with_observation_processor(self):
        """Test that mine processor integrates with observation processor."""
        # This is a placeholder for integration testing
        # In actual implementation, would test full observation processing pipeline
        pass
    
    def test_mine_features_in_reachability(self):
        """Test mine features in reachability analysis."""
        # This is a placeholder for integration testing
        # In actual implementation, would test reachability with mine obstacles
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
