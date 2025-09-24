"""
Simplified Node Feature Extractor for NPP-RL.

This module replaces the complex physics state extractor with a simplified
approach that extracts only essential features, allowing the HGT multimodal
network to learn movement patterns emergently.

Key simplifications:
- 8 features instead of 31 complex physics features
- Basic position and entity information
- Logical relationships (switches, exits)
- Simple distance metrics
- No complex physics calculations

This aligns with HGT design principles: provide basic connectivity and let
the network learn complex patterns through multimodal training.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

# Use nclone physics constants
FULL_MAP_WIDTH_PX = 1056  # Standard N++ level width
FULL_MAP_HEIGHT_PX = 600   # Standard N++ level height  
TILE_PIXEL_SIZE = 24       # Standard N++ tile size


@dataclass
class EntityInfo:
    """Simple entity information for node feature extraction."""
    entity_type: int
    position: Tuple[float, float]
    state: int
    radius: float


class PhysicsStateExtractor:
    """
    Simplified node feature extractor for NPP-RL.
    
    Replaces complex physics modeling with 8 strategic features that enable
    HGT multimodal learning of movement patterns. This approach aligns with
    transformer design principles: provide basic information and let the
    network discover complex patterns through attention mechanisms.
    
    Features extracted (8 total):
    1. x_position_norm: Normalized X position [0,1]
    2. y_position_norm: Normalized Y position [0,1]  
    3. tile_type_norm: Normalized tile type [0,1]
    4. has_entity: Binary flag for entity presence
    5. entity_type_norm: Normalized entity type [0,1]
    6. switch_distance_norm: Distance to nearest switch [0,1]
    7. exit_distance_norm: Distance to exit [0,1]
    8. switch_activated: Binary switch activation state
    """
    
    def __init__(self, debug: bool = False):
        """Initialize simplified node feature extractor."""
        self.debug = debug
        self.feature_names = [
            'x_position_norm',
            'y_position_norm', 
            'tile_type_norm',
            'has_entity',
            'entity_type_norm',
            'switch_distance_norm',
            'exit_distance_norm',
            'switch_activated'
        ]
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
    
    def extract_node_features(
        self,
        position: Tuple[float, float],
        tile_type: int,
        entities: List[EntityInfo],
        game_state: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract simplified node features for a position.
        
        Args:
            position: Tile position (x, y) in pixels
            tile_type: Tile type ID (0=empty, 1=wall, etc.)
            entities: List of entities at or near this position
            game_state: Current game state (switches, etc.)
            
        Returns:
            Array of 8 normalized features [0,1]
        """
        x, y = position
        
        # 1-2: Normalized position
        x_norm = x / FULL_MAP_WIDTH_PX
        y_norm = y / FULL_MAP_HEIGHT_PX
        
        # 3: Normalized tile type (assume max 20 tile types)
        tile_type_norm = tile_type / 20.0
        
        # 4-5: Entity information
        has_entity = 1.0 if entities else 0.0
        entity_type_norm = 0.0
        if entities:
            # Use first entity if multiple (most relevant)
            entity_type_norm = entities[0].entity_type / 10.0  # Assume max 10 entity types
            
        # 6: Distance to nearest switch (normalized)
        switch_distance_norm = self._get_switch_distance_norm(position, entities)
        
        # 7: Distance to exit (normalized)
        exit_distance_norm = self._get_exit_distance_norm(position, entities)
        
        # 8: Switch activation state
        switch_activated = 1.0 if game_state.get("exit_switch_activated", False) else 0.0
        
        features = np.array([
            x_norm,
            y_norm,
            tile_type_norm,
            has_entity,
            entity_type_norm,
            switch_distance_norm,
            exit_distance_norm,
            switch_activated
        ], dtype=np.float32)
        
        # Ensure all features are in [0,1] range
        features = np.clip(features, 0.0, 1.0)
        
        if self.debug:
            print(f"Node features for {position}: {features}")
            
        return features
    
    def _get_switch_distance_norm(self, position: Tuple[float, float], entities: List[EntityInfo]) -> float:
        """Get normalized distance to nearest switch."""
        x, y = position
        min_distance = float('inf')
        
        for entity in entities:
            if entity.entity_type == 4:  # Switch entity type
                ex, ey = entity.position
                distance = math.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 1.0  # No switch found, max distance
            
        # Normalize by diagonal map distance
        max_distance = math.sqrt(FULL_MAP_WIDTH_PX ** 2 + FULL_MAP_HEIGHT_PX ** 2)
        return min(min_distance / max_distance, 1.0)
    
    def _get_exit_distance_norm(self, position: Tuple[float, float], entities: List[EntityInfo]) -> float:
        """Get normalized distance to exit."""
        x, y = position
        min_distance = float('inf')
        
        for entity in entities:
            if entity.entity_type == 3:  # Exit entity type
                ex, ey = entity.position
                distance = math.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 1.0  # No exit found, max distance
            
        # Normalize by diagonal map distance
        max_distance = math.sqrt(FULL_MAP_WIDTH_PX ** 2 + FULL_MAP_HEIGHT_PX ** 2)
        return min(min_distance / max_distance, 1.0)

    # Legacy compatibility methods for existing code
    def extract_ninja_physics_state(
        self,
        ninja_position: Tuple[float, float],
        ninja_velocity: Tuple[float, float],
        ninja_state: Dict[str, Any],
        level_data: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Legacy compatibility method.
        
        This method maintains the original interface but returns simplified features.
        For new code, use extract_node_features() directly.
        """
        # Convert to simplified format
        entities = []
        game_state = {"exit_switch_activated": False}
        
        # Extract basic tile type (default to 0)
        tile_type = 0
        
        return self.extract_node_features(ninja_position, tile_type, entities, game_state)