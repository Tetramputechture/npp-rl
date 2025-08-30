"""
Entity type system for heterogeneous graph processing.

This module defines entity types, categories, and specialized processing
for different game elements in N++ levels.
"""

from enum import IntEnum
from typing import Dict, Tuple
import torch
import torch.nn as nn


class NodeType(IntEnum):
    """High-level node types in the heterogeneous graph."""
    GRID_CELL = 0
    ENTITY = 1
    NINJA = 2


class EntityType(IntEnum):
    """Specific entity types from N++ game."""
    # Interactive elements
    TOGGLE_MINE = 1
    GOLD = 2
    EXIT = 3
    EXIT_SWITCH = 4
    DOOR_REGULAR = 5
    DOOR_LOCKED = 6
    DOOR_TRAP = 8
    LAUNCH_PAD = 10
    ONE_WAY_PLATFORM = 11
    
    # Enemies/Hazards
    DRONE_ZAP = 14
    DRONE_CHASER = 15
    BOUNCE_BLOCK = 17
    THWUMP = 20
    ACTIVE_MINE = 21
    LASER = 23
    BOOST_PAD = 24
    DEATH_BALL = 25
    MINI_DRONE = 26
    SHOVE_THWUMP = 28


class EntityCategory(IntEnum):
    """Entity categories for specialized processing."""
    COLLECTIBLE = 0    # Gold, rewards
    MOVEMENT = 1       # Platforms, launch pads, boost pads, traversable entities
    HAZARD = 2         # Enemies, mines, dangerous elements
    INTERACTIVE = 3    # Doors, switches, exits
    GRID_TILE = 4      # Regular grid cells
    CONDITIONAL = 5    # Conditionally hazardous/traversable entities


class EntityTypeSystem:
    """
    System for managing entity types and their specialized processing.
    
    Provides mapping between entity types and processing categories,
    as well as specialized embedding and attention mechanisms.
    """
    
    def __init__(self):
        """Initialize entity type system."""
        self._entity_to_category = self._build_entity_category_mapping()
        self._category_properties = self._build_category_properties()
    
    def _build_entity_category_mapping(self) -> Dict[int, EntityCategory]:
        """Build mapping from entity types to categories."""
        mapping = {}
        
        # Collectibles
        collectibles = {EntityType.GOLD}
        for entity_type in collectibles:
            mapping[entity_type] = EntityCategory.COLLECTIBLE
        
        # Movement aids (always safe to traverse)
        movement_aids = {
            EntityType.LAUNCH_PAD, EntityType.ONE_WAY_PLATFORM, 
            EntityType.BOUNCE_BLOCK, EntityType.BOOST_PAD
        }
        for entity_type in movement_aids:
            mapping[entity_type] = EntityCategory.MOVEMENT
        
        # Conditional entities (directionally hazardous/traversable)
        conditional = {
            EntityType.THWUMP, EntityType.SHOVE_THWUMP
        }
        for entity_type in conditional:
            mapping[entity_type] = EntityCategory.CONDITIONAL
        
        # Hazards (fully dangerous entities)
        hazards = {
            EntityType.DRONE_ZAP, EntityType.DRONE_CHASER, EntityType.DEATH_BALL, 
            EntityType.MINI_DRONE, EntityType.TOGGLE_MINE, EntityType.ACTIVE_MINE, 
            EntityType.LASER
        }
        for entity_type in hazards:
            mapping[entity_type] = EntityCategory.HAZARD
        
        # Note: THWUMP and SHOVE_THWUMP are now in MOVEMENT category due to 
        # their conditional traversability (safe from sides/back)
        
        # Interactive elements
        interactive = {
            EntityType.EXIT, EntityType.EXIT_SWITCH, EntityType.DOOR_REGULAR,
            EntityType.DOOR_LOCKED, EntityType.DOOR_TRAP
        }
        for entity_type in interactive:
            mapping[entity_type] = EntityCategory.INTERACTIVE
        
        return mapping
    
    def _build_category_properties(self) -> Dict[EntityCategory, Dict]:
        """Build properties for each entity category."""
        return {
            EntityCategory.COLLECTIBLE: {
                'attention_weight': 1.2,  # Higher attention for rewards
                'hazard_level': 0.0,
                'interaction_range': 1.0,
                'movement_impact': False,
                'traversable': True
            },
            EntityCategory.MOVEMENT: {
                'attention_weight': 1.1,
                'hazard_level': 0.0,
                'interaction_range': 2.0,  # Larger interaction range
                'movement_impact': True,
                'traversable': True,
                'platform_capable': True  # Movement entities can be platforms
            },
            EntityCategory.HAZARD: {
                'attention_weight': 1.5,  # Highest attention for dangers
                'hazard_level': 1.0,
                'interaction_range': 1.5,
                'movement_impact': True,
                'traversable': False
            },
            EntityCategory.INTERACTIVE: {
                'attention_weight': 1.0,
                'hazard_level': 0.0,
                'interaction_range': 1.0,
                'movement_impact': False,
                'traversable': True
            },
            EntityCategory.GRID_TILE: {
                'attention_weight': 0.8,  # Lower attention for basic tiles
                'hazard_level': 0.0,
                'interaction_range': 0.5,
                'movement_impact': False,
                'traversable': True
            },
            EntityCategory.CONDITIONAL: {
                'attention_weight': 1.3,  # High attention for complex entities
                'hazard_level': 0.5,  # Partially hazardous
                'interaction_range': 2.0,  # Large range due to directional effects
                'movement_impact': True,
                'traversable': True,  # Conditionally traversable
                'directional_hazard': True,  # Special property for directional entities
                'platform_capable': True   # Can be used as platforms
            }
        }
    
    def get_entity_category(self, entity_type: int) -> EntityCategory:
        """Get category for an entity type."""
        return self._entity_to_category.get(entity_type, EntityCategory.INTERACTIVE)
    
    def get_category_properties(self, category: EntityCategory) -> Dict:
        """Get properties for an entity category."""
        return self._category_properties[category]
    
    def get_attention_weight(self, entity_type: int) -> float:
        """Get attention weight for an entity type."""
        category = self.get_entity_category(entity_type)
        return self._category_properties[category]['attention_weight']
    
    def is_hazardous(self, entity_type: int) -> bool:
        """Check if an entity type is hazardous."""
        category = self.get_entity_category(entity_type)
        return self._category_properties[category]['hazard_level'] > 0.0
    
    def affects_movement(self, entity_type: int) -> bool:
        """Check if an entity type affects movement."""
        category = self.get_entity_category(entity_type)
        return self._category_properties[category]['movement_impact']
    
    def is_traversable(self, entity_type: int) -> bool:
        """Check if an entity type is generally traversable."""
        category = self.get_entity_category(entity_type)
        return self._category_properties[category]['traversable']
    
    def is_directional_hazard(self, entity_type: int) -> bool:
        """Check if an entity type has directional hazard properties."""
        category = self.get_entity_category(entity_type)
        return self._category_properties[category].get('directional_hazard', False)
    
    def is_platform_capable(self, entity_type: int) -> bool:
        """Check if an entity type can be used as a platform."""
        category = self.get_entity_category(entity_type)
        return self._category_properties[category].get('platform_capable', False)
    
    def get_interaction_range(self, entity_type: int) -> float:
        """Get the interaction range for an entity type."""
        category = self.get_entity_category(entity_type)
        return self._category_properties[category]['interaction_range']


class EntitySpecializedEmbedding(nn.Module):
    """
    Specialized embedding layer for different entity types.
    
    Creates separate embedding spaces for different entity categories
    and applies category-specific transformations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        entity_type_system: EntityTypeSystem,
        dropout: float = 0.1
    ):
        """
        Initialize specialized embedding layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
            entity_type_system: Entity type system for categorization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.entity_type_system = entity_type_system
        
        # Category-specific embedding layers
        self.category_embeddings = nn.ModuleDict()
        for category in EntityCategory:
            self.category_embeddings[category.name.lower()] = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
        
        # Attention mechanism for category importance
        self.category_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        entity_types: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through specialized embedding.
        
        Args:
            node_features: Input node features [batch_size, num_nodes, input_dim]
            node_types: Node types [batch_size, num_nodes]
            entity_types: Entity types for entity nodes [batch_size, num_nodes]
            
        Returns:
            Specialized embeddings [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, _ = node_features.shape
        output = torch.zeros(batch_size, num_nodes, self.output_dim, device=node_features.device)
        
        # Process each node based on its type and category
        for b in range(batch_size):
            for n in range(num_nodes):
                node_type = node_types[b, n].item()
                
                if node_type == NodeType.GRID_CELL:
                    category = EntityCategory.GRID_TILE
                elif node_type == NodeType.ENTITY and entity_types is not None:
                    entity_type = entity_types[b, n].item()
                    category = self.entity_type_system.get_entity_category(entity_type)
                else:
                    category = EntityCategory.INTERACTIVE  # Default
                
                # Apply category-specific embedding
                category_name = category.name.lower()
                embedding = self.category_embeddings[category_name](node_features[b:b+1, n:n+1])
                output[b, n] = embedding.squeeze()
        
        # Apply cross-category attention
        output, _ = self.category_attention(output, output, output)
        
        # Final projection
        output = self.output_projection(output)
        
        return output


class HazardAwareAttention(nn.Module):
    """
    Hazard-aware attention mechanism that gives higher attention to dangerous elements.
    
    This attention mechanism uses entity type information to bias attention
    towards hazardous elements, helping the agent better avoid dangers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        entity_type_system: EntityTypeSystem,
        dropout: float = 0.1
    ):
        """
        Initialize hazard-aware attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            entity_type_system: Entity type system for hazard detection
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.entity_type_system = entity_type_system
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Hazard bias parameters
        self.hazard_bias = nn.Parameter(torch.zeros(1))
        self.movement_bias = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        entity_types: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hazard-aware attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            entity_types: Entity types [batch_size, seq_len]
            key_padding_mask: Padding mask [batch_size, seq_len]
            
        Returns:
            Attention output and attention weights
        """
        # Standard attention
        attn_output, attn_weights = self.attention(
            query, key, value, key_padding_mask=key_padding_mask
        )
        
        # Apply hazard-aware bias if entity types are provided
        if entity_types is not None:
            batch_size, seq_len = entity_types.shape
            
            # Create bias mask based on entity types
            hazard_mask = torch.zeros_like(entity_types, dtype=torch.float)
            movement_mask = torch.zeros_like(entity_types, dtype=torch.float)
            
            for b in range(batch_size):
                for s in range(seq_len):
                    entity_type = entity_types[b, s].item()
                    if self.entity_type_system.is_hazardous(entity_type):
                        hazard_mask[b, s] = 1.0
                    if self.entity_type_system.affects_movement(entity_type):
                        movement_mask[b, s] = 1.0
            
            # Apply bias to attention weights
            bias = (hazard_mask * self.hazard_bias + 
                   movement_mask * self.movement_bias).unsqueeze(1)
            
            # Re-weight attention output
            bias_weights = torch.softmax(attn_weights + bias, dim=-1)
            attn_output = torch.bmm(bias_weights, value)
        
        return attn_output, attn_weights


def create_entity_type_system() -> EntityTypeSystem:
    """Create and return an entity type system instance."""
    return EntityTypeSystem()