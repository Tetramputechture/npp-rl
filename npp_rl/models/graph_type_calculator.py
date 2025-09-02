"""
Graph Type Calculator - Proper approach for determining node and edge types.

This module demonstrates how node and edge types should be calculated directly
from map data during graph construction, rather than inferred from features.
This approach is more accurate, robust, and efficient.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from nclone.graph.common import NodeType, EdgeType
from nclone.constants.entity_types import EntityType

def calculate_node_types_from_map_data(
    level_data: Dict[str, Any],
    ninja_position: Tuple[float, float],
    node_positions: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Calculate node types directly from map data.
    
    This is the CORRECT approach - determine types based on what each node
    actually represents in the game world, not by analyzing processed features.
    
    Args:
        level_data: Raw level data with tiles and entities
        ninja_position: Current ninja position
        node_positions: List of (x, y) positions for each node in the graph
        
    Returns:
        Array of NodeType enum values for each node
    """
    node_types = np.zeros(len(node_positions), dtype=np.int32)
    
    # Get entities from level data
    entities = level_data.get('entities', [])
    entity_positions = set()
    for entity in entities:
        if isinstance(entity, dict):
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            entity_positions.add((entity_x, entity_y))
    
    # Classify each node based on its position and what it represents
    for i, (node_x, node_y) in enumerate(node_positions):
        # Check if this node represents the ninja
        ninja_x, ninja_y = ninja_position
        if abs(node_x - ninja_x) < 12 and abs(node_y - ninja_y) < 12:  # Within ninja radius
            node_types[i] = NodeType.NINJA
            continue
        
        # Check if this node represents an entity
        is_entity = False
        for entity_x, entity_y in entity_positions:
            if abs(node_x - entity_x) < 12 and abs(node_y - entity_y) < 12:  # Within entity radius
                node_types[i] = NodeType.ENTITY
                is_entity = True
                break
        
        if not is_entity:
            # This is a regular grid cell
            node_types[i] = NodeType.GRID_CELL
    
    return node_types


def calculate_edge_types_from_movement_data(
    edge_list: List[Tuple[int, int]],
    node_positions: List[Tuple[float, float]],
    movement_data: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Calculate edge types directly from movement mechanics.
    
    This is the CORRECT approach - determine edge types based on the actual
    movement mechanics required to traverse between nodes.
    
    Args:
        edge_list: List of (source_node, target_node) pairs
        node_positions: List of (x, y) positions for each node
        movement_data: Optional movement analysis data
        
    Returns:
        Array of EdgeType enum values for each edge
    """
    edge_types = np.zeros(len(edge_list), dtype=np.int32)
    
    for i, (source_idx, target_idx) in enumerate(edge_list):
        source_pos = node_positions[source_idx]
        target_pos = node_positions[target_idx]
        
        # Calculate movement vector
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Classify based on movement characteristics
        if abs(dy) < 6:  # Mostly horizontal movement
            edge_types[i] = EdgeType.WALK
        elif dy < -24:  # Significant upward movement (negative Y is up)
            if abs(dx) > 12:  # Horizontal component suggests wall jump
                edge_types[i] = EdgeType.WALL_SLIDE  # Wall jump/slide
            else:
                edge_types[i] = EdgeType.JUMP
        elif dy > 24:  # Downward movement
            edge_types[i] = EdgeType.FALL
        elif distance > 100:  # Very long distance suggests special movement
            # Check if there's a launch pad at source position
            # This would require checking entity data
            edge_types[i] = EdgeType.FUNCTIONAL  # Could be launch pad or other special
        else:
            # Default to walk for unclear cases
            edge_types[i] = EdgeType.WALK
    
    return edge_types


def calculate_functional_edge_types(
    edge_list: List[Tuple[int, int]],
    node_positions: List[Tuple[float, float]],
    level_data: Dict[str, Any]
) -> np.ndarray:
    """
    Calculate functional edge types based on entity interactions.
    
    Functional edges represent non-movement relationships like:
    - Switch -> Door activation
    - Launch pad -> Target position
    - Key -> Lock relationships
    
    Args:
        edge_list: List of (source_node, target_node) pairs
        node_positions: List of (x, y) positions for each node
        level_data: Level data with entity information
        
    Returns:
        Array indicating which edges are functional (1) vs movement (0)
    """
    functional_edges = np.zeros(len(edge_list), dtype=np.int32)
    
    entities = level_data.get('entities', [])
    
    # Build entity position map
    entity_map = {}
    for entity in entities:
        if isinstance(entity, dict):
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            entity_type = entity.get('type', None)
            entity_map[(entity_x, entity_y)] = entity_type
    
    for i, (source_idx, target_idx) in enumerate(edge_list):
        source_pos = node_positions[source_idx]
        target_pos = node_positions[target_idx]
        
        # Check if source is a functional entity (switch, launch pad, etc.)
        source_entity_type = None
        for (entity_x, entity_y), entity_type in entity_map.items():
            if abs(source_pos[0] - entity_x) < 12 and abs(source_pos[1] - entity_y) < 12:
                source_entity_type = entity_type
                break
        
        # Determine if this creates a functional relationship
        if source_entity_type == EntityType.LAUNCH_PAD:
            # Launch pad creates functional edge to landing position
            distance = np.sqrt((target_pos[0] - source_pos[0])**2 + (target_pos[1] - source_pos[1])**2)
            if distance > 50:  # Long distance suggests launch pad trajectory
                functional_edges[i] = 1
        
        # Switch -> Door functional relationships using actual entity associations
        elif source_entity_type in ['exit_switch', EntityType.LOCKED_DOOR, EntityType.TRAP_DOOR]:
            # Find the corresponding door entity using proper associations
            target_entity_type = None
            for (entity_x, entity_y), entity_type in entity_map.items():
                if abs(target_pos[0] - entity_x) < 12 and abs(target_pos[1] - entity_y) < 12:
                    target_entity_type = entity_type
                    break
            
            # Check for proper switch->door relationships
            if source_entity_type == 'exit_switch' and target_entity_type == 'exit_door':
                functional_edges[i] = 1
            elif (source_entity_type == EntityType.LOCKED_DOOR and 
                  target_entity_type == 'door_segment_locked'):
                # Verify this is the correct door using entity data
                source_entity = next((e for e in entities if 
                                    abs(e.get('x', 0) - source_pos[0]) < 12 and 
                                    abs(e.get('y', 0) - source_pos[1]) < 12), None)
                if source_entity:
                    door_pos = (source_entity.get('door_x', 0), source_entity.get('door_y', 0))
                    if abs(door_pos[0] - target_pos[0]) < 12 and abs(door_pos[1] - target_pos[1]) < 12:
                        functional_edges[i] = 1
            elif (source_entity_type == EntityType.TRAP_DOOR and 
                  target_entity_type == 'door_segment_trap'):
                # Verify this is the correct door using entity data
                source_entity = next((e for e in entities if 
                                    abs(e.get('x', 0) - source_pos[0]) < 12 and 
                                    abs(e.get('y', 0) - source_pos[1]) < 12), None)
                if source_entity:
                    door_pos = (source_entity.get('door_x', 0), source_entity.get('door_y', 0))
                    if abs(door_pos[0] - target_pos[0]) < 12 and abs(door_pos[1] - target_pos[1]) < 12:
                        functional_edges[i] = 1
    
    return functional_edges


def demonstrate_proper_type_calculation():
    """
    Demonstrate the proper approach to type calculation.
    
    This shows how types should be calculated during graph construction
    rather than inferred from features later.
    """
    # Example level data
    level_data = {
        'tiles': np.zeros((20, 20)),
        'entities': [
            {'type': EntityType.LAUNCH_PAD, 'x': 100.0, 'y': 100.0},
            {'type': EntityType.TOGGLE_MINE, 'x': 200.0, 'y': 150.0, 'state': 1},
            {'type': EntityType.DRONE_ZAP, 'x': 300.0, 'y': 200.0}
        ]
    }
    
    ninja_position = (50.0, 50.0)
    
    # Example graph nodes (in a real graph builder, these would be generated systematically)
    node_positions = [
        (50.0, 50.0),   # Ninja position
        (100.0, 100.0), # Launch pad
        (200.0, 150.0), # Toggle mine
        (75.0, 75.0),   # Grid cell
        (125.0, 125.0), # Grid cell
    ]
    
    # Calculate node types directly from map data
    node_types = calculate_node_types_from_map_data(level_data, ninja_position, node_positions)
    
    print("Node Types (calculated from map data):")
    for i, node_type in enumerate(node_types):
        pos = node_positions[i]
        type_name = NodeType(node_type).name
        print(f"  Node {i} at {pos}: {type_name}")
    
    # Example edges
    edge_list = [
        (0, 3),  # Ninja to grid cell (walk)
        (3, 4),  # Grid cell to grid cell (walk)
        (0, 1),  # Ninja to launch pad (functional)
        (1, 4),  # Launch pad to target (launch)
    ]
    
    # Calculate edge types from movement mechanics
    edge_types = calculate_edge_types_from_movement_data(edge_list, node_positions)
    
    print("\nEdge Types (calculated from movement mechanics):")
    for i, edge_type in enumerate(edge_types):
        source, target = edge_list[i]
        type_name = EdgeType(edge_type).name
        print(f"  Edge {i} ({source}->{target}): {type_name}")
    
    return node_types, edge_types


if __name__ == "__main__":
    demonstrate_proper_type_calculation()