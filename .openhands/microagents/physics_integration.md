---
agent: 'CodeActAgent'
triggers: ['physics', 'nclone', 'constants', 'movement', 'trajectory', 'ninja', 'gravity', 'collision', 'jump', 'n++']
---

# N++ Physics Integration Guidelines

## Critical Rule: Always Use nclone.constants

**NEVER redefine physics constants that already exist in nclone.constants. Always import and use the official values.**

### Essential Physics Constants Import Pattern

```python
from nclone.constants import (
    # Ninja physical properties
    NINJA_RADIUS,                    # 10 pixels (collision radius)
    
    # Gravity constants
    GRAVITY_FALL,                   # 0.06666... (falling/not jumping)
    GRAVITY_JUMP,                   # 0.01111... (actively jumping)
    
    # Movement acceleration
    GROUND_ACCEL,                   # 0.06666... (horizontal on ground)
    AIR_ACCEL,                      # 0.04444... (horizontal in air)
    
    # Drag and friction
    DRAG_REGULAR,                   # 0.9933... (normal drag)
    DRAG_SLOW,                      # 0.8617... (slow motion areas)
    FRICTION_GROUND,                # 0.9459... (ground friction)
    FRICTION_WALL,                  # 0.9113... (wall friction)
    
    # Speed limits
    MAX_HOR_SPEED,                  # 3.333... (maximum horizontal velocity)
    
    # Jump mechanics
    MAX_JUMP_DURATION,              # 45 frames (maximum jump hold time)
    JUMP_FLAT_GROUND_Y,            # -2 (basic jump velocity)
    
    # Wall jump constants
    JUMP_WALL_SLIDE_X,             # 2/3 (wall slide jump horizontal)
    JUMP_WALL_SLIDE_Y,             # -1 (wall slide jump vertical)
    JUMP_WALL_REGULAR_X,           # 1 (wall jump horizontal)
    JUMP_WALL_REGULAR_Y,           # -1.4 (wall jump vertical)
    
    # Collision and damage
    MAX_SURVIVABLE_IMPACT,         # 6 (maximum impact velocity)
    MIN_SURVIVABLE_CRUSHING,       # 0.05 (minimum crushing threshold)
)
```

## N++ Movement States

The ninja has 9 distinct movement states that affect physics calculations:

```python
from enum import IntEnum

class MovementState(IntEnum):
    """Official N++ movement states from simulation."""
    IMMOBILE = 0        # Stationary on ground, no input
    RUNNING = 1         # Moving horizontally with input
    GROUND_SLIDING = 2  # Sliding with momentum, input opposite
    JUMPING = 3         # Actively jumping (reduced gravity)
    FALLING = 4         # In air, falling or moving without jump
    WALL_SLIDING = 5    # Sliding down wall while holding toward it
    DEAD = 6           # Post-death ragdoll physics
    AWAITING_DEATH = 7 # Single-frame transition to death
    CELEBRATING = 8    # Victory state, reduced drag
    DISABLED = 9       # Inactive state
```

## Game Mechanics Integration

### Level Structure Constants

```python
# N++ level structure (from nclone documentation)
LEVEL_WIDTH = 42        # Grid cells
LEVEL_HEIGHT = 23       # Grid cells  
CELL_SIZE = 24         # Pixels per cell
LEVEL_PIXEL_WIDTH = 1056   # 42 * 24
LEVEL_PIXEL_HEIGHT = 600   # 23 * 24

MAX_LEVEL_TIME = 20000  # Maximum frames per level (20,000 @ 60fps = ~5.5 minutes)
FRAME_RATE = 60        # Standard game speed
```

### Input Timing and Buffering

```python
# N++ input buffering constants (critical for AI timing)
JUMP_BUFFER_FRAMES = 5      # Jump input buffer window
WALL_BUFFER_FRAMES = 5      # Wall grab buffer window
FLOOR_BUFFER_FRAMES = 5     # Floor contact buffer window
LAUNCH_PAD_BUFFER_FRAMES = 5 # Launch pad boost buffer window

def check_input_timing(current_frame: int, input_frame: int, buffer_type: str) -> bool:
    """Check if input timing is within buffer window."""
    buffer_size = {
        'jump': JUMP_BUFFER_FRAMES,
        'wall': WALL_BUFFER_FRAMES,
        'floor': FLOOR_BUFFER_FRAMES,
        'launch': LAUNCH_PAD_BUFFER_FRAMES
    }.get(buffer_type, 0)
    
    return abs(current_frame - input_frame) <= buffer_size
```

### Physics State Extraction

```python
class PhysicsStateExtractor:
    """Extract physics-relevant state information for RL observations."""
    
    def extract_ninja_physics_state(self, ninja) -> np.ndarray:
        """Extract physics state vector from ninja object."""
        return np.array([
            ninja.x / LEVEL_PIXEL_WIDTH,           # Normalized X position
            ninja.y / LEVEL_PIXEL_HEIGHT,          # Normalized Y position
            ninja.v_x / MAX_HOR_SPEED,             # Normalized X velocity
            ninja.v_y / abs(JUMP_FLAT_GROUND_Y),   # Normalized Y velocity
            
            # Movement state (one-hot encoded)
            float(ninja.movement_state == MovementState.IMMOBILE),
            float(ninja.movement_state == MovementState.RUNNING),
            float(ninja.movement_state == MovementState.JUMPING),
            float(ninja.movement_state == MovementState.FALLING),
            float(ninja.movement_state == MovementState.WALL_SLIDING),
            
            # Contact states
            float(ninja.ground_contact),
            float(ninja.wall_contact),
            float(ninja.ceiling_contact),
            
            # Jump state
            ninja.jump_duration / MAX_JUMP_DURATION,
            
            # Timing information
            ninja.time_since_ground_contact / 60.0,  # Convert frames to seconds
            ninja.time_since_wall_contact / 60.0,
        ], dtype=np.float32)
    
    def extract_level_physics_features(self, level_data, ninja_pos) -> np.ndarray:
        """Extract physics-relevant level features around ninja."""
        # Extract local geometry that affects physics
        x, y = int(ninja_pos[0] // CELL_SIZE), int(ninja_pos[1] // CELL_SIZE)
        
        # Sample 5x5 grid around ninja for local physics context
        local_features = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell_x, cell_y = x + dx, y + dy
                
                if 0 <= cell_x < LEVEL_WIDTH and 0 <= cell_y < LEVEL_HEIGHT:
                    cell_value = level_data[cell_y * LEVEL_WIDTH + cell_x]
                    
                    # Extract physics-relevant properties
                    is_solid = cell_value in SOLID_TILE_VALUES
                    is_hazard = cell_value in HAZARD_TILE_VALUES
                    is_slope = cell_value in SLOPE_TILE_VALUES
                    
                    local_features.extend([
                        float(is_solid),
                        float(is_hazard),
                        float(is_slope)
                    ])
                else:
                    # Out of bounds - treat as solid
                    local_features.extend([1.0, 0.0, 0.0])
        
        return np.array(local_features, dtype=np.float32)
```

## Physics Validation and Testing

### Trajectory Validation Tests

```python
def test_physics_accurate_trajectory_calculation():
    """Test that trajectory calculations match actual N++ physics."""
    
    # Test horizontal movement (should not require jumping)
    src = (100.0, 100.0)
    tgt = (200.0, 100.0)  # 100 pixels right, same height
    
    result = calculate_trajectory(src, tgt)
    
    # Should be feasible walk
    assert result.feasible, "Horizontal movement should be feasible"
    assert not result.requires_jump, "Horizontal movement should not require jumping"
    assert result.time_estimate > 0, "Should have positive time estimate"
    
    # Test upward jump
    src = (100.0, 100.0)
    tgt = (150.0, 50.0)   # 50 pixels up, 50 right
    
    result = calculate_trajectory(src, tgt)
    
    # Should require jumping and use correct physics
    assert result.requires_jump, "Upward movement should require jumping"
    assert result.time_of_flight > 0, "Jump should have positive flight time"
    
    # Verify physics: y = v0*t + 0.5*g*t^2
    expected_dy = (JUMP_FLAT_GROUND_Y * result.time_of_flight + 
                   0.5 * GRAVITY_JUMP * result.time_of_flight**2)
    actual_dy = tgt[1] - src[1]
    
    assert abs(expected_dy - actual_dy) < 1.0, "Physics calculation should be accurate"

def test_movement_state_physics_consistency():
    """Test that movement states correspond to correct physics."""
    classifier = MovementClassifier()
    
    # Test different movement scenarios
    test_cases = [
        # (src, tgt, expected_movement_type, expected_physics_props)
        ((100, 100), (200, 100), MovementType.WALK, {'no_jumping': True}),
        ((100, 100), (150, 50), MovementType.JUMP, {'requires_gravity': True}),
        ((100, 100), (100, 150), MovementType.FALL, {'uses_gravity': True}),
    ]
    
    for src, tgt, expected_type, expected_props in test_cases:
        movement_type, params = classifier.classify_movement(src, tgt)
        
        assert movement_type == expected_type, f"Expected {expected_type}, got {movement_type}"
        
        # Verify physics properties match movement type
        if expected_props.get('no_jumping'):
            assert not params.get('requires_jump', False)
        if expected_props.get('requires_gravity'):
            assert params.get('energy_cost', 0) > 0
```

## Integration with RL Training

### Physics-Informed Reward Shaping

```python
def compute_physics_informed_rewards(ninja_state, action, next_ninja_state) -> Dict[str, float]:
    """Compute rewards that respect N++ physics principles."""
    
    rewards = {}
    
    # Efficiency reward: encourage physics-optimal movements
    velocity_magnitude = math.sqrt(next_ninja_state.v_x**2 + next_ninja_state.v_y**2)
    max_possible_velocity = math.sqrt(MAX_HOR_SPEED**2 + abs(JUMP_FLAT_GROUND_Y)**2)
    velocity_efficiency = min(velocity_magnitude / max_possible_velocity, 1.0)
    
    rewards['velocity_efficiency'] = velocity_efficiency * 0.1
    
    # Penalize physics violations (if any)
    if abs(next_ninja_state.v_x) > MAX_HOR_SPEED * 1.1:  # Small tolerance for numerical errors
        rewards['speed_violation'] = -1.0
    
    # Reward appropriate use of jump mechanics
    if action == Action.JUMP and ninja_state.ground_contact:
        rewards['proper_jump_timing'] = 0.05
    
    # Reward wall jump usage when appropriate
    if (action in [Action.JUMP_LEFT, Action.JUMP_RIGHT] and 
        ninja_state.wall_contact and ninja_state.movement_state == MovementState.WALL_SLIDING):
        rewards['wall_jump_technique'] = 0.1
    
    return rewards
```

This physics integration guide ensures that all AI development respects the precise N++ physics simulation, leading to more realistic and effective agent behavior that can transfer to the actual game.
