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
