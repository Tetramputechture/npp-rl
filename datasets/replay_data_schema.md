# Human Replay Data Schema

This document defines the expected format and schema for human replay data ingestion into the N++ RL training pipeline.

## Overview

Human replay data is used for behavioral cloning (BC) pretraining to provide the RL agent with expert demonstrations. The data must be processed to align with the `NppEnvironment` environment's observation space and action space.

## Input Format Specification

### Raw Replay Format (JSONL)

Raw replay files should be in JSON Lines format (`.jsonl`), where each line represents a single frame/timestep:

```json
{
  "timestamp": 1692345678.123,
  "level_id": "level_001",
  "frame_number": 42,
  "player_state": {
    "position": {"x": 150.5, "y": 200.3},
    "velocity": {"x": 2.1, "y": -0.5},
    "on_ground": true,
    "wall_sliding": false,
    "jump_time_remaining": 0.0
  },
  "player_inputs": {
    "left": false,
    "right": true,
    "jump": false,
    "restart": false
  },
  "entities": [
    {
      "type": "mine",
      "position": {"x": 180.0, "y": 220.0},
      "active": true
    },
    {
      "type": "exit_door",
      "position": {"x": 300.0, "y": 100.0},
      "active": true
    }
  ],
  "level_bounds": {
    "width": 400,
    "height": 300
  },
  "meta": {
    "session_id": "session_abc123",
    "player_id": "player_xyz789",
    "quality_score": 0.85,
    "completion_status": "in_progress"
  }
}
```

### Required Fields

#### Core Fields
- `timestamp`: Unix timestamp with millisecond precision
- `level_id`: Unique identifier for the level being played
- `frame_number`: Sequential frame number within the level attempt

#### Player State
- `player_state.position`: Player position in level coordinates
  - `x`: Horizontal position (float, level-specific range)
  - `y`: Vertical position (float, level-specific range)
- `player_state.velocity`: Player velocity vector
  - `x`: Horizontal velocity (float, typically -10.0 to 10.0)
  - `y`: Vertical velocity (float, typically -15.0 to 15.0)
- `player_state.on_ground`: Boolean indicating if player is on solid ground
- `player_state.wall_sliding`: Boolean indicating if player is wall-sliding
- `player_state.jump_time_remaining`: Remaining jump time (float, 0.0 to 1.0)

#### Player Inputs
- `player_inputs.left`: Boolean for left movement key
- `player_inputs.right`: Boolean for right movement key  
- `player_inputs.jump`: Boolean for jump key
- `player_inputs.restart`: Boolean for restart key

#### Entities
Array of game entities with:
- `type`: Entity type string ("mine", "exit_door", "switch", etc.)
- `position`: Entity position with `x` and `y` coordinates
- `active`: Boolean indicating if entity is active/enabled

#### Level Information
- `level_bounds.width`: Level width in pixels
- `level_bounds.height`: Level height in pixels

#### Metadata
- `meta.session_id`: Unique session identifier
- `meta.player_id`: Unique player identifier
- `meta.quality_score`: Quality score (0.0 to 1.0, higher is better)
- `meta.completion_status`: "in_progress", "completed", "failed", "abandoned"

### Optional Fields

- `camera_position`: Camera position for view calculations
- `level_geometry`: Static level geometry data (walls, platforms)
- `performance_metrics`: Frame timing, input latency, etc.
- `annotations`: Human-added labels or comments

## Output Format Specification

### Processed Dataset Format

After processing, replay data is converted to structured datasets compatible with the RL environment:

#### NPZ Format (Recommended)
```python
# Each .npz file contains:
{
    'observations': {
        'player_frame': np.array,      # Shape: (N, 64, 64, 3), uint8
        'global_view': np.array,       # Shape: (N, 128, 128, 3), uint8  
        'game_state': np.array         # Shape: (N, 31), float32 (rich profile)
    },
    'actions': np.array,               # Shape: (N,), int32, values 0-5
    'meta': {
        'level_ids': np.array,         # Shape: (N,), string
        'timestamps': np.array,        # Shape: (N,), float64
        'trajectory_lengths': np.array, # Shape: (num_trajectories,), int32
        'quality_scores': np.array,    # Shape: (num_trajectories,), float32
        'session_ids': np.array        # Shape: (num_trajectories,), string
    }
}
```

#### Parquet Format (Alternative)
For larger datasets, Parquet format with columnar storage:
- `observations/player_frame/`: Binary blobs of compressed image data
- `observations/global_view/`: Binary blobs of compressed image data
- `observations/game_state/`: Float32 arrays
- `actions`: Int32 action indices
- `meta/*`: Metadata columns

## Data Alignment with Environment

### Observation Space Alignment

The processed observations must match `NppEnvironment.observation_space`:

#### Player Frame (64x64x3)
- Rendered view centered on player
- RGB format, uint8 values [0, 255]
- Consistent with `get_player_frame()` output

#### Global View (128x128x3)  
- Full level view with player and entities
- RGB format, uint8 values [0, 255]
- Consistent with `get_global_view()` output

#### Game State Vector
**Minimal Profile (17 dimensions):**
1. Player position (x, y) - normalized to [0, 1]
2. Player velocity (x, y) - normalized to [-1, 1]
3. Player state flags (on_ground, wall_sliding) - binary
4. Jump time remaining - normalized to [0, 1]
5. Nearest mine distance - normalized to [0, 1]
6. Exit door distance - normalized to [0, 1]
7. Level progress - normalized to [0, 1]
8. Entity counts (mines, switches, etc.) - normalized

**Rich Profile (31 dimensions):**
- All minimal profile features (17)
- Extended ninja physics state (7 additional)
- Enhanced entity features (7 additional)

### Action Space Alignment

Actions must be mapped to the 6-action discrete space:

```python
ACTION_MAPPING = {
    0: "no_action",      # No input
    1: "left",           # Left movement only
    2: "right",          # Right movement only  
    3: "jump",           # Jump only
    4: "left_jump",      # Left + Jump
    5: "right_jump"      # Right + Jump
}
```

Input combinations are mapped as follows:
- `{left: False, right: False, jump: False}` → Action 0
- `{left: True, right: False, jump: False}` → Action 1
- `{left: False, right: True, jump: False}` → Action 2
- `{left: False, right: False, jump: True}` → Action 3
- `{left: True, right: False, jump: True}` → Action 4
- `{left: False, right: True, jump: True}` → Action 5

Invalid combinations (e.g., left + right) are mapped to Action 0.

## Data Quality Requirements

### Validation Criteria

1. **Action Validity**: All actions must be in range [0, 5]
2. **Observation Completeness**: All required observation keys present
3. **Numeric Bounds**: All numeric values within expected ranges
4. **Temporal Consistency**: Timestamps must be monotonically increasing
5. **Trajectory Completeness**: No missing frames in trajectories

### Quality Scoring

Quality scores (0.0 to 1.0) are assigned based on:
- **Completion Rate**: Did the player complete the level?
- **Efficiency**: Time to completion vs. optimal time
- **Smoothness**: Consistency of inputs and movement
- **Exploration**: Coverage of level areas
- **Error Rate**: Frequency of deaths/restarts

### De-duplication

Near-identical trajectories are identified and deduplicated based on:
- Level ID matching
- Similar action sequences (>95% overlap)
- Similar completion times (within 10%)
- Same player ID (optional)

## File Organization

### Directory Structure
```
datasets/
├── raw/                    # Raw JSONL replay files
│   ├── level_001/
│   ├── level_002/
│   └── ...
├── processed/              # Processed NPZ/Parquet files
│   ├── train/
│   ├── validation/
│   └── test/
├── metadata/               # Dataset metadata and indices
│   ├── dataset_info.json
│   ├── quality_report.json
│   └── deduplication_log.json
└── quarantine/             # Failed/invalid samples
    └── invalid_samples.jsonl
```

### Naming Conventions

- Raw files: `{level_id}_{session_id}_{timestamp}.jsonl`
- Processed files: `{level_id}_{split}_{shard:04d}.npz`
- Metadata files: `{dataset_name}_{version}_{field}.json`

## Usage Examples

### Loading Processed Data
```python
import numpy as np
from pathlib import Path

# Load a processed dataset shard
data = np.load('datasets/processed/train/level_001_train_0001.npz')
observations = data['observations'].item()
actions = data['actions']
meta = data['meta'].item()

print(f"Loaded {len(actions)} samples")
print(f"Observation keys: {list(observations.keys())}")
print(f"Action distribution: {np.bincount(actions)}")
```

### Validating Raw Data
```python
import json
from tools.replay_ingest import validate_replay_frame

with open('datasets/raw/level_001/replay.jsonl', 'r') as f:
    for line_num, line in enumerate(f):
        frame = json.loads(line)
        is_valid, errors = validate_replay_frame(frame)
        if not is_valid:
            print(f"Line {line_num}: {errors}")
```

## Version History

- **v1.0**: Initial schema definition
- **v1.1**: Added rich observation profile support
- **v1.2**: Enhanced quality scoring criteria
- **v1.3**: Added Parquet format specification

## Notes

- All position coordinates are in level pixel coordinates
- Velocity values are in pixels per frame
- Timestamps should be consistent within each session
- Quality scores are subjective and may need tuning based on data analysis
- The schema is designed to be extensible for future enhancements