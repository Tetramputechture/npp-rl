# N++ Binary Replay Parser

This tool converts N++ binary replay files ("trace" mode) to JSONL format compatible with the npp-rl training pipeline.

## Overview

The N++ Binary Replay Parser processes original N++ replay files and converts them to the JSONL format expected by the `replay_ingest.py` tool. It simulates the game frame-by-frame to extract complete state information.

## Input Format

The parser expects N++ "trace" mode replay files with the following structure:

```
replay_directory/
├── inputs_0        # Binary file: zlib-compressed input sequence for replay 1
├── inputs_1        # Binary file: zlib-compressed input sequence for replay 2
├── inputs_2        # Binary file: zlib-compressed input sequence for replay 3
├── inputs_3        # Binary file: zlib-compressed input sequence for replay 4
└── map_data        # Binary file: Raw map geometry data
```

### Input Encoding

Each byte in the input files represents a combined input state (0-7):

| Value | Horizontal | Jump | Description |
|-------|------------|------|-------------|
| 0     | 0          | 0    | No input |
| 1     | 0          | 1    | Jump only |
| 2     | 1          | 0    | Right only |
| 3     | 1          | 1    | Right + Jump |
| 4     | -1         | 0    | Left only |
| 5     | -1         | 1    | Left + Jump |
| 6     | -1         | 0    | Left only (alternate) |
| 7     | -1         | 1    | Left + Jump (alternate) |

## Output Format

The parser generates JSONL files compatible with the existing replay ingestion pipeline. Each line represents a single frame:

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
    }
  ],
  "level_bounds": {
    "width": 1056,
    "height": 600
  },
  "meta": {
    "session_id": "level_001_session_000",
    "player_id": "binary_replay",
    "quality_score": 0.8,
    "completion_status": "in_progress"
  }
}
```

## Usage

### Basic Usage

```bash
# Process a single replay directory
python tools/binary_replay_parser.py --input replays/level_001 --output datasets/raw/

# Process multiple replay directories
python tools/binary_replay_parser.py --input replays/ --output datasets/raw/

# Enable verbose logging
python tools/binary_replay_parser.py --input replays/ --output datasets/raw/ --verbose
```

### Arguments

- `--input`: Input directory containing binary replay files (required)
- `--output`: Output directory for JSONL files (required)
- `--verbose, -v`: Enable verbose logging (optional)

## Processing Pipeline

1. **Detection**: Scan for trace mode files (`inputs_*` and `map_data`)
2. **Loading**: Load and decompress input sequences and map data
3. **Simulation**: Run nclone simulator frame-by-frame with decoded inputs
4. **Extraction**: Extract player state, inputs, and entity data at each frame
5. **Output**: Generate JSONL files with timestamped frame data

## Integration with Existing Tools

After generating JSONL files, process them with the existing replay ingestion tool:

```bash
# Convert binary replays to JSONL
python tools/binary_replay_parser.py --input replays/ --output datasets/raw/

# Process JSONL files for training
python tools/replay_ingest.py --input datasets/raw/ --output datasets/processed/ --profile rich
```

## Entity Type Mapping

The parser maps numeric entity types to names:

| Type ID | Name | Description |
|---------|------|-------------|
| 1 | mine | Toggle mine |
| 2 | gold | Gold collectible |
| 3 | exit_door | Level exit |
| 4 | exit_switch | Switch to activate exit |
| 5 | door | Regular door |
| 6 | locked_door | Locked door |
| 8 | trap_door | Trap door |
| 10 | launch_pad | Launch pad |
| 11 | one_way_platform | One-way platform |
| 14 | drone | Regular drone |
| 17 | bounce_block | Bounce block |
| 20 | thwump | Thwump enemy |
| 21 | toggle_mine | Toggle mine |
| 25 | death_ball | Death ball |
| 26 | mini_drone | Mini drone |

## Dependencies

- nclone package (simulator and physics)
- Python 3.7+
- Standard library modules: json, zlib, pathlib, etc.

## Assumptions

- Level dimensions are fixed at 1056x600 pixels
- Frame rate is 60 FPS for timestamp calculation
- Default quality score is 0.8 (0.9 for completed levels)
- Player ID is set to "binary_replay" for all converted replays

## Error Handling

- Invalid or corrupted replay files are skipped with error logging
- Missing input files are handled gracefully
- Simulation failures are logged and tracked in statistics

## Statistics

The parser outputs processing statistics including:
- Directories processed
- Replays processed/failed
- Frames generated
- Success rate
- Average frames per replay

## Limitations

- Currently only supports "trace" mode (not "splits" mode)
- Fixed level dimensions assumption
- Limited entity state extraction (could be enhanced)
- No support for custom quality scoring based on actual gameplay
