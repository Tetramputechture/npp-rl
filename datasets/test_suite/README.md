# NPP-RL Test Suite (Task 3.3)

This directory contains a comprehensive, deterministic test suite of 250 N++ levels designed for evaluating NPP-RL agent performance across different complexity categories.

## Overview

The test suite provides a well-formed baseline dataset for training and evaluating reinforcement learning agents on N++. All maps are generated deterministically using fixed seeds to ensure reproducibility across runs and experiments.

### Total Levels: 250

- **50 Simple Levels** - Single switch, direct paths, minimal obstacles
- **100 Medium Levels** - 1-3 switches, simple dependencies, may require jumps
- **50 Complex Levels** - 4+ switches, complex dependencies, multi-chamber layouts
- **30 Mine-Heavy Levels** - Significant mine obstacles requiring precise navigation
- **20 Exploration Levels** - Hidden switches, extensive exploration required

## Level Categories

### Simple (50 levels)
**Purpose**: Test basic navigation and switch activation

- **Levels 0-14**: Minimal chambers with exit switch only (1-3 tiles high, 3-12 tiles wide)
  - Ninja on one side, switch in middle, door on other side
  - Flat surfaces, no jumps required
  - **No locked doors** - only exit door activation (type 3 + 4)
  - Difficulty Tier 1

- **Levels 15-24**: Single chamber with vertical deviations
  - Small vertical obstacles to navigate
  - Single exit switch controlling door
  - Difficulty Tier 2

- **Levels 25-39**: Locked door corridor - **INTRODUCES TYPE 6 DOORS**
  - **First appearance of locked doors (type 6)**
  - Layout: Ninja -> Switch -> Locked Door -> Exit Switch -> Exit
  - 16-24 tiles wide, 3-4 tiles high
  - Teaches switch dependency: must collect switch to open locked door
  - **Critical for learning switch-dependent progression**
  - Difficulty Tier 3

- **Levels 40-49**: Simple jump required
  - Small pits (2-3 tiles wide)
  - Minimal mines (0-2 per platform)
  - Difficulty Tier 4

### Medium (100 levels)
**Purpose**: Test navigation with multiple objectives and basic planning

Mixed types distributed evenly:
- **Small mazes** (25 levels): Compact maze layouts with 1-2 switches
- **Multi-chamber** (25 levels): 2 connected chambers with switch dependencies
- **Jump with mines** (25 levels): Moderate jump challenges with mine obstacles
- **Large chambers with obstacles** (25 levels): 2-3 switches with mine hazards

**Difficulty Tiers**: 1-4 (progressive difficulty across the 100 levels)

### Complex (50 levels)
**Purpose**: Test advanced planning and multi-step problem solving

Alternating types:
- **Large multi-chamber** (17 levels): 3-4 connected chambers, complex switch chains
- **Large mazes** (17 levels): Extensive maze navigation with multiple objectives
- **Complex jump sequences** (16 levels): Multi-platform navigation with switches

**Difficulty Tiers**: 1-3

### Mine-Heavy (30 levels)
**Purpose**: Test hazard avoidance and precise navigation

Alternating types:
- **Mine mazes** (15 levels): High-density mine columns requiring careful pathing
- **Mine jump levels** (15 levels): Jump sequences with heavy mine obstacles

**Difficulty Tiers**: 1-3 (increasing mine density)

### Exploration (20 levels)
**Purpose**: Test exploration strategies and discovery

Alternating types:
- **Large mazes** (10 levels): Maximum size mazes with distant objectives
- **Sprawling multi-chamber** (10 levels): 4 chambers with long corridors

**Difficulty Tiers**: 1-3

## File Format

### Directory Structure
```
test_suite/
├── simple/
│   ├── simple_000.pkl
│   ├── simple_001.pkl
│   └── ...
├── medium/
│   ├── medium_000.pkl
│   └── ...
├── complex/
│   ├── complex_000.pkl
│   └── ...
├── mine_heavy/
│   ├── mine_heavy_000.pkl
│   └── ...
├── exploration/
│   ├── exploration_000.pkl
│   └── ...
└── test_suite_metadata.json
```

### Level File Format
Each `.pkl` file contains a pickled dictionary with:
```python
{
    'level_id': str,           # e.g., 'simple_000'
    'seed': int,               # Deterministic seed used for generation
    'category': str,           # Category name
    'map_data': List[int],     # Raw map data (list of integers)
    'metadata': {
        'description': str,     # Human-readable description
        'difficulty_tier': int  # 1-4, sub-category difficulty
    }
}
```

### Metadata File
`test_suite_metadata.json` contains:
```json
{
    "total_levels": 250,
    "categories": {
        "simple": {
            "count": 50,
            "level_ids": ["simple_000", "simple_001", ...]
        },
        ...
    },
    "generation_info": {
        "script_version": "1.0",
        "description": "NPP-RL Task 3.3 comprehensive test suite",
        "deterministic": true
    }
}
```

## Usage

### Loading with TestSuiteLoader

```python
from npp_rl.evaluation import TestSuiteLoader

# Initialize loader
loader = TestSuiteLoader('/path/to/test_suite')

# Get all simple levels
simple_levels = loader.get_category('simple')

# Get a specific level
level = loader.get_level('simple_000')

# Load into environment
env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])
```

### Direct Loading

```python
import pickle

with open('test_suite/simple/simple_000.pkl', 'rb') as f:
    level = pickle.load(f)

# Access map data
map_data = level['map_data']
seed = level['seed']
description = level['metadata']['description']
```

### Running Test Suite Loader CLI

```bash
python -m npp_rl.evaluation.test_suite_loader /path/to/test_suite
```

This will display:
- Total level count
- Breakdown by category
- Sample level information

## Generation

The test suite was generated using the deterministic generation script:

```bash
cd /workspace/nclone
python -m nclone.map_generation.generate_test_suite_maps \
    --output_dir /workspace/npp-rl/datasets/test_suite
```

### Reproducibility

All maps use fixed base seeds:
- Simple: 1000-1049
- Medium: 2000-2099
- Complex: 3000-3049
- Mine-Heavy: 4000-4029
- Exploration: 5000-5019

To regenerate the exact same test suite, run the generation script with the same output directory. The deterministic seeds ensure identical maps every time.

## Evaluation Recommendations

### Baseline Testing
Test agents on the full 250-level suite to establish baseline performance across all complexity categories.

### Category-Specific Training
Train specialized agents on specific categories:
- Train on `simple` → evaluate on `medium` to test generalization
- Train on `medium` → evaluate on `complex` to test scaling
- Train on `mine_heavy` → evaluate hazard avoidance learning

### Difficulty Progression
Use difficulty tiers within each category to:
- Curriculum learning (start with tier 1, progress to tier 3-4)
- Difficulty-based performance curves
- Identify performance plateaus

### Success Metrics
For each level, track:
- Completion rate (% of levels completed)
- Average steps to completion
- Average reward
- Exploration efficiency (unique tiles visited)
- Switch activation order (for dependency analysis)

## Notes

- **Map Validity**: All maps have been verified to be loadable and playable
- **Deterministic**: Same seed always generates the same map
- **Variety**: Within each category, levels vary in layout, size, and obstacle placement
- **Baseline Quality**: Maps are generated to be challenging but solvable
- **Format Compatibility**: Map data format is compatible with nclone's `load_map_from_map_data()`

### Locked Door Implementation

This test suite includes proper locked door (type 6) entities for switch-dependent progression:

- **Entity Format**: Locked doors use 9 bytes: `[type(6), door_x, door_y, orientation, mode, switch_x, switch_y, 0, 0]`
- **Purpose**: Block passage until the player collects the associated switch
- **First Appearance**: Simple levels 25-39 introduce locked doors
- **vs Exit Doors**: Exit doors (type 3 + 4) only block the level exit, while locked doors (type 6) can block internal passages
- **Learning Objective**: Agents must learn that switches can open doors mid-level, not just at the exit

This distinction is critical for agents to understand switch dependencies and plan optimal routes through levels with multiple objectives.

## Future Enhancements

Potential expansions to the test suite:
- Additional categories (e.g., time-pressure, gold-collection focus)
- Difficulty ratings based on agent performance data
- Human playability ratings
- Solution path annotations
- Performance benchmarks from trained agents

## References

- **Task**: Phase 3, Task 3.3 - Comprehensive Evaluation Framework
- **Generation Script**: `nclone/nclone/map_generation/generate_test_suite_maps.py`
- **Loader**: `npp_rl/npp_rl/evaluation/test_suite_loader.py`
- **Documentation**: `npp-rl/docs/tasks/PHASE_3_ROBUSTNESS_OPTIMIZATION.md`
