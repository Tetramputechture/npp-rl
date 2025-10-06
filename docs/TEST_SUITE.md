# Test Suite (Task 3.3)

## Location

The comprehensive test suite (250 deterministic levels) is now located in the **nclone repository**:

```
nclone/datasets/test_suite/
```

## Structure

All test suite components are consolidated in nclone:
- **Generation Script**: `nclone/map_generation/generate_test_suite_maps.py`
- **Dataset**: `nclone/datasets/test_suite/` (250 .pkl files)
- **Loader**: `nclone/evaluation/test_suite_loader.py`
- **Documentation**: 
  - `nclone/datasets/test_suite/README.md`
  - `nclone/datasets/test_suite/QUICKSTART.md`

## Usage in npp-rl

To use the test suite in your npp-rl training/evaluation:

```python
# Import from nclone
from nclone.evaluation import TestSuiteLoader

# Load test suite (provide path to nclone dataset)
loader = TestSuiteLoader('/path/to/nclone/datasets/test_suite')

# Get levels
simple_levels = loader.get_category('simple')
level = loader.get_level('simple_025')  # Locked door level

# Load into environment
env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])
```

## Dataset Composition

- **Total Levels**: 250
- **Simple**: 50 levels (progressive difficulty with locked doors)
- **Medium**: 100 levels (mazes, chambers, mines)
- **Complex**: 50 levels (large multi-chamber, complex jumps)
- **Mine-Heavy**: 30 levels (mine obstacles)
- **Exploration**: 20 levels (large exploration-focused)

## Why in nclone?

The test suite is kept in nclone to:
1. Keep generation script and dataset together
2. Make the dataset reusable across projects (not just npp-rl)
3. Centralize all N++ level generation/management tools
4. Allow npp-rl to focus on RL-specific code (training, evaluation scripts)

## Regenerating

To regenerate the test suite:

```bash
cd /path/to/nclone
python -m nclone.map_generation.generate_test_suite_maps \
  --output_dir datasets/test_suite
```

All levels use deterministic seeds (1000-5019) for reproducibility.

## Pull Requests

- **nclone PR #32**: Test suite generation and dataset
- **npp-rl PR #37**: (No longer contains dataset, only references it)

## Future Evaluation Tools

Future additions to npp-rl will include:
- Agent evaluation scripts that use the test suite
- Performance metrics collection
- Benchmark comparison tools
- Visualization utilities for test results

These will be added to `npp_rl/evaluation/` and will import from nclone.
