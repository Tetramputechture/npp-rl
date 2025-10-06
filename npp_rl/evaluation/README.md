# NPP-RL Evaluation Module

This module contains utilities for evaluating NPP-RL agents against the comprehensive test suite.

## Test Suite Location

The test suite dataset (250 deterministic levels) is located in the **nclone repository**:
```
nclone/datasets/test_suite/
```

This consolidation keeps the generation script and dataset together in nclone, while evaluation/testing utilities remain in npp-rl.

## Structure

- **test_suite_loader.py**: Utilities for loading test suite levels from the nclone dataset
- Future evaluation scripts will be added here for agent testing

## Usage

### Loading Test Suite Levels

```python
from npp_rl.evaluation.test_suite_loader import TestSuiteLoader

# Point to the nclone dataset location
loader = TestSuiteLoader('/path/to/nclone/datasets/test_suite')

# Get all simple levels
simple_levels = loader.get_category('simple')

# Get a specific level
level = loader.get_level('simple_025')  # Locked door level

# Load into environment
env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])
```

### Viewing Test Suite Info

```bash
python -m npp_rl.evaluation.test_suite_loader /path/to/nclone/datasets/test_suite
```

## Dataset Documentation

For complete documentation about the test suite structure, level categories, and generation:
- See `nclone/datasets/test_suite/README.md`
- See `nclone/datasets/test_suite/QUICKSTART.md`

## Regenerating the Dataset

To regenerate the test suite (if needed):
```bash
cd nclone
python -m nclone.map_generation.generate_test_suite_maps \
  --output_dir datasets/test_suite
```

## Future Evaluation Tools

This module will be extended with:
- Agent evaluation scripts
- Performance metrics collection
- Benchmark comparison tools
- Visualization utilities for test results
