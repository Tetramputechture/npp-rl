# Test Suite Quick Start Guide

## Installation

The test suite is already included in this repository. No additional installation needed.

## Quick Examples

### 1. Load and Inspect a Level

```python
import pickle

# Load a level
with open('datasets/test_suite/simple/simple_000.pkl', 'rb') as f:
    level = pickle.load(f)

print(f"Level: {level['level_id']}")
print(f"Description: {level['metadata']['description']}")
print(f"Seed: {level['seed']}")
```

### 2. Use TestSuiteLoader

```python
from npp_rl.evaluation import TestSuiteLoader

# Initialize
loader = TestSuiteLoader('datasets/test_suite')

# Load a specific level
level = loader.get_level('medium_042')

# Load all levels in a category
simple_levels = loader.get_category('simple')

# Get metadata
print(f"Total levels: {loader.get_level_count()}")
print(f"Simple levels: {loader.get_level_count('simple')}")
```

### 3. Load into Environment

```python
from npp_rl.evaluation import TestSuiteLoader
from nclone.gym_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig

# Create environment
config = EnvironmentConfig.for_training()
env = NppEnvironment(config)

# Load test suite level
loader = TestSuiteLoader('datasets/test_suite')
level = loader.get_level('complex_010')
env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])

# Now play!
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

### 4. Evaluate Agent on Category

```python
from npp_rl.evaluation import TestSuiteLoader

loader = TestSuiteLoader('datasets/test_suite')

# Evaluate on all simple levels
results = []
for level in loader.get_category('simple'):
    env.unwrapped.nplay_headless.load_map_from_map_data(level['map_data'])
    obs, info = env.reset()
    
    # Run your agent
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action = agent.get_action(obs)  # Your agent here
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if truncated:
            break
    
    results.append({
        'level_id': level['level_id'],
        'completed': done and not truncated,
        'reward': total_reward,
        'steps': steps
    })

# Analyze results
completion_rate = sum(r['completed'] for r in results) / len(results)
avg_reward = sum(r['reward'] for r in results) / len(results)
print(f"Completion Rate: {completion_rate:.2%}")
print(f"Average Reward: {avg_reward:.2f}")
```

### 5. CLI Inspection

```bash
# View test suite info
python -m npp_rl.evaluation.test_suite_loader datasets/test_suite
```

Output:
```
======================================================================
NPP-RL Test Suite Information
======================================================================

Test suite directory: datasets/test_suite
Total levels: 250

Levels by category:
  simple         :  50 levels
  medium         : 100 levels
  complex        :  50 levels
  mine_heavy     :  30 levels
  exploration    :  20 levels
...
```

## Level Categories at a Glance

| Category | Count | Description | Best For |
|----------|-------|-------------|----------|
| simple | 50 | Single switch, direct paths | Basic navigation testing |
| medium | 100 | 1-3 switches, some planning | Multi-objective planning |
| complex | 50 | 4+ switches, complex layout | Advanced problem solving |
| mine_heavy | 30 | Heavy mine obstacles | Hazard avoidance |
| exploration | 20 | Large, exploration-focused | Discovery strategies |

## Common Use Cases

### Curriculum Learning
```python
# Train progressively through difficulty tiers
for tier in [1, 2, 3]:
    simple_levels = [l for l in loader.get_category('simple') 
                     if l['metadata']['difficulty_tier'] == tier]
    train_agent(simple_levels)
```

### Generalization Testing
```python
# Train on simple, test on medium
train_on = loader.get_category('simple')
test_on = loader.get_category('medium')
```

### Category-Specific Evaluation
```python
# Compare performance across categories
for category in ['simple', 'medium', 'complex', 'mine_heavy', 'exploration']:
    levels = loader.get_category(category)
    performance = evaluate_agent(agent, levels)
    print(f"{category}: {performance['completion_rate']:.2%}")
```

## Regenerating the Dataset

If you need to regenerate (creates identical levels):

```bash
cd /path/to/nclone
python -m nclone.map_generation.generate_test_suite_maps \
    --output_dir /path/to/npp-rl/datasets/test_suite
```

## File Structure

```
datasets/test_suite/
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
├── test_suite_metadata.json    # Level index
├── simple/
│   ├── simple_000.pkl
│   ├── simple_001.pkl
│   └── ... (50 files)
├── medium/
│   └── ... (100 files)
├── complex/
│   └── ... (50 files)
├── mine_heavy/
│   └── ... (30 files)
└── exploration/
    └── ... (20 files)
```

## Level ID Format

- `simple_000` to `simple_049` - Simple levels
- `medium_000` to `medium_099` - Medium levels
- `complex_000` to `complex_049` - Complex levels
- `mine_heavy_000` to `mine_heavy_029` - Mine-heavy levels
- `exploration_000` to `exploration_019` - Exploration levels

## Need More Info?

- Full documentation: See `README.md` in this directory
- Task specification: See `npp-rl/docs/tasks/PHASE_3_ROBUSTNESS_OPTIMIZATION.md`
- Loader source: See `npp_rl/evaluation/test_suite_loader.py`
- Generation script: See `nclone/nclone/map_generation/generate_test_suite_maps.py`

## Questions?

Common questions:

**Q: How do I know which levels are hardest?**
A: Check the `difficulty_tier` in each level's metadata. Higher numbers are harder within a category.

**Q: Can I add my own levels?**
A: Yes! Create a `.pkl` file with the same structure and place it in the appropriate category folder.

**Q: Are these levels solvable?**
A: All levels are generated to be structurally valid with reachable objectives. However, optimal solutions may vary.

**Q: Can I modify the generation script?**
A: Yes! The script is in `nclone/nclone/map_generation/generate_test_suite_maps.py`. You can adjust parameters or add new categories.

**Q: How do I create my own test suite?**
A: Run the generation script with a different output directory and/or modify the seed ranges.
