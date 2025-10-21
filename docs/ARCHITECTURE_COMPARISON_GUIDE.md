# Architecture Comparison Guide

## Overview

These tools enable systematic comparison of different model architectures to determine which features and design choices are most effective for N++ gameplay learning.

## Key Research Questions

The architecture comparison framework addresses these questions:

1. **Graph Neural Network Simplification**: Is the full HGT complexity necessary, or can simpler GNN architectures (GAT, GCN) achieve comparable performance?

2. **Vision-Free Learning**: Can the agent learn effectively without visual input by relying solely on graph, game state, and reachability features?

3. **Modality Importance**: Which input modalities (temporal frames, global view, graph, state, reachability) contribute most to learning?

4. **Efficiency vs Performance Trade-offs**: What is the optimal balance between model complexity and inference speed?

## Architecture Variants

The framework includes 8 predefined architecture variants:

### 1. Full HGT (`full_hgt`)
- **Description**: Full Heterogeneous Graph Transformer with all modalities
- **Modalities**: Temporal frames, global view, graph (HGT), game state, reachability
- **Use Case**: Current baseline architecture
- **Expected**: Best performance but highest computational cost

### 2. Simplified HGT (`simplified_hgt`)
- **Description**: Reduced complexity HGT (fewer layers, smaller dimensions)
- **Modalities**: All modalities with reduced graph processing
- **Use Case**: Faster inference while maintaining heterogeneous graph reasoning
- **Expected**: Good balance of performance and efficiency

### 3. GAT (`gat`)
- **Description**: Graph Attention Network (homogeneous, no type-specific processing)
- **Modalities**: All modalities with GAT graph processing
- **Use Case**: Simpler attention mechanism than HGT
- **Expected**: Faster than HGT, competitive performance

### 4. GCN (`gcn`)
- **Description**: Graph Convolutional Network (simplest graph baseline)
- **Modalities**: All modalities with GCN graph processing
- **Use Case**: Minimal graph processing complexity
- **Expected**: Fastest graph-based approach, lower performance ceiling

### 5. MLP Baseline (`mlp_baseline`)
- **Description**: No graph processing, only vision and state
- **Modalities**: Temporal frames, global view, game state, reachability (no graph)
- **Use Case**: Baseline to measure graph contribution
- **Expected**: Fast but may struggle with spatial reasoning

### 6. Vision-Free (`vision_free`)
- **Description**: No visual input
- **Modalities**: Graph (HGT), game state, reachability only
- **Use Case**: Test if explicit features suffice without vision
- **Expected**: 60-70% faster inference, uncertain performance impact

### 7. No Global View (`no_global_view`)
- **Description**: Remove global view, keep local temporal frames (Scenario 1)
- **Modalities**: Temporal frames, graph, state, reachability
- **Use Case**: Test if global view is redundant with graph
- **Expected**: 30-40% faster, minimal performance loss

### 8. Local Frames Only (`local_frames_only`)
- **Description**: Local temporal awareness + graph + state
- **Modalities**: Temporal frames, graph, state, reachability
- **Use Case**: Alternative to no_global_view
- **Expected**: Similar to no_global_view

## Quick Start

### Installation

Ensure NPP-RL is properly installed with all dependencies:

```bash
cd npp-rl
pip install -r requirements.txt
```

### List Available Architectures

```bash
python tools/compare_architectures.py --list
```

### Compare All Architectures

```bash
python tools/compare_architectures.py --all
```

### Compare Specific Architectures

```bash
python tools/compare_architectures.py --architectures full_hgt vision_free simplified_hgt
```

### Save Results for Analysis

```bash
python tools/compare_architectures.py --all --save-results results/architecture_comparison/
```

## Command-Line Options

```
--architectures ARCH [ARCH ...]   Architecture names to compare
--all                             Compare all available architectures
--list                            List available architectures and exit
--iterations N                    Number of benchmark iterations (default: 100)
--batch-size N                    Batch size for inference (default: 1)
--device {auto,cuda,cpu}          Device to run benchmarks on (default: auto)
--save-results DIR                Directory to save benchmark results (JSON)
```

## Understanding Results

### Benchmark Metrics

The comparison tool measures:

1. **Inference Time**
   - Mean, std, median, p95, p99 (in milliseconds)
   - Lower is better
   - Target: <10ms for real-time gameplay

2. **Memory Usage**
   - Number of parameters
   - Parameter memory (MB)
   - Peak memory during forward pass (MB)
   - Lower is better for deployment

3. **Model Complexity**
   - Estimated FLOPs (billions)
   - Proxy for computational cost
   - Lower enables faster training

### Interpreting Comparison Table

```
Architecture         Time (ms)    Params       Memory (MB)  Modalities
--------------------------------------------------------------------------------
vision_free          2.34 ± 0.15  1,234,567    4.71         graph,state,reach
simplified_hgt       3.45 ± 0.23  2,345,678    8.94         all (reduced)
full_hgt             5.67 ± 0.34  5,678,901    21.67        all
```

**Key Observations:**
- `vision_free` is fastest and most memory-efficient (no CNN processing)
- `simplified_hgt` provides middle ground
- `full_hgt` is slowest but potentially highest performance ceiling

**Current benchmarks measure efficiency only. Full evaluation requires training on actual levels.**

## Integration with Training

### Using Selected Architecture in Training

After selecting an architecture, integrate it into training:

```python
from npp_rl.optimization import (
    get_architecture_config,
    ConfigurableMultimodalExtractor
)

# Get configuration for selected architecture
config = get_architecture_config("vision_free")  # or other architecture

# Create feature extractor
from stable_baselines3 import PPO

policy_kwargs = {
    'features_extractor_class': ConfigurableMultimodalExtractor,
    'features_extractor_kwargs': {'config': config}
}

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    # ... other hyperparameters
)
```

### Experiment Tracking

For systematic comparison across training runs:

1. Run architecture comparison to select candidates
2. Train each candidate on standardized level set
3. Track metrics:
   - Success rate per level category
   - Training timesteps to convergence
   - Final evaluation performance
   - Inference time during rollout
4. Apply weighted criteria to select final architecture

## Vision-Free Architecture Analysis

### Research Question

Can NPP-RL learn effectively without visual input by relying on:
- Graph representation (tile positions, entity locations, connectivity)
- Game state vector (velocity, surface contact, physics state, proximity)
- Reachability features (accessible areas, objective distances)

### Hypothesis

Vision may be redundant because:
- N++ uses discrete 24x24 pixel tiles → perfectly captured by graph nodes
- All entity positions explicitly in graph
- Physics state (velocity, slopes, contact) in game state vector
- Strategic navigation encoded in reachability features

### Expected Outcomes

**If vision_free performs well (within 10% of full_hgt):**
- 60-70% inference speedup
- Simpler architecture for deployment
- Graph + explicit features sufficient

**If vision_free underperforms (>20% degradation):**
- Visual processing captures critical spatial patterns
- Immediate spatial awareness requires visual input
- Hybrid approach needed (e.g., no_global_view)

### Recommended Experimental Protocol

**Phase 1: Efficiency Validation** (current)
```bash
python tools/compare_architectures.py --architectures full_hgt vision_free no_global_view
```

**Phase 2: Training Comparison** (once training set ready)
1. Train each architecture for 5M timesteps
2. Evaluate on test suite with diverse level types
3. Analyze failure modes per architecture
4. Compare learning curves

**Phase 3: Architecture Selection**
1. Apply weighted criteria (40% perf, 30% efficiency, 20% speed, 10% generalization)
2. Select winner or hybrid approach
3. Fine-tune selected architecture


## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:
```bash
python tools/compare_architectures.py --batch-size 1 --device cpu
```

### Import Errors

Ensure NPP-RL is in Python path:
```bash
export PYTHONPATH=/path/to/npp-rl:$PYTHONPATH
```
