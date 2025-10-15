# Task: Optimize Graph Processing Performance and Add Lightweight Vision-Free Architectures

## Context

We've successfully fixed dimension mismatches in graph observation spaces and tensor type errors that were preventing training. However, we've discovered a critical performance issue: **graph-based architectures hang/are extremely slow on CPU during training**, while non-graph architectures (like `mlp_baseline`) work fine.

## Current Status

### ✅ Working Architectures
- **mlp_baseline**: Successfully trains 100 timesteps in ~20 seconds on CPU
  - Uses: temporal frames, global view, game state, reachability (NO GRAPH)

### ❌ Problematic Architectures (All Graph-Based)
The following architectures hang during training (likely in the first rollout collection):
- **vision_free**: Uses graph (full_hgt), game state, reachability
- **full_hgt**: Uses graph (full_hgt) + all modalities
- **gat**: Uses graph (gat) + all modalities
- **gcn**: Uses graph (gcn) + all modalities
- **simplified_hgt**: Uses graph (simplified_hgt) + all modalities
- **local_frames_only**: Uses graph (full_hgt) + temporal frames, game state, reachability
- **no_global_view**: Uses graph (full_hgt) + temporal frames, game state, reachability

### Recent Fixes Applied
1. **Fixed graph observation space dimensions** in `nclone/gym_environment/npp_environment.py`:
   - Changed from hardcoded `(N_MAX_NODES, 3)` to `(N_MAX_NODES, NODE_FEATURE_DIM)` where `NODE_FEATURE_DIM=55`
   - Changed from hardcoded `(E_MAX_EDGES, 1)` to `(E_MAX_EDGES, EDGE_FEATURE_DIM)` where `EDGE_FEATURE_DIM=6`

2. **Fixed tensor indexing** in `npp_rl/feature_extractors/configurable_extractor.py`:
   - Added `.long()` conversion for `edge_index` tensors in all graph encoders (HGT, GAT, GCN)
   - Fixed PyTorch error: "tensors used as indices must be long, int, byte or bool tensors"

3. **Added dynamic n_steps/batch_size adjustment** for minimal training runs:
   - Modified `architecture_trainer.py` to handle very small `total_timesteps` values
   - Added `--skip-final-eval` flag for quick validation tests

## The Problem

### Graph Dimensions
```python
N_MAX_NODES = 15856        # Maximum nodes in graph
E_MAX_EDGES = 126848       # Maximum edges (N_MAX_NODES * 8)
NODE_FEATURE_DIM = 55      # Node features per node
EDGE_FEATURE_DIM = 6       # Edge features per edge

# Total size per observation:
# - Node features: 15856 * 55 = 872,080 values (~0.83MB as float32)
# - Edge features: 126848 * 6 = 761,088 values (~0.73MB as float32)
```

The graph processing appears to be extremely slow or hanging on CPU, likely due to:
- Large graph sizes (15,856 nodes, 126,848 edges)
- Complex graph neural network operations (especially HGT with heterogeneous graph transformers)
- CPU-only execution (no GPU acceleration for graph operations)

## Your Tasks

### Task 1: Investigate Graph Dimension Reduction 🔍

**Objective**: Determine if we can reduce graph dimensions without losing critical information.

**Investigation Steps**:
1. Analyze actual graph usage in typical N++ levels:
   - What's the average number of nodes/edges in real levels?
   - How many levels actually use close to N_MAX_NODES?
   - Profile memory and compute usage during graph processing

2. Explore dimension reduction strategies:
   - Can we use dynamic graphs that only include relevant nodes/edges?
   - Can we reduce NODE_FEATURE_DIM (currently 55) by removing redundant features?
   - Can we reduce EDGE_FEATURE_DIM (currently 6)?
   - Can we use graph pooling or coarsening techniques?

3. Benchmark different graph sizes:
   - Test training speed with reduced dimensions
   - Measure impact on model performance
   - Find optimal trade-off between speed and expressiveness

**Key Files to Examine**:
- `nclone/graph/common.py` - Graph dimension constants
- `nclone/graph/feature_extraction.py` - Node/edge feature computation
- `nclone/gym_environment/npp_environment.py` - Graph observation space definition
- `npp_rl/feature_extractors/configurable_extractor.py` - Graph encoder implementations

**Expected Output**:
- Report documenting current vs. proposed graph dimensions
- Performance benchmarks (training speed, memory usage)
- Recommendation on dimension reduction approach

### Task 2: Optimize Graph Processing Performance ⚡

**Objective**: Make graph-based architectures train efficiently on CPU.

**Optimization Strategies**:
1. **Profile the bottleneck**:
   - Use `py-spy` or `cProfile` to identify where time is spent
   - Check if it's in graph construction, feature extraction, or GNN forward pass

2. **CPU-specific optimizations**:
   - Consider using sparse tensor operations for graph processing
   - Investigate PyTorch Geometric CPU performance settings
   - Check if graph operations can be batched more efficiently

3. **Architecture-specific fixes**:
   - HGT (Heterogeneous Graph Transformer) is particularly complex - can we simplify?
   - Check if attention mechanisms in GAT are causing the slowdown
   - Consider using message passing networks (MPNs) instead of attention

4. **Code-level optimizations**:
   - Review `configurable_extractor.py` lines 354, 369, 385 where edge_index conversion happens
   - Check for unnecessary tensor copies or device transfers
   - Look for opportunities to cache graph computations

**Key Files**:
- `npp_rl/feature_extractors/configurable_extractor.py` - Main graph encoders
- `npp_rl/models/hgt_encoder.py`, `hgt_layer.py` - HGT implementation
- `npp_rl/models/gat.py` - Graph Attention Network
- `npp_rl/models/gcn.py` - Graph Convolutional Network

**Expected Output**:
- Performance profile showing bottlenecks
- Optimized graph encoder implementations
- Benchmark showing training speed improvement

### Task 3: Add Lightweight Vision-Free Architectures 🆕

**Objective**: Create alternative vision-free architectures using GAT and GCN instead of full HGT.

**Implementation Plan**:

1. **Add `vision_free_gat` architecture**:
   ```python
   # In npp_rl/training/architecture_configs.py
   ArchitectureConfig(
       name="vision_free_gat",
       description="Vision-free with Graph Attention Network (lighter than HGT)",
       use_temporal_frames=False,
       use_global_view=False,
       use_graph=True,
       graph_encoder_type="gat",  # Use GAT instead of full_hgt
       use_game_state=True,
       use_reachability=True,
       features_dim=384,
   )
   ```

2. **Add `vision_free_gcn` architecture**:
   ```python
   ArchitectureConfig(
       name="vision_free_gcn",
       description="Vision-free with Graph Convolutional Network (fastest graph option)",
       use_temporal_frames=False,
       use_global_view=False,
       use_graph=True,
       graph_encoder_type="gcn",  # Use GCN for maximum speed
       use_game_state=True,
       use_reachability=True,
       features_dim=256,  # Even smaller for efficiency
   )
   ```

3. **Add `vision_free_simplified` architecture**:
   ```python
   ArchitectureConfig(
       name="vision_free_simplified",
       description="Vision-free with simplified HGT (reduced complexity)",
       use_temporal_frames=False,
       use_global_view=False,
       use_graph=True,
       graph_encoder_type="simplified_hgt",
       use_game_state=True,
       use_reachability=True,
       features_dim=256,
   )
   ```

**Testing Requirements**:
- All new architectures must train successfully with minimal settings:
  ```bash
  python scripts/train_and_compare.py \
      --experiment-name "vision_free_test" \
      --architectures vision_free_gat vision_free_gcn vision_free_simplified \
      --no-pretraining \
      --train-dataset ../nclone/datasets/train \
      --test-dataset ../nclone/datasets/test \
      --total-timesteps 100 \
      --num-envs 2 \
      --skip-final-eval \
      --output-dir experiments/
  ```
- Training should complete in under 1 minute on CPU for 100 timesteps

**Key Files to Modify**:
- `npp_rl/training/architecture_configs.py` - Add new architecture configs
- Update `scripts/list_architectures.py` output to show new architectures

**Expected Output**:
- 3 new working vision-free architectures
- Performance comparison showing training speed vs. original vision_free
- Documentation in `docs/ARCHITECTURE_COMPARISON_GUIDE.md` about when to use each

### Task 4: Validate All Architectures 🧪

**Objective**: Ensure all architectures (existing + new) can train successfully.

**Validation Script**:
Create a comprehensive test that validates all architectures:

```bash
# Test all non-graph architectures (should be fast)
python scripts/train_and_compare.py \
    --experiment-name "validation_no_graph" \
    --architectures mlp_baseline \
    --total-timesteps 100 \
    --num-envs 2 \
    --skip-final-eval \
    --output-dir experiments/validation/

# Test lightweight graph architectures (after optimization)
python scripts/train_and_compare.py \
    --experiment-name "validation_light_graph" \
    --architectures vision_free_gcn vision_free_gat \
    --total-timesteps 100 \
    --num-envs 2 \
    --skip-final-eval \
    --output-dir experiments/validation/

# Test heavy graph architectures (may need more optimization)
python scripts/train_and_compare.py \
    --experiment-name "validation_heavy_graph" \
    --architectures vision_free full_hgt gat gcn \
    --total-timesteps 100 \
    --num-envs 2 \
    --skip-final-eval \
    --output-dir experiments/validation/
```

**Success Criteria**:
- All architectures complete 100 timesteps of training without hanging
- Training time on CPU: <1 min for non-graph, <2 min for light graph, <5 min for heavy graph
- No errors or warnings related to tensor dimensions or types

## Technical References

### Graph Neural Network Background
- **GCN (Graph Convolutional Network)**: Simplest, fastest graph encoder. Good baseline.
- **GAT (Graph Attention Network)**: Medium complexity with attention mechanism. Good balance.
- **HGT (Heterogeneous Graph Transformer)**: Most complex, handles different node/edge types. Slowest but most expressive.

### Relevant Documentation
- `docs/ARCHITECTURE_COMPARISON_GUIDE.md` - Architecture selection guide
- `docs/OBSERVATION_SPACE_GUIDE.md` - Observation space details including graphs
- `docs/TRAINING_SYSTEM.md` - Training system architecture

### Related Code Locations
```
npp-rl/
├── npp_rl/
│   ├── feature_extractors/
│   │   └── configurable_extractor.py         # Lines 354, 369, 385 (edge_index fixes)
│   ├── models/
│   │   ├── hgt_encoder.py, hgt_layer.py     # HGT implementation
│   │   ├── gat.py                            # GAT implementation
│   │   ├── gcn.py                            # GCN implementation
│   │   └── simplified_hgt.py                 # Lightweight HGT variant
│   └── training/
│       ├── architecture_configs.py           # Architecture definitions
│       └── architecture_trainer.py           # Training orchestration

nclone/
└── nclone/
    ├── graph/
    │   ├── common.py                         # Graph dimension constants
    │   └── feature_extraction.py             # Node/edge feature computation
    └── gym_environment/
        └── npp_environment.py                # Lines 133-146 (observation space fix)
```

## Testing & Validation

### Current Validation Results
```bash
# ✅ WORKING: MLP Baseline (no graph)
$ python scripts/train_and_compare.py \
    --experiment-name "mlp_test" \
    --architectures mlp_baseline \
    --total-timesteps 100 \
    --num-envs 2 \
    --skip-final-eval
# Result: SUCCESS - 100 timesteps in ~20 seconds

# ❌ HANGING: Vision-Free (uses HGT)
$ python scripts/train_and_compare.py \
    --experiment-name "vision_free_test" \
    --architectures vision_free \
    --total-timesteps 100 \
    --num-envs 2 \
    --skip-final-eval
# Result: HANGS - No progress after "Starting training"
```

### Debugging Commands
```bash
# Profile graph processing
python -m cProfile -o profile.stats scripts/train_and_compare.py \
    --architectures vision_free --total-timesteps 100 --num-envs 2 --skip-final-eval

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(50)"

# Check actual graph sizes in dataset
python -c "
from nclone.gym_environment import create_training_env
from nclone.gym_environment.config import EnvironmentConfig
import numpy as np

config = EnvironmentConfig(level_dir='../nclone/datasets/train/simple')
env = create_training_env(config)
obs, _ = env.reset()
print(f\"Node mask sum: {obs['graph_obs']['node_mask'].sum()} / {len(obs['graph_obs']['node_mask'])} nodes\")
print(f\"Edge mask sum: {obs['graph_obs']['edge_mask'].sum()} / {len(obs['graph_obs']['edge_mask'])} edges\")
"
```

## Success Metrics

1. **Performance**:
   - All architectures complete 100 timesteps training on CPU without hanging
   - Graph-based architectures train within reasonable time (<5 min for 100 steps)

2. **New Architectures**:
   - 3 new vision-free architectures (GAT, GCN, simplified_hgt) working
   - Performance comparison documented

3. **Optimization**:
   - Identified and resolved performance bottleneck
   - Graph dimension reduction strategy (if applicable)
   - Performance improvements quantified (e.g., "2x faster", "50% memory reduction")

4. **Documentation**:
   - Updated architecture comparison guide with new vision-free options
   - Performance benchmarks for all architectures
   - Recommendations for CPU vs. GPU training

## Branch & Commit Information

**Current Branch**: `fix/cpu-minimal-training-validation`
**Base Branch**: `main`

**Changes Already Made**:
- Fixed graph observation space dimensions in nclone
- Fixed edge_index tensor type conversion
- Added dynamic n_steps/batch_size adjustment
- Added --skip-final-eval flag

**Your Changes Should**:
- Build on existing fixes
- Focus on optimization and new architectures
- Include comprehensive tests
- Update relevant documentation

## Getting Started

1. **Checkout the branch**:
   ```bash
   cd /workspace/npp-rl
   git pull origin fix/cpu-minimal-training-validation
   ```

2. **Reproduce the issue**:
   ```bash
   # This should work (mlp_baseline)
   python scripts/train_and_compare.py --experiment-name "test_mlp" \
       --architectures mlp_baseline --total-timesteps 100 --num-envs 2 --skip-final-eval
   
   # This should hang (vision_free)
   timeout 60 python scripts/train_and_compare.py --experiment-name "test_vision_free" \
       --architectures vision_free --total-timesteps 100 --num-envs 2 --skip-final-eval
   ```

3. **Profile the bottleneck**:
   ```bash
   pip install py-spy  # If not already installed
   
   # Run with profiling
   py-spy record -o profile.svg -- python scripts/train_and_compare.py \
       --architectures vision_free --total-timesteps 100 --num-envs 2 --skip-final-eval
   ```

4. **Start with Task 1 or Task 2** (investigation/optimization) before adding new architectures

## Questions or Issues?

- Check `docs/TRAINING_SYSTEM.md` for training architecture details
- Check `docs/OBSERVATION_SPACE_GUIDE.md` for graph observation space details
- Review recent commits on this branch for context on fixes already applied
- Test changes incrementally - don't optimize and add features at the same time

---

**Document Version**: 1.0  
**Created**: 2025-10-15  
**Last Updated**: 2025-10-15  
**Author**: OpenHands Agent (CPU validation task)
