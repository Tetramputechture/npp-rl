# Architecture Optimization Module

This module provides tools for systematically comparing and optimizing model architectures for NPP-RL.

## Components

### 1. Architecture Configurations (`architecture_configs.py`)
Defines standardized configurations for different architecture variants:
- Full HGT (baseline)
- Simplified HGT (reduced complexity)
- GAT (Graph Attention Network)
- GCN (Graph Convolutional Network)
- MLP baseline (no graph)
- Vision-free (no visual input)
- No global view
- Local frames only

Each configuration specifies:
- Which modalities to use (temporal, global, graph, state, reachability)
- Graph architecture type and parameters
- Fusion mechanism
- Feature dimensions

### 2. Configurable Feature Extractor (`configurable_extractor.py`)
`ConfigurableMultimodalExtractor` - A flexible feature extractor that enables/disables modalities based on configuration. This allows testing different combinations of input features without rewriting model code.

### 3. Benchmarking Utilities (`benchmarking.py`)
Tools for measuring architecture performance:
- `ArchitectureBenchmark` - Comprehensive benchmarking suite
- `BenchmarkResults` - Structured results with metrics
- `create_mock_observations()` - Generate test data

Measured metrics:
- Inference time (mean, std, percentiles)
- Memory usage (parameters, peak memory)
- Model complexity (FLOPs estimate)

## Quick Example

```python
from npp_rl.optimization import (
    get_architecture_config,
    ConfigurableMultimodalExtractor,
    ArchitectureBenchmark,
    create_mock_observations
)
from gymnasium.spaces import Dict as SpacesDict, Box

# Get a predefined configuration
config = get_architecture_config("vision_free")

# Create observation space (simplified example)
obs_space = SpacesDict({
    "game_state": Box(low=-float('inf'), high=float('inf'), shape=(30,)),
    "reachability_features": Box(low=-float('inf'), high=float('inf'), shape=(8,)),
    # ... other components
})

# Build feature extractor with this configuration
extractor = ConfigurableMultimodalExtractor(obs_space, config)

# Benchmark it
benchmark = ArchitectureBenchmark(device='cuda')
sample_obs = create_mock_observations(batch_size=4)

results = benchmark.benchmark_model(
    model=extractor,
    sample_observations=sample_obs,
    num_iterations=100,
    architecture_name=config.name,
    config=config
)

print(f"Inference time: {results.mean_inference_time_ms:.2f} ms")
print(f"Parameters: {results.num_parameters:,}")
```

## Command-Line Tool

See `tools/compare_architectures.py` for a complete comparison tool:

```bash
# List available architectures
python tools/compare_architectures.py --list

# Compare all architectures
python tools/compare_architectures.py --all

# Compare specific architectures
python tools/compare_architectures.py --architectures full_hgt vision_free

# Save results
python tools/compare_architectures.py --all --save-results results/
```

## Adding Custom Architectures

1. Create configuration in `architecture_configs.py`:
```python
def create_my_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        name="my_arch",
        description="My custom architecture",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=False,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(architecture=GraphArchitectureType.GAT),
        # ... other configs
        features_dim=512,
    )

ARCHITECTURE_REGISTRY["my_arch"] = create_my_architecture()
```

2. Use in comparison:
```bash
python tools/compare_architectures.py --architectures my_arch full_hgt
```
