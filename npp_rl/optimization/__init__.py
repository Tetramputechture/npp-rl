"""
Architecture optimization and comparison tools for Task 3.1.

This module provides tools for comparing different model architectures,
benchmarking performance, and analyzing which features are necessary for
effective N++ learning.
"""

from .architecture_configs import (
    ArchitectureConfig,
    ModalityConfig,
    GraphConfig,
    VisualConfig,
    StateConfig,
    FusionConfig,
    GraphArchitectureType,
    FusionType,
    get_architecture_config,
    list_available_architectures,
    print_architecture_summary,
    ARCHITECTURE_REGISTRY,
)

from .configurable_extractor import (
    ConfigurableMultimodalExtractor,
)

from .benchmarking import (
    BenchmarkResults,
    ArchitectureBenchmark,
    create_mock_observations,
)

__all__ = [
    # Configurations
    "ArchitectureConfig",
    "ModalityConfig",
    "GraphConfig",
    "VisualConfig",
    "StateConfig",
    "FusionConfig",
    "GraphArchitectureType",
    "FusionType",
    "get_architecture_config",
    "list_available_architectures",
    "print_architecture_summary",
    "ARCHITECTURE_REGISTRY",
    # Extractor
    "ConfigurableMultimodalExtractor",
    # Benchmarking
    "BenchmarkResults",
    "ArchitectureBenchmark",
    "create_mock_observations",
]