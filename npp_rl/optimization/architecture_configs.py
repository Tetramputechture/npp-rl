"""
Architecture configuration system for Task 3.1 model optimization.

This module defines standardized configurations for different architecture variants
to enable systematic comparison and benchmarking. Each configuration specifies
which modalities to use and how to process them.

Based on Task 3.1 requirements from PHASE_3_ROBUSTNESS_OPTIMIZATION.md
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum


class GraphArchitectureType(Enum):
    """Types of graph neural network architectures."""
    FULL_HGT = "full_hgt"  # Full Heterogeneous Graph Transformer
    SIMPLIFIED_HGT = "simplified_hgt"  # Simplified HGT with reduced complexity
    GAT = "gat"  # Graph Attention Network
    GCN = "gcn"  # Graph Convolutional Network
    NONE = "none"  # No graph processing (MLP baseline)


class FusionType(Enum):
    """Types of multimodal fusion mechanisms."""
    CONCAT = "concat"  # Simple concatenation
    SINGLE_HEAD_ATTENTION = "single_head"  # Single-head cross-modal attention
    MULTI_HEAD_ATTENTION = "multi_head"  # Multi-head cross-modal attention
    HIERARCHICAL = "hierarchical"  # Hierarchical attention
    ADAPTIVE = "adaptive"  # Adaptive attention with learned gating


@dataclass(frozen=True)
class ModalityConfig:
    """Configuration for which input modalities to use."""
    use_temporal_frames: bool = True  # 3D CNN on 84x84x12 temporal frames
    use_global_view: bool = True  # 2D CNN on 176x100 global view
    use_graph: bool = True  # Graph neural network
    use_game_state: bool = True  # Game state vector (26 features after redundancy removal)
    use_reachability: bool = True  # Reachability features (8 features)
    
    def get_enabled_modalities(self) -> List[str]:
        """Return list of enabled modality names."""
        modalities = []
        if self.use_temporal_frames:
            modalities.append("temporal")
        if self.use_global_view:
            modalities.append("global")
        if self.use_graph:
            modalities.append("graph")
        if self.use_game_state:
            modalities.append("state")
        if self.use_reachability:
            modalities.append("reachability")
        return modalities
    
    def count_modalities(self) -> int:
        """Return count of enabled modalities."""
        return sum([
            self.use_temporal_frames,
            self.use_global_view,
            self.use_graph,
            self.use_game_state,
            self.use_reachability
        ])


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for graph neural network architecture."""
    architecture: GraphArchitectureType = GraphArchitectureType.FULL_HGT
    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 256
    num_heads: int = 8  # For attention-based architectures
    dropout: float = 0.1
    
    # HGT-specific: node and edge types
    num_node_types: int = 6  # tile, ninja, mine, exit_switch, exit_door, locked_door
    num_edge_types: int = 2  # adjacent, reachable (simplified from 3)
    
    # Simplification options
    use_type_embeddings: bool = True  # Whether to use separate embeddings per type
    use_edge_features: bool = True  # Whether to use edge features


@dataclass(frozen=True)
class VisualConfig:
    """Configuration for visual processing (CNNs)."""
    # 3D CNN (temporal frames)
    temporal_output_dim: int = 512
    temporal_channels: tuple[int, int, int] = (32, 64, 128)
    
    # 2D CNN (global view)
    global_output_dim: int = 256
    global_channels: tuple[int, int, int] = (32, 64, 128)
    
    # Dropout rates
    cnn_dropout: float = 0.1


@dataclass(frozen=True)
class StateConfig:
    """Configuration for state vector processing."""
    game_state_dim: int = 26  # Input dimension (after redundancy removal)
    reachability_dim: int = 8  # Input dimension
    hidden_dim: int = 128
    output_dim: int = 128


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for multimodal fusion."""
    fusion_type: FusionType = FusionType.MULTI_HEAD_ATTENTION
    num_attention_heads: int = 8
    dropout: float = 0.1
    use_residual: bool = True


@dataclass(frozen=True)
class ArchitectureConfig:
    """
    Complete architecture configuration for model variants.
    
    This dataclass encapsulates all parameters needed to construct
    a specific architecture variant for comparison.
    """
    name: str
    description: str
    modalities: ModalityConfig
    graph: GraphConfig
    visual: VisualConfig
    state: StateConfig
    fusion: FusionConfig
    features_dim: int = 512  # Final output dimension
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            "name": self.name,
            "description": self.description,
            "modalities": self.modalities.__dict__,
            "graph": self.graph.__dict__,
            "visual": self.visual.__dict__,
            "state": self.state.__dict__,
            "fusion": self.fusion.__dict__,
            "features_dim": self.features_dim,
        }


# ===== Predefined Architecture Variants =====

def create_full_hgt_config() -> ArchitectureConfig:
    """Full HGT architecture with all modalities (current baseline)."""
    return ArchitectureConfig(
        name="full_hgt",
        description="Full HGT with all modalities: temporal frames, global view, graph, state, reachability",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=True,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.FULL_HGT,
            hidden_dim=256,
            num_layers=3,
            output_dim=256,
            num_heads=8,
        ),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.MULTI_HEAD_ATTENTION),
        features_dim=512,
    )


def create_simplified_hgt_config() -> ArchitectureConfig:
    """Simplified HGT with reduced complexity."""
    return ArchitectureConfig(
        name="simplified_hgt",
        description="Simplified HGT: reduced layers and dimensions",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=True,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.SIMPLIFIED_HGT,
            hidden_dim=128,
            num_layers=2,
            output_dim=128,
            num_heads=4,
        ),
        visual=VisualConfig(
            temporal_output_dim=256,
            global_output_dim=128,
        ),
        state=StateConfig(hidden_dim=64, output_dim=64),
        fusion=FusionConfig(
            fusion_type=FusionType.SINGLE_HEAD_ATTENTION,
            num_attention_heads=4,
        ),
        features_dim=256,
    )


def create_gat_config() -> ArchitectureConfig:
    """GAT architecture as alternative to HGT."""
    return ArchitectureConfig(
        name="gat",
        description="Graph Attention Network with all modalities",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=True,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.GAT,
            hidden_dim=256,
            num_layers=3,
            output_dim=256,
            num_heads=8,
            use_type_embeddings=False,  # GAT doesn't use heterogeneous types
        ),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.SINGLE_HEAD_ATTENTION),
        features_dim=512,
    )


def create_gcn_config() -> ArchitectureConfig:
    """GCN architecture as simpler graph baseline."""
    return ArchitectureConfig(
        name="gcn",
        description="Graph Convolutional Network (simplest graph baseline)",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=True,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.GCN,
            hidden_dim=256,
            num_layers=3,
            output_dim=256,
            use_type_embeddings=False,
            use_edge_features=False,
        ),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=512,
    )


def create_mlp_baseline_config() -> ArchitectureConfig:
    """MLP baseline without graph processing."""
    return ArchitectureConfig(
        name="mlp_baseline",
        description="MLP baseline: no graph processing, only vision and state",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=True,
            use_graph=False,  # No graph processing
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(architecture=GraphArchitectureType.NONE),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=512,
    )


def create_vision_free_config() -> ArchitectureConfig:
    """Vision-free architecture: graph + state only (Task 3.1 research question)."""
    return ArchitectureConfig(
        name="vision_free",
        description="Vision-free: graph + state + reachability only",
        modalities=ModalityConfig(
            use_temporal_frames=False,  # No vision
            use_global_view=False,  # No vision
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.FULL_HGT,
            hidden_dim=256,
            num_layers=3,
            output_dim=256,
        ),
        visual=VisualConfig(),  # Not used
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=384,  # Smaller since no vision
    )


def create_no_global_view_config() -> ArchitectureConfig:
    """Remove global view only (Task 3.1 Scenario 1)."""
    return ArchitectureConfig(
        name="no_global_view",
        description="Full architecture but without global view (Scenario 1 from Task 3.1)",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=False,  # Remove global view
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.FULL_HGT,
            hidden_dim=256,
            num_layers=3,
            output_dim=256,
        ),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.MULTI_HEAD_ATTENTION),
        features_dim=512,
    )


def create_local_frames_only_config() -> ArchitectureConfig:
    """Keep only local temporal frames for immediate spatial awareness."""
    return ArchitectureConfig(
        name="local_frames_only",
        description="Temporal frames + graph + state (no global view)",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=False,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(architecture=GraphArchitectureType.FULL_HGT),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.MULTI_HEAD_ATTENTION),
        features_dim=512,
    )


# ===== Configuration Registry =====

ARCHITECTURE_REGISTRY: Dict[str, ArchitectureConfig] = {
    "full_hgt": create_full_hgt_config(),
    "simplified_hgt": create_simplified_hgt_config(),
    "gat": create_gat_config(),
    "gcn": create_gcn_config(),
    "mlp_baseline": create_mlp_baseline_config(),
    "vision_free": create_vision_free_config(),
    "no_global_view": create_no_global_view_config(),
    "local_frames_only": create_local_frames_only_config(),
}


def get_architecture_config(name: str) -> ArchitectureConfig:
    """
    Get architecture configuration by name.
    
    Args:
        name: Architecture name (e.g., 'full_hgt', 'vision_free')
        
    Returns:
        ArchitectureConfig object
        
    Raises:
        ValueError if architecture name not found
    """
    if name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(ARCHITECTURE_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture '{name}'. Available: {available}"
        )
    return ARCHITECTURE_REGISTRY[name]


def list_available_architectures() -> List[str]:
    """Return list of available architecture names."""
    return list(ARCHITECTURE_REGISTRY.keys())


def print_architecture_summary(config: ArchitectureConfig) -> None:
    """Print human-readable summary of architecture configuration."""
    print(f"\n{'='*60}")
    print(f"Architecture: {config.name}")
    print(f"{'='*60}")
    print(f"Description: {config.description}")
    print(f"\nModalities ({config.modalities.count_modalities()} enabled):")
    for modality in config.modalities.get_enabled_modalities():
        print(f"  ✓ {modality}")
    print(f"\nGraph Architecture: {config.graph.architecture.value}")
    if config.modalities.use_graph:
        print(f"  - Hidden dim: {config.graph.hidden_dim}")
        print(f"  - Num layers: {config.graph.num_layers}")
        print(f"  - Num heads: {config.graph.num_heads}")
    print(f"\nFusion Type: {config.fusion.fusion_type.value}")
    print(f"Final Feature Dimension: {config.features_dim}")
    print(f"{'='*60}\n")
