"""
Architecture configuration system for model optimization.

This module defines standardized configurations for different architecture variants
to enable systematic comparison and benchmarking. Each configuration specifies
which modalities to use and how to process them.

Note: PBRS (Potential-Based Reward Shaping) is enabled by default in training
to provide dense reward signals for improved learning. Graph building for PBRS
path distance calculations is automatically enabled when PBRS is active, but
graph observations are separately controlled based on whether the architecture
uses graph modality (see ModalityConfig.use_graph).
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
    ADAPTIVE = "adaptive"  # Adaptive attention with learned gating


@dataclass(frozen=True)
class ModalityConfig:
    """Configuration for which input modalities to use."""

    use_player_frame: bool = False  # 2D CNN on 84x84x1 single grayscale frame
    use_global_view: bool = False  # 2D CNN on 176x100x1 grayscale global view
    use_graph: bool = True  # Graph neural network
    use_game_state: bool = (
        True  # Game state vector (NINJA_STATE_DIM=29, ninja physics only)
    )
    use_reachability: bool = True  # Reachability features (6 features)

    def get_enabled_modalities(self) -> List[str]:
        """Return list of enabled modality names."""
        modalities = []
        if self.use_player_frame:
            modalities.append("player_frame")
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
        return sum(
            [
                self.use_player_frame,
                self.use_global_view,
                self.use_graph,
                self.use_game_state,
                self.use_reachability,
            ]
        )


@dataclass(frozen=True)
class GraphConfig:
    """
    Configuration for graph neural network architecture.

    Note: Input dimensions come from nclone (GCN-optimized):
    - node_feature_dim = 6 (NODE_FEATURE_DIM from nclone.graph.common)
      6 dims: 2 spatial + 2 mine + 2 entity_state
      (reachability removed - all nodes in graph are reachable via flood fill)
    - edge_feature_dim = 0 (no edge features for GCN)
    - No node/edge types (GCN learns from features and structure)
    These are used in ConfigurableMultimodalExtractor._create_graph_encoder()
    """

    architecture: GraphArchitectureType = GraphArchitectureType.FULL_HGT
    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 256
    num_heads: int = 8  # For attention-based architectures
    dropout: float = 0.1

    # HGT-specific: node and edge types
    num_node_types: int = 6  # EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT
    num_edge_types: int = 4  # ADJACENT, REACHABLE, FUNCTIONAL, BLOCKED

    # Simplification options
    use_type_embeddings: bool = True  # Whether to use separate embeddings per type
    use_edge_features: bool = True  # Whether to use edge features


@dataclass(frozen=True)
class VisualConfig:
    """Configuration for visual processing (CNNs)."""

    # 2D CNN (single grayscale frame - player view)
    # Changed from 3D CNN (12 frames) for 6.66x speedup and 50% memory reduction
    player_frame_output_dim: int = 512
    player_frame_channels: tuple[int, int, int] = (32, 64, 128)

    # 2D CNN (global view - also grayscale now)
    global_output_dim: int = 256
    global_channels: tuple[int, int, int] = (32, 64, 128)

    # Dropout rates
    cnn_dropout: float = 0.1


@dataclass(frozen=True)
class StateConfig:
    """Configuration for state vector processing."""

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
    detailed_description: (
        str  # Comprehensive description for human-readable documentation
    )
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
            "detailed_description": self.detailed_description,
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
        description="Full HGT with all modalities: player frame, global view, graph, state, reachability",
        detailed_description="""
        Baseline architecture with maximum capacity. Uses all modalities for comprehensive context.
        
        Modalities:
        - Player Frame: 512-dim (84x84x1 single grayscale frame for local spatial awareness)
        - Global View: 256-dim (176x100x1 grayscale full level for strategic planning)
        - Graph (Full HGT): 3 layers, 256 hidden, 8 heads (heterogeneous types)
        - Game State: 128-dim (29-dim ninja physics: position, velocity, normals, etc.)
        - Reachability: 6-dim (4 base + 2 mine context)
        
        Multi-head cross-modal attention fusion (8 heads) learns complex interactions between modalities.
        Highest capacity but largest computational cost. Feature dim: 512.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
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
        detailed_description="""
        Lighter HGT variant balancing performance and efficiency. All modalities with reduced capacity.
        
        Modalities (vs full_hgt):
        - Player Frame: 256-dim (vs 512, single grayscale)
        - Global View: 128-dim (vs 256)
        - Graph (Simplified HGT): 2 layers (vs 3), 128 hidden (vs 256), 4 heads (vs 8)
        - Game State: 64-dim (vs 128, from 29-dim ninja physics)
        - Reachability: 6-dim (4 base + 2 mine context)
        
        Single-head attention fusion (4 heads). Feature dim: 256 (vs 512). Faster training and inference
        with limited resources.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
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
            player_frame_output_dim=256,
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
        detailed_description="""
        Homogeneous graph alternative to HGT. All modalities with GAT for graph processing.
        
        Modalities:
        - Player Frame: 512-dim (84x84x1 grayscale)
        - Global View: 256-dim
        - Graph (GAT): 3 layers, 256 hidden, 8 heads (homogeneous - no type-specific parameters)
        - Game State: 128-dim
        - Reachability: 8-dim
        
        GAT uses attention for aggregation but treats all nodes/edges uniformly, reducing complexity
        vs HGT. Single-head cross-modal attention fusion. Feature dim: 512. Compares heterogeneous
        vs homogeneous graph processing.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
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
        detailed_description="""
        Simplest graph architecture. All modalities with basic GCN (mean aggregation only).
        
        Modalities:
        - Player Frame: 512-dim (84x84x1 grayscale)
        - Global View: 256-dim
        - Graph (GCN): 3 layers, 256 hidden (no attention, no edge features, no type embeddings)
        - Game State: 128-dim
        - Reachability: 8-dim
        
        GCN uses simple mean aggregation. Most lightweight graph approach. Concatenation fusion.
        Feature dim: 512. Tests if sophisticated graph mechanisms (HGT, GAT) provide benefits over
        basic convolutions.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
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


def create_mlp_cnn_config() -> ArchitectureConfig:
    """MLP baseline without graph processing."""
    return ArchitectureConfig(
        name="mlp_cnn",
        description="MLP baseline: no graph processing, only vision and state",
        detailed_description="""
        Non-graph baseline measuring GNN contribution. Vision and state only.
        
        Modalities:
        - Player Frame: 512-dim (84x84x1 grayscale)
        - Global View: 256-dim
        - Graph: DISABLED
        - Game State: 128-dim
        - Reachability: 8-dim
        
        Concatenation fusion. Feature dim: 512. Tests if graph-based relational reasoning is necessary
        or if spatial relationships can be learned from visual inputs alone.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
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
    """Vision-free architecture: graph + state only."""
    return ArchitectureConfig(
        name="vision_free",
        description="Vision-free: graph + state + reachability only",
        detailed_description="""
        Vision-free with symbolic representations only. 5-10x faster inference than full_hgt.
        
        Modalities:
        - Player Frames: DISABLED
        - Global View: DISABLED
        - Graph (Full HGT): 3 layers, 256 hidden, 8 heads
        - Game State: 128-dim
        - Reachability: 8-dim
        
        No CNN processing for maximum speed. Concatenation fusion. Feature dim: 384. Tests if
        structured graph representations suffice without pixel-based vision. Trade-off: potentially
        reduced spatial awareness of fine-grained obstacles.
        """,
        modalities=ModalityConfig(
            use_player_frame=False,  # No vision
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
    """Remove global view only."""
    return ArchitectureConfig(
        name="no_global_view",
        description="Full architecture without global view (local frames + graph + state)",
        detailed_description="""
        Ablation removing global view to test if local frames + graph suffice for navigation.
        
        Modalities:
        - Player Frames: 512-dim (84x84x1 grayscale local view)
        - Global View: DISABLED
        - Graph (Full HGT): 3 layers, 256 hidden, 8 heads
        - Game State: 128-dim
        - Reachability: 8-dim
        
        Tests whether graph structure compensates for lack of strategic level-wide context.
        Multi-head attention fusion (8 heads). Feature dim: 512.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
            use_global_view=False,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.GCN,
            hidden_dim=256,
            num_layers=3,
            output_dim=256,
        ),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.MULTI_HEAD_ATTENTION),
        features_dim=512,
    )


def create_vision_free_gat_config() -> ArchitectureConfig:
    """Vision-free architecture using GAT instead of HGT (lighter)."""
    return ArchitectureConfig(
        name="vision_free_gat",
        description="Vision-free with Graph Attention Network (lighter than HGT)",
        detailed_description="""
        Vision-free with GAT for lighter computation than HGT. Symbolic only.
        
        Modalities:
        - Player Frames: DISABLED
        - Global View: DISABLED
        - Graph (GAT): 3 layers, 256 hidden, 8 heads (homogeneous)
        - Game State: 128-dim
        - Reachability: 8-dim
        
        GAT reduces complexity vs HGT while maintaining attention-based aggregation. No type-specific
        parameters. Concatenation fusion. Feature dim: 384. Tests if heterogeneous types (HGT) are
        necessary in vision-free settings.
        """,
        modalities=ModalityConfig(
            use_player_frame=False,  # No vision
            use_global_view=False,  # No vision
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
        visual=VisualConfig(),  # Not used
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=384,  # Smaller since no vision
    )


def create_vision_free_gcn_config() -> ArchitectureConfig:
    """Vision-free architecture using GCN (fastest graph option)."""
    return ArchitectureConfig(
        name="vision_free_gcn",
        description="Vision-free with Graph Convolutional Network (fastest graph option)",
        detailed_description="""
        Fastest architecture. Vision-free with simplest graph processing. 10-15x faster than full_hgt.
        
        Modalities:
        - Player Frames: DISABLED
        - Global View: DISABLED
        - Graph (GCN): 3 layers, 256 hidden (mean aggregation, no attention/edge features/types)
        - Game State: 128-dim
        - Reachability: 8-dim
        
        Most lightweight overall. Concatenation fusion. Feature dim: 256 (vs 384/512). Ideal for
        real-time deployment where milliseconds matter. Trades performance for extreme efficiency.
        """,
        modalities=ModalityConfig(
            use_player_frame=False,  # No vision
            use_global_view=False,  # No vision
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
        visual=VisualConfig(),  # Not used
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=256,  # Even smaller for efficiency
    )


def create_vision_free_simplified_config() -> ArchitectureConfig:
    """Vision-free architecture using simplified HGT (reduced complexity)."""
    return ArchitectureConfig(
        name="vision_free_simplified",
        description="Vision-free with simplified HGT (reduced complexity)",
        detailed_description="""
        Vision-free with reduced HGT. Balances speed and capability between vision_free and vision_free_gcn.
        
        Modalities:
        - Player Frames: DISABLED
        - Global View: DISABLED
        - Graph (Simplified HGT): 2 layers (vs 3), 128 hidden (vs 256), 4 heads (vs 8)
        - Game State: 64-dim (vs 128)
        - Reachability: 8-dim
        
        Maintains heterogeneous processing with reduced capacity. Faster than vision_free (full HGT),
        more expressive than vision_free_gcn. Concatenation fusion. Feature dim: 256 (vs 384).
        For resource-constrained deployment needing heterogeneous reasoning.
        """,
        modalities=ModalityConfig(
            use_player_frame=False,  # No vision
            use_global_view=False,  # No vision
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
        visual=VisualConfig(),  # Not used
        state=StateConfig(hidden_dim=64, output_dim=64),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=256,
    )


def create_full_hgt_frame_stacked_config() -> ArchitectureConfig:
    """Full HGT architecture designed for use with frame stacking.

    Note: This config defines the architecture that processes stacked frames.
    Frame stacking must be enabled separately via command-line flags:
        --enable-visual-frame-stacking --visual-stack-size 4
        --enable-state-stacking --state-stack-size 4
    """
    return ArchitectureConfig(
        name="full_hgt_frame_stacked",
        description="Full HGT with 4-frame stacking for temporal dynamics (velocity, acceleration)",
        detailed_description="""
        Full HGT architecture enhanced with frame stacking for temporal reasoning.
        Follows DQN approach (Mnih et al. 2015) of stacking 4 frames to capture motion.
        
        IMPORTANT: Frame stacking is NOT automatically enabled by this config.
        You must also pass frame stacking flags when training:
            --enable-visual-frame-stacking --visual-stack-size 4
            --enable-state-stacking --state-stack-size 4
        
        Frame Stacking (4 frames recommended):
        - Visual: 4 consecutive frames stacked along channel dimension (4x input channels)
        - State: 4 consecutive states concatenated (4x state_dim)
        - Enables inference of velocity and acceleration without explicit features
        
        Modalities:
        - Player Frame: 512-dim (processes 4-stacked 84x84x1 frames)
        - Global View: 256-dim (processes 4-stacked 176x100x1 frames)
        - Graph (Full HGT): 3 layers, 256 hidden, 8 heads
        - Game State: 128-dim (processes 4-stacked 26-dim vectors)
        - Reachability: 8-dim
        
        Best for: Tasks requiring motion understanding (jumping, navigation timing).
        Trade-off: 4x visual input channels increases computation slightly.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
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


def create_vision_free_frame_stacked_config() -> ArchitectureConfig:
    """Vision-free architecture designed for use with state stacking.

    Note: This config defines the architecture. Frame stacking must be enabled
    separately via: --enable-state-stacking --state-stack-size 4
    """
    return ArchitectureConfig(
        name="vision_free_frame_stacked",
        description="Vision-free with 4-state stacking for velocity/acceleration inference",
        detailed_description="""
        Fast vision-free architecture using only physics states with frame stacking.
        Demonstrates whether temporal state information alone is sufficient.
        
        IMPORTANT: State stacking is NOT automatically enabled by this config.
        You must also pass: --enable-state-stacking --state-stack-size 4
        
        Frame Stacking (4 frames recommended):
        - State: 4 consecutive states stacked (4x26 = 104 total dimensions)
        - Enables velocity and acceleration inference without visual data
        
        Modalities:
        - Game State: 128-dim (processes 4-stacked 26-dim vectors)
        - Reachability: 8-dim
        - No Visual Processing (fastest)
        - No Graph Processing
        
        Best for: Rapid experimentation, baseline for temporal reasoning value.
        Fastest architecture with temporal understanding.
        """,
        modalities=ModalityConfig(
            use_player_frame=False,
            use_global_view=False,
            use_graph=False,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(architecture=GraphArchitectureType.NONE),
        visual=VisualConfig(),
        state=StateConfig(),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=256,
    )


def create_visual_frame_stacked_only_config() -> ArchitectureConfig:
    """Visual frames with stacking only - tests if visual temporal info is sufficient.

    Note: This config defines the architecture. Visual frame stacking must be enabled
    separately via: --enable-visual-frame-stacking --visual-stack-size 4
    (Do NOT enable state stacking for this config)
    """
    return ArchitectureConfig(
        name="visual_frame_stacked_only",
        description="Visual frames with 4-frame stacking, no state stacking",
        detailed_description="""
        Tests whether visual frame stacking alone provides sufficient temporal information.
        Isolates the contribution of visual temporal patterns vs state temporal patterns.
        
        IMPORTANT: Enable ONLY visual stacking for this config:
            --enable-visual-frame-stacking --visual-stack-size 4
        Do NOT enable state stacking (--enable-state-stacking should be omitted).
        
        Frame Stacking:
        - Visual: 4 frames stacked (player_frame and global_view)
        - State: Single frame only (no stacking)
        
        Modalities:
        - Player Frame: 512-dim (4-stacked frames)
        - Global View: 256-dim (4-stacked frames)
        - Graph (Full HGT): 3 layers, 256 hidden, 8 heads
        - Game State: 128-dim (single state, no stacking)
        - Reachability: 8-dim
        
        Best for: Understanding importance of visual vs state temporal information.
        """,
        modalities=ModalityConfig(
            use_player_frame=True,
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


def create_attention_config() -> ArchitectureConfig:
    """Create simplified graph-based architecture with DeepResNet and Dueling."""
    return ArchitectureConfig(
        name="attention",
        description="Deep ResNet architecture with GNN and Dueling value decomposition",
        detailed_description="""
        Simplified architecture focusing on proven improvements:
        
        1. GCN Graph Encoder (Spatial Aggregation):
           - Graph Convolutional Network with mean aggregation
           - 2 layers, 128 hidden, 256 output (optimized for performance)
           - Input: 7-dim node features (GCN-optimized)
           - Handles variable graph sizes with masking
        
        2. Simple State MLP:
           - Processes 41-dim game state (ninja physics)
           - 2-layer MLP: 41 → 128 → 128
           - Efficient encoding without attention overhead
        
        3. MultiHeadFusion (Cross-Modal Attention):
           - Integrates graph, state, and reachability features
           - Multi-head attention (8 heads)
           - Output: 512-dim final features
        
        4. DeepResNet Policy/Value Networks:
           - Policy: [512, 512, 384, 256, 256] with residual connections
           - Value: [512, 384, 256] with residual connections
           - SiLU activation, LayerNorm
        
        5. Dueling Value Head:
           - Separate V(s) and A(s,a) streams
           - Better value estimation, faster convergence
        
        Total parameters: ~9-11M (down from 15-18M)
        - DeepResNet MLPs: ~6-8M
        - Feature extractors: ~3-4M
        - 30-40% reduction from attention removal
        
        Design Rationale:
        - DeepResNet enables deep reasoning without gradient issues
        - Dueling improves value estimation
        - GNN encodes spatial structure (shortest paths from graph)
        - Simple MLP sufficient for 41D physics
        - No attention redundancy (graph already has shortest paths)
        """,
        modalities=ModalityConfig(
            use_player_frame=False,
            use_global_view=False,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        visual=VisualConfig(),
        graph=GraphConfig(
            architecture=GraphArchitectureType.GCN,
            hidden_dim=128,
            output_dim=256,
            num_layers=2,  # Reduced from 3 for 33% speedup (profiling showed 44.9% time in GCN)
            dropout=0.1,
            use_type_embeddings=False,
            use_edge_features=False,  # GCN uses only connectivity, not edge features
        ),
        state=StateConfig(
            hidden_dim=128,
            output_dim=128,
        ),
        fusion=FusionConfig(
            fusion_type=FusionType.MULTI_HEAD_ATTENTION,
            num_attention_heads=8,
            dropout=0.1,
        ),
        features_dim=512,
    )


# ===== Configuration Registry =====

ARCHITECTURE_REGISTRY: Dict[str, ArchitectureConfig] = {
    "full_hgt": create_full_hgt_config(),
    "simplified_hgt": create_simplified_hgt_config(),
    "gat": create_gat_config(),
    "gcn": create_gcn_config(),
    "mlp_cnn": create_mlp_cnn_config(),
    "vision_free": create_vision_free_config(),
    "vision_free_gat": create_vision_free_gat_config(),
    "vision_free_gcn": create_vision_free_gcn_config(),
    "vision_free_simplified": create_vision_free_simplified_config(),
    "no_global_view": create_no_global_view_config(),
    # Frame stacking variants
    "full_hgt_frame_stacked": create_full_hgt_frame_stacked_config(),
    "vision_free_frame_stacked": create_vision_free_frame_stacked_config(),
    "visual_frame_stacked_only": create_visual_frame_stacked_only_config(),
    # Attention-based architecture
    "attention": create_attention_config(),
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
        raise ValueError(f"Unknown architecture '{name}'. Available: {available}")
    return ARCHITECTURE_REGISTRY[name]


def list_available_architectures() -> List[str]:
    """Return list of available architecture names."""
    return list(ARCHITECTURE_REGISTRY.keys())


def print_architecture_summary(config: ArchitectureConfig) -> None:
    """Print human-readable summary of architecture configuration."""
    print(f"\n{'=' * 60}")
    print(f"Architecture: {config.name}")
    print(f"{'=' * 60}")
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
    print(f"{'=' * 60}\n")


def list_architectures(detailed: bool = True) -> None:
    """
    Display all available architectures with their descriptions.

    Args:
        detailed: If True, show detailed descriptions. If False, show only brief descriptions.
    """
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE ARCHITECTURES ({len(ARCHITECTURE_REGISTRY)} total)")
    print(f"{'=' * 80}\n")

    for name, config in ARCHITECTURE_REGISTRY.items():
        print(f"{'─' * 80}")
        print(f"[{name}]")
        print(f"{'─' * 80}")
        print(f"Brief: {config.description}")

        if detailed:
            print("\nDetailed Description:")
            # Clean up the detailed description (remove extra indentation)
            detailed_lines = config.detailed_description.strip().split("\n")
            for line in detailed_lines:
                print(f"  {line.strip()}")

        print(f"\nModalities: {', '.join(config.modalities.get_enabled_modalities())}")
        print(f"Graph Type: {config.graph.architecture.value}")
        print(f"Fusion Type: {config.fusion.fusion_type.value}")
        print(f"Feature Dim: {config.features_dim}\n")

    print(f"{'=' * 80}\n")
