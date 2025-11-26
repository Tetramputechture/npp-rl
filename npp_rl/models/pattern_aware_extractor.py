"""Pattern-aware feature extractor for generalized N++ navigation.

This module extends the ConfigurableMultimodalExtractor with learned tile/entity
pattern recognition capabilities that enable zero-shot generalization to unseen
level configurations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import logging

from ..feature_extractors.configurable_extractor import ConfigurableMultimodalExtractor
from ..path_prediction import (
    GeneralizedPatternExtractor,
    TilePattern,
    ProbabilisticPathPredictor,
)

logger = logging.getLogger(__name__)


class TilePatternEncoder(nn.Module):
    """Neural encoder for tile pattern features."""

    def __init__(
        self,
        pattern_vocab_size: int = 100,  # Max unique patterns
        pattern_embed_dim: int = 32,
        context_dim: int = 64,
    ):
        super().__init__()

        # Embedding for tile pattern IDs
        self.pattern_embedding = nn.Embedding(pattern_vocab_size, pattern_embed_dim)

        # MLP for processing pattern context
        self.context_mlp = nn.Sequential(
            nn.Linear(pattern_embed_dim + 16, context_dim),  # +16 for pattern metadata
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(
        self, pattern_ids: torch.Tensor, pattern_metadata: torch.Tensor
    ) -> torch.Tensor:
        """Encode tile patterns into dense features.

        Args:
            pattern_ids: Pattern identifiers [batch, max_patterns]
            pattern_metadata: Pattern metadata features [batch, max_patterns, 16]

        Returns:
            Encoded pattern features [batch, context_dim]
        """
        # Embed pattern IDs
        pattern_embeds = self.pattern_embedding(
            pattern_ids
        )  # [batch, max_patterns, embed_dim]

        # Combine with metadata
        combined = torch.cat([pattern_embeds, pattern_metadata], dim=-1)

        # Apply context MLP and pool across patterns
        context_features = self.context_mlp(
            combined
        )  # [batch, max_patterns, context_dim]

        # Max pooling across patterns (handles variable number of patterns)
        pooled_features = torch.max(context_features, dim=1)[0]  # [batch, context_dim]

        return pooled_features


class EntityContextEncoder(nn.Module):
    """Neural encoder for entity context features."""

    def __init__(
        self,
        max_entities: int = 20,
        entity_feature_dim: int = 8,  # x, y, type, distance
        output_dim: int = 32,
    ):
        super().__init__()

        self.max_entities = max_entities

        # MLP for entity features
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_feature_dim, 16), nn.ReLU(), nn.Linear(16, 32)
        )

        # Attention for entity importance
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(32, output_dim)

    def forward(self, entity_features: torch.Tensor) -> torch.Tensor:
        """Encode entity context into dense features.

        Args:
            entity_features: Entity context features [batch, max_entities, entity_feature_dim]

        Returns:
            Encoded entity features [batch, output_dim]
        """
        batch_size = entity_features.size(0)

        # Process entity features
        entity_embeds = self.entity_mlp(entity_features)  # [batch, max_entities, 32]

        # Apply attention to weight entity importance
        attended, _ = self.entity_attention(entity_embeds, entity_embeds, entity_embeds)

        # Pool across entities (mean pooling)
        pooled = torch.mean(attended, dim=1)  # [batch, 32]

        # Output projection
        output = self.output_projection(pooled)  # [batch, output_dim]

        return output


class WaypointDistributionPredictor(nn.Module):
    """Predicts probabilistic waypoint preferences based on patterns."""

    def __init__(
        self,
        input_dim: int,
        num_directions: int = 8,  # 8 cardinal/ordinal directions
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_directions = num_directions

        # Direction preference head
        self.direction_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_directions),
            nn.Softmax(dim=-1),
        )

        # Movement type preference head
        self.movement_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # walk, jump, wall_slide
            nn.Softmax(dim=-1),
        )

        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, pattern_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict waypoint distributions from pattern features.

        Args:
            pattern_features: Combined pattern features [batch, input_dim]

        Returns:
            Dictionary with direction preferences, movement types, and risk
        """
        return {
            "direction_prefs": self.direction_head(pattern_features),
            "movement_prefs": self.movement_head(pattern_features),
            "risk_assessment": self.risk_head(pattern_features),
        }


class PatternAwareExtractor(ConfigurableMultimodalExtractor):
    """Feature extractor enhanced with learned tile/entity pattern recognition.

    This extractor extends the standard multimodal approach by:
    1. Recognizing learned tile/entity patterns from demonstrations
    2. Predicting waypoint preferences based on recognized patterns
    3. Integrating pattern-based guidance with standard modalities
    4. Supporting zero-shot generalization to unseen level configurations
    """

    def __init__(
        self,
        observation_space,
        features_dim: int,
        architecture_config,
        pattern_extractor: Optional[GeneralizedPatternExtractor] = None,
        path_predictor: Optional[ProbabilisticPathPredictor] = None,
        pattern_feature_dim: int = 128,
        max_patterns_per_obs: int = 10,
        max_entities_per_obs: int = 20,
        **kwargs,
    ):
        """Initialize pattern-aware feature extractor.

        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
            architecture_config: Architecture configuration
            pattern_extractor: Pre-trained pattern extractor (optional)
            path_predictor: Multi-path predictor for candidate paths (optional)
            pattern_feature_dim: Dimension of pattern-based features
            max_patterns_per_obs: Maximum patterns to process per observation
            max_entities_per_obs: Maximum entities to process per observation
            **kwargs: Additional arguments for parent class
        """
        super().__init__(observation_space, features_dim, architecture_config, **kwargs)

        self.pattern_extractor = pattern_extractor
        self.path_predictor = path_predictor
        self.pattern_feature_dim = pattern_feature_dim
        self.max_patterns_per_obs = max_patterns_per_obs
        self.max_entities_per_obs = max_entities_per_obs

        # Pattern encoding components
        self.tile_pattern_encoder = TilePatternEncoder(
            pattern_embed_dim=32, context_dim=64
        )

        self.entity_context_encoder = EntityContextEncoder(
            max_entities=max_entities_per_obs, output_dim=32
        )

        # Waypoint distribution predictor
        combined_pattern_dim = 64 + 32  # tile patterns + entity context
        self.waypoint_predictor = WaypointDistributionPredictor(
            input_dim=combined_pattern_dim, hidden_dim=128
        )

        # Integration with existing features
        # Update fusion layer to include pattern features
        original_fusion_input = (
            self.fusion.mlp[0].in_features
            if hasattr(self.fusion, "mlp")
            else features_dim
        )
        new_fusion_input = original_fusion_input + pattern_feature_dim

        # Replace fusion layer to accommodate pattern features
        if hasattr(self.fusion, "mlp"):
            # For concatenation fusion
            self.fusion.mlp[0] = nn.Linear(
                new_fusion_input, self.fusion.mlp[0].out_features
            )
        else:
            # For attention-based fusion, need to rebuild
            self._rebuild_fusion_for_patterns(new_fusion_input, features_dim)

        # Pattern memory for caching
        self.pattern_cache = {}
        self.cache_size_limit = 1000

        logger.info(
            f"Initialized PatternAwareExtractor with {pattern_feature_dim}D pattern features"
        )

    def _rebuild_fusion_for_patterns(self, new_input_dim: int, output_dim: int) -> None:
        """Rebuild fusion layer to accommodate pattern features."""
        from ..feature_extractors.configurable_extractor import ConcatenationFusion

        # Create new concatenation fusion that handles pattern features
        self.fusion = ConcatenationFusion(new_input_dim, output_dim)

        logger.debug(
            f"Rebuilt fusion layer for pattern integration: {new_input_dim} -> {output_dim}"
        )

    def extract_local_patterns(
        self, tile_data: torch.Tensor, entity_positions: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract generalizable local patterns from tile data and entity positions.

        Args:
            tile_data: Tile configuration tensor [height, width]
            entity_positions: List of entity dictionaries with positions and types

        Returns:
            Tuple of (pattern_features, entity_features) tensors
        """
        batch_size = tile_data.size(0) if tile_data.dim() > 2 else 1

        # Initialize feature tensors
        pattern_ids = torch.zeros(
            batch_size, self.max_patterns_per_obs, dtype=torch.long
        )
        pattern_metadata = torch.zeros(batch_size, self.max_patterns_per_obs, 16)
        entity_features = torch.zeros(batch_size, self.max_entities_per_obs, 8)

        for batch_idx in range(batch_size):
            if tile_data.dim() > 2:
                current_tile_data = tile_data[batch_idx].cpu().numpy()
                current_entities = (
                    entity_positions[batch_idx]
                    if batch_idx < len(entity_positions)
                    else []
                )
            else:
                current_tile_data = tile_data.cpu().numpy()
                current_entities = entity_positions[0] if entity_positions else []

            # Extract patterns using the pattern extractor if available
            if self.pattern_extractor is not None:
                patterns = self._extract_patterns_from_tiles(
                    current_tile_data, current_entities
                )

                # Convert patterns to tensor format
                for i, pattern in enumerate(patterns[: self.max_patterns_per_obs]):
                    pattern_ids[batch_idx, i] = (
                        hash(pattern.pattern_id) % 100
                    )  # Map to vocab

                    # Pattern metadata: size, center_tile, confidence, etc.
                    pattern_metadata[batch_idx, i, 0] = pattern.pattern_size
                    pattern_metadata[batch_idx, i, 1] = pattern.center_tile_type
                    # Additional metadata can be added here

            # Process entity features
            for i, entity in enumerate(current_entities[: self.max_entities_per_obs]):
                entity_features[batch_idx, i, 0] = (
                    entity.get("x", 0) / 1000.0
                )  # Normalize position
                entity_features[batch_idx, i, 1] = entity.get("y", 0) / 1000.0
                entity_features[batch_idx, i, 2] = (
                    entity.get("type", 0) / 50.0
                )  # Normalize type
                entity_features[batch_idx, i, 3] = 1.0  # Presence indicator
                # Distance and other features can be computed here

        return pattern_ids, pattern_metadata, entity_features

    def predict_waypoint_distributions(
        self, pattern_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict waypoint preferences based on learned patterns.

        Args:
            pattern_features: Combined pattern features [batch, pattern_dim]

        Returns:
            Dictionary with waypoint distribution predictions
        """
        return self.waypoint_predictor(pattern_features)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with pattern-aware feature integration.

        Args:
            observations: Dictionary of observation tensors

        Returns:
            Enhanced features including pattern-based guidance
        """
        # Extract standard multimodal features using parent method
        standard_features = super().forward(observations)

        # Extract pattern-based features if tile data is available
        pattern_features = None

        if "tile_data" in observations and self.pattern_extractor is not None:
            try:
                tile_data = observations["tile_data"]
                entities = observations.get("entities", [])

                # Extract local patterns
                pattern_ids, pattern_metadata, entity_features = (
                    self.extract_local_patterns(tile_data, entities)
                )

                # Move to appropriate device
                device = standard_features.device
                pattern_ids = pattern_ids.to(device)
                pattern_metadata = pattern_metadata.to(device)
                entity_features = entity_features.to(device)

                # Encode patterns and entities
                tile_pattern_features = self.tile_pattern_encoder(
                    pattern_ids, pattern_metadata
                )
                entity_context_features = self.entity_context_encoder(entity_features)

                # Combine pattern features
                combined_pattern_features = torch.cat(
                    [tile_pattern_features, entity_context_features], dim=-1
                )

                # Predict waypoint distributions
                waypoint_distributions = self.predict_waypoint_distributions(
                    combined_pattern_features
                )

                # Create pattern feature vector that includes guidance information
                direction_prefs = waypoint_distributions["direction_prefs"]
                movement_prefs = waypoint_distributions["movement_prefs"]
                risk_assessment = waypoint_distributions["risk_assessment"]

                # Concatenate all pattern-based features
                pattern_features = torch.cat(
                    [
                        combined_pattern_features,  # 96D
                        direction_prefs,  # 8D
                        movement_prefs,  # 3D
                        risk_assessment,  # 1D
                    ],
                    dim=-1,
                )  # Total: 108D

                # Pad or truncate to match expected dimension
                if pattern_features.size(-1) < self.pattern_feature_dim:
                    padding = torch.zeros(
                        pattern_features.size(0),
                        self.pattern_feature_dim - pattern_features.size(-1),
                        device=device,
                    )
                    pattern_features = torch.cat([pattern_features, padding], dim=-1)
                elif pattern_features.size(-1) > self.pattern_feature_dim:
                    pattern_features = pattern_features[:, : self.pattern_feature_dim]

            except Exception as e:
                logger.warning(f"Pattern extraction failed: {e}, using zero features")
                pattern_features = torch.zeros(
                    standard_features.size(0),
                    self.pattern_feature_dim,
                    device=standard_features.device,
                )

        # If no pattern features available, use zero padding
        if pattern_features is None:
            pattern_features = torch.zeros(
                standard_features.size(0),
                self.pattern_feature_dim,
                device=standard_features.device,
            )

        # Combine standard and pattern features
        enhanced_features = torch.cat([standard_features, pattern_features], dim=-1)

        # Apply updated fusion layer
        output = self.fusion(enhanced_features)

        return output

    def _extract_patterns_from_tiles(
        self, tile_data: np.ndarray, entities: List[Dict[str, Any]]
    ) -> List[TilePattern]:
        """Extract tile patterns from raw tile data."""
        if self.pattern_extractor is None:
            return []

        patterns = []
        height, width = tile_data.shape

        # Sample key positions for pattern extraction (avoid processing every tile)
        sample_positions = []
        step_size = max(1, min(height, width) // 10)  # Sample ~100 positions max

        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                sample_positions.append(
                    (x * 24, y * 24)
                )  # Convert to pixel coordinates

        # Extract patterns at sampled positions
        for pos in sample_positions[: self.max_patterns_per_obs]:
            tile_patterns = self.pattern_extractor._extract_tile_patterns(
                pos, tile_data
            )
            patterns.extend(tile_patterns)

        return patterns[: self.max_patterns_per_obs]

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern recognition and usage."""
        stats = {
            "cache_size": len(self.pattern_cache),
            "max_cache_size": self.cache_size_limit,
        }

        if self.pattern_extractor is not None:
            stats.update(self.pattern_extractor.get_statistics())

        return stats

    def clear_pattern_cache(self) -> None:
        """Clear pattern cache to free memory."""
        self.pattern_cache.clear()
        logger.debug("Cleared pattern cache")


def create_pattern_aware_extractor(
    observation_space,
    features_dim: int,
    architecture_config,
    pattern_extractor: Optional[GeneralizedPatternExtractor] = None,
    path_predictor: Optional[ProbabilisticPathPredictor] = None,
    **kwargs,
) -> PatternAwareExtractor:
    """Factory function to create PatternAwareExtractor.

    Args:
        observation_space: Gym observation space
        features_dim: Output feature dimension
        architecture_config: Architecture configuration
        pattern_extractor: Pre-trained pattern extractor
        path_predictor: Multi-path predictor
        **kwargs: Additional arguments

    Returns:
        Configured PatternAwareExtractor instance
    """
    return PatternAwareExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        architecture_config=architecture_config,
        pattern_extractor=pattern_extractor,
        path_predictor=path_predictor,
        **kwargs,
    )
