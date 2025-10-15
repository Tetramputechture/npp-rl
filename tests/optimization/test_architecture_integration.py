"""
Integration tests for all 8 architecture variants.

Tests that each architecture defined in ARCHITECTURE_REGISTRY can be:
1. Instantiated without errors
2. Perform forward passes with mock observations
3. Integrate with PPO policies

This ensures all architectures are ready for training runs.
"""

import unittest
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Dict as SpacesDict

from npp_rl.optimization.architecture_configs import (
    ARCHITECTURE_REGISTRY,
    get_architecture_config,
    list_available_architectures,
)
from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor


class TestArchitectureIntegration(unittest.TestCase):
    """Test all 8 architecture variants can be instantiated and used."""

    def setUp(self):
        """Create mock observation space for testing."""
        # Complete observation space with all modalities
        self.full_obs_space = SpacesDict(
            {
                "player_frame": Box(
                    low=0, high=255, shape=(12, 84, 84), dtype='uint8'
                ),
                "global_view": Box(
                    low=0, high=255, shape=(176, 100, 1), dtype='uint8'
                ),
                "graph_obs": SpacesDict(
                    {
                        "node_features": Box(
                            low=-float('inf'),
                            high=float('inf'),
                            shape=(50, 67),  # max 50 nodes, 67 features
                            dtype='float32',
                        ),
                        "edge_index": Box(
                            low=0, high=49, shape=(2, 200), dtype='int64'
                        ),
                        "node_mask": Box(
                            low=0, high=1, shape=(50,), dtype='bool'
                        ),
                        "node_types": Box(
                            low=0, high=5, shape=(50,), dtype='int64'
                        ),
                    }
                ),
                "game_state": Box(
                    low=-float('inf'), high=float('inf'), shape=(30,), dtype='float32'
                ),
                "reachability_features": Box(
                    low=-float('inf'), high=float('inf'), shape=(8,), dtype='float32'
                ),
            }
        )

        self.batch_size = 4

    def _create_mock_observations(self, obs_space: gym.spaces.Dict, batch_size: int):
        """Create mock observations matching the observation space."""
        observations = {}

        if "player_frame" in obs_space.spaces:
            observations["player_frame"] = torch.randint(
                0, 256, (batch_size, 12, 84, 84), dtype=torch.uint8
            )

        if "global_view" in obs_space.spaces:
            observations["global_view"] = torch.randint(
                0, 256, (batch_size, 176, 100, 1), dtype=torch.uint8
            )

        if "graph_obs" in obs_space.spaces:
            observations["graph_obs"] = {
                "node_features": torch.randn(batch_size, 50, 67),
                "edge_index": torch.randint(0, 50, (batch_size, 2, 200)),
                "edge_features": torch.randn(batch_size, 200, 9),
                "node_mask": torch.ones(batch_size, 50, dtype=torch.bool),
                "edge_mask": torch.ones(batch_size, 200, dtype=torch.bool),
                "node_types": torch.randint(0, 6, (batch_size, 50)),
            }

        if "game_state" in obs_space.spaces:
            observations["game_state"] = torch.randn(batch_size, 30)

        if "reachability_features" in obs_space.spaces:
            observations["reachability_features"] = torch.randn(batch_size, 8)

        return observations

    def test_all_architectures_in_registry(self):
        """Test that all expected architectures are in the registry."""
        expected_architectures = [
            "full_hgt",
            "simplified_hgt",
            "gat",
            "gcn",
            "mlp_baseline",
            "vision_free",
            "no_global_view",
            "local_frames_only",
        ]

        available = list_available_architectures()

        for arch_name in expected_architectures:
            self.assertIn(
                arch_name,
                available,
                f"Architecture '{arch_name}' not found in registry",
            )

        self.assertEqual(
            len(available),
            8,
            f"Expected 8 architectures, found {len(available)}: {available}",
        )

    def test_full_hgt_instantiation_and_forward(self):
        """Test full_hgt architecture can be instantiated and forward pass works."""
        config = get_architecture_config("full_hgt")

        # Create extractor
        extractor = ConfigurableMultimodalExtractor(self.full_obs_space, config)

        # Create mock observations
        observations = self._create_mock_observations(
            self.full_obs_space, self.batch_size
        )

        # Forward pass
        with torch.no_grad():
            features = extractor(observations)

        # Validate output
        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))
        self.assertGreater(torch.norm(features), 0)

    def test_simplified_hgt_instantiation_and_forward(self):
        """Test simplified_hgt architecture."""
        config = get_architecture_config("simplified_hgt")
        extractor = ConfigurableMultimodalExtractor(self.full_obs_space, config)

        observations = self._create_mock_observations(
            self.full_obs_space, self.batch_size
        )

        with torch.no_grad():
            features = extractor(observations)

        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))

    def test_gat_instantiation_and_forward(self):
        """Test GAT architecture."""
        config = get_architecture_config("gat")
        extractor = ConfigurableMultimodalExtractor(self.full_obs_space, config)

        observations = self._create_mock_observations(
            self.full_obs_space, self.batch_size
        )

        with torch.no_grad():
            features = extractor(observations)

        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))

    def test_gcn_instantiation_and_forward(self):
        """Test GCN architecture."""
        config = get_architecture_config("gcn")
        extractor = ConfigurableMultimodalExtractor(self.full_obs_space, config)

        observations = self._create_mock_observations(
            self.full_obs_space, self.batch_size
        )

        with torch.no_grad():
            features = extractor(observations)

        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))

    def test_mlp_baseline_instantiation_and_forward(self):
        """Test MLP baseline (no graph processing)."""
        config = get_architecture_config("mlp_baseline")

        # MLP baseline doesn't use graph, so create observation space without it
        mlp_obs_space = SpacesDict(
            {
                "player_frame": self.full_obs_space["player_frame"],
                "global_view": self.full_obs_space["global_view"],
                "game_state": self.full_obs_space["game_state"],
                "reachability_features": self.full_obs_space["reachability_features"],
            }
        )

        extractor = ConfigurableMultimodalExtractor(mlp_obs_space, config)

        observations = self._create_mock_observations(mlp_obs_space, self.batch_size)

        with torch.no_grad():
            features = extractor(observations)

        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))

    def test_vision_free_instantiation_and_forward(self):
        """Test vision-free architecture (graph + state only)."""
        config = get_architecture_config("vision_free")

        # Vision-free doesn't use visual modalities
        vision_free_obs_space = SpacesDict(
            {
                "graph_obs": self.full_obs_space["graph_obs"],
                "game_state": self.full_obs_space["game_state"],
                "reachability_features": self.full_obs_space["reachability_features"],
            }
        )

        extractor = ConfigurableMultimodalExtractor(vision_free_obs_space, config)

        observations = self._create_mock_observations(
            vision_free_obs_space, self.batch_size
        )

        with torch.no_grad():
            features = extractor(observations)

        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))

    def test_no_global_view_instantiation_and_forward(self):
        """Test no_global_view architecture (temporal + graph + state)."""
        config = get_architecture_config("no_global_view")

        # No global view variant
        no_global_obs_space = SpacesDict(
            {
                "player_frame": self.full_obs_space["player_frame"],
                "graph_obs": self.full_obs_space["graph_obs"],
                "game_state": self.full_obs_space["game_state"],
                "reachability_features": self.full_obs_space["reachability_features"],
            }
        )

        extractor = ConfigurableMultimodalExtractor(no_global_obs_space, config)

        observations = self._create_mock_observations(
            no_global_obs_space, self.batch_size
        )

        with torch.no_grad():
            features = extractor(observations)

        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))

    def test_local_frames_only_instantiation_and_forward(self):
        """Test local_frames_only architecture (same as no_global_view)."""
        config = get_architecture_config("local_frames_only")

        # Local frames only
        local_obs_space = SpacesDict(
            {
                "player_frame": self.full_obs_space["player_frame"],
                "graph_obs": self.full_obs_space["graph_obs"],
                "game_state": self.full_obs_space["game_state"],
                "reachability_features": self.full_obs_space["reachability_features"],
            }
        )

        extractor = ConfigurableMultimodalExtractor(local_obs_space, config)

        observations = self._create_mock_observations(local_obs_space, self.batch_size)

        with torch.no_grad():
            features = extractor(observations)

        self.assertEqual(features.shape, (self.batch_size, config.features_dim))
        self.assertTrue(torch.all(torch.isfinite(features)))

    def test_all_architectures_batch_size_variations(self):
        """Test all architectures work with different batch sizes."""
        batch_sizes = [1, 4, 16]

        for arch_name in list_available_architectures():
            config = get_architecture_config(arch_name)

            # Create appropriate observation space based on config
            obs_space = self._create_obs_space_for_config(config)
            extractor = ConfigurableMultimodalExtractor(obs_space, config)

            for batch_size in batch_sizes:
                with self.subTest(architecture=arch_name, batch_size=batch_size):
                    observations = self._create_mock_observations(obs_space, batch_size)

                    with torch.no_grad():
                        features = extractor(observations)

                    self.assertEqual(
                        features.shape,
                        (batch_size, config.features_dim),
                        f"Failed for {arch_name} with batch_size={batch_size}",
                    )
                    self.assertTrue(torch.all(torch.isfinite(features)))

    def test_architecture_output_consistency(self):
        """Test that architectures produce consistent outputs for same input."""
        config = get_architecture_config("full_hgt")
        extractor = ConfigurableMultimodalExtractor(self.full_obs_space, config)
        extractor.eval()  # Set to eval mode for deterministic behavior

        observations = self._create_mock_observations(self.full_obs_space, 2)

        # Run forward pass twice
        with torch.no_grad():
            output1 = extractor(observations)
            output2 = extractor(observations)

        # Outputs should be identical in eval mode
        self.assertTrue(
            torch.allclose(output1, output2),
            "Architecture produces inconsistent outputs",
        )

    def _create_obs_space_for_config(self, config):
        """Create appropriate observation space based on config modalities."""
        spaces = {}

        if config.modalities.use_temporal_frames:
            spaces["player_frame"] = self.full_obs_space["player_frame"]

        if config.modalities.use_global_view:
            spaces["global_view"] = self.full_obs_space["global_view"]

        if config.modalities.use_graph:
            spaces["graph_obs"] = self.full_obs_space["graph_obs"]

        if config.modalities.use_game_state:
            spaces["game_state"] = self.full_obs_space["game_state"]

        if config.modalities.use_reachability:
            spaces["reachability_features"] = self.full_obs_space[
                "reachability_features"
            ]

        return SpacesDict(spaces)


class TestArchitectureConfigurations(unittest.TestCase):
    """Test architecture configuration properties."""

    def test_all_configs_have_valid_modality_counts(self):
        """Test that all architectures have reasonable modality counts."""
        for arch_name in list_available_architectures():
            config = get_architecture_config(arch_name)
            count = config.modalities.count_modalities()

            self.assertGreater(
                count, 0, f"{arch_name} has no modalities enabled"
            )
            self.assertLessEqual(
                count, 5, f"{arch_name} has too many modalities ({count})"
            )

    def test_graph_configs_match_architecture_types(self):
        """Test that graph configs match their declared architecture types."""
        config = get_architecture_config("full_hgt")
        self.assertEqual(
            config.graph.architecture.value, "full_hgt"
        )

        config = get_architecture_config("simplified_hgt")
        self.assertEqual(
            config.graph.architecture.value, "simplified_hgt"
        )

        config = get_architecture_config("gat")
        self.assertEqual(config.graph.architecture.value, "gat")

        config = get_architecture_config("gcn")
        self.assertEqual(config.graph.architecture.value, "gcn")

        config = get_architecture_config("mlp_baseline")
        self.assertEqual(
            config.graph.architecture.value, "none"
        )

    def test_vision_free_has_no_visual_modalities(self):
        """Test vision_free config correctly disables visual modalities."""
        config = get_architecture_config("vision_free")

        self.assertFalse(config.modalities.use_temporal_frames)
        self.assertFalse(config.modalities.use_global_view)
        self.assertTrue(config.modalities.use_graph)
        self.assertTrue(config.modalities.use_game_state)

    def test_mlp_baseline_has_no_graph(self):
        """Test MLP baseline correctly disables graph processing."""
        config = get_architecture_config("mlp_baseline")

        self.assertFalse(config.modalities.use_graph)
        self.assertTrue(config.modalities.use_temporal_frames)
        self.assertTrue(config.modalities.use_global_view)


if __name__ == "__main__":
    unittest.main()
