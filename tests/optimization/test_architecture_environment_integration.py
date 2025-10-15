"""
Integration tests for architectures with real nclone environment.

These tests validate that all 8 architecture variants can:
1. Accept observations from the actual nclone environment
2. Process them through ConfigurableMultimodalExtractor
3. Produce valid feature outputs
4. Work with correct dimensions from nclone constants
"""

import unittest
import torch
import numpy as np
from nclone.gym_environment import create_reachability_aware_env
from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM

from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
from npp_rl.optimization.architecture_configs import (
    get_architecture_config,
    ARCHITECTURE_REGISTRY,
)


class TestArchitectureEnvironmentIntegration(unittest.TestCase):
    """Test all architectures with real environment observations."""

    def setUp(self):
        """Create environment for testing."""
        self.env = create_reachability_aware_env()
        self.obs_space = self.env.observation_space

    def tearDown(self):
        """Clean up environment."""
        if hasattr(self, "env"):
            self.env.close()

    def _get_observation_tensor(self):
        """Get observation from environment and convert to torch tensors."""
        obs, _ = self.env.reset()

        # Convert to torch tensors with batch dimension
        torch_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                torch_obs[key] = torch.from_numpy(value).unsqueeze(0)
            else:
                torch_obs[key] = torch.tensor(value).unsqueeze(0)

        return torch_obs

    def test_nclone_dimensions_match_constants(self):
        """Verify nclone environment provides correct dimensions."""
        obs, _ = self.env.reset()

        # Check graph dimensions
        self.assertEqual(
            obs["graph_node_feats"].shape[1],
            NODE_FEATURE_DIM,
            f"Node features should be {NODE_FEATURE_DIM}-dimensional",
        )
        self.assertEqual(
            obs["graph_edge_feats"].shape[1],
            EDGE_FEATURE_DIM,
            f"Edge features should be {EDGE_FEATURE_DIM}-dimensional",
        )

        # Check state dimensions
        self.assertEqual(obs["game_state"].shape[0], 30, "Game state should be 30-dimensional")
        self.assertEqual(
            obs["reachability_features"].shape[0],
            8,
            "Reachability should be 8-dimensional",
        )

    def test_all_architectures_process_environment_observations(self):
        """Test that all 8 architectures can process real environment observations."""
        torch_obs = self._get_observation_tensor()

        for arch_name in ARCHITECTURE_REGISTRY.keys():
            with self.subTest(architecture=arch_name):
                config = get_architecture_config(arch_name)
                extractor = ConfigurableMultimodalExtractor(self.obs_space, config)

                # Forward pass
                with torch.no_grad():
                    features = extractor(torch_obs)

                # Verify output
                self.assertEqual(
                    features.shape,
                    (1, config.features_dim),
                    f"{arch_name} should output (1, {config.features_dim})",
                )
                self.assertTrue(
                    torch.all(torch.isfinite(features)),
                    f"{arch_name} produced non-finite values",
                )
                self.assertGreater(
                    torch.norm(features).item(),
                    0,
                    f"{arch_name} produced all-zero features",
                )

    def test_graph_architectures_use_correct_dimensions(self):
        """Test that graph architectures properly use nclone dimensions."""
        graph_architectures = ["full_hgt", "simplified_hgt", "gat", "gcn"]

        for arch_name in graph_architectures:
            with self.subTest(architecture=arch_name):
                config = get_architecture_config(arch_name)
                extractor = ConfigurableMultimodalExtractor(self.obs_space, config)

                # Get graph encoder
                self.assertIsNotNone(
                    extractor.graph_encoder,
                    f"{arch_name} should have graph encoder",
                )

                # Test with real observations
                torch_obs = self._get_observation_tensor()
                with torch.no_grad():
                    features = extractor(torch_obs)

                self.assertIsNotNone(features)
                self.assertEqual(features.shape[0], 1)

    def test_vision_architectures_process_frames(self):
        """Test that architectures with vision process temporal and global frames."""
        vision_architectures = [
            "full_hgt",
            "simplified_hgt",
            "gat",
            "gcn",
            "mlp_baseline",
        ]

        for arch_name in vision_architectures:
            with self.subTest(architecture=arch_name):
                config = get_architecture_config(arch_name)
                extractor = ConfigurableMultimodalExtractor(self.obs_space, config)

                # Check that CNNs are initialized
                if config.modalities.use_temporal_frames:
                    self.assertIsNotNone(
                        extractor.temporal_cnn,
                        f"{arch_name} should have temporal CNN",
                    )
                if config.modalities.use_global_view:
                    self.assertIsNotNone(
                        extractor.global_cnn,
                        f"{arch_name} should have global CNN",
                    )

    def test_non_graph_architectures_work_without_graph(self):
        """Test that non-graph architectures (MLP baseline) work correctly."""
        config = get_architecture_config("mlp_baseline")
        extractor = ConfigurableMultimodalExtractor(self.obs_space, config)

        # Should not have graph encoder
        self.assertIsNone(extractor.graph_encoder, "MLP baseline should not have graph encoder")

        # Should still work with environment observations
        torch_obs = self._get_observation_tensor()
        with torch.no_grad():
            features = extractor(torch_obs)

        self.assertEqual(features.shape, (1, config.features_dim))

    def test_vision_free_architecture(self):
        """Test vision-free architecture that uses graph + state only."""
        config = get_architecture_config("vision_free")
        extractor = ConfigurableMultimodalExtractor(self.obs_space, config)

        # Should not have vision CNNs
        self.assertIsNone(extractor.temporal_cnn, "Vision-free should not have temporal CNN")
        self.assertIsNone(extractor.global_cnn, "Vision-free should not have global CNN")

        # Should have graph encoder
        self.assertIsNotNone(extractor.graph_encoder, "Vision-free should have graph encoder")

        # Should work with environment observations
        torch_obs = self._get_observation_tensor()
        with torch.no_grad():
            features = extractor(torch_obs)

        self.assertEqual(features.shape, (1, config.features_dim))

    def test_batch_processing(self):
        """Test that architectures handle batched observations correctly."""
        batch_sizes = [1, 4, 8]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                config = get_architecture_config("mlp_baseline")
                extractor = ConfigurableMultimodalExtractor(self.obs_space, config)

                # Create batched observations
                obs, _ = self.env.reset()
                torch_obs = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        # Repeat for batch
                        batched = np.stack([value] * batch_size)
                        torch_obs[key] = torch.from_numpy(batched)

                # Forward pass
                with torch.no_grad():
                    features = extractor(torch_obs)

                self.assertEqual(
                    features.shape,
                    (batch_size, config.features_dim),
                    f"Should handle batch size {batch_size}",
                )


if __name__ == "__main__":
    unittest.main()
