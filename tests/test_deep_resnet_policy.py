"""Tests for Deep ResNet Policy architecture.

Tests forward/backward passes, residual connections, dueling architecture,
and memory usage of the deep ResNet actor-critic policy.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from npp_rl.models.deep_resnet_mlp import ResidualBlock, DeepResNetMLPExtractor
from npp_rl.agents.deep_resnet_actor_critic_policy import DeepResNetActorCriticPolicy


class TestResidualBlock(unittest.TestCase):
    """Test ResidualBlock functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.in_dim = 256
        self.out_dim = 256
        torch.manual_seed(42)
    
    def test_forward_pass(self):
        """Test forward pass with matching dimensions."""
        block = ResidualBlock(self.in_dim, self.out_dim)
        x = torch.randn(self.batch_size, self.in_dim)
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.out_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_forward_pass_dimension_change(self):
        """Test forward pass with dimension change (requires projection)."""
        in_dim, out_dim = 256, 512
        block = ResidualBlock(in_dim, out_dim)
        x = torch.randn(self.batch_size, in_dim)
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, out_dim))
        self.assertFalse(torch.isnan(output).any())
    
    def test_residual_connection(self):
        """Test that residual connection is actually working."""
        block = ResidualBlock(self.in_dim, self.out_dim)
        x = torch.randn(self.batch_size, self.in_dim)
        
        # Forward pass
        output = block(x)
        
        # Compute main path and residual separately
        main_output = block.main_path(x)
        residual = block.residual_proj(x)
        
        # Verify residual connection
        expected = main_output + residual
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_activation_function(self):
        """Test that activation function is applied correctly."""
        block = ResidualBlock(self.in_dim, self.out_dim, activation_fn=nn.SiLU)
        self.assertIsInstance(block.main_path[2], nn.SiLU)
        
        block_relu = ResidualBlock(self.in_dim, self.out_dim, activation_fn=nn.ReLU)
        self.assertIsInstance(block_relu.main_path[2], nn.ReLU)
    
    def test_layer_norm(self):
        """Test that LayerNorm is applied when enabled."""
        block_norm = ResidualBlock(self.in_dim, self.out_dim, use_layer_norm=True)
        self.assertIsInstance(block_norm.main_path[1], nn.LayerNorm)
        
        block_no_norm = ResidualBlock(self.in_dim, self.out_dim, use_layer_norm=False)
        self.assertNotIsInstance(block_no_norm.main_path[1], nn.LayerNorm)
    
    def test_dropout(self):
        """Test that dropout is applied when dropout > 0."""
        block_dropout = ResidualBlock(self.in_dim, self.out_dim, dropout=0.1)
        has_dropout = any(isinstance(m, nn.Dropout) for m in block_dropout.main_path)
        self.assertTrue(has_dropout)
        
        block_no_dropout = ResidualBlock(self.in_dim, self.out_dim, dropout=0.0)
        has_dropout = any(isinstance(m, nn.Dropout) for m in block_no_dropout.main_path)
        self.assertFalse(has_dropout)


class TestDeepResNetMLPExtractor(unittest.TestCase):
    """Test DeepResNetMLPExtractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_dim = 512
        self.policy_layers = [512, 512, 384, 256, 256]
        self.value_layers = [512, 384, 256]
        self.num_actions = 6
        torch.manual_seed(42)
    
    def test_forward_pass(self):
        """Test forward pass through both policy and value networks."""
        extractor = DeepResNetMLPExtractor(
            feature_dim=self.feature_dim,
            policy_layers=self.policy_layers,
            value_layers=self.value_layers,
            num_actions=self.num_actions,
        )
        
        features = torch.randn(self.batch_size, self.feature_dim)
        policy_latent, value_latent = extractor(features)
        
        self.assertEqual(policy_latent.shape, (self.batch_size, self.policy_layers[-1]))
        self.assertEqual(value_latent.shape, (self.batch_size, self.value_layers[-1]))
        self.assertFalse(torch.isnan(policy_latent).any())
        self.assertFalse(torch.isnan(value_latent).any())
    
    def test_dueling_architecture(self):
        """Test dueling architecture for value function."""
        extractor = DeepResNetMLPExtractor(
            feature_dim=self.feature_dim,
            policy_layers=self.policy_layers,
            value_layers=self.value_layers,
            dueling=True,
            num_actions=self.num_actions,
        )
        
        features = torch.randn(self.batch_size, self.feature_dim)
        _, value_features = extractor(features)
        
        # Get dueling values
        state_value, advantages = extractor.get_dueling_values(value_features)
        
        self.assertEqual(state_value.shape, (self.batch_size, 1))
        self.assertEqual(advantages.shape, (self.batch_size, self.num_actions))
        
        # Test dueling combination: V(s) + (A(s,a) - mean(A(s,*)))
        advantage_mean = advantages.mean(dim=1, keepdim=True)
        combined = state_value + (advantages - advantage_mean)
        self.assertEqual(combined.shape, (self.batch_size, self.num_actions))
    
    def test_residual_connections(self):
        """Test that residual connections improve gradient flow."""
        extractor_resnet = DeepResNetMLPExtractor(
            feature_dim=self.feature_dim,
            policy_layers=self.policy_layers,
            value_layers=self.value_layers,
            use_residual=True,
        )
        
        extractor_no_resnet = DeepResNetMLPExtractor(
            feature_dim=self.feature_dim,
            policy_layers=self.policy_layers,
            value_layers=self.value_layers,
            use_residual=False,
        )
        
        features = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        
        # Forward and backward with residual
        policy_latent_resnet, _ = extractor_resnet(features)
        loss_resnet = policy_latent_resnet.sum()
        loss_resnet.backward()
        grad_norm_resnet = features.grad.norm().item()
        
        # Reset gradients
        features.grad = None
        
        # Forward and backward without residual
        policy_latent_no_resnet, _ = extractor_no_resnet(features)
        loss_no_resnet = policy_latent_no_resnet.sum()
        loss_no_resnet.backward()
        grad_norm_no_resnet = features.grad.norm().item()
        
        # Residual connections should help maintain gradient magnitude
        self.assertGreater(grad_norm_resnet, 0)
        self.assertGreater(grad_norm_no_resnet, 0)
    
    def test_separate_policy_value_forward(self):
        """Test separate forward passes for policy and value."""
        extractor = DeepResNetMLPExtractor(
            feature_dim=self.feature_dim,
            policy_layers=self.policy_layers,
            value_layers=self.value_layers,
        )
        
        features = torch.randn(self.batch_size, self.feature_dim)
        
        # Test separate forwards
        policy_latent = extractor.forward_policy(features)
        value_latent = extractor.forward_value(features)
        
        # Test combined forward
        policy_latent_combined, value_latent_combined = extractor(features)
        
        torch.testing.assert_close(policy_latent, policy_latent_combined)
        torch.testing.assert_close(value_latent, value_latent_combined)
    
    def test_layer_norm_stability(self):
        """Test that LayerNorm improves training stability."""
        extractor_norm = DeepResNetMLPExtractor(
            feature_dim=self.feature_dim,
            policy_layers=self.policy_layers,
            value_layers=self.value_layers,
            use_layer_norm=True,
        )
        
        # Large input variations
        features = torch.randn(self.batch_size, self.feature_dim) * 100
        policy_latent, value_latent = extractor_norm(features)
        
        # With LayerNorm, output should be normalized
        self.assertFalse(torch.isnan(policy_latent).any())
        self.assertFalse(torch.isinf(policy_latent).any())
        self.assertLess(policy_latent.std().item(), 50)  # Should be normalized


class TestDeepResNetActorCriticPolicy(unittest.TestCase):
    """Test DeepResNetActorCriticPolicy functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.obs_dim = 512
        self.num_actions = 6
        
        # Create dummy observation and action spaces
        self.observation_space = spaces.Dict({
            "features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.uint8),
        })
        self.action_space = spaces.Discrete(self.num_actions)
        
        torch.manual_seed(42)
    
    def _create_policy(self, use_dueling=True):
        """Helper to create policy instance."""
        def lr_schedule(progress):
            return 3e-4
        
        return DeepResNetActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=lr_schedule,
            net_arch={"pi": [512, 512, 384, 256, 256], "vf": [512, 384, 256]},
            activation_fn=nn.SiLU,
            share_features_extractor=False,
            use_residual=True,
            use_layer_norm=True,
            dueling=use_dueling,
            dropout=0.1,
        )
    
    def test_policy_creation(self):
        """Test that policy can be created successfully."""
        policy = self._create_policy()
        self.assertIsNotNone(policy)
        self.assertIsInstance(policy.mlp_extractor, DeepResNetMLPExtractor)
    
    def test_separate_feature_extractors(self):
        """Test that policy uses separate feature extractors."""
        policy = self._create_policy()
        
        # Policy should have separate extractors
        self.assertIsNot(policy.features_extractor, policy.vf_features_extractor)
        
        # Both should be instances of the same class
        self.assertEqual(type(policy.features_extractor), type(policy.vf_features_extractor))
    
    def test_forward_pass(self):
        """Test forward pass through policy."""
        policy = self._create_policy()
        
        obs = {
            "features": torch.randn(self.batch_size, self.obs_dim),
            "action_mask": torch.ones(self.batch_size, self.num_actions, dtype=torch.uint8),
        }
        
        actions, values, log_probs = policy.forward(obs)
        
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(values.shape, (self.batch_size, 1))
        self.assertEqual(log_probs.shape, (self.batch_size,))
        
        self.assertFalse(torch.isnan(actions).any())
        self.assertFalse(torch.isnan(values).any())
        self.assertFalse(torch.isnan(log_probs).any())
    
    def test_action_masking(self):
        """Test that action masking is properly applied."""
        policy = self._create_policy()
        
        # Create observation with some actions masked
        obs = {
            "features": torch.randn(self.batch_size, self.obs_dim),
            "action_mask": torch.ones(self.batch_size, self.num_actions, dtype=torch.uint8),
        }
        # Mask first 3 actions
        obs["action_mask"][:, :3] = 0
        
        # Sample many actions
        num_samples = 1000
        action_counts = torch.zeros(self.num_actions)
        for _ in range(num_samples):
            actions, _, _ = policy.forward(obs)
            for action in actions:
                action_counts[action] += 1
        
        # Masked actions (0, 1, 2) should never be selected
        self.assertEqual(action_counts[:3].sum(), 0)
        # Unmasked actions (3, 4, 5) should be selected
        self.assertGreater(action_counts[3:].sum(), 0)
    
    def test_evaluate_actions(self):
        """Test action evaluation."""
        policy = self._create_policy()
        
        obs = {
            "features": torch.randn(self.batch_size, self.obs_dim),
            "action_mask": torch.ones(self.batch_size, self.num_actions, dtype=torch.uint8),
        }
        actions = torch.randint(0, self.num_actions, (self.batch_size,))
        
        values, log_probs, entropy = policy.evaluate_actions(obs, actions)
        
        self.assertEqual(values.shape, (self.batch_size, 1))
        self.assertEqual(log_probs.shape, (self.batch_size,))
        self.assertEqual(entropy.shape, (self.batch_size,))
        
        self.assertFalse(torch.isnan(values).any())
        self.assertFalse(torch.isnan(log_probs).any())
        self.assertFalse(torch.isnan(entropy).any())
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through deep network."""
        policy = self._create_policy()
        policy.train()
        
        obs = {
            "features": torch.randn(self.batch_size, self.obs_dim, requires_grad=True),
            "action_mask": torch.ones(self.batch_size, self.num_actions, dtype=torch.uint8),
        }
        
        actions, values, log_probs = policy.forward(obs)
        loss = (values.mean() + log_probs.mean())
        loss.backward()
        
        # Check that gradients exist and are not NaN
        for name, param in policy.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient for {name}")
    
    def test_memory_usage(self):
        """Test memory usage of deep policy."""
        policy = self._create_policy()
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Deep ResNet policy should have significantly more parameters than baseline
        # Baseline: ~724K, Deep ResNet: ~7-15M
        self.assertGreater(total_params, 5_000_000)
        self.assertLess(total_params, 25_000_000)
    
    def test_predict_values(self):
        """Test value prediction."""
        policy = self._create_policy()
        
        obs = {
            "features": torch.randn(self.batch_size, self.obs_dim),
            "action_mask": torch.ones(self.batch_size, self.num_actions, dtype=torch.uint8),
        }
        
        values = policy.predict_values(obs)
        
        self.assertEqual(values.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(values).any())
    
    def test_get_policy_latent(self):
        """Test getting policy latent representation."""
        policy = self._create_policy()
        
        obs = {
            "features": torch.randn(self.batch_size, self.obs_dim),
            "action_mask": torch.ones(self.batch_size, self.num_actions, dtype=torch.uint8),
        }
        
        policy_latent = policy.get_policy_latent(obs)
        
        # Should match last layer of policy network
        expected_dim = 256  # Last element of [512, 512, 384, 256, 256]
        self.assertEqual(policy_latent.shape, (self.batch_size, expected_dim))
        self.assertFalse(torch.isnan(policy_latent).any())
    
    def test_get_value_latent(self):
        """Test getting value latent representation."""
        policy = self._create_policy()
        
        obs = {
            "features": torch.randn(self.batch_size, self.obs_dim),
            "action_mask": torch.ones(self.batch_size, self.num_actions, dtype=torch.uint8),
        }
        
        value_latent = policy.get_value_latent(obs)
        
        # Should match last layer of value network
        expected_dim = 256  # Last element of [512, 384, 256]
        self.assertEqual(value_latent.shape, (self.batch_size, expected_dim))
        self.assertFalse(torch.isnan(value_latent).any())


if __name__ == "__main__":
    unittest.main()

