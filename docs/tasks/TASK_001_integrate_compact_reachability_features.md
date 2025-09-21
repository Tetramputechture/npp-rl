# TASK 001: Integrate Compact Reachability Features with HGT Architecture

## Overview
Integrate the compact reachability features from nclone with the existing HGT-based multimodal feature extractor, enabling reachability-aware spatial reasoning for the RL agent.

## Context & Justification

### Current HGT Architecture
Based on analysis of `/workspace/npp-rl/npp_rl/feature_extractors/hgt_multimodal.py`:
- **Primary Architecture**: Heterogeneous Graph Transformer with type-specific attention
- **Multi-modal Processing**: Visual frames (3D CNN), graph features (HGT), game state (MLP)
- **Advanced Fusion**: Cross-modal attention with spatial awareness
- **Performance Target**: Real-time processing for 60 FPS gameplay

### Integration Strategy
- **Compact Features**: 64-dimensional reachability encoding
- **Guidance Over Ground Truth**: Approximate features for learned spatial reasoning
- **HGT Compatibility**: Leverage graph transformer's ability to learn from compact representations
- **Performance Priority**: <2ms feature extraction for real-time RL

### Research Foundation
- **Heterogeneous Graph Transformers**: Excel at learning spatial relationships from compact numerical features
- **Multi-modal Fusion**: Reachability features complement visual and graph modalities
- **Attention Mechanisms**: Cross-modal attention can learn to weight reachability guidance appropriately

## Technical Specification

### Code Organization and Documentation Requirements

**Import Requirements**:
- Always prefer top-level imports at the module level
- Import all dependencies at the top of the file before any class or function definitions
- Group imports by: standard library, third-party, nclone, npp_rl modules

**Documentation Requirements**:
- **Top-level module docstrings**: Every modified module must have comprehensive docstrings explaining the integration approach and theoretical foundation
- **Inline documentation**: Complex algorithms and model architectures require detailed inline comments explaining the rationale
- **Paper references**: All techniques must reference original research papers in docstrings and comments
- **Integration notes**: Document how each component integrates with existing nclone systems

**Module Modification Approach**:
- **Update existing modules in place** rather than creating separate "enhanced" or "aware" versions
- Extend existing classes with new functionality while maintaining backward compatibility
- Add reachability integration directly to `HGTMultimodalExtractor` in `/npp_rl/feature_extractors/hgt_multimodal.py`

### HGT Architecture Modifications
**Target File**: `/npp_rl/feature_extractors/hgt_multimodal.py`

**Required Documentation Additions**:
```python
"""
HGT Multimodal Feature Extractor with Reachability Integration

This module implements a Heterogeneous Graph Transformer (HGT) based feature extractor
that processes multiple modalities (visual, graph, state) with integrated compact
reachability features from the nclone physics engine.

Integration Strategy:
- Compact reachability features (64-dim) provide spatial guidance without ground truth dependency
- Cross-modal attention mechanisms learn to weight reachability information appropriately
- Performance-optimized for real-time RL training (<2ms feature extraction)

Theoretical Foundation:
- Heterogeneous Graph Transformers: Hu et al. (2020) "Heterogeneous Graph Transformer"
- Multi-modal fusion: Baltrusaitis et al. (2018) "Multimodal Machine Learning: A Survey"
- Attention mechanisms: Vaswani et al. (2017) "Attention Is All You Need"
- Spatial reasoning: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"

nclone Integration:
- Uses TieredReachabilitySystem for performance-tuned feature extraction
- Integrates with compact feature representations from nclone.graph.reachability
- Maintains compatibility with existing NPP physics constants and game state
"""

# Standard library imports
import math
import time
from typing import Dict, Tuple, Optional, Any

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# nclone imports (top-level imports preferred)
from nclone.constants import NINJA_RADIUS, GRAVITY_FALL, MAX_HOR_SPEED
from nclone.graph.reachability.compact_features import ReachabilityFeatureExtractor
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem

# npp_rl imports
from npp_rl.feature_extractors.base import BaseFeatureExtractor
from npp_rl.models.attention import CrossModalAttention


class HGTMultimodalExtractor(BaseFeatureExtractor):
    """
    HGT-based multimodal feature extractor with integrated reachability features.
    
    This extractor processes visual frames, graph structures, game state, and compact
    reachability features through a unified Heterogeneous Graph Transformer architecture.
    
    Architecture Components:
    1. Visual Processing: 3D CNN for temporal frames + 2D CNN for global view
    2. Graph Processing: HGT with type-specific attention mechanisms
    3. State Processing: MLP for physics/game state features  
    4. Reachability Processing: Compact feature integration with cross-modal attention
    5. Multimodal Fusion: Attention-based fusion with reachability awareness
    
    Performance Optimizations:
    - Lazy initialization of reachability extractor for dependency management
    - Caching mechanisms for repeated reachability computations
    - Tier-1 reachability extraction for real-time performance (<2ms target)
    
    References:
    - HGT Architecture: Hu et al. (2020) "Heterogeneous Graph Transformer"
    - Cross-modal Attention: Baltrusaitis et al. (2018) "Multimodal Machine Learning"
    - Reachability Integration: Custom integration with nclone physics system
    """
    
    def __init__(self, observation_space, features_dim=512, 
                 hgt_hidden_dim=256, hgt_num_layers=3,
                 reachability_dim=64, **kwargs):
        super().__init__(observation_space, features_dim, hgt_hidden_dim, 
                        hgt_num_layers, **kwargs)
        
        # Reachability feature processing components
        # Based on compact feature dimensionality from nclone.graph.reachability
        self.reachability_dim = reachability_dim
        
        # Multi-layer reachability encoder with batch normalization for stability
        # Architecture inspired by residual connections from He et al. (2016)
        self.reachability_encoder = nn.Sequential(
            nn.Linear(reachability_dim, 128),
            nn.BatchNorm1d(128),  # Stabilize training with reachability features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Compact representation for attention mechanisms
        )
        
        # Cross-modal attention for reachability integration
        # Implementation based on Baltrusaitis et al. (2018) multimodal fusion principles
        self.reachability_attention = ReachabilityAttentionModule(
            visual_dim=self.cnn_output_dim,
            graph_dim=self.hgt_output_dim,
            state_dim=self.mlp_output_dim,
            reachability_dim=32
        )
        
        # Update final fusion layer to accommodate reachability features
        # Maintain output dimensionality while integrating new modality
        total_dim = (self.cnn_output_dim + self.hgt_output_dim + 
                    self.mlp_output_dim + 32)
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        # Lazy initialization to handle nclone dependency gracefully
        # Allows fallback behavior if reachability system is unavailable
        self.reachability_extractor = None
    
    def _initialize_reachability_extractor(self):
        """
        Lazy initialization of reachability feature extractor.
        
        This approach allows the model to be instantiated even when nclone
        reachability components are not available, enabling development and
        testing in environments without full physics integration.
        
        Implementation follows the dependency injection pattern for better
        testability and modularity as described in Martin (2008) "Clean Code".
        """
        if self.reachability_extractor is None:
            try:
                # Import at runtime to avoid hard dependency on nclone at module level
                # Use TieredReachabilitySystem for performance-optimized feature extraction
                tiered_system = TieredReachabilitySystem()
                self.reachability_extractor = ReachabilityFeatureExtractor(tiered_system)
            except ImportError:
                # Graceful degradation: use dummy extractor during development
                # This ensures the model can still function for testing and development
                self.reachability_extractor = DummyReachabilityExtractor()
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with integrated reachability feature processing.
        
        This method extends the base HGT forward pass to incorporate compact
        reachability features as an additional modality. The integration follows
        the multimodal fusion approach from Baltrusaitis et al. (2018).
        
        Args:
            observations: Dict containing visual frames, graph data, state info,
                         and optionally pre-computed reachability features
                         
        Returns:
            torch.Tensor: Fused feature representation for policy/value networks
            
        Note:
            Performance target is <2ms for real-time RL training compatibility.
            Reachability extraction uses Tier-1 algorithms for speed optimization.
        """
        # Lazy initialization ensures graceful handling of missing dependencies
        if self.reachability_extractor is None:
            self._initialize_reachability_extractor()
        
        # Process each modality through dedicated pathways
        # Visual: 3D CNN for temporal dynamics, 2D CNN for global spatial context
        visual_features = self._process_visual_observations(observations)
        
        # Graph: HGT with heterogeneous attention for structural relationships
        # Based on Hu et al. (2020) type-aware graph transformer architecture
        graph_features = self._process_graph_observations(observations)
        
        # State: MLP processing of physics and game state variables
        state_features = self._process_state_observations(observations)
        
        # Reachability: Extract compact spatial reasoning features from nclone
        # These provide approximate reachability guidance without ground truth dependency
        reachability_features = self._extract_reachability_features(observations)
        
        # Encode reachability features through dedicated neural pathway
        # Batch normalization stabilizes training with heterogeneous feature scales
        processed_reachability = self.reachability_encoder(reachability_features)
        
        # Cross-modal attention fusion integrating all four modalities
        # Allows model to learn optimal weighting of reachability guidance
        fused_features = self.reachability_attention(
            visual_features, graph_features, state_features, processed_reachability
        )
        
        # Final transformation to target feature dimensionality
        output = self.final_fusion(fused_features)
        
        return output
    
    def _extract_reachability_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract compact reachability features from observations.
        """
        batch_size = observations['player_frames'].shape[0]
        device = observations['player_frames'].device
        
        # Extract game state information
        ninja_positions = self._extract_ninja_positions(observations)
        level_data = self._extract_level_data(observations)
        switch_states = self._extract_switch_states(observations)
        
        # Batch process reachability features
        reachability_features = []
        
        for i in range(batch_size):
            try:
                features = self.reachability_extractor.extract_features(
                    ninja_pos=ninja_positions[i],
                    level_data=level_data[i],
                    switch_states=switch_states[i],
                    performance_target="fast"  # Use Tier 1 for real-time performance
                )
                reachability_features.append(features)
            except Exception as e:
                # Fallback: zero features if extraction fails
                print(f"Warning: Reachability extraction failed: {e}")
                features = torch.zeros(self.reachability_dim, device=device)
                reachability_features.append(features)
        
        return torch.stack(reachability_features).to(device)
```

### Reachability Attention Module
**Add to the same file**: `/npp_rl/feature_extractors/hgt_multimodal.py`

```python
class ReachabilityAttentionModule(nn.Module):
    """
    Cross-modal attention module for integrating reachability features with multimodal representations.
    
    This module implements the attention-based fusion mechanism from Baltrusaitis et al. (2018)
    "Multimodal Machine Learning: A Survey", adapted for spatial reasoning integration in RL.
    
    The architecture uses cross-modal attention to allow each modality to attend to
    reachability features, followed by gating mechanisms to modulate feature importance
    based on spatial reachability context.
    
    Architecture Components:
    - Cross-modal attention between each modality and reachability features
    - Learned gating mechanism for reachability-guided feature enhancement
    - Multi-layer fusion network for final feature integration
    
    References:
    - Multimodal fusion: Baltrusaitis et al. (2018) "Multimodal Machine Learning: A Survey"
    - Attention mechanisms: Vaswani et al. (2017) "Attention Is All You Need"
    - Gating networks: Dauphin et al. (2017) "Language Modeling with Gated Convolutional Networks"
    
    Note:
        This module should be integrated directly into the HGTMultimodalExtractor
        rather than created as a separate file to maintain architectural cohesion.
    """
    
    def __init__(self, visual_dim, graph_dim, state_dim, reachability_dim):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.graph_dim = graph_dim
        self.state_dim = state_dim
        self.reachability_dim = reachability_dim
        
        # Attention mechanisms
        self.visual_reachability_attention = CrossModalAttention(
            query_dim=visual_dim, key_dim=reachability_dim, hidden_dim=128
        )
        self.graph_reachability_attention = CrossModalAttention(
            query_dim=graph_dim, key_dim=reachability_dim, hidden_dim=128
        )
        self.state_reachability_attention = CrossModalAttention(
            query_dim=state_dim, key_dim=reachability_dim, hidden_dim=64
        )
        
        # Reachability-guided feature enhancement
        self.reachability_gate = nn.Sequential(
            nn.Linear(reachability_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 gates for visual, graph, state
            nn.Sigmoid()
        )
        
        # Final fusion
        total_dim = visual_dim + graph_dim + state_dim + reachability_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_dim // 2, total_dim // 2)
        )
    
    def forward(self, visual_features, graph_features, state_features, reachability_features):
        """
        Fuse multimodal features with reachability awareness.
        """
        # Apply cross-modal attention
        visual_attended = self.visual_reachability_attention(
            visual_features, reachability_features
        )
        graph_attended = self.graph_reachability_attention(
            graph_features, reachability_features
        )
        state_attended = self.state_reachability_attention(
            state_features, reachability_features
        )
        
        # Compute reachability-based gating
        gates = self.reachability_gate(reachability_features)
        visual_gate, graph_gate, state_gate = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        
        # Apply gating to enhance relevant features
        visual_enhanced = visual_attended * (1 + visual_gate)
        graph_enhanced = graph_attended * (1 + graph_gate)
        state_enhanced = state_attended * (1 + state_gate)
        
        # Concatenate all features
        fused = torch.cat([
            visual_enhanced, graph_enhanced, state_enhanced, reachability_features
        ], dim=1)
        
        # Final fusion
        output = self.fusion_layer(fused)
        
        return output

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for feature fusion.
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, query_dim)
        
        self.scale = hidden_dim ** -0.5
    
    def forward(self, query, key):
        """
        Apply cross-modal attention.
        
        Args:
            query: Features to be attended (e.g., visual features)
            key: Features providing attention context (e.g., reachability features)
        """
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(key)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Project back to query dimension
        output = self.output_proj(attended)
        
        # Residual connection
        return query + output
```

## Implementation Plan

### Core Integration
**Objective**: Integrate reachability features into existing HGT architecture

**Approach**:
- **Update existing modules in place** rather than creating new enhanced versions
- Maintain backward compatibility while adding reachability functionality
- Follow top-level import patterns and comprehensive documentation standards

**Key Modifications**:
1. **HGT Feature Extractor**: Extend `HGTMultimodalExtractor` in `/npp_rl/feature_extractors/hgt_multimodal.py`
2. **Attention Integration**: Add `ReachabilityAttentionModule` to same file for architectural cohesion  
3. **Interface Components**: Create utility functions for nclone integration within existing modules

**Documentation Requirements**:
- Add comprehensive module-level docstrings with theoretical foundations
- Include inline documentation explaining integration approach and performance considerations
- Reference all relevant research papers in docstrings and comments
- Document nclone integration points and fallback mechanisms

### Environment Integration  
**Objective**: Extend environment observations to include reachability information

**Approach**:
- **Modify existing environment classes** rather than creating wrappers
- Extend observation spaces to include reachability features
- Implement efficient batch processing for real-time performance

**Target Environment**: Modify existing `NPPEnv` class in place

**Implementation Example** (modify existing environment):
```python
# Add to existing NPPEnv class in npp_rl/environments/npp_env.py

class NPPEnv:
    """
    NPP Environment with integrated reachability feature support.
    
    This environment extends the base NPP gameplay with compact reachability
    features extracted from the nclone physics system. Reachability information
    is provided as part of the observation space to guide spatial reasoning.
    
    Integration Components:
    - Reachability feature extraction using TieredReachabilitySystem
    - Extended observation space including 64-dimensional reachability features
    - Performance-optimized caching for real-time extraction (<2ms target)
    - Graceful fallback when nclone components are unavailable
    
    References:
    - Reachability analysis: Custom integration with nclone physics system
    - Observation space design: Gym environment standards
    - Performance optimization: Real-time RL training requirements
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize reachability components
        self.reachability_extractor = self._initialize_reachability_extractor()
        self.reachability_cache = {}
        self.cache_ttl = 100  # milliseconds
        
        # Extend observation space
        self._extend_observation_space()
    
    def _initialize_reachability_extractor(self):
        """Initialize reachability feature extractor."""
        try:
            from nclone.graph.reachability.compact_features import ReachabilityFeatureExtractor
            from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
            
            tiered_system = TieredReachabilitySystem()
            return ReachabilityFeatureExtractor(tiered_system)
        except ImportError:
            print("Warning: nclone reachability system not available, using dummy extractor")
            return DummyReachabilityExtractor()
    
    def _extend_observation_space(self):
        """Extend observation space to include reachability features."""
        # Add reachability features to observation space
        reachability_space = spaces.Box(
            low=0.0, high=2.0, shape=(64,), dtype=np.float32
        )
        
        # Update observation space
        if isinstance(self.observation_space, spaces.Dict):
            self.observation_space.spaces['reachability_features'] = reachability_space
        else:
            # Convert to Dict space if not already
            original_space = self.observation_space
            self.observation_space = spaces.Dict({
                'original_obs': original_space,
                'reachability_features': reachability_space
            })
    
    def step(self, action):
        """Enhanced step with reachability feature extraction."""
        obs, reward, done, info = super().step(action)
        
        # Extract reachability features
        reachability_features = self._extract_reachability_features()
        
        # Add to observation
        if isinstance(obs, dict):
            obs['reachability_features'] = reachability_features
        else:
            obs = {
                'original_obs': obs,
                'reachability_features': reachability_features
            }
        
        # Add reachability info for debugging
        info['reachability'] = {
            'extraction_time_ms': getattr(reachability_features, 'extraction_time', 0.0),
            'cache_hit': getattr(reachability_features, 'from_cache', False),
            'confidence': getattr(reachability_features, 'confidence', 1.0)
        }
        
        return obs, reward, done, info
    
    def _extract_reachability_features(self) -> np.ndarray:
        """Extract reachability features for current game state."""
        # Get current game state
        ninja_pos = self._get_ninja_position()
        level_data = self._get_level_data()
        switch_states = self._get_switch_states()
        
        # Check cache first
        cache_key = self._generate_cache_key(ninja_pos, switch_states)
        if self._is_cache_valid(cache_key):
            return self.reachability_cache[cache_key]['features']
        
        # Extract features
        try:
            start_time = time.perf_counter()
            features_tensor = self.reachability_extractor.extract_features(
                ninja_pos=ninja_pos,
                level_data=level_data,
                switch_states=switch_states,
                performance_target="fast"
            )
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            # Convert to numpy
            features = features_tensor.detach().cpu().numpy().astype(np.float32)
            
            # Cache result
            self._cache_features(cache_key, features, extraction_time)
            
            return features
            
        except Exception as e:
            print(f"Warning: Reachability feature extraction failed: {e}")
            # Return zero features as fallback
            return np.zeros(64, dtype=np.float32)
```

### Training Integration
**Objective**: Integrate reachability-aware feature extraction into existing training pipeline

**Approach**:
- **Modify existing training scripts** to support reachability feature extraction
- Update existing configuration systems to include reachability parameters
- Add reachability-specific monitoring to existing performance tracking

**Key Modifications**:
1. **Training Scripts**: Update existing training modules to support reachability-aware extractors
2. **Configuration**: Extend existing config files with reachability integration parameters
3. **Monitoring**: Add reachability feature metrics to existing training monitoring systems

**Documentation Requirements**:
- Document reachability hyperparameters and their effects on training
- Include performance benchmarking guidelines for reachability integration
- Reference relevant RL and multimodal learning literature in training modifications

### Evaluation and Optimization
**Objective**: Assess integration impact and optimize performance

**Evaluation Approach**:
- Compare reachability-integrated models with baseline HGT performance
- Analyze attention patterns to understand reachability feature utilization
- Profile computational overhead and optimize for real-time performance requirements

**Optimization Focus**:
- Memory efficiency for batch processing of reachability features
- Computational optimization for <2ms feature extraction target
- Attention mechanism efficiency for real-time RL training

## Testing Strategy

### Unit Tests
**Test existing modified classes with reachability integration**

```python
class TestHGTMultimodalExtractorWithReachability(unittest.TestCase):
    """Test reachability integration in the modified HGTMultimodalExtractor."""
    
    def setUp(self):
        self.observation_space = create_mock_observation_space()
        # Test the modified existing extractor with reachability features enabled
        self.extractor = HGTMultimodalExtractor(
            observation_space=self.observation_space,
            features_dim=512,
            reachability_dim=64  # Enable reachability integration
        )
    
    def test_forward_pass(self):
        """Test that forward pass works with reachability features."""
        batch_size = 4
        observations = create_mock_observations(batch_size)
        
        # Add reachability features
        observations['reachability_features'] = torch.randn(batch_size, 64)
        
        # Forward pass
        features = self.extractor(observations)
        
        self.assertEqual(features.shape, (batch_size, 512))
        self.assertTrue(torch.all(torch.isfinite(features)))
    
    def test_reachability_attention(self):
        """Test reachability attention mechanism."""
        batch_size = 2
        visual_features = torch.randn(batch_size, 256)
        graph_features = torch.randn(batch_size, 128)
        state_features = torch.randn(batch_size, 64)
        reachability_features = torch.randn(batch_size, 32)
        
        attention_module = ReachabilityAttentionModule(256, 128, 64, 32)
        
        output = attention_module(
            visual_features, graph_features, state_features, reachability_features
        )
        
        expected_dim = 256 + 128 + 64 + 32
        self.assertEqual(output.shape, (batch_size, expected_dim // 2))
    
    def test_fallback_behavior(self):
        """Test fallback behavior when reachability extraction fails."""
        # Mock failed reachability extraction
        self.extractor.reachability_extractor = FailingReachabilityExtractor()
        
        batch_size = 2
        observations = create_mock_observations(batch_size)
        
        # Should not crash, should use zero features
        features = self.extractor(observations)
        
        self.assertEqual(features.shape, (batch_size, 512))
        self.assertTrue(torch.all(torch.isfinite(features)))
```

### Integration Tests
```python
class TestNPPEnvironmentReachabilityIntegration(unittest.TestCase):
    """Test reachability feature integration in the modified NPPEnv."""
    
    def setUp(self):
        # Test the modified existing environment with reachability features enabled
        self.env = NPPEnv(
            render_mode='rgb_array',
            custom_map_path='test_maps/simple_level',
        )
    
    def test_observation_space_extension(self):
        """Test that observation space includes reachability features."""
        obs_space = self.env.observation_space
        
        self.assertIsInstance(obs_space, spaces.Dict)
        self.assertIn('reachability_features', obs_space.spaces)
        
        reachability_space = obs_space.spaces['reachability_features']
        self.assertEqual(reachability_space.shape, (64,))
        self.assertEqual(reachability_space.dtype, np.float32)
    
    def test_step_with_reachability(self):
        """Test that step returns reachability features."""
        obs = self.env.reset()
        
        # Take a step
        action = self.env.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        
        # Check reachability features
        self.assertIn('reachability_features', obs)
        reachability_features = obs['reachability_features']
        
        self.assertEqual(reachability_features.shape, (64,))
        self.assertTrue(np.all(np.isfinite(reachability_features)))
        
        # Check info
        self.assertIn('reachability', info)
        reachability_info = info['reachability']
        self.assertIn('extraction_time_ms', reachability_info)
        self.assertIn('cache_hit', reachability_info)
    
    def test_performance_requirements(self):
        """Test that reachability extraction meets performance requirements."""
        obs = self.env.reset()
        
        extraction_times = []
        for _ in range(100):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            extraction_time = info['reachability']['extraction_time_ms']
            extraction_times.append(extraction_time)
            
            if done:
                obs = self.env.reset()
        
        # Check performance targets
        avg_time = np.mean(extraction_times)
        p95_time = np.percentile(extraction_times, 95)
        
        self.assertLess(avg_time, 2.0, f"Average extraction time too high: {avg_time}ms")
        self.assertLess(p95_time, 5.0, f"95th percentile time too high: {p95_time}ms")
```

### Training Tests
```python
class TestReachabilityIntegratedTraining(unittest.TestCase):
    """Test training compatibility with reachability-integrated components."""
    
    def test_training_compatibility(self):
        """Test that modified HGT extractor works with PPO training."""
        # Create environment with reachability features enabled
        env = NPPEnv(
            render_mode='rgb_array',
        )
        
        # Create model with modified HGT extractor (reachability integration enabled)
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs={
                "features_extractor_class": HGTMultimodalExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 512,
                    "reachability_dim": 64  # Enable reachability integration
                }
            },
            verbose=1
        )
        
        # Short training run
        model.learn(total_timesteps=1000)
        
        # Test prediction
        obs = env.reset()
        action, _states = model.predict(obs)
        
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < env.action_space.n)
    
    def test_feature_learning(self):
        """Test that model learns to use reachability features."""
        # This would be a longer test to verify that the model
        # actually learns to use reachability information effectively
        # Could compare performance with/without reachability features
        pass
```

## Success Criteria

### Performance Requirements
- **Feature Extraction**: <2ms average for reachability features
- **Training Speed**: <10% slowdown compared to standard HGT
- **Memory Usage**: <100MB additional memory for reachability processing

### Quality Requirements
- **Integration Stability**: No crashes or errors during training
- **Feature Quality**: Reachability features show appropriate variance and sensitivity
- **Attention Learning**: Cross-modal attention learns meaningful patterns

### Training Requirements
- **Convergence**: Model converges with reachability features
- **Performance**: Comparable or better level completion rates
- **Generalization**: Improved performance on unseen levels

## Risk Mitigation

### Technical Risks
1. **Performance Overhead**: Continuous monitoring and optimization
2. **Feature Quality**: Validation of reachability feature informativeness
3. **Integration Complexity**: Comprehensive testing and fallback mechanisms

### Training Risks
1. **Convergence Issues**: A/B testing with standard HGT
2. **Overfitting**: Regularization and validation monitoring
3. **Feature Noise**: Robust feature encoding and error handling

## Deliverables

1. **Modified HGTMultimodalExtractor**: Existing HGT architecture extended with reachability integration
2. **Enhanced NPPEnv**: Existing environment updated with reachability feature support
3. **Training Pipeline Updates**: Modified training modules supporting reachability-aware feature extraction
4. **Performance Analysis**: Comprehensive evaluation of integration impact on existing systems
5. **Documentation**: Updated module docstrings and inline documentation with theoretical foundations
6. **Integration Testing**: Test suites validating reachability integration in existing components

**Key Principle**: All modifications maintain backward compatibility while extending functionality

## Dependencies

### Internal Dependencies
- **nclone Tasks**: TASK_001 (Tiered Reachability System), TASK_003 (Compact Features)
- **HGT Architecture**: `/workspace/npp-rl/npp_rl/feature_extractors/hgt_multimodal.py`
- **Training Pipeline**: `/workspace/npp-rl/npp_rl/agents/training.py`

### External Dependencies
- **PyTorch**: Neural network implementation
- **Stable Baselines3**: PPO integration
- **NumPy**: Numerical operations

## References

1. **Strategic Analysis**: `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`
2. **HGT Implementation**: `/workspace/npp-rl/npp_rl/feature_extractors/hgt_multimodal.py`
3. **nclone Integration**: `/workspace/npp-rl/tasks/TASK_002_integrate_reachability_system.md`
4. **Compact Features**: `/workspace/nclone/docs/tasks/TASK_003_create_compact_reachability_features.md`