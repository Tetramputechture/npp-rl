# TASK 002: Implement Reachability-Aware Curiosity Module

## Overview
Enhance the existing intrinsic motivation system with reachability-aware curiosity that avoids wasting exploration on unreachable areas, improving sample efficiency and training performance.

## Context & Justification

### Current Intrinsic Motivation System
Based on analysis of `/workspace/npp-rl/npp_rl/intrinsic/` and README.md:
- **ICM (Intrinsic Curiosity Module)**: Forward/inverse model prediction errors
- **Novelty Detection**: Count-based exploration with state discretization
- **Adaptive Scaling**: Dynamic adjustment based on training progress
- **Integration**: Combined with extrinsic rewards for exploration guidance

### Problem with Current Approach
- **Wasted Exploration**: Agent explores unreachable areas, reducing sample efficiency
- **Uniform Curiosity**: Equal curiosity for all novel states regardless of reachability
- **No Strategic Guidance**: Curiosity doesn't consider level completion objectives
- **Inefficient Learning**: Time spent on impossible areas delays meaningful progress

### Research Foundation
From `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`:
- **Reachability-Guided Exploration**: Focus curiosity on reachable but unexplored areas
- **Frontier Exploration**: Prioritize areas that become reachable after switch activation
- **Strategic Curiosity**: Weight exploration by proximity to level objectives
- **Efficiency Gains**: 2-3x improvement in sample efficiency on complex levels

### Theoretical Justification
- **Pathak et al. (2017)**: ICM works best when prediction errors correlate with meaningful exploration
- **Ecoffet et al. (2019)**: Go-Explore shows benefits of avoiding impossible states
- **Burda et al. (2018)**: Random Network Distillation benefits from reachability constraints

## Technical Specification

### Code Organization and Documentation Requirements

**Import Requirements**:
- Always prefer top-level imports at the module level
- Import all dependencies at the top of the file before any class or function definitions
- Group imports by: standard library, third-party, nclone, npp_rl modules

**Documentation Requirements**:
- **Top-level module docstrings**: Every modified module must have comprehensive docstrings explaining the reachability integration approach and theoretical foundation
- **Inline documentation**: Complex curiosity algorithms require detailed inline comments explaining the reachability-aware modifications
- **Paper references**: All curiosity and exploration techniques must reference original research papers in docstrings and comments
- **Integration notes**: Document how each component integrates with existing nclone reachability systems

**Module Modification Approach**:
- **Update existing intrinsic motivation modules in place** rather than creating separate reachability-aware versions
- Extend existing ICM and novelty detection classes with reachability functionality while maintaining backward compatibility
- Add reachability integration directly to existing modules in `/npp_rl/intrinsic/`

### Intrinsic Motivation Module Modifications
**Target Files**: 
- `/npp_rl/intrinsic/icm.py` (existing ICM module)
- `/npp_rl/intrinsic/novelty.py` (existing novelty detection)
- `/npp_rl/intrinsic/base.py` (existing base classes)

**Required Documentation Additions**:
```python
"""
Intrinsic Curiosity Module with Reachability Integration

This module enhances the existing Intrinsic Curiosity Module (ICM) with reachability-aware
exploration that avoids wasting curiosity on unreachable areas, improving sample efficiency.

Integration Strategy:
- Reachability scaling modulates base curiosity based on accessibility analysis
- Frontier detection boosts exploration of newly reachable areas from nclone physics
- Strategic weighting prioritizes exploration near level completion objectives
- Performance-optimized for real-time RL training (<1ms curiosity computation)

Theoretical Foundation:
- Base ICM: Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"  
- Reachability guidance: Ecoffet et al. (2019) "Go-Explore: a New Approach for Hard-Exploration Problems"
- Strategic exploration: Burda et al. (2018) "Exploration by Random Network Distillation"
- Frontier exploration: Stanton & Clune (2018) "Deep curiosity search"

nclone Integration:
- Uses compact reachability features from nclone.graph.reachability for accessibility assessment
- Integrates with TieredReachabilitySystem for performance-optimized reachability queries
- Maintains compatibility with existing NPP physics constants and level objectives
"""

# Standard library imports
import math
import time
from collections import deque
from typing import Dict, Tuple, Optional, Set, List

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
from npp_rl.intrinsic.base import BaseIntrinsicModule
from npp_rl.intrinsic.utils import compute_prediction_error


class ICMModule(BaseIntrinsicModule):
    """
    Intrinsic Curiosity Module with integrated reachability awareness.
    
    This module extends the base ICM architecture from Pathak et al. (2017) with
    reachability-aware exploration that scales curiosity based on accessibility.
    
    Architecture Enhancements:
    1. Base ICM: Forward/inverse model prediction errors (existing functionality)
    2. Reachability Scaling: Modulate curiosity based on reachability analysis
    3. Frontier Detection: Boost curiosity for newly accessible areas
    4. Strategic Weighting: Prioritize exploration near level objectives
    
    Performance Optimizations:
    - Lazy initialization of reachability extractor for dependency management
    - Caching mechanisms for repeated reachability computations
    - <1ms curiosity computation target for real-time RL training
    
    References:
    - ICM Foundation: Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"
    - Reachability Guidance: Ecoffet et al. (2019) "Go-Explore: a New Approach for Hard-Exploration Problems"
    - Strategic Exploration: Custom integration with nclone physics system
    """
    
    def __init__(self, observation_space, action_space, feature_dim=256, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        
        # Base ICM components (existing functionality preserved)
        self.feature_dim = feature_dim
        self.forward_model = self._build_forward_model()
        self.inverse_model = self._build_inverse_model()
        
        # Reachability integration components (new functionality)
        # Lazy initialization to handle nclone dependency gracefully
        # This allows the model to function without reachability components during development
        self.reachability_extractor = None  
        
        # Reachability-aware enhancement components
        # Based on theoretical foundations from Go-Explore and frontier exploration literature
        self.reachability_predictor = ReachabilityPredictor(
            observation_space, hidden_dim=128
        )
        self.frontier_detector = FrontierDetector(
            reachability_dim=64, memory_size=1000
        )
        self.strategic_weighter = StrategicWeighter(
            reachability_dim=64, objective_dim=32
        )
        
        # Reachability scaling hyperparameters
        # These parameters modulate base ICM curiosity based on accessibility
        self.reachability_scale_factor = 2.0      # Boost for reachable areas
        self.frontier_boost_factor = 3.0          # Extra boost for newly accessible areas
        self.strategic_weight_factor = 1.5        # Weight for objective-proximate areas
        self.unreachable_penalty = 0.1            # Penalty for confirmed unreachable areas
        
        # Exploration state tracking for frontier detection
        # Maintains history of reachable positions to detect newly accessible areas
        self.exploration_history = ExplorationHistory(max_size=10000)
        self.last_reachable_positions = set()
    
    def compute_intrinsic_reward(self, obs, action, next_obs, info=None):
        """
        Compute intrinsic reward with optional reachability awareness.
        
        This method extends the base ICM intrinsic reward computation from Pathak et al. (2017)
        with reachability-aware modulation when enabled. The integration follows the exploration
        efficiency principles from Ecoffet et al. (2019) "Go-Explore".
        
        Args:
            obs: Current observation batch
            action: Action taken
            next_obs: Next observation batch  
            info: Environment info containing optional reachability data
        
        Returns:
            Intrinsic reward tensor, optionally enhanced with reachability awareness
            
        Note:
            Falls back to base ICM computation when reachability awareness is disabled
            or when reachability information is unavailable.
        """
        # Compute base ICM curiosity reward using forward/inverse model prediction errors
        # This preserves existing ICM functionality from Pathak et al. (2017)
        base_reward = self._compute_base_icm_reward(obs, action, next_obs)
        
        # Initialize reachability extractor if needed (lazy loading for dependency management)
        if self.reachability_extractor is None:
            self._initialize_reachability_extractor()
        
        # Extract reachability information from environment info or compute from observations
        reachability_info = self._extract_reachability_info(obs, next_obs, info)

        # Apply reachability-aware modulation to base curiosity
        # Each component implements a specific aspect of reachability-guided exploration
        
        # 1. Scale curiosity based on reachability status (avoid unreachable areas)
        # Implementation based on Go-Explore's accessibility filtering principles
        reachability_scale = self._compute_reachability_scale(reachability_info)
        
        # 2. Boost curiosity for frontier exploration (newly accessible areas)
        # Based on frontier-based exploration from Stanton & Clune (2018)
        frontier_boost = self._compute_frontier_boost(reachability_info)
        
        # 3. Weight exploration by strategic value (proximity to objectives)
        # Guides exploration toward level completion rather than random novelty
        strategic_weight = self._compute_strategic_weight(reachability_info)
        
        # Combine all reachability factors with base ICM reward
        # Multiplicative combination allows each factor to modulate the base curiosity
        enhanced_reward = (base_reward * reachability_scale * 
                          frontier_boost * strategic_weight)
        
        # Update exploration history for frontier detection and strategic planning
        # This maintains state for detecting newly accessible areas in future steps
        self._update_exploration_history(reachability_info, enhanced_reward)
        
        return enhanced_reward
    
    def _compute_reachability_scale(self, reachability_info):
        """
        Scale curiosity based on reachability status.
        
        Scaling Strategy:
        - Reachable areas: 1.0x (full curiosity)
        - Frontier areas: 0.5x (moderate curiosity)
        - Unreachable areas: 0.1x (minimal curiosity)
        """
        current_pos = reachability_info['current_position']
        target_pos = reachability_info['target_position']
        reachable_positions = reachability_info['reachable_positions']
        
        # Check if target position is reachable
        if target_pos in reachable_positions:
            return 1.0  # Full curiosity for reachable areas
        
        # Check if target is on exploration frontier
        if self._is_on_exploration_frontier(target_pos, reachable_positions):
            return 0.5  # Moderate curiosity for frontier
        
        # Check if target has been confirmed unreachable
        if self._is_confirmed_unreachable(target_pos):
            return self.unreachable_penalty  # Minimal curiosity
        
        # Default: unknown area gets moderate curiosity
        return 0.3
    
    def _compute_frontier_boost(self, reachability_info):
        """
        Boost curiosity for newly reachable areas (frontiers).
        """
        current_reachable = set(reachability_info['reachable_positions'])
        
        # Detect newly reachable areas
        newly_reachable = current_reachable - self.last_reachable_positions
        
        if newly_reachable:
            # Update frontier detector
            self.frontier_detector.update_frontiers(newly_reachable)
            
            # Check if current exploration is in frontier area
            target_pos = reachability_info['target_position']
            if self.frontier_detector.is_in_frontier(target_pos):
                return self.frontier_boost_factor
        
        # Update last reachable positions
        self.last_reachable_positions = current_reachable
        
        return 1.0  # No boost
    
    def _compute_strategic_weight(self, reachability_info):
        """
        Weight curiosity based on strategic value (proximity to objectives).
        """
        target_pos = reachability_info['target_position']
        objective_distances = reachability_info.get('objective_distances', [])
        
        if not objective_distances:
            return 1.0  # No strategic information available
        
        # Find closest objective distance
        min_objective_distance = min(objective_distances)
        
        # Strategic weight: higher for areas closer to objectives
        # Use exponential decay: weight = exp(-distance / scale)
        distance_scale = 100.0  # pixels
        strategic_weight = math.exp(-min_objective_distance / distance_scale)
        
        # Scale by strategic weight factor
        return 1.0 + (strategic_weight * (self.strategic_weight_factor - 1.0))
```

### Supporting Components
**Add to existing intrinsic module files** rather than creating separate modules

#### 1. Reachability Predictor  
**Target File**: Add to existing `/npp_rl/intrinsic/icm.py`

```python
class ReachabilityPredictor(nn.Module):
    """
    Neural predictor for position reachability assessment within ICM framework.
    
    This component helps the ICM module make reachability assessments when detailed
    reachability analysis from nclone is not available or too computationally expensive.
    The predictor learns from ground truth reachability data during training.
    
    Architecture based on standard supervised learning for binary classification,
    adapted for spatial reasoning tasks in NPP environments.
    
    References:
    - Binary classification: Standard neural network literature
    - Spatial reasoning: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"
    - Integration pattern: Custom design for ICM enhancement
    
    Note:
        This class should be integrated into the existing ICM module file
        to maintain architectural cohesion and avoid module proliferation.
    """
    
    def __init__(self, observation_space, hidden_dim=128):
        super().__init__()
        
        # Extract observation dimensions
        self.obs_dim = self._get_observation_dim(observation_space)
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(self.obs_dim + 2, hidden_dim),  # +2 for target position
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Binary reachability prediction
            nn.Sigmoid()
        )
        
        # Training components
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def predict_reachability(self, obs, target_pos):
        """
        Predict if target position is reachable from current observation.
        
        Returns:
            Probability that target position is reachable (0-1)
        """
        # Flatten observation
        obs_flat = self._flatten_observation(obs)
        
        # Normalize target position
        target_normalized = self._normalize_position(target_pos)
        
        # Combine inputs
        input_tensor = torch.cat([obs_flat, target_normalized], dim=-1)
        
        # Predict reachability
        reachability_prob = self.predictor(input_tensor)
        
        return reachability_prob.squeeze(-1)
    
    def update(self, obs, target_pos, is_reachable):
        """
        Update predictor based on ground truth reachability.
        """
        # Get prediction
        pred_reachability = self.predict_reachability(obs, target_pos)
        
        # Convert ground truth to tensor
        target_tensor = torch.tensor(is_reachable, dtype=torch.float32)
        
        # Compute loss
        loss = self.criterion(pred_reachability, target_tensor)
        
        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

#### 2. Frontier Detector
**Target File**: Add to existing `/npp_rl/intrinsic/icm.py`

```python
class FrontierDetector:
    """
    Detection and tracking of exploration frontiers for reachability-aware ICM.
    
    This component identifies newly reachable areas that become accessible due to
    environment changes (e.g., switch activation in NPP levels). Frontier detection
    is inspired by frontier-based exploration from robotics and adapted for RL.
    
    The detector maintains a temporal history of reachable positions and identifies
    areas that transition from unreachable to reachable, boosting curiosity for
    these newly accessible regions.
    
    References:
    - Frontier exploration: Yamauchi (1997) "A frontier-based approach for autonomous exploration"
    - Temporal tracking: Stanton & Clune (2018) "Deep curiosity search"
    - NPP integration: Custom design for switch-based level progression
    
    Note:
        Integrated into ICM module for direct access to exploration state
        and efficient frontier detection during curiosity computation.
    """
    
    def __init__(self, reachability_dim=64, memory_size=1000):
        self.reachability_dim = reachability_dim
        self.memory_size = memory_size
        
        # Frontier tracking
        self.current_frontiers = set()
        self.frontier_history = deque(maxlen=memory_size)
        self.frontier_decay_time = 500  # frames
        
        # Frontier analysis
        self.frontier_analyzer = FrontierAnalyzer()
    
    def update_frontiers(self, newly_reachable_positions):
        """
        Update frontier areas based on newly reachable positions.
        """
        # Add new frontiers
        for pos in newly_reachable_positions:
            frontier_info = {
                'position': pos,
                'discovered_time': time.time(),
                'exploration_count': 0
            }
            self.current_frontiers.add(pos)
            self.frontier_history.append(frontier_info)
        
        # Decay old frontiers
        self._decay_old_frontiers()
    
    def is_in_frontier(self, position):
        """
        Check if position is in current frontier areas.
        """
        return position in self.current_frontiers
    
    def get_frontier_value(self, position):
        """
        Get exploration value for frontier position.
        """
        if position not in self.current_frontiers:
            return 0.0
        
        # Find frontier info
        for frontier_info in reversed(self.frontier_history):
            if frontier_info['position'] == position:
                # Value decreases with exploration count and age
                age_factor = self._compute_age_factor(frontier_info['discovered_time'])
                exploration_factor = 1.0 / (1.0 + frontier_info['exploration_count'])
                
                return age_factor * exploration_factor
        
        return 0.5  # Default value
    
    def _decay_old_frontiers(self):
        """
        Remove old frontiers that are no longer novel.
        """
        current_time = time.time()
        expired_frontiers = set()
        
        for pos in self.current_frontiers:
            # Find frontier info
            for frontier_info in reversed(self.frontier_history):
                if frontier_info['position'] == pos:
                    age = current_time - frontier_info['discovered_time']
                    if age > self.frontier_decay_time or frontier_info['exploration_count'] > 10:
                        expired_frontiers.add(pos)
                    break
        
        # Remove expired frontiers
        self.current_frontiers -= expired_frontiers
```

#### 3. Strategic Weighter
**Target File**: Add to existing `/npp_rl/intrinsic/icm.py`

```python
class StrategicWeighter:
    """
    Strategic weighting for objective-oriented exploration within ICM framework.
    
    This component weights curiosity based on strategic value for level completion,
    prioritizing exploration of areas that are likely to contribute to progress.
    The weighter considers proximity to objectives, critical path analysis, and
    unlock potential for new areas.
    
    Strategic weighting is inspired by goal-directed exploration and adapted for
    NPP levels where completion requires specific sequences of actions and area
    accessibility patterns.
    
    References:
    - Goal-directed exploration: Andrychowicz et al. (2017) "Hindsight Experience Replay"
    - Strategic planning: Sutton & Barto (2018) "Reinforcement Learning: An Introduction"
    - NPP-specific objectives: Custom analysis of level completion patterns
    
    Note:
        Integrated into ICM module to directly influence curiosity computation
        based on level-specific strategic considerations.
    """
    
    def __init__(self, reachability_dim=64, objective_dim=32):
        self.reachability_dim = reachability_dim
        self.objective_dim = objective_dim
        
        # Strategic analysis
        self.objective_analyzer = ObjectiveAnalyzer()
        self.path_analyzer = PathAnalyzer()
    
    def compute_strategic_weight(self, position, reachability_features, level_objectives):
        """
        Compute strategic weight for exploring a position.
        
        Factors:
        1. Distance to key objectives (switches, doors, exit)
        2. Position on critical paths
        3. Potential to unlock new areas
        4. Historical exploration success
        """
        weights = []
        
        # Objective proximity weight
        objective_weight = self._compute_objective_proximity_weight(
            position, level_objectives
        )
        weights.append(objective_weight)
        
        # Critical path weight
        path_weight = self._compute_critical_path_weight(
            position, reachability_features
        )
        weights.append(path_weight)
        
        # Unlock potential weight
        unlock_weight = self._compute_unlock_potential_weight(
            position, reachability_features
        )
        weights.append(unlock_weight)
        
        # Combine weights (geometric mean for balanced influence)
        combined_weight = np.prod(weights) ** (1.0 / len(weights))
        
        return combined_weight
    
    def _compute_objective_proximity_weight(self, position, objectives):
        """
        Weight based on proximity to level objectives.
        """
        if not objectives:
            return 1.0
        
        # Find closest objective
        min_distance = float('inf')
        for objective in objectives:
            distance = self._compute_distance(position, objective['position'])
            min_distance = min(min_distance, distance)
        
        # Convert distance to weight (closer = higher weight)
        max_distance = 500.0  # Maximum meaningful distance
        proximity_weight = 1.0 + (max_distance - min_distance) / max_distance
        
        return max(proximity_weight, 0.1)  # Minimum weight
    
    def _compute_critical_path_weight(self, position, reachability_features):
        """
        Weight based on position being on critical paths.
        """
        # Extract path-related features from reachability features
        # This is a simplified heuristic - could be enhanced with graph analysis
        
        # Use area connectivity features (indices 40-47)
        area_connectivity = reachability_features[40:48]
        
        # Positions in well-connected areas get higher weight
        connectivity_score = np.mean(area_connectivity)
        
        # Convert to weight
        path_weight = 0.5 + connectivity_score
        
        return path_weight
    
    def _compute_unlock_potential_weight(self, position, reachability_features):
        """
        Weight based on potential to unlock new areas.
        """
        # Extract switch-related features (indices 8-23)
        switch_features = reachability_features[8:24]
        
        # Positions near inactive but reachable switches get higher weight
        inactive_reachable_switches = np.sum((switch_features > 0.1) & (switch_features < 0.9))
        
        # Convert to weight
        unlock_weight = 1.0 + 0.5 * inactive_reachable_switches
        
        return unlock_weight
```

## Implementation Plan

### Core ICM Enhancement
**Objective**: Integrate reachability awareness into existing ICM architecture

**Approach**:
- **Modify existing ICM module in place** rather than creating separate reachability-aware version
- Follow top-level import patterns and comprehensive documentation standards

**Key Modifications**:
1. **ICM Module**: Extend existing `ICMModule` in `/npp_rl/intrinsic/icm.py`
2. **Supporting Components**: Add `ReachabilityPredictor`, `FrontierDetector`, `StrategicWeighter` to same file for architectural cohesion
3. **Base Classes**: Update existing base classes in `/npp_rl/intrinsic/base.py` if needed

**Documentation Requirements**:
- Add comprehensive module-level docstrings with theoretical foundations
- Include inline documentation explaining reachability integration approach and performance considerations
- Reference all relevant research papers (Pathak et al. 2017, Ecoffet et al. 2019, etc.) in docstrings and comments
- Document nclone integration points and fallback mechanisms

### Novelty Detection Enhancement
**Objective**: Extend existing novelty detection with reachability filtering

**Approach**:
- **Modify existing novelty detection modules** to incorporate reachability constraints
- Extend count-based exploration to avoid unreachable state counting
- Implement reachability-filtered state discretization

### Training Integration
**Objective**: Integrate reachability-aware curiosity into existing training pipeline

**Approach**:
- **Modify existing training scripts** to support reachability-aware ICM
- Update existing configuration systems to include reachability awareness parameters
- Add reachability-specific monitoring to existing performance tracking systems

**Training Manager Modification Example** (modify existing training manager):
```python
# Modify existing training manager in npp_rl/agents/training_manager.py

class PPOTrainingManager:
    """
    PPO Training Manager with integrated reachability-aware curiosity support.
    
    This training manager extends the existing PPO training pipeline with support
    for reachability-aware intrinsic motivation. The integration maintains backward
    compatibility while enabling enhanced exploration efficiency.
    
    Integration Components:
    - Modified ICM with reachability awareness (optional via configuration)
    - Enhanced curiosity metrics tracking for reachability analysis
    - Performance monitoring for exploration efficiency assessment
    - Graceful fallback when reachability components are unavailable
    
    References:
    - PPO foundation: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    - Curiosity integration: Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"  
    - Training pipeline: Custom NPP-RL training architecture
    """
    
    def __init__(self, env, model_config, curiosity_config):
        self.env = env
        self.model_config = model_config
        self.curiosity_config = curiosity_config
        
        # Initialize enhanced curiosity
        self.curiosity_module = self._create_curiosity_module()
        
        # Training metrics
        self.exploration_metrics = ExplorationMetrics()
        self.reachability_metrics = ReachabilityMetrics()
    
    def _create_curiosity_module(self):
        """Create reachability-aware curiosity module."""
        # Base curiosity (existing ICM + novelty)
        base_curiosity = self._create_base_curiosity()
        
        # Reachability extractor
        reachability_extractor = self._create_reachability_extractor()
        
        # Enhanced curiosity using modified ICM with reachability awareness
        return ICMModule(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.curiosity_config
        )
    
    def train_step(self, obs, action, next_obs, reward, done, info):
        """
        Enhanced training step with reachability-aware curiosity.
        """
        # Compute intrinsic reward
        intrinsic_reward = self.curiosity_module.compute_intrinsic_reward(
            obs, action, next_obs, info
        )
        
        # Combine with extrinsic reward
        total_reward = reward + intrinsic_reward
        
        # Update metrics
        self.exploration_metrics.update(obs, action, intrinsic_reward)
        self.reachability_metrics.update(info.get('reachability', {}))
        
        return total_reward, intrinsic_reward
    
    def get_training_metrics(self):
        """Get comprehensive training metrics."""
        return {
            'exploration': self.exploration_metrics.get_summary(),
            'reachability': self.reachability_metrics.get_summary(),
            'curiosity': self.curiosity_module.get_metrics()
        }

### Evaluation and Optimization
**Objective**: Assess reachability integration impact and optimize performance

**Evaluation Approach**:
- Compare reachability-aware ICM with baseline ICM performance  
- Analyze attention patterns to understand reachability feature utilization
- Profile computational overhead and optimize for real-time performance requirements

**Optimization Focus**:
- Memory efficiency for reachability feature caching and frontier tracking
- Computational optimization for <1ms curiosity computation target
- Ablation studies on individual reachability components (scaling, frontier, strategic)

**Key Modifications**:
1. **Evaluation Scripts**: Update existing evaluation scripts to support reachability-aware metrics
2. **Performance Profiling**: Extend existing profiling tools with reachability-specific measurements
3. **Ablation Framework**: Add reachability component ablation to existing testing infrastructure

## Testing Strategy

### Unit Tests
**Test existing modified classes with reachability integration**

```python
class TestICMModuleWithReachability(unittest.TestCase):
    """Test reachability integration in the modified ICMModule."""
    
    def setUp(self):
        self.observation_space = create_mock_obs_space()
        self.action_space = create_mock_action_space()
        
        # Test the modified existing ICM with reachability features enabled
        self.icm_with_reachability = ICMModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        
        # Test backward compatibility with reachability disabled
        self.icm_baseline = ICMModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
    
    def test_reachability_scaling(self):
        """Test that curiosity is scaled based on reachability."""
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        
        # Mock reachable target
        info_reachable = {'reachability': {'target_reachable': True}}
        reward_reachable = self.icm_with_reachability.compute_intrinsic_reward(
            obs, action, next_obs, info_reachable
        )
        
        # Mock unreachable target
        info_unreachable = {'reachability': {'target_reachable': False}}
        reward_unreachable = self.icm_with_reachability.compute_intrinsic_reward(
            obs, action, next_obs, info_unreachable
        )
        
        # Reachable should have higher curiosity
        self.assertGreater(reward_reachable, reward_unreachable)
    
    def test_frontier_boost(self):
        """Test that newly reachable areas get curiosity boost."""
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        
        # First call - establish baseline
        info1 = {'reachability': {'reachable_positions': {(10, 10), (10, 11)}}}
        reward1 = self.icm_with_reachability.compute_intrinsic_reward(obs, action, next_obs, info1)
        
        # Second call - new area becomes reachable
        info2 = {'reachability': {'reachable_positions': {(10, 10), (10, 11), (12, 12)}}}
        reward2 = self.icm_with_reachability.compute_intrinsic_reward(obs, action, next_obs, info2)
        
        # Should detect frontier and boost curiosity
        self.assertGreater(reward2, reward1)
    
    def test_strategic_weighting(self):
        """Test that exploration near objectives gets higher weight."""
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        
        # Near objective
        info_near = {
            'reachability': {
                'objective_distances': [10.0, 50.0, 100.0],  # Close to first objective
                'target_position': (10, 10)
            }
        }
        reward_near = self.icm_with_reachability.compute_intrinsic_reward(obs, action, next_obs, info_near)
        
        # Far from objectives
        info_far = {
            'reachability': {
                'objective_distances': [200.0, 250.0, 300.0],  # Far from all objectives
                'target_position': (50, 50)
            }
        }
        reward_far = self.icm_with_reachability.compute_intrinsic_reward(obs, action, next_obs, info_far)
        
        # Near objective should have higher curiosity
        self.assertGreater(reward_near, reward_far)
```

### Integration Tests
```python
class TestICMTrainingIntegration(unittest.TestCase):
    """Test training integration with reachability-aware ICM."""
    
    def setUp(self):
        # Test with modified existing environment and training manager
        self.env = NPPEnv(
            render_mode='rgb_array',
        )
        self.training_manager = PPOTrainingManager(
            env=self.env,
            model_config={'features_dim': 512},
            curiosity_config={
                'reachability_scale_factor': 2.0
            }
        )
    
    def test_training_step_integration(self):
        """Test that training step works with enhanced curiosity."""
        obs = self.env.reset()
        
        for _ in range(100):
            action = self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action)
            
            # Enhanced training step
            total_reward, intrinsic_reward = self.training_manager.train_step(
                obs, action, next_obs, reward, done, info
            )
            
            # Verify rewards are reasonable
            self.assertIsInstance(total_reward, (int, float))
            self.assertIsInstance(intrinsic_reward, (int, float))
            self.assertGreaterEqual(intrinsic_reward, 0.0)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
    
    def test_exploration_efficiency(self):
        """Test that reachability-aware curiosity improves exploration efficiency."""
        # This would be a longer test comparing exploration patterns
        # with and without reachability awareness
        pass
```

### Performance Tests
```python
class TestCuriosityPerformance(unittest.TestCase):
    def test_curiosity_computation_speed(self):
        """Test that curiosity computation meets performance requirements."""
        curiosity = create_test_curiosity_module()
        
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        info = create_mock_reachability_info()
        
        # Benchmark curiosity computation
        times = []
        for _ in range(1000):
            start_time = time.perf_counter()
            reward = curiosity.compute_intrinsic_reward(obs, action, next_obs, info)
            elapsed = (time.perf_counter() - start_time) * 1000
            times.append(elapsed)
        
        # Check performance targets
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        self.assertLess(avg_time, 1.0, f"Average curiosity time too high: {avg_time}ms")
        self.assertLess(p95_time, 2.0, f"95th percentile time too high: {p95_time}ms")
    
    def test_memory_usage(self):
        """Test that curiosity module doesn't use excessive memory."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create curiosity module
        curiosity = create_test_curiosity_module()
        
        # Run for extended period
        for _ in range(10000):
            obs = create_mock_observation()
            action = 0
            next_obs = create_mock_observation()
            info = create_mock_reachability_info()
            
            reward = curiosity.compute_intrinsic_reward(obs, action, next_obs, info)
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        self.assertLess(memory_increase, 100, f"Memory usage too high: {memory_increase}MB")
```

## Success Criteria

### Performance Requirements
- **Curiosity Computation**: <1ms average per step
- **Memory Usage**: <50MB additional memory for curiosity components
- **Training Speed**: <5% slowdown compared to baseline curiosity

### Quality Requirements
- **Exploration Efficiency**: 20-50% improvement in sample efficiency on complex levels
- **Reachability Awareness**: Demonstrable reduction in exploration of unreachable areas
- **Strategic Focus**: Increased exploration near level objectives

### Training Requirements
- **Convergence**: Faster convergence on complex levels
- **Generalization**: Better performance on unseen levels
- **Stability**: Stable training without curiosity-induced instabilities

## Risk Mitigation

### Technical Risks
1. **Complexity Overhead**: Careful performance monitoring and optimization
2. **Hyperparameter Sensitivity**: Extensive hyperparameter tuning and validation
3. **Integration Issues**: Comprehensive testing with existing systems

### Training Risks
1. **Exploration Bias**: Balance reachability awareness with exploration diversity
2. **Overfitting**: Regularization and validation on diverse levels
3. **Reward Scaling**: Careful tuning of intrinsic vs extrinsic reward balance

## Deliverables

1. **Modified ICMModule**: Existing ICM architecture extended with reachability awareness functionality
2. **Enhanced Novelty Detection**: Existing novelty detection updated with reachability filtering  
3. **Training Pipeline Updates**: Modified training managers supporting reachability-aware curiosity
4. **Performance Analysis**: Comprehensive evaluation of exploration efficiency improvements on existing systems
5. **Documentation**: Updated module docstrings and inline documentation with theoretical foundations
6. **Integration Testing**: Test suites validating reachability integration in existing components

## Dependencies

### Internal Dependencies
- **Reachability Features**: TASK_001 (Compact Reachability Features Integration)
- **Base Curiosity**: Existing ICM and novelty detection modules
- **Training Pipeline**: Enhanced training system with reachability support

### External Dependencies
- **PyTorch**: Neural network implementation
- **NumPy**: Numerical operations
- **Collections**: Data structures for history tracking

## References

1. **Strategic Analysis**: `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`
2. **Current ICM**: `/workspace/npp-rl/npp_rl/intrinsic/` (existing curiosity modules)
3. **Pathak et al. (2017)**: "Curiosity-driven Exploration by Self-supervised Prediction"
4. **Ecoffet et al. (2019)**: "Go-Explore: a New Approach for Hard-Exploration Problems"
5. **Burda et al. (2018)**: "Exploration by Random Network Distillation"