"""
Adaptive Exploration with Hierarchical Reachability-Guided Planning (Refactored)

This module extends the existing adaptive exploration strategies with hierarchical
reinforcement learning capabilities based on reachability analysis from nclone.
The integration provides strategic subgoal generation and level completion planning.

Integration Strategy:
- Hierarchical subgoal extraction converts reachability analysis to actionable objectives
- Strategic planning generates level completion sequences based on switch dependencies  
- Dynamic subgoal updates adapt to changing game state for optimal exploration
- Performance-optimized caching for real-time subgoal management (<3ms target)

Theoretical Foundation:
- Options framework: Sutton et al. (1999) "Between MDPs and semi-MDPs"
- Option-Critic: Bacon et al. (2017) "The Option-Critic Architecture"  
- Data-efficient HRL: Nachum et al. (2018) "Data-Efficient Hierarchical Reinforcement Learning"
- Reachability-guided HRL: Levy et al. (2019) "Hierarchical Actor-Critic with reachability constraints"
- Strategic planning: Custom integration with nclone physics and level completion heuristics

nclone Integration:
- Uses compact reachability features from nclone.graph.reachability for subgoal filtering
- Integrates with TieredReachabilitySystem for performance-optimized planning queries
- Maintains compatibility with existing NPP physics constants and level objective analysis
"""

# Standard library imports
import math
import time
from collections import defaultdict, deque
from typing import Dict, Tuple, List

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# nclone imports (reachability and planning components)
from nclone.graph.reachability.compact_features import CompactReachabilityFeatures
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
from nclone.planning import (
    Subgoal, NavigationSubgoal, SwitchActivationSubgoal, CompletionStrategy, LevelCompletionPlanner, SubgoalPrioritizer
)


class CuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for exploration bonus calculation.
    
    Based on Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"
    with adaptations for NPP environment features.
    """
    
    def __init__(self, feature_dim: int, action_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        # Forward model: predicts next state features from current features + action
        self.forward_model = nn.Sequential(
            nn.Linear(64 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        
        # Inverse model: predicts action from current and next state features
        self.inverse_model = nn.Sequential(
            nn.Linear(64 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state, action, next_state):
        """
        Forward pass for ICM training.
        
        Returns:
            forward_loss: Prediction error for forward model
            inverse_loss: Prediction error for inverse model
            intrinsic_reward: Curiosity-based exploration bonus
        """
        # Encode states
        phi_state = self.feature_encoder(state)
        phi_next_state = self.feature_encoder(next_state)
        
        # Forward model prediction
        action_onehot = torch.zeros(action.size(0), 5, device=action.device)
        action_onehot.scatter_(1, action.long().unsqueeze(1), 1)
        
        forward_input = torch.cat([phi_state, action_onehot], dim=1)
        predicted_next_phi = self.forward_model(forward_input)
        
        # Inverse model prediction
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        predicted_action = self.inverse_model(inverse_input)
        
        # Calculate losses
        forward_loss = nn.MSELoss()(predicted_next_phi, phi_next_state.detach())
        inverse_loss = nn.CrossEntropyLoss()(predicted_action, action.long())
        
        # Intrinsic reward is forward prediction error
        intrinsic_reward = torch.norm(predicted_next_phi - phi_next_state.detach(), dim=1)
        
        return forward_loss, inverse_loss, intrinsic_reward


class NoveltyDetector:
    """
    Novelty detection for exploration bonus using state visitation frequency.
    
    Maintains a hash-based count of state visits for novelty estimation.
    """
    
    def __init__(self, hash_dim: int = 1000):
        self.hash_dim = hash_dim
        self.visit_counts = defaultdict(int)
        self.total_visits = 0
        
    def get_novelty_bonus(self, state_features: torch.Tensor) -> float:
        """Calculate novelty bonus based on state visitation frequency."""
        # Simple hash-based novelty detection
        # Handle both single samples and batches
        if state_features.dim() > 1:
            state_features = state_features[0]  # Take first sample from batch
        # Handle GPU tensors by moving to CPU before numpy conversion
        state_hash = hash(tuple(state_features.detach().cpu().numpy().round(2).tolist())) % self.hash_dim
        
        self.visit_counts[state_hash] += 1
        self.total_visits += 1
        
        # Novelty bonus inversely proportional to visit frequency
        visit_frequency = self.visit_counts[state_hash] / self.total_visits
        novelty_bonus = 1.0 / (1.0 + visit_frequency * 100)
        
        return novelty_bonus


class AdaptiveExplorationManager:
    """
    Adaptive exploration manager with integrated hierarchical reachability-guided planning.
    
    This manager extends the existing curiosity-driven exploration with hierarchical
    reinforcement learning capabilities that provide strategic subgoals based on
    reachability analysis for improved sample efficiency.
    
    Architecture Extensions:
    1. Base Exploration: ICM + Novelty detection (existing functionality preserved)
    2. Subgoal Extraction: Convert reachability analysis to actionable hierarchical objectives
    3. Strategic Planning: Level completion strategy generation using switch dependency analysis
    4. Dynamic Updates: Real-time subgoal adaptation to changing game state
    5. Performance Optimization: Efficient subgoal caching and invalidation system
    
    Hierarchical Integration:
    - Direct initialization of reachability extractor (no defensive programming)
    - Subgoal prioritization based on strategic value and completion likelihood
    - Dynamic cache management for real-time performance (<3ms subgoal generation)
    
    References:
    - Base exploration: Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"
    - Hierarchical RL: Sutton et al. (1999) "Between MDPs and semi-MDPs" 
    - Strategic planning: Bacon et al. (2017) "The Option-Critic Architecture"
    - Reachability integration: Custom design for NPP level completion
    """
    
    def __init__(self, 
                 curiosity_weight: float = 0.1,
                 novelty_weight: float = 0.05,
                 progress_window: int = 100):
        
        # Initialize base exploration components (existing functionality preserved)
        self.curiosity_weight = curiosity_weight
        self.novelty_weight = novelty_weight
        self.progress_window = progress_window
        
        # Initialize components
        self.novelty_detector = NoveltyDetector()
        self.curiosity_module = None  # Will be initialized when we have feature dimensions
        
        # Progress tracking
        self.recent_rewards = deque(maxlen=progress_window)
        self.recent_completion_times = deque(maxlen=progress_window)
        self.exploration_scale = 1.0
        
        # Statistics
        self.total_intrinsic_reward = 0.0
        self.episode_count = 0
        
        # Hierarchical planning components (always enabled, no defensive programming)
        # Direct initialization - fail fast if dependencies are missing
        self.reachability_system = TieredReachabilitySystem()
        self.reachability_features = CompactReachabilityFeatures()
        
        # Subgoal management system for hierarchical planning
        # Based on Options framework from Sutton et al. (1999)
        self.subgoal_cache = {}                    # Cached subgoals for performance optimization
        self.cache_ttl = 200                       # Cache time-to-live in milliseconds  
        self.last_switch_states = {}               # State tracking for dynamic updates
        self.last_ninja_pos = None                 # Position tracking for cache invalidation
        
        # Strategic planning components based on NPP level completion heuristics
        # Implementation follows completion strategies from nclone strategic analysis
        self.completion_planner = LevelCompletionPlanner()
        self.subgoal_prioritizer = SubgoalPrioritizer()
        
        # Performance monitoring for real-time optimization
        # Target: <3ms subgoal generation, >70% cache hit rate
        self.cache_hit_rate = 0.0
        self.avg_subgoal_count = 0.0
        self.planning_time_ms = 0.0
        
    def initialize_curiosity_module(self, feature_dim: int, action_dim: int = 5):
        """Initialize curiosity module with known dimensions."""
        self.curiosity_module = CuriosityModule(feature_dim, action_dim)
        
    def get_exploration_bonus(self, state, action, next_state) -> float:
        """
        Calculate exploration bonus combining curiosity and novelty.
        
        Returns combined intrinsic reward for exploration encouragement.
        """
        total_bonus = 0.0
        
        # Curiosity bonus from ICM
        if self.curiosity_module is not None:
            with torch.no_grad():
                _, _, curiosity_reward = self.curiosity_module(state, action, next_state)
                curiosity_bonus = curiosity_reward.mean().item() * self.curiosity_weight
                total_bonus += curiosity_bonus
        
        # Novelty bonus from state visitation
        novelty_bonus = self.novelty_detector.get_novelty_bonus(state) * self.novelty_weight
        total_bonus += novelty_bonus
        
        # Scale by adaptive exploration factor
        total_bonus *= self.exploration_scale
        
        self.total_intrinsic_reward += total_bonus
        return total_bonus
    
    def update_progress(self, episode_reward: float, completion_time: float):
        """Update progress tracking and adjust exploration scale."""
        self.recent_rewards.append(episode_reward)
        self.recent_completion_times.append(completion_time)
        self.episode_count += 1
        
        # Adaptive exploration scaling based on recent performance
        if len(self.recent_rewards) >= 10:
            recent_avg = np.mean(list(self.recent_rewards)[-10:])
            if recent_avg > np.mean(list(self.recent_rewards)):
                # Performance improving, reduce exploration
                self.exploration_scale = max(0.1, self.exploration_scale * 0.95)
            else:
                # Performance stagnating, increase exploration
                self.exploration_scale = min(2.0, self.exploration_scale * 1.05)
    
    def get_available_subgoals(self, ninja_pos: Tuple[float, float], level_data, 
                                 switch_states: Dict, max_subgoals: int = 5) -> List[Subgoal]:
        """
        Generate hierarchical subgoals using reachability-guided planning.
        
        This method implements the core hierarchical RL integration that converts
        reachability analysis into actionable subgoals following the Options framework.
        
        Performance target: <3ms execution time for real-time RL integration.
        """
        start_time = time.time()
        
        # Check cache first for performance optimization
        cache_key = self._generate_cache_key(ninja_pos, switch_states)
        if cache_key in self.subgoal_cache:
            cached_result = self.subgoal_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.cache_ttl / 1000.0:
                self.cache_hit_rate = 0.9 * self.cache_hit_rate + 0.1 * 1.0  # Update hit rate
                return cached_result['subgoals'][:max_subgoals]
        
        # Generate new subgoals using reachability analysis
        subgoals = []
        
        # Extract reachability features for strategic planning
        reachability_result = self.reachability_system.analyze_reachability(
            level_data, ninja_pos, switch_states, performance_target="balanced"
        )
        
        # Encode reachability into compact features for neural processing
        reachability_features_array = self.reachability_features.encode_reachability(
            reachability_result, level_data, [], ninja_pos, switch_states
        )
        reachability_features = torch.tensor(reachability_features_array, dtype=torch.float32)
        
        # Generate completion strategy using NPP level completion algorithm
        completion_strategy = self.completion_planner.plan_completion(
            ninja_pos, level_data, switch_states, 
            self.reachability_system, self.reachability_features
        )
        
        # Convert completion steps to hierarchical subgoals
        for step in completion_strategy.steps[:max_subgoals]:
            if step.action_type == 'navigate_and_activate':
                subgoal = SwitchActivationSubgoal(
                    switch_id=step.target_id,
                    switch_position=step.target_position,
                    switch_type='exit_switch',
                    reachability_score=0.8,
                    priority=step.priority,
                    estimated_time=5.0,
                    success_probability=0.9
                )
                subgoals.append(subgoal)
                
            elif step.action_type == 'navigate_to_exit':
                subgoal = NavigationSubgoal(
                    target_position=step.target_position,
                    target_type='exit_door',
                    distance=math.sqrt((ninja_pos[0] - step.target_position[0])**2 + 
                                     (ninja_pos[1] - step.target_position[1])**2),
                    priority=step.priority,
                    estimated_time=3.0,
                    success_probability=0.95
                )
                subgoals.append(subgoal)
        
        # Prioritize subgoals using strategic analysis
        prioritized_subgoals = self.subgoal_prioritizer.prioritize(
            subgoals, ninja_pos, level_data, reachability_features
        )
        
        # Cache results for performance optimization
        self.subgoal_cache[cache_key] = {
            'subgoals': prioritized_subgoals,
            'timestamp': time.time()
        }
        
        # Update performance metrics
        self.planning_time_ms = (time.time() - start_time) * 1000
        self.avg_subgoal_count = 0.9 * self.avg_subgoal_count + 0.1 * len(prioritized_subgoals)
        self.cache_hit_rate = 0.9 * self.cache_hit_rate + 0.1 * 0.0  # Cache miss
        
        # Invalidate old cache entries to prevent memory growth
        self._cleanup_cache()
        
        return prioritized_subgoals[:max_subgoals]
    
    def get_completion_strategy(self, ninja_pos: Tuple[float, float], level_data, 
                               switch_states: Dict) -> CompletionStrategy:
        """
        Generate strategic completion plan for the current level.
        
        Uses the production-ready NPP level completion algorithm with neural
        reachability features for fast strategic planning.
        """
        return self.completion_planner.plan_completion(
            ninja_pos, level_data, switch_states,
            self.reachability_system, self.reachability_features
        )
    
    def _generate_cache_key(self, ninja_pos: Tuple[float, float], switch_states: Dict, level_data=None) -> str:
        """Generate cache key for subgoal caching."""
        # Round position to reduce cache fragmentation
        pos_key = f"{int(ninja_pos[0]/24)},{int(ninja_pos[1]/24)}"
        
        # Sort switch states for consistent key generation
        switch_key = ",".join(f"{k}:{v}" for k, v in sorted(switch_states.items()))
        
        # Include level data hash if provided for more specific caching
        level_key = ""
        if level_data is not None:
            level_key = f"|{hash(str(level_data))}"
        
        return f"{pos_key}|{switch_key}{level_key}"
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.subgoal_cache.items()
            if current_time - value['timestamp'] > self.cache_ttl / 1000.0
        ]
        
        for key in expired_keys:
            del self.subgoal_cache[key]
    
    def invalidate_cache_on_switch_change(self, switch_states: Dict):
        """Invalidate cache when switch states change."""
        if switch_states != self.last_switch_states:
            self.subgoal_cache.clear()
            self.last_switch_states = switch_states.copy()
    
    def update_subgoals_on_switch_change(self, ninja_pos: Tuple[float, float], level_data, 
                                       old_switch_states: Dict, new_switch_states: Dict) -> Tuple[List[Subgoal], List[Subgoal]]:
        """
        Update subgoals when switch states change.
        
        Args:
            ninja_pos: Current ninja position
            level_data: Level data for reachability analysis
            old_switch_states: Previous switch states
            new_switch_states: New switch states
            
        Returns:
            Tuple of (new_subgoals, newly_available_subgoals)
        """
        # Invalidate cache due to switch state change
        self.invalidate_cache_on_switch_change(new_switch_states)
        
        # Get old subgoals for comparison
        old_subgoals = self.get_available_subgoals(ninja_pos, level_data, old_switch_states)
        
        # Get new subgoals with updated switch states
        new_subgoals = self.get_available_subgoals(ninja_pos, level_data, new_switch_states)
        
        # Find newly available subgoals (simple comparison by position for now)
        old_positions = {(s.position[0], s.position[1]) for s in old_subgoals}
        newly_available = [s for s in new_subgoals if (s.position[0], s.position[1]) not in old_positions]
        
        return new_subgoals, newly_available
    
    def get_hierarchical_stats(self) -> Dict[str, Any]:
        """Get hierarchical planning statistics."""
        return {
            'cache_size': len(self.subgoal_cache),
            'cache_hit_rate': self.cache_hit_rate,
            'avg_subgoal_count': self.avg_subgoal_count,
            'planning_time_ms': self.planning_time_ms,
            'total_subgoals_generated': getattr(self, 'total_subgoals_generated', 0)
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get performance and exploration statistics."""
        return {
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'episode_count': self.episode_count,
            'exploration_scale': self.exploration_scale,
            'cache_hit_rate': self.cache_hit_rate,
            'avg_subgoal_count': self.avg_subgoal_count,
            'planning_time_ms': self.planning_time_ms,
            'avg_recent_reward': np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0,
            'avg_completion_time': np.mean(list(self.recent_completion_times)) if self.recent_completion_times else 0.0
        }