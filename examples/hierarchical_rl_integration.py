"""
Hierarchical RL Integration Example

This example demonstrates how to integrate the hierarchical reachability manager
with PPO training for improved level completion performance in N++ levels.

The integration shows:
1. Hierarchical subgoal generation during training
2. Reward shaping based on subgoal completion
3. Strategic level completion planning
4. Performance monitoring and optimization

Usage:
    python examples/hierarchical_rl_integration.py

Requirements:
    - npp-rl package with hierarchical reachability manager
    - nclone package with compact reachability features
    - PyTorch for RL training
    - Stable-baselines3 for PPO implementation

References:
    - Options framework: Sutton et al. (1999) "Between MDPs and semi-MDPs"
    - Hierarchical RL: Bacon et al. (2017) "The Option-Critic Architecture"
    - PPO: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import hierarchical components
from npp_rl.agents.adaptive_exploration import AdaptiveExplorationManager

# Import planning components from nclone
from nclone.planning import (
    Subgoal, NavigationSubgoal, SwitchActivationSubgoal, CollectionSubgoal,
    CompletionStrategy, CompletionStep
)

# Mock imports for demonstration (replace with actual imports in production)
class MockPPOAgent:
    """Mock PPO agent for demonstration purposes."""
    
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = None
        
    def predict(self, observation, deterministic=False):
        # Mock action prediction
        return np.random.randint(0, self.action_space), None
    
    def learn(self, total_timesteps):
        print(f"Training PPO agent for {total_timesteps} timesteps...")
        return self


class MockEnvironment:
    """Mock N++ environment for demonstration."""
    
    def __init__(self, level_id="demo_level"):
        self.level_id = level_id
        self.ninja_pos = (150, 150)
        self.switch_states = {'switch_1': False, 'switch_2': False}
        self.level_data = self._create_mock_level_data()
        self.step_count = 0
        self.max_steps = 1000
        
    def _create_mock_level_data(self):
        """Create mock level data for demonstration."""
        class MockLevelData:
            def __init__(self):
                self.level_id = "demo_level"
                self.objectives = [
                    {'type': 'exit_door', 'id': 'exit_door_1', 'x': 500, 'y': 300}
                ]
                self.switches = [
                    {'type': 'door_switch', 'id': 'switch_1', 'x': 200, 'y': 200, 'controls_exit': True},
                    {'type': 'door_switch', 'id': 'switch_2', 'x': 300, 'y': 400, 'controls_exit': False}
                ]
                self.collectibles = [
                    {'type': 'gold', 'id': 'gold_1', 'x': 250, 'y': 250, 'value': 10, 'collected': False}
                ]
        
        return MockLevelData()
    
    def reset(self):
        """Reset environment to initial state."""
        self.ninja_pos = (150, 150)
        self.switch_states = {'switch_1': False, 'switch_2': False}
        self.step_count = 0
        
        # Return mock observation
        return np.random.random((64,))  # 64-dimensional observation
    
    def step(self, action):
        """Execute one environment step."""
        self.step_count += 1
        
        # Mock environment dynamics
        self.ninja_pos = (
            self.ninja_pos[0] + np.random.randint(-5, 6),
            self.ninja_pos[1] + np.random.randint(-5, 6)
        )
        
        # Mock switch activation
        if np.random.random() < 0.01:  # 1% chance per step
            switch_id = np.random.choice(list(self.switch_states.keys()))
            self.switch_states[switch_id] = True
        
        # Mock reward calculation
        base_reward = -0.01  # Time penalty
        completion_reward = 100.0 if self._is_level_complete() else 0.0
        
        done = self._is_level_complete() or self.step_count >= self.max_steps
        info = {
            'ninja_pos': self.ninja_pos,
            'switch_states': self.switch_states.copy(),
            'level_data': self.level_data
        }
        
        return np.random.random((64,)), base_reward + completion_reward, done, info
    
    def _is_level_complete(self):
        """Check if level is complete."""
        # Simple completion check: ninja near exit and exit switch activated
        exit_pos = (500, 300)
        distance_to_exit = np.sqrt((self.ninja_pos[0] - exit_pos[0])**2 + 
                                 (self.ninja_pos[1] - exit_pos[1])**2)
        
        return distance_to_exit < 50 and self.switch_states.get('switch_1', False)


@dataclass
class HierarchicalTrainingConfig:
    """Configuration for hierarchical RL training."""
    
    # Training parameters
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    
    # Hierarchical parameters
    subgoal_update_frequency: int = 100  # Update subgoals every N steps
    reward_shaping_weight: float = 0.1   # Weight for subgoal reward shaping
    max_subgoals: int = 8                # Maximum subgoals to consider
    
    # Performance monitoring
    log_frequency: int = 1000            # Log stats every N steps
    performance_target_ms: float = 3.0   # Target subgoal generation time


class HierarchicalRLTrainer:
    """
    Hierarchical RL trainer integrating subgoal-based planning with PPO.
    
    This trainer demonstrates how to use the hierarchical reachability manager
    to improve RL training performance through strategic subgoal generation
    and reward shaping.
    """
    
    def __init__(self, config: HierarchicalTrainingConfig):
        self.config = config
        
        # Initialize environment and agent
        self.env = MockEnvironment()
        self.agent = MockPPOAgent(
            observation_space=64,  # Mock observation space
            action_space=8         # Mock action space (8 directions)
        )
        
        # Initialize hierarchical reachability manager
        # Note: In production, this would use real nclone dependencies
        # For this demo, we use a mock manager to avoid complex level data requirements
        print("⚠ Using mock hierarchical manager for demonstration")
        print("  (In production, use real AdaptiveExplorationManager with proper level data)")
        self.hierarchical_manager = self._create_mock_hierarchical_manager()
        
        # Training state
        self.current_subgoals: List[Subgoal] = []
        self.subgoal_completion_history: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'subgoal_generation_time': [],
            'reward_shaping_bonus': [],
            'level_completion_rate': [],
            'cache_hit_rate': []
        }
    
    def _create_mock_hierarchical_manager(self):
        """Create mock hierarchical manager for demonstration."""
        class MockHierarchicalManager:
            def get_available_subgoals(self, ninja_pos, level_data, switch_states, max_subgoals=8):
                # Return mock subgoals
                return [
                    NavigationSubgoal(0.9, 20.0, 0.8, (500, 300), 'exit_door', 400.0),
                    SwitchActivationSubgoal(0.8, 30.0, 0.9, 'switch_1', (200, 200), 'door_switch', 0.85)
                ]
            
            def get_completion_strategy(self, ninja_pos, level_data, switch_states):
                from npp_rl.agents.adaptive_exploration import CompletionStrategy, CompletionStep
                return CompletionStrategy(
                    steps=[
                        CompletionStep('navigate_and_activate', (200, 200), 'switch_1', 'Activate exit switch', 1.0),
                        CompletionStep('navigate_to_exit', (500, 300), 'exit_door_1', 'Go to exit door', 1.0)
                    ],
                    confidence=0.85,
                    description="Mock completion strategy"
                )
            
            def get_hierarchical_stats(self):
                return {
                    'planning_time_ms': np.random.uniform(0.5, 2.0),
                    'cache_hit_rate': np.random.uniform(0.3, 0.8),
                    'avg_subgoal_count': np.random.uniform(2, 6),
                    'cache_size': np.random.randint(5, 20)
                }
        
        return MockHierarchicalManager()
    
    def train(self):
        """
        Main training loop with hierarchical RL integration.
        
        This method demonstrates the complete integration of hierarchical
        subgoal planning with PPO training for improved level completion.
        """
        print("🚀 Starting Hierarchical RL Training")
        print(f"Configuration: {self.config}")
        print("-" * 60)
        
        # Training statistics
        episode_count = 0
        total_steps = 0
        episode_rewards = []
        level_completions = 0
        
        # Main training loop
        while total_steps < self.config.total_timesteps:
            episode_count += 1
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment and get initial state
            observation = self.env.reset()
            done = False
            
            print(f"\n📍 Episode {episode_count} - Starting level completion")
            
            # Generate initial subgoals for the episode
            self._update_subgoals()
            
            # Episode loop
            while not done and total_steps < self.config.total_timesteps:
                # Get action from PPO agent
                action, _ = self.agent.predict(observation, deterministic=False)
                
                # Execute action in environment
                next_observation, base_reward, done, info = self.env.step(action)
                
                # Apply hierarchical reward shaping
                shaped_reward = self._apply_reward_shaping(
                    base_reward, info['ninja_pos'], info['level_data'], info['switch_states']
                )
                
                # Update training statistics
                episode_reward += shaped_reward
                episode_steps += 1
                total_steps += 1
                
                # Update subgoals periodically
                if total_steps % self.config.subgoal_update_frequency == 0:
                    self._update_subgoals(info)
                
                # Log progress periodically
                if total_steps % self.config.log_frequency == 0:
                    self._log_training_progress(episode_count, total_steps, episode_rewards)
                
                observation = next_observation
            
            # Episode completed
            episode_rewards.append(episode_reward)
            
            if done and self.env._is_level_complete():
                level_completions += 1
                print(f"✅ Level completed! Total completions: {level_completions}")
            else:
                print(f"⏰ Episode timeout after {episode_steps} steps")
            
            print(f"Episode reward: {episode_reward:.2f}")
            
            # Update completion rate metric
            completion_rate = level_completions / episode_count
            self.performance_metrics['level_completion_rate'].append(completion_rate)
        
        # Training completed
        print("\n" + "=" * 60)
        print("🎯 Hierarchical RL Training Completed!")
        print(f"Total episodes: {episode_count}")
        print(f"Total steps: {total_steps}")
        print(f"Level completion rate: {level_completions/episode_count:.2%}")
        print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
        
        # Display performance metrics
        self._display_performance_summary()
        
        return self.agent
    
    def _update_subgoals(self, info: Optional[Dict] = None):
        """Update current subgoals based on game state."""
        if info is None:
            ninja_pos = self.env.ninja_pos
            level_data = self.env.level_data
            switch_states = self.env.switch_states
        else:
            ninja_pos = info['ninja_pos']
            level_data = info['level_data']
            switch_states = info['switch_states']
        
        # Measure subgoal generation performance
        start_time = time.perf_counter()
        
        # Get hierarchical subgoals
        self.current_subgoals = self.hierarchical_manager.get_available_subgoals(
            ninja_pos, level_data, switch_states, max_subgoals=self.config.max_subgoals
        )
        
        # Record performance metrics
        generation_time = (time.perf_counter() - start_time) * 1000
        self.performance_metrics['subgoal_generation_time'].append(generation_time)
        
        # Log subgoal update
        if len(self.current_subgoals) > 0:
            print(f"🎯 Updated subgoals: {len(self.current_subgoals)} available")
            for i, subgoal in enumerate(self.current_subgoals[:3]):  # Show top 3
                print(f"   {i+1}. {type(subgoal).__name__} (priority: {subgoal.priority:.2f})")
    
    def _apply_reward_shaping(self, base_reward: float, ninja_pos, level_data, switch_states) -> float:
        """Apply hierarchical reward shaping based on subgoal progress."""
        if not self.current_subgoals:
            return base_reward
        
        # Calculate reward shaping bonus from subgoals
        total_shaping_bonus = 0.0
        
        for subgoal in self.current_subgoals:
            # Get reward shaping from subgoal
            shaping_reward = subgoal.get_reward_shaping(ninja_pos)
            
            # Weight by subgoal priority and configuration
            weighted_reward = shaping_reward * subgoal.priority * self.config.reward_shaping_weight
            total_shaping_bonus += weighted_reward
            
            # Check for subgoal completion
            if subgoal.is_completed(ninja_pos, level_data, switch_states):
                completion_bonus = 10.0 * subgoal.priority  # Completion bonus
                total_shaping_bonus += completion_bonus
                
                # Log subgoal completion
                print(f"🏆 Subgoal completed: {type(subgoal).__name__} (+{completion_bonus:.1f} reward)")
        
        # Record shaping bonus for metrics
        self.performance_metrics['reward_shaping_bonus'].append(total_shaping_bonus)
        
        return base_reward + total_shaping_bonus
    
    def _log_training_progress(self, episode_count: int, total_steps: int, episode_rewards: List[float]):
        """Log training progress and hierarchical metrics."""
        print(f"\n📊 Training Progress (Step {total_steps})")
        print(f"Episodes completed: {episode_count}")
        
        if episode_rewards:
            recent_rewards = episode_rewards[-10:]  # Last 10 episodes
            print(f"Recent average reward: {np.mean(recent_rewards):.2f}")
        
        # Get hierarchical statistics
        hierarchical_stats = self.hierarchical_manager.get_hierarchical_stats()
        
        print(f"Hierarchical metrics:")
        print(f"  Subgoal generation time: {hierarchical_stats.get('planning_time_ms', 0):.2f}ms")
        print(f"  Cache hit rate: {hierarchical_stats.get('cache_hit_rate', 0):.2%}")
        print(f"  Average subgoals: {hierarchical_stats.get('avg_subgoal_count', 0):.1f}")
        
        # Record cache hit rate
        self.performance_metrics['cache_hit_rate'].append(
            hierarchical_stats.get('cache_hit_rate', 0)
        )
    
    def _display_performance_summary(self):
        """Display comprehensive performance summary."""
        print("\n📈 Performance Summary")
        print("-" * 40)
        
        # Subgoal generation performance
        if self.performance_metrics['subgoal_generation_time']:
            avg_time = np.mean(self.performance_metrics['subgoal_generation_time'])
            p95_time = np.percentile(self.performance_metrics['subgoal_generation_time'], 95)
            
            print(f"Subgoal Generation Performance:")
            print(f"  Average time: {avg_time:.2f}ms")
            print(f"  95th percentile: {p95_time:.2f}ms")
            print(f"  Target met: {'✅' if avg_time < self.config.performance_target_ms else '❌'}")
        
        # Reward shaping effectiveness
        if self.performance_metrics['reward_shaping_bonus']:
            avg_bonus = np.mean(self.performance_metrics['reward_shaping_bonus'])
            print(f"\nReward Shaping:")
            print(f"  Average bonus per step: {avg_bonus:.3f}")
            print(f"  Total shaping contribution: {sum(self.performance_metrics['reward_shaping_bonus']):.1f}")
        
        # Cache performance
        if self.performance_metrics['cache_hit_rate']:
            avg_hit_rate = np.mean(self.performance_metrics['cache_hit_rate'])
            print(f"\nCaching Performance:")
            print(f"  Average hit rate: {avg_hit_rate:.2%}")
            print(f"  Target met: {'✅' if avg_hit_rate > 0.7 else '❌'}")
        
        # Level completion
        if self.performance_metrics['level_completion_rate']:
            final_completion_rate = self.performance_metrics['level_completion_rate'][-1]
            print(f"\nLevel Completion:")
            print(f"  Final completion rate: {final_completion_rate:.2%}")


def demonstrate_hierarchical_integration():
    """
    Demonstrate hierarchical RL integration with comprehensive examples.
    
    This function shows how to use the hierarchical reachability manager
    in different training scenarios and configurations.
    """
    print("🎮 Hierarchical RL Integration Demonstration")
    print("=" * 60)
    
    # Configuration for fast demonstration
    demo_config = HierarchicalTrainingConfig(
        total_timesteps=5000,      # Reduced for demo
        subgoal_update_frequency=50,
        log_frequency=500,
        reward_shaping_weight=0.2
    )
    
    # Create and run trainer
    trainer = HierarchicalRLTrainer(demo_config)
    
    print("\n🔧 Trainer Configuration:")
    print(f"  Total timesteps: {demo_config.total_timesteps}")
    print(f"  Subgoal update frequency: {demo_config.subgoal_update_frequency}")
    print(f"  Reward shaping weight: {demo_config.reward_shaping_weight}")
    print(f"  Performance target: {demo_config.performance_target_ms}ms")
    
    # Run training
    trained_agent = trainer.train()
    
    print("\n🎯 Integration demonstration completed!")
    print("Key features demonstrated:")
    print("  ✅ Hierarchical subgoal generation")
    print("  ✅ Strategic reward shaping")
    print("  ✅ Performance monitoring")
    print("  ✅ Real-time cache optimization")
    print("  ✅ Level completion planning")
    
    return trained_agent


def demonstrate_subgoal_types():
    """Demonstrate different types of hierarchical subgoals."""
    print("\n🎯 Subgoal Types Demonstration")
    print("-" * 40)
    
    # Create example subgoals
    navigation_subgoal = NavigationSubgoal(
        priority=0.9,
        estimated_time=20.0,
        success_probability=0.8,
        target_position=(500, 300),
        target_type='exit_door',
        distance=400.0
    )
    
    switch_subgoal = SwitchActivationSubgoal(
        priority=0.8,
        estimated_time=30.0,
        success_probability=0.9,
        switch_id='switch_1',
        switch_position=(200, 200),
        switch_type='door_switch',
        reachability_score=0.85
    )
    
    collection_subgoal = CollectionSubgoal(
        priority=0.3,
        estimated_time=15.0,
        success_probability=0.7,
        target_position=(250, 250),
        item_type='gold',
        value=10.0,
        area_connectivity=0.6
    )
    
    subgoals = [navigation_subgoal, switch_subgoal, collection_subgoal]
    
    print("Example hierarchical subgoals:")
    for i, subgoal in enumerate(subgoals, 1):
        print(f"\n{i}. {type(subgoal).__name__}")
        print(f"   Priority: {subgoal.priority:.2f}")
        print(f"   Target: {subgoal.get_target_position()}")
        print(f"   Estimated time: {subgoal.estimated_time:.1f}s")
        print(f"   Success probability: {subgoal.success_probability:.2f}")
        
        # Demonstrate reward shaping
        ninja_pos = (200, 200)
        reward = subgoal.get_reward_shaping(ninja_pos)
        print(f"   Reward shaping (from {ninja_pos}): {reward:.3f}")


if __name__ == "__main__":
    print("🚀 Hierarchical RL Integration Example")
    print("This example demonstrates the integration of hierarchical reachability")
    print("management with PPO training for improved N++ level completion.\n")
    
    # Run demonstrations
    demonstrate_subgoal_types()
    trained_agent = demonstrate_hierarchical_integration()
    
    print("\n" + "=" * 60)
    print("✨ Example completed successfully!")
    print("\nTo use this in production:")
    print("1. Replace mock classes with real environment and PPO implementation")
    print("2. Ensure nclone dependencies are properly installed")
    print("3. Adjust hyperparameters based on your specific levels")
    print("4. Monitor performance metrics and optimize cache settings")
    print("\nFor more information, see the hierarchical reachability manager documentation.")