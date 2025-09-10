# Task 003: Implement Hierarchical RL Framework

## Overview
Implement a hierarchical reinforcement learning (HRL) framework that leverages reachability-aware subgoal selection for efficient learning on complex N++ levels. This includes creating high-level and low-level policies, subtask environments, and the coordination mechanism between hierarchical levels.

## Context Reference
See [npp-rl comprehensive technical roadmap](../docs/comprehensive_technical_roadmap.md) Section 2.1: "HRL Framework Design with Reachability Integration" and Section 1.3: "Integration with RL Architecture"

## Requirements

### Primary Objectives
1. **Implement Reachability-Aware Hierarchical Agent** with high-level and low-level policies
2. **Create Subtask Environment Wrappers** for specialized training
3. **Build Hierarchical Coordination System** for policy switching and communication
4. **Integrate with Reachability System** for intelligent subgoal selection
5. **Optimize for Sample Efficiency** compared to flat RL approaches

### Hierarchical Architecture Design
The HRL system will have two levels:
- **High-Level Policy**: Selects subgoals from reachable options (longer time horizons)
- **Low-Level Policies**: Execute specific subtasks (shorter time horizons)
- **Coordination Layer**: Manages policy switching and subgoal completion detection

### Components to Implement

#### 1. Reachability-Aware Hierarchical Agent
**New File**: `npp_rl/agents/hierarchical_agent.py`

**Core Architecture**:
```python
class ReachabilityAwareHierarchicalAgent:
    def __init__(self, observation_space: SpacesDict, action_space: gym.Space, config: dict):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        
        # Reachability system for intelligent subgoal selection
        self.reachability_manager = HierarchicalReachabilityManager(
            ReachabilityAnalyzer(TrajectoryCalculator())
        )
        
        # High-level policy selects from REACHABLE subtasks only
        self.high_level_policy = PPO(
            policy=HGTMultimodalExtractor,
            env=ReachabilityFilteredSubtaskEnv(),
            n_steps=config['high_level_steps'],  # Longer time horizons (2048)
            **config['high_level_ppo_params']
        )
        
        # Low-level policies execute subtasks with physics awareness
        self.low_level_policies = {}
        for subtask in SUBTASK_DEFINITIONS:
            self.low_level_policies[subtask] = PPO(
                policy=HGTMultimodalExtractor,
                env=PhysicsAwareSubtaskEnv(subtask),
                n_steps=config['low_level_steps'],  # Shorter time horizons (512)
                **config['low_level_ppo_params']
            )
        
        # Level completion planner for strategic guidance
        self.completion_planner = PhysicsAwareLevelCompletionPlanner(
            self.reachability_manager.analyzer
        )
        
        # Coordination state
        self.current_subtask = None
        self.subtask_start_time = 0
        self.subtask_timeout = config.get('subtask_timeout', 1000)  # Max steps per subtask
        self.subgoal_completion_detector = SubgoalCompletionDetector()
        
    def select_action(self, observation: dict) -> int:
        """Reachability-aware hierarchical action selection."""
        # Extract current game state
        ninja_pos = self._extract_ninja_position(observation)
        level_data = self._extract_level_data(observation)
        switch_states = self._extract_switch_states(observation)
        
        # Check if we need to select a new subtask
        if self._should_select_new_subtask(observation):
            # Get only reachable subgoals
            reachable_subgoals = self.reachability_manager.get_reachable_subgoals(
                ninja_pos, level_data, switch_states
            )
            
            # Use completion planner for strategic guidance
            strategic_plan = self.completion_planner.plan_completion_strategy(
                ninja_pos, level_data, switch_states
            )
            
            # High-level policy selects from reachable options, guided by strategy
            filtered_observation = self._filter_observation_by_reachability(
                observation, reachable_subgoals, strategic_plan
            )
            
            subtask_id = self.high_level_policy.predict(filtered_observation)[0]
            self.current_subtask = reachable_subgoals[subtask_id]
            self.subtask_start_time = self._get_current_step()
            
            if self.config.get('debug', False):
                print(f"Selected subtask: {self.current_subtask} from {reachable_subgoals}")
        
        # Execute current subtask with low-level policy
        if self.current_subtask in self.low_level_policies:
            action = self.low_level_policies[self.current_subtask].predict(observation)[0]
        else:
            # Fallback to random action if subtask not found
            action = self.action_space.sample()
            
        return action
    
    def _should_select_new_subtask(self, observation: dict) -> bool:
        """Determine if we should select a new subtask."""
        # First time or no current subtask
        if self.current_subtask is None:
            return True
        
        # Subtask timeout
        current_step = self._get_current_step()
        if current_step - self.subtask_start_time > self.subtask_timeout:
            return True
        
        # Subtask completion detection
        if self.subgoal_completion_detector.is_subtask_completed(
            self.current_subtask, observation
        ):
            return True
        
        # Subtask became unreachable (environment changed)
        ninja_pos = self._extract_ninja_position(observation)
        level_data = self._extract_level_data(observation)
        switch_states = self._extract_switch_states(observation)
        
        reachable_subgoals = self.reachability_manager.get_reachable_subgoals(
            ninja_pos, level_data, switch_states
        )
        
        if self.current_subtask not in reachable_subgoals:
            return True
        
        return False
    
    def _filter_observation_by_reachability(self, observation: dict, 
                                          reachable_subgoals: List[str],
                                          strategic_plan: List[str]) -> dict:
        """Filter observation to only show reachable subgoals to high-level policy."""
        filtered_obs = observation.copy()
        
        # Encode available subgoals
        subgoal_encoding = np.zeros(len(SUBTASK_DEFINITIONS))
        for i, subtask in enumerate(SUBTASK_DEFINITIONS):
            if subtask in reachable_subgoals:
                subgoal_encoding[i] = 1.0
        
        # Add strategic guidance
        strategic_encoding = np.zeros(len(SUBTASK_DEFINITIONS))
        if strategic_plan:
            next_strategic_action = strategic_plan[0]
            if next_strategic_action in SUBTASK_DEFINITIONS:
                strategic_idx = SUBTASK_DEFINITIONS.index(next_strategic_action)
                strategic_encoding[strategic_idx] = 1.0
        
        # Add to observation
        filtered_obs['available_subgoals'] = subgoal_encoding
        filtered_obs['strategic_guidance'] = strategic_encoding
        
        return filtered_obs
```

#### 2. Subtask Environment Wrappers
**New File**: `npp_rl/environments/subtask_envs.py`

**Subtask Definitions**:
```python
SUBTASK_DEFINITIONS = [
    'navigate_to_exit_switch',
    'navigate_to_exit_door', 
    'activate_door_switch',
    'collect_gold',
    'avoid_hazard',
    'perform_wall_jump',
    'use_launch_pad',
    'navigate_bounce_blocks'
]

class PhysicsAwareSubtaskEnv(gym.Wrapper):
    """Environment wrapper for training low-level policies on specific subtasks."""
    
    def __init__(self, env, subtask: str, config: dict = None):
        super().__init__(env)
        self.subtask = subtask
        self.config = config or {}
        self.subtask_reward_shaper = SubtaskRewardShaper(subtask)
        self.completion_detector = SubgoalCompletionDetector()
        
        # Subtask-specific parameters
        self.max_episode_steps = self.config.get('max_subtask_steps', 1000)
        self.current_step = 0
        self.subtask_start_pos = None
        self.target_position = None
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_step = 0
        
        # Set subtask-specific target
        self.target_position = self._identify_subtask_target(obs)
        self.subtask_start_pos = self._extract_ninja_position(obs)
        
        # Add subtask information to observation
        obs = self._add_subtask_info(obs)
        
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        
        # Apply subtask-specific reward shaping
        shaped_reward = self.subtask_reward_shaper.shape_reward(
            obs, reward, self.target_position, self.subtask_start_pos
        )
        
        # Check subtask completion
        subtask_completed = self.completion_detector.is_subtask_completed(
            self.subtask, obs
        )
        
        # Subtask timeout
        subtask_timeout = self.current_step >= self.max_episode_steps
        
        # Update done condition
        done = done or subtask_completed or subtask_timeout
        
        # Add subtask information
        obs = self._add_subtask_info(obs)
        info['subtask'] = {
            'name': self.subtask,
            'completed': subtask_completed,
            'timeout': subtask_timeout,
            'progress': self._compute_subtask_progress(obs),
            'shaped_reward': shaped_reward
        }
        
        return obs, shaped_reward, done, info
    
    def _identify_subtask_target(self, obs) -> Tuple[float, float]:
        """Identify the target position for the current subtask."""
        level_data = self._extract_level_data(obs)
        
        if self.subtask == 'navigate_to_exit_switch':
            return self._find_exit_switch_position(level_data)
        elif self.subtask == 'navigate_to_exit_door':
            return self._find_exit_door_position(level_data)
        elif self.subtask.startswith('activate_door_switch'):
            door_id = self._extract_door_id_from_subtask(self.subtask)
            return self._find_door_switch_position(level_data, door_id)
        elif self.subtask == 'collect_gold':
            return self._find_nearest_gold_position(obs)
        elif self.subtask == 'avoid_hazard':
            return self._find_safe_position(obs)
        else:
            # Default to current position for complex subtasks
            return self._extract_ninja_position(obs)
    
    def _add_subtask_info(self, obs) -> dict:
        """Add subtask-specific information to observation."""
        obs = obs.copy()
        
        # Add target position
        if self.target_position:
            obs['subtask_target'] = np.array(self.target_position, dtype=np.float32)
        
        # Add subtask encoding
        subtask_encoding = np.zeros(len(SUBTASK_DEFINITIONS))
        if self.subtask in SUBTASK_DEFINITIONS:
            subtask_idx = SUBTASK_DEFINITIONS.index(self.subtask)
            subtask_encoding[subtask_idx] = 1.0
        obs['current_subtask'] = subtask_encoding
        
        # Add progress information
        obs['subtask_progress'] = np.array([self._compute_subtask_progress(obs)], dtype=np.float32)
        
        return obs

class ReachabilityFilteredSubtaskEnv(gym.Wrapper):
    """Environment wrapper that only presents reachable subtasks to high-level policy."""
    
    def __init__(self, env, reachability_manager: HierarchicalReachabilityManager):
        super().__init__(env)
        self.reachability_manager = reachability_manager
        
        # Modify action space to be subtask selection
        self.action_space = gym.spaces.Discrete(len(SUBTASK_DEFINITIONS))
        
    def step(self, action):
        # High-level policy selects subtask index
        # This wrapper doesn't actually execute actions, just provides subtask selection interface
        obs, reward, done, info = self.env.step(0)  # No-op action
        
        # Add reachability information
        ninja_pos = self._extract_ninja_position(obs)
        level_data = self._extract_level_data(obs)
        switch_states = self._extract_switch_states(obs)
        
        reachable_subgoals = self.reachability_manager.get_reachable_subgoals(
            ninja_pos, level_data, switch_states
        )
        
        # Encode reachable subgoals
        subgoal_mask = np.zeros(len(SUBTASK_DEFINITIONS))
        for subtask in reachable_subgoals:
            if subtask in SUBTASK_DEFINITIONS:
                idx = SUBTASK_DEFINITIONS.index(subtask)
                subgoal_mask[idx] = 1.0
        
        obs['reachable_subgoals_mask'] = subgoal_mask
        
        # Reward based on selecting reachable vs unreachable subgoals
        if action < len(SUBTASK_DEFINITIONS):
            selected_subtask = SUBTASK_DEFINITIONS[action]
            if selected_subtask in reachable_subgoals:
                reward += 1.0  # Bonus for selecting reachable subgoal
            else:
                reward -= 1.0  # Penalty for selecting unreachable subgoal
        
        return obs, reward, done, info
```

#### 3. Subgoal Completion Detection
**New File**: `npp_rl/utils/subgoal_completion.py`

**Core Functionality**:
```python
class SubgoalCompletionDetector:
    """Detects when subtasks/subgoals have been completed."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.completion_thresholds = {
            'position_threshold': 20.0,  # pixels
            'switch_activation_delay': 5,  # frames
            'gold_collection_delay': 3,   # frames
        }
        
    def is_subtask_completed(self, subtask: str, observation: dict) -> bool:
        """Check if a specific subtask has been completed."""
        if subtask == 'navigate_to_exit_switch':
            return self._is_navigation_completed(observation, 'exit_switch')
        elif subtask == 'navigate_to_exit_door':
            return self._is_navigation_completed(observation, 'exit_door')
        elif subtask.startswith('activate_door_switch'):
            door_id = self._extract_door_id_from_subtask(subtask)
            return self._is_switch_activated(observation, f'door_{door_id}')
        elif subtask == 'collect_gold':
            return self._is_gold_collected(observation)
        elif subtask == 'avoid_hazard':
            return self._is_hazard_avoided(observation)
        elif subtask == 'perform_wall_jump':
            return self._is_wall_jump_completed(observation)
        elif subtask == 'use_launch_pad':
            return self._is_launch_pad_used(observation)
        elif subtask == 'navigate_bounce_blocks':
            return self._is_bounce_sequence_completed(observation)
        else:
            return False
    
    def _is_navigation_completed(self, observation: dict, target_type: str) -> bool:
        """Check if navigation to a target has been completed."""
        ninja_pos = self._extract_ninja_position(observation)
        level_data = self._extract_level_data(observation)
        
        if target_type == 'exit_switch':
            target_pos = self._find_exit_switch_position(level_data)
        elif target_type == 'exit_door':
            target_pos = self._find_exit_door_position(level_data)
        else:
            return False
        
        if target_pos is None:
            return False
        
        # Check distance to target
        distance = np.sqrt((ninja_pos[0] - target_pos[0])**2 + (ninja_pos[1] - target_pos[1])**2)
        return distance < self.completion_thresholds['position_threshold']
    
    def _is_switch_activated(self, observation: dict, switch_id: str) -> bool:
        """Check if a switch has been activated."""
        switch_states = self._extract_switch_states(observation)
        return switch_states.get(switch_id, False)
    
    def _is_gold_collected(self, observation: dict) -> bool:
        """Check if gold has been collected (simplified - check score increase)."""
        # This would need to track score changes or gold entity states
        current_score = self._extract_score(observation)
        return hasattr(self, 'last_score') and current_score > self.last_score
    
    def _is_hazard_avoided(self, observation: dict) -> bool:
        """Check if hazard has been successfully avoided."""
        ninja_pos = self._extract_ninja_position(observation)
        hazards = self._extract_hazard_positions(observation)
        
        # Check if ninja is safe distance from all hazards
        safe_distance = 50.0  # pixels
        for hazard_pos in hazards:
            distance = np.sqrt((ninja_pos[0] - hazard_pos[0])**2 + (ninja_pos[1] - hazard_pos[1])**2)
            if distance < safe_distance:
                return False
        
        return True
    
    def _is_wall_jump_completed(self, observation: dict) -> bool:
        """Check if a wall jump maneuver has been completed."""
        # This would need to track ninja state and detect wall jump physics
        ninja_state = self._extract_ninja_physics_state(observation)
        
        # Check for wall jump indicators (velocity patterns, contact states)
        return (ninja_state.get('wall_contact', False) and 
                ninja_state.get('upward_velocity', 0) > 100)
    
    def _is_launch_pad_used(self, observation: dict) -> bool:
        """Check if launch pad has been successfully used."""
        ninja_state = self._extract_ninja_physics_state(observation)
        
        # Check for launch pad boost indicators
        return ninja_state.get('launch_pad_boost', False)
    
    def _is_bounce_sequence_completed(self, observation: dict) -> bool:
        """Check if bounce block sequence has been completed."""
        ninja_state = self._extract_ninja_physics_state(observation)
        
        # Check for successful bounce block interaction
        return ninja_state.get('bounce_block_hits', 0) > 0
```

#### 4. Subtask Reward Shaping
**New File**: `npp_rl/utils/subtask_rewards.py`

**Core Functionality**:
```python
class SubtaskRewardShaper:
    """Provides reward shaping for specific subtasks to improve learning efficiency."""
    
    def __init__(self, subtask: str, config: dict = None):
        self.subtask = subtask
        self.config = config or {}
        self.last_distance_to_target = None
        self.last_ninja_pos = None
        
    def shape_reward(self, observation: dict, base_reward: float, 
                    target_position: Tuple[float, float], 
                    start_position: Tuple[float, float]) -> float:
        """Apply subtask-specific reward shaping."""
        shaped_reward = base_reward
        
        if self.subtask.startswith('navigate_to'):
            shaped_reward += self._navigation_reward_shaping(
                observation, target_position, start_position
            )
        elif self.subtask.startswith('activate_'):
            shaped_reward += self._activation_reward_shaping(observation)
        elif self.subtask == 'collect_gold':
            shaped_reward += self._collection_reward_shaping(observation)
        elif self.subtask == 'avoid_hazard':
            shaped_reward += self._avoidance_reward_shaping(observation)
        elif self.subtask.startswith('perform_'):
            shaped_reward += self._skill_reward_shaping(observation)
        
        return shaped_reward
    
    def _navigation_reward_shaping(self, observation: dict, 
                                 target_position: Tuple[float, float],
                                 start_position: Tuple[float, float]) -> float:
        """Reward shaping for navigation subtasks."""
        ninja_pos = self._extract_ninja_position(observation)
        
        # Distance-based reward shaping
        current_distance = np.sqrt(
            (ninja_pos[0] - target_position[0])**2 + 
            (ninja_pos[1] - target_position[1])**2
        )
        
        # Progress reward
        if self.last_distance_to_target is not None:
            progress = self.last_distance_to_target - current_distance
            progress_reward = progress * 0.01  # Scale factor
        else:
            progress_reward = 0.0
        
        self.last_distance_to_target = current_distance
        
        # Proximity bonus
        max_distance = np.sqrt(
            (start_position[0] - target_position[0])**2 + 
            (start_position[1] - target_position[1])**2
        )
        
        if max_distance > 0:
            proximity_bonus = (1.0 - current_distance / max_distance) * 0.1
        else:
            proximity_bonus = 0.1
        
        return progress_reward + proximity_bonus
    
    def _activation_reward_shaping(self, observation: dict) -> float:
        """Reward shaping for switch activation subtasks."""
        # Reward for being near switch
        ninja_pos = self._extract_ninja_position(observation)
        switch_positions = self._extract_switch_positions(observation)
        
        min_distance = float('inf')
        for switch_pos in switch_positions:
            distance = np.sqrt(
                (ninja_pos[0] - switch_pos[0])**2 + 
                (ninja_pos[1] - switch_pos[1])**2
            )
            min_distance = min(min_distance, distance)
        
        # Proximity reward
        if min_distance < 30.0:  # Close to switch
            return 0.1
        elif min_distance < 60.0:  # Moderately close
            return 0.05
        else:
            return 0.0
    
    def _collection_reward_shaping(self, observation: dict) -> float:
        """Reward shaping for gold collection subtasks."""
        ninja_pos = self._extract_ninja_position(observation)
        gold_positions = self._extract_gold_positions(observation)
        
        if not gold_positions:
            return 0.0
        
        # Find nearest gold
        min_distance = float('inf')
        for gold_pos in gold_positions:
            distance = np.sqrt(
                (ninja_pos[0] - gold_pos[0])**2 + 
                (ninja_pos[1] - gold_pos[1])**2
            )
            min_distance = min(min_distance, distance)
        
        # Distance-based reward
        if min_distance < 20.0:
            return 0.2  # Very close to gold
        elif min_distance < 50.0:
            return 0.1  # Close to gold
        else:
            return 0.0
    
    def _avoidance_reward_shaping(self, observation: dict) -> float:
        """Reward shaping for hazard avoidance subtasks."""
        ninja_pos = self._extract_ninja_position(observation)
        hazard_positions = self._extract_hazard_positions(observation)
        
        # Penalty for being too close to hazards
        penalty = 0.0
        safe_distance = 50.0
        
        for hazard_pos in hazard_positions:
            distance = np.sqrt(
                (ninja_pos[0] - hazard_pos[0])**2 + 
                (ninja_pos[1] - hazard_pos[1])**2
            )
            
            if distance < safe_distance:
                penalty -= (safe_distance - distance) * 0.01
        
        return penalty
    
    def _skill_reward_shaping(self, observation: dict) -> float:
        """Reward shaping for skill-based subtasks (wall jump, launch pad, etc.)."""
        ninja_state = self._extract_ninja_physics_state(observation)
        
        if self.subtask == 'perform_wall_jump':
            # Reward for wall contact and upward movement
            if ninja_state.get('wall_contact', False):
                reward = 0.1
                if ninja_state.get('upward_velocity', 0) > 50:
                    reward += 0.2  # Bonus for successful wall jump
                return reward
        
        elif self.subtask == 'use_launch_pad':
            # Reward for launch pad interaction
            if ninja_state.get('launch_pad_contact', False):
                return 0.2
        
        elif self.subtask == 'navigate_bounce_blocks':
            # Reward for bounce block hits
            bounce_hits = ninja_state.get('bounce_block_hits', 0)
            return bounce_hits * 0.1
        
        return 0.0
```

## Acceptance Criteria

### Functional Requirements
1. **Hierarchical Structure**: Clear separation between high-level and low-level policies
2. **Reachability Integration**: High-level policy only selects from reachable subgoals
3. **Subtask Specialization**: Low-level policies specialize for their assigned subtasks
4. **Completion Detection**: Accurate detection of subtask completion
5. **Strategic Guidance**: Integration with level completion planner

### Technical Requirements
1. **Sample Efficiency**: >2x sample efficiency compared to flat RL
2. **Decision Speed**: <10ms for hierarchical action selection
3. **Memory Efficiency**: <500MB additional memory for hierarchical components
4. **Scalability**: Support for adding new subtasks without major refactoring

### Quality Requirements
1. **Modular Design**: Clean separation of concerns between components
2. **Debugging Support**: Comprehensive logging and visualization
3. **Configuration**: Flexible configuration for different HRL strategies
4. **Testing**: Full unit and integration test coverage

## Test Scenarios

### Unit Tests
**File**: `tests/test_hierarchical_rl.py`

```python
class TestHierarchicalRL(unittest.TestCase):
    def test_hierarchical_agent_initialization(self):
        """Test proper initialization of hierarchical agent components."""
        config = self._get_test_config()
        agent = ReachabilityAwareHierarchicalAgent(
            observation_space=self._get_test_obs_space(),
            action_space=gym.spaces.Discrete(6),
            config=config
        )
        
        # Validate component initialization
        self.assertIsNotNone(agent.reachability_manager)
        self.assertIsNotNone(agent.high_level_policy)
        self.assertIsNotNone(agent.low_level_policies)
        self.assertIsNotNone(agent.completion_planner)
        
        # Validate subtask definitions
        for subtask in SUBTASK_DEFINITIONS:
            self.assertIn(subtask, agent.low_level_policies)
    
    def test_subtask_environment_wrapper(self):
        """Test subtask environment wrapper functionality."""
        base_env = self._create_mock_env()
        subtask_env = PhysicsAwareSubtaskEnv(base_env, 'navigate_to_exit_switch')
        
        obs = subtask_env.reset()
        
        # Should include subtask information
        self.assertIn('subtask_target', obs)
        self.assertIn('current_subtask', obs)
        self.assertIn('subtask_progress', obs)
        
        # Test step
        action = subtask_env.action_space.sample()
        obs, reward, done, info = subtask_env.step(action)
        
        # Should include subtask info
        self.assertIn('subtask', info)
        self.assertIn('shaped_reward', info['subtask'])
    
    def test_subgoal_completion_detection(self):
        """Test subgoal completion detection."""
        detector = SubgoalCompletionDetector()
        
        # Test navigation completion
        obs = self._create_test_observation_near_exit_switch()
        is_completed = detector.is_subtask_completed('navigate_to_exit_switch', obs)
        self.assertTrue(is_completed)
        
        # Test switch activation
        obs_with_switch = self._create_test_observation_with_activated_switch()
        is_completed = detector.is_subtask_completed('activate_door_switch_1', obs_with_switch)
        self.assertTrue(is_completed)
    
    def test_reward_shaping(self):
        """Test subtask reward shaping."""
        shaper = SubtaskRewardShaper('navigate_to_exit_switch')
        
        obs = self._create_test_observation()
        base_reward = 0.0
        target_pos = (200, 300)
        start_pos = (50, 400)
        
        shaped_reward = shaper.shape_reward(obs, base_reward, target_pos, start_pos)
        
        # Should provide additional reward signal
        self.assertNotEqual(shaped_reward, base_reward)
        self.assertIsInstance(shaped_reward, float)
```

### Integration Tests
**File**: `tests/test_hrl_integration.py`

```python
class TestHRLIntegration(unittest.TestCase):
    def test_hierarchical_training_loop(self):
        """Test hierarchical training with reachability constraints."""
        config = self._get_test_config()
        env = self._create_test_env()
        agent = ReachabilityAwareHierarchicalAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=config
        )
        
        # Run short training episode
        obs = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Validate training metrics
        self.assertGreater(steps, 10)  # Should take some steps
        self.assertGreater(total_reward, -1000)  # Should not fail immediately
        
        # Validate hierarchical structure was used
        self.assertIsNotNone(agent.current_subtask)
    
    def test_reachability_subgoal_filtering(self):
        """Test that only reachable subgoals are selected."""
        config = self._get_test_config()
        agent = ReachabilityAwareHierarchicalAgent(
            observation_space=self._get_test_obs_space(),
            action_space=gym.spaces.Discrete(6),
            config=config
        )
        
        # Create observation with known reachability constraints
        obs = self._create_test_observation_blocked_exit()
        
        # Mock reachability manager to return specific subgoals
        agent.reachability_manager.get_reachable_subgoals = Mock(
            return_value=['activate_door_switch_1', 'collect_gold']
        )
        
        # Force subtask selection
        agent.current_subtask = None
        action = agent.select_action(obs)
        
        # Should have selected a reachable subtask
        self.assertIn(agent.current_subtask, ['activate_door_switch_1', 'collect_gold'])
    
    def test_subtask_specialization(self):
        """Test that low-level policies specialize for their subtasks."""
        # Create environments for different subtasks
        base_env = self._create_mock_env()
        nav_env = PhysicsAwareSubtaskEnv(base_env, 'navigate_to_exit_switch')
        jump_env = PhysicsAwareSubtaskEnv(base_env, 'perform_wall_jump')
        
        # Train simple policies (mock training)
        nav_policy = self._train_mock_policy(nav_env)
        jump_policy = self._train_mock_policy(jump_env)
        
        # Test on navigation scenario
        nav_obs = self._create_navigation_test_observation()
        nav_action = nav_policy.predict(nav_obs)[0]
        jump_action = jump_policy.predict(nav_obs)[0]
        
        # Policies should make different decisions
        # (This is a simplified test - in practice, we'd measure performance differences)
        self.assertIsInstance(nav_action, int)
        self.assertIsInstance(jump_action, int)
```

### Performance Tests
**File**: `tests/test_hrl_performance.py`

```python
class TestHRLPerformance(unittest.TestCase):
    def test_hierarchical_decision_speed(self):
        """Test decision-making speed for hierarchical agent."""
        config = self._get_test_config()
        agent = ReachabilityAwareHierarchicalAgent(
            observation_space=self._get_test_obs_space(),
            action_space=gym.spaces.Discrete(6),
            config=config
        )
        
        obs = self._create_test_observation()
        
        # Time action selection
        start_time = time.time()
        for _ in range(100):
            action = agent.select_action(obs)
        decision_time = time.time() - start_time
        
        # Should make decisions quickly (< 10ms per decision)
        avg_decision_time = decision_time / 100
        self.assertLess(avg_decision_time, 0.01)
    
    def test_sample_efficiency_comparison(self):
        """Test sample efficiency compared to flat RL."""
        # This would be a longer integration test comparing learning curves
        # between hierarchical and flat RL agents
        
        # Create both agents
        hrl_agent = self._create_hrl_agent()
        flat_agent = self._create_flat_agent()
        
        # Train both for same number of steps
        hrl_performance = self._train_agent(hrl_agent, steps=10000)
        flat_performance = self._train_agent(flat_agent, steps=10000)
        
        # HRL should achieve better performance with same number of samples
        self.assertGreater(hrl_performance['final_reward'], flat_performance['final_reward'])
```

## Implementation Steps

### Phase 1: Core Hierarchical Framework (1 week)
1. **Implement Hierarchical Agent**
   - Create agent class with high-level and low-level policies
   - Add reachability integration
   - Implement policy coordination

2. **Create Subtask Definitions**
   - Define standard subtask set
   - Create subtask encoding/decoding
   - Add subtask metadata

### Phase 2: Subtask Environments (1 week)
1. **Implement Subtask Wrappers**
   - Create physics-aware subtask environments
   - Add reward shaping
   - Implement completion detection

2. **Create Reachability Filtering**
   - Implement high-level policy environment
   - Add subgoal masking
   - Create strategic guidance integration

### Phase 3: Coordination and Completion (3-4 days)
1. **Implement Completion Detection**
   - Create completion detector for all subtasks
   - Add physics-based completion criteria
   - Implement timeout handling

2. **Add Reward Shaping**
   - Implement subtask-specific reward shaping
   - Add progress tracking
   - Create skill-based rewards

### Phase 4: Integration and Testing (1 week)
1. **Integration with Existing Systems**
   - Integrate with HGT architecture
   - Add to training pipeline
   - Create configuration system

2. **Comprehensive Testing**
   - Unit tests for all components
   - Integration tests with RL training
   - Performance benchmarks

## Success Metrics
- **Sample Efficiency**: >2x improvement over flat RL
- **Subtask Completion**: >75% subtask completion rate
- **Reachability Compliance**: >95% of selected subtasks are reachable
- **Decision Speed**: <10ms for hierarchical decisions
- **Scalability**: Easy addition of new subtasks

## Dependencies
- Reachability system integration (Task 002)
- Enhanced HGT multimodal architecture
- Existing PPO implementation
- Subtask environment infrastructure

## Estimated Effort
- **Time**: 3-4 weeks
- **Complexity**: High (complex hierarchical coordination)
- **Risk**: Medium-High (novel HRL architecture)

## Notes
- Start with simple subtask set and expand gradually
- Consider curriculum learning for subtask difficulty
- Plan for extensive hyperparameter tuning
- Coordinate with reachability system development
- Consider multi-agent extensions for future work