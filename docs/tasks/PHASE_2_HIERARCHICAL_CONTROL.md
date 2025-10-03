# Phase 2: Hierarchical Control

**Dependencies**: Phase 1 completed  

## Overview

This phase implements a sophisticated two-level hierarchical reinforcement learning architecture that leverages the completion planner to handle complex multi-switch levels. The system will coordinate high-level strategic decisions with low-level movement execution, using ICM-enhanced exploration to handle physics uncertainty.

## Objectives

1. Implement robust two-level policy architecture
2. Create subtask-specific reward functions for dense feedback
3. Integrate mine avoidance with hierarchical navigation
4. Ensure stable training across both policy levels
5. Achieve reliable completion of multi-switch dependency levels

## Task Breakdown

### Task 2.1: Two-Level Policy Architecture

**What we want to do**: Implement a sophisticated hierarchical architecture where a high-level policy selects subtasks based on reachability analysis, and a low-level policy executes movement actions with ICM-enhanced exploration.

**Current state**: 
- Basic hierarchical controller from Phase 1 provides subtask selection
- Single PPO policy handles all decisions
- No separation between strategic and tactical decision making
- ICM exists but not integrated with hierarchical structure

**Files to create/modify**:
- `npp_rl/agents/hierarchical_ppo.py` (enhance from Phase 1)
- `npp_rl/models/hierarchical_policy.py` (new)
- `npp_rl/hrl/subtask_policies.py` (new)
- `npp_rl/hrl/high_level_policy.py` (new)
- `npp_rl/agents/training.py` (modify for hierarchical training)

**Detailed architecture design**:

#### High-Level Policy
**Input features**:
- 8D reachability features (primary decision input)
  - Switch accessibility [2]: Can exit switch be reached?
  - Exit accessibility [3]: Can exit door be reached?
  - Objective distance [1]: Distance to current objective
  - Connectivity score [5]: Overall level connectivity
- Switch state vector (exit switch activated, locked door states)
- Current ninja position (normalized)
- Time remaining in episode

**Output**: Subtask selection (4 discrete actions)
1. `navigate_to_exit_switch` - Priority when exit switch reachable
2. `navigate_to_locked_door_switch` - When exit switch unreachable, find nearest locked door switch
3. `navigate_to_exit_door` - When exit switch activated and exit reachable
4. `explore_for_switches` - When no clear path exists

**Decision logic** (based on completion planner heuristic):
```python
def select_subtask(self, reachability_features, switch_states):
    exit_switch_reachable = reachability_features[2] > 0.5
    exit_door_reachable = reachability_features[3] > 0.5
    exit_switch_activated = switch_states['exit_switch']
    
    if not exit_switch_activated:
        if exit_switch_reachable:
            return 'navigate_to_exit_switch'
        else:
            return 'navigate_to_locked_door_switch'
    else:  # exit switch activated
        if exit_door_reachable:
            return 'navigate_to_exit_door'
        else:
            return 'navigate_to_locked_door_switch'
```

#### Low-Level Policy
**Input features**:
- Full multimodal observations from HGTMultimodalExtractor (512D)
- Current subtask embedding (64D learned embedding)
- Subtask-specific context:
  - Target position for current subtask
  - Distance to target
  - Mine proximity warnings
  - Time since subtask started

**Output**: Movement actions (6 discrete actions)
- NOOP, Left, Right, Jump, Jump+Left, Jump+Right

**ICM Integration**:
- ICM operates at low-level policy level
- Curiosity rewards modulated by current subtask
- Higher exploration when subtask target is uncertain
- Reachability-aware curiosity focusing on relevant areas

#### Shared Components
**Feature Extractor**: Single HGTMultimodalExtractor shared between policies
**Memory**: Separate experience buffers for high-level and low-level policies
**Training**: Alternating updates with careful learning rate scheduling

**Implementation details**:

1. **Policy Network Architecture**:
```python
class HierarchicalPolicy(nn.Module):
    def __init__(self, feature_extractor, high_level_dim=128, low_level_dim=512):
        self.shared_extractor = feature_extractor
        self.high_level_policy = HighLevelPolicy(high_level_dim)
        self.low_level_policy = LowLevelPolicy(low_level_dim)
        self.subtask_embeddings = nn.Embedding(4, 64)  # 4 subtasks
    
    def forward(self, obs, current_subtask=None):
        shared_features = self.shared_extractor(obs)
        
        # High-level decision (every N steps)
        if self.should_update_subtask():
            subtask = self.high_level_policy(shared_features[:high_level_dim])
        
        # Low-level action (every step)
        subtask_embed = self.subtask_embeddings(current_subtask)
        low_level_input = torch.cat([shared_features, subtask_embed], dim=-1)
        action = self.low_level_policy(low_level_input)
        
        return action, subtask
```

2. **Training Coordination**:
- High-level policy updates every 50-100 steps
- Low-level policy updates every step
- Separate PPO instances with coordinated learning rates
- Experience replay buffers maintain subtask context

3. **Subtask Transition Logic**:
- Automatic transitions based on completion planner output
- Timeout-based transitions (max 500 steps per subtask)
- Success-based transitions (switch activated, exit reached)
- Failure-based transitions (stuck detection, when the exit door or exit switch is not reachable and no locked door switch is reachable)

**Acceptance criteria**:
- [ ] High-level policy successfully selects appropriate subtasks
- [ ] Low-level policy executes actions for current subtask
- [ ] ICM integration provides effective exploration
- [ ] Subtask transitions occur at appropriate times
- [ ] Training remains stable across both policy levels
- [ ] Multi-switch levels show improved completion rates

**Testing requirements**:
- Unit tests for policy architecture components


---

### Task 2.2: Subtask-Specific Reward Functions

**What we want to do**: Implement dense reward functions that provide clear feedback for each subtask, enabling efficient learning of hierarchical behaviors.

**Current state**:
- Basic completion rewards from Phase 1 (+0.1 switch, +1.0 exit)
- PBRS provides distance-based shaping
- No subtask-specific reward structure
- Rewards not aligned with hierarchical decision making

**Files to modify**:
- `nclone/gym_environment/reward_calculation/main_reward_calculator.py`
- `nclone/gym_environment/reward_calculation/pbrs_potentials.py`
- `npp_rl/hrl/subtask_rewards.py` (new)

**Detailed reward structure**:

#### Base Rewards (unchanged from Phase 1)
- Switch activation: +0.1
- Exit completion: +1.0
- Death penalty: -0.5
- Time penalty: -0.01 per step

#### Subtask-Specific Dense Rewards

1. **navigate_to_exit_switch**:
   - Progress reward: +0.02 per unit distance reduction to exit switch
   - Proximity bonus: +0.05 when within 2 tiles of exit switch
   - Activation bonus: +0.1 (base reward) when switch activated
   - Timeout penalty: -0.1 if subtask exceeds 300 steps

2. **navigate_to_locked_door_switch**:
   - Progress reward: +0.02 per unit distance reduction to target locked door switch
   - Switch selection bonus: +0.01 for approaching nearest reachable locked door switch
   - Activation bonus: +0.05 when locked door switch activated
   - Door opening bonus: +0.03 when locked door opens (enabling new paths)

3. **navigate_to_exit_door**:
   - Progress reward: +0.03 per unit distance reduction to exit door
   - Proximity bonus: +0.1 when within 1 tile of exit door
   - Completion bonus: +1.0 (base reward) when exit reached
   - Efficiency bonus: +0.2 if completed quickly after switch activation

4. **explore_for_switches**:
   - Exploration reward: +0.01 for visiting new areas
   - Discovery bonus: +0.05 for finding previously unknown locked doors
   - Connectivity bonus: +0.02 for improving reachability score
   - Timeout transition: Automatic switch to specific subtask after 200 steps

#### PBRS Integration
**Subtask-aware potential functions**:
```python
def calculate_subtask_potential(self, obs, current_subtask):
    if current_subtask == 'navigate_to_exit_switch':
        return -distance_to_exit_switch(obs) * 0.1
    elif current_subtask == 'navigate_to_locked_switch':
        return -distance_to_nearest_locked_switch(obs) * 0.1
    elif current_subtask == 'navigate_to_exit_door':
        return -distance_to_exit_door(obs) * 0.15
    elif current_subtask == 'explore_for_switches':
        return exploration_potential(obs) * 0.05
```

#### Mine Avoidance Rewards
- Mine proximity penalty: -0.02 when within 1.5 tiles of toggled mine
- Safe navigation bonus: +0.01 for maintaining safe distance from mines
- Mine state awareness: +0.005 for correctly identifying mine states

**Implementation details**:

1. **Reward Calculator Enhancement**:
```python
class HierarchicalRewardCalculator:
    def __init__(self):
        self.base_calculator = RewardCalculator()
        self.subtask_rewards = SubtaskRewardCalculator()
    
    def calculate_reward(self, obs, prev_obs, current_subtask):
        base_reward = self.base_calculator.calculate_reward(obs, prev_obs)
        subtask_reward = self.subtask_rewards.calculate_subtask_reward(
            obs, prev_obs, current_subtask
        )
        return base_reward + subtask_reward
```

2. **Subtask Progress Tracking**:
- Track distance to objectives over time
- Detect progress vs. stagnation
- Provide bonus for consistent progress
- Penalize repetitive behaviors

3. **Reward Balancing**:
- Subtask rewards scaled to avoid overwhelming base rewards
- Progressive reward scaling based on level difficulty
- Adaptive reward weights based on training progress

**Acceptance criteria**:
- [ ] Each subtask provides clear, dense reward signals
- [ ] Reward structure encourages efficient subtask completion
- [ ] PBRS potentials align with current subtask objectives
- [ ] Mine avoidance integrated into reward structure
- [ ] Reward balance maintains training stability
- [ ] Subtask completion rates improve with dense rewards

**Testing requirements**:
- Unit tests for each subtask reward function
- Balance tests ensuring no reward component dominates

---

### Task 2.3: Mine Avoidance Integration

**What we want to do**: Integrate mine state awareness and avoidance behavior into the hierarchical navigation system, using reachability analysis to plan safe paths.

**Current state**:
- Mine entities processed in simplified entity system
- No mine-specific avoidance behavior
- Reachability analysis doesn't account for mine states
- No mine state tracking in hierarchical decisions

**Files to modify**:
- `nclone/graph/reachability/compact_features.py`
- `npp_rl/hrl/subtask_policies.py`
- `nclone/gym_environment/observation_processor.py`
- `npp_rl/models/entity_type_system.py`

**Mine state integration requirements**:

#### Mine State Representation
**Toggle Mine States** (from sim_mechanics_doc.md):
- Untoggled: Safe (3.5px radius), can be touched
- Toggling: Transitioning (4.5px radius), avoid during transition
- Toggled: Deadly (4px radius), must avoid completely

**State tracking in observations**:
```python
mine_state_vector = [
    mine_x, mine_y,           # Position
    mine_state,               # 0=untoggled, 1=toggling, 2=toggled
    mine_radius,              # Current radius based on state
    time_in_state,            # How long in current state
    ninja_distance,           # Distance from ninja
    is_blocking_path          # Whether mine blocks path to objective
]
```

#### Reachability Analysis Enhancement
**Mine-aware flood fill**:
- Treat toggled mines as obstacles in reachability analysis
- Account for mine radius in path planning
- Update reachability when mine states change
- Provide mine proximity warnings in 8D features

**Enhanced reachability features**:
- Feature [4] Hazard proximity: Distance to nearest dangerous mine
- Add mine-specific sub-features:
  - Nearest toggled mine distance
  - Number of mines blocking current path
  - Safe path availability score

#### Hierarchical Mine Avoidance

**High-level policy mine awareness**:
- Consider mine positions when selecting subtasks
- Prefer subtasks with safer paths
- Switch subtasks if path becomes blocked by mines

**Low-level policy mine avoidance**:
- Immediate mine avoidance in action selection
- Safe distance maintenance (2x mine radius)
- Strategic mine toggling when beneficial
- Emergency avoidance behaviors

**Implementation details**:

1. **Mine State Processor**:
```python
class MineStateProcessor:
    def __init__(self):
        self.mine_states = {}
        self.mine_positions = {}
    
    def update_mine_states(self, entities):
        for entity in entities:
            if entity.type == 'mine':
                self.mine_states[entity.id] = {
                    'state': entity.state,  # 0, 1, 2
                    'position': (entity.x, entity.y),
                    'radius': entity.radius,
                    'last_update': time.time()
                }
    
    def get_dangerous_mines(self, ninja_pos, safety_radius=2.0):
        dangerous = []
        for mine_id, mine_data in self.mine_states.items():
            if mine_data['state'] >= 1:  # toggling or toggled
                distance = calculate_distance(ninja_pos, mine_data['position'])
                if distance < safety_radius * mine_data['radius']:
                    dangerous.append(mine_data)
        return dangerous
```

2. **Safe Path Planning**:
```python
def plan_safe_path(self, start, goal, mine_states):
    # Use reachability analysis with mine obstacles
    safe_reachability = self.reachability_system.analyze_with_mines(
        start, goal, mine_states
    )
    
    # Prefer paths that avoid toggled mines
    if safe_reachability.has_safe_path:
        return safe_reachability.safe_path
    else:
        # Find path that requires mine state changes
        return self.plan_mine_manipulation_path(start, goal, mine_states)
```

3. **Mine Avoidance Actions**:
- **Immediate avoidance**: Override low-level actions when mine too close
- **Strategic positioning**: Position to safely toggle mines when needed
- **Path replanning**: Update subtask when mines block current path

#### Integration with ICM
**Mine-aware curiosity**:
- Reduce curiosity near dangerous mines
- Increase curiosity for safe mine manipulation opportunities
- Focus exploration on mine-free areas when possible

**Curiosity modulation**:
```python
def modulate_curiosity_for_mines(self, curiosity_reward, mine_proximity):
    if mine_proximity < DANGER_THRESHOLD:
        return curiosity_reward * 0.1  # Reduce exploration near danger
    elif mine_proximity < SAFE_THRESHOLD:
        return curiosity_reward * 0.5  # Moderate exploration
    else:
        return curiosity_reward  # Full exploration when safe
```

**Acceptance criteria**:
- [ ] Mine states accurately tracked and represented
- [ ] Reachability analysis accounts for mine obstacles
- [ ] High-level policy considers mine positions in subtask selection
- [ ] Low-level policy maintains safe distance from dangerous mines
- [ ] Strategic mine toggling when beneficial for path planning
- [ ] ICM curiosity appropriately modulated by mine proximity

**Testing requirements**:
- Unit tests for mine state processing

---

### Task 2.4: Training Stability and Optimization

**What we want to do**: Ensure stable training across both hierarchical policy levels with proper hyperparameter tuning and learning coordination.

**Current state**:
- Hierarchical architecture implemented but not optimized
- No specialized training procedures for two-level policies
- Standard PPO hyperparameters may not suit hierarchical structure
- No stability monitoring or adaptive learning rates

**Files to modify**:
- `npp_rl/agents/training.py`
- `npp_rl/agents/hierarchical_ppo.py`
- `npp_rl/agents/hyperparameters/hierarchical_hyperparameters.py` (new)
- `npp_rl/callbacks/hierarchical_callbacks.py` (new)

**Training stability requirements**:

#### Hierarchical Training Coordination
**Learning rate scheduling**:
- High-level policy: Lower learning rate (1e-4) for stable strategic decisions
- Low-level policy: Higher learning rate (3e-4) for responsive action learning
- Adaptive scheduling based on performance metrics

**Update frequency coordination**:
- High-level policy: Update every 50-100 steps
- Low-level policy: Update every step
- Synchronized experience collection with proper temporal credit assignment

**Experience buffer management**:
- Separate buffers for high-level and low-level experiences
- Proper temporal alignment of hierarchical decisions
- Balanced sampling to prevent policy level dominance

#### Hyperparameter Optimization
**PPO-specific parameters for hierarchical training**:
```python
HIERARCHICAL_HYPERPARAMETERS = {
    # High-level policy
    'high_level': {
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 4,
        'clip_range': 0.1,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    },
    
    # Low-level policy
    'low_level': {
        'learning_rate': 3e-4,
        'n_steps': 1024,
        'batch_size': 256,
        'n_epochs': 10,
        'clip_range': 0.2,
        'ent_coef': 0.02,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    },
    
    # ICM parameters
    'icm': {
        'alpha': 0.1,      # Intrinsic reward weight
        'eta': 0.01,       # ICM learning rate
        'lambda_inv': 0.1, # Inverse model loss weight
        'lambda_fwd': 0.9  # Forward model loss weight
    }
}
```

#### Stability Monitoring
**Training metrics to track**:
- Policy gradient norms for both levels
- Value function loss convergence
- Subtask transition frequencies
- Exploration efficiency metrics
- Mine avoidance success rates

**Stability indicators**:
- Consistent improvement in completion rates
- Stable subtask selection patterns
- Balanced exploration vs exploitation
- No catastrophic forgetting between policy levels

#### Adaptive Training Procedures
**Dynamic hyperparameter adjustment**:
```python
class AdaptiveHierarchicalTrainer:
    def __init__(self):
        self.performance_history = []
        self.stability_metrics = {}
    
    def adjust_learning_rates(self, performance_metrics):
        # Reduce learning rates if training becomes unstable
        if self.detect_instability(performance_metrics):
            self.reduce_learning_rates(factor=0.8)
        
        # Increase learning rates if learning stagnates
        elif self.detect_stagnation(performance_metrics):
            self.increase_learning_rates(factor=1.1)
    
    def balance_policy_updates(self, high_level_loss, low_level_loss):
        # Ensure neither policy dominates training
        if high_level_loss > 2 * low_level_loss:
            self.increase_high_level_update_frequency()
        elif low_level_loss > 2 * high_level_loss:
            self.increase_low_level_update_frequency()
```

#### Training Procedures
**Warm-up phase**:
1. Train low-level policy first on simple navigation tasks
2. Gradually introduce high-level policy decisions
3. Full hierarchical training once both policies stable

**Curriculum progression**:
1. Single-switch levels for initial hierarchical coordination
2. Two-switch dependency levels for strategic planning
3. Complex multi-switch levels for full capability testing

**Regularization techniques**:
- Entropy regularization to maintain exploration
- Gradient clipping to prevent instability
- Experience replay balancing between policy levels


## Dependencies and Prerequisites

**Phase 1 completion requirements**:
- Gold collection system removed
- Entity processing simplified
- Basic hierarchical controller implemented
- Completion-focused reward system working

**External dependencies**:
- Stable nclone environment with mine processing
- ICM integration from Phase 1
- Performance monitoring infrastructure

## Risk Mitigation

**Training instability risks**:
- **Mitigation**: Careful hyperparameter tuning and adaptive procedures
- **Fallback**: Reduce to simpler hierarchy if training becomes unstable
- **Monitoring**: Continuous stability metrics tracking

**Hierarchical coordination risks**:
- **Mitigation**: Proper experience buffer management and update scheduling
- **Fallback**: Increase high-level policy update frequency if coordination fails
- **Testing**: Extensive integration testing of policy coordination

**Mine avoidance complexity risks**:
- **Mitigation**: Start with simple mine avoidance, gradually increase complexity
- **Fallback**: Reduce mine avoidance sophistication if training suffers
- **Validation**: Safety testing to ensure mine avoidance works correctly

## Success Criteria for Phase 2

**Primary objectives**:
- [ ] Two-level hierarchical architecture trains stably
- [ ] Subtask-specific rewards improve learning efficiency
- [ ] Mine avoidance integrated successfully
- [ ] Multi-switch levels completed with >60% success rate
- [ ] Training stability maintained over long runs

**Performance targets**:
- [ ] Multi-switch level completion: >60% success rate
- [ ] Training stability: 5000+ episodes without collapse
- [ ] Mine safety: <5% deaths due to mine contact
- [ ] Efficiency: Competitive steps-to-completion vs baseline

**Quality gates**:
- [ ] All hierarchical components tested and validated
- [ ] Training procedures documented and reproducible
- [ ] Performance benchmarks established
- [ ] Code review and documentation completed

This phase establishes sophisticated hierarchical control that can handle complex multi-switch levels while maintaining training stability and safety around mine hazards.