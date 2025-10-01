# Phase 1: Completion-Focused Foundation

**Timeline**: 2-3 weeks  
**Priority**: Critical  
**Dependencies**: None  

## Overview

This phase establishes the foundation for completion-focused NPP-RL training by simplifying the system to focus solely on level completion (switch activation → exit door) without gold collection. The goal is to remove unnecessary complexity and integrate the existing completion planner with the RL training pipeline.

## Objectives

1. Remove all gold collection functionality from the system
2. Simplify entity processing to handle only mines, switches, and exits
3. Implement completion-focused reward system
4. Integrate the existing completion planner with PPO training
5. Validate end-to-end training pipeline for single-switch levels

## Task Breakdown

### Task 1.1: Remove Gold Collection System (2-3 days)

**What we want to do**: Completely eliminate gold collection from observations, rewards, and feature extraction to focus solely on level completion.

**Current state**: 
- Gold collection rewards in `nclone/gym_environment/reward_calculation/main_reward_calculator.py`
- Gold-related observations in `nclone/gym_environment/observation_processor.py`
- Gold features in `npp_rl/feature_extractors/hgt_multimodal.py`
- Gold entity processing in entity extraction pipeline

**Files to modify**:
- `nclone/gym_environment/reward_calculation/main_reward_calculator.py`
- `nclone/gym_environment/reward_calculation/exploration_reward_calculator.py`
- `nclone/gym_environment/observation_processor.py`
- `nclone/gym_environment/entity_extractor.py`
- `npp_rl/feature_extractors/hgt_multimodal.py`
- `nclone/gym_environment/constants.py` (remove gold-related constants)

**Specific changes needed**:
1. Remove gold collection rewards from reward calculator
2. Remove gold-related observations from state representation
3. Remove gold entity processing from entity extractor
4. Remove gold-related features from multimodal feature extractor
5. Update observation space definitions to exclude gold data
6. Remove gold-related constants and configuration

**Acceptance criteria**:
- [ ] No gold-related rewards in reward calculation
- [ ] No gold observations in environment state
- [ ] No gold features in neural network inputs
- [ ] Environment runs without gold-related errors
- [ ] Observation space dimensions reduced appropriately
- [ ] All tests pass with gold functionality removed

**Testing requirements**:
- Unit tests for reward calculator without gold
- Integration tests for environment without gold observations
- Feature extractor tests with simplified inputs

---

### Task 1.2: Simplify Entity Processing (3-4 days)

**What we want to do**: Reduce entity complexity to handle only mines, switches, and exits, removing thwump and drone processing entirely.

**Current state**:
- Complex entity type system in `npp_rl/models/entity_type_system.py` with 6 node types
- Entity extraction handles thwumps, drones, and other complex entities
- Graph processing includes complex entity relationships
- HGT model processes 6 node types with specialized attention

**Files to modify**:
- `npp_rl/models/entity_type_system.py`
- `nclone/gym_environment/entity_extractor.py`
- `npp_rl/models/hgt_config.py`
- `npp_rl/models/hgt_gnn.py`
- `nclone/gym_environment/observation_processor.py`

**Specific changes needed**:
1. Reduce entity types to 4: tile, ninja, mine, objective (switch/exit)
2. Remove thwump and drone processing from entity extractor
3. Simplify HGT configuration to handle 4 node types
4. Update entity type embeddings and attention mechanisms
5. Remove complex entity state tracking (thwump states, drone modes)
6. Simplify graph edge types to 2: adjacent, reachable

**Current entity types (to remove)**:
- Thwumps (Type 20) - complex movement states and deadly face logic
- Drones (Types 14, 26) - patrol patterns and movement modes
- Death balls (Type 25) - seeking behavior
- Complex collectibles beyond basic switches

**Target entity types (to keep)**:
- Tiles (collision geometry)
- Ninja (player character)
- Mines (toggle mines and regular mines only)
- Objectives (switches and exit doors)

**Acceptance criteria**:
- [ ] Entity type system reduced to 4 types
- [ ] No thwump or drone processing in entity extraction
- [ ] HGT model handles simplified entity types
- [ ] Graph construction uses simplified node/edge types
- [ ] Entity observations contain only relevant entities
- [ ] Performance improvement in feature extraction time

**Testing requirements**:
- Entity extraction tests with simplified types
- HGT model tests with 4 node types
- Graph construction tests with simplified entities
- Performance benchmarks showing improved inference time

---

### Task 1.3: Implement Simplified Reward System (1-2 days)

**What we want to do**: Create a completion-focused reward system that provides clear signals for switch activation and exit reaching.

**Current state**:
- Complex reward system in `nclone/gym_environment/reward_calculation/main_reward_calculator.py`
- PBRS potentials include gold collection and complex hazard avoidance
- Navigation rewards include gold-seeking behavior
- Exploration rewards consider gold discovery

**Files to modify**:
- `nclone/gym_environment/reward_calculation/main_reward_calculator.py`
- `nclone/gym_environment/reward_calculation/pbrs_potentials.py`
- `nclone/gym_environment/reward_calculation/navigation_reward_calculator.py`

**Specific changes needed**:
1. Implement completion-focused reward structure:
   - Switch activation: +0.1 immediate reward
   - Exit door reached: +1.0 terminal reward
   - Time penalty: -0.01 per step
   - Death penalty: -0.5
2. Remove gold-related PBRS potentials
3. Focus PBRS on switch/exit distance only
4. Remove gold-seeking navigation rewards
5. Simplify exploration rewards to focus on switch/exit discovery

**Target reward structure**:
```python
# Terminal rewards
SWITCH_ACTIVATION_REWARD = 0.1
EXIT_COMPLETION_REWARD = 1.0
DEATH_PENALTY = -0.5

# Step-based rewards
TIME_PENALTY = -0.01  # Encourage efficiency
PBRS_SWITCH_DISTANCE = 0.05  # Distance-based shaping to switches
PBRS_EXIT_DISTANCE = 0.05    # Distance-based shaping to exit
```

**Acceptance criteria**:
- [ ] Switch activation provides +0.1 reward
- [ ] Exit completion provides +1.0 reward
- [ ] Time penalty of -0.01 per step implemented
- [ ] PBRS focuses only on switch/exit objectives
- [ ] No gold-related rewards remain
- [ ] Reward signals are clear and consistent

**Testing requirements**:
- Unit tests for each reward component
- Integration tests showing correct reward calculation
- Validation that rewards align with completion objectives

---

### Task 1.4: Integrate Completion Planner with RL Training (1-2 weeks)

**What we want to do**: Create a hierarchical RL system that uses the existing completion planner to generate subgoals for PPO training.

**Current state**:
- `nclone/planning/completion_planner.py` implements the exact completion heuristic
- `LevelCompletionPlanner` provides strategic planning but not integrated with RL
- PPO training in `ppo_train.py` and `npp_rl/agents/training.py` is flat (non-hierarchical)
- No mechanism to use completion planner output in RL training

**Files to create/modify**:
- `npp_rl/hrl/completion_controller.py` (new)
- `npp_rl/hrl/__init__.py` (new)
- `npp_rl/agents/training.py`
- `ppo_train.py`
- `npp_rl/agents/hierarchical_ppo.py` (new)

**Specific implementation needed**:

1. **Create Hierarchical Controller** (`npp_rl/hrl/completion_controller.py`):
   ```python
   class CompletionController:
       """Hierarchical controller using completion planner for subgoal generation."""
       
       def __init__(self, completion_planner):
           self.completion_planner = completion_planner
           self.current_subtask = None
           self.subtask_start_time = 0
       
       def get_current_subtask(self, obs, info):
           """Use completion planner to determine current subtask."""
           # Implementation using completion planner heuristic
           
       def should_switch_subtask(self, obs, info):
           """Determine if subtask should change based on completion planner."""
           # Implementation of switching logic
   ```

2. **Create Hierarchical PPO** (`npp_rl/agents/hierarchical_ppo.py`):
   - High-level policy: Subtask selection based on reachability features
   - Low-level policy: Action execution for current subtask
   - Shared feature extractor with separate policy heads

3. **Integration with Training Loop**:
   - Modify training to use hierarchical controller
   - Implement subtask-specific reward shaping
   - Add subtask transition logging and metrics

**Subtasks to implement**:
1. `navigate_to_exit_switch` - Move to and activate exit switch
2. `navigate_to_locked_switch` - Move to and activate locked door switch
3. `navigate_to_exit_door` - Move to exit door after switch activation
4. `avoid_mine` - Navigate around mine hazards

**Architecture design**:
- **High-level policy input**: 8D reachability features, switch states
- **High-level policy output**: Subtask selection (4 discrete actions)
- **Low-level policy input**: Full multimodal observations + current subtask
- **Low-level policy output**: Movement actions (6 discrete actions)

**Acceptance criteria**:
- [ ] Hierarchical controller successfully integrates completion planner
- [ ] High-level policy selects appropriate subtasks based on reachability
- [ ] Low-level policy executes actions for current subtask
- [ ] Training loop handles hierarchical architecture
- [ ] Subtask transitions are logged and trackable
- [ ] System successfully completes single-switch levels

**Testing requirements**:
- Unit tests for completion controller logic
- Integration tests for hierarchical PPO training
- End-to-end tests on simple levels
- Performance tests showing stable training

---

### Task 1.5: Testing and Validation (2-3 days)

**What we want to do**: Validate the complete Phase 1 system with comprehensive testing and performance evaluation.

**Current state**: 
- Individual components modified but not tested as integrated system
- No comprehensive test suite for completion-focused functionality
- No performance benchmarks for simplified system

**Testing requirements**:

1. **Unit Testing**:
   - Reward calculator tests without gold functionality
   - Entity extraction tests with simplified types
   - Feature extractor tests with reduced complexity
   - Completion controller tests with mock environments

2. **Integration Testing**:
   - End-to-end training pipeline validation
   - Hierarchical PPO training stability
   - Environment-agent interaction correctness
   - Observation space consistency

3. **Performance Testing**:
   - Feature extraction time < 10ms target
   - Training stability over 1000 episodes
   - Memory usage within acceptable bounds
   - GPU utilization efficiency

4. **Functional Testing**:
   - Single-switch level completion success
   - Proper subtask transitions
   - Reward signal correctness
   - Mine avoidance behavior

**Files to create/modify**:
- `tests/test_completion_focused_training.py` (new)
- `tests/test_hierarchical_controller.py` (new)
- `tests/test_simplified_rewards.py` (new)
- `tests/integration/test_phase1_pipeline.py` (new)

**Acceptance criteria**:
- [ ] All unit tests pass
- [ ] Integration tests demonstrate stable training
- [ ] Performance benchmarks meet targets
- [ ] Single-switch levels completed with >80% success rate
- [ ] No crashes or errors in 1000-episode training run
- [ ] Feature extraction time consistently < 10ms
- [ ] Memory usage stable over long training runs

**Success metrics**:
- **Completion rate**: >80% on single-switch levels
- **Training stability**: Consistent learning curves without crashes
- **Performance**: <10ms feature extraction, stable memory usage
- **Functionality**: Proper subtask selection and execution

## Dependencies and Prerequisites

**External dependencies**:
- nclone environment with completion planner
- Existing PPO training infrastructure
- HGT multimodal feature extractor

**Internal dependencies**:
- Tasks must be completed in order (1.1 → 1.2 → 1.3 → 1.4 → 1.5)
- Task 1.4 depends on completion of simplification tasks
- Task 1.5 requires all previous tasks completed

## Risk Mitigation

**Technical risks**:
- Hierarchical training instability: Start with simple two-level hierarchy
- Integration complexity: Thorough testing at each step
- Performance degradation: Continuous benchmarking

**Mitigation strategies**:
- Incremental development with testing at each step
- Fallback to simpler architectures if hierarchical training fails
- Performance monitoring throughout development

## Success Criteria for Phase 1

**Primary objectives**:
- [ ] Gold collection completely removed from system
- [ ] Entity processing simplified to 4 types
- [ ] Completion-focused reward system implemented
- [ ] Hierarchical controller integrated with training
- [ ] Single-switch levels completed with >80% success rate

**Performance targets**:
- [ ] Feature extraction time < 10ms
- [ ] Training runs stable for >1000 episodes
- [ ] Memory usage remains constant over long runs
- [ ] No system crashes during training

**Quality gates**:
- [ ] All tests pass
- [ ] Code review completed
- [ ] Performance benchmarks met
- [ ] Documentation updated

This phase establishes the foundation for all subsequent development by creating a stable, completion-focused system that can be enhanced with additional capabilities in later phases.