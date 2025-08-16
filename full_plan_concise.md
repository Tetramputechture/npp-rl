# Improvement Plan

## Current Agent Limitations

**Baseline**: PPO agent with CNN processing frame-stacks + velocity input

**Key Problems**:
- **Sparse Rewards**: Single reward at level completion (~20,000 frames)
- **Physics-Awareness Gap**: CNN must implicitly learn physics from pixels
- **Temporal Precision Bottleneck**: Frame-perfect inputs required but hard to learn
- **Exploration Challenges**: Poor in maze-like and non-linear levels  
- **Generalization Issues**: Overfits to simple geometric patterns
- **Untapped Resource**: 100k+ human expert replays unused

## Core Enhancement Strategy

### 1. Enhanced Observation Architecture

**Multi-Modal Fusion (CNN + MLP + GNN)**

```
Input Streams:
├── Visual: Frame-stack (existing CNN)
├── Symbolic: MLP processing:
│   ├── Ninja state: (x,y), velocity, movement_state, input_buffers
│   ├── Entity states: coordinates, types, activation status
│   └── Tile information: types, traversability
└── Structural: GNN processing:
    ├── Nodes: Grid cells + entities with features
    ├── Edges: Physical connectivity + functional relationships
    └── Message passing for spatial reasoning
```

**Implementation**: Concatenate CNN, MLP, and GNN embeddings before policy/value networks

### 2. Hierarchical Reinforcement Learning

**Framework**: ALCS (Automatically Learning to Compose Subtasks) or SHIRO

**Subtasks**:
- `activate_switch`: Navigate to and trigger exit switch
- `collect_gold`: Navigate to and collect gold pieces  
- `reach_exit_door`: Navigate to activated exit
- `navigate_hazard_zone`: Traverse dangerous areas safely
- `perform_wall_jump`: Execute precise wall jump maneuvers
- `utilize_launch_pad`: Use launch pads effectively

**Benefits**: 
- Dense subtask rewards address sparse reward problem
- Low-level policies specialize in precise physical skills
- High-level policy learns task composition

### 3. Advanced Exploration & Reward Shaping

**Intrinsic Motivation**: 
- ICM (Intrinsic Curiosity Module) for novel state exploration
- IEM-PPO for uncertainty-driven exploration of difficult transitions

**Potential-Based Reward Shaping**:
- Distance-to-objective potentials (gold, switch, exit)
- Progress-on-subtasks potentials  
- Hazard-avoidance negative potentials

### 4. Human-Guided Learning

**Phase 1 - Imitation Learning**:
- Behavioral Cloning pre-training on 100k+ human replays
- Extract state-action pairs for supervised learning
- Provides strong behavioral prior

**Phase 2 - RLHF Integration**:
- Train reward model from human replay quality comparisons
- Use learned reward model for denser feedback
- Align agent behavior with human preferences

### 5. Adaptive Training Pipeline

**Curriculum Learning**:
- Automated difficulty metrics: pathfinding cost, hazard count, required maneuvers
- Dynamic difficulty adjustment based on agent performance
- Progressive complexity introduction

**Procedural Content Generation**:
- Conditional GANs generate levels based on difficulty parameters
- Validation via pathfinding algorithms and physics reachability
- Infinite diverse training data

### 6. Hardware Optimization

**Distributed Training**:
- Stable Baselines3 SubprocVecEnv for parallel environments
- Maximize H100 GPU utilization

**Mixed Precision**:
- PyTorch AMP (autocast + GradScaler)
- FP16 operations on Tensor Cores
- Reduced memory, increased throughput

## Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
**Deliverable**: Enhanced observation agent with human data integration

**Tasks**:
1. **Environment Setup**
   - Gym-compatible N++ environment
   - Extract symbolic game state features
   - Process 100k human replay dataset

2. **Multi-Modal Architecture**
   - Implement CNN + MLP fusion
   - Design symbolic feature extraction
   - Integrate PBRS for basic objectives

3. **Hardware Optimization**
   - Setup SubprocVecEnv distributed training
   - Implement PyTorch AMP mixed precision

4. **Human Data Integration**
   - Implement Behavioral Cloning pre-training
   - Create state-action pair extraction pipeline

### Phase 2: Advanced Exploration (Months 4-6) 
**Deliverable**: Agent with improved exploration and structural understanding

**Tasks**:
1. **Intrinsic Motivation**
   - Integrate ICM or IEM-PPO module
   - Modify reward calculation pipeline

2. **Graph Neural Networks**
   - Design N++ graph representation
   - Implement GNN component
   - Integrate with existing multi-modal architecture

3. **Initial Validation**
   - Test on moderately complex levels
   - Measure exploration improvements

### Phase 3: Hierarchical Control (Months 7-9)
**Deliverable**: HRL agent with curriculum learning

**Tasks**:
1. **HRL Implementation**
   - Define N++ subtask structure
   - Implement ALCS or SHIRO framework
   - Design high-level and low-level policies

2. **Curriculum System**
   - Develop automated difficulty metrics
   - Implement adaptive curriculum selection
   - Create level progression pipeline

3. **Procedural Generation**
   - Design conditional GAN architecture
   - Train on existing N++ levels
   - Implement validation pipeline

4. **RLHF Integration**
   - Train reward model from human data
   - Replace/augment sparse environmental rewards

### Phase 4: Full Integration (Months 10-12)
**Deliverable**: Production-ready robust N++ agent

**Tasks**:
1. **System Integration**
   - Combine all components into unified training pipeline
   - Dynamic PCG-curriculum feedback loop

2. **Optimization**
   - Hyperparameter tuning across all components
   - Performance profiling and optimization

3. **Evaluation**
   - Comprehensive testing on diverse level types
   - Ablation studies to measure component contributions
   - Benchmark against baseline agents

## Expected Outcomes

**Performance Gains**:
- **Robustness**: Handle complex maze-like levels and non-linear paths
- **Training Efficiency**: 4x faster initial learning from IL, reduced sample complexity
- **Generalization**: Better performance on unseen level types and geometries
- **Hardware Utilization**: Maximized H100 GPU throughput

**Technical Deliverables**:
- Multi-modal observation architecture
- HRL framework with learned subtasks  
- Adaptive curriculum + PCG pipeline
- Human-guided learning integration
- Hardware-optimized training system

## Key Implementation Notes

**Critical Dependencies**:
- Stable Baselines3 for PPO + distributed training
- PyTorch for neural networks + mixed precision
- Human replay data preprocessing pipeline
- N++ physics simulation access for symbolic features

**Risk Mitigation**:
- Phased approach allows validation at each stage
- Fallback to simpler approaches if complex methods fail
- Modular design enables component-wise debugging

**Success Metrics**:
- Level completion rate across difficulty spectrum
- Training sample efficiency (episodes to competency)
- Generalization to procedurally generated levels
- Hardware utilization (GPU usage %, training time)