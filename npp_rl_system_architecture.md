graph TD
    %% ===== DATA SOURCES & PHYSICS =====
    subgraph "nclone Physics & Environment"
        NCLONE["🎮 nclone NppEnvironment<br/>- Physics simulation<br/>- Level rendering<br/>- Game state management<br/>- N++ mechanics"]
        PHYSICS_CONST["⚙️ nclone.constants<br/>- NINJA_RADIUS<br/>- GRAVITY_FALL/JUMP<br/>- MAX_HOR_SPEED<br/>- JUMP_FLAT_GROUND_Y"]
        ENTITY_TYPES["🏷️ EntityType Constants<br/>- NINJA, TOGGLE_MINE<br/>- GOLD, EXIT_DOOR<br/>- LOCKED_DOOR, THWUMP<br/>- DEATH_BALL, etc."]
        GRAPH_BUILDER["🕸️ HierarchicalGraphBuilder<br/>- Multi-resolution graphs<br/>- Sub-cell (6px)<br/>- Tile (24px)<br/>- Region (96px)"]
    end

    %% ===== ENVIRONMENT WRAPPERS =====
    subgraph "Environment Wrappers Layer"
        DYN_GRAPH["📊 DynamicGraphWrapper<br/>- Graph observations from nclone<br/>- Switch/door state tracking<br/>- Sub-millisecond performance<br/>- N_MAX_NODES=18000<br/>- E_MAX_EDGES=144000"]
        REACH_WRAP["🎯 ReachabilityWrapper<br/>- ReachabilitySystem<br/>- CompactReachabilityFeatures[64]<br/>- Performance target: fast<br/>- Cache TTL: 100ms"]
        VEC_WRAP["⚡ VectorizationWrapper<br/>- Parallel environment processing<br/>- SubprocVecEnv/DummyVecEnv<br/>- VecMonitor/VecCheckNan"]
        ICM_WRAP["🎁 IntrinsicRewardWrapper<br/>- ICMTrainer integration<br/>- Reward combination (α=0.1)<br/>- Experience buffer<br/>- Intrinsic reward clipping"]
    end

    %% ===== MULTIMODAL OBSERVATIONS =====
    subgraph "Multimodal Observation Space"
        TEMPORAL["🎬 Temporal Frames<br/>player_frames [84,84,12]<br/>- 12-frame temporal stack<br/>- Player-centric view<br/>- Grayscale normalized<br/>- Movement patterns"]
        SPATIAL["🗺️ Global View<br/>global_view [176,100,1]<br/>- Downsampled level view<br/>- Full level context<br/>- Spatial relationships<br/>- Strategic overview"]
        GAME_STATE["📊 Game State Vector[16]<br/>- Ninja physics state<br/>- Position, velocity<br/>- Airborne/walled status<br/>- Jump duration<br/>- Applied forces<br/>- Time remaining"]
        REACH_FEAT["🎯 Reachability Features[64]<br/>- CompactReachabilityFeatures<br/>- Multi-tier accessibility<br/>- Frontier detection<br/>- Strategic navigation<br/>- Exploration tracking"]
        GRAPH_OBS["🕸️ Graph Observations<br/>- node_feats: Entity features<br/>- edge_feats: Connectivity<br/>- edge_index: Graph structure<br/>- node_types: 6 types<br/>- edge_types: 3 types<br/>- node/edge_masks: Valid elements"]
    end

    %% ===== HGT MULTIMODAL EXTRACTOR =====
    subgraph "HGTMultimodalExtractor (Primary Feature Processor)"
        CNN3D["🎦 TemporalCNN3D<br/>- Conv3D layers (1→32→64→128)<br/>- Kernel: (4,7,7)→(3,5,5)→(2,3,3)<br/>- BatchNorm + ReLU + Dropout<br/>- Adaptive pooling (1,4,4)<br/>- Output: 512D temporal features"]
        
        CNN2D["🖼️ SpatialCNN2D<br/>- Conv2D layers (1→32→64→128)<br/>- Kernels: 7×7→5×5→3×3<br/>- SpatialAttentionModule<br/>- Adaptive pooling<br/>- Output: 256D spatial features"]
        
        HGT_PROC["🧠 HGT Graph Processor<br/>- ProductionHGTConfig<br/>- Node features: 8D→128D hidden<br/>- Edge features: 4D<br/>- 3 layers, 8 attention heads<br/>- 6 node types, 3 edge types<br/>- Type-specific attention<br/>- Output: 256D graph features"]
        
        STATE_PROC["📊 State Processor<br/>- MLP: 16→128→128<br/>- ReLU activation<br/>- Dropout regularization<br/>- Output: 128D state features"]
        
        FUSION["🔗 CrossModalFusion<br/>- Cross-modal attention mechanisms<br/>- Layer normalization<br/>- Residual connections<br/>- Temporal-Spatial attention<br/>- Graph-Visual attention<br/>- Fusion network<br/>- Output: 512D fused features"]
    end

    %% ===== NEURAL NETWORK COMPONENTS =====
    subgraph "Neural Network Components"
        HGT_FACTORY["🏭 HGTFactory<br/>- ProductionHGTConfig<br/>- HGTLayer creation<br/>- HGTEncoder creation<br/>- MultimodalHGTSystem<br/>- PyTorch Geometric integration"]
        
        HGT_LAYERS["🧩 HGT Core Components<br/>- HGTLayer: Type-specific attention<br/>- HGTEncoder: Multi-layer processing<br/>- TypeSpecificAttention<br/>- HazardAwareAttention<br/>- CrossModalAttention"]
        
        ENTITY_SYSTEM["🏷️ EntityTypeSystem<br/>- 6 node types:<br/>  • tile, ninja, hazard<br/>  • collectible, switch, exit<br/>- Specialized embeddings<br/>- Hazard-aware processing"]
        
        ATTENTION_MECH["👁️ Attention Mechanisms<br/>- Multi-head attention (8 heads)<br/>- Type-aware processing<br/>- Edge-type specialization<br/>- Spatial attention integration"]
    end

    %% ===== REINFORCEMENT LEARNING =====
    subgraph "PPO Reinforcement Learning"
        PPO_POLICY["🎯 PPO Policy Network<br/>- MultiInputPolicy<br/>- MLP layers [256,256,128]<br/>- Action space: 6 discrete<br/>  • NOOP, Left, Right<br/>  • Jump, Jump+Left, Jump+Right<br/>- Softmax action distribution"]
        
        PPO_VALUE["💎 PPO Value Network<br/>- Critic network<br/>- Value function approximation<br/>- Advantage estimation<br/>- GAE (λ=0.95)<br/>- Shared feature extractor"]
    end

    %% ===== INTRINSIC CURIOSITY MODULE =====
    subgraph "Intrinsic Curiosity & Exploration"
        ICM_NET["🔍 ICMNetwork<br/>- Forward model: (state,action)→next_state<br/>- Inverse model: (state,next_state)→action<br/>- Prediction error → intrinsic reward<br/>- Feature dim: 512D<br/>- Hidden dim: 256D"]
        
        REACH_EXPLORE["🎯 ReachabilityAwareExploration<br/>- Integration with nclone systems<br/>- ReachabilitySystem<br/>- FrontierDetector<br/>- ExplorationRewardCalculator<br/>- Spatial modulation"]
        
        ADAPT_EXPLORE["🧭 AdaptiveExplorationManager<br/>- Hierarchical subgoal generation<br/>- EntityInteractionSubgoal<br/>- LevelCompletionPlanner<br/>- Strategic planning<br/>- Performance caching (<3ms)"]
    end

    %% ===== TRAINING & LOGGING =====
    subgraph "Training Pipeline"
        TRAINING["🚀 Training Script<br/>- Main: npp_rl.agents.training<br/>- CLI interface<br/>- Hyperparameter management<br/>- Model checkpointing<br/>- Multi-environment support"]
        
        CALLBACKS["📊 Training Callbacks<br/>- HierarchicalLoggingCallback<br/>- EvalCallback<br/>- StopTrainingOnNoModelImprovement<br/>- Exploration statistics<br/>- Auxiliary loss tracking"]
        
        HYPERPARAM["⚙️ Hyperparameters<br/>- n_steps: 1024<br/>- batch_size: 256<br/>- γ (gamma): 0.999<br/>- learning_rate: 3e-4→1e-6<br/>- clip_range, ent_coef, vf_coef"]
        
        LOGGING["📈 Monitoring & Logging<br/>- TensorBoard integration<br/>- Performance metrics<br/>- Reward tracking<br/>- Model evaluation<br/>- Session management"]
    end

    %% ===== REWARD SYSTEM =====
    subgraph "Reward System"
        EXT_REWARDS["🎁 Extrinsic Rewards<br/>- Time step penalty<br/>- Switch activation bonus<br/>- Exit completion bonus<br/>- Death penalty<br/>- Multi-scale exploration"]
        
        INT_REWARDS["🔍 Intrinsic Rewards<br/>- ICM prediction error<br/>- Novelty detection<br/>- Reachability modulation<br/>- Frontier boost factors<br/>- Strategic weighting"]
        
        REWARD_COMBINE["⚖️ Reward Combination<br/>- RewardCombiner (α=0.1)<br/>- Intrinsic reward clipping<br/>- Adaptive scaling<br/>- Performance optimization"]
    end

    %% ===== DATA FLOW CONNECTIONS =====
    
    %% Environment to Wrappers
    NCLONE --> DYN_GRAPH
    NCLONE --> REACH_WRAP
    GRAPH_BUILDER --> DYN_GRAPH
    PHYSICS_CONST --> NCLONE
    ENTITY_TYPES --> NCLONE
    
    %% Wrapper Chain
    DYN_GRAPH --> REACH_WRAP
    REACH_WRAP --> VEC_WRAP
    VEC_WRAP --> ICM_WRAP
    
    %% Observation Generation
    DYN_GRAPH --> TEMPORAL
    DYN_GRAPH --> SPATIAL
    DYN_GRAPH --> GAME_STATE
    REACH_WRAP --> REACH_FEAT
    DYN_GRAPH --> GRAPH_OBS
    
    %% Feature Processing
    TEMPORAL --> CNN3D
    SPATIAL --> CNN2D
    GRAPH_OBS --> HGT_PROC
    GAME_STATE --> STATE_PROC
    REACH_FEAT --> STATE_PROC
    
    %% HGT Factory Integration
    HGT_FACTORY --> HGT_PROC
    HGT_LAYERS --> HGT_PROC
    ENTITY_SYSTEM --> HGT_PROC
    ATTENTION_MECH --> HGT_PROC
    
    %% Multimodal Fusion
    CNN3D --> FUSION
    CNN2D --> FUSION
    HGT_PROC --> FUSION
    STATE_PROC --> FUSION
    
    %% RL Processing
    FUSION --> PPO_POLICY
    FUSION --> PPO_VALUE
    FUSION --> ICM_NET
    
    %% ICM Integration
    ICM_NET --> REACH_EXPLORE
    REACH_EXPLORE --> ADAPT_EXPLORE
    ADAPT_EXPLORE --> ICM_WRAP
    
    %% Reward Processing
    NCLONE --> EXT_REWARDS
    ICM_NET --> INT_REWARDS
    EXT_REWARDS --> REWARD_COMBINE
    INT_REWARDS --> REWARD_COMBINE
    REWARD_COMBINE --> ICM_WRAP
    
    %% Training Integration
    PPO_POLICY --> TRAINING
    PPO_VALUE --> TRAINING
    ICM_WRAP --> TRAINING
    CALLBACKS --> TRAINING
    HYPERPARAM --> TRAINING
    TRAINING --> LOGGING

    %% ===== PERFORMANCE ANNOTATIONS =====
    classDef performance fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef neural fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef data fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef processing fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class CNN3D,CNN2D,HGT_PROC,STATE_PROC,FUSION neural
    class TEMPORAL,SPATIAL,GAME_STATE,REACH_FEAT,GRAPH_OBS data
    class DYN_GRAPH,REACH_WRAP,VEC_WRAP,ICM_WRAP processing
    class TRAINING,CALLBACKS,LOGGING performance
