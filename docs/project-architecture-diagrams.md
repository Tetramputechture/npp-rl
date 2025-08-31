# NPP-RL Project Architecture: Complete Mermaid Diagram Guide

This document provides comprehensive Mermaid diagram instructions for visualizing the entire NPP-RL project architecture. The diagrams progress from a global system overview to detailed component breakdowns, showcasing the consolidated HGT-based multimodal architecture.

## Table of Contents

1. [Global System Architecture](#1-global-system-architecture)
2. [NPP-RL Agent Architecture](#2-npp-rl-agent-architecture)
3. [HGT Multimodal Feature Extractor (PRIMARY)](#3-hgt-multimodal-feature-extractor-primary)
4. [Hierarchical Feature Extractor (SECONDARY)](#4-hierarchical-feature-extractor-secondary)
5. [NClone Environment System](#5-nclone-environment-system)
6. [Graph Processing Pipeline](#6-graph-processing-pipeline)
7. [Training and Exploration Systems](#7-training-and-exploration-systems)
8. [Data Flow and Integration](#8-data-flow-and-integration)

---

## 1. Global System Architecture

This diagram shows the complete NPP-RL project ecosystem, including both repositories and their consolidated architectures.

```mermaid
graph TB
    %% Global System Overview
    subgraph ECOSYSTEM ["🌐 NPP-RL Project Ecosystem"]
        subgraph NPP_RL_REPO ["🧠 NPP-RL Repository"]
            subgraph AGENTS ["🎯 Agent Components"]
                TRAINING["training.py<br/>🌟 PRIMARY TRAINING<br/>HGT + Hierarchical Support"]
                EXPLORATION_MGR["adaptive_exploration.py<br/>ICM + Novelty Detection"]
                HYPERPARAMS["hyperparameters/<br/>Optimized PPO Config"]
            end
            
            subgraph FEATURE_EXTRACTORS ["🔍 Feature Extractors"]
                HGT_EXTRACTOR["hgt_multimodal.py<br/>🌟 PRIMARY EXTRACTOR<br/>Heterogeneous Graph Transformers<br/>Type-specific Attention"]
                HIERARCHICAL_EXTRACTOR["hierarchical_multimodal.py<br/>⭐ SECONDARY EXTRACTOR<br/>Multi-resolution GNNs<br/>DiffPool Architecture"]
            end
            
            subgraph MODELS ["🧮 Neural Network Models"]
                HGT_GNN["hgt_gnn.py<br/>HGT Implementation<br/>469 lines of advanced GNN"]
                SPATIAL_ATTN["spatial_attention.py<br/>Cross-modal Attention"]
                GNN_MODELS["gnn.py<br/>GraphSAGE + Utilities"]
            end
            
            subgraph ARCHIVE_NPP ["📦 Archive (Deprecated)"]
                LEGACY_EXTRACTORS["temporal.py, multimodal.py<br/>Legacy Feature Extractors"]
                LEGACY_TRAINING["training.py, npp_agent_ppo.py<br/>Legacy Training Scripts"]
            end
        end
        
        subgraph NCLONE_REPO ["🎮 NClone Repository"]
            subgraph SIMULATION_CORE ["⚙️ Core Simulation"]
                NSIM["nsim.py<br/>Physics Engine<br/>Collision Detection"]
                NINJA_LOGIC["ninja.py<br/>Player Physics<br/>Movement Mechanics"]
                ENTITIES_SYS["entities.py<br/>Game Objects<br/>Interactive Elements"]
                RENDERER["Rendering Pipeline<br/>Visual Output"]
            end
            
            subgraph RL_ENVIRONMENTS ["🌍 RL Environment Interface"]
                BASE_ENV["base_environment.py<br/>Gym Interface<br/>Observation Processing"]
                SPECIFIC_ENVS["basic_level_no_gold/<br/>Concrete Environments<br/>Level Management"]
                OBS_PROCESSOR["observation_processor.py<br/>Multi-modal State Processing"]
            end
            
            subgraph GRAPH_SYSTEM ["📊 Graph Processing System"]
                HIERARCHICAL_BUILDER["hierarchical_builder.py<br/>🌟 PRIMARY BUILDER<br/>Multi-resolution Graphs"]
                COMMON_COMPONENTS["common.py<br/>Shared Graph Components<br/>GraphData, NodeType, EdgeType"]
                GRAPH_OBSERVATION["graph_observation.py<br/>Graph State Integration"]
            end
            
            subgraph ARCHIVE_NCLONE ["📦 Archive (Deprecated)"]
                LEGACY_GRAPH["graph_builder.py<br/>Legacy Standard Builder"]
                LEGACY_PATHFINDING["pathfinding/<br/>Legacy A* Navigation<br/>Surface Parsing"]
            end
        end
    end
    
    %% Cross-Repository Connections
    NCLONE_REPO -.->|"Environment Interface"| NPP_RL_REPO
    BASE_ENV --> TRAINING
    HIERARCHICAL_BUILDER --> HGT_EXTRACTOR
    HIERARCHICAL_BUILDER --> HIERARCHICAL_EXTRACTOR
    GRAPH_OBSERVATION --> HGT_EXTRACTOR
    HGT_GNN --> HGT_EXTRACTOR
    SPATIAL_ATTN --> HGT_EXTRACTOR
    EXPLORATION_MGR --> TRAINING
    
    %% Architecture Hierarchy
    subgraph ARCHITECTURE_LEVELS ["🏗️ Architecture Hierarchy"]
        PRIMARY_ARCH["PRIMARY: HGT-based<br/>Type-specific Attention"]
        SECONDARY_ARCH["SECONDARY: Hierarchical<br/>Multi-resolution Processing<br/>DiffPool GNNs"]
        ARCHIVED_ARCH["ARCHIVED: Legacy<br/>Basic CNN/MLP<br/>Temporal Processing"]
    end
    
    HGT_EXTRACTOR -.-> PRIMARY_ARCH
    HIERARCHICAL_EXTRACTOR -.-> SECONDARY_ARCH
    LEGACY_EXTRACTORS -.-> ARCHIVED_ARCH
    
    %% Styling
    classDef primary fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef secondary fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef archive fill:#9E9E9E,stroke:#616161,stroke-width:2px,color:#fff
    classDef system fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    
    class HGT_EXTRACTOR,TRAINING,HIERARCHICAL_BUILDER,PRIMARY_ARCH primary
    class HIERARCHICAL_EXTRACTOR,SECONDARY_ARCH secondary
    class ARCHIVE_NPP,ARCHIVE_NCLONE,LEGACY_EXTRACTORS,LEGACY_TRAINING,LEGACY_GRAPH,LEGACY_PATHFINDING,ARCHIVED_ARCH archive
    class SIMULATION_CORE,RL_ENVIRONMENTS,GRAPH_SYSTEM system
```

---

## 2. NPP-RL Agent Architecture

This diagram focuses on the complete RL agent architecture, showing the data flow from environment observations through feature extraction to policy decisions.

```mermaid
graph TB
    subgraph ENVIRONMENT ["🎮 NClone Environment Interface"]
        ENV_STATE["Environment State<br/>Physics + Entities + Level"]
        REWARD["Reward Signal<br/>Goal Achievement<br/>Survival Bonus"]
        DONE["Episode Termination<br/>Success/Failure/Timeout"]
    end
    
    subgraph OBSERVATIONS ["📊 Multi-modal Observations"]
        VISUAL_FRAMES["Visual Frames<br/>84x84x12 Stack<br/>Player-centric View<br/>Temporal Context"]
        GLOBAL_VIEW["Global View<br/>176x100 Downsampled<br/>Full Level Overview<br/>Strategic Context"]
        GAME_STATE["Game State Vector<br/>Physics State<br/>Entity Status<br/>Objective Progress"]
        GRAPH_REPR["Heterogeneous Graph<br/>Node Types: Grid/Entity/Hazard<br/>Edge Types: Movement/Functional<br/>Structural Representation"]
    end
    
    subgraph FEATURE_EXTRACTION ["🔍 Feature Extraction (Configurable)"]
        subgraph HGT_PRIMARY ["🌟 HGT Multimodal Extractor (PRIMARY)"]
            subgraph HGT_VISUAL ["Visual Processing"]
                HGT_CNN_3D["3D CNN Branch<br/>Temporal Modeling<br/>Spatiotemporal Features"]
                HGT_CNN_2D["2D CNN Branch<br/>Global Context<br/>Spatial Features"]
            end
            
            subgraph HGT_GRAPH ["HGT Graph Processing"]
                HGT_ENCODER["Heterogeneous Graph Transformer<br/>Type-specific Attention<br/>Multi-head Processing<br/>Entity-aware Embeddings"]
                NODE_TYPE_PROC["Node Type Processing<br/>Grid Cells, Entities<br/>Hazards, Switches"]
                EDGE_TYPE_PROC["Edge Type Processing<br/>Movement: walk/jump/fall<br/>Functional: activate/trigger"]
            end
            
            subgraph HGT_FUSION ["Advanced Multimodal Fusion"]
                CROSS_MODAL_ATTN["Cross-modal Attention<br/>Spatial Awareness<br/>Type-aware Integration"]
                HGT_FUSION_LAYER["Feature Fusion<br/>HGT-enhanced Integration"]
            end
        end
        
        subgraph HIERARCHICAL_SECONDARY ["⭐ Hierarchical Extractor (SECONDARY)"]
            subgraph HIER_VISUAL ["Visual Processing"]
                HIER_CNN_3D["3D CNN Branch<br/>Temporal Processing"]
                HIER_CNN_2D["2D CNN Branch<br/>Global Processing"]
            end
            
            subgraph HIER_GRAPH ["Multi-resolution Graph Processing"]
                GNN_SUBCELL["Sub-cell GNN<br/>6px Resolution<br/>Fine Details"]
                GNN_TILE["Tile GNN<br/>24px Resolution<br/>Navigation"]
                GNN_REGION["Region GNN<br/>96px Resolution<br/>Strategy"]
                DIFFPOOL["DiffPool Layers<br/>Hierarchical Pooling<br/>Learnable Coarsening"]
            end
            
            subgraph HIER_FUSION ["Context-aware Fusion"]
                HIER_ATTENTION["Physics-adaptive Attention<br/>Dynamic Weighting"]
                HIER_FUSION_LAYER["Multi-scale Integration"]
            end
        end
        
        subgraph STATE_PROCESSING ["State Processing (Shared)"]
            MLP_STATE["MLP Branch<br/>Game State Processing<br/>Physics Features"]
        end
    end
    
    subgraph PPO_AGENT ["🧠 PPO Agent"]
        POLICY_HEAD["Policy Head (Actor)<br/>Action Probabilities<br/>Discrete Action Space<br/>Movement Commands"]
        VALUE_HEAD["Value Head (Critic)<br/>State Value Estimation<br/>Advantage Calculation<br/>Return Prediction"]
    end
    
    subgraph EXPLORATION ["🔍 Adaptive Exploration System"]
        ICM["Intrinsic Curiosity Module<br/>Forward/Inverse Models<br/>Prediction Error Reward"]
        NOVELTY["Novelty Detection<br/>Count-based Exploration<br/>State Visit Tracking"]
        ADAPTIVE_SCALE["Adaptive Scaling<br/>Dynamic Bonus Adjustment<br/>Exploration Decay"]
    end
    
    subgraph TRAINING ["🎯 Training Components"]
        EXPERIENCE_BUFFER["Experience Buffer<br/>Trajectory Collection<br/>GAE Computation<br/>Advantage Estimation"]
        PPO_LOSS["PPO Loss Function<br/>Policy + Value + Entropy<br/>Clipped Objective<br/>KL Divergence"]
        OPTIMIZER["Adam Optimizer<br/>Learning Rate Scheduling<br/>Gradient Clipping<br/>Weight Updates"]
    end
    
    %% Data Flow
    ENV_STATE --> OBSERVATIONS
    VISUAL_FRAMES --> HGT_CNN_3D
    VISUAL_FRAMES --> HIER_CNN_3D
    GLOBAL_VIEW --> HGT_CNN_2D
    GLOBAL_VIEW --> HIER_CNN_2D
    GAME_STATE --> MLP_STATE
    GRAPH_REPR --> HGT_ENCODER
    GRAPH_REPR --> GNN_SUBCELL
    GRAPH_REPR --> GNN_TILE
    GRAPH_REPR --> GNN_REGION
    
    %% HGT Path
    HGT_CNN_3D --> CROSS_MODAL_ATTN
    HGT_CNN_2D --> CROSS_MODAL_ATTN
    HGT_ENCODER --> CROSS_MODAL_ATTN
    MLP_STATE --> CROSS_MODAL_ATTN
    CROSS_MODAL_ATTN --> HGT_FUSION_LAYER
    
    %% Hierarchical Path
    HIER_CNN_3D --> HIER_ATTENTION
    HIER_CNN_2D --> HIER_ATTENTION
    GNN_SUBCELL --> DIFFPOOL
    GNN_TILE --> DIFFPOOL
    GNN_REGION --> DIFFPOOL
    DIFFPOOL --> HIER_ATTENTION
    MLP_STATE --> HIER_ATTENTION
    HIER_ATTENTION --> HIER_FUSION_LAYER
    
    %% Policy and Training
    HGT_FUSION_LAYER --> POLICY_HEAD
    HGT_FUSION_LAYER --> VALUE_HEAD
    HIER_FUSION_LAYER --> POLICY_HEAD
    HIER_FUSION_LAYER --> VALUE_HEAD
    
    POLICY_HEAD --> EXPLORATION
    ICM --> ADAPTIVE_SCALE
    NOVELTY --> ADAPTIVE_SCALE
    ADAPTIVE_SCALE --> EXPERIENCE_BUFFER
    
    EXPERIENCE_BUFFER --> PPO_LOSS
    PPO_LOSS --> OPTIMIZER
    OPTIMIZER --> POLICY_HEAD
    OPTIMIZER --> VALUE_HEAD
    
    %% Styling
    classDef primary fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef secondary fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef processing fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef training fill:#FF5722,stroke:#D84315,stroke-width:2px,color:#fff
    classDef environment fill:#607D8B,stroke:#455A64,stroke-width:2px,color:#fff
    
    class HGT_PRIMARY,HGT_ENCODER,CROSS_MODAL_ATTN,HGT_FUSION_LAYER primary
    class HIERARCHICAL_SECONDARY,DIFFPOOL,HIER_ATTENTION,HIER_FUSION_LAYER secondary
    class OBSERVATIONS,STATE_PROCESSING,MLP_STATE processing
    class TRAINING,PPO_LOSS,OPTIMIZER,EXPERIENCE_BUFFER training
    class ENVIRONMENT,ENV_STATE,REWARD,DONE environment
```

---

## 3. HGT Multimodal Feature Extractor (PRIMARY)

This diagram provides detailed insight into the Heterogeneous Graph Transformer architecture, which is the primary and recommended approach.

```mermaid
graph TB
    subgraph INPUT_DATA ["📥 Input Data Streams"]
        VISUAL_INPUT["Visual Observations<br/>84x84x12 Temporal Stack<br/>176x100 Global View"]
        GRAPH_INPUT["Heterogeneous Graph<br/>Node Features + Types<br/>Edge Features + Types<br/>Adjacency Structure"]
        STATE_INPUT["Game State Vector<br/>Physics State<br/>Entity Status<br/>Objective Progress"]
    end
    
    subgraph HGT_ARCHITECTURE ["🌟 HGT Multimodal Architecture"]
        subgraph VISUAL_BRANCH ["Visual Processing Branch"]
            TEMPORAL_CNN["3D CNN Pipeline<br/>Conv3d(1→32→64→128)<br/>Temporal Modeling<br/>Adaptive Pooling"]
            GLOBAL_CNN["2D CNN Pipeline<br/>Conv2d(1→32→64→128)<br/>Spatial Context<br/>Global Understanding"]
        end
        
        subgraph HGT_BRANCH ["HGT Graph Processing Branch"]
            subgraph TYPE_EMBEDDINGS ["Type-Specific Embeddings"]
                NODE_EMBEDDINGS["Node Type Embeddings<br/>Grid Cells: Spatial Properties<br/>Entities: Dynamic State<br/>Hazards: Danger Level<br/>Switches: Activation State"]
                EDGE_EMBEDDINGS["Edge Type Embeddings<br/>Movement: walk/jump/fall<br/>Functional: activate/trigger<br/>Spatial: adjacency/distance"]
            end
            
            subgraph HGT_LAYERS ["HGT Transformer Layers"]
                TYPE_ATTENTION["Type-aware Attention<br/>Node-type × Edge-type<br/>Specialized Attention Heads<br/>Heterogeneous Message Passing"]
                ENTITY_ATTENTION["Entity-aware Processing<br/>Hazard Detection Focus<br/>Goal-oriented Attention<br/>Dynamic Entity Tracking"]
                MULTI_HEAD_ATTN["Multi-head Attention<br/>8 Attention Heads<br/>Parallel Processing<br/>Feature Diversity"]
            end
            
            subgraph AGGREGATION ["Feature Aggregation"]
                MESSAGE_PASSING["Heterogeneous Message Passing<br/>Type-specific Messages<br/>Attention-weighted Aggregation<br/>Neighborhood Information"]
                GLOBAL_POOLING["Global Graph Pooling<br/>Mean-Max Pooling<br/>Graph-level Representation<br/>2 × output_dim Features"]
            end
        end
        
        subgraph STATE_BRANCH ["State Processing Branch"]
            STATE_MLP["State MLP<br/>Linear(state_dim → hidden)<br/>ReLU Activation<br/>Dropout Regularization<br/>Physics Feature Extraction"]
        end
        
        subgraph MULTIMODAL_FUSION ["Advanced Multimodal Fusion"]
            SPATIAL_ATTENTION["Spatial Attention Module<br/>Graph-Visual Alignment<br/>Spatial Correspondence<br/>Cross-modal Enhancement"]
            CROSS_MODAL_ATTN["Cross-modal Attention<br/>Multi-head Attention<br/>Visual ↔ Graph ↔ State<br/>Feature Integration"]
            ATTENTION_NORM["Layer Normalization<br/>Residual Connections<br/>Stable Training<br/>Gradient Flow"]
        end
        
        subgraph OUTPUT_NETWORK ["Output Processing"]
            FUSION_NETWORK["Fusion Network<br/>Linear(total_dim → 2×features_dim)<br/>ReLU → Dropout<br/>Linear(2×features_dim → features_dim)<br/>Final Feature Vector"]
        end
    end
    
    subgraph HGT_ADVANTAGES ["🎯 HGT Advantages"]
        TYPE_SPECIALIZATION["Type Specialization<br/>Dedicated processing for<br/>different node/edge types<br/>Optimal feature learning"]
        ATTENTION_MECHANISMS["Advanced Attention<br/>Multi-head attention<br/>Type-aware processing<br/>Entity-specific focus"]
        SCALABILITY["Scalability<br/>Handles large graphs<br/>Efficient computation<br/>Parallel processing"]
        PERFORMANCE["Superior Performance<br/>Complex spatial reasoning<br/>Robust generalization"]
    end
    
    %% Data Flow
    VISUAL_INPUT --> TEMPORAL_CNN
    VISUAL_INPUT --> GLOBAL_CNN
    GRAPH_INPUT --> NODE_EMBEDDINGS
    GRAPH_INPUT --> EDGE_EMBEDDINGS
    STATE_INPUT --> STATE_MLP
    
    NODE_EMBEDDINGS --> TYPE_ATTENTION
    EDGE_EMBEDDINGS --> TYPE_ATTENTION
    TYPE_ATTENTION --> ENTITY_ATTENTION
    ENTITY_ATTENTION --> MULTI_HEAD_ATTN
    MULTI_HEAD_ATTN --> MESSAGE_PASSING
    MESSAGE_PASSING --> GLOBAL_POOLING
    
    TEMPORAL_CNN --> SPATIAL_ATTENTION
    GLOBAL_CNN --> SPATIAL_ATTENTION
    GLOBAL_POOLING --> SPATIAL_ATTENTION
    SPATIAL_ATTENTION --> CROSS_MODAL_ATTN
    STATE_MLP --> CROSS_MODAL_ATTN
    
    CROSS_MODAL_ATTN --> ATTENTION_NORM
    ATTENTION_NORM --> FUSION_NETWORK
    
    %% Styling
    classDef primary fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef hgt fill:#8BC34A,stroke:#689F38,stroke-width:2px,color:#fff
    classDef attention fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef processing fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef advantages fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    
    class HGT_ARCHITECTURE,FUSION_NETWORK primary
    class HGT_BRANCH,TYPE_EMBEDDINGS,HGT_LAYERS,AGGREGATION hgt
    class TYPE_ATTENTION,ENTITY_ATTENTION,MULTI_HEAD_ATTN,CROSS_MODAL_ATTN,SPATIAL_ATTENTION attention
    class VISUAL_BRANCH,STATE_BRANCH,MULTIMODAL_FUSION processing
    class HGT_ADVANTAGES,TYPE_SPECIALIZATION,ATTENTION_MECHANISMS,SCALABILITY,PERFORMANCE advantages
```

---

## 4. Hierarchical Feature Extractor (SECONDARY)

This diagram shows the hierarchical multi-resolution approach as a secondary architecture option.

```mermaid
graph TB
    subgraph INPUT_STREAMS ["📥 Input Data Streams"]
        VISUAL_STACK["Visual Observations<br/>84x84x12 Temporal Stack<br/>176x100 Global View"]
        HIERARCHICAL_GRAPH["Hierarchical Graph<br/>Multi-resolution Structure<br/>6px → 24px → 96px<br/>Cross-resolution Links"]
        GAME_STATE_VEC["Game State Vector<br/>Physics + Objectives<br/>Normalized Features"]
    end
    
    subgraph HIERARCHICAL_ARCHITECTURE ["⭐ Hierarchical Multimodal Architecture"]
        subgraph VISUAL_PROCESSING ["Visual Processing Branch"]
            TEMPORAL_3D["3D CNN Branch<br/>Conv3d Layers<br/>Temporal Modeling<br/>Spatiotemporal Features"]
            GLOBAL_2D["2D CNN Branch<br/>Conv2d Layers<br/>Global Context<br/>Strategic Overview"]
        end
        
        subgraph MULTI_RESOLUTION_GNN ["Multi-resolution GNN Processing"]
            subgraph RESOLUTION_LEVELS ["Resolution Levels"]
                SUBCELL_LEVEL["Sub-cell Level (6px)<br/>Fine-grained Details<br/>Precise Collision Detection<br/>Local Movement Planning"]
                TILE_LEVEL["Tile Level (24px)<br/>Navigation Planning<br/>Path Finding<br/>Tactical Decisions"]
                REGION_LEVEL["Region Level (96px)<br/>Strategic Planning<br/>High-level Goals<br/>Area Control"]
            end
            
            subgraph GNN_PROCESSING ["GNN Processing Layers"]
                SUBCELL_GNN["Sub-cell GNN<br/>GraphSAGE Layers<br/>Local Feature Aggregation<br/>Fine Detail Processing"]
                TILE_GNN["Tile GNN<br/>GraphSAGE Layers<br/>Navigation Features<br/>Movement Planning"]
                REGION_GNN["Region GNN<br/>GraphSAGE Layers<br/>Strategic Features<br/>Goal-oriented Processing"]
            end
            
            subgraph DIFFPOOL_HIERARCHY ["DiffPool Hierarchical Pooling"]
                POOL_1["DiffPool Layer 1<br/>Sub-cell → Tile<br/>Learnable Coarsening<br/>Information Preservation"]
                POOL_2["DiffPool Layer 2<br/>Tile → Region<br/>Strategic Abstraction<br/>High-level Features"]
                AUX_LOSSES["Auxiliary Losses<br/>Link Prediction<br/>Entropy Regularization<br/>Orthogonality Constraint"]
            end
        end
        
        subgraph STATE_PROCESSING_HIER ["State Processing Branch"]
            STATE_MLP_HIER["State MLP<br/>Physics Processing<br/>Objective Tracking<br/>Normalized Features"]
        end
        
        subgraph HIERARCHICAL_FUSION ["Context-aware Fusion"]
            PHYSICS_ATTENTION["Physics-adaptive Attention<br/>Dynamic Weighting<br/>Context-aware Integration<br/>State-dependent Fusion"]
            MULTI_SCALE_FUSION["Multi-scale Fusion<br/>Resolution Integration<br/>Feature Combination<br/>Hierarchical Understanding"]
        end
        
        subgraph OUTPUT_PROCESSING ["Output Processing"]
            FINAL_FUSION["Final Fusion Network<br/>Feature Integration<br/>Dimensionality Reduction<br/>Output Generation"]
        end
    end
    
    subgraph HIERARCHICAL_BENEFITS ["🎯 Hierarchical Benefits"]
        MULTI_SCALE["Multi-scale Understanding<br/>Fine details + Strategy<br/>Local + Global context<br/>Comprehensive reasoning"]
        LEARNABLE_POOLING["Learnable Pooling<br/>DiffPool coarsening<br/>Information preservation<br/>Adaptive abstraction"]
        AUXILIARY_TRAINING["Auxiliary Training<br/>Additional supervision<br/>Improved representations<br/>Stable learning"]
        INTERPRETABILITY["Interpretability<br/>Clear resolution levels<br/>Understandable hierarchy<br/>Debuggable features"]
    end
    
    %% Data Flow
    VISUAL_STACK --> TEMPORAL_3D
    VISUAL_STACK --> GLOBAL_2D
    HIERARCHICAL_GRAPH --> SUBCELL_GNN
    HIERARCHICAL_GRAPH --> TILE_GNN
    HIERARCHICAL_GRAPH --> REGION_GNN
    GAME_STATE_VEC --> STATE_MLP_HIER
    
    SUBCELL_GNN --> POOL_1
    TILE_GNN --> POOL_1
    POOL_1 --> POOL_2
    REGION_GNN --> POOL_2
    POOL_1 --> AUX_LOSSES
    POOL_2 --> AUX_LOSSES
    
    TEMPORAL_3D --> PHYSICS_ATTENTION
    GLOBAL_2D --> PHYSICS_ATTENTION
    POOL_2 --> PHYSICS_ATTENTION
    STATE_MLP_HIER --> PHYSICS_ATTENTION
    
    PHYSICS_ATTENTION --> MULTI_SCALE_FUSION
    MULTI_SCALE_FUSION --> FINAL_FUSION
    
    %% Styling
    classDef secondary fill:#2196F3,stroke:#1976D2,stroke-width:3px,color:#fff
    classDef hierarchical fill:#03A9F4,stroke:#0288D1,stroke-width:2px,color:#fff
    classDef diffpool fill:#00BCD4,stroke:#0097A7,stroke-width:2px,color:#fff
    classDef processing fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef benefits fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    
    class HIERARCHICAL_ARCHITECTURE,FINAL_FUSION secondary
    class MULTI_RESOLUTION_GNN,RESOLUTION_LEVELS,GNN_PROCESSING hierarchical
    class DIFFPOOL_HIERARCHY,POOL_1,POOL_2,AUX_LOSSES diffpool
    class VISUAL_PROCESSING,STATE_PROCESSING_HIER,HIERARCHICAL_FUSION processing
    class HIERARCHICAL_BENEFITS,MULTI_SCALE,LEARNABLE_POOLING,AUXILIARY_TRAINING,INTERPRETABILITY benefits
```

---

## 5. NClone Environment System

This diagram details the simulation environment and its integration with the RL agent.

```mermaid
graph TB
    subgraph NCLONE_SYSTEM ["🎮 NClone Environment System"]
        subgraph CORE_SIMULATION ["⚙️ Core Simulation Engine"]
            NSIM_ENGINE["nsim.py<br/>Physics Engine<br/>Collision Detection<br/>Entity Updates<br/>Game Logic"]
            NINJA_PHYSICS["ninja.py<br/>Player Physics<br/>Movement Mechanics<br/>Jump/Wall Dynamics<br/>State Management"]
            ENTITY_SYSTEM["entities.py<br/>Game Objects<br/>Switches, Hazards<br/>Interactive Elements<br/>State Tracking"]
            LEVEL_MANAGER["Level Management<br/>Level Loading<br/>Geometry Processing<br/>Spawn Points"]
        end
        
        subgraph RENDERING_SYSTEM ["🖼️ Rendering System"]
            RENDERER["nsim_renderer.py<br/>Visual Rendering<br/>Frame Generation<br/>Debug Overlays"]
            VISUAL_OUTPUT["Visual Output<br/>Player-centric View<br/>Global View<br/>Debug Information"]
        end
        
        subgraph RL_INTERFACE ["🤖 RL Environment Interface"]
            BASE_ENVIRONMENT["base_environment.py<br/>Gym Interface<br/>Action/Observation Space<br/>Episode Management<br/>Reward Calculation"]
            SPECIFIC_ENVS["Specific Environments<br/>basic_level_no_gold/<br/>Environment Variants<br/>Task Definitions"]
            OBS_PROCESSOR["observation_processor.py<br/>Multi-modal Processing<br/>State Normalization<br/>Feature Extraction"]
        end
        
        subgraph GRAPH_GENERATION ["📊 Graph Generation System"]
            HIERARCHICAL_GRAPH_BUILDER["hierarchical_builder.py<br/>🌟 PRIMARY BUILDER<br/>Multi-resolution Graphs<br/>6px/24px/96px Levels<br/>Cross-resolution Links"]
            COMMON_GRAPH_COMPONENTS["common.py<br/>Shared Components<br/>GraphData Structure<br/>NodeType/EdgeType Enums<br/>Graph Utilities"]
            GRAPH_OBSERVATION_INTEGRATION["graph_observation.py<br/>Graph State Integration<br/>Dynamic Updates<br/>RL Interface"]
        end
        
        subgraph OBSERVATION_PIPELINE ["📡 Observation Pipeline"]
            VISUAL_OBS["Visual Observations<br/>Player Frame Stack<br/>Global Level View<br/>Temporal Context"]
            STATE_OBS["State Observations<br/>Physics State<br/>Entity Status<br/>Objective Progress"]
            GRAPH_OBS["Graph Observations<br/>Heterogeneous Structure<br/>Node/Edge Features<br/>Dynamic Updates"]
        end
        
        subgraph ARCHIVE_COMPONENTS ["📦 Archived Components"]
            LEGACY_GRAPH_BUILDER["graph_builder.py<br/>Legacy Standard Builder<br/>Single-resolution<br/>Basic Graph Structure"]
            LEGACY_PATHFINDING["pathfinding/<br/>Legacy A* Navigation<br/>Surface Parsing<br/>Path Planning"]
        end
    end
    
    subgraph ENVIRONMENT_FEATURES ["🎯 Environment Features"]
        PHYSICS_ACCURACY["Physics Accuracy<br/>Precise Collision<br/>Realistic Movement<br/>Authentic N++ Feel"]
        MULTI_MODAL_OBS["Multi-modal Observations<br/>Visual + State + Graph<br/>Rich Information<br/>Comprehensive Context"]
        DYNAMIC_GRAPHS["Dynamic Graphs<br/>Real-time Updates<br/>Entity Tracking<br/>State Changes"]
        SCALABLE_LEVELS["Scalable Levels<br/>Various Difficulties<br/>Different Layouts<br/>Diverse Challenges"]
    end
    
    %% Data Flow
    NSIM_ENGINE --> NINJA_PHYSICS
    NSIM_ENGINE --> ENTITY_SYSTEM
    NINJA_PHYSICS --> RENDERER
    ENTITY_SYSTEM --> RENDERER
    LEVEL_MANAGER --> NSIM_ENGINE
    
    RENDERER --> VISUAL_OUTPUT
    VISUAL_OUTPUT --> VISUAL_OBS
    
    NSIM_ENGINE --> BASE_ENVIRONMENT
    BASE_ENVIRONMENT --> SPECIFIC_ENVS
    SPECIFIC_ENVS --> OBS_PROCESSOR
    
    NSIM_ENGINE --> HIERARCHICAL_GRAPH_BUILDER
    ENTITY_SYSTEM --> HIERARCHICAL_GRAPH_BUILDER
    HIERARCHICAL_GRAPH_BUILDER --> COMMON_GRAPH_COMPONENTS
    COMMON_GRAPH_COMPONENTS --> GRAPH_OBSERVATION_INTEGRATION
    GRAPH_OBSERVATION_INTEGRATION --> GRAPH_OBS
    
    NINJA_PHYSICS --> STATE_OBS
    ENTITY_SYSTEM --> STATE_OBS
    
    VISUAL_OBS --> OBS_PROCESSOR
    STATE_OBS --> OBS_PROCESSOR
    GRAPH_OBS --> OBS_PROCESSOR
    
    %% Styling
    classDef primary fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef simulation fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef interface fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef graph fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef archive fill:#9E9E9E,stroke:#616161,stroke-width:2px,color:#fff
    classDef features fill:#607D8B,stroke:#455A64,stroke-width:2px,color:#fff
    
    class HIERARCHICAL_GRAPH_BUILDER,COMMON_GRAPH_COMPONENTS primary
    class CORE_SIMULATION,NSIM_ENGINE,NINJA_PHYSICS,ENTITY_SYSTEM simulation
    class RL_INTERFACE,BASE_ENVIRONMENT,OBS_PROCESSOR interface
    class GRAPH_GENERATION,GRAPH_OBSERVATION_INTEGRATION graph
    class ARCHIVE_COMPONENTS,LEGACY_GRAPH_BUILDER,LEGACY_PATHFINDING archive
    class ENVIRONMENT_FEATURES,PHYSICS_ACCURACY,MULTI_MODAL_OBS,DYNAMIC_GRAPHS features
```

---

## 6. Graph Processing Pipeline

This diagram shows the detailed graph processing pipeline from level geometry to RL-ready graph observations.

```mermaid
graph TB
    subgraph LEVEL_INPUT ["🗺️ Level Input"]
        LEVEL_GEOMETRY["Level Geometry<br/>Walls, Platforms<br/>Static Elements<br/>Collision Boundaries"]
        DYNAMIC_ENTITIES["Dynamic Entities<br/>Ninja Position<br/>Switches, Hazards<br/>Interactive Objects"]
        GAME_STATE_INFO["Game State<br/>Physics State<br/>Entity Status<br/>Objective Progress"]
    end
    
    subgraph GRAPH_CONSTRUCTION ["🏗️ Graph Construction Pipeline"]
        subgraph SPATIAL_ANALYSIS ["Spatial Analysis"]
            COLLISION_DETECTION["Collision Detection<br/>Walkable Surfaces<br/>Jump Possibilities<br/>Movement Constraints"]
            CONNECTIVITY_ANALYSIS["Connectivity Analysis<br/>Reachable Areas<br/>Movement Paths<br/>Navigation Links"]
            MULTI_RESOLUTION_GRID["Multi-resolution Grid<br/>6px: Fine Details<br/>24px: Navigation<br/>96px: Strategy"]
        end
        
        subgraph NODE_GENERATION ["Node Generation"]
            GRID_NODES["Grid Nodes<br/>Spatial Positions<br/>Walkable Cells<br/>Movement Points"]
            ENTITY_NODES["Entity Nodes<br/>Dynamic Objects<br/>Interactive Elements<br/>Goal Locations"]
            HAZARD_NODES["Hazard Nodes<br/>Danger Zones<br/>Obstacle Areas<br/>Avoidance Points"]
            SWITCH_NODES["Switch Nodes<br/>Activation Points<br/>Door Controls<br/>Mechanism Triggers"]
        end
        
        subgraph EDGE_GENERATION ["Edge Generation"]
            MOVEMENT_EDGES["Movement Edges<br/>Walk Connections<br/>Jump Possibilities<br/>Fall Trajectories"]
            FUNCTIONAL_EDGES["Functional Edges<br/>Switch Activation<br/>Door Opening<br/>Trigger Relations"]
            HIERARCHICAL_EDGES["Hierarchical Edges<br/>Cross-resolution Links<br/>Abstraction Connections<br/>Multi-scale Relations"]
        end
        
        subgraph FEATURE_COMPUTATION ["Feature Computation"]
            NODE_FEATURES["Node Features<br/>Position Coordinates<br/>Type Information<br/>State Properties<br/>Physics Attributes"]
            EDGE_FEATURES["Edge Features<br/>Movement Cost<br/>Action Type<br/>Relationship Strength<br/>Traversal Difficulty"]
            GRAPH_METADATA["Graph Metadata<br/>Adjacency Matrix<br/>Type Mappings<br/>Mask Information"]
        end
    end
    
    subgraph GRAPH_PROCESSING ["🧠 Graph Processing"]
        subgraph HGT_PROCESSING ["HGT Processing (PRIMARY)"]
            TYPE_SPECIFIC_EMBED["Type-specific Embeddings<br/>Node Type Projections<br/>Edge Type Projections<br/>Specialized Processing"]
            HETEROGENEOUS_ATTENTION["Heterogeneous Attention<br/>Type-aware Mechanisms<br/>Multi-head Processing<br/>Entity-focused Attention"]
            GLOBAL_GRAPH_POOLING["Global Graph Pooling<br/>Mean-Max Pooling<br/>Graph-level Features<br/>Comprehensive Representation"]
        end
        
        subgraph HIERARCHICAL_PROCESSING ["Hierarchical Processing (SECONDARY)"]
            RESOLUTION_SPECIFIC_GNN["Resolution-specific GNNs<br/>Sub-cell Processing<br/>Tile Processing<br/>Region Processing"]
            DIFFPOOL_COARSENING["DiffPool Coarsening<br/>Learnable Pooling<br/>Information Preservation<br/>Hierarchical Abstraction"]
            MULTI_SCALE_INTEGRATION["Multi-scale Integration<br/>Cross-resolution Fusion<br/>Feature Combination<br/>Unified Representation"]
        end
    end
    
    subgraph OUTPUT_INTEGRATION ["🎯 Output Integration"]
        GRAPH_FEATURES_OUT["Graph Features<br/>Processed Representations<br/>Spatial Understanding<br/>Structural Knowledge"]
        RL_OBSERVATION["RL Observation<br/>Graph Component<br/>Multi-modal Integration<br/>Agent Input"]
        DYNAMIC_UPDATES["Dynamic Updates<br/>Real-time Changes<br/>Entity Movement<br/>State Evolution"]
    end
    
    %% Data Flow
    LEVEL_GEOMETRY --> COLLISION_DETECTION
    DYNAMIC_ENTITIES --> CONNECTIVITY_ANALYSIS
    GAME_STATE_INFO --> MULTI_RESOLUTION_GRID
    
    COLLISION_DETECTION --> GRID_NODES
    CONNECTIVITY_ANALYSIS --> MOVEMENT_EDGES
    MULTI_RESOLUTION_GRID --> HIERARCHICAL_EDGES
    
    DYNAMIC_ENTITIES --> ENTITY_NODES
    DYNAMIC_ENTITIES --> HAZARD_NODES
    DYNAMIC_ENTITIES --> SWITCH_NODES
    
    GRID_NODES --> NODE_FEATURES
    ENTITY_NODES --> NODE_FEATURES
    HAZARD_NODES --> NODE_FEATURES
    SWITCH_NODES --> NODE_FEATURES
    
    MOVEMENT_EDGES --> EDGE_FEATURES
    FUNCTIONAL_EDGES --> EDGE_FEATURES
    HIERARCHICAL_EDGES --> EDGE_FEATURES
    
    NODE_FEATURES --> TYPE_SPECIFIC_EMBED
    EDGE_FEATURES --> TYPE_SPECIFIC_EMBED
    GRAPH_METADATA --> HETEROGENEOUS_ATTENTION
    
    TYPE_SPECIFIC_EMBED --> HETEROGENEOUS_ATTENTION
    HETEROGENEOUS_ATTENTION --> GLOBAL_GRAPH_POOLING
    
    NODE_FEATURES --> RESOLUTION_SPECIFIC_GNN
    EDGE_FEATURES --> RESOLUTION_SPECIFIC_GNN
    RESOLUTION_SPECIFIC_GNN --> DIFFPOOL_COARSENING
    DIFFPOOL_COARSENING --> MULTI_SCALE_INTEGRATION
    
    GLOBAL_GRAPH_POOLING --> GRAPH_FEATURES_OUT
    MULTI_SCALE_INTEGRATION --> GRAPH_FEATURES_OUT
    GRAPH_FEATURES_OUT --> RL_OBSERVATION
    
    DYNAMIC_ENTITIES --> DYNAMIC_UPDATES
    DYNAMIC_UPDATES --> RL_OBSERVATION
    
    %% Styling
    classDef input fill:#607D8B,stroke:#455A64,stroke-width:2px,color:#fff
    classDef construction fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef hgt fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef hierarchical fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef output fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    
    class LEVEL_INPUT,LEVEL_GEOMETRY,DYNAMIC_ENTITIES,GAME_STATE_INFO input
    class GRAPH_CONSTRUCTION,SPATIAL_ANALYSIS,NODE_GENERATION,EDGE_GENERATION,FEATURE_COMPUTATION construction
    class HGT_PROCESSING,TYPE_SPECIFIC_EMBED,HETEROGENEOUS_ATTENTION,GLOBAL_GRAPH_POOLING hgt
    class HIERARCHICAL_PROCESSING,RESOLUTION_SPECIFIC_GNN,DIFFPOOL_COARSENING,MULTI_SCALE_INTEGRATION hierarchical
    class OUTPUT_INTEGRATION,GRAPH_FEATURES_OUT,RL_OBSERVATION,DYNAMIC_UPDATES output
```

---

## 7. Training and Exploration Systems

This diagram shows the complete training pipeline including adaptive exploration mechanisms.

```mermaid
graph TB
    subgraph TRAINING_ECOSYSTEM ["🎯 Training Ecosystem"]
        subgraph ENVIRONMENT_INTERACTION ["🎮 Environment Interaction"]
            PARALLEL_ENVS["Parallel Environments<br/>64 Concurrent Instances<br/>Vectorized Processing<br/>Efficient Sampling"]
            EPISODE_MANAGEMENT["Episode Management<br/>Reset Handling<br/>Termination Detection<br/>Reward Collection"]
            OBSERVATION_COLLECTION["Observation Collection<br/>Multi-modal Data<br/>Temporal Stacking<br/>Batch Processing"]
        end
        
        subgraph EXPERIENCE_PROCESSING ["📊 Experience Processing"]
            TRAJECTORY_BUFFER["Trajectory Buffer<br/>Experience Storage<br/>Rollout Collection<br/>Batch Organization"]
            GAE_COMPUTATION["GAE Computation<br/>Advantage Estimation<br/>Return Calculation<br/>Bias-Variance Trade-off"]
            NORMALIZATION["Normalization<br/>Advantage Normalization<br/>Reward Scaling<br/>Observation Preprocessing"]
        end
        
        subgraph ADAPTIVE_EXPLORATION ["🔍 Adaptive Exploration System"]
            subgraph ICM_MODULE ["Intrinsic Curiosity Module"]
                FORWARD_MODEL["Forward Model<br/>State Prediction<br/>Dynamics Learning<br/>Environment Modeling"]
                INVERSE_MODEL["Inverse Model<br/>Action Prediction<br/>Feature Learning<br/>Representation Quality"]
                PREDICTION_ERROR["Prediction Error<br/>Curiosity Reward<br/>Exploration Bonus<br/>Novel State Detection"]
            end
            
            subgraph NOVELTY_DETECTION ["Novelty Detection"]
                STATE_COUNTING["State Counting<br/>Visit Frequency<br/>Exploration Tracking<br/>Coverage Measurement"]
                HASH_ENCODING["Hash Encoding<br/>State Representation<br/>Efficient Storage<br/>Fast Lookup"]
                COUNT_BONUS["Count-based Bonus<br/>Inverse Visit Count<br/>Exploration Incentive<br/>Coverage Reward"]
            end
            
            subgraph ADAPTIVE_SCALING ["Adaptive Scaling"]
                EXPLORATION_DECAY["Exploration Decay<br/>Annealing Schedule<br/>Curriculum Learning<br/>Progressive Focus"]
                BONUS_INTEGRATION["Bonus Integration<br/>Intrinsic + Extrinsic<br/>Weighted Combination<br/>Balanced Exploration"]
                PERFORMANCE_MONITORING["Performance Monitoring<br/>Success Rate Tracking<br/>Exploration Effectiveness<br/>Adaptive Adjustment"]
            end
        end
        
        subgraph PPO_TRAINING ["🧠 PPO Training Pipeline"]
            subgraph POLICY_OPTIMIZATION ["Policy Optimization"]
                POLICY_LOSS["Policy Loss<br/>Clipped Objective<br/>Importance Sampling<br/>Conservative Updates"]
                VALUE_LOSS["Value Loss<br/>MSE Regression<br/>Return Prediction<br/>Baseline Learning"]
                ENTROPY_LOSS["Entropy Loss<br/>Exploration Regularization<br/>Policy Diversity<br/>Action Variety"]
            end
            
            subgraph OPTIMIZATION ["Optimization"]
                GRADIENT_COMPUTATION["Gradient Computation<br/>Backpropagation<br/>Loss Aggregation<br/>Parameter Updates"]
                GRADIENT_CLIPPING["Gradient Clipping<br/>Stability Enhancement<br/>Explosion Prevention<br/>Smooth Learning"]
                ADAM_OPTIMIZER["Adam Optimizer<br/>Adaptive Learning Rate<br/>Momentum Integration<br/>Parameter Updates"]
            end
            
            subgraph LEARNING_SCHEDULE ["Learning Schedule"]
                LR_SCHEDULING["Learning Rate Scheduling<br/>Linear Decay<br/>Performance-based<br/>Adaptive Adjustment"]
                CLIP_SCHEDULING["Clip Range Scheduling<br/>Conservative Updates<br/>Stability Maintenance<br/>Trust Region"]
                ENTROPY_SCHEDULING["Entropy Scheduling<br/>Exploration Control<br/>Exploitation Balance<br/>Curriculum Learning"]
            end
        end
        
        subgraph MONITORING_EVALUATION ["📈 Monitoring & Evaluation"]
            PERFORMANCE_METRICS["Performance Metrics<br/>Episode Rewards<br/>Success Rates<br/>Learning Progress"]
            EXPLORATION_METRICS["Exploration Metrics<br/>State Coverage<br/>Novelty Scores<br/>Curiosity Rewards"]
            MODEL_CHECKPOINTING["Model Checkpointing<br/>Best Model Saving<br/>Training Resumption<br/>Evaluation Points"]
            TENSORBOARD_LOGGING["TensorBoard Logging<br/>Real-time Monitoring<br/>Metric Visualization<br/>Training Analysis"]
        end
    end
    
    subgraph TRAINING_FEATURES ["🎯 Training Features"]
        SCALABILITY["Scalability<br/>Parallel Processing<br/>Efficient Sampling<br/>Fast Training"]
        STABILITY["Stability<br/>Clipped Updates<br/>Gradient Control<br/>Robust Learning"]
        EXPLORATION_QUALITY["Exploration Quality<br/>Curiosity-driven<br/>Coverage-based<br/>Adaptive Scaling"]
        MONITORING_DEPTH["Monitoring Depth<br/>Comprehensive Metrics<br/>Real-time Feedback<br/>Performance Tracking"]
    end
    
    %% Data Flow
    PARALLEL_ENVS --> EPISODE_MANAGEMENT
    EPISODE_MANAGEMENT --> OBSERVATION_COLLECTION
    OBSERVATION_COLLECTION --> TRAJECTORY_BUFFER
    
    TRAJECTORY_BUFFER --> GAE_COMPUTATION
    GAE_COMPUTATION --> NORMALIZATION
    
    OBSERVATION_COLLECTION --> FORWARD_MODEL
    OBSERVATION_COLLECTION --> INVERSE_MODEL
    FORWARD_MODEL --> PREDICTION_ERROR
    INVERSE_MODEL --> PREDICTION_ERROR
    
    OBSERVATION_COLLECTION --> STATE_COUNTING
    STATE_COUNTING --> HASH_ENCODING
    HASH_ENCODING --> COUNT_BONUS
    
    PREDICTION_ERROR --> BONUS_INTEGRATION
    COUNT_BONUS --> BONUS_INTEGRATION
    BONUS_INTEGRATION --> EXPLORATION_DECAY
    EXPLORATION_DECAY --> PERFORMANCE_MONITORING
    
    NORMALIZATION --> POLICY_LOSS
    NORMALIZATION --> VALUE_LOSS
    NORMALIZATION --> ENTROPY_LOSS
    
    POLICY_LOSS --> GRADIENT_COMPUTATION
    VALUE_LOSS --> GRADIENT_COMPUTATION
    ENTROPY_LOSS --> GRADIENT_COMPUTATION
    
    GRADIENT_COMPUTATION --> GRADIENT_CLIPPING
    GRADIENT_CLIPPING --> ADAM_OPTIMIZER
    
    ADAM_OPTIMIZER --> LR_SCHEDULING
    LR_SCHEDULING --> CLIP_SCHEDULING
    CLIP_SCHEDULING --> ENTROPY_SCHEDULING
    
    PERFORMANCE_MONITORING --> PERFORMANCE_METRICS
    BONUS_INTEGRATION --> EXPLORATION_METRICS
    ADAM_OPTIMIZER --> MODEL_CHECKPOINTING
    PERFORMANCE_METRICS --> TENSORBOARD_LOGGING
    
    %% Styling
    classDef environment fill:#607D8B,stroke:#455A64,stroke-width:2px,color:#fff
    classDef processing fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef exploration fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef training fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef monitoring fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef features fill:#795548,stroke:#5D4037,stroke-width:2px,color:#fff
    
    class ENVIRONMENT_INTERACTION,PARALLEL_ENVS,EPISODE_MANAGEMENT environment
    class EXPERIENCE_PROCESSING,TRAJECTORY_BUFFER,GAE_COMPUTATION,NORMALIZATION processing
    class ADAPTIVE_EXPLORATION,ICM_MODULE,NOVELTY_DETECTION,ADAPTIVE_SCALING exploration
    class PPO_TRAINING,POLICY_OPTIMIZATION,OPTIMIZATION,LEARNING_SCHEDULE training
    class MONITORING_EVALUATION,PERFORMANCE_METRICS,MODEL_CHECKPOINTING monitoring
    class TRAINING_FEATURES,SCALABILITY,STABILITY,EXPLORATION_QUALITY features
```

---

## 8. Data Flow and Integration

This final diagram shows the complete data flow through the entire system, from environment to trained agent.

```mermaid
graph TB
    subgraph COMPLETE_SYSTEM ["🌐 Complete NPP-RL System Data Flow"]
        subgraph ENV_LAYER ["🎮 Environment Layer"]
            NCLONE_SIM["NClone Simulation<br/>Physics Engine<br/>Entity Management<br/>Level Processing"]
            MULTI_MODAL_OBS["Multi-modal Observations<br/>Visual: 84x84x12 + 176x100<br/>State: Physics + Objectives<br/>Graph: Heterogeneous Structure"]
            REWARD_SIGNAL["Reward Signal<br/>Goal Achievement<br/>Survival Bonus<br/>Exploration Reward"]
        end
        
        subgraph PROCESSING_LAYER ["🔍 Processing Layer"]
            subgraph FEATURE_EXTRACTION_CHOICE ["Feature Extraction (Configurable)"]
                HGT_PATH["🌟 HGT Path (PRIMARY)<br/>Type-specific Attention<br/>Entity-aware Processing<br/>Advanced Fusion"]
                HIERARCHICAL_PATH["⭐ Hierarchical Path (SECONDARY)<br/>Multi-resolution GNNs<br/>DiffPool Coarsening<br/>Context-aware Fusion"]
            end
            
            UNIFIED_FEATURES["Unified Features<br/>512-dimensional<br/>Rich Representation<br/>Multi-modal Integration"]
        end
        
        subgraph AGENT_LAYER ["🧠 Agent Layer"]
            PPO_POLICY["PPO Policy Network<br/>Actor-Critic Architecture<br/>Shared Feature Extractor<br/>Separate Heads"]
            ACTION_SELECTION["Action Selection<br/>Stochastic Policy<br/>Exploration Balance<br/>Discrete Actions"]
            VALUE_ESTIMATION["Value Estimation<br/>State Value Function<br/>Advantage Computation<br/>Return Prediction"]
        end
        
        subgraph EXPLORATION_LAYER ["🔍 Exploration Layer"]
            CURIOSITY_SYSTEM["Curiosity System<br/>ICM Forward/Inverse<br/>Prediction Error Reward<br/>Novel State Detection"]
            NOVELTY_SYSTEM["Novelty System<br/>Count-based Exploration<br/>State Visit Tracking<br/>Coverage Incentive"]
            ADAPTIVE_BONUS["Adaptive Bonus<br/>Dynamic Scaling<br/>Performance-based<br/>Curriculum Learning"]
        end
        
        subgraph TRAINING_LAYER ["🎯 Training Layer"]
            EXPERIENCE_COLLECTION["Experience Collection<br/>Trajectory Rollouts<br/>Parallel Environments<br/>Batch Processing"]
            PPO_OPTIMIZATION["PPO Optimization<br/>Clipped Objective<br/>Value + Policy + Entropy<br/>Conservative Updates"]
            MODEL_UPDATES["Model Updates<br/>Gradient Descent<br/>Parameter Updates<br/>Learning Progress"]
        end
        
        subgraph MONITORING_LAYER ["📈 Monitoring Layer"]
            PERFORMANCE_TRACKING["Performance Tracking<br/>Episode Rewards<br/>Success Rates<br/>Learning Curves"]
            EXPLORATION_ANALYSIS["Exploration Analysis<br/>State Coverage<br/>Novelty Metrics<br/>Curiosity Effectiveness"]
            MODEL_MANAGEMENT["Model Management<br/>Checkpointing<br/>Best Model Selection<br/>Training Resumption"]
        end
    end
    
    subgraph SYSTEM_PROPERTIES ["🎯 System Properties"]
        MODULARITY["Modularity<br/>Configurable Components<br/>Swappable Architectures<br/>Flexible Design"]
        SCALABILITY_PROP["Scalability<br/>Parallel Processing<br/>Efficient Computation<br/>Large-scale Training"]
        ROBUSTNESS["Robustness<br/>Stable Training<br/>Error Handling<br/>Graceful Degradation"]
        PERFORMANCE_PROP["Performance<br/>Sample Efficiency<br/>Fast Convergence"]
    end
    
    %% Primary Data Flow (HGT Path)
    NCLONE_SIM --> MULTI_MODAL_OBS
    MULTI_MODAL_OBS --> HGT_PATH
    HGT_PATH --> UNIFIED_FEATURES
    UNIFIED_FEATURES --> PPO_POLICY
    PPO_POLICY --> ACTION_SELECTION
    PPO_POLICY --> VALUE_ESTIMATION
    
    %% Secondary Data Flow (Hierarchical Path)
    MULTI_MODAL_OBS -.-> HIERARCHICAL_PATH
    HIERARCHICAL_PATH -.-> UNIFIED_FEATURES
    
    %% Exploration Integration
    UNIFIED_FEATURES --> CURIOSITY_SYSTEM
    UNIFIED_FEATURES --> NOVELTY_SYSTEM
    CURIOSITY_SYSTEM --> ADAPTIVE_BONUS
    NOVELTY_SYSTEM --> ADAPTIVE_BONUS
    ADAPTIVE_BONUS --> REWARD_SIGNAL
    
    %% Training Loop
    ACTION_SELECTION --> NCLONE_SIM
    REWARD_SIGNAL --> EXPERIENCE_COLLECTION
    VALUE_ESTIMATION --> EXPERIENCE_COLLECTION
    EXPERIENCE_COLLECTION --> PPO_OPTIMIZATION
    PPO_OPTIMIZATION --> MODEL_UPDATES
    MODEL_UPDATES --> PPO_POLICY
    
    %% Monitoring Integration
    EXPERIENCE_COLLECTION --> PERFORMANCE_TRACKING
    ADAPTIVE_BONUS --> EXPLORATION_ANALYSIS
    MODEL_UPDATES --> MODEL_MANAGEMENT
    
    %% System Properties
    HGT_PATH -.-> MODULARITY
    EXPERIENCE_COLLECTION -.-> SCALABILITY_PROP
    PPO_OPTIMIZATION -.-> ROBUSTNESS
    MODEL_UPDATES -.-> PERFORMANCE_PROP
    
    %% Styling
    classDef primary fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef secondary fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef environment fill:#607D8B,stroke:#455A64,stroke-width:2px,color:#fff
    classDef processing fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef agent fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef exploration fill:#8BC34A,stroke:#689F38,stroke-width:2px,color:#fff
    classDef training fill:#FF5722,stroke:#D84315,stroke-width:2px,color:#fff
    classDef monitoring fill:#795548,stroke:#5D4037,stroke-width:2px,color:#fff
    classDef properties fill:#E91E63,stroke:#C2185B,stroke-width:2px,color:#fff
    
    class HGT_PATH,UNIFIED_FEATURES primary
    class HIERARCHICAL_PATH secondary
    class ENV_LAYER,NCLONE_SIM,MULTI_MODAL_OBS,REWARD_SIGNAL environment
    class PROCESSING_LAYER,FEATURE_EXTRACTION_CHOICE processing
    class AGENT_LAYER,PPO_POLICY,ACTION_SELECTION,VALUE_ESTIMATION agent
    class EXPLORATION_LAYER,CURIOSITY_SYSTEM,NOVELTY_SYSTEM,ADAPTIVE_BONUS exploration
    class TRAINING_LAYER,EXPERIENCE_COLLECTION,PPO_OPTIMIZATION,MODEL_UPDATES training
    class MONITORING_LAYER,PERFORMANCE_TRACKING,EXPLORATION_ANALYSIS,MODEL_MANAGEMENT monitoring
    class SYSTEM_PROPERTIES,MODULARITY,SCALABILITY_PROP,ROBUSTNESS,PERFORMANCE_PROP properties
```

---

## Usage Instructions

To generate these diagrams:

1. **Copy the desired Mermaid code** from any section above
2. **Paste into a Mermaid renderer** such as:
   - [Mermaid Live Editor](https://mermaid.live/)
   - GitHub/GitLab markdown (supports Mermaid natively)
   - VS Code with Mermaid extension
   - Documentation platforms (Notion, Confluence, etc.)

3. **Customize as needed**:
   - Modify colors by changing the `classDef` definitions
   - Add or remove components based on your focus area
   - Adjust layout by changing node connections
   - Scale complexity up or down for different audiences

## Diagram Hierarchy

The diagrams are organized in a hierarchical manner:

1. **Global System Architecture** - Complete ecosystem overview
2. **NPP-RL Agent Architecture** - Agent-specific components and data flow
3. **HGT/Hierarchical Details** - Deep dives into feature extraction approaches
4. **Environment System** - NClone simulation and RL interface details
5. **Graph Processing** - Graph construction and processing pipeline
6. **Training Systems** - Training loop and exploration mechanisms
7. **Data Flow Integration** - Complete system data flow and integration

This structure allows you to start with the big picture and drill down into specific areas of interest, providing both strategic overview and implementation details as needed.