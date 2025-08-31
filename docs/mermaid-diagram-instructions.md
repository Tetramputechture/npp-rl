# Mermaid Diagram Instructions: NPP-RL Agent Architecture

This document provides instructions for generating Mermaid diagrams specifically focused on the NPP-RL agent architecture and its HGT-based multimodal feature extraction system.

## Agent-Specific Architecture Diagrams

### 1. Complete NPP-RL Agent Architecture

```mermaid
graph TB
    subgraph ENVIRONMENT ["🎮 NClone Environment Interface"]
        ENV_STATE["Environment State"]
        REWARD["Reward Signal"]
        DONE["Episode Termination"]
    end
    
    subgraph OBSERVATIONS ["📊 Multi-modal Observations"]
        VISUAL_FRAMES["Visual Frames<br/>84x84x12 Stack<br/>Player-centric View"]
        GLOBAL_VIEW["Global View<br/>176x100 Downsampled<br/>Full Level"]
        GAME_STATE["Game State Vector<br/>Physics + Objectives<br/>Normalized Features"]
        GRAPH_REPR["Heterogeneous Graph<br/>Node/Edge Types<br/>Structural Representation"]
    end
    
    subgraph FEATURE_EXTRACTION ["🔍 HGT Multimodal Feature Extractor (PRIMARY)"]
        subgraph VISUAL_PROC ["Visual Processing"]
            CNN_3D["3D CNN Branch<br/>Temporal Modeling<br/>Spatiotemporal Features"]
            CNN_2D["2D CNN Branch<br/>Global Context<br/>Spatial Features"]
        end
        
        subgraph GRAPH_PROC ["HGT Graph Processing"]
            HGT_ENCODER["Heterogeneous Graph Transformer<br/>Type-specific Attention<br/>Multi-head Processing"]
            NODE_TYPES["Node Types<br/>Grid Cells, Entities<br/>Hazards, Switches"]
            EDGE_TYPES["Edge Types<br/>Movement (walk/jump/fall)<br/>Functional Relations"]
            ENTITY_EMBED["Entity Embeddings<br/>Specialized Processing<br/>Hazard-aware Attention"]
        end
        
        subgraph STATE_PROC ["State Processing"]
            MLP_STATE["MLP Branch<br/>Game State<br/>Physics Features"]
        end
        
        subgraph FUSION ["Advanced Multimodal Fusion"]
            CROSS_MODAL["Cross-modal Attention<br/>Spatial Awareness<br/>Type-aware Integration"]
            FUSION_LAYER["Feature Fusion<br/>HGT-enhanced Integration"]
        end
    end
    
    subgraph PPO_AGENT ["🧠 PPO Agent"]
        POLICY_HEAD["Policy Head (Actor)<br/>Action Probabilities<br/>Discrete Action Space"]
        VALUE_HEAD["Value Head (Critic)<br/>State Value Estimation<br/>Advantage Calculation"]
    end
    
    subgraph EXPLORATION ["🔍 Adaptive Exploration"]
        ICM["Intrinsic Curiosity Module<br/>Forward/Inverse Models<br/>Prediction Error Reward"]
        NOVELTY["Novelty Detection<br/>Count-based Exploration<br/>State Visit Tracking"]
        ADAPTIVE_SCALE["Adaptive Scaling<br/>Dynamic Bonus Adjustment"]
    end
    
    subgraph TRAINING ["🎯 Training Components"]
        EXPERIENCE_BUFFER["Experience Buffer<br/>Trajectory Collection<br/>GAE Computation"]
        PPO_LOSS["PPO Loss<br/>Policy + Value + Entropy<br/>Clipped Objective"]
        OPTIMIZER["Adam Optimizer<br/>Learning Rate Scheduling<br/>Gradient Clipping"]
    end
    
    %% Data Flow
    ENV_STATE --> OBSERVATIONS
    VISUAL_FRAMES --> CNN_3D
    GLOBAL_VIEW --> CNN_2D
    GAME_STATE --> MLP_STATE
    GRAPH_REPR --> GNN_SUBCELL
    GRAPH_REPR --> GNN_TILE
    GRAPH_REPR --> GNN_REGION
    
    GNN_SUBCELL --> DIFFPOOL
    GNN_TILE --> DIFFPOOL
    GNN_REGION --> DIFFPOOL
    
    CNN_3D --> ATTENTION
    CNN_2D --> ATTENTION
    MLP_STATE --> ATTENTION
    DIFFPOOL --> ATTENTION
    
    ATTENTION --> FUSION_LAYER
    FUSION_LAYER --> POLICY_HEAD
    FUSION_LAYER --> VALUE_HEAD
    
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
    classDef processing fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef training fill:#FF5722,stroke:#D84315,stroke-width:2px,color:#fff
    
    class FEATURE_EXTRACTION,PPO_AGENT primary
    class VISUAL_PROC,GRAPH_PROC,STATE_PROC,FUSION processing
    class TRAINING,EXPLORATION training
```

### 2. Heterogeneous Graph Transformer (HGT) Detail - PRIMARY

```mermaid
graph TB
    subgraph INPUT_GRAPH ["📊 Input: Heterogeneous Graph Data"]
        NODE_TYPES_INPUT["Node Types<br/>Grid Cells, Entities<br/>Hazards, Switches, Exits"]
        EDGE_TYPES_INPUT["Edge Types<br/>Movement: walk/jump/fall<br/>Functional: activate/trigger"]
        NODE_FEATURES["Node Features<br/>Position, Type, State<br/>Physics Properties"]
        EDGE_FEATURES["Edge Features<br/>Movement Cost<br/>Relationship Type"]
    end

    subgraph HGT_LAYERS ["🧠 Heterogeneous Graph Transformer Layers"]
        subgraph TYPE_SPECIFIC ["Type-Specific Processing"]
            NODE_PROJ["Node Type Projections<br/>Specialized Embeddings<br/>Per-type Linear Layers"]
            EDGE_PROJ["Edge Type Projections<br/>Relationship Embeddings<br/>Per-type Linear Layers"]
        end
        
        subgraph ATTENTION ["Multi-Head Attention"]
            TYPE_ATTN["Type-aware Attention<br/>Node-type × Edge-type<br/>Specialized Attention Heads"]
            ENTITY_ATTN["Entity-aware Attention<br/>Hazard Detection<br/>Goal-oriented Focus"]
        end
        
        subgraph AGGREGATION ["Feature Aggregation"]
            MSG_PASS["Message Passing<br/>Type-specific Messages<br/>Heterogeneous Aggregation"]
            GLOBAL_POOL["Global Pooling<br/>Mean-Max Pooling<br/>Graph-level Features"]
        end
    end

    subgraph OUTPUT ["🎯 HGT Output"]
        GRAPH_FEATURES["Graph Features<br/>Type-aware Representations<br/>Spatial Understanding"]
        ENTITY_FEATURES["Entity Features<br/>Hazard Awareness<br/>Goal Recognition"]
    end

    INPUT_GRAPH --> HGT_LAYERS
    HGT_LAYERS --> OUTPUT
```

### 3. Hierarchical Graph Neural Network Detail - SECONDARY

```mermaid
graph TB
    subgraph INPUT_GRAPH ["📊 Input: Hierarchical Graph Data"]
        SUBCELL_NODES["Sub-cell Nodes<br/>Fine-grained Spatial<br/>6px Resolution"]
        TILE_NODES["Tile Nodes<br/>Navigation Points<br/>24px Resolution"]
        REGION_NODES["Region Nodes<br/>Strategic Areas<br/>96px Resolution"]
        
        SUBCELL_EDGES["Sub-cell Edges<br/>Local Connectivity"]
        TILE_EDGES["Tile Edges<br/>Navigation Links"]
        REGION_EDGES["Region Edges<br/>Strategic Connections"]
        HIER_EDGES["Hierarchical Edges<br/>Cross-resolution Links"]
    end
    
    subgraph GNN_LAYERS ["🧠 Graph Neural Network Layers"]
        subgraph LEVEL_1 ["Level 1: Sub-cell Processing"]
            GCN_1["Graph Convolution<br/>Message Passing<br/>Local Feature Aggregation"]
            POOL_1["DiffPool Layer 1<br/>Learnable Coarsening<br/>Sub-cell → Tile"]
        end
        
        subgraph LEVEL_2 ["Level 2: Tile Processing"]
            GCN_2["Graph Convolution<br/>Navigation Reasoning<br/>Path Planning Features"]
            POOL_2["DiffPool Layer 2<br/>Learnable Coarsening<br/>Tile → Region"]
        end
        
        subgraph LEVEL_3 ["Level 3: Region Processing"]
            GCN_3["Graph Convolution<br/>Strategic Planning<br/>High-level Reasoning"]
            GLOBAL_POOL["Global Pooling<br/>Graph-level Features"]
        end
    end
    
    subgraph AUXILIARY_LOSSES ["📈 Auxiliary Training Objectives"]
        LINK_PRED["Link Prediction Loss<br/>Edge Reconstruction<br/>Structural Learning"]
        ENTROPY_REG["Entropy Regularization<br/>Assignment Diversity<br/>Pooling Quality"]
        ORTHO_REG["Orthogonality Loss<br/>Feature Diversity<br/>Representation Quality"]
    end
    
    subgraph OUTPUT_FEATURES ["📤 Output Features"]
        MULTI_SCALE["Multi-scale Features<br/>Hierarchical Representations"]
        GRAPH_EMBEDDING["Graph Embedding<br/>Structural Understanding"]
        ATTENTION_WEIGHTS["Attention Weights<br/>Importance Scores"]
    end
    
    %% Data Flow
    SUBCELL_NODES --> GCN_1
    SUBCELL_EDGES --> GCN_1
    GCN_1 --> POOL_1
    POOL_1 --> GCN_2
    
    TILE_NODES --> GCN_2
    TILE_EDGES --> GCN_2
    GCN_2 --> POOL_2
    POOL_2 --> GCN_3
    
    REGION_NODES --> GCN_3
    REGION_EDGES --> GCN_3
    HIER_EDGES --> GCN_1
    HIER_EDGES --> GCN_2
    HIER_EDGES --> GCN_3
    
    GCN_3 --> GLOBAL_POOL
    GLOBAL_POOL --> MULTI_SCALE
    
    POOL_1 --> LINK_PRED
    POOL_2 --> LINK_PRED
    POOL_1 --> ENTROPY_REG
    POOL_2 --> ENTROPY_REG
    GCN_1 --> ORTHO_REG
    GCN_2 --> ORTHO_REG
    GCN_3 --> ORTHO_REG
    
    MULTI_SCALE --> GRAPH_EMBEDDING
    MULTI_SCALE --> ATTENTION_WEIGHTS
```

### 3. Training Loop and Data Flow

```mermaid
graph LR
    subgraph EPISODE ["🎮 Episode Execution"]
        RESET["Environment Reset<br/>Initial State"]
        STEP["Environment Step<br/>Action → Observation"]
        COLLECT["Experience Collection<br/>Trajectory Building"]
    end
    
    subgraph PROCESSING ["🔄 Batch Processing"]
        BATCH_PREP["Batch Preparation<br/>Trajectory Segmentation"]
        ADVANTAGE["Advantage Estimation<br/>GAE Computation"]
        NORMALIZE["Normalization<br/>Observation Preprocessing"]
    end
    
    subgraph OPTIMIZATION ["⚡ Optimization"]
        FORWARD["Forward Pass<br/>Policy + Value Prediction"]
        LOSS_COMP["Loss Computation<br/>PPO + Auxiliary Losses"]
        BACKWARD["Backward Pass<br/>Gradient Computation"]
        UPDATE["Parameter Update<br/>Adam Optimizer"]
    end
    
    subgraph EVALUATION ["📊 Evaluation & Logging"]
        METRICS["Metrics Computation<br/>Rewards, Success Rate"]
        LOGGING["Logging & Visualization<br/>TensorBoard, Wandb"]
        CHECKPOINT["Model Checkpointing<br/>Best Model Saving"]
    end
    
    RESET --> STEP
    STEP --> COLLECT
    COLLECT --> BATCH_PREP
    BATCH_PREP --> ADVANTAGE
    ADVANTAGE --> NORMALIZE
    NORMALIZE --> FORWARD
    FORWARD --> LOSS_COMP
    LOSS_COMP --> BACKWARD
    BACKWARD --> UPDATE
    UPDATE --> METRICS
    METRICS --> LOGGING
    LOGGING --> CHECKPOINT
    CHECKPOINT --> RESET
    
    %% Styling
    classDef episode fill:#4CAF50,stroke:#2E7D32,stroke-width:2px,color:#fff
    classDef processing fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef optimization fill:#FF5722,stroke:#D84315,stroke-width:2px,color:#fff
    classDef evaluation fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    
    class EPISODE episode
    class PROCESSING processing
    class OPTIMIZATION optimization
    class EVALUATION evaluation
```

### 4. Feature Extractor Component Breakdown

```mermaid
graph TB
    subgraph INPUTS ["📥 Multimodal Inputs"]
        VISUAL["Visual Input<br/>Shape: (B, 1, 12, 84, 84)<br/>Temporal Frame Stack"]
        GLOBAL["Global Input<br/>Shape: (B, 1, 176, 100)<br/>Full Level View"]
        STATE["State Input<br/>Shape: (B, state_dim)<br/>Physics Vector"]
        GRAPH["Graph Input<br/>HierarchicalGraphData<br/>Multi-resolution Structure"]
    end
    
    subgraph VISUAL_BRANCH ["🖼️ Visual Processing Branch"]
        CONV3D_1["3D Conv Layer 1<br/>Kernel: (4,7,7)<br/>Temporal-Spatial Features"]
        CONV3D_2["3D Conv Layer 2<br/>Kernel: (3,5,5)<br/>Refined Features"]
        ADAPTIVE_POOL_3D["Adaptive 3D Pooling<br/>Fixed Output Size"]
        FLATTEN_3D["Flatten<br/>Feature Vector"]
    end
    
    subgraph GLOBAL_BRANCH ["🌍 Global Processing Branch"]
        CONV2D_1["2D Conv Layer 1<br/>Spatial Feature Extraction"]
        CONV2D_2["2D Conv Layer 2<br/>Higher-level Features"]
        ADAPTIVE_POOL_2D["Adaptive 2D Pooling<br/>Fixed Output Size"]
        FLATTEN_2D["Flatten<br/>Feature Vector"]
    end
    
    subgraph STATE_BRANCH ["📊 State Processing Branch"]
        MLP_1["Linear Layer 1<br/>State Embedding"]
        RELU_1["ReLU Activation"]
        MLP_2["Linear Layer 2<br/>Feature Refinement"]
        RELU_2["ReLU Activation"]
    end
    
    subgraph GRAPH_BRANCH ["🕸️ Graph Processing Branch"]
        GRAPH_CONV_1["Graph Conv 1<br/>Sub-cell Level"]
        GRAPH_CONV_2["Graph Conv 2<br/>Tile Level"]
        GRAPH_CONV_3["Graph Conv 3<br/>Region Level"]
        DIFFPOOL_1["DiffPool 1<br/>Sub-cell → Tile"]
        DIFFPOOL_2["DiffPool 2<br/>Tile → Region"]
        GRAPH_GLOBAL["Global Graph Pool<br/>Graph-level Features"]
    end
    
    subgraph FUSION_LAYER ["🔗 Multimodal Fusion"]
        CONCAT["Concatenation<br/>All Feature Vectors"]
        ATTENTION_WEIGHTS["Attention Mechanism<br/>Physics-aware Weighting"]
        FUSION_MLP["Fusion MLP<br/>Final Feature Integration"]
        OUTPUT_FEATURES["Output Features<br/>Unified Representation"]
    end
    
    %% Connections
    VISUAL --> CONV3D_1
    CONV3D_1 --> CONV3D_2
    CONV3D_2 --> ADAPTIVE_POOL_3D
    ADAPTIVE_POOL_3D --> FLATTEN_3D
    
    GLOBAL --> CONV2D_1
    CONV2D_1 --> CONV2D_2
    CONV2D_2 --> ADAPTIVE_POOL_2D
    ADAPTIVE_POOL_2D --> FLATTEN_2D
    
    STATE --> MLP_1
    MLP_1 --> RELU_1
    RELU_1 --> MLP_2
    MLP_2 --> RELU_2
    
    GRAPH --> GRAPH_CONV_1
    GRAPH --> GRAPH_CONV_2
    GRAPH --> GRAPH_CONV_3
    GRAPH_CONV_1 --> DIFFPOOL_1
    GRAPH_CONV_2 --> DIFFPOOL_2
    DIFFPOOL_1 --> GRAPH_CONV_2
    DIFFPOOL_2 --> GRAPH_CONV_3
    GRAPH_CONV_3 --> GRAPH_GLOBAL
    
    FLATTEN_3D --> CONCAT
    FLATTEN_2D --> CONCAT
    RELU_2 --> CONCAT
    GRAPH_GLOBAL --> CONCAT
    
    CONCAT --> ATTENTION_WEIGHTS
    ATTENTION_WEIGHTS --> FUSION_MLP
    FUSION_MLP --> OUTPUT_FEATURES
```

## Usage Instructions

### Generating Diagrams
1. Copy the desired Mermaid code block
2. Use any Mermaid-compatible tool:
   - **Online**: https://mermaid.live/
   - **VS Code**: Mermaid extension
   - **CLI**: `mmdc -i input.mmd -o output.png`

### Customization
- Modify colors by changing `classDef` statements
- Add/remove components as architecture evolves
- Update connections to reflect data flow changes
- Adjust layout by changing graph direction (`TB`, `LR`, etc.)

### Integration
- Include generated diagrams in documentation
- Reference in README files
- Use in presentations and papers
- Embed in technical specifications

This provides comprehensive visual documentation of the NPP-RL agent's hierarchical multimodal architecture.