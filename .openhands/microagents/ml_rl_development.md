---
agent: 'CodeActAgent'
triggers: ['machine learning', 'reinforcement learning', 'neural network', 'model', 'training', 'ppo', 'feature extractor', 'gnn', 'icm', 'pytorch', 'stable baselines']
---

# ML/RL Development Guidelines for NPP-RL

## Neural Network Architecture Standards

### Feature Extractor Design Patterns

**Always inherit from appropriate base classes and use clear naming:**

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class 3DFeatureExtractor(BaseFeaturesExtractor):
    """
    3D CNN feature extractor for temporal N++ observations.
    
    Based on research showing 37.9% improvement with:
    - 12-frame temporal stacking (Cobbe et al., 2020)
    - 3D convolutions for spatiotemporal features (Ji et al., 2013)
    - Multi-modal fusion (visual + physics state)
    """
    
    def __init__(self, observation_space: SpacesDict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # Implementation follows established patterns...
```

### Model Configuration Standards

**Store hyperparameters in dedicated modules with research justification:**

```python
# npp_rl/agents/hyperparameters/ppo_hyperparameters.py
HYPERPARAMETERS = {
    # Research-backed parameter choices
    "n_steps": 1024,  # Increased from 512 (scaling law research)
    "batch_size": 256,  # Must be <= n_steps for proper batching
    "gamma": 0.999,  # High for long-horizon N++ levels
    "learning_rate": get_linear_fn(3e-4, 1e-6),  # Linear decay schedule
    
    # Optuna-tuned values
    "clip_range": 0.3892025290555702,
    "ent_coef": 0.002720504247658009,
    "gae_lambda": 0.998801,
}

# Network architecture scaling
NET_ARCH_SIZE = [256, 256, 128]  # Scaled for complex multi-modal observations
```

## Reinforcement Learning Patterns

### PPO Agent Creation Pattern

```python
def create_enhanced_ppo_agent(env) -> PPO:
    """Create PPO agent with NPP-RL optimizations."""
    

    features_extractor_class = 3DFeatureExtractor

    policy_kwargs = {
        'features_extractor_class': features_extractor_class,
        'features_extractor_kwargs': {'features_dim': 512},
        'net_arch': {
            'pi': NET_ARCH_SIZE,  # Policy network
            'vf': NET_ARCH_SIZE   # Value network
        },
        'activation_fn': nn.ReLU,
    }
    
    model = PPO(
        policy="MultiInputPolicy",  # Required for multi-modal observations
        env=env,
        policy_kwargs=policy_kwargs,
        **HYPERPARAMETERS,
        tensorboard_log=log_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return model
```

### Training Pipeline Structure

```python
def train_agent_with_monitoring(
    env, 
    total_timesteps: int,
    eval_freq: int = 10000,
    checkpoint_freq: int = 50000
) -> PPO:
    """Standard training pipeline with proper monitoring."""
    
    # Create model
    model = create_enhanced_ppo_agent(env)
    
    # Setup comprehensive callbacks
    callbacks = [
        EvalCallback(
            eval_env=create_eval_env(),
            eval_freq=eval_freq,
            best_model_save_path=f"{log_dir}/best_model",
            deterministic=True,
            render=False
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=f"{log_dir}/checkpoints"
        ),
        NPPMetricsCallback(),  # Game-specific metrics
        TensorboardLoggingCallback()
    ]
    
    # Train with progress tracking
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    return model
```

## Graph Neural Networks (Phase 2)

### GNN Architecture for N++ Levels

```python
class GraphEncoder(nn.Module):
    """
    GraphSAGE-style encoder for N++ level structure.
    
    Converts level geometry and entity relationships into
    graph representations for structural understanding.
    """
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Message passing layers
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        
        # GraphSAGE layers with masking support
        self.gnn_layers = nn.ModuleList([
            MaskedGraphSAGELayer(hidden_dim, hidden_dim)
            for _ in range(3)  # 3 layers typical for good performance
        ])
        
        # Global pooling for graph-level features
        self.global_pool = GlobalMeanMaxPool()
    
    def forward(self, graph_obs):
        """Process graph observations with proper masking."""
        # Handle variable-sized graphs through masking
        node_features = graph_obs['node_features']  # [batch, max_nodes, node_dim]
        edge_index = graph_obs['edge_index']        # [batch, 2, max_edges]
        node_mask = graph_obs['node_mask']          # [batch, max_nodes]
        edge_mask = graph_obs['edge_mask']          # [batch, max_edges]
        
        # Encode features
        h_nodes = self.node_encoder(node_features)
        
        # Message passing with masking
        for gnn_layer in self.gnn_layers:
            h_nodes = gnn_layer(h_nodes, edge_index, node_mask, edge_mask)
        
        # Global graph representation
        graph_embedding = self.global_pool(h_nodes, node_mask)
        
        return graph_embedding
```

### Graph Observation Integration

```python
def create_multimodal_feature_extractor(observation_space, use_graph: bool = True):
    """Create feature extractor combining visual, state, and graph observations."""
    
    class MultiModalExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=512):
            super().__init__(observation_space, features_dim)
            
            # Visual encoders
            self.cnn_player = build_3d_cnn(observation_space['player_frame'].shape)
            self.cnn_global = build_2d_cnn(observation_space['global_view'].shape)
            
            # State encoder
            state_dim = observation_space['game_state'].shape[0]
            self.mlp_state = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            
            # Graph encoder (optional)
            if use_graph and 'graph_obs' in observation_space.spaces:
                self.graph_encoder = GraphEncoder(
                    node_feature_dim=67,  # N++ level node features
                    edge_feature_dim=9    # N++ level edge features
                )
                fusion_input_dim = 512 + 256 + 128 + 256  # All modalities
            else:
                self.graph_encoder = None
                fusion_input_dim = 512 + 256 + 128  # Visual + state only
            
            # Fusion network
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, features_dim)
            )
        
        def forward(self, observations):
            # Process each modality
            player_features = self.cnn_player(observations['player_frame'])
            global_features = self.cnn_global(observations['global_view'])
            state_features = self.mlp_state(observations['game_state'])
            
            # Combine features
            features = [player_features, global_features, state_features]
            
            if self.graph_encoder is not None and 'graph_obs' in observations:
                graph_features = self.graph_encoder(observations['graph_obs'])
                features.append(graph_features)
            
            # Fuse all modalities
            combined = torch.cat(features, dim=1)
            output = self.fusion(combined)
            
            return output
    
    return MultiModalExtractor
```

## Intrinsic Motivation and Exploration

### ICM (Intrinsic Curiosity Module) Implementation

```python
class ICMNetwork(nn.Module):
    """
    Intrinsic Curiosity Module based on Pathak et al. (2017).
    
    Provides exploration bonuses in sparse reward environments
    by learning to predict state transitions.
    """
    
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        
        # Inverse model: predicts action from state transition
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Forward model: predicts next state features
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
    
    def compute_intrinsic_reward(self, state_features, next_state_features, actions):
        """Compute intrinsic reward from forward model prediction error."""
        # One-hot encode actions
        actions_onehot = F.one_hot(actions, num_classes=6).float()
        
        # Forward model prediction
        forward_input = torch.cat([state_features, actions_onehot], dim=1)
        predicted_next_state = self.forward_model(forward_input)
        
        # Intrinsic reward is prediction error
        prediction_error = F.mse_loss(
            predicted_next_state, 
            next_state_features.detach(), 
            reduction='none'
        ).mean(dim=1)
        
        return prediction_error.detach()
    
    def compute_losses(self, state_features, next_state_features, actions):
        """Compute ICM training losses."""
        batch_size = state_features.shape[0]
        actions_onehot = F.one_hot(actions, num_classes=6).float()
        
        # Forward model loss
        forward_input = torch.cat([state_features, actions_onehot], dim=1)
        predicted_next_state = self.forward_model(forward_input)
        forward_loss = F.mse_loss(predicted_next_state, next_state_features)
        
        # Inverse model loss
        inverse_input = torch.cat([state_features, next_state_features], dim=1)
        predicted_actions = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(predicted_actions, actions)
        
        return {
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'total_icm_loss': 0.9 * forward_loss + 0.1 * inverse_loss
        }
```

### Exploration Management

```python
class AdaptiveExplorationManager:
    """
    Combines multiple exploration strategies with adaptive weighting.
    
    Strategies:
    - ICM-based curiosity
    - Count-based novelty detection  
    - Adaptive scaling based on training progress
    """
    
    def __init__(self, icm_weight: float = 0.1, novelty_weight: float = 0.05):
        self.icm = None  # Initialized later
        self.novelty_detector = CountBasedNoveltyDetector()
        self.icm_weight = icm_weight
        self.novelty_weight = novelty_weight
        
        # Adaptive scaling
        self.performance_history = deque(maxlen=1000)
        self.exploration_decay = 0.999
    
    def initialize_curiosity_module(self, feature_dim: int, action_dim: int):
        """Initialize ICM after knowing feature dimensions."""
        self.icm = ICMNetwork(feature_dim, action_dim)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=1e-3)
    
    def compute_exploration_reward(self, observations, actions, next_observations):
        """Compute combined exploration bonus."""
        exploration_bonus = torch.zeros(len(observations))
        
        if self.icm is not None:
            # Extract features for ICM
            state_features = self.extract_features(observations)
            next_state_features = self.extract_features(next_observations)
            
            # ICM curiosity bonus
            icm_bonus = self.icm.compute_intrinsic_reward(
                state_features, next_state_features, actions
            )
            exploration_bonus += self.icm_weight * icm_bonus
        
        # Count-based novelty bonus
        novelty_bonus = self.novelty_detector.compute_novelty(observations)
        exploration_bonus += self.novelty_weight * novelty_bonus
        
        # Adaptive scaling based on performance
        adaptive_scale = self.get_adaptive_scale()
        exploration_bonus *= adaptive_scale
        
        return exploration_bonus.clamp(max=1.0)  # Clip to prevent instability
```

## Behavioral Cloning Integration

### BC Dataset and Pretraining

```python
class NPPBehavioralCloningDataset(Dataset):
    """Dataset for loading human N++ replay data."""
    
    def __init__(self, data_dir: str, quality_threshold: float = 0.8):
        self.data_dir = Path(data_dir)
        self.quality_threshold = quality_threshold
        self.episodes = self.load_episodes()
    
    def load_episodes(self):
        """Load and filter high-quality episodes."""
        episodes = []
        
        for replay_file in self.data_dir.glob("*.npz"):
            data = np.load(replay_file)
            
            # Quality filtering
            success_rate = data.get('success_rate', 0)
            completion_time = data.get('completion_time', float('inf'))
            
            if success_rate >= self.quality_threshold:
                episodes.append({
                    'observations': data['observations'],
                    'actions': data['actions'],
                    'quality_score': success_rate
                })
        
        return episodes

def pretrain_with_behavioral_cloning(model: PPO, bc_dataset, epochs: int = 10):
    """Pretrain policy using behavioral cloning."""
    
    bc_dataloader = DataLoader(bc_dataset, batch_size=64, shuffle=True)
    bc_optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    
    model.policy.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in bc_dataloader:
            observations = batch['observations']
            expert_actions = batch['actions']
            
            # Forward pass through policy
            features = model.policy.extract_features(observations)
            action_logits = model.policy.mlp_extractor.policy_net(features)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            
            # BC loss (negative log likelihood)
            bc_loss = -action_dist.log_prob(expert_actions).mean()
            
            # Backward pass
            bc_optimizer.zero_grad()
            bc_loss.backward()
            bc_optimizer.step()
            
            total_loss += bc_loss.item()
        
        avg_loss = total_loss / len(bc_dataloader)
        print(f"BC Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    model.policy.eval()
    return model
```

## Model Testing Patterns

### Architecture Validation Tests

```python
def test_feature_extractor_forward_pass():
    """Test that feature extractors handle multi-modal observations correctly."""
    # Create realistic observation space
    obs_space = SpacesDict({
        'player_frame': Box(low=0, high=255, shape=(84, 84, 12), dtype=np.uint8),
        'global_view': Box(low=0, high=255, shape=(176, 100, 1), dtype=np.uint8),
        'game_state': Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
    })
    
    # Test 3D feature extractor
    extractor = 3DFeatureExtractor(obs_space, features_dim=512)
    
    # Create realistic batch
    batch_size = 4
    mock_obs = {
        'player_frame': torch.randint(0, 256, (batch_size, 84, 84, 12), dtype=torch.uint8),
        'global_view': torch.randint(0, 256, (batch_size, 176, 100, 1), dtype=torch.uint8),
        'game_state': torch.randn(batch_size, 32)
    }
    
    # Test forward pass
    with torch.no_grad():
        features = extractor(mock_obs)
    
    # Validate output
    assert features.shape == (batch_size, 512), f"Expected shape (4, 512), got {features.shape}"
    assert torch.all(torch.isfinite(features)), "Features contain NaN or inf values"
    assert torch.norm(features) > 0, "Features are all zeros"

def test_ppo_integration():
    """Test that custom feature extractors work with SB3 PPO."""
    env = DummyVecEnv([lambda: create_test_env()])
    
    # Create agent with custom feature extractor
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            'features_extractor_class': 3DFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 512}
        },
        n_steps=64,  # Small for testing
        batch_size=32
    )
    
    # Test learning step doesn't crash
    model.learn(total_timesteps=128)
    
    # Test prediction works
    obs = env.reset()
    action, _ = model.predict(obs)
    assert action is not None
```

## Performance Optimization Guidelines

### Memory Management
```python
# For large batch training, use gradient accumulation
def train_with_large_effective_batch(model, data_loader, effective_batch_size=1024):
    """Train with large effective batch size using gradient accumulation."""
    accumulation_steps = effective_batch_size // data_loader.batch_size
    
    model.zero_grad()
    for i, batch in enumerate(data_loader):
        loss = compute_loss(model, batch)
        loss = loss / accumulation_steps  # Scale loss
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            model.optimizer.step()
            model.zero_grad()
```

### GPU Optimization
```python
# Enable mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

@autocast()
def forward_pass(model, observations):
    return model(observations)

# Use in training loop
with autocast():
    loss = compute_loss(model, batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This guide ensures all ML/RL development follows established patterns, uses proper PyTorch/SB3 integration, and maintains the high standards required for reproducible research-quality implementations.
