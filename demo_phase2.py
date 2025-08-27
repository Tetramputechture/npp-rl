"""
Phase 2 Demo Script

This script demonstrates how to use the Phase 2 features including ICM,
graph observations, and enhanced training capabilities.
"""

import torch
import numpy as np
from pathlib import Path

from npp_rl.config.phase2_config import Phase2Config, get_config_presets
from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.models.gnn import create_graph_encoder
from npp_rl.feature_extractors import create_feature_extractor
from npp_rl.eval.exploration_metrics import ExplorationMetrics
from npp_rl.data.bc_dataset import create_mock_replay_data, create_bc_dataloader
from nclone.graph.graph_builder import GraphBuilder


def demo_icm():
    """Demonstrate ICM functionality."""
    print("🧠 ICM (Intrinsic Curiosity Module) Demo")
    print("=" * 50)
    
    # Create ICM network
    icm = ICMNetwork(
        feature_dim=512,
        action_dim=6,
        hidden_dim=256,
        eta=0.01,  # Intrinsic reward scaling
        lambda_inv=0.1,  # Inverse model weight
        lambda_fwd=0.9   # Forward model weight
    )
    
    # Create trainer
    trainer = ICMTrainer(icm, learning_rate=1e-3)
    
    print(f"✅ Created ICM with {sum(p.numel() for p in icm.parameters()):,} parameters")
    
    # Simulate training step
    batch_size = 32
    current_features = torch.randn(batch_size, 512)
    next_features = torch.randn(batch_size, 512)
    actions = torch.randint(0, 6, (batch_size,))
    
    # Update ICM
    stats = trainer.update(current_features, next_features, actions)
    
    print(f"📊 ICM Training Stats:")
    print(f"   Inverse Loss: {stats['inverse_loss']:.4f}")
    print(f"   Forward Loss: {stats['forward_loss']:.4f}")
    print(f"   Total Loss: {stats['total_loss']:.4f}")
    print(f"   Mean Intrinsic Reward: {stats['mean_intrinsic_reward']:.4f}")
    
    # Get intrinsic rewards
    rewards = trainer.get_intrinsic_reward(current_features, next_features, actions)
    print(f"🎁 Intrinsic Rewards: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
    print()


def demo_graph_observations():
    """Demonstrate graph observation functionality."""
    print("🕸️ Graph Observations Demo")
    print("=" * 50)
    
    # Create graph builder
    builder = GraphBuilder()
    
    print(f"✅ Created GraphBuilder")
    print(f"   Node feature dim: {builder.node_feature_dim}")
    print(f"   Edge feature dim: {builder.edge_feature_dim}")
    
    # Create mock level data
    level_data = {
        'tiles': np.zeros((23, 42), dtype=np.int32),
        'width': 42,
        'height': 23
    }
    
    # Add some entities
    entities = [
        {'type': 'exit_switch', 'x': 100, 'y': 200, 'active': False, 'state': 0.0},
        {'type': 'exit_door', 'x': 300, 'y': 400, 'active': True, 'state': 0.0},
        {'type': 'gold', 'x': 200, 'y': 300, 'active': True, 'state': 1.0}
    ]
    
    ninja_position = (150, 250)
    
    # Build graph
    graph_data = builder.build_graph(level_data, ninja_position, entities)
    
    print(f"📊 Graph Statistics:")
    print(f"   Nodes: {graph_data.num_nodes}")
    print(f"   Edges: {graph_data.num_edges}")
    print(f"   Node features shape: {graph_data.node_features.shape}")
    print(f"   Edge features shape: {graph_data.edge_features.shape}")
    
    # Create GNN encoder
    gnn = create_graph_encoder(
        node_feature_dim=builder.node_feature_dim,
        edge_feature_dim=builder.edge_feature_dim,
        hidden_dim=128,
        num_layers=3,
        output_dim=256
    )
    
    print(f"✅ Created GNN with {sum(p.numel() for p in gnn.parameters()):,} parameters")
    
    # Process graph with GNN
    batch_size = 4
    graph_obs = {
        'graph_node_feats': torch.from_numpy(graph_data.node_features).unsqueeze(0).repeat(batch_size, 1, 1),
        'graph_edge_index': torch.from_numpy(graph_data.edge_index).unsqueeze(0).repeat(batch_size, 1, 1),
        'graph_edge_feats': torch.from_numpy(graph_data.edge_features).unsqueeze(0).repeat(batch_size, 1, 1),
        'graph_node_mask': torch.from_numpy(graph_data.node_mask).unsqueeze(0).repeat(batch_size, 1),
        'graph_edge_mask': torch.from_numpy(graph_data.edge_mask).unsqueeze(0).repeat(batch_size, 1)
    }
    
    with torch.no_grad():
        graph_embedding = gnn(graph_obs)
    
    print(f"🧠 Graph embedding shape: {graph_embedding.shape}")
    print()


def demo_exploration_metrics():
    """Demonstrate exploration metrics functionality."""
    print("🗺️ Exploration Metrics Demo")
    print("=" * 50)
    
    # Create exploration metrics tracker
    metrics = ExplorationMetrics(
        grid_width=42,
        grid_height=23,
        cell_size=24,
        window_size=100
    )
    
    print("✅ Created ExplorationMetrics tracker")
    
    # Simulate an episode with interesting movement pattern
    metrics.reset_episode()
    
    # Simulate spiral movement pattern
    positions = []
    center_x, center_y = 500, 300
    
    for i in range(50):
        angle = i * 0.3
        radius = i * 2
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        positions.append((x, y))
        
        # Decreasing intrinsic reward (agent gets less curious about familiar areas)
        intrinsic_reward = 0.5 * np.exp(-i * 0.05)
        metrics.update_step((x, y), intrinsic_reward)
    
    # End episode
    episode_metrics = metrics.end_episode(success=True)
    
    print(f"📊 Episode Metrics:")
    print(f"   Coverage: {episode_metrics['coverage']:.3f}")
    print(f"   Unique cells visited: {episode_metrics['unique_cells_visited']}")
    print(f"   Visitation entropy: {episode_metrics['visitation_entropy']:.3f}")
    print(f"   Mean intrinsic reward: {episode_metrics['mean_intrinsic_reward']:.4f}")
    print(f"   Episode length: {episode_metrics['episode_length']}")
    print(f"   Success: {bool(episode_metrics['success'])}")
    
    # Get rolling metrics
    rolling = metrics.get_rolling_metrics()
    print(f"🔄 Rolling Metrics:")
    print(f"   Rolling coverage: {rolling['rolling_coverage']:.3f}")
    print(f"   Rolling entropy: {rolling['rolling_entropy']:.3f}")
    print()


def demo_behavioral_cloning():
    """Demonstrate behavioral cloning functionality."""
    print("🎭 Behavioral Cloning Demo")
    print("=" * 50)
    
    # Create temporary directory for mock data
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print("✅ Creating mock replay data...")
        
        # Create mock replay data
        create_mock_replay_data(
            output_dir=temp_dir,
            num_episodes=20,
            episode_length=100
        )
        
        print(f"📁 Created 20 mock episodes in {temp_dir}")
        
        # Define observation space (simplified)
        from gymnasium.spaces import Box, Dict as SpacesDict
        observation_space = {
            'player_frame': Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8),
            'global_view': Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
            'game_state': Box(low=-1, high=1, shape=(32,), dtype=np.float32)
        }
        
        from gymnasium.spaces import Discrete
        action_space = Discrete(6)
        
        # Create dataloader
        dataloader = create_bc_dataloader(
            data_dir=temp_dir,
            observation_space=observation_space,
            action_space=action_space,
            batch_size=8,
            max_episodes=10
        )
        
        print(f"✅ Created BC dataloader")
        
        # Get dataset statistics
        stats = dataloader.dataset.get_statistics()
        print(f"📊 Dataset Statistics:")
        print(f"   Episodes: {stats['num_episodes']}")
        print(f"   Transitions: {stats['num_transitions']}")
        print(f"   Avg episode length: {stats['episode_length']['mean']:.1f}")
        print(f"   Action distribution: {stats['action_distribution']}")
        
        # Test loading a batch
        for batch_idx, (observations, actions) in enumerate(dataloader):
            print(f"🎯 Loaded batch {batch_idx + 1}:")
            print(f"   Batch size: {actions.shape[0]}")
            print(f"   Observation keys: {list(observations.keys())}")
            print(f"   Action range: {actions.min().item()}-{actions.max().item()}")
            
            if batch_idx >= 2:  # Just show a few batches
                break
    
    print()


def demo_configuration():
    """Demonstrate configuration system."""
    print("⚙️ Configuration System Demo")
    print("=" * 50)
    
    # Show available presets
    presets = get_config_presets()
    print(f"✅ Available configuration presets:")
    for name, config in presets.items():
        icm_status = "✓" if config.icm.enabled else "✗"
        graph_status = "✓" if config.graph.enabled else "✗"
        bc_status = "✓" if config.bc.enabled else "✗"
        print(f"   {name:12} - ICM:{icm_status} Graph:{graph_status} BC:{bc_status}")
    
    # Create custom configuration
    config = Phase2Config()
    config.icm.enabled = True
    config.icm.alpha = 0.15  # Higher intrinsic reward weight
    config.graph.enabled = True
    config.graph.num_layers = 4  # Deeper GNN
    config.bc.enabled = False
    config.experiment_name = "custom_demo"
    config.total_timesteps = 500_000
    
    print(f"\n📝 Custom Configuration:")
    print(f"   Experiment: {config.experiment_name}")
    print(f"   ICM enabled: {config.icm.enabled} (alpha={config.icm.alpha})")
    print(f"   Graph enabled: {config.graph.enabled} (layers={config.graph.num_layers})")
    print(f"   BC enabled: {config.bc.enabled}")
    print(f"   Total timesteps: {config.total_timesteps:,}")
    
    # Validate configuration
    from npp_rl.config.phase2_config import validate_config
    messages = validate_config(config)
    if messages:
        print(f"⚠️ Validation messages:")
        for msg in messages:
            print(f"   {msg}")
    else:
        print("✅ Configuration is valid")
    
    print()


def demo_integration():
    """Demonstrate integration of multiple components."""
    print("🔗 Component Integration Demo")
    print("=" * 50)
    
    # Create configuration
    config = Phase2Config()
    config.icm.enabled = True
    config.graph.enabled = True
    
    print("✅ Created Phase 2 configuration")
    
    # Create mock observation space
    from gymnasium.spaces import Box, Dict as SpacesDict
    from nclone.graph.graph_builder import N_MAX_NODES, E_MAX_EDGES
    
    observation_space = SpacesDict({
        'player_frame': Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8),
        'global_view': Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
        'game_state': Box(low=-1, high=1, shape=(32,), dtype=np.float32),
        'graph_node_feats': Box(low=-1, high=1, shape=(N_MAX_NODES, 67), dtype=np.float32),
        'graph_edge_index': Box(low=0, high=N_MAX_NODES-1, shape=(2, E_MAX_EDGES), dtype=np.int32),
        'graph_edge_feats': Box(low=-1, high=1, shape=(E_MAX_EDGES, 9), dtype=np.float32),
        'graph_node_mask': Box(low=0, high=1, shape=(N_MAX_NODES,), dtype=np.float32),
        'graph_edge_mask': Box(low=0, high=1, shape=(E_MAX_EDGES,), dtype=np.float32)
    })
    
    # Create multimodal feature extractor
    extractor = create_feature_extractor(
        observation_space=observation_space,
        features_dim=512,
        use_graph_obs=True
    )
    
    print(f"✅ Created multimodal feature extractor with {sum(p.numel() for p in extractor.parameters()):,} parameters")
    
    # Create ICM
    icm = ICMNetwork(feature_dim=512, action_dim=6)
    icm_trainer = ICMTrainer(icm)
    
    print("✅ Created ICM trainer")
    
    # Create exploration metrics
    metrics = ExplorationMetrics()
    
    print("✅ Created exploration metrics tracker")
    
    # Simulate training loop
    print("\n🔄 Simulating training steps...")
    
    batch_size = 4
    for step in range(5):
        # Create mock observations
        observations = {
            'player_frame': torch.randint(0, 256, (batch_size, 64, 64, 4), dtype=torch.uint8),
            'global_view': torch.randint(0, 256, (batch_size, 128, 128, 1), dtype=torch.uint8),
            'game_state': torch.randn(batch_size, 32),
            'graph_node_feats': torch.randn(batch_size, N_MAX_NODES, 67),
            'graph_edge_index': torch.randint(0, N_MAX_NODES, (batch_size, 2, E_MAX_EDGES)),
            'graph_edge_feats': torch.randn(batch_size, E_MAX_EDGES, 9),
            'graph_node_mask': torch.ones(batch_size, N_MAX_NODES),
            'graph_edge_mask': torch.ones(batch_size, E_MAX_EDGES)
        }
        
        # Extract features
        with torch.no_grad():
            features = extractor(observations)
        
        # Update ICM (simulate consecutive states)
        if step > 0:
            actions = torch.randint(0, 6, (batch_size,))
            stats = icm_trainer.update(prev_features, features, actions)
            
            # Update exploration metrics
            for i in range(batch_size):
                pos = (100 + step * 50 + i * 10, 100 + step * 30)
                metrics.update_step(pos, stats['mean_intrinsic_reward'])
        
        prev_features = features
        
        print(f"   Step {step + 1}: Features shape {features.shape}")
    
    # End episode and get metrics
    episode_metrics = metrics.end_episode(success=True)
    
    print(f"\n📊 Final Integration Results:")
    print(f"   Feature extraction: ✅")
    print(f"   ICM training: ✅")
    print(f"   Exploration tracking: ✅")
    print(f"   Coverage achieved: {episode_metrics['coverage']:.3f}")
    print(f"   Mean intrinsic reward: {episode_metrics['mean_intrinsic_reward']:.4f}")
    
    print()


def main():
    """Run all Phase 2 demos."""
    print("🚀 Phase 2 Feature Demonstration")
    print("=" * 60)
    print("This demo showcases the key Phase 2 components and their integration.")
    print("=" * 60)
    print()
    
    try:
        demo_icm()
        demo_graph_observations()
        demo_exploration_metrics()
        demo_behavioral_cloning()
        demo_configuration()
        demo_integration()
        
        print("🎉 Phase 2 Demo Complete!")
        print("\nNext steps:")
        print("1. Run 'python train_phase2.py --preset icm_only' for ICM training")
        print("2. Run 'python bc_pretrain.py --create_mock_data' for BC pretraining")
        print("3. Run 'python test_phase2_simple.py' for comprehensive testing")
        print("4. Check docs/PHASE_2_IMPLEMENTATION_COMPLETE.md for full documentation")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()