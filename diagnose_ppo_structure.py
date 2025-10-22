#!/usr/bin/env python3
"""
Diagnose the actual PPO model structure to understand feature extractor organization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from npp_rl.agents.hierarchical_ppo import HierarchicalActorCriticPolicy
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor

def create_dummy_env():
    """Create a minimal dummy environment for testing."""
    from gymnasium import spaces
    import numpy as np
    
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict({
                'player_frame': spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8),
                'entity_positions': spaces.Box(-1, 1, (6,), dtype=np.float32),
                'reachability_map': spaces.Box(0, 1, (21, 21, 3), dtype=np.float32),
            })
            self.action_space = spaces.Discrete(5)
        
        def reset(self, **kwargs):
            obs = {
                'player_frame': np.zeros((84, 84, 3), dtype=np.uint8),
                'entity_positions': np.zeros(6, dtype=np.float32),
                'reachability_map': np.zeros((21, 21, 3), dtype=np.float32),
            }
            return obs, {}
        
        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}
    
    return DummyEnv()

def analyze_model_structure():
    """Analyze the PPO model structure with HierarchicalActorCriticPolicy."""
    
    print("="*60)
    print("Creating HierarchicalPPO Model")
    print("="*60)
    
    env = create_dummy_env()
    
    # Create policy kwargs
    policy_kwargs = {
        'features_extractor_class': ConfigurableMultimodalExtractor,
        'features_extractor_kwargs': {
            'player_frame_cnn_channels': [32, 64, 64],
            'reachability_cnn_channels': [16, 32],
            'use_batch_norm': True,
        },
        'net_arch': [256, 256, 128],
        'high_level_update_frequency': 50,
        'max_steps_per_subtask': 500,
        'use_icm': True,
    }
    
    # Create model with hierarchical policy
    model = PPO(
        policy=HierarchicalActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        device='cpu',
        verbose=0,
    )
    
    print("✓ Model created successfully\n")
    
    # Analyze policy structure
    print("="*60)
    print("Policy State Dict Analysis")
    print("="*60)
    
    policy_state = model.policy.state_dict()
    all_keys = list(policy_state.keys())
    
    print(f"Total keys: {len(all_keys)}\n")
    
    # Categorize keys
    categories = {
        'mlp_extractor.features_extractor.*': [],
        'features_extractor.*': [],
        'pi_features_extractor.*': [],
        'vf_features_extractor.*': [],
        'mlp_extractor.high_level_policy.*': [],
        'mlp_extractor.low_level_policy.*': [],
        'mlp_extractor.current_subtask': [],
        'action_net.*': [],
        'value_net.*': [],
        'other': [],
    }
    
    for key in all_keys:
        categorized = False
        
        if 'mlp_extractor.features_extractor.' in key:
            categories['mlp_extractor.features_extractor.*'].append(key)
            categorized = True
        if key.startswith('features_extractor.') and 'mlp_extractor' not in key:
            categories['features_extractor.*'].append(key)
            categorized = True
        if key.startswith('pi_features_extractor.'):
            categories['pi_features_extractor.*'].append(key)
            categorized = True
        if key.startswith('vf_features_extractor.'):
            categories['vf_features_extractor.*'].append(key)
            categorized = True
        if 'mlp_extractor.high_level_policy.' in key:
            categories['mlp_extractor.high_level_policy.*'].append(key)
            categorized = True
        if 'mlp_extractor.low_level_policy.' in key:
            categories['mlp_extractor.low_level_policy.*'].append(key)
            categorized = True
        if key == 'mlp_extractor.current_subtask':
            categories['mlp_extractor.current_subtask'].append(key)
            categorized = True
        if key.startswith('action_net.'):
            categories['action_net.*'].append(key)
            categorized = True
        if key.startswith('value_net.'):
            categories['value_net.*'].append(key)
            categorized = True
        
        if not categorized:
            categories['other'].append(key)
    
    # Print category summary
    for category, keys in categories.items():
        if keys:
            print(f"\n{category}: {len(keys)} keys")
            print(f"  First 5: {keys[:5]}")
    
    # Check for overlaps
    print("\n" + "="*60)
    print("Overlap Analysis")
    print("="*60)
    
    has_hierarchical = len(categories['mlp_extractor.features_extractor.*']) > 0
    has_shared = len(categories['features_extractor.*']) > 0
    has_separate = len(categories['pi_features_extractor.*']) > 0 or len(categories['vf_features_extractor.*']) > 0
    
    print(f"Has hierarchical extractor: {has_hierarchical}")
    print(f"Has shared extractor: {has_shared}")
    print(f"Has separate extractors: {has_separate}")
    
    if has_hierarchical and has_shared:
        print("\n⚠️  WARNING: Model has BOTH hierarchical AND shared extractors!")
        print("This will cause BC weights to be mapped to both, creating 2x duplication.")
    
    if has_hierarchical and has_separate:
        print("\n⚠️  WARNING: Model has BOTH hierarchical AND separate extractors!")
        print("This will cause BC weights to be mapped to multiple locations.")
    
    # Check if shared/separate are actually references to hierarchical
    print("\n" + "="*60)
    print("Reference Analysis")
    print("="*60)
    
    # Check if features_extractor is a reference to mlp_extractor.features_extractor
    if has_hierarchical and has_shared:
        # Get actual objects
        if hasattr(model.policy, 'features_extractor') and hasattr(model.policy, 'mlp_extractor'):
            if hasattr(model.policy.mlp_extractor, 'features_extractor'):
                is_same = model.policy.features_extractor is model.policy.mlp_extractor.features_extractor
                print(f"policy.features_extractor is policy.mlp_extractor.features_extractor: {is_same}")
                
                if is_same:
                    print("\n✓ They are the SAME object (reference)")
                    print("  This means both key prefixes point to the same weights")
                    print("  We should map BC weights to ONLY ONE of them to avoid duplication")
                else:
                    print("\n✗ They are DIFFERENT objects")
                    print("  This is unexpected and problematic")
    
    return model, categories

if __name__ == "__main__":
    try:
        model, categories = analyze_model_structure()
        
        print("\n" + "="*60)
        print("Recommendation")
        print("="*60)
        
        has_hierarchical = len(categories['mlp_extractor.features_extractor.*']) > 0
        has_shared = len(categories['features_extractor.*']) > 0
        has_separate = len(categories['pi_features_extractor.*']) > 0 or len(categories['vf_features_extractor.*']) > 0
        
        if has_hierarchical:
            print("✓ Use ONLY hierarchical mapping:")
            print("  BC: feature_extractor.* → PPO: mlp_extractor.features_extractor.*")
            if has_shared:
                print("\n  Ignore 'features_extractor.*' keys (they are references)")
            if has_separate:
                print("\n  Ignore separate extractor keys (not used in hierarchical)")
        elif has_shared:
            print("✓ Use ONLY shared mapping:")
            print("  BC: feature_extractor.* → PPO: features_extractor.*")
        elif has_separate:
            print("✓ Use ONLY separate mapping:")
            print("  BC: feature_extractor.* → PPO: pi_features_extractor.* and vf_features_extractor.*")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
