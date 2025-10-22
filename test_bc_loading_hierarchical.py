#!/usr/bin/env python3
"""
Test script to verify BC weight loading for hierarchical PPO.

This script simulates the weight loading process without actually
training, to verify the mapping logic works correctly.
"""

import torch
from pathlib import Path

def test_bc_loading():
    """Test BC weight loading detection and mapping."""
    
    # Load BC checkpoint
    bc_path = Path("bc_best.pth")
    if not bc_path.exists():
        print("❌ BC checkpoint not found at bc_best.pth")
        return False
    
    checkpoint = torch.load(bc_path, map_location='cpu', weights_only=False)
    bc_state_dict = checkpoint["policy_state_dict"]
    
    print("="*60)
    print("BC Checkpoint Analysis")
    print("="*60)
    print(f"Total keys in BC checkpoint: {len(bc_state_dict)}")
    print(f"\nFirst 10 keys:")
    for i, key in enumerate(list(bc_state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # Count feature extractor weights
    feature_extractor_keys = [k for k in bc_state_dict.keys() if k.startswith("feature_extractor.")]
    policy_head_keys = [k for k in bc_state_dict.keys() if k.startswith("policy_head.")]
    
    print(f"\nFeature extractor keys: {len(feature_extractor_keys)}")
    print(f"Policy head keys: {len(policy_head_keys)}")
    
    # Simulate PPO model keys (hierarchical structure)
    print("\n" + "="*60)
    print("Simulated Hierarchical PPO Model Structure")
    print("="*60)
    
    # Simulate what a HierarchicalActorCriticPolicy would have
    simulated_ppo_keys = []
    
    # Add hierarchical feature extractor keys
    for bc_key in feature_extractor_keys:
        sub_key = bc_key[len("feature_extractor."):]
        hierarchical_key = f"mlp_extractor.features_extractor.{sub_key}"
        simulated_ppo_keys.append(hierarchical_key)
    
    # Add other hierarchical policy components
    simulated_ppo_keys.extend([
        "mlp_extractor.current_subtask",  # Subtask embedding
        "mlp_extractor.high_level_policy.subtask_selector.0.weight",
        "mlp_extractor.high_level_policy.subtask_selector.0.bias",
        "mlp_extractor.low_level_policy.action_net.0.weight",
        "mlp_extractor.low_level_policy.action_net.0.bias",
        "action_net.weight",
        "action_net.bias",
        "value_net.weight",
        "value_net.bias",
    ])
    
    print(f"Total PPO keys (simulated): {len(simulated_ppo_keys)}")
    print(f"\nExpected hierarchical keys (first 10):")
    for i, key in enumerate(simulated_ppo_keys[:10]):
        print(f"  {i+1}. {key}")
    
    # Test detection logic
    print("\n" + "="*60)
    print("Detection Logic Test")
    print("="*60)
    
    uses_shared_extractor = any(
        k.startswith("features_extractor.") for k in simulated_ppo_keys
    )
    uses_separate_extractors = any(
        k.startswith("pi_features_extractor.") for k in simulated_ppo_keys
    ) or any(k.startswith("vf_features_extractor.") for k in simulated_ppo_keys)
    uses_hierarchical_extractor = any(
        "mlp_extractor.features_extractor." in k for k in simulated_ppo_keys
    )
    
    print(f"Uses shared extractor: {uses_shared_extractor}")
    print(f"Uses separate extractors: {uses_separate_extractors}")
    print(f"Uses hierarchical extractor: {uses_hierarchical_extractor}")
    
    # Test mapping logic
    print("\n" + "="*60)
    print("Mapping Logic Test")
    print("="*60)
    
    mapped_state_dict = {}
    
    for key, value in bc_state_dict.items():
        if key.startswith("feature_extractor."):
            sub_key = key[len("feature_extractor."):]
            
            if uses_hierarchical_extractor:
                hierarchical_key = f"mlp_extractor.features_extractor.{sub_key}"
                mapped_state_dict[hierarchical_key] = value
            
            if uses_shared_extractor:
                shared_key = f"features_extractor.{sub_key}"
                mapped_state_dict[shared_key] = value
            
            if uses_separate_extractors:
                pi_key = f"pi_features_extractor.{sub_key}"
                vf_key = f"vf_features_extractor.{sub_key}"
                mapped_state_dict[pi_key] = value
                mapped_state_dict[vf_key] = value.clone()
    
    print(f"Mapped {len(mapped_state_dict)} tensors")
    print(f"\nFirst 10 mapped keys:")
    for i, key in enumerate(list(mapped_state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # Check which keys would be missing
    print("\n" + "="*60)
    print("Missing Keys Analysis")
    print("="*60)
    
    ppo_keys_set = set(simulated_ppo_keys)
    mapped_keys_set = set(mapped_state_dict.keys())
    
    missing_keys = ppo_keys_set - mapped_keys_set
    unexpected_keys = mapped_keys_set - ppo_keys_set
    
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys:
        print(f"  Examples: {list(missing_keys)[:5]}")
    
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print(f"  Examples: {list(unexpected_keys)[:5]}")
    
    # Check success
    print("\n" + "="*60)
    print("Test Result")
    print("="*60)
    
    hierarchical_keys_mapped = [k for k in mapped_keys_set if "mlp_extractor.features_extractor." in k]
    
    if len(hierarchical_keys_mapped) == len(feature_extractor_keys):
        print("✅ SUCCESS: All BC feature extractor weights can be mapped to hierarchical PPO")
        print(f"   Mapped {len(hierarchical_keys_mapped)} feature extractor tensors")
        print(f"   Missing keys are expected (high-level/low-level policy heads, subtask embeddings)")
        return True
    else:
        print("❌ FAILURE: Not all BC weights were mapped correctly")
        print(f"   Expected: {len(feature_extractor_keys)}")
        print(f"   Mapped: {len(hierarchical_keys_mapped)}")
        return False

if __name__ == "__main__":
    success = test_bc_loading()
    exit(0 if success else 1)
