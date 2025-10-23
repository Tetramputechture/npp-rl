#!/usr/bin/env python3
"""
Simple validation script for BC checkpoint analysis without heavy dependencies.

This script validates:
1. BC checkpoint structure
2. CNN layer input channels (to detect frame stacking)
3. State dimensions (to detect state stacking)
4. Compatibility with target frame stacking configuration

Usage:
    python scripts/validate_checkpoint_simple.py --checkpoint bc_best.pth_testing
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch


def analyze_checkpoint(checkpoint_path: str) -> Dict:
    """Analyze BC checkpoint structure and frame stacking configuration."""
    print(f"\n{'='*80}")
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if not isinstance(checkpoint, dict):
        print("ERROR: Checkpoint is not a dictionary")
        return {}
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if "policy_state_dict" not in checkpoint:
        print("ERROR: Missing 'policy_state_dict' key")
        return {}
    
    policy_dict = checkpoint["policy_state_dict"]
    print(f"Number of tensors in policy_state_dict: {len(policy_dict)}\n")
    
    results = {
        'total_tensors': len(policy_dict),
        'visual_channels': {},
        'state_dimensions': {},
    }
    
    # Analyze player_frame CNN
    player_frame_key = "feature_extractor.player_frame_cnn.conv_layers.0.weight"
    if player_frame_key in policy_dict:
        weight_shape = tuple(policy_dict[player_frame_key].shape)
        # Conv2d weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        out_ch, in_ch, kh, kw = weight_shape
        results['visual_channels']['player_frame'] = {
            'weight_shape': weight_shape,
            'in_channels': in_ch,
            'frame_stack_size': in_ch,  # For grayscale, 1 channel = 1 frame
        }
        print(f"✓ Player Frame CNN:")
        print(f"  First conv layer: {weight_shape}")
        print(f"  Input channels: {in_ch}")
        print(f"  → Frame stack size: {in_ch}")
    else:
        print(f"✗ Player frame CNN not found")
    
    # Analyze global_view CNN
    global_view_key = "feature_extractor.global_view_cnn.conv_layers.0.weight"
    if global_view_key in policy_dict:
        weight_shape = tuple(policy_dict[global_view_key].shape)
        out_ch, in_ch, kh, kw = weight_shape
        results['visual_channels']['global_view'] = {
            'weight_shape': weight_shape,
            'in_channels': in_ch,
            'frame_stack_size': in_ch,
        }
        print(f"\n✓ Global View CNN:")
        print(f"  First conv layer: {weight_shape}")
        print(f"  Input channels: {in_ch}")
        print(f"  → Frame stack size: {in_ch}")
    else:
        print(f"\n✗ Global view CNN not found")
    
    # Analyze game_state MLP
    state_key = "feature_extractor.game_state_mlp.0.weight"
    if state_key in policy_dict:
        weight_shape = tuple(policy_dict[state_key].shape)
        # Linear weight shape: [out_features, in_features]
        out_feat, in_feat = weight_shape
        results['state_dimensions']['game_state'] = {
            'weight_shape': weight_shape,
            'in_features': in_feat,
        }
        print(f"\n✓ Game State MLP:")
        print(f"  First linear layer: {weight_shape}")
        print(f"  Input features: {in_feat}")
        
        # Estimate state stacking
        # Base game state is approximately 35-45 features
        base_state_size = 35
        if in_feat > base_state_size * 1.5:
            estimated_stack = in_feat // base_state_size
            results['state_dimensions']['game_state']['estimated_stack_size'] = estimated_stack
            print(f"  → Estimated state stack size: {estimated_stack}")
        else:
            results['state_dimensions']['game_state']['estimated_stack_size'] = 1
            print(f"  → No state stacking detected (single state)")
    else:
        print(f"\n✗ Game state MLP not found")
    
    # Show all feature extractor keys
    print(f"\n{'='*80}")
    print("All feature_extractor keys in checkpoint:")
    print(f"{'='*80}")
    feature_keys = [k for k in policy_dict.keys() if k.startswith("feature_extractor.")]
    for i, key in enumerate(feature_keys, 1):
        shape = tuple(policy_dict[key].shape) if hasattr(policy_dict[key], 'shape') else 'scalar'
        print(f"  {i:2d}. {key}")
        print(f"      Shape: {shape}")
    
    return results


def check_compatibility(
    results: Dict,
    target_visual_stack: int,
    target_state_stack: int,
) -> Tuple[bool, List[str]]:
    """Check if checkpoint is compatible with target frame stacking config."""
    print(f"\n{'='*80}")
    print("Compatibility Check")
    print(f"{'='*80}\n")
    
    print(f"Target configuration:")
    print(f"  Visual frame stack size: {target_visual_stack}")
    print(f"  State stack size: {target_state_stack}")
    print()
    
    issues = []
    
    # Check player_frame compatibility
    if 'player_frame' in results.get('visual_channels', {}):
        checkpoint_frames = results['visual_channels']['player_frame']['frame_stack_size']
        print(f"Player frame:")
        print(f"  Checkpoint has: {checkpoint_frames} stacked frames")
        print(f"  Target expects: {target_visual_stack} stacked frames")
        if checkpoint_frames != target_visual_stack:
            issue = f"MISMATCH: player_frame has {checkpoint_frames} frames, target expects {target_visual_stack}"
            issues.append(issue)
            print(f"  ✗ {issue}")
        else:
            print(f"  ✓ Compatible")
    
    # Check global_view compatibility
    if 'global_view' in results.get('visual_channels', {}):
        checkpoint_frames = results['visual_channels']['global_view']['frame_stack_size']
        print(f"\nGlobal view:")
        print(f"  Checkpoint has: {checkpoint_frames} stacked frames")
        print(f"  Target expects: {target_visual_stack} stacked frames")
        if checkpoint_frames != target_visual_stack:
            issue = f"MISMATCH: global_view has {checkpoint_frames} frames, target expects {target_visual_stack}"
            issues.append(issue)
            print(f"  ✗ {issue}")
        else:
            print(f"  ✓ Compatible")
    
    # Check game_state compatibility
    if 'game_state' in results.get('state_dimensions', {}):
        checkpoint_stack = results['state_dimensions']['game_state'].get('estimated_stack_size', 1)
        print(f"\nGame state:")
        print(f"  Checkpoint has: ~{checkpoint_stack} stacked states (estimated)")
        print(f"  Target expects: {target_state_stack} stacked states")
        if checkpoint_stack != target_stack and checkpoint_stack > 1:
            issue = f"POSSIBLE MISMATCH: game_state has ~{checkpoint_stack} states, target expects {target_state_stack}"
            issues.append(issue)
            print(f"  ⚠ {issue}")
        else:
            print(f"  ✓ Likely compatible")
    
    print(f"\n{'='*80}")
    if issues:
        print(f"✗ Found {len(issues)} compatibility issue(s)")
        return False, issues
    else:
        print(f"✓ Checkpoint appears compatible with target configuration")
        return True, []


def analyze_weight_mapping(checkpoint_path: str, use_hierarchical: bool = True):
    """Analyze how weights would map from BC to PPO."""
    print(f"\n{'='*80}")
    print("Weight Mapping Analysis")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bc_state_dict = checkpoint["policy_state_dict"]
    
    print(f"Target policy type: {'Hierarchical PPO' if use_hierarchical else 'Standard PPO'}")
    print(f"\nBC checkpoint has {len(bc_state_dict)} tensors")
    
    # Count feature extractor weights
    feature_extractor_keys = [k for k in bc_state_dict.keys() if k.startswith("feature_extractor.")]
    print(f"  Feature extractor tensors: {len(feature_extractor_keys)}")
    
    # Count policy head weights
    policy_head_keys = [k for k in bc_state_dict.keys() if k.startswith("policy_head.")]
    print(f"  Policy head tensors: {len(policy_head_keys)} (will NOT be transferred)")
    
    if use_hierarchical:
        print(f"\nFor Hierarchical PPO, BC weights will be mapped as:")
        print(f"  BC: feature_extractor.*")
        print(f"  → PPO: mlp_extractor.features_extractor.*")
        print(f"\nExpected behavior:")
        print(f"  ✓ Feature extractor weights will be loaded")
        print(f"  ✓ High-level and low-level policy heads will be randomly initialized")
        print(f"  ✓ This is CORRECT behavior - policy heads should learn from scratch")
    else:
        print(f"\nFor Standard PPO, BC weights will be mapped as:")
        print(f"  BC: feature_extractor.*")
        print(f"  → PPO: features_extractor.* (shared)")
        print(f"         OR pi_features_extractor.* and vf_features_extractor.* (separate)")
    
    # Simulate mapping
    mapped_count = 0
    for key in bc_state_dict.keys():
        if key.startswith("feature_extractor."):
            mapped_count += 1
    
    print(f"\n{mapped_count} tensors will be mapped from BC to PPO")


def provide_recommendations(results: Dict, is_compatible: bool, issues: List[str]):
    """Provide recommendations based on analysis."""
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if is_compatible:
        print("✓ The checkpoint appears compatible with your target configuration.")
        print("\nYou can proceed with training using this checkpoint for pretraining.")
        print("\nExpected log output when loading:")
        print("  ✓ Loaded BC pretrained feature extractor weights")
        print("  ✓ Loaded N weight tensors (BC → hierarchical)")
        print("  ✓ Missing keys (will use random init): M")
        print("    → Hierarchical policy keys missing: X (expected)")
        print("    → Action/value head keys missing: Y (expected)")
        print("\nThe 'missing keys' are EXPECTED and CORRECT - these components should")
        print("be trained from scratch during RL training.")
    else:
        print("✗ The checkpoint has compatibility issues with your target configuration.")
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nPossible solutions:")
        print("\n1. RECOMMENDED: Re-train BC checkpoint with frame stacking enabled")
        print("   Run BC pretraining with the same frame stacking flags:")
        print("   --enable-visual-frame-stacking --visual-stack-size 4")
        print("   --enable-state-stacking --state-stack-size 4")
        
        print("\n2. ALTERNATIVE: Train without pretraining")
        print("   Use --no-pretraining flag to skip BC pretraining")
        print("   The agent will learn from scratch via RL")
        
        print("\n3. ALTERNATIVE: Match frame stacking to checkpoint")
        print("   If checkpoint has no frame stacking (1 channel),")
        print("   train without frame stacking flags")


def main():
    parser = argparse.ArgumentParser(
        description="Validate BC checkpoint for frame stacking compatibility"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="bc_best.pth_testing",
        help="Path to BC checkpoint"
    )
    parser.add_argument(
        "--visual-stack-size",
        type=int,
        default=4,
        help="Target visual frame stack size"
    )
    parser.add_argument(
        "--state-stack-size",
        type=int,
        default=4,
        help="Target state stack size"
    )
    parser.add_argument(
        "--enable-visual-stacking",
        action="store_true",
        help="Enable visual frame stacking in target config"
    )
    parser.add_argument(
        "--enable-state-stacking",
        action="store_true",
        help="Enable state stacking in target config"
    )
    
    args = parser.parse_args()
    
    # Determine target stack sizes
    target_visual_stack = args.visual_stack_size if args.enable_visual_stacking else 1
    target_state_stack = args.state_stack_size if args.enable_state_stacking else 1
    
    print("\n" + "="*80)
    print("BC Checkpoint Frame Stacking Validation")
    print("="*80)
    
    # Analyze checkpoint
    results = analyze_checkpoint(args.checkpoint)
    
    if not results:
        print("\n✗ Failed to analyze checkpoint")
        return 1
    
    # Check compatibility
    is_compatible, issues = check_compatibility(
        results,
        target_visual_stack,
        target_state_stack,
    )
    
    # Analyze weight mapping
    analyze_weight_mapping(args.checkpoint, use_hierarchical=True)
    
    # Provide recommendations
    provide_recommendations(results, is_compatible, issues)
    
    print(f"\n{'='*80}\n")
    
    return 0 if is_compatible else 1


if __name__ == "__main__":
    sys.exit(main())
