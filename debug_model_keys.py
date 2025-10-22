#!/usr/bin/env python3
"""Debug script to inspect actual model keys."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path

# Load BC checkpoint
bc_path = Path(__file__).parent / "bc_best.pth"
bc_checkpoint = torch.load(bc_path, map_location='cpu', weights_only=False)
bc_keys = list(bc_checkpoint['policy_state_dict'].keys())

print("="*80)
print("BC Checkpoint Structure")
print("="*80)
print(f"Total keys: {len(bc_keys)}")

feature_extractor_keys = [k for k in bc_keys if k.startswith('feature_extractor.')]
policy_head_keys = [k for k in bc_keys if k.startswith('policy_head.')]

print(f"Feature extractor keys: {len(feature_extractor_keys)}")
print(f"Policy head keys: {len(policy_head_keys)}")

print("\nFirst 10 BC feature extractor keys:")
for i, key in enumerate(feature_extractor_keys[:10]):
    print(f"  {i+1}. {key}")

print("\n" + "="*80)
print("Error Analysis")
print("="*80)
print("Error says:")
print("  Loaded 58 weight tensors (BC → hierarchical)")
print("  Missing keys (will use random init): 206")
print("  Examples: ['features_extractor.player_frame_cnn.0.weight', ...]")
print()
print("Analysis:")
print("  1. 58 tensors loaded to mlp_extractor.features_extractor.* ✓")
print("  2. 206 keys missing, including features_extractor.* ✗")
print()
print("Hypothesis:")
print("  The model has BOTH structures:")
print("    - mlp_extractor.features_extractor.* (58 keys) ← Loaded from BC")
print("    - features_extractor.* (58 keys) ← MISSING (not loaded)")
print()
print("  If they're the same object (reference), only 58 keys should be missing.")
print("  If 206 keys are missing, that means:")
print("    206 = 58 (features_extractor) + 58 (pi_features_extractor)")
print("          + 58 (vf_features_extractor) + other")
print()
print("  OR:")
print("    The model structure has features_extractor.* as SEPARATE from")
print("    mlp_extractor.features_extractor.*")

print("\n" + "="*80)
print("Solution")
print("="*80)
print("We need to:")
print("  1. Check if features_extractor and mlp_extractor.features_extractor are the same")
print("  2. If they're references: mapping to one should satisfy both")
print("  3. If they're separate: need to map BC weights to BOTH")
print()
print("Current priority logic maps to ONLY hierarchical when detected.")
print("We may need to map to hierarchical AND shared if both exist.")
