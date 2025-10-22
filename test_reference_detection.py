#!/usr/bin/env python3
"""
Test the reference detection logic for BC weight loading.
"""

import torch
import torch.nn as nn

print("="*80)
print("Test: Reference Detection Logic")
print("="*80)

# Simulate different model structures
class MockFeatureExtractor(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.layer = nn.Linear(10, 10)

class MockMLPExtractor(nn.Module):
    def __init__(self, features_extractor):
        super().__init__()
        self.features_extractor = features_extractor

class MockPolicy_Reference(nn.Module):
    """Model where features_extractor and mlp_extractor.features_extractor are the SAME"""
    def __init__(self):
        super().__init__()
        self.features_extractor = MockFeatureExtractor("shared")
        self.mlp_extractor = MockMLPExtractor(self.features_extractor)  # Reference

class MockPolicy_Separate(nn.Module):
    """Model where they are DIFFERENT objects"""
    def __init__(self):
        super().__init__()
        self.features_extractor = MockFeatureExtractor("top_level")
        hierarchical_extractor = MockFeatureExtractor("hierarchical")
        self.mlp_extractor = MockMLPExtractor(hierarchical_extractor)  # Separate

print("\n" + "-"*80)
print("Case 1: Reference Model (expected in hierarchical)")
print("-"*80)

policy1 = MockPolicy_Reference()
is_same = policy1.features_extractor is policy1.mlp_extractor.features_extractor
print(f"features_extractor is mlp_extractor.features_extractor: {is_same}")
print(f"features_extractor name: {policy1.features_extractor.name}")
print(f"mlp_extractor.features_extractor name: {policy1.mlp_extractor.features_extractor.name}")

# Test the detection logic
uses_hierarchical = True  # Simulated
uses_shared = True  # Simulated (would detect features_extractor.* in state_dict)

map_hierarchical = uses_hierarchical
map_shared = False

if uses_hierarchical:
    if hasattr(policy1, 'features_extractor') and \
       hasattr(policy1, 'mlp_extractor') and \
       hasattr(policy1.mlp_extractor, 'features_extractor'):
        is_same_object = (
            policy1.features_extractor is policy1.mlp_extractor.features_extractor
        )
        if not is_same_object and uses_shared:
            map_shared = True
            print("Decision: Map to BOTH hierarchical and shared (they're separate)")
        else:
            print("Decision: Map to ONLY hierarchical (shared is a reference)")

print(f"map_hierarchical: {map_hierarchical}")
print(f"map_shared: {map_shared}")

print("\n" + "-"*80)
print("Case 2: Separate Model (if bug exists)")
print("-"*80)

policy2 = MockPolicy_Separate()
is_same = policy2.features_extractor is policy2.mlp_extractor.features_extractor
print(f"features_extractor is mlp_extractor.features_extractor: {is_same}")
print(f"features_extractor name: {policy2.features_extractor.name}")
print(f"mlp_extractor.features_extractor name: {policy2.mlp_extractor.features_extractor.name}")

# Test the detection logic
map_hierarchical = uses_hierarchical
map_shared = False

if uses_hierarchical:
    if hasattr(policy2, 'features_extractor') and \
       hasattr(policy2, 'mlp_extractor') and \
       hasattr(policy2.mlp_extractor, 'features_extractor'):
        is_same_object = (
            policy2.features_extractor is policy2.mlp_extractor.features_extractor
        )
        if not is_same_object and uses_shared:
            map_shared = True
            print("Decision: Map to BOTH hierarchical and shared (they're separate)")
        else:
            print("Decision: Map to ONLY hierarchical (shared is a reference)")

print(f"map_hierarchical: {map_hierarchical}")
print(f"map_shared: {map_shared}")

print("\n" + "="*80)
print("Expected Behavior")
print("="*80)
print("Case 1 (Reference): Map to hierarchical only")
print("  - features_extractor.* will show as missing but it's OK (reference)")
print("  - Missing keys are expected and harmless")
print()
print("Case 2 (Separate): Map to both hierarchical and shared")
print("  - features_extractor.* really needs loading")
print("  - Both hierarchical and shared get BC weights")
print()
print("The fix detects which case applies using 'is' operator!")
