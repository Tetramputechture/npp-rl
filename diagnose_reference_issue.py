#!/usr/bin/env python3
"""
Diagnose whether features_extractor and mlp_extractor.features_extractor
are references to the same object or separate objects.
"""

import torch
import torch.nn as nn

print("="*80)
print("Understanding PyTorch State Dict with Shared References")
print("="*80)

class ChildModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

class ParentModule(nn.Module):
    def __init__(self, shared_child):
        super().__init__()
        self.shared_child = shared_child  # Store reference

class Model1_Reference(nn.Module):
    """Model where both paths point to the SAME object"""
    def __init__(self):
        super().__init__()
        self.child = ChildModule()  # Create once
        self.parent = ParentModule(self.child)  # Pass reference

class Model2_Separate(nn.Module):
    """Model where both paths are SEPARATE objects"""
    def __init__(self):
        super().__init__()
        self.child = ChildModule()  # Create one
        child2 = ChildModule()  # Create another
        self.parent = ParentModule(child2)  # Pass different one

print("\n" + "-"*80)
print("Test 1: Reference Model (same object)")
print("-"*80)
model1 = Model1_Reference()
keys1 = list(model1.state_dict().keys())
print(f"State dict keys: {len(keys1)}")
for k in keys1:
    print(f"  {k}")

# Check if they're the same object
is_same = model1.child is model1.parent.shared_child
print(f"\nmodel.child is model.parent.shared_child: {is_same}")

print("\n" + "-"*80)
print("Test 2: Separate Model (different objects)")
print("-"*80)
model2 = Model2_Separate()
keys2 = list(model2.state_dict().keys())
print(f"State dict keys: {len(keys2)}")
for k in keys2:
    print(f"  {k}")

print("\n" + "-"*80)
print("Key Insight")
print("-"*80)
print("In Model1 (reference): both 'child.*' and 'parent.shared_child.*' keys exist")
print("  Even though they point to the same object!")
print()
print("PyTorch state_dict serializes ALL module paths, even if they reference")
print("the same underlying parameters.")
print()
print("When using load_state_dict(state_dict, strict=False):")
print("  - If you provide 'child.layer.weight', it loads to model.child.layer.weight")
print("  - model.parent.shared_child.layer.weight is automatically the same (reference)")
print("  - So 'parent.shared_child.layer.weight' shows as MISSING but it's actually loaded!")

print("\n" + "-"*80)
print("Test 3: Verify Loading Behavior")
print("-"*80)
model1 = Model1_Reference()
print("Original weight:", model1.child.layer.weight[0, 0].item())

# Create state dict with only 'child.*' keys
partial_state = {}
for k, v in model1.state_dict().items():
    if k.startswith('child.'):
        # Use different values
        partial_state[k] = v * 2

print(f"\nLoading state dict with only 'child.*' keys ({len(partial_state)} keys)")
result = model1.load_state_dict(partial_state, strict=False)
print(f"Missing keys: {len(result.missing_keys)}")
print(f"  Examples: {result.missing_keys[:3]}")

print(f"\nAfter loading:")
print(f"  model.child.layer.weight[0,0]: {model1.child.layer.weight[0, 0].item()}")
print(f"  model.parent.shared_child.layer.weight[0,0]: {model1.parent.shared_child.layer.weight[0, 0].item()}")
print(f"  Are they equal? {torch.equal(model1.child.layer.weight, model1.parent.shared_child.layer.weight)}")

print("\n" + "="*80)
print("Conclusion for HierarchicalActorCriticPolicy")
print("="*80)
print("If features_extractor shows as MISSING after loading mlp_extractor.features_extractor,")
print("it could mean:")
print()
print("1. They ARE references (like Model1):")
print("   - Missing key is reported but weights are actually loaded")
print("   - This is EXPECTED and OK")
print()
print("2. They are SEPARATE (like Model2):")
print("   - Missing key means those weights are truly not loaded")
print("   - We need to map BC weights to BOTH locations")
print()
print("To determine which case applies, we need to check if:")
print("  policy.features_extractor is policy.mlp_extractor.features_extractor")
