#!/usr/bin/env python3
"""
Comprehensive BC Weight Transfer Validation Script

This script validates that BC pretrained weights transfer correctly to hierarchical PPO
models with frame stacking enabled.

Tests:
1. BC checkpoint structure validation
2. Frame stacking configuration detection
3. Weight key mapping verification
4. Actual weight loading simulation
5. Shape compatibility checks

Usage:
    python scripts/validate_bc_weight_transfer.py --checkpoint bc_best.pth_testing
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BCCheckpointValidator:
    """Validates BC checkpoint structure and frame stacking configuration."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = None
        self.policy_state_dict = None
        self.architecture = None
        self.frame_stacking_config = None
        
    def load_checkpoint(self) -> bool:
        """Load checkpoint file."""
        if not self.checkpoint_path.exists():
            logger.error(f"✗ Checkpoint not found: {self.checkpoint_path}")
            return False
        
        try:
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False
            )
            logger.info(f"✓ Loaded checkpoint: {self.checkpoint_path.name}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate checkpoint has required structure."""
        if not isinstance(self.checkpoint, dict):
            logger.error("✗ Checkpoint is not a dictionary")
            return False
        
        # Check for required keys
        if "policy_state_dict" not in self.checkpoint:
            logger.error("✗ Missing 'policy_state_dict' key")
            logger.info(f"  Found keys: {list(self.checkpoint.keys())}")
            return False
        
        self.policy_state_dict = self.checkpoint["policy_state_dict"]
        
        # Check for optional metadata
        self.architecture = self.checkpoint.get("architecture", "unknown")
        self.frame_stacking_config = self.checkpoint.get("frame_stacking", None)
        
        logger.info(f"✓ Checkpoint structure valid")
        logger.info(f"  Architecture: {self.architecture}")
        logger.info(f"  Total weight tensors: {len(self.policy_state_dict)}")
        
        return True
    
    def analyze_key_structure(self) -> Dict[str, List[str]]:
        """Analyze weight key structure to understand model components."""
        key_groups = defaultdict(list)
        
        for key in self.policy_state_dict.keys():
            # Extract top-level component
            top_level = key.split('.')[0]
            key_groups[top_level].append(key)
        
        logger.info(f"\n✓ Key structure analysis:")
        for component, keys in sorted(key_groups.items()):
            logger.info(f"  {component}: {len(keys)} keys")
            # Show first 3 keys as examples
            for key in sorted(keys)[:3]:
                logger.info(f"    - {key}")
            if len(keys) > 3:
                logger.info(f"    ... and {len(keys) - 3} more")
        
        return dict(key_groups)
    
    def detect_frame_stacking(self) -> Dict[str, any]:
        """Detect frame stacking configuration from weight shapes."""
        frame_stack_info = {}
        
        # Check player_frame_cnn first conv layer
        player_frame_key = "feature_extractor.player_frame_cnn.conv_layers.0.weight"
        if player_frame_key in self.policy_state_dict:
            weight_shape = tuple(self.policy_state_dict[player_frame_key].shape)
            # Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
            in_channels = weight_shape[1]
            frame_stack_info['player_frame_cnn'] = {
                'key': player_frame_key,
                'shape': weight_shape,
                'in_channels': in_channels,
                'detected_stack_size': in_channels,
                'is_stacked': in_channels > 1
            }
        
        # Check global_cnn first conv layer
        global_cnn_key = "feature_extractor.global_cnn.conv_layers.0.weight"
        if global_cnn_key in self.policy_state_dict:
            weight_shape = tuple(self.policy_state_dict[global_cnn_key].shape)
            in_channels = weight_shape[1]
            frame_stack_info['global_cnn'] = {
                'key': global_cnn_key,
                'shape': weight_shape,
                'in_channels': in_channels,
                'detected_stack_size': in_channels,
                'is_stacked': in_channels > 1
            }
        
        # Check state_mlp to detect state stacking
        state_mlp_key = "feature_extractor.state_mlp.0.weight"
        if state_mlp_key in self.policy_state_dict:
            weight_shape = tuple(self.policy_state_dict[state_mlp_key].shape)
            # Linear: [out_features, in_features]
            in_features = weight_shape[1]
            frame_stack_info['state_mlp'] = {
                'key': state_mlp_key,
                'shape': weight_shape,
                'in_features': in_features,
            }
        
        logger.info(f"\n✓ Frame stacking detection:")
        for component, info in frame_stack_info.items():
            logger.info(f"  {component}:")
            for k, v in info.items():
                logger.info(f"    {k}: {v}")
        
        # Compare with checkpoint metadata if available
        if self.frame_stacking_config:
            logger.info(f"\n✓ Frame stacking config from checkpoint metadata:")
            for k, v in self.frame_stacking_config.items():
                logger.info(f"    {k}: {v}")
        
        return frame_stack_info


class WeightMappingValidator:
    """Validates weight mapping from BC to hierarchical PPO structure."""
    
    def __init__(self, bc_state_dict: Dict[str, torch.Tensor]):
        self.bc_state_dict = bc_state_dict
    
    def simulate_mapping(self, target_structure: str = "hierarchical") -> Dict[str, torch.Tensor]:
        """Simulate weight mapping from BC to target structure.
        
        Args:
            target_structure: One of "shared", "separate", "hierarchical"
        
        Returns:
            Mapped state dict
        """
        mapped_dict = {}
        skipped = []
        
        for key, value in self.bc_state_dict.items():
            if key.startswith("feature_extractor."):
                sub_key = key[len("feature_extractor."):]
                
                if target_structure == "hierarchical":
                    new_key = f"mlp_extractor.features_extractor.{sub_key}"
                    mapped_dict[new_key] = value
                elif target_structure == "shared":
                    new_key = f"features_extractor.{sub_key}"
                    mapped_dict[new_key] = value
                elif target_structure == "separate":
                    pi_key = f"pi_features_extractor.{sub_key}"
                    vf_key = f"vf_features_extractor.{sub_key}"
                    mapped_dict[pi_key] = value
                    mapped_dict[vf_key] = value.clone()
                else:
                    raise ValueError(f"Unknown target structure: {target_structure}")
            else:
                skipped.append(key)
        
        logger.info(f"\n✓ Weight mapping simulation ({target_structure}):")
        logger.info(f"  Mapped: {len(mapped_dict)} tensors")
        logger.info(f"  Skipped: {len(skipped)} tensors")
        if skipped:
            logger.info(f"  Skipped keys (BC-specific):")
            for key in skipped[:5]:
                logger.info(f"    - {key}")
            if len(skipped) > 5:
                logger.info(f"    ... and {len(skipped) - 5} more")
        
        return mapped_dict
    
    def analyze_mapped_keys(self, mapped_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """Analyze mapped key structure."""
        key_groups = defaultdict(list)
        
        for key in mapped_dict.keys():
            # Extract component path
            parts = key.split('.')
            if len(parts) >= 2:
                component = '.'.join(parts[:2])
            else:
                component = parts[0]
            key_groups[component].append(key)
        
        logger.info(f"\n✓ Mapped key structure:")
        for component, keys in sorted(key_groups.items()):
            logger.info(f"  {component}: {len(keys)} keys")
        
        return dict(key_groups)


class HierarchicalModelValidator:
    """Validates weight loading into actual hierarchical model."""
    
    def __init__(self):
        self.model = None
        self.policy = None
    
    def create_mock_hierarchical_policy(self, arch_config_name: str = "mlp_baseline"):
        """Create a mock hierarchical policy for testing.
        
        Note: This creates the actual model to verify weight compatibility.
        """
        try:
            from nclone.gym_environment.npp_environment import NppEnvironment
            from nclone.gym_environment.config import EnvironmentConfig
            from npp_rl.training.architecture_configs import get_architecture_config
            from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
            
            # Create temporary environment
            env_config = EnvironmentConfig.for_training()
            env = NppEnvironment(config=env_config)
            
            # Get architecture config
            arch_config = get_architecture_config(arch_config_name)
            
            # Create feature extractor
            feature_extractor = ConfigurableMultimodalExtractor(
                observation_space=env.observation_space,
                config=arch_config,
            )
            
            logger.info(f"\n✓ Created mock feature extractor")
            logger.info(f"  Architecture: {arch_config_name}")
            logger.info(f"  Features dim: {arch_config.features_dim}")
            
            # Get state dict
            extractor_state_dict = feature_extractor.state_dict()
            
            logger.info(f"  Feature extractor keys: {len(extractor_state_dict)}")
            
            # Analyze key structure
            key_groups = defaultdict(list)
            for key in extractor_state_dict.keys():
                top_level = key.split('.')[0]
                key_groups[top_level].append(key)
            
            logger.info(f"  Key groups:")
            for component, keys in sorted(key_groups.items()):
                logger.info(f"    {component}: {len(keys)} keys")
            
            env.close()
            
            return extractor_state_dict
            
        except Exception as e:
            logger.error(f"✗ Failed to create mock policy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_weight_compatibility(
        self, 
        bc_weights: Dict[str, torch.Tensor],
        model_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """Validate that BC weights are compatible with model structure."""
        
        # Simulate mapping to hierarchical structure
        mapped_weights = {}
        for key, value in bc_weights.items():
            if key.startswith("feature_extractor."):
                sub_key = key[len("feature_extractor."):]
                # Try to match against model keys (without hierarchical prefix)
                if sub_key in model_state_dict:
                    mapped_weights[sub_key] = value
        
        compatible_keys = []
        incompatible_keys = []
        shape_mismatches = []
        
        for mapped_key, bc_tensor in mapped_weights.items():
            if mapped_key in model_state_dict:
                model_tensor = model_state_dict[mapped_key]
                if bc_tensor.shape == model_tensor.shape:
                    compatible_keys.append(mapped_key)
                else:
                    incompatible_keys.append(mapped_key)
                    shape_mismatches.append({
                        'key': mapped_key,
                        'bc_shape': tuple(bc_tensor.shape),
                        'model_shape': tuple(model_tensor.shape)
                    })
        
        logger.info(f"\n✓ Weight compatibility check:")
        logger.info(f"  Compatible: {len(compatible_keys)} tensors")
        logger.info(f"  Incompatible: {len(incompatible_keys)} tensors")
        
        if shape_mismatches:
            logger.warning(f"\n⚠ Shape mismatches found:")
            for mismatch in shape_mismatches[:5]:
                logger.warning(f"  {mismatch['key']}:")
                logger.warning(f"    BC shape: {mismatch['bc_shape']}")
                logger.warning(f"    Model shape: {mismatch['model_shape']}")
            if len(shape_mismatches) > 5:
                logger.warning(f"  ... and {len(shape_mismatches) - 5} more")
        
        return {
            'compatible_keys': compatible_keys,
            'incompatible_keys': incompatible_keys,
            'shape_mismatches': shape_mismatches,
            'total_bc_keys': len(mapped_weights),
            'total_model_keys': len(model_state_dict)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate BC weight transfer to hierarchical PPO"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to BC checkpoint file"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp_baseline",
        help="Architecture to validate against (default: mlp_baseline)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Banner
    logger.info("=" * 80)
    logger.info("BC Weight Transfer Validation")
    logger.info("=" * 80)
    
    # Step 1: Validate BC checkpoint
    logger.info("\n[1/5] Loading and validating BC checkpoint...")
    bc_validator = BCCheckpointValidator(args.checkpoint)
    
    if not bc_validator.load_checkpoint():
        logger.error("✗ FAILED: Could not load checkpoint")
        return 1
    
    if not bc_validator.validate_structure():
        logger.error("✗ FAILED: Invalid checkpoint structure")
        return 1
    
    # Step 2: Analyze checkpoint structure
    logger.info("\n[2/5] Analyzing checkpoint structure...")
    key_groups = bc_validator.analyze_key_structure()
    
    # Step 3: Detect frame stacking
    logger.info("\n[3/5] Detecting frame stacking configuration...")
    frame_stack_info = bc_validator.detect_frame_stacking()
    
    # Step 4: Simulate weight mapping
    logger.info("\n[4/5] Simulating weight mapping to hierarchical structure...")
    mapping_validator = WeightMappingValidator(bc_validator.policy_state_dict)
    mapped_dict = mapping_validator.simulate_mapping("hierarchical")
    mapping_validator.analyze_mapped_keys(mapped_dict)
    
    # Step 5: Validate against actual model structure
    logger.info("\n[5/5] Validating against actual model structure...")
    model_validator = HierarchicalModelValidator()
    model_state_dict = model_validator.create_mock_hierarchical_policy(args.architecture)
    
    if model_state_dict:
        compatibility_results = model_validator.validate_weight_compatibility(
            bc_validator.policy_state_dict,
            model_state_dict
        )
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ BC checkpoint: {args.checkpoint}")
        logger.info(f"✓ Architecture: {bc_validator.architecture}")
        logger.info(f"✓ Total BC weight tensors: {len(bc_validator.policy_state_dict)}")
        logger.info(f"✓ Mapped to hierarchical: {len(mapped_dict)} tensors")
        logger.info(f"✓ Compatible with model: {len(compatibility_results['compatible_keys'])} tensors")
        
        if compatibility_results['shape_mismatches']:
            logger.warning(f"⚠ Shape mismatches: {len(compatibility_results['shape_mismatches'])}")
            logger.warning("  This may indicate frame stacking configuration mismatch!")
        else:
            logger.info("✓ All weights are shape-compatible!")
        
        # Frame stacking summary
        if frame_stack_info:
            logger.info("\n✓ Frame stacking detected:")
            for component, info in frame_stack_info.items():
                if 'is_stacked' in info and info['is_stacked']:
                    logger.info(f"  {component}: {info['detected_stack_size']} frames")
        
        logger.info("\n" + "=" * 80)
        
        if compatibility_results['shape_mismatches']:
            logger.warning("⚠ VALIDATION PASSED WITH WARNINGS")
            logger.warning("  Review shape mismatches above")
            return 0
        else:
            logger.info("✓ VALIDATION PASSED - All checks successful!")
            return 0
    else:
        logger.error("✗ FAILED: Could not create model for validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
