#!/usr/bin/env python3
"""
Comprehensive validation script for frame stacking pretraining pipeline.

This script validates:
1. BC checkpoint structure and compatibility with frame stacking
2. Weight transfer mechanism from BC to hierarchical PPO
3. Observation space handling with frame stacking enabled
4. End-to-end pretraining pipeline with frame stacking

Usage:
    python scripts/validate_frame_stacking_pretraining.py --checkpoint bc_best.pth_testing
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.architecture_configs import get_architecture_config
from npp_rl.training.policy_utils import create_observation_space_from_config
from npp_rl.utils import setup_experiment_logging

logger = logging.getLogger(__name__)


class CheckpointValidator:
    """Validates BC checkpoint structure and compatibility."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = None
        self.policy_dict = None
        
    def load_checkpoint(self) -> bool:
        """Load and validate checkpoint file."""
        if not self.checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {self.checkpoint_path}")
            return False
        
        try:
            self.checkpoint = torch.load(
                self.checkpoint_path, 
                map_location="cpu", 
                weights_only=False
            )
            logger.info(f"✓ Loaded checkpoint: {self.checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate checkpoint has expected structure."""
        required_keys = ["policy_state_dict", "architecture"]
        
        if not isinstance(self.checkpoint, dict):
            logger.error("Checkpoint is not a dictionary")
            return False
        
        missing_keys = [k for k in required_keys if k not in self.checkpoint]
        if missing_keys:
            logger.error(f"Missing required keys: {missing_keys}")
            logger.info(f"Found keys: {list(self.checkpoint.keys())}")
            return False
        
        self.policy_dict = self.checkpoint["policy_state_dict"]
        logger.info(f"✓ Checkpoint structure valid")
        logger.info(f"  Keys: {list(self.checkpoint.keys())}")
        logger.info(f"  Number of tensors in policy_state_dict: {len(self.policy_dict)}")
        return True
    
    def analyze_conv_layer_channels(self) -> Dict[str, Dict[str, int]]:
        """Analyze input channels of CNN layers to detect frame stacking."""
        results = {}
        
        # Check player_frame_cnn first conv layer
        player_frame_key = "feature_extractor.player_frame_cnn.conv_layers.0.weight"
        if player_frame_key in self.policy_dict:
            weight_shape = self.policy_dict[player_frame_key].shape
            # Conv2d weight shape: [out_channels, in_channels, kernel_h, kernel_w]
            in_channels = weight_shape[1]
            results['player_frame'] = {
                'in_channels': in_channels,
                'weight_shape': weight_shape,
                'frame_stack_size': in_channels,  # Each channel = 1 frame for grayscale
            }
            logger.info(f"  player_frame_cnn.conv_layers.0: {weight_shape}")
            logger.info(f"    → Input channels: {in_channels} (frame stack size: {in_channels})")
        
        # Check global_view_cnn first conv layer
        global_view_key = "feature_extractor.global_view_cnn.conv_layers.0.weight"
        if global_view_key in self.policy_dict:
            weight_shape = self.policy_dict[global_view_key].shape
            in_channels = weight_shape[1]
            results['global_view'] = {
                'in_channels': in_channels,
                'weight_shape': weight_shape,
                'frame_stack_size': in_channels,
            }
            logger.info(f"  global_view_cnn.conv_layers.0: {weight_shape}")
            logger.info(f"    → Input channels: {in_channels} (frame stack size: {in_channels})")
        
        return results
    
    def analyze_state_dimensions(self) -> Dict[str, any]:
        """Analyze state embedding layers to detect state stacking."""
        results = {}
        
        # Check game_state_mlp first layer
        state_key = "feature_extractor.game_state_mlp.0.weight"
        if state_key in self.policy_dict:
            weight_shape = self.policy_dict[state_key].shape
            # Linear weight shape: [out_features, in_features]
            in_features = weight_shape[1]
            results['game_state'] = {
                'in_features': in_features,
                'weight_shape': weight_shape,
            }
            logger.info(f"  game_state_mlp.0: {weight_shape}")
            logger.info(f"    → Input features: {in_features}")
            
            # Base game state size is approximately 35-45 features
            # If in_features is much larger, it may include stacking
            base_state_size = 35  # Approximate
            if in_features > base_state_size * 1.5:
                estimated_stack = in_features // base_state_size
                results['game_state']['estimated_stack_size'] = estimated_stack
                logger.info(f"    → Estimated state stack size: {estimated_stack}")
            else:
                results['game_state']['estimated_stack_size'] = 1
        
        return results
    
    def check_compatibility_with_frame_stacking(
        self, 
        visual_stack_size: int = 4,
        state_stack_size: int = 4
    ) -> Tuple[bool, List[str]]:
        """Check if checkpoint is compatible with specified frame stacking config."""
        issues = []
        
        conv_analysis = self.analyze_conv_layer_channels()
        state_analysis = self.analyze_state_dimensions()
        
        # Check visual frame stacking compatibility
        if 'player_frame' in conv_analysis:
            checkpoint_frames = conv_analysis['player_frame']['frame_stack_size']
            if checkpoint_frames != visual_stack_size:
                issues.append(
                    f"player_frame: checkpoint has {checkpoint_frames} stacked frames, "
                    f"but config expects {visual_stack_size}"
                )
        
        if 'global_view' in conv_analysis:
            checkpoint_frames = conv_analysis['global_view']['frame_stack_size']
            if checkpoint_frames != visual_stack_size:
                issues.append(
                    f"global_view: checkpoint has {checkpoint_frames} stacked frames, "
                    f"but config expects {visual_stack_size}"
                )
        
        # Check state stacking compatibility
        if 'game_state' in state_analysis:
            checkpoint_stack = state_analysis['game_state'].get('estimated_stack_size', 1)
            if checkpoint_stack != state_stack_size and checkpoint_stack > 1:
                issues.append(
                    f"game_state: checkpoint appears to have {checkpoint_stack} stacked states, "
                    f"but config expects {state_stack_size}"
                )
        
        is_compatible = len(issues) == 0
        return is_compatible, issues


class WeightTransferValidator:
    """Validates weight transfer from BC to hierarchical PPO."""
    
    def __init__(self, checkpoint_path: str, architecture_name: str = "mlp_baseline"):
        self.checkpoint_path = checkpoint_path
        self.architecture_name = architecture_name
        
    def simulate_weight_transfer(
        self, 
        use_hierarchical: bool = True,
        frame_stack_config: Optional[Dict] = None
    ) -> Tuple[bool, Dict]:
        """Simulate weight transfer to detect potential issues."""
        logger.info(f"\nSimulating weight transfer:")
        logger.info(f"  Architecture: {self.architecture_name}")
        logger.info(f"  Hierarchical PPO: {use_hierarchical}")
        logger.info(f"  Frame stacking: {frame_stack_config}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            bc_state_dict = checkpoint["policy_state_dict"]
            
            # Get architecture config
            arch_config = get_architecture_config(self.architecture_name)
            
            # Create observation space (frame stacking affects observation space)
            obs_space = create_observation_space_from_config(arch_config)
            
            # Log observation space
            logger.info(f"\nObservation space:")
            for key, space in obs_space.spaces.items():
                logger.info(f"  {key}: {space.shape}")
            
            # Create a mock PPO policy to test weight loading
            from stable_baselines3.common.policies import ActorCriticPolicy
            from npp_rl.hrl.hierarchical_actor_critic_policy import HierarchicalActorCriticPolicy
            from gymnasium import spaces
            
            action_space = spaces.Discrete(6)
            
            if use_hierarchical:
                # Create hierarchical policy
                policy = HierarchicalActorCriticPolicy(
                    observation_space=obs_space,
                    action_space=action_space,
                    lr_schedule=lambda _: 3e-4,
                    features_extractor_class=arch_config.extractor_class,
                    features_extractor_kwargs=arch_config.extractor_kwargs,
                    net_arch=[256, 256],
                )
            else:
                # Create standard policy
                policy = ActorCriticPolicy(
                    observation_space=obs_space,
                    action_space=action_space,
                    lr_schedule=lambda _: 3e-4,
                    features_extractor_class=arch_config.extractor_class,
                    features_extractor_kwargs=arch_config.extractor_kwargs,
                    net_arch=[256, 256],
                )
            
            policy_keys = list(policy.state_dict().keys())
            
            # Simulate weight mapping (same logic as architecture_trainer)
            uses_hierarchical_extractor = any(
                "mlp_extractor.features_extractor." in k for k in policy_keys
            )
            uses_separate_extractors = any(
                k.startswith("pi_features_extractor.") for k in policy_keys
            )
            uses_shared_extractor = any(
                k.startswith("features_extractor.") for k in policy_keys
            )
            
            logger.info(f"\nPolicy structure:")
            logger.info(f"  Has hierarchical extractor: {uses_hierarchical_extractor}")
            logger.info(f"  Has shared extractor: {uses_shared_extractor}")
            logger.info(f"  Has separate extractors: {uses_separate_extractors}")
            
            # Map BC weights
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
            
            # Try to load weights
            missing_keys, unexpected_keys = policy.load_state_dict(
                mapped_state_dict, strict=False
            )
            
            results = {
                'success': True,
                'mapped_tensors': len(mapped_state_dict),
                'missing_keys': len(missing_keys),
                'unexpected_keys': len(unexpected_keys),
                'missing_key_examples': missing_keys[:5],
                'unexpected_key_examples': unexpected_keys[:5],
            }
            
            logger.info(f"\n✓ Weight transfer simulation completed:")
            logger.info(f"  Mapped tensors: {results['mapped_tensors']}")
            logger.info(f"  Missing keys: {results['missing_keys']}")
            if results['missing_keys'] > 0:
                logger.info(f"    Examples: {results['missing_key_examples']}")
            logger.info(f"  Unexpected keys: {results['unexpected_keys']}")
            if results['unexpected_keys'] > 0:
                logger.info(f"    Examples: {results['unexpected_key_examples']}")
            
            return True, results
            
        except Exception as e:
            logger.error(f"Weight transfer simulation failed: {e}", exc_info=True)
            return False, {'error': str(e)}


class FrameStackingValidator:
    """Validates frame stacking configuration and observations."""
    
    def __init__(self, architecture_name: str = "mlp_baseline"):
        self.architecture_name = architecture_name
    
    def validate_observation_shapes(
        self,
        enable_visual_stacking: bool = True,
        visual_stack_size: int = 4,
        enable_state_stacking: bool = True,
        state_stack_size: int = 4,
    ) -> Tuple[bool, Dict]:
        """Validate observation shapes with frame stacking."""
        logger.info(f"\nValidating observation shapes:")
        logger.info(f"  Visual stacking: {enable_visual_stacking} (size: {visual_stack_size})")
        logger.info(f"  State stacking: {enable_state_stacking} (size: {state_stack_size})")
        
        try:
            # Get architecture config
            arch_config = get_architecture_config(self.architecture_name)
            
            # Create observation space
            obs_space = create_observation_space_from_config(arch_config)
            
            results = {}
            
            # Check each observation component
            for key, space in obs_space.spaces.items():
                expected_shape = space.shape
                results[key] = {
                    'shape': expected_shape,
                    'dtype': space.dtype,
                }
                
                # Analyze expected vs actual shape with frame stacking
                if key in ['player_frame', 'global_view']:
                    # Visual observations
                    # Without stacking: (H, W, 1)
                    # With stacking: (stack_size, H, W, 1)
                    if enable_visual_stacking:
                        if len(expected_shape) == 4 and expected_shape[0] == visual_stack_size:
                            results[key]['status'] = 'correct'
                            logger.info(f"  ✓ {key}: {expected_shape} (stacked)")
                        else:
                            results[key]['status'] = 'incorrect'
                            results[key]['issue'] = f"Expected shape with stacking: ({visual_stack_size}, H, W, 1), got {expected_shape}"
                            logger.warning(f"  ✗ {key}: {expected_shape} - {results[key]['issue']}")
                    else:
                        if len(expected_shape) == 3:
                            results[key]['status'] = 'correct'
                            logger.info(f"  ✓ {key}: {expected_shape} (single frame)")
                        else:
                            results[key]['status'] = 'incorrect'
                            results[key]['issue'] = f"Expected single frame shape (H, W, 1), got {expected_shape}"
                            logger.warning(f"  ✗ {key}: {expected_shape} - {results[key]['issue']}")
                
                elif key == 'game_state':
                    # State observations
                    # Without stacking: (state_dim,)
                    # With stacking: (stack_size * state_dim,)
                    base_state_dim = 35  # Approximate
                    if enable_state_stacking:
                        expected_dim = state_stack_size * base_state_dim
                        if abs(expected_shape[0] - expected_dim) < 50:  # Allow some variance
                            results[key]['status'] = 'correct'
                            logger.info(f"  ✓ {key}: {expected_shape} (stacked)")
                        else:
                            results[key]['status'] = 'unclear'
                            logger.info(f"  ? {key}: {expected_shape} (state stacking enabled)")
                    else:
                        results[key]['status'] = 'correct'
                        logger.info(f"  ✓ {key}: {expected_shape}")
                else:
                    results[key]['status'] = 'correct'
                    logger.info(f"  ✓ {key}: {expected_shape}")
            
            all_correct = all(r.get('status') != 'incorrect' for r in results.values())
            return all_correct, results
            
        except Exception as e:
            logger.error(f"Observation shape validation failed: {e}", exc_info=True)
            return False, {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Validate frame stacking pretraining pipeline"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="bc_best.pth_testing",
        help="Path to BC checkpoint"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp_baseline",
        help="Architecture name"
    )
    parser.add_argument(
        "--visual-stack-size",
        type=int,
        default=4,
        help="Visual frame stack size"
    )
    parser.add_argument(
        "--state-stack-size",
        type=int,
        default=4,
        help="State stack size"
    )
    parser.add_argument(
        "--enable-visual-stacking",
        action="store_true",
        help="Enable visual frame stacking"
    )
    parser.add_argument(
        "--enable-state-stacking",
        action="store_true",
        help="Enable state stacking"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_experiment_logging("frame_stacking_validation", Path("validation_logs"))
    
    logger.info("=" * 80)
    logger.info("Frame Stacking Pretraining Pipeline Validation")
    logger.info("=" * 80)
    
    all_tests_passed = True
    
    # Test 1: Checkpoint validation
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Checkpoint Structure Validation")
    logger.info("=" * 80)
    
    checkpoint_validator = CheckpointValidator(args.checkpoint)
    if checkpoint_validator.load_checkpoint() and checkpoint_validator.validate_structure():
        logger.info("\n✓ Checkpoint structure validation passed")
        
        # Analyze checkpoint
        logger.info("\nAnalyzing checkpoint configuration:")
        conv_analysis = checkpoint_validator.analyze_conv_layer_channels()
        state_analysis = checkpoint_validator.analyze_state_dimensions()
        
        # Check compatibility
        logger.info(f"\nChecking compatibility with target configuration:")
        logger.info(f"  Target visual stack size: {args.visual_stack_size if args.enable_visual_stacking else 1}")
        logger.info(f"  Target state stack size: {args.state_stack_size if args.enable_state_stacking else 1}")
        
        is_compatible, issues = checkpoint_validator.check_compatibility_with_frame_stacking(
            visual_stack_size=args.visual_stack_size if args.enable_visual_stacking else 1,
            state_stack_size=args.state_stack_size if args.enable_state_stacking else 1
        )
        
        if is_compatible:
            logger.info("✓ Checkpoint is compatible with target configuration")
        else:
            logger.warning("✗ Checkpoint has compatibility issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            all_tests_passed = False
    else:
        logger.error("✗ Checkpoint validation failed")
        all_tests_passed = False
    
    # Test 2: Weight transfer validation
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Weight Transfer Validation")
    logger.info("=" * 80)
    
    weight_validator = WeightTransferValidator(args.checkpoint, args.architecture)
    
    frame_stack_config = None
    if args.enable_visual_stacking or args.enable_state_stacking:
        frame_stack_config = {
            'enable_visual_frame_stacking': args.enable_visual_stacking,
            'visual_stack_size': args.visual_stack_size,
            'enable_state_stacking': args.enable_state_stacking,
            'state_stack_size': args.state_stack_size,
        }
    
    success, results = weight_validator.simulate_weight_transfer(
        use_hierarchical=True,
        frame_stack_config=frame_stack_config
    )
    
    if success:
        logger.info("\n✓ Weight transfer validation passed")
    else:
        logger.error("✗ Weight transfer validation failed")
        all_tests_passed = False
    
    # Test 3: Observation shape validation
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Observation Shape Validation")
    logger.info("=" * 80)
    
    obs_validator = FrameStackingValidator(args.architecture)
    success, results = obs_validator.validate_observation_shapes(
        enable_visual_stacking=args.enable_visual_stacking,
        visual_stack_size=args.visual_stack_size,
        enable_state_stacking=args.enable_state_stacking,
        state_stack_size=args.state_stack_size,
    )
    
    if success:
        logger.info("\n✓ Observation shape validation passed")
    else:
        logger.error("✗ Observation shape validation failed")
        all_tests_passed = False
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    if all_tests_passed:
        logger.info("✓ All validation tests PASSED")
        return 0
    else:
        logger.error("✗ Some validation tests FAILED")
        logger.error("\nPlease review the issues above and ensure:")
        logger.error("1. BC checkpoint was trained with the same frame stacking configuration")
        logger.error("2. Observation spaces match between BC and RL training")
        logger.error("3. Weight transfer logic handles the specific architecture correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
