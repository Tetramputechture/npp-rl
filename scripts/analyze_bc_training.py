#!/usr/bin/env python3
"""
Analyze BC training to understand why loss remains high (~0.5) after 10 epochs.
"""

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def analyze_bc_checkpoint(checkpoint_path: Path):
    """Analyze BC checkpoint for training metrics."""
    logger.info("=" * 80)
    logger.info("BC TRAINING ANALYSIS")
    logger.info("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    logger.info(f"\n1. CHECKPOINT METADATA:")
    if 'metadata' in checkpoint:
        metadata = checkpoint['metadata']
        logger.info(f"  Architecture: {metadata.get('architecture', 'N/A')}")
        logger.info(f"  Training epochs: {metadata.get('epochs', 'N/A')}")
        logger.info(f"  Total samples seen: {metadata.get('total_samples', 'N/A')}")
        logger.info(f"  Final loss: {metadata.get('final_loss', 'N/A')}")
        logger.info(f"  Best loss: {metadata.get('best_loss', 'N/A')}")
        
        if 'training_history' in metadata:
            history = metadata['training_history']
            logger.info(f"\n2. TRAINING HISTORY:")
            logger.info(f"  Number of recorded epochs: {len(history)}")
            if history:
                logger.info(f"  First epoch loss: {history[0].get('loss', 'N/A')}")
                logger.info(f"  Last epoch loss: {history[-1].get('loss', 'N/A')}")
                logger.info(f"  Loss improvement: {history[0].get('loss', 0) - history[-1].get('loss', 0):.4f}")
    
    # Analyze action distribution if available
    logger.info(f"\n3. ACTION SPACE ANALYSIS:")
    if 'policy_net' in checkpoint or 'state_dict' in checkpoint:
        state_dict = checkpoint.get('state_dict', checkpoint.get('policy_net', {}))
        
        # Look for policy head weights
        policy_weights = None
        for key in state_dict:
            if 'policy_head' in key and 'weight' in key and 'bias' not in key:
                policy_weights = state_dict[key]
                break
        
        if policy_weights is not None:
            logger.info(f"  Policy head output shape: {policy_weights.shape}")
            logger.info(f"  Number of actions: {policy_weights.shape[0]}")
            
            # Analyze weight magnitudes
            weight_norms = torch.norm(policy_weights, dim=1)
            logger.info(f"  Weight norms per action:")
            for i, norm in enumerate(weight_norms):
                logger.info(f"    Action {i}: {norm:.4f}")
    
    return checkpoint


def analyze_action_distribution():
    """Analyze expected action distribution for N++."""
    logger.info("\n4. EXPECTED ACTION DISTRIBUTION FOR N++:")
    logger.info("  N++ is a fast-paced platformer where:")
    logger.info("  - Players are ALWAYS moving (left/right)")
    logger.info("  - Jump is frequently pressed (but not every frame)")
    logger.info("  - No-action (standing still) is RARE")
    logger.info("")
    logger.info("  Expected distribution:")
    logger.info("    - no_action (0):     ~5%   (very rare)")
    logger.info("    - left (1):          ~20%  (moving left)")
    logger.info("    - right (2):         ~20%  (moving right)")
    logger.info("    - jump (3):          ~10%  (neutral jump)")
    logger.info("    - left_jump (4):     ~20%  (most common)")
    logger.info("    - right_jump (5):    ~25%  (most common)")
    logger.info("")
    logger.info("  ⚠️  If BC model predicts uniform distribution, loss ~0.5 is expected!")
    logger.info("      Cross-entropy loss for 6-class uniform: -log(1/6) ≈ 1.79")
    logger.info("      But if target is heavily skewed, loss could be lower")


def analyze_potential_issues():
    """Analyze potential causes for high BC loss."""
    logger.info("\n5. POTENTIAL CAUSES FOR HIGH BC LOSS (~0.5):")
    logger.info("")
    logger.info("  A. DATA QUALITY ISSUES:")
    logger.info("     1. Noisy labels - Are replay actions correct?")
    logger.info("     2. Ambiguous states - Multiple valid actions for same observation")
    logger.info("     3. Missing context - Frame stacking not properly implemented")
    logger.info("     4. Observation mismatch - BC sees different obs than during replay")
    logger.info("")
    logger.info("  B. MODEL CAPACITY ISSUES:")
    logger.info("     1. Model too small - Cannot learn complex mapping")
    logger.info("     2. Model too large - Overfitting to noise")
    logger.info("     3. Wrong architecture - Visual features not properly extracted")
    logger.info("")
    logger.info("  C. TRAINING ISSUES:")
    logger.info("     1. Learning rate too high/low")
    logger.info("     2. Batch size suboptimal")
    logger.info("     3. Data imbalance - Some actions underrepresented")
    logger.info("     4. Insufficient epochs - Model hasn't converged")
    logger.info("     5. Dataset too diverse - Mixing different skill levels")
    logger.info("")
    logger.info("  D. FRAME STACKING ISSUES:")
    logger.info("     1. BC trained WITH frame stacking, but observations WITHOUT")
    logger.info("     2. BC trained WITHOUT frame stacking, but observations WITH")
    logger.info("     3. Stack size mismatch between training and data")
    logger.info("     4. Padding strategy wrong (zero vs repeat)")


def check_frame_stacking_consistency(checkpoint_path: Path):
    """Check if frame stacking is properly configured."""
    logger.info("\n6. FRAME STACKING CONFIGURATION CHECK:")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('policy_net', {}))
    
    # Check CNN input channels
    player_cnn_found = False
    global_cnn_found = False
    
    for key, tensor in state_dict.items():
        if 'player_frame_cnn.conv_layers.0.weight' in key:
            player_cnn_found = True
            in_channels = tensor.shape[1]
            logger.info(f"  ✓ Player frame CNN: {in_channels} input channels")
            if in_channels == 1:
                logger.warning("    ⚠️  Frame stacking NOT enabled (expected 4 channels)")
            elif in_channels == 4:
                logger.info("    ✓ Frame stacking enabled (4 channels)")
        
        if 'global_cnn.conv_layers.0.weight' in key:
            global_cnn_found = True
            in_channels = tensor.shape[1]
            logger.info(f"  ✓ Global CNN: {in_channels} input channels")
            if in_channels == 1:
                logger.warning("    ⚠️  Frame stacking NOT enabled (expected 4 channels)")
            elif in_channels == 4:
                logger.info("    ✓ Frame stacking enabled (4 channels)")
    
    if not player_cnn_found and not global_cnn_found:
        logger.error("  ✗ No CNN layers found! Using non-visual architecture?")
    
    # Check metadata
    if 'metadata' in checkpoint and 'frame_stacking' in checkpoint['metadata']:
        fs_config = checkpoint['metadata']['frame_stacking']
        logger.info(f"\n  Metadata frame stacking config:")
        logger.info(f"    Visual stacking: {fs_config.get('enable_visual_frame_stacking', False)}")
        logger.info(f"    Visual stack size: {fs_config.get('visual_stack_size', 'N/A')}")
        logger.info(f"    State stacking: {fs_config.get('enable_state_stacking', False)}")
        logger.info(f"    State stack size: {fs_config.get('state_stack_size', 'N/A')}")


def recommendations():
    """Provide recommendations for improving BC training."""
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS TO IMPROVE BC TRAINING")
    logger.info("=" * 80)
    logger.info("")
    logger.info("1. CHECK DATA QUALITY:")
    logger.info("   - Inspect a few samples visually")
    logger.info("   - Verify actions make sense for observations")
    logger.info("   - Check action distribution matches expectations")
    logger.info("")
    logger.info("2. VERIFY OBSERVATION CONSISTENCY:")
    logger.info("   - BC training observations MUST match replay observations")
    logger.info("   - Frame stacking must be identical")
    logger.info("   - Preprocessing must be identical")
    logger.info("")
    logger.info("3. TUNE HYPERPARAMETERS:")
    logger.info("   - Try different learning rates (1e-3, 5e-4, 1e-4)")
    logger.info("   - Adjust batch size (32, 64, 128)")
    logger.info("   - Train for more epochs (20-50)")
    logger.info("")
    logger.info("4. ADDRESS DATA IMBALANCE:")
    logger.info("   - Use weighted cross-entropy loss")
    logger.info("   - Oversample rare actions")
    logger.info("   - Use focal loss for hard examples")
    logger.info("")
    logger.info("5. IMPROVE MODEL:")
    logger.info("   - Add dropout for regularization")
    logger.info("   - Use batch normalization")
    logger.info("   - Try larger feature extractors")
    logger.info("")
    logger.info("6. VALIDATE PRETRAINING:")
    logger.info("   - Run validate_bc_weight_transfer.py")
    logger.info("   - Ensure weights are properly loaded")
    logger.info("   - Check that pretrained model helps RL performance")
    logger.info("")
    logger.info("7. BENCHMARK:")
    logger.info("   - Compare BC loss to random policy (~1.79 for 6 classes)")
    logger.info("   - Loss of 0.5 means model is MUCH better than random")
    logger.info("   - But may not be optimal for all levels")


def main():
    parser = argparse.ArgumentParser(description="Analyze BC training")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to BC checkpoint')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Run analyses
    analyze_bc_checkpoint(checkpoint_path)
    analyze_action_distribution()
    check_frame_stacking_consistency(checkpoint_path)
    analyze_potential_issues()
    recommendations()
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
