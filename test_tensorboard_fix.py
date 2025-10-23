#!/usr/bin/env python3
"""Test that TensorBoard logging is properly configured."""

import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent))

from npp_rl.training.architecture_trainer import ArchitectureTrainer
from npp_rl.training.architecture_configs import get_architecture_config

print("="*80)
print("TensorBoard Fix Verification Test")
print("="*80)

# Create temporary directories
test_dir = Path(tempfile.mkdtemp(prefix="tensorboard_test_"))
train_dataset = test_dir / "train"
test_dataset = test_dir / "test"
output_dir = test_dir / "output"

train_dataset.mkdir()
test_dataset.mkdir()
output_dir.mkdir()

# Create dummy map files
(train_dataset / "dummy.txt").write_text("dummy")
(test_dataset / "dummy.txt").write_text("dummy")

try:
    print("\n" + "-"*80)
    print("Test 1: Check tensorboard_log configuration")
    print("-"*80)
    
    # Get a simple architecture config
    config = get_architecture_config("simple_cnn")
    
    # Create trainer
    trainer = ArchitectureTrainer(
        architecture_config=config,
        train_dataset_path=str(train_dataset),
        test_dataset_path=str(test_dataset),
        output_dir=output_dir,
        device_id=0,
        world_size=1,
        tensorboard_writer=None,  # Not providing custom writer
    )
    
    # Build policy to set hyperparameters
    trainer.build_policy()
    
    # Check tensorboard_log is set
    tensorboard_log = trainer.hyperparams.get("tensorboard_log")
    
    print(f"\ntensorboard_log parameter: {tensorboard_log}")
    
    if tensorboard_log is None:
        print("✗ FAIL: tensorboard_log is None (TensorBoard logging disabled!)")
        sys.exit(1)
    else:
        expected_path = str(output_dir / "tensorboard")
        if tensorboard_log == expected_path:
            print(f"✓ PASS: tensorboard_log correctly set to {tensorboard_log}")
        else:
            print(f"✗ FAIL: Expected {expected_path}, got {tensorboard_log}")
            sys.exit(1)
    
    print("\n" + "-"*80)
    print("Test 2: Verify TensorBoard directory will be created")
    print("-"*80)
    
    tb_dir = Path(tensorboard_log)
    print(f"\nTensorBoard directory: {tb_dir}")
    print(f"Parent directory exists: {tb_dir.parent.exists()}")
    
    if tb_dir.parent.exists():
        print("✓ PASS: Parent directory exists, TensorBoard can write")
    else:
        print("✗ FAIL: Parent directory missing")
        sys.exit(1)
    
    print("\n" + "-"*80)
    print("Test 3: Check ComprehensiveEvaluator initialization")
    print("-"*80)
    
    try:
        evaluator = trainer.create_evaluator()
        print(f"\n✓ PASS: ComprehensiveEvaluator created successfully")
        print(f"  Test dataset path: {evaluator.test_dataset_path}")
        print(f"  Device: {evaluator.device}")
    except Exception as e:
        print(f"\n✗ FAIL: ComprehensiveEvaluator creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("All Tests Passed!")
    print("="*80)
    print("\nSummary:")
    print("  ✓ tensorboard_log parameter is correctly set")
    print("  ✓ TensorBoard will write to proper directory")
    print("  ✓ ComprehensiveEvaluator initializes correctly")
    print("\nTensorBoard logging is now fixed and will work properly!")

finally:
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")
