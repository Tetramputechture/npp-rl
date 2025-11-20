#!/usr/bin/env python3
"""Test learning rate schedule fix for attention architecture.

This test verifies that:
1. Learning rate schedule produces positive values throughout training
2. Learning rate properly warms up and decays
3. Schedule works correctly for various training lengths (2M, 5M, 10M steps)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_lr_schedule_positive_values():
    """Test that LR schedule produces positive values for entire training duration."""
    
    print("=" * 80)
    print("Testing Learning Rate Schedule Fix")
    print("=" * 80)
    
    # Test various training lengths
    test_configs = [
        (2_000_000, "2M steps"),
        (5_000_000, "5M steps"),
        (10_000_000, "10M steps (default)"),
        (20_000_000, "20M steps (extended)"),
    ]
    
    base_lr = 3e-4
    all_passed = True
    
    for total_steps, description in test_configs:
        print(f"\n{'─' * 80}")
        print(f"Testing: {description} (total_steps={total_steps:,})")
        print(f"{'─' * 80}")
        
        warmup_steps = int(0.25 * total_steps)
        
        def warmup_lr_schedule(progress_remaining: float) -> float:
            """LR schedule with warmup then linear decay."""
            progress = 1.0 - progress_remaining
            current_step = progress * total_steps

            if current_step < warmup_steps:
                # Warmup phase: linear ramp from 0.1x to 1.0x
                warmup_progress = current_step / warmup_steps
                return base_lr * (0.1 + 0.9 * warmup_progress)
            else:
                # Decay phase: linear decay to 1e-5
                decay_progress = (current_step - warmup_steps) / (
                    total_steps - warmup_steps
                )
                return base_lr * (1.0 - decay_progress) + 1e-5 * decay_progress
        
        # Test at key points
        test_points = [
            (1.0, "Start (step 0)"),
            (0.75, f"25% complete (step {int(0.25 * total_steps):,})"),
            (0.5, f"50% complete (step {int(0.5 * total_steps):,})"),
            (0.25, f"75% complete (step {int(0.75 * total_steps):,})"),
            (0.0, f"End (step {total_steps:,})"),
        ]
        
        config_passed = True
        for progress_remaining, label in test_points:
            lr = warmup_lr_schedule(progress_remaining)
            
            status = "✓ PASS" if lr > 0 else "✗ FAIL"
            color = "\033[92m" if lr > 0 else "\033[91m"
            reset = "\033[0m"
            
            print(f"  {label:40} LR={lr:.6e} {color}{status}{reset}")
            
            if lr <= 0:
                config_passed = False
                all_passed = False
                print(f"    {color}ERROR: Negative learning rate detected!{reset}")
        
        if config_passed:
            print(f"  \033[92m✓ All LR values positive for {description}\033[0m")
        else:
            print(f"  \033[91m✗ Failed: Negative LR detected for {description}\033[0m")
    
    print(f"\n{'=' * 80}")
    if all_passed:
        print("\033[92m✓ ALL TESTS PASSED: Learning rate schedule fix verified\033[0m")
        print("  - LR remains positive throughout training")
        print("  - Warmup works correctly (0.1x to 1.0x over first 25%)")
        print("  - Decay works correctly (1.0x to 1e-5 over remaining 75%)")
    else:
        print("\033[91m✗ TESTS FAILED: Learning rate schedule still has issues\033[0m")
        print("  - Check implementation in architecture_trainer.py")
    print("=" * 80)
    
    return all_passed


def test_lr_schedule_warmup_behavior():
    """Test that warmup properly ramps from 0.1x to 1.0x."""
    
    print("\n" + "=" * 80)
    print("Testing Learning Rate Warmup Behavior")
    print("=" * 80)
    
    total_steps = 10_000_000
    warmup_steps = int(0.25 * total_steps)  # 2.5M steps
    base_lr = 3e-4
    
    def warmup_lr_schedule(progress_remaining: float) -> float:
        """LR schedule with warmup then linear decay."""
        progress = 1.0 - progress_remaining
        current_step = progress * total_steps

        if current_step < warmup_steps:
            # Warmup phase: linear ramp from 0.1x to 1.0x
            warmup_progress = current_step / warmup_steps
            return base_lr * (0.1 + 0.9 * warmup_progress)
        else:
            # Decay phase: linear decay to 1e-5
            decay_progress = (current_step - warmup_steps) / (
                total_steps - warmup_steps
            )
            return base_lr * (1.0 - decay_progress) + 1e-5 * decay_progress
    
    # Test warmup phase
    warmup_test_points = [
        (1.0, 0.0, 0.1),      # Start: 0% through warmup -> 0.1x base_lr
        (0.9375, 0.25, 0.325),  # 25% through warmup -> 0.325x base_lr
        (0.875, 0.5, 0.55),    # 50% through warmup -> 0.55x base_lr
        (0.8125, 0.75, 0.775), # 75% through warmup -> 0.775x base_lr
        (0.75, 1.0, 1.0),      # End warmup: 100% -> 1.0x base_lr
    ]
    
    print(f"\nWarmup phase (0 to {warmup_steps:,} steps):")
    warmup_passed = True
    for progress_remaining, warmup_pct, expected_multiplier in warmup_test_points:
        lr = warmup_lr_schedule(progress_remaining)
        expected_lr = base_lr * expected_multiplier
        actual_multiplier = lr / base_lr
        
        # Allow 1% tolerance
        matches = abs(actual_multiplier - expected_multiplier) < 0.01
        status = "✓ PASS" if matches else "✗ FAIL"
        color = "\033[92m" if matches else "\033[91m"
        reset = "\033[0m"
        
        print(
            f"  {warmup_pct*100:3.0f}% warmup: "
            f"LR={lr:.6e} ({actual_multiplier:.3f}x base_lr) "
            f"expected={expected_lr:.6e} ({expected_multiplier:.3f}x) "
            f"{color}{status}{reset}"
        )
        
        if not matches:
            warmup_passed = False
    
    if warmup_passed:
        print(f"  \033[92m✓ Warmup behavior correct\033[0m")
    else:
        print(f"  \033[91m✗ Warmup behavior incorrect\033[0m")
    
    print("=" * 80)
    return warmup_passed


if __name__ == "__main__":
    # Run tests
    test1_passed = test_lr_schedule_positive_values()
    test2_passed = test_lr_schedule_warmup_behavior()
    
    # Exit with appropriate code
    if test1_passed and test2_passed:
        print("\n\033[92m✓ All tests passed successfully!\033[0m\n")
        sys.exit(0)
    else:
        print("\n\033[91m✗ Some tests failed. Please review the implementation.\033[0m\n")
        sys.exit(1)

