"""
FIXED Reward Constants - Addressing Critical Training Issues

This file contains the corrected reward values based on comprehensive training analysis.
See COMPREHENSIVE_TRAINING_ANALYSIS.md for full details.

KEY CHANGES FROM ORIGINAL:
1. Reduced time penalty 100x (was -0.01, now -0.0001)
2. Increased completion reward 10x (was 1.0, now 10.0)
3. Increased switch reward 10x (was 0.1, now 1.0)
4. Increased dense shaping 10x for better learning signal
5. Boosted exploration rewards for better coverage

CRITICAL FIX: Original reward scaling made level completion impossible to learn!
- Old: Fast completion (1000 steps) = +1.0 - 10.0 = -9.0 (NEGATIVE!)
- New: Fast completion (1000 steps) = +10.0 - 0.1 = +9.9 (POSITIVE!)

Apply these changes to: nclone/nclone/gym_environment/reward_calculation/reward_constants.py
"""

# =============================================================================
# TERMINAL REWARD CONSTANTS (FIXED)
# =============================================================================

# Level completion reward - INCREASED 10x
# Was: 1.0 (insufficient to overcome time penalty)
# Now: 10.0 (ensures positive returns for successful completion)
LEVEL_COMPLETION_REWARD = 10.0

# Death penalty - Keep proportional to completion
# Was: -0.5 (reasonable)
# Now: -0.5 (unchanged, still proportional)
DEATH_PENALTY = -0.5

# Switch activation reward - INCREASED 10x
# Was: 0.1 (too small)
# Now: 1.0 (meaningful milestone reward)
SWITCH_ACTIVATION_REWARD = 1.0


# =============================================================================
# TIME-BASED REWARDS (CRITICAL FIX)
# =============================================================================

# Time penalty per step - REDUCED 100x
# Was: -0.01 (catastrophic - accumulated to -200 over max episode)
# Now: -0.0001 (manageable - accumulates to -2.0 over max episode)
#
# Impact analysis:
# Fast completion (1000 steps):  +10.0 - 0.1 = +9.9 ✓
# Medium (5000 steps):           +10.0 - 0.5 = +9.5 ✓
# Slow (10000 steps):            +10.0 - 1.0 = +9.0 ✓
# Very slow (15000 steps):       +10.0 - 1.5 = +8.5 ✓
# Max episode (20000 steps):     +10.0 - 2.0 = +8.0 ✓
#
# Death scenarios:
# Early death (2000 steps):      -0.5 - 0.2 = -0.7 (clear penalty)
# Mid death (10000 steps):       -0.5 - 1.0 = -1.5 (stronger penalty)
TIME_PENALTY_PER_STEP = -0.0001


# =============================================================================
# NAVIGATION REWARD SHAPING CONSTANTS (INCREASED 10x)
# =============================================================================

# Distance improvement scale - INCREASED 10x
# Was: 0.0001 (too weak to provide meaningful signal)
# Now: 0.001 (provides dense feedback)
#
# Example: Moving 50 pixels toward objective = 0.05 reward
# This is now comparable to time penalty and provides clear learning signal
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.001

# Minimum distance threshold (unchanged - already reasonable)
NAVIGATION_MIN_DISTANCE_THRESHOLD = 20.0

# Potential-based shaping scale (unchanged - needs testing)
NAVIGATION_POTENTIAL_SCALE = 0.0005


# =============================================================================
# EXPLORATION REWARD CONSTANTS (INCREASED 10x)
# =============================================================================

# Grid dimensions (unchanged)
EXPLORATION_GRID_WIDTH = 44
EXPLORATION_GRID_HEIGHT = 25
EXPLORATION_CELL_SIZE = 24.0

# Exploration rewards at different scales - INCREASED 10x
# Was: 0.001 each (too weak)
# Now: 0.01 each (meaningful exploration incentive)
#
# Total max exploration per step: 0.04 (vs time penalty 0.0001)
# This encourages exploration while maintaining completion focus
EXPLORATION_CELL_REWARD = 0.01
EXPLORATION_AREA_4X4_REWARD = 0.01
EXPLORATION_AREA_8X8_REWARD = 0.01
EXPLORATION_AREA_16X16_REWARD = 0.01


# =============================================================================
# PBRS CONSTANTS (INCREASED 10x)
# =============================================================================

# PBRS gamma (unchanged)
PBRS_GAMMA = 0.99

# PBRS weights - RECOMMENDED ACTIVE CONFIGURATION
PBRS_OBJECTIVE_WEIGHT = 1.0      # Enable objective shaping (was: 1.0)
PBRS_HAZARD_WEIGHT = 0.1         # Mild hazard avoidance (was: 0.0)
PBRS_IMPACT_WEIGHT = 0.0         # Disable for completion focus (was: 0.0)
PBRS_EXPLORATION_WEIGHT = 0.2    # Mild exploration bonus (was: 0.0)

# PBRS scaling - INCREASED 10x
# Was: 0.05 (too weak)
# Now: 0.5 (provides meaningful gradient)
PBRS_SWITCH_DISTANCE_SCALE = 0.5
PBRS_EXIT_DISTANCE_SCALE = 0.5


# =============================================================================
# REWARD VALIDATION
# =============================================================================

def validate_fixed_rewards():
    """Validate that reward fixes achieve positive returns."""
    
    print("="*80)
    print("REWARD SCALING VALIDATION")
    print("="*80)
    
    scenarios = [
        ("Fast completion", 1000, True, False),
        ("Medium completion", 5000, True, False),
        ("Slow completion", 10000, True, False),
        ("Very slow completion", 15000, True, False),
        ("Max length completion", 20000, True, False),
        ("Early death", 2000, False, True),
        ("Mid death", 10000, False, True),
        ("Late death", 18000, False, True),
    ]
    
    print("\nScenario Analysis:")
    print("-" * 80)
    
    all_valid = True
    
    for name, steps, completed, died in scenarios:
        time_penalty = TIME_PENALTY_PER_STEP * steps
        completion = LEVEL_COMPLETION_REWARD if completed else 0
        death = DEATH_PENALTY if died else 0
        switch = SWITCH_ACTIVATION_REWARD if completed else 0
        
        total = completion + switch + death + time_penalty
        
        # Expected outcomes
        if completed:
            expected = "POSITIVE"
            valid = total > 0
        else:
            expected = "NEGATIVE"
            valid = total < 0
            
        status = "✓" if valid else "✗ ERROR"
        all_valid = all_valid and valid
        
        print(f"{name:25} ({steps:5} steps):")
        print(f"  Completion: {completion:+7.2f}")
        print(f"  Switch:     {switch:+7.2f}")
        print(f"  Death:      {death:+7.2f}")
        print(f"  Time:       {time_penalty:+7.2f}")
        print(f"  TOTAL:      {total:+7.2f}  Expected: {expected:8}  {status}")
        print()
    
    print("="*80)
    if all_valid:
        print("✓ ALL SCENARIOS VALID - Reward scaling is CORRECT")
    else:
        print("✗ VALIDATION FAILED - Reward scaling has issues")
    print("="*80)
    
    return all_valid


if __name__ == "__main__":
    validate_fixed_rewards()
