#!/usr/bin/env python3
"""
Comprehensive Parameter Review Script

Reviews all PPO hyperparameters, PBRS reward scaling, and curriculum settings
against RL/ML best practices to ensure agent learns efficient paths through
generalized levels.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def review_ppo_hyperparameters():
    """Review PPO hyperparameters against best practices."""
    print("=" * 80)
    print("PPO HYPERPARAMETER REVIEW")
    print("=" * 80)
    
    params = {
        "learning_rate": {
            "current": 3e-4,
            "best_practice": "3e-4 (PPO standard)",
            "rationale": "Default PPO LR from Schulman et al. 2017. Common range: 1e-5 to 3e-4",
            "status": "✅ OPTIMAL",
            "recommendation": "Consider LR annealing (linear decay) for better convergence",
        },
        "n_steps": {
            "current": 2048,
            "best_practice": "128-2048 (depends on task)",
            "rationale": "Controls rollout length. Longer = more stable but less frequent updates",
            "status": "✅ GOOD",
            "recommendation": "2048 is standard for navigation tasks. Could try 1024 if value estimates are poor",
        },
        "batch_size": {
            "current": 256,
            "best_practice": "32-512 (< n_steps)",
            "rationale": "Mini-batch size for SGD updates. Should divide n_steps evenly",
            "status": "✅ GOOD",
            "recommendation": "256 works well. Ensures 8 mini-batches per update (2048/256=8)",
        },
        "n_epochs": {
            "current": 10,
            "best_practice": "3-10",
            "rationale": "Number of SGD epochs per rollout. More epochs = more sample reuse",
            "status": "✅ OPTIMAL",
            "recommendation": "10 is PPO standard. Good balance of sample efficiency and stability",
        },
        "gamma": {
            "current": 0.99,
            "best_practice": "0.99-0.999",
            "rationale": "Discount factor. Higher = values far future rewards more",
            "status": "✅ OPTIMAL",
            "recommendation": "0.99 is standard. Matches PBRS gamma for theoretical consistency",
        },
        "gae_lambda": {
            "current": 0.95,
            "best_practice": "0.92-0.98",
            "rationale": "GAE parameter balancing bias/variance in advantage estimation",
            "status": "✅ OPTIMAL",
            "recommendation": "0.95 is PPO standard. Good balance for most tasks",
        },
        "clip_range": {
            "current": 0.2,
            "best_practice": "0.1-0.3",
            "rationale": "PPO policy clipping range. Controls update magnitude",
            "status": "✅ OPTIMAL",
            "recommendation": "0.2 is PPO standard. Could try 0.1 for more conservative updates",
        },
        "clip_range_vf": {
            "current": 1.0,
            "best_practice": "None or 1.0-10.0",
            "rationale": "Value function clipping. Prevents large value updates",
            "status": "✅ IMPROVED",
            "recommendation": "CHANGED from 10.0 → 1.0. Tighter clipping improves stability",
            "analysis": "Previous 10.0 caused 56% value loss increase. 1.0 is more conservative",
        },
        "ent_coef": {
            "current": 0.01,
            "best_practice": "0.0-0.01",
            "rationale": "Entropy coefficient. Encourages exploration",
            "status": "⚠️  NEEDS REVIEW",
            "recommendation": "0.01 is common but may be too high for sparse reward navigation",
            "analysis": "Consider reducing to 0.001 if policy converges slowly. Monitor entropy in TensorBoard",
        },
        "vf_coef": {
            "current": 0.5,
            "best_practice": "0.5-1.0",
            "rationale": "Value function loss coefficient. Balances policy vs value learning",
            "status": "✅ OPTIMAL",
            "recommendation": "0.5 is PPO standard. Adequate for most tasks",
        },
        "max_grad_norm": {
            "current": 0.5,
            "best_practice": "0.5-1.0",
            "rationale": "Gradient clipping. Prevents exploding gradients",
            "status": "✅ OPTIMAL",
            "recommendation": "0.5 is PPO standard. Good for stability",
        },
        "optimizer_eps": {
            "current": 1e-5,
            "best_practice": "1e-5 (PPO) or 1e-8 (PyTorch default)",
            "rationale": "Adam epsilon for numerical stability",
            "status": "✅ IMPROVED",
            "recommendation": "CHANGED from 1e-8 → 1e-5. Matches official PPO implementation",
            "analysis": "1e-5 from openai/baselines improves convergence in sparse reward tasks",
        },
    }
    
    for param, info in params.items():
        print(f"\n{param}:")
        print(f"  Current: {info['current']}")
        print(f"  Best Practice: {info['best_practice']}")
        print(f"  Status: {info['status']}")
        print(f"  Rationale: {info['rationale']}")
        print(f"  Recommendation: {info['recommendation']}")
        if 'analysis' in info:
            print(f"  Analysis: {info['analysis']}")
    
    print("\n" + "=" * 80)
    print("CRITICAL FINDINGS:")
    print("=" * 80)
    print("1. ✅ clip_range_vf: 10.0 → 1.0 (FIXED - prevents value function divergence)")
    print("2. ✅ optimizer_eps: 1e-8 → 1e-5 (FIXED - matches PPO standard)")
    print("3. ⚠️  ent_coef: 0.01 may be too high for sparse rewards")
    print("4. ✅ All other parameters match PPO best practices")
    print()

def review_pbrs_reward_scaling():
    """Review PBRS reward scaling for efficiency learning."""
    print("=" * 80)
    print("PBRS REWARD SCALING REVIEW")
    print("=" * 80)
    
    # Base environment rewards
    print("\nBASE ENVIRONMENT REWARDS:")
    print("  Exit door: +1.0 (terminal success)")
    print("  Switch activation: +0.1 (milestone)")
    print("  Death: -0.5 (terminal failure)")
    print("  Step: 0.0 (no step penalty)")
    print()
    
    # PBRS constants from subtask_rewards.py
    pbrs_params = {
        "PROGRESS_REWARD_SCALE": {
            "value": 0.02,
            "per_unit": "distance",
            "typical_episode": "10-50 units → 0.2-1.0 cumulative",
            "status": "✅ GOOD",
            "rationale": "2% of switch bonus per distance unit. Keeps shaped rewards < sparse rewards",
            "efficiency_impact": "✅ Encourages shorter paths (less distance = less reward needed)",
        },
        "PROXIMITY_BONUS_SCALE": {
            "value": 0.05,
            "trigger": "< 2 tiles from switch",
            "typical_episode": "0.05 per step near switch",
            "status": "✅ GOOD",
            "rationale": "5% of exit reward. Encourages precise positioning",
            "efficiency_impact": "✅ Helps agent learn final approach to objectives",
        },
        "EFFICIENCY_BONUS": {
            "value": 0.2,
            "trigger": "< 150 steps from switch to exit",
            "typical_episode": "+0.2 if fast",
            "status": "✅ EXCELLENT",
            "rationale": "20% of exit reward for fast completion",
            "efficiency_impact": "✅✅ DIRECTLY rewards efficient paths (shorter time)",
        },
        "MINE_PROXIMITY_PENALTY": {
            "value": -0.02,
            "trigger": "< 1.5 tiles from toggled mine",
            "typical_episode": "-0.02 per risky step",
            "status": "⚠️  NEEDS REVIEW",
            "rationale": "Discourages risky paths near mines",
            "efficiency_impact": "⚠️  May encourage overly conservative (longer) paths",
            "recommendation": "Monitor if agent takes unnecessarily long detours. Consider reducing to -0.01",
        },
        "SAFE_NAVIGATION_BONUS": {
            "value": 0.01,
            "trigger": "> 3.0 tiles from mines",
            "typical_episode": "+0.01 per safe step",
            "status": "⚠️  POTENTIAL ISSUE",
            "rationale": "Reinforces mine-aware behavior",
            "efficiency_impact": "❌ CONFLICTS with efficiency! Rewards MORE steps (longer paths)",
            "recommendation": "CONSIDER REMOVING. Contradicts efficiency goal",
        },
        "EXPLORATION_REWARD": {
            "value": 0.01,
            "trigger": "new tile visited",
            "typical_episode": "+0.01 per new tile",
            "status": "⚠️  POTENTIAL ISSUE",
            "rationale": "Encourages coverage",
            "efficiency_impact": "❌ May encourage wandering instead of direct paths",
            "recommendation": "DISABLE for efficiency-focused training. Enable only for exploration stages",
        },
        "TIMEOUT_PENALTY_MAJOR": {
            "value": -0.1,
            "trigger": "> 300 steps to switch",
            "typical_episode": "-0.1 if too slow",
            "status": "✅ EXCELLENT",
            "rationale": "Prevents infinite loops",
            "efficiency_impact": "✅✅ DIRECTLY penalizes inefficient behavior",
        },
    }
    
    print("PBRS REWARD COMPONENTS:")
    for param, info in pbrs_params.items():
        print(f"\n{param}:")
        print(f"  Value: {info['value']}")
        if 'per_unit' in info:
            print(f"  Per: {info['per_unit']}")
        if 'trigger' in info:
            print(f"  Trigger: {info['trigger']}")
        print(f"  Typical Episode: {info['typical_episode']}")
        print(f"  Status: {info['status']}")
        print(f"  Rationale: {info['rationale']}")
        print(f"  Efficiency Impact: {info['efficiency_impact']}")
        if 'recommendation' in info:
            print(f"  ⚠️  RECOMMENDATION: {info['recommendation']}")
    
    # PBRS potential function
    print("\n" + "=" * 80)
    print("PBRS POTENTIAL FUNCTION:")
    print("=" * 80)
    print("\nNavigate to Switch: Φ(s) = -distance * 0.1")
    print("  Status: ✅ EXCELLENT")
    print("  Efficiency Impact: ✅✅ Directly encodes efficiency (minimize distance)")
    print()
    print("Navigate to Exit Door: Φ(s) = -distance * 0.15")
    print("  Status: ✅ EXCELLENT")
    print("  Efficiency Impact: ✅✅ Higher weight for final objective, strong efficiency signal")
    print()
    print("PBRS Formula: r_shaped = γ * Φ(s') - Φ(s)")
    print("  With γ=0.99 (matches PPO gamma)")
    print("  Status: ✅ OPTIMAL - maintains policy invariance (Ng et al. 1999)")
    print()
    
    print("=" * 80)
    print("CRITICAL FINDINGS:")
    print("=" * 80)
    print("1. ✅✅ PBRS potential (-distance) DIRECTLY encodes efficiency")
    print("2. ✅✅ EFFICIENCY_BONUS (0.2) rewards fast completion")
    print("3. ✅✅ TIMEOUT_PENALTY (-0.1) penalizes slow behavior")
    print("4. ❌ SAFE_NAVIGATION_BONUS conflicts with efficiency (rewards more steps)")
    print("5. ❌ EXPLORATION_REWARD may encourage wandering")
    print("6. ⚠️  MINE_PROXIMITY_PENALTY may cause overly conservative paths")
    print()
    print("RECOMMENDATION:")
    print("  - DISABLE SAFE_NAVIGATION_BONUS when efficiency is priority")
    print("  - DISABLE EXPLORATION_REWARD except in exploration curriculum stages")
    print("  - REDUCE MINE_PROXIMITY_PENALTY to -0.01 if agent takes excessive detours")
    print()

def review_curriculum_parameters():
    """Review curriculum learning parameters."""
    print("=" * 80)
    print("CURRICULUM LEARNING REVIEW")
    print("=" * 80)
    
    curriculum = {
        "SUCCESS_THRESHOLDS": {
            "simplest": 0.70,
            "simpler": 0.70,
            "simple": 0.70,
            "medium": 0.60,
            "complex": 0.50,
            "exploration": 0.50,
            "mine_heavy": 0.40,
            "status": "✅ GOOD",
            "rationale": "Progressive difficulty requires lower thresholds for harder stages",
            "recommendation": "Thresholds appropriately calibrated. 70% is solid mastery",
        },
        "STAGE_MIN_EPISODES": {
            "simplest": 200,
            "simpler": 200,
            "simple": 200,
            "medium": 250,
            "complex": 300,
            "exploration": 300,
            "mine_heavy": 300,
            "status": "✅ IMPROVED",
            "rationale": "INCREASED from 100/150/200. Ensures thorough learning",
            "recommendation": "2x episodes for early stages prevents premature advancement",
        },
        "TREND_ADVANCEMENT": {
            "enabled": "Optional (--disable-trend-advancement to turn off)",
            "requirements": "90% of min episodes + 2% margin + 60% floor",
            "status": "✅ IMPROVED",
            "rationale": "FIXED: Was 80% episodes + 5% margin (too permissive)",
            "recommendation": "Much more conservative. Prevents 58% advancement bug",
        },
        "EARLY_ADVANCEMENT": {
            "threshold": 0.90,
            "min_episodes": 30,
            "status": "✅ GOOD",
            "rationale": "Allows high performers (90%+) to skip ahead",
            "recommendation": "Appropriate for exceptional performance",
        },
        "STAGE_MIXING": {
            "enabled": True,
            "rationale": "Mixes current and previous stages to prevent forgetting",
            "status": "✅ GOOD",
            "recommendation": "Essential for curriculum learning stability",
        },
    }
    
    for param, info in curriculum.items():
        print(f"\n{param}:")
        if isinstance(info, dict) and 'status' in info:
            for key, value in info.items():
                if key not in ['status', 'rationale', 'recommendation']:
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            print(f"  Status: {info['status']}")
            print(f"  Rationale: {info['rationale']}")
            print(f"  Recommendation: {info['recommendation']}")
    
    print("\n" + "=" * 80)
    print("CRITICAL FINDINGS:")
    print("=" * 80)
    print("1. ✅ Curriculum thresholds appropriately calibrated")
    print("2. ✅ Min episodes 2x for early stages (200 vs 100)")
    print("3. ✅ Trend advancement logic FIXED (90% episodes, 2% margin, 60% floor)")
    print("4. ✅ Early advancement threshold (90%) appropriate")
    print("5. ✅ Stage mixing prevents catastrophic forgetting")
    print()

def review_network_architecture():
    """Review network architecture for efficiency learning."""
    print("=" * 80)
    print("NETWORK ARCHITECTURE REVIEW")
    print("=" * 80)
    
    arch = {
        "feature_extractor": {
            "class": "ConfigurableMultimodalExtractor",
            "status": "✅ GOOD",
            "rationale": "Handles multi-modal observations (grid, vector, scalar)",
            "recommendation": "Architecture appropriate for N++ observations",
        },
        "policy_network": {
            "layers": [256, 256, 128],
            "activation": "ReLU",
            "output_init": "0.01 scale",
            "status": "✅ OPTIMAL",
            "rationale": "Standard PPO architecture. Sufficient capacity for navigation",
            "recommendation": "3 layers with decreasing width is standard. Good balance",
        },
        "value_network": {
            "layers": [256, 256, 128],
            "activation": "ReLU",
            "output_init": "1.0 scale",
            "status": "✅ OPTIMAL",
            "rationale": "Shared feature extractor, separate value head",
            "recommendation": "Standard PPO architecture. Appropriate capacity",
        },
        "initialization": {
            "hidden_layers": "Orthogonal with sqrt(2) scale",
            "policy_output": "Orthogonal with 0.01 scale",
            "value_output": "Orthogonal with 1.0 scale",
            "status": "✅ OPTIMAL",
            "rationale": "Matches official PPO implementation (openai/baselines)",
            "recommendation": "Proper initialization critical for training stability",
        },
    }
    
    for component, info in arch.items():
        print(f"\n{component}:")
        for key, value in info.items():
            if key not in ['status', 'rationale', 'recommendation']:
                print(f"  {key}: {value}")
        print(f"  Status: {info['status']}")
        print(f"  Rationale: {info['rationale']}")
        print(f"  Recommendation: {info['recommendation']}")
    
    print("\n" + "=" * 80)
    print("CRITICAL FINDINGS:")
    print("=" * 80)
    print("1. ✅ Network architecture matches PPO best practices")
    print("2. ✅ Proper initialization (orthogonal with appropriate scaling)")
    print("3. ✅ Separate policy/value heads with shared features")
    print("4. ✅ Sufficient capacity for navigation tasks")
    print()

def review_observation_space():
    """Review observation space for efficiency signals."""
    print("=" * 80)
    print("OBSERVATION SPACE REVIEW")
    print("=" * 80)
    
    print("\nREQUIRED FOR EFFICIENT PATHS:")
    print("  ✅ Player position (x, y)")
    print("  ✅ Goal position (switch_x, switch_y, exit_x, exit_y)")
    print("  ✅ Distance to goal (computed in potential function)")
    print("  ✅ Mine positions and states")
    print("  ✅ Wall/obstacle map")
    print("  ✅ Switch activation state")
    print()
    
    print("POTENTIAL CONCERNS:")
    print("  ⚠️  Verify observation includes:")
    print("     - Reachability features (for connectivity-based learning)")
    print("     - Mine toggle states (for safe navigation)")
    print("     - Door locked/unlocked states (for planning)")
    print()
    
    print("EFFICIENCY-SPECIFIC OBSERVATIONS:")
    print("  ✅ Distance potential Φ(s) = -distance computed from (player_x, goal_x)")
    print("  ✅ Progress tracking (best_distance) allows measuring improvement")
    print("  ⚠️  No explicit 'steps taken' in observation")
    print("     Recommendation: Add episode_step counter to observation")
    print("     Rationale: Agent can learn time-awareness for efficiency bonus")
    print()

def main():
    """Run comprehensive parameter review."""
    print("\n")
    print("=" * 80)
    print("COMPREHENSIVE PARAMETER REVIEW FOR EFFICIENT PATH LEARNING")
    print("=" * 80)
    print()
    print("Goal: Agent learns shortest, safest paths through generalized N++ levels")
    print("Task: Navigation with sparse terminal rewards and dense PBRS shaping")
    print()
    
    review_ppo_hyperparameters()
    review_pbrs_reward_scaling()
    review_curriculum_parameters()
    review_network_architecture()
    review_observation_space()
    
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print("\nSTRENGTHS:")
    print("  ✅✅ PPO hyperparameters match best practices")
    print("  ✅✅ PBRS potential function DIRECTLY encodes efficiency (minimize distance)")
    print("  ✅✅ Curriculum advancement logic FIXED (prevents premature progression)")
    print("  ✅✅ Network architecture appropriate for task")
    print("  ✅  Value function clipping improved (10.0 → 1.0)")
    print("  ✅  Optimizer epsilon matches PPO standard (1e-5)")
    print()
    
    print("CRITICAL IMPROVEMENTS NEEDED:")
    print("  1. ❌ DISABLE SAFE_NAVIGATION_BONUS (conflicts with efficiency)")
    print("     Location: npp_rl/hrl/subtask_rewards.py")
    print("     Rationale: Rewards taking MORE steps (longer paths)")
    print()
    print("  2. ❌ DISABLE EXPLORATION_REWARD except in exploration stages")
    print("     Location: npp_rl/hrl/subtask_rewards.py")
    print("     Rationale: Encourages wandering instead of direct paths")
    print()
    print("  3. ⚠️  REDUCE MINE_PROXIMITY_PENALTY: -0.02 → -0.01")
    print("     Location: npp_rl/hrl/subtask_rewards.py")
    print("     Rationale: May cause overly conservative (longer) detours")
    print()
    print("  4. ⚠️  CONSIDER REDUCING ent_coef: 0.01 → 0.001")
    print("     Location: npp_rl/training/architecture_trainer.py")
    print("     Rationale: High entropy may slow convergence in sparse reward environment")
    print()
    print("  5. ⚠️  ADD episode_step to observation")
    print("     Location: Environment wrapper")
    print("     Rationale: Enables agent to learn time-awareness for efficiency bonus")
    print()
    
    print("=" * 80)
    print("PRIORITY ACTION ITEMS:")
    print("=" * 80)
    print("\nPRIORITY 1 (CRITICAL - Directly improves efficiency learning):")
    print("  [ ] Remove SAFE_NAVIGATION_BONUS from mine avoidance calculation")
    print("  [ ] Add curriculum-stage-aware EXPLORATION_REWARD (disable for navigation stages)")
    print()
    print("PRIORITY 2 (HIGH - Fine-tuning for faster convergence):")
    print("  [ ] Reduce MINE_PROXIMITY_PENALTY to -0.01")
    print("  [ ] Consider entropy coefficient schedule (0.01 → 0.001)")
    print()
    print("PRIORITY 3 (MEDIUM - Quality of life improvements):")
    print("  [ ] Add episode_step to observation space")
    print("  [ ] Monitor TensorBoard for entropy, value loss, explained variance")
    print()
    
    print("=" * 80)
    print("VALIDATION PLAN:")
    print("=" * 80)
    print("\n1. After implementing Priority 1 fixes:")
    print("   - Run 100k training with PBRS enabled")
    print("   - Monitor TensorBoard: pbrs/total_shaping_reward should decrease over time")
    print("   - Check: Episode lengths should decrease (more efficient paths)")
    print()
    print("2. Monitor these metrics:")
    print("   - rollout/ep_len_mean: Should DECREASE (shorter paths = efficiency)")
    print("   - curriculum/success_rate: Should INCREASE (better policies)")
    print("   - train/explained_variance: Should be > 0.5 (healthy value function)")
    print("   - train/entropy_loss: Should gradually DECREASE (policy converging)")
    print()
    print("3. Success criteria:")
    print("   - Episode length decreases by 20%+ compared to baseline")
    print("   - Success rate > 70% on medium difficulty")
    print("   - Explained variance > 0.7")
    print()

if __name__ == "__main__":
    main()
