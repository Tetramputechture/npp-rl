# NPP-RL Training Analysis Report
================================================================================

## 1. Training Configuration
--------------------------------------------------------------------------------
**Experiment Name**: mlp-1029-f3-corridors-2
**Total Timesteps**: 1,000,000
**Number of Environments**: 28
**Batch Size**: 256
**Learning Rate**: 0.0003
**N-Steps**: 1024

**Feature Configuration**:
  - Pretraining: True
  - BC Epochs: 20
  - Curriculum Learning: True
  - Curriculum Threshold: 0.5
  - PBRS (Potential-Based Reward Shaping): True
  - PBRS Gamma: 0.995
  - Mine Avoidance Reward: True
  - Visual Frame Stacking: True
  - Stack Size: 3

## 2. Final Evaluation Results
--------------------------------------------------------------------------------
**Architecture**: mlp_baseline
**Training Status**: completed
**Success Rate**: 0.0%
**Average Steps**: 10000.0
**Total Episodes**: 14

## 3. Curriculum Learning Analysis
--------------------------------------------------------------------------------

### 3.1 Overall Success Rate by Stage

**simple**: 26.6% (25/94 episodes)
**simpler**: 45.0% (86/191 episodes)
**simplest**: 78.3% (101/129 episodes)

### 3.2 Top 10 Level Generators by Success Rate

- **horizontal_corridor:minimal**: 92.7% (38/41 episodes)
- **vertical_corridor:minimal**: 73.9% (34/46 episodes)
- **corridors:simplest**: 69.0% (29/42 episodes)
- **single_chamber:obstacle**: 68.8% (11/16 episodes)
- **horizontal_corridor:simple**: 64.7% (33/51 episodes)
- **vertical_corridor:simpler**: 54.3% (25/46 episodes)
- **vertical_corridor:simpler_with_mines**: 48.8% (21/43 episodes)
- **hills:simple**: 42.9% (3/7 episodes)
- **single_chamber:gap**: 31.2% (5/16 episodes)
- **vertical_corridor:simple**: 25.0% (3/12 episodes)

### 3.3 Bottom 10 Level Generators by Success Rate

- **maze:tiny**: 8.3% (1/12 episodes)
- **jump_required:simple**: 9.1% (1/11 episodes)
- **vertical_corridor:platforms**: 9.1% (1/11 episodes)
- **corridors:simple**: 11.7% (7/60 episodes)
- **vertical_corridor:simple**: 25.0% (3/12 episodes)
- **single_chamber:gap**: 31.2% (5/16 episodes)
- **hills:simple**: 42.9% (3/7 episodes)
- **vertical_corridor:simpler_with_mines**: 48.8% (21/43 episodes)
- **vertical_corridor:simpler**: 54.3% (25/46 episodes)
- **horizontal_corridor:simple**: 64.7% (33/51 episodes)

## 4. Training Metrics Summary
--------------------------------------------------------------------------------

### 4.1 Reward Metrics

**pbrs_rewards/exploration_mean**:
  - Mean: 0.0001
  - Std: 0.0001
  - Min: 0.0000
  - Max: 0.0050

**pbrs_rewards/navigation_mean**:
  - Mean: 0.0000
  - Std: 0.0000
  - Min: -0.0000
  - Max: 0.0004

**pbrs_rewards/pbrs_mean**:
  - Mean: -0.0043
  - Std: 0.0001
  - Min: -0.0048
  - Max: -0.0039

**pbrs_rewards/total_mean**:
  - Mean: -0.0019
  - Std: 0.0060
  - Min: -0.0047
  - Max: 0.0701

**rewards/hierarchical_mean**:
  - Mean: -40.2556
  - Std: 19.0921
  - Min: -83.7548
  - Max: 6.0473

**rewards/hierarchical_std**:
  - Mean: 148.6402
  - Std: 51.2954
  - Min: 2.7184
  - Max: 205.3505

### 4.2 Loss Metrics

**loss/entropy**:
  - Mean: -1.511140
  - Std: 0.135401
  - Final: -1.454031

**loss/total**:
  - Mean: -0.035325
  - Std: 0.070558
  - Final: -0.055791

**loss/value**:
  - Mean: 0.031794
  - Std: 0.083514
  - Final: 0.019218

**train/entropy_loss**:
  - Mean: -1.511140
  - Std: 0.137421
  - Final: -1.454031

**train/loss**:
  - Mean: -0.035325
  - Std: 0.071611
  - Final: -0.055791

**train/policy_gradient_loss**:
  - Mean: -0.027689
  - Std: 0.007266
  - Final: -0.033994

**train/value_loss**:
  - Mean: 0.031794
  - Std: 0.084761
  - Final: 0.019218
