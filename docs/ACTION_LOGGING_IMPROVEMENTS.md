# Action Logging Improvements for TensorBoard

## Overview

This document describes the enhanced action logging in the `EnhancedTensorBoardCallback` that provides clearer, more informative visualizations of agent behavior in TensorBoard.

## N++ Action Space

The N++ environment uses a discrete action space with 6 actions:

| Action Index | Name | Description | Keyboard Equivalent |
|--------------|------|-------------|---------------------|
| 0 | NOOP | No action | No keys pressed |
| 1 | Left | Move left | 'A' key |
| 2 | Right | Move right | 'D' key |
| 3 | Jump | Jump in place | Space key |
| 4 | Jump+Left | Jump and move left | Space + 'A' |
| 5 | Jump+Right | Jump and move right | Space + 'D' |

## Previous Implementation

The old implementation logged:
- `actions/frequency_action_0` through `actions/frequency_action_5` - Not intuitive
- `actions/entropy` - Overall action diversity
- `actions/distribution` - Histogram of action counts

**Problems:**
1. Action names were numeric indices - hard to interpret
2. No breakdown of movement patterns (horizontal vs vertical)
3. No analysis of directional bias
4. No insight into action sequences/transitions
5. Difficult to understand what the agent is actually doing

## New Implementation

### 1. Descriptive Action Names

**TensorBoard Path**: `actions/frequency/`

Now logs with clear names:
- `actions/frequency/NOOP`
- `actions/frequency/Left`
- `actions/frequency/Right`
- `actions/frequency/Jump`
- `actions/frequency/Jump+Left`
- `actions/frequency/Jump+Right`

**Benefit**: Immediately understand which gameplay movements are being used.

### 2. Movement Analysis

**TensorBoard Path**: `actions/movement/`

Tracks how the agent moves through space:

- **`stationary_pct`**: Percentage of time doing NOOP (standing still)
- **`active_pct`**: Percentage of time moving (1.0 - stationary_pct)
- **`left_bias`**: Of all horizontal movement, what % is leftward
- **`right_bias`**: Of all horizontal movement, what % is rightward

**Use Cases:**
- Detect if agent is too passive (high stationary_pct)
- Identify directional bias (e.g., agent prefers moving right)
- Track if agent learns to be more active over time
- Debug exploration issues

**Example Insights:**
```
stationary_pct: 0.25  → Agent is idle 25% of the time
active_pct: 0.75      → Agent is moving 75% of the time
left_bias: 0.48       → 48% of movement is leftward
right_bias: 0.52      → 52% of movement is rightward (slight right bias)
```

### 3. Jump Analysis

**TensorBoard Path**: `actions/jump/`

Analyzes jumping behavior:

- **`frequency`**: Overall jump frequency (% of all actions)
- **`directional_pct`**: How often jumps include horizontal movement
- **`vertical_only_pct`**: How often agent jumps straight up

**Use Cases:**
- Track if agent learns when to jump
- Identify if agent uses directional jumps effectively
- Detect inefficient behavior (too many vertical-only jumps)
- Monitor progression from random jumping to strategic jumping

**Example Insights:**
```
frequency: 0.35           → Agent jumps 35% of the time
directional_pct: 0.82     → 82% of jumps include horizontal movement (good!)
vertical_only_pct: 0.18   → 18% are vertical-only jumps
```

### 4. Action Transitions

**TensorBoard Path**: `actions/transitions/`

Tracks action sequences to understand behavior patterns:

- **Format**: `{PreviousAction}_to_{NextAction}`
- **Example**: `Left_to_Jump+Left` shows probability of jumping left after moving left
- **Filtering**: Only logs transitions with >1% probability to reduce noise

**Use Cases:**
- Identify action chains (e.g., "move left, then jump left")
- Detect inefficient patterns (e.g., alternating left/right rapidly)
- Understand agent's planning horizon
- Spot emergent behaviors

**Example Transitions:**
```
Left_to_Left: 0.65             → Agent tends to continue moving left
Left_to_Jump+Left: 0.25        → Often jumps while moving left (good!)
Left_to_Right: 0.03            → Rarely reverses direction (efficient)
Right_to_Jump+Right: 0.28      → Jumps while moving right (consistent)
NOOP_to_Jump: 0.15             → Sometimes jumps from standstill
Jump+Left_to_Left: 0.55        → Continues moving left after jumping
```

### 5. Action Entropy

**TensorBoard Path**: `actions/entropy`

Measures policy diversity (unchanged from previous implementation):
- **High entropy** (~1.79 for 6 actions): Agent explores many actions
- **Low entropy** (~0.5): Agent converges to few preferred actions

**Use Cases:**
- Monitor exploration vs exploitation
- Track policy convergence
- Identify if agent gets stuck in repetitive behavior

## Visualization Guide

### Essential Plots to Monitor

#### 1. Movement Behavior Dashboard
Create a TensorBoard custom scalar chart with:
```yaml
Movement Overview:
  - actions/movement/active_pct
  - actions/movement/stationary_pct
  - actions/jump/frequency
```

**What to look for:**
- Early training: High stationary_pct (agent hasn't learned to move)
- Mid training: Decreasing stationary_pct (agent learns movement is useful)
- Late training: Balanced active_pct based on what's optimal

#### 2. Action Usage Dashboard
```yaml
Action Frequencies:
  - actions/frequency/NOOP
  - actions/frequency/Left
  - actions/frequency/Right
  - actions/frequency/Jump
  - actions/frequency/Jump+Left
  - actions/frequency/Jump+Right
```

**What to look for:**
- Uniform distribution → Still exploring
- Concentrated on jump+directions → Agent learned platforming basics
- High NOOP → Potential issue or level-specific strategy

#### 3. Directional Bias Dashboard
```yaml
Directional Preferences:
  - actions/movement/left_bias
  - actions/movement/right_bias
```

**What to look for:**
- 0.5/0.5 split → No directional bias (good for varied levels)
- Strong bias → Possible issue or level-specific pattern

#### 4. Jump Intelligence Dashboard
```yaml
Jump Patterns:
  - actions/jump/frequency
  - actions/jump/directional_pct
  - actions/jump/vertical_only_pct
```

**What to look for:**
- Increasing directional_pct → Agent learns directional jumps are effective
- High vertical_only_pct → Inefficient, agent hasn't learned optimal jumping

#### 5. Behavior Patterns (Transitions)
Filter TensorBoard for `actions/transitions/` and focus on:
- Continuation patterns (Left_to_Left, Right_to_Right)
- Jump transitions (Left_to_Jump+Left, Right_to_Jump+Right)
- Direction changes (Left_to_Right, Right_to_Left)

**What to look for:**
- Coherent sequences → Agent planning ahead
- Random transitions → Still exploring
- Repetitive loops → Potential stuck behavior

## Interpreting Training Progress

### Early Training (0-10% of training)
**Expected Patterns:**
```
stationary_pct: 0.30-0.50      (agent hesitant to move)
action_entropy: 1.6-1.79       (high exploration)
directional_pct: 0.40-0.60     (random jumping)
transitions: Nearly uniform    (no clear patterns)
```

**Red Flags:**
- stationary_pct > 0.80 → Agent not learning to move
- action_entropy < 1.0 → Not exploring enough

### Mid Training (10-50% of training)
**Expected Patterns:**
```
stationary_pct: 0.15-0.30      (more active)
action_entropy: 1.2-1.5        (some specialization)
directional_pct: 0.70-0.85     (learning directional jumps)
transitions: Clear patterns    (e.g., Left→Jump+Left)
```

**Red Flags:**
- action_entropy < 0.8 → Collapsing to single action
- directional_pct < 0.50 → Not learning jump+movement

### Late Training (50-100% of training)
**Expected Patterns:**
```
stationary_pct: 0.05-0.20      (active, but strategic)
action_entropy: 0.8-1.3        (specialized but adaptive)
directional_pct: 0.80-0.95     (efficient jumping)
transitions: Specialized       (level-specific patterns)
```

**Red Flags:**
- stationary_pct > 0.40 → Agent giving up or stuck
- action_entropy < 0.5 → Over-specialized, not generalizing

## Common Patterns and Diagnoses

### Pattern: High NOOP Usage
**Symptoms:**
- `actions/frequency/NOOP` > 0.4
- `actions/movement/stationary_pct` > 0.4

**Possible Causes:**
1. Agent hasn't learned that movement leads to rewards
2. Observation space too complex
3. Reward shaping issue
4. Agent stuck in local minimum

**Diagnosis Steps:**
1. Check episode success rate (low success → needs more training)
2. Check value estimates (all near zero → reward signal too sparse)
3. Review reward function configuration

### Pattern: Left-Right Oscillation
**Symptoms:**
- `actions/transitions/Left_to_Right` > 0.3
- `actions/transitions/Right_to_Left` > 0.3
- Low episode success rate

**Possible Causes:**
1. Agent indecisive about direction
2. Insufficient temporal information
3. Need for frame stacking or recurrent policy

**Diagnosis Steps:**
1. Check if using frame stacking (should see velocity)
2. Increase training time
3. Consider adding position history to observations

### Pattern: Jump Spam
**Symptoms:**
- `actions/jump/frequency` > 0.6
- `actions/jump/vertical_only_pct` > 0.5
- Poor episode success

**Possible Causes:**
1. Agent discovered jumping gives positive signal
2. Reward shaping over-incentivizes jumping
3. Agent hasn't learned horizontal control

**Diagnosis Steps:**
1. Review PBRS weights (check if jump reward too high)
2. Check horizontal movement usage
3. Increase training time for horizontal control

### Pattern: Single Action Dominance
**Symptoms:**
- One action > 0.6 frequency
- `actions/entropy` < 0.8
- Plateau in success rate

**Possible Causes:**
1. Over-specialization to training levels
2. Insufficient exploration
3. Action space mismatch with curriculum

**Diagnosis Steps:**
1. Check curriculum difficulty progression
2. Increase entropy coefficient in PPO
3. Add more diverse training levels

## Implementation Details

### Memory Usage
- **Action counts**: ~48 bytes (6 actions × 8 bytes)
- **Action transitions**: ~288 bytes (6×6 matrix × 8 bytes)
- **Transition tracking**: ~number_of_envs × 8 bytes

**Total**: <500 bytes per callback instance (negligible)

### Performance Impact
- **Action counting**: O(num_envs) per step
- **Transition tracking**: O(num_envs) per step
- **Logging**: Only at log_freq intervals
- **Histogram logging**: Only at histogram_freq intervals

**Total overhead**: <0.2% CPU for typical configurations

### Configuration

Default configuration (no changes needed):
```python
callback = EnhancedTensorBoardCallback(
    log_freq=100,          # Log scalars every 100 steps
    histogram_freq=1000,   # Log histograms every 1000 steps
    verbose=1
)
```

Action logging happens automatically - no additional parameters needed.

## TensorBoard Tips

### 1. Smoothing
Action metrics can be noisy. Use TensorBoard smoothing:
- **Frequency metrics**: 0.8-0.9 smoothing
- **Transition metrics**: 0.9-0.95 smoothing (very noisy)
- **Entropy**: 0.7 smoothing

### 2. Multi-Run Comparison
When comparing architectures or hyperparameters:
- Filter by `actions/movement/` to compare activity levels
- Compare `actions/jump/directional_pct` to see learning efficiency
- Look at entropy curves to understand exploration differences

### 3. Custom Scalars
Create a custom scalar layout file:

```json
{
  "Action Analysis": {
    "Movement": ["actions/movement/.*"],
    "Jumps": ["actions/jump/.*"],
    "Frequencies": ["actions/frequency/.*"]
  },
  "Behavior Patterns": {
    "Top Transitions": [
      "actions/transitions/Left_to_Jump+Left",
      "actions/transitions/Right_to_Jump+Right",
      "actions/transitions/Left_to_Left",
      "actions/transitions/Right_to_Right"
    ]
  }
}
```

Load with: `tensorboard --logdir=experiments --custom_scalars=action_layout.json`

## Troubleshooting

### Transitions Not Appearing
**Symptom**: No `actions/transitions/` metrics in TensorBoard

**Causes**:
1. Not enough training steps (need > histogram_freq steps)
2. All transitions < 1% probability (very early training)

**Solution**: Wait for more training steps or lower the 0.01 threshold in code.

### Action Names Show as "Action0", "Action1", etc.
**Symptom**: Generic action names instead of "NOOP", "Left", etc.

**Causes**:
1. Action space size doesn't match N++ (modified environment?)
2. Using wrong environment type

**Solution**: Verify action space is `Discrete(6)` and environment is nclone-based.

### Metrics Not Updating
**Symptom**: Action metrics stuck at initial values

**Causes**:
1. Callback not properly initialized
2. TensorBoard writer not found

**Solution**: Check logs for initialization warnings. Ensure model has TensorBoard logger configured.

## Future Enhancements

Possible future additions:
1. **Action heatmaps by level section** - Which actions used in which areas
2. **Temporal action patterns** - Action sequences over longer time windows
3. **Per-episode action summaries** - Track action usage within successful vs failed episodes
4. **Action efficiency metrics** - Actions per unit progress
5. **Comparative baselines** - Show optimal human action distributions

## Summary

The enhanced action logging provides:

✅ **Intuitive Names** - "Jump+Left" instead of "Action 4"
✅ **Movement Insights** - Understand horizontal vs vertical behavior
✅ **Directional Analysis** - Detect biases and patterns
✅ **Jump Intelligence** - Track jump learning progression
✅ **Behavior Sequences** - Understand action chains and planning
✅ **Minimal Overhead** - <0.2% CPU, <500 bytes memory
✅ **Easy Analysis** - Clear TensorBoard organization
✅ **Debugging Support** - Quickly diagnose training issues

These improvements make it much easier to understand what the agent is learning and identify problems during training.
