"""
Hierarchical PPO Hyperparameters for Two-Level Policy Training

This module defines optimized hyperparameters for hierarchical reinforcement learning
with separate high-level (subtask selection) and low-level (action execution) policies.

Based on Task 2.4 requirements:
- High-level policy: Lower learning rate for stable strategic decisions
- Low-level policy: Higher learning rate for responsive action learning
- Coordinated update frequencies and experience buffer management
- ICM parameters for exploration enhancement
- Adaptive training procedures for stability

Research foundations:
- PPO: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Hierarchical RL: Nachum et al. (2018) "Data-Efficient Hierarchical RL"
- ICM: Pathak et al. (2017) "Curiosity-driven Exploration"
"""

from stable_baselines3.common.utils import LinearSchedule


# High-Level Policy Hyperparameters
# Strategic subtask selection based on reachability features
HIGH_LEVEL_HYPERPARAMETERS = {
    # Learning rate: Lower for stable strategic decisions
    # Linear decay from 1e-4 to 1e-5 over training
    "learning_rate": LinearSchedule(1e-4, 1e-5, 1.0),
    # Steps per update: Longer horizon for strategic decisions
    # High-level makes decisions every 50-100 steps, so collect more experience
    "n_steps": 2048,
    # Batch size: Smaller for strategic decisions (fewer decisions to learn from)
    "batch_size": 64,
    # Epochs: Fewer epochs to prevent overfitting on sparse high-level decisions
    "n_epochs": 4,
    # Discount factor: High for long-term strategic planning
    "gamma": 0.999,
    # GAE lambda: Slightly lower for more stable advantage estimates
    "gae_lambda": 0.95,
    # Clip range: Conservative to prevent large strategy shifts
    "clip_range": 0.1,
    # Clip range for value function
    "clip_range_vf": 0.1,
    # Entropy coefficient: Moderate exploration for subtask selection
    "ent_coef": 0.01,
    # Value function coefficient: Balanced importance
    "vf_coef": 0.5,
    # Gradient clipping: Conservative for stability
    "max_grad_norm": 0.5,
    # Normalize advantages
    "normalize_advantage": True,
    # Verbosity
    "verbose": 1,
}

# Low-Level Policy Hyperparameters
# Tactical action execution with ICM-enhanced exploration
LOW_LEVEL_HYPERPARAMETERS = {
    # Learning rate: Higher for responsive action learning
    # Linear decay from 3e-4 to 1e-5 over training
    "learning_rate": LinearSchedule(3e-4, 1e-5, 1.0),
    # Steps per update: Standard PPO horizon
    "n_steps": 1024,
    # Batch size: Larger for better gradient estimates with complex policy
    "batch_size": 256,
    # Epochs: More epochs for thorough action learning
    "n_epochs": 10,
    # Discount factor: High for long-term task completion
    "gamma": 0.999,
    # GAE lambda: High as in standard PPO
    "gae_lambda": 0.998801,
    # Clip range: Standard PPO clipping
    "clip_range": 0.2,
    # Clip range for value function
    "clip_range_vf": 0.1,
    # Entropy coefficient: Higher for more exploration at action level
    "ent_coef": 0.02,
    # Value function coefficient: Balanced importance
    "vf_coef": 0.5,
    # Gradient clipping: Standard PPO value
    "max_grad_norm": 0.5,
    # Normalize advantages
    "normalize_advantage": True,
    # Verbosity
    "verbose": 1,
}

# ICM (Intrinsic Curiosity Module) Hyperparameters
# For exploration enhancement at low-level policy
ICM_HYPERPARAMETERS = {
    # Intrinsic reward weight (alpha)
    # Balance between extrinsic and intrinsic rewards
    # Lower values prioritize task completion, higher values prioritize exploration
    "alpha": 0.1,
    # ICM learning rate (eta)
    # Separate from policy learning rate for stable curiosity learning
    "eta": 1e-3,
    # Inverse model loss weight (lambda_inv)
    # Weight for inverse model (predicts action from state transition)
    "lambda_inv": 0.1,
    # Forward model loss weight (lambda_fwd)
    # Weight for forward model (predicts next state from action)
    # Higher weight emphasizes prediction error = curiosity
    "lambda_fwd": 0.9,
    # Feature dimension for ICM
    # Dimension of learned state representations
    "feature_dim": 128,
    # ICM update frequency
    # How often to update ICM (every N steps)
    "update_frequency": 4,
    # Mine-aware curiosity modulation
    # Reduce curiosity near dangerous mines
    "mine_danger_threshold": 2.0,  # Distance in tiles
    "mine_curiosity_scale": 0.1,  # Curiosity scale near mines
    "safe_curiosity_scale": 1.0,  # Curiosity scale in safe areas
}

# Hierarchical Training Coordination
# Parameters for coordinating high-level and low-level training
HIERARCHICAL_COORDINATION = {
    # High-level update frequency
    # How many low-level steps between high-level decisions
    "high_level_update_frequency": 50,
    # Maximum steps per subtask
    # Automatic timeout if subtask takes too long
    "max_steps_per_subtask": 500,
    # Minimum steps between subtask switches
    # Cooldown to prevent rapid switching
    "min_steps_between_switches": 50,
    # Experience buffer coordination
    # Ratio of high-level to low-level experiences to collect
    "experience_ratio": 0.1,  # 10% high-level, 90% low-level
    # Warm-up phase parameters
    "warmup_steps": 100000,  # Steps to train low-level before introducing high-level
    "warmup_high_level_lr_scale": 0.1,  # Scale high-level LR during warmup
    # Curriculum progression thresholds
    "simple_level_threshold": 0.3,  # Success rate to advance from simple levels
    "medium_level_threshold": 0.5,  # Success rate to advance from medium levels
    "curriculum_evaluation_episodes": 100,  # Episodes to evaluate progression
}

# Adaptive Training Parameters
# For dynamic hyperparameter adjustment based on training metrics
ADAPTIVE_TRAINING = {
    # Stability detection thresholds
    "instability_gradient_norm_threshold": 10.0,  # Flag if gradient norm exceeds this
    "instability_value_loss_threshold": 5.0,  # Flag if value loss exceeds this
    "instability_window_size": 1000,  # Steps to check for instability
    # Stagnation detection thresholds
    "stagnation_improvement_threshold": 0.01,  # Minimum improvement rate
    "stagnation_window_size": 10000,  # Steps to check for stagnation
    # Learning rate adjustment parameters
    "lr_decrease_factor": 0.8,  # Multiply LR by this when unstable
    "lr_increase_factor": 1.1,  # Multiply LR by this when stagnating
    "lr_min": 1e-6,  # Minimum learning rate
    "lr_max": 3e-4,  # Maximum learning rate
    # Policy balance parameters
    "policy_loss_ratio_threshold": 2.0,  # Flag if one policy loss >> other
    "policy_balance_window_size": 500,  # Steps to check policy balance
}

# Network Architecture for Hierarchical Policies
# Sized appropriately for each policy level
HIERARCHICAL_NET_ARCH = {
    # High-level policy network (subtask selection)
    # Smaller network for strategic decisions
    "high_level": {
        "pi": [128, 128],  # Policy network
        "vf": [128, 128],  # Value network
    },
    # Low-level policy network (action execution)
    # Larger network for complex action selection
    "low_level": {
        "pi": [256, 256, 128],  # Policy network
        "vf": [256, 256, 128],  # Value network
    },
}

# Logging and Monitoring Configuration
LOGGING_CONFIG = {
    # TensorBoard logging frequency
    "tensorboard_log_freq": 100,
    # Console logging frequency
    "console_log_freq": 1000,
    # Checkpoint save frequency
    "checkpoint_freq": 50000,
    # Evaluation frequency
    "eval_freq": 10000,
    # Evaluation episodes
    "eval_episodes": 10,
    # Metrics to track
    "track_metrics": [
        # High-level metrics
        "high_level/policy_loss",
        "high_level/value_loss",
        "high_level/entropy",
        "high_level/gradient_norm",
        "high_level/learning_rate",
        "high_level/subtask_distribution",
        # Low-level metrics
        "low_level/policy_loss",
        "low_level/value_loss",
        "low_level/entropy",
        "low_level/gradient_norm",
        "low_level/learning_rate",
        "low_level/action_distribution",
        # ICM metrics
        "icm/intrinsic_reward_mean",
        "icm/intrinsic_reward_std",
        "icm/forward_loss",
        "icm/inverse_loss",
        "icm/curiosity_scale",
        # Hierarchical coordination metrics
        "hierarchical/subtask_transitions",
        "hierarchical/avg_subtask_duration",
        "hierarchical/subtask_success_rate",
        "hierarchical/coordination_efficiency",
        # Environment metrics
        "environment/episode_reward",
        "environment/episode_length",
        "environment/success_rate",
        "environment/switch_activations",
        "environment/exit_completions",
        # Stability metrics
        "stability/gradient_norm_ratio",
        "stability/value_loss_change",
        "stability/policy_loss_change",
        "stability/is_stable",
    ],
}

# GPU Optimization Settings
# For H100 or better GPUs
GPU_OPTIMIZATION = {
    # Enable TF32 for faster matrix multiplication on A100/H100
    "use_tf32": True,
    # Mixed precision training
    "use_amp": False,  # Keep False for stability, enable if memory constrained
    # Number of parallel environments
    # Scale based on CPU cores and memory
    "num_envs": 64,  # Default for H100 instance
    # Pin memory for faster GPU transfer
    "pin_memory": True,
    # Number of dataloader workers
    "num_workers": 4,
    # Gradient accumulation steps (if memory constrained)
    "gradient_accumulation_steps": 1,
}

# Complete hierarchical configuration
# Combines all parameters for easy access
HIERARCHICAL_CONFIG = {
    "high_level": HIGH_LEVEL_HYPERPARAMETERS,
    "low_level": LOW_LEVEL_HYPERPARAMETERS,
    "icm": ICM_HYPERPARAMETERS,
    "coordination": HIERARCHICAL_COORDINATION,
    "adaptive": ADAPTIVE_TRAINING,
    "net_arch": HIERARCHICAL_NET_ARCH,
    "logging": LOGGING_CONFIG,
    "gpu": GPU_OPTIMIZATION,
}
