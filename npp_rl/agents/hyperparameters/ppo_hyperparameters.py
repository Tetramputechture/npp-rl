"""
Hyperparameters for PPO agent training.
This module contains the configuration parameters for training a PPO agent,
with detailed documentation for each parameter explaining its purpose and impact.
"""

HYPERPARAMETERS = {
    # Number of steps to run for each environment per update
    # Increased from 512 to 1024 based on research showing better performance
    # (e.g., "Scaling Laws for Neural Language Models", OpenAI, 2020; general trend in RL)
    # This is the batch size = n_steps * n_env where n_env is number of environment copies running in parallel
    # Larger values -> more stable training but slower convergence
    "n_steps": 1024,
    # Minibatch size for each optimization step
    # Increased from 128 to 256 for better gradient estimates with larger networks
    # (Matches trends in large-scale RL training, e.g., IMPALA, DeepMind, 2018)
    # This is the number of samples processed in each optimization iteration
    # Smaller batch sizes -> more noisy updates but faster training
    # Should be <= n_steps
    "batch_size": 256,
    # Number of epochs when optimizing the surrogate loss
    # More epochs -> more optimization steps on the same data
    # Higher values can lead to overfitting but better sample efficiency
    "n_epochs": 5,
    # Discount factor for future rewards
    # UPDATED: Reduced from 0.999 to 0.995 for better credit assignment in sparse reward setting
    # Analysis showed high gamma (0.999) was amplifying noise in sparse rewards
    # (See training analysis Oct 28, 2025)
    # Range: 0.0 to 1.0
    # Higher values -> agent cares more about long-term rewards
    # Lower values -> agent focuses more on immediate rewards
    "gamma": 0.995,
    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    # UPDATED: Reduced from 0.9988 to 0.97 for lower variance advantage estimates
    # High GAE lambda was causing unstable learning with high gamma
    # (See training analysis Oct 28, 2025)
    # Range: 0.0 to 1.0
    # Higher values -> less bias, more variance
    # Lower values -> more bias, less variance
    "gae_lambda": 0.97,
    # Clipping parameter for PPO loss
    # UPDATED: Reduced from 0.389 to 0.2 for more conservative policy updates
    # Standard PPO uses 0.2, higher values were causing instability
    # (See training analysis Oct 28, 2025)
    # Limits the amount the policy can change in one update
    # Smaller values -> more conservative updates
    # Typical range: 0.1 to 0.3
    "clip_range": 0.2,
    # Clipping parameter for value function
    # Similar to clip_range but for value function
    # None means no clipping
    # IMPORTANT: this clipping depends on the reward scaling
    "clip_range_vf": 0.1,
    # Entropy coefficient for the loss calculation
    # CRITICAL UPDATE: Increased from 0.00272 to 0.02 (7x increase)
    # Analysis showed policy was converging prematurely due to low entropy
    # Agent was avoiding jump actions and getting stuck in local optima
    # Higher entropy maintains exploration for longer duration
    # (See training analysis Oct 28, 2025 - this was identified as critical issue #4)
    # Encourages exploration by penalizing deterministic policies
    # Higher values -> more exploration
    # Lower values -> more exploitation
    "ent_coef": 0.02,
    # Value function coefficient for the loss calculation
    # UPDATED: Increased from 0.469 to 0.5 for more standard value
    # Controls importance of value function loss vs policy loss
    # Higher values -> agent focuses more on getting value estimates right
    "vf_coef": 0.5,
    # Maximum value for gradient clipping
    # UPDATED: Reduced from 2.566 to 2.0 for cleaner gradient clipping
    # Prevents too large policy updates
    # Helps stability by limiting the magnitude of parameter changes
    "max_grad_norm": 2.0,
    # Whether to normalize the advantage
    # Normalizing advantages can help stabilize training
    # Especially useful when reward scales vary during training
    "normalize_advantage": True,
    # Verbosity level
    # 0: no output
    # 1: info
    # 2: debug
    "verbose": 1,
}

# NOTE: These hyperparameters were updated based on comprehensive training analysis
# performed on Oct 28, 2025. Key changes address:
# 1. Premature convergence (increased ent_coef)
# 2. High variance advantage estimates (reduced gae_lambda)
# 3. Poor credit assignment (reduced gamma)
# 4. Aggressive policy updates (reduced clip_range)
#
# See docs/TRAINING_ANALYSIS_2025-10-28.md for full details.

# Scaled up network architecture based on research findings
# (e.g., ProcGen benchmark results showing benefits of larger networks)
# Increased from [128, 128] to [256, 256, 128] for better feature extraction
NET_ARCH_SIZE = [256, 256, 128]
