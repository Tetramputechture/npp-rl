"""
Hyperparameters for PPO agent training.
This module contains the configuration parameters for training a PPO agent,
with detailed documentation for each parameter explaining its purpose and impact.
"""

HYPERPARAMETERS = {
    # Number of steps to run for each environment per update
    # This is the batch size = n_steps * n_env where n_env is number of environment copies running in parallel
    # Larger values -> more stable training but slower convergence
    "n_steps": 512,

    # Minibatch size for each optimization step
    # This is the number of samples processed in each optimization iteration
    # Smaller batch sizes -> more noisy updates but faster training
    # Should be <= n_steps
    "batch_size": 128,

    # Number of epochs when optimizing the surrogate loss
    # More epochs -> more optimization steps on the same data
    # Higher values can lead to overfitting but better sample efficiency
    "n_epochs": 5,

    # Discount factor for future rewards
    # Range: 0.0 to 1.0
    # Higher values -> agent cares more about long-term rewards
    # Lower values -> agent focuses more on immediate rewards
    "gamma": 0.996797,

    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    # Range: 0.0 to 1.0
    # Higher values -> less bias, more variance
    # Lower values -> more bias, less variance
    "gae_lambda": 0.998801,

    # Clipping parameter for PPO loss
    # Limits the amount the policy can change in one update
    # Smaller values -> more conservative updates
    # Typical range: 0.1 to 0.3
    "clip_range": 0.3892025290555702,

    # Clipping parameter for value function
    # Similar to clip_range but for value function
    # None means no clipping
    # IMPORTANT: this clipping depends on the reward scaling
    "clip_range_vf": 0.1,

    # Entropy coefficient for the loss calculation
    # Encourages exploration by penalizing deterministic policies
    # Higher values -> more exploration
    # Lower values -> more exploitation
    "ent_coef": 0.002720504247658009,

    # Value function coefficient for the loss calculation
    # Controls importance of value function loss vs policy loss
    # Higher values -> agent focuses more on getting value estimates right
    "vf_coef": 0.46935642684352785,

    # Maximum value for gradient clipping
    # Prevents too large policy updates
    # Helps stability by limiting the magnitude of parameter changes
    "max_grad_norm": 2.5658831600273806,

    # Whether to normalize the advantage
    # Normalizing advantages can help stabilize training
    # Especially useful when reward scales vary during training
    "normalize_advantage": True,

    # Verbosity level
    # 0: no output
    # 1: info
    # 2: debug
    "verbose": 1
}

NET_ARCH_SIZE = [128, 128]
