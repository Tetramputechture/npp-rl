"""Hyperparameter search space definitions for Optuna optimization."""

from typing import Dict, Any


def get_search_space(architecture_name: str) -> Dict[str, Any]:
    """
    Get hyperparameter search space for architecture.

    Returns dict with structure:
    {
        "parameter_name": {
            "type": "float" | "int" | "categorical",
            "low": float,  # for float/int
            "high": float,  # for float/int
            "log": bool,   # for float/int (log scale)
            "choices": list,  # for categorical
        }
    }

    Args:
        architecture_name: Name of architecture from ARCHITECTURE_REGISTRY

    Returns:
        Dictionary mapping parameter names to their search space definitions
    """
    # Base PPO search space (applies to all architectures)
    base_space = {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "n_steps": {"type": "categorical", "choices": [512, 1024, 2048, 4096]},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "gamma": {"type": "float", "low": 0.95, "high": 0.999, "log": False},
        "gae_lambda": {"type": "float", "low": 0.90, "high": 0.99, "log": False},
        "clip_range": {"type": "float", "low": 0.1, "high": 0.4, "log": False},
        "clip_range_vf": {"type": "float", "low": 0.1, "high": 2.0, "log": False},
        "ent_coef": {"type": "float", "low": 1e-4, "high": 0.1, "log": True},
        "vf_coef": {"type": "float", "low": 0.3, "high": 0.8, "log": False},
        "max_grad_norm": {"type": "float", "low": 0.3, "high": 5.0, "log": False},
        "n_epochs": {"type": "int", "low": 3, "high": 10},
        # Network architecture
        "net_arch_depth": {"type": "int", "low": 2, "high": 4},
        "net_arch_width": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "features_dim": {"type": "categorical", "choices": [256, 512, 768, 1024]},
        # Training settings
        "num_envs": {"type": "categorical", "choices": [32, 64, 128, 256]},
        "lr_schedule": {"type": "categorical", "choices": ["constant", "linear"]},
    }

    # Architecture-specific additions
    if (
        "gat" in architecture_name
        or "gcn" in architecture_name
        or "hgt" in architecture_name
    ):
        base_space.update(
            {
                "gnn_num_layers": {"type": "int", "low": 2, "high": 4},
                "gnn_hidden_dim": {"type": "categorical", "choices": [64, 128, 256]},
            }
        )

        if "gat" in architecture_name or "hgt" in architecture_name:
            base_space["gnn_num_heads"] = {"type": "int", "low": 2, "high": 8}

    # Add CNN-specific params if architecture uses vision
    if "vision_free" not in architecture_name and "mlp" not in architecture_name:
        base_space.update(
            {
                "cnn_base_channels": {"type": "categorical", "choices": [16, 32, 64]},
                "cnn_num_layers": {"type": "int", "low": 2, "high": 4},
            }
        )

    return base_space
