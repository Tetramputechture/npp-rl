# Policy Architecture Guide

## Overview

This guide documents the deep ResNet policy architecture used for PPO training on N++ navigation tasks. The architecture is designed for sequential planning in levels with variable numbers of locked doors (1-16), toggle mines, exit switches, and exit doors.

For information on feature extractors and multimodal architectures, see [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) and [ARCHITECTURE_COMPARISON_GUIDE.md](ARCHITECTURE_COMPARISON_GUIDE.md).

## Deep ResNet Policy Architecture

### Core Components

The policy architecture consists of:

1. **Deep ResNet MLP Extractor** (`npp_rl/models/deep_resnet_mlp.py`)
   - 5-layer policy network: `[512, 512, 384, 256, 256]`
   - 3-layer value network: `[512, 384, 256]`
   - Residual connections every 2 layers
   - LayerNorm normalization
   - SiLU activation function
   - Optional dueling architecture for value decomposition

2. **Deep ResNet Actor-Critic Policy** (`npp_rl/agents/deep_resnet_actor_critic_policy.py`)
   - Inherits from `MaskedActorCriticPolicy` for action masking support
   - Separate feature extractors for policy and value streams
   - Compatible with stable-baselines3 PPO
   - Supports optional auxiliary tasks and distributional value functions

### Architecture Details

**Policy Network:**
```
features [batch, feature_dim] →
  ResBlock(512) → ResBlock(512) → ResBlock(384) → ResBlock(256) → ResBlock(256) →
  Linear(256, num_actions) → action_logits [batch, 6]
```

**Value Network (Standard):**
```
features [batch, feature_dim] →
  ResBlock(512) → ResBlock(384) → ResBlock(256) →
  Linear(256, 1) → value [batch, 1]
```

**Value Network (Dueling):**
```
features [batch, feature_dim] →
  ResBlock(512) → ResBlock(384) → ResBlock(256) →
    ├─→ Linear(256, 1) → V(s)
    └─→ Linear(256, num_actions) → A(s,a)
  Combine: V(s) + (A(s,a) - mean(A(s,*)))
```

## Design Rationale

### Why Residual Networks for Action Prediction

The policy head uses deep residual networks rather than attention mechanisms for action prediction. This design choice is based on several considerations:

**Action Space Characteristics**

N++ uses a discrete action space with 6 actions (combinations of horizontal movement {-1, 0, 1} and jump {0, 1}). The policy head maps a feature vector directly to action logits, which is a function approximation problem rather than a sequence-to-sequence or set-processing task.

Multi-head attention is most effective when:
- Processing variable-length sequences or sets
- Selecting from multiple candidates with dynamic relevance
- Handling permutation-invariant structures

For discrete action prediction, attention would require artificially creating sequence structure where none exists in the problem domain.

**Hierarchical Feature Abstraction**

N++ requires multi-step sequential planning (e.g., reach switch → avoid hazards → navigate to exit). Deep residual networks (He et al., 2016) enable hierarchical feature learning:
- Early layers: Low-level physics (velocity, surface contact)
- Middle layers: Tactical decisions (jump timing, wall interactions)
- Late layers: Strategic planning (route selection, risk assessment)

Residual connections maintain gradient flow through 5 layers, enabling credit assignment over long episodes (up to 20,000 frames).

**Domain Requirements**

N++ gameplay requires precise, deterministic physics reasoning:
- Frame-perfect jump timing (5-frame input buffers)
- Velocity-dependent collision detection
- Complex state transitions (9 movement states)
- Lethal impact thresholds

Residual networks provide:
- Deterministic transformations for consistent behavior
- Smooth gradients (SiLU activations) for fine-tuning
- Hierarchical structure matching physics abstraction levels

**Computational Efficiency**

Residual networks are more efficient than attention for this use case:
- Policy head: ~625K parameters (5 linear layers)
- Attention alternative: ~260K parameters per layer + O(n²) attention computation
- Memory: ResNet uses linear memory; attention requires caching Q/K/V matrices

For training with 128 parallel environments, the ResNet architecture uses approximately 40GB VRAM on H100 GPUs.

### Where Attention Is Used

Attention mechanisms are used appropriately in other parts of the architecture:

**Objective Attention** (`npp_rl/models/objective_attention.py`):
- Multi-head attention over variable-length objective sequences (1-16 locked doors)
- Permutation-invariant processing of door ordering
- Learns sequential goal prioritization

**Graph Attention** (GAT encoder):
- Attention over graph neighbors with variable node degrees
- Dynamic weighting of edge features
- Handles heterogeneous graph structures

These use cases have genuine sequence/set structure that benefits from attention mechanisms.

### Key Architectural Choices

**Separate Feature Extractors**

Policy and value networks use separate `ConfigurableMultimodalExtractor` instances to avoid gradient conflicts. This isolation may improve training stability and convergence speed.

**Residual Connections**

Residual connections (He et al., 2016) are applied every 2 layers to enable deeper networks without gradient degradation. When layer dimensions change, a learned linear projection is used for the residual path.

**LayerNorm**

Layer normalization (Ba et al., 2016) is applied after each linear layer, before activation. This stabilizes training with large batch sizes (512+) and is more appropriate than BatchNorm for reinforcement learning.

**Dueling Architecture**

The value network uses dueling architecture (Wang et al., 2016) to separate state value V(s) from action advantages A(s,a). This decomposition may improve value estimation by learning state value independent of action selection.

**SiLU Activation**

The SiLU (Swish) activation function (Hendrycks & Gimpel, 2016) is used instead of ReLU. SiLU is smooth and non-monotonic, which may improve gradient flow in deep networks.

## Usage

### Training with ArchitectureTrainer

```python
from npp_rl.training.architecture_trainer import ArchitectureTrainer
from npp_rl.training.architecture_configs import get_architecture_config

architecture_config = get_architecture_config("attention")

trainer = ArchitectureTrainer(
    architecture_config=architecture_config,
    train_dataset_path="path/to/train",
    test_dataset_path="path/to/test",
    output_dir="outputs/deep_resnet",
    use_deep_resnet_policy=True,
    device_id=0,
)

trainer.setup_model()
trainer.setup_environments(num_envs=128)
trainer.train(total_timesteps=10_000_000)
```

### Direct PPO Usage

```python
import torch.nn as nn
from stable_baselines3 import PPO
from npp_rl.agents.deep_resnet_actor_critic_policy import DeepResNetActorCriticPolicy
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
from npp_rl.training.architecture_configs import get_architecture_config

config = get_architecture_config("attention")

policy_kwargs = {
    "features_extractor_class": ConfigurableMultimodalExtractor,
    "features_extractor_kwargs": {"config": config},
    "share_features_extractor": False,
    "net_arch": {
        "pi": [512, 512, 384, 256, 256],
        "vf": [512, 384, 256],
    },
    "activation_fn": nn.SiLU,
    "use_residual": True,
    "use_layer_norm": True,
    "dueling": True,
    "dropout": 0.1,
}

model = PPO(
    policy=DeepResNetActorCriticPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=2.0,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=1.0,
    device="cuda:0",
)

model.learn(total_timesteps=10_000_000)
```

### Command Line Training

```bash
python -m npp_rl.agents.training \
    --architecture attention \
    --use-deep-resnet-policy \
    --num-envs 128 \
    --total-timesteps 10000000 \
    --learning-rate 1e-4 \
    --batch-size 512 \
    --n-steps 4096
```

## Hyperparameters

### Recommended Configuration (H100 80GB)

```python
hyperparameters = {
    "learning_rate": 1e-4,
    "n_steps": 4096,
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": 2.0,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
}

training_config = {
    "num_envs": 128,
    "total_timesteps": 10_000_000,
    "eval_freq": 50_000,
    "save_freq": 100_000,
}
```

### Smaller GPUs (16-24GB)

```python
hyperparameters = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 8,
    "num_envs": 64,
}
```

## Architecture Comparison

| Component | Baseline | Deep ResNet |
|-----------|----------|-------------|
| Policy Depth | 3 layers | 5 layers |
| Policy Width | [256,256,128] | [512,512,384,256,256] |
| Value Network | Shared features | Separate features |
| Residual Connections | None | Every 2 layers |
| Normalization | None | LayerNorm |
| Activation | ReLU | SiLU |
| Value Architecture | Standard | Dueling |
| Total Parameters | ~724K | ~7-15M |

## Monitoring

### Key Metrics

- Policy loss: Should decrease and stabilize
- Value loss: Should decrease steadily
- Policy entropy: Should decay gradually (not collapse)
- Gradient norms: Should be stable
- Success rate: Should increase over training
- Episode length: Should decrease as agent learns
- Death rate: Should decrease

### TensorBoard

```bash
tensorboard --logdir outputs/deep_resnet/tensorboard
```

## Troubleshooting

### NaN in Losses

- Lower learning rate (1e-4 → 5e-5)
- Increase gradient clipping (1.0 → 0.5)
- Check for NaN in observations
- Verify action masking is working

### Training Instability

- Increase batch size (512 → 1024)
- Lower learning rate
- Add more gradient clipping
- Verify LayerNorm is enabled

### Poor Generalization

- Increase dropout (0.1 → 0.2)
- Add weight decay (1e-4)
- Train on more diverse levels
- Use curriculum learning

### Memory Issues

- Reduce batch size
- Reduce num_envs
- Reduce n_steps
- Enable gradient accumulation

## Advanced Features

### Auxiliary Tasks

Optional auxiliary prediction tasks may improve representation learning:

```python
policy_kwargs["enable_auxiliary_tasks"] = True
policy_kwargs["auxiliary_weights"] = {
    "death": 0.1,
    "time": 0.1,
    "subgoal": 0.1,
}
```

Available tasks:
- Death prediction (binary classification)
- Time-to-goal estimation (regression)
- Next subgoal classification (multi-class)

### Time-Conditional Gating

Urgency-aware decision making based on time remaining:

```python
policy_kwargs["enable_time_gating"] = True
```

See `npp_rl/models/time_conditional_policy.py` for implementation details.

### Objective Attention

Multi-head attention over variable-length objective sequences:

```python
policy_kwargs["enable_objective_attention"] = True
policy_kwargs["max_locked_doors"] = 16
```

See `npp_rl/models/objective_attention.py` for implementation details.

### Distributional Value Function

Quantile regression for value estimation:

```python
policy_kwargs["use_distributional_value"] = True
policy_kwargs["num_quantiles"] = 51
```

See `npp_rl/models/distributional_value.py` for implementation details (Dabney et al., 2018).

## Implementation Files

- `npp_rl/models/deep_resnet_mlp.py` - ResNet MLP extractor
- `npp_rl/agents/deep_resnet_actor_critic_policy.py` - Main policy class
- `npp_rl/models/objective_attention.py` - Attention over objectives
- `npp_rl/models/distributional_value.py` - Quantile value function
- `npp_rl/models/auxiliary_tasks.py` - Auxiliary prediction heads
- `npp_rl/models/time_conditional_policy.py` - Time-aware gating
- `tests/test_deep_resnet_policy.py` - Tests

## References

1. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.

2. Dabney, W., Rowland, M., Bellemare, M. G., & Munos, R. (2018). Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1).

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

4. Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415.

5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

7. Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

