"""Deep ResNet MLP extractor for PPO policy and value networks.

This module implements deep residual MLPs with LayerNorm and modern activations
for improved gradient flow and learning stability in complex sequential tasks.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Type


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and configurable activation.
    
    Architecture:
        input → Linear → LayerNorm → Activation → output
                 ↓                                    ↑
                 └────── projection (if needed) ──────┘
    
    The residual connection uses identity mapping if dimensions match,
    or a learned linear projection if dimensions differ.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_fn: Type[nn.Module] = nn.SiLU,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize residual block.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            activation_fn: Activation function class (e.g., nn.SiLU, nn.ReLU)
            use_layer_norm: Whether to use LayerNorm
            dropout: Dropout rate (0 = no dropout)
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Main path
        layers = [nn.Linear(in_dim, out_dim)]
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        
        layers.append(activation_fn())
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.main_path = nn.Sequential(*layers)
        
        # Residual path (projection if dimensions differ)
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor [batch, in_dim]
        
        Returns:
            Output tensor [batch, out_dim]
        """
        return self.main_path(x) + self.residual_proj(x)


class DeepResNetMLPExtractor(nn.Module):
    """Deep ResNet MLP for policy and value streams with optional dueling.
    
    This extractor creates separate deep residual networks for the policy
    and value functions, with optional dueling architecture for the value head.
    
    Policy Network:
        features → [ResBlock1 → ResBlock2 → ... → ResBlockN] → policy_latent
    
    Value Network (Standard):
        features → [ResBlock1 → ResBlock2 → ... → ResBlockN] → value_latent
    
    Value Network (Dueling):
        features → [ResBlock1 → ResBlock2 → ... → ResBlockN] 
                        ├─→ state_value_stream → V(s)
                        └─→ advantage_stream → A(s,a)
        output: V(s) + (A(s,a) - mean(A(s,*)))
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        policy_layers: List[int] = [512, 512, 384, 256, 256],
        value_layers: List[int] = [512, 384, 256],
        activation_fn: Type[nn.Module] = nn.SiLU,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        dueling: bool = True,
        num_actions: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize deep ResNet MLP extractor.
        
        Args:
            feature_dim: Dimension of input features
            policy_layers: List of hidden dimensions for policy network
            value_layers: List of hidden dimensions for value network
            activation_fn: Activation function class
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use LayerNorm
            dueling: Whether to use dueling architecture for value
            num_actions: Number of actions (for dueling advantage stream)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.policy_layers = policy_layers
        self.value_layers = value_layers
        self.dueling = dueling
        self.num_actions = num_actions
        
        # Build policy network
        self.policy_net = self._build_network(
            feature_dim,
            policy_layers,
            activation_fn,
            use_residual,
            use_layer_norm,
            dropout,
        )
        
        # Build value network
        if dueling:
            # Shared value trunk
            self.value_trunk = self._build_network(
                feature_dim,
                value_layers,
                activation_fn,
                use_residual,
                use_layer_norm,
                dropout,
            )
            
            # State value stream: V(s)
            self.state_value_head = nn.Linear(value_layers[-1], 1)
            
            # Advantage stream: A(s,a)
            self.advantage_head = nn.Linear(value_layers[-1], num_actions)
        else:
            # Standard value network
            self.value_net = self._build_network(
                feature_dim,
                value_layers,
                activation_fn,
                use_residual,
                use_layer_norm,
                dropout,
            )
        
        # Output dimensions
        self.latent_dim_pi = policy_layers[-1]
        self.latent_dim_vf = value_layers[-1]
    
    def _build_network(
        self,
        input_dim: int,
        layer_dims: List[int],
        activation_fn: Type[nn.Module],
        use_residual: bool,
        use_layer_norm: bool,
        dropout: float,
    ) -> nn.Module:
        """Build a deep residual network.
        
        Args:
            input_dim: Input dimension
            layer_dims: List of layer dimensions
            activation_fn: Activation function class
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use LayerNorm
            dropout: Dropout rate
        
        Returns:
            Sequential network or ModuleList of residual blocks
        """
        if not layer_dims:
            return nn.Identity()
        
        if use_residual:
            # Build network with residual blocks
            # Add residual connections every 2 layers for optimal gradient flow
            blocks = []
            prev_dim = input_dim
            
            for i, dim in enumerate(layer_dims):
                # Decide whether this block should have a residual connection
                # We add residuals every 2 layers and at dimension changes
                if i > 0 and (i % 2 == 0 or prev_dim != dim):
                    blocks.append(
                        ResidualBlock(
                            prev_dim,
                            dim,
                            activation_fn,
                            use_layer_norm,
                            dropout if i < len(layer_dims) - 1 else 0,  # No dropout on last layer
                        )
                    )
                else:
                    # Regular layer without explicit residual
                    layer_list = [nn.Linear(prev_dim, dim)]
                    if use_layer_norm:
                        layer_list.append(nn.LayerNorm(dim))
                    layer_list.append(activation_fn())
                    if dropout > 0 and i < len(layer_dims) - 1:
                        layer_list.append(nn.Dropout(dropout))
                    blocks.append(nn.Sequential(*layer_list))
                
                prev_dim = dim
            
            return nn.Sequential(*blocks)
        else:
            # Standard MLP without residual connections
            layers = []
            prev_dim = input_dim
            
            for i, dim in enumerate(layer_dims):
                layers.append(nn.Linear(prev_dim, dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dim))
                layers.append(activation_fn())
                if dropout > 0 and i < len(layer_dims) - 1:
                    layers.append(nn.Dropout(dropout))
                prev_dim = dim
            
            return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy and value networks.
        
        Args:
            features: Input features [batch, feature_dim]
        
        Returns:
            Tuple of (policy_latent, value_latent)
            - policy_latent: [batch, latent_dim_pi]
            - value_latent: [batch, latent_dim_vf] (or [batch, 1] if dueling and final output)
        """
        # Policy forward
        policy_latent = self.policy_net(features)
        
        # Value forward
        if self.dueling:
            # Dueling architecture
            value_features = self.value_trunk(features)
            
            # State value: V(s)
            state_value = self.state_value_head(value_features)  # [batch, 1]
            
            # Advantages: A(s,a)
            advantages = self.advantage_head(value_features)  # [batch, num_actions]
            
            # Combine: V(s) + (A(s,a) - mean(A(s,*)))
            # This is the dueling architecture from Wang et al. (2016)
            advantage_mean = advantages.mean(dim=1, keepdim=True)
            value_latent = state_value + (advantages - advantage_mean)
            
            # For compatibility with SB3, return the trunk features for value
            # The actual value computation happens in the policy class
            return policy_latent, value_features
        else:
            # Standard value network
            value_latent = self.value_net(features)
            return policy_latent, value_latent
    
    def forward_policy(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy network only.
        
        Args:
            features: Input features [batch, feature_dim]
        
        Returns:
            Policy latent representation [batch, latent_dim_pi]
        """
        return self.policy_net(features)
    
    def forward_value(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network only.
        
        Args:
            features: Input features [batch, feature_dim]
        
        Returns:
            Value latent representation [batch, latent_dim_vf]
        """
        if self.dueling:
            return self.value_trunk(features)
        else:
            return self.value_net(features)
    
    def get_dueling_values(self, value_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get state value and advantages from value features (dueling only).
        
        Args:
            value_features: Value trunk output [batch, latent_dim_vf]
        
        Returns:
            Tuple of (state_value, advantages)
            - state_value: [batch, 1]
            - advantages: [batch, num_actions]
        """
        if not self.dueling:
            raise RuntimeError("get_dueling_values() only works with dueling=True")
        
        state_value = self.state_value_head(value_features)
        advantages = self.advantage_head(value_features)
        return state_value, advantages

