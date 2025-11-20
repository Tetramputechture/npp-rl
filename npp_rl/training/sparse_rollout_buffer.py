"""Sparse Rollout Buffer for handling variable-sized graph observations.

This module provides a memory-optimized rollout buffer that stores sparse graph
observations in COO format, achieving ~95% memory reduction compared to the
standard RolloutBuffer with padded graph observations.

The buffer handles variable-sized arrays for graph components while maintaining
compatibility with Stable-Baselines3's PPO algorithm.
"""

import numpy as np
import torch
from typing import Dict, Optional, Generator, Union, List
from stable_baselines3.common.buffers import DictRolloutBuffer, DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces


class SparseGraphRolloutBuffer(DictRolloutBuffer):
    """Rollout buffer that handles sparse graph observations efficiently.

    Extends SB3's DictRolloutBuffer to store variable-sized graph observations in
    sparse COO format, avoiding memory waste from padding. Non-graph observations
    are stored normally in fixed-size arrays.

    Memory savings: ~95% reduction for graph observations
    - Dense format: 4500 nodes * 17 features * 4 bytes = ~77KB per observation
    - Sparse format: ~200 nodes * 17 features * 4 bytes = ~3.4KB per observation

    The buffer stores:
    - Standard SB3 data (observations, actions, rewards, etc.) in base buffer
    - Sparse graph data in separate lists (one entry per step)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        """Initialize sparse graph rollout buffer.

        Args:
            buffer_size: Max number of elements in buffer
            observation_space: Observation space
            action_space: Action space
            device: PyTorch device
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
            n_envs: Number of parallel environments
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

        # Separate storage for sparse graph observations
        # Each entry is a list of n_envs sparse observations for that timestep
        # Format: List[timestep][env_idx] -> Dict with sparse graph arrays
        self.sparse_graph_obs: List[List[Optional[Dict[str, np.ndarray]]]] = []

        # Track which observation keys are sparse graph data
        self.sparse_keys = {
            "graph_node_feats_sparse",
            "graph_edge_index_sparse",
            "graph_edge_feats_sparse",
            "graph_node_types_sparse",
            "graph_edge_types_sparse",
            "graph_num_nodes",
            "graph_num_edges",
        }

    def reset(self) -> None:
        """Reset buffer to empty state."""
        super().reset()
        self.sparse_graph_obs = []

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """Add new transition to buffer.

        Separates sparse graph observations from regular observations and stores
        them in separate data structures for memory efficiency.

        Args:
            obs: Observation dict from environment (may contain sparse graph data)
            action: Action taken
            reward: Reward received
            episode_start: Episode start flags
            value: Value estimate
            log_prob: Log probability of action
        """
        # Extract and store sparse graph data separately
        sparse_graph_data = []

        for env_idx in range(self.n_envs):
            # Extract sparse graph data for this environment
            env_sparse_data = {}
            has_sparse = False

            for key in self.sparse_keys:
                if key in obs:
                    # VecEnv format: obs[key] is array/list with one entry per env
                    if self.n_envs == 1:
                        env_sparse_data[key] = obs[key]
                    else:
                        env_sparse_data[key] = obs[key][env_idx]
                    has_sparse = True

            sparse_graph_data.append(env_sparse_data if has_sparse else None)

        # Store sparse graph data in separate structure
        if len(self.sparse_graph_obs) <= self.pos:
            self.sparse_graph_obs.append(sparse_graph_data)
        else:
            self.sparse_graph_obs[self.pos] = sparse_graph_data

        # Remove sparse keys from observation dict before passing to parent
        obs_non_sparse = {k: v for k, v in obs.items() if k not in self.sparse_keys}

        # Store non-sparse observations using parent class
        super().add(obs_non_sparse, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator:
        """Generate batches of transitions for training.

        Reconstructs full observations by combining non-sparse data from parent
        buffer with sparse graph data from separate storage.

        Args:
            batch_size: Size of batches to generate (None = full buffer)

        Yields:
            RolloutBufferSamples with reconstructed observations
        """
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # If no batch_size specified, return entire buffer
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < len(indices):
            batch_indices = indices[start_idx : start_idx + batch_size]

            # Get non-sparse data from parent buffer using internal method
            # This returns observations without sparse graph data
            yield self._get_samples(batch_indices)

            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ):
        """Get samples at specified indices.

        Reconstructs full observations including sparse graph data.

        Args:
            batch_inds: Indices of samples to retrieve
            env: Optional VecNormalize environment for normalization

        Returns:
            RolloutBufferSamples with full observations
        """
        # DictRolloutBuffer stores observations as [buffer_size, n_envs, ...]
        # but batch_inds are flat indices [0, buffer_size * n_envs)
        # We need to convert them to 2D indices
        buffer_pos = (
            batch_inds // self.n_envs
        )  # Which timestep in buffer [0, buffer_size)
        env_idx = batch_inds % self.n_envs  # Which environment [0, n_envs)

        # Manually get samples using 2D indexing (parent method would fail)
        observations = {}
        for key, obs in self.observations.items():
            # obs is [buffer_size, n_envs, ...]
            # Index with [buffer_pos, env_idx] to get [batch_size, ...]
            observations[key] = self.to_torch(obs[buffer_pos, env_idx])

        # Get other buffer components using 2D indexing
        actions = self.to_torch(self.actions[buffer_pos, env_idx])
        old_values = self.to_torch(self.values[buffer_pos, env_idx].flatten())
        old_log_prob = self.to_torch(self.log_probs[buffer_pos, env_idx].flatten())
        advantages = self.to_torch(self.advantages[buffer_pos, env_idx].flatten())
        returns = self.to_torch(self.returns[buffer_pos, env_idx].flatten())

        # Reconstruct sparse graph observations and add to observations dict
        # Use the same index mapping
        timesteps = buffer_pos
        env_indices = env_idx

        # Collect sparse graph data for this batch
        sparse_batch_data = {}
        for key in self.sparse_keys:
            batch_sparse_data = []
            for t, e in zip(timesteps, env_indices):
                if (
                    t < len(self.sparse_graph_obs)
                    and self.sparse_graph_obs[t][e] is not None
                ):
                    sparse_data = self.sparse_graph_obs[t][e]
                    if key in sparse_data:
                        batch_sparse_data.append(sparse_data[key])
                    else:
                        # Key not in this sparse observation
                        batch_sparse_data.append(self._empty_sparse_array(key))
                else:
                    # No sparse data for this timestep/env
                    batch_sparse_data.append(self._empty_sparse_array(key))
            sparse_batch_data[key] = batch_sparse_data

        # Convert sparse data to dense and update observations dict
        # This ensures observations match the observation space definition
        dense_graphs = self._convert_sparse_batch_to_dense(sparse_batch_data)
        observations.update(dense_graphs)

        # Return DictRolloutBufferSamples with dense graph data
        return DictRolloutBufferSamples(
            observations=observations,
            actions=actions,
            old_values=old_values,
            old_log_prob=old_log_prob,
            advantages=advantages,
            returns=returns,
        )

    def _empty_sparse_array(self, key: str) -> np.ndarray:
        """Create empty sparse array for a given key.

        Args:
            key: Sparse observation key

        Returns:
            Empty numpy array with appropriate shape and dtype
        """
        if key == "graph_node_feats_sparse":
            return np.zeros((0, 17), dtype=np.float32)  # NODE_FEATURE_DIM=17
        elif key == "graph_edge_index_sparse":
            return np.zeros((2, 0), dtype=np.uint16)
        elif key == "graph_edge_feats_sparse":
            return np.zeros((0, 12), dtype=np.float32)  # EDGE_FEATURE_DIM=12
        elif key == "graph_node_types_sparse":
            return np.zeros(0, dtype=np.uint8)
        elif key == "graph_edge_types_sparse":
            return np.zeros(0, dtype=np.uint8)
        elif key in ("graph_num_nodes", "graph_num_edges"):
            return np.array([0], dtype=np.int32)
        else:
            return np.array([])

    def _convert_sparse_batch_to_dense(
        self, sparse_batch_data: Dict[str, List[np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """Convert a batch of sparse graph data to dense padded tensors.

        Args:
            sparse_batch_data: Dictionary mapping sparse keys to lists of numpy arrays

        Returns:
            Dictionary mapping dense keys to padded torch tensors
        """
        # Determine batch size
        batch_size = (
            len(next(iter(sparse_batch_data.values()))) if sparse_batch_data else 0
        )
        if batch_size == 0:
            return {}

        # Get maximum sizes for padding
        max_num_nodes = max(
            arr.shape[0] if arr.ndim > 0 and arr.shape[0] > 0 else 0
            for arr in sparse_batch_data.get("graph_node_feats_sparse", [])
        )
        max_num_edges = max(
            arr.shape[1] if arr.ndim > 1 and arr.shape[1] > 0 else 0
            for arr in sparse_batch_data.get("graph_edge_index_sparse", [])
        )

        # Ensure minimum sizes to avoid zero-size tensors
        max_num_nodes = max(max_num_nodes, 1)
        max_num_edges = max(max_num_edges, 1)

        # Create dense tensors
        dense_graphs = {}

        # Node features: [batch_size, max_num_nodes, node_feat_dim]
        if "graph_node_feats_sparse" in sparse_batch_data:
            node_feat_dim = 17  # NODE_FEATURE_DIM
            node_feats = np.zeros(
                (batch_size, max_num_nodes, node_feat_dim), dtype=np.float32
            )
            for i, arr in enumerate(sparse_batch_data["graph_node_feats_sparse"]):
                if arr.shape[0] > 0:
                    node_feats[i, : arr.shape[0], :] = arr
            dense_graphs["graph_node_feats"] = self.to_torch(node_feats)

        # Edge index: [batch_size, 2, max_num_edges]
        if "graph_edge_index_sparse" in sparse_batch_data:
            edge_index = np.zeros((batch_size, 2, max_num_edges), dtype=np.int64)
            for i, arr in enumerate(sparse_batch_data["graph_edge_index_sparse"]):
                if arr.shape[1] > 0:
                    edge_index[i, :, : arr.shape[1]] = arr
            dense_graphs["graph_edge_index"] = self.to_torch(edge_index)

        # Edge features: [batch_size, max_num_edges, edge_feat_dim]
        if "graph_edge_feats_sparse" in sparse_batch_data:
            edge_feat_dim = 12  # EDGE_FEATURE_DIM
            edge_feats = np.zeros(
                (batch_size, max_num_edges, edge_feat_dim), dtype=np.float32
            )
            for i, arr in enumerate(sparse_batch_data["graph_edge_feats_sparse"]):
                if arr.shape[0] > 0:
                    edge_feats[i, : arr.shape[0], :] = arr
            dense_graphs["graph_edge_feats"] = self.to_torch(edge_feats)

        # Node types: [batch_size, max_num_nodes]
        if "graph_node_types_sparse" in sparse_batch_data:
            node_types = np.zeros((batch_size, max_num_nodes), dtype=np.int64)
            for i, arr in enumerate(sparse_batch_data["graph_node_types_sparse"]):
                if arr.shape[0] > 0:
                    node_types[i, : arr.shape[0]] = arr
            dense_graphs["graph_node_types"] = self.to_torch(node_types)

        # Edge types: [batch_size, max_num_edges]
        if "graph_edge_types_sparse" in sparse_batch_data:
            edge_types = np.zeros((batch_size, max_num_edges), dtype=np.int64)
            for i, arr in enumerate(sparse_batch_data["graph_edge_types_sparse"]):
                if arr.shape[0] > 0:
                    edge_types[i, : arr.shape[0]] = arr
            dense_graphs["graph_edge_types"] = self.to_torch(edge_types)

        # Generate masks to indicate valid (non-padding) elements
        # Node mask: [batch_size, max_num_nodes]
        node_mask = np.zeros((batch_size, max_num_nodes), dtype=np.int32)
        for i, arr in enumerate(sparse_batch_data.get("graph_node_feats_sparse", [])):
            if arr.shape[0] > 0:
                node_mask[i, : arr.shape[0]] = 1
        dense_graphs["graph_node_mask"] = self.to_torch(node_mask)

        # Edge mask: [batch_size, max_num_edges]
        edge_mask = np.zeros((batch_size, max_num_edges), dtype=np.int32)
        for i, arr in enumerate(sparse_batch_data.get("graph_edge_index_sparse", [])):
            if arr.shape[1] > 0:
                edge_mask[i, : arr.shape[1]] = 1
        dense_graphs["graph_edge_mask"] = self.to_torch(edge_mask)

        return dense_graphs

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: np.ndarray
    ) -> None:
        """Compute returns and advantage using GAE.

        This is unchanged from parent implementation since it only uses
        values and rewards, not observations.

        Args:
            last_values: Value estimates for last observations
            dones: Done flags for last observations
        """
        super().compute_returns_and_advantage(last_values, dones)

    def size_bytes(self) -> Dict[str, int]:
        """Calculate memory usage of buffer in bytes.

        Returns:
            Dict with memory usage breakdown
        """
        # Get parent buffer size (non-sparse observations)
        parent_size = 0
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                parent_size += value.nbytes

        # Calculate sparse graph data size
        sparse_size = 0
        for timestep_data in self.sparse_graph_obs:
            for env_data in timestep_data:
                if env_data:
                    for array in env_data.values():
                        if isinstance(array, np.ndarray):
                            sparse_size += array.nbytes

        return {
            "non_sparse_bytes": parent_size,
            "sparse_graph_bytes": sparse_size,
            "total_bytes": parent_size + sparse_size,
        }
