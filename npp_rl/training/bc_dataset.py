"""Behavioral Cloning Dataset for N++ Replay Data.

This module provides a PyTorch Dataset for loading compact replay files
and generating training data for behavioral cloning pretraining.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from nclone.replay import CompactReplay
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig

logger = logging.getLogger(__name__)


class BCReplayDataset(Dataset):
    """PyTorch Dataset for behavioral cloning from N++ replay files.
    
    Loads compact .replay files, generates observations using the N++ simulator,
    and provides (observation, action) pairs for BC training.
    
    Features:
    - Loads .replay files from directory
    - Generates observations deterministically
    - Caches processed data to NPZ files for efficiency
    - Supports filtering by success/quality
    - Handles multimodal observations (vision, graph, state)
    """
    
    def __init__(
        self,
        replay_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        filter_successful_only: bool = True,
        max_replays: Optional[int] = None,
    ):
        """Initialize BC replay dataset.
        
        Args:
            replay_dir: Directory containing .replay files
            cache_dir: Directory for caching processed data (default: replay_dir/cache)
            use_cache: Whether to use cached processed data
            filter_successful_only: Only include successful replays
            max_replays: Maximum number of replays to load (None for all)
        """
        self.replay_dir = Path(replay_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.replay_dir / "cache"
        self.use_cache = use_cache
        self.filter_successful_only = filter_successful_only
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for processed data
        self.samples: List[Tuple[Dict, int]] = []
        
        # Load replay files
        replay_files = self._load_replay_files(max_replays)
        logger.info(f"Found {len(replay_files)} replay files in {self.replay_dir}")
        
        # Process replays
        self._process_replays(replay_files)
        
        logger.info(f"Loaded {len(self.samples)} training samples")
    
    def _load_replay_files(self, max_replays: Optional[int]) -> List[Path]:
        """Load list of replay files from directory.
        
        Args:
            max_replays: Maximum number of replays to load
            
        Returns:
            List of replay file paths
        """
        replay_files = sorted(self.replay_dir.glob("*.replay"))
        
        if max_replays is not None:
            replay_files = replay_files[:max_replays]
        
        return replay_files
    
    def _get_cache_path(self, replay_path: Path) -> Path:
        """Get cache file path for a replay file.
        
        Args:
            replay_path: Path to replay file
            
        Returns:
            Path to cache file
        """
        cache_name = replay_path.stem + ".npz"
        return self.cache_dir / cache_name
    
    def _process_replays(self, replay_files: List[Path]) -> None:
        """Process all replay files and build dataset.
        
        Args:
            replay_files: List of replay file paths
        """
        for replay_path in replay_files:
            try:
                # Check cache first
                cache_path = self._get_cache_path(replay_path)
                
                if self.use_cache and cache_path.exists():
                    samples = self._load_from_cache(cache_path)
                else:
                    samples = self._process_replay_file(replay_path)
                    
                    if self.use_cache:
                        self._save_to_cache(samples, cache_path)
                
                # Add samples to dataset
                self.samples.extend(samples)
                
            except Exception as e:
                logger.warning(f"Failed to process {replay_path.name}: {e}")
                continue
    
    def _process_replay_file(self, replay_path: Path) -> List[Tuple[Dict, int]]:
        """Process a single replay file into training samples.
        
        Args:
            replay_path: Path to replay file
            
        Returns:
            List of (observation, action) tuples
        """
        # Load replay
        with open(replay_path, 'rb') as f:
            replay_data = f.read()
        
        replay = CompactReplay.from_binary(replay_data)
        
        # Filter by success if requested
        if self.filter_successful_only and not replay.success:
            logger.debug(f"Skipping unsuccessful replay: {replay_path.name}")
            return []
        
        # Generate observations by simulating the replay
        samples = self._simulate_replay(replay)
        
        return samples
    
    def _simulate_replay(self, replay: CompactReplay) -> List[Tuple[Dict, int]]:
        """Simulate a replay to generate observations and actions.
        
        Args:
            replay: CompactReplay object
            
        Returns:
            List of (observation, action) tuples
        """
        samples = []
        
        try:
            # Use ReplayExecutor to regenerate observations deterministically
            from nclone.replay.replay_executor import ReplayExecutor
            
            executor = ReplayExecutor()
            
            # Execute replay to get observations
            observations = executor.execute_replay(replay.map_data, replay.input_sequence)
            
            executor.close()
            
            # Pair observations with actions
            # Note: input_sequence has one action per frame, observations align with frames
            for i, (obs, action) in enumerate(zip(observations, replay.input_sequence)):
                samples.append((self._process_observation(obs), int(action)))
            
        except Exception as e:
            logger.warning(f"Failed to simulate replay: {e}")
            # Try fallback method using environment directly
            try:
                samples = self._simulate_replay_with_env(replay)
            except Exception as e2:
                logger.warning(f"Fallback simulation also failed: {e2}")
                return []
        
        return samples
    
    def _simulate_replay_with_env(self, replay: CompactReplay) -> List[Tuple[Dict, int]]:
        """Fallback method: simulate replay using environment directly.
        
        This is less efficient but more compatible if ReplayExecutor has issues.
        
        Args:
            replay: CompactReplay object
            
        Returns:
            List of (observation, action) tuples
        """
        samples = []
        
        # Create a temporary level file from the map data
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npp', delete=False) as tmp:
            tmp.write(replay.map_data)
            tmp_path = tmp.name
        
        try:
            # Create environment
            config = EnvironmentConfig.for_training()
            env = NppEnvironment(config=config)
            
            # Load level from file
            env.load_level(tmp_path)
            
            # Get initial observation
            obs, _ = env.reset()
            
            # Execute each input from the replay
            for action in replay.input_sequence:
                # Store the observation-action pair
                samples.append((self._process_observation(obs), int(action)))
                
                # Step environment
                obs, _, terminated, truncated, _ = env.step(action)
                
                # Stop if episode ends
                if terminated or truncated:
                    break
            
            env.close()
            
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return samples
    
    def _process_observation(self, obs: Dict) -> Dict:
        """Process observation into format suitable for caching.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Processed observation dictionary
        """
        # Extract only the components needed for training
        # Skip complex objects like level_data that can't be easily cached
        processed = {
            'game_state': obs['game_state'].copy(),
            'global_view': obs['global_view'].copy(),
            'reachability_features': obs['reachability_features'].copy(),
            'player_frame': obs['player_frame'].copy(),
            'graph_node_feats': obs['graph_node_feats'].copy(),
            'graph_edge_index': obs['graph_edge_index'].copy(),
            'graph_edge_feats': obs['graph_edge_feats'].copy(),
            'graph_node_mask': obs['graph_node_mask'].copy(),
            'graph_edge_mask': obs['graph_edge_mask'].copy(),
            'graph_node_types': obs['graph_node_types'].copy(),
            'graph_edge_types': obs['graph_edge_types'].copy(),
        }
        
        return processed
    
    def _save_to_cache(self, samples: List[Tuple[Dict, int]], cache_path: Path) -> None:
        """Save processed samples to cache file.
        
        Args:
            samples: List of (observation, action) tuples
            cache_path: Path to cache file
        """
        if not samples:
            return
        
        try:
            # Prepare data for saving
            cache_data = {
                'num_samples': len(samples),
            }
            
            # Store each observation component and action
            for i, (obs, action) in enumerate(samples):
                for key, value in obs.items():
                    cache_key = f'obs_{i}_{key}'
                    cache_data[cache_key] = value
                
                cache_data[f'action_{i}'] = action
            
            # Save to NPZ
            np.savez_compressed(cache_path, **cache_data)
            logger.debug(f"Cached {len(samples)} samples to {cache_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")
    
    def _load_from_cache(self, cache_path: Path) -> List[Tuple[Dict, int]]:
        """Load processed samples from cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            List of (observation, action) tuples
        """
        try:
            data = np.load(cache_path, allow_pickle=False)
            num_samples = int(data['num_samples'])
            
            samples = []
            for i in range(num_samples):
                # Reconstruct observation
                obs = {}
                for key in ['game_state', 'global_view', 'reachability_features',
                           'player_frame', 'graph_node_feats', 'graph_edge_index',
                           'graph_edge_feats', 'graph_node_mask', 'graph_edge_mask',
                           'graph_node_types', 'graph_edge_types']:
                    cache_key = f'obs_{i}_{key}'
                    if cache_key in data:
                        obs[key] = data[cache_key]
                
                # Get action
                action = int(data[f'action_{i}'])
                
                samples.append((obs, action))
            
            logger.debug(f"Loaded {len(samples)} samples from cache {cache_path.name}")
            return samples
            
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return []
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (observation_dict, action_tensor)
        """
        obs, action = self.samples[idx]
        
        # Convert observation to tensors
        obs_tensors = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.from_numpy(value)
            else:
                obs_tensors[key] = torch.tensor(value)
        
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.long)
        
        return obs_tensors, action_tensor
