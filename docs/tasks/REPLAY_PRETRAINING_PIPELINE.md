# Replay-Based Pre-Training Pipeline

**Status**: Planning  
**Priority**: High  
**Complexity**: Medium-High

---

## Overview

This document provides comprehensive technical instructions for implementing a replay-based pre-training system. The pipeline enables:
1. Recording human gameplay demonstrations during test environment runs
2. Pre-training models using behavioral cloning (BC) on recorded demonstrations
3. Fine-tuning pre-trained models on training datasets and evaluating against test datasets
4. Comparing pre-trained models against baseline models without pre-training

This approach leverages imitation learning to initialize policies with human-like behaviors before reinforcement learning fine-tuning, potentially accelerating convergence and improving sample efficiency.

### Key Design Principle: Deterministic Replay

**Critical: The N++ physics simulation is completely deterministic.** Given a map and input sequence, replaying those inputs will always produce identical game states and observations. This enables an extremely efficient storage format:

**Instead of storing:** Full observations (~500KB+ per episode with images, state, etc.)  
**We store:** Map data (1335 bytes) + Input sequence (1 byte per frame)  

**Storage reduction:** 100-1000x compression  
**Benefits:**
- Minimal disk space requirements
- Flexible observation format changes without re-recording
- Different architectures can process the same replay data
- Fast regeneration during training

This follows the same pattern as N++ attract files (`nclone/nclone/replay/npp_attract_decoder.py`).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Phase 1: Recording Gameplay Demonstrations](#phase-1-recording-gameplay-demonstrations)
3. [Phase 2: Pre-Training with Behavioral Cloning](#phase-2-pre-training-with-behavioral-cloning)
4. [Phase 3: Fine-Tuning and Evaluation](#phase-3-fine-tuning-and-evaluation)
5. [Phase 4: Baseline Comparison](#phase-4-baseline-comparison)
6. [File Structure and Organization](#file-structure-and-organization)
7. [Technical References](#technical-references)

---

## System Architecture

### Data Flow Pipeline

```
Human Gameplay (test_environment.py)
    ↓
Recording System (captures ONLY: map data + input sequence)
    ↓
Episode Success Detection
    ↓
Replay File Generation (compact .replay format: map + inputs)
    ↓
Replay Executor (deterministic regeneration)
    ├── Loads map data into environment
    ├── Replays input sequence frame-by-frame
    └── Generates observations on-demand
    ↓
Processed Training Data (NPZ shards with regenerated observations)
    ↓
Behavioral Cloning Pre-Training (bc_pretrain.py)
    ↓
Pre-Trained Model Checkpoints (.pth + .zip)
    ↓
Fine-Tuning on Train Dataset (nclone/datasets/train/)
    ↓
Evaluation on Test Dataset (nclone/datasets/test/)
    ↓
Performance Metrics & Comparison
```

**Key Efficiency:** Replay files are ~1-5KB each (not 500KB+). During training data generation, observations are regenerated deterministically using the replay executor.

### Key Components

**Existing Infrastructure:**
- `nclone/nclone/test_environment.py` - Interactive test environment with keyboard controls
- `npp-rl/tools/replay_ingest.py` - Replay data ingestion and processing
- `npp-rl/bc_pretrain.py` - Behavioral cloning training script
- `npp-rl/npp_rl/training/bc_trainer.py` - BC trainer implementation
- `npp-rl/npp_rl/agents/training.py` - Main RL training script
- `nclone/nclone/evaluation/test_suite_loader.py` - Dataset loading utilities
- `npp-rl/npp_rl/evaluation/test_suite_loader.py` - NPP-RL dataset loader

**Components to Create:**
- `nclone/nclone/replay/gameplay_recorder.py` - Compact replay recording (map + inputs only)
- `nclone/nclone/replay/replay_executor.py` - **Deterministic replay executor** (regenerates observations)
- `npp-rl/tools/generate_bc_data.py` - Batch observation generation from replays
- `npp-rl/scripts/pretrain_and_finetune.py` - End-to-end pre-training + fine-tuning pipeline
- `npp-rl/scripts/evaluate_pretrained_models.py` - Comprehensive evaluation script
- `npp-rl/scripts/compare_baselines.py` - Baseline comparison utilities

---

## Phase 1: Recording Gameplay Demonstrations

### 1.1 Gameplay Recorder Implementation

Create `nclone/nclone/replay/gameplay_recorder.py`:

```python
"""
Compact Gameplay Recorder for Human Demonstrations

Records ONLY map data and input sequence during test environment runs.
Leverages deterministic physics to regenerate observations on-demand.

Storage format mirrors N++ attract files:
- Map data: 1335 bytes (fixed)
- Input sequence: 1 byte per frame
- Total: ~1-5KB per episode (vs 500KB+ with observations)
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CompactReplay:
    """Compact replay format - only inputs and map data."""
    
    episode_id: str
    map_data: bytes  # Raw map data (1335 bytes)
    input_sequence: List[int]  # Input values (0-7 per frame)
    
    # Metadata
    level_id: Optional[str]
    start_time: datetime
    end_time: datetime
    success: bool
    
    def to_binary(self) -> bytes:
        """
        Convert to binary format (similar to N++ attract files).
        
        Format:
        - Header (8 bytes):
            - Map data length (4 bytes, uint32)
            - Input sequence length (4 bytes, uint32)
        - Map data (variable, typically 1335 bytes)
        - Input sequence (variable, 1 byte per frame)
        """
        map_data_len = len(self.map_data)
        input_seq_len = len(self.input_sequence)
        
        # Pack header
        header = struct.pack("<II", map_data_len, input_seq_len)
        
        # Pack input sequence
        inputs_bytes = bytes(self.input_sequence)
        
        # Combine
        return header + self.map_data + inputs_bytes
    
    @classmethod
    def from_binary(cls, data: bytes, episode_id: str = "unknown") -> "CompactReplay":
        """Load from binary format."""
        # Unpack header
        map_data_len, input_seq_len = struct.unpack("<II", data[0:8])
        
        # Extract map data
        map_data = data[8:8+map_data_len]
        
        # Extract input sequence
        input_start = 8 + map_data_len
        input_sequence = list(data[input_start:input_start+input_seq_len])
        
        return cls(
            episode_id=episode_id,
            map_data=map_data,
            input_sequence=input_sequence,
            level_id=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
        )
    
    def get_file_size(self) -> int:
        """Get file size in bytes."""
        return 8 + len(self.map_data) + len(self.input_sequence)


def map_action_to_input(action: int) -> int:
    """
    Map discrete action (0-5) to N++ input byte (0-7).
    
    Actions:
        0: NOOP
        1: LEFT
        2: RIGHT
        3: JUMP
        4: LEFT + JUMP
        5: RIGHT + JUMP
    
    Input encoding (bit flags):
        Bit 0: Jump
        Bit 1: Right
        Bit 2: Left
    """
    action_to_input_map = {
        0: 0,  # NOOP: 000
        1: 4,  # LEFT: 100
        2: 2,  # RIGHT: 010
        3: 1,  # JUMP: 001
        4: 5,  # LEFT+JUMP: 101
        5: 3,  # RIGHT+JUMP: 011
    }
    return action_to_input_map.get(action, 0)


class GameplayRecorder:
    """Compact gameplay recorder - stores only map data and input sequence."""
    
    def __init__(self, output_dir: str = "datasets/human_replays"):
        """Initialize gameplay recorder.
        
        Args:
            output_dir: Directory to save recorded replays
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.current_map_data: Optional[bytes] = None
        self.current_input_sequence: List[int] = []
        self.current_episode_id: Optional[str] = None
        self.current_level_id: Optional[str] = None
        self.episode_start_time: Optional[datetime] = None
        
        # Statistics
        self.total_episodes_recorded = 0
        self.successful_episodes_recorded = 0
        
    def start_recording(self, map_data: bytes, map_name: str = "unknown", level_id: Optional[str] = None):
        """Start recording a new episode.
        
        Args:
            map_data: Raw map data bytes (1335 bytes)
            map_name: Name/identifier for the map
            level_id: Optional level ID from test suite
        """
        if self.is_recording:
            print("⚠️  Already recording, stopping previous episode")
            self.stop_recording(success=False)
        
        self.is_recording = True
        self.current_map_data = map_data
        self.current_input_sequence = []
        self.episode_start_time = datetime.now()
        self.current_level_id = level_id
        
        # Generate unique episode ID
        self.current_episode_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{map_name}"
        
        print(f"🔴 Recording started: {self.current_episode_id}")
        print(f"   Map size: {len(map_data)} bytes")
    
    def record_action(self, action: int):
        """Record a single action.
        
        Args:
            action: Discrete action taken (0-5)
        """
        if not self.is_recording:
            return
        
        # Convert action to input byte
        input_byte = map_action_to_input(action)
        self.current_input_sequence.append(input_byte)
    
    def stop_recording(self, success: bool, save: bool = True) -> Optional[str]:
        """Stop recording and optionally save the replay.
        
        Args:
            success: Whether the episode ended in success (level completed)
            save: Whether to save the replay to disk
        
        Returns:
            Path to saved replay file, or None if not saved
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        # Update statistics
        self.total_episodes_recorded += 1
        if success:
            self.successful_episodes_recorded += 1
        
        # Save replay if successful and save flag is True
        saved_path = None
        if save and success and self.current_map_data is not None:
            saved_path = self._save_replay()
            
            duration = (datetime.now() - self.episode_start_time).total_seconds()
            file_size = 8 + len(self.current_map_data) + len(self.current_input_sequence)
            
            print(f"✅ Replay saved: {saved_path}")
            print(f"   - Frames: {len(self.current_input_sequence)}")
            print(f"   - Duration: {duration:.1f}s")
            print(f"   - File size: {file_size} bytes ({file_size/1024:.1f} KB)")
        elif not success:
            print(f"❌ Episode failed, not saved")
        
        # Reset state
        self.current_map_data = None
        self.current_input_sequence = []
        self.current_episode_id = None
        
        return saved_path
    
    def _save_replay(self) -> str:
        """Save replay to disk in compact binary format.
        
        Returns:
            Path to saved replay file
        """
        if self.current_map_data is None or self.current_episode_id is None:
            raise ValueError("Cannot save replay: missing data")
        
        # Create compact replay
        replay = CompactReplay(
            episode_id=self.current_episode_id,
            map_data=self.current_map_data,
            input_sequence=self.current_input_sequence,
            level_id=self.current_level_id,
            start_time=self.episode_start_time,
            end_time=datetime.now(),
            success=True,
        )
        
        # Save binary replay file
        replay_path = self.output_dir / f"{self.current_episode_id}.replay"
        with open(replay_path, "wb") as f:
            f.write(replay.to_binary())
        
        return str(replay_path)
    
    def print_statistics(self):
        """Print recording statistics."""
        print("\n" + "=" * 60)
        print("COMPACT REPLAY RECORDING STATISTICS")
        print("=" * 60)
        print(f"Total episodes recorded: {self.total_episodes_recorded}")
        print(f"Successful episodes: {self.successful_episodes_recorded}")
        if self.total_episodes_recorded > 0:
            success_rate = self.successful_episodes_recorded / self.total_episodes_recorded * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        # List replay files and total storage
        replay_files = list(self.output_dir.glob("*.replay"))
        if replay_files:
            total_size = sum(f.stat().st_size for f in replay_files)
            print(f"Replay files: {len(replay_files)}")
            print(f"Total storage: {total_size / 1024:.1f} KB")
            print(f"Average file size: {total_size / len(replay_files) / 1024:.1f} KB")
        print("=" * 60)
```

### 1.2 Replay Executor (Deterministic Observation Generation)

Create `nclone/nclone/replay/replay_executor.py`:

```python
"""
Deterministic Replay Executor

Replays stored input sequences against map data to regenerate observations.
Leverages the completely deterministic nature of N++ physics simulation.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path

from ..nplay_headless import NPlayHeadless
from ..gym_environment.observation_processor import ObservationProcessor


def map_input_to_action(input_byte: int) -> int:
    """
    Map N++ input byte (0-7) to discrete action (0-5).
    
    Inverse of map_action_to_input from gameplay_recorder.py
    """
    input_to_action_map = {
        0: 0,  # 000: NOOP
        1: 3,  # 001: JUMP
        2: 2,  # 010: RIGHT
        3: 5,  # 011: RIGHT+JUMP
        4: 1,  # 100: LEFT
        5: 4,  # 101: LEFT+JUMP
        6: 0,  # 110: Invalid (mapped to NOOP)
        7: 0,  # 111: Invalid (mapped to NOOP)
    }
    return input_to_action_map.get(input_byte, 0)


def decode_input_to_controls(input_byte: int) -> tuple[int, int]:
    """
    Decode input byte to horizontal and jump controls.
    
    Returns:
        (horizontal, jump) where:
            horizontal: -1 (left), 0 (none), 1 (right)
            jump: 0 (no jump), 1 (jump)
    """
    jump = 1 if (input_byte & 0x01) else 0
    right = 1 if (input_byte & 0x02) else 0
    left = 1 if (input_byte & 0x04) else 0
    
    # Handle conflicting inputs
    if left and right:
        horizontal = 0
    elif left:
        horizontal = -1
    elif right:
        horizontal = 1
    else:
        horizontal = 0
    
    return horizontal, jump


class ReplayExecutor:
    """Executes replays deterministically to generate observations."""
    
    def __init__(
        self,
        observation_config: Optional[Dict[str, Any]] = None,
        render_mode: str = "rgb_array",
    ):
        """Initialize replay executor.
        
        Args:
            observation_config: Configuration for observation processor
            render_mode: Rendering mode for environment
        """
        self.observation_config = observation_config or {}
        self.render_mode = render_mode
        
        # Create headless environment
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=False,
            enable_logging=False,
            enable_debug_overlay=False,
            seed=42,  # Fixed seed for determinism
        )
        
        # Create observation processor
        self.obs_processor = ObservationProcessor(
            enable_augmentation=False,  # No augmentation for replay
        )
    
    def execute_replay(
        self,
        map_data: bytes,
        input_sequence: List[int],
    ) -> List[Dict[str, Any]]:
        """Execute a replay and generate observations for each frame.
        
        Args:
            map_data: Raw map data (1335 bytes)
            input_sequence: Input sequence (1 byte per frame)
        
        Returns:
            List of observations, one per frame
        """
        # Load map
        self.nplay_headless.load_map_from_map_data(list(map_data))
        
        observations = []
        
        # Execute each input frame
        for frame_idx, input_byte in enumerate(input_sequence):
            # Decode input to controls
            horizontal, jump = decode_input_to_controls(input_byte)
            
            # Execute one simulation step
            self.nplay_headless.tick(horizontal, jump)
            
            # Get raw observation
            raw_obs = self._get_raw_observation()
            
            # Process observation
            processed_obs = self.obs_processor.process_observation(raw_obs)
            
            # Get discrete action for this frame
            action = map_input_to_action(input_byte)
            
            # Store observation with action
            observations.append({
                "observation": processed_obs,
                "action": action,
                "frame": frame_idx,
            })
        
        return observations
    
    def _get_raw_observation(self) -> Dict[str, Any]:
        """Get raw observation from environment."""
        # Render current frame
        screen = self.nplay_headless.render()
        
        # Get ninja position
        ninja_x, ninja_y = self.nplay_headless.ninja_position()
        
        # Get ninja state
        ninja = self.nplay_headless.sim.ninja
        
        # Build raw observation (similar to npp_environment.py)
        obs = {
            "screen": screen,
            "player_x": ninja_x,
            "player_y": ninja_y,
            "game_state": np.array([
                # Ninja physics state (simplified)
                ninja_x / 1008,  # Normalized position
                ninja_y / 552,
                ninja.xvel / 10.0,  # Normalized velocity
                ninja.yvel / 10.0,
                float(ninja.on_ground),
                float(ninja.wall_sliding),
                # Add more state as needed...
            ] + [0.0] * 24, dtype=np.float32),  # Pad to 30 features
            "reachability_features": np.zeros(8, dtype=np.float32),  # Placeholder
        }
        
        return obs
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'nplay_headless'):
            del self.nplay_headless
```

### 1.3 Integration with Test Environment

Modify `nclone/nclone/test_environment.py` to add compact recording:

```python
# Add at the top with other imports
from nclone.replay.gameplay_recorder import GameplayRecorder

# Add after argument parser setup (around line 170)
parser.add_argument(
    "--record",
    action="store_true",
    help="Enable compact replay recording for behavioral cloning",
)
parser.add_argument(
    "--recording-output",
    type=str,
    default="datasets/human_replays",
    help="Output directory for recorded replays",
)

# Add after environment creation (around line 254)
recorder = None
if args.record:
    recorder = GameplayRecorder(output_dir=args.recording_output)
    print("\n" + "=" * 60)
    print("COMPACT REPLAY RECORDING ENABLED")
    print("=" * 60)
    print(f"Output directory: {args.recording_output}")
    print(f"Storage format: Binary (map + inputs only)")
    print("\nControls:")
    print("  B - Start recording")
    print("  N - Stop recording (without saving)")
    print("  R - Reset (auto-saves if successful)")
    print("=" * 60 + "\n")

# Add in main game loop (around line 680), in keyboard event handling:
if event.key == pygame.K_b and recorder is not None:
    # Start recording
    if not recorder.is_recording:
        map_data = bytes(env.nplay_headless.current_map_data)
        map_name = env.map_loader.current_map_name or "unknown"
        level_id = None  # TODO: Extract from env if using test suite
        recorder.start_recording(map_data, map_name, level_id)
    else:
        print("Already recording!")

if event.key == pygame.K_n and recorder is not None:
    # Stop recording without saving
    if recorder.is_recording:
        recorder.stop_recording(success=False, save=False)
        print("Recording stopped (not saved)")

# Modify reset handler (around line 686) to auto-save successful episodes:
if event.key == pygame.K_r:
    # Check if episode was successful before reset
    if recorder is not None and recorder.is_recording:
        # Determine if episode was successful
        player_won = env.nplay_headless.ninja_has_won()
        recorder.stop_recording(success=player_won, save=player_won)
    
    # Reset environment
    observation, info = env.reset()

# Add in the main step loop (around line 1011), to record each action:
# CRITICAL: Only record the action, not full observations
if recorder is not None and recorder.is_recording:
    recorder.record_action(action)

# Add cleanup at the end (around line 1035):
if recorder is not None:
    recorder.print_statistics()
```

**Key difference:** We only call `recorder.record_action(action)` - no need to extract or store any environment state. Everything will be regenerated deterministically!

### 1.4 Usage Instructions

**Recording human demonstrations:**

```bash
# Start test environment with compact recording enabled
cd /home/tetra/projects/nclone
python -m nclone.test_environment --record --recording-output datasets/human_replays

# Play the level:
# 1. Press 'B' to start recording (map + inputs only)
# 2. Play through the level
# 3. Complete the level successfully
# 4. Press 'R' to reset (automatically saves ~1-5KB replay file)

# Repeat for multiple levels/attempts

# View statistics and storage usage
# Statistics are printed automatically when the program exits
# Example output:
#   Replay files: 50
#   Total storage: 125 KB  (vs ~25 MB with full observations!)
#   Average file size: 2.5 KB
```

**Storage efficiency example:**
```
Traditional approach (storing observations):
- 50 episodes × 500KB/episode = 25 MB

Deterministic replay approach (inputs only):
- 50 episodes × 2.5KB/episode = 125 KB

Reduction: 200x smaller!
```

---

## Phase 2: Pre-Training with Behavioral Cloning

### 2.1 Observation Generation from Compact Replays

Create `npp-rl/tools/generate_bc_data.py`:

```python
#!/usr/bin/env python3
"""
Generate BC Training Data from Compact Replays

Reads compact replay files (.replay format with map + inputs) and uses
deterministic replay execution to generate full observations for training.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
import sys

# Add nclone to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nclone"))

from nclone.replay.gameplay_recorder import CompactReplay
from nclone.replay.replay_executor import ReplayExecutor


def process_replay_file(
    replay_path: Path,
    executor: ReplayExecutor,
) -> List[Dict]:
    """Process a single replay file to generate training samples.
    
    Args:
        replay_path: Path to .replay file
        executor: Replay executor instance
    
    Returns:
        List of training samples
    """
    # Load compact replay
    with open(replay_path, "rb") as f:
        replay_data = f.read()
    
    replay = CompactReplay.from_binary(replay_data, episode_id=replay_path.stem)
    
    # Execute replay to generate observations
    observations = executor.execute_replay(replay.map_data, replay.input_sequence)
    
    return observations


def save_batch_npz(samples: List[Dict], output_path: Path):
    """Save a batch of samples to NPZ format."""
    if not samples:
        return
    
    # Separate observations by key
    obs_dict = defaultdict(list)
    actions = []
    
    for sample in samples:
        obs = sample["observation"]
        action = sample["action"]
        
        # Collect observation components
        for key, value in obs.items():
            obs_dict[key].append(value)
        
        actions.append(action)
    
    # Stack into arrays
    stacked_obs = {}
    for key, values in obs_dict.items():
        stacked_obs[key] = np.stack(values)
    
    actions_array = np.array(actions, dtype=np.int32)
    
    # Save
    np.savez_compressed(
        output_path,
        **stacked_obs,
        actions=actions_array,
    )
    
    print(f"Saved batch with {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BC training data from compact replays"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing .replay files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for NPZ batches",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Samples per output file",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        raise ValueError(f"Input directory not found: {args.input}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Find all replay files
    replay_files = list(args.input.glob("*.replay"))
    print(f"Found {len(replay_files)} replay files")
    print(f"Total input size: {sum(f.stat().st_size for f in replay_files) / 1024:.1f} KB")
    
    # Create replay executor
    print("Initializing replay executor...")
    executor = ReplayExecutor()
    
    # Process all replays
    all_samples = []
    batch_num = 0
    
    for replay_file in tqdm(replay_files, desc="Processing replays"):
        try:
            samples = process_replay_file(replay_file, executor)
            all_samples.extend(samples)
            
            # Save batch if we have enough samples
            if len(all_samples) >= args.batch_size:
                output_path = args.output / f"batch_{batch_num:04d}.npz"
                save_batch_npz(all_samples[:args.batch_size], output_path)
                
                all_samples = all_samples[args.batch_size:]
                batch_num += 1
        
        except Exception as e:
            print(f"Error processing {replay_file}: {e}")
    
    # Save remaining samples
    if all_samples:
        output_path = args.output / f"batch_{batch_num:04d}.npz"
        save_batch_npz(all_samples, output_path)
    
    # Cleanup
    executor.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("OBSERVATION GENERATION COMPLETE")
    print("=" * 60)
    print(f"Replays processed: {len(replay_files)}")
    print(f"Batches created: {batch_num + 1}")
    print(f"Output directory: {args.output}")
    
    # Calculate output size
    output_files = list(args.output.glob("*.npz"))
    total_output_size = sum(f.stat().st_size for f in output_files)
    print(f"Total output size: {total_output_size / (1024*1024):.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
cd /home/tetra/projects/npp-rl

# Generate observations from compact replays
python tools/generate_bc_data.py \
    --input datasets/human_replays \
    --output datasets/bc_training \
    --batch_size 10000

# Expected output:
#   Found 50 replay files
#   Total input size: 125 KB
#   Processing replays: 100%|████████| 50/50
#   Saved batch with 10000 samples to batch_0000.npz
#   ...
#   OBSERVATION GENERATION COMPLETE
#   Replays processed: 50
#   Batches created: 5
#   Total output size: 45 MB
```

**Key advantage:** The same 125KB of replays can be regenerated with different observation configurations by just changing the `ReplayExecutor` settings!

### 2.2 Behavioral Cloning Pre-Training

Create `npp-rl/scripts/pretrain_policies.py`:

```python
#!/usr/bin/env python3
"""
Pre-Train Multiple Policy Architectures Using Behavioral Cloning

This script trains multiple policy architectures on human demonstration data,
supporting various observation configurations and feature extractors.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch

from npp_rl.training.bc_trainer import BCTrainer
from npp_rl.data.bc_dataset import create_bc_dataloader
from nclone.gym_environment.npp_environment import NppEnvironment
from torch.utils.tensorboard import SummaryWriter


def get_architecture_configs() -> Dict[str, Dict]:
    """Define architecture configurations to pre-train.
    
    Returns:
        Dictionary mapping architecture name to configuration
    """
    return {
        "hgt_multimodal": {
            "policy": "npp",
            "extractor_type": "hgt",
            "observation_config": {
                "enable_graph_updates": True,
                "enable_reachability": True,
                "enable_hierarchical": False,
            },
            "feature_dim": 512,
            "description": "HGT-based multimodal with graph and reachability",
        },
        "hierarchical_multimodal": {
            "policy": "npp",
            "extractor_type": "hierarchical",
            "observation_config": {
                "enable_graph_updates": False,
                "enable_reachability": True,
                "enable_hierarchical": True,
            },
            "feature_dim": 512,
            "description": "Hierarchical multimodal with reachability",
        },
        "simple_cnn": {
            "policy": "simple",
            "extractor_type": "cnn",
            "observation_config": {
                "enable_graph_updates": False,
                "enable_reachability": False,
                "enable_hierarchical": False,
            },
            "feature_dim": 256,
            "description": "Simple CNN baseline",
        },
    }


def pretrain_architecture(
    arch_name: str,
    arch_config: Dict,
    dataset_dir: Path,
    output_dir: Path,
    args,
) -> Dict:
    """Pre-train a single architecture configuration.
    
    Args:
        arch_name: Architecture identifier
        arch_config: Architecture configuration
        dataset_dir: Directory with BC training data
        output_dir: Output directory for checkpoints
        args: Command-line arguments
    
    Returns:
        Training results dictionary
    """
    print(f"\n{'=' * 70}")
    print(f"PRE-TRAINING: {arch_name}")
    print(f"{'=' * 70}")
    print(f"Description: {arch_config['description']}")
    print(f"Policy: {arch_config['policy']}")
    print(f"Extractor: {arch_config['extractor_type']}")
    print(f"{'=' * 70}\n")
    
    # Create output directory for this architecture
    arch_output_dir = output_dir / arch_name
    arch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment to get observation space
    env = NppEnvironment(
        render_mode="rgb_array",
        enable_graph_updates=arch_config["observation_config"]["enable_graph_updates"],
        enable_reachability=arch_config["observation_config"]["enable_reachability"],
        enable_hierarchical=arch_config["observation_config"]["enable_hierarchical"],
    )
    
    observation_space = env.observation_space
    action_space = env.action_space
    
    # Create dataloader
    train_dataloader = create_bc_dataloader(
        dataset_dir=str(dataset_dir),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # Create BC trainer
    trainer = BCTrainer(
        observation_space=observation_space,
        action_space=action_space,
        policy_class=arch_config["policy"],
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef,
        freeze_backbone_steps=args.freeze_backbone_steps,
        device=args.device,
    )
    
    # Setup logging
    log_dir = arch_output_dir / "logs"
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_dataloader, writer)
        training_history.append(train_metrics)
        
        print(
            f"  Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy']:.3f}"
        )
        
        # Save checkpoint
        is_best = train_metrics["accuracy"] > best_accuracy
        if is_best:
            best_accuracy = train_metrics["accuracy"]
        
        checkpoint_path = arch_output_dir / f"checkpoint_epoch_{epoch}.pth"
        trainer.save_checkpoint(str(checkpoint_path), is_best=is_best)
        
        # Log to TensorBoard
        writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
        writer.add_scalar("epoch/train_accuracy", train_metrics["accuracy"], epoch)
    
    # Export final model for Stable-Baselines3
    export_path = arch_output_dir / "bc_policy.pth"
    trainer.export_for_sb3(str(export_path))
    
    # Save training configuration
    config_path = arch_output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "architecture": arch_name,
            "config": arch_config,
            "training_args": vars(args),
            "final_accuracy": best_accuracy,
        }, f, indent=2)
    
    writer.close()
    env.close()
    
    print(f"\n✅ Pre-training complete for {arch_name}")
    print(f"   Best accuracy: {best_accuracy:.3f}")
    print(f"   Model saved to: {export_path}")
    
    return {
        "architecture": arch_name,
        "best_accuracy": best_accuracy,
        "final_loss": training_history[-1]["loss"],
        "model_path": str(export_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train multiple policy architectures using behavioral cloning"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default="datasets/bc_training",
        help="Directory containing processed BC training data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="pretrained_policies",
        help="Output directory for pre-trained models",
    )
    
    # Architecture selection
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["hgt_multimodal", "hierarchical_multimodal", "simple_cnn"],
        help="Architectures to pre-train (space-separated)",
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Entropy regularization"
    )
    parser.add_argument(
        "--freeze_backbone_steps",
        type=int,
        default=0,
        help="Steps to freeze feature extractor",
    )
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto", help="Training device")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers"
    )
    
    args = parser.parse_args()
    
    # Validate dataset directory
    if not args.dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {args.dataset_dir}")
    
    # Get architecture configurations
    all_configs = get_architecture_configs()
    
    # Filter to requested architectures
    configs_to_train = {
        name: config for name, config in all_configs.items()
        if name in args.architectures
    }
    
    if not configs_to_train:
        raise ValueError(
            f"No valid architectures selected. Available: {list(all_configs.keys())}"
        )
    
    print(f"\n{'=' * 70}")
    print("BEHAVIORAL CLONING PRE-TRAINING")
    print(f"{'=' * 70}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Architectures: {list(configs_to_train.keys())}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'=' * 70}\n")
    
    # Pre-train each architecture
    results = []
    for arch_name, arch_config in configs_to_train.items():
        try:
            result = pretrain_architecture(
                arch_name, arch_config, args.dataset_dir, args.output_dir, args
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error pre-training {arch_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_path = args.output_dir / "pretraining_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("PRE-TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for result in results:
        print(f"{result['architecture']:25s} - Accuracy: {result['best_accuracy']:.3f}")
    print(f"{'=' * 70}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Pre-train all architectures
cd /home/tetra/projects/npp-rl
python scripts/pretrain_policies.py \
    --dataset_dir datasets/bc_training \
    --output_dir pretrained_policies \
    --epochs 20 \
    --batch_size 64 \
    --lr 3e-4

# Pre-train specific architectures only
python scripts/pretrain_policies.py \
    --architectures hgt_multimodal simple_cnn \
    --epochs 15

# Monitor training with TensorBoard
tensorboard --logdir pretrained_policies/
```

---

## Phase 3: Fine-Tuning and Evaluation

### 3.1 Fine-Tuning Script

Create `npp-rl/scripts/finetune_pretrained.py`:

```python
#!/usr/bin/env python3
"""
Fine-Tune Pre-Trained Policies with RL

Loads pre-trained policies from behavioral cloning and fine-tunes them using
PPO on the training dataset, then evaluates on the test dataset.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.evaluation.test_suite_loader import TestSuiteLoader
from npp_rl.feature_extractors import HGTMultimodalExtractor


def load_pretrained_policy(
    pretrained_path: Path,
    env,
    architecture: str,
) -> PPO:
    """Load a pre-trained policy and create PPO model.
    
    Args:
        pretrained_path: Path to pre-trained policy checkpoint
        env: Training environment
        architecture: Architecture name
    
    Returns:
        PPO model with pre-trained weights
    """
    print(f"Loading pre-trained policy from: {pretrained_path}")
    
    # Load pre-trained weights
    pretrained_state = torch.load(pretrained_path)
    
    # Create PPO model with matching architecture
    # Policy kwargs depend on architecture
    if architecture == "hgt_multimodal":
        policy_kwargs = {
            "features_extractor_class": HGTMultimodalExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [256, 256, 128],
        }
    elif architecture == "hierarchical_multimodal":
        policy_kwargs = {
            # TODO: Add hierarchical extractor
            "net_arch": [256, 256, 128],
        }
    else:  # simple_cnn
        policy_kwargs = {
            "net_arch": [256, 128],
        }
    
    # Create model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        verbose=1,
    )
    
    # Load pre-trained weights into policy
    model.policy.load_state_dict(pretrained_state, strict=False)
    print("✅ Pre-trained weights loaded successfully")
    
    return model


def create_train_env(level_ids: list, num_envs: int):
    """Create vectorized training environment with test suite levels."""
    
    def make_env(level_id):
        def _init():
            # Load level from test suite
            loader = TestSuiteLoader("/home/tetra/projects/nclone/datasets/train")
            level = loader.get_level(level_id)
            
            # Create environment
            env = NppEnvironment(render_mode="rgb_array")
            
            # Load level
            env.unwrapped.nplay_headless.load_map_from_map_data(level["map_data"])
            
            return env
        return _init
    
    # Create vectorized environment
    envs = [make_env(level_ids[i % len(level_ids)]) for i in range(num_envs)]
    vec_env = SubprocVecEnv(envs)
    vec_env = VecMonitor(vec_env)
    
    return vec_env


def finetune_model(
    model: PPO,
    train_env,
    eval_env,
    output_dir: Path,
    total_timesteps: int,
) -> Dict:
    """Fine-tune model with PPO.
    
    Args:
        model: Pre-trained PPO model
        train_env: Training environment
        eval_env: Evaluation environment
        output_dir: Output directory for checkpoints
        total_timesteps: Total training timesteps
    
    Returns:
        Training results dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = configure(str(output_dir / "logs"), ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_checkpoint",
    )
    
    # Fine-tune
    print(f"\nFine-tuning for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )
    
    # Save final model
    final_path = output_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"✅ Fine-tuned model saved to: {final_path}")
    
    return {
        "total_timesteps": total_timesteps,
        "final_model_path": str(final_path),
        "best_model_path": str(output_dir / "best_model" / "best_model.zip"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pre-trained policies with RL"
    )
    
    parser.add_argument(
        "--pretrained_model",
        type=Path,
        required=True,
        help="Path to pre-trained BC policy (.pth)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["hgt_multimodal", "hierarchical_multimodal", "simple_cnn"],
        help="Architecture type",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="finetuned_models",
        help="Output directory for fine-tuned models",
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        default="/home/tetra/projects/nclone/datasets/train",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=32,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5000000,
        help="Total training timesteps",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pretrained_model.exists():
        raise ValueError(f"Pre-trained model not found: {args.pretrained_model}")
    if not args.train_dataset.exists():
        raise ValueError(f"Training dataset not found: {args.train_dataset}")
    
    # Load training dataset
    train_loader = TestSuiteLoader(str(args.train_dataset))
    train_level_ids = train_loader.get_level_ids()
    
    print(f"\n{'=' * 70}")
    print("FINE-TUNING PRE-TRAINED POLICY")
    print(f"{'=' * 70}")
    print(f"Pre-trained model: {args.pretrained_model}")
    print(f"Architecture: {args.architecture}")
    print(f"Training levels: {len(train_level_ids)}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"{'=' * 70}\n")
    
    # Create environments
    train_env = create_train_env(train_level_ids, args.num_envs)
    eval_env = create_train_env(train_level_ids[:10], 1)  # Use subset for eval
    
    # Load pre-trained model
    model = load_pretrained_policy(
        args.pretrained_model,
        train_env,
        args.architecture,
    )
    
    # Fine-tune
    results = finetune_model(
        model,
        train_env,
        eval_env,
        args.output_dir / args.architecture,
        args.total_timesteps,
    )
    
    # Save results
    results_path = args.output_dir / args.architecture / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"Results saved to: {results_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
```

### 3.2 Evaluation Script

Create `npp-rl/scripts/evaluate_on_test_suite.py`:

```python
#!/usr/bin/env python3
"""
Evaluate Trained Models on Test Suite

Evaluates fine-tuned models on the test dataset and generates comprehensive
performance metrics.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.evaluation.test_suite_loader import TestSuiteLoader


def evaluate_model_on_level(
    model: PPO,
    level_data: Dict,
    max_steps: int = 1800,
    num_episodes: int = 5,
) -> Dict:
    """Evaluate model on a single level.
    
    Args:
        model: Trained model
        level_data: Level data dictionary
        max_steps: Maximum steps per episode
        num_episodes: Number of evaluation episodes
    
    Returns:
        Evaluation metrics dictionary
    """
    env = NppEnvironment(render_mode="rgb_array")
    env.unwrapped.nplay_headless.load_map_from_map_data(level_data["map_data"])
    
    successes = []
    episode_lengths = []
    episode_rewards = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        success = info.get("is_success", False)
        successes.append(success)
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
    
    env.close()
    
    return {
        "success_rate": np.mean(successes),
        "avg_episode_length": np.mean(episode_lengths),
        "avg_reward": np.mean(episode_rewards),
        "num_episodes": num_episodes,
    }


def evaluate_model_on_dataset(
    model_path: Path,
    test_dataset_path: Path,
    output_path: Path,
    num_episodes_per_level: int = 5,
) -> Dict:
    """Evaluate model on entire test dataset.
    
    Args:
        model_path: Path to trained model
        test_dataset_path: Path to test dataset
        output_path: Path to save results
        num_episodes_per_level: Episodes per level
    
    Returns:
        Comprehensive evaluation results
    """
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(str(model_path))
    
    print(f"Loading test dataset from: {test_dataset_path}")
    test_loader = TestSuiteLoader(str(test_dataset_path))
    
    # Evaluate on each category
    results_by_category = {}
    all_results = []
    
    for category in test_loader.CATEGORIES:
        print(f"\nEvaluating category: {category}")
        levels = test_loader.get_category(category)
        
        category_results = []
        for level in tqdm(levels, desc=f"Evaluating {category}"):
            level_result = evaluate_model_on_level(
                model, level, num_episodes=num_episodes_per_level
            )
            level_result["level_id"] = level["level_id"]
            level_result["category"] = category
            category_results.append(level_result)
            all_results.append(level_result)
        
        # Compute category statistics
        results_by_category[category] = {
            "levels": category_results,
            "avg_success_rate": np.mean([r["success_rate"] for r in category_results]),
            "avg_episode_length": np.mean([r["avg_episode_length"] for r in category_results]),
            "avg_reward": np.mean([r["avg_reward"] for r in category_results]),
            "num_levels": len(category_results),
        }
        
        print(f"  Category {category} - Success Rate: {results_by_category[category]['avg_success_rate']:.3f}")
    
    # Compute overall statistics
    overall_stats = {
        "overall_success_rate": np.mean([r["success_rate"] for r in all_results]),
        "overall_avg_episode_length": np.mean([r["avg_episode_length"] for r in all_results]),
        "overall_avg_reward": np.mean([r["avg_reward"] for r in all_results]),
        "total_levels_evaluated": len(all_results),
        "episodes_per_level": num_episodes_per_level,
    }
    
    # Combine results
    final_results = {
        "model_path": str(model_path),
        "test_dataset_path": str(test_dataset_path),
        "overall": overall_stats,
        "by_category": results_by_category,
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Overall Success Rate: {overall_stats['overall_success_rate']:.3f}")
    print(f"Results saved to: {output_path}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on test suite"
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--test_dataset",
        type=Path,
        default="/home/tetra/projects/nclone/datasets/test",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="evaluation_results/results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--episodes_per_level",
        type=int,
        default=5,
        help="Number of episodes per level",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model.exists():
        raise ValueError(f"Model not found: {args.model}")
    if not args.test_dataset.exists():
        raise ValueError(f"Test dataset not found: {args.test_dataset}")
    
    # Evaluate
    evaluate_model_on_dataset(
        args.model,
        args.test_dataset,
        args.output,
        args.episodes_per_level,
    )


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Fine-tune pre-trained model
cd /home/tetra/projects/npp-rl
python scripts/finetune_pretrained.py \
    --pretrained_model pretrained_policies/hgt_multimodal/bc_policy.pth \
    --architecture hgt_multimodal \
    --output_dir finetuned_models \
    --total_timesteps 5000000 \
    --num_envs 32

# Evaluate fine-tuned model on test suite
python scripts/evaluate_on_test_suite.py \
    --model finetuned_models/hgt_multimodal/best_model/best_model.zip \
    --test_dataset /home/tetra/projects/nclone/datasets/test \
    --output evaluation_results/hgt_multimodal_results.json \
    --episodes_per_level 10
```

---

## Phase 4: Baseline Comparison

### 4.1 Training Baseline Models

Create `npp-rl/scripts/train_baseline.py`:

```python
#!/usr/bin/env python3
"""
Train Baseline Models (No Pre-Training)

Trains models from scratch without pre-training for comparison.
"""

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.evaluation.test_suite_loader import TestSuiteLoader
from npp_rl.feature_extractors import HGTMultimodalExtractor


def create_train_env(level_ids: list, num_envs: int):
    """Create vectorized training environment."""
    
    def make_env(level_id):
        def _init():
            loader = TestSuiteLoader("/home/tetra/projects/nclone/datasets/train")
            level = loader.get_level(level_id)
            env = NppEnvironment(render_mode="rgb_array")
            env.unwrapped.nplay_headless.load_map_from_map_data(level["map_data"])
            return env
        return _init
    
    envs = [make_env(level_ids[i % len(level_ids)]) for i in range(num_envs)]
    vec_env = SubprocVecEnv(envs)
    vec_env = VecMonitor(vec_env)
    return vec_env


def train_baseline(
    architecture: str,
    output_dir: Path,
    num_envs: int,
    total_timesteps: int,
):
    """Train baseline model from scratch.
    
    Args:
        architecture: Architecture name
        output_dir: Output directory
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
    """
    # Create output directory
    output_dir = output_dir / f"baseline_{architecture}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training dataset
    train_loader = TestSuiteLoader("/home/tetra/projects/nclone/datasets/train")
    train_level_ids = train_loader.get_level_ids()
    
    # Create environments
    train_env = create_train_env(train_level_ids, num_envs)
    eval_env = create_train_env(train_level_ids[:10], 1)
    
    # Configure policy based on architecture
    if architecture == "hgt_multimodal":
        policy_kwargs = {
            "features_extractor_class": HGTMultimodalExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [256, 256, 128],
        }
    elif architecture == "hierarchical_multimodal":
        policy_kwargs = {
            "net_arch": [256, 256, 128],
        }
    else:  # simple_cnn
        policy_kwargs = {
            "net_arch": [256, 128],
        }
    
    # Create model
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        verbose=1,
    )
    
    # Setup logger
    logger = configure(str(output_dir / "logs"), ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_checkpoint",
    )
    
    # Train
    print(f"\nTraining baseline {architecture} for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )
    
    # Save final model
    final_path = output_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"✅ Baseline model saved to: {final_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["hgt_multimodal", "hierarchical_multimodal", "simple_cnn"],
        help="Architecture to train",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="baseline_models",
        help="Output directory",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=32,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5000000,
        help="Total training timesteps",
    )
    
    args = parser.parse_args()
    
    train_baseline(
        args.architecture,
        args.output_dir,
        args.num_envs,
        args.total_timesteps,
    )


if __name__ == "__main__":
    main()
```

### 4.2 Comparison Script

Create `npp-rl/scripts/compare_pretrained_vs_baseline.py`:

```python
#!/usr/bin/env python3
"""
Compare Pre-Trained vs Baseline Models

Generates comprehensive comparison between models trained with and without
pre-training from human demonstrations.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def load_evaluation_results(results_path: Path) -> Dict:
    """Load evaluation results from JSON."""
    with open(results_path, "r") as f:
        return json.load(f)


def compare_models(pretrained_results: Dict, baseline_results: Dict) -> Dict:
    """Compare pretrained and baseline model results.
    
    Args:
        pretrained_results: Evaluation results for pre-trained model
        baseline_results: Evaluation results for baseline model
    
    Returns:
        Comparison statistics
    """
    comparison = {
        "overall": {},
        "by_category": {},
    }
    
    # Overall comparison
    pretrained_sr = pretrained_results["overall"]["overall_success_rate"]
    baseline_sr = baseline_results["overall"]["overall_success_rate"]
    improvement = ((pretrained_sr - baseline_sr) / baseline_sr * 100) if baseline_sr > 0 else 0.0
    
    comparison["overall"] = {
        "pretrained_success_rate": pretrained_sr,
        "baseline_success_rate": baseline_sr,
        "improvement_percent": improvement,
        "pretrained_avg_reward": pretrained_results["overall"]["overall_avg_reward"],
        "baseline_avg_reward": baseline_results["overall"]["overall_avg_reward"],
    }
    
    # Category-wise comparison
    for category in pretrained_results["by_category"].keys():
        pretrained_cat = pretrained_results["by_category"][category]
        baseline_cat = baseline_results["by_category"][category]
        
        cat_pretrained_sr = pretrained_cat["avg_success_rate"]
        cat_baseline_sr = baseline_cat["avg_success_rate"]
        cat_improvement = ((cat_pretrained_sr - cat_baseline_sr) / cat_baseline_sr * 100) if cat_baseline_sr > 0 else 0.0
        
        comparison["by_category"][category] = {
            "pretrained_success_rate": cat_pretrained_sr,
            "baseline_success_rate": cat_baseline_sr,
            "improvement_percent": cat_improvement,
        }
    
    return comparison


def plot_comparison(comparison: Dict, output_path: Path):
    """Generate comparison plots.
    
    Args:
        comparison: Comparison statistics
        output_path: Path to save plot
    """
    categories = list(comparison["by_category"].keys())
    pretrained_rates = [comparison["by_category"][c]["pretrained_success_rate"] for c in categories]
    baseline_rates = [comparison["by_category"][c]["baseline_success_rate"] for c in categories]
    
    # Create bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, pretrained_rates, width, label="Pre-Trained", color="green", alpha=0.7)
    bars2 = ax.bar(x + width/2, baseline_rates, width, label="Baseline", color="blue", alpha=0.7)
    
    ax.set_xlabel("Category")
    ax.set_ylabel("Success Rate")
    ax.set_title("Pre-Trained vs Baseline: Success Rate by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare pre-trained and baseline models"
    )
    
    parser.add_argument(
        "--pretrained_results",
        type=Path,
        required=True,
        help="Path to pre-trained model evaluation results",
    )
    parser.add_argument(
        "--baseline_results",
        type=Path,
        required=True,
        help="Path to baseline model evaluation results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="comparison_results",
        help="Output directory for comparison results",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pretrained_results.exists():
        raise ValueError(f"Pre-trained results not found: {args.pretrained_results}")
    if not args.baseline_results.exists():
        raise ValueError(f"Baseline results not found: {args.baseline_results}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading evaluation results...")
    pretrained_results = load_evaluation_results(args.pretrained_results)
    baseline_results = load_evaluation_results(args.baseline_results)
    
    # Compare
    print("Comparing models...")
    comparison = compare_models(pretrained_results, baseline_results)
    
    # Save comparison
    comparison_path = args.output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Generate plot
    plot_path = args.output_dir / "comparison_plot.png"
    plot_comparison(comparison, plot_path)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"Overall Success Rate:")
    print(f"  Pre-Trained: {comparison['overall']['pretrained_success_rate']:.3f}")
    print(f"  Baseline:    {comparison['overall']['baseline_success_rate']:.3f}")
    print(f"  Improvement: {comparison['overall']['improvement_percent']:+.1f}%")
    print(f"\nBy Category:")
    for category, stats in comparison["by_category"].items():
        print(f"  {category:15s}: {stats['improvement_percent']:+.1f}%")
    print(f"{'=' * 70}")
    print(f"\nDetailed results saved to: {comparison_path}")


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Train baseline model
cd /home/tetra/projects/npp-rl
python scripts/train_baseline.py \
    --architecture hgt_multimodal \
    --output_dir baseline_models \
    --total_timesteps 5000000 \
    --num_envs 32

# Evaluate baseline model
python scripts/evaluate_on_test_suite.py \
    --model baseline_models/baseline_hgt_multimodal/best_model/best_model.zip \
    --test_dataset /home/tetra/projects/nclone/datasets/test \
    --output evaluation_results/baseline_hgt_multimodal_results.json

# Compare pre-trained vs baseline
python scripts/compare_pretrained_vs_baseline.py \
    --pretrained_results evaluation_results/hgt_multimodal_results.json \
    --baseline_results evaluation_results/baseline_hgt_multimodal_results.json \
    --output_dir comparison_results
```

---

## File Structure and Organization

### Expected Directory Layout

```
npp-rl/
├── datasets/
│   ├── human_replays/           # Recorded gameplay demonstrations
│   │   ├── 20241207_153045_level_001/
│   │   │   ├── replay.jsonl
│   │   │   ├── episode.pkl
│   │   │   ├── map_data.bin
│   │   │   └── metadata.json
│   │   └── ...
│   └── bc_training/             # Processed BC training data
│       ├── batch_0000.npz
│       ├── batch_0001.npz
│       └── ...
├── pretrained_policies/         # Pre-trained BC models
│   ├── hgt_multimodal/
│   │   ├── bc_policy.pth
│   │   ├── config.json
│   │   └── logs/
│   ├── hierarchical_multimodal/
│   └── simple_cnn/
├── finetuned_models/            # Fine-tuned RL models
│   ├── hgt_multimodal/
│   │   ├── best_model/
│   │   ├── final_model.zip
│   │   ├── results.json
│   │   └── logs/
│   └── ...
├── baseline_models/             # Baseline models (no pre-training)
│   ├── baseline_hgt_multimodal/
│   │   ├── best_model/
│   │   ├── final_model.zip
│   │   └── logs/
│   └── ...
├── evaluation_results/          # Evaluation results on test suite
│   ├── hgt_multimodal_results.json
│   ├── baseline_hgt_multimodal_results.json
│   └── ...
├── comparison_results/          # Comparison analysis
│   ├── comparison.json
│   └── comparison_plot.png
└── scripts/                     # Utility scripts
    ├── pretrain_policies.py
    ├── finetune_pretrained.py
    ├── train_baseline.py
    ├── evaluate_on_test_suite.py
    └── compare_pretrained_vs_baseline.py

nclone/
├── datasets/
│   ├── train/                   # Training levels (250 levels)
│   │   ├── simple/
│   │   ├── medium/
│   │   ├── complex/
│   │   ├── mine_heavy/
│   │   ├── exploration/
│   │   └── metadata.json
│   └── test/                    # Test levels (250 levels)
│       ├── simple/
│       ├── medium/
│       ├── complex/
│       ├── mine_heavy/
│       ├── exploration/
│       └── metadata.json
└── nclone/
    └── replay/
        └── gameplay_recorder.py  # NEW: Gameplay recording system
```

---

## Technical References

### Key Concepts

**Behavioral Cloning (BC)**:
- Supervised learning approach to imitation learning
- Trains policy to mimic expert demonstrations
- Provides good initialization for RL fine-tuning
- Can suffer from distributional shift without RL fine-tuning

**Transfer Learning Benefits**:
- Faster convergence during RL training
- Better sample efficiency
- Improved initial policy quality
- Reduced exploration required

**Evaluation Metrics**:
- **Success Rate**: Percentage of levels completed successfully
- **Episode Length**: Average steps to completion
- **Episode Reward**: Average cumulative reward
- **Category Performance**: Success rates broken down by difficulty

### Related Research

1. **Behavioral Cloning for RL**:
   - "Learning from Demonstrations for Real World Reinforcement Learning" (Hester et al., 2018)
   - "Deep Q-Learning from Demonstrations" (Hester et al., 2017)

2. **Imitation Learning**:
   - "A Survey of Imitation Learning" (Hussein et al., 2017)
   - "Dataset Aggregation (DAgger)" (Ross et al., 2011)

3. **Transfer Learning in RL**:
   - "Transfer Learning for Reinforcement Learning Domains: A Survey" (Taylor & Stone, 2009)
   - "Progressive Neural Networks" (Rusu et al., 2016)

### Best Practices

1. **Data Quality**:
   - Record diverse demonstrations across difficulty levels
   - Ensure demonstrations are high-quality (successful completions)
   - Balance dataset across level categories
   - Filter out low-quality demonstrations using quality scores

2. **Pre-Training**:
   - Use entropy regularization to encourage exploration
   - Monitor overfitting through validation metrics
   - Consider data augmentation (minor perturbations)
   - Freeze feature extractors initially if needed

3. **Fine-Tuning**:
   - Start with lower learning rates than training from scratch
   - Monitor for catastrophic forgetting of BC behavior
   - Use curriculum learning (easy to hard levels)
   - Evaluate frequently on held-out validation set

4. **Evaluation**:
   - Test on diverse level categories
   - Use multiple episodes per level for statistical significance
   - Compare against meaningful baselines
   - Analyze failure cases to identify weaknesses

---

## Summary Checklist

### Phase 1: Recording
- [ ] Implement `GameplayRecorder` class in `nclone/nclone/replay/gameplay_recorder.py`
- [ ] Integrate recording into `test_environment.py`
- [ ] Record 50+ successful demonstrations across diverse levels
- [ ] Verify recorded data quality and format

### Phase 2: Pre-Training
- [ ] Process recordings with `replay_ingest.py`
- [ ] Create `pretrain_policies.py` script
- [ ] Pre-train HGT, hierarchical, and simple CNN architectures
- [ ] Validate pre-trained model checkpoints

### Phase 3: Fine-Tuning
- [ ] Create `finetune_pretrained.py` script
- [ ] Create `evaluate_on_test_suite.py` script
- [ ] Fine-tune all pre-trained models on train dataset
- [ ] Evaluate fine-tuned models on test dataset

### Phase 4: Baseline Comparison
- [ ] Create `train_baseline.py` script
- [ ] Create `compare_pretrained_vs_baseline.py` script
- [ ] Train baseline models (no pre-training)
- [ ] Evaluate baseline models on test dataset
- [ ] Generate comparison analysis and visualizations

---

## Additional Notes

**Performance Expectations**:
- BC pre-training typically achieves 60-80% accuracy on demonstrations
- Fine-tuned models should show 10-30% improvement over baselines
- Improvement magnitude depends on demonstration quality and quantity

**Computational Requirements**:
- Recording: Minimal overhead (<5% performance impact)
- BC pre-training: 2-4 hours on GPU for 20 epochs
- RL fine-tuning: 8-12 hours on GPU for 5M timesteps (32 parallel envs)
- Evaluation: 2-3 hours for 250 levels × 10 episodes

**Troubleshooting**:
- If BC accuracy is low (<40%), check demonstration quality
- If fine-tuning doesn't improve over baseline, reduce learning rate
- If catastrophic forgetting occurs, use lower learning rate and more gradual fine-tuning
- If evaluation success rates are inconsistent, increase episodes per level

---

---

## Critical Implementation Note: Deterministic Replay Benefits

This pipeline leverages the **completely deterministic** nature of the N++ physics simulation as a core design principle:

### Storage Efficiency
| Component | Traditional Approach | Deterministic Replay | Reduction |
|-----------|---------------------|----------------------|-----------|
| Per episode | ~500 KB (observations) | ~2.5 KB (map + inputs) | **200x smaller** |
| 100 episodes | ~50 MB | ~250 KB | **200x smaller** |
| 1000 episodes | ~500 MB | ~2.5 MB | **200x smaller** |

### Flexibility Benefits

**Problem with traditional approach:**
- Changing observation format requires re-recording ALL demonstrations
- Different architectures need separate demonstration datasets
- No way to fix observation bugs in existing data

**Solution with deterministic replay:**
- Single replay file works with ANY observation configuration
- HGT, hierarchical, and simple CNN architectures use the SAME replay files
- Can fix observation processor bugs and regenerate data instantly
- Can add new observation features retroactively

### Implementation Checklist

**Phase 1 - Recording:**
- [ ] Implement `CompactReplay` dataclass with binary serialization
- [ ] Implement `GameplayRecorder` with `record_action()` method
- [ ] Verify replay files are ~1-5KB each (NOT hundreds of KB)
- [ ] Test: Load replay, verify map_data + input_sequence only

**Phase 2 - Replay Execution:**
- [ ] Implement `ReplayExecutor` class
- [ ] Implement `decode_input_to_controls()` for input → controls mapping
- [ ] Test: Same replay file generates identical observations on multiple runs
- [ ] Test: Different observation configs work with same replay file

**Phase 3 - Observation Generation:**
- [ ] Implement `generate_bc_data.py` script
- [ ] Verify determinism: Run same replay twice, compare observations (should be identical)
- [ ] Measure performance: Should process ~10-50 replays/second
- [ ] Test different architecture configs use same replay inputs

**Phase 4 - Integration:**
- [ ] Update test_environment.py with compact recording
- [ ] Verify: `recorder.record_action(action)` is the ONLY recording call
- [ ] Test: Successfully record and replay a complete episode
- [ ] Verify: File size is ~1-5KB, not 500KB+

### Verification Tests

```bash
# Test 1: Verify determinism
python tools/generate_bc_data.py --input replays/ --output bc_data_v1/
python tools/generate_bc_data.py --input replays/ --output bc_data_v2/
diff -r bc_data_v1/ bc_data_v2/  # Should be identical

# Test 2: Verify observation format flexibility
# Change ObservationProcessor settings
python tools/generate_bc_data.py --input replays/ --output bc_data_new_format/
# Same replays, different observations - no re-recording needed!

# Test 3: Verify file sizes
du -sh datasets/human_replays/  # Should be ~KB range
du -sh datasets/bc_training/     # Should be ~MB range (100x+ larger)
```

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Ready for Implementation  
**Critical Design Principle**: Deterministic replay with compact storage

