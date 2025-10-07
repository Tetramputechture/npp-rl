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
        "--batch-size",
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
