#!/usr/bin/env python3
"""Profile replay ingestion for path prediction to identify bottlenecks."""

import cProfile
import pstats
import io
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
nclone_root = project_root.parent / "nclone"
sys.path.insert(0, str(nclone_root))

from npp_rl.data.replay_dataset import PathReplayDataset


def profile_replay_loading():
    """Profile replay dataset loading with cProfile."""

    replay_dir = "/home/tetra/projects/nclone/datasets/path-replays"

    print("=" * 80)
    print("PROFILING REPLAY DATASET LOADING")
    print("=" * 80)
    print(f"Replay directory: {replay_dir}")
    print("Max replays: 1 (for quick profiling)")
    print()

    # Create profiler
    profiler = cProfile.Profile()

    # Profile dataset creation (which processes all replays)
    print("Starting profiling...")
    profiler.enable()

    start_time = time.time()
    dataset = PathReplayDataset(
        replay_dir=replay_dir,
        waypoint_interval=5,
        min_trajectory_length=20,
        enable_rendering=False,
        max_replays=1,
    )
    end_time = time.time()

    profiler.disable()

    elapsed = end_time - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
    print(f"Samples loaded: {len(dataset)}")

    # Print profiling results
    print("\n" + "=" * 80)
    print("TOP 50 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(50)
    print(s.getvalue())

    print("\n" + "=" * 80)
    print("TOP 50 FUNCTIONS BY INTERNAL TIME")
    print("=" * 80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("time")
    ps.print_stats(50)
    print(s.getvalue())

    # Save profile data to file
    profile_file = "/tmp/replay_ingestion_profile.prof"
    profiler.dump_stats(profile_file)
    print(f"\nProfile data saved to: {profile_file}")
    print("You can visualize with: snakeviz", profile_file)


if __name__ == "__main__":
    profile_replay_loading()
