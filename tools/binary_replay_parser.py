#!/usr/bin/env python3
"""
N++ Binary Replay Parser

This tool converts N++ binary replay files ("trace" mode) to JSONL format
compatible with the npp-rl training pipeline. It parses the original N++
replay format and simulates the level to extract frame-by-frame data.

Usage:
    python tools/binary_replay_parser.py --input replays/ --output datasets/raw/
    python tools/binary_replay_parser.py --help
"""

import argparse
import json
import logging
import os
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import hashlib

# Add nclone to path for imports
import sys
project_root = Path(__file__).parent.parent
nclone_path = project_root.parent / "nclone"
sys.path.insert(0, str(nclone_path))

from nclone.nsim import Simulator
from nclone.sim_config import SimConfig

logger = logging.getLogger(__name__)


class BinaryReplayParser:
    """Parser for N++ binary replay files to JSONL format."""
    
    # Input encoding dictionaries (from ntrace.py)
    HOR_INPUTS_DIC = {0: 0, 1: 0, 2: 1, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1}
    JUMP_INPUTS_DIC = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}
    
    # Required file names for trace mode
    RAW_INPUTS = ["inputs_0", "inputs_1", "inputs_2", "inputs_3"]
    RAW_MAP_DATA = "map_data"
    
    # Fixed level dimensions (as specified by user)
    LEVEL_WIDTH = 1056
    LEVEL_HEIGHT = 600
    
    def __init__(self):
        """Initialize the parser."""
        self.stats = {
            'files_processed': 0,
            'frames_generated': 0,
            'replays_processed': 0,
            'replays_failed': 0
        }
    
    def detect_trace_mode(self, replay_dir: Path) -> bool:
        """
        Check if directory contains trace mode files.
        
        Args:
            replay_dir: Directory to check
            
        Returns:
            True if trace mode files are present
        """
        map_data_exists = (replay_dir / self.RAW_MAP_DATA).exists()
        inputs_exist = any((replay_dir / inp).exists() for inp in self.RAW_INPUTS)
        
        return map_data_exists and inputs_exist
    
    def load_inputs_and_map(self, replay_dir: Path) -> Tuple[List[List[int]], List[int]]:
        """
        Load input sequences and map data from binary files.
        
        Args:
            replay_dir: Directory containing replay files
            
        Returns:
            Tuple of (inputs_list, map_data)
        """
        inputs_list = []
        
        # Load input files
        for input_file in self.RAW_INPUTS:
            input_path = replay_dir / input_file
            if input_path.exists():
                with open(input_path, "rb") as f:
                    # Decompress and convert to integers
                    raw_data = zlib.decompress(f.read())
                    inputs = [int(b) for b in raw_data]
                    inputs_list.append(inputs)
            else:
                break
        
        # Load map data
        map_path = replay_dir / self.RAW_MAP_DATA
        with open(map_path, "rb") as f:
            map_data = [int(b) for b in f.read()]
        
        logger.info(f"Loaded {len(inputs_list)} input sequences and map data")
        return inputs_list, map_data
    
    def decode_inputs(self, raw_inputs: List[int]) -> Tuple[List[int], List[int]]:
        """
        Decode raw input values to horizontal and jump components.
        
        Args:
            raw_inputs: List of raw input values (0-7)
            
        Returns:
            Tuple of (horizontal_inputs, jump_inputs)
        """
        hor_inputs = [self.HOR_INPUTS_DIC[inp] for inp in raw_inputs]
        jump_inputs = [self.JUMP_INPUTS_DIC[inp] for inp in raw_inputs]
        return hor_inputs, jump_inputs
    
    def get_entity_data(self, sim: Simulator) -> List[Dict[str, Any]]:
        """
        Extract entity information from simulator.
        
        Args:
            sim: Simulator instance
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        # Iterate through all entity types
        for entity_type, entity_list in sim.entity_dic.items():
            for entity in entity_list:
                entity_data = {
                    "type": self._get_entity_type_name(entity_type),
                    "position": {
                        "x": float(entity.xpos),
                        "y": float(entity.ypos)
                    },
                    "active": True  # Default to active, could be refined based on entity state
                }
                
                # Add type-specific data
                if hasattr(entity, 'state'):
                    entity_data["state"] = entity.state
                if hasattr(entity, 'radius'):
                    entity_data["radius"] = entity.radius
                
                entities.append(entity_data)
        
        return entities
    
    def _get_entity_type_name(self, entity_type: int) -> str:
        """
        Convert entity type ID to name.
        
        Args:
            entity_type: Numeric entity type
            
        Returns:
            Entity type name string
        """
        # Entity type mapping based on N++ documentation
        type_mapping = {
            1: "mine",
            2: "gold",
            3: "exit_door",
            4: "exit_switch",
            5: "door",
            6: "locked_door",
            8: "trap_door",
            10: "launch_pad",
            11: "one_way_platform",
            14: "drone",
            17: "bounce_block",
            20: "thwump",
            21: "toggle_mine",
            25: "death_ball",
            26: "mini_drone"
        }
        
        return type_mapping.get(entity_type, f"unknown_{entity_type}")
    
    def simulate_replay(self, inputs: List[int], map_data: List[int], 
                       level_id: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Simulate a replay and extract frame-by-frame data.
        
        Args:
            inputs: Raw input sequence
            map_data: Map data
            level_id: Level identifier
            session_id: Session identifier
            
        Returns:
            List of frame dictionaries in JSONL format
        """
        frames = []
        
        # Decode inputs
        hor_inputs, jump_inputs = self.decode_inputs(inputs)
        inp_len = len(inputs)
        
        # Initialize simulator (disable animation for parsing)
        sim_config = SimConfig(enable_anim=False)
        sim = Simulator(sim_config)
        sim.load(map_data)
        
        # Track initial state
        start_time = time.time()
        
        # Execute simulation frame by frame
        while sim.frame < inp_len:
            # Get current inputs
            hor_input = hor_inputs[sim.frame]
            jump_input = jump_inputs[sim.frame]
            
            # Create frame data before simulation step
            frame_data = {
                "timestamp": start_time + (sim.frame * (1.0 / 60.0)),  # 60 FPS
                "level_id": level_id,
                "frame_number": sim.frame,
                "player_state": {
                    "position": {
                        "x": float(sim.ninja.xpos),
                        "y": float(sim.ninja.ypos)
                    },
                    "velocity": {
                        "x": float(sim.ninja.xspeed),
                        "y": float(sim.ninja.yspeed)
                    },
                    "on_ground": sim.ninja.state in [0, 1, 2],  # Ground states
                    "wall_sliding": sim.ninja.state == 5,       # Wall sliding state
                    "jump_time_remaining": max(0.0, (45 - sim.ninja.jump_timer) / 45.0) if hasattr(sim.ninja, 'jump_timer') else 0.0
                },
                "player_inputs": {
                    "left": hor_input == -1,
                    "right": hor_input == 1,
                    "jump": jump_input == 1,
                    "restart": False
                },
                "entities": self.get_entity_data(sim),
                "level_bounds": {
                    "width": self.LEVEL_WIDTH,
                    "height": self.LEVEL_HEIGHT
                },
                "meta": {
                    "session_id": session_id,
                    "player_id": "binary_replay",
                    "quality_score": 0.8,  # Default quality score
                    "completion_status": "in_progress"
                }
            }
            
            frames.append(frame_data)
            
            # Execute simulation step
            sim.tick(hor_input, jump_input)
            
            # Check for death or completion
            if sim.ninja.state == 6:  # Dead
                # Update final frame status
                frames[-1]["meta"]["completion_status"] = "failed"
                break
            elif sim.ninja.state == 8:  # Celebrating (completed)
                # Update final frame status
                frames[-1]["meta"]["completion_status"] = "completed"
                frames[-1]["meta"]["quality_score"] = 0.9  # Higher score for completion
                break
        
        logger.info(f"Generated {len(frames)} frames for replay")
        return frames
    
    def parse_replay_directory(self, replay_dir: Path, output_dir: Path, 
                              level_id: Optional[str] = None) -> bool:
        """
        Parse a single replay directory and generate JSONL files.
        
        Args:
            replay_dir: Directory containing binary replay files
            output_dir: Output directory for JSONL files
            level_id: Optional level identifier (auto-generated if None)
            
        Returns:
            True if parsing succeeded
        """
        try:
            # Check if this is a trace mode directory
            if not self.detect_trace_mode(replay_dir):
                logger.warning(f"Directory {replay_dir} does not contain trace mode files")
                return False
            
            # Load data
            inputs_list, map_data = self.load_inputs_and_map(replay_dir)
            
            # Generate level ID if not provided
            if level_id is None:
                # Use directory name or generate from map data hash
                if replay_dir.name != "." and replay_dir.name:
                    level_id = replay_dir.name
                else:
                    map_hash = hashlib.md5(str(map_data).encode()).hexdigest()[:8]
                    level_id = f"level_{map_hash}"
            
            # Process each input sequence
            session_counter = 0
            for i, inputs in enumerate(inputs_list):
                session_id = f"{level_id}_session_{session_counter:03d}"
                
                logger.info(f"Processing replay {i+1}/{len(inputs_list)} (session: {session_id})")
                
                try:
                    # Simulate and extract frames
                    frames = self.simulate_replay(inputs, map_data, level_id, session_id)
                    
                    if frames:
                        # Save to JSONL file
                        output_file = output_dir / f"{session_id}.jsonl"
                        self.save_frames_to_jsonl(frames, output_file)
                        
                        self.stats['frames_generated'] += len(frames)
                        self.stats['replays_processed'] += 1
                        session_counter += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process replay {i}: {e}")
                    self.stats['replays_failed'] += 1
            
            self.stats['files_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to parse replay directory {replay_dir}: {e}")
            return False
    
    def save_frames_to_jsonl(self, frames: List[Dict[str, Any]], output_file: Path):
        """
        Save frames to JSONL file.
        
        Args:
            frames: List of frame dictionaries
            output_file: Output file path
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for frame in frames:
                f.write(json.dumps(frame) + '\n')
        
        logger.info(f"Saved {len(frames)} frames to {output_file}")
    
    def process_directory(self, input_dir: Path, output_dir: Path):
        """
        Process all replay directories in the input directory.
        
        Args:
            input_dir: Directory containing replay subdirectories
            output_dir: Output directory for JSONL files
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if input_dir itself is a replay directory
        if self.detect_trace_mode(input_dir):
            logger.info(f"Processing single replay directory: {input_dir}")
            self.parse_replay_directory(input_dir, output_dir)
        else:
            # Look for subdirectories containing replays
            replay_dirs = []
            for item in input_dir.iterdir():
                if item.is_dir() and self.detect_trace_mode(item):
                    replay_dirs.append(item)
            
            if not replay_dirs:
                logger.warning(f"No trace mode replay directories found in {input_dir}")
                return
            
            logger.info(f"Found {len(replay_dirs)} replay directories")
            
            for replay_dir in replay_dirs:
                logger.info(f"Processing replay directory: {replay_dir}")
                self.parse_replay_directory(replay_dir, output_dir, level_id=replay_dir.name)
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*50)
        print("BINARY REPLAY PARSER STATISTICS")
        print("="*50)
        print(f"Directories processed: {self.stats['files_processed']}")
        print(f"Replays processed: {self.stats['replays_processed']}")
        print(f"Replays failed: {self.stats['replays_failed']}")
        print(f"Frames generated: {self.stats['frames_generated']}")
        
        if self.stats['replays_processed'] > 0:
            avg_frames = self.stats['frames_generated'] / self.stats['replays_processed']
            print(f"Average frames per replay: {avg_frames:.1f}")
        
        total_replays = self.stats['replays_processed'] + self.stats['replays_failed']
        if total_replays > 0:
            success_rate = self.stats['replays_processed'] / total_replays * 100
            print(f"Success rate: {success_rate:.1f}%")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert N++ binary replay files to JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single replay directory
  python tools/binary_replay_parser.py --input replays/level_001 --output datasets/raw/
  
  # Process multiple replay directories
  python tools/binary_replay_parser.py --input replays/ --output datasets/raw/
  
  # Enable verbose logging
  python tools/binary_replay_parser.py --input replays/ --output datasets/raw/ --verbose
        """
    )
    
    parser.add_argument('--input', type=Path, required=True,
                       help='Input directory containing binary replay files')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for JSONL files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")
    
    # Initialize parser
    replay_parser = BinaryReplayParser()
    
    # Process data
    logger.info(f"Processing binary replay data from {args.input}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        replay_parser.process_directory(args.input, args.output)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    # Print statistics
    replay_parser.print_statistics()
    
    return 0


if __name__ == '__main__':
    exit(main())
