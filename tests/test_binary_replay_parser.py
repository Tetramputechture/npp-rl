"""
Tests for Binary Replay Parser.

This module tests the binary replay parser functionality for converting N++
binary replay files to JSONL format compatible with the training pipeline.
"""

import json
import tempfile
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.binary_replay_parser import BinaryReplayParser


class TestBinaryReplayParser:
    """Test cases for BinaryReplayParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = BinaryReplayParser()
        self.example_replay_dir = Path("example_replays/r1")
    
    def test_input_decoding(self):
        """Test input decoding functionality."""
        # Test various input combinations
        test_inputs = [0, 1, 2, 3, 4, 5, 6, 7]
        
        hor_inputs, jump_inputs = self.parser.decode_inputs(test_inputs)
        
        # Check horizontal inputs
        expected_hor = [0, 0, 1, 1, -1, -1, -1, -1]
        assert hor_inputs == expected_hor
        
        # Check jump inputs  
        expected_jump = [0, 1, 0, 1, 0, 1, 0, 1]
        assert jump_inputs == expected_jump
    
    def test_entity_type_name_mapping(self):
        """Test entity type name mapping."""
        # Test known entity types
        assert self.parser._get_entity_type_name(1) == "mine"
        assert self.parser._get_entity_type_name(2) == "gold"
        assert self.parser._get_entity_type_name(3) == "exit_door"
        assert self.parser._get_entity_type_name(4) == "exit_switch"
        assert self.parser._get_entity_type_name(10) == "launch_pad"
        assert self.parser._get_entity_type_name(20) == "thwump"
        
        # Test unknown entity type
        assert self.parser._get_entity_type_name(999) == "unknown_999"
    
    def test_detect_trace_mode(self):
        """Test trace mode detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test without files - should return False
            assert not self.parser.detect_trace_mode(temp_path)
            
            # Create map_data file
            (temp_path / "map_data").write_bytes(b"test_map_data")
            assert not self.parser.detect_trace_mode(temp_path)  # Still missing inputs
            
            # Create inputs_0 file
            (temp_path / "inputs_0").write_bytes(b"test_inputs")
            assert self.parser.detect_trace_mode(temp_path)  # Should return True now
    
    def test_example_replay_existence(self):
        """Test that the example replay directory exists and has expected files."""
        if not self.example_replay_dir.exists():
            pytest.skip(f"Example replay directory {self.example_replay_dir} not found")
        
        # Check for required files
        assert (self.example_replay_dir / "map_data").exists(), "map_data file missing"
        
        # Check for at least one input file
        input_files = [self.example_replay_dir / f for f in self.parser.RAW_INPUTS]
        input_files_exist = [f.exists() for f in input_files]
        assert any(input_files_exist), "No input files found"
    
    def test_load_inputs_and_map_with_example(self):
        """Test loading inputs and map data with the example replay."""
        if not self.example_replay_dir.exists():
            pytest.skip(f"Example replay directory {self.example_replay_dir} not found")
        
        if not self.parser.detect_trace_mode(self.example_replay_dir):
            pytest.skip(f"Example replay directory {self.example_replay_dir} is not in trace mode")
        
        try:
            inputs_list, map_data = self.parser.load_inputs_and_map(self.example_replay_dir)
            
            # Validate inputs
            assert len(inputs_list) > 0, "No input sequences loaded"
            assert all(isinstance(inputs, list) for inputs in inputs_list), "Input sequences should be lists"
            assert all(all(isinstance(inp, int) for inp in inputs) for inputs in inputs_list), "Input values should be integers"
            assert all(all(0 <= inp <= 7 for inp in inputs) for inputs in inputs_list), "Input values should be in range 0-7"
            
            # Validate map data
            assert isinstance(map_data, list), "Map data should be a list"
            assert len(map_data) > 0, "Map data should not be empty"
            assert all(isinstance(b, int) for b in map_data), "Map data values should be integers"
            
        except Exception as e:
            pytest.fail(f"Failed to load example replay data: {e}")
    
    @patch('tools.binary_replay_parser.Simulator')
    @patch('tools.binary_replay_parser.SimConfig')
    def test_simulate_replay_mock(self, mock_sim_config, mock_simulator_class):
        """Test replay simulation with mocked simulator."""
        # Create mock simulator instance
        mock_sim = mock_simulator_class.return_value
        mock_sim.frame = 0
        mock_sim.ninja.xpos = 100.0
        mock_sim.ninja.ypos = 200.0
        mock_sim.ninja.xspeed = 1.5
        mock_sim.ninja.yspeed = -0.5
        mock_sim.ninja.state = 1  # Running state
        mock_sim.ninja.jump_timer = 0  # Add jump_timer attribute
        mock_sim.entity_dic = {}
        
        # Test input sequence
        test_inputs = [0, 2, 3, 2, 0]  # No input, right, right+jump, right, no input
        test_map_data = [1, 2, 3, 4, 5]  # Dummy map data
        
        # Mock the simulation loop
        def mock_tick(hor_input, jump_input):
            mock_sim.frame += 1
            # Simulate ninja movement
            mock_sim.ninja.xpos += hor_input * 2.0
            if jump_input:
                mock_sim.ninja.yspeed = -2.0
                mock_sim.ninja.jump_timer += 1
            else:
                mock_sim.ninja.yspeed += 0.1  # Gravity
                mock_sim.ninja.jump_timer = max(0, mock_sim.ninja.jump_timer - 1)
        
        mock_sim.tick = mock_tick
        
        # Run simulation
        frames = self.parser.simulate_replay(
            test_inputs, test_map_data, "test_level", "test_session"
        )
        
        # Validate output
        assert len(frames) == len(test_inputs), f"Expected {len(test_inputs)} frames, got {len(frames)}"
        
        for i, frame in enumerate(frames):
            # Check required fields
            assert "timestamp" in frame
            assert "level_id" in frame
            assert "frame_number" in frame
            assert "player_state" in frame
            assert "player_inputs" in frame
            assert "entities" in frame
            assert "level_bounds" in frame
            assert "meta" in frame
            
            # Check frame number
            assert frame["frame_number"] == i
            
            # Check level ID
            assert frame["level_id"] == "test_level"
            
            # Check player inputs match decoded values
            expected_hor = self.parser.HOR_INPUTS_DIC[test_inputs[i]]
            expected_jump = self.parser.JUMP_INPUTS_DIC[test_inputs[i]]
            
            assert frame["player_inputs"]["left"] == (expected_hor == -1)
            assert frame["player_inputs"]["right"] == (expected_hor == 1)
            assert frame["player_inputs"]["jump"] == (expected_jump == 1)
            assert frame["player_inputs"]["restart"] is False
            
            # Check level bounds
            assert frame["level_bounds"]["width"] == 1056
            assert frame["level_bounds"]["height"] == 600
            
            # Check meta fields
            assert frame["meta"]["session_id"] == "test_session"
            assert frame["meta"]["player_id"] == "binary_replay"
            assert frame["meta"]["completion_status"] == "in_progress"
    
    def test_save_frames_to_jsonl(self):
        """Test saving frames to JSONL format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_file = temp_path / "test_output.jsonl"
            
            # Create test frames
            test_frames = [
                {
                    "timestamp": 1692345678.0,
                    "level_id": "test_level",
                    "frame_number": 0,
                    "player_state": {
                        "position": {"x": 100.0, "y": 200.0},
                        "velocity": {"x": 1.0, "y": -0.5},
                        "on_ground": True,
                        "wall_sliding": False,
                        "jump_time_remaining": 0.0
                    },
                    "player_inputs": {
                        "left": False,
                        "right": True,
                        "jump": False,
                        "restart": False
                    },
                    "entities": [],
                    "level_bounds": {"width": 1056, "height": 600},
                    "meta": {
                        "session_id": "test_session",
                        "player_id": "binary_replay",
                        "quality_score": 0.8,
                        "completion_status": "in_progress"
                    }
                }
            ]
            
            # Save frames
            self.parser.save_frames_to_jsonl(test_frames, output_file)
            
            # Verify file was created
            assert output_file.exists()
            
            # Read back and verify content
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 1
            loaded_frame = json.loads(lines[0].strip())
            assert loaded_frame == test_frames[0]
    
    def test_process_example_replay_end_to_end(self):
        """Test end-to-end processing of the example replay."""
        if not self.example_replay_dir.exists():
            pytest.skip(f"Example replay directory {self.example_replay_dir} not found")
        
        if not self.parser.detect_trace_mode(self.example_replay_dir):
            pytest.skip(f"Example replay directory {self.example_replay_dir} is not in trace mode")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            try:
                # Process the example replay
                success = self.parser.parse_replay_directory(
                    self.example_replay_dir, output_dir, level_id="test_level"
                )
                
                assert success, "Replay processing should succeed"
                
                # Check that output files were created
                output_files = list(output_dir.glob("*.jsonl"))
                assert len(output_files) > 0, "No output files were created"
                
                # Validate the first output file
                output_file = output_files[0]
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                
                assert len(lines) > 0, "Output file should contain frames"
                
                # Validate first frame structure
                first_frame = json.loads(lines[0].strip())
                required_fields = [
                    "timestamp", "level_id", "frame_number", "player_state",
                    "player_inputs", "entities", "level_bounds", "meta"
                ]
                
                for field in required_fields:
                    assert field in first_frame, f"Missing required field: {field}"
                
                # Validate player state structure
                player_state = first_frame["player_state"]
                assert "position" in player_state
                assert "velocity" in player_state
                assert "x" in player_state["position"]
                assert "y" in player_state["position"]
                assert "x" in player_state["velocity"]
                assert "y" in player_state["velocity"]
                
                # Validate level bounds
                assert first_frame["level_bounds"]["width"] == 1056
                assert first_frame["level_bounds"]["height"] == 600
                
                # Check statistics were updated
                assert self.parser.stats['files_processed'] > 0
                assert self.parser.stats['frames_generated'] > 0
                assert self.parser.stats['replays_processed'] > 0
                
            except Exception as e:
                pytest.fail(f"End-to-end test failed: {e}")
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        initial_stats = self.parser.stats.copy()
        
        # All stats should start at 0
        for key, value in initial_stats.items():
            assert value == 0, f"Initial stat {key} should be 0"
        
        # Test print_statistics doesn't crash
        try:
            self.parser.print_statistics()
        except Exception as e:
            pytest.fail(f"print_statistics should not raise exception: {e}")
