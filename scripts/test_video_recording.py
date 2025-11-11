#!/usr/bin/env python3
"""Test script to verify video recording functionality.

This script creates a simple test to ensure:
1. VideoRecorder can be imported and initialized
2. Frames can be recorded
3. Videos can be saved to MP4 format
4. imageio and imageio-ffmpeg are properly installed
"""

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_video_recording():
    """Test video recording with synthetic frames."""
    logger.info("=" * 70)
    logger.info("Video Recording Test")
    logger.info("=" * 70)

    # Test imports
    logger.info("Testing imports...")
    try:
        import imageio
        import imageio_ffmpeg

        logger.info(f"  ✓ imageio version: {imageio.__version__}")
        logger.info(f"  ✓ imageio-ffmpeg version: {imageio_ffmpeg.__version__}")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("Install with: pip install imageio>=2.31.0 imageio-ffmpeg>=0.4.8")
        return False

    # Test VideoRecorder import
    try:
        from npp_rl.utils.video_recorder import VideoRecorder, create_video_recorder

        logger.info("  ✓ VideoRecorder imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import VideoRecorder: {e}")
        return False

    # Create test output directory
    output_dir = Path("test_output/video_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  ✓ Created output directory: {output_dir}")

    # Test video creation with synthetic frames
    logger.info("\nTesting video creation...")
    video_path = output_dir / "test_video.mp4"

    try:
        recorder = create_video_recorder(str(video_path), fps=30)
        if recorder is None:
            print("  ✗ Failed to create video recorder")
            return False

        logger.info("  ✓ VideoRecorder initialized")

        # Start recording
        recorder.start_recording()
        logger.info("  ✓ Recording started")

        # Generate synthetic frames (moving gradient)
        width, height = 640, 480
        num_frames = 90  # 3 seconds at 30 fps

        for i in range(num_frames):
            # Create a frame with a moving gradient
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Horizontal gradient that moves
            offset = int((i / num_frames) * width)
            for x in range(width):
                intensity = int(255 * ((x + offset) % width) / width)
                frame[:, x, 0] = intensity  # Red channel
                frame[:, x, 1] = 255 - intensity  # Green channel
                frame[:, x, 2] = 128  # Blue channel (constant)

            recorder.record_frame(frame)

            if (i + 1) % 30 == 0:
                logger.info(f"    Recording frame {i + 1}/{num_frames}...")

        logger.info(f"  ✓ Recorded {num_frames} frames")

        # Stop recording and save
        success = recorder.stop_recording(save=True)

        if success and video_path.exists():
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ Video saved successfully: {video_path}")
            logger.info(f"    Size: {file_size_mb:.2f} MB")
            logger.info(f"    Frames: {num_frames}")
            logger.info(f"    Duration: {num_frames / 30:.1f} seconds")
            return True
        else:
            print("  ✗ Failed to save video")
            return False

    except Exception as e:
        print(f"  ✗ Video recording failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_evaluation_video_integration():
    """Test that evaluation can create videos (imports only)."""
    logger.info("\nTesting evaluation video integration...")

    try:
        from npp_rl.evaluation.comprehensive_evaluator import ComprehensiveEvaluator

        logger.info("  ✓ ComprehensiveEvaluator imported successfully")

        # Check that the class has video recording parameters
        import inspect

        sig = inspect.signature(ComprehensiveEvaluator.evaluate_model)
        params = sig.parameters

        required_params = ["record_videos", "video_output_dir", "video_fps"]
        for param in required_params:
            if param in params:
                logger.info(f"    ✓ Parameter '{param}' available")
            else:
                print(f"    ⚠ Parameter '{param}' not found")

        return True

    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting video recording tests...\n")

    # Run tests
    test1_passed = test_video_recording()
    test2_passed = test_evaluation_video_integration()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    logger.info(
        f"Video Recording Test:     {'✓ PASSED' if test1_passed else '✗ FAILED'}"
    )
    logger.info(
        f"Evaluation Integration:   {'✓ PASSED' if test2_passed else '✗ FAILED'}"
    )

    if test1_passed and test2_passed:
        logger.info("\n✓ All tests passed!")
        logger.info(
            "\nVideo recording is ready. Use --record-eval-videos flag during training."
        )
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
