import subprocess
from pathlib import Path


def process_video(input_path: Path, output_path: Path, rotate: bool = True, mirror: bool = True):
    """Process a video using ffmpeg to rotate 90 degrees clockwise and/or mirror it."""
    # Build the ffmpeg command
    filter_complex = []
    if mirror:
        filter_complex.append("hflip")
    if rotate:
        filter_complex.append("transpose=2")  # 1 = 90 degrees clockwise

    filter_str = ",".join(filter_complex) if filter_complex else "copy"

    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264",  # Use H.264 codec
        "-preset", "medium",  # Balance between speed and quality
        "-y",  # Overwrite output file if it exists
        str(output_path)
    ]

    print(f"\nProcessing {input_path.name}")
    try:
        # Run ffmpeg command
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Finished processing {input_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path.name}:")
        print(e.stderr)


def main():
    # Base directory for videos
    videos_dir = Path("./videos")

    # Create dir for processed videos
    processed_videos_dir = videos_dir / "processed"
    processed_videos_dir.mkdir(exist_ok=True)

    # Find all MP4 files in subdirectories
    for video_file in videos_dir.glob("**/*.mp4"):
        if "processed" not in str(video_file):  # Skip processed directory
            # Create output path
            rel_path = video_file.relative_to(videos_dir)
            output_path = processed_videos_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Process video (both rotate and mirror)
            process_video(video_file, output_path)


if __name__ == "__main__":
    main()
