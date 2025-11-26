#!/usr/bin/env python3
"""Visualize expert paths from replay demonstrations.

This script loads replay files and visualizes the extracted expert paths
overlaid on level geometry to validate data ingestion.

Usage:
    python scripts/visualize_replay_paths.py \
        --replay-dir /path/to/replays \
        --output-dir visualizations/ \
        --max-replays 10 \
        --waypoint-interval 5
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Set matplotlib to use non-interactive backend
matplotlib.use("Agg")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
nclone_root = project_root.parent / "nclone"
sys.path.insert(0, str(nclone_root))

from npp_rl.path_prediction.path_replay_dataset import PathReplayDataset
from npp_rl.path_prediction.coordinate_utils import denormalize_waypoints
from npp_rl.rendering.matplotlib_tile_renderer import (
    render_tiles_to_axis,
    render_mines_to_axis,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Standard N++ level dimensions
LEVEL_WIDTH = 44 * 24  # 1056 pixels
LEVEL_HEIGHT = 25 * 24  # 600 pixels

# Border offset: tiles are inner playable area, waypoints span full level
# Need to offset tiles by 24px to align with waypoint coordinate system
BORDER_OFFSET = 24  # 1 tile = 24 pixels border


def tiles_array_to_dict(tiles: np.ndarray, border_offset: int = 24) -> dict:
    """Convert tiles numpy array to dictionary format for rendering.

    Tiles array is (height, width) = (23, 42) representing inner playable area.
    Grid coordinates (x, y) need to be converted to pixel coordinates for rendering.
    The renderer expects grid coordinates and multiplies by 24 internally.

    To align with waypoints that span the full level (including border), we offset
    tile grid coordinates by 1 tile (border_offset/24) so they render at the correct
    pixel positions.

    Args:
        tiles: 2D numpy array of shape (height, width) with tile types
        border_offset: Border offset in pixels (default 24)

    Returns:
        Dictionary mapping (x, y) grid coordinates to tile type values
        Grid coordinates are offset by border_offset/24 to account for border
    """
    tile_dict = {}
    height, width = tiles.shape
    grid_offset = border_offset // 24  # Convert pixels to grid units (should be 1)

    for y in range(height):
        for x in range(width):
            tile_type = int(tiles[y, x])
            if tile_type != 0:  # Skip empty tiles
                # Tiles array is indexed as [y, x], store as (x, y) grid coordinates
                # Offset by grid_offset to account for border when rendering
                # Renderer will convert grid coords to pixel coords by multiplying by 24
                tile_dict[(x + grid_offset, y + grid_offset)] = tile_type
    return tile_dict


def render_entities(ax, entities: list) -> None:
    """Render entities (exit doors, switches) on the axis.

    Args:
        ax: Matplotlib axis to render on
        entities: List of entity dictionaries with type, x, y fields
        Entity types are integers: EXIT_DOOR=3, EXIT_SWITCH=4
    """
    # EntityType constants
    EXIT_DOOR = 3
    EXIT_SWITCH = 4

    for entity in entities:
        entity_type = entity.get("type")
        x = entity.get("x", 0)
        y = entity.get("y", 0)

        # Handle both integer enum values and string fallbacks
        if (
            entity_type == EXIT_DOOR
            or entity_type == "exit"
            or entity_type == "exit_door"
        ):
            # Exit door - green circle
            circle = mpatches.Circle(
                (x, y),
                12,  # radius
                facecolor="green",
                edgecolor="black",
                linewidth=2,
                alpha=0.8,
                zorder=4,
            )
            ax.add_patch(circle)
        elif (
            entity_type == EXIT_SWITCH
            or entity_type == "switch"
            or entity_type == "exit_switch"
        ):
            # Exit switch - blue square
            square = mpatches.Rectangle(
                (x - 8, y - 8),
                16,
                16,
                facecolor="blue",
                edgecolor="black",
                linewidth=2,
                alpha=0.8,
                zorder=4,
            )
            ax.add_patch(square)


def visualize_replay_sample(sample: dict, output_path: Path, replay_id: str) -> None:
    """Create visualization for a single replay sample.

    Args:
        sample: Sample dictionary from PathReplayDataset
        output_path: Path to save visualization image
        replay_id: Replay identifier for title
    """
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

    # Set axis limits and invert Y-axis to match N++ coordinate system
    ax.set_xlim(0, LEVEL_WIDTH)
    ax.set_ylim(LEVEL_HEIGHT, 0)  # Invert Y axis
    ax.set_aspect("equal")

    # Render tiles
    # Tiles array is (23, 42) inner playable area in grid coordinates
    # Waypoints are in tile data coordinates (full level dimensions 1056x600)
    # The renderer expects grid coordinates and multiplies by 24 to get pixel coordinates
    tiles = sample.get("tiles")
    if tiles is not None:
        # Debug: log tile data info
        if isinstance(tiles, np.ndarray):
            logger.debug(
                f"Tiles shape: {tiles.shape}, dtype: {tiles.dtype}, "
                f"min: {tiles.min()}, max: {tiles.max()}, non-zero: {np.count_nonzero(tiles)}"
            )
        tile_dict = tiles_array_to_dict(tiles, border_offset=BORDER_OFFSET)
        logger.debug(f"Converted to {len(tile_dict)} non-empty tiles in dictionary")
        # Render tiles - grid coordinates are offset by 1 tile to account for border
        # This aligns tiles with waypoints that span the full level (1056x600)
        render_tiles_to_axis(
            ax,
            tile_dict,
            tile_size=24.0,
            tile_color="#606060",
            alpha=0.9,
            show_tile_labels=False,
        )
    else:
        logger.warning(f"No tiles data available for {replay_id}")

    # Render entities (exit, switches)
    # Entities are in tile-data coordinates, same as waypoints (no offset needed)
    entities = sample.get("entities", [])
    if entities:
        logger.debug(f"Found {len(entities)} entities to render")
        # Debug: log entity types
        entity_types = [e.get("type") for e in entities]
        logger.debug(f"Entity types: {entity_types[:10]}")  # First 10
        render_entities(ax, entities)
    else:
        logger.debug(f"No entities found for {replay_id}")

    # Extract mines from entities
    mines = []
    for entity in entities:
        if entity.get("type") == "mine":
            mines.append(
                {
                    "x": entity.get("x", 0),
                    "y": entity.get("y", 0),
                    "radius": entity.get("radius", 18),
                    "state": entity.get("state", 1),
                }
            )

    if mines:
        render_mines_to_axis(
            ax, mines, tile_color="#FF4444", safe_color="#44AAFF", alpha=0.8
        )

    # Get snapped waypoints (normalized) and denormalize to tile data coordinates
    expert_waypoints_normalized = sample.get("expert_waypoints", [])
    snapped_waypoints = denormalize_waypoints(
        expert_waypoints_normalized, LEVEL_WIDTH, LEVEL_HEIGHT
    )

    # Plot snapped waypoints with sequential color coding
    if snapped_waypoints:
        snapped_xs = [w[0] for w in snapped_waypoints]
        snapped_ys = [w[1] for w in snapped_waypoints]
        num_waypoints = len(snapped_waypoints)

        # Create color gradient from blue (start) to red (end)
        # Using 'plasma' colormap for better visibility
        colors = plt.cm.plasma(np.linspace(0, 1, num_waypoints))

        # Plot waypoints with sequential colors
        for i, (x, y) in enumerate(snapped_waypoints):
            ax.scatter(
                [x],
                [y],
                c=[colors[i]],
                s=40,
                alpha=0.9,
                edgecolors="white",
                linewidths=1.5,
                zorder=5,
            )

        # Connect snapped waypoints with gradient-colored lines
        if len(snapped_waypoints) > 1:
            # Draw segments with gradient colors
            for i in range(len(snapped_waypoints) - 1):
                ax.plot(
                    [snapped_xs[i], snapped_xs[i + 1]],
                    [snapped_ys[i], snapped_ys[i + 1]],
                    color=colors[i],
                    linewidth=2.5,
                    alpha=0.7,
                    zorder=4,
                )

        # Add colorbar to show sequence progression
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=num_waypoints - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Waypoint Sequence", rotation=270, labelpad=15)

    # NOTE: Raw waypoints are not stored in the dataset after snapping.
    # To show raw waypoints, we would need to re-execute the replay,
    # which is expensive. For now, we only show snapped waypoints.

    # Mark start position (green star)
    # Start position is in tile-data coordinates, same as waypoints (no offset needed)
    start_pos = sample.get("start_pos")
    if start_pos:
        ax.scatter(
            [start_pos[0]],
            [start_pos[1]],
            c="green",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            zorder=6,
            label="Start",
        )

    # Mark goal positions (red stars)
    # Goal positions are in tile-data coordinates, same as waypoints (no offset needed)
    goal_positions = sample.get("goal_positions", [])
    if goal_positions:
        goal_xs = [g[0] for g in goal_positions]
        goal_ys = [g[1] for g in goal_positions]
        ax.scatter(
            goal_xs,
            goal_ys,
            c="red",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            zorder=6,
            label="Goals",
        )

    # Create title with sample info
    trajectory_length = sample.get("trajectory_length", 0)
    success = sample.get("success", False)
    num_waypoints = len(snapped_waypoints)
    title = (
        f"{replay_id}\n"
        f"Success: {success}, "
        f"Trajectory: {trajectory_length} frames, "
        f"Waypoints: {num_waypoints}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend
    ax.legend(loc="upper right", fontsize=10)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved visualization to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize expert paths from replay demonstrations"
    )

    parser.add_argument(
        "--replay-dir",
        type=str,
        required=True,
        help="Directory containing .replay files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/",
        help="Output directory for visualization images (default: visualizations/)",
    )
    parser.add_argument(
        "--max-replays",
        type=int,
        default=None,
        help="Maximum number of replays to visualize (for testing)",
    )
    parser.add_argument(
        "--waypoint-interval",
        type=int,
        default=5,
        help="Extract waypoint every N frames (default: 5)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info("Loading replay dataset...")
    dataset = PathReplayDataset(
        replay_dir=args.replay_dir,
        waypoint_interval=args.waypoint_interval,
        max_replays=args.max_replays,
        enable_rendering=False,  # Don't need rendering for visualization
        enable_augmentation=False,  # Don't augment for visualization
    )

    logger.info(f"Loaded {len(dataset)} samples from dataset")

    # Visualize each sample
    logger.info("Creating visualizations...")
    for idx in tqdm(range(len(dataset)), desc="Visualizing"):
        try:
            sample = dataset[idx]
            replay_id = sample.get("replay_id", f"replay_{idx}")

            # Create output filename
            output_filename = f"{replay_id}.png"
            output_path = output_dir / output_filename

            # Create visualization
            visualize_replay_sample(sample, output_path, replay_id)

        except Exception as e:
            logger.warning(f"Failed to visualize sample {idx}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            continue

    logger.info(
        f"Visualization complete! Saved {len(list(output_dir.glob('*.png')))} images to {output_dir}"
    )


if __name__ == "__main__":
    main()
