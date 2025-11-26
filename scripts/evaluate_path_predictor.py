#!/usr/bin/env python3
"""Evaluation and visualization script for path predictor network.

This script:
1. Loads a trained path predictor checkpoint
2. Loads levels from N++ official maps
3. Generates candidate path predictions
4. Visualizes predictions with tiles, entities, and paths
5. Analyzes prediction quality

Usage:
    python scripts/evaluate_path_predictor.py \\
        --checkpoint models/path_predictor/checkpoint_epoch_10.pt \\
        --num-levels 5 \\
        --output-dir visualizations/path_predictions
"""

import argparse
import logging
import sys
import random
from pathlib import Path
import numpy as np
import torch

# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
nclone_root = project_root.parent / "nclone"
sys.path.insert(0, str(nclone_root))

from npp_rl.path_prediction.multipath_predictor import create_multipath_predictor
from npp_rl.path_prediction.graph_observation_utils import extract_graph_observation
from npp_rl.path_prediction.tile_feature_utils import extract_tile_features_from_tiles
from npp_rl.path_prediction.path_analysis import (
    compute_path_properties,
    characterize_path_strategy,
)
from npp_rl.rendering.matplotlib_tile_renderer import (
    render_tiles_to_axis,
    render_mines_to_axis,
)
from nclone.nplay_headless import NPlayHeadless
from nclone.graph.reachability.graph_builder import GraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize path predictor predictions"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--maps-dir",
        type=str,
        default=str(nclone_root / "nclone" / "maps" / "official" / "S"),
        help="Directory containing official maps",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=5,
        help="Number of levels to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/path_predictions",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for level selection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load trained path predictor from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded ProbabilisticPathPredictor model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same config
    predictor_config = {
        "graph_feature_dim": 256,
        "tile_pattern_dim": 64,
        "entity_feature_dim": 32,
        "num_path_candidates": 4,
        "max_waypoints": 20,
        "hidden_dim": 512,
    }

    predictor = create_multipath_predictor(predictor_config)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.to(device)
    predictor.eval()

    logger.info(
        f"Model loaded successfully (trained for {checkpoint['epoch'] + 1} epochs)"
    )

    return predictor


def load_random_levels(maps_dir: str, num_levels: int, seed: int = 42):
    """Load random levels from official maps directory.

    Args:
        maps_dir: Directory containing map files
        num_levels: Number of levels to load
        seed: Random seed

    Returns:
        List of (map_name, map_data) tuples
    """
    maps_path = Path(maps_dir)

    if not maps_path.exists():
        raise ValueError(f"Maps directory not found: {maps_dir}")

    # Get all map files
    map_files = sorted([f for f in maps_path.iterdir() if f.is_file()])

    if len(map_files) == 0:
        raise ValueError(f"No map files found in {maps_dir}")

    logger.info(f"Found {len(map_files)} map files in {maps_dir}")

    # Select random maps
    random.seed(seed)
    selected_files = random.sample(map_files, min(num_levels, len(map_files)))

    # Load map data
    levels = []
    for map_file in selected_files:
        with open(map_file, "rb") as f:
            map_data = [int(b) for b in f.read()]
        levels.append((map_file.name, map_data))
        logger.info(f"  Loaded: {map_file.name}")

    return levels


def build_graph_for_level(map_data):
    """Build adjacency graph for a level.

    Args:
        map_data: Level map data

    Returns:
        Tuple of (adjacency_dict, spatial_hash, sim)
    """
    from nclone.graph.level_data import extract_start_position_from_map_data

    # N++ standard level dimensions (interior playable area)
    MAP_TILE_HEIGHT = 23
    MAP_TILE_WIDTH = 42

    # Create simulator and load map
    sim = NPlayHeadless(
        render_mode="grayscale_array",
        enable_animation=False,
        enable_logging=False,
        enable_debug_overlay=False,
        enable_rendering=False,
    )
    sim.load_map_from_map_data(map_data)

    # Extract tile data as 2D array (following ReplayExecutor pattern)
    tile_dic = sim.get_tile_data()
    tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    # Simulator tiles include a 1-tile border; map inner (1..42, 1..23) -> (0..41, 0..22)
    for (x, y), tile_id in tile_dic.items():
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            # Handle NaN values (shouldn't happen but safety check)
            if np.isnan(tile_id):
                tiles[inner_y, inner_x] = 0  # Default to empty tile
            else:
                tiles[inner_y, inner_x] = int(tile_id)

    # Extract entities
    entities = []
    for entity in sim.sim.entity_dic.values():
        for e in entity:
            entities.append(
                {
                    "type": e.type,
                    "x": e.xpos,
                    "y": e.ypos,
                    "toggled": getattr(e, "state", 0)
                    == 0,  # state 0 = toggled (deadly)
                }
            )

    # Extract start position from map data
    start_position = extract_start_position_from_map_data(map_data)

    # Build level data for graph
    level_data = {
        "tiles": tiles,
        "entities": entities,
        "start_position": start_position,
        "switch_states": {},  # Empty for now
    }

    # Build graph
    graph_builder = GraphBuilder(debug=False)
    graph_data = graph_builder.build_graph(level_data)

    adjacency = graph_data["adjacency"]
    # Create a simple spatial hash (not used but kept for API compatibility)
    spatial_hash = None

    logger.debug(f"Built graph with {len(adjacency)} nodes")

    return adjacency, spatial_hash, sim


def extract_features_for_prediction(sim, adjacency, spatial_hash, device):
    """Extract features needed for path prediction.

    Args:
        sim: Simulator instance
        adjacency: Graph adjacency dictionary
        spatial_hash: Spatial hash for node lookups
        device: Device to place tensors on

    Returns:
        Tuple of (graph_obs, tile_patterns, entity_features)
    """
    # Build graph data structure for observation extraction
    graph_data = {
        "adjacency": adjacency,
        "num_nodes": len(adjacency) if adjacency else 0,
        "num_edges": sum(len(neighbors) for neighbors in adjacency.values()) // 2
        if adjacency
        else 0,
    }

    # Use consistent graph observation extraction
    graph_obs_single = extract_graph_observation(
        graph_data, target_dim=256, device=device
    )
    graph_obs = graph_obs_single.unsqueeze(0)  # Add batch dimension

    # Extract actual tile pattern features from level tiles
    tile_dic = sim.get_tile_data()
    # Convert tile_dic to 2D array (N++ levels are 42x23 interior tiles)
    MAP_TILE_HEIGHT = 23
    MAP_TILE_WIDTH = 42
    tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    for (x, y), tile_id in tile_dic.items():
        # Simulator uses 1-based indexing with border; map to 0-based interior
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            # Handle NaN values (shouldn't happen but safety check)
            if np.isnan(tile_id):
                tiles[inner_y, inner_x] = 0  # Default to empty tile
            else:
                tiles[inner_y, inner_x] = int(tile_id)

    tile_patterns_single = extract_tile_features_from_tiles(
        tiles, target_dim=64, device=device
    )
    tile_patterns = tile_patterns_single.unsqueeze(0)  # Add batch dimension

    # Entity features: encode entity counts
    entity_features = torch.zeros(1, 32, device=device)
    if hasattr(sim.sim, "entity_dic"):
        # Count different entity types
        num_mines = 0
        num_doors = 0
        num_switches = 0

        for entity_list in sim.sim.entity_dic.values():
            for entity in entity_list:
                if entity.type in [1, 21]:  # Toggle mines
                    num_mines += 1
                elif entity.type in [5, 6, 8]:  # Doors
                    num_doors += 1
                elif entity.type in [10, 11]:  # Switches
                    num_switches += 1

        entity_features[0, 0] = min(num_mines / 20.0, 1.0)
        entity_features[0, 1] = min(num_doors / 10.0, 1.0)
        entity_features[0, 2] = min(num_switches / 10.0, 1.0)

    return graph_obs, tile_patterns, entity_features


def predict_paths(
    predictor, graph_obs, tile_patterns, entity_features, adjacency, spatial_hash
):
    """Generate path predictions.

    Args:
        predictor: Path predictor model
        graph_obs: Graph observation features
        tile_patterns: Tile pattern features
        entity_features: Entity context features
        adjacency: Graph adjacency for validation
        spatial_hash: Spatial hash for validation

    Returns:
        List of CandidatePath objects
    """
    # Set graph data for validation
    predictor.set_graph_data(adjacency, spatial_hash)

    # Generate predictions
    with torch.no_grad():
        candidate_paths = predictor.predict_candidate_paths(
            graph_obs, tile_patterns, entity_features
        )

    return candidate_paths


def visualize_predictions(
    sim,
    adjacency,
    candidate_paths,
    output_path,
    map_name,
):
    """Create visualization of predicted paths.

    Args:
        sim: Simulator instance
        adjacency: Graph adjacency
        candidate_paths: List of CandidatePath objects
        output_path: Path to save visualization
        map_name: Name of the map
    """
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)

    # Get level bounds (N++ levels are 42x23 tiles interior + 1 tile border = 44x25)
    level_width = 44 * 24  # 44 tiles wide (including border)
    level_height = 25 * 24  # 25 tiles tall (including border)

    # Set axis limits
    ax.set_xlim(0, level_width)
    ax.set_ylim(level_height, 0)  # Invert Y axis
    ax.set_aspect("equal")

    # Render tiles (tile_dic is already a dictionary from sim.get_tile_data())
    tile_dic = sim.get_tile_data()

    render_tiles_to_axis(
        ax,
        tile_dic,
        tile_size=24.0,
        tile_color="#606060",
        alpha=0.8,
        show_tile_labels=False,
    )

    # Render mines
    mines = []
    for entity_list in sim.sim.entity_dic.values():
        for entity in entity_list:
            if entity.type in [1, 21]:  # Toggle mines
                mines.append(
                    {
                        "x": entity.xpos,
                        "y": entity.ypos,
                        "radius": 18,  # Approximate mine radius
                        "state": getattr(
                            entity, "state", 1
                        ),  # 0=toggled(deadly), 1=untoggled(safe), 2=toggling
                    }
                )

    if mines:
        render_mines_to_axis(ax, mines, tile_color="#FF4444", safe_color="#44AAFF")

    # Prepare tiles and entities for path analysis
    tile_dic = sim.get_tile_data()
    MAP_TILE_HEIGHT = 23
    MAP_TILE_WIDTH = 42
    tiles_array = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    for (x, y), tile_id in tile_dic.items():
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            if not np.isnan(tile_id):
                tiles_array[inner_y, inner_x] = int(tile_id)

    # Extract entities for analysis
    entities_list = []
    for entity_list in sim.sim.entity_dic.values():
        for entity in entity_list:
            entities_list.append(
                {
                    "type": entity.type,
                    "x": entity.xpos,
                    "y": entity.ypos,
                }
            )

    # Render predicted paths with learned characterization
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    for idx, path in enumerate(candidate_paths[:4]):  # Limit to 4 paths
        if len(path.waypoints) < 2:
            continue

        color = colors[idx % len(colors)]

        # Compute path properties and characterization
        path_properties = compute_path_properties(
            path.waypoints, tiles=tiles_array, entities=entities_list
        )
        path_characterization = characterize_path_strategy(path_properties)

        # Extract x, y coordinates
        xs = [wp[0] for wp in path.waypoints]
        ys = [wp[1] for wp in path.waypoints]

        # Plot lines connecting waypoints
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.5,
            alpha=0.6,
            linestyle="-",
            label=f"{path_characterization}: conf={path.confidence:.2f}, cost={path.estimated_cost:.1f}",
            zorder=5,
        )

        # Plot all waypoints as nodes
        ax.scatter(
            xs,
            ys,
            color=color,
            s=80,  # Size of waypoint markers
            alpha=0.8,
            edgecolors="white",
            linewidths=1.5,
            zorder=8,
        )

        # Mark start position (green circle)
        ax.plot(
            xs[0],
            ys[0],
            marker="o",
            color="#00FF00",
            markersize=14,
            markeredgecolor="black",
            markeredgewidth=2.5,
            zorder=10,
            label=f"Start {idx + 1}" if idx == 0 else "",
        )

        # Mark end position/goal (red star)
        ax.plot(
            xs[-1],
            ys[-1],
            marker="*",
            color="#FF0000",
            markersize=18,
            markeredgecolor="black",
            markeredgewidth=2.5,
            zorder=10,
            label=f"Goal {idx + 1}" if idx == 0 else "",
        )

        # Add waypoint numbers for clarity (optional, only for first path to avoid clutter)
        if idx == 0 and len(xs) <= 15:  # Only label if not too many waypoints
            for wp_idx, (x, y) in enumerate(
                zip(xs[1:-1], ys[1:-1]), start=1
            ):  # Skip start and end
                ax.annotate(
                    f"{wp_idx}",
                    xy=(x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color="white",
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.7,
                    ),
                    zorder=9,
                )

    # Add legend
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Title and labels
    ax.set_title(f"Path Predictions: {map_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


def analyze_predictions(candidate_paths, adjacency):
    """Analyze prediction quality.

    Args:
        candidate_paths: List of CandidatePath objects
        adjacency: Graph adjacency for validation

    Returns:
        Dictionary with analysis results
    """
    from npp_rl.path_prediction.graph_utils import validate_path_on_graph

    analysis = {
        "num_paths": len(candidate_paths),
        "valid_paths": 0,
        "avg_confidence": 0.0,
        "avg_path_length": 0.0,
        "avg_waypoints": 0.0,
        "diversity_score": 0.0,
    }

    if not candidate_paths:
        return analysis

    # Validate paths on graph
    for path in candidate_paths:
        if len(path.waypoints) >= 2:
            is_valid, graph_dist = validate_path_on_graph(
                path.waypoints, adjacency, None
            )
            if is_valid:
                analysis["valid_paths"] += 1

    # Compute averages
    confidences = [p.confidence for p in candidate_paths]
    path_lengths = [len(p.waypoints) for p in candidate_paths]

    analysis["avg_confidence"] = np.mean(confidences) if confidences else 0.0
    analysis["avg_waypoints"] = np.mean(path_lengths) if path_lengths else 0.0

    # Compute diversity (average pairwise distance between paths)
    if len(candidate_paths) >= 2:
        diversities = []
        for i, path1 in enumerate(candidate_paths):
            for path2 in candidate_paths[i + 1 :]:
                if len(path1.waypoints) > 0 and len(path2.waypoints) > 0:
                    # Simple diversity: distance between path endpoints
                    dist = np.sqrt(
                        (path1.waypoints[-1][0] - path2.waypoints[-1][0]) ** 2
                        + (path1.waypoints[-1][1] - path2.waypoints[-1][1]) ** 2
                    )
                    diversities.append(dist)
        analysis["diversity_score"] = np.mean(diversities) if diversities else 0.0

    return analysis


def main():
    """Main evaluation loop."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PATH PREDICTOR EVALUATION")
    logger.info("=" * 60)

    # Load model
    predictor = load_model(args.checkpoint, args.device)

    # Load random levels
    levels = load_random_levels(args.maps_dir, args.num_levels, args.seed)

    # Evaluate each level
    all_analyses = []

    for level_idx, (map_name, map_data) in enumerate(levels):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating Level {level_idx + 1}/{len(levels)}: {map_name}")
        logger.info(f"{'=' * 60}")

        try:
            # Build graph
            logger.info("Building graph...")
            adjacency, spatial_hash, sim = build_graph_for_level(map_data)

            # Extract features
            logger.info("Extracting features...")
            graph_obs, tile_patterns, entity_features = extract_features_for_prediction(
                sim, adjacency, spatial_hash, args.device
            )

            # Predict paths
            logger.info("Generating path predictions...")
            candidate_paths = predict_paths(
                predictor,
                graph_obs,
                tile_patterns,
                entity_features,
                adjacency,
                spatial_hash,
            )

            logger.info(f"Generated {len(candidate_paths)} candidate paths")

            # Visualize
            output_path = (
                output_dir / f"level_{level_idx:03d}_{map_name.replace(' ', '_')}.png"
            )
            logger.info("Creating visualization...")
            visualize_predictions(
                sim, adjacency, candidate_paths, str(output_path), map_name
            )

            # Analyze
            logger.info("Analyzing predictions...")
            analysis = analyze_predictions(candidate_paths, adjacency)
            all_analyses.append(analysis)

            # Log analysis
            logger.info(
                f"  Valid paths: {analysis['valid_paths']}/{analysis['num_paths']}"
            )
            logger.info(f"  Avg confidence: {analysis['avg_confidence']:.3f}")
            logger.info(f"  Avg waypoints: {analysis['avg_waypoints']:.1f}")
            logger.info(f"  Diversity score: {analysis['diversity_score']:.1f} px")

            # Cleanup (NPlayHeadless uses exit() not close())
            if hasattr(sim, "exit"):
                sim.exit()

        except Exception as e:
            logger.error(f"Error evaluating {map_name}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    # Overall statistics
    logger.info(f"\n{'=' * 60}")
    logger.info("OVERALL STATISTICS")
    logger.info(f"{'=' * 60}")

    if all_analyses:
        avg_valid = np.mean(
            [a["valid_paths"] / max(a["num_paths"], 1) for a in all_analyses]
        )
        avg_confidence = np.mean([a["avg_confidence"] for a in all_analyses])
        avg_waypoints = np.mean([a["avg_waypoints"] for a in all_analyses])
        avg_diversity = np.mean([a["diversity_score"] for a in all_analyses])

        logger.info(f"Average valid path rate: {avg_valid:.2%}")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        logger.info(f"Average waypoints per path: {avg_waypoints:.1f}")
        logger.info(f"Average diversity: {avg_diversity:.1f} px")

    logger.info(f"\nVisualizations saved to {output_dir}")
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
