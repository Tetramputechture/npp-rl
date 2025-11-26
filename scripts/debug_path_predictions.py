#!/usr/bin/env python3
"""Debug script for analyzing path prediction quality.

This script provides detailed analysis of a trained path predictor:
- Loads model and test maps
- Generates predictions and compares with ground truth (if available)
- Computes metrics: node validity, connectivity, start/goal proximity
- Visualizes predictions with detailed annotations
- Generates analysis reports

Usage:
    python scripts/debug_path_predictions.py \
        --model-path path-pred-test/best_model.pt \
        --test-maps-dir ../nclone/nclone/maps/test-maps \
        --output-dir debug-analysis \
        --num-maps 10
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
import json

# Set matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project roots to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
nclone_root = project_root.parent / "nclone"
sys.path.insert(0, str(nclone_root))

from npp_rl.path_prediction.graph_path_predictor import create_graph_path_predictor
from npp_rl.rendering.matplotlib_tile_renderer import (
    render_tiles_to_axis,
    render_mines_to_axis,
)
from nclone.nplay_headless import NPlayHeadless
from nclone.graph.reachability.graph_builder import GraphBuilder
from nclone.graph.level_data import extract_start_position_from_map_data
from nclone.constants.physics_constants import MAP_TILE_HEIGHT, MAP_TILE_WIDTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Debug and analyze path predictor")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--test-maps-dir",
        type=str,
        required=True,
        help="Directory containing test maps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug-analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--num-maps",
        type=int,
        default=10,
        help="Number of maps to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu)",
    )

    return parser.parse_args()


def analyze_path_quality(
    positions: list,
    adjacency: dict,
    start_pos: tuple,
    goal_positions: list,
) -> dict:
    """Analyze quality metrics for a predicted path.

    Args:
        positions: List of predicted waypoint positions
        adjacency: Graph adjacency dictionary
        start_pos: Start position (x, y)
        goal_positions: List of goal positions

    Returns:
        Dictionary with quality metrics
    """
    metrics = {}

    if not positions or len(positions) == 0:
        return {
            "num_waypoints": 0,
            "valid_nodes_ratio": 0.0,
            "connectivity_ratio": 0.0,
            "start_proximity": float("inf"),
            "goal_proximity": float("inf"),
            "path_length_pixels": 0.0,
            "unique_nodes_ratio": 0.0,
        }

    # Check how many positions are valid graph nodes
    valid_nodes = sum(1 for p in positions if p in adjacency)
    metrics["num_waypoints"] = len(positions)
    metrics["valid_nodes_ratio"] = valid_nodes / len(positions)

    # Check connectivity (consecutive nodes are adjacent in graph)
    connected_pairs = 0
    total_pairs = 0
    for i in range(len(positions) - 1):
        if positions[i] in adjacency and positions[i + 1] in adjacency:
            total_pairs += 1
            # Check if there's an edge between them
            neighbors = [n[0] for n in adjacency.get(positions[i], [])]
            if positions[i + 1] in neighbors:
                connected_pairs += 1

    metrics["connectivity_ratio"] = (
        connected_pairs / total_pairs if total_pairs > 0 else 0.0
    )

    # Start proximity (distance from first waypoint to start)
    if positions:
        first_pos = positions[0]
        dist_to_start = np.sqrt(
            (first_pos[0] - start_pos[0]) ** 2 + (first_pos[1] - start_pos[1]) ** 2
        )
        metrics["start_proximity"] = dist_to_start
    else:
        metrics["start_proximity"] = float("inf")

    # Goal proximity (distance from last waypoint to nearest goal)
    if positions and goal_positions:
        last_pos = positions[-1]
        min_goal_dist = min(
            np.sqrt((last_pos[0] - gx) ** 2 + (last_pos[1] - gy) ** 2)
            for gx, gy in goal_positions
        )
        metrics["goal_proximity"] = min_goal_dist
    else:
        metrics["goal_proximity"] = float("inf")

    # Path length (total Euclidean distance)
    path_length = 0.0
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        path_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    metrics["path_length_pixels"] = path_length

    # Unique nodes ratio (detect repetitive predictions)
    unique_positions = len(set(positions))
    metrics["unique_nodes_ratio"] = unique_positions / len(positions)

    return metrics


def visualize_with_analysis(
    map_file: Path,
    positions_all_heads: list,
    adjacency: dict,
    start_pos: tuple,
    goal_positions: list,
    metrics_all_heads: list,
    output_path: Path,
):
    """Create detailed visualization with metrics overlaid.

    Args:
        map_file: Path to map file
        positions_all_heads: List of position lists (one per head)
        adjacency: Graph adjacency
        start_pos: Start position
        goal_positions: Goal positions
        metrics_all_heads: List of metrics dicts (one per head)
        output_path: Where to save visualization
    """
    # Load map
    with open(map_file, "rb") as f:
        map_data_bytes = f.read()
    map_data = [int(b) for b in map_data_bytes]

    nplay = NPlayHeadless(enable_rendering=False)
    nplay.load_map_from_map_data(map_data)

    tile_dic = nplay.get_tile_data()
    tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    for (x, y), tile_id in tile_dic.items():
        inner_x = x - 1
        inner_y = y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            tiles[inner_y, inner_x] = int(tile_id)

    # Create figure with subplots for each head + summary
    num_heads = len(positions_all_heads)
    fig = plt.figure(figsize=(20, 5 * ((num_heads + 1) // 2)))

    BORDER_OFFSET = 24
    colors = ["red", "blue", "orange", "purple"]

    for head_idx in range(num_heads):
        ax = fig.add_subplot(2, (num_heads + 1) // 2, head_idx + 1)

        # Render tiles
        render_tiles_to_axis(ax, tile_dic)
        render_mines_to_axis(ax, nplay.get_all_mine_data_for_visualization())

        # Draw graph nodes
        if adjacency:
            node_xs = [pos[0] + BORDER_OFFSET for pos in adjacency.keys()]
            node_ys = [pos[1] + BORDER_OFFSET for pos in adjacency.keys()]
            ax.scatter(node_xs, node_ys, c="gray", s=1, alpha=0.3, zorder=2)

        # Draw start position
        ax.scatter(
            [start_pos[0] + BORDER_OFFSET],
            [start_pos[1] + BORDER_OFFSET],
            c="green",
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=2,
            zorder=10,
            label="Start",
        )

        # Draw goal positions
        if goal_positions:
            for i, (gx, gy) in enumerate(goal_positions):
                ax.scatter(
                    gx + BORDER_OFFSET,
                    gy + BORDER_OFFSET,
                    c="red",
                    s=150,
                    marker="D",
                    edgecolors="orange",
                    linewidths=2,
                    zorder=10,
                    label=f"Goal {i + 1}" if i == 0 else "",
                )

        # Draw predicted path
        positions = positions_all_heads[head_idx]
        if positions and len(positions) > 1:
            xs = [pos[0] + BORDER_OFFSET for pos in positions]
            ys = [pos[1] + BORDER_OFFSET for pos in positions]

            ax.plot(
                xs,
                ys,
                color=colors[head_idx % len(colors)],
                linewidth=2,
                alpha=0.7,
                marker="o",
                markersize=4,
                zorder=5,
            )

        # Add metrics as text
        metrics = metrics_all_heads[head_idx]
        metrics_text = f"Head {head_idx + 1}\n"
        metrics_text += f"Waypoints: {metrics['num_waypoints']}\n"
        metrics_text += f"Valid nodes: {metrics['valid_nodes_ratio']:.1%}\n"
        metrics_text += f"Connected: {metrics['connectivity_ratio']:.1%}\n"
        metrics_text += f"Start dist: {metrics['start_proximity']:.1f}px\n"
        metrics_text += f"Goal dist: {metrics['goal_proximity']:.1f}px\n"
        metrics_text += f"Path length: {metrics['path_length_pixels']:.1f}px\n"
        metrics_text += f"Unique: {metrics['unique_nodes_ratio']:.1%}"

        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
            family="monospace",
        )

        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlim(0, 44 * 24)
        ax.set_ylim(0, 25 * 24)
        ax.invert_yaxis()
        ax.axis("equal")

    plt.suptitle(f"Path Prediction Analysis: {map_file.stem}", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Main analysis function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Path Prediction Debugging and Analysis")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test maps: {args.test_maps_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device)

    # Extract config from checkpoint
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        # Default config
        model_config = {
            "node_feature_dim": 16,
            "hidden_dim": 256,
            "output_dim": 256,
            "num_gnn_layers": 3,
            "num_path_candidates": 4,
            "max_waypoints": 20,
            "gnn_type": "gcn",
            "num_gat_heads": 8,
            "dropout": 0.1,
            "use_fusion": True,
            "context_dim": 256,
            "fusion_hidden_dim": 128,
        }

    predictor = create_graph_path_predictor(model_config)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.to(args.device)
    predictor.eval()

    logger.info("Model loaded successfully")
    logger.info(f"Model config: {model_config}")

    # Load test maps
    test_maps_path = Path(args.test_maps_dir)
    map_files = [f for f in test_maps_path.iterdir() if f.is_file() and f.suffix == ""]
    map_files.sort()
    map_files = map_files[: args.num_maps]

    logger.info(f"Found {len(map_files)} test maps")

    # Analyze each map
    all_metrics = []
    graph_builder = GraphBuilder()

    for map_idx, map_file in enumerate(map_files):
        logger.info(f"\n[{map_idx + 1}/{len(map_files)}] Analyzing {map_file.name}...")

        try:
            # Load map
            with open(map_file, "rb") as f:
                map_data_bytes = f.read()
            map_data = [int(b) for b in map_data_bytes]

            nplay = NPlayHeadless(enable_rendering=False)
            nplay.load_map_from_map_data(map_data)

            # Get tile data
            tile_dic = nplay.get_tile_data()
            tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
            for (x, y), tile_id in tile_dic.items():
                inner_x = x - 1
                inner_y = y - 1
                if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
                    tiles[inner_y, inner_x] = int(tile_id)

            # Extract start position
            start_pos = extract_start_position_from_map_data(nplay.current_map_data)
            if start_pos is None:
                logger.warning("  No start position found, skipping")
                continue

            # Extract goal positions
            goal_positions = []
            all_entities = nplay.get_all_entities()
            for entity in all_entities:
                entity_type = entity.get("type", -1)
                if entity_type == 3:  # Exit door
                    exit_door_pos = (int(entity["x"]) - 24, int(entity["y"]) - 24)
                    goal_positions.append(exit_door_pos)
                    if "switch_x" in entity and "switch_y" in entity:
                        exit_switch_pos = (
                            int(entity["switch_x"]) - 24,
                            int(entity["switch_y"]) - 24,
                        )
                        goal_positions.append(exit_switch_pos)
                elif entity_type == 11:  # Standalone exit switch
                    exit_switch_pos = (int(entity["x"]) - 24, int(entity["y"]) - 24)
                    goal_positions.append(exit_switch_pos)

            # Build graph
            level_data = {"tiles": tiles, "entities": []}
            result = graph_builder.build_graph(
                level_data=level_data,
                ninja_pos=start_pos,
                filter_by_reachability=True,
            )
            adjacency = result.get("adjacency", {})

            if not adjacency:
                logger.warning("  Empty graph, skipping")
                continue

            logger.info(f"  Graph: {len(adjacency)} nodes")
            logger.info(f"  Start: {start_pos}, Goals: {goal_positions}")

            # Generate predictions
            ninja_state = torch.zeros(40, dtype=torch.float32)

            with torch.no_grad():
                node_indices, positions_all_heads, metadata = (
                    predictor.forward_from_adjacency(
                        adjacency=adjacency,
                        start_pos=start_pos,
                        goal_positions=goal_positions,
                        ninja_state=ninja_state,
                        temperature=1.0,
                        device=args.device,
                    )
                )

            # Analyze each head
            metrics_all_heads = []
            for head_idx, positions in enumerate(positions_all_heads):
                metrics = analyze_path_quality(
                    positions, adjacency, start_pos, goal_positions
                )
                metrics_all_heads.append(metrics)

                logger.info(
                    f"  Head {head_idx + 1}: {metrics['num_waypoints']} waypoints, "
                    f"{metrics['valid_nodes_ratio']:.1%} valid, "
                    f"{metrics['connectivity_ratio']:.1%} connected, "
                    f"goal_dist={metrics['goal_proximity']:.1f}px"
                )

            # Visualize
            viz_path = output_path / f"{map_file.stem}_analysis.png"
            visualize_with_analysis(
                map_file,
                positions_all_heads,
                adjacency,
                start_pos,
                goal_positions,
                metrics_all_heads,
                viz_path,
            )
            logger.info(f"  Saved visualization: {viz_path}")

            # Store results
            all_metrics.append(
                {
                    "map_name": map_file.stem,
                    "graph_nodes": len(adjacency),
                    "heads": metrics_all_heads,
                }
            )

        except Exception as e:
            logger.error(f"  Failed to analyze {map_file.name}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    # Generate summary report
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 80)

    if all_metrics:
        # Aggregate metrics across all maps and heads
        all_valid_ratios = []
        all_connectivity_ratios = []
        all_start_proximities = []
        all_goal_proximities = []

        for map_result in all_metrics:
            for head_metrics in map_result["heads"]:
                all_valid_ratios.append(head_metrics["valid_nodes_ratio"])
                all_connectivity_ratios.append(head_metrics["connectivity_ratio"])
                all_start_proximities.append(head_metrics["start_proximity"])
                all_goal_proximities.append(head_metrics["goal_proximity"])

        logger.info(f"Maps analyzed: {len(all_metrics)}")
        logger.info(f"Average valid nodes ratio: {np.mean(all_valid_ratios):.1%}")
        logger.info(
            f"Average connectivity ratio: {np.mean(all_connectivity_ratios):.1%}"
        )
        logger.info(f"Average start proximity: {np.mean(all_start_proximities):.1f}px")
        logger.info(f"Average goal proximity: {np.mean(all_goal_proximities):.1f}px")

        # Save detailed report
        report_path = output_path / "analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "num_maps": len(all_metrics),
                        "avg_valid_nodes_ratio": float(np.mean(all_valid_ratios)),
                        "avg_connectivity_ratio": float(
                            np.mean(all_connectivity_ratios)
                        ),
                        "avg_start_proximity": float(np.mean(all_start_proximities)),
                        "avg_goal_proximity": float(np.mean(all_goal_proximities)),
                    },
                    "per_map": all_metrics,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved detailed report: {report_path}")

    logger.info("=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
