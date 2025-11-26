#!/usr/bin/env python3
"""Training script for graph-based path predictor network.

This script trains the GraphPathPredictor using GNN + Pointer Networks
to predict paths as sequences of discrete graph nodes from expert demonstrations.

The graph-based approach ensures geometric validity by constraining predictions
to valid graph nodes (no predictions in walls).

Usage:
    # Train on full dataset
    python scripts/train_path_predictor.py \
        --replay-dir /path/to/replays \
        --output-dir graph-path-model \
        --num-epochs 100 \
        --batch-size 32 \
        --device cuda
    
    # Train with visualization
    python scripts/train_path_predictor.py \
        --replay-dir /path/to/replays \
        --output-dir graph-path-model \
        --num-epochs 50 \
        --batch-size 32 \
        --device cuda \
        --visualize-after-training \
        --test-maps-dir ../nclone/nclone/maps/test-maps \
        --num-viz-samples 10
"""

import argparse
import logging
import sys
import random
from pathlib import Path
import torch
import json
import numpy as np

# Set matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
nclone_root = project_root.parent / "nclone"
sys.path.insert(0, str(nclone_root))

from npp_rl.path_prediction.graph_path_predictor import create_graph_path_predictor
from npp_rl.path_prediction.graph_trainer import GraphPathPredictorTrainer
from npp_rl.path_prediction.replay_dataset import PathReplayDataset
from npp_rl.rendering.matplotlib_tile_renderer import (
    render_tiles_to_axis,
    render_mines_to_axis,
)
from nclone.nplay_headless import NPlayHeadless
from nclone.graph.reachability.graph_builder import GraphBuilder
from nclone.graph.level_data import extract_start_position_from_map_data
from nclone.constants.physics_constants import MAP_TILE_HEIGHT, MAP_TILE_WIDTH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train graph-based path predictor from replay demonstrations"
    )

    # Data arguments
    parser.add_argument(
        "--replay-dir",
        type=str,
        required=True,
        help="Directory containing .replay files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--max-replays",
        type=int,
        default=None,
        help="Maximum number of replays to load (for debugging)",
    )

    # Model arguments
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gcn",
        choices=["gcn", "gat"],
        help="GNN encoder type (gcn or gat)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for GNN",
    )
    parser.add_argument(
        "--num-gnn-layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--num-path-candidates",
        type=int,
        default=4,
        help="Number of diverse path candidates to generate",
    )
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=20,
        help="Maximum waypoints per path",
    )
    parser.add_argument(
        "--node-feature-dim",
        type=int,
        default=8,
        help="Dimension of node features",
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cuda or cpu)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Loss weights
    parser.add_argument(
        "--node-loss-weight",
        type=float,
        default=1.0,
        help="Weight for node classification loss",
    )
    parser.add_argument(
        "--connectivity-loss-weight",
        type=float,
        default=0.5,
        help="Weight for connectivity loss",
    )
    parser.add_argument(
        "--diversity-loss-weight",
        type=float,
        default=0.3,
        help="Weight for diversity loss",
    )

    # Dataset arguments
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest for validation)",
    )
    parser.add_argument(
        "--waypoint-interval",
        type=int,
        default=5,
        help="Extract waypoint every N frames",
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize-after-training",
        action="store_true",
        help="Generate visualizations after training completes",
    )
    parser.add_argument(
        "--test-maps-dir",
        type=str,
        default=None,
        help="Directory containing test maps for visualization",
    )
    parser.add_argument(
        "--num-viz-samples",
        type=int,
        default=10,
        help="Number of samples to visualize",
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def visualize_graph_predictions(
    predictor,
    test_maps_dir,
    output_dir,
    num_samples,
    device,
):
    """Visualize predictions on test maps.

    Args:
        predictor: Trained GraphPathPredictor
        test_maps_dir: Directory with test map files
        output_dir: Output directory for visualizations
        num_samples: Number of maps to visualize
        device: Device for inference
    """
    logger.info("=" * 80)
    logger.info("Generating visualizations on test maps")
    logger.info("=" * 80)

    test_maps_path = Path(test_maps_dir)
    if not test_maps_path.exists():
        logger.warning(f"Test maps directory not found: {test_maps_dir}")
        return
    # get all map files in directory (they have no extension)
    map_files = [f for f in test_maps_path.iterdir() if f.is_file() and f.suffix == ""]
    map_files.sort()

    # Limit number of samples
    map_files = map_files[:num_samples]
    logger.info(f"Visualizing {len(map_files)} test maps")

    # Create visualizations directory
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    predictor.eval()

    for map_file in map_files:
        try:
            logger.info(f"\nVisualizing: {map_file.name}")

            # Load map as binary data
            try:
                with open(map_file, "rb") as f:
                    map_data_bytes = f.read()
                # Convert bytes to list of integers
                map_data = [int(b) for b in map_data_bytes]
            except Exception as e:
                logger.warning(f"Could not read {map_file.name}: {e}, skipping")
                continue

            # Initialize NPlay environment and load map
            nplay = NPlayHeadless(enable_rendering=False)
            nplay.load_map_from_map_data(map_data)

            # Get tile data as dict and convert to numpy array
            tile_dic = nplay.get_tile_data()

            # Convert tile dict to numpy array
            tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
            for (x, y), tile_id in tile_dic.items():
                inner_x = x - 1  # Simulator tiles include 1-tile border
                inner_y = y - 1
                if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
                    tiles[inner_y, inner_x] = int(tile_id)

            # Extract start position from stored map_data
            start_pos = extract_start_position_from_map_data(nplay.current_map_data)
            if start_pos is None:
                logger.warning(f"No start position found in {map_file.name}")
                continue

            # CRITICAL FIX: Extract goal positions (exit switch, exit door) from map entities
            # The model was TRAINED with these, so we MUST pass them during inference!
            goal_positions = []
            exit_switch_pos = None
            exit_door_pos = None

            # Get all entities from the environment
            all_entities = nplay.get_all_entities()
            for entity in all_entities:
                entity_type = entity.get("type", -1)
                # Entity type 3 = Exit door (has switch), type 11 = Exit switch alone
                if entity_type == 3:  # Exit door entity (includes switch position)
                    # Exit door position - convert from world coords to tile data coords
                    exit_door_pos = (int(entity["x"]) - 24, int(entity["y"]) - 24)
                    goal_positions.append(exit_door_pos)
                    # Switch position is stored separately in entity data
                    if "switch_x" in entity and "switch_y" in entity:
                        exit_switch_pos = (
                            int(entity["switch_x"]) - 24,
                            int(entity["switch_y"]) - 24,
                        )
                        goal_positions.append(exit_switch_pos)
                elif entity_type == 11:  # Standalone exit switch
                    exit_switch_pos = (int(entity["x"]) - 24, int(entity["y"]) - 24)
                    goal_positions.append(exit_switch_pos)

            logger.info(
                f"  Extracted {len(goal_positions)} goal positions: {goal_positions}"
            )

            # Build graph with proper LevelData dict
            # GraphBuilder expects level_data dict with 'tiles' as numpy array
            graph_builder = GraphBuilder()
            level_data = {
                "tiles": tiles,
                "entities": [],  # No entities needed for basic graph building
            }
            # CRITICAL FIX: Use filter_by_reachability=True for consistency with training!
            result = graph_builder.build_graph(
                level_data=level_data,
                ninja_pos=start_pos,
                filter_by_reachability=True,  # MATCH TRAINING!
            )
            adjacency = result.get("adjacency", {})
            spatial_hash = result.get("spatial_hash")

            if not adjacency:
                logger.warning(f"Empty graph for {map_file.name}")
                continue

            logger.info(f"  Graph: {len(adjacency)} nodes")

            # Debug logging: Validate coordinate system
            logger.info(f"=== COORDINATE VALIDATION for {map_file.name} ===")
            logger.info(f"  Start position (tile coords): {start_pos}")
            logger.info(f"  Goal positions (tile coords): {goal_positions}")
            logger.info(f"  Graph node count: {len(adjacency)}")
            if adjacency:
                node_xs = [p[0] for p in adjacency.keys()]
                node_ys = [p[1] for p in adjacency.keys()]
                logger.info(f"  Graph node X range: [{min(node_xs)}, {max(node_xs)}]")
                logger.info(f"  Graph node Y range: [{min(node_ys)}, {max(node_ys)}]")

            # Create minimal ninja_state (zeros for visualization - model trained with real states)
            # Using zeros is suboptimal but better than None which breaks fusion
            ninja_state = torch.zeros(40, dtype=torch.float32)

            # Generate predictions WITH goal positions and ninja state for proper fusion!
            with torch.no_grad():
                node_indices, positions_all_heads, metadata = (
                    predictor.forward_from_adjacency(
                        adjacency=adjacency,
                        start_pos=start_pos,  # Pass start position for fusion
                        goal_positions=goal_positions,  # CRITICAL: Pass goals for fusion!
                        ninja_state=ninja_state,  # Pass ninja state (zeros) for fusion
                        temperature=1.0,
                        device=device,
                    )
                )

            logger.info(f"  Predicted {len(positions_all_heads)} path candidates")

            # Debug logging: Validate predictions
            for head_idx, positions in enumerate(positions_all_heads):
                logger.info(f"  Head {head_idx}: {len(positions)} waypoints")
                if positions:
                    logger.info(f"    First waypoint: {positions[0]}")
                    logger.info(f"    Last waypoint: {positions[-1]}")
                    # Check if positions are valid graph nodes
                    valid_nodes = sum(1 for p in positions if p in adjacency)
                    logger.info(
                        f"    Valid graph nodes: {valid_nodes}/{len(positions)} ({100 * valid_nodes / len(positions):.1f}%)"
                    )

            # Check if we got valid predictions
            if not positions_all_heads or all(len(p) == 0 for p in positions_all_heads):
                logger.warning(f"No valid path predictions for {map_file.name}")
                continue

            # Create visualization
            fig, ax = plt.subplots(figsize=(15, 10))

            # Render level - pass tile_dic (dictionary), not tiles (numpy array)
            render_tiles_to_axis(ax, tile_dic)
            render_mines_to_axis(ax, nplay.get_all_mine_data_for_visualization())

            # COORDINATE SYSTEM ALIGNMENT:
            # - tile_dic includes 1-tile (24px) border, renders at grid_coord * 24
            # - Graph nodes are built from inner tiles array (no border), in inner coordinate space
            # - Need to offset graph nodes by +24px to align with rendered tiles that include border
            BORDER_OFFSET = 24

            # Draw graph nodes (small gray dots) with border offset
            node_positions = list(adjacency.keys())
            if node_positions:
                node_xs = [pos[0] + BORDER_OFFSET for pos in node_positions]
                node_ys = [pos[1] + BORDER_OFFSET for pos in node_positions]
                ax.scatter(node_xs, node_ys, c="gray", s=1, alpha=0.3, zorder=2)

            # Draw start position with border offset
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

            # Draw predicted paths (different colors for each head)
            colors = ["red", "blue", "orange", "purple"]
            paths_drawn = 0
            for head_idx, positions in enumerate(positions_all_heads):
                if not positions or len(positions) == 0:
                    continue

                # Remove duplicates while preserving order
                unique_positions = []
                seen = set()
                for pos in positions:
                    if pos not in seen:
                        unique_positions.append(pos)
                        seen.add(pos)

                if len(unique_positions) < 2:
                    continue

                # Apply border offset to align with rendered tiles
                xs = [pos[0] + BORDER_OFFSET for pos in unique_positions]
                ys = [pos[1] + BORDER_OFFSET for pos in unique_positions]

                # Draw path
                ax.plot(
                    xs,
                    ys,
                    color=colors[head_idx % len(colors)],
                    linewidth=2,
                    alpha=0.7,
                    marker="o",
                    markersize=4,
                    label=f"Path {head_idx + 1} ({len(unique_positions)} nodes)",
                    zorder=5,
                )
                paths_drawn += 1

            # Log if no paths were drawable
            if paths_drawn == 0:
                logger.warning(
                    f"No drawable paths for {map_file.name} (all paths < 2 unique nodes)"
                )

            ax.set_title(f"{map_file.stem}\nGraph: {len(adjacency)} nodes", fontsize=14)
            ax.legend(loc="upper right")
            ax.axis("equal")

            # Coordinate system:
            # - Tiles from get_tile_data() have grid coords (0,0) to (43,23) including 1-tile border
            # - These get rendered at pixel coords (0,0) to (1032,552) [43*24, 23*24]
            # - Graph nodes are in pixel/world space which also includes the 24px border
            # - So axis limits should cover: 0 to (MAP_TILE_WIDTH+2)*24 for full map with borders
            # Standard map: 42 inner tiles + 2 border tiles = 44 tiles wide = 1056px
            # Standard map: 23 inner tiles + 2 border tiles = 25 tiles tall = 600px
            # But tile_dic goes (0-43, 0-23) = 44x24 tiles = 1056x576 pixels
            ax.set_xlim(0, 44 * 24)  # 1056 pixels - full width including borders
            ax.set_ylim(0, 25 * 24)  # 600 pixels - full height including borders
            ax.invert_yaxis()

            # Save
            output_path = viz_dir / f"{map_file.stem}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"  Saved visualization to {output_path}")

        except Exception as e:
            logger.error(f"Failed to visualize {map_file.name}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    logger.info("=" * 80)
    logger.info(f"Visualizations saved to {viz_dir}")
    logger.info("=" * 80)


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    logger.info("=" * 80)
    logger.info("Graph-Based Path Predictor Training")
    logger.info("=" * 80)
    logger.info(f"Replay directory: {args.replay_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"GNN type: {args.gnn_type.upper()}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 80)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Load dataset
    logger.info("\nLoading replay dataset...")
    full_dataset = PathReplayDataset(
        replay_dir=args.replay_dir,
        waypoint_interval=args.waypoint_interval,
        max_replays=args.max_replays,
        node_feature_dim=16,  # Enhanced node features with start/goal context
    )

    logger.info(f"Loaded {len(full_dataset)} samples")

    # Split into train/val
    if args.train_split < 1.0:
        train_size = int(len(full_dataset) * args.train_split)
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info("No validation split (train_split=1.0)")

    # Create model with multimodal fusion
    logger.info("\nCreating model with multimodal fusion...")
    model_config = {
        "node_feature_dim": 16,  # Enhanced features with start/goal context
        "hidden_dim": args.hidden_dim,
        "output_dim": args.hidden_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "num_path_candidates": args.num_path_candidates,
        "max_waypoints": args.max_waypoints,
        "gnn_type": args.gnn_type,
        "num_gat_heads": 8,
        "dropout": 0.1,
        # Multimodal fusion parameters
        "use_fusion": True,  # Enable multimodal fusion architecture
        "context_dim": 256,  # Fusion output dimension
        "fusion_hidden_dim": 128,  # Physics encoder hidden dimension
    }

    predictor = create_graph_path_predictor(model_config)

    # Log model statistics
    stats = predictor.get_statistics()
    logger.info("Model Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Create trainer with improved loss weights
    # CRITICAL: Adjusted based on analysis - connectivity and goal reaching are more important
    logger.info("\nCreating trainer...")
    loss_weights = {
        "node": args.node_loss_weight,  # Base node classification
        "connectivity": 2.0,  # INCREASED - penalize invalid paths harder (was 0.5)
        "diversity": 0.1,  # DECREASED - less important initially (was 0.3)
        "start_goal": 3.0,  # INCREASED - emphasize start/goal waypoints (was 1.0)
        "goal_reaching": 2.0,  # INCREASED - paths must reach goals (was 0.5)
    }
    logger.info(f"Loss weights: {loss_weights}")

    trainer = GraphPathPredictorTrainer(
        predictor=predictor,
        learning_rate=args.learning_rate,
        device=args.device,
        loss_weights=loss_weights,
    )

    # Train
    logger.info("\nStarting training...")
    training_stats = trainer.train_offline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_dir=args.output_dir,
        save_every=args.save_every,
    )

    # Save training statistics
    stats_path = output_path / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "train_losses": training_stats["train_losses"],
                "val_losses": training_stats["val_losses"],
                "best_val_loss": training_stats["best_val_loss"],
                "epochs_trained": training_stats["epochs_trained"],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved training statistics to {stats_path}")

    # Visualize if requested
    if args.visualize_after_training and args.test_maps_dir:
        visualize_graph_predictions(
            predictor=predictor,
            test_maps_dir=args.test_maps_dir,
            output_dir=args.output_dir,
            num_samples=args.num_viz_samples,
            device=args.device,
        )

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete")
    logger.info(f"Best validation loss: {training_stats['best_val_loss']:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
