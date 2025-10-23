"""
TensorBoard logging utilities for frame stacking visualization and monitoring.

Provides functions to log stacked frames as images and track frame stacking
configuration parameters for analysis and debugging.
"""

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
import matplotlib.pyplot as plt
import io
from PIL import Image


def log_frame_stack_config(
    writer: SummaryWriter, config: Dict[str, Any], global_step: int = 0
) -> None:
    """
    Log frame stacking configuration to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        config: Configuration dictionary with frame stacking parameters
        global_step: Global training step
    """
    # Log frame stacking scalars
    if "enable_visual_frame_stacking" in config:
        writer.add_scalar(
            "config/frame_stack/visual_enabled",
            int(config["enable_visual_frame_stacking"]),
            global_step,
        )

    if "visual_stack_size" in config:
        writer.add_scalar(
            "config/frame_stack/visual_stack_size",
            config["visual_stack_size"],
            global_step,
        )

    if "enable_state_stacking" in config:
        writer.add_scalar(
            "config/frame_stack/state_enabled",
            int(config["enable_state_stacking"]),
            global_step,
        )

    if "state_stack_size" in config:
        writer.add_scalar(
            "config/frame_stack/state_stack_size",
            config["state_stack_size"],
            global_step,
        )

    if "frame_stack_padding_type" in config:
        # Log padding type as text
        writer.add_text(
            "config/frame_stack/padding_type",
            config["frame_stack_padding_type"],
            global_step,
        )

    # Create a configuration summary text
    frame_stack_summary = []
    if config.get("enable_visual_frame_stacking", False):
        frame_stack_summary.append(
            f"Visual Frame Stacking: Enabled ({config.get('visual_stack_size', 0)} frames)"
        )
    else:
        frame_stack_summary.append("Visual Frame Stacking: Disabled")

    if config.get("enable_state_stacking", False):
        frame_stack_summary.append(
            f"State Stacking: Enabled ({config.get('state_stack_size', 0)} states)"
        )
    else:
        frame_stack_summary.append("State Stacking: Disabled")

    frame_stack_summary.append(
        f"Padding Type: {config.get('frame_stack_padding_type', 'N/A')}"
    )

    writer.add_text(
        "config/frame_stack/summary", "\n".join(frame_stack_summary), global_step
    )


def visualize_stacked_frames(
    stacked_frames: np.ndarray, title: str = "Stacked Frames"
) -> np.ndarray:
    """
    Create a visualization of stacked frames as a grid.

    Args:
        stacked_frames: Stacked frames array with shape:
            - (stack_size, H, W, 1) for grayscale
            - (stack_size, H, W, 3) for RGB
        title: Title for the visualization

    Returns:
        RGB image array ready for TensorBoard logging
    """
    if stacked_frames.ndim == 4:
        stack_size = stacked_frames.shape[0]

        # Remove channel dimension if present
        if stacked_frames.shape[-1] == 1:
            frames = stacked_frames[..., 0]  # (stack_size, H, W)
        else:
            frames = stacked_frames

        # Determine grid layout (prefer horizontal layout for <= 6 frames)
        if stack_size <= 6:
            nrows, ncols = 1, stack_size
        else:
            nrows = (stack_size + 3) // 4
            ncols = min(4, stack_size)

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        fig.suptitle(title, fontsize=16)

        # Flatten axes for easier iteration
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx in range(stack_size):
            ax = axes[idx]
            frame = frames[idx]

            # Normalize for display
            if frame.max() > 1.0:
                frame = frame / 255.0

            ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Frame t-{stack_size - idx - 1}")
            ax.axis("off")

        # Hide unused subplots
        for idx in range(stack_size, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        # Convert plot to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close(fig)

        return img_array

    return stacked_frames


def log_stacked_observations(
    writer: SummaryWriter,
    observations: Dict[str, Any],
    global_step: int,
    max_samples: int = 4,
) -> None:
    """
    Log stacked observations to TensorBoard as images.

    Args:
        writer: TensorBoard SummaryWriter
        observations: Observation dictionary containing stacked frames
        global_step: Global training step
        max_samples: Maximum number of samples to log (from batch)
    """
    # Log stacked player frames
    if "player_frame" in observations:
        player_frames = observations["player_frame"]

        # Convert torch tensor to numpy if needed
        if torch.is_tensor(player_frames):
            player_frames = player_frames.detach().cpu().numpy()

        # Check if frames are stacked (5D: batch, stack, H, W, C)
        if player_frames.ndim == 5:
            batch_size = min(player_frames.shape[0], max_samples)

            for i in range(batch_size):
                stacked_frame = player_frames[i]  # (stack_size, H, W, C)
                vis_img = visualize_stacked_frames(
                    stacked_frame, title=f"Player Frame Stack (Sample {i})"
                )

                # Log to TensorBoard (HWC format)
                writer.add_image(
                    f"observations/player_frame_stack/sample_{i}",
                    vis_img,
                    global_step,
                    dataformats="HWC",
                )

        elif player_frames.ndim == 4:
            # Single frame per sample: (batch, H, W, C)
            batch_size = min(player_frames.shape[0], max_samples)

            for i in range(batch_size):
                frame = player_frames[i]  # (H, W, C)

                # Normalize for display
                if frame.max() > 1.0:
                    frame = frame / 255.0

                # Log to TensorBoard
                if frame.shape[-1] == 1:
                    # Grayscale: convert to (H, W)
                    writer.add_image(
                        f"observations/player_frame/sample_{i}",
                        frame[..., 0],
                        global_step,
                        dataformats="HW",
                    )
                else:
                    # RGB: (H, W, C)
                    writer.add_image(
                        f"observations/player_frame/sample_{i}",
                        frame,
                        global_step,
                        dataformats="HWC",
                    )

    # Log stacked global views
    if "global_view" in observations:
        global_views = observations["global_view"]

        if torch.is_tensor(global_views):
            global_views = global_views.detach().cpu().numpy()

        if global_views.ndim == 5:
            batch_size = min(global_views.shape[0], max_samples)

            for i in range(batch_size):
                stacked_view = global_views[i]
                vis_img = visualize_stacked_frames(
                    stacked_view, title=f"Global View Stack (Sample {i})"
                )

                writer.add_image(
                    f"observations/global_view_stack/sample_{i}",
                    vis_img,
                    global_step,
                    dataformats="HWC",
                )


def log_state_stack_statistics(
    writer: SummaryWriter,
    game_state: np.ndarray,
    global_step: int,
    prefix: str = "observations/game_state",
) -> None:
    """
    Log statistics about stacked game states.

    Args:
        writer: TensorBoard SummaryWriter
        game_state: Game state array, shape:
            - (batch, state_dim) for single state
            - (batch, stack_size, state_dim) for stacked states
        global_step: Global training step
        prefix: Prefix for TensorBoard tags
    """
    if torch.is_tensor(game_state):
        game_state = game_state.detach().cpu().numpy()

    # Check if states are stacked
    if game_state.ndim == 3:
        # Stacked states: (batch, stack_size, state_dim)
        batch_size, stack_size, state_dim = game_state.shape

        writer.add_scalar(f"{prefix}_stack/stack_size", stack_size, global_step)
        writer.add_scalar(f"{prefix}_stack/state_dim", state_dim, global_step)

        # Log statistics for each time step in the stack
        for t in range(stack_size):
            state_t = game_state[:, t, :]  # (batch, state_dim)

            writer.add_scalar(
                f"{prefix}_stack/mean_t_{t}", np.mean(state_t), global_step
            )
            writer.add_scalar(f"{prefix}_stack/std_t_{t}", np.std(state_t), global_step)

        # Log temporal differences (velocity proxy)
        if stack_size > 1:
            state_diff = game_state[:, 1:, :] - game_state[:, :-1, :]
            writer.add_scalar(
                f"{prefix}_stack/temporal_change_mean",
                np.mean(np.abs(state_diff)),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}_stack/temporal_change_std", np.std(state_diff), global_step
            )

    elif game_state.ndim == 2:
        # Single state: (batch, state_dim)
        writer.add_scalar(f"{prefix}/mean", np.mean(game_state), global_step)
        writer.add_scalar(f"{prefix}/std", np.std(game_state), global_step)
