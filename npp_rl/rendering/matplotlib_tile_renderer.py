"""
Matplotlib-based tile and entity rendering for route visualizations.

This module provides efficient rendering of N++ tiles and entities using matplotlib,
adapted from the Cairo-based rendering in nclone.shared_tile_renderer.
"""

import math
from typing import Dict, List, Tuple, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


def group_tiles_by_type(tile_dic: Dict[Tuple[int, int], int]) -> Dict[int, List[Tuple[int, int]]]:
    """Group tiles by type for efficient batch rendering.
    
    Args:
        tile_dic: Dictionary mapping (x, y) coordinates to tile type values
        
    Returns:
        Dictionary mapping tile type to list of coordinates
    """
    tile_groups = {}
    for coords, tile_val in tile_dic.items():
        # Skip empty tiles and glitched tiles (>33)
        if tile_val != 0 and tile_val <= 33:
            if tile_val not in tile_groups:
                tile_groups[tile_val] = []
            tile_groups[tile_val].append(coords)
    return tile_groups


def render_tiles_to_axis(
    ax,
    tile_dic: Dict[Tuple[int, int], int],
    tile_size: float = 24.0,
    tile_color: str = "#808080",
    alpha: float = 1.0,
) -> None:
    """Render tiles to a matplotlib axis.
    
    Args:
        ax: Matplotlib axis to render on
        tile_dic: Dictionary mapping (x, y) grid coordinates to tile type values
        tile_size: Size of each tile in pixels (default 24)
        tile_color: Color for tiles (default gray)
        alpha: Transparency level (default 1.0 for solid)
    """
    # Group tiles by type for efficient rendering
    tile_groups = group_tiles_by_type(tile_dic)
    
    # Render each tile type group
    for tile_type, coords_list in tile_groups.items():
        render_tile_type_group(ax, tile_type, coords_list, tile_size, tile_color, alpha)


def render_tile_type_group(
    ax,
    tile_type: int,
    coords_list: List[Tuple[int, int]],
    tile_size: float,
    tile_color: str,
    alpha: float,
) -> None:
    """Render a group of tiles of the same type.
    
    Args:
        ax: Matplotlib axis to render on
        tile_type: Tile type ID (1-33)
        coords_list: List of (x, y) grid coordinates for this tile type
        tile_size: Size of each tile in pixels
        tile_color: Color for tiles
        alpha: Transparency level
    """
    if tile_type == 1:
        # Full solid tiles - batch render with rectangles
        patches = []
        for x, y in coords_list:
            rect = mpatches.Rectangle(
                (x * tile_size, y * tile_size),
                tile_size,
                tile_size,
                facecolor=tile_color,
                edgecolor=None,
                linewidth=0,
                alpha=alpha,
            )
            patches.append(rect)
        
        # Add all patches at once with consistent zorder
        for patch in patches:
            ax.add_patch(patch)
            patch.set_zorder(1)
    
    elif tile_type < 6:
        # Half tiles (types 2-5) - batch render
        patches = []
        for x, y in coords_list:
            dx = tile_size / 2 if tile_type == 3 else 0
            dy = tile_size / 2 if tile_type == 4 else 0
            w = tile_size if tile_type % 2 == 0 else tile_size / 2
            h = tile_size / 2 if tile_type % 2 == 0 else tile_size
            
            rect = mpatches.Rectangle(
                (x * tile_size + dx, y * tile_size + dy),
                w,
                h,
                facecolor=tile_color,
                edgecolor=None,
                linewidth=0,
                alpha=alpha,
            )
            patches.append(rect)
        
        for patch in patches:
            ax.add_patch(patch)
            patch.set_zorder(1)
    
    else:
        # Complex tiles (slopes, curves, etc.) - render individually
        for x, y in coords_list:
            draw_complex_tile(ax, tile_type, x, y, tile_size, tile_color, alpha)


def draw_complex_tile(
    ax,
    tile_type: int,
    x: int,
    y: int,
    tile_size: float,
    tile_color: str,
    alpha: float,
) -> None:
    """Draw a complex tile shape (slopes, curves, etc.).
    
    Args:
        ax: Matplotlib axis to render on
        tile_type: Tile type ID (6-33)
        x: Grid x coordinate
        y: Grid y coordinate
        tile_size: Size of each tile in pixels
        tile_color: Color for tiles
        alpha: Transparency level
    """
    base_x = x * tile_size
    base_y = y * tile_size
    
    if tile_type < 10:
        # Triangular tiles (types 6-9) - 45 degree slopes
        dx1 = 0
        dy1 = tile_size if tile_type == 8 else 0
        dx2 = 0 if tile_type == 9 else tile_size
        dy2 = tile_size if tile_type == 9 else 0
        dx3 = 0 if tile_type == 6 else tile_size
        dy3 = tile_size
        
        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
        ]
        
        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(1)
    
    elif tile_type < 14:
        # Quarter circle tiles (types 10-13) - convex corners
        dx = tile_size if (tile_type == 11 or tile_type == 12) else 0
        dy = tile_size if (tile_type == 12 or tile_type == 13) else 0
        theta1 = 90 * (tile_type - 10)
        theta2 = 90 * (tile_type - 9)
        
        wedge = mpatches.Wedge(
            (base_x + dx, base_y + dy),
            tile_size,
            theta1,
            theta2,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(wedge)
        wedge.set_zorder(1)
    
    elif tile_type < 18:
        # Inverted quarter circle tiles (types 14-17) - concave corners (quarter pipes)
        # These are full tiles with a circular cutout, so we render as complex polygon
        dx1 = tile_size if (tile_type == 15 or tile_type == 16) else 0
        dy1 = tile_size if (tile_type == 16 or tile_type == 17) else 0
        dx2 = tile_size if (tile_type == 14 or tile_type == 17) else 0
        dy2 = tile_size if (tile_type == 14 or tile_type == 15) else 0
        theta1 = 180 + 90 * (tile_type - 10)
        theta2 = 180 + 90 * (tile_type - 9)
        
        wedge = mpatches.Wedge(
            (base_x + dx2, base_y + dy2),
            tile_size,
            theta1,
            theta2,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(wedge)
        wedge.set_zorder(1)
    
    elif tile_type < 22:
        # Sloped triangular tiles (types 18-21) - mild slopes
        dx1 = 0
        dy1 = tile_size if (tile_type == 20 or tile_type == 21) else 0
        dx2 = tile_size
        dy2 = tile_size if (tile_type == 20 or tile_type == 21) else 0
        dx3 = tile_size if (tile_type == 19 or tile_type == 20) else 0
        dy3 = tile_size / 2
        
        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
        ]
        
        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(1)
    
    elif tile_type < 26:
        # Quadrilateral tiles (types 22-25) - raised mild slopes
        dx1 = 0
        dy1 = tile_size / 2 if (tile_type == 23 or tile_type == 24) else 0
        dx2 = 0 if tile_type == 23 else tile_size
        dy2 = tile_size / 2 if tile_type == 25 else 0
        dx3 = tile_size
        dy3 = (tile_size / 2 if tile_type == 22 else 0) if tile_type < 24 else tile_size
        dx4 = tile_size if tile_type == 23 else 0
        dy4 = tile_size
        
        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
            (base_x + dx4, base_y + dy4),
        ]
        
        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(1)
    
    elif tile_type < 30:
        # Triangular tiles with midpoint (types 26-29) - steep slopes
        dx1 = tile_size / 2
        dy1 = tile_size if (tile_type == 28 or tile_type == 29) else 0
        dx2 = tile_size if (tile_type == 27 or tile_type == 28) else 0
        dy2 = 0
        dx3 = tile_size if (tile_type == 27 or tile_type == 28) else 0
        dy3 = tile_size
        
        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
        ]
        
        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(1)
    
    elif tile_type <= 33:
        # Complex quadrilateral tiles (types 30-33) - raised steep slopes
        dx1 = tile_size / 2
        dy1 = tile_size if (tile_type == 30 or tile_type == 31) else 0
        dx2 = tile_size if (tile_type == 31 or tile_type == 33) else 0
        dy2 = tile_size
        dx3 = tile_size if (tile_type == 31 or tile_type == 32) else 0
        dy3 = tile_size if (tile_type == 32 or tile_type == 33) else 0
        dx4 = tile_size if (tile_type == 30 or tile_type == 32) else 0
        dy4 = 0
        
        vertices = [
            (base_x + dx1, base_y + dy1),
            (base_x + dx2, base_y + dy2),
            (base_x + dx3, base_y + dy3),
            (base_x + dx4, base_y + dy4),
        ]
        
        polygon = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=tile_color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(polygon)
        polygon.set_zorder(1)


def render_mines_to_axis(
    ax,
    mines: List[Dict],
    tile_color: str = "#FF6B6B",
    safe_color: str = "#4ECDC4",
    alpha: float = 0.8,
) -> None:
    """Render mines to a matplotlib axis with visibility culling.
    
    Only renders mines that are visible within the current axis limits
    for better performance on large levels.
    
    Args:
        ax: Matplotlib axis to render on
        mines: List of mine dicts with keys: x, y, state, radius
        tile_color: Color for dangerous (toggled) mines (default red)
        safe_color: Color for safe (untoggled/toggling) mines (default cyan)
        alpha: Transparency level (default 0.8)
    """
    # Get current axis limits for culling
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Add some padding to include mines partially visible at edges
    padding = 50  # pixels
    x_min, x_max = min(xlim) - padding, max(xlim) + padding
    y_min, y_max = min(ylim) - padding, max(ylim) + padding
    
    for mine in mines:
        x = mine["x"]
        y = mine["y"]
        
        # Cull mines outside visible area
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        
        radius = mine["radius"]
        state = mine["state"]
        
        # State 0 = toggled (dangerous), 1 = untoggled (safe), 2 = toggling (safe)
        is_dangerous = (state == 0)
        color = tile_color if is_dangerous else safe_color
        
        # Render mine as circle
        circle = mpatches.Circle(
            (x, y),
            radius,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=alpha,
        )
        ax.add_patch(circle)
        circle.set_zorder(3)  # Above tiles (1) and paths (2), below markers (5+)

