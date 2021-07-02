import numpy as np
from matplotlib import patches


def draw_bbox(
        ax,
        bbox: dict,
        category: str,
        color: str = 'black',
        background_color: str = 'white',
    ):
    """Draw bounding box on matplotlib axes.

    Args:
        ax (matplotlib.axes):
        bbox (dict):
        color (str):
        category (str):
    """

    linewidth = 2 # TODO
    alpha = 0.7

    bbox_patch = patches.Rectangle(
        xy=(bbox['xmin'], bbox['ymin']),
        width=bbox['xmax'] - bbox['xmin'],
        height=bbox['ymax'] - bbox['ymin'],
        linewidth=linewidth,
        alpha=alpha,
        linestyle='dashed',
        edgecolor='white',
        facecolor='none',
    )

    ax.add_patch(bbox_patch)
    ax.text(
        x=bbox['xmin'],
        y=bbox['ymin'] - 8,
        s=category,
        color=color,
        size=10,
        backgroundcolor=background_color,
    )

    return ax


def draw_polygons(
        ax,
        polygons: list,
        color: str = 'white',
    ):
    """Draw (segmentation) mask on matplotlib axes.

    Args:
        ax (matplotlib.axes):
        polygons (list):
        color (str):
    """

    for polygon in polygons:
        contour_patch = patches.Polygon(np.array(polygon).reshape(-1, 2), facecolor='none', edgecolor=color)
        mask_patch = patches.Polygon(np.array(polygon).reshape(-1, 2), facecolor=color, alpha=0.5)

        ax.add_patch(contour_patch)
        ax.add_patch(mask_patch)

    return ax