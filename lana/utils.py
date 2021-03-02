"""Utilities"""
from string import ascii_uppercase

import numpy as np
import matplotlib.pyplot as plt


def track_identifiers(tracks):
    """List criteria that identify a track"""
    return [
        identifier
        for identifier in ["Condition", "Sample", "Tissue", "Track_ID", "Source"]
        if identifier in tracks.dropna(axis=1).columns
    ]


def equalize_axis3d(source_ax, zoom=1, target_ax=None):
    """Equalize axis for a mpl3d plot; after
    http://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio"""
    if target_ax == None:
        target_ax = source_ax
    elif zoom != 1:
        print("Zoom ignored when target axis is provided.")
        zoom = 1
    source_extents = np.array(
        [getattr(source_ax, "get_{}lim".format(dim))() for dim in "xyz"]
    )
    target_extents = np.array(
        [getattr(target_ax, "get_{}lim".format(dim))() for dim in "xyz"]
    )
    spread = target_extents[:, 1] - target_extents[:, 0]
    max_spread = max(abs(spread))
    r = max_spread / 2
    centers = np.mean(source_extents, axis=1)
    for center, dim in zip(centers, "xyz"):
        getattr(source_ax, "set_{}lim".format(dim))(
            center - r / zoom, center + r / zoom
        )
    source_ax.set_aspect("equal")


def label_axes():
    """Label panels; after http://stackoverflow.com/questions/25543978/"""
    # TODO: not 3D ready, works only for axis w/o ticks
    for i, ax in enumerate(plt.gcf().axes):
        ax.text(
            -0.075,
            0.96,
            ascii_uppercase[i],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )
