"""Watershed segmentation on U* — optional alternative to flood-fill."""

from __future__ import annotations

import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def watershed_labels(
    landscape: np.ndarray,
    *,
    footprint: tuple[int, int] = (3, 3),
    min_distance: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Marker-controlled watershed on ``landscape`` (e.g. normalized U*).

    Returns a label grid (``-1`` = ridge / background) and the count of markers.
    """
    landscape = np.asarray(landscape, dtype=np.float64)
    lm = peak_local_max(-landscape, footprint=np.ones(footprint), min_distance=min_distance)
    coords = np.column_stack(np.where(lm))
    markers = np.zeros(landscape.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1
    labels = watershed(landscape, markers=markers)
    # boundaries often separate segments — optional dilation step omitted for simplicity
    return labels.astype(np.int32), len(coords)
