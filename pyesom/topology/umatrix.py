"""U-matrix from SOM neuron weights (inter-neuron distances on the grid)."""

from __future__ import annotations

import numpy as np


def _fast_norm(v: np.ndarray) -> np.floating:
    return np.sqrt(np.dot(v, v))


def compute_umatrix(
    weights: np.ndarray,
    scaling: str = "sum",
    toroidal: bool = False,
) -> np.ndarray:
    """
    Compute the U-matrix for a rectangular 2-D SOM grid.

    Parameters
    ----------
    weights
        Shape ``(x, y, n_features)`` — neuron prototype vectors.
    scaling
        ``'sum'`` (default): each cell is the normalized sum of distances to
        neighbours (MiniSom default). ``'mean'``: average neighbour distance.
    toroidal
        When ``True``, edges wrap Pac-Man style so every neuron has the same
        number of neighbours. Set this when the SOM was trained on a toroidal
        grid (e.g. intrasom with ``mapshape="toroid"``).

    Returns
    -------
    ndarray of shape ``(x, y)``, values normalized to ``[0, 1]``.
    """
    if weights.ndim != 3:
        raise ValueError("weights must have shape (x, y, n_features)")
    if scaling not in ("sum", "mean"):
        raise ValueError('scaling must be "sum" or "mean"')

    x, y, _ = weights.shape
    um = np.full((x, y, 8), np.nan, dtype=np.float64)

    ii = [[0, -1, -1, -1, 0, 1, 1, 1]] * 2
    jj = [[-1, -1, 0, 1, 1, 1, 0, -1]] * 2

    for xi in range(x):
        for yj in range(y):
            w_2 = weights[xi, yj]
            e = yj % 2 == 0
            for k, (di, dj) in enumerate(zip(ii[e], jj[e])):
                ni, nj = xi + di, yj + dj
                if toroidal:
                    ni, nj = ni % x, nj % y
                    w_1 = weights[ni, nj]
                    um[xi, yj, k] = _fast_norm(w_2 - w_1)
                elif 0 <= ni < x and 0 <= nj < y:
                    w_1 = weights[ni, nj]
                    um[xi, yj, k] = _fast_norm(w_2 - w_1)

    if scaling == "mean":
        u = np.nanmean(um, axis=2)
    else:
        u = np.nansum(um, axis=2)
    mx = np.nanmax(u)
    if mx > 0:
        u = u / mx
    return u.astype(np.float64, copy=False)
