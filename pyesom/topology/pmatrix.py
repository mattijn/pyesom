"""P-matrix (data density per neuron) — hit map and PDE-style variants."""

from __future__ import annotations

import numpy as np


def compute_pmatrix(
    weights: np.ndarray,
    data: np.ndarray,
    pareto_fraction: float | None = 0.2013,
) -> np.ndarray:
    """
    Empirical density height at each SOM node.

    * If ``pareto_fraction`` is ``None`` (simplified): each sample is assigned
      to its nearest prototype; the P-height is the hit count per node.
    * Otherwise: for each node, Pareto radius is the distance to the
      ``floor(pareto_fraction * n_samples)``-th nearest data point (by distance
      to the node weight); P-height counts points inside that hypersphere.

    Parameters
    ----------
    weights
        Shape ``(x, y, n_features)``.
    data
        Shape ``(n_samples, n_features)``.
    pareto_fraction
        Used only in the PDE-style path; ``None`` selects the hit-map shortcut.

    Returns
    -------
    ndarray of shape ``(x, y)`` with non-negative density values.
    """
    weights = np.asarray(weights, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (x, y, n_features)")
    if data.ndim != 2 or data.shape[1] != weights.shape[2]:
        raise ValueError("data must be (n_samples, n_features) matching weights")

    x, y, _ = weights.shape
    n = data.shape[0]
    flat_w = weights.reshape(-1, weights.shape[2])

    if pareto_fraction is None:
        # nearest-neuron (Voronoi) counts
        dists = np.linalg.norm(data[:, None, :] - flat_w[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        counts = np.bincount(nearest, minlength=x * y).astype(np.float64)
        return counts.reshape(x, y)

    pf = float(pareto_fraction)
    if not (0 < pf < 1):
        raise ValueError("pareto_fraction must be in (0, 1)")

    rank = min(max(int(np.floor(pf * n)), 1), n) - 1
    out = np.zeros((x, y), dtype=np.float64)

    for i in range(x):
        for j in range(y):
            w = weights[i, j]
            d = np.linalg.norm(data - w, axis=1)
            d_sorted = np.partition(d, rank)
            radius = float(np.sort(d_sorted[: rank + 1])[-1])
            out[i, j] = float(np.count_nonzero(d <= radius))

    return out
