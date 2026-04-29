"""Trained SOM lattice — analysis from weights only (trainer-agnostic)."""

from __future__ import annotations

import numpy as np

from pyesom.topology.umatrix import compute_umatrix


def reshape_flat_codebook(
    flat: np.ndarray, grid_rows: int, grid_cols: int
) -> np.ndarray:
    """
    Reshape a row-major codebook ``(n_nodes, n_features)`` to ``(rows, cols, n_features)``.

    Uses C-order indexing: node ``k`` maps to ``(k // cols, k % cols)``.
    Same layout as MiniSom ``get_weights()`` and TorchSOM ``weights`` when read as numpy.
    """
    flat = np.asarray(flat)
    if flat.ndim != 2:
        raise ValueError("flat codebook must have shape (n_nodes, n_features)")
    n_nodes, _ = flat.shape
    if n_nodes != grid_rows * grid_cols:
        raise ValueError(
            f"n_nodes ({n_nodes}) must equal grid_rows * grid_cols ({grid_rows * grid_cols})"
        )
    return flat.reshape(grid_rows, grid_cols, flat.shape[1])


def bmus_from_weights(weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    """BMU grid coordinates ``(n_samples, 2)`` as ``(row, col)`` via Euclidean distance."""
    data = np.asarray(data, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    if W.ndim != 3:
        raise ValueError("weights must have shape (x, y, n_features)")
    x, y, d = W.shape
    Wf = W.reshape(-1, d)
    d2 = (
        np.sum(data * data, axis=1, keepdims=True)
        + np.sum(Wf * Wf, axis=1, keepdims=False)
        - 2.0 * (data @ Wf.T)
    )
    idx = np.argmin(d2, axis=1).astype(np.int64)
    rows = idx // y
    cols = idx % y
    return np.column_stack((rows, cols))


class SomLattice:
    """
    Rectangular SOM grid represented only by prototype weights ``(x, y, n_features)``.

    Lightweight helper when you already have a weight tensor (e.g. NumPy export from
    another library) and want pyesom topology methods without constructing :class:`~pyesom.projection.esom.ESOM`.
    If you train inside pyesom, prefer ``ESOM(..., backend="sompy")`` or ``"minisom"``.
    """

    __slots__ = ("_weights",)

    def __init__(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 3:
            raise ValueError("weights must have shape (x, y, n_features)")
        self._weights = w

    @property
    def weights(self) -> np.ndarray:
        """Prototype weights ``(rows, cols, features)``."""
        return self._weights

    @property
    def shape(self) -> tuple[int, int, int]:
        x, y, d = self._weights.shape
        return x, y, d

    @classmethod
    def from_esom(cls, esom: object) -> SomLattice:
        """Wrap any trained map that exposes ``weights`` shaped ``(x, y, n_features)`` (e.g. :class:`~pyesom.projection.esom.ESOM`)."""
        w = getattr(esom, "weights", None)
        if w is None:
            raise TypeError("esom must have a .weights ndarray")
        return cls(np.asarray(w, dtype=np.float64))

    @classmethod
    def from_flat(
        cls, flat: np.ndarray, grid_rows: int, grid_cols: int
    ) -> SomLattice:
        """Build from a flattened codebook (e.g. SOMPY ``codebook.matrix``)."""
        return cls(reshape_flat_codebook(flat, grid_rows, grid_cols))

    def u_matrix(self, scaling: str = "sum") -> np.ndarray:
        """Distance-based U-matrix, same normalization as MiniSom ``distance_map``."""
        return compute_umatrix(self._weights, scaling=scaling)

    def bmu_indices(self, data: np.ndarray) -> np.ndarray:
        """BMU grid coordinates ``(n_samples, 2)`` as ``(row, col)``."""
        return bmus_from_weights(self._weights, data)

    def hit_map(self, data: np.ndarray) -> np.ndarray:
        """Hit counts per neuron (same semantics as MiniSom ``activation_response``)."""
        bm = self.bmu_indices(data)
        x, y = self._weights.shape[0], self._weights.shape[1]
        hits = np.zeros((x, y), dtype=np.int64)
        np.add.at(hits, (bm[:, 0], bm[:, 1]), 1)
        return hits

    def component_plane(self, data: np.ndarray, feature_idx: int) -> np.ndarray:
        """Mean value of ``feature_idx`` over samples mapped to each neuron."""
        data = np.asarray(data, dtype=np.float64)
        rx, ry = self._weights.shape[0], self._weights.shape[1]
        bm = self.bmu_indices(data)
        acc = np.zeros((rx, ry), dtype=np.float64)
        cnt = np.zeros((rx, ry), dtype=np.float64)
        fi = int(feature_idx)
        for k in range(len(data)):
            r, c = int(bm[k, 0]), int(bm[k, 1])
            acc[r, c] += data[k, fi]
            cnt[r, c] += 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            plane = np.where(cnt > 0, acc / cnt, np.nan)
        return plane
