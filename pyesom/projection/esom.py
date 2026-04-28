"""Emergent SOM — large-grid MiniSom wrapper."""

from __future__ import annotations

import numpy as np
from minisom import MiniSom


class ESOM:
    """
    Emergent Self-Organizing Map (large grid, topology-forward interpretation).

    Thin wrapper around MiniSom with helpers for U-matrix, hit counts, BMUs,
    and component planes.
    """

    def __init__(
        self,
        x: int,
        y: int,
        n_features: int,
        *,
        sigma: float | None = None,
        learning_rate: float = 0.5,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        x, y
            Grid dimensions (rows × cols).
        n_features
            Input dimensionality.
        sigma
            Initial neighbourhood radius; default ``max(x, y) / 2``.
        random_seed
            Passed through to MiniSom when supported (``random_seed`` kwarg).
        **kwargs
            Additional MiniSom constructor arguments.
        """
        self._shape = (int(x), int(y))
        self._n_features = int(n_features)
        if sigma is None:
            sigma = max(x, y) / 2.0

        rs = kwargs.pop("random_seed", random_seed)
        self._som = MiniSom(
            x,
            y,
            n_features,
            sigma=sigma,
            learning_rate=learning_rate,
            random_seed=rs,
            **kwargs,
        )

    @property
    def weights(self) -> np.ndarray:
        """Prototype weights ``(x, y, n_features)``."""
        return np.asarray(self._som.get_weights())

    def fit(self, data: np.ndarray, iterations: int | None = None) -> ESOM:
        """Train on ``data`` (``n_samples``, ``n_features``)."""
        data = np.asarray(data, dtype=np.float64)
        if iterations is None:
            iterations = max(10_000, 50 * len(data))
        if data.shape[1] >= 2 and len(data) >= 2:
            self._som.pca_weights_init(data)
        else:
            self._som.random_weights_init(data)
        self._som.train_random(data, int(iterations))
        return self

    def u_matrix(self) -> np.ndarray:
        """Distance-based U-matrix ``(x, y)``, normalized like MiniSom."""
        return np.asarray(self._som.distance_map(scaling="sum"))

    def hit_map(self, data: np.ndarray) -> np.ndarray:
        """Hit counts per neuron (same semantics as MiniSom ``activation_response``)."""
        data = np.asarray(data, dtype=np.float64)
        return np.asarray(self._som.activation_response(data))

    def bmu_indices(self, data: np.ndarray) -> np.ndarray:
        """BMU grid coordinates ``(n_samples, 2)`` as ``(row, col)``."""
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        out = np.empty((n, 2), dtype=np.int64)
        for i in range(n):
            wx, wy = self._som.winner(data[i])
            out[i] = np.int64(wx), np.int64(wy)
        return out

    def component_plane(self, data: np.ndarray, feature_idx: int) -> np.ndarray:
        """Mean value of ``feature_idx`` over samples mapped to each neuron."""
        data = np.asarray(data, dtype=np.float64)
        x, y = self._shape
        bm = self.bmu_indices(data)
        acc = np.zeros((x, y), dtype=np.float64)
        cnt = np.zeros((x, y), dtype=np.float64)
        fi = int(feature_idx)
        for k in range(len(data)):
            r, c = int(bm[k, 0]), int(bm[k, 1])
            acc[r, c] += data[k, fi]
            cnt[r, c] += 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            plane = np.where(cnt > 0, acc / cnt, np.nan)
        return plane
