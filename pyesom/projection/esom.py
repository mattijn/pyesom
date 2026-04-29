"""Emergent SOM — trainer-pluggable large-grid wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from minisom import MiniSom

from pyesom.projection.lattice import bmus_from_weights
from pyesom.topology.umatrix import compute_umatrix

_SUPPORTED_BACKENDS = frozenset({"minisom", "sompy"})


class ESOM:
    """
    Emergent Self-Organizing Map (large grid, topology-forward interpretation).

    Training is selected via ``backend`` (default ``minisom``). Helpers cover
    U-matrix, hit counts, BMUs, and component planes.
    """

    def __init__(
        self,
        x: int,
        y: int,
        n_features: int,
        *,
        backend: str = "minisom",
        sigma: float | None = None,
        learning_rate: float = 0.5,
        random_seed: int | None = None,
        sompy_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        x, y
            Grid dimensions (rows × cols).
        n_features
            Expected input dimensionality (checked against ``fit(data)``).
        backend
            ``\"minisom\"`` (default): sequential MiniSom training.

            ``\"sompy\"``: batch SOMPY trainer (:mod:`sompy`). Requires installing sompy,
            e.g. ``pip install 'sompy @ git+https://github.com/sevamoo/SOMPY.git'``.
            Extra keyword arguments for ``SOMFactory.build`` go in ``sompy_kwargs``.
        sigma
            Initial neighbourhood radius (``minisom`` only); default ``max(x, y) / 2``.
        learning_rate
            Passed to MiniSom when ``backend=\"minisom\"``.
        random_seed
            Used where the backend supports it (MiniSom ``random_seed``; NumPy seed before sompy train).
        sompy_kwargs
            Forwarded to ``sompy.SOMFactory.build`` when ``backend=\"sompy``.
        **kwargs
            Additional MiniSom constructor arguments when ``backend=\"minisom\"``.
        """
        self._shape = (int(x), int(y))
        self._n_features = int(n_features)
        b = backend.strip().lower().replace("-", "_")
        if b not in _SUPPORTED_BACKENDS:
            opts = ", ".join(sorted(_SUPPORTED_BACKENDS))
            raise ValueError(f"unsupported backend {backend!r}; use one of: {opts}")

        self._backend = b
        rs = kwargs.pop("random_seed", random_seed)
        self._random_seed = rs

        self._som: MiniSom | None = None
        self._sompy = None
        self._sompy_factory_kw = dict(sompy_kwargs or {})

        if b == "minisom":
            if sigma is None:
                sigma = max(x, y) / 2.0
            self._som = MiniSom(
                x,
                y,
                n_features,
                sigma=sigma,
                learning_rate=learning_rate,
                random_seed=rs,
                **kwargs,
            )
        elif b == "sompy":
            if kwargs:
                names = ", ".join(sorted(kwargs.keys()))
                raise TypeError(
                    f"unexpected keyword arguments for backend='sompy': {names}"
                )

    @property
    def backend(self) -> str:
        """Active training backend name."""
        return self._backend

    @property
    def weights(self) -> np.ndarray:
        """Prototype weights ``(x, y, n_features)``."""
        if self._backend == "sompy":
            if self._sompy is None:
                raise RuntimeError("call fit() before accessing weights")
            from pyesom.projection import sompy_backend

            return sompy_backend.weights_grid(self._sompy)
        assert self._som is not None
        return np.asarray(self._som.get_weights())

    def fit(self, data: np.ndarray, iterations: int | None = None) -> ESOM:
        """Train on ``data`` (``n_samples``, ``n_features``)."""
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self._n_features:
            raise ValueError(
                f"data must have shape (n_samples, {self._n_features}), got {data.shape}"
            )
        if iterations is None:
            iterations = max(10_000, 50 * len(data))

        if self._backend == "minisom":
            assert self._som is not None
            if data.shape[1] >= 2 and len(data) >= 2:
                self._som.pca_weights_init(data)
            else:
                self._som.random_weights_init(data)
            self._som.train_random(data, int(iterations))
            return self

        if self._backend == "sompy":
            from pyesom.projection import sompy_backend

            self._sompy = sompy_backend.train_sompy(
                data,
                self._shape,
                iterations=int(iterations),
                random_seed=self._random_seed,
                factory_kw=self._sompy_factory_kw,
            )
            return self

        raise RuntimeError(f"unhandled backend {self._backend!r}")

    def u_matrix(self) -> np.ndarray:
        """Distance-based U-matrix ``(x, y)``, MiniSom-compatible normalization."""
        return compute_umatrix(self.weights, scaling="sum")

    def hit_map(self, data: np.ndarray) -> np.ndarray:
        """Hit counts per neuron."""
        data = np.asarray(data, dtype=np.float64)
        bm = bmus_from_weights(self.weights, data)
        gx, gy = self._shape
        hits = np.zeros((gx, gy), dtype=np.int64)
        np.add.at(hits, (bm[:, 0], bm[:, 1]), 1)
        return hits

    def bmu_indices(self, data: np.ndarray) -> np.ndarray:
        """BMU grid coordinates ``(n_samples, 2)`` as ``(row, col)``."""
        data = np.asarray(data, dtype=np.float64)
        return bmus_from_weights(self.weights, data)

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
