"""Emergent SOM — trainer-pluggable large-grid wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyesom.projection.lattice import bmus_from_weights
from pyesom.topology.umatrix import compute_umatrix

_SUPPORTED_BACKENDS = frozenset({"minisom", "sompy", "torchsom", "intrasom"})


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
        torchsom_kwargs: dict[str, Any] | None = None,
        intrasom_kwargs: dict[str, Any] | None = None,
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

            ``\"torchsom\"``: PyTorch batch trainer (:mod:`torchsom`). Requires
            ``pip install -e ".[torchsom]"`` (PyPI ``torchsom`` + ``torch``).
            Extra keyword arguments for ``SOM.__init__`` go in ``torchsom_kwargs``
            (e.g. ``epochs``, ``batch_size``, ``topology``).

            ``\"intrasom\"``: IntraSOM batch trainer (:mod:`intrasom`). Supports
            **toroidal** and hexagonal grids. Requires ``pip install -e ".[intrasom]"``.
            Defaults: ``mapshape="toroid"``, ``lattice="hexa"``. Extra keyword arguments
            go in ``intrasom_kwargs`` (e.g. ``mapshape``, ``lattice``,
            ``train_rough_len``, ``train_finetune_len``).
        sigma
            Initial neighbourhood radius.  MiniSom default: ``max(x, y) / 2``.
            TorchSOM default: ``1.0``.  Ignored by sompy.
        learning_rate
            Passed to MiniSom or TorchSOM (sompy controls this internally).
        random_seed
            Used where the backend supports it.
        sompy_kwargs
            Forwarded to ``sompy.SOMFactory.build`` when ``backend=\"sompy\"``.
        torchsom_kwargs
            Forwarded to ``torchsom.SOM.__init__`` when ``backend=\"torchsom\"``.
            Supported keys include ``epochs`` (default 10), ``batch_size`` (default 32),
            ``topology``, ``neighborhood_function``, ``distance_function``,
            ``initialization_mode``, etc.
        intrasom_kwargs
            Forwarded to ``intrasom.SOMFactory.build`` and ``som.train`` when
            ``backend=\"intrasom\"``. Supported keys include ``mapshape`` (default
            ``\"toroid\"``), ``lattice`` (default ``\"hexa\"``), ``train_rough_len``,
            ``train_finetune_len``, ``normalization``, ``initialization``, etc.
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
        self._sigma = sigma
        self._learning_rate = learning_rate

        self._som = None
        self._sompy = None
        self._torchsom = None
        self._intrasom = None
        self._sompy_factory_kw = dict(sompy_kwargs or {})
        self._torchsom_factory_kw = dict(torchsom_kwargs or {})
        self._intrasom_factory_kw = dict(intrasom_kwargs or {})

        if b == "minisom":
            self._minisom_extra_kw: dict[str, Any] = dict(kwargs)
        elif b in ("sompy", "torchsom", "intrasom"):
            if kwargs:
                names = ", ".join(sorted(kwargs.keys()))
                raise TypeError(
                    f"unexpected keyword arguments for backend={backend!r}: {names}"
                )
            self._minisom_extra_kw = {}

    @property
    def backend(self) -> str:
        """Active training backend name."""
        return self._backend

    @property
    def weights(self) -> np.ndarray:
        """Prototype weights ``(x, y, n_features)``."""
        if self._backend == "minisom":
            if self._som is None:
                raise RuntimeError("call fit() before accessing weights")
            from pyesom.projection import minisom_backend
            return minisom_backend.weights_grid(self._som)
        if self._backend == "sompy":
            if self._sompy is None:
                raise RuntimeError("call fit() before accessing weights")
            from pyesom.projection import sompy_backend
            return sompy_backend.weights_grid(self._sompy)
        if self._backend == "torchsom":
            if self._torchsom is None:
                raise RuntimeError("call fit() before accessing weights")
            from pyesom.projection import torchsom_backend
            return torchsom_backend.weights_grid(self._torchsom)
        if self._backend == "intrasom":
            if self._intrasom is None:
                raise RuntimeError("call fit() before accessing weights")
            from pyesom.projection import intrasom_backend
            return intrasom_backend.weights_grid(self._intrasom)
        raise RuntimeError(f"unhandled backend {self._backend!r}")

    def fit(
        self,
        data: np.ndarray,
        iterations: int | None = None,
        epochs: int | None = None,
    ) -> ESOM:
        """Train on ``data`` (``n_samples``, ``n_features``).

        Each backend uses its own natural training-duration unit:

        * **minisom** — uses ``iterations`` (individual sample presentations).
          Default: ``max(10_000, 50 × n_samples)``.  ``epochs`` is ignored.
        * **torchsom** — uses ``epochs`` (full passes through the dataset).
          Default: ``torchsom_kwargs["epochs"]`` if set, otherwise ``10``.
          ``iterations`` is ignored.
        * **sompy** — manages its own epoch budget based on the grid/data ratio
          (``train_len_factor=1.0``).  Both ``iterations`` and ``epochs`` are
          ignored; adjust training length via ``sompy_kwargs["train_len_factor"]``.
        * **intrasom** — uses ``epochs`` split evenly between rough and fine-tuning
          phases (default ``10``).  Override via ``intrasom_kwargs["train_rough_len"]``
          and ``intrasom_kwargs["train_finetune_len"]``.  ``iterations`` is ignored.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self._n_features:
            raise ValueError(
                f"data must have shape (n_samples, {self._n_features}), got {data.shape}"
            )

        if self._backend == "minisom":
            if iterations is None:
                iterations = max(10_000, 50 * len(data))
            from pyesom.projection import minisom_backend
            self._som = minisom_backend.train_minisom(
                data,
                self._shape,
                sigma=self._sigma,
                learning_rate=self._learning_rate,
                iterations=int(iterations),
                random_seed=self._random_seed,
                kwargs=self._minisom_extra_kw,
            )
            return self

        if self._backend == "sompy":
            from pyesom.projection import sompy_backend
            self._sompy = sompy_backend.train_sompy(
                data,
                self._shape,
                random_seed=self._random_seed,
                factory_kw=self._sompy_factory_kw,
            )
            return self

        if self._backend == "torchsom":
            kw = dict(self._torchsom_factory_kw)
            _epochs = epochs if epochs is not None else int(kw.pop("epochs", 10))
            sigma = self._sigma if self._sigma is not None else 1.0
            seed = self._random_seed if self._random_seed is not None else 42
            from pyesom.projection import torchsom_backend
            self._torchsom = torchsom_backend.train_torchsom(
                data,
                self._shape,
                epochs=_epochs,
                sigma=sigma,
                learning_rate=self._learning_rate,
                random_seed=seed,
                factory_kw=kw,
            )
            return self

        if self._backend == "intrasom":
            kw = dict(self._intrasom_factory_kw)
            _epochs = epochs if epochs is not None else int(kw.pop("epochs", 10))
            from pyesom.projection import intrasom_backend
            self._intrasom = intrasom_backend.train_intrasom(
                data,
                self._shape,
                epochs=_epochs,
                random_seed=self._random_seed,
                factory_kw=kw,
            )
            # IntraSOM may adjust the grid (e.g. odd→even rows for hexa lattice);
            # sync _shape from the actual trained weights so all derived methods agree.
            actual = intrasom_backend.weights_grid(self._intrasom)
            self._shape = (actual.shape[0], actual.shape[1])
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
