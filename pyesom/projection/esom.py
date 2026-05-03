"""Emergent SOM — trainer-pluggable large-grid wrapper."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from pyesom.projection.lattice import bmus_from_weights
from pyesom.topology.umatrix import compute_umatrix

_SUPPORTED_BACKENDS = frozenset({"minisom", "sompy", "torchsom", "intrasom"})


class ESOM:
    """
    Emergent Self-Organizing Map — train, inspect, and export SOM projections.

    Grid size can be specified in two ways:

    * **Explicit grid** — pass ``x`` and ``y`` (rows × cols).  Use this when you
      want direct control over the map dimensions, e.g. for ESOM visualisation.
    * **Node count** — pass ``n_nodes``.  The grid side is computed automatically
      as ``ceil(sqrt(n_nodes))``.  Use this for the TopoSwarm pipeline where only
      the number of representative prototypes matters, not the grid shape.

    Parameters
    ----------
    x, y
        Grid dimensions (rows × cols).  Mutually exclusive with ``n_nodes``.
    n_features
        Expected input dimensionality.  Optional — inferred from ``fit(data)`` if
        omitted.  When provided, ``fit`` validates the data shape against it.
    n_nodes
        Target number of prototype nodes.  Mutually exclusive with ``x``/``y``.
        Actual node count after training may differ slightly (e.g. IntraSOM rounds
        rows to even for hexagonal lattices).
    backend
        ``"minisom"`` (default), ``"sompy"``, ``"torchsom"``, or ``"intrasom"``.
    sigma
        Initial neighbourhood radius.
    learning_rate
        Passed to MiniSom or TorchSOM.
    random_seed
        Reproducibility seed where the backend supports it.
    sompy_kwargs, torchsom_kwargs, intrasom_kwargs
        Extra keyword arguments forwarded to the respective backend.
    """

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        n_features: int | None = None,
        *,
        n_nodes: int | None = None,
        backend: str = "minisom",
        sigma: float | None = None,
        learning_rate: float = 0.5,
        random_seed: int | None = None,
        sompy_kwargs: dict[str, Any] | None = None,
        torchsom_kwargs: dict[str, Any] | None = None,
        intrasom_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # ── grid size resolution ──────────────────────────────────────
        explicit = (x is not None) or (y is not None)
        if explicit and n_nodes is not None:
            raise ValueError("specify either (x, y) or n_nodes, not both")
        if not explicit and n_nodes is None:
            raise ValueError("specify either (x, y) or n_nodes")
        if n_nodes is not None:
            side = math.ceil(math.sqrt(int(n_nodes)))
            x = y = side
        if x is None or y is None:
            raise ValueError("both x and y must be provided when not using n_nodes")

        # ── backend validation ────────────────────────────────────────
        b = backend.strip().lower().replace("-", "_")
        if b not in _SUPPORTED_BACKENDS:
            opts = ", ".join(sorted(_SUPPORTED_BACKENDS))
            raise ValueError(f"unsupported backend {backend!r}; use one of: {opts}")

        self._shape = (int(x), int(y))
        self._n_features = int(n_features) if n_features is not None else None
        self._backend = b
        self._random_seed = random_seed
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

    # ── properties ───────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Active training backend name."""
        return self._backend

    @property
    def weights(self) -> np.ndarray:
        """Prototype weights as a grid ``(rows, cols, n_features)``."""
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

    @property
    def node_weights(self) -> np.ndarray:
        """Prototype weights as a flat codebook ``(n_nodes, n_features)``.

        This is the TopoSwarm-facing view of the same weights returned by
        :attr:`weights`.  Row ``k`` is the prototype for grid cell
        ``(k // cols, k % cols)``.
        """
        W = self.weights
        return W.reshape(-1, W.shape[-1])

    # ── training ─────────────────────────────────────────────────────

    def fit(
        self,
        data: np.ndarray,
        iterations: int | None = None,
        epochs: int | None = None,
    ) -> ESOM:
        """Train on ``data`` (``n_samples × n_features``).

        Each backend uses its own natural training-duration unit:

        * **minisom** — ``iterations`` (individual sample presentations).
          Default: ``max(10_000, 50 × n_samples)``.
        * **torchsom** — ``epochs`` (full dataset passes). Default: 10.
        * **sompy** — manages its own budget; both args are ignored.
        * **intrasom** — ``epochs`` split evenly between rough and fine-tuning.
          Default: 10.  Override via ``intrasom_kwargs["train_rough_len"]`` etc.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {data.shape}")
        if self._n_features is not None and data.shape[1] != self._n_features:
            raise ValueError(
                f"data has {data.shape[1]} features, expected {self._n_features}"
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

        elif self._backend == "sompy":
            from pyesom.projection import sompy_backend
            self._sompy = sompy_backend.train_sompy(
                data,
                self._shape,
                random_seed=self._random_seed,
                factory_kw=self._sompy_factory_kw,
            )

        elif self._backend == "torchsom":
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

        elif self._backend == "intrasom":
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
            actual = intrasom_backend.weights_grid(self._intrasom)
            self._shape = (actual.shape[0], actual.shape[1])

        else:
            raise RuntimeError(f"unhandled backend {self._backend!r}")

        return self

    # ── analysis ─────────────────────────────────────────────────────

    def u_matrix(self) -> np.ndarray:
        """Distance-based U-matrix ``(rows, cols)``."""
        return compute_umatrix(self.weights, scaling="sum")

    def hit_map(self, data: np.ndarray) -> np.ndarray:
        """Hit counts per neuron ``(rows, cols)``."""
        data = np.asarray(data, dtype=np.float64)
        bm = bmus_from_weights(self.weights, data)
        rows, cols = self._shape
        hits = np.zeros((rows, cols), dtype=np.int64)
        np.add.at(hits, (bm[:, 0], bm[:, 1]), 1)
        return hits

    def bmu_indices(self, data: np.ndarray) -> np.ndarray:
        """BMU grid coordinates ``(n_samples, 2)`` as ``(row, col)``."""
        data = np.asarray(data, dtype=np.float64)
        return bmus_from_weights(self.weights, data)

    def quantization_error(self, data: np.ndarray) -> float:
        """Mean distance from each sample to its BMU prototype vector.

        Lower is better — use this to choose ``n_nodes``: plot QE vs n_nodes
        and pick the elbow where additional nodes stop reducing the error.
        A value below 5 % of the mean pairwise distance in the raw data is
        generally sufficient for TopoSwarm.
        """
        data = np.asarray(data, dtype=np.float64)
        W = self.weights
        bm = bmus_from_weights(W, data)
        bmu_w = W[bm[:, 0], bm[:, 1]]
        return float(np.mean(np.linalg.norm(data - bmu_w, axis=1)))

    def component_plane(self, data: np.ndarray, feature_idx: int) -> np.ndarray:
        """Mean value of ``feature_idx`` over samples mapped to each neuron."""
        data = np.asarray(data, dtype=np.float64)
        rows, cols = self._shape
        bm = bmus_from_weights(self.weights, data)
        acc = np.zeros((rows, cols), dtype=np.float64)
        cnt = np.zeros((rows, cols), dtype=np.float64)
        fi = int(feature_idx)
        np.add.at(acc, (bm[:, 0], bm[:, 1]), data[:, fi])
        np.add.at(cnt, (bm[:, 0], bm[:, 1]), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cnt > 0, acc / cnt, np.nan)

    # ── TopoSwarm bridge ──────────────────────────────────────────────

    def export_npz(
        self,
        path: str | Path,
        data: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> Path:
        """Save the arrays the TopoSwarm Julia pipeline needs.

        Writes a single ``.npz`` file containing:

        * ``node_weights`` — ``(n_nodes, n_features)`` prototype vectors
        * ``hit_map``      — ``(n_nodes,)`` population per node (float64)
        * ``bmu_indices``  — ``(n_samples,)`` flat node index per raw sample
        * ``labels``       — ``(n_samples,)`` int class labels (optional, used by
          ``show_grid`` for visualisation in Julia)

        Parameters
        ----------
        path
            Output path.  ``.npz`` is appended by NumPy if not present.
        data
            The raw training data used to compute BMUs and hit counts.
        labels
            Optional integer class labels per sample.  When provided, Julia's
            ``show_grid`` uses them to label bot positions on the grid.

        Returns
        -------
        Path
            Resolved path of the written file (with ``.npz`` suffix).

        Examples
        --------
        >>> esom = ESOM(n_nodes=1000, backend="intrasom").fit(X)
        >>> esom.export_npz("bridge.npz", X, labels=y)
        """
        data = np.asarray(data, dtype=np.float64)
        _, cols = self._shape
        W = self.node_weights                               # (n_nodes, d)
        hm = self.hit_map(data).ravel().astype(np.float64) # (n_nodes,)
        bm2d = bmus_from_weights(self.weights, data)        # (n_samples, 2)
        bm_flat = (bm2d[:, 0] * cols + bm2d[:, 1]).astype(np.int64)

        out = Path(path)
        arrays: dict = dict(node_weights=W, hit_map=hm, bmu_indices=bm_flat)
        if labels is not None:
            # aggregate per-sample labels to per-node majority class
            lbl = np.asarray(labels, dtype=np.int64)
            n_nodes = W.shape[0]
            node_labels = np.full(n_nodes, -1, dtype=np.int64)
            for k in range(n_nodes):
                mask = bm_flat == k
                if mask.any():
                    vals, counts = np.unique(lbl[mask], return_counts=True)
                    node_labels[k] = vals[counts.argmax()]
            arrays["labels"] = node_labels
        np.savez(out, **arrays)
        resolved = out if out.suffix == ".npz" else out.with_suffix(out.suffix + ".npz")
        return resolved
