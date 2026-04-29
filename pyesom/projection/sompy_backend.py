"""Train / inspect maps via the `sompy` package (optional dependency).

PyPI ``sompy`` wheels may fail to import; install from GitHub::

    pip install 'sompy @ git+https://github.com/sevamoo/SOMPY.git'

Requires numpy/scipy/scikit-learn (sompy runtime deps).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyesom.projection.lattice import reshape_flat_codebook


def import_sompy_factory():  # pragma: no cover - exercised only when backend=sompy
    import numpy as np

    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]  # SomPy legacy (NumPy < 2)

    try:
        from sompy import SOMFactory
    except ImportError as e:
        raise ImportError(
            "backend='sompy' requires the sompy package. Install with:\n"
            "  pip install 'sompy @ git+https://github.com/sevamoo/SOMPY.git'\n"
            "(PyPI 'sompy' may fail with ImportError.)"
        ) from e
    return SOMFactory


def default_epoch_budget(
    mapsize: tuple[int, int], n_samples: int, initialization: str
) -> int:
    """Match rough+fine rough+fine epoch counts used by sompy ``train()`` defaults."""
    rows, cols = mapsize
    mn = min(rows, cols)
    nnodes = rows * cols
    max_s = max(rows, cols)
    if mn == 1:
        mpd = float(nnodes * 10) / float(max(n_samples, 1))
    else:
        mpd = float(nnodes) / float(max(n_samples, 1))

    rough = int(np.ceil(30.0 * mpd))
    if initialization == "random":
        fine = int(np.ceil(50.0 * mpd))
    else:
        fine = int(np.ceil(40.0 * mpd))
    return max(rough + fine, 1)


def train_sompy(
    data: np.ndarray,
    mapsize: tuple[int, int],
    *,
    iterations: int,
    random_seed: int | None,
    factory_kw: dict[str, Any],
) -> Any:
    """Build and train a rectangular planar ``sompy.SOM``."""
    SOMFactory = import_sompy_factory()
    data = np.asarray(data, dtype=np.float64)
    defaults: dict[str, Any] = {
        "normalization": None,
        "lattice": "rect",
        "mapshape": "planar",
        "training": "batch",
        "initialization": "pca",
        "neighborhood": "gaussian",
    }
    merged = {**defaults, **factory_kw}
    init_mode = str(merged.get("initialization", "pca"))
    budget = default_epoch_budget(mapsize, len(data), init_mode)
    factor = max(float(iterations) / float(budget), 1e-6)

    if random_seed is not None:
        np.random.seed(int(random_seed))

    som = SOMFactory.build(data, mapsize=list(mapsize), **merged)
    som.train(verbose=None, train_len_factor=factor, n_job=1)
    return som


def weights_grid(som: Any) -> np.ndarray:
    """``(rows, cols, n_features)`` from sompy ``codebook``."""
    rows = int(som.codebook.mapsize[0])
    cols = int(som.codebook.mapsize[1])
    mat = np.asarray(som.codebook.matrix, dtype=np.float64)
    return reshape_flat_codebook(mat, rows, cols)
