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


def train_sompy(
    data: np.ndarray,
    mapsize: tuple[int, int],
    *,
    random_seed: int | None,
    factory_kw: dict[str, Any],
) -> Any:
    """Build and train a rectangular planar ``sompy.SOM``."""
    SOMFactory = import_sompy_factory()
    data = np.asarray(data, dtype=np.float64)
    kw = dict(factory_kw)
    train_len_factor = float(kw.pop("train_len_factor", 1.0))
    defaults: dict[str, Any] = {
        "normalization": None,
        "lattice": "rect",
        "mapshape": "planar",
        "training": "batch",
        "initialization": "pca",
        "neighborhood": "gaussian",
    }
    merged = {**defaults, **kw}

    if random_seed is not None:
        np.random.seed(int(random_seed))

    som = SOMFactory.build(data, mapsize=list(mapsize), **merged)
    som.train(verbose=None, train_len_factor=train_len_factor, n_job=1)
    return som


def weights_grid(som: Any) -> np.ndarray:
    """``(rows, cols, n_features)`` from sompy ``codebook``."""
    rows = int(som.codebook.mapsize[0])
    cols = int(som.codebook.mapsize[1])
    mat = np.asarray(som.codebook.matrix, dtype=np.float64)
    return reshape_flat_codebook(mat, rows, cols)
