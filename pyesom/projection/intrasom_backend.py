"""Train / inspect maps via the ``intrasom`` package (optional dependency).

IntraSOM supports toroidal and hexagonal grids, making it the recommended backend
for datasets where border effects matter (e.g. the Atom FCPS benchmark).

Install with::

    pip install -e ".[intrasom]"

or directly: ``pip install intrasom``
"""

from __future__ import annotations

from typing import Any

import numpy as np


def import_intrasom_factory():  # pragma: no cover - exercised only when backend=intrasom
    try:
        from intrasom import SOMFactory
    except ImportError as e:
        raise ImportError(
            "backend='intrasom' requires the intrasom package. Install with:\n"
            "  pip install -e \".[intrasom]\"\n"
            "or: pip install intrasom\n"
            "(see https://github.com/InTRA-USP/IntraSOM)"
        ) from e
    return SOMFactory


def train_intrasom(
    data: np.ndarray,
    mapsize: tuple[int, int],
    *,
    epochs: int,
    random_seed: int | None,
    factory_kw: dict[str, Any],
) -> Any:
    """Build and train an IntraSOM SOM instance.

    ``mapsize`` follows the ESOM convention ``(rows, cols)``.
    IntraSOM expects ``(cols, rows)`` internally; this function converts.
    """
    SOMFactory = import_intrasom_factory()
    data = np.asarray(data, dtype=np.float64)

    if random_seed is not None:
        np.random.seed(int(random_seed))

    kw = dict(factory_kw)
    rough_len = int(kw.pop("train_rough_len", max(1, epochs // 2)))
    fine_len = int(kw.pop("train_finetune_len", epochs - rough_len))

    defaults: dict[str, Any] = {
        "mapshape": "toroid",
        "lattice": "hexa",
        "normalization": None,
        "initialization": "pca",
        "training": "batch",
    }
    merged = {**defaults, **kw}

    rows, cols = mapsize
    lattice = str(merged.get("lattice", "hexa"))
    # IntraSOM hexagonal lattice requires an even number of rows; round up silently.
    if lattice == "hexa" and rows % 2 != 0:
        rows += 1
    # IntraSOM mapsize convention: [cols, rows]
    som = SOMFactory.build(data, mapsize=[cols, rows], **merged)
    som.train(
        bootstrap=False,
        train_rough_len=rough_len,
        train_finetune_len=fine_len,
        save=False,
        summary=False,
    )
    return som


def weights_grid(som: Any) -> np.ndarray:
    """``(rows, cols, n_features)`` from a trained IntraSOM SOM."""
    # som.mapsize is [cols, rows]; neuron_matrix is (n_neurons, n_features)
    cols, rows = int(som.mapsize[0]), int(som.mapsize[1])
    nm = np.asarray(som.neuron_matrix, dtype=np.float64)
    return nm.reshape(rows, cols, -1)
