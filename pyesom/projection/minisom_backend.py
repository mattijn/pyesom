"""Train / inspect maps via the `minisom` package (core dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np


def import_minisom():  # pragma: no cover - exercised only when backend=minisom
    try:
        from minisom import MiniSom
    except ImportError as e:
        raise ImportError(
            "minisom is a required dependency. Install with:\n"
            "  pip install 'minisom>=2.3'"
        ) from e
    return MiniSom


def train_minisom(
    data: np.ndarray,
    mapsize: tuple[int, int],
    *,
    sigma: float | None,
    learning_rate: float,
    iterations: int,
    random_seed: int | None,
    kwargs: dict[str, Any],
) -> Any:
    """Build and train a MiniSom instance."""
    MiniSom = import_minisom()
    x, y = mapsize
    n_features = data.shape[1]
    if sigma is None:
        sigma = max(x, y) / 2.0
    som = MiniSom(
        x, y, n_features,
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=random_seed,
        **kwargs,
    )
    if data.shape[1] >= 2 and len(data) >= 2:
        som.pca_weights_init(data)
    else:
        som.random_weights_init(data)
    som.train_random(data, int(iterations))
    return som


def weights_grid(som: Any) -> np.ndarray:
    """``(rows, cols, n_features)`` from a trained MiniSom."""
    return np.asarray(som.get_weights(), dtype=np.float64)
