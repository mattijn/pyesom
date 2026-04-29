"""Train / inspect maps via the ``torchsom`` package (optional dependency).

Install with ``pip install -e ".[torchsom]"`` (pulls PyTorch and PyPI ``torchsom``), or
see https://pypi.org/project/torchsom/
"""

from __future__ import annotations

from typing import Any

import numpy as np


def import_torchsom():  # pragma: no cover - exercised only when backend=torchsom
    try:
        from torchsom.core import SOM
    except ImportError as e:
        raise ImportError(
            "backend='torchsom' requires PyTorch and the PyPI package torchsom. Install:\n"
            '  pip install -e ".[torchsom]"\n'
            "or: pip install 'torch>=2' torchsom\n"
            "(see https://pypi.org/project/torchsom/)"
        ) from e
    return SOM


def train_torchsom(
    data: np.ndarray,
    mapsize: tuple[int, int],
    *,
    epochs: int,
    sigma: float,
    learning_rate: float,
    random_seed: int,
    factory_kw: dict[str, Any],
) -> Any:
    """Build and train a TorchSOM SOM instance."""
    import torch

    SOM = import_torchsom()
    x, y = mapsize
    n_features = data.shape[1]

    defaults: dict[str, Any] = {
        "batch_size": 32,
        "initialization_mode": "pca",
        "topology": "rectangular",
        "neighborhood_function": "gaussian",
        "distance_function": "euclidean",
    }
    kw = {**defaults, **factory_kw}

    som = SOM(
        x=x,
        y=y,
        num_features=n_features,
        epochs=epochs,
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=random_seed,
        **kw,
    )

    data_t = torch.as_tensor(data.astype(np.float32), dtype=torch.float32).to(som.device)
    if data.shape[1] >= 2 and len(data) >= 2:
        som.initialize_weights(data_t)  # uses self.initialization_mode

    som.fit(data_t, verbose=False)
    return som


def weights_grid(som: Any) -> np.ndarray:
    """``(rows, cols, n_features)`` from a trained TorchSOM SOM."""
    return som.weights.detach().cpu().numpy().astype(np.float64)
