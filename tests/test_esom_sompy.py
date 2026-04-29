"""Optional ``backend='sompy'`` smoke test (needs ``sompy`` importable).

Install extra::

    pip install '.[sompy]'

If ``sompy`` fails to import (e.g. NumPy 2 incompatibility in older forks), tests skip.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import sompy  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"sompy not available: {exc}", allow_module_level=True)

from pyesom import ESOM


def test_esom_sompy_train_and_topology():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((80, 4))
    som = ESOM(6, 5, 4, backend="sompy", random_seed=0)
    som.fit(data, iterations=3000)
    assert som.backend == "sompy"
    w = som.weights
    assert w.shape == (6, 5, 4)
    u = som.u_matrix()
    assert u.shape == (6, 5)
    bm = som.bmu_indices(data)
    assert bm.shape == (len(data), 2)
    hits = som.hit_map(data)
    assert int(hits.sum()) == len(data)
