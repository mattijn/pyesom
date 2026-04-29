"""Optional ``backend='intrasom'`` smoke test (needs ``intrasom`` importable).

Install extra::

    pip install '.[intrasom]'
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import intrasom  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"intrasom not available: {exc}", allow_module_level=True)

from pyesom import ESOM


def test_esom_intrasom_train_and_topology():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((80, 4))
    som = ESOM(6, 5, 4, backend="intrasom", random_seed=0)
    som.fit(data, epochs=4)
    assert som.backend == "intrasom"
    w = som.weights
    assert w.shape == (6, 5, 4)
    u = som.u_matrix()
    assert u.shape == (6, 5)
    bm = som.bmu_indices(data)
    assert bm.shape == (len(data), 2)
    hits = som.hit_map(data)
    assert int(hits.sum()) == len(data)


def test_esom_intrasom_toroid_default():
    """mapshape defaults to toroid."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((40, 2))
    som = ESOM(4, 5, 2, backend="intrasom", random_seed=0)
    som.fit(data, epochs=2)
    assert som._intrasom.mapshape == "toroid"


def test_esom_intrasom_planar_override():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((40, 2))
    som = ESOM(4, 5, 2, backend="intrasom",
               intrasom_kwargs={"mapshape": "planar"}, random_seed=0)
    som.fit(data, epochs=2)
    assert som._intrasom.mapshape == "planar"


def test_esom_intrasom_epochs_split():
    """train_rough_len and train_finetune_len override epoch split."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((40, 2))
    som = ESOM(4, 5, 2, backend="intrasom",
               intrasom_kwargs={"train_rough_len": 3, "train_finetune_len": 1},
               random_seed=0)
    som.fit(data)  # epochs kwarg not needed when lengths are explicit
    assert som._intrasom.train_rough_len == 3
    assert som._intrasom.train_finetune_len == 1


def test_esom_intrasom_component_plane():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((60, 3))
    som = ESOM(6, 6, 3, backend="intrasom", random_seed=0)
    som.fit(data, epochs=2)
    plane = som.component_plane(data, 0)
    assert plane.shape == (6, 6)


def test_esom_intrasom_odd_rows_rounded_up():
    """Hexagonal IntraSOM rounds odd row counts to even; weights shape reflects this."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((40, 2))
    som = ESOM(5, 4, 2, backend="intrasom", random_seed=0)
    som.fit(data, epochs=2)
    # rows=5 is odd → rounded to 6 internally; weights shape is (6, 4, 2)
    assert som.weights.shape == (6, 4, 2)
    assert som.hit_map(data).shape == (6, 4)
    assert int(som.hit_map(data).sum()) == len(data)
