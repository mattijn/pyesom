"""FCPS archive + full pipeline smoke tests (``resources/fcps.npz``, not shipped in the wheel).

Paper-grade purity benchmarks depend strongly on ESOM size, iteration budget, and the
projection implementation; reproduce extended experiments via ``scripts/run_fcps_benchmark.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyesom.clustering.ustar_flood import UStarFloodClustering
from fcps_npz_io import load_fcps, resolve_fcps_npz_path
from pyesom.projection.esom import ESOM
from pyesom.topology.pmatrix import compute_pmatrix

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    _FCPS_NPZ = resolve_fcps_npz_path(search_from=REPO_ROOT)
except FileNotFoundError:
    _FCPS_NPZ = None

pytestmark = pytest.mark.skipif(
    _FCPS_NPZ is None,
    reason="Missing resources/fcps.npz — run scripts/export_fcps_npz.py or set PYESOM_FCPS_NPZ.",
)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


def test_fcps_shapes_and_labels():
    assert _FCPS_NPZ is not None
    data, cls = load_fcps("atom", npz_path=_FCPS_NPZ)
    assert data.shape == (800, 3)
    assert cls.shape == (800,)
    assert set(np.unique(cls).tolist()) == {0, 1}


def test_fcps_esom_pipeline_smoke():
    """Fast smoke test — validates imports and wiring."""
    assert _FCPS_NPZ is not None
    data, cls = load_fcps("atom", npz_path=_FCPS_NPZ)
    data = _zscore(data)
    som = ESOM(14, 16, data.shape[1], random_seed=0)
    som.fit(data, iterations=800)
    u = som.u_matrix()
    p = compute_pmatrix(som.weights, data, pareto_fraction=None)
    clf = UStarFloodClustering(min_cluster_size=2, n_threshold_steps=40)
    clf.fit(u, p)
    pred = clf.predict(som.bmu_indices(data))
    assert pred.shape == cls.shape
    assert np.isfinite(clf.ustar_).all()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PYESOM_FULL_FCPS") != "1",
    reason="Set PYESOM_FULL_FCPS=1 for extended benchmark (~minutes).",
)
def test_fcps_atom_adjusted_rand():
    pytest.importorskip("sklearn")
    from sklearn.metrics import adjusted_rand_score

    assert _FCPS_NPZ is not None
    data, cls = load_fcps("atom", npz_path=_FCPS_NPZ)
    data = _zscore(data)
    gx, gy = 55, 65
    som = ESOM(gx, gy, data.shape[1], random_seed=42)
    som.fit(data, iterations=40_000)
    u = som.u_matrix()
    p = compute_pmatrix(som.weights, data, pareto_fraction=None)
    clf = UStarFloodClustering(min_cluster_size=4, n_threshold_steps=120)
    clf.fit(u, p)
    pred = clf.predict(som.bmu_indices(data))
    mask = pred >= 0
    ari = adjusted_rand_score(cls[mask], pred[mask])
    assert ari >= 0.15


@pytest.mark.slow
def test_hepta_reasonable():
    """Seven-cluster toy — coarse grid sometimes merges parties; sanity bounds only."""
    assert _FCPS_NPZ is not None
    data, cls = load_fcps("hepta", npz_path=_FCPS_NPZ)
    data = _zscore(data)
    gx, gy = 32, 36
    som = ESOM(gx, gy, data.shape[1], random_seed=7)
    som.fit(data, iterations=15_000)
    u = som.u_matrix()
    p = compute_pmatrix(som.weights, data, pareto_fraction=None)
    clf = UStarFloodClustering(min_cluster_size=2, n_threshold_steps=100)
    clf.fit(u, p)
    pred = clf.predict(som.bmu_indices(data))
    assert clf.n_clusters_ >= 2
    assert np.mean(pred < 0) <= 0.45
