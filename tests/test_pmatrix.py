import numpy as np
import pytest

from pyesom.topology.pmatrix import compute_pmatrix


# ── hit-map (pareto_fraction=None) ────────────────────────────────────────────

def test_pmatrix_hits_sum_to_sample_count():
    rng = np.random.default_rng(1)
    weights = rng.standard_normal((5, 6, 3))
    data = rng.standard_normal((50, 3))
    hits = compute_pmatrix(weights, data, pareto_fraction=None)
    assert hits.shape == (5, 6)
    assert hits.sum() == len(data)


def test_pmatrix_hits_shape():
    rng = np.random.default_rng(2)
    weights = rng.standard_normal((4, 7, 2))
    data = rng.standard_normal((30, 2))
    hits = compute_pmatrix(weights, data, pareto_fraction=None)
    assert hits.shape == (4, 7)


def test_pmatrix_hits_non_negative():
    rng = np.random.default_rng(3)
    weights = rng.standard_normal((4, 5, 2))
    data = rng.standard_normal((30, 2))
    hits = compute_pmatrix(weights, data, pareto_fraction=None)
    assert np.all(hits >= 0)


def test_pmatrix_nearest_neuron_assignment():
    """Data concentrated near one node should hit that node most."""
    weights = np.array([[[0.0, 0.0], [10.0, 10.0]]])  # shape (1, 2, 2)
    rng = np.random.default_rng(4)
    data = rng.normal(loc=[0.0, 0.0], scale=0.05, size=(100, 2))
    hits = compute_pmatrix(weights, data, pareto_fraction=None)
    assert hits[0, 0] > hits[0, 1]
    assert int(hits.sum()) == 100


def test_pmatrix_single_node_captures_all():
    """Single-node SOM → all samples hit that one node."""
    weights = np.zeros((1, 1, 3))
    data = np.random.default_rng(5).standard_normal((25, 3))
    hits = compute_pmatrix(weights, data, pareto_fraction=None)
    assert hits[0, 0] == 25


# ── PDE-style (pareto_fraction given) ─────────────────────────────────────────

def test_pmatrix_pde_shape():
    rng = np.random.default_rng(6)
    weights = rng.standard_normal((4, 5, 2))
    data = rng.standard_normal((40, 2))
    p = compute_pmatrix(weights, data, pareto_fraction=0.2013)
    assert p.shape == (4, 5)


def test_pmatrix_pde_non_negative():
    rng = np.random.default_rng(7)
    weights = rng.standard_normal((3, 4, 2))
    data = rng.standard_normal((30, 2))
    p = compute_pmatrix(weights, data, pareto_fraction=0.2)
    assert np.all(p >= 0)


def test_pmatrix_pde_count_near_pareto_fraction():
    """Node-local PDE: each node's count ≈ floor(pareto_fraction * n_samples).

    The implementation uses a per-node Pareto radius (distance to the k-th nearest
    data point from that node's weight), so the count is approximately pf*n for
    most nodes regardless of their position in data space.
    """
    rng = np.random.default_rng(8)
    weights = rng.standard_normal((2, 3, 2))
    n_samples = 100
    data = rng.standard_normal((n_samples, 2))
    pf = 0.2
    p = compute_pmatrix(weights, data, pareto_fraction=pf)
    expected = int(np.floor(pf * n_samples))
    # Each node should have count >= expected (the Pareto radius includes at least
    # floor(pf*n) samples by construction)
    assert np.all(p >= expected)


def test_pmatrix_pde_radius_shrinks_for_nearby_data():
    """Nodes embedded within dense data should have smaller Pareto radii,
    while the total count is always approximately floor(pf * n).

    We can't directly observe the radius, but we CAN observe that both
    the dense-adjacent and distant nodes get roughly the same count.
    """
    rng = np.random.default_rng(9)
    weights = np.array([[[0.0, 0.0], [50.0, 50.0]]])  # (1, 2, 2): one near, one far
    data = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(80, 2))
    pf = 0.2
    p = compute_pmatrix(weights, data, pareto_fraction=pf)
    expected = int(np.floor(pf * 80))  # = 16
    # Both nodes get approximately the expected count (node-local PDE property)
    assert p[0, 0] >= expected
    assert p[0, 1] >= expected


# ── Error paths ───────────────────────────────────────────────────────────────

def test_pmatrix_wrong_weights_ndim_raises():
    with pytest.raises(ValueError, match="weights must have shape"):
        compute_pmatrix(np.zeros((3, 3)), np.zeros((10, 2)))


def test_pmatrix_wrong_data_ndim_raises():
    with pytest.raises(ValueError):
        compute_pmatrix(np.zeros((3, 3, 2)), np.zeros(10))  # 1-D data


def test_pmatrix_feature_mismatch_raises():
    weights = np.zeros((3, 3, 2))
    data = np.zeros((10, 4))  # 4 features, weights have 2
    with pytest.raises(ValueError):
        compute_pmatrix(weights, data, pareto_fraction=None)


def test_pmatrix_invalid_pareto_zero_raises():
    weights = np.zeros((3, 3, 2))
    data = np.zeros((10, 2))
    with pytest.raises(ValueError, match="pareto_fraction"):
        compute_pmatrix(weights, data, pareto_fraction=0.0)


def test_pmatrix_invalid_pareto_one_raises():
    weights = np.zeros((3, 3, 2))
    data = np.zeros((10, 2))
    with pytest.raises(ValueError, match="pareto_fraction"):
        compute_pmatrix(weights, data, pareto_fraction=1.0)


def test_pmatrix_invalid_pareto_above_one_raises():
    weights = np.zeros((3, 3, 2))
    data = np.zeros((10, 2))
    with pytest.raises(ValueError, match="pareto_fraction"):
        compute_pmatrix(weights, data, pareto_fraction=1.5)
