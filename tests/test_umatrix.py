import numpy as np
import pytest
from minisom import MiniSom

from pyesom.topology.umatrix import compute_umatrix


def test_umatrix_matches_minisom_sum():
    rng = np.random.default_rng(0)
    x, y, f = 7, 9, 4
    weights = rng.standard_normal((x, y, f))
    som = MiniSom(x, y, f, random_seed=0)
    som._weights[:] = weights

    ours = compute_umatrix(weights, scaling="sum")
    ref = np.asarray(som.distance_map(scaling="sum"))
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_umatrix_matches_minisom_mean():
    rng = np.random.default_rng(1)
    x, y, f = 5, 8, 3
    weights = rng.standard_normal((x, y, f))
    som = MiniSom(x, y, f, random_seed=0)
    som._weights[:] = weights

    ours = compute_umatrix(weights, scaling="mean")
    ref = np.asarray(som.distance_map(scaling="mean"))
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_umatrix_output_shape():
    weights = np.random.default_rng(2).standard_normal((5, 7, 4))
    u = compute_umatrix(weights)
    assert u.shape == (5, 7)


def test_umatrix_output_in_unit_interval():
    rng = np.random.default_rng(3)
    weights = rng.standard_normal((8, 10, 3))
    u = compute_umatrix(weights)
    assert u.min() >= 0.0
    assert u.max() <= 1.0 + 1e-12


def test_umatrix_max_is_one():
    rng = np.random.default_rng(4)
    weights = rng.standard_normal((6, 6, 4))
    u = compute_umatrix(weights)
    assert abs(u.max() - 1.0) < 1e-12


def test_umatrix_sum_and_mean_differ():
    rng = np.random.default_rng(5)
    weights = rng.standard_normal((8, 10, 4))
    u_sum = compute_umatrix(weights, scaling="sum")
    u_mean = compute_umatrix(weights, scaling="mean")
    assert not np.allclose(u_sum, u_mean)


def test_umatrix_constant_weights_all_zero():
    # Identical prototypes → all distances 0 → U-matrix all 0
    weights = np.ones((4, 5, 3))
    u = compute_umatrix(weights)
    np.testing.assert_array_equal(u, 0.0)


def test_umatrix_dtype_float64():
    weights = np.ones((3, 3, 2), dtype=np.float32)
    u = compute_umatrix(weights)
    assert u.dtype == np.float64


def test_umatrix_wrong_ndim_raises():
    with pytest.raises(ValueError, match="weights must have shape"):
        compute_umatrix(np.zeros((5, 5)))


def test_umatrix_wrong_scaling_raises():
    with pytest.raises(ValueError, match="scaling must be"):
        compute_umatrix(np.zeros((3, 3, 2)), scaling="invalid")


def test_umatrix_single_row_grid():
    # Edge case: 1×N grid — boundary neurons have only right/left neighbours
    weights = np.random.default_rng(6).standard_normal((1, 6, 3))
    u = compute_umatrix(weights)
    assert u.shape == (1, 6)
    assert u.min() >= 0.0
    assert u.max() <= 1.0 + 1e-12


def test_umatrix_non_negative():
    weights = np.random.default_rng(7).standard_normal((5, 5, 5))
    u = compute_umatrix(weights)
    assert np.all(u >= 0.0)


def test_umatrix_high_distance_regions_have_high_values():
    """Two weight clusters far apart should produce high U-values at their boundary."""
    weights = np.zeros((3, 4, 2))
    weights[:, :2, :] = 0.0     # left half: weights at origin
    weights[:, 2:, :] = 100.0  # right half: weights far away
    u = compute_umatrix(weights)
    # Boundary columns (1 and 2) should be highest
    boundary_max = max(u[:, 1].max(), u[:, 2].max())
    interior_max = max(u[:, 0].max(), u[:, 3].max())
    assert boundary_max >= interior_max
