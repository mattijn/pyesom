"""SomLattice agrees with ESOM / MiniSom when given the same weights."""

from __future__ import annotations

import numpy as np
import pytest

from pyesom import ESOM, SomLattice, reshape_flat_codebook


def test_umatrix_matches_esom():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((40, 5))
    som = ESOM(6, 7, 5, random_seed=1)
    som.fit(data, iterations=500)
    lat = SomLattice.from_esom(som)
    u_som = som.u_matrix()
    u_lat = lat.u_matrix()
    assert u_som.shape == u_lat.shape
    np.testing.assert_allclose(u_som, u_lat, rtol=0, atol=0)


def test_bmu_hit_component_match_esom():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((25, 4))
    som = ESOM(5, 4, 4, random_seed=3)
    som.fit(data, iterations=400)
    lat = SomLattice.from_esom(som)
    np.testing.assert_array_equal(som.bmu_indices(data), lat.bmu_indices(data))
    np.testing.assert_array_equal(som.hit_map(data), lat.hit_map(data))
    for fi in range(4):
        np.testing.assert_allclose(
            som.component_plane(data, fi),
            lat.component_plane(data, fi),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )


def test_reshape_flat_codebook_roundtrip():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((4, 5, 6))
    flat = w.reshape(-1, 6)
    back = reshape_flat_codebook(flat, 4, 5)
    np.testing.assert_array_equal(back, w)


def test_reshape_flat_wrong_size_raises():
    with pytest.raises(ValueError, match="must equal"):
        reshape_flat_codebook(np.zeros((10, 3)), 3, 3)
