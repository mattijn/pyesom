import numpy as np
import pytest

from pyesom.topology.ustar import compute_ustar


# ── formula correctness (Ultsch 2003) ─────────────────────────────────────────

def test_ustar_uniform_p_is_identity():
    """Uniform P → ScaleFactor = 1 everywhere → U* == U (no median filter)."""
    rng = np.random.default_rng(0)
    u = rng.random((4, 5))
    p = np.ones_like(u) * 5.0
    out = compute_ustar(u, p, median_filter_size=1, use_robust_mean=False)
    np.testing.assert_allclose(out, u, rtol=1e-12)


def test_ustar_mean_density_identity():
    """Original paper test: P == mean(P) everywhere → U* == U."""
    rng = np.random.default_rng(2)
    u = rng.random((4, 5))
    p = np.ones_like(u) * 5.0
    out = compute_ustar(u, p, use_robust_mean=False)
    np.testing.assert_allclose(out, u, rtol=1e-12)


def test_ustar_max_density_gives_zero():
    """At the densest node (P == P_max), ScaleFactor = 0 → U* = 0."""
    u = np.ones((5, 5)) * 0.5
    p = np.ones((5, 5)) * 2.0
    p[2, 2] = 10.0          # single node with P = P_max
    out = compute_ustar(u, p, median_filter_size=1, use_robust_mean=False)
    assert abs(out[2, 2]) < 1e-10


def test_ustar_high_density_suppresses():
    """Nodes with P > mean(P) should get U* < U."""
    u = np.full((5, 5), 0.5)
    p = np.full((5, 5), 1.0)
    p[2, 2] = 8.0           # much higher density → suppressed
    out = compute_ustar(u, p, median_filter_size=1, use_robust_mean=False)
    assert out[2, 2] < u[2, 2]


def test_ustar_low_density_amplifies():
    """Nodes with P < mean(P) should get U* > U (boundary amplified)."""
    u = np.full((5, 5), 0.5)
    p = np.full((5, 5), 5.0)
    p[2, 2] = 0.0           # sparse node at boundary → amplified
    out = compute_ustar(u, p, median_filter_size=1, use_robust_mean=False)
    assert out[2, 2] > u[2, 2]


def test_ustar_scalemax_clips():
    """Scale factor must never exceed scalemax, so U* <= U * scalemax."""
    u = np.full((5, 5), 1.0)
    p = np.full((5, 5), 5.0)
    p[2, 2] = 0.0           # would produce very high scale without clip
    scalemax = 2.0
    out = compute_ustar(u, p, scalemax=scalemax, median_filter_size=1)
    # U* = U * scale, U = 1 everywhere → U* <= scalemax
    assert np.all(out <= scalemax + 1e-12)


# ── output shape and dtype ─────────────────────────────────────────────────────

def test_ustar_output_shape():
    rng = np.random.default_rng(3)
    u = rng.random((6, 8))
    p = rng.random((6, 8)) + 0.1
    out = compute_ustar(u, p)
    assert out.shape == (6, 8)


def test_ustar_all_finite():
    rng = np.random.default_rng(4)
    u = rng.random((7, 9))
    p = rng.random((7, 9)) * 10
    out = compute_ustar(u, p)
    assert np.all(np.isfinite(out))


def test_ustar_non_negative():
    rng = np.random.default_rng(5)
    u = np.abs(rng.random((6, 6)))
    p = np.abs(rng.random((6, 6))) + 0.1
    out = compute_ustar(u, p)
    assert np.all(out >= 0.0)


# ── median filter path ─────────────────────────────────────────────────────────

def test_ustar_no_median_filter_size_one():
    rng = np.random.default_rng(6)
    u = rng.random((5, 5))
    p = rng.random((5, 5)) * 10
    out = compute_ustar(u, p, median_filter_size=1)
    assert out.shape == (5, 5)
    assert np.all(np.isfinite(out))


def test_ustar_median_filter_smooths_spike():
    """Isolated P spike should be smoothed by median filter."""
    u = np.full((7, 7), 0.5)
    p = np.full((7, 7), 2.0)
    p[3, 3] = 1000.0        # extreme spike in the middle
    out_filtered = compute_ustar(u, p, median_filter_size=3, use_robust_mean=False)
    out_raw = compute_ustar(u, p, median_filter_size=1, use_robust_mean=False)
    # With the filter, the spike's effect on the center node is suppressed
    assert out_filtered[3, 3] > out_raw[3, 3]


# ── robust mean ───────────────────────────────────────────────────────────────

def test_ustar_robust_mean_vs_plain_mean():
    """Skewed P distribution: when median > mean, robust mean uses median → different U*.

    A few very sparse nodes (low P) pull mean below median.
    p_ref_robust = max(mean, median) = median > mean = p_ref_plain.
    """
    rng = np.random.default_rng(7)
    u = rng.random((5, 5))
    p = np.full((5, 5), 5.0)
    # Two very sparse nodes drag mean below median
    p[0, 0] = 0.0
    p[0, 1] = 0.0
    # mean ≈ (23*5 + 0 + 0)/25 = 4.6, median = 5.0 → max differs from mean
    assert np.median(p) > np.mean(p)   # confirm setup
    out_robust = compute_ustar(u, p, median_filter_size=1, use_robust_mean=True)
    out_plain = compute_ustar(u, p, median_filter_size=1, use_robust_mean=False)
    assert not np.allclose(out_robust, out_plain)


# ── error handling ─────────────────────────────────────────────────────────────

def test_ustar_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        compute_ustar(np.zeros((3, 4)), np.zeros((3, 5)))


def test_ustar_degenerate_p_no_crash():
    """All-zero P (degenerate): denom <= 1e-15 → scale = 1 → U* == U."""
    u = np.random.default_rng(8).random((4, 4))
    p = np.zeros((4, 4))
    out = compute_ustar(u, p, median_filter_size=1, use_robust_mean=False)
    np.testing.assert_allclose(out, u, rtol=1e-12)
