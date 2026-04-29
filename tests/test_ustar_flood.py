import numpy as np
import pytest

from pyesom.clustering.ustar_flood import UStarFloodClustering


def _two_valley_grid(size=10):
    """Synthetic U-matrix with two clearly separated low basins."""
    u = np.ones((size, size), dtype=np.float64) * 0.9
    u[1:4, 1:4] = 0.1   # basin A
    u[6:9, 6:9] = 0.1   # basin B — separated by high ridge
    p = np.ones_like(u) * 0.5
    return u, p


# ── fit attributes ─────────────────────────────────────────────────────────────

def test_two_valleys_two_clusters():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=50, threshold_anchor="upper")
    clf.fit(u, p)
    assert clf.n_clusters_ >= 2


def test_labels_grid_shape():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=30)
    clf.fit(u, p)
    assert clf.labels_.shape == u.shape


def test_ustar_attribute_set():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering()
    clf.fit(u, p)
    assert clf.ustar_ is not None
    assert clf.ustar_.shape == u.shape
    assert np.all(np.isfinite(clf.ustar_))
    assert np.all(clf.ustar_ >= 0.0)


def test_threshold_in_unit_interval():
    u = np.random.default_rng(0).random((6, 8))
    p = np.ones_like(u)
    clf = UStarFloodClustering(n_threshold_steps=50)
    clf.fit(u, p)
    assert 0.0 <= clf.threshold_ <= 1.0


def test_auto_threshold_upper_bin_matches_manual_formula():
    rng = np.random.default_rng(42)
    u = rng.random((18, 18)).astype(np.float64)
    p = np.ones_like(u)
    nsteps = 50
    clf = UStarFloodClustering(
        min_cluster_size=1, n_threshold_steps=nsteps, use_ustar=False, threshold_anchor="upper"
    )
    clf.fit(u, p)
    from pyesom.clustering.ustar_flood import _normalize_unit_interval, _flood_region_size

    unorm = _normalize_unit_interval(clf.ustar_)
    thresholds = np.linspace(0.0, 1.0, nsteps)
    sizes = [_flood_region_size(unorm, float(t)) for t in thresholds]
    grad = np.diff(np.asarray(sizes, dtype=np.float64))
    k = int(np.argmax(grad))
    expected = float(thresholds[min(k + 1, nsteps - 1)])
    assert clf.threshold_ == expected


def test_auto_threshold_lower_bin_matches_manual_formula():
    rng = np.random.default_rng(42)
    u = rng.random((18, 18)).astype(np.float64)
    p = np.ones_like(u)
    nsteps = 50
    clf = UStarFloodClustering(
        min_cluster_size=1, n_threshold_steps=nsteps, use_ustar=False, threshold_anchor="lower"
    )
    clf.fit(u, p)
    from pyesom.clustering.ustar_flood import _normalize_unit_interval, _flood_region_size

    unorm = _normalize_unit_interval(clf.ustar_)
    thresholds = np.linspace(0.0, 1.0, nsteps)
    sizes = [_flood_region_size(unorm, float(t)) for t in thresholds]
    grad = np.diff(np.asarray(sizes, dtype=np.float64))
    k = int(np.argmax(grad))
    assert clf.threshold_ == float(thresholds[k])


def test_threshold_anchor_invalid_raises():
    with pytest.raises(ValueError, match="threshold_anchor"):
        UStarFloodClustering(threshold_anchor="middle")
def test_manual_threshold_override():
    """fit(..., threshold=t) pins threshold_ and skips auto gradient rule."""
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=50)
    clf.fit(u, p, threshold=0.48)
    assert clf.threshold_ == 0.48


def test_manual_threshold_out_of_range_raises():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1)
    with pytest.raises(ValueError, match="threshold"):
        clf.fit(u, p, threshold=1.01)


def test_labels_contain_only_valid_ids():
    """All labels must be either -1 (unassigned) or a non-negative cluster index."""
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=50)
    clf.fit(u, p)
    assert np.all(clf.labels_ >= -1)
    assert clf.n_clusters_ == len(np.unique(clf.labels_[clf.labels_ >= 0]))


def test_cluster_ids_contiguous():
    """Cluster IDs after fit must form a contiguous range 0..K-1."""
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=50)
    clf.fit(u, p)
    assigned = clf.labels_[clf.labels_ >= 0]
    if len(assigned) > 0:
        assert set(assigned.tolist()) == set(range(clf.n_clusters_))


# ── use_ustar=False ────────────────────────────────────────────────────────────

def test_use_ustar_false_stores_raw_u():
    """When use_ustar=False, the stored ustar_ equals the raw input U-matrix."""
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, use_ustar=False)
    clf.fit(u, p)
    np.testing.assert_array_equal(clf.ustar_, u)


def test_use_ustar_false_still_clusters():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=50, use_ustar=False, threshold_anchor="upper")
    clf.fit(u, p)
    assert clf.n_clusters_ >= 1


# ── min_cluster_size ─────────────────────────────────────────────────────────────

def test_min_cluster_size_enforced():
    """No cluster after pruning should have fewer nodes than min_cluster_size."""
    u, p = _two_valley_grid()
    min_size = 3
    clf = UStarFloodClustering(min_cluster_size=min_size, n_threshold_steps=50)
    clf.fit(u, p)
    for cid in range(clf.n_clusters_):
        count = int(np.count_nonzero(clf.labels_ == cid))
        assert count >= min_size, f"Cluster {cid} has only {count} nodes"


def test_min_cluster_size_one_allows_singletons():
    """With min_cluster_size=1, even single-node basins are kept."""
    u = np.ones((6, 6)) * 0.9
    u[0, 0] = 0.0   # isolated single node as a basin
    u[5, 5] = 0.0
    p = np.ones_like(u) * 0.5
    clf_strict = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=50)
    clf_any = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=50)
    clf_strict.fit(u, p)
    clf_any.fit(u, p)
    assert clf_any.n_clusters_ >= clf_strict.n_clusters_


# ── predict ────────────────────────────────────────────────────────────────────

def test_predict_shape():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=30)
    clf.fit(u, p)
    bmu = np.array([[2, 2], [7, 7], [0, 9]], dtype=np.int64)
    labels = clf.predict(bmu)
    assert labels.shape == (3,)


def test_predict_assigned_nodes_are_valid():
    """Samples whose BMU is a labelled node should get a non-negative label."""
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=50, threshold_anchor="upper")
    clf.fit(u, p)
    # Feed BMU coordinates from each known cluster centre
    centres = np.array([[2, 2], [7, 7]], dtype=np.int64)
    preds = clf.predict(centres)
    # These nodes sit in the valleys — at least one should be assigned
    assert np.any(preds >= 0)


def test_predict_out_of_bounds_gives_minus_one():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=30)
    clf.fit(u, p)
    bmu = np.array([[999, 999]], dtype=np.int64)
    assert clf.predict(bmu)[0] == -1


def test_predict_before_fit_raises():
    clf = UStarFloodClustering()
    with pytest.raises(RuntimeError, match="fit"):
        clf.predict(np.array([[0, 0]]))


def test_predict_bad_shape_raises():
    u, p = _two_valley_grid()
    clf = UStarFloodClustering(min_cluster_size=1).fit(u, p)
    with pytest.raises(ValueError, match="shape"):
        clf.predict(np.array([0, 1]))   # 1-D, not (n, 2)


# ── boundary nodes ─────────────────────────────────────────────────────────────

def test_high_ridge_leaves_boundary_unassigned():
    """A landscape that is mostly ridge should leave many nodes as -1."""
    u = np.ones((10, 10)) * 0.95   # almost entirely ridge
    u[2, 2] = 0.0
    u[8, 8] = 0.0
    p = np.ones_like(u) * 0.5
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=60)
    clf.fit(u, p)
    unassigned_frac = np.mean(clf.labels_ == -1)
    assert unassigned_frac > 0.5   # most nodes are on the ridge


# ── flat / degenerate grids ───────────────────────────────────────────────────

def test_flat_landscape_no_crash():
    """A completely uniform U-matrix should not raise and gives ≥ 0 clusters."""
    u = np.full((6, 6), 0.5)
    p = np.ones_like(u) * 2.0
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=30)
    clf.fit(u, p)
    assert clf.n_clusters_ >= 0
    assert clf.labels_.shape == (6, 6)


def test_single_basin_one_cluster():
    """Single global minimum → exactly one cluster found (if large enough)."""
    u = np.ones((8, 8)) * 0.8
    u[3:6, 3:6] = 0.05   # one clear basin
    p = np.ones_like(u) * 0.5
    clf = UStarFloodClustering(min_cluster_size=1, n_threshold_steps=60, threshold_anchor="upper")
    clf.fit(u, p)
    assert clf.n_clusters_ >= 1


# ── min_samples ─────────────────────────────────────────────────────────────

def test_min_samples_preserves_data_rich_small_cluster():
    """
    Edge cluster: 2 SOM nodes, but high catchment count.
    With min_cluster_size=10 alone it would be pruned.
    With min_samples=50 the AND-rescue condition keeps it.

    p_median_size=1 disables the 3×3 median filter so the density signal
    of the isolated 2-node cluster is not smeared out by its sparse neighbours.
    """
    u = np.ones((10, 10)) * 0.9
    # Basin A — large (16 nodes), easily survives min_cluster_size=10
    u[1:5, 1:5] = 0.1
    p = np.ones((10, 10)) * 0.5
    p[1:5, 1:5] = 5.0

    # Basin B — only 2 nodes, but each holds 100 catchments
    u[8, 8] = 0.05
    u[8, 9] = 0.07
    p[8, 8] = 100.0
    p[8, 9] = 100.0

    common = dict(n_threshold_steps=50, p_median_size=1, threshold_anchor="upper")

    clf_strict = UStarFloodClustering(min_cluster_size=10, min_samples=0, **common)
    clf_strict.fit(u, p)
    # Basin B should be pruned with min_samples=0 (old behaviour)
    basin_b_survived_strict = any(
        clf_strict.labels_[8, j] >= 0 for j in (8, 9)
    )
    assert not basin_b_survived_strict, "Basin B should be pruned without min_samples"

    clf_rescue = UStarFloodClustering(min_cluster_size=10, min_samples=50, **common)
    clf_rescue.fit(u, p)
    # Basin B should survive because its catchment sum (200) >= min_samples=50
    basin_b_survived_rescue = any(
        clf_rescue.labels_[8, j] >= 0 for j in (8, 9)
    )
    assert basin_b_survived_rescue, "Basin B should survive with min_samples=50"


def test_min_samples_zero_behaves_like_original():
    """min_samples=0 (default) must give the same result as the original pruning."""
    u, p = _two_valley_grid()
    clf_orig = UStarFloodClustering(min_cluster_size=5, min_samples=0,
                                    n_threshold_steps=50)
    clf_compat = UStarFloodClustering(min_cluster_size=5,
                                      n_threshold_steps=50)
    clf_orig.fit(u, p)
    clf_compat.fit(u, p)
    np.testing.assert_array_equal(clf_orig.labels_, clf_compat.labels_)
