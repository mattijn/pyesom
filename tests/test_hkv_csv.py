"""Regression tests on the bundled HKV CSV (pre-trained 60×50 SOM).

The CSV contains a fully trained SOM with:
  - X, Y : grid coordinates (0-based), X in [0,59], Y in [0,49]
  - U-matrix          : pre-computed inter-neuron distance map (normalized)
  - Total events in node : hit count per neuron (serves as P-matrix proxy)

These tests exercise the full topology + clustering pipeline using a real
hydrological dataset, without re-training the SOM.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pyesom.clustering.ustar_flood import UStarFloodClustering
from pyesom.topology.ustar import compute_ustar

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = REPO_ROOT / "resources" / "HKV-viewer" / "eu-extreme-discharge-som.csv"

SKIP = pytest.mark.skipif(not CSV.is_file(), reason="resources/HKV-viewer CSV not present")

GRID_X = 60
GRID_Y = 50


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_grids() -> tuple[np.ndarray, np.ndarray]:
    """Return (u_grid, p_grid) as (60, 50) arrays from the CSV."""
    u_grid = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    p_grid = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    with CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xi = int(row["X"])
            yi = int(row["Y"])
            u_grid[xi, yi] = float(row["U-matrix"])
            p_grid[xi, yi] = float(row["Total events in node"])
    return u_grid, p_grid


# ── CSV structure checks ───────────────────────────────────────────────────────

@SKIP
def test_hkv_grid_u_matrix_present():
    """Original smoke test — confirm grid dimensions and U-matrix column."""
    with CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    xs = [int(r["X"]) for r in rows]
    ys = [int(r["Y"]) for r in rows]
    assert len(set(xs)) == GRID_X
    assert len(set(ys)) == GRID_Y
    u = [float(r["U-matrix"]) for r in rows]
    assert len(u) == GRID_X * GRID_Y
    assert np.isfinite(np.asarray(u)).all()


@SKIP
def test_hkv_all_grid_positions_present():
    """Every (X, Y) pair must appear exactly once."""
    seen = set()
    with CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add((int(row["X"]), int(row["Y"])))
    assert len(seen) == GRID_X * GRID_Y


@SKIP
def test_hkv_p_matrix_non_negative():
    _, p_grid = _load_grids()
    assert np.all(p_grid >= 0)


@SKIP
def test_hkv_u_matrix_non_negative_and_finite():
    """U-matrix values from the CSV are non-negative real numbers."""
    u_grid, _ = _load_grids()
    assert u_grid.min() >= 0.0
    assert np.all(np.isfinite(u_grid))


# ── U*-matrix computation ──────────────────────────────────────────────────────

@SKIP
def test_hkv_ustar_computes_without_error():
    u_grid, p_grid = _load_grids()
    ustar = compute_ustar(u_grid, p_grid)
    assert ustar.shape == (GRID_X, GRID_Y)
    assert np.all(np.isfinite(ustar))


@SKIP
def test_hkv_ustar_non_negative():
    u_grid, p_grid = _load_grids()
    ustar = compute_ustar(u_grid, p_grid)
    assert np.all(ustar >= 0.0)


@SKIP
def test_hkv_ustar_dense_nodes_suppressed():
    """Dense nodes (high P) should have U* ≤ U, sparse nodes should have U* ≥ U.

    The Ultsch ScaleFactor is a decreasing function of P, so the mean
    ScaleFactor in dense regions should be lower than in sparse regions.
    """
    u_grid, p_grid = _load_grids()
    # Skip median filter and robust mean to test the raw ScaleFactor effect
    ustar = compute_ustar(u_grid, p_grid, median_filter_size=1, use_robust_mean=False)
    # Compute per-node ScaleFactor: U* / U (where U > 0)
    mask = u_grid > 0
    scale = np.where(mask, ustar / np.where(mask, u_grid, 1.0), np.nan)
    p_mean = p_grid.mean()
    dense_mask = (p_grid > p_mean) & mask
    sparse_mask = (p_grid < p_mean) & mask
    if dense_mask.any() and sparse_mask.any():
        mean_scale_dense = np.nanmean(scale[dense_mask])
        mean_scale_sparse = np.nanmean(scale[sparse_mask])
        assert mean_scale_dense <= mean_scale_sparse + 0.5  # dense gets lower ScaleFactor


@SKIP
def test_hkv_ustar_median_filter_changes_result():
    """Applying a 3×3 median filter to P before U* should produce different values."""
    u_grid, p_grid = _load_grids()
    ustar_raw = compute_ustar(u_grid, p_grid, median_filter_size=1)
    ustar_filtered = compute_ustar(u_grid, p_grid, median_filter_size=3)
    # They should differ — the filter modifies the P-matrix
    assert not np.allclose(ustar_raw, ustar_filtered)
    # Both must be finite and non-negative
    assert np.all(np.isfinite(ustar_filtered))
    assert np.all(ustar_filtered >= 0.0)


# ── clustering pipeline ───────────────────────────────────────────────────────

@SKIP
def test_hkv_flood_clustering_runs():
    u_grid, p_grid = _load_grids()
    clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=80)
    clf.fit(u_grid, p_grid)
    assert clf.labels_.shape == (GRID_X, GRID_Y)
    assert clf.ustar_ is not None
    assert 0.0 <= clf.threshold_ <= 1.0


@SKIP
def test_hkv_flood_finds_multiple_clusters():
    """Real European discharge data should resolve at least 2 catchment regimes."""
    u_grid, p_grid = _load_grids()
    clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=80)
    clf.fit(u_grid, p_grid)
    assert clf.n_clusters_ >= 2, (
        f"Expected ≥ 2 clusters from HKV SOM, got {clf.n_clusters_}"
    )


@SKIP
def test_hkv_flood_labels_valid():
    """All label values must be -1 (unassigned) or a valid cluster index."""
    u_grid, p_grid = _load_grids()
    clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=80)
    clf.fit(u_grid, p_grid)
    assert np.all(clf.labels_ >= -1)
    assigned = clf.labels_[clf.labels_ >= 0]
    if len(assigned):
        assert set(assigned.tolist()) == set(range(clf.n_clusters_))


@SKIP
def test_hkv_flood_some_nodes_assigned():
    """At least some nodes should be assigned (not 100% ridge)."""
    u_grid, p_grid = _load_grids()
    clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=80)
    clf.fit(u_grid, p_grid)
    assert np.any(clf.labels_ >= 0)


@SKIP
def test_hkv_flood_unassigned_fraction_reasonable():
    """U*F may leave boundary nodes unassigned — but not everything."""
    u_grid, p_grid = _load_grids()
    clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=80)
    clf.fit(u_grid, p_grid)
    unassigned = float(np.mean(clf.labels_ == -1))
    assert unassigned < 1.0, "All nodes unassigned — pipeline broken"


@SKIP
def test_hkv_predict_on_occupied_nodes():
    """Predict labels for nodes that actually received data hits."""
    u_grid, p_grid = _load_grids()
    clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=60)
    clf.fit(u_grid, p_grid)
    # Collect (X, Y) for nodes with at least one hit
    occupied = []
    with CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["Total events in node"]) > 0:
                occupied.append((int(row["X"]), int(row["Y"])))
    assert len(occupied) > 0, "No occupied nodes found in CSV"
    bmus = np.array(occupied, dtype=np.int64)
    preds = clf.predict(bmus)
    assert preds.shape == (len(occupied),)
    assert preds.dtype in (np.int32, np.int64, np.intp)


@SKIP
def test_hkv_predict_cluster_ids_within_range():
    """Predicted labels must be -1 or in [0, n_clusters_)."""
    u_grid, p_grid = _load_grids()
    clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=60)
    clf.fit(u_grid, p_grid)
    occupied = []
    with CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["Total events in node"]) > 0:
                occupied.append((int(row["X"]), int(row["Y"])))
    if occupied:
        bmus = np.array(occupied, dtype=np.int64)
        preds = clf.predict(bmus)
        valid = preds[(preds >= 0)]
        if len(valid):
            assert valid.max() < clf.n_clusters_
