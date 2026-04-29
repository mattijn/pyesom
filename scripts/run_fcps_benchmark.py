#!/usr/bin/env python3
"""Print clustering-quality metrics on FCPS sets (manual / opt-in).

  PYTHONPATH=. python scripts/run_fcps_benchmark.py

Requires ``tests/fixtures/fcps.npz`` (see ``scripts/export_fcps_npz.py``) or ``PYESOM_FCPS_NPZ``.
Requires optional ``scikit-learn`` for Adjusted Rand Index.

Training defaults favour larger grids and iteration budgets suitable for exploratory runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


def contamination_fraction(pred: np.ndarray, true: np.ndarray) -> float:
    mixed = total = 0
    for c in np.unique(pred):
        if c < 0:
            continue
        m = pred == c
        u = np.unique(true[m])
        total += int(np.sum(m))
        if len(u) > 1:
            mixed += int(np.sum(m))
    return mixed / total if total else 1.0


def main() -> None:
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        adjusted_rand_score = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="atom")
    parser.add_argument("--iterations", type=int, default=40_000)
    parser.add_argument("--gx", type=int, default=55)
    parser.add_argument("--gy", type=int, default=65)
    parser.add_argument(
        "--fcps-npz",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to fcps.npz (default: resolve tests/fixtures or PYESOM_FCPS_NPZ)",
    )
    args = parser.parse_args()

    _scripts = Path(__file__).resolve().parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))

    from fcps_npz_io import load_fcps, resolve_fcps_npz_path
    from pyesom.clustering.ustar_flood import UStarFloodClustering
    from pyesom.projection.esom import ESOM
    from pyesom.topology.pmatrix import compute_pmatrix

    repo_root = Path(__file__).resolve().parents[1]
    fcps_npz = args.fcps_npz or resolve_fcps_npz_path(search_from=repo_root)
    data, cls = load_fcps(args.dataset, npz_path=fcps_npz)  # type: ignore[arg-type]
    data = zscore(data)
    som = ESOM(args.gx, args.gy, data.shape[1], random_seed=42)
    som.fit(data, iterations=args.iterations)
    u = som.u_matrix()
    p = compute_pmatrix(som.weights, data, pareto_fraction=None)
    clf = UStarFloodClustering(min_cluster_size=4, n_threshold_steps=120)
    clf.fit(u, p)
    pred = clf.predict(som.bmu_indices(data))

    mask = pred >= 0
    print(f"dataset={args.dataset} grid={args.gx}x{args.gy} iterations={args.iterations}")
    print(f"n_clusters={clf.n_clusters_} threshold={clf.threshold_:.4f}")
    print(f"fraction_unassigned={np.mean(pred < 0):.4f}")
    print(f"contamination_fraction={contamination_fraction(pred, cls):.4f}")
    if adjusted_rand_score is not None:
        print(f"adjusted_rand_score={adjusted_rand_score(cls[mask], pred[mask]):.4f}")
    else:
        print("adjusted_rand_score=(install scikit-learn)", file=sys.stderr)


if __name__ == "__main__":
    main()
