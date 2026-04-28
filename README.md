# pyesom

Python implementation of the **Emergent Self-Organizing Map (ESOM)** workflow: large-grid SOM projection, **U\***-matrix (U + P density), and **U\*F** flood-fill clustering with an automatic threshold (Moutarde & Ultsch, WSOM 2005). The design mirrors the Thrun/Ultsch R ecosystem conceptually (`DatabionicSwarm`, `GeneralizedUmatrix`, `ProjectionBasedClustering`) while staying MIT-licensed and implemented from published descriptions—not translated from GPL R sources.

## Install

```bash
pip install -e ".[dev]"
```

Optional: `pip install -e ".[bench]"` for extended benchmark scripts using scikit-learn metrics.

## Quick start

```python
import numpy as np
from pyesom import ESOM, UStarFloodClustering, compute_pmatrix

data = np.random.randn(500, 4)
data = (data - data.mean(0)) / (data.std(0) + 1e-9)

som = ESOM(35, 42, data.shape[1], random_seed=0)
som.fit(data, iterations=20_000)

u = som.u_matrix()
p = compute_pmatrix(som.weights, data, pareto_fraction=None)

clf = UStarFloodClustering(min_cluster_size=5, n_threshold_steps=100)
clf.fit(u, p)
labels = clf.predict(som.bmu_indices(data))
```

## Layout

| Module | Role |
|--------|------|
| `pyesom.projection.esom` | Large-grid MiniSom wrapper (`PCA` weight init before training) |
| `pyesom.topology` | `compute_umatrix`, `compute_pmatrix`, `compute_ustar` |
| `pyesom.clustering` | `UStarFloodClustering` |
| `pyesom.visualization` | Altair topographic maps & component planes |

Out of scope for v0.1 (placeholders): Pswarm projection, DBS geodesic clustering (`projection/pswarm.py`, `clustering/dbs.py`).

## FCPS benchmarks

FCPS-related helpers live **outside** the installable package: `scripts/fcps_npz_io.py` (load/export utilities only). Generate `resources/fcps.npz` via `scripts/export_fcps_npz.py` (reads `resources/FCPS-master/data` by default; requires `pip install '.[export-fcps]'`). Set `PYESOM_FCPS_NPZ` or place `resources/fcps.npz` for tests and benchmarks. Benchmark tests skip when the archive is absent.

- Heavy metric checks: `PYESOM_FULL_FCPS=1 pytest -m integration` (needs `scikit-learn`).
- Manual exploration: `python scripts/run_fcps_benchmark.py`.

Paper-level purity depends on ESOM training budget and grid size; treat benchmarks as regression tooling rather than strict reproduction of legacy Java/R experiments unless you tune hyperparameters deliberately.

## Optional local assets

If you populate `resources/` (gitignored here) with FCPS exports and optional HKV assets, tests pick them up — for example `resources/fcps.npz` for FCPS benchmarks and `tests/test_hkv_csv.py` for the EU discharge SOM CSV.

## Development

```bash
pytest
```

See `CITATIONS.md` for references.
