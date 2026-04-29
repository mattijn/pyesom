# pyesom

**pyesom** implements large-grid **Emergent Self-Organizing Maps (ESOM)** together with **U\*** (combined U- and P-matrix topography) and **U\*F** flood-fill clustering with an automatic segmentation threshold. The methods follow the published formulations referenced below—especially Ultsch on the U-matrix and density scaling, Moutarde & Ultsch on U\*F clustering, and ESOM-style maps as described in the Ultsch & Mörchen technical report.

**ESOM training** is **backend-pluggable**: you choose who trains the weight lattice (`ESOM(..., backend=...)`). Topology helpers (**U-matrix, BMUs, hit counts**) use the same definitions regardless of backend once weights share the rectangular `(rows, cols, features)` layout.

## Install

```bash
pip install -e ".[dev]"
```

Optional extras:

- **`[bench]`** — extended benchmark scripts using scikit-learn metrics (`pip install -e ".[bench]"`).
- **`[sompy]`** — [SomPy](https://github.com/sevamoo/SOMPY) batch trainer as an ESOM backend (`pip install -e ".[sompy]"`). Pulls SomPy from GitHub (PyPI `sompy` is often broken).

## ESOM backends

| `backend` | Trainer | Notes |
|-----------|-----------|--------|
| **`minisom`** (default) | [MiniSom](https://github.com/JustGlowing/minisom) | Sequential updates; core dependency. |
| **`sompy`** | SomPy | Batch updates; requires `pip install '.[sompy]'` in the same environment as your notebook or tests. |

Example:

```python
som = ESOM(35, 42, data.shape[1], random_seed=0, backend="minisom")  # default
# som = ESOM(35, 42, data.shape[1], random_seed=0, backend="sompy")   # optional
som.fit(data, iterations=20_000)
```

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
| `pyesom.projection.esom` | ESOM (`ESOM`): MiniSom or SomPy backend; PCA/random weight init for MiniSom |
| `pyesom.topology` | `compute_umatrix`, `compute_pmatrix`, `compute_ustar` |
| `pyesom.clustering` | `UStarFloodClustering` |
| `pyesom.visualization` | Altair topographic maps & component planes |

Out of scope for v0.1 (placeholders): Pswarm projection, DBS geodesic clustering (`projection/pswarm.py`, `clustering/dbs.py`).

## Benchmarks

We benchmark against selected setups described by **Moutarde & Ultsch (2005)** ([HAL](https://hal.science/hal-00435726)). Tests load **`tests/fixtures/fcps.npz`** (tensor export from the FCPS R package, [GPL-3](https://cran.r-project.org/package=FCPS)); see `tests/fixtures/README.md`.

## References

1. Ultsch, A. (2003). *U*-Matrix: a Tool to visualize Clusters in high dimensional Data. Technical Report Nr. 36, University of Marburg.  
   <https://www.cs.ubbcluj.ro/~gabis/DocDiplome/SOM/ultsch03ustar.pdf>

2. Moutarde, F. & Ultsch, A. (2005). U\*F clustering: a new performant “cluster-mining” method based on segmentation of Self-Organizing Maps. WSOM 2005, Paris.  
   HAL: <https://hal.science/hal-00435726>

3. Ultsch, A. & Mörchen, F. (2005). ESOM-Maps: tools for clustering, visualization, and classification with Emergent SOM. Technical Report Nr. 46, University of Marburg.

4. Ultsch, A. (2005). Clustering with SOM: U\*C. In *Proceedings of the 5th Workshop on Self-Organizing Maps (WSOM 2005)*, Paris, pp. 75–82.

5. Thrun, M.C. & Ultsch, A. (2021). Swarm Intelligence for Self-Organized Clustering. *Artificial Intelligence*, Vol. 290, 103237. DOI: [10.1016/j.artint.2020.103237](https://doi.org/10.1016/j.artint.2020.103237)

6. Thrun, M.C. (2018). *Projection-Based Clustering through Self-Organization and Swarm Intelligence.* Springer Vieweg. DOI: [10.1007/978-3-658-20540-9](https://doi.org/10.1007/978-3-658-20540-9) · open-access PDF: <https://link.springer.com/content/pdf/10.1007/978-3-658-20540-9.pdf>

7. Vettigli, G. (2018). MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map.  
   <https://github.com/JustGlowing/minisom>

## Development

```bash
pytest
```
