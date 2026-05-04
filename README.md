# pyesom

> **Work in progress.** This is an ongoing personal project, not a finished product. The note below is just me writing down how things unfolded and where I’m heading.

## How I got here

I want to see if we now can find an automated way to cluster the U-matrix that comes out of a SOM. I used SOMPy years ago and liked the results, but the lack of documentation always made it feel inaccessible. Coming back after some time away, I expected things to have settled a bit more, but instead found that the ecosystem is more fragmented than before. More libraries, still no clear standard, and the clustering results and quality differ more across backends than I expected. Some experiments on FCPS datasets (results in [`tuning/results/`](tuning/results/README.md)) confirmed that. So I tried to build a **pluggable backend structure** with consistent U-matrix, P-matrix, and U\* helpers underneath, so at least the topology layer stays the same regardless of who trained the weights.

From Moutarde & Ultsch on U\*F segmentation I ended up at Thrun & Ultsch on **swarm-based self-organized clustering**: the idea of data bots moving through the SOM lattice as a swarm. I wanted to actually build it rather than just read about it, and I also wanted an excuse to try something new, so the swarm experiments live in **Julia** under [`toposwarm/`](toposwarm/). Good choice; Julia made me think more carefully about the geometry.

Now I’m applying what I have to real cases. One thing I already run into: **the ridges that U\*F flood-fill relies on are often not prominent enough**, even when the U-matrix looks fine visually. The method assumes more structure than practice reliably delivers. Worth keeping in mind before expecting the automatic threshold to just work.

**Things I still want to explore:**

- **1D swarming.** What happens when the swarm runs on a 1D lattice? My guess is it should sort itself into something like a bead-chain ordering; curious to see if that holds.
- **Triplet loss and human similarity judgments.** Garrido Valenzuela et al. (see reference 8) use a triplet loss to embed spaces according to *human* perception. I keep thinking that could help with automated clustering: either as a refinement step or as a weight on the U-matrix landscape.
- **More on automated clustering beyond U\*F.** Thrun & Ultsch (2020, reference 9) on projection-based clustering looks promising for finding both distance- and density-based clusters in high-dimensional data, possibly a useful complement or alternative.

One thing from Thrun’s writing that genuinely stuck with me: using **complete numbers** (I prefer that over “complex”) to store 2D coordinates, *x* in the real part, *y* in the imaginary part. When I read that I just thought: yes, of course; the type carries the geometry for free. I want to see where else that idea fits naturally.

---

**pyesom** implements large-grid **Emergent Self-Organizing Maps (ESOM)** together with **U\*** (combined U- and P-matrix topography) and **U\*F** flood-fill clustering with an automatic segmentation threshold. The methods follow the published formulations referenced below, especially Ultsch on the U-matrix and density scaling, Moutarde & Ultsch on U\*F clustering, and ESOM-style maps as described in the Ultsch & Mörchen technical report.

**ESOM training** is **backend-pluggable**: you choose who trains the weight lattice (`ESOM(..., backend=...)`). Topology helpers (**U-matrix, BMUs, hit counts**) use the same definitions regardless of backend once weights share the rectangular `(rows, cols, features)` layout.

## Install

```bash
pip install -e "."          # core (minisom backend included)
pip install -e ".[dev]"     # + pytest, ruff, jupyter
```

Optional extras:

| extra | installs | when you need it |
|---|---|---|
| `[intrasom]` | [IntraSOM](https://github.com/InTRA-USP/IntraSOM) | toroidal + hexagonal lattice; recommended for TopoSwarm |
| `[bench]` | scikit-learn | ARI / NMI benchmark metrics |
| `[sompy]` | SomPy (from GitHub) | SomPy batch trainer backend |
| `[torchsom]` | torchsom + PyTorch | GPU-capable batch trainer backend |

```bash
pip install -e ".[intrasom]"           # TopoSwarm recommended
pip install -e ".[dev,bench]"          # development + benchmarks
pip install -e ".[dev,bench,intrasom]" # everything useful
```

## ESOM backends

| `backend` | Trainer | Notes |
|-----------|-----------|--------|
| **`minisom`** (default) | [MiniSom](https://github.com/JustGlowing/minisom) | Sequential updates; lightweight core dependency. |
| **`intrasom`** | [IntraSOM](https://github.com/InTRA-USP/IntraSOM) | Batch trainer; supports toroidal and hexagonal lattice. Recommended for TopoSwarm. |
| **`sompy`** | SomPy | Batch updates; requires `pip install '.[sompy]'`. |
| **`torchsom`** | [torchsom](https://pypi.org/project/torchsom/) | PyTorch batch trainer; requires `pip install '.[torchsom]'`. |

Example:

```python
som = ESOM(n_nodes=50, random_seed=0)                      # minisom by default
# som = ESOM(n_nodes=50, backend="intrasom",               # recommended for TopoSwarm
#            intrasom_kwargs={"mapshape": "toroid", "lattice": "hexa"})
som.fit(data, epochs=20)
```

## Quick start

```python
import numpy as np
from pyesom import ESOM, UStarFloodClustering, compute_pmatrix

data = np.random.randn(500, 4)
data = (data - data.mean(0)) / (data.std(0) + 1e-9)

som = ESOM(n_nodes=50, random_seed=0)
som.fit(data, epochs=20)

u = som.u_matrix()
p = compute_pmatrix(som.weights, data, pareto_fraction=None)

clf = UStarFloodClustering(min_cluster_size=5, threshold_anchor="upper")
clf.fit(u, p)
labels = clf.predict(som.bmu_indices(data))
```

## Layout

| Module / path | Role |
|--------|------|
| `pyesom.projection.esom` | `ESOM`: pluggable backends (minisom, intrasom, sompy, torchsom) |
| `pyesom.topology` | `compute_umatrix`, `compute_pmatrix`, `compute_ustar` |
| `pyesom.clustering` | `UStarFloodClustering`: U*F flood-fill with automatic threshold |
| `pyesom.visualization` | Altair topographic maps and component planes |
| `toposwarm/` | Julia swarm-projection stage; see [`toposwarm/README.md`](toposwarm/README.md) |

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

8. Garrido Valenzuela, F.O., Cats, O. & van Cranenburgh, S. (2025). From pixels to perceptions: using human similarity judgments to enrich urban space embeddings. *International Journal of Geographical Information Science*. DOI: [10.1080/13658816.2025.2595658](https://doi.org/10.1080/13658816.2025.2595658)

9. Thrun, M.C. & Ultsch, A. (2020). Using Projection based Clustering to Find Distance and Density based Clusters in High-Dimensional Data. *Journal of Classification*. DOI: [10.1007/s00357-020-09373-2](https://doi.org/10.1007/s00357-020-09373-2)

10. Moosavi, V., Packmann, S. & Vallés, I. (2014). *SOMPY: A Python Library for Self Organizing Map (SOM)*. GitHub.  
    <https://github.com/sevamoo/SOMPY>

11. de Gouvêa, R.C.T., Gioria, R.S., Marques, G.R. & Carneiro, C.C. (2023). IntraSOM: A comprehensive Python library for Self-Organizing Maps with hexagonal toroidal maps training and missing data handling. *Software Impacts*, 17, 100570. DOI: [10.1016/j.simpa.2023.100570](https://doi.org/10.1016/j.simpa.2023.100570)

12. Berthier, L., Shokry, A., Moreaud, M., Ramelet, G. & Moulines, E. (2025). torchsom: The Reference PyTorch Library for Self-Organizing Maps. *arXiv preprint* arXiv:2510.11147. DOI: [10.48550/arXiv.2510.11147](https://doi.org/10.48550/arXiv.2510.11147)

## Development

```bash
pytest
```
