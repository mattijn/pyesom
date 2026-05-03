# TopoSwarm

Julia implementation of the swarm-projection step in the **TopoSwarm** pipeline.
Here is what it looks like live: the swarm projecting 212 points from the FCPS hepta benchmark in real time:

<img src="toposwarm.gif" width="50%" alt="TopoSwarm live, hepta, 7 clusters, no k" />

[Watch on asciinema](https://asciinema.org/a/1007636)

Given a pairwise distance matrix over SOM prototype nodes, TopoSwarm places each node as a *bot* on a toroidal grid and iteratively minimises a sum-of-squared-distance-residuals stress, producing a topology-preserving 2-D layout at O(n²) cost in the number of nodes (not raw samples).

## How it fits in the pipeline

```
Raw data (N × d)
    ↓  ESOM.fit()          [Python / pyesom]
SOM prototypes (k × d)
    ↓  ESOM.export_npz()   [Python / pyesom]
bridge.npz
    ↓  run_pswarm.jl       [Julia / TopoSwarm]  ← this package
result.npz
    ↓  UStarFloodClustering [Python / pyesom]
Cluster labels (N,)
```

## Installation

```julia
# from the repo root
julia --project=toposwarm -e 'using Pkg; Pkg.instantiate()'
```

Requires Julia ≥ 1.9.  Dependencies: `NPZ`, `Printf`, `Random`, `Statistics`.

## Usage

### From the command line (Quarto / subprocess)

```bash
julia --project=toposwarm toposwarm/scripts/run_pswarm.jl <bridge.npz> <result.npz>
```

**Input** `bridge.npz`:

| array | shape | description |
|---|---|---|
| `node_weights` | `(k, d)` | SOM prototype vectors |
| `hit_map` | `(k,)` | samples mapped to each node |
| `bmu_indices` | `(N,)` | flat node index per raw sample |
| `labels` | `(k,)` | optional per-node majority class label |

**Output** `result.npz`:

| array | shape | description |
|---|---|---|
| `node_rows` / `node_cols` | `(k,)` | 1-based grid coordinates per node |
| `sample_rows` / `sample_cols` | `(N,)` | grid coordinates per raw sample |
| `umatrix` | `(rows, cols)` | U-matrix (NaN for unoccupied cells) |
| `stress` | `(1,)` | final projection stress (sum-of-squared normalised distance residuals) |

### From Julia

```julia
include("toposwarm/src/TopoSwarm.jl")
using .TopoSwarm

D      = pairwise_euclidean(node_weights)   # (k × k) distance matrix
result = pswarm!(D; pop_weights=hit_map ./ sum(hit_map), seed=42)

show_stress(result.stress_history)          # ASCII convergence plot
show_grid(result; labels=node_labels)       # ASCII grid map
U = umatrix(result)                         # sparse U-matrix
```

## Key functions

| function | description |
|---|---|
| `pswarm!(D)` | main driver, returns `PswarmResult` |
| `pswarm_live!(D)` | same as `pswarm!` but redraws the grid after every epoch so you can watch the bots settle in real time |
| `pairwise_euclidean(X)` | build distance matrix from weight matrix |
| `stress_weighted(D_data, D_grid, pop_weights)` | population-weighted stress (used as the objective) |
| `umatrix(result)` | compute U-matrix over the swarm grid |
| `assign_raw_samples(result, bmu_indices)` | map raw samples to grid positions |
| `show_grid(result)` | ASCII visualisation of the grid |
| `show_stress(history)` | ASCII convergence plot |

### Live demo

Watch the swarm self-organise in real time. Works in both a terminal and a Jupyter notebook.

**Terminal:**

```bash
# toy example (9 points, 3 clusters)
julia --project=toposwarm toposwarm/scripts/run_pswarm_live.jl

# FCPS benchmark dataset
julia --project=toposwarm toposwarm/scripts/run_pswarm_live.jl hepta
# available: toy | atom | chainlink | hepta | twodiamonds | wingnut | lsun
```

**Jupyter notebook** (see `notebooks/06_showcasting_agent_swarm.ipynb`):

```julia
pswarm_live!(D; labels=labels, seed=42)
```

## Paper

The companion blog post lives in `paper/toposwarm.qmd`. It is a [Quarto](https://quarto.org) document that runs all Python and Julia code cells end-to-end and produces a self-contained `toposwarm.html`.

### Rendering locally

You need the `pyesom` Jupyter kernel, Julia, and Quarto installed.

```bash
cd toposwarm/paper
quarto render toposwarm.qmd
```

This runs the full pipeline (SOM training, swarm projection, U*F clustering, metrics) and writes `toposwarm.html`. The intermediate files `bridge.npz` and `result.npz` are gitignored.

### Publishing to GitHub Pages

The rendered HTML is committed directly to the repo (not gitignored). A GitHub Actions workflow at `.github/workflows/publish-paper.yml` triggers whenever `toposwarm/paper/toposwarm.html` is pushed to `main`. It copies the file to `_site/index.html` and deploys it via `actions/deploy-pages`. No Julia or Python is needed in CI.

```bash
# after rendering locally:
git add toposwarm/paper/toposwarm.html
git commit -m "render paper"
git push
```
