"""
Experiment configuration — this is the file to iterate on (autoresearch style).

Defines the parameter GRID for each backend. The search runner in search.py
will sweep every combination and log results to tuning/results/*.tsv.

Inspired by autoresearch/train.py: keep the scope tight, one file to hack.

WHAT TO CHANGE HERE:
  - Add/remove entries in the per-backend grids
  - Adjust shared_grid for pareto_fraction / threshold sweep
  - Change dataset_grid_sizes to try non-paper grid dimensions
  - Add new backend sections following the existing pattern
"""

from __future__ import annotations

from itertools import product
from typing import Any


# ---------------------------------------------------------------------------
# Grid sizes per dataset (rows, cols)
# Paper: 50×82 for most, 40×50 for TwoDiamonds, toroidal 50×82 for Atom
# We also try ±10% variants to see sensitivity.
# ---------------------------------------------------------------------------

DATASET_GRID_SIZES: dict[str, list[tuple[int, int]]] = {
    "atom":        [(50, 82)],                        # paper size only (toroid via intrasom)
    "lsun":        [(50, 82), (40, 60), (60, 100)],
    "wingnut":     [(50, 82), (40, 60), (60, 100)],
    "chainlink":   [(50, 82), (40, 60), (60, 100)],
    "chainlink_u": [(50, 82)],
    "twodiamonds": [(40, 50), (30, 40), (50, 60)],
}


# ---------------------------------------------------------------------------
# Shared U*F parameters (swept for all backends)
# ---------------------------------------------------------------------------

SHARED_GRID: list[dict[str, Any]] = [
    {"pareto_fraction": pf, "n_threshold_steps": ns, "threshold_anchor": ta}
    for pf, ns, ta in product(
        [0.15, 0.2013, 0.25],           # pareto density radius fraction
        [100, 200],                      # threshold resolution
        ["upper"],                       # paper uses upper
    )
]


# ---------------------------------------------------------------------------
# Backend-specific training parameter grids
# ---------------------------------------------------------------------------

# MiniSom — sequential stochastic, iterations = sample presentations
MINISOM_GRID: list[dict[str, Any]] = [
    {
        "iterations": itr,
        "sigma": sigma,
        "learning_rate": lr,
        "neighborhood_function": nbf,
    }
    for itr, sigma, lr, nbf in product(
        [50_000, 150_000, 400_000],            # total sample presentations
        [None, 15, 25],                         # None = max(rows,cols)/2 auto
        [0.3, 0.5, 0.7],                        # initial learning rate
        ["gaussian"],                            # bubble is faster but coarser
    )
]

# SOMPY — batch, self-managed epoch budget scaled by train_len_factor
SOMPY_GRID: list[dict[str, Any]] = [
    {
        "train_len_factor": tlf,
        "initialization": init,
    }
    for tlf, init in product(
        [1.0, 3.0, 8.0, 20.0],
        ["pca", "random"],
    )
]

# TorchSOM — mini-batch, epochs = full data passes
TORCHSOM_GRID: list[dict[str, Any]] = [
    {
        "epochs": ep,
        "sigma": sigma,
        "learning_rate": lr,
        "batch_size": bs,
    }
    for ep, sigma, lr, bs in product(
        [50, 150, 400],
        [1.0, 8.0, 20.0],
        [0.3, 0.5],
        [32],
    )
]

# IntraSOM — batch hexagonal/toroidal; used primarily for Atom
# For non-Atom datasets we also try planar to compare fairly.
INTRASOM_GRID: list[dict[str, Any]] = [
    {
        "epochs": ep,
        "mapshape": ms,
    }
    for ep, ms in product(
        [10, 30, 80],
        ["toroid", "planar"],
    )
]

# IntraSOM for Atom specifically: always toroid (paper topology)
INTRASOM_ATOM_GRID: list[dict[str, Any]] = [
    {
        "epochs": ep,
        "mapshape": "toroid",
    }
    for ep in [10, 20, 50, 100]
]


# ---------------------------------------------------------------------------
# Registry — maps backend name → (training_grid, dataset_override_grids)
# dataset_override_grids: {dataset_name: grid} overrides the default grid
# ---------------------------------------------------------------------------

BACKEND_GRIDS: dict[str, dict[str, Any]] = {
    "minisom": {
        "training_grid": MINISOM_GRID,
        "dataset_overrides": {},
    },
    "sompy": {
        "training_grid": SOMPY_GRID,
        "dataset_overrides": {},
    },
    "torchsom": {
        "training_grid": TORCHSOM_GRID,
        "dataset_overrides": {},
    },
    "intrasom": {
        "training_grid": INTRASOM_GRID,
        "dataset_overrides": {
            "atom": INTRASOM_ATOM_GRID,
        },
    },
}


def training_configs(backend: str, dataset: str) -> list[dict[str, Any]]:
    """Return the list of training parameter dicts for (backend, dataset)."""
    entry = BACKEND_GRIDS[backend]
    overrides = entry["dataset_overrides"]
    return overrides.get(dataset, entry["training_grid"])


def all_configs(backend: str, dataset: str) -> list[dict[str, Any]]:
    """
    Cross product of training configs × shared U*F configs × grid sizes.
    Each element is a flat dict with all parameters for one experiment.
    """
    configs = []
    for train_kw in training_configs(backend, dataset):
        for shared_kw in SHARED_GRID:
            for grid_size in DATASET_GRID_SIZES[dataset]:
                configs.append({
                    "grid_size": grid_size,
                    **train_kw,
                    **shared_kw,
                })
    return configs
