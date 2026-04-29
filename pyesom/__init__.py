"""pyesom — ESOM topology, U*, and U*F clustering in Python."""

from __future__ import annotations

from pyesom.clustering.ustar_flood import UStarFloodClustering
from pyesom.projection.esom import ESOM
from pyesom.projection.lattice import SomLattice, reshape_flat_codebook
from pyesom.topology.pmatrix import compute_pmatrix
from pyesom.topology.umatrix import compute_umatrix
from pyesom.topology.ustar import compute_ustar

__version__ = "0.1.0"

__all__ = [
    "ESOM",
    "SomLattice",
    "reshape_flat_codebook",
    "UStarFloodClustering",
    "compute_pmatrix",
    "compute_umatrix",
    "compute_ustar",
]
