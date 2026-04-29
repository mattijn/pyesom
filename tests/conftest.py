"""Pytest hooks — NumPy 2 / sompy compat and optional bundled ``SOMPY-master`` on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# SomPy uses ``np.Inf`` (removed in NumPy 2.0).
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_SOMPY = _ROOT / "resources" / "SOMPY-master"
if _BUNDLED_SOMPY.is_dir():
    sys.path.insert(0, str(_BUNDLED_SOMPY))
