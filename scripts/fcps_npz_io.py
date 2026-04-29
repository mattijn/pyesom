"""Load FCPS-shaped benchmark tensors from ``tests/fixtures/fcps.npz`` (repo checkout).

Not part of the ``pyesom`` wheel — keep benchmark data loading out of the library package.

See ``tests/fixtures/README.md`` and ``scripts/export_fcps_npz.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

_DATASET_NAMES = (
    "atom",
    "chainlink",
    "hepta",
    "lsun",
    "twodiamonds",
    "wingnut",
)

DatasetName = Literal["atom", "chainlink", "hepta", "lsun", "twodiamonds", "wingnut"]

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_FIXTURE_REL = Path("tests") / "fixtures" / "fcps.npz"
FIXTURE_FCPS_NPZ = _REPO_ROOT / _FIXTURE_REL


def resolve_fcps_npz_path(*, search_from: Path | None = None) -> Path:
    """
    Locate ``fcps.npz``.

    Resolution order:

    1. Environment variable ``PYESOM_FCPS_NPZ``.
    2. ``<repo>/tests/fixtures/fcps.npz`` where ``repo`` is the parent of ``scripts/``
       (this file lives in ``scripts/``).
    3. Walk upward from ``search_from`` (default: current working directory);
       ``tests/fixtures/fcps.npz`` at each ancestor (running notebooks from subdirs).
    """
    env = os.environ.get("PYESOM_FCPS_NPZ")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"PYESOM_FCPS_NPZ points to missing file: {p}")

    candidate = _REPO_ROOT / _FIXTURE_REL
    if candidate.is_file():
        return candidate

    anchor = (search_from or Path.cwd()).resolve()
    for directory in [anchor, *anchor.parents]:
        c = directory / _FIXTURE_REL
        if c.is_file():
            return c

    raise FileNotFoundError(
        "Could not find fcps.npz (tests/fixtures/fcps.npz). "
        "Regenerate with: pip install '.[export-fcps]' && python scripts/export_fcps_npz.py "
        "or set PYESOM_FCPS_NPZ."
    )


def load_fcps(name: DatasetName, *, npz_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(data, cls)`` with ``data`` shape ``(n, d)`` and integer labels ``0 .. k-1``.
    """
    name_s = name.lower()
    if name_s not in _DATASET_NAMES:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {_DATASET_NAMES}")

    path = Path(npz_path).expanduser().resolve() if npz_path is not None else resolve_fcps_npz_path()
    z = np.load(path)
    return z[f"{name_s}_data"].copy(), z[f"{name_s}_cls"].copy()


def load_fcps_fixture(name: DatasetName) -> tuple[np.ndarray, np.ndarray]:
    """Load tensors from the resolved committed fixture (same default NPZ as :func:`load_fcps`)."""
    return load_fcps(name)
