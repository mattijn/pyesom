"""Load FCPS-shaped benchmark arrays from a local ``.npz`` export (development / benchmarks only).

Not part of the ``pyesom`` installable package: keep FCPS-derived assets and GPL-related
tooling out of the library wheel. See ``export_fcps_npz.py`` and ``resources/fcps.npz``.
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


def resolve_fcps_npz_path(*, search_from: Path | None = None) -> Path:
    """
    Locate ``fcps.npz`` for :func:`load_fcps`.

    Resolution order:

    1. Environment variable ``PYESOM_FCPS_NPZ`` (path to the file).
    2. Starting at ``search_from`` (default: current working directory),
       walk upward; at each directory, if ``<dir>/resources/fcps.npz`` exists,
       return it.

    Raises
    ------
    FileNotFoundError
        If no file is found. The message lists how to export the archive
        (``scripts/export_fcps_npz.py``) and set ``PYESOM_FCPS_NPZ``.
    """
    env = os.environ.get("PYESOM_FCPS_NPZ")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"PYESOM_FCPS_NPZ points to missing file: {p}")

    anchor = (search_from or Path.cwd()).resolve()
    for directory in [anchor, *anchor.parents]:
        candidate = directory / "resources" / "fcps.npz"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find FCPS data archive (resources/fcps.npz). "
        "From repo root: pip install '.[export-fcps]' && python scripts/export_fcps_npz.py "
        "or set PYESOM_FCPS_NPZ to the path of fcps.npz."
    )


def load_fcps(name: DatasetName, *, npz_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(data, cls)`` with ``data`` shape ``(n, d)`` and integer labels ``0 .. k-1``.

    Parameters
    ----------
    name
        One of the FCPS dataset keys in the export.
    npz_path
        Path to ``fcps.npz``. If omitted, :func:`resolve_fcps_npz_path` is used.

    Notes
    -----
    The ``lsun`` export uses the first two dimensions of FCPS ``Lsun3D``; class ``3`` is a
    small outlier group (4 points). When benchmarking against ``true_k=3``, restrict
    evaluation to samples with ``cls < 3``.
    """
    name_s = name.lower()
    if name_s not in _DATASET_NAMES:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {_DATASET_NAMES}")

    path = Path(npz_path).expanduser().resolve() if npz_path is not None else resolve_fcps_npz_path()
    z = np.load(path)
    return z[f"{name_s}_data"].copy(), z[f"{name_s}_cls"].copy()
