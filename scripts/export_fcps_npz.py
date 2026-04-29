#!/usr/bin/env python3
"""Export FCPS .rda datasets to ``tests/fixtures/fcps.npz`` (committed test fixture; not in wheel).

Requires: pip install rdata
Run from repo root with local FCPS ``.rda`` files (default reads ``resources/FCPS-master/data``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fcps-data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "resources" / "FCPS-master" / "data",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fcps.npz",
    )
    args = parser.parse_args()

    import rdata

    names = ["Atom", "Chainlink", "Hepta", "Lsun3D", "TwoDiamonds", "WingNut"]
    arrays: dict[str, np.ndarray] = {}

    for name in names:
        parsed = rdata.parser.parse_file(args.fcps_data / f"{name}.rda")
        conv = rdata.conversion.convert(parsed)[name]
        data = np.asarray(conv["Data"], dtype=np.float64)
        cls = np.asarray(conv["Cls"], dtype=np.int64).ravel()
        cls = cls - cls.min()
        key = name.lower()
        arrays[f"{key}_data"] = data
        arrays[f"{key}_cls"] = cls

    lsun_data = arrays.pop("lsun3d_data")
    lsun_cls = arrays.pop("lsun3d_cls")
    arrays["lsun_data"] = lsun_data[:, :2].copy()
    arrays["lsun_cls"] = lsun_cls.copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
