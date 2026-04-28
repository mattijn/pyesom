"""Component planes — feature-wise maps over the ESOM grid."""

from __future__ import annotations

from typing import Sequence

import altair as alt
import numpy as np
import pandas as pd

from pyesom.projection.esom import ESOM


def plot_component_planes(
    som: ESOM,
    data: np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    cell_pixels: float = 8.0,
) -> alt.Chart:
    """
    Faceted rect charts — one plane per input feature — shared quantitative scale.
    """
    alt.data_transformers.disable_max_rows()

    data = np.asarray(data, dtype=np.float64)
    n_feat = data.shape[1]
    xdim, ydim = som.weights.shape[:2]

    if feature_names is None:
        names = [str(i) for i in range(n_feat)]
    else:
        names = list(feature_names)

    planes = []
    for fi in range(n_feat):
        planes.append(som.component_plane(data, fi))

    rows = []
    for fi in range(n_feat):
        plane = planes[fi]
        for i in range(xdim):
            for j in range(ydim):
                rows.append(
                    {
                        "col": int(j),
                        "row": int(i),
                        "feature": names[fi],
                        "value": float(plane[i, j]),
                    }
                )
    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("col:O", sort=None, title=None),
            y=alt.Y("row:O", sort="descending", title=None),
            column=alt.Column("feature:N", title=""),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["feature", "row", "col", "value"],
        )
        .properties(
            width={"step": cell_pixels},
            height={"step": cell_pixels},
        )
    )
    return chart.resolve_scale(color="shared")
