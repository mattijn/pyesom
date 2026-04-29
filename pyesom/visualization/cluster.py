"""Cluster label map — Altair rect chart with proper legend."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd


def plot_cluster_map(
    labels: np.ndarray,
    title: str = "Cluster map",
    *,
    x_col: str = "col",
    y_col: str = "row",
    cell_pixels: float = 7.0,
    unassigned_color: str = "#d8d8d8",
    scheme: str = "category20",
) -> alt.LayerChart:
    """
    Rect heatmap of a cluster label grid.

    Unassigned nodes (label == -1) are drawn in ``unassigned_color`` and are
    intentionally excluded from the legend.  Assigned clusters use the
    ``scheme`` nominal color palette (default: ``"category20"``).

    Parameters
    ----------
    labels:
        2-D integer array of shape ``(rows, cols)``; -1 means unassigned.
    title:
        Chart title.
    x_col / y_col:
        Column names to use in the produced DataFrame.  Override to ``"X"``
        and ``"Y"`` when the grid uses geographic coordinate names.
    cell_pixels:
        Width/height of each cell in pixels.
    unassigned_color:
        Fill colour for boundary / unassigned nodes.
    scheme:
        Vega-Lite color scheme for the cluster nominal encoding.
        ``"category20"`` gives 20 visually distinct colours.

    X/Y ordinal axes are omitted (``axis=None``).
    """
    alt.data_transformers.disable_max_rows()

    labels = np.asarray(labels, dtype=np.int32)
    h, w = labels.shape

    rows = [
        {x_col: int(j), y_col: int(i), "cluster": int(labels[i, j])}
        for i in range(h)
        for j in range(w)
    ]
    df = pd.DataFrame(rows)

    xy_enc = dict(
        x=alt.X(f"{x_col}:O", sort=None, axis=None),
        y=alt.Y(f"{y_col}:O", sort="descending", axis=None),
    )
    props = dict(width={"step": cell_pixels}, height={"step": cell_pixels})

    # Layer 1 — grey background for *all* nodes (including unassigned)
    base = (
        alt.Chart(df)
        .mark_rect()
        .encode(**xy_enc, color=alt.value(unassigned_color))
        .properties(**props)
    )

    # Layer 2 — assigned nodes only; legend shows only real cluster IDs
    overlay = (
        alt.Chart(df)
        .transform_filter(alt.datum.cluster >= 0)
        .mark_rect()
        .encode(
            **xy_enc,
            color=alt.Color(
                "cluster:N",
                scale=alt.Scale(scheme=scheme),
                legend=alt.Legend(title="Cluster"),
            ),
            tooltip=[x_col, y_col, "cluster"],
        )
        .properties(**props, title=title)
    )

    return alt.layer(base, overlay)
