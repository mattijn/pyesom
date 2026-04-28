"""Topographic U* maps (Altair)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd


def plot_topographic_map(
    ustar: np.ndarray,
    bestmatches: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    hypsometric: bool = True,
    cell_pixels: float = 6.0,
) -> alt.Chart:
    """
    Rect heatmap of ``ustar`` with optional BMU scatter (and label color).

    Uses a hypsometric-style palette: low elevations (cluster cores) read as valley
    tones, highs as ridges.
    """
    alt.data_transformers.disable_max_rows()

    ustar = np.asarray(ustar, dtype=np.float64)
    h, w = ustar.shape
    rows = []
    for i in range(h):
        for j in range(w):
            rows.append({"col": int(j), "row": int(i), "ustar": float(ustar[i, j])})
    grid = pd.DataFrame(rows)

    chart = alt.Chart(grid).mark_rect().encode(
        x=alt.X("col:O", sort=None, title=None),
        y=alt.Y("row:O", sort="descending", title=None),
        color=alt.Color(
            "ustar:Q",
            scale=(
                alt.Scale(
                    range=["#2166ac", "#4dac26", "#b8860b", "#f5f5f5"],
                )
                if hypsometric
                else alt.Scale(scheme="blues")
            ),
            legend=alt.Legend(title="U*"),
        ),
        tooltip=["row", "col", "ustar"],
    )

    bm = np.asarray(bestmatches, dtype=np.float64)
    bdf = pd.DataFrame({"col": bm[:, 1], "row": bm[:, 0]})
    if labels is not None:
        bdf["cluster"] = np.asarray(labels)
        pts = (
            alt.Chart(bdf)
            .mark_circle(opacity=0.35, stroke="white", strokeWidth=0.3, size=24)
            .encode(
                x=alt.X("col:O", sort=None),
                y=alt.Y("row:O", sort="descending"),
                color=alt.Color("cluster:N", legend=None),
                tooltip=["row", "col", "cluster"],
            )
        )
    else:
        pts = alt.Chart(bdf).mark_circle(color="black", opacity=0.25, size=20).encode(
            x=alt.X("col:O", sort=None),
            y=alt.Y("row:O", sort="descending"),
        )

    layered = alt.layer(chart, pts).properties(
        width={"step": cell_pixels},
        height={"step": cell_pixels},
        title="Topographic map (U*)",
    )
    return layered
