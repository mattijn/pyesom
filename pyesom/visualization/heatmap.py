"""Grid heatmap — 2-D numpy array → Altair rect chart."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd


def plot_grid_heatmap(
    arr: np.ndarray,
    title: str = "",
    *,
    scheme: str = "viridis",
    value_name: str = "value",
    cell_pixels: float = 7.0,
    x_field: str = "col",
    y_field: str = "row",
    x_title: str = "SOM column",
    y_title: str = "SOM row",
) -> alt.Chart:
    """
    Rect heatmap of a 2-D numpy array.

    Parameters
    ----------
    arr:
        2-D array of shape ``(rows, cols)``.
    title:
        Chart title.
    scheme:
        Vega-Lite colour scheme for the quantitative colour encoding.
    value_name:
        Name used for the value column in the tooltip and colour legend.
    cell_pixels:
        Width and height of each cell in pixels.
    x_field / y_field:
        Column names to use in the produced DataFrame.
        Defaults to ``"col"`` / ``"row"``; override to ``"X"`` / ``"Y"``
        when the grid uses geographic coordinate names.
    x_title / y_title:
        Axis label text shown in the chart.
    """
    alt.data_transformers.disable_max_rows()

    arr = np.asarray(arr, dtype=np.float64)
    h, w = arr.shape
    df = pd.DataFrame(
        [
            {x_field: int(j), y_field: int(i), value_name: float(arr[i, j])}
            for i in range(h)
            for j in range(w)
        ]
    )

    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{x_field}:O", sort=None, title=x_title),
            y=alt.Y(f"{y_field}:O", sort="descending", title=y_title),
            color=alt.Color(
                f"{value_name}:Q",
                scale=alt.Scale(scheme=scheme),
                legend=alt.Legend(title=value_name),
            ),
            tooltip=[
                x_field,
                y_field,
                alt.Tooltip(f"{value_name}:Q", format=".4f"),
            ],
        )
        .properties(
            width={"step": cell_pixels},
            height={"step": cell_pixels},
            title=title,
        )
    )
