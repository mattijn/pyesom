"""Cluster label map — Altair rect charts with optional U*-matrix overlay."""

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


def plot_ustar_cluster_overlay(
    ustar: np.ndarray,
    labels: np.ndarray,
    title: str = "U*F cluster overlay",
    *,
    bmu_indices: np.ndarray | None = None,
    sample_labels: np.ndarray | None = None,
    x_col: str = "col",
    y_col: str = "row",
    cell_pixels: float = 7.0,
    opacity_min: float = 0.25,
    scheme: str = "category20",
    dot_scheme: str = "tableau10",
    dot_size: float = 15.0,
    dot_opacity: float = 0.7,
) -> alt.LayerChart:
    """
    Cluster colours overlaid on the U\\*-matrix, as in Moutarde & Ultsch (2005).

    Three (optionally four) visual layers:

    1. **U\\*-matrix greyscale** — dark ridges mark cluster boundaries, light
       valleys are cluster cores.
    2. **Cluster colour fill** with opacity ``1 − ustar_norm``, clamped to
       ``[opacity_min, 1]`` — neurons deep in a basin show near-opaque colour;
       neurons near a ridge that were still assigned become semi-transparent,
       revealing uncertainty.
    3. **Unassigned neurons** (``labels == -1``) — fully transparent, so only
       the dark U\\* ridge is visible, making the natural moat between clusters
       apparent.
    4. *(optional)* **BMU scatter** — one circle per sample placed at its
       best-matching unit grid position, coloured by ``sample_labels`` (or by
       the neuron's cluster label if ``sample_labels`` is not provided).

    Parameters
    ----------
    ustar:
        2-D float array of shape ``(rows, cols)``, normalised to ``[0, 1]``.
        Typically ``clf.ustar_`` from a fitted :class:`UStarFloodClustering`.
    labels:
        2-D integer array of shape ``(rows, cols)``; -1 means unassigned.
    title:
        Chart title.
    bmu_indices:
        Optional ``(n_samples, 2)`` integer array of ``(row, col)`` BMU
        positions — e.g. ``som.bmu_indices(data)``.  When provided, each
        sample is drawn as a circle on the grid.
    sample_labels:
        Optional 1-D array of length ``n_samples`` with per-sample labels
        (int or str).  Used to colour the BMU dots.  When omitted, dots
        inherit the cluster label of their BMU neuron from ``labels``.
    x_col / y_col:
        Column names (override to ``"X"``/``"Y"`` for geographic grids).
    cell_pixels:
        Width/height of each cell in pixels.
    opacity_min:
        Minimum opacity for assigned neurons (default 0.25).  Prevents very
        high-U\\* assigned neurons from disappearing entirely.
    scheme:
        Vega-Lite color scheme for the cluster rect fill (default ``"category20"``).
    dot_scheme:
        Vega-Lite color scheme for the BMU dots (default ``"tableau10"``).
        Deliberately different from ``scheme`` so dot colours cannot be
        confused with region colours — the two numbering systems are
        independent and should not share a palette.
    dot_size:
        Dot area in Vega-Lite size units (default 15).
    dot_opacity:
        Opacity of the BMU dots (default 0.7).
    """
    alt.data_transformers.disable_max_rows()

    ustar = np.asarray(ustar, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    h, w = ustar.shape

    # Normalise U* to [0, 1] in case caller passes un-normalised values
    u_min, u_max = float(ustar.min()), float(ustar.max())
    u_range = u_max - u_min if u_max > u_min else 1.0
    ustar_norm = (ustar - u_min) / u_range

    rows = []
    for i in range(h):
        for j in range(w):
            lbl = int(labels[i, j])
            u = float(ustar_norm[i, j])
            # Assigned: opacity fades toward ridges; unassigned: fully transparent
            opacity = max(opacity_min, 1.0 - u) if lbl >= 0 else 0.0
            rows.append({
                x_col: int(j),
                y_col: int(i),
                "cluster": lbl,
                "ustar": float(ustar[i, j]),
                "opacity": opacity,
            })
    df = pd.DataFrame(rows)

    xy_enc = dict(
        x=alt.X(f"{x_col}:O", sort=None, axis=None),
        y=alt.Y(f"{y_col}:O", sort="descending", axis=None),
    )
    props = dict(width={"step": cell_pixels}, height={"step": cell_pixels})

    # Layer 1 — U*-matrix greyscale background
    bg = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            **xy_enc,
            color=alt.Color(
                "ustar:Q",
                scale=alt.Scale(scheme="greys"),
                legend=alt.Legend(title="U*"),
            ),
            tooltip=[x_col, y_col, alt.Tooltip("ustar:Q", format=".4f")],
        )
        .properties(**props)
    )

    # Layer 2 — cluster colour with per-neuron opacity
    fg = (
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
            opacity=alt.Opacity("opacity:Q", scale=alt.Scale(domain=[0, 1]), legend=None),
            tooltip=[x_col, y_col, "cluster", alt.Tooltip("ustar:Q", format=".4f")],
        )
        .properties(**props, title=title)
    )

    if bmu_indices is None:
        return alt.layer(bg, fg)

    # Layer 3 — BMU scatter
    bmu_indices = np.asarray(bmu_indices, dtype=np.int32)
    if sample_labels is not None:
        slabels = np.asarray(sample_labels)
        # normalise to str so nominal colour scale works for both int and str
        dot_color_values = [str(s) for s in slabels]
        color_title = "Label"
    else:
        # look up each sample's cluster from the grid labels
        dot_color_values = [
            str(int(labels[bmu_indices[k, 0], bmu_indices[k, 1]]))
            for k in range(len(bmu_indices))
        ]
        color_title = "Cluster"

    bmu_df = pd.DataFrame({
        x_col: bmu_indices[:, 1].tolist(),
        y_col: bmu_indices[:, 0].tolist(),
        color_title: dot_color_values,
    })

    scatter = (
        alt.Chart(bmu_df)
        .mark_circle(size=dot_size, opacity=dot_opacity, stroke="white", strokeWidth=0.3)
        .encode(
            x=alt.X(f"{x_col}:O", sort=None, axis=None),
            y=alt.Y(f"{y_col}:O", sort="descending", axis=None),
            color=alt.Color(
                f"{color_title}:N",
                scale=alt.Scale(scheme=dot_scheme),
                legend=alt.Legend(title=color_title),
            ),
            tooltip=[x_col, y_col, color_title],
        )
        .properties(width={"step": cell_pixels}, height={"step": cell_pixels})
    )

    return alt.layer(bg, fg, scatter)
