from __future__ import annotations

from pathlib import Path
import sys

import altair as alt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from tuning.harness import TARGETS_BY_NAME


RESULTS_DIR = Path(__file__).parent / "results"
OUT_HTML = RESULTS_DIR / "analysis_altair.html"


def load_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(RESULTS_DIR.glob("*.tsv")):
        df = pd.read_csv(path, sep="\t")
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise SystemExit("No TSV results found in tuning/results.")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["backend_dataset"] = all_df["backend"] + " / " + all_df["dataset"]
    all_df["elapsed_s"] = pd.to_numeric(all_df["elapsed_s"], errors="coerce")
    all_df["paper_score"] = pd.to_numeric(all_df["paper_score"], errors="coerce")
    all_df["coverage"] = pd.to_numeric(all_df["coverage"], errors="coerce")
    all_df["purity"] = pd.to_numeric(all_df["purity"], errors="coerce")
    all_df["n_wrong"] = pd.to_numeric(all_df["n_wrong"], errors="coerce")
    all_df["n_clusters"] = pd.to_numeric(all_df["n_clusters"], errors="coerce")
    all_df["n_assigned"] = pd.to_numeric(all_df["n_assigned"], errors="coerce")
    all_df = all_df.dropna(subset=["paper_score", "elapsed_s", "coverage", "n_wrong"])
    all_df["paper_k"] = all_df["dataset"].map(lambda d: TARGETS_BY_NAME[d].paper_k)
    all_df["paper_wrong"] = all_df["dataset"].map(lambda d: TARGETS_BY_NAME[d].paper_wrong)
    all_df["paper_unassigned"] = all_df["dataset"].map(lambda d: TARGETS_BY_NAME[d].paper_unassigned)
    all_df["paper_n_samples"] = all_df["dataset"].map(lambda d: TARGETS_BY_NAME[d].n_samples)
    all_df["unassigned"] = all_df["paper_n_samples"] - all_df["n_assigned"]
    all_df["k_gap"] = (all_df["n_clusters"] - all_df["paper_k"]).abs()
    all_df["strict_match"] = (
        (all_df["n_clusters"] == all_df["paper_k"])
        & (all_df["n_wrong"] == all_df["paper_wrong"])
        & (all_df["unassigned"] == all_df["paper_unassigned"])
    )
    return all_df


def top_score_bar(df: pd.DataFrame) -> alt.Chart:
    best = (
        df.sort_values("paper_score", ascending=False)
        .groupby(["backend", "dataset"], as_index=False)
        .first()
    )
    return (
        alt.Chart(best)
        .mark_bar()
        .encode(
            x=alt.X("paper_score:Q", title="Best paper_score"),
            y=alt.Y("backend:N", sort="-x", title="Backend"),
            color=alt.Color("backend:N", legend=None),
            column=alt.Column("dataset:N", title="Dataset"),
            tooltip=["backend", "dataset", "paper_score", "coverage", "purity", "n_wrong", "n_clusters", "elapsed_s"],
        )
        .properties(title="Best score per backend/dataset", width=180, height=160)
        .resolve_scale(x="independent", y="independent")
    )


def elapsed_distribution_boxplot(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_boxplot(extent=1.5)
        .encode(
            x=alt.X("backend:N", title="Backend"),
            y=alt.Y("elapsed_s:Q", title="elapsed_s distribution", scale=alt.Scale(type="log")),
            color=alt.Color("backend:N", legend=None),
            column=alt.Column("dataset:N", title="Dataset"),
            tooltip=["backend", "dataset"],
        )
        .properties(title="Runtime distribution by backend (boxplot)", width=180, height=220)
        .resolve_scale(y="independent")
    )


def assignment_distribution_boxplot(df: pd.DataFrame) -> alt.Chart:
    long_df = pd.concat(
        [
            df[["backend", "dataset", "coverage"]].rename(columns={"coverage": "value"}).assign(metric="coverage"),
            df[["backend", "dataset", "n_wrong"]].rename(columns={"n_wrong": "value"}).assign(metric="n_wrong"),
        ],
        ignore_index=True,
    )
    return (
        alt.Chart(long_df)
        .mark_boxplot(extent=1.5)
        .encode(
            x=alt.X("backend:N", title="Backend"),
            y=alt.Y("value:Q", title="distribution"),
            color=alt.Color("backend:N", legend=None),
            column=alt.Column("dataset:N", title="Dataset"),
            row=alt.Row("metric:N", title=None, sort=["coverage", "n_wrong"]),
            tooltip=["backend", "dataset", "metric"],
        )
        .properties(title="Assignment metrics distribution (boxplots)", width=180, height=120)
        .resolve_scale(y="independent")
    )


def score_distribution_boxplot(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_boxplot(extent=1.5)
        .encode(
            x=alt.X("backend:N", title="Backend"),
            y=alt.Y("paper_score:Q", title="paper_score distribution"),
            color=alt.Color("backend:N", legend=None),
            column=alt.Column("dataset:N", title="Dataset"),
            tooltip=["backend", "dataset"],
        )
        .properties(title="Score distribution by backend (boxplot)", width=180, height=220)
        .resolve_scale(y="independent")
    )


def k_gap_best(df: pd.DataFrame) -> alt.Chart:
    best = (
        df.sort_values("paper_score", ascending=False)
        .groupby(["backend", "dataset"], as_index=False)
        .first()
    )
    best["wrong_gap"] = (best["n_wrong"] - best["paper_wrong"]).abs()
    best["unassigned_gap"] = (best["unassigned"] - best["paper_unassigned"]).abs()
    # Weighted total gap for quick visual ranking (K exactness is most important here).
    best["repro_gap_score"] = best["k_gap"] * 3 + best["wrong_gap"] + 0.25 * best["unassigned_gap"]

    best["label"] = best["repro_gap_score"].round(0).astype(int).astype(str)
    base = alt.Chart(best).encode(
        x=alt.X("backend:N", title="Backend"),
        y=alt.Y("dataset:N", title="Dataset"),
        tooltip=[
            "backend",
            "dataset",
            "paper_score",
            "n_clusters",
            "paper_k",
            "k_gap",
            "n_wrong",
            "paper_wrong",
            "wrong_gap",
            "unassigned",
            "paper_unassigned",
            "unassigned_gap",
            "repro_gap_score",
        ],
    )
    heat = base.mark_rect().encode(
        color=alt.Color(
            "repro_gap_score:Q",
            title="Reproduction gap score (lower better)",
            scale=alt.Scale(scheme="yelloworangered"),
        )
    )
    text = base.mark_text(fontSize=10).encode(
        text="label:N",
        color=alt.condition("datum.repro_gap_score > 1.5", alt.value("white"), alt.value("black")),
    )
    return (heat + text).properties(
        title="Reproduction gap heatmap (best config per backend/dataset)",
        width=alt.Step(40),
        height=alt.Step(40),
    )


def strict_reproduction_progress(df: pd.DataFrame) -> alt.Chart:
    work = df.copy()
    work["wrong_gap"] = (work["n_wrong"] - work["paper_wrong"]).abs()
    work["unassigned_gap"] = (work["unassigned"] - work["paper_unassigned"]).abs()
    work["components_hit"] = (
        (work["k_gap"] == 0).astype(int)
        + (work["wrong_gap"] == 0).astype(int)
        + (work["unassigned_gap"] == 0).astype(int)
    )
    # same weighting as reproduction-gap heatmap for consistency
    work["strict_gap_score"] = (
        work["k_gap"] * 3 + work["wrong_gap"] + 0.25 * work["unassigned_gap"]
    )
    # Keep the closest run to exact paper target for each backend/dataset
    closest = (
        work.sort_values("strict_gap_score", ascending=True)
        .groupby(["backend", "dataset"], as_index=False)
        .first()
    )
    closest["label"] = closest["components_hit"].astype(str) + "/3"

    base = alt.Chart(closest).encode(
        x=alt.X("backend:N", title="Backend"),
        y=alt.Y("dataset:N", title="Dataset"),
        tooltip=[
            "backend",
            "dataset",
            "label",
            "strict_gap_score",
            "k_gap",
            "wrong_gap",
            "unassigned_gap",
            "paper_score",
            "n_clusters",
            "n_wrong",
            "unassigned",
        ],
    )
    heat = base.mark_rect().encode(
        color=alt.Color(
            "components_hit:Q",
            title="Exact target components hit",
            scale=alt.Scale(domain=[0, 3], scheme="blues"),
        )
    )
    txt = base.mark_text(fontSize=11).encode(
        text="label:N",
        color=alt.condition("datum.components_hit >= 2", alt.value("white"), alt.value("black")),
    )
    return (heat + txt).properties(
        title="Strict reproduction progress (closest run per backend/dataset)",
        width=alt.Step(42),
        height=alt.Step(34),
    )


def backend_reliability(df: pd.DataFrame) -> alt.Chart:
    best = (
        df.sort_values("paper_score", ascending=False)
        .groupby(["backend", "dataset"], as_index=False)
        .first()
    )
    # Rank backends within each dataset by best paper_score
    best["dataset_rank"] = best.groupby("dataset")["paper_score"].rank(
        method="min", ascending=False
    )
    rel = (
        best.groupby("backend", as_index=False)
        .agg(
            wins=("dataset_rank", lambda s: int((s == 1).sum())),
            top2=("dataset_rank", lambda s: int((s <= 2).sum())),
            datasets_with_results=("dataset", "nunique"),
            mean_best_score=("paper_score", "mean"),
        )
    )
    rel["wins_rate"] = rel["wins"] / 6.0
    rel["top2_rate"] = rel["top2"] / 6.0
    rel["coverage_rate"] = rel["datasets_with_results"] / 6.0
    rel["summary_score"] = 0.5 * rel["wins_rate"] + 0.3 * rel["top2_rate"] + 0.2 * rel["coverage_rate"]

    # Heatmap-like table of backend reliability components
    table = rel.melt(
        id_vars=["backend", "wins", "top2", "datasets_with_results", "mean_best_score", "summary_score"],
        value_vars=["wins_rate", "top2_rate", "coverage_rate", "mean_best_score"],
        var_name="metric",
        value_name="rate",
    )
    table["metric_label"] = table["metric"].map(
        {
            "wins_rate": "wins/6",
            "top2_rate": "top2/6",
            "coverage_rate": "coverage/6",
            "mean_best_score": "mean best score",
        }
    )
    table["label"] = table.apply(
        lambda r: (
            f"{int(r['wins'])}/6"
            if r["metric"] == "wins_rate"
            else f"{int(r['top2'])}/6"
            if r["metric"] == "top2_rate"
            else f"{int(r['datasets_with_results'])}/6"
            if r["metric"] == "coverage_rate"
            else f"{r['mean_best_score']:.3f}"
        ),
        axis=1,
    )

    table_base = alt.Chart(table).encode(
        x=alt.X("metric_label:N", title=None),
        y=alt.Y("backend:N", sort=alt.SortField(field="summary_score", order="descending"), title="Backend"),
        tooltip=[
            "backend",
            "metric_label",
            "rate",
            "wins",
            "top2",
            "datasets_with_results",
            "mean_best_score",
            "summary_score",
        ],
    )
    table_rect = table_base.mark_rect().encode(
        color=alt.Color("rate:Q", scale=alt.Scale(scheme="blues"), title="Value")
    )
    table_text = table_base.mark_text(fontSize=11).encode(
        text="label:N",
        color=alt.condition("datum.rate > 0.6", alt.value("white"), alt.value("black")),
    )
    table_chart = (table_rect + table_text).properties(
        title="Backend reliability table",
        width=alt.Step(90),
        height=alt.Step(28),
    )

    rank_bar = (
        alt.Chart(rel)
        .mark_bar()
        .encode(
            x=alt.X("summary_score:Q", title="Composite reliability score"),
            y=alt.Y("backend:N", sort=alt.SortField(field="summary_score", order="descending"), title="Backend"),
            color=alt.Color("backend:N", legend=None),
            tooltip=[
                "backend",
                "summary_score",
                "wins",
                "top2",
                "datasets_with_results",
                "mean_best_score",
            ],
        )
        .properties(title="Backend reliability ranking", width=420, height=140)
    )
    return alt.vconcat(table_chart, rank_bar, spacing=8)


def main() -> None:
    alt.data_transformers.disable_max_rows()
    df = load_results()
    chart = alt.vconcat(
        backend_reliability(df),
        top_score_bar(df),
        score_distribution_boxplot(df),
        k_gap_best(df),
        strict_reproduction_progress(df),
        elapsed_distribution_boxplot(df),
        assignment_distribution_boxplot(df),
        spacing=24,
    ).resolve_scale(color="shared", x="independent", y="independent")
    chart.save(str(OUT_HTML))
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()

