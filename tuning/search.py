"""
Autonomous hyperparameter search for paper reproduction.

Adapted from the autoresearch pattern:
  - Fixed metric / evaluation (harness.py)
  - Iterable config grid    (experiment.py)
  - This script = the loop

Usage
-----
# All backends, all datasets (long — hours):
    python tuning/search.py

# Use several CPU cores (each config is independent):
    python tuning/search.py --jobs 8

# Parallelize by (backend,dataset) combos instead of per-config:
    python tuning/search.py --jobs 6 --parallel-scope combo

# Single backend:
    python tuning/search.py --backend minisom
    python tuning/search.py --backend intrasom

# Single dataset:
    python tuning/search.py --dataset atom

# Limit experiments per (backend, dataset) combo for a quick scan:
    python tuning/search.py --max-per-combo 20

# Show current best results from TSV without running:
    python tuning/search.py --report

# Re-run after interruption: skips configs already present in the TSV (default).
# Force full re-run from scratch:
    python tuning/search.py --no-resume

Results are logged to tuning/results/<backend>_<dataset>.tsv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
import traceback
import warnings
from pathlib import Path

# Cap BLAS/OpenMP threads per process by default. This avoids oversubscription
# when we already parallelize with multiple Python subprocesses.
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repo root is on path when run directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from tuning.harness import (
    PAPER_TARGETS,
    TARGETS_BY_NAME,
    DatasetResult,
    evaluate_result,
    format_result_row,
    load_dataset,
    mean_paper_score,
    zscore,
)
from tuning.experiment import BACKEND_GRIDS, all_configs

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DEFAULT_RUN_LOG = RESULTS_DIR / "run.log"

ALL_BACKENDS = list(BACKEND_GRIDS.keys())
ALL_DATASETS = [t.dataset for t in PAPER_TARGETS]

TSV_FIELDS = [
    "backend", "dataset", "grid_rows", "grid_cols",
    # training params
    "iterations", "epochs", "sigma", "learning_rate",
    "neighborhood_function", "batch_size",
    "train_len_factor", "initialization",
    "mapshape",
    # U*F params
    "pareto_fraction", "n_threshold_steps", "threshold_anchor",
    # results
    "n_clusters", "n_assigned", "n_wrong",
    "purity", "coverage", "paper_score",
    "elapsed_s",
]

# Columns that uniquely identify a training + U*F config (used for resume / skip).
TSV_IDENTITY_FIELDS = [
    "backend", "dataset", "grid_rows", "grid_cols",
    "iterations", "epochs", "sigma", "learning_rate",
    "neighborhood_function", "batch_size",
    "train_len_factor", "initialization", "mapshape",
    "pareto_fraction", "n_threshold_steps", "threshold_anchor",
]


def _norm_identity_scalar(field: str, value: object) -> str:
    """Normalize a cell so config dicts and TSV rows compare equal."""
    if value is None:
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    if field in ("threshold_anchor", "initialization", "neighborhood_function", "mapshape"):
        return s
    try:
        x = float(s)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.12g}"
    except ValueError:
        return s


def identity_key_from_config(backend: str, dataset: str, config: dict) -> tuple[str, ...]:
    r, c = config["grid_size"]
    return (
        _norm_identity_scalar("backend", backend),
        _norm_identity_scalar("dataset", dataset),
        _norm_identity_scalar("grid_rows", r),
        _norm_identity_scalar("grid_cols", c),
        _norm_identity_scalar("iterations", config.get("iterations", "")),
        _norm_identity_scalar("epochs", config.get("epochs", "")),
        _norm_identity_scalar("sigma", config.get("sigma", "")),
        _norm_identity_scalar("learning_rate", config.get("learning_rate", "")),
        _norm_identity_scalar("neighborhood_function", config.get("neighborhood_function", "")),
        _norm_identity_scalar("batch_size", config.get("batch_size", "")),
        _norm_identity_scalar("train_len_factor", config.get("train_len_factor", "")),
        _norm_identity_scalar("initialization", config.get("initialization", "")),
        _norm_identity_scalar("mapshape", config.get("mapshape", "")),
        _norm_identity_scalar("pareto_fraction", config["pareto_fraction"]),
        _norm_identity_scalar("n_threshold_steps", config["n_threshold_steps"]),
        _norm_identity_scalar("threshold_anchor", config["threshold_anchor"]),
    )


def identity_key_from_tsv_row(row: dict) -> tuple[str, ...]:
    return tuple(
        _norm_identity_scalar(f, row.get(f, ""))
        for f in TSV_IDENTITY_FIELDS
    )


def dataset_result_from_tsv_row(dataset: str, row: dict) -> DatasetResult:
    return DatasetResult(
        dataset=dataset,
        n_clusters=int(row["n_clusters"]),
        n_assigned=int(row["n_assigned"]),
        n_wrong=int(row["n_wrong"]),
        purity=float(row["purity"]),
        coverage=float(row["coverage"]),
        paper_score=float(row["paper_score"]),
    )


def load_resume_state(path: Path) -> tuple[
    set[tuple[str, ...]],
    dict[tuple[str, ...], float],
    float,
    DatasetResult | None,
]:
    """
    Read an existing results TSV: completed config keys, best score per key,
    and overall best row (for summaries when everything is skipped).
    """
    completed: set[tuple[str, ...]] = set()
    best_score_by_key: dict[tuple[str, ...], float] = {}
    best_overall = -1.0
    best_result: DatasetResult | None = None
    if not path.is_file():
        return completed, best_score_by_key, best_overall, best_result

    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = identity_key_from_tsv_row(row)
            completed.add(key)
            score = float(row["paper_score"])
            prev = best_score_by_key.get(key)
            if prev is None or score > prev:
                best_score_by_key[key] = score
            if score > best_overall:
                best_overall = score
                ds = row.get("dataset", "")
                best_result = dataset_result_from_tsv_row(ds, row)
    return completed, best_score_by_key, best_overall, best_result


# ---------------------------------------------------------------------------
# Console → file mirroring (run.log)
# ---------------------------------------------------------------------------

class _TeeStream(io.TextIOBase):
    """Mirror writes to two text streams (stdout/stderr + log file)."""

    def __init__(self, left: io.TextIOBase, right: io.TextIOBase) -> None:
        self.left = left
        self.right = right

    def write(self, s: str) -> int:
        self.left.write(s)
        self.right.write(s)
        return len(s)

    def flush(self) -> None:
        self.left.flush()
        self.right.flush()


def enable_run_log(path: Path, append: bool) -> io.TextIOBase | None:
    """
    Mirror all console output to `path`.
    Returns the opened log handle (caller should close).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    log_file = open(path, mode, encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)
    print(f"[run-log] Writing console output to {path}")
    return log_file


def disable_run_log(log_file: io.TextIOBase | None) -> None:
    """Restore real stdout/stderr, then close the log stream."""
    if log_file is None:
        return
    if isinstance(sys.stdout, _TeeStream):
        sys.stdout = sys.stdout.left
    if isinstance(sys.stderr, _TeeStream):
        sys.stderr = sys.stderr.left
    log_file.close()


@contextlib.contextmanager
def maybe_suppress_output(enabled: bool):
    """Optionally silence stdout/stderr (used in worker subprocesses)."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# ---------------------------------------------------------------------------
# Combo-level worker (for --parallel-scope combo)
# ---------------------------------------------------------------------------

def run_combo(
    backend: str,
    dataset: str,
    max_per_combo: int | None,
    resume: bool,
    quiet_worker_output: bool,
) -> tuple[DatasetResult | None, int, int, int]:
    """
    Run one (backend, dataset) combo end-to-end.
    Returns: (best_result, n_run, n_skipped, total_configs)
    """
    with maybe_suppress_output(quiet_worker_output):
        configs = all_configs(backend, dataset)
        if max_per_combo:
            configs = configs[:max_per_combo]
        total = len(configs)

        result_path = tsv_path(backend, dataset)
        if resume:
            completed, _, best_score, best_result = load_resume_state(result_path)
        else:
            completed = set()
            best_score = -1.0
            best_result = None

        n_run = 0
        n_skipped = 0
        for config in configs:
            key = identity_key_from_config(backend, dataset, config)
            if resume and key in completed:
                n_skipped += 1
                continue
            outcome = run_one(backend, dataset, config)
            if outcome is None:
                continue
            result, elapsed = outcome
            write_tsv_row(backend, dataset, config, result, elapsed)
            n_run += 1
            if result.paper_score > best_score:
                best_score = result.paper_score
                best_result = result

    return best_result, n_run, n_skipped, total


def _pool_run_combo(
    payload: tuple[str, str, int | None, bool, bool],
) -> tuple[str, str, DatasetResult | None, int, int, int]:
    backend, dataset, max_per_combo, resume, quiet_worker_output = payload
    best_result, n_run, n_skipped, total = run_combo(
        backend=backend,
        dataset=dataset,
        max_per_combo=max_per_combo,
        resume=resume,
        quiet_worker_output=quiet_worker_output,
    )
    return backend, dataset, best_result, n_run, n_skipped, total


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_one(
    backend: str,
    dataset: str,
    config: dict,
) -> tuple[DatasetResult, float] | None:
    """
    Train one SOM + cluster + evaluate.
    Returns (DatasetResult, elapsed_seconds) or None on failure.
    """
    from pyesom import ESOM, UStarFloodClustering
    from pyesom.topology.pmatrix import compute_pmatrix

    target = TARGETS_BY_NAME[dataset]
    real_dataset = "chainlink" if dataset == "chainlink_u" else dataset
    data_raw, true_labels = load_dataset(real_dataset)
    data = zscore(data_raw)

    rows, cols = config["grid_size"]
    pareto_fraction = config["pareto_fraction"]
    n_threshold_steps = config["n_threshold_steps"]
    threshold_anchor = config["threshold_anchor"]

    # --- build backend kwargs ---
    som_kwargs: dict = {}
    fit_kwargs: dict = {}

    if backend == "minisom":
        if config.get("sigma") is not None:
            som_kwargs["sigma"] = config["sigma"]
        som_kwargs["learning_rate"] = config.get("learning_rate", 0.5)
        nbf = config.get("neighborhood_function", "gaussian")
        if nbf != "gaussian":
            som_kwargs["neighborhood_function"] = nbf
        fit_kwargs["iterations"] = config.get("iterations", 50_000)

    elif backend == "sompy":
        skw = {}
        if "train_len_factor" in config:
            skw["train_len_factor"] = config["train_len_factor"]
        if "initialization" in config:
            skw["initialization"] = config["initialization"]
        som_kwargs["sompy_kwargs"] = skw

    elif backend == "torchsom":
        tkw = {"batch_size": config.get("batch_size", 32)}
        if config.get("sigma") is not None:
            som_kwargs["sigma"] = config["sigma"]
        som_kwargs["learning_rate"] = config.get("learning_rate", 0.5)
        som_kwargs["torchsom_kwargs"] = tkw
        fit_kwargs["epochs"] = config.get("epochs", 10)

    elif backend == "intrasom":
        ikw = {"mapshape": config.get("mapshape", "toroid")}
        som_kwargs["intrasom_kwargs"] = ikw
        fit_kwargs["epochs"] = config.get("epochs", 10)

    t0 = time.perf_counter()
    try:
        som = ESOM(rows, cols, data.shape[1], backend=backend, random_seed=42, **som_kwargs)
        som.fit(data, **fit_kwargs)

        u = som.u_matrix()
        p = compute_pmatrix(som.weights, data, pareto_fraction=pareto_fraction)

        clf = UStarFloodClustering(
            min_cluster_size=1,
            n_threshold_steps=n_threshold_steps,
            threshold_anchor=threshold_anchor,
            use_ustar=target.use_ustar,
        )
        clf.fit(u, p)
        pred = clf.predict(som.bmu_indices(data))

        elapsed = time.perf_counter() - t0
        result = evaluate_result(dataset, true_labels, pred, clf.n_clusters_)
        return result, elapsed

    except Exception:
        traceback.print_exc()
        return None


def _pool_run_one(
    payload: tuple[int, str, str, dict],
) -> tuple[int, str, str, dict, DatasetResult | None, float | None]:
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    i, backend, dataset, config = payload
    out = run_one(backend, dataset, config)
    if out is None:
        return i, backend, dataset, config, None, None
    result, elapsed = out
    return i, backend, dataset, config, result, elapsed


# ---------------------------------------------------------------------------
# TSV logging
# ---------------------------------------------------------------------------

def tsv_path(backend: str, dataset: str) -> Path:
    return RESULTS_DIR / f"{backend}_{dataset}.tsv"


def write_tsv_row(backend: str, dataset: str, config: dict, result: DatasetResult, elapsed: float) -> None:
    path = tsv_path(backend, dataset)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        if write_header:
            w.writeheader()
        row = {
            "backend": backend,
            "dataset": dataset,
            "grid_rows": config["grid_size"][0],
            "grid_cols": config["grid_size"][1],
            "iterations": config.get("iterations", ""),
            "epochs": config.get("epochs", ""),
            "sigma": config.get("sigma", ""),
            "learning_rate": config.get("learning_rate", ""),
            "neighborhood_function": config.get("neighborhood_function", ""),
            "batch_size": config.get("batch_size", ""),
            "train_len_factor": config.get("train_len_factor", ""),
            "initialization": config.get("initialization", ""),
            "mapshape": config.get("mapshape", ""),
            "pareto_fraction": config["pareto_fraction"],
            "n_threshold_steps": config["n_threshold_steps"],
            "threshold_anchor": config["threshold_anchor"],
            "n_clusters": result.n_clusters,
            "n_assigned": result.n_assigned,
            "n_wrong": result.n_wrong,
            "purity": f"{result.purity:.4f}",
            "coverage": f"{result.coverage:.4f}",
            "paper_score": f"{result.paper_score:.4f}",
            "elapsed_s": f"{elapsed:.1f}",
        }
        w.writerow(row)


# ---------------------------------------------------------------------------
# Report: read TSV files and show best per (backend, dataset)
# ---------------------------------------------------------------------------

def report() -> None:
    tsvs = sorted(RESULTS_DIR.glob("*.tsv"))
    if not tsvs:
        print("No results yet — run the search first.")
        return

    # aggregate best score per (backend, dataset)
    best: dict[tuple[str, str], dict] = {}
    counts: dict[tuple[str, str], int] = {}

    for path in tsvs:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = (row["backend"], row["dataset"])
                score = float(row["paper_score"])
                counts[key] = counts.get(key, 0) + 1
                if key not in best or score > float(best[key]["paper_score"]):
                    best[key] = dict(row)

    print("\n=== Best results per (backend, dataset) ===\n")
    target_by = {t.dataset: t for t in PAPER_TARGETS}
    prev_backend = None
    for (backend, dataset), row in sorted(best.items()):
        if backend != prev_backend:
            print(f"\n── {backend} ──")
            prev_backend = backend
        target = target_by[dataset]
        n_runs = counts[(backend, dataset)]
        wrong_s = row["n_wrong"]
        cov_s = float(row["coverage"])
        pur_s = float(row["purity"])
        k_s = int(row["n_clusters"])
        score_s = float(row["paper_score"])
        k_marker = "✓" if k_s == target.paper_k else f"✗(want {target.paper_k})"
        wrong_marker = "✓" if int(wrong_s) == 0 else f"✗({wrong_s})"
        print(
            f"  {dataset:<14} [{n_runs:>4} runs]  "
            f"K={k_s:<3}{k_marker:<12}  wrong={wrong_marker:<10}  "
            f"cov={cov_s:.3f}  pur={pur_s:.3f}  score={score_s:.3f}"
        )
        # show best config
        cfg_parts = []
        for k in ("grid_rows", "grid_cols", "iterations", "epochs", "sigma",
                  "learning_rate", "train_len_factor", "initialization",
                  "mapshape", "pareto_fraction", "threshold_anchor"):
            v = row.get(k, "")
            if v:
                cfg_parts.append(f"{k}={v}")
        print(f"    config: {', '.join(cfg_parts)}")

    print()


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def search(
    backends: list[str],
    datasets: list[str],
    max_per_combo: int | None,
    jobs: int = 1,
    resume: bool = True,
    parallel_scope: str = "config",
    quiet_worker_output: bool = True,
) -> None:
    print(f"\n{'='*70}")
    print(f"  U*F paper reproduction search")
    print(f"  backends : {backends}")
    print(f"  datasets : {datasets}")
    if max_per_combo:
        print(f"  max/combo: {max_per_combo}")
    if jobs > 1:
        print(f"  jobs     : {jobs} (parallel)")
    print(f"  scope    : {parallel_scope}")
    if parallel_scope == "combo":
        print(f"  workers  : {'quiet' if quiet_worker_output else 'verbose'} output")
    if not resume:
        print(f"  resume   : off (re-run all configs)")
    print(f"{'='*70}\n")

    grand_results: list[DatasetResult] = []

    if parallel_scope == "combo" and jobs > 1:
        print("  Running combos in parallel subprocesses...")
        backend_results_map: dict[str, list[DatasetResult]] = {b: [] for b in backends}
        payloads = [
            (backend, dataset, max_per_combo, resume, quiet_worker_output)
            for backend in backends
            for dataset in datasets
        ]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut_map = {ex.submit(_pool_run_combo, p): p for p in payloads}
            total_futures = len(fut_map)
            done_count = 0
            while fut_map:
                done_set, _ = wait(
                    fut_map.keys(), timeout=5.0, return_when=FIRST_COMPLETED
                )
                if not done_set:
                    print(
                        f"  progress: done={done_count}/{total_futures}, "
                        f"running/pending={len(fut_map)}"
                    )
                    continue
                for fut in done_set:
                    backend, dataset = fut_map.pop(fut)[0:2]
                    try:
                        be, ds, best_result, n_run, n_skipped, total = fut.result()
                    except Exception:
                        traceback.print_exc()
                        print(f"  [combo crashed] {backend}/{dataset}")
                        done_count += 1
                        continue
                    print(
                        f"  done {be}/{ds}: ran={n_run}, skipped={n_skipped}, total={total}"
                    )
                    if best_result is not None:
                        backend_results_map[be].append(best_result)
                        grand_results.append(best_result)
                    done_count += 1

        for backend in backends:
            backend_results = backend_results_map[backend]
            if backend_results:
                print(f"\n  {backend} summary — best per dataset:")
                for r in sorted(backend_results, key=lambda x: datasets.index(x.dataset)):
                    print(format_result_row(r, TARGETS_BY_NAME[r.dataset]))
                print(f"  mean paper_score = {mean_paper_score(backend_results):.3f}")

        if grand_results:
            print(f"\n{'='*70}")
            print("  OVERALL BEST (one per dataset, best backend wins)")
            print(f"{'='*70}")
            best_per_dataset: dict[str, DatasetResult] = {}
            for r in grand_results:
                if r.dataset not in best_per_dataset or r.paper_score > best_per_dataset[r.dataset].paper_score:
                    best_per_dataset[r.dataset] = r
            for ds in datasets:
                if ds in best_per_dataset:
                    print(format_result_row(best_per_dataset[ds], TARGETS_BY_NAME[ds]))
            print(f"  mean paper_score = {mean_paper_score(list(best_per_dataset.values())):.3f}")
        return

    for backend in backends:
        print(f"\n{'─'*60}")
        print(f"  BACKEND: {backend}")
        print(f"{'─'*60}")

        backend_results: list[DatasetResult] = []

        for dataset in datasets:
            target = TARGETS_BY_NAME[dataset]
            configs = all_configs(backend, dataset)
            if max_per_combo:
                configs = configs[:max_per_combo]

            result_path = tsv_path(backend, dataset)
            if resume:
                completed, _, file_best_score, file_best_result = load_resume_state(
                    result_path
                )
                best_score = file_best_score
                best_result = file_best_result
            else:
                completed = set()
                score_by_key = {}
                best_score = -1.0
                best_result = None

            print(f"\n  Dataset: {dataset}  ({len(configs)} configs)")
            print(f"  Paper target: K={target.paper_k}, wrong={target.paper_wrong}, "
                  f"unassigned={target.paper_unassigned}/{target.n_samples}")

            if resume and completed:
                n_pending = sum(
                    1
                    for c in configs
                    if identity_key_from_config(backend, dataset, c) not in completed
                )
                n_skip = len(configs) - n_pending
                if n_skip:
                    print(
                        f"  Resume: {n_skip}/{len(configs)} already in {result_path.name} "
                        f"→ {n_pending} to run"
                    )

            def _print_config_line(i: int, config: dict) -> None:
                rows, cols = config["grid_size"]
                print(f"    [{i:>3}/{len(configs)}] grid=({rows},{cols})", end="", flush=True)
                kp = []
                for k in ("iterations", "epochs", "sigma", "learning_rate",
                          "train_len_factor", "mapshape", "pareto_fraction"):
                    v = config.get(k)
                    if v is not None:
                        kp.append(f"{k}={v}")
                print(f"  {', '.join(kp)}", end="", flush=True)

            def _run_and_log(i: int, config: dict, outcome: tuple[DatasetResult, float] | None) -> None:
                nonlocal best_score, best_result
                _print_config_line(i, config)
                if outcome is None:
                    print("  → FAILED")
                    return
                result, elapsed = outcome
                write_tsv_row(backend, dataset, config, result, elapsed)
                marker = "★" if result.paper_score > best_score else " "
                print(
                    f"  → K={result.n_clusters} wrong={result.n_wrong} "
                    f"cov={result.coverage:.3f} pur={result.purity:.3f} "
                    f"score={result.paper_score:.3f} ({elapsed:.0f}s) {marker}"
                )
                if result.paper_score > best_score:
                    best_score = result.paper_score
                    best_result = result

            if jobs <= 1:
                for i, config in enumerate(configs, 1):
                    key = identity_key_from_config(backend, dataset, config)
                    if resume and key in completed:
                        continue
                    _run_and_log(i, config, run_one(backend, dataset, config))
            else:
                pool_payloads: list[tuple[int, str, str, dict]] = []
                for i, config in enumerate(configs, 1):
                    key = identity_key_from_config(backend, dataset, config)
                    if resume and key in completed:
                        continue
                    pool_payloads.append((i, backend, dataset, config))
                if pool_payloads:
                    with ProcessPoolExecutor(max_workers=jobs) as ex:
                        future_to_payload = {
                            ex.submit(_pool_run_one, p): p for p in pool_payloads
                        }
                        for fut in as_completed(future_to_payload):
                            try:
                                i, be, ds, cfg, result, elapsed = fut.result()
                            except Exception:
                                traceback.print_exc()
                                print("    [pool] worker crashed")
                                continue
                            if result is None or elapsed is None:
                                _run_and_log(i, cfg, None)
                            else:
                                _run_and_log(i, cfg, (result, elapsed))

            if best_result is not None:
                backend_results.append(best_result)
                grand_results.append(best_result)

        if backend_results:
            print(f"\n  {backend} summary — best per dataset:")
            for r in backend_results:
                print(format_result_row(r, TARGETS_BY_NAME[r.dataset]))
            print(f"  mean paper_score = {mean_paper_score(backend_results):.3f}")

    if grand_results:
        print(f"\n{'='*70}")
        print("  OVERALL BEST (one per dataset, best backend wins)")
        print(f"{'='*70}")
        # keep best result per dataset across all backends
        best_per_dataset: dict[str, DatasetResult] = {}
        for r in grand_results:
            if r.dataset not in best_per_dataset or r.paper_score > best_per_dataset[r.dataset].paper_score:
                best_per_dataset[r.dataset] = r
        for ds in datasets:
            if ds in best_per_dataset:
                print(format_result_row(best_per_dataset[ds], TARGETS_BY_NAME[ds]))
        print(f"  mean paper_score = {mean_paper_score(list(best_per_dataset.values())):.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="U*F hyperparameter search — autoresearch style"
    )
    parser.add_argument(
        "--backend", default="all",
        help=f"Backend to search (default: all). Choices: {ALL_BACKENDS + ['all']}"
    )
    parser.add_argument(
        "--dataset", default="all",
        help=f"Dataset to search (default: all). Choices: {ALL_DATASETS + ['all']}"
    )
    parser.add_argument(
        "--max-per-combo", type=int, default=None,
        metavar="N",
        help="Limit to first N configs per (backend, dataset) — for quick scan"
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, metavar="N",
        help="Parallel worker processes (default: 1). Each config is independent; "
             "try a few less than your CPU core count.",
    )
    parser.add_argument(
        "--parallel-scope", choices=("config", "combo"), default="config",
        help="Parallelize per-config within a combo (default) or per (backend,dataset) combo.",
    )
    parser.add_argument(
        "--verbose-workers", action="store_true",
        help="In combo scope, keep worker subprocess stdout/stderr visible. "
             "Default is quiet workers for cleaner logs.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not skip configs already present in the results TSV (full re-run). "
             "Default is to resume: only run missing configs.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print best results from existing TSV files and exit"
    )
    parser.add_argument(
        "--run-log", default=str(DEFAULT_RUN_LOG),
        help="Path for mirrored console log (default: tuning/results/run.log). "
             "Use --run-log '' to disable.",
    )
    parser.add_argument(
        "--append-run-log", action="store_true",
        help="Append to run log instead of overwriting it each start.",
    )
    args = parser.parse_args()

    run_log_handle: io.TextIOBase | None = None
    if args.run_log.strip():
        run_log_handle = enable_run_log(Path(args.run_log), append=args.append_run_log)

    if args.report:
        try:
            report()
        finally:
            disable_run_log(run_log_handle)
        return

    backends = ALL_BACKENDS if args.backend == "all" else [args.backend]
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for b in backends:
        if b not in ALL_BACKENDS:
            parser.error(f"Unknown backend {b!r}. Choose from: {ALL_BACKENDS}")
    for d in datasets:
        if d not in ALL_DATASETS:
            parser.error(f"Unknown dataset {d!r}. Choose from: {ALL_DATASETS}")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    try:
        search(
            backends,
            datasets,
            args.max_per_combo,
            jobs=args.jobs,
            resume=not args.no_resume,
            parallel_scope=args.parallel_scope,
            quiet_worker_output=not args.verbose_workers,
        )
    finally:
        disable_run_log(run_log_handle)


if __name__ == "__main__":
    main()
