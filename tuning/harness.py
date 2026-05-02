"""Fixed evaluation harness — do NOT modify during search.

Analogous to prepare.py in autoresearch: datasets, paper targets, metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _fcps_path() -> Path:
    for d in [Path.cwd(), *Path.cwd().parents]:
        p = d / "tests" / "fixtures" / "fcps.npz"
        if p.is_file():
            return p
    raise FileNotFoundError("tests/fixtures/fcps.npz not found — run from pyesom root")


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(_fcps_path())
    return z[f"{name}_data"].copy(), z[f"{name}_cls"].copy()


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


# ---------------------------------------------------------------------------
# Paper targets  (from §3 of Moutarde & Ultsch 2005)
# ---------------------------------------------------------------------------

class PaperTarget(NamedTuple):
    dataset: str
    n_true_clusters: int
    paper_k: int           # clusters the paper found
    paper_wrong: int       # misclassified samples
    paper_unassigned: int  # samples left outside any cluster
    n_samples: int
    use_ustar: bool        # True = U*F on U*, False = U*F on U-matrix


PAPER_TARGETS: list[PaperTarget] = [
    PaperTarget("atom",        2, 2, 0,  0,   800, True),
    PaperTarget("lsun",        3, 3, 0,  5,   404, True),
    PaperTarget("wingnut",     2, 2, 0,  89,  1016, True),
    PaperTarget("chainlink",   2, 3, 0,  55,  1000, True),   # §3.2: 3 regions, split
    PaperTarget("chainlink_u", 2, 2, 0,  0,   1000, False),  # §3.3 variant: U-matrix
    PaperTarget("twodiamonds", 2, 2, 0,  71,  800, True),
]

TARGETS_BY_NAME = {t.dataset: t for t in PAPER_TARGETS}


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class DatasetResult(NamedTuple):
    dataset: str
    n_clusters: int
    n_assigned: int
    n_wrong: int
    purity: float          # fraction correct among assigned (0–1)
    coverage: float        # fraction assigned (0–1)
    paper_score: float     # composite 0–1 vs paper target (see below)


def _hungarian_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[int, float]:
    """
    Optimal assignment purity among assigned samples (pred != -1).

    Returns (n_wrong, purity) where n_wrong = samples misclassified under
    the best cluster-to-class mapping.
    """
    mask = pred_labels != -1
    if mask.sum() == 0:
        return 0, 0.0

    true_a = true_labels[mask]
    pred_a = pred_labels[mask]

    true_ids = np.unique(true_a)
    pred_ids = np.unique(pred_a)
    n_true = len(true_ids)
    n_pred = len(pred_ids)

    # cost matrix: rows = pred clusters, cols = true clusters
    size = max(n_pred, n_true)
    cost = np.zeros((size, size), dtype=np.int64)
    for pi, p in enumerate(pred_ids):
        for ti, t in enumerate(true_ids):
            cost[pi, ti] = int(np.sum((pred_a == p) & (true_a == t)))

    row_ind, col_ind = linear_sum_assignment(-cost)
    n_correct = int(cost[row_ind, col_ind].sum())
    n_assigned = int(mask.sum())
    n_wrong = n_assigned - n_correct
    purity = n_correct / n_assigned
    return n_wrong, purity


def evaluate_result(
    dataset: str,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    n_clusters: int,
) -> DatasetResult:
    target = TARGETS_BY_NAME[dataset]
    n_samples = len(true_labels)
    n_assigned = int((pred_labels != -1).sum())
    coverage = n_assigned / n_samples

    n_wrong, purity = _hungarian_purity(true_labels, pred_labels)

    # Paper score: weighted composite
    #   40% — purity (0 misclassifications is the key paper claim)
    #   30% — coverage (how many samples got assigned)
    #   30% — correct number of clusters
    k_score = 1.0 if n_clusters == target.paper_k else max(0.0, 1.0 - abs(n_clusters - target.paper_k) / target.paper_k)
    paper_score = 0.4 * purity + 0.3 * coverage + 0.3 * k_score

    return DatasetResult(
        dataset=dataset,
        n_clusters=n_clusters,
        n_assigned=n_assigned,
        n_wrong=n_wrong,
        purity=purity,
        coverage=coverage,
        paper_score=paper_score,
    )


def mean_paper_score(results: list[DatasetResult]) -> float:
    return float(np.mean([r.paper_score for r in results]))


def format_result_row(r: DatasetResult, target: PaperTarget) -> str:
    k_marker = "✓" if r.n_clusters == target.paper_k else f"✗(want {target.paper_k})"
    wrong_marker = "✓" if r.n_wrong == 0 else f"✗({r.n_wrong})"
    return (
        f"  {r.dataset:<14} K={r.n_clusters:<3}{k_marker:<12} "
        f"wrong={wrong_marker:<10} "
        f"coverage={r.coverage:.3f}  purity={r.purity:.3f}  score={r.paper_score:.3f}"
    )
