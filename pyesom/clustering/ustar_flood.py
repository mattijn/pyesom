"""U*F clustering — flood-fill segmentation with automatic threshold."""

from __future__ import annotations

from collections import deque

import numpy as np

from pyesom.topology.ustar import compute_ustar


def _normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _neighbours(i: int, j: int, h: int, w: int, toroidal: bool):
    if toroidal:
        yield (i + 1) % h, j
        yield (i - 1) % h, j
        yield i, (j + 1) % w
        yield i, (j - 1) % w
    else:
        if i + 1 < h: yield i + 1, j
        if i - 1 >= 0: yield i - 1, j
        if j + 1 < w: yield i, j + 1
        if j - 1 >= 0: yield i, j - 1


def _flood_region_size(norm: np.ndarray, threshold: float, toroidal: bool = False) -> int:
    """Region grown from global minimum with 4-connectivity and ``value < threshold``."""
    h, w = norm.shape
    gy, gx = divmod(int(np.nanargmin(norm)), w)
    stack = deque([(gy, gx)])
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    thr = float(threshold)
    while stack:
        i, j = stack.popleft()
        if visited[i, j]:
            continue
        if np.isnan(norm[i, j]):
            continue
        if norm[i, j] >= thr:
            continue
        visited[i, j] = True
        count += 1
        stack.extend(_neighbours(i, j, h, w, toroidal))
    return count


def _local_minima_mask(norm: np.ndarray, toroidal: bool = False) -> np.ndarray:
    h, w = norm.shape
    m = np.ones((h, w), dtype=bool)
    for i in range(h):
        for j in range(w):
            v = norm[i, j]
            if np.isnan(v):
                m[i, j] = False
                continue
            for ni, nj in _neighbours(i, j, h, w, toroidal):
                if norm[ni, nj] < v:
                    m[i, j] = False
                    break
    return m


class UStarFloodClustering:
    """
    U*F clustering (Moutarde & Ultsch, WSOM 2005).

    Automatic threshold: scan normalized U*, build ``region_size(t)`` from the global
    minimum (Moutarde & Ultsch §2.4), take discrete gradients. The bin with the largest
    gradient is where overflow begins; by default we anchor at the **lower** edge of
    that bin (just **before** the sharpest rise — stricter, matches the paper's intent).
    This can yield empty segmentation on noisy inputs where no cells satisfy
    ``norm < threshold``; use ``threshold_anchor="upper"`` (just **after** the rise) for
    more permissive valley filling, or override with ``threshold=`` on :meth:`fit`.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 0,
        n_threshold_steps: int = 100,
        use_ustar: bool = True,
        scalemax: float = 3.0,
        p_median_size: int = 3,
        threshold_anchor: str = "lower",
        toroidal: bool = False,
    ) -> None:
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.n_threshold_steps = int(n_threshold_steps)
        self.use_ustar = bool(use_ustar)
        self.scalemax = float(scalemax)
        self.p_median_size = int(p_median_size)
        ta = str(threshold_anchor).lower()
        if ta not in ("upper", "lower"):
            raise ValueError('threshold_anchor must be "upper" or "lower"')
        self.threshold_anchor = ta
        self.toroidal = bool(toroidal)

        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.ustar_: np.ndarray | None = None
        self.threshold_: float | None = None
        self._unorm_: np.ndarray | None = None

    def fit(
        self,
        u_matrix: np.ndarray,
        p_matrix: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> UStarFloodClustering:
        u_matrix = np.asarray(u_matrix, dtype=np.float64)
        p_matrix = np.asarray(p_matrix, dtype=np.float64)

        if self.use_ustar:
            raw = compute_ustar(
                u_matrix,
                p_matrix,
                scalemax=self.scalemax,
                median_filter_size=self.p_median_size,
            )
        else:
            raw = u_matrix.copy()

        self.ustar_ = raw
        unorm = _normalize_unit_interval(raw)
        self._unorm_ = unorm

        if threshold is not None:
            t_star = float(threshold)
            if not (0.0 <= t_star <= 1.0):
                raise ValueError("threshold must be between 0 and 1 (normalized U*)")
        else:
            thresholds = np.linspace(0.0, 1.0, self.n_threshold_steps)
            sizes = [_flood_region_size(unorm, float(t), self.toroidal) for t in thresholds]
            grad = np.diff(np.asarray(sizes, dtype=np.float64))
            if grad.size == 0 or np.nanmax(grad) <= 0:
                t_star = 0.5
            else:
                t_star_idx = int(np.argmax(grad))
                if self.threshold_anchor == "lower":
                    t_star = float(thresholds[t_star_idx])
                else:
                    t_star = float(thresholds[min(t_star_idx + 1, len(thresholds) - 1)])

        self.threshold_ = t_star
        labels, n_clu = self._label_nodes(unorm, t_star, p_matrix)
        self.labels_ = labels
        self.n_clusters_ = n_clu
        return self

    def _label_nodes(
        self,
        unorm: np.ndarray,
        threshold: float,
        p_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        h, w = unorm.shape
        label_grid = np.full((h, w), -1, dtype=np.int32)
        min_mask = _local_minima_mask(unorm, self.toroidal)
        thr = float(threshold)

        candidates = [(unorm[i, j], i, j) for i, j in zip(*np.where(min_mask))]
        candidates.sort(key=lambda t: t[0])

        cluster_id = 0
        for _v, i, j in candidates:
            if label_grid[i, j] >= 0:
                continue
            _flood_fill_assign(unorm, thr, i, j, label_grid, cluster_id, self.toroidal)
            cluster_id += 1

        # remove small parties
        # A cluster is pruned only when BOTH conditions hold:
        #   node count < min_cluster_size  AND  catchment count < min_samples
        # This preserves edge clusters that span few SOM nodes but hold many data points.
        if cluster_id > 0 and (self.min_cluster_size > 1 or self.min_samples > 0):
            for cid in range(cluster_id):
                mask = label_grid == cid
                node_count = int(np.count_nonzero(mask))
                too_few_nodes = node_count < self.min_cluster_size
                if self.min_samples > 0 and p_matrix is not None:
                    catchment_count = int(np.round(p_matrix[mask].sum()))
                    too_few_catchments = catchment_count < self.min_samples
                    if too_few_nodes and too_few_catchments:
                        label_grid[mask] = -1
                else:
                    if too_few_nodes:
                        label_grid[mask] = -1

        # renumber contiguous 0..K-1
        uniq = np.unique(label_grid[label_grid >= 0])
        new_labels = np.full_like(label_grid, -1)
        for new_id, uid in enumerate(uniq):
            new_labels[label_grid == uid] = new_id

        n_clusters = int(len(uniq))
        return new_labels, n_clusters

    def predict(self, bmu_indices: np.ndarray) -> np.ndarray:
        if self.labels_ is None:
            raise RuntimeError("Call fit before predict.")
        bmu_indices = np.asarray(bmu_indices, dtype=np.int64)
        if bmu_indices.ndim != 2 or bmu_indices.shape[1] != 2:
            raise ValueError("bmu_indices must have shape (n, 2) with (row, col).")

        out = np.empty(bmu_indices.shape[0], dtype=np.int64)
        h, w = self.labels_.shape
        for k in range(bmu_indices.shape[0]):
            r, c = int(bmu_indices[k, 0]), int(bmu_indices[k, 1])
            if 0 <= r < h and 0 <= c < w:
                out[k] = self.labels_[r, c]
            else:
                out[k] = -1
        return out


def _flood_fill_assign(
    norm: np.ndarray,
    threshold: float,
    start_i: int,
    start_j: int,
    label_grid: np.ndarray,
    cluster_id: int,
    toroidal: bool = False,
) -> None:
    h, w = norm.shape
    dq = deque([(start_i, start_j)])
    visited = np.zeros((h, w), dtype=bool)
    thr = float(threshold)
    while dq:
        i, j = dq.popleft()
        if visited[i, j]:
            continue
        visited[i, j] = True
        if label_grid[i, j] >= 0:
            continue
        if np.isnan(norm[i, j]):
            continue
        if norm[i, j] >= thr:
            continue
        label_grid[i, j] = cluster_id
        dq.extend(_neighbours(i, j, h, w, toroidal))
