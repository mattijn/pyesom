"""U*-matrix: combine U-matrix heights with P-matrix density."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def compute_ustar(
    u_matrix: np.ndarray,
    p_matrix: np.ndarray,
    scalemax: float = 3.0,
    median_filter_size: int = 3,
    use_robust_mean: bool = True,
) -> np.ndarray:
    """
    Combine U-heights with P-heights (Ultsch 2003; appendix linear form).

    ``ScaleFactor(n) = (p_max - P(n)) / (p_max - p_ref)``, clipped at
    ``scalemax``, with ``p_ref = max(mean(P), median(P))`` when
    ``use_robust_mean`` is True (otherwise ``mean(P)``).

    ``U*(n) = U(n) * ScaleFactor(n)``.
    """
    u_matrix = np.asarray(u_matrix, dtype=np.float64)
    p_matrix = np.asarray(p_matrix, dtype=np.float64)
    if u_matrix.shape != p_matrix.shape:
        raise ValueError("u_matrix and p_matrix must have the same shape")

    if median_filter_size and median_filter_size > 1:
        p_f = median_filter(p_matrix, size=int(median_filter_size))
    else:
        p_f = p_matrix.copy()

    p_max = float(np.max(p_f))
    if use_robust_mean:
        p_ref = float(max(np.mean(p_f), np.median(p_f)))
    else:
        p_ref = float(np.mean(p_f))

    denom = p_max - p_ref
    if denom <= 1e-15:
        scale = np.ones_like(p_f)
    else:
        scale = (p_max - p_f) / denom

    scale = np.minimum(scale, float(scalemax))
    return u_matrix * scale
