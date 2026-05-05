"""Iterative Least Squares (Gauss-Newton) trilateration.

Aligned with the aircraft-cabin reference (SVD-based step via np.linalg.lstsq,
centroid initialisation). Retains two pragmatic safeguards that the reference
did not need because the cabin measurements were cleaner: per-iteration step
clipping and a sanity-bounds rejection on the final estimate. These keep the
solver from diverging when BLE (sigma ~4.76 m) or WiFi (sigma ~1.46 m) fixes
are NLOS-corrupted.
"""
import numpy as np


def _centroid_init(anchor_xy):
    return anchor_xy.mean(axis=0)


def _sanity_bounds(anchor_xy, slack=5.0):
    lo = anchor_xy.min(axis=0)
    hi = anchor_xy.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    return lo - slack * span, hi + slack * span


def solve_ils(anchor_xy, ranges, init=None, max_iter=7, tol=1e-6,
              min_anchors=3, max_step=10.0):
    """
    Gauss-Newton minimisation of sum_i (r_i - ||p - a_i||)^2.

    Returns dict: x, y, converged, n_iter, residual_norm.
    """
    n = anchor_xy.shape[0]
    if n < min_anchors:
        return {"x": np.nan, "y": np.nan, "converged": False, "n_iter": 0, "residual_norm": np.nan}

    p = _centroid_init(anchor_xy) if init is None else np.asarray(init, dtype=float).copy()
    lo, hi = _sanity_bounds(anchor_xy)
    converged = False
    it = 0
    last_res_norm = np.nan
    for it in range(1, max_iter + 1):
        diff = p - anchor_xy
        d = np.linalg.norm(diff, axis=1)
        d_safe = np.maximum(d, 1e-10)
        res = ranges - d
        J = diff / d_safe[:, None]

        try:
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            return {"x": np.nan, "y": np.nan, "converged": False, "n_iter": it, "residual_norm": np.nan}

        step_norm = np.linalg.norm(delta)
        if step_norm > max_step:
            delta = delta * (max_step / step_norm)

        p = p - delta
        last_res_norm = float(np.linalg.norm(res))
        if np.linalg.norm(delta) < tol:
            converged = True
            break

    if not np.all(np.isfinite(p)) or np.any(p < lo) or np.any(p > hi):
        return {"x": np.nan, "y": np.nan, "converged": False, "n_iter": it, "residual_norm": last_res_norm}
    return {"x": float(p[0]), "y": float(p[1]), "converged": converged, "n_iter": it, "residual_norm": last_res_norm}
