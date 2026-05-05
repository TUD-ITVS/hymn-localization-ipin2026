"""Robust Least Squares via scipy.optimize.least_squares with Huber loss.

Aligned with the aircraft-cabin reference (trust-region solver, Huber kernel
with f_scale = delta, centroid initialisation). Retains a sanity-bounds
rejection on the final estimate so divergent fixes on NLOS-corrupted BLE/WiFi
are returned as NaN rather than being counted as valid outliers.
"""
import numpy as np
from scipy.optimize import least_squares

from hymn.methods.ils import _sanity_bounds


def _centroid_init(anchor_xy):
    return anchor_xy.mean(axis=0)


def _residuals(p, anchor_xy, ranges):
    return ranges - np.linalg.norm(p - anchor_xy, axis=1)


def solve_rls(anchor_xy, ranges, delta=1.0, min_anchors=3, max_nfev_per_anchor=50):
    n = anchor_xy.shape[0]
    if n < min_anchors:
        return {"x": np.nan, "y": np.nan, "converged": False, "n_iter": 0, "residual_norm": np.nan}

    x0 = _centroid_init(anchor_xy)
    lo, hi = _sanity_bounds(anchor_xy)
    try:
        result = least_squares(
            _residuals, x0=x0,
            args=(anchor_xy, ranges),
            loss="huber", f_scale=float(delta),
            max_nfev=int(max_nfev_per_anchor * n),
        )
    except (ValueError, np.linalg.LinAlgError):
        return {"x": np.nan, "y": np.nan, "converged": False, "n_iter": 0, "residual_norm": np.nan}

    p = np.asarray(result.x, dtype=float)
    if (not np.all(np.isfinite(p))) or np.any(p < lo) or np.any(p > hi):
        return {"x": np.nan, "y": np.nan, "converged": False,
                "n_iter": int(result.nfev),
                "residual_norm": float(np.linalg.norm(result.fun))}
    return {
        "x": float(p[0]), "y": float(p[1]),
        "converged": bool(result.success),
        "n_iter": int(result.nfev),
        "residual_norm": float(np.linalg.norm(result.fun)),
    }
