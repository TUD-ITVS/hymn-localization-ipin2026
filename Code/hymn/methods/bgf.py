"""
Recursive Bayesian Grid Filter with constant-velocity motion model.

Ports the structure of the aircraft-cabin `GridBasedPositioning` filter, but
fixes the likelihood model to a Gaussian (no pluggable measurement models)
and uses auto-computed grid bounds from the caller.

The filter carries a persistent posterior `self.p` across calls. `reset()`
must be invoked at every reference-point transition to clear the state.
"""
import numpy as np


class BayesianGrid:
    def __init__(
        self,
        bounds,
        grid_res=0.1,
        enable_motion=True,
        velocity=0.1,
        velocity_std=0.2,
        dt_min=0.2,
        confidence_radius=2.0,
    ):
        xmin, xmax, ymin, ymax = bounds
        self.grid_res = grid_res
        self.enable_motion = enable_motion
        self.velocity = velocity
        self.velocity_std = velocity_std
        self.dt_min = dt_min
        self.confidence_radius = confidence_radius

        xs = np.arange(xmin, xmax + grid_res / 2.0, grid_res)
        ys = np.arange(ymin, ymax + grid_res / 2.0, grid_res)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        self.grid_points = np.column_stack((XX.ravel(), YY.ravel()))
        self.n_cells = self.grid_points.shape[0]

        self.p_uniform = np.ones(self.n_cells) / self.n_cells
        self.p = self.p_uniform.copy()
        self.t_current = 0.0
        self.t_previous = 0.0

    def reset(self):
        self.p = self.p_uniform.copy()
        self.t_current = 0.0
        self.t_previous = 0.0

    def _motion_update(self):
        if np.allclose(self.p, self.p_uniform):
            return

        n_probable = max(1, int(np.ceil(0.01 * self.n_cells)))
        idx_probable = np.argpartition(-self.p, n_probable - 1)[:n_probable]

        x_diff = self.grid_points[:, [0]] - self.grid_points[idx_probable, 0]
        y_diff = self.grid_points[:, [1]] - self.grid_points[idx_probable, 1]
        distances = np.sqrt(x_diff ** 2 + y_diff ** 2)

        dt = float(max(self.t_current - self.t_previous, self.dt_min))
        sigma_mot = self.velocity_std * dt
        residuals = distances - self.velocity * dt
        idx_valid = np.abs(residuals) < 3.0 * sigma_mot

        p_trans_valid = np.exp(
            -residuals[idx_valid] ** 2 / (2.0 * sigma_mot ** 2)
        ) / np.sqrt(2.0 * np.pi * sigma_mot ** 2)
        p_transition = np.zeros((self.n_cells, n_probable))
        p_transition[idx_valid] = p_trans_valid

        self.p = np.sum(self.p[idx_probable] * p_transition, axis=1)
        self.p[self.p < 1e-5] = 1e-5

    def _measurement_update(self, anchor_xy, ranges, std):
        n = anchor_xy.shape[0]
        dx = anchor_xy[:, 0].reshape(n, 1) - self.grid_points[:, 0]
        dy = anchor_xy[:, 1].reshape(n, 1) - self.grid_points[:, 1]
        expected = np.sqrt(dx ** 2 + dy ** 2)  # (n, n_cells)
        res = ranges[:, None] - expected

        sig = np.asarray(std, dtype=float).reshape(n, 1)
        sig = np.maximum(sig, 1e-12)
        log_lik = np.sum(-0.5 * (res / sig) ** 2, axis=0)  # (n_cells,)

        log_lik -= log_lik.max()
        lik = np.exp(log_lik)

        self.p = self.p * lik
        s = self.p.sum()
        if s > 0.0 and np.isfinite(s):
            self.p = self.p / s
        else:
            self.p = self.p_uniform.copy()

    def _extract_position(self):
        idx_map = int(np.argmax(self.p))
        pos_map = self.grid_points[idx_map]

        d = np.linalg.norm(self.grid_points - pos_map, axis=1)
        idx_conf = d < self.confidence_radius
        w = self.p[idx_conf]
        pos = self.grid_points[idx_conf]
        wsum = w.sum()
        if wsum > 1e-300:
            return (
                float(np.sum(w * pos[:, 0]) / wsum),
                float(np.sum(w * pos[:, 1]) / wsum),
            )
        return float(pos_map[0]), float(pos_map[1])

    def step(self, anchor_xy, ranges, std, timestamp):
        self.t_current = float(timestamp)
        if self.enable_motion:
            self._motion_update()
        self._measurement_update(anchor_xy, ranges, std)
        x, y = self._extract_position()
        self.t_previous = self.t_current
        return x, y


def solve_bgf(grid, anchor_xy, ranges, std, timestamp, min_anchors=3):
    """Thin wrapper matching the ILS/RLS return schema."""
    n = anchor_xy.shape[0]
    if n < min_anchors:
        return {"x": np.nan, "y": np.nan, "converged": False, "n_iter": 0, "residual_norm": np.nan}
    x, y = grid.step(anchor_xy, ranges, std, timestamp)
    return {"x": x, "y": y, "converged": True, "n_iter": 0, "residual_norm": np.nan}
