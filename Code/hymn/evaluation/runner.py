"""
Runs every (technology x classical method) combination over HYMN measurements
and writes a long-format results_long.csv. ResNet variants are merged in by
resnet_ingest after classical methods have finished.
"""
import os
import time
import numpy as np
import pandas as pd

from hymn.config import EVAL, OUT_DIR, RESULT_SCHEMA
from hymn.evaluation import data_interface as di
from hymn.evaluation import resnet_ingest
from hymn.methods.bgf import BayesianGrid, solve_bgf
from hymn.methods.ils import solve_ils
from hymn.methods.rls import solve_rls


def _row(tech, method, meas, est):
    x, y = est["x"], est["y"]
    if np.isfinite(x) and np.isfinite(y):
        err = float(np.hypot(x - meas["ref_xy"][0], y - meas["ref_xy"][1]))
    else:
        err = np.nan
    return {
        "technology": tech,
        "method": method,
        "point_id": meas["point_id"],
        "ts": meas["ts"],
        "ref_x": float(meas["ref_xy"][0]),
        "ref_y": float(meas["ref_xy"][1]),
        "x_est": x,
        "y_est": y,
        "error": err,
        "n_anchors": int(meas["n_anchors"]),
        "converged": bool(est.get("converged", True)),
        "n_iter": int(est.get("n_iter", 0)),
    }


def _build_grid(cfg):
    bgf_cfg = cfg["bgf"]
    bounds = di.get_grid_bounds(pad=bgf_cfg["pad_m"])
    return BayesianGrid(
        bounds,
        grid_res=bgf_cfg["grid_res"],
        enable_motion=bgf_cfg["enable_motion"],
        velocity=bgf_cfg["velocity"],
        velocity_std=bgf_cfg["velocity_std"],
        dt_min=bgf_cfg["dt_min"],
        confidence_radius=bgf_cfg["confidence_radius"],
    )


def run_classical(cfg=EVAL, techs=None, methods=None, verbose=True):
    techs = techs or cfg["technologies"]
    methods = methods or cfg["methods"]
    grid = _build_grid(cfg) if "BGF" in methods else None

    rows = []
    for tech in techs:
        t0 = time.time()
        df = di.load_tech(tech)
        df = df.sort_values(["point_id", "ts"], kind="stable").reset_index(drop=True)
        if verbose:
            print(f"[runner] {tech}: {len(df)} rows loaded")
        if grid is not None:
            grid.reset()
        prev_point_id = None
        for meas in di.iter_measurements(df, tech):
            if grid is not None and meas["point_id"] != prev_point_id:
                grid.reset()
                prev_point_id = meas["point_id"]
            for method in methods:
                if method == "ILS":
                    est = solve_ils(
                        meas["anchor_xy"], meas["ranges"],
                        max_iter=cfg["ils"]["max_iter"], tol=cfg["ils"]["tol"],
                        min_anchors=cfg["ils"]["min_anchors"],
                    )
                elif method == "RLS":
                    est = solve_rls(
                        meas["anchor_xy"], meas["ranges"],
                        delta=cfg["rls"]["huber_delta"],
                        min_anchors=cfg["rls"]["min_anchors"],
                        max_nfev_per_anchor=cfg["rls"]["max_nfev_per_anchor"],
                    )
                elif method == "BGF":
                    est = solve_bgf(
                        grid, meas["anchor_xy"], meas["ranges"], meas["std"],
                        timestamp=meas["ts"],
                        min_anchors=cfg["ils"]["min_anchors"],
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                rows.append(_row(tech, method, meas, est))
        if verbose:
            print(f"[runner] {tech}: classical methods done in {time.time()-t0:.1f}s")
    return pd.DataFrame(rows, columns=RESULT_SCHEMA)


def run_all(cfg=EVAL, techs=None, methods=None, include_resnet=True, verbose=True):
    di.ensure_dirs()
    long_df = run_classical(cfg=cfg, techs=techs, methods=methods, verbose=verbose)

    if include_resnet:
        resnet_df = resnet_ingest.ingest_all(cfg, verbose=verbose)
        if resnet_df is not None and len(resnet_df) > 0:
            long_df = pd.concat([long_df, resnet_df], ignore_index=True)

    out_path = os.path.join(OUT_DIR, "results_long.csv")
    long_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[runner] wrote {len(long_df)} rows -> {out_path}")
    return long_df
