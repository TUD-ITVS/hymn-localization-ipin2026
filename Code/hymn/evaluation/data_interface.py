"""Data interface layer. Wraps hymn.io / hymn.preprocessing and exposes a
per-row iterator that yields the arrays each positioning method consumes.
Keeps method implementations free of DataFrame logic.

Anchor-name normalization and surveyed-position remap happen at load time
in hymn.io, so this layer no longer needs its own remap.
"""
import os

import numpy as np
import pandas as pd

from hymn.config import EVAL, OUT_DIR, FIG_DIR
from hymn.io import get_anchors_positions, load_hymn_data
from hymn.preprocessing import compute_bounds


def _anchor_xy_dict(tech):
    """Return {anchor_id: (x, y)} for a single tech, dropping Z."""
    raw = get_anchors_positions(tech)
    return {aid: (float(v[0]), float(v[1])) for aid, v in raw.items()}


def get_anchor_xy(tech):
    """
    Return (ids, xy, std) for a tech.
      - ids: list[str], canonical anchor ids.
      - xy:  np.ndarray (N, 2).
      - std: np.ndarray (N,) with the per-anchor range std.
    For 'fused', concatenates BLE + UWB + WiFi anchors in that order.
    """
    stds = EVAL["range_std"]
    if tech == "fused":
        parts = []
        for t in ("ble", "uwb", "wifi"):
            ids_t, xy_t, std_t = get_anchor_xy(t)
            parts.append((ids_t, xy_t, std_t))
        ids = sum([p[0] for p in parts], [])
        xy = np.vstack([p[1] for p in parts])
        std = np.concatenate([p[2] for p in parts])
        return ids, xy, std
    d = _anchor_xy_dict(tech)
    ids = list(d.keys())
    xy = np.array([d[k] for k in ids], dtype=float)
    std = np.full(len(ids), stds[tech], dtype=float)
    return ids, xy, std


def get_grid_bounds(pad=2.0):
    """
    Return (xmin, xmax, ymin, ymax) around all anchors/reference points.
    Reuses preprocessing.compute_bounds (which returns square min/max scalars)
    then pads by `pad` meters on each side.
    """
    anchors_ble = list(_anchor_xy_dict("ble").values())
    anchors_uwb = list(_anchor_xy_dict("uwb").values())
    anchors_wifi = list(_anchor_xy_dict("wifi").values())
    min_all, max_all = compute_bounds(anchors_ble, anchors_uwb, anchors_wifi)
    return (min_all - pad, max_all + pad, min_all - pad, max_all + pad)


def _load_fused_df():
    """
    Build a row-aligned fused dataframe where each row pools BLE+UWB+WiFi
    anchor ids and ranges. Uses match_dataframe_sizes() as in resnet.data_split.
    Ground truth = BLE's (ref_x, ref_y), matching build_Xy().
    """
    from hymn.methods.resnet.data_split import match_dataframe_sizes
    df_ble, df_uwb, df_wifi = load_hymn_data()
    ble_m, uwb_m, wifi_m = match_dataframe_sizes(df_ble, df_uwb, df_wifi, "point_id")
    n = min(len(ble_m), len(uwb_m), len(wifi_m))
    ble_m = ble_m.iloc[:n].reset_index(drop=True)
    uwb_m = uwb_m.iloc[:n].reset_index(drop=True)
    wifi_m = wifi_m.iloc[:n].reset_index(drop=True)

    rows = []
    for i in range(n):
        b, u, w = ble_m.iloc[i], uwb_m.iloc[i], wifi_m.iloc[i]
        ids = list(b["anchor_ids"]) + list(u["anchor_ids"]) + list(w["anchor_ids"])
        rng = list(b["ranges"]) + list(u["ranges"]) + list(w["ranges"])
        rows.append({
            "point_id": b["point_id"],
            "ts": b["ts"],
            "anchor_ids": ids,
            "ranges": rng,
            "ref_x": float(b["ref_x"]),
            "ref_y": float(b["ref_y"]),
        })
    return pd.DataFrame(rows)


def load_tech(tech):
    """
    Return a dataframe with columns
        point_id, ts, anchor_ids, ranges, ref_x, ref_y
    for the given tech. anchor_ids are already canonical (BLE_0N, WIFI_0N
    matching the surveyed positions table) — normalization and remap happen
    in hymn.io.load_hymn_data.
    """
    if tech == "fused":
        return _load_fused_df()
    df_ble, df_uwb, df_wifi = load_hymn_data()
    if tech == "ble":
        df = df_ble
    elif tech == "uwb":
        df = df_uwb
    elif tech == "wifi":
        df = df_wifi
    else:
        raise ValueError(f"Unsupported tech: {tech}")
    keep = ["point_id", "ts", "anchor_ids", "ranges", "ref_x", "ref_y"]
    return df[keep].reset_index(drop=True)


def iter_measurements(df, tech):
    """
    Yield one dict per row with the inputs a positioning solver needs:
        point_id, ts, ref_xy (2,), anchor_xy (K, 2), ranges (K,), std (K,), n_anchors
    NaN ranges are filtered out and anchors missing from the reference table
    are skipped. If fewer than 3 valid anchors remain, n_anchors<3 is returned
    and the caller can decide to emit NaN estimates.
    """
    ids_ref, xy_ref, std_ref = get_anchor_xy(tech)
    id_to_idx = {aid: i for i, aid in enumerate(ids_ref)}

    for _, row in df.iterrows():
        aids = row["anchor_ids"]
        rngs = row["ranges"]
        xy, rs, ss = [], [], []
        for aid, r in zip(aids, rngs):
            if aid not in id_to_idx:
                continue
            if r is None or (isinstance(r, float) and np.isnan(r)):
                continue
            j = id_to_idx[aid]
            xy.append(xy_ref[j])
            rs.append(float(r))
            ss.append(float(std_ref[j]))
        xy = np.array(xy, dtype=float).reshape(-1, 2)
        rs = np.array(rs, dtype=float)
        ss = np.array(ss, dtype=float)
        yield {
            "point_id": row["point_id"],
            "ts": row["ts"],
            "ref_xy": np.array([row["ref_x"], row["ref_y"]], dtype=float),
            "anchor_xy": xy,
            "ranges": rs,
            "std": ss,
            "n_anchors": len(rs),
        }


def compute_ranging_long(techs=("ble", "uwb", "wifi")):
    """
    Build a long-format dataframe of per-anchor ranging residuals for single-
    technology data. For each valid (measurement, anchor) pair, the true range
    is the Euclidean distance from the reference point to the anchor; the
    residual is measured_range minus true_range.

    Columns: technology, point_id, ts, anchor_id, ref_x, ref_y,
             true_range, measured_range, residual, abs_residual.
    """
    rows = []
    for tech in techs:
        df = load_tech(tech)
        ids_ref, xy_ref, _ = get_anchor_xy(tech)
        id_to_idx = {aid: i for i, aid in enumerate(ids_ref)}
        for _, row in df.iterrows():
            ref_xy = np.array([row["ref_x"], row["ref_y"]], dtype=float)
            for aid, r in zip(row["anchor_ids"], row["ranges"]):
                if aid not in id_to_idx:
                    continue
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                a_xy = xy_ref[id_to_idx[aid]]
                true_r = float(np.hypot(a_xy[0] - ref_xy[0], a_xy[1] - ref_xy[1]))
                res = float(r) - true_r
                rows.append({
                    "technology": tech,
                    "point_id": row["point_id"],
                    "ts": row["ts"],
                    "anchor_id": aid,
                    "ref_x": float(row["ref_x"]),
                    "ref_y": float(row["ref_y"]),
                    "true_range": true_r,
                    "measured_range": float(r),
                    "residual": res,
                    "abs_residual": abs(res),
                })
    return pd.DataFrame(rows)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
