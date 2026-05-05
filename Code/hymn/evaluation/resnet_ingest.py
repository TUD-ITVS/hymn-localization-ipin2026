"""
Ingest ResNet predictions into the long-format evaluation schema.

If per-point prediction CSVs exist at the expected paths, load them directly.
Otherwise, regenerate them by calling predict_data.model_predict_csv() for each
trained variant. Results are always reported as technology='fused' because the
existing ResNet models are trained on fused BLE+UWB+WiFi channels.
"""
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

from hymn.config import PREDICTIONS_DIR, RESULT_SCHEMA, TRAINING_SUMMARIES_DIR


AGGREGATE_CSV_GLOB = {
    "ResNet-SpatialHoldout": "results_split-spatial-*.csv",
    "ResNet-RandomSplit": "results_split-random-*.csv",
}


def _resolve_model_path(variant, spec):
    """Locate the checkpoint for ``variant``.

    Priority:
    1. ``spec["model"]`` if set and the file exists on disk.
    2. The ``model_path`` cell of the most recent training-summary CSV that
       matches the variant's protocol (sorted lexicographically, last wins —
       the timestamp suffix in the filename ensures latest-by-mtime is also
       latest-by-name).

    Returns ``None`` if neither source resolves to an existing file. Lets the
    pipeline run after a fresh training pass without manual edits to config.
    """
    explicit = spec.get("model")
    if explicit and os.path.exists(explicit):
        return explicit
    pattern = AGGREGATE_CSV_GLOB.get(variant)
    if pattern is None:
        return None
    matches = sorted(glob.glob(os.path.join(TRAINING_SUMMARIES_DIR, pattern)))
    if not matches:
        return None
    try:
        df = pd.read_csv(matches[-1])
        candidate = df.loc[0, "model_path"]
    except Exception:
        return None
    if isinstance(candidate, str) and os.path.exists(candidate):
        return candidate
    return None


def _parse_aggregate_errors(csv_path):
    """
    Extract the raw per-sample error list from an aggregate training-summary CSV
    (the `ecdf_errors` cell). The cell is a Python tuple
        ([json.dumps(ecdf.x.tolist())], [json.dumps(errors.tolist())])
    serialized to string; we want the second list.
    """
    df = pd.read_csv(csv_path)
    cell = df.loc[0, "ecdf_errors"]
    # Extract all JSON lists in the cell; the second one is the raw errors.
    matches = re.findall(r"'(\[[^']*\])'", cell)
    if len(matches) < 2:
        return None
    errors_json = matches[1].replace("Infinity", "1e308").replace("-1e308", "-1e308")
    errors = json.loads(errors_json)
    errors = np.array(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    return errors


def _aggregate_to_long(variant, errors):
    """Wrap a bare error array into the long-format schema, with NaN placeholders
    for fields not recoverable from the aggregate blob."""
    n = len(errors)
    return pd.DataFrame({
        "technology": ["fused"] * n,
        "method": [variant] * n,
        "point_id": [np.nan] * n,
        "ts": [np.nan] * n,
        "ref_x": [np.nan] * n,
        "ref_y": [np.nan] * n,
        "x_est": [np.nan] * n,
        "y_est": [np.nan] * n,
        "error": errors,
        "n_anchors": [0] * n,
        "converged": [True] * n,
        "n_iter": [0] * n,
    })[RESULT_SCHEMA]


def _pred_to_long(df, method):
    out = pd.DataFrame({
        "technology": "fused",
        "method": method,
        "point_id": df["point_id"].astype(str),
        "ts": df.get("timestamp_ble", df.get("ts", pd.Series([np.nan] * len(df)))),
        "ref_x": df["ref_x"].astype(float),
        "ref_y": df["ref_y"].astype(float),
        "x_est": df["x_pred"].astype(float),
        "y_est": df["y_pred"].astype(float),
        "error": df["error(m)"].astype(float),
        "n_anchors": df.filter(like="n_anchors_").sum(axis=1).astype(int) if any(c.startswith("n_anchors_") for c in df.columns) else 0,
        "converged": True,
        "n_iter": 0,
    })
    return out[RESULT_SCHEMA]


def _regenerate(variant, spec, verbose=True):
    """
    Run TF inference once via predict_data.model_predict_csv, then move the
    produced 'both_predictions.csv' to the variant-specific output path.
    """
    from hymn.io import get_anchors_positions
    from hymn.methods.resnet.predict import model_predict_csv

    model_path = _resolve_model_path(variant, spec)
    if model_path is None:
        if verbose:
            print(f"[resnet_ingest] no checkpoint found for {variant}; cannot regenerate")
        return None
    test_pickle = spec["test_pickle"]
    pred_out = spec["pred_out"]

    if verbose:
        print(f"[resnet_ingest] regenerating ResNet predictions for {variant} from {os.path.basename(model_path)}")

    anchors_ble = get_anchors_positions("ble")
    anchors_uwb = get_anchors_positions("uwb")
    anchors_wifi = get_anchors_positions("wifi")

    model_predict_csv(
        model_path=model_path,
        test_data_path=test_pickle,
        anchors_ble=anchors_ble,
        anchors_uwb=anchors_uwb,
        anchors_wifi=anchors_wifi,
        tech="both",
    )

    default_csv = os.path.join(PREDICTIONS_DIR, "both_predictions.csv")
    default_pkl = os.path.join(PREDICTIONS_DIR, "both_predictions.pkl")
    out_csv = pred_out
    out_pkl = out_csv.replace(".csv", ".pkl")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.exists(default_csv):
        shutil.move(default_csv, out_csv)
    if os.path.exists(default_pkl):
        shutil.move(default_pkl, out_pkl)
    return out_csv


def _try_aggregate_fallback(variant, verbose=True):
    """Load errors from the training-summary CSV. Returns long-format df or None."""
    pattern = AGGREGATE_CSV_GLOB.get(variant)
    if pattern is None:
        return None
    matches = sorted(glob.glob(os.path.join(TRAINING_SUMMARIES_DIR, pattern)))
    if not matches:
        return None
    try:
        errors = _parse_aggregate_errors(matches[-1])
    except Exception as e:
        if verbose:
            print(f"[resnet_ingest] aggregate parse failed for {variant}: {e}")
        return None
    if errors is None or len(errors) == 0:
        return None
    if verbose:
        print(f"[resnet_ingest] aggregate fallback: {variant} -> {len(errors)} errors from {os.path.basename(matches[-1])}")
    return _aggregate_to_long(variant, errors)


def ingest_all(cfg, verbose=True, force_regenerate=False):
    """
    Prefer per-point prediction CSVs (regenerated via TF if missing). Fall back
    to parsing the aggregate training-summary CSV (`ecdf_errors` blob) when TF
    is unavailable — ECDF and stats work; spatial plots will skip ResNet rows
    because their ref_x/ref_y are NaN.
    """
    variants = cfg["resnet_variants"]
    frames = []
    for variant, spec in variants.items():
        pred_csv = spec["pred_out"]
        if force_regenerate or not os.path.exists(pred_csv):
            try:
                _regenerate(variant, spec, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[resnet_ingest] WARNING: could not regenerate {variant}: {e}")
        if os.path.exists(pred_csv):
            df = pd.read_csv(pred_csv)
            frames.append(_pred_to_long(df, variant))
            if verbose:
                print(f"[resnet_ingest] loaded {len(df)} per-point rows for {variant}")
            continue
        fallback = _try_aggregate_fallback(variant, verbose=verbose)
        if fallback is not None:
            frames.append(fallback)
        elif verbose:
            print(f"[resnet_ingest] WARNING: no data for {variant}; skipping")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)
