"""Single source of truth for paths, ranging sigmas, training hyperparameters,
and evaluation parameters. All paths are absolute and cwd-independent.

Merged from the legacy ``Code/config.py`` (RESNET training settings) and
``Code/evaluation/config.py`` (EVAL_CONFIG + RESULT_SCHEMA).
"""
import os

# --- Paths (all absolute, anchored at this file's location) ---
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Code/hymn
CODE_DIR = os.path.dirname(PACKAGE_DIR)                       # .../Code
DATA_DIR = os.path.join(CODE_DIR, "data")
OUTPUTS_DIR = os.path.join(CODE_DIR, "outputs")
OUT_DIR = os.path.join(OUTPUTS_DIR, "evaluation")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TRAINED_DIR = os.path.join(OUTPUTS_DIR, "trained")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
PROCESSED_DATA_DIR = os.path.join(OUTPUTS_DIR, "processed_data")
TRAINING_SUMMARIES_DIR = os.path.join(OUTPUTS_DIR, "training_summaries")

# --- Per-tech ranging sigmas (single source of truth, empirical) ---
# Used by both the residual-map calculator (ResNet pipeline) and the BGF solver.
RANGE_STD = {"ble": 4.76, "uwb": 0.27, "wifi": 1.46}

# --- ResNet training ---
RESNET = {
    "random_seed": 42,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.2,
    "epochs": 50,
    "test_split": 0.3,
    "trained_models_path": TRAINED_DIR,
}

# --- Evaluation pipeline ---
EVAL = {
    "technologies": ["fused"],
    "methods": ["ILS", "RLS", "BGF"],
    # Per-variant ResNet specs. ``model`` is optional — when None, the eval
    # pipeline auto-discovers the checkpoint by reading ``model_path`` from the
    # most recent matching training-summary CSV under ``outputs/training_summaries/``.
    # Set ``model`` to a specific .h5 path only to pin a particular checkpoint.
    "resnet_variants": {
        "ResNet-SpatialHoldout": {
            "model": None,
            "test_pickle": os.path.join(PROCESSED_DATA_DIR, "spatial_holdout", "fused_test.pkl"),
            "pred_out": os.path.join(PREDICTIONS_DIR, "fused_spatial_holdout.csv"),
        },
        "ResNet-RandomSplit": {
            "model": None,
            "test_pickle": os.path.join(PROCESSED_DATA_DIR, "random_split", "fused_test.pkl"),
            "pred_out": os.path.join(PREDICTIONS_DIR, "fused_random_split.csv"),
        },
    },
    "range_std": RANGE_STD,
    "ils": {"max_iter": 7, "tol": 1e-6, "min_anchors": 3},
    "rls": {"huber_delta": 1.0, "max_nfev_per_anchor": 50, "min_anchors": 3},
    "bgf": {
        "grid_res": 0.5, "pad_m": 2.0,
        "confidence_radius": 2.0,
        "enable_motion": True,
        "velocity": 0.1, "velocity_std": 0.2, "dt_min": 0.2,
    },
    "plot": {
        "dpi": 300,
        "font_family": "serif",
        "font_size": 9,
        "grid_alpha": 0.3,
        "method_colors": {
            "ILS": "#0072B2",
            "RLS": "#009E73",
            "BGF": "#D55E00",
            "ResNet-SpatialHoldout": "#CC79A7",
            "ResNet-RandomSplit": "#56B4E9",
        },
        "method_linestyles": {
            "ILS": "-",
            "RLS": "--",
            "BGF": "-.",
            "ResNet-SpatialHoldout": ":",
            "ResNet-RandomSplit": (0, (3, 1, 1, 1)),
        },
        "ranging_ecdf_xmax": 20.0,
        "spatial_grid_resolution": (120, 140),
        "spatial_vmax": None,
        "spatial_vmax_percentile": 98.0,
        "spatial_contour_levels": [0.5, 1.0, 2.0, 3.0, 5.0],
        "spatial_hull_pad": 0.5,
    },
}

RESULT_SCHEMA = [
    "technology", "method", "point_id", "ts",
    "ref_x", "ref_y", "x_est", "y_est", "error",
    "n_anchors", "converged", "n_iter",
]
