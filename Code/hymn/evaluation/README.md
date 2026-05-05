# HYMN Positioning Evaluation

Evaluation pipeline for the IPIN 2026 paper. Runs ILS, RLS (Huber IRLS), and
Bayesian Grid on the HYMN dataset, ingests ResNet predictions from two
training splits, and produces ECDF + spatial error plots and a statistics
table. A separate pass computes per-anchor ranging residuals for BLE, UWB,
and WiFi and emits a per-technology ECDF and stats table.

This subpackage is `hymn.evaluation`; the entry point is
`Code/scripts/run_evaluation.py`. See the top-level `README.md` for the
overall code map.

## Scope

**Positioning:** 5 methods on the fused BLE+UWB+WiFi measurements (ILS, RLS,
BGF, ResNet-SpatialHoldout, ResNet-RandomSplit). Single-technology positioning
is out of scope — the need for multi-technology fusion is argued in prose
rather than driven by per-tech result tables. The full `results_long.csv`
still contains single-tech rows as an audit trail, but stats and figures
filter to fused only via `EVAL["technologies"]`.

**Ranging:** per-anchor absolute residual for BLE, UWB, WiFi. GNSS is skipped
(no LOS range observable indoors). Feeds the Ranging Performance subsection
of the paper; deeper ranging physics is out of scope (see Michler2025).

## Run

From `Code/`:

```bash
python scripts/run_evaluation.py                 # full pipeline (~12 min for classical methods)
python scripts/run_evaluation.py --quick         # UWB+ILS only (~<30s, no TF, no ResNet)
python scripts/run_evaluation.py --no-resnet
python scripts/run_evaluation.py --no-plots
python scripts/run_evaluation.py --skip-classical  # reuse results_long.csv, re-ingest ResNet + re-plot
```

Both `python scripts/run_evaluation.py` and `python -m scripts.run_evaluation`
work. All paths in `hymn.config` are absolute (anchored at `__file__`), so
the scripts run from any cwd.

## Methods

| Method | Implementation | Notes |
|---|---|---|
| `ILS` | `hymn/methods/ils.py` | Gauss-Newton, initialized at anchor centroid. |
| `RLS` | `hymn/methods/rls.py` | IRLS with Huber loss (δ=1.0 m), centroid init, sanity-bounds rejection. |
| `BGF` | `hymn/methods/bgf.py` | Gaussian likelihood grid (res 0.5 m), constant-velocity motion model, weighted-centroid extractor. |
| `ResNet-SpatialHoldout` | `hymn/evaluation/resnet_ingest.py` | Inference from a checkpoint trained on the leave-points-out split (extrapolation regime). Path is configured via `EVAL["resnet_variants"]`. |
| `ResNet-RandomSplit` | `hymn/evaluation/resnet_ingest.py` | Inference from a checkpoint trained on the random epoch split (interpolation regime, optimistic baseline). |

Per-anchor range std is the single source of truth `hymn.config.RANGE_STD`
= `{"ble": 4.76, "uwb": 0.27, "wifi": 1.46}` (empirical). These broadcast to
the fused case as a per-anchor vector so each measurement is weighted by its
own technology's noise model. The same `RANGE_STD` is used by `hymn.grid_calc`
for the ResNet residual maps.

## Outputs

All artifacts land in `Code/outputs/evaluation/`:

**Positioning**
- `results_long.csv` — single source of truth. One row per (sample, method).
  Columns: `technology, method, point_id, ts, ref_x, ref_y, x_est, y_est, error, n_anchors, converged, n_iter`.
- `stats_table.csv` and `stats_table.tex` — Mean / Median / Std / P75 / P95
  per (tech, method). Best-per-tech values bolded in the LaTeX version.

**Ranging** (produced by `data_interface.compute_ranging_long()` +
`stats.write_ranging_stats()` + `plotting.figures.plot_ecdf_ranging()`)
- `ranging_long.csv` — one row per (measurement, anchor). Columns:
  `technology, point_id, ts, anchor_id, ref_x, ref_y, true_range,
  measured_range, residual, abs_residual`.
- `ranging_stats.csv` and `ranging_stats.tex` — n / Mean / Median / Std /
  P75 / P95 of `abs_residual` per technology.

**Figures** (`figures/`)
- `ecdf_fused.pdf` — positioning ECDF on fused input, all 5 methods.
- `ecdf_ranging.pdf` — ranging ECDF per technology (BLE / UWB / WiFi),
  log-scaled x.
- `spatial_error_<tech>_<method>.pdf` — per-point mean error on the 2D
  layout with anchors overlaid.
- `boxplot_per_method.pdf` — per-method positioning error, fused.
- `anchors_overview.pdf` — anchor and reference-point geometry sanity check.

## Quick verification

1. `python scripts/run_evaluation.py --quick` should finish in under 30 seconds and
   print a summary with UWB-ILS median in the 0.3–0.6 m range. Quick mode
   skips the ranging pass.
2. In `results_long.csv`, for any finite row, `error ≈ hypot(x_est-ref_x, y_est-ref_y)`.
3. The fused-ResNet-SpatialHoldout mean in `stats_table.csv` should agree with
   the `mean` field in the latest `outputs/training_summaries/results_split-spatial-*.csv`
   to within a couple of centimeters (identical samples, identical model).
4. `ranging_stats.csv` should show UWB median ≲ 0.5 m, WiFi median ~6–7 m,
   BLE median ~10 m; row counts in the tens of thousands per technology.
5. `anchors_overview.pdf` shows 5 BLE + 10 UWB + 6 WiFi anchors and the
   reference-point grid on the HYMN measurement plate.

## Design notes

- Long-format CSV is the single source of truth; stats and plots derive from it.
- Anchor-name normalization (`BLE1` → `BLE_01`) and the surveyed-position
  remap (`WIFI_01` → `WIFI_03`, etc.) happen at load time in `hymn.io`, so
  every downstream consumer (this evaluation, the ResNet pipeline) sees
  canonical anchor labels matching `data/reference/csv/anchor_coordinates.csv`.
- `data_interface.iter_measurements` filters NaN ranges and pre-computes the
  per-anchor std vector, so solver functions take only `(anchor_xy, ranges,
  std)` arrays.
- The BGF grid is built once per run from `preprocessing.compute_bounds()`
  plus a 2 m pad, so all methods and all techs share the same spatial frame.
- Fused ground truth uses BLE's `(ref_x, ref_y)`, matching the convention
  used in `hymn.methods.resnet.data_split.build_Xy` for consistency with
  ResNet training.

## Caveats

- ResNet per-point CSVs (`outputs/predictions/fused_*.csv`) are regenerated
  once via `hymn.methods.resnet.predict.model_predict_csv` on first run,
  **if TensorFlow is importable**. Subsequent runs reuse the cached CSVs.
  When TF is unavailable (e.g. a fresh venv without TF installed),
  `resnet_ingest` falls back to parsing the raw error list from the aggregate
  training-summary CSVs (`outputs/training_summaries/results_split-*.csv`).
  In fallback mode, ECDF and stats include ResNet; spatial plots do not
  (ref_x/ref_y are NaN for those rows). To upgrade to full spatial plots,
  run the pipeline once in a TF-equipped environment.
- Spatial plots for the fused case collapse to BLE's reference XY; UWB and
  WiFi per-point plots use their own tech-specific reference positions (as in
  the ResNet training pipeline).
