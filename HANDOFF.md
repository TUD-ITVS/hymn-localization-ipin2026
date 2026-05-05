# Reproducibility-test handoff

Thanks for running this on a stronger machine. The IPIN 2026 paper companion code is in this repo and we'd like to verify the full ResNet+classical pipeline produces sensible numbers before we tag and archive a release.

## What you'd do

1. **Clone and set up the environment**
   ```bash
   git clone https://github.com/TUD-ITVS/hymn-localization-ipin2026.git
   cd hymn-localization-ipin2026
   python -m venv .venv-tf
   # Linux/macOS:
   source .venv-tf/bin/activate
   # Windows PowerShell:
   . .venv-tf\Scripts\Activate.ps1
   pip install -r Code/requirements.txt
   ```
   Python 3.10–3.13 should all work (TensorFlow 2.21 has wheels for each). A CUDA-enabled GPU is auto-detected by TF if present.

2. **Train both ResNet protocols**
   ```bash
   cd Code
   python scripts/train_resnet.py
   ```
   The script loops both `SpatialHoldout` and `RandomSplit` protocols. Total runtime on CPU is roughly 3–4 hours; with a recent GPU it's well under an hour. Checkpoints land in `Code/outputs/trained/`, and a per-protocol summary CSV in `Code/outputs/training_summaries/`.

3. **Run the full evaluation**
   ```bash
   python scripts/run_evaluation.py
   ```
   This runs the classical methods (ILS, RLS, BGF), ingests the ResNet predictions, writes `Code/outputs/evaluation/stats_table.{csv,tex}` and the manuscript figures into `Code/outputs/evaluation/figures/`. Should take 5–10 minutes.

   The evaluation pipeline auto-discovers the latest checkpoint per protocol from the training-summary CSVs — no need to edit any config.

## What to send back

Zip the entire `Code/outputs/evaluation/` directory and send it back. That's the minimal set:

- `stats_table.csv` and `stats_table.tex` — the post-refactor positioning numbers
- `ranging_stats.csv` and `ranging_stats.tex` — per-technology ranging residuals
- `figures/*.pdf` — the four manuscript figures (`ecdf_ranging`, `ecdf_fused`, `spatial_heatmap_fused`, plus per-method spatial heatmaps)

If anything fails, the full stdout of the failing command plus a description of the host (OS, Python version, GPU/CPU) is enough for us to diagnose remotely.

## What we'll do with the result

Diff your `stats_table.csv` against the inline numbers in the paper, update the manuscript if anything moved meaningfully, then tag the release on GitHub. A snapshot is then archived to Zenodo with a DOI that the paper cites.
