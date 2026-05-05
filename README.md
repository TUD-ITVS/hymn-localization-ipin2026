# HYMN Multi-Technology Indoor Positioning Evaluation

Companion code for the IPIN 2026 paper *From Least Squares to Deep Learning: Benchmarking Indoor Positioning on the HYMN Multi-Technology Dataset*. Implements iterative least squares (ILS), robust least squares (RLS), Bayesian grid filtering (BGF), and two ResNet variants (RandomSplit and SpatialHoldout protocols), and reproduces every number, table, and figure in the manuscript from the bundled CSV inputs.

- **Paper authors**: Paul Schwarzbach and Muhammad Ammad — Institute of Traffic Telematics, TUD Dresden University of Technology
- **License**: MIT (see [`LICENSE`](LICENSE))
- **Source code**: [github.com/TUD-ITVS/hymn-localization-ipin2026](https://github.com/TUD-ITVS/hymn-localization-ipin2026)
- **Code archive**: Zenodo DOI `10.5281/zenodo.XXXXXXX` *(populated after the v1.0-ipin2026 GitHub release)*
- **HYMN dataset**: Zenodo DOI [`10.5281/zenodo.17979434`](https://zenodo.org/doi/10.5281/zenodo.17979434), data descriptor [arXiv:2604.20349](https://arxiv.org/abs/2604.20349)

## Repository layout

```
hymn-localization-ipin2026/
├── Code/
│   ├── data/csv/                  # HYMN measurements (BLE, UWB, WiFi)
│   ├── data/reference/            # Anchor positions, reference-point coordinates
│   ├── hymn/                      # The library — import from any script via `import hymn`
│   │   ├── io.py                  # CSV ingestion + WIFI/BLE anchor-name & position remap
│   │   ├── preprocessing.py       # Bounds + range residuals
│   │   ├── grid_calc.py           # Per-anchor likelihood maps (uses hymn.config.RANGE_STD)
│   │   ├── config.py              # Paths, RANGE_STD, RESNET hyperparams, EVAL parameters
│   │   ├── methods/               # ALL positioning methods in one place
│   │   │   ├── ils.py, rls.py, bgf.py     # Classical (used by evaluation)
│   │   │   └── resnet/                    # ResNet (architectures, training, inference)
│   │   ├── evaluation/            # Classical+learning evaluation pipeline
│   │   │   ├── runner.py          # Dispatch over (technology × method)
│   │   │   ├── stats.py           # stats_table.{csv,tex} (Table I)
│   │   │   ├── data_interface.py  # Per-row solver inputs
│   │   │   └── resnet_ingest.py   # Folds ResNet predictions into long format
│   │   └── plotting/
│   │       ├── diagnostics.py     # Training-time sample plots (was plotting.py)
│   │       └── figures.py         # Publication figures (was evaluation/plotting.py)
│   ├── scripts/                   # Thin entry points
│   │   ├── train_resnet.py        # ResNet training (was main.py)
│   │   └── run_evaluation.py      # Stats + figures (was evaluation/run.py)
│   ├── outputs/                   # All artefacts produced by the pipelines
│   │   ├── trained/               # ResNet .h5 checkpoints (regenerable)
│   │   ├── predictions/           # ResNet inference CSVs
│   │   ├── training_summaries/    # results_split-{random,spatial}-*.csv
│   │   └── evaluation/            # results_long.csv, stats_table.{csv,tex}, figures/*.pdf
│   └── requirements.txt           # Pinned dependencies
├── LICENSE                        # MIT
├── CITATION.cff
└── README.md
```

### Code map (where to look first)

| If you want to… | Look at |
|---|---|
| Add a new positioning method | `Code/hymn/methods/` (drop in a new module alongside `ils.py`/`rls.py`/`bgf.py`) |
| Change a hyperparameter or path | `Code/hymn/config.py` (single source of truth: `RESNET`, `EVAL`, `RANGE_STD`, all paths) |
| Add a new technology | `Code/hymn/io.py` (loaders + remap) and `RANGE_STD` in `Code/hymn/config.py` |
| Re-run the paper end-to-end | `Code/scripts/train_resnet.py` then `Code/scripts/run_evaluation.py` |
| Add a publication figure | `Code/hymn/plotting/figures.py` |
| Inspect inputs to a single solver call | `Code/hymn/evaluation/data_interface.iter_measurements` |

The CSV files in `Code/data/csv/` are a subset of the HYMN Zenodo deposit (DOI `10.5281/zenodo.17979434`), restricted to the BLE, UWB, and WiFi technologies that this paper evaluates. The full set including 5G NR and GNSS measurements is available in the upstream deposit. The manuscript source is maintained separately and is not part of this repository.

## Hardware and software requirements

- **Operating system**: developed on Windows 11. Linux and macOS should work, since no Windows-specific paths or APIs are used in the code.
- **Python**: any version supported by the TensorFlow release pinned in `Code/requirements.txt`. The classical methods (ILS, RLS, BGF) work on any current Python without TensorFlow.
- **Compute**: the classical methods are CPU-only and should run on a conventional laptop or workstation. ResNet training is more demanding and benefits substantially from a CUDA-enabled GPU. CPU-only training is supported, but expect significantly longer runtimes.
- **Disk**: the repository itself is small. Trained checkpoints and intermediate caches are written under `Code/outputs/` and are gitignored.

## Quick start

```bash
git clone https://github.com/TUD-ITVS/hymn-localization-ipin2026.git
cd hymn-localization-ipin2026

# Create a fresh environment
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
. .venv\Scripts\Activate.ps1

pip install -r Code/requirements.txt

# Step 1 — train and predict the ResNet variants
cd Code
python scripts/train_resnet.py        # runs the configured scenarios end-to-end

# Step 2 — run the classical methods, ingest ResNet outputs, produce stats and figures
python scripts/run_evaluation.py
```

The unified evaluation pipeline supports faster iteration:

```bash
python scripts/run_evaluation.py --quick          # UWB + ILS smoke test (~1 min)
python scripts/run_evaluation.py --no-resnet      # classical methods only
python scripts/run_evaluation.py --skip-classical # reuse cached classical results, re-ingest ResNet + re-plot
```

Both scripts also work as modules (e.g. `python -m scripts.run_evaluation --quick`).

## Indicative execution times

Stage runtimes vary substantially by hardware. As rough guidance:

- The smoke test (`python scripts/run_evaluation.py --quick`) completes in well under a minute on any modern machine.
- The classical-only run (`python scripts/run_evaluation.py --no-resnet`) completes in minutes and is dominated by the BGF grid evaluation.
- ResNet training is the long pole. With a recent GPU it is a matter of minutes per protocol. CPU-only training takes substantially longer and is best left to run unattended.

ResNet training time also depends on `RESNET["epochs"]` in [`Code/hymn/config.py`](Code/hymn/config.py) and on when early stopping kicks in.

## Reproducibility — code-to-paper mapping

Every figure and table in the manuscript is produced by the scripts below.

| Manuscript artefact | LaTeX label | Producing script | Output file |
|---|---|---|---|
| Fig. 1 (ECDF of ranging residuals per technology) | `fig:ecdf_ranging` | [`Code/hymn/plotting/figures.py`](Code/hymn/plotting/figures.py) — `plot_ecdf_ranging` | `Code/outputs/evaluation/figures/ecdf_ranging.pdf` |
| Table I (positioning error stats) | `tab:positioning_stats` | [`Code/hymn/evaluation/stats.py`](Code/hymn/evaluation/stats.py) — `write_stats` | `Code/outputs/evaluation/stats_table.csv` (and `.tex`) |
| Fig. 2 (ECDF of horizontal positioning error) | `fig:ecdf_fused` | [`Code/hymn/plotting/figures.py`](Code/hymn/plotting/figures.py) — `plot_ecdf_fused` | `Code/outputs/evaluation/figures/ecdf_fused.pdf` |
| Fig. 3 (spatial median-error heatmap) | `fig:spatial` | [`Code/hymn/plotting/figures.py`](Code/hymn/plotting/figures.py) — `plot_spatial_heatmap_fused` | `Code/outputs/evaluation/figures/spatial_heatmap_fused.pdf` |

All paths in the *Output file* column are populated when the pipeline is run. The repository ships only the source code and the input CSVs.

ResNet results come from `python scripts/train_resnet.py` runs whose summary CSVs are then ingested by `hymn.evaluation.resnet_ingest` (which auto-discovers the latest checkpoint per protocol — no config edits needed after retraining). Trained `.h5` checkpoints are not redistributed because they regenerate from the bundled CSV inputs and the random seed pinned in [`Code/hymn/config.py`](Code/hymn/config.py).

## Reproducibility — randomness

A single integer constant `RESNET["random_seed"] = 42` in [`Code/hymn/config.py`](Code/hymn/config.py) controls:

- pandas-side `train_test_split` and `shuffle` calls in [`Code/hymn/methods/resnet/data_split.py`](Code/hymn/methods/resnet/data_split.py)
- TensorFlow weight initialisation, dropout masking, and shuffling, applied through `tf.keras.utils.set_random_seed(...)` at the start of `model_training` in [`Code/hymn/methods/resnet/train.py`](Code/hymn/methods/resnet/train.py)

Bit-exact reproduction of ResNet outputs depends on TensorFlow version, hardware, and whether op-level GPU determinism is enabled. Cross-platform variation in the last two decimal places is expected.

## Citation

If you use this code, please cite the IPIN 2026 paper and the archived release. The [`CITATION.cff`](CITATION.cff) file is the machine-readable source. GitHub renders it as a *Cite this repository* button.

```bibtex
@inproceedings{Schwarzbach2026_HYMN_benchmark,
  author    = {Schwarzbach, Paul and Ammad, Muhammad},
  title     = {From Least Squares to Deep Learning: Benchmarking Indoor Positioning on the {HYMN} Multi-Technology Dataset},
  booktitle = {Proceedings of the 15th International Conference on Indoor Positioning and Indoor Navigation (IPIN 2026)},
  year      = {2026}
}

@misc{HYMN_Code_zenodo_2026,
  author    = {Schwarzbach, Paul and Ammad, Muhammad},
  title     = {{HYMN} Multi-Technology Indoor Positioning Evaluation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

The HYMN dataset itself, and its data descriptor, should be cited separately:

```bibtex
@misc{HYMN_Dataset_zenodo_2025,
  author    = {Michler, Albrecht and Ammad, Muhammad and Schwarzbach, Paul and Ninnemann, Jonas and {Ußler}, Hagen},
  title     = {TUD-ITVS/HYMN-dataset},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17979434}
}

@misc{Ammad2026_HYMN_descriptor,
  author    = {Ammad, Muhammad and Michler, Albrecht and Schwarzbach, Paul and Ninnemann, Jonas and {Ußler}, Hagen and Michler, Oliver},
  title     = {Descriptor: A Hybrid Indoor and Indoor-Outdoor Positioning Multi-Technology Dataset ({HYMN})},
  year      = {2026},
  publisher = {arXiv},
  doi       = {10.48550/arXiv.2604.20349}
}
```

## Acknowledgement

The authors thank Albrecht Michler, Jonas Ninnemann, and Hagen Ußler, co-authors of the HYMN dataset, for their contributions to the dataset collection, ground truth acquisition, and measurement campaigns on which this evaluation is built.

## Declaration on Generative AI

During the development of this repository, the authors used Anthropic's Claude (via the Claude Code CLI) to assist with the design, refactoring, and documentation of the evaluation pipeline. All AI-generated content was reviewed, verified, and edited by the authors, who take full responsibility for the final content of this repository.
