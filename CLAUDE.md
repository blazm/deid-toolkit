# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

**deid-toolkit** (v0.2.0) is a toolkit for running and evaluating privacy-preserving de-identification techniques in facial biometrics. It provides two modes:

- **`deid run`** — Pipeline execution (select datasets/techniques/evals, run the full pipeline)
- **`deid explore`** / **`deid serve`** — Streamlit web UI for browsing results

## Quick Start

```bash
conda env create -f environment.yml && conda activate deid-toolkit
pip install -e .                        # or [explore] / [reports] / [full] extras
deid migrate --yes                      # legacy config → deid-config.yaml
deid select wizard                      # interactive selection
deid run all                            # full pipeline
deid explore                            # or: deid serve (auto-reload)
```

## Installation Tiers

| Command | Includes | Use case |
|---------|----------|----------|
| `pip install -e .` | Core CLI + config | Run pipeline, list/select commands |
| `pip install -e ".[reports]"` | + matplotlib, seaborn | PDF report generation |
| `pip install -e ".[explore]"` | + streamlit, plotly | Web UI (reads results CSVs) |
| `pip install -e ".[full]"` | Everything | Complete toolkit |

## CLI Commands

```bash
# List
deid list datasets|techniques|evaluation|results|selected

# Select
deid select datasets arface lfw         # By name or index
deid select techniques deepprivacy2
deid select evaluation ssim lpips
deid select all -d arface -t dp2 -e ssim  # All in one command
deid select wizard                      # Interactive guided selection

# Run
deid run all                            # Full pipeline (preprocess + techniques + evaluation)
deid run preprocess                     # Alignment + pair generation
deid run techniques                     # Techniques only
deid run evaluation                     # Evaluation only
deid run validation                     # Preprocess + eval on aligned images (reference baseline)
deid run selected                       # Resume incomplete stages
deid run logs                           # Show latest pipeline log

# Config & UI
deid show                               # Current config
deid migrate [--yes]                    # Migrate legacy config.ini → deid-config.yaml
deid migrate-structure                  # Create workspace dirs with .gitkeep files
deid explore [--port 8501]              # Launch Streamlit (blocks terminal)
deid serve [--port 8501]                # Auto-reload server (watchdog-based file watching)
```

## Architecture

```
deid/                          # CLI package + built-in scripts
  __init__.py                  # __version__ = "0.2.0"
  __main__.py                  # Entry: `python -m deid`
  cli/                         # Typer CLI
    main.py                    # Typer app root (run/list/select top-level + serve)
    commands.py                # list/select/run subcommands + show/migrate/explore/top-level cmds
    serve.py                   # Auto-reload Streamlit server (watchdog-based file watching)
  config/                      # Unified config layer
    models.py                  # Pydantic Settings model (DatasetSelection, TechniqueSelection, etc.)
    loader.py                  # Config loading (YAML preferred, INI+pipeline.yml fallback)
    migrator.py                # config.ini + pipeline.yml → deid-config.yaml
  pipeline.py                  # Pipeline orchestrator (cross-platform subprocess, conda env activation)
  utils/                       # Ported utilities from legacy modules/
    align_face_mtcnn.py        # Multiprocessing face alignment (mp_main)
    generate_img_pairs_all.py  # Genuine/impostor pair generation
  reports/                     # PDF report generation (independent of explore/ — compute-side)
    pdf_export.py              # ROC, CMC, distribution plots, summary tables
  techniques/                  # Built-in DEID technique scripts (13 techniques)
  evaluation/                  # Built-in evaluation scripts
    FID.py, ssim.py, lpips.py, mse.py, dan.py, hsemotion.py
    vggface.py, vggface_optimized.py, adaface_iv.py, adaface_optimized.py
    arcface.py, deepface_vggface.py  # VGG-Face verification (arcface = ONNX, deepface_vggface = DeepFace H5)
    deepface_GD.py, deepface_expression.py, deepface_age.py, deepface_race.py  # DeepFace demographics
    ediffiqa.py, identification.py, restnet18_GD.py, pytorchFid.py
    identity_verification/     # AdaFace, SWINface, InsightFace full protocol code
  environments/                # Built-in conda env YAML configs
  explore/                     # Streamlit app (independent of reports/)
    app.py                     # Main layout: Home/Benchmarks/Survey/Login/Docs/Results/Datasets
    landing.py                 # Public landing page
    public_benchmarks.py       # Public benchmarks (no auth required)
    survey.py, survey_api.py, survey_config.py, survey_analysis.py  # Human verification survey
    login_page.py, auth.py     # Authentication
    docs.py                    # Toolkit documentation tab
    datasets.py                # Dataset browser tab
    compare.py                 # Before/after image comparison
    summary.py                 # Score tables, ROC, distance histograms
    metrics.py                 # Per-metric detailed viewer
    clusters.py                # Embedding clustering (t-SNE/UMAP)
    unsupervised.py            # Unsupervised embedding analysis
    reid_risk.py               # Re-identification risk assessment
    techniques_grid.py         # Techniques comparison grid
    radar_charts.py            # Radar/spider charts for technique comparison
    gallery.py                 # Filterable image gallery
    embedding_analysis.py      # Core: load cached embeddings, project, compute displacement/collapse metrics
    embedding_viz_cli.py       # CLI entry point: `python -m deid.explore.embedding_viz_cli` — also saves `_data.csv` files
    embedding_analysis_tab.py  # Streamlit tab: static matplotlib charts for embedding analysis
    interactive_embedding_tab.py  # Streamlit tab: Plotly-based interactive viewer (reads `_data.csv` from CLI output)
    data_loader.py             # Discovery of datasets, techniques, results (singleton ConfigLoader)
    viz.py                     # Plotting helpers (matplotlib/seaborn)
    pdf_export.py              # Backwards-compat shim → deid.reports.pdf_export
    assets.py                  # Static assets (logo, etc.)

label_generation_csv/          # Dataset-specific label extraction scripts
  {dataset}_labels.py          # One per dataset (Ar-face, LFW, CK+, UTKFace, etc.)

docker/                        # Docker-specific configuration
  conda-env.yml                # Frozen conda export for reproducible builds
  README.md                    # Docker setup documentation

root_dir/                      # Data directory (user extension points)
  deid-config.yaml             # Active config (selections + settings)
  datasets/                    # original/, aligned/, labels/, pairs/, deidentified/
  techniques/                  # User extension: custom technique scripts (priority over built-in)
  evaluation/                  # User extension: custom eval scripts (priority over built-in)
  environments/                # User extension: custom conda env YAML configs
  results/                     # Evaluation CSVs ({technique}/{dataset}/*.csv)
```

## Key Files

| File | Purpose |
|------|---------|
| `root_dir/deid-config.yaml` | Active config (selections + settings). Edit directly or via CLI. |
| `pyproject.toml` | Package deps, entry point (`deid = "deid.cli.main:app"`), optional extras. |
| `environment.yml` | Conda environment (Python 3.9+, PyTorch, pytorch-cuda=12.1). |
| `docker/conda-env.yml` | Frozen conda export for Docker builds (pinned versions). |
| `deid/pipeline.py` | Pipeline orchestrator — `run_preprocess`, `run_techniques`, `run_evaluations`, `run_all`. |
| `deid/config/models.py` | Pydantic Settings model with computed path properties. |
| `deid/config/loader.py` | Config loading (YAML preferred → INI+pipeline.yml fallback). Dual-discovery logic. |
| `deid/utils/align_face_mtcnn.py` | Multiprocessing face alignment (`mp_main`). |

## Config Loading

`root_dir/deid-config.yaml` is the single source of truth. Falls back to `config.ini` + `root_dir/pipeline.yml` (legacy). Environment variables prefixed with `DEID_` override settings (via pydantic-settings).

```yaml
root_dir: root_dir
result_dir: results
logs_dir: logs
datasets:
  selected: [arface, ck+_fix]
techniques:
  selected: [deepprivacy2, ksamenet]
  args: {ksamenet: "--postprocessing_alignment yes"}
evaluation:
  selected: [ssim, lpips]
```

## Pipeline Flow

1. **Preprocess**: MTCNN alignment (`align_face_mtcnn.mp_main`) + pair generation (`generate_img_pairs_all.main`). Checks `label_generation_csv/` for label extraction scripts and warns if missing.
2. **Techniques**: Runs each technique script via `run_technique()`, activating the conda env from `environments/`. Scripts receive: `"{script}" "{aligned_path}" "{save_path}" {extra_args}`. Deepprivacy2 gets special `--dataset_filetype` handling.
3. **Evaluations**: Runs each eval script via `run_evaluation()`. Scripts receive 9 CLI args (aligned/deid paths, dataset/technique names, pairs, save path, root_dir, eval_package_dir). Results go to `results/{tech}/{ds}/{eval}.csv`.
4. **Validation**: Always runs evaluations on aligned images as a reference baseline (`results/validation/{ds}/`).
5. **Post-run**: `run_all()` writes `manifest.json` and auto-generates PDF reports via `deid.reports.pdf_export` (non-critical; failures logged as warnings).

## Dual-Discovery (User vs Built-in)

Scripts in `root_dir/techniques/`, `root_dir/evaluation/`, and `root_dir/environments/` take **priority** over built-in `deid/techniques/`, `deid/evaluation/`, and `deid/environments/`. The loader (`ConfigLoader`) unions both sources, with user scripts winning on name collisions.

## Adding a New Technique

**Built-in** — add to `deid/techniques/` and `deid/environments/`.
**User custom** — place `my_technique.py` in `root_dir/techniques/` and `my_technique.yml` in `root_dir/environments/`. User scripts take priority.

Script signature: `python script.py {aligned_path} {save_path} [extra_args]`

## Adding a New Evaluation

**Built-in** — add to `deid/evaluation/`.
**User custom** — place in `root_dir/evaluation/`. User scripts take priority.

Script receives 9 CLI args:
`--aligned_path`, `--deid_path`, `--dataset_name`, `--technique_name`,
`--impostor_pairs_filepath`, `--genuine_pairs_filepath`,
`--save_path`, `--root_dir`, `--eval_package_dir`

## Available Evaluation Models

All scripts use `utils.read_args()` for CLI parsing (positional: `aligned_path`, `deidentified_path`; named: `--save_path`, `--dir_to_log`, `--root_dir`, pair filepaths).

### Identity Verification (pair-based)

Compares face embeddings of aligned vs de-identified image pairs. Requires genuine + impostor pair files.

| Script | Select name | Backend | Status | Notes |
|--------|-------------|---------|--------|-------|
| `arcface.py` | `arcface` | ONNX | Working | Fastest (~24 min / 18k pairs). Auto-detects GPU. |
| `adaface_optimized.py` | `adaface_optimized` | PyTorch + MTCNN | Working | Feature caching (pickle). Slower first run, fast on rerun. |
| `adaface_iv.py` | `adaface_iv` | PyTorch + MTCNN | Working | Full protocol (re-aligns each image via MTCNN internally). ~75 min / 18k pairs. |
| `swinface.py` | `swinface` | PyTorch (SwinT) | Working | Feature caching. ~50 min / 18k pairs. Needs `--root_dir`. |
| `deepface_vggface.py` | `deepface_vggface` | TensorFlow (DeepFace H5) | Working | Slowest (~3+ hr / 18k pairs). Auto-downloads VGG-Face H5 weights. |
| `vggface.py` / `vggface_optimized.py` | `vggface` / `vggface_optimized` | PyTorch (.t7) | Broken | Requires `torchfile` fix for .t7 loader. Use `deepface_vggface` instead. |

### Image Quality (per-image, no pairs needed)

Compare aligned vs de-identified images file-by-file. No pair files required.

| Script | Select name | Backend | Status | Notes |
|--------|-------------|---------|--------|-------|
| `ssim.py` | `ssim` | PyTorch (pytorch-msssim) | Working | Outputs SSIM + MS-SSIM (two CSVs). |
| `lpips.py` | `lpips` | PyTorch (lpips) | Working | LPIPS-Alex. CPU-only by default (`use_gpu=False` in code). |
| `mse.py` | `mse` | PyTorch | Working | Pixel-wise mean squared error. |
| `FID.py` | `fid` | PyTorch (pytorch-fid / InceptionV3) | Working | Distribution-level metric (directory-vs-directory, single score). |
| `ediffiqa.py` | `ediffiqa` | PyTorch | Working | No-reference face IQ. Outputs 3 CSVs: `_aligned`, `_deid`, `_delta`. Variants: T/S/M/L (default S). |

### Data Utility — Attribute Preservation (per-image, no pairs needed)

Classify aligned vs de-identified images independently, measure attribute match rate.

| Script | Select name | Backend | Status | Notes |
|--------|-------------|---------|--------|-------|
| `dan.py` | `dan` | PyTorch (AffecNet) | Working | Facial expression (7 classes). ~23 min / 1k images. |
| `deepface_GD.py` | `deepface_gender` | TensorFlow (DeepFace) | Working | Gender classification. Uses `detector_backend=skip`. |
| `deepface_expression.py` | `deepface_expression` | TensorFlow (DeepFace) | Working | Expression classification. |
| `deepface_age.py` | `deepface_age` | TensorFlow (DeepFace) | Working | Age bucket: child/teen/adult/mid_adult/elderly. |
| `deepface_race.py` | `deepface_race` | TensorFlow (DeepFace) | Working | Race classification. |
| `restnet18_GD.py` | `restnet18_GD` | PyTorch (ResNet18) | Unverified | Gender via ResNet18. Needs external weights. |
| `hsemotion.py` | `hsemotion` | External module | Unverified | Needs external dependency. |

### Weights location

Pre-trained checkpoints: `deid/evaluation/weights/`. DeepFace weights auto-download to `~/.deepface/weights/` on first use.

## Embedding Cache Structure

Verification models with caching store `.pkl` files under `root_dir/preprocess/temp/{model}/`:

```
root_dir/preprocess/temp/
  {model}/
    {dataset}/original/              # Original face embeddings (shared across techniques)
    {dataset}/deid/{technique}/      # De-identified embeddings (computed per technique)
```

All three models (AdaFace, SWINFace, DeepFace VGG-Face) share the same layout. Originals computed once per dataset, reused across techniques.

**Cache format by model:**
- **SWINFace**: `dict[str, torch.Tensor]` with key `"Recognition"` (512-d)
- **AdaFace**: `torch.Tensor` (512-d) — only `adaface_optimized.py` caches; `adaface_iv.py` does not
- **DeepFace VGG-Face**: raw `numpy.ndarray` (4096-d)

**ArcFace** does NOT cache embeddings — ONNX inference without pickle caching.

## Embedding Space Analysis / Visualization

CLI and Streamlit tab for per-image displacement fields, identity collapse detection, and multi-technique comparison. Three projection methods: UMAP (default), PCA (interpretable axes), t-SNE.

```bash
# CLI (PDF + PNG + CSV output)
python -m deid.explore.embedding_viz_cli \
  --dataset celeba-test_aligned --model swinface \
  --techniques blur pixelize --method umap

# Interactive: deid serve → Results → Embedding Analysis / Interactive Viewer
```

Output: `root_dir/results/viz/{dataset}_{model}/` with:
- `displacement_{m}_{ds}_{tech}.pdf/.png` — Static displacement figure
- `displacement_{m}_{ds}_{tech}_data.csv` — Per-image data (orig/deid coords, cos_sim, euclidean_dist) for interactive viewer
- `collapse_{m}_{ds}_{tech}.pdf/.png/.csv` — Collapse chart + metrics table
- `comparison_{m}_{ds}_multi.pdf/.png` — Multi-technique overlay figure
- `comparison_{m}_{ds}_multi_data.csv` — Per-image data for all techniques (for interactive viewer)

The `_data.csv` files are read by the "Interactive Viewer" tab in Streamlit (Plotly-based, hover tooltips, identity filtering).

Key files: `deid/explore/embedding_analysis.py`, `deid/explore/embedding_viz_cli.py`, `deid/explore/embedding_analysis_tab.py`, `deid/explore/interactive_embedding_tab.py`

## Explore App (Streamlit)

| Entry | Mode |
|-------|------|
| `deid explore` | Launches Streamlit; blocks terminal. |
| `deid serve` | Auto-reload server. Watches `deid/explore`, `deid/evaluation`, `deid/techniques`, `root_dir/datasets/labels`. Falls back to restart-on-exit loop if watchdog is not installed. |

**Auth**: Login gate protects Docs, Results, and Datasets tabs. Public tabs: Home, Benchmarks, Survey, Login.

**Data access**: Uses a singleton `ConfigLoader` via `data_loader.get_loader()`. Call `reset_loader()` to clear (e.g., on logout).

## Two-Machine Deployment

```
Machine A (Compute workstation):  pip install -e ".[full]"    → runs pipeline
                                    Output: root_dir/results/ + datasets/deidentified/
                                    Transfer: robocopy/rsync results + deidentified images to Machine B

Machine B (Web visualization):    pip install -e ".[explore]"  → lighter deps, reads CSVs
                                    root_dir/deid-config.yaml points to local workspace
                                    Runs: deid explore
```

## Subprocess Details

`run_streamed()` in `pipeline.py` handles cross-platform shells (Git Bash on Windows, /bin/bash on Unix) with tqdm progress bar support (buffers `\r` lines, flushes every 1s). Conda env activation via `_find_conda_sh()` checks Miniforge3, Anaconda3, Miniconda3 in both `~` and `%LOCALAPPDATA%` on Windows. Falls back to current Python if the specific conda env doesn't exist.

## Results Hierarchy

```
results/
  manifest.json                    # Written by run_all() — run metadata
  {technique}/
    {dataset}/
      {evaluation}.csv             # e.g., swinface.csv, arcface.csv, ssim.csv
  validation/
    {dataset}/
      {evaluation}.csv              # Reference baseline (aligned images only)
  viz/
    {dataset}_{model}/             # Embedding space visualizations
      displacement_{model}_{dataset}_{tech}.pdf/.png
      collapse_{model}_{dataset}_{tech}.pdf/.png/.csv
      comparison_{model}_{dataset}_multi.pdf/.png
```

`deid run selected` checks results/ hierarchy to determine which stages to skip.

## CSV Result Format

Evaluation scripts output CSV files with a standardized schema:

| Column | Description | Example |
|--------|-------------|---------|
| `image` | Image filename (aligned side of pair) | `00172.jpg` |
| `{metric}` | Primary metric column name | `cossim`, `ssim`, `mse`, `lpips`, `isMatch` |
| `img_b` | De-identified image filename (verification only) | `00172.jpg` |
| `ground_truth` | Binary label: 1=genuine, 0=impostor (verification only) | `1` |

**Metric-specific schemas:**
- Verification scripts (SWINFace, ArcFace, AdaFace): `image,cossim,img_b,ground_truth`
- Image quality scripts (SSIM, LPIPS, MSE): `image,{metric}`
- Attribute preservation (DAN, DeepFace gender/age/expression/race): `image,isMatch`
- eDifFIQA: Three separate CSVs — `_aligned`, `_deid`, `_delta` (per-image quality scores)

## Docker

```bash
docker compose -f docker-compose-dev.yml up -d --build
docker exec -it deidtoolkit /bin/bash
docker compose -f docker-compose-dev.yml down
```

Uses `docker/conda-env.yml` (frozen export for reproducible builds). For local dev, use `environment.yml`.

## Testing

No formal test suite. Test via:
- `deid list datasets` / `deid list techniques` / `deid list evaluation` — verify discovery
- `deid explore` or `deid serve` — verify Streamlit app
- Manual pipeline runs on small dataset subsets (FRI is the toy dataset)

## TODO / Upcoming

See `TODO.md`: Pose estimation, gaze estimation, and FIQ (Face Image Quality) metrics before vs. after de-identification.
