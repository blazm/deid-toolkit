# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
    swinface.py, identification.py, deepface_GD.py, restnet18_GD.py
    identity_verification/     # AdaFace, SWINface, InsightFace full protocol code
  environments/                # Built-in conda env YAML configs
  explore/                     # Streamlit app (public portal + protected toolkit)
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
  environments/                # User extension: custom conda env YAMLs
  results/                     # Evaluation CSVs ({technique}/{dataset}/*.csv)
```

## Key Files

| File | Purpose |
|------|---------|
| `root_dir/deid-config.yaml` | Active config (selections + settings). Edit directly or via CLI. |
| `pyproject.toml` | Package deps. Entry point: `deid = "deid.cli.main:app"`. |
| `environment.yml` | Conda environment (Python 3.9, PyTorch, CUDA 12.1). |
| `docker/conda-env.yml` | Frozen conda export for Docker builds (pinned versions). |
| `deid/pipeline.py` | Pipeline orchestrator — `run_preprocess`, `run_techniques`, `run_evaluations`, `run_all`. |
| `deid/config/models.py` | Pydantic Settings model with computed path properties. |
| `deid/config/loader.py` | Config loading (YAML preferred → INI+pipeline.yml fallback). Dual-discovery logic. |
| `deid/utils/align_face_mtcnn.py` | Multiprocessing face alignment (`mp_main`). |

## Configuration

`root_dir/deid-config.yaml` is the single source of truth:

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

Environment variables prefixed with `DEID_` override settings (via pydantic-settings).

## Pipeline Flow

1. **Preprocess**: MTCNN alignment (`align_face_mtcnn.mp_main`) + pair generation (`generate_img_pairs_all.main`). Checks `label_generation_csv/` for label extraction scripts and warns if missing.
2. **Techniques**: Runs each technique script via `run_technique()`, activating the conda env from `environments/`. Scripts receive: `"{script}" "{aligned_path}" "{save_path}" {extra_args}`. Deepprivacy2 gets special `--dataset_filetype` handling.
3. **Evaluations**: Runs each eval script via `run_evaluation()`. Scripts receive 9 CLI args (aligned/deid paths, dataset/technique names, pairs, save path, root_dir, eval_package_dir). Results go to `results/{tech}/{ds}/{eval}.csv`.
4. **Validation**: Always runs evaluations on aligned images as a reference baseline (outputs to `results/validation/{ds}/`).
5. **Post-run**: `run_all()` writes `manifest.json` and auto-generates PDF reports via `deid.reports.pdf_export` (non-critical; failures logged as warnings).

## Dual-Discovery (User vs Built-in)

User scripts in `root_dir/techniques/` and `root_dir/evaluation/` take **priority** over built-in `deid/techniques/` and `deid/evaluation/`. Same for `root_dir/environments/` over `deid/environments/`. The loader unions both sources, with user scripts winning on name collisions.

## Adding a New Technique

**Built-in** — add to `deid/techniques/` and `deid/environments/`
**User custom** — place `my_technique.py` in `root_dir/techniques/` and `my_technique.yml` in `root_dir/environments/`. User scripts take priority over built-in.

Script signature: `python script.py {aligned_path} {save_path} [extra_args]`

## Adding a New Evaluation

**Built-in** — add to `deid/evaluation/`
**User custom** — place in `root_dir/evaluation/`. User scripts take priority over built-in.

Script receives: `--aligned_path`, `--deid_path`, `--dataset_name`, `--technique_name`, `--impostor_pairs_filepath`, `--genuine_pairs_filepath`, `--save_path`, `--root_dir`, `--eval_package_dir`

## Two-Machine Deployment

```
Machine A (Compute workstation):  pip install -e ".[full]"    → runs pipeline
                                    Output: root_dir/results/ + datasets/deidentified/
                                    Transfer: robocopy/rsync results + deid images to Machine B

Machine B (Web visualization):    pip install -e ".[explore]"  → lighter deps, reads CSVs
                                    root_dir/deid-config.yaml points to local workspace
                                    Runs: deid explore
```

## Important Implementation Details

- **Config loading**: `deid-config.yaml` (preferred) → `config.ini` + `pipeline.yml` (legacy fallback).
- **Subprocess runner**: `run_streamed()` handles cross-platform shells (Git Bash on Windows, /bin/bash on Unix) with tqdm progress bar support.
- **Conda env discovery**: `_find_conda_sh()` checks Miniforge3, Anaconda3, Miniconda3 in both `~` and `%LOCALAPPDATA%` on Windows. Falls back to current Python if the specific conda env doesn't exist.
- **Results hierarchy**: `results/{technique}/{dataset}/*.csv` with `manifest.json` at the root.
- **`deid run selected`**: Resume logic — checks existing CSVs in `results/` and only runs incomplete stages.
- **Explore app auth**: Login gate protects Docs, Results, and Datasets tabs. Public tabs: Home, Benchmarks, Survey, Login.
- **Explore app data**: Uses a singleton `ConfigLoader` via `data_loader.get_loader()`. Call `reset_loader()` to clear it (e.g., on logout).
- **`deid serve`**: Watchdog-based file watching on `deid/explore`, `deid/evaluation`, `deid/techniques`, `root_dir/datasets/labels`. Falls back to simple restart-on-exit loop if watchdog is not installed.

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
