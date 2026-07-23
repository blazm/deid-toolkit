# deid-toolkit

A toolkit for running and evaluating privacy-preserving techniques in facial biometrics.

## Two Modes

- **Pipeline mode** (`deid run`) — Select datasets, techniques, and evaluation metrics; run the full de-identification pipeline
- **Explore mode** (`deid explore`) — Interactive web UI for browsing results (before/after image comparison, Summary tab, metric tables, galleries)

## Quick Start

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate deid-toolkit

# 2. Install the CLI
pip install -e .

# 3. Initialize the config
deid config migrate --yes

# 4. Select datasets, techniques, evaluations
deid list datasets
deid list techniques
deid list evaluation

deid select datasets arface ck+_fix
deid select techniques deepprivacy2 ksamenet
deid select evaluation ssim lpips vggface

# 5. Run the pipeline
deid run all

# 6. Explore results
deid explore
```

## CLI Reference

```bash
# List
deid list datasets                 # Available datasets
deid list techniques               # Available techniques
deid list evaluation               # Available metrics
deid list results                  # Available results
deid list selected                 # Preview what 'deid run selected' would do

# Select
deid select datasets arface lfw     # By name
deid select datasets 0 1            # By index
deid select techniques deepprivacy2
deid select evaluation ssim lpips

# Run
deid run all                       # Full pipeline (preprocess + techniques + evaluation)
deid run preprocess                # Alignment + pair generation
deid run techniques                # Techniques only
deid run evaluation                # Evaluation only
deid run selected                  # Resume incomplete stages
deid run logs                      # Latest pipeline log

# Config
deid show                          # Current configuration
deid migrate [--yes]               # Migrate legacy config.ini → deid-config.yaml
deid explore [--port 8501]         # Streamlit result browser
```

## Architecture

```
deid/                          # CLI package + built-in scripts
  cli/                         # Typer CLI commands
  config/                      # Unified config (Pydantic + YAML)
  pipeline.py                  # Pipeline orchestrator
  utils/                       # Ported utilities (align_face_mtcnn, pair generation)
  techniques/                  # Built-in DEID technique scripts
  evaluation/                  # Built-in evaluation scripts + identity verification protocols
  environments/                # Built-in conda env configs
  explore/                     # Streamlit app (Compare/Summary/Embeddings/Metrics/Gallery)
  __main__.py                  # Entry: `python -m deid`

legacy/                        # Deprecated old codebase (preserved for reference)
  modules/                     # Legacy cmd.Cmd shell
  evaluations/, techniques/    # Legacy scripts
  environments/                # Legacy conda configs
  visualization/               # Legacy plot scripts

root_dir/                      # Data directory (user extension points)
  deid-config.yaml             # Active config (selections + settings)
  pipeline.yml                 # Rename mappings + technique args
  datasets/                    -- original/, aligned/, labels/, pairs/
  techniques/                  -- User extension: custom technique scripts
  evaluation/                  -- User extension: custom evaluation scripts
  environments/                -- User extension: custom conda env YAMLs
  results/                     -- Evaluation CSVs
```

## Adding a New Technique

**Built-in** — add to `deid/techniques/` and `deid/environments/`
**User custom** — place `my_technique.py` in `root_dir/techniques/` and `my_technique.yml` in `root_dir/environments/`. User scripts take priority over built-in.

## Adding a New Evaluation

**Built-in** — add to `deid/evaluation/`
**User custom** — place `my_metric.py` in `root_dir/evaluation/`. User scripts take priority over built-in.

Script receives: `--aligned_path`, `--deid_path`, `--dataset_name`, `--technique_name`, `--impostor_pairs_filepath`, `--genuine_pairs_filepath`, `--save_path`, `--root_dir`, `--eval_package_dir`

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

Legacy `config.ini` is still read as a fallback during migration.

## Prerequisites

- **Python 3.9+**
- **Conda/Mamba** (see `environment.yml`)
- **CUDA-capable GPU** for deep learning techniques

## Installation

```bash
# 1. Create the conda environment (named "deid-toolkit")
conda env create -f environment.yml
conda activate deid-toolkit

# 2. Install the toolkit package (choose one):
pip install -e .                       # Core CLI only (compute pipeline + config)
pip install -e ".[reports]"            # + PDF report generation (matplotlib, seaborn)
pip install -e ".[explore]"            # + Streamlit web UI (browse results in browser)
pip install -e ".[full]"               # Everything (deep learning deps for all techniques)

# 3. Set up the workspace directory
deid migrate-structure                 # Creates root_dir/ with needed subdirectories
cp examples/workspace/deid-config.yaml root_dir/  # Copy config template

# 4. Initialize config (if migrating from legacy config.ini)
deid migrate --yes
```

### Installation Tiers

| Command | Includes | Use case |
|---------|----------|----------|
| `pip install -e .` | Core CLI + config | Run pipeline, list/select commands |
| `pip install -e ".[reports]"` | + matplotlib, seaborn | PDF report generation |
| `pip install -e ".[explore]"` | + streamlit, plotly | Web UI (reads results CSVs) |
| `pip install -e ".[full]"` | Everything | Complete toolkit (all techniques + evals) |

### Two-Machine Deployment

For the typical workflow (compute on workstation A, serve visualizations on server B):

```bash
# Machine A — Compute workstation (GPU required)
conda activate deid-toolkit
pip install -e ".[full]"
deid run all                           # Runs full pipeline
# Transfer results to Machine B:
robocopy root_dir/results user@B:/workspace/root_dir/results /E
robocopy root_dir/datasets/deidentified user@B:/workspace/root_dir/datasets/deidentified /E

# Machine B — Visualization server (no GPU needed)
conda activate deid-toolkit
pip install -e ".[explore]"
deid explore --port 8501               # Serves web UI at localhost:8501
```

## Daemon Mode (`deid serve`)

Run the explore app continuously with auto-reload on file changes — ideal for development and long-running demo setups.

```bash
# Start in a tmux session (background)
tmux new -d -s deid
deid serve

# Attach back anytime
tmux attach -t deid

# Stop
tmux kill-session -t deid
```

Or run directly (blocks the terminal):

```bash
deid serve --port 8501
```

See [`DAEMON.md`](DAEMON.md) for full details.

## Data Paths

All data lives under `root_dir/` (your workspace folder — configurable via `deid-config.yaml`):

| Data | Path |
|------|------|
| Original images | `root_dir/datasets/original/{dataset_name}/img/` |
| Aligned images | `root_dir/datasets/aligned/{dataset_name}/` |
| De-identified images | `root_dir/datasets/deidentified/{technique_name}/{dataset_name}/` |
| Labels | `root_dir/datasets/labels/{dataset_name}_labels.csv` |
| Pairs | `root_dir/datasets/pairs/{dataset_name}_{impostor\|genuine}_pairs.txt` |
| Results | `root_dir/results/` |
| Config | `root_dir/deid-config.yaml` |
