# deid-toolkit

A framework for running and evaluating privacy-preserving techniques in facial biometrics.
A reference implementation of state-of-the-art de-identification and face-obfuscation
techniques together with the framework around them. Results of the research runs are
stored under `root_dir/`.
The toolkit itself is the **framework only**: MTCNN alignment, pipeline
orchestration, built-in evaluation metrics, and the SOTA embedding-space evaluators
(`sota_evaluators/`). The 20 heavy de-identification baselines are maintained
separately (one conda env + batch script per method; our runner scripts for each
method live in `baselines/`) and drop their outputs into `root_dir/datasets/`.

## Two Modes

- **Pipeline mode** (`deid run`) — Select datasets, techniques, and evaluation metrics; run the full de-identification pipeline
- **Explore mode** (`deid explore`) — Interactive web UI for browsing results (before/after image comparison, Summary tab, metric tables, galleries)

## Quick Start

```bash
conda env create -f environment.yml && conda activate deid-toolkit
pip install -e .
deid migrate --yes
deid verify          # NEW: check that your datasets/labels/pairs/technique outputs
                     # are properly prepared (read-only diagnostics, exit code 1 on FAIL)
deid list datasets   # e.g. arface lfw fri mug-still
deid select datasets arface ck+_fix
deid select techniques deepprivacy2    # one or more by name/index
deid select evaluation ssim lpips vggface adaface_iv arcface swinface
deid run all
deid explore
```

Or from the Windows launcher at the repo root (any current directory):

```bat
run_pipeline.bat           " no argument: runs the FULL pipeline (deid run all)
run_toolkit.bat            " no argument: runs the full preparation check (deid verify --all)
run_toolkit.bat verify     " same, explicit (or --all / --quiet / --detail)
run_toolkit.bat list datasets
run_toolkit.bat explore
rem - everything after the script name is passed to the deid CLI
```

**`deid verify`** — preparation diagnostics before running anything:
per selected dataset it checks the aligned image count, label-CSV discovery and
row/path coverage (including whether the gender column is actually filled),
and sample-validates the genuine/impostor pair files; per selected technique it
locates the output folder (legacy `deidentified/{tech}/{ds}` or dataset-root
`datasets/{Technique}/{ds}`) and compares output vs aligned counts (shortfalls
are WARN — technique failure logs are a supported condition). It also reports
SOTA-stack artifacts (`root_dir/predictions/`, `root_dir/embeddings/`) and the
Python/torch/CUDA environment. **Strictly read-only**: it never writes under
`root_dir/`. Flags: `--all` (every aligned dataset, not just selected),
`--detail`, `--quiet` (non-PASS lines only).

## Acknowledgments

This work is part of the research project **“Enhancing Biometrics with Diffusion Models
and Differential Privacy”**.

We are grateful to **NVIDIA Corporation** for the donation of GPU hardware used in
this work, under the **NVIDIA Academic Grant Program**.

## SOTA Embedding-Space Evaluators

`deid explore` and the built-in `deid select evaluation` metrics below are the
**lightweight** stack. The main SOTA numbers come from the independent
embedding-space stack in **`sota_evaluators/`**: two probe models (SwinFace + TransFace)
feeding verification, cross-condition linkability, identity-mapping structure,
drift / retrieval / collapse diagnostics, and SwinFace gender + expression attribute
utility on RaFD / MUG-Still / KDEF. Usage, environments, run scripts, and the
(junction-based, git-ignored) model-weight layout are documented in
`sota_evaluators/README.md`. Its outputs land in `root_dir/predictions/` and
`root_dir/embeddings/`.

## Baselines — the 20 evaluated approaches (our batch scripts)

**Positioning:** the toolkit itself ships the framework plus the two basic
built-in techniques (`blur`, `pixelize` — they run out of the box via `deid run`).
Everything in `baselines/` is *supporting material for the methods evaluated in
the study*, not part of the toolkit: we offer the batch-runner scripts only —
running a method on your side means obtaining its official repository and
weights (linked per method below) and placing them per the manifest.

`baselines/` contains the toolkit's own batch runner for each of the 20
de-identification approaches evaluated in the study (one `deidentify_batch*.py`
per method, plus our local helper scripts where needed — e.g. NullFace's
local face-embedding extractor, RP's SDXL pipeline loader).
The full official method repos and model weights are **not** shipped (they are
large); per method you get our script + a `weights_manifest.txt` (exact weight
files, paths, sizes) to place from the method's official release. All runners
share the same contract: aligned face crops in `--input` → same-basename PNGs
in `--output`, skip-existing, failure log, 256² output unless the method's
protocol dictates otherwise. The full list — paper, required official code,
per-method conda environment, weight sources — is in **`baselines/README.md`**:

> DeepPrivacy · CLEANIR · IPFA · RiDDLE · AMT-GAN · FALCO · CPP-DeID · LDFA ·
> DeepPrivacy2 · G2Face · GANonymization · FADM · DiffPrivate · FAMS · AIDPro ·
> NullFace · RP · AnonNET · iFADIT · PRO-Face

To run a method, you need our runner script plus (for 17 of the 20) the method's
official repository and weights — all linked per method in `baselines/README.md`;
the weights manifest (`weights_manifest.txt` per method) lists exactly which
files to place and where. Three runners need **only** the files in `baselines/`
plus off-the-shelf model weights (no official-repo code): **LDFA** (Stable
Diffusion 2 inpainting), **RP** (SDXL), and **PRO-Face** (blur + IResNet-50
restore configuration). Run each method inside its own conda env. The two basic
built-ins (`deid/techniques/blur.py`, `pixelize.py`) accept the same
`--input/--output` batch contract (as well as the legacy positional pipeline
interface), so you can call them exactly like the baseline runners.
```
python deid/techniques/blur.py --input <aligned_dir> --output <deid_dir>
```

## CLI Reference

```bash
# List
deid list datasets|techniques|evaluation|results|selected

# Select
deid select datasets arface lfw       # By name or index
deid select techniques deepprivacy2
deid select evaluation ssim lpips
deid select all -d arface -t dp2 -e ssim  # All in one command
deid select wizard                     # Interactive guided selection

# Run
deid run all                           # Full pipeline (preprocess + techniques + eval)
deid run preprocess                    # Alignment + pair generation
deid run techniques                    # Techniques only
deid run evaluation                    # Evaluation only
deid run validation                    # Preprocess + evaluation only (aligned as reference)
deid run selected                      # Resume incomplete stages
deid run logs                          # Show latest pipeline log

# Config & UI
deid show                              # Current config
deid verify [--all] [--detail] [--quiet]  # Dataset/technique preparation diagnostics (read-only)
deid migrate [--yes]                   # Migrate legacy config.ini → deid-config.yaml
deid migrate-structure                 # Create workspace dirs with .gitkeep files
deid explore [--port 8501]             # Launch Streamlit web UI
deid serve [--port 8501]               # Auto-reload server (watchdog-based)
```

## Available Evaluation Metrics

### Image Quality (perceptual similarity)

| Script | Name for `deid select` | What it measures |
|--------|----------------------|-----------------|
| `ssim.py` | `ssim` | Structural Similarity Index |
| `lpips.py` | `lpips` | Learned Perceptual Image Patch Similarity |
| `mse.py` | `mse` | Mean Squared Error |
| `FID.py` | `fid` | Fréchet Inception Distance |
| `ediffiqa.py` | `ediffiqa` | E-DIFFIQA (no-reference image quality) |

### Identity Verification (cosine similarity of face embeddings)

| Script | Name for `deid select` | What it measures | Status |
|--------|----------------------|-----------------|--------|
| `adaface_iv.py` | `adaface_iv` | AdaFace full verification protocol | ✅ Working |
| `adaface_optimized.py` | `adaface_optimized` | AdaFace with feature caching (faster) | ✅ Working |
| `arcface.py` | `arcface` | ArcFace ONNX-based verification (fastest) | ✅ Working |
| `swinface.py` | `swinface` | SWINFace full verification protocol | ✅ Working |
| `deepface_vggface.py` | `deepface_vggface` | VGG-Face via DeepFace (H5 weights, bypasses broken .t7 loader) | ✅ Working |
| `vggface.py` / `vggface_optimized.py` | `vggface` / `vggface_optimized` | VGG-Face (.t7 loader — needs torchfile fix) | ⚠️ Loader issue |

### Data Utility (attribute preservation after de-identification)

| Script | Name for `deid select` | What it measures |
|--------|----------------------|-----------------|
| **Identity** | | |
| `deepface_GD.py` | `deepface_gender` | Gender classification match rate |
| `dan.py` | `dan` | Facial expression (emotion) via AffecNet |
| **DeepFace Demographics** | | |
| `deepface_expression.py` | `deepface_expression` | Facial expression classification |
| `deepface_age.py` | `deepface_age` | Age bucket classification (child/teen/adult/mid_adult/elderly) |
| `deepface_race.py` | `deepface_race` | Race classification |

### Other Evaluations

| Script | Name for `deid select` | What it measures |
|--------|----------------------|-----------------|
| `hsemotion.py` | `hsemotion` | HSEmotion facial emotions (needs external module) |
| `restnet18_GD.py` | `restnet18_GD` | Gender via ResNet18 (needs external module + weight) |
| `deepface_GD.py` | `deepface_GD` | Legacy DeepFace gender (alias for `deepface_gender`) |

## Running the Pipeline — Full Example

```bash
# 1. Set up workspace
deid migrate-structure

# 2. Edit config or use CLI to select
deid select datasets fri mug-still
deid select techniques blur pixelize
deid select evaluation ssim lpips arcface adaface_iv swinface \
                       deepface_gender deepface_expression deepface_age dan

# 3. Run full pipeline
deid run all

# Results appear in root_dir/results/{technique}/{dataset}/*.csv

# 4. Browse results in browser
deid explore
```

## Pipeline Flow

1. **Preprocess**: MTCNN alignment (`align_face_mtcnn.py`) + pair generation (`generate_img_pairs_all.py`)
2. **Techniques**: Runs each technique script (user scripts in `root_dir/techniques/` first, then the basic built-in scripts in `deid/techniques/` — `blur`, `pixelize`; the heavyweight baselines run via the external scripts documented in `baselines/README.md`, and their outputs are consumed from `root_dir/datasets/`)
3. **Evaluations**: Runs selected eval scripts on each dataset/technique combo (lightweight stack; the SOTA stack lives in `sota_evaluators/`)
4. **Validation**: Always runs evaluations on aligned images as reference baseline (`results/validation/{ds}/`)
5. **Reports**: Auto-generates PDF reports via `deid.reports.pdf_export`

## Data Paths

| Data | Path |
|------|------|
| Original images | `root_dir/datasets/original/{dataset_name}/img/` |
| Aligned images | `root_dir/datasets/aligned/{dataset_name}/` |
| De-identified images | `root_dir/datasets/deidentified/{technique_name}/{dataset_name}/` |
| Labels | `root_dir/datasets/labels/{dataset_name}_labels.csv` |
| Pairs | `root_dir/datasets/pairs/{dataset_name}_{impostor\|genuine}_pairs.txt` |
| Results (evaluation) | `root_dir/results/{technique}/{dataset}/{metric}.csv` |
| Results (SOTA predictions) | `root_dir/predictions/{dataset}/` (per-technique gender/expression CSVs — paper data) |
| Results (SOTA embeddings) | `root_dir/embeddings/{SwinFace,TransFace}/` |
| Results (visualization) | `root_dir/results/viz/{dataset}_{model}/` |
| Embedding cache | `root_dir/preprocess/temp/{model}/{dataset}/` |
| Config | `root_dir/deid-config.yaml` |

`root_dir/` is the project's research data/results store (datasets, labels, pairs,
predictions, embeddings, results, logs). It is a **no-touch zone for toolkit code**:
nothing in `deid/` or `sota_evaluators/` may move, rename, or delete files under it.
Binary contents (original/aligned images, per-baseline de-identified sets, technique
outputs, prediction CSVs, embedding caches) are git-ignored; only small structural
files (labels, .gitkeep, the workspace README) are committed.

## CSV Result Format

Evaluation scripts produce CSVs with a per-metric schema:

| Eval type | Columns | Example |
|-----------|---------|---------|
| Verification (SWINFace, AdaFace Opt.) | `image,cossim,img_b,ground_truth` (+ `cossim_originals` for adaface_optimized) | `00172.jpg,0.493,00172.jpg,1` |
| Verification (ArcFace) | `image,cossim,ground_truth` — no `img_b` column | `00172.jpg,0.493,1` |
| Image quality (SSIM, LPIPS, MSE) | `image,{metric}` — per-image score | `00172.jpg,0.82` |
| Attribute preservation (DAN, DeepFace*) | `image,isMatch` — 1 if attribute preserved | `00172.jpg,1` |
| eDifFIQA | Three separate CSVs: `*_aligned.csv`, `*_deid.csv`, `*_delta.csv` with `image,quality_{x}` columns | `00172.jpg,3.45` |
| FID | Single row with `image` = directory path and metric column = distribution-level score | Not per-image |

## Model Weights

Pre-trained weights live in `deid/evaluation/weights/`:

| File | Used by | Status |
|------|---------|--------|
| `adaface_ir50_ms1mv2.ckpt` | AdaFace eval scripts | ✅ Present |
| `checkpoint_step_79999_gpu_0.pt` | SWINFace | ✅ Present |
| `model.onnx` | ArcFace (ONNX) | ✅ Present |
| `VGG_FACE.t7` | VGG-Face (.t7 — loader broken) | ⚠️ Loader issue |
| `affecnet8_epoch5_acc0.6209.pth` | DAN (emotion) | ✅ Present |
| `face_gender_classification_transfer_learning_with_ResNet18.pth` | ResNet18 gender | ✅ Present |

DeepFace weights (`~/.deepface/weights/`) are auto-downloaded on first use:
- `vgg_face_weights.h5` — VGG-Face (via DeepFace)
- `gender_model_weights.h5`, `age_model_weights.h5`, `facial_expression_model_weights.h5`, `race_model_single_batch.h5` — Demographic classifiers

## Embedding Cache Structure

Verification models cache embeddings as `.pkl` files for reuse across evaluations:

```
root_dir/preprocess/temp/
  {model}/
    {dataset}/original/              # Original face embeddings (shared across techniques)
    {dataset}/deid/{technique}/      # De-identified embeddings (computed per technique)
```

All three models (AdaFace, SWINFace, DeepFace VGG-Face) use the same cache layout. Originals are shared across techniques — each image's original embedding is computed once and reused. De-identified embeddings are computed independently per technique.

**Note:** ArcFace does NOT cache embeddings — it uses ONNX inference without pickle caching.

**Embedding formats:**
| Model | Dimension | Pickle Format |
|-------|-----------|---------------|
| SWINFace | 512-d | `dict[str, torch.Tensor]` with `"Recognition"` key |
| AdaFace | 512-d | `torch.Tensor` |
| DeepFace VGG-Face | 4096-d | raw `numpy.ndarray` |

**Important:** `adaface_iv.py` (full verification protocol) does NOT cache embeddings — it re-aligns and extracts features every run via internal MTCNN. Only `adaface_optimized.py` and `deepface_vggface.py` cache embeddings.

## Embedding Space Visualization

CLI tool and Streamlit tab for analyzing how de-identification techniques manipulate identity embeddings:

```bash
python -m deid.explore.embedding_viz_cli \
  --dataset celeba-test --model swinface \
  --techniques blur pixelize --method umap
```

Output: `root_dir/results/viz/{dataset}_{model}/` — PDF/PNG figures + CSV metrics (displacement fields, identity collapse charts, multi-technique comparison overlays). Projection methods: UMAP, PCA, t-SNE. Interactive version available via `deid serve` → Results → Embedding Analysis tab.

## Configuration

`root_dir/deid-config.yaml`:

```yaml
root_dir: root_dir
result_dir: results
logs_dir: logs
datasets:
  selected: [arface, ck+_fix]
techniques:
  selected: [blur, pixelize]
evaluation:
  selected: [ssim, lpips, arcface, adaface_iv, deepface_gender]
```

## Architecture

```
deid/                          # CLI package + built-in scripts
  cli/                         # Typer CLI commands
  config/                      # Unified config (Pydantic + YAML)
  pipeline.py                  # Pipeline orchestrator
  utils/                       # Ported utilities (align_face_mtcnn, pair generation)
  techniques/                  # Built-in DEID technique scripts
  evaluation/                  # Built-in evaluation scripts
    data_utility/              # DAN emotion classifier module
      DAN/networks/dan.py      # Attention-based emotion network
    identity_verification/     # Full ID verification protocols
      AdaFace/                 # AdaFace (MTCNN alignment + IR-50 backbone)
      swinface/                # SWINFace (SwinT backbone + attention module)
      vgg-face.pytorch/        # VGG-Face PyTorch port (.t7 loader — has issues)
    weights/                   # Pre-trained model checkpoints
  environments/                # Built-in conda env configs
  explore/                     # Streamlit app (Compare/Summary/Embeddings/Metrics)
  __main__.py                  # Entry: `python -m deid`

root_dir/                      # Data directory (user extension points)
  deid-config.yaml             # Active config (selections + settings)
  datasets/                    -- original/, aligned/, labels/, pairs/
  techniques/                  -- User custom technique scripts
  evaluation/                  -- User custom evaluation scripts
  environments/                -- User custom conda env YAMLs
  results/                     -- Evaluation CSVs
```

## Two-Machine Deployment

```bash
# Machine A — Compute (GPU required)
pip install -e ".[full]"
deid run all
robocopy root_dir/results user@B:/workspace/root_dir/results /E

# Machine B — Visualization (no GPU needed)
pip install -e ".[explore]"
deid explore --port 8501
```

## Adding Evaluations

**Built-in** — add to `deid/evaluation/`
**User custom** — place in `root_dir/evaluation/`. User scripts take priority.

Script receives: `--aligned_path`, `--deid_path`, `--dataset_name`, `--technique_name`, `--impostor_pairs_filepath`, `--genuine_pairs_filepath`, `--save_path`, `--root_dir`, `--eval_package_dir`

## Prerequisites

- **Python 3.9+**
- **Conda/Mamba** (see `environment.yml`)
- **CUDA-capable GPU** for deep learning techniques (PyTorch 2.7+ recommended for sm_120 GPUs)
