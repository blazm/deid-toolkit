# SOTA Embedding-Space Evaluators

The state-of-the-art evaluation stack (SOTA evaluators):
two independent face-recognition **probes** (SwinFace, TransFace) plus the analysis
scripts that turn their embeddings into the paper's verification, cross-condition
linkability, identity-mapping, drift, retrieval, collapse-ratio and gender/expression
results.

This is a **copy of the working stack** (`D:\dev\deid-toolkit_evaluators`, which remains
the live working copy during the research period — sync changes from there back into
this folder). Only the code was copied; **no binary model weights are stored or
committed here** (see *Model weights* below). The `root_dir/` results store of the
toolkit is the single shared output location and must not be modified from here.

## Environments

Independent conda envs (see the `environment_*.yml` files; the live machines use
`swinface` with torch cu128 on Python 3.10):

```bash
conda env create -f environment_swinface.yml   # name: swinface
conda env create -f environment_transface.yml  # name: transface
```

## Model weights (not committed)

The scripts resolve weights **relative to this directory** (`models/...`). On the
research machine `models/swinface` and `models/transface` are **directory junctions**
to the working copy's weight folders:

```
models/swinface  ==> D:\dev\deid-toolkit_evaluators\models\swinface
models/transface ==> D:\dev\deid-toolkit_evaluators\models\transface
```

Re-create on another machine with (PowerShell/CMD):

```bat
mklink /J <this_dir>\models\swinface  D:\dev\deid-toolkit_evaluators\models\swinface
mklink /J <this_dir>\models\transface D:\dev\deid-toolkit_evaluators\models\transface
```

or simply copy the two used checkpoints:
- `models/swinface/checkpoint_step_79999_gpu_0.pt` — SwinFace (882 MB; identical to
  `deid/evaluation/weights/checkpoint_step_79999_gpu_0.pt` already shipped locally)
- `models/transface/glint360k_model_TransFace_L.pt` — TransFace-L (1.1 GB)

`models/` is included in `.gitignore`.

## Manuscript usage (exactly how the paper's numbers were produced)

Datasets: aligned face crops in
`D:\dev\deid-toolkit\root_dir\datasets\aligned\{rafd,mug-still,kdef}` plus de-identified
per-technique sets in `D:\dev\deid-toolkit\root_dir\datasets\<Technique>\{...}` —
20 baselines (all four of RP, AnonNET, iFADIT, PRO-Face included; the stack and
its `run_*.bat` launchers support any number via the technique folder scan).

```bat
:: 1) Gender + expression (SwinFace attribute heads) per technique, per dataset
conda run -n swinface python evaluate_gender_expression.py --dataset rafd  --output D:\dev\deid-toolkit\root_dir\predictions\rafd
conda run -n swinface python evaluate_gender_expression.py --dataset mug-still --output D:\dev\deid-toolkit\root_dir\predictions\mug-still
conda run -n swinface python evaluate_gender_expression.py --dataset kdef  --output D:\dev\deid-toolkit\root_dir\predictions\kdef
::    -> per-technique gender_results.csv / expression_results.csv + confusion matrices

:: 2) Paper figure: gender accuracy bar + expression accuracy table
conda run -n swinface python plot_gender_expression_summary.py ^
    --rafd-dir   D:\dev\deid-toolkit\root_dir\predictions\rafd ^
    --celeba-dir D:\dev\deid-toolkit\root_dir\predictions\celeba-test ^
    --mug-dir    D:\dev\deid-toolkit\root_dir\predictions\mug-still ^
    --kdef-dir   D:\dev\deid-toolkit\root_dir\predictions\kdef ^
    --output     gender_expression_summary

:: 3) Embeddings for the identity analyses (CelebA-test)
conda run -n swinface python generate_embeddings_swinface.py  ...
conda run -n transface python generate_embeddings_transface.py --weight models\transface\glint360k_model_TransFace_L.pt ...

:: 4) Analyses (all read embeddings + pairs from root_dir, write CSV/HTML/PDF)
conda run -n swinface python verify_embeddings.py ...                :: verification (same-condition) ROC/AUC/EER
conda run -n swinface python analyze_linkability.py ...              :: cross-condition linkability (de-anonymization)
conda run -n swinface python analyze_id_mapping.py ...               :: 1:1 / 1:N / N:1 / M:N mapping structure
conda run -n swinface python analyze_collapse_ratio.py ...           :: per-identity collapse ratio
conda run -n swinface python analyze_identity_drift.py ...           :: displacement fields (violin/CDF/compass)
conda run -n swinface python analyze_retrieval.py ...
conda run -n swinface python analyze_compactness_separation.py ...
conda run -n swinface python analyze_gender_preservation.py ...
conda run -n swinface python visualize_embedding_space.py ...        :: UMAP density projections (explorer: HTML report, overlay, compactness)
conda run -n swinface python drift_direction_v2.py --probe swinface|transface --what density --cols 7 --ref-panel
                                                              :: manuscript UMAP plates: 21-panel 3x7 (cols 7) / 7x3 (cols 3) density +
                                                              :: displacement arrows, reference panel labelled "Reference"
conda run -n swinface python visualize_qualitative_grid.py ...       :: qualitative comparison grid
```

(The `run_*.bat` files in this folder are the exact parameterised invocations used on the
research machine; their `--weight`/`--output` arguments follow the same conventions.)

## Script reference

| Script | Role |
|---|---|
| `evaluate_gender_expression.py` | SwinFace gender + expression (7-class) vs GT labels; datasets rafd / mug-still / kdef; writes the manuscript's `*_results.csv` pairs to `root_dir/predictions/{dataset}/` |
| `extract_attributes_swinface.py` | SwinFace model build + preprocessing (imported by the above) |
| `generate_embeddings_{swinface,transface}.py` | 512-d embeddings per image pair set |
| `batch_run_{swinface,transface}.py`, `batch_verify_all.py`, `verify_embeddings.py` | batch verification (ROC/AUC/EER) |
| `analyze_identity_drift.py`, `drift_direction_v2.py`, `drift_analysis_*` (gen.) | displacement / drift diagnostics |
| `analyze_linkability.py` | cross-condition linkability (EER vs originals, R@1, CMC) |
| `analyze_id_mapping.py` | person ↔ de-identified-identity mapping regimes |
| `analyze_collapse_ratio.py` | per-identity compaction/expansion (paper's "collapse ratio") |
| `analyze_retrieval.py`, `analyze_compactness_separation.py` | open-set retrieval; spread diagnostics |
| `analyze_gender_preservation.py`, `plot_gender_expression_summary.py` | attribute-utility figures/tables for the paper |
| `visualize_embedding_space.py`, `visualize_qualitative_grid.py` | UMAP explorer (HTML report, overlay/compactness); qualitative grids |
| `drift_direction_v2.py` | manuscript UMAP density plates + displacement arrows (`--cols 3/7 --ref-panel` → 7×3 / 3×7) |
| `make_pipeline_figure.py`, `make_vertical_figures.py` | paper layout figures |

## Conventions

- All model loads are local (no downloads at run time: no HF cache, no torch.hub,
  no `~/.deepface`/`~/.insightface` fetches in this stack).
- Outputs go to `D:\dev\deid-toolkit\root_dir\{predictions,embeddings,results}` —
  the toolkit's data store. **Never move, rename or delete anything under `root_dir/`
  from this stack**; it is the research data record.
- Windows note: OpenMP duplication between torch and opencv is handled in-script
  (`KMP_DUPLICATE_LIB_OK=TRUE`).
