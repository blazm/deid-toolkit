# Handover State — 2026-08-12

## 1. GOAL

Build SwinFace-based gender and expression evaluation scripts to compare anonymization techniques across datasets, producing tabular results and publication-ready visualizations saved in `D:\dev\deid-toolkit\root_dir\predictions\`.

Two new scripts were created:
- **`evaluate_gender_expression.py`** — Runs SwinFace inference on aligned + de-identified images, compares predictions against ground-truth labels from CSVs, outputs per-dataset CSV tables and confusion matrix plots.
- **`plot_gender_expression_summary.py`** — Reads per-dataset CSV results and produces a grouped bar chart (RaFD vs CelebA gender accuracy), an expression accuracy table across RaFD/MUG/KDEF, and LaTeX versions of both tables.

## 2. ARCHITECTURAL DECISIONS & CONSTRAINTS

### Datasets
| Dataset | Aligned images path | Labels file | Gender GT | Expression GT |
|---------|-------------------|-------------|-----------|---------------|
| **RaFD** | `datasets/aligned/rafd/` (1,406 jpg) | `labels/rafd-frontal_aligned_labels.csv` | ✅ Yes (-1=F, 1=M) | ✅ Yes (7 exprs in binary columns) |
| **CelebA test** | `datasets/aligned/celeba/` (~2,824 jpg) | `labels/celeba-test_labels.csv` | ✅ Yes (-1=F, 1=M) | ❌ No |
| **MUG-Still** | `datasets/aligned/mug-still/` (986 jpg) | `labels/mug-still_labels.csv` | ❌ No (empty column) | ✅ Yes (Emotion_code→expr mapping: 0=Neutral, 1=Anger, 4=Disgust, 5=Fear, 6=Happy, 7=Sadness, 8=Surprise) |
| **KDEF** | `datasets/aligned/kdef/` (2,934 jpg) | `labels/kdef_labels.csv` | ❌ No (empty column) | ✅ Yes — **BUT excludes Scream(2) and Contempt(3)** as no SwinFace equivalent. Mapping: 0=Neutral, 1=Anger, 4=Disgust, 5=Fear, 6=Happy, 7=Sadness, 8=Surprise |

### Supported Techniques (9 total, customizable via `--techniques`)
`AIDPro AMT-GAN DeepPrivacy DeepPrivacy2 G2Face GANonymization IPFA RiDDLE CLEANIR`

Each technique's de-identified images are at `datasets/{technique}/{dataset_name}/` — RaFD uses `.png`, most others use `.jpg`. The script handles both.

### SwinFace Prediction Model
- Gender: 0=Female, 1=Male (binary logits → argmax)
- Expression: 7 classes: Angry(0), Disgust(1), Fear(2), Happy(3), Sad(4), Surprise(5), Neutral(6)
- Uses `extract_attributes_swinface.py` for model building (`build_swinface_model`) and preprocessing
- Checkpoint: `models/swinface/checkpoint_step_79999_gpu_0.pt`

### Output Directory Structure
```
D:\dev\deid-toolkit\root_dir\predictions\
├── rafd/
│   ├── ground_truth.csv
│   ├── gender_results.csv
│   ├── expression_results.csv
│   ├── expression_confusion_matrix.png + .pdf
│   └── expression_confusion_matrix_caption.txt
├── mug-still/  (same structure; gender shows N/A)
├── kdef/       (same structure; gender shows N/A, Scream/Contempt excluded from expr)
├── celeba-test/ (same structure; expression shows N/A)
│
├── gender_expression_summary_gender_accuracy.png + .pdf   ← grouped bar chart
├── gender_expression_summary_gender_accuracy_caption.txt  ← caption text
├── gender_expression_summary_gender_accuracy.tex          ← LaTeX table
├── gender_expression_summary_expression_accuracy.csv      ← cross-dataset expr table
└── gender_expression_summary_expression_accuracy.tex      ← LaTeX table
```

### Visual Design Rules (agreed)
- **Confusion matrix plots:** No title, tightly cropped (tight subplots_adjust), no suptitle. Caption in separate `.txt` file with full description + interpretation suitable for scientific manuscript.
- **Grouped bar chart (RaFD vs CelebA):** No title, no x-axis label ("Anonymization Technique"). Caption fully describes the figure. Dataset names on y-ticks, technique names on x-axis. Two bars per technique (RaFD + CelebA). Dashed lines for aligned baseline accuracy.
- **Grid layout:** `ceil(sqrt(n_techs))` — 9 techs = 3x3, 16 techs = 4x4.
- All plots: PNG @ 300dpi + PDF vector version.

### Label File Format (all datasets share same schema)
CSV columns: Name, Path, Identity, Gender_code, Gender, Age, Race_code, Race, Emotion_code, Neutral, Anger, Scream, Contempt, Disgust, Fear, Happy, Sadness, Surprise, ...

Gender_code: 1=Male, -1=Female → maps to SwinFace (0=Female, 1=Male) via GT_Male = (Gender_code == 1)
Emotion_code values vary per dataset but map to same 7 SwinFace expression classes.

## 3. CURRENT STATUS OF FILES MODIFIED

### New files created:
1. **`D:\dev\deid-toolkit_evaluators\evaluate_gender_expression.py`** (~850 lines)
   - Handles all 4 datasets (rafd, mug-still, kdef, celeba-test) via `DATASET_LOADERS` dict
   - Dataset-specific label loaders: `load_rafd_labels`, `load_mug_labels`, `load_kdef_labels`, `load_celeba_labels`
   - Expression handling when GT is None: CSV shows N/A, caption says "no expr GT", confusion matrix plot skipped with placeholder caption
   - Gender handling when GT is None: CSV shows N/A for all gender fields, caption includes "(no gender GT)"
   - `generate_confusion_matrix()` — 3x3 grid for 9 techniques, tight layout, no title, per-subtitle = technique name + accuracy %, cell annotations with count + %
   - Generates PNG + PDF confusion matrix + `.txt` caption file

2. **`D:\dev\deid-toolkit_evaluators\plot_gender_expression_summary.py`** (~340 lines)
   - `read_gender_csv()` / `read_expression_csv()` — parse per-dataset CSVs
   - `plot_gender_grouped()` — grouped bar chart, no title, no x-label, expanded caption
   - `write_expression_table()` — CSV with 6 columns (RaFD/MUG/KDEF × aligned/deid)
   - `write_gender_tex()` — LaTeX table: technique | RaFD aligned | RaFD DeID | CelebA aligned | CelebA DeID
   - `write_expression_tex()` — LaTeX table: technique | RaFD(align,deid) | MUG(align,deid) | KDEF(align,deid)

### CLAUDE.md already exists at repo root (created earlier)

## 4. IMMEDIATE NEXT STEPS

1. **Grouped bar chart title bug** — user said they see a title text "Gender Classification Accuracy on De-Identified Faces" still appearing despite `ax.set_title()` being removed. Need to investigate:
   - Possible Python bytecode cache issue (old .pyc)
   - Or matplotlib auto-generating something via `bbox_inches='tight'` or default figure attributes
   - Fix: explicitly call `ax.set_title("")` after removing the comment line, and/or check if `fig.add_subplot()` defaults are adding it

2. **Run KDEF evaluation** — was not run during this session due to Scream/Contempt exclusion concern. Should be verified now that the code handles it correctly. ✅ Done (12 techniques).

3. **Consider adding more techniques** — user mentioned 16 techniques for final publication. The script already supports arbitrary lists via `--techniques`.

4. **Potential improvements:**
   - Add a convenience wrapper/bat file to run all datasets at once
   - Consider combining RaFD + CelebA gender results with technique-level aggregation (weighted by image count)
   - The LaTeX tables could use better formatting (e.g., `\multicolumn` for "Baseline" row, or `siunitx` package for aligned decimals)

## 5. TODO — PENDING DATASET PROCESSING

### CelebA-test: de-identified images needed for 3 techniques
KDEF and RaFD already have all 12 technique results. **CelebA-test** still only has the original 9 techniques (AIDPro through RiDDLE). Need to process CelebA-test de-identified images for:

- **NullFace** — `datasets/NullFace/celeba-test/`
- **FADM** — `datasets/FADM/celeba-test/`
- **FAMS** — `datasets/FAMS/celeba-test/`

These must be generated via the deid-toolkit anonymization pipeline before running:
```bash
conda run -n swinface python evaluate_gender_expression.py \
    --dataset celeba-test \
    --techniques AIDPro AMT-GAN DeepPrivacy DeepPrivacy2 G2Face GANonymization IPFA RiDDLE CLEANIR NullFace FADM FAMS \
    --output D:\dev\deid-toolkit\root_dir\predictions\celeba-test
```

After CelebA-test is updated, rerun the gender summary visualization:
```bash
conda run -n swinface python plot_gender_expression_summary.py \
    --rafd-dir   D:\dev\deid-toolkit\root_dir\predictions\rafd \
    --celeba-dir D:\dev\deid-toolkit\root_dir\predictions\celeba-test \
    --mug-dir    D:\dev\deid-toolkit\root_dir\predictions\mug-still \
    --kdef-dir   D:\dev\deid-toolkit\root_dir\predictions\kdef \
    --output     gender_expression_summary
```

## 6. QUICK REFERENCE COMMANDS

```bash
# Evaluate a single dataset
conda run -n swinface python evaluate_gender_expression.py \
    --dataset rafd | mug-still | kdef | celeba-test \
    --techniques AIDPro AMT-GAN DeepPrivacy DeepPrivacy2 G2Face GANonymization IPFA RiDDLE CLEANIR NullFace FADM FAMS \
    --output D:\dev\deid-toolkit\root_dir\predictions\<dataset>

# Generate summary plots + LaTeX tables from existing results
conda run -n swinface python plot_gender_expression_summary.py \
    --rafd-dir   D:\dev\deid-toolkit\root_dir\predictions\rafd \
    --celeba-dir D:\dev\deid-toolkit\root_dir\predictions\celeba-test \
    --mug-dir    D:\dev\deid-toolkit\root_dir\predictions\mug-still \
    --kdef-dir   D:\dev\deid-toolkit\root_dir\predictions\kdef \
    --output     gender_expression_summary

# Copy outputs to predictions dir after running summary script
cp gender_expression_summary* D:/dev/deid-toolkit/root_dir/predictions/
```
