# Embedding Space Visualization — Session Summary

**Date:** 2026-08-08  
**Purpose:** Record all decisions, fixes, and patterns for the UMAP embedding space visualization system so future work can proceed without re-deriving everything.

---

## Key Architecture

### `visualize_embedding_space.py` — Main visualization script

#### Workflow
1. Loads aligned (original) + anonymized embeddings from both SwinFace and TransFace model directories
2. Fits UMAP on ALL 2,824 aligned embeddings (n_neighbors=30, min_dist=0.1, cosine metric, random_state=42)
3. Projects each technique's embeddings through the fitted UMAP transformer
4. Renders 4 figures: density panels, overlay, drift arrows, compactness bar chart

#### Output files (per model)
| File | Description |
|------|-------------|
| `{prefix}_density.png/.pdf` | 4×4 grid of KDE density maps (16 baselines, Original as background contour) |
| `{prefix}_overlay.png/.pdf` | All techniques overlaid as smooth KDE contours |
| `{prefix}_arrows.png/.pdf` | Drift vectors (sampled subset, max 350 per technique) |
| `{prefix}_compactness.png/.pdf` | Mean distance from centroid per technique |
| `{prefix}.html` | HTML report with all figures |

#### CLI Usage
```bash
cd /d/dev/deid-toolkit_evaluators
conda run -n swinface python visualize_embedding_space.py \
  --aligned "D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/aligned/celeba-test" \
  --techniques-dir "D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/datasets" \
  --output embedding_projection_swinface.html

conda run -n transface python visualize_embedding_space.py \
  --aligned "D:/dev/deid-toolkit/root_dir/embeddings/TransFace/aligned/celeba-test" \
  --techniques-dir "D:/dev/deid-toolkit/root_dir/embeddings/TransFace/datasets" \
  --output embedding_projection_transface.html
```

---

## Design Decisions

### Color System (consistently applied across all figures)
- **Original**: Gray (#777777), grayscale colormap
- **16 baselines**: ColorBrewer palette with explicit override for FALCO
  - FALCO: `#17becf` (teal, PuBu colormap) — was yellow-orange (YlOrBr) before

### Figure 1: Density Panels (4×4 grid)
- Fixed axis limits: GLOBAL_XLIM/YLIM from aligned UMAP space (+10% margin)
- KDE density rendered on fixed grid matching bounds → density rectangle always fills the panel
- Original embedded as dashed gray contour in every panel (drawn BEFORE technique density)
- Technique name as colored inset label in top-left corner (no title, no image count)
- 3.4"×3.0" per subplot → tight packing for manuscript inclusion

### Figure 1b: Overlay
- KDE contour-only overlay with very low alpha fill (0.015) + solid contour lines
- Reference as faint gray dashed contour in background
- Legend at bottom-right

### Figure 2: Drift Arrows
- Up to 350 sampled arrows per technique (n_aligned//8, max 350)
- Technique density rendered with same fixed bounds
- Original contour in background (dashed gray)
- Arrows drawn from aligned → anonymized position

### Figure 3: Compactness
- Mean distance from centroid in UMAP 2D space (not original 512-d)
- Bar chart with error bars showing std deviation
- Technique names displayed as "Original" for the reference baseline

---

## Bug Fixes Applied

### Fix 1: gaussian_kde ValueError
**Symptom:** `ValueError: points have dimension 2, dataset has dimension 1`  
**Root cause:** `list(x) + list(y)` concatenates into flat array; scipy expects `(dim, n_points)` shape  
**Fix:** Use `np.vstack([x, y])` which produces correct `(2, N)` shape

### Fix 2: Original contour invisible
**Symptom:** Reference/Original dashed contour not visible in density panels  
**Root cause:** Two issues: (1) contour drawn AFTER imshow coverage, (2) too few levels (4), low alpha  
**Fix:** Draw contour BEFORE density; increase to 12 levels, colors="#555555", alpha=0.6, linewidths=1.0

### Fix 3: Density rectangle misaligned with axes
**Symptom:** KDE image doesn't fill the panel when axis limits are fixed  
**Root cause:** KDE grid spanned technique's data range, not global bounds  
**Fix:** Added `x_bounds`/`y_bounds` params to `_density_kde()`, pass GLOBAL_XLIM/YLIM

### Fix 4: PDF file locked (PermissionError)
**Symptom:** Scripts crash trying to write PDF while browser/File Explorer has it open  
**Root cause:** Windows file locking  
**Fix:** Added `_savefig_safe()` helper with retry loop (6 attempts, 0.3s delay between retries)

---

## Environment Setup

### Conda environments used
| Env | Python | numpy | umap-learn | scipy | matplotlib |
|-----|--------|-------|------------|-------|------------|
| swinface | 3.x | 2.2.6 | ✓ | 1.15.2 | ✓ |
| transface | 3.x | 2.2.6 | ✓ | 1.15.2 | ✓ |

### System Python NOT used
- numpy 2.5.1 → incompatible with numba (requires ≤2.4)
- pip installed umap via system Python → downgraded numpy to 2.4.6
- **Always use conda envs:** `conda run -n swinface ...` or `conda run -n transface ...`

### Known scipy version compatibility
- scipy 1.15.2: requires proper `(dim, n_points)` shape for gaussian_kde
- Always verify with: `conda run -n <env> python -c "from scipy.stats import gaussian_kde; ..."`

---

## File Structure

```
deid-toolkit_evaluators/
├── visualize_embedding_space.py   # Main UMAP visualization (4 figures)
├── batch_verify_all.py            # Batch ROC/AUC verification across techniques
├── verify_embeddings.py           # Single technique ROC/AUC from pairs files
├── analyze_identity_drift.py      # Identity preservation analysis
├── analyze_compactness_separation.py  # Compactness-separation analysis
└── analyze_retrieval.py           # Identity retrieval accuracy (k-NN)

Output files:
├── embedding_projection_swinface_*.png/.pdf
├── embedding_projection_transface_*.png/.pdf
├── embedding_projection_swinface.html
├── embedding_projection_transface.html
└── verification_*.html
```

---

## LaTeX Manuscript Notes

### Naming conventions in paper
- **"Original"** — not "Aligned" or "Reference"; refers to un-anonymized faces
- All technique names as-is (AIDPro, AMT-GAN, CLEANIR, etc.)
- Colormaps use sequential variants for Original: Greys
- Each baseline uses distinct colormap from its assigned palette color

### Figure captions (suggested)
- **Fig. X:** Per-baseline embedding density — 4×4 grid with uniform UMAP coordinates. Original distribution shown as dashed contour in each panel. Technique labels in inset corner.
- **Fig. Y:** KDE contour overlay of all techniques in unified embedding space.

---

## Old Files (Removed)
- `embedding_projection_*_scatter.png/.pdf` — from earlier 500-image sample runs using scatter plots (replaced by KDE density)

---

## Qualitative Grid Visualization

### Script: `visualize_qualitative_grid.py`

Generates a labelled image grid for manuscript inclusion comparing anonymized baseline outputs against Reference (aligned face images).

#### Usage
```bash
cd /d/dev/deid-toolkit_evaluators

conda run -n swinface python visualize_qualitative_grid.py ^
    --ref-dir     "D:/dev/deid-toolkit/root_dir/datasets/aligned/fri" ^
    --baselines   "D:/dev/deid-toolkit/root_dir/datasets" ^
    --subjects    SandiLjubic KristijanLenac ZigaEmersic MatejVitek PeterPeer VitomirStruc2 BlazMeden ^
    --orientation horizontal  | vertical ^
    --cell-size   256 256 ^
    --output      qualitative_comparison.pdf
```

- **Subjects**: full filenames without extension (no glob/wildcards). Must exist in aligned/fri/.
- **Orientation**: `horizontal` — Reference first + baselines rightward, subjects down; `vertical` — similar but taller layout.
- **Cell size**: default 256×256 px square images.
- **Reference column**: `aligned/fri/` is displayed as "Reference" and excluded from baseline list. The `original/` folder is explicitly skipped.
- **Output**: PNG or PDF (extension determines format).

#### How it works
1. Reads subject stems from `--ref-dir` (e.g., aligned/fri/)
2. For each baseline subdirectory in `--baselines`, checks for a `{baseline}/fri/` folder with matching filename stem (tries .png, .tiff, .bmp, .jpg in order)
3. Loads all images, resizes to cell-size, assembles into labelled grid canvas with PIL
4. Saves as PDF or PNG

#### Known baselines without data
- `CPP-DeID/fri/` — exists but empty (0 files)
- `deidentified/fri/` — exists but not a DEID baseline (skip)

---

## Old Files (Removed)
- `embedding_projection_*_scatter.png/.pdf` — from earlier 500-image sample runs using scatter plots (replaced by KDE density)


---


---

## Clean Mode (`--clean`)

Tightest-possible layout **for density panels only** (not overlay/drift/compactness). Removes all tick marks, axis labels, grid lines, and the suptitle from the figure. Writes a descriptive caption to a companion `.txt` file instead.

### Usage
```bash
# SwinFace clean mode
conda run -n swinface python visualize_embedding_space.py ^
    --aligned "D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/aligned/celeba-test" ^
    --techniques-dir "D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/datasets" ^
    --output embedding_projection_swinface_clean.html --clean

# TransFace clean mode
conda run -n transface python visualize_embedding_space.py ^
    --aligned "D:/dev/deid-toolkit/root_dir/embeddings/TransFace/aligned/celeba-test" ^
    --techniques-dir "D:/dev/deid-toolkit/root_dir/embeddings/TransFace/datasets" ^
    --output embedding_projection_transface_clean.html --clean
```

### What --clean does on density panels (Fig 1 only)

| Property | Normal mode | Clean mode |
|----------|-------------|------------|
| `figsize` | `(13.6, 12.0)` inches | `(10.4, 9.6)` inches |
| Ticks/labels | Full axis ticks + numbers | `set_xticks([])`, `set_yticks([])` |
| Grid | `alpha=0.15` | Removed |
| suptitle | Present (title text on image) | **Removed** — caption written to `.txt` file instead |
| Layout rect | `[0, 0, 1, 0.96]` | `[0, 0, 1, 0.99]` |

### Other figures (overlay / drift / compactness)
Unchanged regardless of `--clean`. They always use their normal size, labels, and layout.

### Caption file
Generated alongside the density panel: `{prefix}_clean_density_caption.txt` (e.g. `embedding_projection_swinface_clean_density_caption.txt`, 551 bytes). Contains a full manuscript-ready figure caption with all technical details (UMAP params, dataset name, N count, description of Original contour and density encoding).

