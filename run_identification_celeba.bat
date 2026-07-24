@echo off
REM =============================================================
REM  Identification + CMC Curves — CelebA-test (validation baseline)
REM  Runs gallery search using cached embeddings from verification.
REM  Models: ArcFace, AdaFace, SWINFace, DeepFace VGG-Face
REM
REM  Usage:  run_identification_celeba.bat
REM  Requires: conda env "deid-toolkit" activated.
REM            Run verification first (caches .pkl embeddings).
REM =============================================================

set DEID_FORCE_CPU=1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

echo.
echo ========================================
echo  Identification + CMC — CelebA-test (Validation)
echo ========================================
echo Models:     ArcFace, AdaFace, SWINFace, DeepFace VGG-Face
echo Labels:     celeba-test_labels.csv (579 identities)
echo Mode:       Validation (aligned vs aligned gallery search)
echo Output:     root_dir/results/validation/celeba-test/identification_{model}.csv
echo             root_dir/results/viz/cmc_validation_celeba.{pdf,png}
echo ========================================
echo.

set ROOT=root_dir
set ALIGNED=%ROOT%/datasets/aligned/celeba-test
set LABELS=%ROOT%/datasets/labels/celeba-test_labels.csv
set SAVE_DIR=%ROOT%/results/validation/celeba-test
set LOGS=%ROOT%/logs
set MODELS=arcface adaface_optimized swinface deepface_vggface

if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM ---------------------------------------------------------------
REM  Run identification for each model (reads cached .pkl embeddings)
REM ---------------------------------------------------------------
echo [1/4] SWINFace identification ...
python -u deid/evaluation/identification.py ^
    "%ALIGNED%" "%ALIGNED%" ^
    --save_path "%SAVE_DIR%/identification_swinface.csv" ^
    --root_dir "%ROOT%" ^
    --model swinface ^
    --labels_path "%LABELS%" ^
    --gallery_ratio 0.5

echo.
echo [2/4] AdaFace identification ...
python -u deid/evaluation/identification.py ^
    "%ALIGNED%" "%ALIGNED%" ^
    --save_path "%SAVE_DIR%/identification_adaface_optimized.csv" ^
    --root_dir "%ROOT%" ^
    --model adaface_optimized ^
    --labels_path "%LABELS%" ^
    --gallery_ratio 0.5

echo.
echo [3/4] ArcFace identification ...
python -u deid/evaluation/identification.py ^
    "%ALIGNED%" "%ALIGNED%" ^
    --save_path "%SAVE_DIR%/identification_arcface.csv" ^
    --root_dir "%ROOT%" ^
    --model arcface ^
    --labels_path "%LABELS%" ^
    --gallery_ratio 0.5

echo.
echo [4/4] DeepFace VGG-Face identification ...
python -u deid/evaluation/identification.py ^
    "%ALIGNED%" "%ALIGNED%" ^
    --save_path "%SAVE_DIR%/identification_deepface_vggface.csv" ^
    --root_dir "%ROOT%" ^
    --model deepface_vggface ^
    --labels_path "%LABELS%" ^
    --gallery_ratio 0.5

REM ---------------------------------------------------------------
REM  Generate combined CMC curve plot (all models on one figure)
REM ---------------------------------------------------------------
echo.
echo ========================================
echo  Generating CMC curves ...
echo ========================================
echo.

python -u -c "^
import matplotlib; matplotlib.use('Agg');^
import matplotlib.pyplot as plt, pandas as pd, os, glob, numpy as np;^
save_dir = '%SAVE_DIR%';^
fig, ax = plt.subplots(figsize=(9, 7));^
colors = {'swinface':'#1f77b4','adaface_optimized':'#ff7f0e','arcface':'#2ca02c','deepface_vggface':'#d62728'};^
for csv_path in sorted(glob.glob(os.path.join(save_dir, 'identification_*.csv'))):^
    df = pd.read_csv(csv_path);^
    total = len(df);^
    if total == 0: continue;^
    rank_cols = [c for c in df.columns if c.startswith('rank') and c.endswith('_correct')];^
    if not rank_cols: continue;^
    ranks = sorted([int(c.split('_')[0].replace('rank','')) for c in rank_cols]);^
    cmc = np.array([df[f'rank{k}_correct'].sum()/total for k in ranks]);^
    model_name = os.path.basename(csv_path).replace('identification_','').replace('.csv','');^
    label = model_name.replace('_optimized','').replace('deepface_vggface','DeepFace VGG-Face').replace('arcface','ArcFace').replace('swinface','SWINFace');^
    ax.plot(ranks, cmc, 'o-', linewidth=2, markersize=4, label=f'{label} (R@1={cmc[0]:.1%})', color=colors.get(model_name,'#333'));^
ax.set_xlabel('Rank', fontsize=13);^
ax.set_ylabel('Cumulative Match Rate', fontsize=13);^
ax.set_title('CMC Curves — CelebA-test (Validation Baseline)', fontsize=14, fontweight='bold');^
ax.legend(frameon=True, fontsize=10, loc='lower right');^
ax.grid(True, alpha=0.3);^
ax.set_xlim(left=0); ax.set_ylim(bottom=0, top=1.05);^
viz_dir = '%ROOT%/results/viz'; os.makedirs(viz_dir, exist_ok=True);^
fig.savefig(os.path.join(viz_dir, 'cmc_validation_celeba.pdf'), dpi=200);^
fig.savefig(os.path.join(viz_dir, 'cmc_validation_celeba.png'), dpi=200);^
plt.close(); print(f'CMC curves saved to {viz_dir}/cmc_validation_celeba.pdf + .png');^
"

echo.
echo ========================================
echo  Identification + CMC — DONE
echo ========================================
echo Results:   %SAVE_DIR%/identification_*.csv
echo CMC plots: %ROOT%/results/viz/cmc_validation_celeba.{pdf,png}
pause
