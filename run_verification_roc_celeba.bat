@echo off
REM =============================================================
REM  Verification + ROC Curves — CelebA-test (validation baseline)
REM  Runs ArcFace, AdaFace, SWINFace, DeepFace VGG-Face on
REM  aligned vs. aligned images, then plots all ROC curves.
REM
REM  Usage:  run_verification_roc_celeba.bat
REM  Requires: conda env "deid-toolkit" activated.
REM =============================================================

set DEID_FORCE_CPU=1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

echo.
echo ========================================
echo  Verification + ROC — CelebA-test (Validation)
echo ========================================
echo Models:     ArcFace, AdaFace, SWINFace, DeepFace VGG-Face
echo Mode:       Validation (aligned vs. aligned baseline)
echo Output:     root_dir/results/validation/celeba-test/
echo             root_dir/results/viz/roc_curves.pdf/.png
echo ========================================
echo.

set ROOT=root_dir
set ALIGNED=%ROOT%/datasets/aligned/celeba-test
set GENUINE=%ROOT%/datasets/pairs/celeba-test_genuine_pairs.txt
set IMPOSTOR=%ROOT%/datasets/pairs/celeba-test_impostor_pairs.txt
set SAVE_DIR=%ROOT%/results/validation/celeba-test
set LOGS=%ROOT%/logs

if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM ---------------------------------------------------------------
REM  ArcFace — ONNX, fastest (~1 min / 8226 pairs on CPU)
REM ---------------------------------------------------------------
echo.
echo [1/4] ArcFace (ONNX) ...
python -u deid/evaluation/arcface.py ^
    "%ALIGNED%" ^
    "%ALIGNED%" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%SAVE_DIR%/arcface.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  AdaFace Optimized — caches embeddings (~75 min first run, fast on rerun)
REM ---------------------------------------------------------------
echo.
echo [2/4] AdaFace Optimized ...
python -u deid/evaluation/adaface_optimized.py ^
    "%ALIGNED%" ^
    "%ALIGNED%" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%SAVE_DIR%/adaface_optimized.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  SWINFace — caches embeddings (~50 min first run, fast on rerun)
REM ---------------------------------------------------------------
echo.
echo [3/4] SWINFace ...
python -u deid/evaluation/swinface.py ^
    "%ALIGNED%" ^
    "%ALIGNED%" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%SAVE_DIR%/swinface.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  DeepFace VGG-Face — TensorFlow (~3+ hr first run, fast on rerun)
REM ---------------------------------------------------------------
echo.
echo [4/4] DeepFace VGG-Face ...
python -u deid/evaluation/deepface_vggface.py ^
    "%ALIGNED%" ^
    "%ALIGNED%" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%SAVE_DIR%/deepface_vggface.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  Generate ROC curve plots (all models on one figure)
REM ---------------------------------------------------------------
echo.
echo ========================================
echo  Generating ROC curves ...
echo ========================================
echo.

python -u -c "^
import matplotlib; matplotlib.use('Agg');^
import matplotlib.pyplot as plt;^
import numpy as np, pandas as pd, os, glob;^
save_dir = '%SAVE_DIR%';^
fig, ax = plt.subplots(figsize=(8, 8));^
for csv_path in sorted(glob.glob(os.path.join(save_dir, '*.csv'))):^
    df = pd.read_csv(csv_path);^
    if 'ground_truth' not in df.columns: continue;^
    score_col = next((c for c in ['cossim','score','similarity'] if c in df.columns), None);^
    if score_col is None: continue;^
    scores, labels = df[score_col].values, df['ground_truth'].values.astype(int);^
    idx = np.argsort(scores)[::-1];^
    sl = labels[idx];^
    tp, tn = sl.sum(), len(sl) - sl.sum();^
    if tp == 0 or tn == 0: continue;^
    tpr, fpr = [], [];^
    t, f = 0.0, 0.0;^
    for i in range(len(sl)):^
        if sl[i] == 1: t += 1.0/tp; else: f += 1.0/tn;^
        tpr.append(t); fpr.append(f);^
    fpr = [0.0] + fpr; tpr = [0.0] + tpr;^
    auc = float(-(np.array(fpr)[1:]-np.array(fpr)[:-1]).sum() * ((np.array(tpr)[1:]+np.array(tpr)[:-1])/2).sum());^
    label = os.path.basename(csv_path).replace('.csv','').replace('_optimized','');^
    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC={auc:.4f})');^
ax.plot([0,1],[0,1],'k--',alpha=0.3,label='Random');^
ax.set_xlabel('False Positive Rate', fontsize=13);^
ax.set_ylabel('True Positive Rate', fontsize=13);^
ax.set_title('ROC Curves — CelebA-test (Validation Baseline)', fontsize=14, fontweight='bold');^
ax.legend(frameon=True, fontsize=11);^
ax.grid(True, alpha=0.3);^
viz_dir = '%ROOT%/results/viz'; os.makedirs(viz_dir, exist_ok=True);^
fig.savefig(os.path.join(viz_dir, 'roc_validation_celeba.pdf'), dpi=200);^
fig.savefig(os.path.join(viz_dir, 'roc_validation_celeba.png'), dpi=200);^
plt.close(); print(f'ROC curves saved to {viz_dir}/roc_validation_celeba.pdf + .png');^
"

echo.
echo ========================================
echo  Validation + ROC — DONE
echo ========================================
echo Results:   %SAVE_DIR%/*.csv
echo ROC plots: %ROOT%/results/viz/roc_validation_celeba.{pdf,png}
pause
