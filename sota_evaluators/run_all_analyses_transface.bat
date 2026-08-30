@echo off
echo ============================================
echo Run All Visualizations — TransFace
echo ============================================
echo.
echo This runs all 4 analysis scripts sequentially.
echo Results will be saved as HTML + PNG files.
echo.

set ALIGNED_DIR=D:\dev\deid-toolkit\root_dir\embeddings\TransFace\aligned\celeba-test
set TECHNIQUES_DIR=D:\dev\deid-toolkit\root_dir\embeddings\TransFace\datasets
set SCRIPT_DIR=%~dp0

call conda activate transface
if errorlevel 1 (
    echo ERROR: Failed to activate 'transface' conda environment.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

echo [1/4] Identity Drift Analysis...
python analyze_identity_drift.py ^
    --aligned "%ALIGNED_DIR%" ^
    --techniques-dir "%TECHNIQUES_DIR%" ^
    --output drift_analysis_transface.html
echo.

echo [2/4] Embedding Space Projection (UMAP)...
python visualize_embedding_space.py ^
    --aligned "%ALIGNED_DIR%" ^
    --techniques-dir "%TECHNIQUES_DIR%" ^
    --output embedding_projection_transface.html ^
    --sample 500
echo.

echo [3/4] Retrieval Accuracy...
python analyze_retrieval.py ^
    --aligned "%ALIGNED_DIR%" ^
    --techniques-dir "%TECHNIQUES_DIR%" ^
    --output retrieval_analysis_transface.html ^
    --sample 1000
echo.

echo [4/4] Compactness-Separation Analysis...
python analyze_compactness_separation.py ^
    --aligned "%ALIGNED_DIR%" ^
    --techniques-dir "%TECHNIQUES_DIR%" ^
    --output compactness_separation_transface.html
echo.

echo ============================================
echo All TransFace analyses complete!
echo Open the HTML files in your browser.
echo ============================================
pause