@echo off
echo ========================================
echo Identity Drift Analysis — TransFace
echo ========================================
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

python analyze_identity_drift.py ^
    --aligned "%ALIGNED_DIR%" ^
    --techniques-dir "%TECHNIQUES_DIR%" ^
    --output drift_analysis_transface.html

echo.
pause