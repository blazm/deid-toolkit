@echo off
echo ========================================
echo Retrieval Accuracy — SwinFace
echo ========================================
echo.

set ALIGNED_DIR=D:\dev\deid-toolkit\root_dir\embeddings\SwinFace\aligned\celeba-test
set TECHNIQUES_DIR=D:\dev\deid-toolkit\root_dir\embeddings\SwinFace\datasets
set SCRIPT_DIR=%~dp0

call conda activate swinface
if errorlevel 1 (
    echo ERROR: Failed to activate 'swinface' conda environment.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

python analyze_retrieval.py ^
    --aligned "%ALIGNED_DIR%" ^
    --techniques-dir "%TECHNIQUES_DIR%" ^
    --output retrieval_analysis_swinface.html ^
    --sample 1000

echo.
pause