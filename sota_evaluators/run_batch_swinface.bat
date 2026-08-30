@echo off
echo ========================================
echo SwinFace Batch Embedding Generation
echo ========================================
echo.

set DATASETS_DIR=D:\dev\deid-toolkit\root_dir\datasets
set OUTPUT_ROOT=D:\dev\deid-toolkit\root_dir\embeddings\SwinFace
set SCRIPT_DIR=%~dp0

call conda activate swinface
if errorlevel 1 (
    echo ERROR: Failed to activate 'swinface' conda environment.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

python batch_run_swinface.py ^
    --datasets-dir "%DATASETS_DIR%" ^
    --output-root "%OUTPUT_ROOT%"

echo.
pause
