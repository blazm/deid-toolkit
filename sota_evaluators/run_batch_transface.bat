@echo off
echo ========================================
echo TransFace Batch Embedding Generation
echo ========================================
echo.

set DATASETS_DIR=D:\dev\deid-toolkit\root_dir\datasets
set OUTPUT_ROOT=D:\dev\deid-toolkit\root_dir\embeddings\TransFace
set SCRIPT_DIR=%~dp0

call conda activate transface
if errorlevel 1 (
    echo ERROR: Failed to activate 'transface' conda environment.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

python batch_run_transface.py ^
    --datasets-dir "%DATASETS_DIR%" ^
    --output-root "%OUTPUT_ROOT%"

echo.
pause
