@echo off
echo ========================================
echo Batch Verification — TransFace (All Techniques)
echo ========================================
echo.

set EMBEDDINGS_ROOT=D:\dev\deid-toolkit\root_dir\embeddings\TransFace
set PAIRS_DIR=D:\dev\deid-toolkit\root_dir\datasets\pairs
set SCRIPT_DIR=%~dp0

call conda activate transface
if errorlevel 1 (
    echo ERROR: Failed to activate 'transface' conda environment.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

python batch_verify_all.py ^
    --embeddings-root "%EMBEDDINGS_ROOT%" ^
    --pairs "%PAIRS_DIR%" ^
    --output verification_transface_all.html

echo.
pause
