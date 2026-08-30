@echo off
echo ========================================
echo SwinFace Embedding Generator
echo ========================================
echo.

set INPUT_DIR=D:\dev\deid-toolkit\root_dir\datasets\FALCO\celeba-test
set OUTPUT_DIR=D:\dev\deid-toolkit\root_dir\embeddings\SwinFace\datasets\FALCO
set SCRIPT_DIR=%~dp0

call conda activate swinface
if errorlevel 1 (
    echo ERROR: Failed to activate 'swinface' conda environment.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

REM Note: script auto-appends last 2 input folder levels, so output will be:
REM   %OUTPUT_DIR%\aligned\celeba-test\*.npy

python generate_embeddings_swinface.py ^
    --input "%INPUT_DIR%" ^
    --output "%OUTPUT_DIR%"

echo.
pause
