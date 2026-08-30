@echo off
echo ========================================
echo TransFace Embedding Generator
echo ========================================
echo.

set INPUT_DIR=D:\dev\deid-toolkit\root_dir\datasets\FALCO\celeba-test
set OUTPUT_DIR=D:\dev\deid-toolkit\root_dir\embeddings\TransFace\datasets
set SCRIPT_DIR=%~dp0

call conda activate transface
if errorlevel 1 (
    echo ERROR: Failed to activate 'transface' conda environment.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

REM Note: script auto-appends last 2 input folder levels, so output will be:
REM   %OUTPUT_DIR%\aligned\celeba-test\*.npy

python generate_embeddings_transface.py ^
    --input "%INPUT_DIR%" ^
    --output "%OUTPUT_DIR%" ^
    --network vit_l_dp005_mask_005 ^
    --weight models\transface\glint360k_model_TransFace_L.pt

echo.
pause
