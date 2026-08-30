@echo off
echo ========================================
echo SwinFace Attribute Extractor
echo (Age, Gender, Expression)
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

python extract_attributes_swinface.py ^
    --input "%INPUT_DIR%" ^
    --output "%OUTPUT_DIR%\attributes.csv"

echo.
pause
