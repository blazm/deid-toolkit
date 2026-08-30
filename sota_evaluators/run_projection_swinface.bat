@echo off
echo ========================================
echo Embedding Space Projection — SwinFace
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

python visualize_embedding_space.py ^
    --aligned "%ALIGNED_DIR%" ^
    --techniques-dir "%TECHNIQUES_DIR%" ^
    --output embedding_projection_swinface.html

echo.
pause