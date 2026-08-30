@echo off
echo ========================================
echo Verification — SwinFace ROC/AUC
echo ========================================
echo.

set EMBEDDINGS_DIR=D:\dev\deid-toolkit\root_dir\embeddings\SwinFace\aligned\celeba-test
set PAIRS_DIR=D:\dev\deid-toolkit\root_dir\datasets\pairs
set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR%"

python verify_embeddings.py ^
    --embeddings "%EMBEDDINGS_DIR%" ^
    --pairs "%PAIRS_DIR%" ^
    --output verification_swinface.html

echo.
pause
