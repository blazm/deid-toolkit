@echo off
REM ============================================================
REM  Embedding Space Analysis — Generate visualization PDFs/PNGs
REM
REM  Requires: conda env "deid-toolkit" activated, SWINFace cache exists.
REM            Run the evaluation batch first (run_mug_still_evals.bat).
REM
REM  Usage:  run_embedding_viz.bat              (auto-discovers techniques)
REM          run_embedding_viz.bat blur pixelize (specific techniques)
REM ============================================================

cd /d "%~dp0"

echo Using: python (from activated conda env)
python --version
echo.

if "%~1"=="" (
  echo Auto-discovering techniques from cache...
  echo.
  python -m deid.explore.embedding_viz_cli ^
    --dataset mug-still --model swinface --root-dir root_dir
) else (
  echo Techniques: %*
  echo.
  python -m deid.explore.embedding_viz_cli ^
    --dataset mug-still --model swinface --root-dir root_dir ^
    --techniques %*
)

echo.
pause
