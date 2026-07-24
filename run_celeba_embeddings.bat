@echo off
REM ============================================================
REM  CelebA Test — Embedding Space Analysis Pipeline
REM  Datasets: celeba-test (2824 images, 579 identities)
REM  Techniques: blur, pixelize
REM  Model: SWINFace (optimized for embedding space viz)
REM
REM  Usage:  run_celeba_embeddings.bat
REM  Requires: conda env "deid-toolkit" activated.
REM ============================================================

cd /d "%~dp0"

for %%I in (.) do set ROOT=%CD%

set DATASET=celeba-test
set TECHS=blur pixelize

echo Selected dataset: %DATASET%
echo Techniques: %TECHS%
echo.

echo Using: python (from activated conda env)
python --version
echo.

set EVAL_DIR=%ROOT%\deid\evaluation
set ALIGNED=%ROOT%\root_dir\datasets\aligned\%DATASET%
set LABELS=%ROOT%\root_dir\datasets\labels
set PAIRS=%ROOT%\root_dir\datasets\pairs
set LOGS=%ROOT%\root_dir\logs

REM Force CPU — sm_120 not supported by PyTorch ^< 2.7
set CUDA_VISIBLE_DEVICES=
set DEID_FORCE_CPU=1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

REM ============================================================
REM  Step 1: Generate genuine/impostor pair files
REM ============================================================
echo ============================================================
echo  STEP 1: Generating pair files for %DATASET%
echo ============================================================
echo.

python -u -c "import sys, os; from deid.utils import generate_img_pairs_all as gp; gp.main([sys.argv[1]], sys.argv[2].replace(chr(92), chr(47)), sys.argv[3].replace(chr(92), chr(47)))" %DATASET% "%LABELS%" "%PAIRS%"

if exist "%PAIRS%\%DATASET%_genuine_pairs.txt" (
  echo. [OK] Genuine pairs created.
) else (
  echo. [ERROR] Pair generation failed. Check label CSV format.
  pause
  exit /b 1
)

REM Count pairs for info
for %%F in ("%PAIRS%\%DATASET%_genuine_pairs.txt") do set SIZE=%%~zF
echo. [INFO] Genuine pair file size: %SIZE% bytes

REM ============================================================
REM  Step 2: Run de-identification techniques
REM ============================================================
echo.
echo ============================================================
echo  STEP 2: Running de-identification techniques
echo ============================================================
echo.

for %%T in (%TECHS%) do (
  call :run_technique %%T
)

REM ============================================================
REM  Step 3: Run SWINFace evaluation (caches embeddings)
REM ============================================================
echo.
echo ============================================================
echo  STEP 3: Running SWINFace for embedding caching
echo  NOTE: This caches embeddings required for visualization.
echo        ~50 min per technique on CPU (2824 images x 2 sets).
echo ============================================================
echo.

for %%T in (%TECHS%) do (
  call :run_swinface %%T
)

REM ============================================================
REM  Step 4: Generate embedding visualizations
REM ============================================================
echo.
echo ============================================================
echo  STEP 4: Generating embedding space visualizations
echo ============================================================
echo.

python -m deid.explore.embedding_viz_cli ^
  --dataset %DATASET% ^
  --model swinface ^
  --techniques %TECHS% ^
  --root-dir root_dir ^
  --method umap

echo.
echo ============================================================
echo  ALL DONE!
echo  Dataset   : %DATASET%
echo  Finished  : %date% %time%
echo ============================================================
echo.
for %%T in (%TECHS%) do (
  echo  %%T SWINFace results:
  if exist "%ROOT%\root_dir\results\%%T\%DATASET%" (
    dir "%ROOT%\root_dir\results\%%T\%DATASET%\*.csv" /b 2>nul || echo  (no CSVs)
  ) else (
    echo  (no results — SWINFace may still be running or failed)
  )
  echo.
)

echo  Visualization outputs:
  dir "%ROOT%\root_dir\results\viz\%DATASET%_swinface\" /b 2>nul || echo  (no viz files yet)
echo.

pause
goto :eof

REM ---- subroutines ----

:run_technique
set TECH=%1
set DEID=%ROOT%\root_dir\datasets\deidentified\%TECH%\%DATASET%

if exist "%DEID%" (
  echo [SKIP] Technique %TECH% — already processed.
  goto :eof
)

echo.
echo [RUN] Technique: %TECH% ...

python -u "%ROOT%\deid\techniques\%TECH%.py" ^
  "%ALIGNED%" "%DEID%"

if exist "%DEID%" (
  echo [OK] %TECH% complete.
) else (
  echo [WARN] %TECH% output dir missing — check technique script.
)
goto :eof

:run_swinface
set TECH=%1
set DEID=%ROOT%\root_dir\datasets\deidentified\%TECH%\%DATASET%
set RESULTS=%ROOT%\root_dir\results\%TECH%\%DATASET%
set SAVE=%RESULTS%\swinface.csv

if exist "%SAVE%" (
  echo.
  echo [SKIP] SWINFace %TECH% — results already exist: %SAVE%
  REM Check if cache dirs exist too
  if not exist "%ROOT%\root_dir\preprocess\temp\swinface" (
    echo [WARN] Results CSV exists but cache dir missing. Re-run for embeddings.
  )
  goto :eof
)

echo.
echo [RUN] SWINFace: %TECH% (~50 min on CPU / %DATASET%) ...

if not exist "%RESULTS%" mkdir "%RESULTS%"

set PYTHONPATH=%EVAL_DIR%

python -u "%EVAL_DIR%\swinface.py" ^
  "%ALIGNED%" "%DEID%" ^
  --genuine_pairs_filepath "%PAIRS%\%DATASET%_genuine_pairs.txt" ^
  --impostor_pairs_filepath "%PAIRS%\%DATASET%_impostor_pairs.txt" ^
  --save_path "%SAVE%" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"

if exist "%SAVE%" (
  echo [OK] SWINFace %TECH% complete.
) else (
  echo [WARN] SWINFace %TECH% finished but no results CSV found.
)
goto :eof
