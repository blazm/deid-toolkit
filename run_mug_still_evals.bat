@echo off

REM ============================================================
REM  Mug-still batch evaluation — ALL available models
REM  Categories: Verification + Image Quality + Data Utility
REM
REM  Usage:  run_mug_still_evals.bat
REM          (or pass technique(s) as arguments: run_mug_still_evals.bat blur pixelize)
REM  Requires: conda env "deid-toolkit" activated.
REM ============================================================

cd /d "%~dp0"

for %%I in (.) do set ROOT=%CD%

set DATASET=mug-still

if "%~1"=="" (
  set TECHS=blur pixelize
) else (
  set TECHS=%*
)

echo Selected techniques: %TECHS%
echo.

echo Using: python (from activated conda env)
python --version
echo.

set EVAL_DIR=%ROOT%\deid\evaluation
set ALIGNED=%ROOT%\root_dir\datasets\aligned\%DATASET%
set PAIRS_GENUINE=%ROOT%\root_dir\datasets\pairs\%DATASET%_genuine_pairs.txt
set PAIRS_IMPOSTOR=%ROOT%\root_dir\datasets\pairs\%DATASET%_impostor_pairs.txt
set LOGS=%ROOT%\root_dir\logs

if not exist "%ALIGNED%" (
  echo ERROR: Aligned dataset not found: %ALIGNED%
  pause
  exit /b 1
)

if not exist "%PAIRS_GENUINE%" (
  echo ERROR: Genuine pairs not found: %PAIRS_GENUINE%
  pause
  exit /b 1
)

if not exist "%LOGS%" mkdir "%LOGS%"
set PYTHONPATH=%EVAL_DIR%

REM Force CPU — sm_120 not supported by PyTorch < 2.7
REM Remove this once you upgrade: pip install torch --index-url https://download.pytorch.org/whl/cu128
set CUDA_VISIBLE_DEVICES=
set DEID_FORCE_CPU=1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

echo ============================================================
echo  Full evaluation — %DATASET%
echo  Dataset   : %DATASET%
echo  Started   : %date% %time%
echo ============================================================
echo.

for %%T in (%TECHS%) do (
  call :run_tech %%T
)

REM ================================================================
call :summary
pause
goto :eof

REM ================================================================
:run_tech
set TECH=%1
set DEID=%ROOT%\root_dir\datasets\deidentified\%TECH%\%DATASET%
set RESULTS=%ROOT%\root_dir\results\%TECH%\%DATASET%

if not exist "%DEID%" (
  echo.
  echo [SKIP] %TECH% — de-identified images not found: %DEID%
  echo        Run technique first, then re-run this script.
  echo.
  goto :eof
)

if not exist "%RESULTS%" mkdir "%RESULTS%"

echo ============================================================
echo  Technique: %TECH%
echo  Results  : %RESULTS%
echo ============================================================
echo.

call :eval_mse
call :eval_ssim
call :eval_arcface
call :eval_deepface_gender
call :eval_deepface_expression
call :eval_deepface_age
call :eval_deepface_race
call :eval_ediffiqa
call :eval_fid
call :eval_dan
call :eval_lpips
call :eval_swinface
call :eval_adaface_optimized
call :eval_adaface_iv
call :eval_deepface_vggface

echo === %TECH% complete ===
echo.
goto :eof

REM ---- individual eval subroutines ----

:eval_mse
echo [ 1/15] MSE               (~2   min) Image quality  ...
python -u "%EVAL_DIR%\mse.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\mse.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_ssim
echo [ 2/15] SSIM              (~3   min) Image quality  ...
python -u "%EVAL_DIR%\ssim.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\ssim.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_arcface
echo [ 3/15] ArcFace (ONNX)    (~24  min) Verification   ...
python -u "%EVAL_DIR%\arcface.py" ^
  "%ALIGNED%" "%DEID%" ^
  --genuine_pairs_filepath "%PAIRS_GENUINE%" ^
  --impostor_pairs_filepath "%PAIRS_IMPOSTOR%" ^
  --save_path "%RESULTS%\arcface.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_deepface_gender
echo [ 4/15] DeepFace Gender   (~5   min) Data utility   ...
python -u "%EVAL_DIR%\deepface_GD.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\deepface_gender.csv" ^
  --dir_to_log "%LOGS%"
goto :eof

:eval_deepface_expression
echo [ 5/15] DeepFace Expr     (~5   min) Data utility   ...
python -u "%EVAL_DIR%\deepface_expression.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\deepface_expression.csv" ^
  --dir_to_log "%LOGS%"
goto :eof

:eval_deepface_age
echo [ 6/15] DeepFace Age      (~5   min) Data utility   ...
python -u "%EVAL_DIR%\deepface_age.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\deepface_age.csv" ^
  --dir_to_log "%LOGS%"
goto :eof

:eval_deepface_race
echo [ 7/15] DeepFace Race     (~5   min) Data utility   ...
python -u "%EVAL_DIR%\deepface_race.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\deepface_race.csv" ^
  --dir_to_log "%LOGS%"
goto :eof

:eval_ediffiqa
echo [ 8/15] eDifFIQA          (~5   min) Image quality  ...
python -u "%EVAL_DIR%\ediffiqa.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\ediffiqa.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_fid
echo [ 9/15] FID               (~5   min) Image quality  ...
python -u "%EVAL_DIR%\FID.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\fid.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_dan
echo [10/15] DAN (AffecNet)    (~23  min) Data utility   ...
python -u "%EVAL_DIR%\dan.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\dan.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_lpips
echo [11/15] LPIPS (CPU)       (~15  min) Image quality  ...
python -u "%EVAL_DIR%\lpips.py" ^
  "%ALIGNED%" "%DEID%" ^
  --save_path "%RESULTS%\lpips.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_swinface
echo [12/15] SWINFace          (~50  min) Verification   ...
python -u "%EVAL_DIR%\swinface.py" ^
  "%ALIGNED%" "%DEID%" ^
  --genuine_pairs_filepath "%PAIRS_GENUINE%" ^
  --impostor_pairs_filepath "%PAIRS_IMPOSTOR%" ^
  --save_path "%RESULTS%\swinface.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_adaface_optimized
echo [13/15] AdaFace Opt       (~75  min, cached rerun fast) Verification ...
python -u "%EVAL_DIR%\adaface_optimized.py" ^
  "%ALIGNED%" "%DEID%" ^
  --genuine_pairs_filepath "%PAIRS_GENUINE%" ^
  --impostor_pairs_filepath "%PAIRS_IMPOSTOR%" ^
  --save_path "%RESULTS%\adaface_optimized.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

:eval_adaface_iv
echo [14/15] AdaFace IV        (~75  min) Verification   ...
python -u "%EVAL_DIR%\adaface_iv.py" ^
  "%ALIGNED%" "%DEID%" ^
  --genuine_pairs_filepath "%PAIRS_GENUINE%" ^
  --impostor_pairs_filepath "%PAIRS_IMPOSTOR%" ^
  --save_path "%RESULTS%\adaface_iv.csv" ^
  --dir_to_log "%LOGS%"
goto :eof

:eval_deepface_vggface
echo [15/15] DeepFace VGG-Face (~3+ hr, cached rerun fast) Verification ...
python -u "%EVAL_DIR%\deepface_vggface.py" ^
  "%ALIGNED%" "%DEID%" ^
  --genuine_pairs_filepath "%PAIRS_GENUINE%" ^
  --impostor_pairs_filepath "%PAIRS_IMPOSTOR%" ^
  --save_path "%RESULTS%\deepface_vggface.csv" ^
  --dir_to_log "%LOGS%" ^
  --root_dir "%ROOT%\root_dir"
goto :eof

REM ---- summary ----

:summary
echo ============================================================
echo  ALL DONE!
echo  Dataset   : %DATASET%
echo  Finished  : %date% %time%
echo ============================================================
echo.
for %%T in (%TECHS%) do (
  set RESULTS=%ROOT%\root_dir\results\%%T\%DATASET%
  echo  %%T:
  if exist "%RESULTS%" (
    dir "%RESULTS%\*.csv" /b 2>nul || echo  (no CSVs)
  ) else (
    echo  (no results dir — technique was skipped)
  )
  echo.
)
goto :eof
