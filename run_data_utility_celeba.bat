@echo off
REM =============================================================
REM  Data Utility — CelebA-test (blur, pixelize)
REM  DAN (expression), DeepFace gender/expression/age/race
REM  ~2824 images per technique. CPU for PyTorch scripts.
REM  Result format: results/{technique}/{dataset}/{eval}.csv
REM =============================================================
set DEID_FORCE_CPU=1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

echo.
echo ========================================
echo  Data Utility — CelebA-test
echo ========================================
echo Techniques: blur, pixelize
echo Images:     ~2824 per technique
echo Metrics:    DAN (expression), Gender, Expression, Age, Race
echo.

set ROOT=root_dir
set ALIGNED=%ROOT%/datasets/aligned/celeba-test
set LOGS=%ROOT%/logs

REM ---------------------------------------------------------------
REM  DAN — blur (PyTorch, expression/emotion via AffecNet)
REM ---------------------------------------------------------------
echo.
echo [1/10] DAN | blur
python deid/evaluation/dan.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --save_path "%ROOT%/results/blur/celeba-test/dan.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DAN — pixelize
REM ---------------------------------------------------------------
echo.
echo [2/10] DAN | pixelize
python deid/evaluation/dan.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/dan.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Gender — blur (TensorFlow)
REM ---------------------------------------------------------------
echo.
echo [3/10] DeepFace Gender | blur
python deid/evaluation/deepface_GD.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --save_path "%ROOT%/results/blur/celeba-test/deepface_gender.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Gender — pixelize
REM ---------------------------------------------------------------
echo.
echo [4/10] DeepFace Gender | pixelize
python deid/evaluation/deepface_GD.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/deepface_gender.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Expression — blur
REM ---------------------------------------------------------------
echo.
echo [5/10] DeepFace Expression | blur
python deid/evaluation/deepface_expression.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --save_path "%ROOT%/results/blur/celeba-test/deepface_expression.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Expression — pixelize
REM ---------------------------------------------------------------
echo.
echo [6/10] DeepFace Expression | pixelize
python deid/evaluation/deepface_expression.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/deepface_expression.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Age — blur
REM ---------------------------------------------------------------
echo.
echo [7/10] DeepFace Age | blur
python deid/evaluation/deepface_age.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --save_path "%ROOT%/results/blur/celeba-test/deepface_age.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Age — pixelize
REM ---------------------------------------------------------------
echo.
echo [8/10] DeepFace Age | pixelize
python deid/evaluation/deepface_age.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/deepface_age.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Race — blur
REM ---------------------------------------------------------------
echo.
echo [9/10] DeepFace Race | blur
python deid/evaluation/deepface_race.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --save_path "%ROOT%/results/blur/celeba-test/deepface_race.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  DeepFace Race — pixelize
REM ---------------------------------------------------------------
echo.
echo [10/10] DeepFace Race | pixelize
python deid/evaluation/deepface_race.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/deepface_race.csv" ^
    --dir_to_log "%LOGS%"

echo.
echo ========================================
echo  Data Utility — DONE
echo ========================================
echo Results: %ROOT%/results/{blur,pixelize}/celeba-test/*.csv
pause
