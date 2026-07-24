@echo off
REM =============================================================
REM  Image Quality — CelebA-test (blur, pixelize)
REM  SSIM, MS-SSIM, LPIPS, MSE, FID, eDifFIQA
REM  ~2824 image pairs per technique. CPU for PyTorch scripts.
REM  Result format: results/{technique}/{dataset}/{eval}.csv
REM =============================================================
set DEID_FORCE_CPU=1

echo.
echo ========================================
echo  Image Quality — CelebA-test
echo ========================================
echo Techniques: blur, pixelize
echo Images:     ~2824 per technique
echo Metrics:    SSIM, MS-SSIM, LPIPS, MSE, FID, eDifFIQA
echo.

set ROOT=root_dir
set ALIGNED=%ROOT%/datasets/aligned/celeba-test_aligned
set LOGS=%ROOT%/logs

REM ---------------------------------------------------------------
REM  SSIM — blur (outputs: ssim.csv + msssim.csv)
REM ---------------------------------------------------------------
echo.
echo [1/12] SSIM | blur
python deid/evaluation/ssim.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test_aligned" ^
    --save_path "%ROOT%/results/blur/celeba-test_aligned/ssim.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  SSIM — pixelize
REM ---------------------------------------------------------------
echo.
echo [2/12] SSIM | pixelize
python deid/evaluation/ssim.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test_aligned" ^
    --save_path "%ROOT%/results/pixelize/celeba-test_aligned/ssim.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  LPIPS — blur (CPU only by default)
REM ---------------------------------------------------------------
echo.
echo [3/12] LPIPS | blur
python deid/evaluation/lpips.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test_aligned" ^
    --save_path "%ROOT%/results/blur/celeba-test_aligned/lpips.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  LPIPS — pixelize
REM ---------------------------------------------------------------
echo.
echo [4/12] LPIPS | pixelize
python deid/evaluation/lpips.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test_aligned" ^
    --save_path "%ROOT%/results/pixelize/celeba-test_aligned/lpips.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  MSE — blur
REM ---------------------------------------------------------------
echo.
echo [5/12] MSE | blur
python deid/evaluation/mse.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test_aligned" ^
    --save_path "%ROOT%/results/blur/celeba-test_aligned/mse.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  MSE — pixelize
REM ---------------------------------------------------------------
echo.
echo [6/12] MSE | pixelize
python deid/evaluation/mse.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test_aligned" ^
    --save_path "%ROOT%/results/pixelize/celeba-test_aligned/mse.csv" ^
    --dir_to_log "%LOGS%"

REM ---------------------------------------------------------------
REM  FID — blur (distribution-level, single score)
REM ---------------------------------------------------------------
echo.
echo [7/12] FID | blur
python deid/evaluation/FID.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test_aligned" ^
    --save_path "%ROOT%/results/blur/celeba-test_aligned/fid.csv"

REM ---------------------------------------------------------------
REM  FID — pixelize
REM ---------------------------------------------------------------
echo.
echo [8/12] FID | pixelize
python deid/evaluation/FID.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test_aligned" ^
    --save_path "%ROOT%/results/pixelize/celeba-test_aligned/fid.csv"

REM ---------------------------------------------------------------
REM  eDifFIQA — blur (3 CSVs: _aligned, _deid, _delta)
REM ---------------------------------------------------------------
echo.
echo [9/12] eDifFIQA | blur
python deid/evaluation/ediffiqa.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test_aligned" ^
    --save_path "%ROOT%/results/blur/celeba-test_aligned/ediffiqa.csv" ^
    --eval_package_dir "deid/evaluation"

REM ---------------------------------------------------------------
REM  eDifFIQA — pixelize
REM ---------------------------------------------------------------
echo.
echo [10/12] eDifFIQA | pixelize
python deid/evaluation/ediffiqa.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test_aligned" ^
    --save_path "%ROOT%/results/pixelize/celeba-test_aligned/ediffiqa.csv" ^
    --eval_package_dir "deid/evaluation"

REM ---------------------------------------------------------------
REM  Validation baselines — aligned vs. aligned (reference quality)
REM ---------------------------------------------------------------
echo.
echo [11/12] Validation SSIM (aligned vs. aligned)
python deid/evaluation/ssim.py ^
    "%ALIGNED%" ^
    "%ALIGNED%" ^
    --save_path "%ROOT%/results/validation/celeba-test_aligned/ssim.csv" ^
    --dir_to_log "%LOGS%"

echo.
echo [12/12] Validation LPIPS (aligned vs. aligned)
python deid/evaluation/lpips.py ^
    "%ALIGNED%" ^
    "%ALIGNED%" ^
    --save_path "%ROOT%/results/validation/celeba-test_aligned/lpips.csv" ^
    --dir_to_log "%LOGS%"

echo.
echo ========================================
echo  Image Quality — DONE
echo ========================================
echo Results: %ROOT%/results/{blur,pixelize}/celeba-test_aligned/*.csv
pause
