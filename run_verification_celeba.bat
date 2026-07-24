@echo off
REM =============================================================
REM  Verification + Embeddings — CelebA-test (blur, pixelize)
REM  ArcFace, AdaFace Optimized, SWINFace (+ embedding viz)
REM  ~8226 pairs per model. CPU only (DEID_FORCE_CPU=1).
REM  Result format: results/{technique}/{dataset}/{eval}.csv
REM =============================================================
set DEID_FORCE_CPU=1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

echo.
echo ========================================
echo  Verification + Embeddings — CelebA-test
echo ========================================
echo Techniques: blur, pixelize
echo Pairs:      8226 (4113 genuine + 4113 impostor)
echo Models:     ArcFace, AdaFace Optimized, SWINFace
echo.

set ROOT=root_dir
set ALIGNED=%ROOT%/datasets/aligned/celeba-test
set GENUINE=%ROOT%/datasets/pairs/celeba-test_genuine_pairs.txt
set IMPOSTOR=%ROOT%/datasets/pairs/celeba-test_impostor_pairs.txt
set LOGS=%ROOT%/logs

REM ---------------------------------------------------------------
REM  ArcFace — blur (ONNX, no caching, fastest)
REM ---------------------------------------------------------------
echo.
echo [1/6] ArcFace | blur
python deid/evaluation/arcface.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%ROOT%/results/blur/celeba-test/arcface.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  ArcFace — pixelize
REM ---------------------------------------------------------------
echo.
echo [2/6] ArcFace | pixelize
python deid/evaluation/arcface.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/arcface.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  AdaFace Optimized — blur (caches embeddings)
REM ---------------------------------------------------------------
echo.
echo [3/6] AdaFace Optimized | blur
python deid/evaluation/adaface_optimized.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%ROOT%/results/blur/celeba-test/adaface_optimized.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  AdaFace Optimized — pixelize (caches embeddings)
REM ---------------------------------------------------------------
echo.
echo [4/6] AdaFace Optimized | pixelize
python deid/evaluation/adaface_optimized.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/adaface_optimized.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  SWINFace — blur (caches embeddings)
REM ---------------------------------------------------------------
echo.
echo [5/6] SWINFace | blur
python deid/evaluation/swinface.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/blur/celeba-test" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%ROOT%/results/blur/celeba-test/swinface.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  SWINFace — pixelize (caches embeddings)
REM ---------------------------------------------------------------
echo.
echo [6/6] SWINFace | pixelize
python deid/evaluation/swinface.py ^
    "%ALIGNED%" ^
    "%ROOT%/datasets/deidentified/pixelize/celeba-test" ^
    --genuine_pairs_filepath "%GENUINE%" ^
    --impostor_pairs_filepath "%IMPOSTOR%" ^
    --save_path "%ROOT%/results/pixelize/celeba-test/swinface.csv" ^
    --dir_to_log "%LOGS%" ^
    --root_dir "%ROOT%"

REM ---------------------------------------------------------------
REM  Embedding Visualization — SWINFace (cached embeddings)
REM ---------------------------------------------------------------
echo.
echo [7a] Generating embedding visualizations (SWINFace)...
python -m deid.explore.embedding_viz_cli ^
    --dataset celeba-test ^
    --model swinface ^
    --techniques blur pixelize ^
    --method umap

REM ---------------------------------------------------------------
REM  Embedding Visualization — AdaFace (cached embeddings)
REM ---------------------------------------------------------------
echo.
echo [7b] Generating embedding visualizations (AdaFace)...
python -m deid.explore.embedding_viz_cli ^
    --dataset celeba-test ^
    --model adaface ^
    --techniques blur pixelize ^
    --method umap

echo.
echo ========================================
echo  Verification + Embeddings — DONE
echo ========================================
echo Results:    %ROOT%/results/{blur,pixelize}/celeba-test/*.csv
echo Embeddings: %ROOT%/preprocess/temp/{swinface,adaface}/
echo Visuals:    %ROOT%/results/viz/celeba-test_{swinface,adaface}/
pause
