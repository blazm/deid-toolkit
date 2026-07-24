@echo off
REM =============================================================
REM  AdaFace Optimized — CelebA-test Dataset (Full Run)
REM  ~8226 pairs (4113 genuine + 4113 impostor), ~30-35 min on CPU.
REM  Techniques: blur, pixelize
REM  Outputs: verification CSVs + embedding cache + viz figures
REM =============================================================

set DEID_FORCE_CPU=1

echo.
echo ========================================
echo  AdaFace Optimized — CelebA-test Run
echo ========================================
echo Pairs:     ~8226 (4113 genuine + 4113 impostor)
echo ETA:       ~30-35 min per technique (CPU, first run)
echo Reruns:    Faster (AdaFace caches features as .pkl)
echo.

REM --- Step 1: AdaFace verification (blur) ---
echo [1/2] AdaFace verification: blur | celeba-test_aligned
python -m deid.evaluation.adaface_optimized ^
    "root_dir/datasets/aligned/celeba-test_aligned" ^
    "root_dir/datasets/deidentified/blur/celeba-test_aligned" ^
    --dataset_name celeba-test_aligned ^
    --technique_name blur ^
    --genuine_pairs_filepath "root_dir/datasets/pairs/celeba-test_aligned_genuine_pairs.txt" ^
    --impostor_pairs_filepath "root_dir/datasets/pairs/celeba-test_aligned_impostor_pairs.txt" ^
    --save_path "root_dir/results/adaface_optimized/blur/celeba-test_aligned/adaface_optimized.csv" ^
    --dir_to_log "root_dir/logs" ^
    --root_dir "root_dir"
echo [1/2] Done.

REM --- Step 2: AdaFace verification (pixelize) ---
echo [2/2] AdaFace verification: pixelize | celeba-test_aligned
python -m deid.evaluation.adaface_optimized ^
    "root_dir/datasets/aligned/celeba-test_aligned" ^
    "root_dir/datasets/deidentified/pixelize/celeba-test_aligned" ^
    --dataset_name celeba-test_aligned ^
    --technique_name pixelize ^
    --genuine_pairs_filepath "root_dir/datasets/pairs/celeba-test_aligned_genuine_pairs.txt" ^
    --impostor_pairs_filepath "root_dir/datasets/pairs/celeba-test_aligned_impostor_pairs.txt" ^
    --save_path "root_dir/results/adaface_optimized/pixelize/celeba-test_aligned/adaface_optimized.csv" ^
    --dir_to_log "root_dir/logs" ^
    --root_dir "root_dir"
echo [2/2] Done.

REM --- Step 3: Embedding visualization (displacement + collapse + comparison) ---
echo.
echo [3/3] Generating embedding visualizations...
python -m deid.explore.embedding_viz_cli ^
    --dataset celeba-test_aligned ^
    --model adaface ^
    --techniques blur pixelize ^
    --method umap
echo [3/3] Done.

echo.
echo ========================================
echo  AdaFace CelebA run complete!
echo ========================================
echo Results:   root_dir/results/adaface_optimized/{blur,pixelize}/celeba-test_aligned/
echo Embeddings: root_dir/preprocess/temp/adaface/
echo Visuals:   root_dir/results/viz/celeba-test_aligned_adaface/
echo.
pause
