@echo off
rem Identity mapping structure (1:1/1:N/N:1/M:N) analysis — TransFace
cd /d D:\dev\deid-toolkit_evaluators
conda run -n transface python analyze_id_mapping.py ^
    --embeddings-root D:\dev\deid-toolkit\root_dir\embeddings\TransFace ^
    --pairs           D:\dev\deid-toolkit\root_dir\datasets\pairs ^
    --output          D:\dev\deid-toolkit\root_dir\predictions\id_mapping_transface
pause
