@echo off
rem Cross-condition linkability (de-anonymization) analysis — TransFace
cd /d D:\dev\deid-toolkit_evaluators
conda run -n transface python analyze_linkability.py ^
    --embeddings-root D:\dev\deid-toolkit\root_dir\embeddings\TransFace ^
    --pairs           D:\dev\deid-toolkit\root_dir\datasets\pairs ^
    --output          D:\dev\deid-toolkit\root_dir\predictions\linkability_transface
pause
