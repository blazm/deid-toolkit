@echo off
rem Cross-condition linkability (de-anonymization) analysis — SwinFace
cd /d D:\dev\deid-toolkit_evaluators
conda run -n swinface python analyze_linkability.py ^
    --embeddings-root D:\dev\deid-toolkit\root_dir\embeddings\SwinFace ^
    --pairs           D:\dev\deid-toolkit\root_dir\datasets\pairs ^
    --output          D:\dev\deid-toolkit\root_dir\predictions\linkability_swinface
pause
