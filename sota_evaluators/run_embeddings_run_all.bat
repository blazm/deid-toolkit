@echo off
REM Embeddings for kdef / rafd / mug-still (aligned + 20 techniques),
REM both backbones, sequentially on one GPU. Resumable driver inside.
cd /d D:\dev\deid-toolkit\sota_evaluators
echo === SWINFACE START %date% %time% ===
C:\Users\b\anaconda3\envs\swinface\python.exe -u run_embeddings_extra_ds.py --model swinface
echo === SWINFACE DONE %date% %time% ===
echo === TRANSFACE START %date% %time% ===
C:\Users\b\anaconda3\envs\transface\python.exe -u run_embeddings_extra_ds.py --model transface
echo === TRANSFACE DONE %date% %time% ===
echo ALL_DONE

