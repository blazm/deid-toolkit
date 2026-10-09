@echo off
REM Re-embed LDFA LFW from the regenerated @512 de-id set (datasets\LDFA\lfw).
REM Overwrites the first-round 256-scale LDFA/lfw embeddings in place.
REM aligned\lfw is NOT re-embedded (originals unchanged).
set TF=C:\Users\b\anaconda3\envs\transface\python.exe
set SWIN=C:\Users\b\anaconda3\envs\swinface\python.exe
set ROOT=D:\dev\deid-toolkit\root_dir
cd /d D:\dev\deid-toolkit\sota_evaluators

echo [TransFace] LDFA lfw @512
%TF% -u generate_embeddings_transface.py --input "%ROOT%\datasets\LDFA\lfw" --output "%ROOT%\embeddings\TransFace\datasets" --network vit_l_dp005_mask_005 --weight models\transface\glint360k_model_TransFace_L.pt
if errorlevel 1 goto :fail

echo [SwinFace] LDFA lfw @512
%SWIN% -u generate_embeddings_swinface.py --input "%ROOT%\datasets\LDFA\lfw" --output "%ROOT%\embeddings\SwinFace\datasets"
if errorlevel 1 goto :fail

echo LDFA_LFW_512_EMBEDS_DONE
goto :eof
:fail
echo EMBED_FAIL
exit /b 1
