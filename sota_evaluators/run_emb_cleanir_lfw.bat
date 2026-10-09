@echo off
REM FINAL: TransFace + SwinFace embeddings for CLEANIR on LFW.
REM Run ONLY after CLEANIR/lfw reaches 13,233 (check: dir /b ...CLEANIR\lfw | find /c).
set TF=C:\Users\b\anaconda3\envs\transface\python.exe
set SWIN=C:\Users\b\anaconda3\envs\swinface\python.exe
set ROOT=D:\dev\deid-toolkit\root_dir
cd /d D:\dev\deid-toolkit\sota_evaluators

echo [TransFace] CLEANIR lfw
%TF% -u generate_embeddings_transface.py --input "%ROOT%\datasets\CLEANIR\lfw" --output "%ROOT%\embeddings\TransFace\datasets" --network vit_l_dp005_mask_005 --weight models\transface\glint360k_model_TransFace_L.pt

echo [SwinFace] CLEANIR lfw
%SWIN% -u generate_embeddings_swinface.py --input "%ROOT%\datasets\CLEANIR\lfw" --output "%ROOT%\embeddings\SwinFace\datasets"
echo CLEANIR_EMBEDS_LFW_DONE
