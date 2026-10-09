@echo off
REM SwinFace embeddings: aligned + 19 COMPLETED techniques x {lfw}
REM Same invocation pattern as run_emb_swinface_ds.bat (kdef/rafd/mug-still).
REM CLEANIR is excluded (incomplete); run run_emb_cleanir_lfw.bat when it is done.
set SWIN=C:\Users\b\anaconda3\envs\swinface\python.exe
set ROOT=D:\dev\deid-toolkit\root_dir
cd /d D:\dev\deid-toolkit\sota_evaluators

echo [aligned] lfw
%SWIN% -u generate_embeddings_swinface.py --input "%ROOT%\datasets\aligned\lfw" --output "%ROOT%\embeddings\SwinFace"

for %%T in (AIDPro AMT-GAN AnonNET CPP-DeID DeepPrivacy DeepPrivacy2 DiffPrivate FADM FALCO FAMS G2Face GANonymization IPFA LDFA NullFace PRO-Face RP RiDDLE iFADIT) do (
  echo [%%T] lfw
  %SWIN% -u generate_embeddings_swinface.py --input "%ROOT%\datasets\%%T\lfw" --output "%ROOT%\embeddings\SwinFace\datasets"
)
echo SWINFACE_EMBEDS_LFW_DONE
