@echo off
REM SwinFace embeddings: aligned + 20 techniques x {kdef, rafd, mug-still}
REM Same invocation pattern as run_swinface.bat (auto-appends last 2 input paths)
set SWIN=C:\Users\b\anaconda3\envs\swinface\python.exe
set ROOT=D:\dev\deid-toolkit\root_dir
cd /d D:\dev\deid-toolkit\sota_evaluators

for %%D in (kdef rafd mug-still) do (
  echo [aligned] %%D
  %SWIN% -u generate_embeddings_swinface.py --input "%ROOT%\datasets\aligned\%%D" --output "%ROOT%\embeddings\SwinFace"
)
for %%D in (kdef rafd mug-still) do for %%T in (AIDPro AMT-GAN AnonNET CLEANIR CPP-DeID DeepPrivacy DeepPrivacy2 DiffPrivate FADM FALCO FAMS G2Face GANonymization IPFA LDFA NullFace PRO-Face RP RiDDLE iFADIT) do (
  echo [%%T] %%D
  %SWIN% -u generate_embeddings_swinface.py --input "%ROOT%\datasets\%%T\%%D" --output "%ROOT%\embeddings\SwinFace\datasets"
)
echo SWINFACE_EMBEDS_DONE
