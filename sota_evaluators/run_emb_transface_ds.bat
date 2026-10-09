@echo off
REM TransFace embeddings: aligned + 20 techniques x {kdef, rafd, mug-still}
REM Same invocation pattern as run_transface.bat
set TF=C:\Users\b\anaconda3\envs\transface\python.exe
set ROOT=D:\dev\deid-toolkit\root_dir
cd /d D:\dev\deid-toolkit\sota_evaluators

for %%D in (kdef rafd mug-still) do (
  echo [aligned] %%D
  %TF% -u generate_embeddings_transface.py --input "%ROOT%\datasets\aligned\%%D" --output "%ROOT%\embeddings\TransFace" --network vit_l_dp005_mask_005 --weight models\transface\glint360k_model_TransFace_L.pt
)
for %%D in (kdef rafd mug-still) do for %%T in (AIDPro AMT-GAN AnonNET CLEANIR CPP-DeID DeepPrivacy DeepPrivacy2 DiffPrivate FADM FALCO FAMS G2Face GANonymization IPFA LDFA NullFace PRO-Face RP RiDDLE iFADIT) do (
  echo [%%T] %%D
  %TF% -u generate_embeddings_transface.py --input "%ROOT%\datasets\%%T\%%D" --output "%ROOT%\embeddings\TransFace\datasets" --network vit_l_dp005_mask_005 --weight models\transface\glint360k_model_TransFace_L.pt
)
echo TRANSFACE_EMBEDS_DONE
