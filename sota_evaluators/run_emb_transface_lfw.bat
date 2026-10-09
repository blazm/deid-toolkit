@echo off
REM TransFace embeddings: aligned + 19 COMPLETED techniques x {lfw}
REM Same invocation pattern as run_emb_transface_ds.bat (kdef/rafd/mug-still).
REM CLEANIR is excluded (incomplete); run run_emb_cleanir_lfw.bat when it is done.
set TF=C:\Users\b\anaconda3\envs\transface\python.exe
set ROOT=D:\dev\deid-toolkit\root_dir
cd /d D:\dev\deid-toolkit\sota_evaluators

echo [aligned] lfw
%TF% -u generate_embeddings_transface.py --input "%ROOT%\datasets\aligned\lfw" --output "%ROOT%\embeddings\TransFace" --network vit_l_dp005_mask_005 --weight models\transface\glint360k_model_TransFace_L.pt

for %%T in (AIDPro AMT-GAN AnonNET CPP-DeID DeepPrivacy DeepPrivacy2 DiffPrivate FADM FALCO FAMS G2Face GANonymization IPFA LDFA NullFace PRO-Face RP RiDDLE iFADIT) do (
  echo [%%T] lfw
  %TF% -u generate_embeddings_transface.py --input "%ROOT%\datasets\%%T\lfw" --output "%ROOT%\embeddings\TransFace\datasets" --network vit_l_dp005_mask_005 --weight models\transface\glint360k_model_TransFace_L.pt
)
echo TRANSFACE_EMBEDS_LFW_DONE
