@echo off
rem DeepFace gender+expression evaluation on all 4 datasets (CPU TF).
rem Run after faceXformer/other GPU jobs if CPU cores are contended (still OK to overlap:
rem FaceXFormer is GPU-bound, DeepFace is CPU-bound).
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
set PY=C:\Users\b\anaconda3\envs\deepface\python.exe
set SD=%~dp0
set OUT=D:\dev\deid-toolkit\root_dir\predictions

for %%d in (kdef rafd mug-still celeba-test) do (
    %PY% -u %SD%evaluate_gender_expression_deepface.py --dataset %%d --output %OUT%\%%d
)
