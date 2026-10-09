@echo off
rem UniFace gender+emotion evaluation on all 4 datasets.
rem Protocol B (canonical UniFace: SCRFD detection + per-model crops);
rem detection rates are part of the output CSVs.
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set PY=C:\Users\b\anaconda3\envs\uniface\python.exe
set SD=%~dp0
set OUT=D:\dev\deid-toolkit\root_dir\predictions

for %%d in (kdef rafd mug-still celeba-test) do (
    %PY% -u %SD%evaluate_gender_expression_uniface.py --dataset %%d --output %OUT%\%%d
)
