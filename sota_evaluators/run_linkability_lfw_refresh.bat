@echo off
REM Refresh LFW linkability (both backbones) after the LDFA @512 re-embed.
REM RUN AFTER run_emb_ldfa_lfw_512.bat. All 20 techniques are re-scored
REM against the (unavoidably unchanged) lfw pairs; only LDFA's numbers move.
set PY=C:\Users\b\anaconda3\envs\deid-toolkit\python.exe
set ROOT=D:\dev\deid-toolkit\root_dir
cd /d D:\dev\deid-toolkit\sota_evaluators

REM Back up first-round (256-scale interim) numbers for A/B
if not exist "%ROOT%\predictions\linkability_lfw_round1_256scale" mkdir "%ROOT%\predictions\linkability_lfw_round1_256scale"
copy /Y linkability_TransFace_lfw.csv          "%ROOT%\predictions\linkability_lfw_round1_256scale\" >nul
copy /Y linkability_TransFace_lfw_results.json "%ROOT%\predictions\linkability_lfw_round1_256scale\" >nul
copy /Y linkability_SwinFace_lfw.csv           "%ROOT%\predictions\linkability_lfw_round1_256scale\" >nul
copy /Y linkability_SwinFace_lfw_results.json  "%ROOT%\predictions\linkability_lfw_round1_256scale\" >nul

echo [TransFace] linkability lfw
%PY% -u analyze_linkability.py --model --embeddings-root "%ROOT%\embeddings\TransFace" --dataset lfw --output linkability_TransFace_lfw
if errorlevel 1 goto :fail

echo [SwinFace] linkability lfw
%PY% -u analyze_linkability.py --model --embeddings-root "%ROOT%\embeddings\SwinFace" --dataset lfw --output linkability_SwinFace_lfw
if errorlevel 1 goto :fail

echo LFW_LINKABILITY_REFRESH_DONE
goto :eof
:fail
echo ANALYSIS_FAIL
exit /b 1
