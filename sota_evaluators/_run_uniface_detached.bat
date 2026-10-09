@echo off
start "UniFace-eval" /MIN cmd /c "cd /d D:\dev\deid-toolkit\sota_evaluators && run_uniface_all.bat > _uniface_all_run.log 2>&1"
