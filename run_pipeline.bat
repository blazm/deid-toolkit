@echo off
setlocal
set PYTHONIOENCODING=utf-8

REM deid-toolkit pipeline launcher (keep this file pure ASCII - cmd.com chokes on UTF-8 here)
REM
REM Usage (from any folder):
REM   run_pipeline.bat            no argument: full pipeline (deid run all)
REM   run_pipeline.bat run techniques
REM   run_pipeline.bat run evaluation
REM
REM Run the preparation check first:  run_toolkit.bat
REM
REM Requires: conda on PATH and the 'deid-toolkit' env (conda env create -f environment.yml).
REM All arguments after the script name are passed straight through to `python -m deid`.

cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo [run_pipeline] "conda" was not found on PATH.
    echo [run_pipeline] Add your Anaconda/Miniconda "Scripts" directory to PATH, or run manually:
    echo [run_pipeline]     conda activate deid-toolkit ^&^& python -m deid %*
    exit /b 1
)

call conda activate deid-toolkit
if errorlevel 1 (
    echo [run_pipeline] Could not activate conda env "deid-toolkit".
    echo [run_pipeline] Create it first:  conda env create -f environment.yml
    exit /b 1
)

if "%~1"=="" goto :no_command
set "args=%*"
goto :run_cli

:no_command
echo [run_pipeline] No command given - running the FULL pipeline now: deid run all
echo [run_pipeline] Pre-check first with: run_toolkit.bat   |   interrupt with Ctrl+C
set "args=run all"

:run_cli
python -m deid %args%
set RC=%errorlevel%

call conda deactivate
endlocal & exit /b %RC%
