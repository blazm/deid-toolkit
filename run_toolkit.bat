@echo off
setlocal
set PYTHONIOENCODING=utf-8

REM deid-toolkit launcher (keep this file pure ASCII - cmd.com chokes on UTF-8 here)
REM
REM Usage (from any folder):
REM   run_toolkit.bat             no argument: full preparation check (deid verify --all, no running)
REM   run_toolkit.bat verify      explicit preparation check (--all / --quiet / --detail)
REM   run_toolkit.bat list datasets
REM   run_toolkit.bat explore
REM
REM To actually run the pipeline use run_pipeline.bat (or: run_toolkit.bat run all).
REM
REM Requires: conda on PATH and the 'deid-toolkit' env (conda env create -f environment.yml).
REM All arguments after the script name are passed straight through to `python -m deid`.

cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo [run_toolkit] "conda" was not found on PATH.
    echo [run_toolkit] Add your Anaconda/Miniconda "Scripts" directory to PATH, or run manually:
    echo [run_toolkit]     conda activate deid-toolkit ^&^& python -m deid %*
    exit /b 1
)

call conda activate deid-toolkit
if errorlevel 1 (
    echo [run_toolkit] Could not activate conda env "deid-toolkit".
    echo [run_toolkit] Create it first:  conda env create -f environment.yml
    exit /b 1
)

if "%~1"=="" goto :no_command
set "args=%*"
goto :run_cli

:no_command
echo [run_toolkit] No command given - running the FULL preparation check (no pipeline): deid verify --all
echo [run_toolkit] To start the pipeline itself:  run_pipeline.bat
set "args=verify --all"

:run_cli
python -m deid %args%
set RC=%errorlevel%

call conda deactivate
endlocal & exit /b %RC%
