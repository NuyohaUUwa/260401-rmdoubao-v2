@echo off
REM Start Doubao Watermark Remover FastAPI Service

REM Switch to the directory of the current script (project root)
cd /d "%~dp0"

REM Activate conda environment (run in Anaconda Prompt or terminal with conda initialized)
call conda activate redoubao

REM Run FastAPI application in module mode to avoid package import issues
python -m app.main

pause
