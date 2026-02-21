@echo off
setlocal
cd /d "%~dp0\.."
python ml_pipeline_complete.py
endlocal
