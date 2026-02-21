@echo off
setlocal
cd /d "%~dp0\.."
python -m py_compile project_config.py ml_pipeline_complete.py app.py
if errorlevel 1 exit /b %errorlevel%
python ml_pipeline_complete.py
endlocal
