@echo off
setlocal
cd /d "%~dp0\.."
python -m scripts.run_app
endlocal
