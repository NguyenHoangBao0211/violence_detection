@echo off
cd /d "%~dp0"
echo Starting Violence Detection Web App...
echo Open http://127.0.0.1:8000
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
pause
