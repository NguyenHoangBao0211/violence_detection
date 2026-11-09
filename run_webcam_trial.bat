@echo off
cd /d "%~dp0"
echo Webcam trial: http://127.0.0.1:8001/live
"%~dp0.venv\Scripts\python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8001 --loop asyncio:SelectorEventLoop
pause
