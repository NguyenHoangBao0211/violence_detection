@echo off
cd /d "%~dp0"
echo Starting Violence Detection Web App...
echo Open http://127.0.0.1:8001
rem SelectorEventLoop: the default Windows Proactor loop logs WinError 10054
rem and can stop accepting connections when the browser drops a request.
"%~dp0.venv\Scripts\python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8001 --loop asyncio:SelectorEventLoop
pause
