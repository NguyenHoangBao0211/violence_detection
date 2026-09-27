@echo off
cd /d "%~dp0"
echo Starting Violence Detection Web App...
echo Open http://127.0.0.1:8000
rem SelectorEventLoop: the default Windows Proactor loop logs WinError 10054
rem and can stop accepting connections when the browser drops a request.
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
pause
