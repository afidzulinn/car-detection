@echo off
title Car Component AI Detector (PA-SCAN)
cd /d "%~dp0"
echo =================================================================
echo   Car Component AI Detector (Multi-Scale PA-SCAN)
echo   Dashboard : http://localhost:8000
echo   Health    : http://localhost:8000/health
echo =================================================================
start "" "http://localhost:8000"
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000
pause
