$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Car Component AI Detector (Multi-Scale PA-SCAN)" -ForegroundColor Cyan
Write-Host "  Dashboard : http://localhost:8000" -ForegroundColor Green
Write-Host "  Health    : http://localhost:8000/health" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
Start-Job -ScriptBlock { Start-Sleep -Seconds 2; Start-Process "http://localhost:8000" } | Out-Null
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000
