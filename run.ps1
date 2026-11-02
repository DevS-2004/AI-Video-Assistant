# Run AI Video Assistant with Streamlit
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting AI Video Assistant..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$streamlitPath = "C:\Users\Devendra Singh\.venvs\AI-Video-Assistant\Scripts\streamlit.exe"
if (-not (Test-Path $streamlitPath)) {
    $streamlitPath = ".\.venv\Scripts\streamlit.exe"
}
if (-not (Test-Path $streamlitPath)) {
    $streamlitPath = "streamlit"
}

& $streamlitPath run app.py
