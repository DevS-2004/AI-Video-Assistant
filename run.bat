@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo Starting AI Video Assistant...
echo ===================================================

if exist "C:\Users\Devendra Singh\.venvs\AI-Video-Assistant\Scripts\streamlit.exe" (
    set "STREAMLIT_CMD=C:\Users\Devendra Singh\.venvs\AI-Video-Assistant\Scripts\streamlit.exe"
) else if exist ".venv\Scripts\streamlit.exe" (
    set "STREAMLIT_CMD=.venv\Scripts\streamlit.exe"
) else (
    set "STREAMLIT_CMD=streamlit"
)

"%STREAMLIT_CMD%" run app.py
pause
