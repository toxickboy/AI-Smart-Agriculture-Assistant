@echo off
echo ============================================
echo   AI Smart Agriculture Assistant
echo ============================================
echo.

:: Change to the folder where this script lives
cd /d "%~dp0"

:: Check if streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    echo.
)

echo Starting app at http://localhost:8501
echo Press Ctrl+C to stop.
echo.

python -m streamlit run app.py
pause
