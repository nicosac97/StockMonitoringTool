@echo off
rem Start Stock Ledger. First run creates the virtual environment and installs dependencies.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    python -m venv .venv || exit /b 1
    .venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
)
start "" http://localhost:8501
.venv\Scripts\streamlit run app.py
