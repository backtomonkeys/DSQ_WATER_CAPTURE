@echo off
rem Double-click to open the DSQ core web page (Windows)
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
if exist "..\.venv\Scripts\activate.bat" call "..\.venv\Scripts\activate.bat"
python -m streamlit run app.py
pause
