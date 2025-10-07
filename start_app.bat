@echo off
echo Starting AI Self-Checkout System...
echo.
echo Installing/Updating dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit application...
echo The application will open in your default web browser.
echo Press Ctrl+C to stop the application.
echo.
streamlit run streamlit_app.py
pause