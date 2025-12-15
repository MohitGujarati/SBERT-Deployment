@echo off
echo Starting AI Recommendation System...
echo ---------------------------------------------------
echo Please wait for the server to initialize...
echo The browser should open automatically at http://127.0.0.1:5000/
echo If it doesn't, please open that URL manually.
echo ---------------------------------------------------

start http://127.0.0.1:5000/

python app.py
pause
