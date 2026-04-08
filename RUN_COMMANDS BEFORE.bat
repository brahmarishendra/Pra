@echo off
REM Update pip to ensure compatibility
python -m pip install --upgrade pip

REM Install NumPy and Matplotlib and sklearn
python -m pip install numpy matplotlib scikit-learn

echo Installation complete.
pause   