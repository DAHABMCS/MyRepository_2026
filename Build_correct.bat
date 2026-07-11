@echo off
echo ========================================
echo Building Trading Application
echo ========================================
echo.

REM Change to project directory
cd /d "C:\Users\dahab\PyCharm_2026.2.23\New_Bollinger_bands\9.00Trading_ML_Prediction_Widget_Demo_Live_backtrade"

echo Current directory: %CD%
echo.

REM Use the correct Python path with quotes
set PYTHON_EXE="C:\Program Files\Python311\python.exe"

echo Step 1: Checking Python...
%PYTHON_EXE% --version
echo.

echo Step 2: Checking NumPy...
%PYTHON_EXE% -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
%PYTHON_EXE% -c "import numpy; print(f'NumPy location: {numpy.__file__}')"
echo.

echo Step 3: Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
del /q *.spec 2>nul
echo.

echo Step 4: Building with PyInstaller...
%PYTHON_EXE% -m PyInstaller --clean --noconfirm ^
    --name "Professional_Trading_ML_AI" ^
    --onefile ^
    --console ^
    --icon "trade.ico" ^
    --add-data "images;images" ^
    --add-data "models;models" ^
    --add-data "strategies;strategies" ^
    --add-data "utils;utils" ^
    --add-data "backtest_params.json;." ^
    --add-data "config.json;." ^
    --add-data "launcher_config.json;." ^
    --add-data "strategy_settings.json;." ^
    --collect-all numpy ^
    --collect-all pandas ^
    --collect-all scipy ^
    --collect-all sklearn ^
    --collect-all xgboost ^
    --collect-all matplotlib ^
    --collect-all PIL ^
    --hidden-import numpy.core._multiarray_umath ^
    --hidden-import numpy.core.umath ^
    --hidden-import numpy.random.mtrand ^
    --hidden-import scipy.special._ufuncs_cxx ^
    --hidden-import scipy.sparse._sparsetools ^
    --hidden-import sklearn.utils._cython_blas ^
    --hidden-import sklearn.utils._openmp_helpers ^
    --hidden-import sklearn.tree._tree ^
    --hidden-import pandas._libs.tslibs.base ^
    --hidden-import pandas._libs.tslibs.ccalendar ^
    --hidden-import pandas._libs.tslibs.np_datetime ^
    --hidden-import xgboost._dll ^
    --exclude-module IPython ^
    --exclude-module notebook ^
    --exclude-module jupyter ^
    --exclude-module jupyter_client ^
    --exclude-module jupyter_core ^
    --exclude-module PyQt5 ^
    --exclude-module PyQt6 ^
    --exclude-module PySide2 ^
    --exclude-module PySide6 ^
    --exclude-module wx ^
    main_New_MACD_HybridScore_Latest.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo Executable: dist\Professional_Trading_ML_AI.exe
    echo ========================================
) else (
    echo.
    echo ========================================
    echo BUILD FAILED with error code %errorlevel%
    echo ========================================
)

echo.
pause