# build_fixed.ps1
$projectPath = "C:\Users\dahab\PyCharm_2026.2.23\New_Bollinger_bands\9.00Trading_ML_Prediction_Widget_Demo_Live_backtrade"
$pythonExe = "C:\Program Files\Python311\python.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Trading Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $projectPath
Write-Host "Current directory: $PWD" -ForegroundColor Green
Write-Host ""

# Check Python
Write-Host "Step 1: Checking Python..." -ForegroundColor Yellow
& $pythonExe --version
Write-Host ""

# Check NumPy
Write-Host "Step 2: Checking NumPy..." -ForegroundColor Yellow
& $pythonExe -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
Write-Host ""

# Clean previous builds
Write-Host "Step 3: Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" -ErrorAction SilentlyContinue }
Remove-Item "*.spec" -ErrorAction SilentlyContinue
Write-Host "Cleaned successfully" -ForegroundColor Green
Write-Host ""

# Build with PyInstaller
Write-Host "Step 4: Building with PyInstaller..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
Write-Host ""

& $pythonExe -m PyInstaller --clean --noconfirm `
    --name "Professional_Trading_ML_AI" `
    --onefile `
    --console `
    --icon "trade.ico" `
    --add-data "images;images" `
    --add-data "models;models" `
    --add-data "strategies;strategies" `
    --add-data "utils;utils" `
    --add-data "backtest_params.json;." `
    --add-data "config.json;." `
    --add-data "launcher_config.json;." `
    --add-data "strategy_settings.json;." `
    --collect-all numpy `
    --collect-all pandas `
    --collect-all scipy `
    --collect-all sklearn `
    --collect-all matplotlib `
    --collect-all PIL `
    --collect-all backtesting `
    --hidden-import numpy.core._multiarray_umath `
    --hidden-import numpy.core.umath `
    --hidden-import scipy.special._ufuncs_cxx `
    --hidden-import sklearn.utils._cython_blas `
    --hidden-import sklearn.utils._openmp_helpers `
    --exclude-module IPython `
    --exclude-module notebook `
    --exclude-module jupyter `
    --exclude-module jupyter_client `
    --exclude-module jupyter_core `
    --exclude-module PyQt5 `
    --exclude-module PyQt6 `
    --exclude-module PySide2 `
    --exclude-module PySide6 `
    --exclude-module wx `
    --exclude-module hypothesis `
    --exclude-module pytest `
    --exclude-module dask `
    --exclude-module distributed `
    --exclude-module xgboost.testing `
    main_New_MACD_HybridScore_Latest.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Executable location:" -ForegroundColor Cyan
    Write-Host "dist\Professional_Trading_ML_AI.exe" -ForegroundColor White
    Write-Host ""

    if (Test-Path "dist\Professional_Trading_ML_AI.exe") {
        $size = [math]::Round((Get-Item "dist\Professional_Trading_ML_AI.exe").Length / 1MB, 2)
        Write-Host "File size: ${size} MB" -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "BUILD FAILED with error code $LASTEXITCODE" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"