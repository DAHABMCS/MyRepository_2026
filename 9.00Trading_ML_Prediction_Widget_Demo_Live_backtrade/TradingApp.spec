# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

datas = [('images', 'images'), ('models', 'models'), ('strategies', 'strategies'), ('utils', 'utils')]
binaries = [('C:\\Users\\dahab\\AppData\\Roaming\\Python\\Python311\\site-packages\\xgboost\\lib\\xgboost.dll', 'xgboost\\lib')]
hiddenimports = ['talib', 'talib.stream', 'talib.abstract', 'talib._ta_lib', 'xgboost', 'xgboost.sklearn', 'xgboost.training', 'xgboost.core', 'xgboost.compat', 'xgboost.data', 'xgboost.callback', 'backtesting', 'backtesting.backtesting', 'backtesting.lib', 'scipy', 'scipy.stats', 'scipy.signal', 'scipy.optimize', 'scipy.linalg', 'scipy.sparse', 'scipy.special', 'scipy._lib.messagestream', 'sklearn.utils._cython_blas', 'sklearn.neighbors._partition_nodes', 'sklearn.tree._utils', 'sklearn.utils._weight_vector', 'pandas._libs.tslibs.base', 'pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.nattype', 'pandas._libs.tslibs.timedeltas']
datas += collect_data_files('xgboost')
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pandas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('joblib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('backtesting')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('keras')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('talib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main_New_MACD_HybridScore_Latest.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'PyQt5', 'PyQt6', 'xgboost.testing'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TradingApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
