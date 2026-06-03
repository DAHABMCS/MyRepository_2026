# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# BlockDiag and related libraries (often needed for Streamlit)
hiddenimports = [
    'streamlit',
    'streamlit.web',
    'streamlit.web.cli',
    'streamlit.runtime',
    'streamlit.runtime.scriptrunner',
    'streamlit.runtime.websocket_session_manager',
    'streamlit.runtime.media_file_manager',
    'streamlit.logger',
    'streamlit.proto',
    'streamlit.proto.Alerts_pb2',
    'streamlit.proto.ForwardMsg_pb2',
    'streamlit.proto.PageConfig_pb2',
    'streamlit.proto.SessionStatus_pb2',
    'streamlit.proto.BackMsg_pb2',
    'streamlit.elements',
    'streamlit.elements.lib',
    'altair',
    'blinker',
    'cachetools',
    'click',
    'gitpython',
    'importlib_metadata',
    'numpy',
    'packaging',
    'pandas',
    'pillow',
    'protobuf',
    'pyarrow',
    'requests',
    'tenacity',
    'toml',
    'tornado',
    'tzlocal',
    'validators',
    'watchdog',
    'plotly',           # If you use Plotly charts
    'matplotlib',       # If you use Matplotlib
    'yfinance',         # For trading data
    'TA_Lib',          # If you use TA-Lib for indicators
    'ta',              # Alternative technical analysis library
    'bokeh',           # If you use Bokeh
]

# Collect Streamlit's metadata and data files
streamlit_datas = copy_metadata('streamlit')
streamlit_datas += collect_data_files('streamlit')

# Add your app's specific data files and directories
app_datas = [
    # Your main app file (change 'streamlit_app.py' to your actual main filename)
    ('app.py', '.'),
    ('main.py', '.'),
    ('streamlit_app.py', '.'),

    # Include pages folder if you have multi-page app
    ('pages', 'pages'),

    # Include any subdirectories your app uses
    ('utils', 'utils'),
    ('components', 'components'),
    ('data', 'data'),
    ('config', 'config'),

    # Include static assets
    ('assets', 'assets'),
    ('images', 'images'),
    ('css', 'css'),

    # Include .streamlit folder if it exists
    ('.streamlit', '.streamlit'),

    # Include any CSV, JSON, or other data files
    ('*.csv', '.'),
    ('*.json', '.'),
    ('*.yaml', '.'),
    ('*.toml', '.'),
]

# Combine all datas
all_datas = app_datas + streamlit_datas

# Python paths to include
python_paths = [
    'C:\\Users\\dahab\\PyCharm_2026.2.23\\New_Bollinger_bands\\Web_TradingApp',
]

# Binary files (if needed for specific packages like TA-Lib)
binaries = []

# Add TA-Lib DLL if you're using it (uncomment and adjust path as needed)
# ta_lib_path = r'C:\Users\dahab\PyCharm_2026.2.23\New_Bollinger_bands\Web_TradingApp\lib\ta_lib.dll'
# if os.path.exists(ta_lib_path):
#     binaries.append((ta_lib_path, '.'))

a = Analysis(
    ['run.py'],  # Launcher script name - make sure this exists
    pathex=python_paths,
    binaries=binaries,
    datas=all_datas,
    hiddenimports=hiddenimports,
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5',
        'PySide2',
        'PySide6',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'unittest',
        'doctest',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Web_TradingApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want to see console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon file path if you have one: 'icon.ico'
)

# Optional: Create a folder distribution instead of single file
# Uncomment below for --onedir mode (more stable)
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='Web_TradingApp',
# )