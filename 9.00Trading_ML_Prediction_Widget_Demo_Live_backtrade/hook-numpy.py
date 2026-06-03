from PyInstaller.utils.hooks import collect_all
datas, binaries, hiddenimports = collect_all('numpy')
from PyInstaller.utils.hooks import collect_data_files
datas = collect_data_files('backtesting')