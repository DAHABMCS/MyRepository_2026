"""
Professional Trading Platform — Web Edition
Streamlit conversion of ProfessionalTradingPlatformV9
Author: Amr Aboueldahab
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import time
import threading
import numpy as np
import ccxt
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import importlib
import importlib.util

# Anchor all relative file/folder lookups (config.json, strategies/, etc.) to
# this script's own directory, and make sure that directory is on sys.path so
# the strategies/, models/, and utils/ packages import correctly regardless
# of the working directory the app was launched from.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# websocket-client is optional — the app works fully without it (falls back
# to clock-aligned REST polling only). Install with: pip install websocket-client
try:
    import websocket as _ws_client

    _WS_CLIENT_AVAILABLE = True
except ImportError:
    _ws_client = None
    _WS_CLIENT_AVAILABLE = False

st.set_page_config(
    page_title="Professional Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family:'Exo 2',sans-serif; background:#080a12; color:#ccd6f6; }

[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b0d1a,#0e1120); border-right:1px solid #1e3a6e; }
[data-testid="stSidebar"] * { color:#e2eaff !important; }
[data-testid="stSidebar"] label { color:#8eaad4 !important; font-size:0.8rem !important; }

.main .block-container { background:#080a12; padding-top:1rem; }

[data-testid="metric-container"] { background:linear-gradient(135deg,#0e1528,#111a30); border:1px solid #1e4080; border-radius:8px; padding:12px 16px; box-shadow:0 0 10px #0050aa22; }
[data-testid="metric-container"] label { color:#7eb8e8 !important; font-size:0.5rem !important; font-weight:600 !important; letter-spacing:0.08rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#00e5ff !important; font-family:'Share Tech Mono',monospace !important; font-size:1.5rem !important; text-shadow:0 0 8px #00e5ff88 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:0.85rem !important; }

[data-testid="stTabs"] [role="tab"] { color:#6a90c0; font-weight:700; letter-spacing:0.06em; font-size:0.85rem; text-transform:uppercase; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#00e5ff !important; border-bottom:2px solid #00e5ff !important; text-shadow:0 0 8px #00e5ff66; }
[data-testid="stTabs"] [role="tab"]:hover { color:#a0d0ff !important; }

.stButton > button { background:linear-gradient(135deg,#0e1e3a,#0a1428); color:#00e5ff; border:1px solid #00e5ff66; border-radius:6px; font-family:'Share Tech Mono',monospace; font-size:0.85rem; letter-spacing:0.08em; transition:all 0.2s; font-weight:600; }
.stButton > button:hover { background:linear-gradient(135deg,#00e5ff20,#0a1428); border-color:#00e5ff; box-shadow:0 0 16px #00e5ff55; color:#ffffff; }

.log-box { background:#060810; border:1px solid #1a3060; border-radius:6px; padding:12px 16px; font-family:'Share Tech Mono',monospace; font-size:0.8rem; height:300px; overflow-y:auto; line-height:1.7; }

.status-badge { display:inline-block; padding:5px 16px; border-radius:20px; font-family:'Share Tech Mono',monospace; font-size:0.82rem; font-weight:bold; letter-spacing:0.12em; }
.status-buy  { background:#003a1a; color:#00ff99; border:1px solid #00ff99; box-shadow:0 0 10px #00ff9955; text-shadow:0 0 6px #00ff99; }
.status-sell { background:#3a0015; color:#ff3366; border:1px solid #ff3366; box-shadow:0 0 10px #ff336655; text-shadow:0 0 6px #ff3366; }
.status-parking { background:#0c0e1a; color:#7090c0; border:1px solid #304070; }

.section-header { color:#00e5ff; font-size:0.72rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; border-bottom:1px solid #1a3060; padding-bottom:6px; margin-bottom:10px; margin-top:14px; text-shadow:0 0 6px #00e5ff44; }

.top-banner { background:linear-gradient(90deg,#080e20,#0c1530,#080e20); border:1px solid #1a3a70; border-radius:8px; padding:14px 24px; display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; box-shadow:0 0 20px #0040aa18; }
.banner-title { font-family:'Share Tech Mono',monospace; font-size:1.15rem; color:#00e5ff; letter-spacing:0.14em; text-shadow:0 0 10px #00e5ff66; }

[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label,[data-testid="stSlider"] label,[data-testid="stRadio"] label { color:#8eaad4 !important; font-weight:600 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] { border-color:#1e3a6e !important; background:#0b0f1e !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] * { color:#ccd6f6 !important; }
.stRadio div[role="radiogroup"] label { color:#b0c8f0 !important; }

p, span, div, li { color:#b8ccee; }

[data-testid="stDateInput"] input { 
    color: #ccd6f6 !important; 
    background: #0b0f1e !important; 
    cursor: pointer !important;
    -webkit-appearance: none !important;
}
[data-testid="stDateInput"] button { 
    color: #00e5ff !important; 
    background: transparent !important;
}
[data-baseweb="popover"] { 
    z-index: 99999 !important; 
    background: #0e1120 !important; 
    border: 1px solid #1e3a6e !important;
    border-radius: 8px !important;
}
[data-baseweb="popover"] * { 
    color: #ccd6f6 !important; 
}
[data-baseweb="calendar"] * { 
    color: #ccd6f6 !important; 
    background: #0e1120 !important;
}
[data-baseweb="calendar"] button:hover { 
    background: #1a2a4a !important; 
    border-radius: 4px !important;
}
.stCaption,[data-testid="stCaptionContainer"] { color:#7090b8 !important; }
h1,h2,h3,h4 { color:#d0e4ff !important; }
strong { color:#e8f0ff; }
code { color:#00e5ff; background:#0a1428; border-radius:3px; padding:1px 5px; }

[data-testid="stExpander"] summary { color:#a0c0e8 !important; font-weight:600; }

hr { border-color:#1a3060 !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#060810; }
::-webkit-scrollbar-thumb { background:#1e3a6e; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#00e5ff55; }

.pulse-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#00ff99; margin-right:7px; box-shadow:0 0 8px #00ff99; animation:pulse 1.2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.75)} }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY LOADER - DYNAMICALLY LOADS DESKTOP STRATEGY FILES
# ═══════════════════════════════════════════════════════════════════════════

def find_strategy_files():
    """Find all strategy files in the project"""
    strategies = {}

    # Get the current directory where the script is running
    current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

    # Look in strategies/ folder (primary location based on your file structure)
    strategies_dir = current_dir / "strategies"
    if strategies_dir.exists():
        for file in strategies_dir.glob('*Strategy*.py'):
            strategies[file.stem] = str(file)
        for file in strategies_dir.glob('*strategy*.py'):
            strategies[file.stem] = str(file)

    # Also look in current directory
    for file in current_dir.glob('*Strategy*.py'):
        strategies[file.stem] = str(file)
    for file in current_dir.glob('*strategy*.py'):
        strategies[file.stem] = str(file)

    # Also look in parent directory (in case we're in a subfolder)
    parent_dir = current_dir.parent
    strategies_parent = parent_dir / "strategies"
    if strategies_parent.exists():
        for file in strategies_parent.glob('*Strategy*.py'):
            strategies[file.stem] = str(file)

    return strategies


def load_strategy_class(file_path: str, class_name: str = None):
    """Load a strategy class from a .py file.

    Strategy files (KalmanTrendStrategy_New.py, MomentumStrategy_MACD_HybridScore_Latest.py,
    scalping_strategy.py) use relative imports like `from .base3_New import BaseStrategy`,
    which only resolve if the file is imported as a real submodule of its package
    (e.g. `strategies.KalmanTrendStrategy_New`), not as a bare top-level module loaded
    directly from a file path. importlib.import_module() below preserves that package
    context; the old __import__/spec_from_file_location approach did not, which is why
    strategies would silently fail to load ("Desktop strategy not found").
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        package_name = file_path.parent.name  # e.g. "strategies"
        module_name = f"{package_name}.{file_path.stem}"  # e.g. "strategies.KalmanTrendStrategy_New"

        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            add_log(f"Error importing {module_name}: {e}", "warn")
            return None

        if class_name:
            return getattr(mod, class_name, None)

        # Find any Strategy class
        for attr_name in dir(mod):
            if 'Strategy' in attr_name and not attr_name.startswith('_'):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type):
                    return attr
        return None

    except Exception as e:
        add_log(f"Error loading strategy from {file_path}: {e}", "warn")
        return None


def get_strategy_class(strategy_name: str, for_backtest: bool = False):
    """
    Get the appropriate strategy class for the given strategy name
    Returns: (class, error_message)
    """
    strategy_map = {
        "Momentum": {
            "files": [
                "MomentumStrategy_MACD_HybridScore_Latest",
                "MomentumStrategy",
                "momentum_strategy",
                "TradingStrategy3",
                "base3_New",
            ],
            "backtest_class": "BacktestMomentumStrategy",
            "live_class": "MomentumStrategy"
        },
        "Kalman": {
            "files": [
                "KalmanTrendStrategy_New",
                "KalmanTrendStrategy",
                "kalman_strategy",
            ],
            "backtest_class": "BacktestKalmanTrendStrategy",
            "live_class": "KalmanTrendStrategy"
        },
        "Scalping": {
            "files": [
                "scalping_strategy",
                "ScalpingStrategy",
            ],
            "backtest_class": "BacktestScalpingStrategy",
            "live_class": "ScalpingStrategy"
        }
    }

    if strategy_name not in strategy_map:
        return None, f"Unknown strategy: {strategy_name}"

    config = strategy_map[strategy_name]
    class_name = config["backtest_class"] if for_backtest else config["live_class"]

    # Get the current directory
    current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

    # Try to find and load the strategy
    for file_name in config["files"]:
        # Try multiple locations
        locations = [
            current_dir / "strategies" / f"{file_name}.py",
            current_dir / f"{file_name}.py",
            current_dir.parent / "strategies" / f"{file_name}.py",
        ]

        for location in locations:
            if location.exists():
                cls = load_strategy_class(location, class_name)
                if cls:
                    add_log(f"✅ Loaded {strategy_name} from {location}", "info")
                    return cls, None

        # Try without path
        file_path = ROOT_DIR / f"{file_name}.py"
        if file_path.exists():
            cls = load_strategy_class(file_path, class_name)
            if cls:
                return cls, None

    return None, f"Could not find strategy files for {strategy_name} in strategies/ folder"


# Cache for strategy classes
_STRATEGY_CLASS_CACHE = {}


def get_cached_strategy_class(strategy_name: str, for_backtest: bool = False):
    """Get cached strategy class"""
    cache_key = f"{strategy_name}_{'backtest' if for_backtest else 'live'}"

    if cache_key in _STRATEGY_CLASS_CACHE:
        return _STRATEGY_CLASS_CACHE[cache_key], None

    cls, error = get_strategy_class(strategy_name, for_backtest)
    if cls:
        _STRATEGY_CLASS_CACHE[cache_key] = cls
        add_log(f"✅ Loaded {strategy_name} strategy", "info")
        st.session_state.strategy_status[strategy_name] = "Desktop Strategy Loaded ✅"
    else:
        add_log(f"⚠️ Could not load {strategy_name}: {error}", "warn")
        st.session_state.strategy_status[strategy_name] = f"Not found: {error[:50]}"

    return cls, error


# ═══════════════════════════════════════════════════════════════════════════
# CLOCK-ALIGNED CANDLE SYNC
# ═══════════════════════════════════════════════════════════════════════════

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240, "1D": 1440}


def seconds_until_next_candle_close(interval: str, buffer_sec: float = 3.0) -> float:
    period_sec = TF_MINUTES.get(interval, 15) * 60
    now = datetime.now(timezone.utc).timestamp()
    next_boundary = (int(now // period_sec) + 1) * period_sec
    return max(0.5, next_boundary - now + buffer_sec)


# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET CANDLE-CLOSE TRIGGER
# ═══════════════════════════════════════════════════════════════════════════

OKX_WS_BUSINESS_URL = "wss://ws.okx.com:8443/ws/v5/business"


class _CandleWSBridge:
    _registry = {}
    _registry_lock = threading.Lock()

    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self._lock = threading.Lock()
        self._connected = False
        self._last_confirmed_ts = None
        self._last_message_at = 0.0
        self._thread = None
        self._stop = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "last_confirmed_ts": self._last_confirmed_ts,
                "last_message_at": self._last_message_at,
            }

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        if not _WS_CLIENT_AVAILABLE:
            return
        bar_channel = f"candle{self.interval}"
        backoff = 1.0
        while not self._stop:
            ws = None
            try:
                ws = _ws_client.create_connection(OKX_WS_BUSINESS_URL, timeout=10)
                ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [{"channel": bar_channel, "instId": self.symbol}],
                }))
                with self._lock:
                    self._connected = True
                    self._last_message_at = time.time()
                backoff = 1.0
                ws.settimeout(25)

                while not self._stop:
                    try:
                        raw = ws.recv()
                    except Exception:
                        ws.send("ping")
                        with self._lock:
                            self._last_message_at = time.time()
                        continue

                    if raw == "pong":
                        with self._lock:
                            self._last_message_at = time.time()
                        continue

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    with self._lock:
                        self._last_message_at = time.time()

                    data = msg.get("data")
                    if not data:
                        continue
                    candle = data[0]
                    if len(candle) >= 2 and candle[-1] == "1":
                        with self._lock:
                            self._last_confirmed_ts = int(candle[0]) / 1000.0
            except Exception:
                pass
            finally:
                with self._lock:
                    self._connected = False
                try:
                    if ws is not None:
                        ws.close()
                except Exception:
                    pass
            if self._stop:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def stop(self):
        self._stop = True

    @classmethod
    def get(cls, symbol: str, interval: str) -> "_CandleWSBridge":
        key = (symbol, interval)
        with cls._registry_lock:
            bridge = cls._registry.get(key)
            if bridge is None:
                bridge = cls(symbol, interval)
                cls._registry[key] = bridge
                bridge.start()
            return bridge


def wait_for_next_trigger(symbol: str, interval: str) -> str:
    if not st.session_state.get("sync_to_clock", True):
        time.sleep(st.session_state.refresh_interval)
        return "fixed"

    buffer_sec = st.session_state.get("clock_sync_buffer", 3)
    period_sec = TF_MINUTES.get(interval, 15) * 60
    now = time.time()
    boundary_epoch = (int(now // period_sec) + 1) * period_sec
    deadline = boundary_epoch + buffer_sec

    use_ws = st.session_state.get("use_ws_trigger", True) and _WS_CLIENT_AVAILABLE
    if not use_ws:
        time.sleep(max(0.5, deadline - time.time()))
        return "fallback"

    bridge = _CandleWSBridge.get(symbol, interval)
    poll_step = 0.5
    while time.time() < deadline:
        snap = bridge.snapshot()
        confirmed_ts = snap.get("last_confirmed_ts")
        if confirmed_ts and confirmed_ts >= boundary_epoch - 2:
            return "websocket"
        time.sleep(poll_step)
    return "fallback"


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "trading_running": False,
        "backtest_running": False,
        "connected": False,
        "connection_mode": None,
        "logs": [],
        "trade_history": [],
        "current_status": "PARKING",
        "position": None,
        "position_long": None,
        "position_short": None,
        "stats": {
            "trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0,
            "long_trades": 0, "long_wins": 0, "long_pnl": 0.0,
            "short_trades": 0, "short_wins": 0, "short_pnl": 0.0,
        },
        "virtual_balance": {"USDT": 5000.0, "COIN": 0.0},
        "ml_prediction": None,
        "ml_confidence": 0.0,
        "backtest_results": None,
        "market_data": None,
        "config": {},
        "strategy_settings": {},
        "last_price": None,
        "trained_ml_model": None,
        "bars_processed": 0,
        "last_cycle_time": None,
        "last_signal": "—",
        "last_bar_ts": None,
        "cycle_count": 0,
        "refresh_interval": 10,
        "sync_to_clock": True,
        "use_ws_trigger": True,
        "last_trigger_source": None,
        "clock_sync_buffer": 3,
        "chart_markers": [],
        "trading_windows": {
            "Momentum": {"enabled": False, "start": "00:00", "end": "23:59",
                         "days": [0, 1, 2, 3, 4, 5, 6], "tz": "UTC"},
            "Kalman": {"enabled": False, "start": "00:00", "end": "23:59",
                       "days": [0, 1, 2, 3, 4, 5, 6], "tz": "UTC"},
            "Scalping": {"enabled": False, "start": "00:00", "end": "23:59",
                         "days": [0, 1, 2, 3, 4, 5, 6], "tz": "UTC"},
            "Enhanced": {"enabled": False, "start": "00:00", "end": "23:59",
                         "days": [0, 1, 2, 3, 4, 5, 6], "tz": "UTC"},
        },
        "tw_panel_open": False,
        "session_start_time": None,
        "session_start_balance": 5000.0,
        "commission_rate": 0.001,
        "session_summary": None,
        "ranging_settings": {
            "enabled": True,
            "momentum_min_adx": 20.0,
            "momentum_max_bb_width": 5.0,
            "momentum_min_slope": 0.15,
            "kalman_min_adx": 20.0,
            "kalman_max_bb_width": 6.0,
            "kalman_min_slope": 0.12,
            "scalping_min_adx": 18.0,
            "scalping_max_bb_width": 4.0,
            "scalping_min_slope": 0.10,
            "enhanced_min_adx": 20.0,
            "enhanced_max_bb_width": 5.0,
            "enhanced_min_slope": 0.15,
            "skip_on_ranging": True,
            "ranging_cooldown_bars": 5,
        },
        "ranging_cooldown_counter": 0,
        "ranging_analysis": None,
        "use_real_engine": True,
        "confidence_threshold": 65.0,
        "_live_strategies": {},
        "_last_symbol": None,
        "_windows_loaded": False,
        "strategy_status": {},
        # Clock sync UI controls
        "show_timing_controls": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def add_log(message: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"ts": ts, "msg": message, "level": level})
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-200:]


def load_config():
    cfg_path = ROOT_DIR / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                st.session_state.config = json.load(f)
            return True
        except Exception as e:
            add_log(f"Config load error: {e}", "err")
    return False


def load_strategy_settings():
    path = ROOT_DIR / "strategy_settings.json"
    if path.exists():
        try:
            with open(path) as f:
                st.session_state.strategy_settings = json.load(f)
        except Exception:
            pass


def save_strategy_settings(data: dict):
    try:
        with open(ROOT_DIR / "strategy_settings.json", "w") as f:
            json.dump(data, f, indent=2)
        add_log("✅ Strategy settings saved.", "info")
    except Exception as e:
        add_log(f"Save error: {e}", "err")


if not st.session_state.config:
    load_config()
if not st.session_state.strategy_settings:
    load_strategy_settings()


@st.cache_data(ttl=120, show_spinner=False)
def _read_windows_file() -> dict:
    path = ROOT_DIR / "trading_windows.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_trading_windows():
    data = _read_windows_file()
    if data:
        for strat, cfg in data.items():
            if strat in st.session_state.trading_windows:
                st.session_state.trading_windows[strat].update(cfg)
            else:
                st.session_state.trading_windows[strat] = cfg
        return True
    return False


if not st.session_state.get("_windows_loaded", False):
    if load_trading_windows():
        add_log("⏱ Trading windows loaded from disk.", "info")
    st.session_state["_windows_loaded"] = True


def save_trading_windows():
    try:
        with open(ROOT_DIR / "trading_windows.json", "w") as f:
            json.dump(st.session_state.trading_windows, f, indent=2)
        add_log("⏱ Trading windows saved to disk.", "info")
    except Exception as e:
        add_log(f"⚠️ Could not save trading_windows.json: {e}", "warn")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE BACKTEST EXCEL
# ═══════════════════════════════════════════════════════════════════════════

def save_backtest_excel(results, symbol, interval, strategy):
    """Save backtest results to Excel file on disk"""
    try:
        from pathlib import Path

        results_dir = ROOT_DIR / "backtest_results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"backtest_{symbol}_{interval}_{strategy}_{timestamp}.xlsx"

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            pd.DataFrame(list(results["metrics"].items()),
                         columns=["Metric", "Value"]).to_excel(writer, sheet_name="Summary", index=False)

            if results.get("trades") is not None:
                results["trades"].to_excel(writer, sheet_name="Trades")

            data_info = pd.DataFrame([
                ["Symbol", symbol],
                ["Interval", interval],
                ["Strategy", strategy],
                ["Start Date", results.get("requested_range", ["?", "?"])[0]],
                ["End Date", results.get("requested_range", ["?", "?"])[1]],
                ["Actual Start", results.get("data_range", ["?", "?", 0])[0]],
                ["Actual End", results.get("data_range", ["?", "?", 0])[1]],
                ["Bars Used", results.get("data_range", ["?", "?", 0])[2]],
                ["Export Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ], columns=["Info", "Value"])
            data_info.to_excel(writer, sheet_name="Info", index=False)

            if results.get("monte_carlo"):
                mc_data = results["monte_carlo"].copy()
                if "final_equities" in mc_data:
                    del mc_data["final_equities"]
                pd.DataFrame([mc_data]).to_excel(writer, sheet_name="MonteCarlo", index=False)

            if results.get("is_optimization") and results.get("optimization_results") is not None:
                results["optimization_results"].to_excel(writer, sheet_name="Optimization", index=False)

        add_log(f"✅ Excel saved: {filename}", "buy")
        return str(filename)
    except Exception as e:
        add_log(f"❌ Excel save error: {e}", "err")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def run_monte_carlo_backtest(df, strategy_class, capital, n_simulations=1000, commission=0.001):
    from backtesting import Backtest

    bt = Backtest(df, strategy_class, cash=float(capital), commission=commission)
    initial_stats = bt.run()

    trades = initial_stats.get("_trades", None)
    if trades is None or len(trades) == 0:
        return None, "No trades to simulate"

    trade_returns = trades["Return [%]"].values if "Return [%]" in trades.columns else []
    if len(trade_returns) == 0:
        return None, "No trade return data"

    final_equities = []
    max_drawdowns = []
    sharpe_ratios = []

    for i in range(n_simulations):
        shuffled_returns = np.random.permutation(trade_returns)
        cumulative = np.cumprod(1 + shuffled_returns / 100)
        final_equity = capital * cumulative[-1]
        final_equities.append(final_equity)

        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak * 100
        max_drawdowns.append(np.max(drawdown))

        returns_pct = shuffled_returns / 100
        if len(returns_pct) > 1:
            sharpe = np.mean(returns_pct) / (np.std(returns_pct) + 1e-9)
            sharpe_ratios.append(sharpe)

        if (i + 1) % 100 == 0:
            add_log(f"   Monte Carlo: {i + 1}/{n_simulations} simulations complete", "sys")

    results = {
        "mean_final_equity": np.mean(final_equities),
        "median_final_equity": np.median(final_equities),
        "std_final_equity": np.std(final_equities),
        "min_final_equity": np.min(final_equities),
        "max_final_equity": np.max(final_equities),
        "percentile_5": np.percentile(final_equities, 5),
        "percentile_25": np.percentile(final_equities, 25),
        "percentile_75": np.percentile(final_equities, 75),
        "percentile_95": np.percentile(final_equities, 95),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "max_drawdown_95": np.percentile(max_drawdowns, 95),
        "mean_sharpe": np.mean(sharpe_ratios) if sharpe_ratios else 0,
        "probability_profit": np.mean(np.array(final_equities) > capital) * 100,
        "n_simulations": n_simulations,
        "final_equities": final_equities,
    }

    return results, None


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_parameter_optimization(df, strategy_class, capital, param_grid, optimization_metric="Return %",
                               commission=0.001):
    from backtesting import Backtest
    import itertools

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    all_results = []
    total_combos = len(combinations)

    add_log(f"🔍 Running parameter optimization: {total_combos} combinations", "info")

    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        class OptimizedStrategy(strategy_class):
            pass

        for param_name, param_value in params.items():
            setattr(OptimizedStrategy, param_name, param_value)

        try:
            bt = Backtest(df, OptimizedStrategy, cash=float(capital), commission=commission)
            stats = bt.run()

            result = {
                **params,
                "Return %": stats['Return [%]'],
                "Sharpe Ratio": stats['Sharpe Ratio'],
                "Win Rate %": stats['Win Rate [%]'],
                "Max Drawdown %": stats['Max. Drawdown [%]'],
                "Total Trades": stats['# Trades'],
                "Profit Factor": stats.get('Profit Factor', 0),
                "Final Equity": stats['Equity Final [$]'],
            }
            all_results.append(result)

            if (idx + 1) % 10 == 0:
                add_log(f"   Optimization progress: {idx + 1}/{total_combos} combinations tested", "sys")

        except Exception as e:
            add_log(f"⚠️ Combination {idx + 1} failed: {e}", "warn")
            continue

    if not all_results:
        return None, None, None

    results_df = pd.DataFrame(all_results)

    metric_map = {
        "Return %": "Return %",
        "Sharpe Ratio": "Sharpe Ratio",
        "Win Rate": "Win Rate %",
        "Profit Factor": "Profit Factor",
        "Final Equity": "Final Equity",
    }

    metric_col = metric_map.get(optimization_metric, "Return %")
    results_df_sorted = results_df.sort_values(metric_col, ascending=False)
    best_params = results_df_sorted.iloc[0].to_dict()

    best_params_dict = {k: best_params[k] for k in param_names}
    best_metrics = {k: best_params[k] for k in
                    ['Return %', 'Sharpe Ratio', 'Win Rate %', 'Max Drawdown %', 'Total Trades', 'Profit Factor',
                     'Final Equity']}

    add_log(f"✅ Optimization complete! Best {optimization_metric}: {best_metrics.get(metric_col, 'N/A')}", "buy")

    return results_df, best_params_dict, best_metrics


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION PARAMETER GRIDS
# ═══════════════════════════════════════════════════════════════════════════

def get_optimization_param_grid(strategy: str) -> dict:
    if strategy == "Momentum":
        return {
            "ema_fast": [5, 8, 10, 12],
            "ema_slow": [20, 26, 30, 40],
            "ema_trend": [50, 60, 75, 100],
            "rsi_min": [30, 35, 40, 45],
            "rsi_max": [65, 70, 75, 80],
        }
    elif strategy == "Kalman":
        return {
            "kalman_q": [0.005, 0.01, 0.02, 0.05],
            "kalman_r": [100, 300, 500, 800],
            "rsi_min": [35, 40, 45],
            "rsi_max": [60, 65, 70],
        }
    elif strategy == "Scalping":
        return {
            "ema_fast": [3, 5, 8, 10],
            "ema_slow": [15, 20, 25, 30],
            "rsi_period": [10, 14, 18],
            "stop_loss_pct": [0.3, 0.5, 0.8, 1.0],
            "take_profit_pct": [0.8, 1.0, 1.5, 2.0],
        }
    else:
        return {
            "ema_fast": [5, 8, 12],
            "ema_slow": [20, 30, 40],
            "rsi_min": [30, 40],
            "rsi_max": [60, 70],
        }


# ═══════════════════════════════════════════════════════════════════════════
# RANGING MARKET DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def detect_ranging_market(df: pd.DataFrame, lookback: int = 50) -> dict:
    if len(df) < lookback:
        return {
            "is_ranging": False,
            "ranging_strength": 0.0,
            "reason": "Insufficient data — filter inactive",
            "adx": 0.0,
            "bb_width_pct": 0.0,
            "slope_pct": 0.0,
            "efficiency_ratio": 0.0,
            "range_width_pct": 0.0,
        }

    d = df.copy().iloc[-lookback:]
    close = d["Close"].values

    if 'adx' in df.columns and not df['adx'].isna().all():
        adx = float(df['adx'].dropna().iloc[-1])
    else:
        high, low = d["High"].values, d["Low"].values
        plus_dm = np.diff(high)
        plus_dm = np.maximum(plus_dm, 0)
        minus_dm = -np.diff(low)
        minus_dm = np.maximum(minus_dm, 0)

        tr = np.maximum(high[1:] - low[1:],
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1]))

        atr = pd.Series(tr).ewm(span=14, adjust=False).mean().iloc[-1]

        plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(span=14, adjust=False).mean().iloc[-1]

    period = 20
    if len(close) >= period:
        std = close[-period:].std()
        bb_middle = close[-period:].mean()
    else:
        std = close.std()
        bb_middle = close.mean()
    bb_width = (2 * std / bb_middle) * 100 if bb_middle != 0 else 0

    x = np.arange(len(close))
    slope, _ = np.polyfit(x, close, 1)
    slope_pct = (slope / (close[-1] + 1e-9)) * 100

    range_high = np.max(close)
    range_low = np.min(close)
    range_width_pct = ((range_high - range_low) / (range_low + 1e-9)) * 100

    gross_move = np.abs(close[-1] - close[0])
    net_move = np.sum(np.abs(np.diff(close)))
    efficiency_ratio = gross_move / (net_move + 1e-9)

    ranging_signals = []
    ranging_score = 0

    if adx < 20:
        ranging_signals.append(f"ADX={adx:.1f} (<20)")
        ranging_score += 35
    elif adx < 25:
        ranging_signals.append(f"ADX={adx:.1f} (<25)")
        ranging_score += 20
    elif adx > 35:
        ranging_score -= 20

    if bb_width < 3.0:
        ranging_signals.append(f"BB squeeze {bb_width:.1f}%")
        ranging_score += 30
    elif bb_width < 5.0:
        ranging_signals.append(f"BB narrow {bb_width:.1f}%")
        ranging_score += 15
    elif bb_width > 10.0:
        ranging_score -= 10

    if abs(slope_pct) < 0.15:
        ranging_signals.append(f"Slope={slope_pct:.2f}%/bar (flat)")
        ranging_score += 25
    elif abs(slope_pct) < 0.3:
        ranging_signals.append(f"Slope={slope_pct:.2f}%/bar (weak)")
        ranging_score += 10
    elif abs(slope_pct) > 0.8:
        ranging_score -= 15

    if efficiency_ratio < 0.3:
        ranging_signals.append(f"ER={efficiency_ratio:.2f} (very choppy)")
        ranging_score += 20
    elif efficiency_ratio < 0.45:
        ranging_signals.append(f"ER={efficiency_ratio:.2f} (choppy)")
        ranging_score += 10
    elif efficiency_ratio > 0.6:
        ranging_score -= 10

    if range_width_pct < 2.0:
        ranging_signals.append(f"Range={range_width_pct:.1f}% (tight)")
        ranging_score += 15
    elif range_width_pct > 8.0:
        ranging_score -= 10

    is_ranging = ranging_score >= 40
    ranging_strength = min(100, ranging_score)

    reason = " | ".join(ranging_signals[:3]) if ranging_signals else "No clear signal"
    if is_ranging:
        reason = f"RANGING: {reason}"
    else:
        reason = f"TRENDING: {reason}"

    return {
        "is_ranging": is_ranging,
        "ranging_strength": ranging_strength,
        "reason": reason,
        "adx": adx,
        "bb_width_pct": bb_width,
        "slope_pct": slope_pct,
        "efficiency_ratio": efficiency_ratio,
        "range_width_pct": range_width_pct,
    }


def should_skip_due_to_ranging(df: pd.DataFrame,
                               strategy: str,
                               min_adx: float = 20.0,
                               max_bb_width: float = 5.0,
                               min_slope_pct: float = 0.15) -> tuple[bool, dict]:
    analysis = detect_ranging_market(df, lookback=50)
    ranging_cfg = st.session_state.get("ranging_settings", {})

    if strategy == "Scalping":
        adx_threshold = ranging_cfg.get("scalping_min_adx", 18.0)
        max_bb = ranging_cfg.get("scalping_max_bb_width", 4.0)
        min_slope = ranging_cfg.get("scalping_min_slope", 0.1)
        strength_limit = ranging_cfg.get("scalping_max_ranging_strength", 45.0)
    elif strategy == "Kalman":
        adx_threshold = ranging_cfg.get("kalman_min_adx", 20.0)
        max_bb = ranging_cfg.get("kalman_max_bb_width", 6.0)
        min_slope = ranging_cfg.get("kalman_min_slope", 0.12)
        strength_limit = ranging_cfg.get("kalman_max_ranging_strength", 70.0)
    else:
        adx_threshold = ranging_cfg.get("momentum_min_adx", min_adx)
        max_bb = ranging_cfg.get("momentum_max_bb_width", max_bb_width)
        min_slope = ranging_cfg.get("momentum_min_slope", min_slope_pct)
        strength_limit = ranging_cfg.get("momentum_max_ranging_strength", 55.0)

    skip = False
    reasons = []

    if analysis["adx"] < adx_threshold:
        skip = True
        reasons.append(f"ADX={analysis['adx']:.1f} < {adx_threshold}")

    if analysis["bb_width_pct"] < max_bb and analysis["adx"] < 25:
        skip = True
        reasons.append(f"BB={analysis['bb_width_pct']:.1f}% (squeeze)")

    if abs(analysis["slope_pct"]) < min_slope and analysis["adx"] < 25:
        skip = True
        reasons.append(f"Slope={analysis['slope_pct']:.3f}%/bar")

    if analysis["ranging_strength"] > strength_limit:
        skip = True
        reasons.append(
            f"Ranging strength={analysis['ranging_strength']:.0f}%"
            f" > {strength_limit:.0f}% [{strategy}]"
        )

    analysis["skip"] = skip
    analysis["skip_reasons"] = reasons

    return skip, analysis


# ═══════════════════════════════════════════════════════════════════════════
# OKX CONNECTION
# ═══════════════════════════════════════════════════════════════════════════

def check_connection(mode: str):
    if mode.lower() == "demo":
        add_log("✅ Demo mode active — no API keys required.", "buy")
        st.session_state.connected = True
        st.session_state.connection_mode = "demo"
        return True
    cfg = st.session_state.config.get(mode, {})
    if not cfg or not cfg.get("api_key"):
        add_log(f"❌ No API config found for {mode} mode.", "err")
        return False
    try:
        from okx import MarketData
        api = MarketData.MarketAPI(
            api_key=cfg["api_key"],
            api_secret_key=cfg["api_secret_key"],
            passphrase=cfg["passphrase"],
            flag="0" if mode == "live" else "1",
        )
        resp = api.get_tickers(instType="SPOT")
        if resp.get("code") == "0":
            add_log(f"✅ Connection successful! ({mode.upper()} mode)", "buy")
            st.session_state.connected = True
            st.session_state.connection_mode = mode
            return True
        else:
            add_log(f"❌ Connection failed: {resp.get('msg', 'unknown')}", "err")
    except ImportError:
        add_log("⚠️ OKX library not installed. Run: pip install python-okx", "warn")
    except Exception as e:
        add_log(f"❌ Connection error: {str(e)}", "err")
    return False


def fetch_public_ohlcv(symbol: str, interval: str, limit: int = 1000,
                       silent: bool = False) -> "pd.DataFrame | None":
    try:
        import requests
        url = "https://www.okx.com/api/v5/market/candles"
        fetched: list = []
        after_param = ""
        remaining = min(limit, 1000)

        while remaining > 0:
            batch = min(remaining, 300)
            params: dict = {"instId": symbol, "bar": interval, "limit": str(batch)}
            if after_param:
                params["after"] = after_param
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                break
            rows = data["data"]
            fetched.extend(rows)
            remaining -= len(rows)
            if len(rows) < batch:
                break
            after_param = rows[-1][0]

        if not fetched:
            return None

        df = pd.DataFrame(fetched, columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume",
            "volCcy", "volBase", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])
        df = (df.sort_values("timestamp")
              .drop_duplicates("timestamp")
              .set_index("timestamp"))

        if not silent:
            add_log(f"📡 OKX public feed: {symbol} {interval} — {len(df)} bars fetched", "info")

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        add_log(f"⚠️ OKX public fetch error: {e}", "warn")
        return None


def fetch_historical_data_ccxt(symbol: str, exchange_name: str = "binance", start=None, end=None,
                               interval: str = "15m", days: int = 360, limit: "int | None" = None) -> "pd.DataFrame":
    import time as _time

    try:
        if exchange_name.lower() == "binance":
            exchange = ccxt.binance({
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "spot",
                }
            })
        elif exchange_name.lower() == "okx":
            exchange = ccxt.okx({"enableRateLimit": True})
        else:
            add_log(f"❌ Unsupported exchange: {exchange_name}", "err")
            return pd.DataFrame()
    except Exception as e:
        add_log(f"❌ Failed to initialize {exchange_name}: {e}", "err")
        return pd.DataFrame()

    ccxt_interval = interval.lower()
    interval_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
        "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
    }
    ccxt_interval = interval_map.get(ccxt_interval, ccxt_interval)

    if end is None:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    else:
        end_ms = int(pd.to_datetime(end, utc=True).timestamp() * 1000)

    if start is None:
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    else:
        start_ms = int(pd.to_datetime(start, utc=True).timestamp() * 1000)

    timeframe_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
        "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
        "1w": 604_800_000, "1M": 2_592_000_000,
    }
    interval_ms = timeframe_ms.get(ccxt_interval, 900_000)

    limit_per_request = 1000
    all_data = []
    since = start_ms
    max_iterations = 1000
    iteration = 0

    ccxt_symbol = symbol.replace("-", "/")

    add_log(
        f"🔍 Fetching {ccxt_symbol} {ccxt_interval} from {exchange_name.upper()} "
        f"({pd.to_datetime(start_ms, unit='ms', utc=True):%Y-%m-%d} → "
        f"{pd.to_datetime(end_ms, unit='ms', utc=True):%Y-%m-%d})",
        "info"
    )

    while since < end_ms and iteration < max_iterations:
        iteration += 1
        try:
            candles = exchange.fetch_ohlcv(
                ccxt_symbol,
                ccxt_interval,
                since=since,
                limit=limit_per_request
            )

            if not candles:
                break

            filtered = [c for c in candles if c[0] <= end_ms]
            if not filtered:
                break

            all_data.extend(filtered)
            last_ts = filtered[-1][0]

            if last_ts >= end_ms:
                break

            since = last_ts + 1

            _time.sleep(exchange.rateLimit / 1000)

            if iteration % 5 == 0:
                progress_pct = min(100, int(((since - start_ms) / (end_ms - start_ms)) * 100))
                add_log(f"   Fetch progress: {progress_pct}% ({len(all_data)} candles so far)", "sys")

        except Exception as e:
            add_log(f"⚠️ ccxt fetch error at iteration {iteration}: {e}", "warn")
            break

    if iteration >= max_iterations:
        add_log(f"⚠️ Reached maximum pagination requests ({max_iterations})", "warn")

    if not all_data:
        add_log(f"⚠️ No data fetched for {ccxt_symbol} {ccxt_interval}", "warn")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    # Convert timestamp properly without stripping timezone info
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Ensure index is a proper DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    start_disp = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_disp = pd.to_datetime(end_ms, unit="ms", utc=True)
    df = df[(df.index >= start_disp) & (df.index <= end_disp)]

    if limit is not None:
        df = df.tail(min(len(df), limit))

    if not df.empty:
        add_log(
            f"✅ {exchange_name.upper()} fetch complete: {len(df)} candles "
            f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})",
            "buy"
        )
    else:
        add_log(f"⚠️ No data found in the requested date range after filtering", "warn")

    return df


def fetch_okx_history_range(symbol: str, interval: str, start_ts: "pd.Timestamp",
                            end_ts: "pd.Timestamp", max_bars: int = 3000) -> "pd.DataFrame | None":
    try:
        import requests, time
        url = "https://www.okx.com/api/v5/market/history-candles"
        fetched: list = []
        after_param = str(int(end_ts.timestamp() * 1000))
        max_pages = max(1, max_bars // 100)

        for _ in range(max_pages):
            params = {"instId": symbol, "bar": interval, "limit": "100", "after": after_param}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                break
            rows = data["data"]
            fetched.extend(rows)
            oldest_ts = pd.to_datetime(float(rows[-1][0]), unit="ms")
            after_param = rows[-1][0]
            if oldest_ts <= start_ts or len(rows) < 100:
                break
            time.sleep(0.05)

        if not fetched:
            return None

        df = pd.DataFrame(fetched, columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume",
            "volCcy", "volBase", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df = (df.sort_values("timestamp")
              .drop_duplicates("timestamp")
              .set_index("timestamp"))
        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        add_log(f"⚠️ OKX history-candles fetch error: {e}", "warn")
        return None


def fetch_market_data(symbol: str, interval: str, limit: int = 500,
                      silent: bool = False):
    try:
        cfg_key = st.session_state.connection_mode or "demo"

        if cfg_key == "demo":
            df = fetch_public_ohlcv(symbol, interval, limit, silent=silent)
            if df is not None and len(df) >= 30:
                return df
            add_log("⚠️ OKX public feed unavailable — falling back to generated demo data.", "warn")
            return generate_demo_data(symbol, interval, limit)

        cfg = st.session_state.config.get(cfg_key, {})
        if not cfg or not cfg.get("api_key"):
            add_log(f"⚠️ No API config for '{cfg_key}' — trying public feed as fallback.", "warn")
            return fetch_public_ohlcv(symbol, interval, limit, silent=silent)

        from okx import MarketData
        api = MarketData.MarketAPI(
            api_key=cfg["api_key"],
            api_secret_key=cfg["api_secret_key"],
            passphrase=cfg["passphrase"],
            flag="0" if cfg_key == "live" else "1",
        )
        resp = api.get_candlesticks(instId=symbol, bar=interval, limit=str(min(limit, 300)))
        if resp.get("code") != "0" or not resp.get("data"):
            add_log(f"⚠️ OKX SDK returned no data (code={resp.get('code')}) — trying public feed.", "warn")
            return fetch_public_ohlcv(symbol, interval, limit, silent=silent)

        df = pd.DataFrame(resp["data"], columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume", "volCcy", "volBase", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])
        df = (df.sort_values("timestamp")
              .drop_duplicates("timestamp")
              .set_index("timestamp"))
        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        add_log(f"⚠️ Market data error: {e}", "warn")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_demo_data(symbol: str, interval: str, limit: int = 300,
                       anchor_end: "pd.Timestamp | None" = None) -> pd.DataFrame:
    base_prices = {
        "SOL-USDT": 142.0, "BTC-USDT": 65000.0, "ETH-USDT": 3200.0,
        "BNB-USDT": 580.0, "XRP-USDT": 0.58,
    }
    base = base_prices.get(symbol, 100.0)
    if anchor_end is not None:
        _end_for_seed = pd.Timestamp(anchor_end)
        _end_for_seed = _end_for_seed.tz_localize(None) if _end_for_seed.tzinfo else _end_for_seed
        rng = np.random.default_rng(int(_end_for_seed.timestamp()) // 60)
    else:
        rng = np.random.default_rng(int(pd.Timestamp.now().timestamp()) // 60)
    rets = rng.normal(0.0001, 0.008, limit)
    trend = np.sin(np.linspace(0, 6 * np.pi, limit)) * 0.0015
    rets = rets + trend

    closes = [base]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    closes = closes[1:]

    interval_mins = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240, "1D": 1440}.get(interval, 15)
    rows = []
    if anchor_end is not None:
        end_ts = pd.Timestamp(anchor_end)
        end_ts = (end_ts.tz_localize(None) if end_ts.tzinfo else end_ts).floor('min')
    else:
        end_ts = pd.Timestamp.now(tz='UTC').floor('min').tz_localize(None)
    for i, close in enumerate(closes):
        ts = end_ts - pd.Timedelta(minutes=interval_mins * (limit - i))
        spread = abs(rng.normal(0, 0.004))
        high = close * (1 + spread)
        low = close * (1 - spread)
        open_ = closes[i - 1] if i > 0 else close * (1 + rng.normal(0, 0.002))
        high = max(high, open_, close)
        low = min(low, open_, close)
        volume = abs(rng.normal(1_000_000, 350_000))
        rows.append([ts, open_, high, low, close, volume])

    df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df = df.set_index("timestamp")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TRADING WINDOW GUARD
# ═══════════════════════════════════════════════════════════════════════════

def is_within_trading_window(strategy: str) -> tuple[bool, str]:
    windows = st.session_state.get("trading_windows", {})
    cfg = windows.get(strategy)
    if not cfg or not cfg.get("enabled", False):
        return True, "No window restriction"

    now_utc = datetime.now(timezone.utc)
    day_now = now_utc.weekday()
    time_now = now_utc.hour * 60 + now_utc.minute

    allowed_days = cfg.get("days", list(range(7)))
    if day_now not in allowed_days:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        allowed_str = " ".join(day_names[d] for d in allowed_days)
        return False, f"Outside trading days ({day_names[day_now]}) — allowed: {allowed_str}"

    start_h, start_m = map(int, cfg["start"].split(":"))
    end_h, end_m = map(int, cfg["end"].split(":"))
    start_mins = start_h * 60 + start_m
    end_mins = end_h * 60 + end_m

    if start_mins <= end_mins:
        in_window = start_mins <= time_now <= end_mins
    else:
        in_window = time_now >= start_mins or time_now <= end_mins

    if not in_window:
        return False, (f"Outside trading hours {cfg['start']}–{cfg['end']} UTC "
                       f"(now {now_utc.strftime('%H:%M')} UTC)")
    return True, "Within trading window"


def render_window_indicator(strategy: str):
    allowed, reason = is_within_trading_window(strategy)
    windows = st.session_state.get("trading_windows", {})
    cfg = windows.get(strategy, {})
    if not cfg.get("enabled", False):
        badge_color = "#304070"
        label = "⏱ 24/7"
    elif allowed:
        badge_color = "#003a1a"
        label = f"⏱ OPEN {cfg['start']}–{cfg['end']}"
    else:
        badge_color = "#3a0015"
        label = f"⏱ CLOSED"
    st.markdown(
        f'<div style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'background:{badge_color};border:1px solid {"#00ff99" if allowed else "#ff2255"};'
        "font-family:Share Tech Mono,monospace;font-size:0.72rem;"
        f'color:{"#00ff99" if allowed else "#ff2255"};margin:4px 0">'
        f'{label}</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame, strategy: str = "Momentum") -> pd.DataFrame:
    d = df.copy()
    if strategy == "Scalping":
        span_fast = int(st.session_state.get("scal_ema_fast", 5.0))
        span_mid = int(st.session_state.get("scal_ema_slow", 20.0))
        span_slow = 60
    else:
        span_fast = int(st.session_state.get("mom_ema_fast", 5.0))
        span_mid = int(st.session_state.get("mom_ema_mid", 26.0))
        span_slow = int(st.session_state.get("mom_ema_slow", 60.0))
    span_fast = max(2, span_fast)
    span_mid = max(span_fast + 1, span_mid)
    span_slow = max(span_mid + 1, span_slow)
    d["ema5"] = d["Close"].ewm(span=span_fast, adjust=False).mean()
    d["ema26"] = d["Close"].ewm(span=span_mid, adjust=False).mean()
    d["ema60"] = d["Close"].ewm(span=span_slow, adjust=False).mean()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    d["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    d["macd"] = d["Close"].ewm(span=12, adjust=False).mean() - d["Close"].ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"] - d["Close"].shift()).abs()
    d["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()
    d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    plus_dm = d["High"].diff().clip(lower=0)
    minus_dm = (-d["Low"].diff()).clip(lower=0)
    tr14 = hl.ewm(span=14, adjust=False).mean()
    d["adx"] = (
            (plus_dm.ewm(span=14, adjust=False).mean() / tr14.replace(0, np.nan) -
             minus_dm.ewm(span=14, adjust=False).mean() / tr14.replace(0, np.nan)).abs() * 100
    ).ewm(span=14, adjust=False).mean()
    return d


# ═══════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED ML ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _safe_predict(model, df: pd.DataFrame) -> tuple[int, float]:
    try:
        result = model.predict(df)
    except Exception as e:
        add_log(f"⚠️ model.predict() raised: {e}", "err")
        return 0, 0.5

    def _to_int(v) -> int:
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float):
            return 1 if v > 0 else -1 if v < 0 else 0
        if isinstance(v, str):
            u = v.strip().upper()
            if u in ("1", "BUY", "LONG", "BULLISH", "UP"):   return 1
            if u in ("-1", "SELL", "SHORT", "BEARISH", "DOWN"): return -1
            return 0
        if hasattr(v, '__len__') and len(v) == 1:
            return _to_int(v[0])
        try:
            return int(v)
        except Exception:
            return 0

    def _to_conf(v) -> float:
        try:
            f = float(v) if not hasattr(v, '__len__') else float(v[0])
            return min(1.0, f / 100.0 if f > 1.0 else f)
        except Exception:
            return 0.6

    if isinstance(result, (tuple, list)):
        n = len(result)
        if n == 0:
            return 0, 0.5
        if n == 1:
            inner = result[0]
            if isinstance(inner, (tuple, list)) and len(inner) >= 2:
                return _to_int(inner[0]), _to_conf(inner[1])
            return _to_int(inner), 0.6
        return _to_int(result[0]), _to_conf(result[1])

    if hasattr(result, '__len__'):
        return _to_int(result[0]) if len(result) > 0 else 0, 0.6

    return _to_int(result), 0.6


def _ml_direction_gate(
        signal: str,
        direction: str,
        ml_pred: "str | None",
        ml_conf: float,
        conf_thr: float,
        auto_exec: bool,
) -> tuple[str, str]:
    if not ml_pred or ml_pred == "NEUTRAL" or ml_conf < conf_thr:
        if ml_pred and ml_pred != "NEUTRAL" and ml_conf < conf_thr:
            neutral_why = "ML below confidence threshold"
        elif ml_pred == "NEUTRAL":
            neutral_why = "ML NEUTRAL — no gate"
        else:
            neutral_why = "ML not trained / no prediction"
        return signal, neutral_why

    ml = ml_pred.upper()
    ml_bull = (ml == "BULLISH")
    ml_bear = (ml == "BEARISH")
    conf_str = f"{ml_conf:.0f}%"

    if direction == "Long":
        if ml_bull:
            if signal == "BUY":
                return "BUY", f"✅ ML ALIGNED: BULLISH ({conf_str}) confirms LONG entry"
            if signal == "HOLD" and auto_exec:
                return "BUY", f"✅ ML AUTO-EXEC: BULLISH ({conf_str}) → BUY (Long mode)"
            return signal, f"ML BULLISH ({conf_str}) — awaiting BUY setup"
        if ml_bear:
            if signal == "BUY":
                return "HOLD", f"⚠️ ML CONFLICT: BEARISH ({conf_str}) suppresses LONG entry"
            return signal, f"ML BEARISH ({conf_str}) — exit/hold passes through (Long mode)"

    elif direction == "Short":
        if ml_bear:
            if signal == "SELL":
                return "SELL", f"✅ ML ALIGNED: BEARISH ({conf_str}) confirms SHORT entry"
            if signal == "HOLD" and auto_exec:
                return "SELL", f"✅ ML AUTO-EXEC: BEARISH ({conf_str}) → SELL (Short mode)"
            return signal, f"ML BEARISH ({conf_str}) — awaiting SELL setup"
        if ml_bull:
            if signal == "SELL":
                return "HOLD", f"⚠️ ML CONFLICT: BULLISH ({conf_str}) suppresses SHORT entry"
            return signal, f"ML BULLISH ({conf_str}) — exit/hold passes through (Short mode)"

    else:
        if signal == "BUY":
            if ml_bull:
                return "BUY", f"✅ ML ALIGNED: BULLISH ({conf_str}) confirms long leg entry"
            if ml_bear:
                return "HOLD", f"⚠️ ML CONFLICT: BEARISH ({conf_str}) suppresses long leg entry"
        if signal == "SELL":
            if ml_bear:
                return "SELL", f"✅ ML ALIGNED: BEARISH ({conf_str}) confirms short leg entry"
            if ml_bull:
                return "HOLD", f"⚠️ ML CONFLICT: BULLISH ({conf_str}) suppresses short leg entry"
        if signal == "HOLD" and auto_exec:
            if ml_bull:
                return "BUY", f"✅ ML AUTO-EXEC: BULLISH ({conf_str}) → BUY (long leg, Both)"
            if ml_bear:
                return "SELL", f"✅ ML AUTO-EXEC: BEARISH ({conf_str}) → SELL (short leg, Both)"

    return signal, f"ML {ml} ({conf_str}) — no gate rule matched for {direction}/{signal}"


def build_ml_features(df: pd.DataFrame):
    d = compute_indicators(df).copy()
    d["ema_spread"] = (d["ema5"] - d["ema60"]) / (d["ema60"] + 1e-9)
    d["price_ema26"] = (d["Close"] - d["ema26"]) / (d["ema26"] + 1e-9)
    d["rsi_prev"] = d["rsi"].shift(1)
    d["macd_hist_prev"] = d["macd_hist"].shift(1)
    d["vol_change"] = d["Volume"].pct_change()
    d["hl_ratio"] = (d["High"] - d["Low"]) / (d["Close"] + 1e-9)
    d["target"] = np.where(d["Close"].shift(-1) > d["Close"], 1, -1)
    d = d.dropna()
    if len(d) < 60:
        return None, None
    feat_cols = ["rsi", "macd", "macd_hist", "atr", "vol_ratio", "adx",
                 "ema_spread", "price_ema26", "rsi_prev", "macd_hist_prev",
                 "vol_change", "hl_ratio"]
    return d[feat_cols].values, d["target"].values


class BuiltinMLModel:
    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.trained = False
        self.accuracy = 0.0

    def train(self, df: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X, y = build_ml_features(df)
        if X is None:
            raise ValueError("Not enough data — load at least 80 candles first.")

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)

        if self.model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=150, max_depth=8,
                                                random_state=42, n_jobs=-1)
        elif self.model_type == "xgb":
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(n_estimators=150, max_depth=5,
                                           learning_rate=0.05, random_state=42,
                                           eval_metric="logloss", verbosity=0)
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                        learning_rate=0.05, random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                    learning_rate=0.03, random_state=42)

        self.model.fit(X_tr, y_tr)
        self.accuracy = float(accuracy_score(y_te, self.model.predict(X_te)))
        self.trained = True
        return self

    def predict(self, df: pd.DataFrame):
        if not self.trained or self.model is None:
            raise RuntimeError("Model not trained yet.")
        X, _ = build_ml_features(df)
        if X is None:
            return 0, 0.5
        X_last = self.scaler.transform(X[-1:])
        pred = int(self.model.predict(X_last)[0])
        proba = self.model.predict_proba(X_last)[0]
        conf = float(np.max(proba))
        return pred, conf

    def forecast(self, df: pd.DataFrame, n: int = 5):
        if not self.trained:
            return []
        forecasts = []
        df_tmp = df.copy()
        for _ in range(n):
            pred, conf = self.predict(df_tmp)
            last_close = float(df_tmp["Close"].iloc[-1])
            atr = float(compute_indicators(df_tmp)["atr"].iloc[-1])
            next_close = last_close * (1 + 0.002 * pred)
            new_row = pd.DataFrame([{
                "Open": last_close, "High": last_close + atr * 0.5,
                "Low": last_close - atr * 0.5, "Close": next_close,
                "Volume": float(df_tmp["Volume"].mean()),
            }], index=[df_tmp.index[-1] + (df_tmp.index[-1] - df_tmp.index[-2])])
            df_tmp = pd.concat([df_tmp, new_row])
            forecasts.append({"close": next_close, "direction": pred, "conf": conf})
        return forecasts


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY SIGNAL FUNCTIONS (SIMPLIFIED BUILT-IN)
# ═══════════════════════════════════════════════════════════════════════════

def momentum_signal(df, stop_loss_pct, order_size_pct):
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row
    rsi_min = float(st.session_state.get("mom_rsi_min", 40.0))
    rsi_max = float(st.session_state.get("mom_rsi_max", 70.0))
    vol_thr = float(st.session_state.get("mom_vol_ratio", 1.3))
    adx_thr = float(st.session_state.get("mom_adx_min", 18.0))

    bull_stack = row["ema5"] > row["ema26"] > row["ema60"]
    bear_stack = row["ema5"] < row["ema26"] < row["ema60"]

    macd_bull = row["macd"] > row["macd_signal"] and prev["macd"] <= prev["macd_signal"]
    macd_bear = row["macd"] < row["macd_signal"] and prev["macd"] >= prev["macd_signal"]
    rsi_buy = rsi_min < row["rsi"] < rsi_max
    rsi_sell = (100 - rsi_max) < row["rsi"] < (100 - rsi_min)

    vol_ok = row["vol_ratio"] > vol_thr
    adx_ok = row["adx"] > adx_thr

    signal = "HOLD"
    reason = "Conditions not met"

    if bull_stack and macd_bull and rsi_buy and vol_ok and adx_ok:
        signal = "BUY"
        reason = f"EMA↑ MACD✅ RSI:{row['rsi']:.0f} Vol:{row['vol_ratio']:.2f}x ADX:{row['adx']:.0f}"
    elif bear_stack and macd_bear and rsi_sell and vol_ok and adx_ok:
        signal = "SELL"
        reason = f"EMA↓ MACD✅ RSI:{row['rsi']:.0f} Vol:{row['vol_ratio']:.2f}x ADX:{row['adx']:.0f}"
    return {"signal": signal, "price": row["Close"], "atr": row["atr"], "reason": reason,
            "rsi": row["rsi"], "adx": row["adx"], "macd_hist": row["macd_hist"],
            "ema_stack": "BULL" if bull_stack else "BEAR" if bear_stack else "FLAT",
            "vol_ratio": row["vol_ratio"]}


def kalman_signal(df, stop_loss_pct, order_size_pct):
    row = df.iloc[-1]
    close = df["Close"].values.astype(float)
    q_val = float(st.session_state.get("kal_proc_noise1", 0.01))
    r_val = float(st.session_state.get("kal_meas_noise", 500.0))
    rsi_min = float(st.session_state.get("kal_rsi_min", 40.0))
    rsi_max = float(st.session_state.get("kal_rsi_max", 65.0))
    str_min = float(st.session_state.get("kal_str_min", 70.0)) / 100.0
    x, p = close[0], 1.0
    for c in close[1:]:
        p += q_val;
        k = p / (p + r_val);
        x = x + k * (c - x);
        p = (1 - k) * p
    diff = row["Close"] - x
    strength = abs(diff) / (row["atr"] + 1e-9)

    rsi_buy = rsi_min < row["rsi"] < rsi_max
    rsi_sell = (100 - rsi_max) < row["rsi"] < (100 - rsi_min)

    signal = "HOLD"
    reason = "Waiting for trend"
    if diff > 0 and strength > str_min and rsi_buy:
        signal = "BUY";
        reason = f"Kalman↑ str:{strength:.2f} RSI:{row['rsi']:.0f}"
    elif diff < 0 and strength > str_min and rsi_sell:
        signal = "SELL";
        reason = f"Kalman↓ str:{strength:.2f} RSI:{row['rsi']:.0f}"
    return {"signal": signal, "price": row["Close"], "atr": row["atr"], "reason": reason,
            "rsi": row["rsi"], "adx": row["adx"], "macd_hist": row["macd_hist"],
            "ema_stack": "—", "vol_ratio": row["vol_ratio"]}


def scalping_signal(df, stop_loss_pct, order_size_pct):
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row
    cross_up = row["ema5"] > row["ema26"] and prev["ema5"] <= prev["ema26"]
    cross_down = row["ema5"] < row["ema26"] and prev["ema5"] >= prev["ema26"]

    rsi_ok_buy = 35.0 < row["rsi"] < 70.0
    rsi_ok_sell = 30.0 < row["rsi"] < 65.0

    signal = "HOLD"
    reason = "No crossover"
    if cross_up and rsi_ok_buy:
        signal = "BUY";
        reason = f"EMA cross↑ RSI:{row['rsi']:.0f}"
    elif cross_down and rsi_ok_sell:
        signal = "SELL";
        reason = f"EMA cross↓ RSI:{row['rsi']:.0f}"
    return {"signal": signal, "price": row["Close"], "atr": row["atr"], "reason": reason,
            "rsi": row["rsi"], "adx": row["adx"], "macd_hist": row["macd_hist"],
            "ema_stack": "BULL" if row["ema5"] > row["ema26"] else "BEAR",
            "vol_ratio": row["vol_ratio"]}


def enhanced_signal(df, stop_loss_pct, order_size_pct, interval="15m"):
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row

    bull_stack = row["ema5"] > row["ema26"] > row["ema60"]
    bear_stack = row["ema5"] < row["ema26"] < row["ema60"]
    macd_bull = row["macd"] > row["macd_signal"] and prev["macd"] <= prev["macd_signal"]
    macd_bear = row["macd"] < row["macd_signal"] and prev["macd"] >= prev["macd_signal"]
    cross_up = row["ema5"] > row["ema26"] and prev["ema5"] <= prev["ema26"]
    cross_down = row["ema5"] < row["ema26"] and prev["ema5"] >= prev["ema26"]
    rsi_buy = 40 < row["rsi"] < 70
    rsi_sell = 30 < row["rsi"] < 60
    vol_ok = row["vol_ratio"] > 1.3
    adx_ok = row["adx"] > 18

    close = df["Close"].values.astype(float)
    x, p, q, r = close[0], 1.0, 0.01, 500.0
    for c in close[1:]:
        p += q;
        k = p / (p + r);
        x = x + k * (c - x);
        p = (1 - k) * p
    kalman_diff = row["Close"] - x
    kalman_strength = abs(kalman_diff) / (row["atr"] + 1e-9)
    kalman_bull = kalman_diff > 0 and kalman_strength > 0.5
    kalman_bear = kalman_diff < 0 and kalman_strength > 0.5

    fast_intervals = {"1m", "5m"}
    medium_intervals = {"15m", "30m"}

    if interval in fast_intervals:
        weights = {
            "cross": (3, 3),
            "rsi": (2, 2),
            "volume": (1, 1),
            "macd": (1, 1),
            "kalman": (1, 1),
            "ema": (1, 1),
            "adx": (1, 1),
        }
        threshold = 7
    elif interval in medium_intervals:
        weights = {
            "cross": (1, 1),
            "rsi": (1, 1),
            "volume": (1, 1),
            "macd": (2, 2),
            "kalman": (2, 2),
            "ema": (2, 2),
            "adx": (1, 1),
        }
        threshold = 7
    else:
        weights = {
            "cross": (0, 0),
            "rsi": (1, 1),
            "volume": (1, 1),
            "macd": (2, 2),
            "kalman": (3, 3),
            "ema": (2, 2),
            "adx": (1, 1),
        }
        threshold = 7

    bull_score = 0
    bear_score = 0
    score_log = []

    def add(name, bull_cond, bear_cond):
        nonlocal bull_score, bear_score
        bw, sw = weights[name]
        if bull_cond and bw:
            bull_score += bw
            score_log.append(f"{name}↑+{bw}")
        if bear_cond and sw:
            bear_score += sw
            score_log.append(f"{name}↓+{sw}")

    add("cross", cross_up, cross_down)
    add("rsi", rsi_buy, rsi_sell)
    add("volume", vol_ok and (bull_stack or macd_bull or cross_up), vol_ok and (bear_stack or macd_bear or cross_down))
    add("macd", macd_bull, macd_bear)
    add("kalman", kalman_bull, kalman_bear)
    add("ema", bull_stack, bear_stack)
    add("adx", adx_ok, adx_ok)

    max_score = sum(w[0] for w in weights.values())

    signal = "HOLD"
    if bull_score >= threshold and bull_score > bear_score:
        signal = "BUY"
    elif bear_score >= threshold and bear_score > bull_score:
        signal = "SELL"

    reason = (
        f"[{interval}] Score B:{bull_score}/{max_score} S:{bear_score}/{max_score} "
        f"thr:{threshold} | {' '.join(score_log) or 'no triggers'}"
    )

    return {
        "signal": signal,
        "price": row["Close"],
        "atr": row["atr"],
        "reason": reason,
        "rsi": row["rsi"],
        "adx": row["adx"],
        "macd_hist": row["macd_hist"],
        "ema_stack": "BULL" if bull_stack else "BEAR" if bear_stack else "FLAT",
        "vol_ratio": row["vol_ratio"],
        "bull_score": bull_score,
        "bear_score": bear_score,
        "max_score": max_score,
    }


STRATEGY_FN = {
    "Momentum": momentum_signal,
    "Kalman": kalman_signal,
    "Scalping": scalping_signal,
    "Enhanced": enhanced_signal,
}

# ═══════════════════════════════════════════════════════════════════════════
# REAL STRATEGY ENGINE - BRIDGES DESKTOP STRATEGIES TO STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

REAL_ENGINE_STRATEGIES = {"Momentum", "Kalman", "Scalping"}


def get_current_momentum_params_for_backtest() -> dict:
    conf = float(st.session_state.get("confidence_threshold", 65.0))
    return {
        "quality_tier2_min": conf,
        "quality_tier1_min": conf + 3,
        "short_quality_tier2_min": conf + 5,
        "short_quality_tier1_min": conf + 8,
    }


def get_live_strategy(name: str):
    """Get or create a live strategy instance from desktop files"""
    cache = st.session_state.setdefault("_live_strategies", {})

    if name not in cache:
        try:
            # Try to get the strategy class from desktop files
            StrategyClass, error = get_cached_strategy_class(name, for_backtest=False)

            if StrategyClass is None:
                add_log(f"⚠️ Could not load real {name} strategy: {error}", "warn")
                add_log(f"   Using simplified built-in strategy instead", "info")
                return None

            # Create adapter and strategy instance
            adapter = LiveTradingAdapter(st.session_state.get("_last_symbol", "SOL-USDT"))
            cache[name] = StrategyClass(adapter)
            add_log(f"⚙️ Real {name} engine initialized from desktop files", "info")
            st.session_state.strategy_status[name] = "Desktop Strategy Loaded ✅"

        except Exception as e:
            add_log(f"❌ Failed to initialize {name} strategy: {e}", "err")
            st.session_state.strategy_status[name] = f"Error: {str(e)[:50]}"
            return None

    strat = cache[name]

    # Apply direction filter if the strategy supports it
    direction_filter = st.session_state.get("direction", "Long").lower()
    if hasattr(strat, "trade_direction"):
        strat.trade_direction = "both" if direction_filter == "both" else direction_filter

    return strat


class LiveTradingAdapter:
    """Bridges desktop strategies to Streamlit's session state"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_data = None

    def log_message(self, message, color="white"):
        level_map = {
            "red": "err", "bold red": "err", "orange": "warn", "yellow": "warn",
            "green": "buy", "bold green": "buy", "cyan": "info", "blue": "info",
            "magenta": "info",
        }
        add_log(f"[{self.symbol}] {message}", level_map.get(color, "sys"))

    def get_current_price(self):
        return st.session_state.get("last_price")

    def get_balance(self, currency, retries=3, delay=2.0):
        return float(st.session_state.virtual_balance.get(currency, 0.0))

    def get_account_balance(self):
        return {"quantity": 0.0}

    def get_volume_ratio(self, df=None, current_data=None, default=1.0):
        vol_ratio = None
        if df is not None and len(df) >= 2:
            for col_name in ["Volume_Ratio", "volume_ratio", "Vol_Ratio"]:
                if col_name in df.columns:
                    try:
                        val = df[col_name].iloc[-2]
                        if pd.notna(val) and val > 0:
                            vol_ratio = float(val)
                            break
                    except Exception:
                        continue
        if vol_ratio is None or vol_ratio <= 0:
            if current_data is not None:
                for col_name in ["Volume_Ratio", "volume_ratio"]:
                    try:
                        val = current_data.get(col_name) if hasattr(current_data, "get") else None
                        if val and float(val) > 0:
                            vol_ratio = float(val)
                            break
                    except Exception:
                        continue
        if vol_ratio is None or vol_ratio <= 0:
            if df is not None and "Volume" in df.columns and len(df) >= 22:
                try:
                    current_vol = float(df["Volume"].iloc[-2])
                    avg_vol = df["Volume"].iloc[-22:-2].mean()
                    if avg_vol > 0:
                        vol_ratio = current_vol / avg_vol
                except Exception:
                    pass
        if vol_ratio is None or vol_ratio <= 0:
            vol_ratio = default
        return max(0.01, min(10.0, vol_ratio))

    def update_status_indicators(self, status):
        st.session_state.current_status = status

    def place_order(self, side, price=None, quantity=None, **kwargs):
        """Virtual fill for demo mode"""
        if price is None:
            price = self.get_current_price()
        if price is None or quantity is None or quantity <= 0:
            return {"success": False}
        commission_rate = float(st.session_state.get("commission_rate", 0.001))
        commission = float(price) * float(quantity) * commission_rate
        st.session_state.virtual_balance["USDT"] -= commission
        return {"success": True, "filled_quantity": float(quantity), "filled_price": float(price),
                "commission": commission}


def _mirror_strategy_position_to_ui(strategy):
    """Reflect the real strategy's internal position into UI"""
    pos = getattr(strategy, "position", None)
    if pos is None:
        pos = getattr(strategy, "position_state", None)
    if pos and pos.get("type"):
        side = pos["type"]
        entry_price = pos.get("entry_price", pos.get("price"))
        st.session_state.position = {
            "price": entry_price,
            "stop_loss": pos.get("stop_loss"),
            "trail_price": pos.get("trailing_stop") or pos.get("stop_loss"),
            "side": side,
            "size_pct": st.session_state.get("order_size_pct", 30),
            "bar_ts": datetime.now().strftime("%H:%M"),
        }
        st.session_state.current_status = "BUY" if side == "long" else "SELL"
    else:
        st.session_state.position = None
        st.session_state.current_status = "PARKING"


def _record_real_trade_close(strategy_name, profit, exit_price, reason, side="EXIT"):
    st.session_state.stats["pnl"] += profit
    st.session_state.stats["trades"] += 1
    if profit > 0:
        st.session_state.stats["wins"] += 1
    t, w = st.session_state.stats["trades"], st.session_state.stats["wins"]
    st.session_state.stats["win_rate"] = (w / t * 100) if t else 0.0
    st.session_state.virtual_balance["USDT"] += profit
    st.session_state.trade_history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Symbol": strategy_name, "Side": side,
        "Entry": "-", "Exit": round(float(exit_price), 4) if exit_price else "-",
        "PnL%": "-", "PnL$": f"${profit:+.2f}",
        "Reason": reason,
    })


def run_real_strategy_cycle(strategy_name, current_data, current_price, df):
    """Run the desktop strategy cycle"""
    strategy = get_live_strategy(strategy_name)
    if strategy is None:
        return False

    try:
        result = strategy.run_analysis_cycle(current_data, current_price, df)
    except Exception as e:
        add_log(f"❌ [{strategy_name}] Strategy error: {e}", "err")
        return False

    if not isinstance(result, tuple) or len(result) < 2:
        return True

    first = result[0]

    # ── EXIT PATH ────────────────────────────────────────────────────────
    if first not in (None, "hold", "HOLD", -1, "buy", "sell", "long", "short", "sell_short"):
        exit_signal, exit_pct = first, result[1]
        add_log(f"🚨 [{strategy_name}] EXECUTING EXIT: {exit_signal}", "warn")
        try:
            success, profit, exit_price = strategy.execute_sell(reason=exit_signal, exit_percentage=exit_pct)
            if success:
                add_log(f"✅ [{strategy_name}] EXIT COMPLETE: P&L ${profit:.2f}", "buy" if profit > 0 else "sell")
                _record_real_trade_close(strategy_name, profit, exit_price, exit_signal)
            else:
                add_log(f"❌ [{strategy_name}] EXIT FAILED", "err")
        except Exception as e:
            add_log(f"❌ [{strategy_name}] EXIT error: {e}", "err")
        _mirror_strategy_position_to_ui(strategy)
        return True

    # ── ENTRY PATH ───────────────────────────────────────────────────────
    if len(result) >= 4 and first in ("buy", "sell", "long", "short", "sell_short"):
        decision, quality_score, shares, reason = result[0], result[1], result[2], result[3]
        if quality_score is None or not shares or shares <= 0:
            return True

        confidence_threshold = float(st.session_state.get("confidence_threshold", 65.0))
        if quality_score < confidence_threshold:
            add_log(
                f"🎯 [{strategy_name}] ENTRY REJECTED — quality {quality_score:.0f} "
                f"< threshold {confidence_threshold:.0f}", "warn")
            return True

        is_short = decision in ("sell", "short", "sell_short")
        atr_val = float(current_data.get("ATR", 1)) if hasattr(current_data, "get") else 1.0
        tier = getattr(strategy, "_last_entry_tier", 1)

        try:
            if is_short:
                if hasattr(strategy, "execute_short"):
                    success, filled_qty, order_id = strategy.execute_short(
                        shares=shares, price=current_price, atr=atr_val,
                        quality_score=quality_score, tier=tier)
                elif hasattr(strategy, "_pending_signal"):
                    strategy._pending_signal = {"direction": "short"}
                    success, filled_qty, order_id = strategy.execute_buy(
                        shares=shares, price=current_price, atr=atr_val,
                        quality_score=quality_score, tier=tier)
                else:
                    add_log(
                        f"⚠️ [{strategy_name}] SHORT signal ignored — no short-entry logic.", "warn")
                    return True
            else:
                success, filled_qty, order_id = strategy.execute_buy(
                    shares=shares, price=current_price, atr=atr_val,
                    quality_score=quality_score, tier=tier)

            if success:
                add_log(
                    f"{'🔴' if is_short else '🟢'} [{strategy_name}] "
                    f"{'SHORT' if is_short else 'LONG'} EXECUTED — qty {filled_qty:.4f} @ "
                    f"{current_price:.4f} (quality {quality_score:.0f}) | {reason}",
                    "sell" if is_short else "buy")
            else:
                add_log(f"❌ [{strategy_name}] ENTRY FAILED", "err")
        except Exception as e:
            add_log(f"❌ [{strategy_name}] ENTRY error: {e}", "err")

        _mirror_strategy_position_to_ui(strategy)
        return True

    _mirror_strategy_position_to_ui(strategy)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# LIVE TRADING CYCLE
# ═══════════════════════════════════════════════════════════════════════════

def run_trading_cycle(symbol, interval, strategy, stop_loss_pct, trailing_pct, order_size_pct, mode):
    _use_real_now = strategy in REAL_ENGINE_STRATEGIES and st.session_state.get("use_real_engine", True)

    ranging_settings = st.session_state.get("ranging_settings", {"enabled": True, "skip_on_ranging": True})

    if not st.session_state.trading_running:
        st.session_state.ranging_cooldown_counter = 0

    if not _use_real_now and st.session_state.ranging_cooldown_counter > 0:
        st.session_state.ranging_cooldown_counter -= 1
        add_log(f"⏸ Ranging cooldown: {st.session_state.ranging_cooldown_counter} bars remaining", "sys")
        return

    allowed, window_reason = is_within_trading_window(strategy)
    if not allowed:
        add_log(f"⏱ WINDOW BLOCKED [{strategy}] — {window_reason}", "warn")
        if st.session_state.trading_running:
            st.session_state.trading_running = False
            st.session_state.session_summary = build_session_summary("Window Closed")
            add_log(f"⏱ Trading auto-stopped — window closed for {strategy}.", "warn")
        return

    CHART_BARS = 300
    df = fetch_market_data(symbol, interval, limit=CHART_BARS, silent=True)
    if df is None or len(df) < 30:
        add_log("⚠️ Could not fetch candles — skipping cycle.", "warn")
        return

    if not _use_real_now and ranging_settings.get("enabled", True) and ranging_settings.get("skip_on_ranging", True):
        skip, ranging_analysis = should_skip_due_to_ranging(
            df,
            strategy=strategy,
            min_adx=ranging_settings.get(f"{strategy.lower()}_min_adx", 20.0),
            max_bb_width=ranging_settings.get(f"{strategy.lower()}_max_bb_width", 5.0),
            min_slope_pct=ranging_settings.get(f"{strategy.lower()}_min_slope", 0.15),
        )

        st.session_state.ranging_analysis = ranging_analysis

        if skip:
            cooldown = ranging_settings.get("ranging_cooldown_bars", 5)
            st.session_state.ranging_cooldown_counter = cooldown

            add_log(
                f"⛔ RANGING MARKET DETECTED — SKIPPING TRADE | "
                f"ADX:{ranging_analysis['adx']:.1f} | "
                f"BB:{ranging_analysis['bb_width_pct']:.1f}% | "
                f"Slope:{ranging_analysis['slope_pct']:.2f}%/bar | "
                f"Reason: {', '.join(ranging_analysis['skip_reasons'])} | "
                f"Cooldown: {cooldown} bars",
                "warn"
            )

            st.session_state.market_data = df
            st.session_state.last_price = float(df["Close"].iloc[-1])

            st.session_state.chart_markers.append({
                "ts": df.index[-1],
                "price": float(df["Close"].iloc[-1]),
                "kind": "RANGING",
                "label": "⛔ RNG",
            })
            if len(st.session_state.chart_markers) > 200:
                st.session_state.chart_markers = st.session_state.chart_markers[-200:]

            return

    st.session_state.market_data = df

    if len(df) < 2:
        add_log('⚠️ Need at least 2 bars to detect candle boundaries.', 'warn')
        return

    closed_bar_ts = str(df.index[-2])
    forming_price = float(df["Close"].iloc[-1])

    st.session_state.last_price = forming_price

    if closed_bar_ts == st.session_state.last_bar_ts:
        add_log(f"⏳ Waiting for new bar | Price: {forming_price:.4f}", "sys")
        return

    st.session_state.last_bar_ts = closed_bar_ts
    st.session_state.bars_processed += 1
    st.session_state.cycle_count += 1
    st.session_state["_beep_on_rerun"] = True
    st.session_state["_last_symbol"] = symbol
    st.session_state.last_cycle_time = datetime.now().strftime("%H:%M:%S")

    # ── REAL ENGINE PATH (Desktop Strategies) ──────────────────────────
    if strategy in REAL_ENGINE_STRATEGIES and st.session_state.get("use_real_engine", True):
        try:
            real_strategy = get_live_strategy(strategy)
            if real_strategy is None:
                # Fall back to simplified if real strategy couldn't load
                add_log(f"ℹ️ Falling back to simplified {strategy} strategy", "info")
            else:
                # Check if strategy has calculate_indicators method
                if hasattr(real_strategy, 'calculate_indicators'):
                    df_real = real_strategy.calculate_indicators(df)
                    if df_real is None or len(df_real) < 2:
                        add_log(f"⚠️ [{strategy}] Real engine: not enough bars for indicators yet.", "warn")
                        return
                    current_data = df_real.iloc[-2]
                    current_price = float(current_data["Close"])
                    st.session_state.last_price = current_price
                    run_real_strategy_cycle(strategy, current_data, current_price, df_real)
                    return
                else:
                    add_log(f"⚠️ [{strategy}] Strategy missing calculate_indicators method", "warn")
        except Exception as e:
            add_log(f"❌ [{strategy}] Real engine error: {e}", "err")
        # Fall through to simplified if real engine fails

    # ── SIMPLIFIED PATH (Built-in Strategies) ──────────────────────────
    df_closed = df.iloc[:-1].copy()
    df_ind = compute_indicators(df_closed, strategy=strategy).bfill().ffill()
    if df_ind.empty or df_ind['rsi'].isna().all():
        add_log('⚠️ Indicators incomplete — need more candle data.', 'warn')
        return

    sig_fn = STRATEGY_FN.get(strategy, momentum_signal)
    if strategy == "Enhanced":
        result = sig_fn(df_ind, stop_loss_pct, order_size_pct, interval=interval)
    else:
        result = sig_fn(df_ind, stop_loss_pct, order_size_pct)

    price = result["price"]
    signal = result["signal"]
    st.session_state.last_signal = signal

    bar_ts_str = pd.Timestamp(closed_bar_ts).strftime("%H:%M")

    score_str = ""
    if strategy == "Enhanced":
        score_str = (
            f" | Score B:{result.get('bull_score', 0)}/{result.get('max_score', 10)} "
            f"S:{result.get('bear_score', 0)}/{result.get('max_score', 10)}"
        )

    add_log(
        f"📊 Bar [{bar_ts_str}] #{st.session_state.bars_processed} | "
        f"O:{float(df_closed['Open'].iloc[-1]):.4f} "
        f"H:{float(df_closed['High'].iloc[-1]):.4f} "
        f"L:{float(df_closed['Low'].iloc[-1]):.4f} "
        f"C:{price:.4f} | "
        f"RSI:{result['rsi']:.0f} ADX:{result['adx']:.0f} "
        f"MACD:{'▲' if result['macd_hist'] > 0 else '▼'} "
        f"Vol:{result['vol_ratio']:.2f}x | EMA:{result['ema_stack']}"
        f"{score_str}",
        "sys"
    )

    direction_filter = st.session_state.get("direction", "Long")

    def _open_pos(slot_key, side_name, entry_price, sig):
        sl = entry_price * (1 - stop_loss_pct / 100) if side_name == "long" \
            else entry_price * (1 + stop_loss_pct / 100)
        st.session_state[slot_key] = {
            "price": entry_price, "stop_loss": sl,
            "trail_price": sl, "side": side_name,
            "size_pct": order_size_pct, "bar_ts": bar_ts_str,
        }
        st.session_state.chart_markers.append({
            "ts": df.index[-1], "price": entry_price,
            "kind": "BUY" if sig == "BUY" else "SELL_ENTRY",
            "label": f"{'L' if side_name == 'long' else 'S'}{entry_price:.4f}",
        })
        if len(st.session_state.chart_markers) > 200:
            st.session_state.chart_markers = st.session_state.chart_markers[-200:]
        add_log(
            f"{'🟢' if sig == 'BUY' else '🔴'} OPEN {side_name.upper()} @ {entry_price:.4f} | "
            f"SL:{sl:.4f} | Size:{order_size_pct}% | {result['reason']}",
            "buy" if sig == "BUY" else "sell"
        )

    def _trail_pos(slot_key):
        p = st.session_state[slot_key]
        if p is None:
            return False
        if p["side"] == "long":
            nt = price * (1 - trailing_pct / 100)
            if nt > p.get("trail_price", p["stop_loss"]):
                p["trail_price"] = nt
                add_log(f"   ↑ {slot_key} trail → {nt:.4f}", "sys")
            return price <= p.get("trail_price", p["stop_loss"])
        else:
            nt = price * (1 + trailing_pct / 100)
            if nt < p.get("trail_price", p["stop_loss"]):
                p["trail_price"] = nt
                add_log(f"   ↓ {slot_key} trail → {nt:.4f}", "sys")
            return price >= p.get("trail_price", p["stop_loss"])

    def _close_pos(slot_key, close_price, reason_exit):
        p = st.session_state[slot_key]
        if p is None:
            return
        side_c = p["side"]
        size_c = p.get("size_pct", order_size_pct)
        e_price = p["price"]
        raw_pct = ((close_price - e_price) / e_price * 100) if side_c == "long" \
            else ((e_price - close_price) / e_price * 100)
        fee_in = float(st.session_state.get("maker_pct", 0.0001)) * 100
        fee_out = float(st.session_state.get("taker_pct", 0.0001)) * 100
        net_pct = raw_pct - fee_in - fee_out
        usdt = st.session_state.virtual_balance["USDT"] * (size_c / 100) * (net_pct / 100)
        st.session_state.virtual_balance["USDT"] += usdt
        st.session_state.stats["pnl"] += usdt
        st.session_state.stats["trades"] += 1
        if net_pct > 0:
            st.session_state.stats["wins"] += 1
        t = st.session_state.stats["trades"]
        w = st.session_state.stats["wins"]
        st.session_state.stats["win_rate"] = (w / t * 100) if t else 0.0
        sk = "long" if side_c == "long" else "short"
        st.session_state.stats[f"{sk}_trades"] += 1
        st.session_state.stats[f"{sk}_pnl"] += usdt
        if net_pct > 0:
            st.session_state.stats[f"{sk}_wins"] += 1
        sign = "+" if net_pct >= 0 else ""
        level = "buy" if net_pct >= 0 else "sell"
        add_log(
            f"{'🟢' if net_pct >= 0 else '🔴'} CLOSE [{reason_exit}] {side_c.upper()} "
            f"Entry:{e_price:.4f} Exit:{close_price:.4f} "
            f"Gross:{'+' if raw_pct >= 0 else ''}{raw_pct:.2f}% "
            f"Fees:{fee_in + fee_out:.3f}% Net:{sign}{net_pct:.2f}% (${usdt:+.2f})", level
        )
        st.session_state.trade_history.append({
            "Time": st.session_state.last_cycle_time,
            "Symbol": symbol, "Side": side_c.upper(),
            "Entry": e_price, "Exit": close_price,
            "PnL%": f"{sign}{net_pct:.2f}%", "PnL$": f"${usdt:+.2f}",
            "Reason": reason_exit,
        })
        st.session_state.chart_markers.append({
            "ts": df.index[-1], "price": close_price,
            "kind": "CLOSE_WIN" if net_pct >= 0 else "CLOSE_LOSS",
            "label": f"X{sign}{net_pct:.1f}%",
            "pnl_pct": net_pct,
        })
        st.session_state[slot_key] = None

    _ml_enabled_now = st.session_state.get("ml_toggle", False)
    if _ml_enabled_now:
        signal, _gate_reason = _ml_direction_gate(
            signal=signal,
            direction=direction_filter,
            ml_pred=st.session_state.ml_prediction,
            ml_conf=st.session_state.ml_confidence,
            conf_thr=float(st.session_state.get("ml_conf_slider", 75)),
            auto_exec=st.session_state.get("auto_exec", False),
        )
        _gate_notable = any(kw in _gate_reason for kw in ("ALIGNED", "CONFLICT", "AUTO-EXEC"))
        if _gate_notable:
            _gate_level = "buy" if "AUTO-EXEC" in _gate_reason or "ALIGNED" in _gate_reason else "warn"
            add_log(f"   🤖 {_gate_reason}", _gate_level)
        elif signal == "HOLD":
            add_log(f"   ⬜ HOLD — {result['reason']}", "sys")
    elif signal == "HOLD":
        add_log(f"   ⬜ HOLD — {result['reason']}", "sys")

    if direction_filter == "Long":
        if st.session_state.position:
            _stop_hit_l = _trail_pos("position")
            if _stop_hit_l or signal == "SELL":
                _close_pos("position", price, "STOP" if _stop_hit_l else "SIGNAL")
                st.session_state.current_status = "PARKING"
        if not st.session_state.position and signal == "BUY":
            _open_pos("position", "long", price, signal)
            st.session_state.current_status = "BUY"

    elif direction_filter == "Short":
        if st.session_state.position:
            _stop_hit_s = _trail_pos("position")
            if _stop_hit_s or signal == "BUY":
                _close_pos("position", price, "STOP" if _stop_hit_s else "SIGNAL")
                st.session_state.current_status = "PARKING"
        if not st.session_state.position and signal == "SELL":
            _open_pos("position", "short", price, signal)
            st.session_state.current_status = "SELL"

    else:
        if st.session_state.position_long:
            _stop_long = _trail_pos("position_long")
            if _stop_long or signal == "SELL":
                _close_pos("position_long", price, "STOP" if _stop_long else "SIGNAL")
        if not st.session_state.position_long and signal == "BUY":
            _open_pos("position_long", "long", price, signal)

        if st.session_state.position_short:
            _stop_short = _trail_pos("position_short")
            if _stop_short or signal == "BUY":
                _close_pos("position_short", price, "STOP" if _stop_short else "SIGNAL")
        if not st.session_state.position_short and signal == "SELL":
            _open_pos("position_short", "short", price, signal)

        has_long = st.session_state.position_long is not None
        has_short = st.session_state.position_short is not None
        if has_long and has_short:
            st.session_state.current_status = "BUY"
        elif has_long:
            st.session_state.current_status = "BUY"
        elif has_short:
            st.session_state.current_status = "SELL"
        else:
            st.session_state.current_status = "PARKING"

        st.session_state.position = (
                st.session_state.position_long or st.session_state.position_short
        )


# ═══════════════════════════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════════════════════════

def render_candlestick_chart(df: pd.DataFrame, symbol: str):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#00ff99", decreasing_line_color="#ff2255",
            increasing_fillcolor="rgba(0,255,153,0.25)", decreasing_fillcolor="rgba(255,34,85,0.25)", name="OHLC",
        ), row=1, col=1)
        df2 = df.copy()
        if len(df2) >= 60:
            df2["EMA5"] = df2["Close"].ewm(span=5, adjust=False).mean()
            df2["EMA26"] = df2["Close"].ewm(span=26, adjust=False).mean()
            df2["EMA60"] = df2["Close"].ewm(span=60, adjust=False).mean()
            for name, col, color, w in [("EMA5", "EMA5", "#ffcc00", 1.8), ("EMA26", "EMA26", "#00ccff", 1.6),
                                        ("EMA60", "EMA60", "#cc44ff", 1.4)]:
                fig.add_trace(go.Scatter(x=df2.index, y=df2[col], name=name,
                                         line=dict(color=color, width=w), opacity=1.0), row=1, col=1)
        pos = st.session_state.position
        if pos:
            fig.add_hline(y=pos["price"], line_color="#ffcc00", line_dash="dash",
                          line_width=2, annotation_text="ENTRY", row=1, col=1)
            fig.add_hline(y=pos["stop_loss"], line_color="#ff2255", line_dash="dot",
                          line_width=2, annotation_text="STOP", row=1, col=1)
            if "trail_price" in pos:
                fig.add_hline(y=pos["trail_price"], line_color="#cc44ff", line_dash="dot",
                              line_width=2, annotation_text="TRAIL", row=1, col=1)
        markers = st.session_state.get("chart_markers", [])
        if markers:
            chart_start = df.index[0]
            chart_end = df.index[-1]

            buy_ts, buy_px, buy_lbl = [], [], []
            sell_entry_ts, sell_px, sell_lbl = [], [], []
            close_win_ts, cw_px, cw_lbl = [], [], []
            close_loss_ts, cl_px, cl_lbl = [], [], []
            ranging_ts, ranging_px, ranging_lbl = [], [], []

            for m in markers:
                ts = pd.Timestamp(m["ts"])
                ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
                if ts < chart_start or ts > chart_end:
                    continue
                k = m["kind"]
                if k == "BUY":
                    buy_ts.append(ts);
                    buy_px.append(m["price"]);
                    buy_lbl.append(m["label"])
                elif k == "SELL_ENTRY":
                    sell_entry_ts.append(ts);
                    sell_px.append(m["price"]);
                    sell_lbl.append(m["label"])
                elif k == "CLOSE_WIN":
                    close_win_ts.append(ts);
                    cw_px.append(m["price"]);
                    cw_lbl.append(m["label"])
                elif k == "CLOSE_LOSS":
                    close_loss_ts.append(ts);
                    cl_px.append(m["price"]);
                    cl_lbl.append(m["label"])
                elif k == "RANGING":
                    ranging_ts.append(ts);
                    ranging_px.append(m["price"]);
                    ranging_lbl.append(m["label"])

            if buy_ts:
                fig.add_trace(go.Scatter(
                    x=buy_ts, y=[p * 0.9985 for p in buy_px],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=14, color="#00ff99",
                                line=dict(color="#ffffff", width=1)),
                    text=buy_lbl, textposition="bottom center",
                    textfont=dict(color="#00ff99", size=9, family="Share Tech Mono"),
                    name="BUY", showlegend=True,
                ), row=1, col=1)

            if sell_entry_ts:
                fig.add_trace(go.Scatter(
                    x=sell_entry_ts, y=[p * 1.0015 for p in sell_px],
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=14, color="#ff2255",
                                line=dict(color="#ffffff", width=1)),
                    text=sell_lbl, textposition="top center",
                    textfont=dict(color="#ff2255", size=9, family="Share Tech Mono"),
                    name="SELL", showlegend=True,
                ), row=1, col=1)

            if close_win_ts:
                fig.add_trace(go.Scatter(
                    x=close_win_ts, y=[p * 1.002 for p in cw_px],
                    mode="markers+text",
                    marker=dict(symbol="circle", size=11, color="#ffcc00",
                                line=dict(color="#ffffff", width=1)),
                    text=cw_lbl, textposition="top center",
                    textfont=dict(color="#ffcc00", size=9, family="Share Tech Mono"),
                    name="WIN", showlegend=True,
                ), row=1, col=1)

            if close_loss_ts:
                fig.add_trace(go.Scatter(
                    x=close_loss_ts, y=[p * 0.998 for p in cl_px],
                    mode="markers+text",
                    marker=dict(symbol="circle", size=11, color="#cc44ff",
                                line=dict(color="#ffffff", width=1)),
                    text=cl_lbl, textposition="bottom center",
                    textfont=dict(color="#cc44ff", size=9, family="Share Tech Mono"),
                    name="LOSS", showlegend=True,
                ), row=1, col=1)

            if ranging_ts:
                fig.add_trace(go.Scatter(
                    x=ranging_ts, y=[p * 1.001 for p in ranging_px],
                    mode="markers+text",
                    marker=dict(symbol="x", size=12, color="#ffffff",
                                line=dict(color="#ffaa00", width=2)),
                    text=ranging_lbl, textposition="top center",
                    textfont=dict(color="#ffaa00", size=9, family="Share Tech Mono"),
                    name="RANGING", showlegend=True,
                ), row=1, col=1)

        colors = ["rgba(0,255,140,0.85)" if c >= o else "rgba(255,30,80,0.85)"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                             marker_color=colors, name="Volume", showlegend=False), row=2, col=1)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#060810", plot_bgcolor="#08091a",
            font=dict(family="Share Tech Mono", color="#a0b8d8", size=11),
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0), height=480,
            legend=dict(bgcolor="rgba(6,8,16,0.8)", bordercolor="#1a3060", borderwidth=1,
                        font=dict(color="#a0b8d8", size=10), x=0.01, y=0.99),
            title=dict(text=f"  {symbol}", font=dict(color="#00e5ff", size=14, family="Share Tech Mono"), x=0),
        )
        fig.update_xaxes(gridcolor="#182040", zeroline=False, tickfont=dict(color="#7090b8"))
        fig.update_yaxes(gridcolor="#182040", zeroline=False, tickfont=dict(color="#7090b8"))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    except ImportError:
        st.info("Install plotly: `pip install plotly`")


# ═══════════════════════════════════════════════════════════════════════════
# TRADING WINDOW FLOATING PANEL
# ═══════════════════════════════════════════════════════════════════════════

def render_trading_window_panel(strategy: str):
    windows = st.session_state.setdefault("trading_windows", {})
    cfg = windows.setdefault(strategy, {
        "enabled": False, "start": "00:00", "end": "23:59",
        "days": list(range(7)), "tz": "UTC"
    })

    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    PRESETS = {
        "24/7 (no restriction)": (False, "00:00", "23:59", list(range(7))),
        "London session": (True, "08:00", "17:00", [0, 1, 2, 3, 4]),
        "New York session": (True, "13:00", "22:00", [0, 1, 2, 3, 4]),
        "Asian session": (True, "00:00", "09:00", [0, 1, 2, 3, 4, 5, 6]),
        "Crypto prime (EU+US)": (True, "07:00", "22:00", [0, 1, 2, 3, 4, 5, 6]),
        "Scalping window": (True, "08:30", "12:00", [0, 1, 2, 3, 4]),
        "Custom": None,
    }

    st.markdown(f"""
<div style="background:linear-gradient(135deg,#080e20,#0a1228);
border:1px solid #00e5ff44;border-radius:10px;padding:18px 22px;margin:10px 0;
box-shadow:0 0 20px #00e5ff18">
<div style="font-family:'Share Tech Mono',monospace;font-size:0.9rem;color:#00e5ff;
letter-spacing:0.12em;margin-bottom:14px">
⏱ TRADING WINDOW — {strategy.upper()}</div>""",
                unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([2, 2, 1.5])

    with col_left:
        st.markdown('<div style="color:#7eb8e8;font-size:0.78rem;font-weight:600;margin-bottom:4px">PRESET</div>',
                    unsafe_allow_html=True)
        preset_choice = st.selectbox("Preset", list(PRESETS.keys()),
                                     key=f"tw_preset_{strategy}", label_visibility="collapsed")

        st.markdown('<div style="color:#7eb8e8;font-size:0.78rem;font-weight:600;margin:8px 0 4px">ENABLE WINDOW</div>',
                    unsafe_allow_html=True)
        enabled = st.toggle("Restrict trading hours", value=cfg.get("enabled", False),
                            key=f"tw_enabled_{strategy}")

        st.markdown(
            '<div style="color:#7eb8e8;font-size:0.78rem;font-weight:600;margin:8px 0 4px">START TIME (UTC)</div>',
            unsafe_allow_html=True)
        start_val = cfg.get("start", "00:00")
        start_time = st.text_input("Start", value=start_val,
                                   key=f"tw_start_{strategy}", label_visibility="collapsed",
                                   placeholder="HH:MM e.g. 09:00")

        st.markdown(
            '<div style="color:#7eb8e8;font-size:0.78rem;font-weight:600;margin:8px 0 4px">END TIME (UTC)</div>',
            unsafe_allow_html=True)
        end_val = cfg.get("end", "23:59")
        end_time = st.text_input("End", value=end_val,
                                 key=f"tw_end_{strategy}", label_visibility="collapsed",
                                 placeholder="HH:MM e.g. 17:00")

    with col_mid:
        st.markdown('<div style="color:#7eb8e8;font-size:0.78rem;font-weight:600;margin-bottom:4px">TRADING DAYS</div>',
                    unsafe_allow_html=True)
        current_days = cfg.get("days", list(range(7)))
        selected_days = []
        for i, day in enumerate(DAY_NAMES):
            checked = i in current_days
            if st.checkbox(day, value=checked, key=f"tw_day_{strategy}_{i}"):
                selected_days.append(i)

        st.markdown(
            '<div style="color:#7eb8e8;font-size:0.78rem;font-weight:600;margin:12px 0 4px">CURRENT STATUS</div>',
            unsafe_allow_html=True)
        allowed, reason = is_within_trading_window(strategy)
        status_color = "#00ff99" if (not enabled or allowed) else "#ff2255"
        status_text = "✅ TRADING ALLOWED" if (not enabled or allowed) else "🚫 TRADING BLOCKED"
        html_status = (
            f'<div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;'
            f'color:{status_color};padding:6px 0">{status_text}</div>'
            f'<div style="color:#5a7a9a;font-size:0.72rem">{reason}</div>'
        )
        st.markdown(html_status, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div style="color:#7eb8e8;font-size:0.78rem;font-weight:600;margin-bottom:8px">ACTIONS</div>',
                    unsafe_allow_html=True)

        if st.button("💾 Save", width="stretch", key=f"tw_save_{strategy}"):
            p = PRESETS.get(preset_choice)
            if p is not None:
                p_enabled, p_start, p_end, p_days = p
                windows[strategy] = {
                    "enabled": p_enabled, "start": p_start,
                    "end": p_end, "days": p_days, "tz": "UTC"
                }
                add_log(f"⏱ [{strategy}] window set to preset: {preset_choice}", "info")
            else:
                import re as _re
                time_ok = bool(_re.match(r"^\d{2}:\d{2}$", start_time)) and bool(_re.match(r"^\d{2}:\d{2}$", end_time))
                if not time_ok:
                    st.error("⚠️ Time format must be HH:MM")
                else:
                    windows[strategy] = {
                        "enabled": enabled,
                        "start": start_time,
                        "end": end_time,
                        "days": selected_days if selected_days else list(range(7)),
                        "tz": "UTC",
                    }
                    days_str = " ".join(DAY_NAMES[d] for d in (selected_days or list(range(7))))
                    status = "ON" if enabled else "OFF"
                    add_log(f"⏱ [{strategy}] window {status}: {start_time}–{end_time} UTC | {days_str}", "info")
            st.session_state.trading_windows = windows
            save_trading_windows()
            st.rerun()

        if st.button("🔄 Reset", width="stretch", key=f"tw_reset_{strategy}"):
            windows[strategy] = {
                "enabled": False, "start": "00:00",
                "end": "23:59", "days": list(range(7)), "tz": "UTC"
            }
            st.session_state.trading_windows = windows
            save_trading_windows()
            add_log(f"⏱ [{strategy}] window reset to 24/7", "info")
            st.rerun()

        if st.button("✖ Close", width="stretch", key=f"tw_close_{strategy}"):
            st.session_state.tw_panel_open = False
            st.rerun()

        st.markdown(
            '<div style="color:#7eb8e8;font-size:0.72rem;font-weight:600;margin:14px 0 4px">ALL STRATEGIES</div>',
            unsafe_allow_html=True)
        for strat in ["Momentum", "Kalman", "Scalping", "Enhanced"]:
            w = windows.get(strat, {})
            if w.get("enabled", False):
                a, _ = is_within_trading_window(strat)
                dot = "🟢" if a else "🔴"
                st.markdown(f'<div style="font-family:monospace;font-size:0.7rem;color:#8a9abc">'
                            f'{dot} {strat}: {w["start"]}–{w["end"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-family:monospace;font-size:0.7rem;color:#3a5a7a">⚪ {strat}: 24/7</div>',
                            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def build_session_summary(stop_reason: str = "Manual Stop") -> dict:
    now = datetime.now(timezone.utc)
    start_time = st.session_state.get("session_start_time")
    duration = ""
    if start_time:
        delta = now - start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        duration = f"{h:02d}:{m:02d}:{s:02d}"

    s = st.session_state.stats
    bal = st.session_state.virtual_balance["USDT"]
    start_bal = st.session_state.get("session_start_balance", 5000.0)
    net_pnl = bal - start_bal

    summary = {
        "stop_reason": stop_reason,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S UTC") if start_time else "—",
        "end_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration": duration,
        "bars_processed": st.session_state.bars_processed,
        "start_balance": start_bal,
        "end_balance": round(bal, 4),
        "net_pnl": round(net_pnl, 4),
        "net_pnl_pct": round((net_pnl / start_bal * 100) if start_bal else 0, 2),
        "total_trades": s["trades"],
        "total_wins": s["wins"],
        "win_rate": round(s["win_rate"], 1),
        "long_trades": s["long_trades"],
        "long_wins": s["long_wins"],
        "long_pnl": round(s["long_pnl"], 4),
        "short_trades": s["short_trades"],
        "short_wins": s["short_wins"],
        "short_pnl": round(s["short_pnl"], 4),
        "trade_history": list(st.session_state.trade_history),
    }
    return summary


def render_session_summary():
    summary = st.session_state.get("session_summary")
    if not summary:
        return

    stop_color = {"Manual Stop": "#ffcc00", "Emergency Stop": "#ff2255",
                  "Window Closed": "#cc44ff"}.get(summary["stop_reason"], "#00e5ff")
    net_col = "#00ff99" if summary["net_pnl"] >= 0 else "#ff2255"

    st.markdown(f"""
<div style="background:linear-gradient(135deg,#08101e,#0c1428);
border:2px solid {stop_color}44;border-radius:10px;padding:20px 24px;margin:12px 0;
box-shadow:0 0 24px {stop_color}18">
<div style="font-family:Share Tech Mono,monospace;font-size:1rem;
color:{stop_color};letter-spacing:0.14em;margin-bottom:16px">
📊 SESSION SUMMARY — {summary['stop_reason'].upper()}</div>""",
                unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown('<div style="color:#7eb8e8;font-size:0.72rem;font-weight:600">SESSION START</div>'
                    f'<div style="font-family:monospace;color:#ccd6f6;font-size:0.85rem">{summary["start_time"]}</div>',
                    unsafe_allow_html=True)
    with r1c2:
        st.markdown('<div style="color:#7eb8e8;font-size:0.72rem;font-weight:600">SESSION END</div>'
                    f'<div style="font-family:monospace;color:#ccd6f6;font-size:0.85rem">{summary["end_time"]}</div>',
                    unsafe_allow_html=True)
    with r1c3:
        dur_html = (
            '<div style="color:#7eb8e8;font-size:0.72rem;font-weight:600">DURATION</div>'
            f'<div style="font-family:Share Tech Mono,monospace;color:#00e5ff;font-size:1rem">{summary["duration"]}</div>'
        )
        st.markdown(dur_html, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    metrics = [
        (m1, "Start Balance", f"${summary['start_balance']:.2f}", "#ccd6f6"),
        (m2, "End Balance", f"${summary['end_balance']:.2f}", "#00e5ff"),
        (m3, "Net P&L", f"${summary['net_pnl']:+.2f}", net_col),
        (m4, "Net P&L %", f"{summary['net_pnl_pct']:+.2f}%", net_col),
        (m5, "Win Rate", f"{summary['win_rate']:.1f}%", "#ffcc00"),
        (m6, "Bars Processed", str(summary["bars_processed"]), "#8a9abc"),
    ]
    for col, label, val, color in metrics:
        with col:
            st.markdown(
                f'<div style="background:#0a1020;border:1px solid #1a3060;border-radius:6px;'
                f'padding:10px 12px;text-align:center">'
                f'<div style="color:#6a8ab0;font-size:0.68rem;font-weight:600;letter-spacing:0.08em">{label}</div>'
                f'<div style="font-family:Share Tech Mono,monospace;color:{color};'
                f'font-size:1.1rem;font-weight:700;margin-top:4px">{val}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    for col, side, trades, wins, pnl in [
        (bc1, "LONG", summary["long_trades"], summary["long_wins"], summary["long_pnl"]),
        (bc2, "SHORT", summary["short_trades"], summary["short_wins"], summary["short_pnl"]),
    ]:
        wr = round(wins / trades * 100, 1) if trades else 0
        col_v = "#00ff99" if pnl >= 0 else "#ff2255"
        side_c = "#00ff99" if side == "LONG" else "#ff2255"
        with col:
            st.markdown(
                f'<div style="background:#08101e;border:1px solid {side_c}33;border-radius:6px;padding:12px 16px">'
                f'<div style="color:{side_c};font-family:Share Tech Mono,monospace;'
                f'font-weight:700;font-size:0.85rem;margin-bottom:8px">{side}</div>'
                f'<div style="color:#8a9abc;font-size:0.78rem;line-height:2">'
                f'Trades: <span style="color:#ccd6f6">{trades}</span> &nbsp;'
                f'Wins: <span style="color:#ccd6f6">{wins}</span> &nbsp;'
                f'WR: <span style="color:#ffcc00">{wr:.1f}%</span><br>'
                f'P&L: <span style="color:{col_v};font-weight:700">${pnl:+.4f}</span>'
                f'</div></div>',
                unsafe_allow_html=True
            )

    if summary["trade_history"]:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        with st.expander(f"📋 Trade History ({len(summary['trade_history'])} trades)", expanded=False):
            st.dataframe(
                pd.DataFrame(summary["trade_history"]),
                width='stretch', hide_index=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns([2, 2, 1])
    with ec1:
        try:
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                rows = [
                    ["Session Summary", ""],
                    ["Stop Reason", summary["stop_reason"]],
                    ["Start Time", summary["start_time"]],
                    ["End Time", summary["end_time"]],
                    ["Duration", summary["duration"]],
                    ["", ""],
                    ["FINANCIAL", ""],
                    ["Start Balance ($)", summary["start_balance"]],
                    ["End Balance ($)", summary["end_balance"]],
                    ["Net P&L ($)", summary["net_pnl"]],
                    ["Net P&L (%)", summary["net_pnl_pct"]],
                    ["", ""],
                    ["TRADES", ""],
                    ["Total Trades", summary["total_trades"]],
                    ["Total Wins", summary["total_wins"]],
                    ["Win Rate (%)", summary["win_rate"]],
                    ["Bars Processed", summary["bars_processed"]],
                    ["", ""],
                    ["LONG SIDE", ""],
                    ["Long Trades", summary["long_trades"]],
                    ["Long Wins", summary["long_wins"]],
                    ["Long P&L ($)", summary["long_pnl"]],
                    ["", ""],
                    ["SHORT SIDE", ""],
                    ["Short Trades", summary["short_trades"]],
                    ["Short Wins", summary["short_wins"]],
                    ["Short P&L ($)", summary["short_pnl"]],
                ]
                pd.DataFrame(rows, columns=["Metric", "Value"]).to_excel(
                    writer, sheet_name="Summary", index=False)
                if summary["trade_history"]:
                    pd.DataFrame(summary["trade_history"]).to_excel(
                        writer, sheet_name="Trades", index=False)
            buf.seek(0)
            fname = datetime.now().strftime("session_%Y%m%d_%H%M.xlsx")
            st.download_button(
                "📥 Download Session Excel",
                data=buf, file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="session_excel_download"
            )
        except Exception as e:
            st.error(f"Excel export error: {e}")
    with ec2:
        if st.button("✖ Dismiss Summary", key="dismiss_summary"):
            st.session_state.session_summary = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# LOG RENDERER
# ═══════════════════════════════════════════════════════════════════════════

LOG_COLORS = {"buy": "#00ff99", "sell": "#ff2255", "info": "#00e5ff", "warn": "#ffcc00", "err": "#ff4444",
              "sys": "#4a8a6a"}


def render_log():
    lines = ""
    for e in reversed(st.session_state.logs[-60:]):
        color = LOG_COLORS.get(e["level"], "#8a9abc")
        msg = e["msg"].replace("<", "&lt;").replace(">", "&gt;")
        lines += f'<div style="color:{color}"><span style="color:#3a6a9a">[{e["ts"]}]</span> {msg}</div>\n'
    st.markdown(f'<div class="log-box">{lines}</div>', unsafe_allow_html=True)


def render_status_badge(status: str):
    cls = {"BUY": "status-buy", "SELL": "status-sell", "PARKING": "status-parking"}.get(status.upper(),
                                                                                        "status-parking")
    st.markdown(f'<span class="status-badge {cls}">⬤ {status.upper()}</span>', unsafe_allow_html=True)


ml_enabled = False
ml_model = "XGBoost"
ml_conf_thr = 75
auto_exec = False

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px">
      <div style="font-family:'Share Tech Mono',monospace;font-size:1.05rem;color:#00d4ff;letter-spacing:0.12em">📈 ABOULDAHAB MCS</div>
      <div style="color:#3a4a6a;font-size:0.68rem;letter-spacing:0.15em;margin-top:4px">TRADING PLATFORM v10.0</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Trading Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("Mode", ["Demo", "Live", "Backtest"], key="mode_select", label_visibility="collapsed")

    st.markdown('<div class="section-header">Market</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        symbol = st.selectbox("Symbol", ["SOL-USDT", "BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT"],
                              key="symbol_select")
    with c2:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1H", "4H", "1D"], index=2,
                                key="interval_select")

    st.markdown('<div class="section-header">Strategy</div>', unsafe_allow_html=True)
    strategy = st.selectbox("Strategy", ["Momentum", "Kalman", "Scalping"], key="strategy_select",
                            label_visibility="collapsed")

    _engine_on = st.session_state.get("use_real_engine", True)
    _is_real_engine_strategy = strategy in REAL_ENGINE_STRATEGIES

    # Eagerly attempt the load as soon as the strategy is selected, so the
    # badge below reflects the real outcome instead of a stale "Unknown"
    # that only updates after Start/backtest is actually clicked.
    if _engine_on and _is_real_engine_strategy and strategy not in st.session_state.strategy_status:
        try:
            get_cached_strategy_class(strategy, for_backtest=False)
        except Exception as e:
            st.session_state.strategy_status[strategy] = f"Error: {str(e)[:50]}"
            add_log(f"❌ Eager load of {strategy} failed: {e}", "err")

    # Check if real strategy is loaded
    _strategy_status = st.session_state.strategy_status.get(strategy, "Unknown")
    _status_color = "🟢" if "Loaded" in _strategy_status else "🟡" if "Error" not in _strategy_status else "🔴"

    _badge = ("🟢 Real engine" if (_engine_on and _is_real_engine_strategy and "Loaded" in _strategy_status)
              else "🟡 Desktop strategy not found" if (_engine_on and _is_real_engine_strategy)
    else "🟡 Simplified (Enhanced has no real file yet)" if strategy == "Enhanced"
    else "⚪ Simplified (real engine off)")

    with st.expander(f"⚙️ Engine — {_badge}", expanded=False):
        st.toggle(
            "Use real desktop strategy classes (Momentum/Kalman/Scalping)",
            key="use_real_engine",
            help="When on, Momentum/Kalman/Scalping run the actual desktop "
                 "strategy classes instead of the simplified built-in signal functions.",
        )

        # Show status of desktop strategy files
        if _is_real_engine_strategy:
            st.markdown(f"**Strategy Status:** {_status_color} {_strategy_status}")

            # Check if files exist
            strategy_files = find_strategy_files()
            if strategy_files:
                st.caption(f"Found {len(strategy_files)} strategy file(s)")
            else:
                st.caption("⚠️ No strategy files found in current directory")

        st.slider(
            "Entry confidence threshold %", 50, 95,
            int(st.session_state.get("confidence_threshold", 65.0)),
            key="confidence_threshold",
            help="Mirrors the desktop app's confidence_var slider.",
        )
        if strategy == "Momentum" and st.session_state.get("direction", "Long") in ("Short", "Both"):
            st.caption("⚠️ Short entries are ignored for Momentum — the supplied "
                       "MomentumStrategy file has no short-entry execution path.")

    st.markdown('<div class="section-header">Trading</div>', unsafe_allow_html=True)
    direction = st.radio("Direction", ["Long", "Short", "Both"], key="direction", horizontal=True)
    stop_loss_pct = st.slider("Stop Loss %", 0.5, 5.0, 1.5, key="stop_loss")
    trailing_pct = st.slider("Trailing %", 0.0, 2.0, 0.8, key="trailing")
    order_size_pct = st.slider("Order Size %", 5, 100, 30, key="order_size")

    # ── CLOCK SYNC CONTROLS ──────────────────────────────────────────────
    with st.expander("⏱ Clock Sync Settings", expanded=False):
        st.toggle(
            "Sync to clock (candle closes)",
            key="sync_to_clock",
            value=st.session_state.get("sync_to_clock", True),
            help="When ON, trades execute exactly at candle closes. When OFF, uses fixed interval polling."
        )

        if not st.session_state.get("sync_to_clock", True):
            st.slider("Refresh interval (seconds)", 1, 60,
                      st.session_state.get("refresh_interval", 10),
                      key="refresh_interval",
                      help="How often to poll for new data when clock sync is OFF")

        if st.session_state.get("sync_to_clock", True):
            st.toggle(
                "Use WebSocket trigger",
                key="use_ws_trigger",
                value=st.session_state.get("use_ws_trigger", True),
                help="When ON, uses OKX WebSocket for precise candle close timing. "
                     "Falls back to clock sync if unavailable."
            )

            st.number_input(
                "Clock sync buffer (seconds)",
                min_value=1, max_value=10,
                value=st.session_state.get("clock_sync_buffer", 3),
                key="clock_sync_buffer",
                help="Seconds to wait after candle close before fetching data"
            )

        # Show current trigger source
        if st.session_state.get("last_trigger_source"):
            trigger_emoji = "📡" if st.session_state.last_trigger_source == "websocket" else "⏰"
            trigger_label = "WebSocket" if st.session_state.last_trigger_source == "websocket" else "Clock"
            st.caption(f"{trigger_emoji} Last trigger: {trigger_label}")

    st.markdown('<div class="section-header">Connect / Control</div>', unsafe_allow_html=True)

    if mode == "Live":
        if st.button("🔌 Connect API", width="stretch"):
            check_connection("live")
    else:
        st.info("ℹ️ Demo mode — no API keys required.", icon="ℹ️")

    # Strategy-specific status
    if _is_real_engine_strategy and _engine_on:
        if st.session_state.strategy_status.get(strategy, ""):
            st.caption(f"📊 {strategy}: {st.session_state.strategy_status[strategy]}")

    st.markdown("---")
    st.markdown("**Controls**")

    topc1, topc2 = st.columns(2)
    with topc1:
        if st.button("▶️ START" if not st.session_state.trading_running else "⏹ STOP", width="stretch"):
            if not st.session_state.trading_running:
                st.session_state.trading_running = True
                st.session_state.session_start_time = datetime.now(timezone.utc)
                st.session_state.session_start_balance = st.session_state.virtual_balance["USDT"]
                st.session_state.bars_processed = 0
                st.session_state.stats = {
                    "trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0,
                    "long_trades": 0, "long_wins": 0, "long_pnl": 0.0,
                    "short_trades": 0, "short_wins": 0, "short_pnl": 0.0,
                }
                st.session_state.trade_history = []
                add_log(f"🚀 Trading started: {symbol} {interval} {strategy} (mode: {mode})", "buy")

                # Run initial cycle
                run_trading_cycle(symbol, interval, strategy, stop_loss_pct, trailing_pct, order_size_pct, mode)
                st.rerun()
            else:
                st.session_state.trading_running = False
                st.session_state.session_summary = build_session_summary("Manual Stop")
                add_log("⏹ Trading stopped.", "warn")
                st.rerun()

    with topc2:
        if st.button("🔄 Force cycle", width="stretch"):
            add_log("🔄 Manual cycle triggered", "sys")
            run_trading_cycle(symbol, interval, strategy, stop_loss_pct, trailing_pct, order_size_pct, mode)
            st.rerun()

    st.caption(f"Balance: ${st.session_state.virtual_balance['USDT']:.2f} USDT")
    st.caption(f"Position: {st.session_state.current_status}")
    if st.session_state.position:
        pos = st.session_state.position
        st.caption(f"Entry: {pos['price']:.4f} SL: {pos['stop_loss']:.4f}")

    if st.button("💾 Save Settings", width="stretch"):
        save_strategy_settings(st.session_state.strategy_settings)
        add_log("✅ Settings saved", "info")

    st.markdown('<div class="section-header">⏱ Trading Windows</div>', unsafe_allow_html=True)
    if st.button("⏱ Open Window Panel" if not st.session_state.tw_panel_open else "⏱ Close Window Panel",
                 width="stretch"):
        st.session_state.tw_panel_open = not st.session_state.tw_panel_open
        st.rerun()

    if st.session_state.tw_panel_open:
        render_trading_window_panel(strategy)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["📈 Live Trading", "📊 Backtest", "🤖 ML Predictions", "⚙️ Parameters", "❓ Help"])

# ─── TAB 0: LIVE TRADING ────────────────────────────────────────────────────
with tabs[0]:
    banner_col1, banner_col2, banner_col3, banner_col4 = st.columns([2.2, 1.6, 1.2, 1])
    with banner_col1:
        st.markdown(f'<div class="banner-title">📈 {symbol} · {interval}</div>', unsafe_allow_html=True)
    with banner_col2:
        if st.session_state.last_price:
            price_color = "#00ff99" if st.session_state.last_price > 0 else "#ff2255"
            st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:1.5rem;'
                        f'color:{price_color};font-weight:700">${st.session_state.last_price:.4f}</div>',
                        unsafe_allow_html=True)
    with banner_col3:
        render_status_badge(st.session_state.current_status)
        render_window_indicator(strategy)
    with banner_col4:
        if st.session_state.trading_running:
            st.markdown('<span class="pulse-dot"></span> LIVE', unsafe_allow_html=True)
        else:
            st.markdown('⏸ IDLE', unsafe_allow_html=True)

    # Live trading stats
    stmt1, stmt2, stmt3, stmt4, stmt5, stmt6 = st.columns(6)
    stats = st.session_state.stats
    stmt1.metric("Trades", stats["trades"])
    stmt2.metric("Wins", stats["wins"])
    stmt3.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    stmt4.metric("P&L", f"${stats['pnl']:.2f}", delta=stats['pnl'])
    stmt5.metric("Balance", f"${st.session_state.virtual_balance['USDT']:.2f}")
    stmt6.metric("Bars", st.session_state.bars_processed)

    # Chart
    if st.session_state.market_data is not None and len(st.session_state.market_data) > 10:
        render_candlestick_chart(st.session_state.market_data, symbol)
    else:
        st.info("📊 Load market data by clicking START or Force cycle.")

    # Logs
    st.markdown('<div class="section-header">📋 Event Log</div>', unsafe_allow_html=True)
    render_log()

    # ── CLOCK-ALIGNED CONTINUOUS AUTO-UPDATE ────────────────────────────
    # While trading is running, we wait for the next candle close using the
    # wait_for_next_trigger() function, then run the trading cycle. This
    # ensures we process exactly at candle boundaries, not on a fixed timer.
    if st.session_state.trading_running:
        # ── Check if we should use clock sync or fixed interval ────────
        use_clock_sync = st.session_state.get("sync_to_clock", True)

        if use_clock_sync:
            # ── CLOCK-ALIGNED MODE ──────────────────────────────────────
            # Wait for the next candle close before running the cycle
            trigger_source = wait_for_next_trigger(symbol, interval)

            # Log trigger source if it changed
            if trigger_source != st.session_state.get("last_trigger_source"):
                st.session_state.last_trigger_source = trigger_source
                trigger_emoji = "📡" if trigger_source == "websocket" else "⏰"
                trigger_label = "WebSocket" if trigger_source == "websocket" else "Clock"
                add_log(f"{trigger_emoji} Trigger: {trigger_label} ({interval} candle closed)", "sys")

            # Run the trading cycle with the new candle data
            run_trading_cycle(symbol, interval, strategy, stop_loss_pct, trailing_pct, order_size_pct, mode)

            # Rerun immediately to process the next candle
            st.rerun()

        else:
            # ── FIXED INTERVAL MODE (Legacy behavior) ──────────────────
            run_trading_cycle(symbol, interval, strategy, stop_loss_pct, trailing_pct, order_size_pct, mode)
            time.sleep(max(1, st.session_state.get("refresh_interval", 10)))
            st.rerun()

# ─── TAB 1: BACKTEST ──────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">📊 Backtest Engine</div>', unsafe_allow_html=True)

    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_symbol = st.selectbox("Symbol", ["SOL-USDT", "BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT"],
                                 key="bt_symbol", index=0)
        bt_interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1H", "4H", "1D"],
                                   key="bt_interval", index=2)
    with bt_col2:
        bt_strategy = st.selectbox("Strategy", ["Momentum", "Kalman", "Scalping"], key="bt_strategy")
        bt_exchange = st.selectbox("Exchange", ["binance", "okx"], key="bt_exchange")
    with bt_col3:
        bt_start = st.date_input("Start", value=datetime.now() - timedelta(days=90), key="bt_start")
        bt_end = st.date_input("End", value=datetime.now(), key="bt_end")
        bt_capital = st.number_input("Capital ($)", min_value=1000, value=10000, step=1000, key="bt_capital")

    bt_col4, bt_col5, bt_col6 = st.columns(3)
    with bt_col4:
        st.toggle("Enable Monte Carlo", key="bt_mc", value=False)
        mc_sims = st.number_input("Simulations", min_value=100, max_value=5000, value=1000, step=100,
                                  key="bt_mc_sims", disabled=not st.session_state.bt_mc)
    with bt_col5:
        st.toggle("Enable Parameter Optimization", key="bt_opt", value=False)
        opt_metric = st.selectbox("Optimize For", ["Return %", "Sharpe Ratio", "Win Rate", "Profit Factor"],
                                  key="bt_opt_metric", disabled=not st.session_state.bt_opt)
    with bt_col6:
        st.toggle("Use Real Strategy Classes", key="bt_use_real", value=True,
                  help="Use desktop strategy files instead of simplified built-in signals")
        if st.button("🚀 Run Backtest", width="stretch", key="bt_run"):
            st.session_state.backtest_running = True
            add_log("🚀 Starting backtest...", "info")

            with st.spinner("Running backtest..."):
                try:
                    start_ts = pd.Timestamp(bt_start).tz_localize('UTC')
                    end_ts = pd.Timestamp(bt_end).tz_localize('UTC') + timedelta(days=1)

                    df = fetch_historical_data_ccxt(
                        bt_symbol, bt_exchange,
                        start=start_ts, end=end_ts,
                        interval=bt_interval
                    )

                    if df is None or len(df) < 30:
                        st.error("❌ Not enough data fetched. Try a different date range or symbol.")
                        add_log("❌ Backtest failed: insufficient data", "err")
                        st.session_state.backtest_running = False
                        st.rerun()

                    # Get strategy class
                    use_real = st.session_state.bt_use_real
                    StrategyClass, error = get_cached_strategy_class(bt_strategy, for_backtest=True) if use_real else (
                        None, None)

                    if use_real and StrategyClass is None:
                        st.warning(f"⚠️ Could not load real strategy: {error}. Falling back to simplified.")
                        add_log(f"⚠️ Backtest: real strategy not found, using simplified", "warn")
                        StrategyClass, error = None, None

                    if StrategyClass is None:
                        # Use simplified built-in strategy
                        if bt_strategy == "Momentum":
                            from backtesting import Strategy


                            class MomentumBacktest(Strategy):
                                def init(self):
                                    self.ema5 = self.I(lambda x: pd.Series(x).ewm(span=5, adjust=False).mean(),
                                                       self.data.Close)
                                    self.ema26 = self.I(lambda x: pd.Series(x).ewm(span=26, adjust=False).mean(),
                                                        self.data.Close)
                                    self.ema60 = self.I(lambda x: pd.Series(x).ewm(span=60, adjust=False).mean(),
                                                        self.data.Close)
                                    self.rsi = self.I(lambda x: 100 - (100 / (
                                                1 + pd.Series(x).diff().clip(lower=0).ewm(span=14).mean() / (
                                            -pd.Series(x).diff().clip(upper=0)).ewm(span=14).mean())), self.data.Close)
                                    self.atr = self.I(lambda x: pd.Series(x).rolling(14).max().ewm(span=14).mean(),
                                                      self.data.High - self.data.Low)

                                def next(self):
                                    if len(self.data) < 2:
                                        return
                                    if self.position:
                                        stop = self.position.entry_price * (1 - 0.015)
                                        if self.data.Close[-1] <= stop:
                                            self.position.close()
                                        return
                                    if self.data.ema5[-1] > self.data.ema26[-1] > self.data.ema60[-1] and self.data.rsi[
                                        -1] > 40 and self.data.rsi[-1] < 70:
                                        self.buy(sl=self.data.Close[-1] * 0.985)
                        elif bt_strategy == "Kalman":
                            from backtesting import Strategy


                            class KalmanBacktest(Strategy):
                                def init(self):
                                    self.rsi = self.I(lambda x: 100 - (100 / (
                                                1 + pd.Series(x).diff().clip(lower=0).ewm(span=14).mean() / (
                                            -pd.Series(x).diff().clip(upper=0)).ewm(span=14).mean())), self.data.Close)
                                    self.atr = self.I(lambda x: pd.Series(x).rolling(14).max().ewm(span=14).mean(),
                                                      self.data.High - self.data.Low)
                                    close_vals = self.data.Close
                                    x_vals = [float(close_vals[0])]
                                    p = 1.0
                                    for c in close_vals[1:]:
                                        p += 0.01
                                        k = p / (p + 500)
                                        x_vals.append(x_vals[-1] + k * (float(c) - x_vals[-1]))
                                        p = (1 - k) * p
                                    self.kalman = self.I(lambda x: x_vals[:len(x)], self.data.Close)

                                def next(self):
                                    if len(self.data) < 2 or self.position:
                                        return
                                    diff = self.data.Close[-1] - self.kalman[-1]
                                    strength = abs(diff) / (self.data.atr[-1] + 1e-9)
                                    if diff > 0 and strength > 0.7 and self.data.rsi[-1] > 40 and self.data.rsi[
                                        -1] < 65:
                                        self.buy(sl=self.data.Close[-1] * 0.985)
                        elif bt_strategy == "Scalping":
                            from backtesting import Strategy


                            class ScalpingBacktest(Strategy):
                                def init(self):
                                    self.ema5 = self.I(lambda x: pd.Series(x).ewm(span=5, adjust=False).mean(),
                                                       self.data.Close)
                                    self.ema26 = self.I(lambda x: pd.Series(x).ewm(span=26, adjust=False).mean(),
                                                        self.data.Close)
                                    self.rsi = self.I(lambda x: 100 - (100 / (
                                                1 + pd.Series(x).diff().clip(lower=0).ewm(span=14).mean() / (
                                            -pd.Series(x).diff().clip(upper=0)).ewm(span=14).mean())), self.data.Close)

                                def next(self):
                                    if len(self.data) < 2:
                                        return
                                    if self.position:
                                        self.position.close()
                                        return
                                    cross_up = self.data.ema5[-1] > self.data.ema26[-1] and self.data.ema5[-2] <= \
                                               self.data.ema26[-2]
                                    cross_down = self.data.ema5[-1] < self.data.ema26[-1] and self.data.ema5[-2] >= \
                                                 self.data.ema26[-2]
                                    if cross_up and self.data.rsi[-1] > 35 and self.data.rsi[-1] < 70:
                                        self.buy(sl=self.data.Close[-1] * 0.99, tp=self.data.Close[-1] * 1.01)
                                    elif cross_down and self.data.rsi[-1] > 30 and self.data.rsi[-1] < 65:
                                        self.sell(sl=self.data.Close[-1] * 1.01, tp=self.data.Close[-1] * 0.99)
                        else:
                            from backtesting import Strategy


                            class SimpleBacktest(Strategy):
                                def next(self):
                                    if self.position:
                                        self.position.close()
                        StrategyClass = SimpleBacktest

                    # Run backtest
                    from backtesting import Backtest

                    bt = Backtest(df, StrategyClass, cash=float(bt_capital), commission=0.001)
                    stats_backtest = bt.run()
                    trades = stats_backtest.get("_trades", None)

                    result = {
                        "metrics": dict(stats_backtest),
                        "trades": trades,
                        "requested_range": [bt_start.strftime("%Y-%m-%d"), bt_end.strftime("%Y-%m-%d")],
                        "data_range": [df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"), len(df)],
                        "is_optimization": False,
                        "optimization_results": None,
                        "monte_carlo": None,
                    }

                    # Monte Carlo
                    if st.session_state.bt_mc:
                        with st.spinner(f"Running {mc_sims} Monte Carlo simulations..."):
                            mc_results, mc_error = run_monte_carlo_backtest(
                                df, StrategyClass, bt_capital, n_simulations=mc_sims, commission=0.001
                            )
                            if mc_results:
                                result["monte_carlo"] = mc_results
                                add_log(f"✅ Monte Carlo complete: {mc_sims} simulations", "buy")
                            else:
                                add_log(f"⚠️ Monte Carlo skipped: {mc_error}", "warn")

                    # Optimization
                    if st.session_state.bt_opt:
                        param_grid = get_optimization_param_grid(bt_strategy)
                        with st.spinner(f"Optimizing {len(param_grid)} parameters..."):
                            opt_df, best_params, best_metrics = run_parameter_optimization(
                                df, StrategyClass, bt_capital, param_grid,
                                optimization_metric=opt_metric, commission=0.001
                            )
                            if opt_df is not None:
                                result["is_optimization"] = True
                                result["optimization_results"] = opt_df
                                result["best_params"] = best_params
                                result["best_metrics"] = best_metrics
                                add_log(
                                    f"✅ Optimization complete! Best {opt_metric}: {best_metrics.get(opt_metric, 'N/A')}",
                                    "buy")
                            else:
                                add_log("⚠️ Optimization produced no results", "warn")

                    st.session_state.backtest_results = result
                    add_log(f"✅ Backtest complete: {stats_backtest['# Trades']} trades, "
                            f"Return: {stats_backtest['Return [%]']:.2f}%", "buy")

                    # Save Excel
                    save_backtest_excel(result, bt_symbol, bt_interval, bt_strategy)

                except Exception as e:
                    st.error(f"❌ Backtest error: {e}")
                    add_log(f"❌ Backtest error: {e}", "err")
                    import traceback

                    st.code(traceback.format_exc())

            st.session_state.backtest_running = False
            st.rerun()

    # Display backtest results
    if st.session_state.backtest_results:
        result = st.session_state.backtest_results
        st.markdown('<div class="section-header">📈 Backtest Results</div>', unsafe_allow_html=True)

        metric_names = {
            'Return [%]': 'Return %',
            'Buy & Hold Return [%]': 'Buy & Hold %',
            'Max. Drawdown [%]': 'Max Drawdown %',
            'Avg. Drawdown [%]': 'Avg Drawdown %',
            'Sharpe Ratio': 'Sharpe Ratio',
            'Win Rate [%]': 'Win Rate %',
            '# Trades': 'Total Trades',
            'Equity Final [$]': 'Final Equity',
            'Profit Factor': 'Profit Factor',
            'Exposure Time [%]': 'Exposure %',
        }

        m1, m2, m3, m4, m5 = st.columns(5)
        metrics = result["metrics"]
        with m1:
            st.metric("Return %", f"{metrics.get('Return [%]', 0):.2f}%")
        with m2:
            st.metric("Sharpe", f"{metrics.get('Sharpe Ratio', 0):.2f}")
        with m3:
            st.metric("Win Rate %", f"{metrics.get('Win Rate [%]', 0):.2f}%")
        with m4:
            st.metric("Max DD %", f"{metrics.get('Max. Drawdown [%]', 0):.2f}%")
        with m5:
            st.metric("Total Trades", f"{metrics.get('# Trades', 0):.0f}")

        if result.get("monte_carlo"):
            st.markdown("### 🎲 Monte Carlo Results")
            mc = result["monte_carlo"]
            mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
            with mc_c1:
                st.metric("Mean Final Equity", f"${mc['mean_final_equity']:,.2f}")
            with mc_c2:
                st.metric("Median Final Equity", f"${mc['median_final_equity']:,.2f}")
            with mc_c3:
                st.metric("Profit Probability", f"{mc['probability_profit']:.1f}%")
            with mc_c4:
                st.metric("Mean Max DD %", f"{mc['mean_max_drawdown']:.2f}%")

        if result.get("is_optimization") and result.get("optimization_results") is not None:
            st.markdown("### 🔧 Parameter Optimization")
            st.dataframe(result["optimization_results"].head(10))

        if result.get("trades") is not None:
            with st.expander(f"📋 Trade Log ({len(result['trades'])} trades)", expanded=False):
                st.dataframe(result["trades"], width='stretch')

        st.success("✅ Backtest complete!")

# ─── TAB 2: ML PREDICTIONS ──────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">🤖 ML Predictions</div>', unsafe_allow_html=True)

    ml_info1, ml_info2, ml_info3 = st.columns(3)
    with ml_info1:
        ml_model_type = st.selectbox("Model", ["Random Forest", "XGBoost", "Gradient Boosting"],
                                     key="ml_model_type")
    with ml_info2:
        ml_train_bars = st.number_input("Training Bars", min_value=100, max_value=2000, value=500,
                                        key="ml_train_bars", step=50)
    with ml_info3:
        ml_conf_thr = st.slider("Confidence Threshold %", 50, 95, 75, key="ml_conf_slider")

    ml_toggle_col, ml_auto_col = st.columns(2)
    with ml_toggle_col:
        st.toggle("Enable ML Gate", key="ml_toggle", value=False,
                  help="When enabled, ML predictions will filter strategy signals")
    with ml_auto_col:
        st.toggle("Auto-Execute ML signals", key="auto_exec", value=False,
                  help="When enabled, ML can convert HOLD → BUY/SELL")

    if st.button("🧠 Train ML Model", key="train_ml"):
        with st.spinner("Training ML model..."):
            df = fetch_market_data(symbol, interval, limit=ml_train_bars, silent=True)
            if df is None or len(df) < 80:
                st.error("Not enough data. Need at least 80 candles.")
            else:
                try:
                    model = BuiltinMLModel(model_type="rf" if ml_model_type == "Random Forest" else "xgb")
                    model.train(df)
                    st.session_state.trained_ml_model = model
                    st.session_state.ml_prediction = None
                    add_log(f"✅ ML model trained! Accuracy: {model.accuracy:.1%}", "buy")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if st.session_state.trained_ml_model:
        model = st.session_state.trained_ml_model
        st.success(f"✅ Model trained — Accuracy: {model.accuracy:.1%}")

        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            if st.button("🔮 Predict Next Bar", key="ml_predict"):
                df = fetch_market_data(symbol, interval, limit=300, silent=True)
                if df is not None and len(df) > 60:
                    try:
                        pred, conf = model.predict(df)
                        direction = "BULLISH 🟢" if pred == 1 else "BEARISH 🔴" if pred == -1 else "NEUTRAL ⚪"
                        st.session_state.ml_prediction = "BULLISH" if pred == 1 else "BEARISH" if pred == -1 else "NEUTRAL"
                        st.session_state.ml_confidence = conf
                        add_log(f"🤖 ML Prediction: {direction} (conf: {conf:.1%})", "info")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        with col_pred2:
            if st.button("📈 Forecast 5 Bars", key="ml_forecast"):
                df = fetch_market_data(symbol, interval, limit=300, silent=True)
                if df is not None and len(df) > 60:
                    try:
                        fc = model.forecast(df, n=5)
                        if fc:
                            st.dataframe(pd.DataFrame(fc))
                            add_log(f"✅ Forecast generated {len(fc)} bars", "info")
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")

        if st.session_state.ml_prediction:
            pred = st.session_state.ml_prediction
            conf = st.session_state.ml_confidence
            color = "#00ff99" if pred == "BULLISH" else "#ff2255" if pred == "BEARISH" else "#ffcc00"
            st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:1.5rem;color:{color}'>"
                        f"📊 {pred} — {conf:.1%} confidence</div>", unsafe_allow_html=True)

        st.caption("ML predictions can be used as a filter for strategy signals when 'Enable ML Gate' is on.")

# ─── TAB 3: PARAMETERS ────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">⚙️ Strategy Parameters</div>', unsafe_allow_html=True)

    param_tabs = st.tabs(["Momentum", "Kalman", "Scalping", "Ranging Filter"])

    # Momentum
    with param_tabs[0]:
        st.markdown("### Momentum Strategy Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("EMA Fast", value=5.0, min_value=2.0, max_value=20.0, step=1.0, key="mom_ema_fast")
            st.number_input("EMA Mid", value=26.0, min_value=5.0, max_value=50.0, step=1.0, key="mom_ema_mid")
        with c2:
            st.number_input("EMA Slow", value=60.0, min_value=10.0, max_value=200.0, step=5.0, key="mom_ema_slow")
            st.number_input("RSI Min", value=40.0, min_value=20.0, max_value=60.0, step=1.0, key="mom_rsi_min")
        with c3:
            st.number_input("RSI Max", value=70.0, min_value=40.0, max_value=90.0, step=1.0, key="mom_rsi_max")
            st.number_input("ADX Min", value=18.0, min_value=10.0, max_value=50.0, step=1.0, key="mom_adx_min")
            st.number_input("Volume Ratio", value=1.3, min_value=0.5, max_value=3.0, step=0.1, key="mom_vol_ratio")

    # Kalman
    with param_tabs[1]:
        st.markdown("### Kalman Strategy Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Process Noise (q)", value=0.01, min_value=0.001, max_value=0.1, step=0.001,
                            key="kal_proc_noise1", format="%.3f")
            st.number_input("Measurement Noise (r)", value=500.0, min_value=50.0, max_value=2000.0, step=50.0,
                            key="kal_meas_noise")
        with c2:
            st.number_input("RSI Min", value=40.0, min_value=20.0, max_value=60.0, step=1.0, key="kal_rsi_min")
            st.number_input("RSI Max", value=65.0, min_value=40.0, max_value=90.0, step=1.0, key="kal_rsi_max")
        with c3:
            st.number_input("Strength Min %", value=70.0, min_value=20.0, max_value=150.0, step=5.0, key="kal_str_min")

    # Scalping
    with param_tabs[2]:
        st.markdown("### Scalping Strategy Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("EMA Fast", value=5.0, min_value=1.0, max_value=20.0, step=1.0, key="scal_ema_fast")
            st.number_input("EMA Slow", value=20.0, min_value=5.0, max_value=50.0, step=1.0, key="scal_ema_slow")
        with c2:
            st.number_input("RSI Min", value=35.0, min_value=20.0, max_value=60.0, step=1.0, key="scal_rsi_min")
            st.number_input("RSI Max", value=70.0, min_value=40.0, max_value=90.0, step=1.0, key="scal_rsi_max")
        with c3:
            st.number_input("ADX Min", value=18.0, min_value=10.0, max_value=50.0, step=1.0, key="scal_adx_min")
            st.number_input("Volume Ratio", value=1.2, min_value=0.5, max_value=3.0, step=0.1, key="scal_vol_ratio")

    # Ranging Filter
    with param_tabs[3]:
        st.markdown("### Ranging Market Filter")
        st.toggle("Enable Ranging Filter", key="ranging_enabled",
                  value=st.session_state.ranging_settings.get("enabled", True))

        st.markdown("#### Momentum")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.number_input("Min ADX", value=20.0, min_value=10.0, max_value=40.0, step=1.0, key="ranging_momentum_adx")
        with rc2:
            st.number_input("Max BB Width %", value=5.0, min_value=1.0, max_value=15.0, step=0.5,
                            key="ranging_momentum_bb")
        with rc3:
            st.number_input("Min Slope %/bar", value=0.15, min_value=0.01, max_value=1.0, step=0.01,
                            key="ranging_momentum_slope")

        st.markdown("#### Kalman")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.number_input("Min ADX", value=20.0, min_value=10.0, max_value=40.0, step=1.0, key="ranging_kalman_adx")
        with rc2:
            st.number_input("Max BB Width %", value=6.0, min_value=1.0, max_value=15.0, step=0.5,
                            key="ranging_kalman_bb")
        with rc3:
            st.number_input("Min Slope %/bar", value=0.12, min_value=0.01, max_value=1.0, step=0.01,
                            key="ranging_kalman_slope")

        st.markdown("#### Scalping")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.number_input("Min ADX", value=18.0, min_value=10.0, max_value=40.0, step=1.0, key="ranging_scalping_adx")
        with rc2:
            st.number_input("Max BB Width %", value=4.0, min_value=1.0, max_value=15.0, step=0.5,
                            key="ranging_scalping_bb")
        with rc3:
            st.number_input("Min Slope %/bar", value=0.10, min_value=0.01, max_value=1.0, step=0.01,
                            key="ranging_scalping_slope")

        st.markdown("#### General")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.toggle("Skip trades on ranging", key="ranging_skip",
                      value=st.session_state.ranging_settings.get("skip_on_ranging", True))
        with gc2:
            st.number_input("Cooldown bars", value=5, min_value=1, max_value=20, step=1, key="ranging_cooldown")

        if st.button("💾 Save Ranging Settings", key="save_ranging"):
            st.session_state.ranging_settings = {
                "enabled": st.session_state.ranging_enabled,
                "momentum_min_adx": st.session_state.ranging_momentum_adx,
                "momentum_max_bb_width": st.session_state.ranging_momentum_bb,
                "momentum_min_slope": st.session_state.ranging_momentum_slope,
                "kalman_min_adx": st.session_state.ranging_kalman_adx,
                "kalman_max_bb_width": st.session_state.ranging_kalman_bb,
                "kalman_min_slope": st.session_state.ranging_kalman_slope,
                "scalping_min_adx": st.session_state.ranging_scalping_adx,
                "scalping_max_bb_width": st.session_state.ranging_scalping_bb,
                "scalping_min_slope": st.session_state.ranging_scalping_slope,
                "skip_on_ranging": st.session_state.ranging_skip,
                "ranging_cooldown_bars": st.session_state.ranging_cooldown,
            }
            add_log("✅ Ranging settings saved", "info")
            st.rerun()

# ─── TAB 4: HELP ──────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">❓ Help & Instructions</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🚀 Quick Start
    1. **Select Trading Mode** — Demo (no API keys) or Live (requires OKX API)
    2. **Choose Market** — Symbol and Timeframe
    3. **Select Strategy** — Momentum, Kalman, or Scalping
    4. **Adjust Parameters** — Stop loss, trailing, order size
    5. **Click START** — Trading begins

    ### ⏱ Clock Sync (NEW)
    - **Sync to clock**: Trading executes exactly at candle closes for perfect alignment
    - **WebSocket trigger**: Uses OKX WebSocket for precise candle close timing
    - **Fallback**: Uses clock alignment if WebSocket is unavailable
    - **Fixed interval**: Legacy polling mode (use for debugging)

    ### 📊 Strategies
    - **Momentum** — EMA crossovers + RSI + MACD + volume confirmation
    - **Kalman** — Kalman filter trend detection + RSI confirmation
    - **Scalping** — Fast EMA crossovers + RSI for quick entries/exits

    ### 🤖 ML Integration
    - Train ML models on historical data
    - Use predictions as a gate for strategy signals
    - Auto-execute mode can convert HOLD → BUY/SELL

    ### ⏱ Trading Windows
    - Restrict trading to specific hours/days
    - Save presets for different sessions (London, New York, Asia)
    - UTC timezone

    ### 📈 Backtest
    - Test strategies on historical data
    - Monte Carlo simulation for robustness
    - Parameter optimization to find best settings

    ### 🔧 Ranging Filter
    - Detects ranging markets using ADX, Bollinger Bands, slope
    - Skips trades during consolidation
    - Configurable per strategy

    ### 💾 Data Persistence
    - Trading windows saved to `trading_windows.json`
    - Strategy settings saved to `strategy_settings.json`
    - Backtest results saved to `backtest_results/` folder
    """)

    st.markdown("---")
    st.caption("📈 ABOULDAHAB MCS  Professional Trading Platform v10.0 ")

# ─── SESSION SUMMARY ──────────────────────────────────────────────────────
if st.session_state.get("session_summary"):
    render_session_summary()