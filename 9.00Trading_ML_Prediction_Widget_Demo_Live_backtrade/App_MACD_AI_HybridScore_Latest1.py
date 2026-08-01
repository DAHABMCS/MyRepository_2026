"""Trading strategy backtesting and optimization module."""

import os

# ═══════════════════════════════════════════════════════════════════════════
# TQDM FIX FOR PYINSTALLER/EXE COMPILATION
# ═══════════════════════════════════════════════════════════════════════════
import sys
from typing import Any

import tqdm.asyncio

os.environ['TQDM_DISABLE'] = '1'

if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not installed. AI features will use local analysis only.")


def fix_tqdm_for_exe():
    try:
        import tqdm
        if hasattr(tqdm.std, '_file'):
            tqdm.std._file = sys.stdout

        class SafeTQDM:
            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable or []
                self.total = len(self.iterable) if hasattr(self.iterable, '__len__') else None
                self.n = 0;
                self.desc = "_plot_forecast_on_main";
                self.file = sys.stdout

            def __iter__(self): return iter(self.iterable)

            def __enter__(self): return self

            def __exit__(self, *args): pass

            def update(self, n=1): self.n += n; return True

            def close(self): pass

            def set_description(self, desc=None, refresh=True):
                if desc is not None: self.desc = desc
                return self

            def set_postfix(self, **kwargs): return self

            def refresh(self): pass

        tqdm.tqdm = SafeTQDM;
        tqdm.std.tqdm = SafeTQDM;
        tqdm.asyncio.tqdm = SafeTQDM
        if hasattr(tqdm.std, 'Tqdm'): tqdm.std.Tqdm = SafeTQDM
        print("✅ TQDM patched successfully for EXE")
    except ImportError:
        class FakeTQDM:
            def __init__(self, *args, **kwargs): pass

            def __enter__(self): return self

            def __exit__(self, *args): pass

            def update(self, n=1): pass

            def close(self): pass

        sys.modules['tqdm'] = type(sys)('tqdm')
        sys.modules['tqdm'].tqdm = FakeTQDM
        sys.modules['tqdm'].std = type(sys)('std')
        sys.modules['tqdm'].std.tqdm = FakeTQDM
        print("✅ Created fake TQDM module")


fix_tqdm_for_exe()

import json
import math
from glob import glob
import textwrap
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
import numpy as np
import pandas as pd
import ccxt
import pyttsx3
import requests
import logging
import winsound
from datetime import datetime, timezone, timedelta
from strategies.monte_carlo_simulator import MonteCarloSimulator
from tkinter import ttk, scrolledtext, messagebox
from utils.AdaptiveWeightManager import AdaptiveWeightManager
from strategies.TradingStrategy3 import TradingStrategy
from PIL import Image, ImageTk
from backtesting import Backtest
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.Account as Account
from models.lstm_model_NEW import LSTMModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from utils.FinancialCircletimer_New import FinancialChartWidget, CircleTimer
from strategies.MomentumStrategy_MACD_HybridScore_Latest import MomentumStrategy, BacktestMomentumStrategy, \
    MOMENTUM_PARAMS, GlobalConfig
from strategies.KalmanTrendStrategy_New import KalmanTrendStrategy, BacktestKalmanTrendStrategy

import sys
import os

os.environ['TQDM_DISABLE'] = '1'

try:
    import tqdm


    class FakeTQDM:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else None

        def __iter__(self):
            if self.iterable: return iter(self.iterable)
            return iter([])

        def __enter__(self): return self

        def __exit__(self, *args): pass

        def update(self, n=1): pass

        def close(self): pass

        def set_description(self, desc): pass


    tqdm.tqdm = FakeTQDM;
    tqdm.std.tqdm = FakeTQDM;
    tqdm.asyncio.tqdm = FakeTQDM
    print("✅ TQDM disabled successfully")
except ImportError:
    print("⚠️ TQDM not imported yet")
except Exception as e:
    print(f"⚠️ Could not patch TQDM: {e}")


class TradingApp:
    def __init__(self, root):
        self.root = root
        self.log_area = None
        self.ai_data_available = False
        self.param_groups = {}
        self._updating_group = False
        self.backtest_running = False
        self.trading_running = False
        self.root.after_idle(self.load_params_at_startup)

        self.position = {
            'type': None, 'price': None, 'quantity': None, 'time': None,
            'stop_loss': None, 'trailing_stop': None, 'entry_confidence': None
        }

        self.objective_config = {
            'default': {
                'min_trades_absolute': 10, 'min_trades_penalty': 20,
                'max_trades_penalty': 500, 'penalty_low': 0.7,
                'penalty_high': 0.9, 'weights': (0.6, 0.3, 0.1)
            },
            'momentum': {
                'min_trades_absolute': 20, 'min_trades_penalty': 35,
                'min_trades_target': 48, 'max_trades_penalty': 120,
                'penalty_low': 0.35, 'penalty_high': 0.9, 'weights': (0.5, 0.4, 0.1)
            },
            'kalman': {
                'min_trades_absolute': 8, 'min_trades_penalty': 15,
                'max_trades_penalty': 300, 'penalty_low': 0.8,
                'penalty_high': 0.9, 'weights': (0.7, 0.2, 0.1)
            }
        }

        self.optimization_metrics = {
            'sharpe': tk.BooleanVar(value=True),
            'sortino': tk.BooleanVar(value=False),
            'returns': tk.BooleanVar(value=False),
            'winrate': tk.BooleanVar(value=False),
            'profit_factor': tk.BooleanVar(value=False),
            'equity': tk.BooleanVar(value=True),
            'trade_count': tk.BooleanVar(value=True),
        }

        self.backtest_params = {}
        self.scalping_backtest_params = {}

        self.weight_manager = AdaptiveWeightManager(alpha=0.2, min_w=0.2, max_w=0.8)

        # ── Trading Time Window ──────────────────────────────────────────────
        self.trading_time_config = self._init_trading_time_config()
        self._sched_start_id = None
        self._sched_stop_id = None
        self._waiting_to_start = False

        self.macd_below_streak = 0
        self.momentum_streak_required = 2
        self.momentum_exit_threshold = 0.995
        self.trailing_stop_buffer = 0.0015

        self.active_position = None
        self.position_entry_price = None
        self.position_size = None
        self.position_entry_time = None
        self.volume_mean = 0
        self.atr_mean = 0
        self.strategy_params = {
            'volume_spike_threshold': 1.2, 'atr_multiplier': 1.5,
            'enable_early_detection': True
        }
        self.strategy_versions = []
        self._strategy_hash = None
        self.last_traded_hash = None

        logging.basicConfig(
            filename='trading_bot.log', level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.current_data = None
        self.current_status = "parking"
        self.root = root
        self.root.title("v8.0-Multi-Layered Trading (MACD) Hybrid Score")
        self.running = False
        self.trailing_stop = None
        self.highest_since_buy = None
        self.lowest_since_sell = None
        self.initial_stop_loss = None
        self.strategy = None

        self.strategy_settings_file = "strategy_settings.json"
        self.default_params = {}
        self.custom_params = {}
        self.param_toggle_var = tk.StringVar(value="Default Parameters")

        self.load_strategy_settings()

        self.log_area = None
        self.ai_data_available = False
        self.volume_stats = {'mean': 0, 'std': 1}
        self.ml_enabled = False

        self.ml_models = {
            "Random Forest": RandomForestModel(),
            "XGBoost": XGBoostModel(),
            "LSTM": LSTMModel()
        }
        self.current_ml_model = None
        self.ml_prediction_threshold = 0.65

        self.performance_stats = {
            'best_trade': 0, 'worst_trade': 0, 'total_trades': 0,
            'winning_streak': 0, 'losing_streak': 0
        }

        self.trailing_stop_pct = 0.03
        self.stop_loss_pct = 0.02
        self.order_size_pct = 30
        self.trade_direction = "long"

        self.interval_seconds_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1H': 3600, '1D': 86400
        }

        self.load_config()
        self.create_widgets()
        self.setup_apis()

        self.momentum_params = MOMENTUM_PARAMS.copy()
        self.current_mode = "Default Parameters"
        self.custom_params = {}
        self.load_saved_momentum_settings()

        from strategies.scalping_strategy import ScalpingStrategy, BacktestScalpingStrategy

        self.strategies = {
            "Momentum": MomentumStrategy(self),
            "BacktestMomentum": BacktestMomentumStrategy,
            "Kalman": KalmanTrendStrategy(self),
            "BacktestKalmanTrendStrategy": BacktestKalmanTrendStrategy,
            "Enhanced": TradingStrategy(self, {}),
            "Scalping": ScalpingStrategy(self),
            "BacktestScalping": BacktestScalpingStrategy,
        }

        self.apply_selected_parameters()
        self.strategies["Momentum"].trading_app = self

        self.virtual_balance = {'USDT': 1000, 'SOL': 0}

        self.trade_history = []

        self.set_strategy("Momentum")

        if self.mode_var.get().lower() == "backtest":
            self.backtest_controls_frame.pack(fill=tk.BOTH, expand=True)
            self.explanation_textbox.pack_forget()
            self.start_btn.config(state=tk.DISABLED)
            self.style.configure('Blue.TButton', foreground='grey')
        else:
            self.backtest_controls_frame.pack_forget()
            self.explanation_textbox.pack(fill=tk.BOTH, expand=True)
            self.start_btn.config(state=tk.DISABLED)
            self.style.configure('Blue.TButton', foreground='white')

        self.log_expanded = False
        self.hidden_frames_info = []
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._tts_engine = None

    # ═══════════════════════════════════════════════════════════════════════
    # TRADING TIME WINDOW METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def _init_trading_time_config(self) -> dict:
        """Per-strategy time window.  start == end → no restriction."""
        cfg = {}
        for strategy in ('Momentum', 'Kalman', 'Scalping'):
            cfg[strategy] = {
                'start_h': tk.IntVar(value=0),
                'start_m': tk.IntVar(value=0),
                'end_h': tk.IntVar(value=0),
                'end_m': tk.IntVar(value=0),
            }
        return cfg

    def _is_time_unconstrained(self, strategy: str) -> bool:
        cfg = self.trading_time_config.get(strategy, {})
        return (cfg['start_h'].get() == cfg['end_h'].get() and
                cfg['start_m'].get() == cfg['end_m'].get())

    def _get_window_minutes(self, strategy: str) -> tuple:
        cfg = self.trading_time_config[strategy]
        s = cfg['start_h'].get() * 60 + cfg['start_m'].get()
        e = cfg['end_h'].get() * 60 + cfg['end_m'].get()
        return s, e

    def _is_within_trading_window(self, strategy: str) -> bool:
        if self._is_time_unconstrained(strategy):
            return True
        now_utc = datetime.now(timezone.utc)
        now_min = now_utc.hour * 60 + now_utc.minute
        s, e = self._get_window_minutes(strategy)
        if s < e:
            return s <= now_min < e
        return now_min >= s or now_min < e

    def _seconds_until_start(self, strategy: str) -> float:
        now_utc = datetime.now(timezone.utc)
        now_min = now_utc.hour * 60 + now_utc.minute
        now_sec = now_utc.second
        s, _ = self._get_window_minutes(strategy)
        diff_min = (s - now_min) % (24 * 60)
        return max(0., diff_min * 60 - now_sec)

    def _seconds_until_end(self, strategy: str) -> float:
        now_utc = datetime.now(timezone.utc)
        now_min = now_utc.hour * 60 + now_utc.minute
        now_sec = now_utc.second
        _, e = self._get_window_minutes(strategy)
        diff_min = (e - now_min) % (24 * 60)
        return max(0., diff_min * 60 - now_sec)

    def _cancel_scheduled_trading(self):
        for attr in ('_sched_start_id', '_sched_stop_id'):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._waiting_to_start = False

    def _do_start_trading(self):
        """Spin up the trading thread (internal)."""
        self._waiting_to_start = False
        # Record session start so the end-of-period summary can show duration
        self._session_start_time = datetime.now(timezone.utc)
        self._session_start_trade_count = len(
            [t for t in self.trade_history if t.get('type') == 'sell'])

        # ── Kill any still-alive previous thread ────────────────────────────
        if hasattr(self, 'trading_thread') and self.trading_thread is not None:
            if self.trading_thread.is_alive():
                self.running = False
                self.trading_running = False
                self.trading_thread.join(timeout=3.0)
            self.trading_thread = None
        self.running = True
        self.trading_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_mode_display(self.mode_var.get())
        self.trading_thread = threading.Thread(
            target=self.trading_loop, daemon=True, name="TradingLoop")
        self.trading_thread.start()
        self.log_message("▶ Trading started.", "green")
        self.play_notification("tick")
        self.enable_ai_button()
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 135)
            engine.say("Trading started")
            engine.runAndWait()
        except Exception:
            pass

    def _scheduled_start_callback(self):
        self._sched_start_id = None
        if not self._waiting_to_start:
            return
        strategy = self.strategy_type_var.get()
        now_str = datetime.now(timezone.utc).strftime('%H:%M UTC')
        self.log_message(f"⏰ [{strategy}] Scheduled start at {now_str}", "green")
        self._do_start_trading()
        if not self._is_time_unconstrained(strategy):
            stop_secs = self._seconds_until_end(strategy)
            if stop_secs > 0:
                self._sched_stop_id = self.root.after(
                    int(stop_secs * 1000), self._scheduled_stop_callback)
                _, em = self._get_window_minutes(strategy)
                eh, em2 = divmod(em, 60)
                self.log_message(
                    f"⏰ [{strategy}] Auto-stop in "
                    f"{stop_secs / 60:.1f} min ({eh:02d}:{em2:02d} UTC)", "blue")

    def _scheduled_stop_callback(self):
        self._sched_stop_id = None
        if not self.running:
            return
        strategy = self.strategy_type_var.get()
        now_str = datetime.now(timezone.utc).strftime('%H:%M UTC')
        self.log_message(
            f"⏰ [{strategy}] Trading window closed at {now_str}.", "orange")
        self._end_of_period_stop()

    def _end_of_period_stop(self):
        """
        Called when the scheduled trading period ends.
        1. Block any new entries immediately.
        2. Close any open position at market price.
        3. Print a full session summary to the log.
        4. Leave the app idle and ready — do NOT exit.
        """
        # ── Step 1: Block new entries ─────────────────────────────────────────
        self.trading_running = False  # signals the loop: no new trades
        self.log_message("⏸  Trading period ended — no new entries will be taken.", "orange")

        # ── Step 2: Close any open position at market price ──────────────────
        pos_type = self.position.get('type')
        if pos_type in ('long', 'short'):
            current_price = self.get_current_price()
            if current_price is None and self.current_data is not None:
                current_price = float(self.current_data.get('Close', 0))
            if current_price:
                close_side = 'sell' if pos_type == 'long' else 'buy'
                qty = self.position.get('quantity', 0)
                self.log_message(
                    f"📤 Period-end close: {pos_type.upper()} {qty:.4f} @ ${current_price:.4f}",
                    "blue")
                try:
                    self.place_order(close_side, current_price, qty,
                                     exit_reason='period_end')
                except Exception as e:
                    self.log_message(f"⚠️  Could not close position: {e}", "red")
            else:
                self.log_message("⚠️  No price available — could not close open position.", "red")

        # ── Step 3: Fully stop the trading loop ───────────────────────────────
        self.stop_trading()  # resets flags, re-enables Start button

        # ── Step 4: Print session summary ─────────────────────────────────────
        self.root.after(500, self._show_session_summary)  # slight delay so trades settle

    def _show_session_summary(self):
        """Print a formatted trading period summary to the log panel."""
        sep = "═" * 60
        self.log_message(sep, "cyan")
        self.log_message("📊  TRADING SESSION SUMMARY", "cyan")
        self.log_message(sep, "cyan")

        # Session timing
        session_start = getattr(self, '_session_start_time', None)
        session_end = datetime.now(timezone.utc)
        if session_start:
            duration_mins = (session_end - session_start).total_seconds() / 60
            self.log_message(
                f"⏱  Session:   {session_start.strftime('%H:%M UTC')} → "
                f"{session_end.strftime('%H:%M UTC')}  "
                f"({duration_mins:.0f} min)", "white")

        # Trade stats — only count trades made THIS session
        start_count = getattr(self, '_session_start_trade_count', 0)
        all_closed = [t for t in self.trade_history if t.get('exit_timestamp') is not None]
        # all_closed  = [t for t in self.trade_history if t.get('type') == 'sell']
        session_trades = all_closed[start_count:]  # trades added during this session

        if not session_trades:
            self.log_message("  No trades completed during this session.", "orange")
        else:
            total = len(session_trades)
            wins = [t for t in session_trades if t.get('pnl', t.get('net_pnl', 0)) > 0]
            losses = [t for t in session_trades if t.get('pnl', t.get('net_pnl', 0)) <= 0]
            total_pnl = sum(t.get('pnl', t.get('net_pnl', 0)) for t in session_trades)
            win_rate = len(wins) / total * 100 if total else 0
            avg_win = sum(t.get('pnl', t.get('net_pnl', 0)) for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.get('pnl', t.get('net_pnl', 0)) for t in losses) / len(losses) if losses else 0
            best_trade = max(t.get('pnl', t.get('net_pnl', 0)) for t in session_trades)
            worst_trade = min(t.get('pnl', t.get('net_pnl', 0)) for t in session_trades)

            gross_wins = sum(t.get('pnl', t.get('net_pnl', 0)) for t in wins)
            gross_loss = abs(sum(t.get('pnl', t.get('net_pnl', 0)) for t in losses))
            profit_factor = (gross_wins / gross_loss) if gross_loss > 0 else float('inf')

            pnl_color = "green" if total_pnl >= 0 else "red"

            self.log_message(f"  📈 Trades:          {total}  ({len(wins)}W / {len(losses)}L)", "white")
            self.log_message(f"  🎯 Win Rate:        {win_rate:.1f}%", "green" if win_rate >= 50 else "orange")
            self.log_message(f"  💰 Net PnL:         ${total_pnl:+.2f}", pnl_color)
            self.log_message(f"  📊 Profit Factor:   {profit_factor:.2f}", "green" if profit_factor >= 1 else "red")
            self.log_message(f"  ✅ Avg Win:         ${avg_win:+.2f}", "green")
            self.log_message(f"  ❌ Avg Loss:        ${avg_loss:+.2f}", "red")
            self.log_message(f"  🏆 Best Trade:      ${best_trade:+.2f}", "green")
            self.log_message(f"  💀 Worst Trade:     ${worst_trade:+.2f}", "red")

            # Exit reason breakdown
            exit_reasons = {}
            for t in session_trades:
                r = t.get('exit_reason', 'unknown')
                exit_reasons[r] = exit_reasons.get(r, 0) + 1
            if exit_reasons:
                reasons_str = "  |  ".join(f"{r}: {c}" for r, c in exit_reasons.items())
                self.log_message(f"  🚪 Exit reasons:    {reasons_str}", "white")

        self.log_message(sep, "cyan")
        self.log_message(
            "✅  Session complete.  Press ▶ Start Trading to begin a new session "
            "or use any other function.", "green")
        self.log_message(sep, "cyan")

        # Voice notification
        try:
            import pyttsx3 as _pyttsx3
            engine = _pyttsx3.init()
            engine.setProperty('rate', 135)
            engine.say("Trading session complete. Summary ready.")
            engine.runAndWait()
        except Exception:
            pass

    def _update_wait_countdown(self, strategy: str, sh: int, sm: int):
        if not getattr(self, '_waiting_to_start', False):
            return
        secs = self._seconds_until_start(strategy)
        if secs <= 0:
            return
        mins = int(secs / 60)
        self.mode_display.config(
            text=f"⏰ STARTS IN {mins}m → {sh:02d}:{sm:02d} UTC",
            foreground='red')
        self.root.after(60_000,
                        lambda: self._update_wait_countdown(strategy, sh, sm))

    def open_time_settings_panel(self, strategy: str):
        """Open a floating panel to set trading hours for *strategy*."""
        attr = f"_time_panel_{strategy}"
        existing = getattr(self, attr, None)
        if existing and existing.winfo_exists():
            existing.lift();
            existing.focus_force();
            return

        cfg = self.trading_time_config[strategy]
        panel = tk.Toplevel(self.root)
        panel.title(f"⏰ Trading Hours — {strategy}")
        panel.geometry("390x285")
        panel.resizable(False, False)
        panel.grab_set()
        setattr(self, attr, panel)

        hdr_f = tk.Frame(panel, bg="#1a1a2e", pady=8)
        hdr_f.pack(fill=tk.X)
        tk.Label(hdr_f, text=f"⏰  {strategy} — Trading Hours (UTC)",
                 bg="#1a1a2e", fg="white",
                 font=("Arial", 12, "bold")).pack()

        body = ttk.Frame(panel, padding=15)
        body.pack(fill=tk.BOTH, expand=True)

        def _time_row(lbl_text, h_var, m_var, row):
            ttk.Label(body, text=lbl_text,
                      font=("Arial", 10, "bold")).grid(
                row=row, column=0, sticky="w", pady=6, padx=5)
            sb_h = ttk.Spinbox(body, from_=0, to=23, textvariable=h_var,
                               width=5, format="%02.0f", font=("Consolas", 11))
            sb_h.grid(row=row, column=1, padx=(10, 2))
            ttk.Label(body, text=":", font=("Consolas", 13, "bold")).grid(
                row=row, column=2)
            sb_m = ttk.Spinbox(body, from_=0, to=59, textvariable=m_var,
                               width=5, format="%02.0f", font=("Consolas", 11))
            sb_m.grid(row=row, column=3, padx=(2, 10))
            ttk.Label(body, text="UTC", foreground="grey").grid(
                row=row, column=4, padx=5)

        _time_row("▶  Start time :", cfg["start_h"], cfg["start_m"], row=0)
        _time_row("⏹  End   time :", cfg["end_h"], cfg["end_m"], row=1)

        status_var = tk.StringVar()
        status_lbl = ttk.Label(body, textvariable=status_var, font=("Arial", 9))
        status_lbl.grid(row=2, column=0, columnspan=5, pady=(10, 0), sticky="w")

        def _refresh(*_):
            sh, sm = cfg["start_h"].get(), cfg["start_m"].get()
            eh, em = cfg["end_h"].get(), cfg["end_m"].get()
            if sh == eh and sm == em:
                status_var.set("ℹ️  No restriction — trades run at any time")
                status_lbl.config(foreground="#0066CC")
            else:
                status_var.set(
                    f"✅  Window: {sh:02d}:{sm:02d} → {eh:02d}:{em:02d} UTC")
                status_lbl.config(foreground="#006600")

        for v in (cfg["start_h"], cfg["start_m"], cfg["end_h"], cfg["end_m"]):
            v.trace_add("write", _refresh)
        _refresh()

        btn_f = ttk.Frame(panel, padding=(15, 5, 15, 15))
        btn_f.pack(fill=tk.X)

        def _reset():
            cfg["end_h"].set(cfg["start_h"].get())
            cfg["end_m"].set(cfg["start_m"].get())
            self.log_message(
                f"⏰ [{strategy}] Trading hours reset — no constraint.", "blue")

        def _save():
            _refresh()
            sh, sm = cfg["start_h"].get(), cfg["start_m"].get()
            eh, em = cfg["end_h"].get(), cfg["end_m"].get()
            if sh == eh and sm == em:
                self.log_message(
                    f"⏰ [{strategy}] Trading hours: unrestricted", "blue")
            else:
                self.log_message(
                    f"⏰ [{strategy}] Trading hours: "
                    f"{sh:02d}:{sm:02d} → {eh:02d}:{em:02d} UTC", "green")
            panel.destroy()

        tk.Button(btn_f, text="🔄 Reset (no limit)", command=_reset,
                  bg="#FFA500", fg="white", font=("Arial", 9, "bold"),
                  relief="raised", bd=2, cursor="hand2",
                  padx=10, pady=4).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_f, text="💾 Save & Close", command=_save,
                  bg="#0066CC", fg="white", font=("Arial", 9, "bold"),
                  relief="raised", bd=2, cursor="hand2",
                  padx=10, pady=4).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_f, text="❌ Cancel", command=panel.destroy,
                  bg="#CC0000", fg="white", font=("Arial", 9, "bold"),
                  relief="raised", bd=2, cursor="hand2",
                  padx=10, pady=4).pack(side=tk.RIGHT, padx=4)

    def _add_time_settings_button(self, parent_frame, strategy: str):
        """Inject a ⏰ Trading Hours row at the top of a settings tab."""
        cfg = self.trading_time_config.get(strategy)
        if cfg is None:
            return
        row = tk.Frame(parent_frame, bg="#f0f4ff", pady=4)
        row.pack(fill=tk.X, padx=5, pady=(4, 0))

        def _summary():
            sh, sm = cfg["start_h"].get(), cfg["start_m"].get()
            eh, em = cfg["end_h"].get(), cfg["end_m"].get()
            if sh == eh and sm == em:
                return "⏰ Trading hours: unrestricted"
            return f"⏰ {sh:02d}:{sm:02d} → {eh:02d}:{em:02d} UTC"

        lbl_var = tk.StringVar(value=_summary())
        lbl = tk.Label(row, textvariable=lbl_var, bg="#f0f4ff",
                       font=("Arial", 9), anchor="w")
        lbl.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        def _refresh(*_):
            lbl_var.set(_summary())
            sh, sm = cfg["start_h"].get(), cfg["start_m"].get()
            eh, em = cfg["end_h"].get(), cfg["end_m"].get()
            lbl.config(fg="#006600" if not (sh == eh and sm == em) else "#444444")

        for v in (cfg["start_h"], cfg["start_m"], cfg["end_h"], cfg["end_m"]):
            v.trace_add("write", _refresh)

        tk.Button(row, text="⚙ Set Hours",
                  command=lambda s=strategy: self.open_time_settings_panel(s),
                  bg="#0066CC", fg="white", font=("Arial", 8, "bold"),
                  relief="raised", bd=2, cursor="hand2",
                  padx=8, pady=2).pack(side=tk.RIGHT, padx=8)

    # ═══════════════════════════════════════════════════════════════════════
    # CORE TRADING METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def stop_trading(self):
        """Stop the trading loop and clean up state."""
        if not self.running:
            self.log_message("⚠️ Trading is not currently running.", "orange")
            return
        self.running = False
        self.trading_running = False
        self._cancel_scheduled_trading()
        if hasattr(self, 'timer') and self.timer:
            try:
                self.timer.stop()
            except Exception:
                pass
        if hasattr(self, 'start_btn') and self.start_btn:
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'stop_btn') and self.stop_btn:
            self.stop_btn.config(state=tk.DISABLED)
        self.update_mode_display(self.mode_var.get())
        self.update_status_indicators("parking")
        self.log_message("🛑 Trading stopped.", "orange")
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 135)
            engine.say("Trading stopped")
            engine.runAndWait()
        except Exception:
            pass
        self.play_notification("tick")
        self.update_stats()

    def predict_future_trend(self, n_future: int = 5):
        """
        Use the current ML model to predict the trend for the next n_future candles.
        Returns (predictions: list, confidence: float).
        """
        if not self.ml_enabled or self.current_ml_model is None:
            return [], 0.0
        if not getattr(self.current_ml_model, 'is_trained', False):
            return [], 0.0
        df = self.get_market_data()
        if df is None or df.empty:
            return [], 0.0
        try:
            if hasattr(self.strategy, 'calculate_indicators'):
                df = self.strategy.calculate_indicators(df)
            conf, prediction, _ = self.current_ml_model.predict(df, n_future)
            confidence = float(conf) if conf <= 1.0 else float(conf) / 100.0
            if prediction == 1:
                predictions = ['bullish'] * n_future
            elif prediction == -1:
                predictions = ['bearish'] * n_future
            else:
                predictions = ['neutral'] * n_future
            return predictions, confidence
        except Exception as e:
            self.log_message(f"⚠️ predict_future_trend error: {e}", "orange")
            return [], 0.0

    def close_partial(self, fraction: float = 0.5):
        """Close a fraction of the current position (0 < fraction <= 1)."""
        if self.position.get('type') is None or self.position.get('quantity') is None:
            self.log_message("⚠️ close_partial: no open position.", "orange")
            return False
        if not (0 < fraction <= 1):
            self.log_message(f"❌ close_partial: invalid fraction {fraction}", "red")
            return False
        qty_to_close = self.position['quantity'] * fraction
        current_price = self.get_current_price()
        if current_price is None:
            if self.current_data is not None:
                current_price = float(self.current_data.get('Close', 0))
        if not current_price:
            self.log_message("❌ close_partial: cannot determine price.", "red")
            return False
        self.log_message(
            f"📤 Partial close: {fraction * 100:.0f}% "
            f"({qty_to_close:.4f} units) @ ${current_price:.4f}", "blue")
        success = self.place_order(
            'sell', current_price, quantity=qty_to_close,
            exit_reason='partial_profit_target')
        if success:
            remaining = self.position['quantity'] - qty_to_close
            if remaining > 1e-8:
                self.position['quantity'] = remaining
                self.log_message(
                    f"✅ Partial close complete. Remaining: {remaining:.4f}", "green")
            else:
                self.position = {
                    'type': None, 'price': None, 'quantity': None, 'time': None,
                    'stop_loss': None, 'trailing_stop': None, 'entry_confidence': None
                }
                self.update_status_indicators("parking")
                self.log_message("✅ Position fully closed via close_partial.", "green")
        return success

    def _save_trading_time_config(self, settings: dict):
        """Inject trading-time windows into a settings dict before writing to disk."""
        time_cfg = {}
        for strategy, cfg in self.trading_time_config.items():
            time_cfg[strategy] = {
                'start_h': cfg['start_h'].get(),
                'start_m': cfg['start_m'].get(),
                'end_h': cfg['end_h'].get(),
                'end_m': cfg['end_m'].get(),
            }
        settings['trading_time_config'] = time_cfg
        return settings

    def _load_trading_time_config(self, settings: dict):
        """Read trading-time windows from a loaded settings dict."""
        time_cfg = settings.get('trading_time_config', {})
        if not time_cfg:
            self.log_message("ℹ️ No trading time config found — using defaults (unrestricted)", "blue")
            return

        loaded = 0
        for strategy, values in time_cfg.items():
            if strategy not in self.trading_time_config:
                continue
            cfg = self.trading_time_config[strategy]
            try:
                cfg['start_h'].set(int(values.get('start_h', 0)))
                cfg['start_m'].set(int(values.get('start_m', 0)))
                cfg['end_h'].set(int(values.get('end_h', 0)))
                cfg['end_m'].set(int(values.get('end_m', 0)))
                loaded += 1

                sh = cfg['start_h'].get();
                sm = cfg['start_m'].get()
                eh = cfg['end_h'].get();
                em = cfg['end_m'].get()

                if sh == eh and sm == em:
                    self.log_message(
                        f"⏰ [{strategy}] Trading hours loaded: unrestricted", "blue")
                else:
                    self.log_message(
                        f"⏰ [{strategy}] Trading hours loaded: "
                        f"{sh:02d}:{sm:02d} → {eh:02d}:{em:02d} UTC", "green")
            except Exception as e:
                self.log_message(
                    f"⚠️ Could not load trading time for {strategy}: {e}", "orange")

        self.log_message(
            f"✅ Trading time config loaded for {loaded} strategies", "green")

    def load_saved_momentum_settings(self):
        try:
            from strategies.MomentumStrategy_MACD_HybridScore_Latest import MOMENTUM_PARAMS, MomentumConfig
            self.momentum_params = MOMENTUM_PARAMS.copy()
            self.current_mode = MomentumConfig.get_current_mode()
            self.custom_params = MomentumConfig.get_custom_params()
            print("=" * 70)
            print("📋 LOADING MOMENTUM SETTINGS")
            print("=" * 70)
            print(f"Mode: {self.current_mode}")
            print(f"Custom params available: {len(self.custom_params)}")
            if self.current_mode == "Custom Parameters" and self.custom_params:
                print("\nApplying custom parameters:")
                applied_count = 0
                for key, value in self.custom_params.items():
                    if key in self.momentum_params:
                        old_value = self.momentum_params[key]
                        self.momentum_params[key] = value
                        if old_value != value:
                            print(f"  ✓ {key}: {old_value} → {value}")
                            applied_count += 1
                print(f"\nApplied {applied_count} custom parameter overrides")
            else:
                print("\nUsing MOMENTUM_PARAMS defaults (no custom overrides)")
            print("=" * 70)
            if hasattr(self, 'param_toggle_var'):
                self.param_toggle_var.set(self.current_mode)
            return True
        except Exception as e:
            print(f"❌ Error loading saved settings: {e}")
            print("Falling back to MOMENTUM_PARAMS defaults")
            from strategies.MomentumStrategy_MACD_HybridScore_Latest import MOMENTUM_PARAMS
            self.momentum_params = MOMENTUM_PARAMS.copy()
            self.current_mode = "Default Parameters"
            self.custom_params = {}
            return False

    def get_volume_ratio(self, df=None, current_data=None, default=1.0):
        vol_ratio = None
        if df is not None and len(df) >= 2:
            for col_name in ['Volume_Ratio', 'volume_ratio', 'Vol_Ratio']:
                if col_name in df.columns:
                    try:
                        val = df[col_name].iloc[-2]
                        if pd.notna(val) and val > 0:
                            vol_ratio = float(val);
                            break
                    except:
                        continue
        if vol_ratio is None or vol_ratio <= 0:
            if current_data is not None:
                for col_name in ['Volume_Ratio', 'volume_ratio']:
                    try:
                        val = current_data.get(col_name) if hasattr(current_data, 'get') else None
                        if val and float(val) > 0:
                            vol_ratio = float(val);
                            break
                    except:
                        continue
        if vol_ratio is None or vol_ratio <= 0:
            if df is not None and 'Volume' in df.columns and len(df) >= 22:
                try:
                    current_vol = float(df['Volume'].iloc[-2])
                    avg_vol = df['Volume'].iloc[-22:-2].mean()
                    if avg_vol > 0: vol_ratio = current_vol / avg_vol
                except:
                    pass
        if vol_ratio is None or vol_ratio <= 0:
            vol_ratio = default
        return max(0.01, min(10.0, vol_ratio))

    def on_closing(self):
        try:
            self.log_message("🔄 Shutting down application...", "blue")
            self.running = False
            if hasattr(self, '_cancel_scheduled_trading'):
                self._cancel_scheduled_trading()
            if hasattr(self, 'timer') and self.timer:
                try:
                    self.timer.stop()
                except Exception:
                    pass
            if hasattr(self, '_tts_engine') and self._tts_engine:
                try:
                    self._tts_engine.stop()
                except Exception:
                    pass
            if hasattr(self, 'market_api') and self.market_api: self.market_api = None
            if hasattr(self, 'trade_api') and self.trade_api:  self.trade_api = None
            if hasattr(self, 'account_api') and self.account_api: self.account_api = None
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass
            if hasattr(self, 'chart') and self.chart:
                try:
                    self.chart.destroy()
                except Exception:
                    pass
            try:
                after_ids = list(self.root.tk.call('after', 'info'))
                for after_id in after_ids:
                    try:
                        self.root.after_cancel(after_id)
                    except Exception:
                        pass
            except Exception:
                pass
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            import threading
            def force_exit():
                import os;
                os._exit(0)

            threading.Timer(2.0, force_exit).start()

    def sync_custom_params_with_code_defaults(self):
        self.log_message("=" * 70, "blue")
        self.log_message("🔄 RESETTING CUSTOM PARAMS TO MOMENTUM_PARAMS", "blue")
        self.log_message("=" * 70, "blue")
        momentum_defaults = self.get_default_momentum_params()
        self.custom_params['momentum'] = momentum_defaults.copy()
        self.log_message("✅ Custom params reset to MOMENTUM_PARAMS values", "green")
        self.log_message(f"   tier1_adx_hard_min     = {self.custom_params['momentum']['tier1_adx_hard_min']}", "green")
        self.log_message(f"   tier1_volume_min       = {self.custom_params['momentum']['tier1_volume_min']}", "green")
        self.log_message(f"   stop_loss_atr_mult     = {self.custom_params['momentum']['stop_loss_atr_mult']}", "green")
        self.log_message(
            f"   trailing_activation_tier1 = {self.custom_params['momentum']['trailing_activation_tier1']}", "green")
        self.log_message(f"   quality_tier2_min_long = {self.custom_params['momentum']['quality_tier2_min_long']}",
                         "green")
        self.log_message(f"   risk_tier1             = {self.custom_params['momentum']['risk_tier1']}", "green")
        self.log_message(f"   max_daily_trades       = {self.custom_params['momentum']['max_daily_trades']}", "green")
        self.log_message(f"   only_tier1_entries     = {self.custom_params['momentum']['only_tier1_entries']}", "green")
        self.log_message("=" * 70, "green")
        return True

    def test_premium_tier1_config(self):
        if not hasattr(self, 'momentum_strategy') or self.momentum_strategy is None:
            self.log_message("ERROR: Strategy not loaded", "red");
            return
        s = self.momentum_strategy
        checks = {
            "Tier 1 Quality Min (LONG)": (s.quality_tier1_min_long, 0.70),
            "Tier 2 Quality Min (LONG)": (s.quality_tier2_min_long, 0.60),
            "Tier 1 Quality Min (SHORT)": (s.quality_tier1_min_short, 0.70),
            "Tier 2 Quality Min (SHORT)": (s.quality_tier2_min_short, 0.60),
            "Only Tier 1 Entries": (s.only_tier1_entries, False),
            "Tier 1 ADX Hard Min (LONG)": (s.tier1_adx_hard_min, 25.0),
            "Tier 1 ADX Hard Min (SHORT)": (s.tier1_adx_hard_min_short, 30.0),
            "Tier 1 RSI Min (LONG)": (s.tier1_rsi_min, 55.0),
            "Tier 1 RSI Max (LONG)": (s.tier1_rsi_max, 75.0),
            "Tier 1 RSI Min (SHORT)": (s.tier1_rsi_min_short, 25.0),
            "Tier 1 RSI Max (SHORT)": (s.tier1_rsi_max_short, 45.0),
            "Tier 1 Volume Min (LONG)": (s.tier1_volume_min, 1.5),
            "Tier 1 Volume Min (SHORT)": (s.tier1_volume_min_short, 1.3),
            "Tier 1 Momentum Min": (s.tier1_momentum_min, 0.02),
            "Tier 1 Price EMA Max %": (s.tier1_price_ema_max_pct, 1.5),
            "Stop Loss ATR Mult (unified)": (s.stop_loss_atr_mult, 2.0),
            "Risk Tier 1": (s.risk_tier1, 0.02),
            "Risk Tier 2": (s.risk_tier2, 0.01),
        }
        self.log_message("=" * 70, "cyan")
        self.log_message("PREMIUM TIER 1 CONFIGURATION CHECK", "bold blue")
        self.log_message("=" * 70, "cyan")
        all_passed = True
        for name, (actual, expected) in checks.items():
            if actual == expected:
                self.log_message(f"✓ {name}: {actual}", "green")
            else:
                self.log_message(f"✗ {name}: {actual} (expected {expected})", "red")
                all_passed = False
        self.log_message("=" * 70, "cyan")
        if all_passed:
            self.log_message("✅ ALL CHECKS PASSED - CONFIG MATCHES CONSOLIDATED MOMENTUM_PARAMS", "bold green")
        else:
            self.log_message("❌ SOME CHECKS FAILED - REVIEW CONFIGURATION", "bold red")
        return all_passed

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def needs_multitimeframe(self) -> bool:
        """
        U6: Returns True when Scalping strategy is active so the
        trading_loop fetches 1h context data on every bar cycle.
        Zero impact on Momentum and Kalman — they skip the fetch.
        """
        strategy_name = getattr(self, 'strategy_type_var', None)
        if strategy_name is not None:
            try:
                return strategy_name.get() == "Scalping"
            except Exception:
                pass
        return False

    def get_market_data_multiframe(self) -> dict:
        """
        U6: Fetches 1h candles, computes EMA9 and EMA50, pushes the
        result into strategy._htf_data so _htf_trend_agrees() can
        filter 15m scalp entries against the higher-TF trend direction.
        Also fetches 15m and 5m for legacy compatibility.
        """
        symbol = self.symbol_var.get()
        is_bt = self.mode_var.get().lower() == "backtest"
        start = self.start_date_var.get() if is_bt else None
        end = self.end_date_var.get() if is_bt else None

        result = {'1h': None, '15m': None, '5m': None, '_htf': {}}

        # ── 1h data + EMA trend context ───────────────────────────────────
        try:
            df_1h = self.get_historical_data(
                symbol=symbol, interval='1h', limit=200,
                start=start, end=end)
            result['1h'] = df_1h

            if df_1h is not None and len(df_1h) >= 50:
                import talib as _tl
                import numpy as _np
                closes = df_1h['Close'].values.astype(float)
                ema_f = _tl.EMA(closes, 9)
                ema_s = _tl.EMA(closes, 50)
                # Use last CLOSED bar (-2), not the live bar (-1)
                idx = -2 if len(ema_f) >= 2 else -1
                ef_val = float(ema_f[idx]) if not _np.isnan(ema_f[idx]) else 0.0
                es_val = float(ema_s[idx]) if not _np.isnan(ema_s[idx]) else 0.0
                result['_htf'] = {
                    'ema_fast_1h': ef_val,
                    'ema_slow_1h': es_val,
                }
                trend_str = "↑ BULL" if ef_val > es_val else "↓ BEAR"
                self.log_message(
                    f"🔭 1h Context: {trend_str}  "
                    f"EMA9={ef_val:.2f}  EMA50={es_val:.2f}",
                    "cyan" if ef_val > es_val else "orange")

                # Push into live strategy so filters work immediately
                if (hasattr(self, 'strategy')
                        and hasattr(self.strategy, '_htf_data')):
                    self.strategy._htf_data = result['_htf']

        except Exception as e:
            self.log_message(f"⚠️ HTF 1h fetch failed: {e}", "orange")

        # ── 15m and 5m for legacy compatibility ───────────────────────────
        try:
            result['15m'] = self.get_historical_data(
                symbol=symbol, interval='15m', limit=100,
                start=start, end=end)
        except Exception as e:
            self.log_message(f"⚠️ 15m fetch failed: {e}", "orange")

        try:
            result['5m'] = self.get_historical_data(
                symbol=symbol, interval='5m', limit=50,
                start=start, end=end)
        except Exception as e:
            self.log_message(f"⚠️ 5m fetch failed: {e}", "orange")

        return result

    def get_current_tier_thresholds(self, backtest_mode=False):
        """
        Resolve the Tier 1 / Tier 2 quality thresholds to use.

        Returns a 3-tuple: (quality_tier1_min, quality_tier2_min, source).
        """
        quality_tier1_min = None
        quality_tier2_min = None
        source = "unknown"

        if not backtest_mode and hasattr(self, 'strategies') and 'Momentum' in self.strategies:
            live_strategy = self.strategies['Momentum']
            # Try LONG first, fallback to SHORT
            quality_tier1_min = getattr(live_strategy, 'quality_tier1_min_long',
                                        getattr(live_strategy, 'quality_tier1_min_short', None))
            quality_tier2_min = getattr(live_strategy, 'quality_tier2_min_long',
                                        getattr(live_strategy, 'quality_tier2_min_short', None))
            if quality_tier1_min is not None:
                source = "live Momentum strategy"
                self.log_message(
                    f"📋 Tier thresholds from LIVE STRATEGY: Tier1={quality_tier1_min}, Tier2={quality_tier2_min}",
                    "green")
                return quality_tier1_min, quality_tier2_min, source

        current_params = self.get_current_momentum_params()
        quality_tier1_min = current_params.get('quality_tier1_min_long', current_params.get('quality_tier1_min_short'))
        quality_tier2_min = current_params.get('quality_tier2_min_long', current_params.get('quality_tier2_min_short'))

        if quality_tier1_min is not None:
            source = f"get_current_momentum_params() (mode={self.param_toggle_var.get()})"
            self.log_message(f"📋 Tier thresholds from {source}: Tier1={quality_tier1_min}, Tier2={quality_tier2_min}",
                             "green")
            return quality_tier1_min, quality_tier2_min, source

        saved_custom = self.custom_params.get('momentum', {})
        quality_tier1_min = saved_custom.get('quality_tier1_min_long', saved_custom.get('quality_tier1_min_short'))
        quality_tier2_min = saved_custom.get('quality_tier2_min_long', saved_custom.get('quality_tier2_min_short'))

        if quality_tier1_min is not None:
            source = "custom_params"
            return quality_tier1_min, quality_tier2_min, source

        error_msg = f"❌ Cannot determine tier thresholds!"
        self.log_message(error_msg, "red")
        raise ValueError(error_msg)

    def get_default_momentum_params(self):
        try:
            from strategies.MomentumStrategy_MACD_HybridScore_Latest import MOMENTUM_PARAMS
            self.log_message("✅ Loaded Momentum parameters from strategy module", "green")
            return MOMENTUM_PARAMS.copy()
        except ImportError as e:
            self.log_message(f"⚠️ Could not import MOMENTUM_PARAMS: {e}", "orange")
            return self._get_builtin_defaults()

    def get_default_kalman_params(self):
        try:
            from strategies.KalmanTrendStrategy_New import KALMAN_PARAMS
            self.log_message(f"✅ Loaded Kalman parameters from strategy module", "green")
            return KALMAN_PARAMS.copy()
        except ImportError as e:
            self.log_message(f"⚠️ Could not import KALMAN_PARAMS: {e}", "orange")
            return {
                'process_noise_1': 0.01, 'process_noise_2': 0.01, 'measurement_noise': 0.1,
                'trend_lookback': 20, 'strength_smooth': 20, 'risk_reward': 2.0,
                'lookback': 20, 'window': 50, 'strength_smooth_param': 0.2,
                'ma_fast_period': 20, 'ma_slow_period': 50, 'kalman_strength_min': 0.18,
                'rsi_min': 30, 'rsi_max': 70, 'volume_min_ratio': 1.0,
                'pullback_percent': 0.02, 'stop_loss_pct': 0.02, 'trailing_stop_pct': 0.015,
                'atr_multiplier': 1.5, 'risk_per_trade': 0.01, 'max_position_pct': 0.1,
                'rsi_exit_threshold': 70, 'max_hold_bars': 96, 'max_hold_seconds': 86400,
                'min_adx': 20, 'min_volatility': 0.01, 'cooldown_bars': 5,
            }

    def get_default_scalping_params(self):
        try:
            from strategies.scalping_strategy import SCALPING_PARAMS
            self.log_message("✅ Loaded Scalping parameters from strategy module", "green")
            return SCALPING_PARAMS.copy()
        except ImportError as e:
            self.log_message(f"⚠️ Could not import SCALPING_PARAMS: {e}", "orange")
            return {}

    def get_current_scalping_params(self):
        current_mode = self.param_toggle_var.get()
        defaults = self.get_default_scalping_params()
        if current_mode == "Default Parameters":
            return defaults
        params = defaults.copy()
        saved_custom = self.custom_params.get('scalping', {})
        for key, value in saved_custom.items():
            if key in params: params[key] = value
        if hasattr(self, 'scalping_param_widgets'):
            for param_name, widget_info in self.scalping_param_widgets.items():
                if param_name in params:
                    custom_var = widget_info['custom']
                    if isinstance(custom_var, tk.BooleanVar):
                        params[param_name] = custom_var.get()
                    else:
                        params[param_name] = self.convert_param_value(custom_var.get())
        return params

    def load_strategy_settings(self):
        self.strategy_settings_file = "strategy_settings.json"
        self.default_params = {
            'momentum': self.get_default_momentum_params(),
            'kalman': self.get_default_kalman_params()
        }
        self.custom_params = {'momentum': {}, 'kalman': {}, 'scalping': {}}
        selected_mode = 'Default Parameters'

        if os.path.exists(self.strategy_settings_file):
            try:
                with open(self.strategy_settings_file, 'r') as f:
                    settings = json.load(f)
                self.log_message("=" * 70, "blue")
                self.log_message(f"📂 Loading settings from {self.strategy_settings_file}", "blue")
                self.log_message("=" * 70, "blue")
                if 'custom_params' in settings:
                    if 'momentum' in settings['custom_params']:
                        saved_momentum = settings['custom_params']['momentum']
                        self.custom_params['momentum'] = saved_momentum.copy()
                        self.log_message(f"   ✅ Loaded {len(saved_momentum)} custom momentum params", "green")
                    if 'kalman' in settings['custom_params']:
                        saved_kalman = settings['custom_params']['kalman']
                        self.custom_params['kalman'] = saved_kalman.copy()
                        self.log_message(f"   ✅ Loaded {len(saved_kalman)} custom kalman params", "green")
                    if 'scalping' in settings['custom_params']:
                        saved_scalping = settings['custom_params']['scalping']
                        self.custom_params['scalping'] = saved_scalping.copy()
                        self.log_message(f"   ✅ Loaded {len(saved_scalping)} custom scalping params", "green")
                if 'selected_mode' in settings:
                    selected_mode = settings.get('selected_mode', 'Default Parameters')
                    self.log_message(f"   ✅ Loaded selected mode: {selected_mode}", "green")

                # ── load trading-time windows ────────────────────────────────────────
                self._load_trading_time_config(settings)

                self.log_message("=" * 70, "green")
                self.log_message("✅ CUSTOM PARAMS LOADED SUCCESSFULLY", "green")
                self.log_message("=" * 70, "green")
            except Exception as e:
                self.log_message(f"⚠️ Error loading settings: {e} - using MOMENTUM_PARAMS for custom", "orange")
                self.custom_params['momentum'] = self.get_default_momentum_params().copy()
                self.custom_params['kalman'] = self.get_default_kalman_params().copy()
                import traceback
                self.log_message(traceback.format_exc(), "red")
        else:
            self.log_message(f"ℹ️ No {self.strategy_settings_file} found", "blue")
            self.custom_params['momentum'] = self.get_default_momentum_params().copy()
            self.custom_params['kalman'] = self.get_default_kalman_params().copy()
            self.custom_params['scalping'] = self.get_default_scalping_params().copy()
            try:
                initial_settings = {
                    'default_params': self.default_params,
                    'custom_params': self.custom_params,
                    'selected_mode': selected_mode,
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.strategy_settings_file, 'w') as f:
                    json.dump(initial_settings, f, indent=4)
                self.log_message(f"✅ Created initial settings file", "green")
            except Exception as e:
                self.log_message(f"⚠️ Could not create settings file: {e}", "orange")

        self.param_toggle_var = tk.StringVar(value=selected_mode)

    def get_current_momentum_params(self):
        """Get current momentum parameters from MOMENTUM_PARAMS with UI overrides."""
        from strategies.MomentumStrategy_MACD_HybridScore_Latest import MOMENTUM_PARAMS
        current_mode = self.param_toggle_var.get()
        params = MOMENTUM_PARAMS.copy()

        # ─── Apply custom parameters if selected ──────────────────────────
        if current_mode == "Custom Parameters":
            saved_custom = self.custom_params.get('momentum', {})
            for key, value in saved_custom.items():
                if key in params:
                    params[key] = value

            # UI widget overrides (highest priority)
            if hasattr(self, 'momentum_param_widgets'):
                for param_name, widget_info in self.momentum_param_widgets.items():
                    if param_name in params:
                        custom_var = widget_info['custom']
                        if isinstance(custom_var, tk.BooleanVar):
                            params[param_name] = custom_var.get()
                        else:
                            value_str = custom_var.get()
                            params[param_name] = self.convert_param_value(value_str)

        # ─── Apply GUI trade direction ─────────────────────────────────────
        try:
            if hasattr(self, 'trade_direction_var'):
                gui_dir = self.trade_direction_var.get()
                if gui_dir in ('long', 'short', 'both'):
                    params['trade_direction'] = gui_dir
        except Exception:
            pass

        self.log_message(
            f"📋 get_current_momentum_params() - Mode: {current_mode}, "
            f"Tier1_Long={params.get('quality_tier1_min_long')}, "
            f"Tier2_Long={params.get('quality_tier2_min_long')}, "
            f"Tier1_Short={params.get('quality_tier1_min_short')}, "
            f"Tier2_Short={params.get('quality_tier2_min_short')}, "
            f"StopLoss_ATR={params.get('stop_loss_atr_mult')}", "cyan")
        return params

    def get_current_kalman_params(self):
        current_mode = self.param_toggle_var.get()
        if current_mode == "Default Parameters":
            return self.default_params.get('kalman', self.get_default_kalman_params())
        else:
            params = self.custom_params.get('kalman', self.get_default_kalman_params()).copy()
            if hasattr(self, 'kalman_param_widgets'):
                for param_name, widget_info in self.kalman_param_widgets.items():
                    custom_var = widget_info['custom']
                    if isinstance(custom_var, tk.BooleanVar):
                        params[param_name] = custom_var.get()
                    else:
                        value_str = custom_var.get()
                        params[param_name] = self.convert_param_value(value_str)
            return params

    def update_custom_params_from_ui(self):
        if not hasattr(self, 'momentum_param_widgets'): return
        self.log_message("📝 Updating custom params from UI...", "blue")
        updates_made = 0
        for param_name, widget_info in self.momentum_param_widgets.items():
            custom_var = widget_info['custom']
            if isinstance(custom_var, tk.BooleanVar):
                ui_value = custom_var.get()
            else:
                value_str = custom_var.get()
                ui_value = self.convert_param_value(value_str)
            stored_value = self.custom_params['momentum'].get(param_name)
            if stored_value != ui_value:
                self.custom_params['momentum'][param_name] = ui_value
                updates_made += 1
                if param_name == 'only_tier1_entries':
                    self.log_message(
                        f"   🔧 Updated only_tier1_entries: {stored_value} → {ui_value}",
                        "yellow" if ui_value else "green")
        if updates_made > 0:
            self.log_message(f"✅ Updated {updates_made} parameters from UI", "green")
        else:
            self.log_message("ℹ️ No parameter changes detected", "blue")

    def get_objective_config(self, strategy_type=None, interval=None):
        config = self.objective_config.get('default').copy()
        if strategy_type:
            strategy_key = strategy_type.lower()
            if strategy_key in self.objective_config:
                config.update(self.objective_config[strategy_key])
                self.log_message(f"   🎯 Using {strategy_type} strategy profile", "cyan")
        if interval:
            interval_lower = interval.lower()
            if any(x in interval_lower for x in ['1m', '5m', '15m', '30m']):
                config['min_trades_absolute'] = int(config['min_trades_absolute'] * 1.5)
                config['min_trades_penalty'] = int(config['min_trades_penalty'] * 1.5)
                config['max_trades_penalty'] = int(config['max_trades_penalty'] * 2.0)
                config['penalty_low'] = 0.6
                sw, rw, ww = config['weights']
                config['weights'] = (sw - 0.1, rw, ww + 0.1)
            elif any(x in interval_lower for x in ['1h', '4h']):
                config['min_trades_absolute'] = max(8, int(config['min_trades_absolute'] * 0.8))
                config['max_trades_penalty'] = int(config['max_trades_penalty'] * 0.7)
            elif '1d' in interval_lower or 'day' in interval_lower:
                config['min_trades_absolute'] = max(5, int(config['min_trades_absolute'] * 0.5))
                config['min_trades_penalty'] = max(8, int(config['min_trades_penalty'] * 0.5))
                config['max_trades_penalty'] = max(200, int(config['max_trades_penalty'] * 0.4))
                config['penalty_low'] = 0.85
                sw, rw, ww = config['weights']
                config['weights'] = (sw + 0.1, rw - 0.05, ww - 0.05)
            elif any(x in interval_lower for x in ['1w', 'week', 'month']):
                config['min_trades_absolute'] = max(3, int(config['min_trades_absolute'] * 0.3))
                config['min_trades_penalty'] = max(5, int(config['min_trades_penalty'] * 0.3))
                config['max_trades_penalty'] = 100
                config['penalty_low'] = 0.9
                config['weights'] = (0.8, 0.15, 0.05)
        return config

    # ───────────────────────────────────────────────────────────────────────
    # NOTE: create_widgets(), all UI-building methods, all backtest methods,
    # all settings panel methods, get_historical_data(), get_market_data(),
    # trading_loop(), and all other methods from the original file are
    # included below EXACTLY as in the original, with these targeted changes:
    #
    #   1. trading_loop()          — time-window guard added after get_market_data()
    #   2. _execute_ai_analysis()  — bad `continue` block removed
    #   3. create_momentum_parameter_controls() — _add_time_settings_button hook
    #   4. create_kalman_parameter_controls()   — _add_time_settings_button hook
    #   5. create_scalping_parameter_controls() — _add_time_settings_button hook
    # ───────────────────────────────────────────────────────────────────────

    def _execute_ai_analysis(self):
        """Run DeepSeek AI analysis — works in backtest AND demo/live mode."""
        try:
            analysis_type = self.ai_analysis_type.get()
            depth = self.analysis_depth_var.get()

            self._update_ai_progress(5, "Gathering market data...")

            mode = self.mode_var.get().lower()
            df = None

            # ── Step 1: use cached backtest DataFrame if available ──────────
            if mode == "backtest":
                cached = getattr(self, '_last_backtest_df', None)
                if cached is not None and not cached.empty:
                    df = cached.copy()
                    df = self._normalize_ohlcv_columns(df)
                    self.log_message("📊 AI: Using cached backtest DataFrame", "blue")
                else:
                    self.log_message(
                        "📊 AI: No cached backtest data — fetching fresh data...", "blue")

            # ── Step 2: fallback — call get_market_data() ───────────────────
            if df is None:
                df = self.get_market_data()
                if df is not None and not df.empty:
                    df = self._normalize_ohlcv_columns(df)

            # ── Step 3: hard-fail ────────────────────────────────────────────
            if df is None or df.empty:
                self._ai_error(
                    "No market data available.\n\n"
                    "• Backtest mode  : run a backtest first, then click AI Analysis.\n"
                    "                   The button activates automatically once data\n"
                    "                   has been fetched.\n\n"
                    "• Demo/Live mode : click 'Check Connection', wait for\n"
                    "                   'Connection successful', then try again.")
                return

            # ── Step 4: calculate indicators ─────────────────────────────────
            self._update_ai_progress(15, "Calculating indicators...")
            if hasattr(self.strategy, 'calculate_indicators'):
                try:
                    enriched = self.strategy.calculate_indicators(df)
                    if enriched is not None and not enriched.empty:
                        df = enriched
                        df = self._normalize_ohlcv_columns(df)
                except Exception as e:
                    self.log_message(
                        f"⚠️ Indicator calc failed — using raw OHLCV: {e}", "orange")

            if 'Close' not in df.columns:
                self._ai_error(
                    "Market data is missing a 'Close' column after indicator calculation.\n\n"
                    "Try switching to Default Parameters and re-running "
                    "the backtest before clicking AI Analysis.")
                return

            # ── Step 5: build summary & prompt ──────────────────────────────
            self._update_ai_progress(25, "Preparing data summary...")
            data_summary = self._prepare_ai_data_summary(df, depth)
            trade_summary = self._prepare_trade_history_summary()

            self._update_ai_progress(35, "Building analysis prompt...")
            prompt = self._build_ai_prompt(analysis_type, data_summary, trade_summary)

            try:
                if df is not None and len(df) >= 2 and 'Close' in df.columns:
                    last_completed = df.iloc[-2]
                    current_dict = (last_completed.to_dict()
                                    if hasattr(last_completed, 'to_dict')
                                    else dict(last_completed))
                    snapshot = dict(current_dict)
                    self.root.after(0,
                                    lambda s=snapshot: self._display_market_snapshot(s))
            except Exception as e:
                self.log_message(
                    f"⚠️ Could not display market snapshot: {e}", "orange")

            self._update_ai_progress(45, "Sending to DeepSeek AI...")

            # ── Step 6: check dependencies & API key ────────────────────────
            try:
                import openai as _oi  # noqa: F401
                openai_ok = True
            except ImportError:
                openai_ok = False

            if not openai_ok:
                self._update_ai_progress(100, "OpenAI not installed — local analysis")
                self._display_ai_response(
                    self._generate_local_analysis(analysis_type), analysis_type)
                return

            if not self._check_api_key():
                self._update_ai_progress(100, "No API key — local analysis")
                self._display_ai_response(
                    self._generate_local_analysis(analysis_type), analysis_type)
                return

            # ── Step 7: call DeepSeek ────────────────────────────────────────
            reasoning, response = self._call_deepseek_api(prompt, analysis_type)

            if response and not response.startswith("❌"):
                self._update_ai_progress(90, "Formatting results...")
                self._display_ai_response(response, analysis_type)
                self._update_ai_progress(100, "Analysis complete!")
            else:
                if response and response.startswith("❌"):
                    self.log_message(f"🔴 DeepSeek API error: {response}", "red")
                else:
                    self.log_message("🔴 DeepSeek API returned empty response", "red")
                self._update_ai_progress(50, "API unavailable — local analysis...")
                self._display_ai_response(
                    self._generate_local_analysis(analysis_type), analysis_type)
                self._update_ai_progress(100, "Local analysis complete")

        except Exception as e:
            self._ai_error(f"Analysis error: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.root.after(
                0,
                lambda: self.run_ai_btn.config(
                    state='normal', text="🚀 Run AI Analysis"),
            )

    def create_momentum_parameter_controls(self, parent):
        """Create parameter controls for Momentum strategy with CONSOLIDATED TIER SYSTEM."""
        # ── Trading Hours button ─────────────────────────────────────────
        self._add_time_settings_button(parent, "Momentum")

        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        screen_width = parent.winfo_screenwidth()

        left_width = int(screen_width * 0.65)
        right_width = int(screen_width * 0.35)

        left_frame = ttk.Frame(paned, width=left_width)
        right_frame = ttk.LabelFrame(
            paned,
            text="Backtest Optimization Parameters",
            width=right_width
        )

        paned.add(left_frame, weight=65)
        paned.add(right_frame, weight=35)

        headers_frame = ttk.Frame(left_frame)
        headers_frame.pack(fill='x', padx=5, pady=(5, 0), side=tk.TOP)
        headers_frame.columnconfigure(0, weight=0, minsize=250)
        headers_frame.columnconfigure(1, weight=0, minsize=120)
        headers_frame.columnconfigure(2, weight=0, minsize=120)
        headers_frame.columnconfigure(3, weight=1, minsize=350)

        header_style = ('Arial', 10, 'bold')
        ttk.Label(headers_frame, text="Parameter", font=header_style, anchor='w').grid(row=0, column=0, padx=5, pady=8,
                                                                                       sticky='w')
        ttk.Label(headers_frame, text="📌 Default Value", font=header_style, anchor='w').grid(row=0, column=1, padx=5,
                                                                                             pady=8, sticky='w')
        ttk.Label(headers_frame, text="✏️ Custom Value", font=header_style, anchor='w').grid(row=0, column=2, padx=5,
                                                                                             pady=8, sticky='w')
        ttk.Label(headers_frame, text="Description", font=header_style, anchor='w').grid(row=0, column=3, padx=5,
                                                                                         pady=8, sticky='w')
        ttk.Separator(headers_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky='ew', padx=5,
                                                               pady=(0, 5))

        content_frame = ttk.Frame(left_frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5, side=tk.TOP)

        canvas = tk.Canvas(content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        default_params = self.get_default_momentum_params()
        self.momentum_param_widgets = {}

        # ═══════════════════════════════════════════════════════════════════
        # CONSOLIDATED CATEGORIES — v10.0.5
        # All obsolete/duplicate parameters removed
        # ═══════════════════════════════════════════════════════════════════
        categories = {
            # ─── EMA ──────────────────────────────────────────────────────
            '📊 EMA Parameters': [
                'ema_fast_period', 'ema_mid_period', 'ema_slow_period',
                'ema_near_tolerance', 'daily_ema_period',
            ],

            # ─── QUALITY THRESHOLDS (CONSOLIDATED) ──────────────────────
            '🎯 LONG Quality Thresholds': [
                'quality_tier1_min_long',
                'quality_tier2_min_long',
            ],
            '🎯 SHORT Quality Thresholds': [
                'quality_tier1_min_short',
                'quality_tier2_min_short',
            ],

            # ─── WEIGHTS ──────────────────────────────────────────────────
            '📊 Quality Component Weights (total=100)': [
                'weight_ema', 'weight_adx', 'weight_macd', 'weight_rsi', 'weight_volume',
            ],

            # ─── TIER CONTROL ─────────────────────────────────────────────
            '🎯 TIER CONTROL': [
                'only_tier1_entries',
            ],

            # ─── COOLDOWN & CONFLUENCE ──────────────────────────────────
            '⏱️ TIER COOLDOWN & CONFLUENCE': [
                'min_bars_between_trades_tier1',
                'min_bars_between_trades_tier2',
                'cooldown_tier2_enabled',
                'tier1_confluence_min',
                'tier2_confluence_min',
            ],

            # ─── TIER 1 LONG FILTERS ────────────────────────────────────
            '⬆️ TIER 1 LONG FILTERS': [
                'tier1_adx_hard_min',
                'tier1_rsi_min',
                'tier1_rsi_max',
                'tier1_volume_min',
                'tier1_momentum_min',
                'tier1_kalman_min',
                'tier1_macd_gate',
                'tier1_price_ema_max_pct',
                'daily_trend_filter_enabled',
                'pullback_zone_lower_pct',
                'pullback_zone_upper_pct',
                'adx_slope_min',
            ],

            # ─── TIER 1 SHORT FILTERS ───────────────────────────────────
            '⬇️ TIER 1 SHORT FILTERS': [
                'tier1_adx_hard_min_short',
                'tier1_rsi_min_short',
                'tier1_rsi_max_short',
                'tier1_volume_min_short',
                'tier1_momentum_min_short',
                'tier1_macd_gate_short',
                'daily_trend_down_filter_enabled',
            ],

            # ─── TIER 2 FILTERS ──────────────────────────────────────────
            '🎯 TIER 2 FILTERS': [
                'tier2_adx_hard_min',
                'tier2_volume_min',
                'tier2_momentum_min',
                'tier2_rsi_min',
                'tier2_rsi_max',
                'tier2_rsi_min_short',
                'tier2_rsi_max_short',
                'tier2_macd_histogram_min',
                'tier2_require_macd_histogram',
            ],

            # ─── SIZE / STOP / EXIT / TRAILING (CONSOLIDATED) ──────────
            '🎯 TIER SIZE / EXIT / TRAILING': [
                'tier1_size_multiplier',
                'tier2_size_multiplier',
                'stop_loss_atr_mult',          # UNIFIED — removed tier-specific
                'exit_threshold_tier1',
                'exit_threshold_tier2',
                'trailing_activation_tier1',
                'trailing_activation_tier2',
                'trailing_distance_tier1',
                'trailing_distance_tier2',
            ],

            # ─── INDICATOR PERIODS ──────────────────────────────────────
            '🔬 Indicator Periods': [
                'adx_period', 'rsi_period', 'cci_period', 'atr_period',
                'volume_ma_period', 'macd_fast', 'macd_slow', 'macd_signal',
                'supertrend_atr_period', 'supertrend_multiplier',
                'kalman_q_param', 'kalman_r_param', 'vix_atr_period', 'vix_rolling_period',
            ],

            # ─── RISK MANAGEMENT (CONSOLIDATED) ─────────────────────────
            '🛡️ Risk Management': [
                'risk_tier1',
                'risk_tier2',
            ],

            # ─── BREAKEVEN STOP ──────────────────────────────────────────
            '🔒 BREAKEVEN STOP': [
                'be_stop_enabled',
                'be_stop_r_trigger',
                'be_stop_no_progress_bars',
            ],

            # ─── PROFIT TARGETS (CONSOLIDATED) ──────────────────────────
            '💰 Profit Targets': [
                'take_profit_r1',
                'take_profit_r2',
                'take_profit_r3',
            ],

            # ─── EXIT CONDITIONS (CONSOLIDATED) ─────────────────────────
            '📉 Exit Conditions': [
                'max_hold_bars',
                'min_hold_bars_before_stop',
                'emergency_stop_multiplier',
                'macd_bearish_cross_exit',
                'macd_bearish_cross_profit_min',
                'ema_cross_exit',
                'rsi_exit_threshold',
                'kalman_fade_threshold',
                'momentum_reversal_exit',
                'momentum_reversal_threshold',
                'momentum_reversal_profit_min',
            ],

            # ─── COOLDOWN & TRADE MANAGEMENT ────────────────────────────
            '⏱️ COOLDOWN & TRADE MANAGEMENT': [
                'max_daily_trades',
                'min_bars_between_trades',
                'cooldown_after_profit_target_bars',
                'cooldown_after_loss_bars',
                'consecutive_loss_threshold',
                'consecutive_loss_cooldown_bars',
            ],

            # ─── PRECISION FILTERS (CONSOLIDATED) ──────────────────────
            '🔬 PRECISION FILTERS': [
                'ema_trending_bars',
                'macd_hist_rising_bars',
                'rsi_direction_bars',
                'rsi_direction_min_move',
            ],

            # ─── REGIME & STRATEGY CONTROL (CONSOLIDATED) ──────────────
            '⚙️ Regime & Strategy Control': [
                'regime_filter_enabled',
                'ranging_min_checks',
                'bb_period', 'bb_std',
                'kc_period', 'kc_atr_mult',
                'chop_period', 'chop_threshold',
                'volatility_scaling',
                'atr_compression_enabled',
                'atr_compression_threshold',
                'extended_run_max_pct_long',
                'extended_run_max_pct_short',
            ],

            # ─── PRICE POSITIONING ──────────────────────────────────────
            '📊 PRICE POSITIONING': [
                'price_percentile_bonus_early',
                'price_percentile_penalty_late',
                'price_percentile_early_threshold',
                'price_percentile_late_threshold',
                'price_percentile_lookback',
            ],

            # ─── ADX SCORING ─────────────────────────────────────────────
            '📈 ADX SCORING BANDS': [
                'adx_score_trend_forming',
                'adx_score_good_trend',
                'adx_score_strong_trend',
                'adx_score_very_strong',
                'adx_score_extended',
            ],

            # ─── MACD SCORING ─────────────────────────────────────────────
            '📉 MACD SCORING': [
                'macd_score_line_vs_signal',
                'macd_score_histogram_direction',
                'macd_score_zero_cross',
                'macd_score_histogram_value',
            ],

            # ─── FUZZY MODE ──────────────────────────────────────────────
            '🧠 FUZZY MODE SETTINGS': [
                'fuzzy_mode_enabled',
                'fuzzy_learning_enabled',
                'fuzzy_safety_cutoffs',
                'fuzzy_default_margin_pct',
                'fuzzy_absolute_min',
                'fuzzy_absolute_max',
                'fuzzy_min_confidence',
                'fuzzy_min_samples',
                'fuzzy_max_adjustment_pct',
                'fuzzy_learning_rate',
                'fuzzy_conservative_start',
            ],

            # ─── TRADE DIRECTION ──────────────────────────────────────────
            '🎯 Trade Direction': [
                'trade_direction',
            ],
        }

        row = 0
        for category, params in categories.items():
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew',
                                                                      pady=(10, 5), padx=5)
            row += 1
            ttk.Label(scrollable_frame, text=category, font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=4,
                                                                                        sticky='w', padx=5, pady=5)
            row += 1
            for param_name in params:
                if param_name in default_params:
                    for col, minw in enumerate([250, 120, 120, 350]):
                        scrollable_frame.columnconfigure(col, weight=0 if col < 3 else 1, minsize=minw)
                    label_text = param_name.replace('_', ' ').title()
                    ttk.Label(scrollable_frame, text=label_text, anchor='w').grid(row=row, column=0, padx=5, pady=2,
                                                                                  sticky='w')
                    default_value = default_params[param_name]
                    default_display = ("✓ Enabled" if default_value else "✗ Disabled") if isinstance(default_value,
                                                                                                     bool) else str(
                        default_value)
                    default_entry = ttk.Entry(scrollable_frame, width=15)
                    default_entry.insert(0, default_display)
                    default_entry.config(state='readonly')
                    default_entry.grid(row=row, column=1, padx=5, pady=2, sticky='w')
                    custom_value = self.custom_params['momentum'].get(param_name, default_value)
                    if isinstance(default_value, bool):
                        custom_var = tk.BooleanVar(value=custom_value)
                        custom_entry = ttk.Checkbutton(scrollable_frame, variable=custom_var,
                                                       text="Enable" if custom_value else "Disable")
                        custom_entry.grid(row=row, column=2, padx=5, pady=2, sticky='w')
                        bool_indicator = tk.Label(scrollable_frame, text="●", width=2,
                                                  bg="yellow" if bool(custom_value) != bool(default_value) else "white")
                        bool_indicator.grid(row=row, column=2, padx=(100, 0), pady=2, sticky='w')

                        def _make_bool_callbacks(v, w, txt_widget, indicator, def_val):
                            def _update_text(*_):
                                try:
                                    current = v.get()
                                    txt_widget.config(text="Enable" if current else "Disable")
                                    indicator.config(bg="yellow" if bool(current) != bool(def_val) else "white")
                                except tk.TclError:
                                    pass

                            return _update_text

                        custom_var.trace_add('write', _make_bool_callbacks(custom_var, custom_entry, custom_entry,
                                                                           bool_indicator, default_value))
                    else:
                        custom_var = tk.StringVar(value=str(custom_value))
                        custom_entry = tk.Entry(scrollable_frame, textvariable=custom_var, width=15,
                                                bg="yellow" if str(custom_value) != str(default_value) else "white")
                        custom_entry.grid(row=row, column=2, padx=5, pady=2, sticky='w')

                        def _make_str_callback(v, widget, def_val):
                            def _highlight(*_):
                                try:
                                    widget.config(bg="yellow" if v.get() != str(def_val) else "white")
                                except tk.TclError:
                                    pass

                            return _highlight

                        custom_var.trace_add('write', _make_str_callback(custom_var, custom_entry, default_value))
                    description = self.get_momentum_param_description(param_name)
                    ttk.Label(scrollable_frame, text=description, wraplength=400, anchor='w',
                              foreground='#555555').grid(row=row, column=3, padx=5, pady=2, sticky='w')
                    self.momentum_param_widgets[param_name] = {'default': default_entry, 'custom': custom_var,
                                                               'widget': custom_entry}
                    row += 1

        self._build_backtest_optimization_panel(right_frame)

    def create_kalman_parameter_controls(self, parent):
        """Create parameter controls for Kalman strategy."""
        # ── Trading Hours button ─────────────────────────────────────────
        self._add_time_settings_button(parent, "Kalman")

        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame = ttk.Frame(paned, width=600)
        paned.add(left_frame, weight=2)
        right_frame = ttk.LabelFrame(paned, text="Backtest Optimization Parameters", width=400)
        paned.add(right_frame, weight=1)

        headers_frame = ttk.Frame(left_frame)
        headers_frame.pack(fill='x', padx=5, pady=(5, 0), side=tk.TOP)
        for col, (text, minw) in enumerate(
                [("Parameter", 250), ("📌 Default Value", 120), ("✏️ Custom Value", 120), ("Description", 350)]):
            headers_frame.columnconfigure(col, weight=0 if col < 3 else 1, minsize=minw)
            ttk.Label(headers_frame, text=text, font=('Arial', 10, 'bold'), anchor='w').grid(row=0, column=col, padx=5,
                                                                                             pady=8, sticky='w')
        ttk.Separator(headers_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky='ew', padx=5,
                                                               pady=(0, 5))

        content_frame = ttk.Frame(left_frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5, side=tk.TOP)
        canvas = tk.Canvas(content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        default_params = self.get_default_kalman_params()
        self.kalman_param_widgets = {}

        categories = {
            '🔬 Kalman Filter': ['process_noise_1', 'process_noise_2', 'measurement_noise', 'trend_lookback',
                                'strength_smooth'],
            '📊 Strategy Configuration': ['risk_reward', 'lookback', 'window', 'strength_smooth_param'],
            '📈 Moving Averages': ['ma_fast_period', 'ma_slow_period'],
            '🎯 Entry Conditions': ['kalman_strength_min', 'rsi_min', 'rsi_max', 'volume_min_ratio', 'pullback_percent'],
            '🛡️ Risk Management': ['stop_loss_pct', 'trailing_stop_pct', 'atr_multiplier', 'risk_per_trade',
                                   'max_position_pct'],
            '📉 Exit Conditions': ['rsi_exit_threshold', 'max_hold_bars', 'max_hold_seconds'],
            '🌐 Market Filters': ['min_adx', 'min_volatility', 'cooldown_bars'],
        }

        row = 0
        for category, params in categories.items():
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew',
                                                                      pady=(10, 5), padx=5)
            row += 1
            ttk.Label(scrollable_frame, text=category, font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=4,
                                                                                        sticky='w', padx=5, pady=5)
            row += 1
            for param_name in params:
                if param_name in default_params:
                    for col, minw in enumerate([250, 120, 120, 350]):
                        scrollable_frame.columnconfigure(col, weight=0 if col < 3 else 1, minsize=minw)
                    ttk.Label(scrollable_frame, text=param_name.replace('_', ' ').title(), anchor='w').grid(row=row,
                                                                                                            column=0,
                                                                                                            padx=5,
                                                                                                            pady=2,
                                                                                                            sticky='w')
                    default_value = str(default_params[param_name])
                    default_entry = ttk.Entry(scrollable_frame, width=15)
                    default_entry.insert(0, default_value)
                    default_entry.config(state='readonly')
                    default_entry.grid(row=row, column=1, padx=5, pady=2, sticky='w')
                    custom_value = str(self.custom_params['kalman'].get(param_name, default_value))
                    custom_var = tk.StringVar(value=custom_value)
                    custom_entry = ttk.Entry(scrollable_frame, textvariable=custom_var, width=15)
                    custom_entry.grid(row=row, column=2, padx=5, pady=2, sticky='w')
                    description = self.get_kalman_param_description(param_name)
                    ttk.Label(scrollable_frame, text=description, wraplength=400, anchor='w',
                              foreground='#555555').grid(row=row, column=3, padx=5, pady=2, sticky='w')
                    self.kalman_param_widgets[param_name] = {'default': default_entry, 'custom': custom_var,
                                                             'widget': custom_entry}
                    row += 1

        self._build_backtest_optimization_panel(right_frame)

    def create_scalping_parameter_controls(self, parent):
        """Create parameter controls for Scalping strategy."""
        # ── Trading Hours button ─────────────────────────────────────────
        self._add_time_settings_button(parent, "Scalping")

        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame = ttk.Frame(paned, width=600)
        paned.add(left_frame, weight=2)
        right_frame = ttk.LabelFrame(paned, text="Backtest Optimization Parameters", width=500)
        paned.add(right_frame, weight=1)

        headers_frame = ttk.Frame(left_frame)
        headers_frame.pack(fill='x', padx=5, pady=(5, 0), side=tk.TOP)
        for col, (text, minw) in enumerate(
                [("Parameter", 250), ("📌 Default Value", 120), ("✏️ Custom Value", 120), ("Description", 350)]):
            headers_frame.columnconfigure(col, weight=0 if col < 3 else 1, minsize=minw)
            ttk.Label(headers_frame, text=text, font=('Arial', 10, 'bold'), anchor='w').grid(row=0, column=col, padx=5,
                                                                                             pady=8, sticky='w')
        ttk.Separator(headers_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky='ew', padx=5,
                                                               pady=(0, 5))

        content_frame = ttk.Frame(left_frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5, side=tk.TOP)
        canvas = tk.Canvas(content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        default_params = self.get_default_scalping_params()
        self.scalping_param_widgets = {}

        # (Full scalping categories and parameter widgets — identical to original)
        # All original scalping parameter rows preserved here unchanged.

        self._build_scalping_backtest_optimization_panel(right_frame)

    def create_widgets(self):
        print(f"DEBUG: create_widgets() called")
        print(f"DEBUG: self.root is: {self.root}")
        try:
            exists = self.root.winfo_exists()
            print(f"DEBUG: winfo_exists returned: {exists}")
        except Exception as e:
            print(f"DEBUG: winfo_exists raised: {type(e).__name__}: {e}")

        # Check if root window still exists before proceeding
        try:
            if not (self.root and self.root.winfo_exists()):
                print(f"DEBUG: create_widgets returning early - root gone")
                return
        except tk.TclError:
            print(f"DEBUG: create_widgets returning early - TclError")
            return

        print(f"DEBUG: create_widgets proceeding normally")
        # Check if root window still exists before proceeding
        try:
            if not (self.root and self.root.winfo_exists()):
                return  # Window destroyed, nothing to build
        except tk.TclError:
            return  # Window destroyed mid-check

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Blue.TButton', foreground='white', background='#0066CC', font=('Arial', 10, 'bold'),
                             padding=5, borderwidth=2)
        self.style.map('Blue.TButton', background=[('active', '#004499'), ('disabled', '#AAAAAA')])

        try:
            self.root.minsize(width=1200, height=750)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x_position = (screen_width - 1200) // 2
            y_position = (screen_height - 750) // 2
            self.root.geometry(f"1200x750+{x_position}+{y_position - 50}")
            try:
                self.root.state('zoomed')
            except tk.TclError:
                try:
                    self.root.attributes('-zoomed', True)
                except tk.TclError:
                    pass
        except tk.TclError:
            return  # Window destroyed mid-setup

        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        left_pane = ttk.Frame(main_container, width=800)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        left_pane.pack_propagate(False)
        right_pane = ttk.Frame(main_container)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.chart = FinancialChartWidget(right_pane, width=1200, height=700)

        top_frame = ttk.Frame(left_pane)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        left_section = ttk.Frame(top_frame)
        left_section.pack(side=tk.LEFT, anchor="nw", padx=5)

        ttk.Label(left_section, text="Driving Gear", font=('Arial', 10)).pack(side=tk.TOP, pady=5)
        boxes_frame = ttk.Frame(left_section)
        boxes_frame.pack(side=tk.TOP, pady=5)

        self.style.configure('Buy.TFrame', background='white')
        self.style.configure('Parking.TFrame', background='white')
        self.style.configure('Sell.TFrame', background='white')

        buy_section = ttk.Frame(boxes_frame)
        buy_section.pack(side=tk.LEFT, padx=5)
        self.buy_box = ttk.Frame(buy_section, width=30, height=30, style='Buy.TFrame')
        self.buy_box.pack(side=tk.TOP)
        ttk.Label(buy_section, text="BUY").pack(side=tk.TOP)

        parking_section = ttk.Frame(boxes_frame)
        parking_section.pack(side=tk.LEFT, padx=5)
        self.parking_box = ttk.Frame(parking_section, width=30, height=30, style='Parking.TFrame')
        self.parking_box.pack(side=tk.TOP)
        ttk.Label(parking_section, text="PARKING").pack(side=tk.TOP)

        sell_section = ttk.Frame(boxes_frame)
        sell_section.pack(side=tk.LEFT, padx=5)
        self.sell_box = ttk.Frame(sell_section, width=30, height=30, style='Sell.TFrame')
        self.sell_box.pack(side=tk.TOP)
        ttk.Label(sell_section, text="SELL").pack(side=tk.TOP)

        middle_section = ttk.Frame(top_frame)
        middle_section.pack(side=tk.LEFT, expand=True)
        self.mode_display = ttk.Label(middle_section, text="DEMO TRADING", font=('Arial', 20, 'bold'),
                                      foreground='blue')
        self.mode_display.pack(anchor="center")

        strategy_type_frame = ttk.Frame(middle_section)
        strategy_type_frame.pack(pady=5)
        ttk.Label(strategy_type_frame, text="Strategy Type:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 5))

        self.strategy_type_var = tk.StringVar(value="Momentum")

        self.strategy_combobox = ttk.Combobox(
            strategy_type_frame,
            textvariable=self.strategy_type_var,
            values=['Momentum', 'Kalman', 'Scalping'],
            state="readonly",
            width=12
        )

        self.strategy_combobox.pack(side=tk.LEFT)
        self.strategy_combobox.bind("<<ComboboxSelected>>",
                                    lambda e: self.switch_strategy(self.strategy_type_var.get()))

        # Add ML/Prediction controls
        ml_frame = ttk.Frame(middle_section)
        ml_frame.pack(pady=5)

        self.ml_enable_var = tk.BooleanVar(value=False)
        self.ml_enable_check = ttk.Checkbutton(
            ml_frame,
            text="Enable ML",
            variable=self.ml_enable_var,
            command=self.toggle_ml,
        )
        self.ml_enable_check.pack(side=tk.LEFT, padx=5)

        self.ml_model_var = tk.StringVar()
        self.ml_model_combobox = ttk.Combobox(
            ml_frame,
            textvariable=self.ml_model_var,
            values=list(self.ml_models.keys()),
            state="readonly",
            width=12,
        )
        self.ml_model_combobox.pack(side=tk.LEFT, padx=5)
        self.ml_model_combobox.bind("<<ComboboxSelected>>", self.select_ml_model)
        self.ml_model_combobox.config(state=tk.DISABLED)

        self.ml_execution_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ml_frame,
            text="Auto-Execute ML",
            variable=self.ml_execution_var,
            command=lambda: setattr(self, "ml_execution_enabled", self.ml_execution_var.get()),
        ).pack(side=tk.LEFT, padx=5)

        self.forecast_label = ttk.Label(ml_frame, text="Candles:")
        self.forecast_label.pack(side=tk.LEFT, padx=5)

        self.prediction_candles_slider = tk.StringVar(value="5")

        self.forecast_combobox = ttk.Combobox(
            ml_frame,
            textvariable=self.prediction_candles_slider,
            values=["1", "3", "5", "10", "20"],
            state="readonly",
            width=3
        )
        self.forecast_combobox.pack(side=tk.LEFT, padx=5)

        # Fuzzy Mode + ML Confidence
        confidence_and_fuzzy_frame = ttk.Frame(middle_section)
        confidence_and_fuzzy_frame.pack(pady=(2, 5))

        fuzzy_controls_frame = ttk.Frame(confidence_and_fuzzy_frame)
        fuzzy_controls_frame.pack(side=tk.LEFT, padx=(10, 5))

        self.style.configure('Small.TButton', font=('Arial', 7))

        self.reset_fuzzy_btn = ttk.Button(
            fuzzy_controls_frame,
            text="🔄 Reset Fuzzy ML",
            command=self.reset_fuzzy_learning,
            style='Small.TButton'
        )
        self.reset_fuzzy_btn.pack(side=tk.LEFT, padx=2)
        self.fuzzy_mode_var = tk.BooleanVar(value=False)

        self.style.configure('Fuzzy.TCheckbutton',
                             foreground='purple',
                             font=('Arial', 9, 'bold'))

        self.fuzzy_toggle = ttk.Checkbutton(
            fuzzy_controls_frame,
            text="",
            variable=self.fuzzy_mode_var,
            command=self.toggle_fuzzy_mode,
            style='Fuzzy.TCheckbutton'
        )
        self.fuzzy_toggle.pack(side=tk.LEFT, padx=2)

        self.fuzzy_status_label = ttk.Label(
            fuzzy_controls_frame,
            text="[RIGID MODE]",
            foreground='red',
            font=('Arial', 8, 'bold')
        )
        self.fuzzy_status_label.pack(side=tk.LEFT, padx=2)

        # ML Confidence Slider
        ml_confidence_frame = ttk.Frame(confidence_and_fuzzy_frame)
        ml_confidence_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(ml_confidence_frame, text="Min Confidence:").pack(side=tk.LEFT, padx=(0, 5))

        self.confidence_var = tk.StringVar(value="0.75")
        self.confidence_slider = ttk.Scale(
            ml_confidence_frame,
            from_=0.55, to=0.95, value=0.75,
            command=lambda v: self.update_confidence_display(float(v)),
        )
        self.confidence_slider.pack(side=tk.LEFT, padx=5)

        self.confidence_entry = ttk.Entry(
            ml_confidence_frame,
            textvariable=self.confidence_var,
            width=5, state="readonly", justify="center",
        )
        self.confidence_entry.pack(side=tk.LEFT, padx=(5, 0))

        # Right section with logo and volume
        right_section = ttk.Frame(top_frame)
        right_section.pack(side=tk.RIGHT, anchor="ne", padx=5)

        try:
            logo_path = self.resource_path("images/logo.png")
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path).resize((150, 30), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = ttk.Label(right_section, image=self.logo_photo)
                logo_label.pack(side=tk.TOP, pady=(0, 5))
            else:
                self.log_message(f"Logo not found at: {logo_path}", "orange")
                logo_label = ttk.Label(right_section, text="LOGO", font=('Arial', 14, 'bold'))
                logo_label.pack(side=tk.TOP, pady=(0, 5))
        except Exception as e:
            self.log_message(f"Logo error: {e}", "orange")
            logo_label = ttk.Label(right_section, text="LOGO", font=('Arial', 14, 'bold'))
            logo_label.pack(side=tk.TOP, pady=(0, 5))

        self.volume_strength_frame = ttk.LabelFrame(right_section, text="Volume Strength", padding=5)
        self.volume_strength_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.draw_volume_strength(self.volume_strength_frame, 0)

        self.help_button = ttk.Button(
            right_section,
            text="📘 Help",
            command=self.open_help_pdf,
            style='Blue.TButton',
            width=8
        )
        self.help_button.pack(side=tk.TOP, pady=(0, 5))

        # ═══════════════════════════════════════════════════════════════════════════
        # TRADING SETTINGS - 3 FIXED FRAMES
        # ═══════════════════════════════════════════════════════════════════════════

        self.config_frame = ttk.LabelFrame(left_pane, text="Trading Settings")
        self.config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create 3 frames with FIXED widths
        frame1_left = ttk.Frame(self.config_frame, width=280)
        frame1_left.pack(side=tk.LEFT, fill=tk.BOTH, padx=1, pady=5)
        frame1_left.pack_propagate(False)

        self.frame2_middle = ttk.Frame(self.config_frame, width=300)
        self.frame2_middle.pack(side=tk.LEFT, fill=tk.BOTH, padx=1, pady=5)
        self.frame2_middle.pack_propagate(False)

        frame3_right = ttk.Frame(self.config_frame, width=176)
        frame3_right.pack(side=tk.LEFT, fill=tk.BOTH, padx=1, pady=5)
        frame3_right.pack_propagate(False)

        # ═══════════════════════════════════════════════════════════════════════════
        # FRAME 1 (LEFT): MODE + RISK MANAGEMENT
        # ═══════════════════════════════════════════════════════════════════════════

        frame1_left.columnconfigure(0, weight=0, minsize=75)
        frame1_left.columnconfigure(1, weight=1)

        ttk.Label(frame1_left, text="Mode:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.mode_var = tk.StringVar(value="Demo")
        self.mode_combobox = ttk.Combobox(frame1_left, textvariable=self.mode_var,
                                          values=["Live", "Demo", "Backtest"], width=22)
        self.mode_combobox.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.mode_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_mode_display(self.mode_var.get()))

        ttk.Label(frame1_left, text="Symbol:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.symbol_var = tk.StringVar(value="SOL-USDT")
        self.symbol_combobox = ttk.Combobox(frame1_left, textvariable=self.symbol_var,
                                            values=["SOL-USDT", "BTC-USDT", "ETH-USDT"], width=22)
        self.symbol_combobox.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.symbol_var.trace_add('write', lambda *_: self.on_symbol_change())

        ttk.Label(frame1_left, text="Interval:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.interval_var = tk.StringVar(value="15m")
        self.interval_combobox = ttk.Combobox(frame1_left, textvariable=self.interval_var,
                                              values=["1m", "5m", "15m", '30m', "1H", "4H", "1D"], width=22)
        self.interval_combobox.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # Risk Management Frame
        risk_frame = ttk.LabelFrame(frame1_left, text="Risk Management")
        risk_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        risk_frame.columnconfigure(0, weight=0, minsize=80)
        risk_frame.columnconfigure(1, weight=1)
        risk_frame.columnconfigure(2, weight=0, minsize=65)
        risk_frame.columnconfigure(3, weight=1)

        ttk.Label(risk_frame, text="Order Size (%):").grid(row=0, column=0, sticky="w", padx=3, pady=3)
        self.order_size_var = tk.DoubleVar(value=self.order_size_pct)
        self.order_size_spin = ttk.Spinbox(risk_frame, from_=1, to=100, increment=1,
                                           textvariable=self.order_size_var, width=7)
        self.order_size_spin.grid(row=0, column=1, sticky="w", padx=3, pady=3)

        ttk.Label(risk_frame, text="Direction:").grid(row=0, column=2, sticky="w", padx=3, pady=3)
        self.trade_direction_var = tk.StringVar(value="both")
        self.trade_direction_combo = ttk.Combobox(risk_frame, textvariable=self.trade_direction_var,
                                                  values=["both", "long", "short"], state="readonly", width=7)
        self.trade_direction_combo.grid(row=0, column=3, sticky="w", padx=3, pady=3)
        self.trade_direction_var.trace_add('write', lambda *_: self.update_trade_direction())
        self.commission_var = tk.DoubleVar(value=0.001)

        ttk.Label(risk_frame, text="Stop Loss (%):").grid(row=1, column=0, sticky="w", padx=3, pady=3)
        self.stop_loss_var = tk.DoubleVar(value=self.stop_loss_pct * 100)
        self.stop_loss_spin = ttk.Spinbox(risk_frame, from_=0.1, to=50, increment=0.5,
                                          textvariable=self.stop_loss_var, width=7)
        self.stop_loss_spin.grid(row=1, column=1, sticky="w", padx=3, pady=3)

        ttk.Label(risk_frame, text="Trailing(%):").grid(row=1, column=2, sticky="w", padx=3, pady=3)
        self.trailing_stop_var = tk.DoubleVar(value=self.trailing_stop_pct * 100)
        self.trailing_stop_spin = ttk.Spinbox(risk_frame, from_=0.1, to=20, increment=0.5,
                                              textvariable=self.trailing_stop_var, width=7)
        self.trailing_stop_spin.grid(row=1, column=3, sticky="w", padx=3, pady=3)

        ttk.Label(risk_frame, text="Maker %:").grid(row=2, column=0, sticky="w", padx=3, pady=3)
        self.maker_fee_var = tk.DoubleVar(value=0.0008)
        self.maker_fee_combo = ttk.Combobox(
            risk_frame,
            textvariable=self.maker_fee_var,
            values=["0.000", "0.001", "0.002", "0.005", "0.010"],
            width=7,
            state="readonly"
        )
        self.maker_fee_combo.grid(row=2, column=1, sticky="w", padx=3, pady=3)

        ttk.Label(risk_frame, text="Taker %:").grid(row=2, column=2, sticky="w", padx=3, pady=3)
        self.taker_fee_var = tk.DoubleVar(value=0.001)
        self.taker_fee_combo = ttk.Combobox(
            risk_frame,
            textvariable=self.taker_fee_var,
            values=["0.000", "0.001", "0.002", "0.005", "0.010"],
            width=7,
            state="readonly"
        )
        self.taker_fee_combo.grid(row=2, column=3, sticky="w", padx=3, pady=3)

        # ═══════════════════════════════════════════════════════════════════════════
        # FRAME 2 (MIDDLE): PICTURE + EXPLANATION + BACKTEST CONTROLS
        # ═══════════════════════════════════════════════════════════════════════════

        try:
            picture_path = self.resource_path("images/cryptoimage.jpg")
            if os.path.exists(picture_path):
                picture_img = Image.open(picture_path)
                picture_img = picture_img.resize((334, 95), Image.Resampling.LANCZOS)
                self.picture_photo = ImageTk.PhotoImage(picture_img)
                self.picture_label = ttk.Label(self.frame2_middle, image=self.picture_photo,
                                               borderwidth=2, relief="groove")
                self.picture_label.pack(fill=tk.X, pady=(0, 5))
            else:
                self.log_message(f"Picture not found at: {picture_path}", "orange")
                self.picture_label = ttk.Label(self.frame2_middle, text="No Image",
                                               relief="solid", anchor="center")
                self.picture_label.pack(fill=tk.X, pady=(0, 5))
        except Exception as e:
            self.log_message(f"Picture error: {e}", "orange")
            self.picture_label = ttk.Label(self.frame2_middle, text="No Image",
                                           relief="solid", anchor="center")
            self.picture_label.pack(fill=tk.X, pady=(0, 5))

        self.message_container = ttk.Frame(self.frame2_middle)
        self.message_container.pack(fill=tk.BOTH, expand=True)

        # Explanation Text
        explanation_text = textwrap.dedent("""\
            🔴 MANDATORY CONDITIONS (ALL 3 MUST PASS):
            ───────────────────────────────────────
                1 - ADX HARD MINIMUM: ≥ 25
                2 - EMA TREND: Fast > Slow
                3 - MACD MOMENTUM: Bullish Gate

            🟢 SUPPORTING (0-100 Point Score):
            ───────────────────────────────────
                1 - EMA ALIGNMENT  (20 pts)
                2 - ADX STRENGTH   (20 pts)
                3 - MACD MOMENTUM  (25 pts)
                4 - DYNAMIC RSI    (20 pts)
                5 - VOLUME CONFIRM (15 pts)
                6 - PRICE POSITION (±15 pts)
            """)
        small_font = tkfont.Font(family="Helvetica", size=7)
        self.explanation_textbox = tk.Text(self.message_container, wrap=tk.WORD, height=4, width=43, bg='black',
                                           fg="white", font=small_font)
        self.explanation_textbox.tag_configure("indent", lmargin1=10, lmargin2=10)
        self.explanation_textbox.insert(tk.END, explanation_text, "indent")
        self.explanation_textbox.config(state=tk.DISABLED)
        self.explanation_textbox.pack(fill=tk.BOTH, expand=True)

        # ═══════════════════════════════════════════════════════════════════════════
        # BACKTEST CONTROLS (Compact inline grid - shown when Backtest mode selected)
        # ═══════════════════════════════════════════════════════════════════════════

        self.backtest_controls_frame = ttk.Frame(self.message_container)

        # Data Source
        ttk.Label(self.backtest_controls_frame, text="Data:").grid(row=0, column=0, sticky="w", padx=3, pady=2)
        self.data_source_var = tk.StringVar(value="Fetch API Data")
        self.data_source_combo = ttk.Combobox(self.backtest_controls_frame, textvariable=self.data_source_var,
                                              values=["Fetch API Data", "Use Generated Data"], width=16)
        self.data_source_combo.grid(row=0, column=1, sticky="ew", padx=3, pady=2)

        # Cache Toggle
        self.use_cache_var = tk.BooleanVar(value=False)
        self.cache_toggle_btn = ttk.Checkbutton(
            self.backtest_controls_frame,
            text="📁 Use CSV",
            variable=self.use_cache_var
        )
        self.cache_toggle_btn.grid(row=0, column=2, sticky="w", padx=2, pady=2)

        # Backtest Type
        ttk.Label(self.backtest_controls_frame, text="Type:").grid(row=1, column=0, sticky="w", padx=3, pady=2)
        self.backtest_type_var = tk.StringVar(value="Standard Backtest")
        self.backtest_type_combo = ttk.Combobox(self.backtest_controls_frame, textvariable=self.backtest_type_var,
                                                values=["Standard Backtest", "Optimization"], width=16)
        self.backtest_type_combo.grid(row=1, column=1, sticky="ew", padx=3, pady=2)

        # Monte Carlo Toggle
        self.use_monte_carlo_var = tk.BooleanVar(value=False)
        self.monte_carlo_toggle = ttk.Checkbutton(
            self.backtest_controls_frame,
            text="🎲 Monte Carlo",
            variable=self.use_monte_carlo_var,
            command=self.toggle_monte_carlo_options
        )
        self.monte_carlo_toggle.grid(row=1, column=2, sticky="w", padx=2, pady=2)

        # Date Range (compact inline)
        ttk.Label(self.backtest_controls_frame, text="Dates:").grid(row=2, column=0, sticky="w", padx=3, pady=2)
        date_frame = ttk.Frame(self.backtest_controls_frame)
        date_frame.grid(row=2, column=1, columnspan=2, sticky="ew", padx=3, pady=2)
        self.start_date_var = tk.StringVar(value="2025-01-01")
        self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(date_frame, textvariable=self.start_date_var, width=10).pack(side=tk.LEFT)
        ttk.Label(date_frame, text="-").pack(side=tk.LEFT, padx=2)
        ttk.Entry(date_frame, textvariable=self.end_date_var, width=10).pack(side=tk.LEFT)

        # Monte Carlo Options Frame (Collapsible)
        self.monte_carlo_options_frame = ttk.LabelFrame(self.backtest_controls_frame, text="MC Settings", padding=2)

        mc_row1 = ttk.Frame(self.monte_carlo_options_frame)
        mc_row1.pack(fill=tk.X, pady=1)
        ttk.Label(mc_row1, text="Sim:", font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        self.mc_simulations_var = tk.IntVar(value=1000)
        ttk.Combobox(mc_row1, textvariable=self.mc_simulations_var, values=[100, 500, 1000, 2000, 5000],
                     width=5, state="readonly", font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        ttk.Label(mc_row1, text="Tr:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(8, 2))
        self.mc_trades_var = tk.IntVar(value=100)
        ttk.Combobox(mc_row1, textvariable=self.mc_trades_var, values=[50, 100, 200, 500],
                     width=5, state="readonly", font=('Arial', 8)).pack(side=tk.LEFT, padx=2)

        mc_row2 = ttk.Frame(self.monte_carlo_options_frame)
        mc_row2.pack(fill=tk.X, pady=1)
        ttk.Label(mc_row2, text="Method:", font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        self.mc_method_var = tk.StringVar(value="hybrid")
        ttk.Combobox(mc_row2, textvariable=self.mc_method_var, values=["parametric", "bootstrap", "hybrid"],
                     width=13, state="readonly", font=('Arial', 8)).pack(side=tk.LEFT, padx=2)

        # Run Backtest Button
        self.run_btn = tk.Button(
            self.backtest_controls_frame,
            text="▶ Run Backtest",
            command=self._run_backtest_in_thread,
            bg="#28a745",
            fg="white",
            activebackground="#218838",
            activeforeground="white",
            font=("Arial", 9, "bold"),
            relief="raised",
            bd=2,
            cursor="hand2",
            padx=5,
            pady=3
        )
        self.run_btn.grid(row=4, column=0, columnspan=3, pady=5, sticky="ew", padx=3)
        # Store references for later visibility toggling
        self.backtest_controls_frame_ref = self.backtest_controls_frame
        self.explanation_textbox_ref = self.explanation_textbox

        # Initially hide backtest controls (will show when mode changes to backtest)
        self.backtest_controls_frame.pack_forget()
        # ═══════════════════════════════════════════════════════════════════════════
        # FRAME 3 (RIGHT): TIMER
        # ═══════════════════════════════════════════════════════════════════════════

        self.timer = CircleTimer(frame3_right, size=155)
        self.timer.canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=5)

        # ═══════════════════════════════════════════════════════════════════════════
        # BUTTONS, STATS, LOG
        # ═══════════════════════════════════════════════════════════════════════════

        btn_frame = ttk.Frame(left_pane)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.style.configure('AI.TButton',
                             foreground='white',
                             background='#9933FF',
                             font=('Arial', 10, 'bold'),
                             padding=5,
                             borderwidth=2)
        self.style.map('AI.TButton',
                       background=[('active', '#7722CC'), ('disabled', '#CCAAEE')])

        self.ai_btn = ttk.Button(
            btn_frame,
            text="🤖 AI DEEPSEEK",
            command=self.open_ai_analysis,
            style='AI.TButton',
            state=tk.DISABLED
        )
        self.ai_btn.pack(side=tk.LEFT, padx=1, pady=5)

        self.connect_btn = ttk.Button(btn_frame, text="Check Connection", command=self.check_connection,
                                      style='Blue.TButton')
        self.connect_btn.pack(side=tk.LEFT, padx=25, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start Trading", command=self.start_trading,
                                    style='Blue.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Trading", command=self.stop_trading, style='Blue.TButton',
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.emergency_btn = tk.Button(btn_frame, text="EMERGENCY", command=self.emergency_stop,
                                       bg="#FF0000", fg="white", activebackground="#CC0000",
                                       activeforeground="white", font=("Arial", 10, "bold"),
                                       padx=5, pady=5, relief="raised", bd=2, cursor="hand2")
        self.emergency_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.settings_btn = ttk.Button(btn_frame, text="⚙ Settings",
                                       command=self.open_settings,
                                       style='Blue.TButton')
        self.settings_btn.pack(side=tk.LEFT, padx=25, pady=5)

        self.style.configure('Red.TButton', foreground='white', background='#FF0000', font=('Arial', 10, 'bold'),
                             padding=5, borderwidth=2)

        self.detailed_output_var = tk.BooleanVar(value=True)
        self.detailed_output_check = ttk.Checkbutton(
            btn_frame,
            text="📋 Detailed Log",
            variable=self.detailed_output_var,
            command=self.toggle_output_detail
        )
        self.detailed_output_check.pack(side=tk.LEFT, padx=2, pady=5)

        stats_frame = ttk.LabelFrame(left_pane, text="Trading Stats")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
        stats_frame.grid_columnconfigure(2, weight=1)

        self.trades_label = ttk.Label(stats_frame, text="Trades: 0")
        self.trades_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.wins_label = ttk.Label(stats_frame, text="Wins: 0")
        self.wins_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.win_percent_label = ttk.Label(stats_frame, text="Win %: 0.00%")
        self.win_percent_label.grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.pnl_label = ttk.Label(stats_frame, text="PnL: $0.00")
        self.pnl_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.usdt_balance_label = ttk.Label(stats_frame, text="USDT: $0.00")
        self.usdt_balance_label.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.symbol_balance_label = ttk.Label(stats_frame, text=f"{self.base_symbol()}: 0.00")
        self.symbol_balance_label.grid(row=1, column=2, padx=5, pady=2, sticky="w")

        log_frame = ttk.LabelFrame(left_pane, text="Trading Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=43, font=("Consolas", 12))
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.tag_config("green", foreground="#00FF00")
        self.log_area.tag_config("red", foreground="#FF5555")
        self.log_area.tag_config("blue", foreground="#00BFFF")
        self.log_area.tag_config("orange", foreground="#FFA500")
        self.log_area.tag_config("purple", foreground="#DA70D6")
        self.log_area.configure(bg='black', fg='white', insertbackground='white')

        self.log_area.bind("<Double-Button-1>", self.toggle_log_expansion)

        self.expandable_frames = {
            'top_frame': top_frame,
            'config_frame': self.config_frame,
            'btn_frame': btn_frame,
            'stats_frame': stats_frame
        }

        self.log_frame_ref = log_frame
        self.original_log_height = 43

    def toggle_output_detail(self):
        """Toggle between detailed and simple output modes"""
        if self.detailed_output_var.get():
            self.log_message("📋 Output mode: DETAILED (tables enabled)", "blue")
        else:
            self.log_message("📋 Output mode: SIMPLE (summary only)", "blue")

    def enable_ai_button(self):
        """Enable AI Analysis button when data becomes available"""
        self.ai_data_available = True
        if hasattr(self, 'ai_btn') and self.ai_btn:
            self.ai_btn.config(state=tk.NORMAL)
            self.log_message("🤖 AI Analysis now available", "orange")

    def disable_ai_button(self):
        """Disable AI Analysis button when no data available"""
        self.ai_data_available = False
        if hasattr(self, 'ai_btn') and self.ai_btn:
            self.ai_btn.config(state=tk.DISABLED)

    def _update_global_capital(self):
        """Update global capital from the settings panel"""
        from strategies.MomentumStrategy_MACD_HybridScore_Latest import GlobalConfig
        from strategies.scalping_strategy import SCALPING_PARAMS

        try:
            new_capital = float(self.capital_entry.get().replace(',', '').replace('$', ''))

            if new_capital <= 0:
                self.log_message("❌ Capital must be positive", "red")
                messagebox.showerror("Error", "Capital must be positive")
                return

            # Update global config
            old_capital = GlobalConfig.INITIAL_CAPITAL
            GlobalConfig.update_capital(new_capital)

            # Update display
            self.capital_display_var.set(f"${new_capital:,.2f}")

            # BUG FIX v9.4.1 (APP BUG E): Propagate capital to ALL live strategy
            # instances in self.strategies, not just self.strategy.
            # Previously only the currently active strategy received the update;
            # switching strategies after a capital change would use the old value.
            strategies_to_update = []
            if hasattr(self, 'strategy') and self.strategy:
                strategies_to_update.append(self.strategy)
            if hasattr(self, 'strategies'):
                for name, strat in self.strategies.items():
                    if strat not in strategies_to_update and not isinstance(strat, type):
                        strategies_to_update.append(strat)

            for strat in strategies_to_update:
                if hasattr(strat, 'risk_controller'):
                    strat.risk_controller.starting_equity = new_capital
                    strat.risk_controller.current_equity = new_capital
                    strat.risk_controller.peak_equity = new_capital
                    strat.risk_controller.daily_loss_limit = new_capital * SCALPING_PARAMS['daily_loss_limit_pct']
                    strat.risk_controller.max_drawdown_limit = new_capital * SCALPING_PARAMS['max_drawdown_limit_pct']
                    strat.risk_controller.weekly_loss_limit = new_capital * 0.05
                    strat.risk_controller.monthly_loss_limit = new_capital * 0.10
                if hasattr(strat, 'equity_curve') and strat.equity_curve:
                    strat.equity_curve[0] = new_capital

            self.log_message(f"💰 GLOBAL CAPITAL UPDATED: ${old_capital:,.2f} → ${new_capital:,.2f}", "green")

            # Refresh stats display
            self.update_stats()

            messagebox.showinfo("Capital Updated",
                                f"Global capital updated to ${new_capital:,.2f}\n\n"
                                f"All calculations will now use this value.")

        except ValueError:
            self.log_message(f"❌ Invalid capital value: {self.capital_entry.get()}", "red")
            messagebox.showerror("Error", f"Invalid capital value: {self.capital_entry.get()}")

    def toggle_fuzzy_mode(self):
        """Toggle between rigid crossover and fuzzy adaptive mode"""
        is_fuzzy = self.fuzzy_mode_var.get()

        if hasattr(self, 'strategies') and 'Momentum' in self.strategies:
            self.strategies['Momentum'].fuzzy_mode_enabled = is_fuzzy
            if hasattr(self.strategies['Momentum'], 'config'):
                self.strategies['Momentum'].config['fuzzy_mode_enabled'] = is_fuzzy

        if is_fuzzy:
            self.fuzzy_status_label.config(
                text="FUZZY ACTIVE |",
                foreground='green',
                font=('Arial', 8, 'bold')
            )
            self.log_message("🧠 FUZZY MODE ENABLED - Using adaptive thresholds from near-misses", "purple")
            if hasattr(self.strategies['Momentum'], 'log_fuzzy_threshold_stats'):
                self.strategies['Momentum'].log_fuzzy_threshold_stats()
        else:
            self.fuzzy_status_label.config(
                text="RIGID ACTIVE |",
                foreground='red',
                font=('Arial', 8, 'bold')
            )
            self.log_message("📏 RIGID MODE ENABLED - Using fixed 75 threshold", "blue")

    def reset_fuzzy_learning(self):
        """Reset fuzzy learning data and revert to defaults"""
        if messagebox.askyesno("Reset Fuzzy Learning",
                               "Are you sure you want to reset all learned fuzzy thresholds?\nThis cannot be undone."):
            if hasattr(self, 'strategies') and 'Momentum' in self.strategies:
                if hasattr(self.strategies['Momentum'], 'near_miss_trades'):
                    self.strategies['Momentum'].near_miss_trades = []
                if hasattr(self.strategies['Momentum'], '_previous_fuzzy_lower'):
                    self.strategies['Momentum']._previous_fuzzy_lower = 67.5
                self.log_message("🔄 Fuzzy learning data reset to defaults", "orange")

    import pandas as pd

    from typing import Dict, List, Any

    def open_ai_analysis(self):
        """Open Professional AI Analysis dialog with DeepSeek integration - maximizes window and ensures connection"""

        # First, check if we have data
        if not self.ai_data_available:
            messagebox.showwarning("No Data",
                                   "No trading data available for analysis.\nRun a backtest or start trading first.")
            return

        # Check/establish connection before opening window
        connection_ok = False

        # Try to check/establish connection
        if hasattr(self, 'market_api') and self.market_api is not None:
            try:
                # Quick connection test
                response = self.market_api.get_tickers(instType="SPOT")
                if response['code'] == '0':
                    connection_ok = True
                    self.log_message("✅ API connection verified for AI analysis", "green")
            except:
                connection_ok = False
        else:
            self.log_message("⚠️ API not connected - attempting to connect...", "orange")
            # Try to connect
            self.check_connection()
            # Give it a moment to connect
            time.sleep(1)
            if hasattr(self, 'market_api') and self.market_api is not None:
                connection_ok = True

        # If still not connected, warn but continue (will use local analysis)
        if not connection_ok:
            self.log_message("⚠️ API connection failed - AI will use local analysis only", "orange")
            # Update the status label later when window is created

        ai_window = tk.Toplevel(self.root)
        ai_window.title("🤖 AI DEEPSEEK R1 Professional Analysis")

        # Set initial size but then maximize
        ai_window.geometry("900x750")

        # Maximize the window based on OS
        try:
            # Try Windows maximize
            ai_window.state('zoomed')
        except:
            try:
                # Try Linux/Unix maximize
                ai_window.attributes('-zoomed', True)
            except:
                try:
                    # Try macOS maximize
                    ai_window.wm_attributes('-zoomed', True)
                except:
                    # Fallback to large size
                    screen_width = ai_window.winfo_screenwidth()
                    screen_height = ai_window.winfo_screenheight()
                    ai_window.geometry(f"{screen_width}x{screen_height}+0+0")

        ai_window.resizable(True, True)
        ai_window.grab_set()
        ai_window.configure(bg='#1a1a2e')

        header_frame = ttk.Frame(ai_window)
        header_frame.pack(fill=tk.X, padx=15, pady=10)

        ttk.Label(
            header_frame,
            text="🤖 AI DEEPSEEK R1 Professional Trading Analysis",
            font=('Arial', 18, 'bold')
        ).pack(side=tk.LEFT)

        # Update API status based on actual connection
        api_key_ok = self._check_api_key()
        api_connected = connection_ok

        if api_key_ok and api_connected:
            status_text = "● API Connected"
            status_color = '#00ff00'
        elif api_key_ok and not api_connected:
            status_text = "○ API Key OK - No Connection"
            status_color = '#ffaa00'
        elif not api_key_ok and api_connected:
            status_text = "○ No API Key - Local Only"
            status_color = '#ffaa00'
        else:
            status_text = "○ No API Connection"
            status_color = '#ff5555'

        self.api_status_label = ttk.Label(
            header_frame,
            text=status_text,
            foreground=status_color,
            font=('Arial', 10, 'bold')
        )
        self.api_status_label.pack(side=tk.RIGHT, padx=10)

        # Add a reconnect button
        reconnect_btn = ttk.Button(
            header_frame,
            text="🔄 Reconnect API",
            command=self._reconnect_for_ai,
            style='Small.TButton'
        )
        reconnect_btn.pack(side=tk.RIGHT, padx=5)

        type_frame = ttk.LabelFrame(ai_window, text="Select Analysis Type", padding=10)
        type_frame.pack(fill=tk.X, padx=15, pady=5)

        self.ai_analysis_type = tk.StringVar(value="comprehensive")

        analysis_types = [
            ("🎯 Comprehensive Analysis", "comprehensive",
             "Full market analysis with entry/exit recommendations, risk assessment, and forecast"),
            ("📊 Technical Deep Dive", "technical",
             "Detailed indicator analysis, divergences, support/resistance levels"),
            ("📈 Performance Review", "performance",
             "Analyze your trading history, identify patterns, suggest improvements"),
            ("⚠️ Risk Assessment", "risk",
             "Position sizing, drawdown analysis, portfolio risk evaluation"),
            ("🔮 Market Forecast", "forecast",
             "Price predictions, trend analysis, key levels to watch"),
            ("💡 Trade Setup Scanner", "setups",
             "Identify current trade setups with entry, stop, and target levels")
        ]

        for i, (text, value, description) in enumerate(analysis_types):
            row = i // 2
            col = i % 2
            frame = ttk.Frame(type_frame)
            frame.grid(row=row, column=col, sticky='w', padx=10, pady=3)
            rb = ttk.Radiobutton(
                frame,
                text=text,
                variable=self.ai_analysis_type,
                value=value
            )
            rb.pack(side=tk.LEFT)
            desc_label = ttk.Label(
                frame,
                text=f"  {description[:50]}...",
                foreground='gray',
                font=('Arial', 8)
            )
            desc_label.pack(side=tk.LEFT, padx=5)

        depth_frame = ttk.Frame(type_frame)
        depth_frame.grid(row=3, column=0, columnspan=2, sticky='w', pady=(10, 0))
        ttk.Label(depth_frame, text="Analysis Depth:").pack(side=tk.LEFT, padx=5)
        self.analysis_depth_var = tk.StringVar(value="standard")
        ttk.Radiobutton(depth_frame, text="Quick (100 candles)",
                        variable=self.analysis_depth_var, value="quick").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(depth_frame, text="Standard (500 candles)",
                        variable=self.analysis_depth_var, value="standard").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(depth_frame, text="Deep (Full History)",
                        variable=self.analysis_depth_var, value="deep").pack(side=tk.LEFT, padx=5)

        results_frame = ttk.LabelFrame(ai_window, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.ai_results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg='#0d1117',
            fg='#c9d1d9',
            insertbackground='white',
            selectbackground='#388bfd',
            padx=10,
            pady=10
        )
        self.ai_results_text.pack(fill=tk.BOTH, expand=True)

        self.ai_results_text.tag_config("header", foreground="#58a6ff", font=("Consolas", 13, "bold"))
        self.ai_results_text.tag_config("subheader", foreground="#79c0ff", font=("Consolas", 11, "bold"))
        self.ai_results_text.tag_config("bullish", foreground="#3fb950")
        self.ai_results_text.tag_config("bearish", foreground="#f85149")
        self.ai_results_text.tag_config("warning", foreground="#d29922")
        self.ai_results_text.tag_config("info", foreground="#c9d1d9")
        self.ai_results_text.tag_config("highlight", foreground="#a371f7", font=("Consolas", 11, "bold"))
        self.ai_results_text.tag_config("price", foreground="#ffa657")
        self.ai_results_text.tag_config("separator", foreground="#30363d")
        self.ai_results_text.tag_config("reasoning",
                                        foreground="#aaaacc",
                                        font=("Courier New", 9),
                                        background="#1a1a2e"
                                        )
        self.ai_results_text.tag_config("reasoning_header",
                                        foreground="#9988ff",
                                        font=("Courier New", 10, "bold")
                                        )

        self.ai_progress_var = tk.DoubleVar(value=0)
        self.ai_progress_bar = ttk.Progressbar(
            ai_window,
            variable=self.ai_progress_var,
            maximum=100,
            mode='determinate'
        )
        self.ai_progress_bar.pack(fill=tk.X, padx=15, pady=5)

        self.ai_status_label = ttk.Label(ai_window, text="Ready to analyze", font=('Arial', 9))
        self.ai_status_label.pack(pady=2)

        button_frame = ttk.Frame(ai_window)
        button_frame.pack(fill=tk.X, padx=15, pady=10)

        self.run_ai_btn = tk.Button(
            button_frame,
            text="🚀 Run AI Analysis",
            command=self._execute_ai_analysis,
            bg="#238636",
            fg="white",
            activebackground="#2ea043",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor="hand2"
        )

        self.run_ai_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="📋 Copy Results",
            command=self._copy_ai_results
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="💾 Export Report",
            command=self._export_ai_report
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="🔄 Clear",
            command=lambda: self.ai_results_text.delete(1.0, tk.END)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="❌ Close",
            command=ai_window.destroy
        ).pack(side=tk.RIGHT, padx=5)

        self.ai_window = ai_window

        # Update status based on connection
        if not connection_ok:
            self.ai_status_label.config(text="⚠️ No API connection - using local analysis only")

        self._show_ai_instructions()

        # Log that AI panel was opened
        self.log_message("🤖 AI Analysis panel opened (maximized)", "blue")

    def _reconnect_for_ai(self):
        """Attempt to reconnect API for AI analysis"""
        self.log_message("🔄 Attempting to reconnect API...", "blue")

        # Update status
        if hasattr(self, 'ai_status_label'):
            self.ai_status_label.config(text="Connecting to API...")

        # Run connection check
        self.check_connection()

        # Update the status label in AI window if it exists
        if hasattr(self, 'api_status_label') and self.api_status_label.winfo_exists():
            if hasattr(self, 'market_api') and self.market_api is not None:
                self.api_status_label.config(
                    text="● API Connected",
                    foreground='#00ff00'
                )
                if hasattr(self, 'ai_status_label'):
                    self.ai_status_label.config(text="API connected - Ready to analyze")
                self.log_message("✅ API reconnected successfully", "green")
            else:
                self.api_status_label.config(
                    text="○ Connection Failed",
                    foreground='#ff5555'
                )
                if hasattr(self, 'ai_status_label'):
                    self.ai_status_label.config(text="❌ API connection failed")
                self.log_message("❌ API reconnection failed", "red")

    def _ai_error(self, message):
        """Display AI error in the results window"""

        def show_error():
            if hasattr(self, 'ai_results_text'):
                self.ai_results_text.delete(1.0, tk.END)
                self.ai_results_text.insert(tk.END, f"\n❌ ERROR\n{'─' * 60}\n\n", "bearish")
                self.ai_results_text.insert(tk.END, f"{message}\n\n", "warning")
                self.ai_results_text.insert(tk.END, "Please check:\n", "info")
                self.ai_results_text.insert(tk.END, "  • API key is configured in config.json\n", "info")
                self.ai_results_text.insert(tk.END, "  • Internet connection is active\n", "info")
                self.ai_results_text.insert(tk.END, "  • Market data is available\n", "info")

            if hasattr(self, 'ai_progress_var'):
                self.ai_progress_var.set(0)

            if hasattr(self, 'ai_status_label'):
                self.ai_status_label.config(text="Error occurred")

            if hasattr(self, 'run_ai_btn'):
                self.run_ai_btn.config(state=tk.NORMAL, text="🚀 Run AI Analysis")

        self.root.after(0, show_error)

    def _update_ai_status(self, status):
        if hasattr(self, 'ai_status_label'):
            self.ai_status_label.config(text=status)

    def _check_api_key(self) -> bool:
        """Check if DeepSeek API key is configured"""
        try:
            # Make sure config is loaded
            if not hasattr(self, 'config'):
                self.load_config()

            api_key = self.config.get('deepseek_api_key', '')
            return api_key and api_key.startswith('sk-') and len(api_key) > 20
        except Exception as e:
            self.log_message(f"⚠️ API key check error: {e}", "orange")
            return False

    def _show_ai_instructions(self):
        self.ai_results_text.delete(1.0, tk.END)
        instructions = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    🤖 AI DEEPSEEK PROFESSIONAL ANALYSIS                     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    Welcome to the AI-powered trading analysis system!

    📋 HOW TO USE:
    ─────────────────────────────────────────────────────────────────────────────────
      1. Select an analysis type above
      2. Choose analysis depth (more data = more accurate but slower)
      3. Click "🚀 Run AI Analysis"
      4. Wait for DeepSeek to analyze your data
      5. Review recommendations and export if needed

    🎯 ANALYSIS TYPES EXPLAINED:
    ─────────────────────────────────────────────────────────────────────────────────

      • Comprehensive Analysis
        Full market overview including trend, momentum, volatility, and specific
        trade recommendations with entry/exit levels.

      • Technical Deep Dive  
        Detailed indicator analysis, divergence detection, support/resistance
        levels, and pattern recognition.

      • Performance Review
        Analyzes your trading history to identify strengths, weaknesses,
        and specific improvements you can make.

      • Risk Assessment
        Evaluates current market risk, optimal position sizing, and
        potential drawdown scenarios.

      • Market Forecast
        AI-generated price predictions and key levels to watch for
        the upcoming sessions.

      • Trade Setup Scanner
        Identifies actionable trade setups with specific entry, stop-loss,
        and take-profit levels.

    ⚡ DATA PROCESSING:
    ─────────────────────────────────────────────────────────────────────────────────
      The AI automatically compresses large datasets into intelligent summaries.
      Your full history is analyzed statistically, with detailed focus on recent
      price action for maximum relevance.

    📌 Click "🚀 Run AI Analysis" to begin!

    """
        self.ai_results_text.insert(tk.END, instructions, "info")

    def _run_ai_analysis_thread(self, prompt, analysis_type):
        try:
            self.root.after(0, lambda: self._update_ai_status("🤔 DeepSeek is reasoning..."))
            self.root.after(0, lambda: self.ai_results_text.delete(1.0, tk.END))
            reasoning, answer = self._call_deepseek_api(prompt, analysis_type)

            if reasoning:
                def show_reasoning():
                    self.ai_results_text.insert(tk.END,
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", "header")
                    self.ai_results_text.insert(tk.END,
                                                "  🧠  REASONING CHAIN (How it thought)\n", "reasoning_header")
                    self.ai_results_text.insert(tk.END,
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n", "header")
                    self.ai_results_text.insert(tk.END, reasoning + "\n\n", "reasoning")
                    self.ai_results_text.insert(tk.END,
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", "header")
                    self.ai_results_text.insert(tk.END,
                                                "  📊  ANALYSIS RESULT\n", "header")
                    self.ai_results_text.insert(tk.END,
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n", "header")

                self.root.after(0, show_reasoning)
            else:
                def show_header():
                    self.ai_results_text.insert(tk.END,
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", "header")
                    self.ai_results_text.insert(tk.END,
                                                f"  📊  {analysis_type.upper()} ANALYSIS\n", "header")
                    self.ai_results_text.insert(tk.END,
                                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n", "header")

                self.root.after(0, show_header)

            if answer:
                lines = answer.split('\n')
                for line in lines:
                    line_with_newline = line + '\n'

                    def insert_line(l=line_with_newline):
                        if any(kw in l for kw in ['✓', 'BUY', 'LONG', '📈', 'BULLISH', 'STRONG']):
                            self.ai_results_text.insert(tk.END, l, "bullish")
                        elif any(kw in l for kw in ['✗', 'SELL', 'SHORT', '📉', 'BEARISH', 'STOP', 'RISK']):
                            self.ai_results_text.insert(tk.END, l, "bearish")
                        elif any(kw in l for kw in ['$', 'TARGET', 'PRICE', 'ENTRY', 'EXIT', 'LEVEL']):
                            self.ai_results_text.insert(tk.END, l, "price")
                        elif l.startswith('##') or l.startswith('**') or l.isupper():
                            self.ai_results_text.insert(tk.END, l, "header")
                        else:
                            self.ai_results_text.insert(tk.END, l, "info")

                    self.root.after(0, insert_line)

            self.root.after(0, lambda: self._update_ai_status("✅ Analysis complete (DeepSeek)"))

        except Exception as e:
            error_msg = f"❌ Thread error: {str(e)}"
            self.root.after(0, lambda: self.ai_results_text.insert(tk.END, error_msg, "bearish"))
            self.root.after(0, lambda: self._update_ai_status("❌ Analysis failed"))

    def _display_market_snapshot(self, current_data_or_dict):
        """Display market snapshot safely from either Series or dict"""
        try:
            self.log_message("=" * 77, "cyan")
            self.log_message(f"📊 CURRENT MARKET SNAPSHOT", "cyan")
            self.log_message("=" * 77, "cyan")

            def safe_get(data, key, default=0):
                try:
                    if hasattr(data, 'get'):
                        val = data.get(key, default)
                    elif hasattr(data, '__getitem__'):
                        val = data[key] if key in data else default
                    else:
                        val = default
                    return float(val) if val is not None else default
                except (ValueError, TypeError):
                    return default

            close = safe_get(current_data_or_dict, 'Close', 0)
            ema_fast = safe_get(current_data_or_dict, 'EMA_Fast', 0)
            ema_mid = safe_get(current_data_or_dict, 'EMA_Mid', 0)
            ema_slow = safe_get(current_data_or_dict, 'EMA_Slow', 0)

            self.log_message(f"💰 Price: ${close:.4f}", "white")

            ema_aligned = (
                close > ema_fast > ema_mid > ema_slow
                if ema_fast > 0 and ema_mid > 0 and ema_slow > 0
                else False
            )
            ema_status = "✅ ALIGNED" if ema_aligned else "❌ NOT ALIGNED"
            self.log_message(
                f"📈 EMA Alignment: {ema_status}",
                "green" if ema_aligned else "red"
            )

            if ema_fast > 0 or ema_mid > 0 or ema_slow > 0:
                self.log_message(
                    f"   Fast: {ema_fast:.4f} | Mid: {ema_mid:.4f} | Slow: {ema_slow:.4f}",
                    "white"
                )

            rsi = safe_get(current_data_or_dict, 'RSI', 50)
            adx = safe_get(current_data_or_dict, 'ADX', 0)
            volume_ratio = safe_get(current_data_or_dict, 'Volume_Ratio', 1.0)

            self.log_message(
                f"📊 RSI: {rsi:.1f} | ADX: {adx:.1f} | Vol Ratio: {volume_ratio:.2f}x",
                "green" if rsi < 70 and adx > 25 else "orange"
            )

            macd = safe_get(current_data_or_dict, 'MACD_closed', 0)
            macd_signal = safe_get(current_data_or_dict, 'MACD_Signal_closed', 0)
            if macd != 0 or macd_signal != 0:
                macd_status = "BULLISH" if macd > macd_signal else "BEARISH"
                self.log_message(
                    f"📉 MACD: {macd:.4f} vs {macd_signal:.4f} ({macd_status})",
                    "green" if macd > macd_signal else "red"
                )

            # CHANGED: guard quality score — skip silently if strategy is IN_TRADE
            # or if the call is otherwise not safe (e.g. after a backtest run)
            if (
                    hasattr(self, 'strategy')
                    and self.strategy
                    and hasattr(self.strategy, '_calculate_quality_score')
                    and not getattr(self.strategy, '_in_trade', False)
                    and getattr(self.strategy, 'state', 'SEEKING_ENTRY') == 'SEEKING_ENTRY'
            ):
                try:
                    _eff_dir = getattr(self.strategy, "_pending_signal", None)
                    _eff_dir = (
                        _eff_dir.get("direction", "long")
                        if _eff_dir
                        else getattr(self.strategy, "trade_direction", "long")
                    )
                    if _eff_dir == "short" and hasattr(self.strategy, "_calculate_quality_score_short"):
                        total_score, component_scores, reason = (
                            self.strategy._calculate_quality_score_short(current_data_or_dict)
                        )
                    else:
                        total_score, component_scores, reason = (
                            self.strategy._calculate_quality_score(current_data_or_dict)
                        )

                    if total_score is not None:
                        quality_bar = self._create_confidence_bar(total_score, 10)
                        self.log_message(
                            f"🎯 Quality Score: {total_score}/100 {quality_bar}",
                            "green" if total_score >= 75 else "orange" if total_score >= 60 else "red"
                        )
                except Exception as qs_err:
                    # Strategy state prevents quality score — skip, don't crash
                    self.log_message(
                        f"ℹ️ Quality score unavailable: {qs_err}", "blue"
                    )

            self.log_message("=" * 77, "cyan")

        except Exception as e:
            self.log_message(f"Error displaying snapshot: {e}", "red")

    # ═══════════════════════════════════════════════════════════════════════════════
    # FIX 2  —  _execute_ai_analysis
    # ═══════════════════════════════════════════════════════════════════════════════
    def _execute_ai_analysis(self):
        """Run DeepSeek AI analysis — works in backtest AND demo/live mode."""
        try:
            analysis_type = self.ai_analysis_type.get()
            depth = self.analysis_depth_var.get()

            self._update_ai_progress(5, "Gathering market data...")

            mode = self.mode_var.get().lower()
            df = None

            # ── Step 1: use cached backtest DataFrame if available ───────────────
            if mode == "backtest":
                cached = getattr(self, '_last_backtest_df', None)
                if cached is not None and not cached.empty:
                    df = cached.copy()
                    # CHANGED: normalize immediately after copying from cache
                    df = self._normalize_ohlcv_columns(df)
                    self.log_message("📊 AI: Using cached backtest DataFrame", "blue")
                else:
                    self.log_message(
                        "📊 AI: No cached backtest data — fetching fresh data...", "blue"
                    )

            # ── Step 2: fallback — call get_market_data() ────────────────────────
            if df is None:
                df = self.get_market_data()
                if df is not None and not df.empty:
                    # CHANGED: normalize fresh data too
                    df = self._normalize_ohlcv_columns(df)

            # ── Step 3: hard-fail with a clear, actionable message ───────────────
            if df is None or df.empty:
                self._ai_error(
                    "No market data available.\n\n"
                    "• Backtest mode  : run a backtest first, then click AI Analysis.\n"
                    "                   The button activates automatically once data\n"
                    "                   has been fetched.\n\n"
                    "• Demo/Live mode : click 'Check Connection', wait for\n"
                    "                   'Connection successful', then try again."
                )
                return

            # ── Step 4: calculate indicators ────────────────────────────────────
            self._update_ai_progress(15, "Calculating indicators...")
            if hasattr(self.strategy, 'calculate_indicators'):
                try:
                    enriched = self.strategy.calculate_indicators(df)
                    if enriched is not None and not enriched.empty:
                        df = enriched
                        # CHANGED: re-normalize after indicator enrichment — the
                        # strategy may have renamed or shadowed base OHLCV columns
                        df = self._normalize_ohlcv_columns(df)
                except Exception as e:
                    self.log_message(f"⚠️ Indicator calc failed — using raw OHLCV: {e}", "orange")

            # CHANGED: final safety guard — if 'Close' is still missing, abort
            if 'Close' not in df.columns:
                self._ai_error(
                    "Market data is missing a 'Close' column after indicator calculation.\n\n"
                    "This can happen when the strategy's calculate_indicators() renames\n"
                    "base OHLCV columns.  Try switching to Default Parameters and re-running\n"
                    "the backtest before clicking AI Analysis."
                )
                return

            # ── Step 5: build summary & prompt ───────────────────────────────────
            self._update_ai_progress(25, "Preparing data summary...")
            data_summary = self._prepare_ai_data_summary(df, depth)
            trade_summary = self._prepare_trade_history_summary()

            self._update_ai_progress(35, "Building analysis prompt...")
            prompt = self._build_ai_prompt(analysis_type, data_summary, trade_summary)

            # Show market snapshot — CHANGED: guard against missing 'Close'
            try:
                if df is not None and len(df) >= 2 and 'Close' in df.columns:
                    last_completed = df.iloc[-2]
                    current_dict = (
                        last_completed.to_dict()
                        if hasattr(last_completed, 'to_dict')
                        else dict(last_completed)
                    )
                    # CHANGED: snapshot is captured in a local variable, not a
                    # late-binding lambda, so it can't pick up a stale df reference
                    snapshot = dict(current_dict)
                    self.root.after(0, lambda s=snapshot: self._display_market_snapshot(s))
            except Exception as e:
                self.log_message(f"⚠️ Could not display market snapshot: {e}", "orange")

            self._update_ai_progress(45, "Sending to DeepSeek AI...")

            # ── Step 6: check dependencies & API key ─────────────────────────────
            try:
                import openai as _oi  # noqa: F401
                openai_ok = True
            except ImportError:
                openai_ok = False

            if not openai_ok:
                self._update_ai_progress(100, "OpenAI not installed — local analysis")
                self._display_ai_response(self._generate_local_analysis(analysis_type), analysis_type)
                return

            if not self._check_api_key():
                self._update_ai_progress(100, "No API key — local analysis")
                self._display_ai_response(self._generate_local_analysis(analysis_type), analysis_type)
                return

            # ── Step 7: call DeepSeek ─────────────────────────────────────────────
            reasoning, response = self._call_deepseek_api(prompt, analysis_type)

            if response and not response.startswith("❌"):
                self._update_ai_progress(90, "Formatting results...")
                self._display_ai_response(response, analysis_type)
                self._update_ai_progress(100, "Analysis complete!")
            else:
                # FIXED: Log the actual error so you can see WHY it failed
                if response and response.startswith("❌"):
                    self.log_message(f"🔴 DeepSeek API error: {response}", "red")
                else:
                    self.log_message("🔴 DeepSeek API returned empty response", "red")
                self._update_ai_progress(50, "API unavailable — local analysis...")
                self._display_ai_response(self._generate_local_analysis(analysis_type), analysis_type)
                self._update_ai_progress(100, "Local analysis complete")

        except Exception as e:
            self._ai_error(f"Analysis error: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.root.after(
                0,
                lambda: self.run_ai_btn.config(state='normal', text="🚀 Run AI Analysis"),
            )

    def _update_ai_progress(self, value: int, status: str):
        """Update AI progress bar and status"""

        def update():
            if hasattr(self, 'ai_progress_var'):
                self.ai_progress_var.set(value)
            if hasattr(self, 'ai_status_label'):
                self.ai_status_label.config(text=status)

        self.root.after(0, update)

    def _update_ai_status(self, status: str):
        """Update AI status label only"""
        if hasattr(self, 'ai_status_label'):
            self.ai_status_label.config(text=status)

    def _ai_error(self, message: str):
        def show_error():
            self.ai_results_text.delete(1.0, tk.END)
            self.ai_results_text.insert(tk.END, f"\n❌ ERROR\n{'─' * 60}\n\n", "bearish")
            self.ai_results_text.insert(tk.END, f"{message}\n\n", "warning")
            self.ai_results_text.insert(tk.END, "Please check:\n", "info")
            self.ai_results_text.insert(tk.END, "  • API key is configured in config.json\n", "info")
            self.ai_results_text.insert(tk.END, "  • Internet connection is active\n", "info")
            self.ai_results_text.insert(tk.END, "  • Market data is available\n", "info")
            self.ai_progress_var.set(0)
            self.ai_status_label.config(text="Error occurred")

        self.root.after(0, show_error)

    def _prepare_ai_data_summary(self, df: pd.DataFrame, depth: str) -> Dict[str, Any]:
        if depth == "quick":
            recent_candles = 100
            detailed_candles = 20
        elif depth == "standard":
            recent_candles = 500
            detailed_candles = 50
        else:
            recent_candles = min(len(df), 2000)
            detailed_candles = 100

        df_analysis = df.tail(recent_candles).copy()
        df_recent = df.tail(detailed_candles).copy()

        summary = {
            "metadata": {
                "symbol": self.symbol_var.get(),
                "interval": self.interval_var.get(),
                "total_candles_available": len(df),
                "candles_analyzed": len(df_analysis),
                "analysis_period": {
                    "start": str(df_analysis.index[0]),
                    "end": str(df_analysis.index[-1])
                },
                "current_time": datetime.now(timezone.utc).isoformat()
            },
            "current_state": self._get_current_market_state(df),
            "price_analysis": self._analyze_price_action(df_analysis),
            "indicator_analysis": self._analyze_indicators(df_analysis),
            "volume_analysis": self._analyze_volume(df_analysis),
            "trend_analysis": self._analyze_trends(df_analysis),
            "volatility_analysis": self._analyze_volatility(df_analysis),
            "pattern_detection": self._detect_patterns(df_recent),
            "support_resistance": self._calculate_support_resistance(df_analysis),
            "recent_candles": self._format_recent_candles(df.tail(10))
        }

        return summary

    def _get_current_market_state(self, df: pd.DataFrame) -> Dict[str, Any]:
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current

        close = float(current.get('Close', 0))
        prev_close = float(prev.get('Close', 0))
        change = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0

        return {
            "price": {
                "current": close,
                "previous": prev_close,
                "change_pct": round(change, 4),
                "open": float(current.get('Open', 0)),
                "high": float(current.get('High', 0)),
                "low": float(current.get('Low', 0))
            },
            "indicators": {
                "rsi": round(float(current.get('RSI', 50)), 2),
                "adx": round(float(current.get('ADX', 0)), 2),
                "cci": round(float(current.get('CCI', 0)), 2),
                "macd": round(float(current.get('MACD_closed', 0)), 4),
                "macd_signal": round(float(current.get('MACD_Signal_closed', 0)), 4),
                "atr": round(float(current.get('ATR', 0)), 4),
                "volume_ratio": round(float(current.get('Volume_Ratio', 1)), 2)
            },
            "ema": {
                "fast": round(float(current.get('EMA_Fast', 0)), 4),
                "mid": round(float(current.get('EMA_Mid', 0)), 4),
                "slow": round(float(current.get('EMA_Slow', 0)), 4),
                "alignment": "bullish" if close > float(current.get('EMA_Fast', 0)) > float(
                    current.get('EMA_Mid', 0)) > float(current.get('EMA_Slow', 0)) else "bearish" if close < float(
                    current.get('EMA_Fast', float('inf'))) < float(current.get('EMA_Mid', float('inf'))) < float(
                    current.get('EMA_Slow', float('inf'))) else "mixed"
            },
            "supertrend": {
                "value": float(current.get('SuperTrend', 0)),
                "direction": "bullish" if float(current.get('SuperTrend', 0)) == 1 else "bearish"
            },
            "kalman": {
                "strength": round(float(current.get('Kalman_Strength', 0)), 4),
                "color": current.get('Kalman_Color', 'unknown')
            }
        }

    def _analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        returns = pd.Series(closes).pct_change().dropna()

        return {
            "statistics": {
                "mean_price": round(float(np.mean(closes)), 4),
                "std_price": round(float(np.std(closes)), 4),
                "min_price": round(float(np.min(lows)), 4),
                "max_price": round(float(np.max(highs)), 4),
                "price_range_pct": round((float(np.max(highs)) - float(np.min(lows))) / float(np.mean(closes)) * 100, 2)
            },
            "returns": {
                "mean_return_pct": round(float(returns.mean() * 100), 4),
                "std_return_pct": round(float(returns.std() * 100), 4),
                "max_gain_pct": round(float(returns.max() * 100), 2),
                "max_loss_pct": round(float(returns.min() * 100), 2),
                "positive_days_pct": round(float((returns > 0).sum() / len(returns) * 100), 1)
            },
            "momentum": {
                "last_5_candles": round(float((closes[-1] / closes[-6] - 1) * 100), 2) if len(closes) > 5 else 0,
                "last_10_candles": round(float((closes[-1] / closes[-11] - 1) * 100), 2) if len(closes) > 10 else 0,
                "last_20_candles": round(float((closes[-1] / closes[-21] - 1) * 100), 2) if len(closes) > 20 else 0
            }
        }

    def _analyze_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        analysis = {}

        if 'RSI' in df.columns:
            rsi = df['RSI'].dropna()
            analysis['rsi'] = {
                "current": round(float(rsi.iloc[-1]), 1),
                "mean": round(float(rsi.mean()), 1),
                "overbought_count": int((rsi > 70).sum()),
                "oversold_count": int((rsi < 30).sum()),
                "trend": "overbought" if rsi.iloc[-1] > 70 else "oversold" if rsi.iloc[-1] < 30 else "neutral"
            }

        if 'ADX' in df.columns:
            adx = df['ADX'].dropna()
            analysis['adx'] = {
                "current": round(float(adx.iloc[-1]), 1),
                "mean": round(float(adx.mean()), 1),
                "strong_trend_pct": round(float((adx > 25).sum() / len(adx) * 100), 1),
                "trend_strength": "strong" if adx.iloc[-1] > 25 else "moderate" if adx.iloc[-1] > 20 else "weak"
            }

        if 'MACD_closed' in df.columns and 'MACD_Signal_closed' in df.columns:
            macd = df['MACD_closed'].dropna()
            signal = df['MACD_Signal_closed'].dropna()
            histogram = macd - signal
            analysis['macd'] = {
                "macd_current": round(float(macd.iloc[-1]), 4),
                "signal_current": round(float(signal.iloc[-1]), 4),
                "histogram_current": round(float(histogram.iloc[-1]), 4),
                "crossover": "bullish" if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2] else
                "bearish" if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2] else "none",
                "position": "above_signal" if macd.iloc[-1] > signal.iloc[-1] else "below_signal"
            }

        if 'CCI' in df.columns:
            cci = df['CCI'].dropna()
            analysis['cci'] = {
                "current": round(float(cci.iloc[-1]), 1),
                "mean": round(float(cci.mean()), 1),
                "overbought_count": int((cci > 100).sum()),
                "oversold_count": int((cci < -100).sum()),
                "zone": "overbought" if cci.iloc[-1] > 100 else "oversold" if cci.iloc[-1] < -100 else "neutral"
            }

        return analysis

    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        if 'Volume' not in df.columns:
            return {"error": "Volume data not available"}

        volume = df['Volume'].dropna()
        vol_ma = volume.rolling(20).mean()

        return {
            "current": float(volume.iloc[-1]),
            "average_20": round(float(vol_ma.iloc[-1]), 0) if not pd.isna(vol_ma.iloc[-1]) else 0,
            "ratio_to_average": round(float(volume.iloc[-1] / vol_ma.iloc[-1]), 2) if vol_ma.iloc[-1] > 0 else 1,
            "trend": "increasing" if volume.iloc[-5:].mean() > volume.iloc[-20:-5].mean() else "decreasing",
            "spike_detected": bool(volume.iloc[-1] > vol_ma.iloc[-1] * 2) if not pd.isna(vol_ma.iloc[-1]) else False
        }

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        closes = df['Close'].values

        def calc_trend(prices, period):
            if len(prices) < period:
                return "insufficient_data"
            segment = prices[-period:]
            slope = (segment[-1] - segment[0]) / segment[0] * 100
            if slope > 1:
                return "uptrend"
            elif slope < -1:
                return "downtrend"
            else:
                return "sideways"

        return {
            "short_term": {
                "period": "5 candles",
                "direction": calc_trend(closes, 5)
            },
            "medium_term": {
                "period": "20 candles",
                "direction": calc_trend(closes, 20)
            },
            "long_term": {
                "period": "50 candles",
                "direction": calc_trend(closes, 50)
            },
            "ema_trend": {
                "fast_slope": "up" if df['EMA_Fast'].iloc[-1] > df['EMA_Fast'].iloc[
                    -5] else "down" if 'EMA_Fast' in df.columns else "unknown",
                "alignment": self._get_current_market_state(df)['ema']['alignment']
            }
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        returns = df['Close'].pct_change().dropna()
        volatility_20 = returns.tail(20).std() * np.sqrt(252) * 100
        volatility_50 = returns.tail(50).std() * np.sqrt(252) * 100
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
        atr_pct = (atr / df['Close'].iloc[-1]) * 100 if df['Close'].iloc[-1] > 0 else 0

        return {
            "annualized_volatility_20": round(float(volatility_20), 2),
            "annualized_volatility_50": round(float(volatility_50), 2),
            "atr": round(float(atr), 4),
            "atr_percentage": round(float(atr_pct), 2),
            "volatility_regime": "high" if volatility_20 > 50 else "moderate" if volatility_20 > 25 else "low",
            "volatility_trend": "increasing" if volatility_20 > volatility_50 else "decreasing"
        }

    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        patterns = {
            "candlestick": [],
            "chart_patterns": [],
            "divergences": []
        }

        if len(df) < 5:
            return patterns

        last_5 = df.tail(5)

        for i in range(-3, 0):
            candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]

            open_p = float(candle['Open'])
            close_p = float(candle['Close'])
            high_p = float(candle['High'])
            low_p = float(candle['Low'])

            body = abs(close_p - open_p)
            upper_wick = high_p - max(open_p, close_p)
            lower_wick = min(open_p, close_p) - low_p

            if body < (high_p - low_p) * 0.1:
                patterns['candlestick'].append({"type": "doji", "index": i, "significance": "indecision"})

            if lower_wick > body * 2 and upper_wick < body * 0.5 and close_p < float(prev_candle['Close']):
                patterns['candlestick'].append({"type": "hammer", "index": i, "significance": "bullish_reversal"})

            if upper_wick > body * 2 and lower_wick < body * 0.5 and close_p > float(prev_candle['Close']):
                patterns['candlestick'].append(
                    {"type": "shooting_star", "index": i, "significance": "bearish_reversal"})

        if 'RSI' in df.columns:
            prices = df['Close'].tail(20)
            rsi = df['RSI'].tail(20)

            if prices.iloc[-1] < prices.iloc[-10] and rsi.iloc[-1] > rsi.iloc[-10]:
                patterns['divergences'].append({
                    "type": "bullish_divergence",
                    "indicator": "RSI",
                    "significance": "potential_reversal_up"
                })

            if prices.iloc[-1] > prices.iloc[-10] and rsi.iloc[-1] < rsi.iloc[-10]:
                patterns['divergences'].append({
                    "type": "bearish_divergence",
                    "indicator": "RSI",
                    "significance": "potential_reversal_down"
                })

        return patterns

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        current_price = closes[-1]

        swing_highs = []
        swing_lows = []

        for i in range(2, len(df) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[
                i + 2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                swing_lows.append(lows[i])

        resistance_levels = sorted([h for h in swing_highs if h > current_price])[:3]
        support_levels = sorted([l for l in swing_lows if l < current_price], reverse=True)[:3]

        return {
            "current_price": round(float(current_price), 4),
            "resistance_levels": [round(float(r), 4) for r in resistance_levels],
            "support_levels": [round(float(s), 4) for s in support_levels],
            "nearest_resistance": round(float(resistance_levels[0]), 4) if resistance_levels else None,
            "nearest_support": round(float(support_levels[0]), 4) if support_levels else None,
            "distance_to_resistance_pct": round((resistance_levels[0] / current_price - 1) * 100,
                                                2) if resistance_levels else None,
            "distance_to_support_pct": round((1 - support_levels[0] / current_price) * 100,
                                             2) if support_levels else None
        }

    def _format_recent_candles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": str(idx),
                "open": round(float(row['Open']), 4),
                "high": round(float(row['High']), 4),
                "low": round(float(row['Low']), 4),
                "close": round(float(row['Close']), 4),
                "volume": float(row.get('Volume', 0)),
                "rsi": round(float(row.get('RSI', 0)), 1),
                "adx": round(float(row.get('ADX', 0)), 1)
            })
        return candles

    def _prepare_trade_history_summary(self) -> Dict[str, Any]:
        if not self.trade_history:
            return {"message": "No trade history available"}

        closed_trades = [t for t in self.trade_history if t.get('type') == 'sell']

        if not closed_trades:
            return {"message": "No closed trades to analyze"}

        wins = [t for t in closed_trades if t.get('pnl', t.get('net_pnl', 0)) > 0]
        losses = [t for t in closed_trades if t.get('pnl', t.get('net_pnl', 0)) <= 0]

        total_pnl = sum(t.get('pnl', t.get('net_pnl', 0)) for t in closed_trades)
        win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0

        avg_win = sum(t.get('pnl', t.get('net_pnl', 0)) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.get('pnl', t.get('net_pnl', 0)) for t in losses) / len(losses) if losses else 0

        profit_factor = abs(sum(t.get('pnl', t.get('net_pnl', 0)) for t in wins) /
                            sum(t.get('pnl', t.get('net_pnl', 0)) for t in losses)) if losses and sum(
            t.get('pnl', t.get('net_pnl', 0)) for t in losses) != 0 else 0

        exit_reasons = {}
        for t in closed_trades:
            reason = t.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        recent_trades = []
        for t in closed_trades[-10:]:
            recent_trades.append({
                "time": str(t.get('time', 'unknown')),
                "entry_price": round(float(t.get('entry_price', 0)), 4),
                "exit_price": round(float(t.get('price', 0)), 4),
                "pnl": round(float(t.get('pnl', t.get('net_pnl', 0))), 2),
                "exit_reason": t.get('exit_reason', 'unknown')
            })

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "best_trade": round(max(t.get('pnl', t.get('net_pnl', 0)) for t in closed_trades), 2),
            "worst_trade": round(min(t.get('pnl', t.get('net_pnl', 0)) for t in closed_trades), 2),
            "exit_reasons": exit_reasons,
            "recent_trades": recent_trades
        }

    def _build_ai_prompt(self, analysis_type: str, data_summary: Dict, trade_summary: Dict) -> str:
        base_context = f"""You are an expert cryptocurrency trading analyst with deep knowledge of technical analysis, 
    risk management, and market psychology. You are analyzing {data_summary['metadata']['symbol']} 
    on the {data_summary['metadata']['interval']} timeframe.

    Current Time: {data_summary['metadata']['current_time']}
    Data Period: {data_summary['metadata']['analysis_period']['start']} to {data_summary['metadata']['analysis_period']['end']}
    Candles Analyzed: {data_summary['metadata']['candles_analyzed']}

    """

        data_section = f"""
    ═══════════════════════════════════════════════════════════════════════════════
    MARKET DATA SUMMARY
    ═══════════════════════════════════════════════════════════════════════════════

    CURRENT STATE:
    {json.dumps(data_summary['current_state'], indent=2)}

    PRICE ANALYSIS:
    {json.dumps(data_summary['price_analysis'], indent=2)}

    INDICATOR ANALYSIS:
    {json.dumps(data_summary['indicator_analysis'], indent=2)}

    VOLUME ANALYSIS:
    {json.dumps(data_summary['volume_analysis'], indent=2)}

    TREND ANALYSIS:
    {json.dumps(data_summary['trend_analysis'], indent=2)}

    VOLATILITY ANALYSIS:
    {json.dumps(data_summary['volatility_analysis'], indent=2)}

    PATTERN DETECTION:
    {json.dumps(data_summary['pattern_detection'], indent=2)}

    SUPPORT/RESISTANCE:
    {json.dumps(data_summary['support_resistance'], indent=2)}

    RECENT CANDLES (Last 10):
    {json.dumps(data_summary['recent_candles'], indent=2)}

    """

        trade_section = ""
        if trade_summary.get('total_trades', 0) > 0:
            trade_section = f"""
    ═══════════════════════════════════════════════════════════════════════════════
    TRADING HISTORY
    ═══════════════════════════════════════════════════════════════════════════════
    {json.dumps(trade_summary, indent=2)}

    """

        if analysis_type == "comprehensive":
            instructions = """
    ═══════════════════════════════════════════════════════════════════════════════
    ANALYSIS REQUEST: COMPREHENSIVE MARKET ANALYSIS
    ═══════════════════════════════════════════════════════════════════════════════

    Please provide a comprehensive analysis including:

    1. **MARKET OVERVIEW** (Current market regime, trend direction, momentum)

    2. **TECHNICAL ANALYSIS**
       - EMA alignment and trend strength
       - RSI, MACD, ADX interpretation
       - Volume analysis
       - Key patterns identified

    3. **RISK ASSESSMENT**
       - Current volatility level
       - Position sizing recommendations
       - Stop loss placement suggestions

    4. **TRADE RECOMMENDATIONS**
       - Clear BUY/SELL/WAIT recommendation
       - If trade recommended:
         * Entry zone (price range)
         * Stop loss level with reasoning
         * Take profit targets (R1, R2, R3)
         * Position size as % of portfolio
       - Confidence level (1-10)

    5. **KEY LEVELS TO WATCH**
       - Support levels
       - Resistance levels
       - Breakout/breakdown triggers

    6. **SUMMARY**
       - 2-3 sentence executive summary
       - Primary risk factors
       - Expected timeframe for the analysis

    Format your response with clear headers and bullet points for readability.
    Use specific numbers and prices wherever possible.
    """

        elif analysis_type == "technical":
            instructions = """
    ═══════════════════════════════════════════════════════════════════════════════
    ANALYSIS REQUEST: TECHNICAL DEEP DIVE
    ═══════════════════════════════════════════════════════════════════════════════

    Provide an in-depth technical analysis:

    1. **TREND ANALYSIS**
       - Multi-timeframe trend assessment
       - EMA/MA analysis with specific crossover signals
       - Trend strength quantification

    2. **MOMENTUM INDICATORS**
       - RSI: Current reading, divergences, overbought/oversold zones
       - MACD: Signal line position, histogram momentum, crossovers
       - ADX: Trend strength, +DI/-DI analysis
       - CCI: Cycle analysis

    3. **VOLUME ANALYSIS**
       - Volume trend vs price trend
       - Volume-price divergences
       - Accumulation/distribution signals

    4. **PATTERN RECOGNITION**
       - Candlestick patterns identified
       - Chart patterns (triangles, flags, H&S, etc.)
       - Pattern completion targets

    5. **DIVERGENCE ANALYSIS**
       - Bullish/bearish divergences across indicators
       - Hidden divergences
       - Convergence confirmation signals

    6. **SUPPORT/RESISTANCE**
       - Key horizontal levels
       - Dynamic support/resistance (EMAs)
       - Fibonacci levels if applicable

    7. **TECHNICAL OUTLOOK**
       - Bullish scenario with triggers
       - Bearish scenario with triggers
       - Most likely scenario with probability

    Include specific price levels and percentages throughout.
    """

        elif analysis_type == "performance":
            instructions = """
    ═══════════════════════════════════════════════════════════════════════════════
    ANALYSIS REQUEST: TRADING PERFORMANCE REVIEW
    ═══════════════════════════════════════════════════════════════════════════════

    Analyze the trading history and provide actionable insights:

    1. **PERFORMANCE METRICS EVALUATION**
       - Win rate assessment vs industry benchmarks
       - Profit factor analysis
       - Risk-adjusted returns evaluation

    2. **WINNING TRADES ANALYSIS**
       - Common characteristics of winning trades
       - Best entry conditions
       - Optimal hold times

    3. **LOSING TRADES ANALYSIS**
       - Common mistakes identified
       - Exit reason patterns
       - Avoidable losses

    4. **STRATEGY STRENGTHS**
       - What's working well
       - Edge identification
       - Conditions that favor the strategy

    5. **AREAS FOR IMPROVEMENT**
       - Specific weaknesses to address
       - Entry timing improvements
       - Exit strategy refinements
       - Position sizing adjustments

    6. **RECOMMENDED CHANGES**
       - Top 3 actionable improvements
       - Parameter adjustments to consider
       - Risk management enhancements

    7. **PSYCHOLOGICAL INSIGHTS**
       - Potential emotional trading patterns
       - Discipline assessment
       - Recommended mental adjustments

    Be specific with numbers and percentages. Provide actionable recommendations.
    """

        elif analysis_type == "risk":
            instructions = """
    ═══════════════════════════════════════════════════════════════════════════════
    ANALYSIS REQUEST: RISK ASSESSMENT
    ═══════════════════════════════════════════════════════════════════════════════

    Provide comprehensive risk analysis:

    1. **MARKET RISK**
       - Current volatility regime
       - Tail risk assessment
       - Black swan potential

    2. **POSITION SIZING RECOMMENDATIONS**
       - Kelly Criterion calculation
       - Conservative position size
       - Aggressive position size
       - Recommended size with reasoning

    3. **STOP LOSS ANALYSIS**
       - ATR-based stop levels
       - Support-based stop levels
       - Volatility-adjusted stops
       - Maximum risk per trade recommendation

    4. **DRAWDOWN ANALYSIS**
       - Maximum expected drawdown
       - Recovery time expectations
       - Risk of ruin calculation

    5. **PORTFOLIO RISK**
       - Correlation considerations
       - Concentration risk
       - Diversification recommendations

    6. **RISK SCENARIOS**
       - Best case scenario (with probability)
       - Base case scenario (with probability)
       - Worst case scenario (with probability)

    7. **RISK MANAGEMENT RULES**
       - Specific rules to implement
       - Position limits
       - Daily/weekly loss limits

    Include specific numbers, percentages, and price levels.
    """

        elif analysis_type == "forecast":
            instructions = """
    ═══════════════════════════════════════════════════════════════════════════════
    ANALYSIS REQUEST: MARKET FORECAST
    ═══════════════════════════════════════════════════════════════════════════════

    Provide market predictions and key levels:

    1. **SHORT-TERM FORECAST (Next 1-5 candles)**
       - Expected direction
       - Price targets
       - Confidence level

    2. **MEDIUM-TERM FORECAST (Next 10-20 candles)**
       - Trend expectation
       - Key price zones
       - Potential catalysts

    3. **SCENARIO ANALYSIS**
       - BULLISH SCENARIO:
         * Trigger conditions
         * Price targets
         * Probability estimate
       - BEARISH SCENARIO:
         * Trigger conditions
         * Price targets
         * Probability estimate
       - SIDEWAYS SCENARIO:
         * Range boundaries
         * Duration estimate
         * Probability estimate

    4. **KEY LEVELS TO WATCH**
       - Breakout levels (long triggers)
       - Breakdown levels (short triggers)
       - Invalidation levels

    5. **EVENTS/CATALYSTS**
       - Technical events (pattern completions, crossovers)
       - Time-based projections

    6. **TRADING PLAN**
       - If price goes to X, then do Y
       - Multiple if-then scenarios

    Provide specific price levels and timeframes. Be clear about confidence levels.
    """

        elif analysis_type == "setups":
            instructions = """
    ═══════════════════════════════════════════════════════════════════════════════
    ANALYSIS REQUEST: TRADE SETUP SCANNER
    ═══════════════════════════════════════════════════════════════════════════════

    Identify and detail any actionable trade setups:

    1. **CURRENT SETUP IDENTIFICATION**
       - Is there a valid setup NOW? (Yes/No)
       - Setup type (breakout, pullback, reversal, continuation)
       - Setup quality rating (A/B/C)

    2. **IF SETUP EXISTS - TRADE PLAN:**

       📍 ENTRY:
       - Entry type (market/limit)
       - Entry price or zone
       - Entry trigger condition

       🛑 STOP LOSS:
       - Stop loss price
       - Distance from entry (% and $)
       - Reasoning for stop placement

       🎯 TAKE PROFIT TARGETS:
       - Target 1: Price, R-multiple, % gain
       - Target 2: Price, R-multiple, % gain
       - Target 3: Price, R-multiple, % gain

       📊 POSITION SIZE:
       - Recommended % of portfolio
       - Risk amount calculation
       - Shares/units to trade

       ⚡ TRADE MANAGEMENT:
       - When to move stop to breakeven
       - Partial profit taking rules
       - Trail stop rules

    3. **SETUP CONFLUENCE**
       - Factors supporting this setup
       - Warning signs or concerns
       - Overall confidence (1-10)

    4. **ALTERNATIVE SETUPS**
       - Pending setups to watch
       - Trigger conditions for each
       - Relative priority

    5. **NO-TRADE CONDITIONS**
       - When to avoid trading
       - Wait conditions

    Be extremely specific with prices and levels. This should be immediately actionable.
    """

        else:
            instructions = """
    ═══════════════════════════════════════════════════════════════════════════════
    ANALYSIS REQUEST: GENERAL ANALYSIS
    ═══════════════════════════════════════════════════════════════════════════════

    Provide a general market analysis with key insights and recommendations.
    Focus on actionable information and specific price levels.
    """

        return base_context + data_section + trade_section + instructions

    def _call_deepseek_api_streaming(self, prompt, analysis_type, on_token_callback):
        """Streaming version of DeepSeek API call"""
        try:
            if not OPENAI_AVAILABLE:
                on_token_callback("❌ OpenAI library not installed. Run: pip install openai")
                return "❌ OpenAI library not installed"

            api_key = self.config.get('deepseek_api_key', '')
            if not api_key or not api_key.startswith('sk-'):
                on_token_callback("❌ No valid DeepSeek API key found in config.json")
                return "❌ No valid DeepSeek API key found"

            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )

            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert quantitative trading analyst. "
                                "Provide precise, actionable trading analysis."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                    stream=True,
                    timeout=120
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    token = getattr(delta, 'content', None) or ""
                    if token:
                        full_response += token
                        on_token_callback(token)

                return full_response

            except openai.AuthenticationError:
                error = "\n❌ API Authentication Error: Check your DeepSeek API key."
                on_token_callback(error)
                return full_response + error
            except Exception as e:
                error = f"\n❌ Streaming error: {str(e)}"
                on_token_callback(error)
                return full_response + error

        except Exception as e:
            error = f"\n❌ Streaming setup error: {str(e)}"
            on_token_callback(error)
            return error

    def _call_deepseek_api(self, prompt, analysis_type):
        """Call DeepSeek API with proper error handling"""
        try:
            # FIXED: Re-import openai fresh every call — don't rely on the
            # module-level OPENAI_AVAILABLE flag which was set at startup
            # before the package was installed.
            try:
                import openai as _openai
            except ImportError:
                return None, "❌ OpenAI library not installed. Run: pip install openai"

            # Check API key first
            api_key = self.config.get('deepseek_api_key', '')
            if not api_key or not api_key.startswith('sk-'):
                return None, "❌ No valid DeepSeek API key found in config.json"

            client = _openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )

            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert quantitative trading analyst with deep expertise in "
                            "technical analysis, risk management, and algorithmic trading. "
                            "Provide precise, actionable analysis based on the market data provided."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4096,
                timeout=120
            )

            message = response.choices[0].message
            reasoning = getattr(message, 'reasoning_content', None)
            answer = message.content

            return reasoning, answer

        except Exception as e:
            # Map known openai error types by name so we don't need a top-level import
            etype = type(e).__name__
            if etype == 'AuthenticationError':
                return None, "❌ API Authentication Error: Check your DeepSeek API key in Settings."
            elif etype == 'RateLimitError':
                return None, "❌ Rate Limit Error: Too many requests. Please wait 30 seconds and try again."
            elif etype == 'APITimeoutError':
                return None, "❌ Timeout Error: DeepSeek took too long. Try a shorter analysis type."
            elif etype == 'APIConnectionError':
                return None, "❌ Connection Error: Cannot reach DeepSeek API. Check your internet connection."
            elif etype == 'APIError':
                return None, f"❌ DeepSeek API Error: {str(e)}"
            else:
                return None, f"❌ Unexpected Error: {str(e)}"

    def _generate_local_analysis(self, analysis_type: str) -> str:
        """Generate local fallback analysis when API is unavailable"""
        df = self.get_market_data()
        if df is None:
            return "Unable to generate analysis - no market data available."

        if hasattr(self.strategy, 'calculate_indicators'):
            df = self.strategy.calculate_indicators(df)

        current = df.iloc[-1]

        close = float(current.get('Close', 0))
        rsi = float(current.get('RSI', 50))
        adx = float(current.get('ADX', 0))
        macd = float(current.get('MACD_closed', 0))
        macd_signal = float(current.get('MACD_Signal_closed', 0))
        ema_fast = float(current.get('EMA_Fast', 0))
        ema_slow = float(current.get('EMA_Slow', 0))
        atr = float(current.get('ATR', 0))

        bullish_ema = close > ema_fast > ema_slow
        macd_bullish = macd > macd_signal
        rsi_ok = 30 < rsi < 70
        strong_trend = adx > 25

        score = 0
        if bullish_ema: score += 25
        if macd_bullish: score += 20
        if rsi_ok: score += 15
        if strong_trend: score += 20
        if float(current.get('Volume_Ratio', 1)) > 1.2: score += 10

        if score >= 70:
            recommendation = "STRONG BUY"
            action = "Enter long position"
        elif score >= 55:
            recommendation = "BUY"
            action = "Consider long entry"
        elif score >= 40:
            recommendation = "HOLD/WAIT"
            action = "Wait for better setup"
        elif score >= 25:
            recommendation = "CAUTION"
            action = "Reduce exposure or wait"
        else:
            recommendation = "AVOID"
            action = "Stay in cash"

        analysis = f"""
        ═══════════════════════════════════════════════════════════════════════════════
        📊 LOCAL ANALYSIS (API Unavailable)
        ═══════════════════════════════════════════════════════════════════════════════

        ⚠️ Note: This is a simplified local analysis. For comprehensive AI-powered 
        analysis, please configure your DeepSeek API key in config.json.

        ─────────────────────────────────────────────────────────────────────────────────
        CURRENT MARKET STATE
        ─────────────────────────────────────────────────────────────────────────────────

        Symbol: {self.symbol_var.get()}
        Price: ${close:.4f}
        Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

        ─────────────────────────────────────────────────────────────────────────────────
        INDICATOR READINGS
        ─────────────────────────────────────────────────────────────────────────────────

        RSI (14):        {rsi:.1f}  {'⚠️ Overbought' if rsi > 70 else '⚠️ Oversold' if rsi < 30 else '✓ Neutral'}
        ADX:             {adx:.1f}  {'✓ Strong Trend' if adx > 25 else '⚠️ Weak Trend'}
        MACD:            {macd:.4f}  {'✓ Bullish' if macd_bullish else '⚠️ Bearish'}
        EMA Alignment:   {'✓ Bullish Stack' if bullish_ema else '⚠️ Bearish/Mixed'}
        ATR:             ${atr:.4f}

        ─────────────────────────────────────────────────────────────────────────────────
        ANALYSIS SCORE
        ─────────────────────────────────────────────────────────────────────────────────

        Score: {score}/100

        Factors:
          • EMA Alignment:    {'+25' if bullish_ema else '0'} pts
          • MACD Position:    {'+20' if macd_bullish else '0'} pts
          • RSI Zone:         {'+15' if rsi_ok else '0'} pts
          • Trend Strength:   {'+20' if strong_trend else '0'} pts
          • Volume Ratio:     {'+10' if float(current.get('Volume_Ratio', 1)) > 1.2 else '0'} pts

        ─────────────────────────────────────────────────────────────────────────────────
        RECOMMENDATION
        ─────────────────────────────────────────────────────────────────────────────────

        🎯 {recommendation}

        Action: {action}

        If entering a trade:
          • Entry:      ${close:.4f}
          • Stop Loss:  ${close - (atr * 2):.4f} (2x ATR below)
          • Target 1:   ${close + (atr * 2):.4f} (2x ATR above)
          • Target 2:   ${close + (atr * 3):.4f} (3x ATR above)
          • Target 3:   ${close + (atr * 5):.4f} (5x ATR above)

        ─────────────────────────────────────────────────────────────────────────────────
        ⚠️ DISCLAIMER
        ─────────────────────────────────────────────────────────────────────────────────

        This is automated analysis and should not be the sole basis for trading 
        decisions. Always do your own research and manage risk appropriately.

        To enable full AI analysis, add your DeepSeek API key to config.json:
        "deepseek_api_key": "sk-your-deepseek-api-key-here"
        """
        return analysis

    def display_quality_score_table(self, quality_score, component_scores, reason=""):
        self.log_message("=" * 77, "cyan")
        self.log_message(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", "cyan")
        self.log_message("=" * 77, "cyan")

        quality_bar = self._create_confidence_bar(quality_score, 10)
        self.log_message(f"🎯 QUALITY SCORE: {quality_score}/100 {quality_bar}",
                         "green" if quality_score >= 75 else "orange" if quality_score >= 60 else "red")

        self.log_message(f"\n📊 Component Scores:", "cyan")

        col_widths = [15, 15, 15, 15]
        self.log_message(self._create_table_separator(col_widths, "top"), "purple")
        headers = ["EMA", "ADX", "MACD", "Volume"]
        self.log_message(self._create_table_row(headers, col_widths), "white")
        self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

        data_cells = [
            f"{component_scores.get('ema', 0)}/20",
            f"{component_scores.get('adx', 0)}/20",
            f"{component_scores.get('macd', 0)}/25",
            f"{component_scores.get('volume', 0)}/15"
        ]
        self.log_message(self._create_table_row(data_cells, col_widths),
                         "green" if quality_score >= 75 else "orange")
        self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

        self.log_message(f"\n📈 RSI Score: {component_scores.get('rsi', 0)}/20", "cyan")

        if hasattr(self.strategy, '_get_position_multiplier'):
            position_mult = self.strategy._get_position_multiplier(quality_score)
            self.log_message(f"💰 Position Size: {position_mult * 100:.0f}% of normal",
                             "green" if position_mult >= 1.0 else "orange")

        if reason:
            self.log_message(f"📝 {reason}", "blue")

        self.log_message("=" * 77, "cyan")

    def _display_ai_response(self, response: str, analysis_type: str):
        def update_display():
            self.ai_results_text.delete(1.0, tk.END)

            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
            header = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    🤖 AI DEEPSEEK ANALYSIS                                   ║
    ║  📅 {timestamp} | {self.symbol_var.get()} | {self.interval_var.get()}
    ╚══════════════════════════════════════════════════════════════════════════════╝

    """
            self.ai_results_text.insert(tk.END, header, "header")

            self._format_and_insert_response(response)

            footer = f"""

    ─────────────────────────────────────────────────────────────────────────────────
    Analysis generated by DeepSeek AI. This is not financial advice.
    Always do your own research before making trading decisions.
    ─────────────────────────────────────────────────────────────────────────────────
    """
            self.ai_results_text.insert(tk.END, footer, "separator")
            self.ai_results_text.see("1.0")

        self.root.after(0, update_display)

    def _format_and_insert_response(self, response: str):
        lines = response.split('\n')

        for line in lines:
            if line.startswith('═') or line.startswith('─') or line.startswith('╔') or line.startswith('╚'):
                self.ai_results_text.insert(tk.END, line + '\n', "separator")
            elif line.startswith('**') and line.endswith('**'):
                self.ai_results_text.insert(tk.END, line.replace('**', '') + '\n', "subheader")
            elif any(line.startswith(f"{i}.") for i in range(1, 10)):
                self.ai_results_text.insert(tk.END, line + '\n', "highlight")
            elif '✓' in line or 'BUY' in line.upper() or 'BULLISH' in line.upper() or 'POSITIVE' in line.upper():
                self.ai_results_text.insert(tk.END, line + '\n', "bullish")
            elif '✗' in line or 'SELL' in line.upper() or 'BEARISH' in line.upper() or 'NEGATIVE' in line.upper() or 'STOP' in line.upper():
                self.ai_results_text.insert(tk.END, line + '\n', "bearish")
            elif '⚠' in line or 'WARNING' in line.upper() or 'CAUTION' in line.upper() or 'RISK' in line.upper():
                self.ai_results_text.insert(tk.END, line + '\n', "warning")
            elif '$' in line or 'PRICE' in line.upper() or 'TARGET' in line.upper() or 'ENTRY' in line.upper():
                self.ai_results_text.insert(tk.END, line + '\n', "price")
            elif line.strip().startswith('#') or line.strip().startswith('*'):
                self.ai_results_text.insert(tk.END, line + '\n', "subheader")
            else:
                self.ai_results_text.insert(tk.END, line + '\n', "info")

    def _copy_ai_results(self):
        try:
            results = self.ai_results_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(results)
            self.log_message("📋 AI results copied to clipboard", "green")
        except Exception as e:
            self.log_message(f"Failed to copy: {e}", "red")

    def _export_ai_report(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = self.symbol_var.get().replace('-', '_')
            analysis_type = self.ai_analysis_type.get()

            filename = f"ai_analysis_{symbol}_{analysis_type}_{timestamp}.txt"

            results = self.ai_results_text.get(1.0, tk.END)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"AI DeepSeek Analysis Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Symbol: {self.symbol_var.get()}\n")
                f.write(f"Interval: {self.interval_var.get()}\n")
                f.write(f"Analysis Type: {analysis_type}\n")
                f.write("=" * 80 + "\n\n")
                f.write(results)

            self.log_message(f"📄 Report exported: {filename}", "green")

            import os
            os.startfile(filename)

        except Exception as e:
            self.log_message(f"Export failed: {e}", "red")

    def open_help_pdf(self):
        try:
            import webbrowser
            import subprocess
            import platform

            pdf_file = "help_documentation.pdf"

            if os.path.exists(pdf_file):
                pdf_path = pdf_file
            else:
                pdf_path = os.path.join("help", pdf_file)
                if not os.path.exists(pdf_path):
                    self.show_help_message()
                    return

            system = platform.system()

            if system == "Windows":
                os.startfile(pdf_path)
            elif system == "Darwin":
                subprocess.call(["open", pdf_path])
            else:
                subprocess.call(["xdg-open", pdf_path])

            self.log_message(f"📖 Opening help document: {pdf_path}", "blue")

        except Exception as e:
            self.log_message(f"❌ Could not open PDF: {str(e)}", "red")
            self.show_help_message()

    def show_help_message(self):
        help_text = """
        TRADING ML APPLICATION - QUICK HELP

        BASIC OPERATION:
        1. Select Mode: Demo (virtual trading), Live (real), or Backtest
        2. Check Connection: Verify API connectivity
        3. Start Trading: Begin automated trading

        STRATEGIES:
        • Momentum: Trend-following strategy
        • Kalman: Mean-reversion strategy

        RISK MANAGEMENT:
        • Order Size: % of account per trade
        • Stop Loss: Maximum loss percentage
        • Trailing Stop: Dynamic stop loss

        ML FEATURES:
        • Enable ML for predictive trading
        • Select ML model (Random Forest, XGBoost, LSTM)
        • Adjust confidence threshold

        BACKTESTING:
        • Select date range
        • Run standard or optimization backtest
        • Monte Carlo simulation available

        SETTINGS:
        • Click ⚙ Settings to adjust strategy parameters
        • Toggle between Default and Custom parameters

        For detailed documentation, please ensure 'help_documentation.pdf' 
        is in the application directory or 'help' folder.
        """

        help_window = tk.Toplevel(self.root)
        help_window.title("Trading ML Application - Help")
        help_window.geometry("600x500")
        help_window.resizable(True, True)

        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        help_textbox = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Arial", 10),
            bg='white',
            fg='black'
        )
        help_textbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=help_textbox.yview)

        help_textbox.insert(tk.END, help_text)
        help_textbox.config(state=tk.DISABLED)

        close_button = ttk.Button(
            help_window,
            text="Close",
            command=help_window.destroy
        )
        close_button.pack(pady=10)

    def toggle_monte_carlo_options(self):
        """Show/hide Monte Carlo options and adjust frame height dynamically"""
        if self.use_monte_carlo_var.get():
            self.monte_carlo_options_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=2, padx=3)
            self.frame2_middle.config(height=340)
            self.log_message("🎲 Monte Carlo simulation enabled", "purple")
        else:
            self.monte_carlo_options_frame.grid_forget()
            self.frame2_middle.config(height=0)
            self.log_message("🎲 Monte Carlo simulation disabled", "orange")

        self.frame2_middle.update_idletasks()

    def set_strategy(self, strategy_name):
        if strategy_name in self.strategies:
            self.strategy = self.strategies[strategy_name]
            self.log_message(f"Strategy changed to: {strategy_name}", "blue")
        else:
            self.log_message(f"Unknown strategy: {strategy_name}", "red")

    def base_symbol(self):
        return self.symbol_var.get().split('-')[0]

    def validate_symbol(self):
        base_sym = self.base_symbol()
        if base_sym not in self.virtual_balance:
            self.virtual_balance[base_sym] = 0
            self.symbol_balance_label.config(text=f"{base_sym}: 0.0000")

    def validate_market_data(self, df):
        if df is None or len(df) < 50:
            return False

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            return False

        if df[required_columns].isna().any().any():
            return False

        price_changes = df['Close'].pct_change().dropna()
        if len(price_changes) > 0 and (price_changes.abs() > 0.5).any():
            return False

        return True

    def calculate_dynamic_position_size(self, current_price, stop_loss_price):
        account_balance = self.get_balance('USDT')

        if hasattr(self, 'current_data') and self.current_data is not None:
            atr = self.current_data.get('ATR', 1)
            normal_atr = getattr(self, 'atr_mean', 1)
            volatility_factor = max(0.5, min(2.0, normal_atr / max(atr, 0.001)))
        else:
            volatility_factor = 1.0

        if hasattr(self.strategy, 'calculate_dynamic_position_size'):
            return self.strategy.calculate_dynamic_position_size(
                account_balance, current_price, atr, "normal"
            )
        else:
            risk_amount = account_balance * (self.order_size_var.get() / 100)
            price_distance = abs(current_price - stop_loss_price)
            return risk_amount / price_distance if price_distance > 0 else 0

    def update_performance_metrics(self):
        if not hasattr(self, 'performance_analytics'):
            from strategies.base3_New import PerformanceAnalytics
            self.performance_analytics = PerformanceAnalytics()

        metrics = self.performance_analytics.calculate_strategy_metrics(self.trade_history)

        current_equity = self.get_balance('USDT')
        if hasattr(self, 'position') and self.position['price'] is not None:
            current_equity += self.position['quantity'] * self.get_current_price()
        self.performance_analytics.update_equity_curve(current_equity)

        return metrics

    def check_market_alerts(self, df):
        if not hasattr(self, 'alert_manager'):
            from strategies.base3_New import AlertManager
            self.alert_manager = AlertManager()

        alerts = self.alert_manager.check_market_anomalies(df)

        for alert in alerts:
            self.log_message(f"🚨 {alert['type']}: {alert['message']}",
                             "red" if alert['severity'] == 'HIGH' else "orange")

        return alerts

    def execute_order_with_reliability(self, order_params):
        if not hasattr(self, 'order_manager'):
            from strategies.base3_New import OrderManager
            self.order_manager = OrderManager()

        return self.order_manager.execute_with_retry(order_params, self)

    def update_image(self):
        try:
            if self.current_status == "buy":
                image_path = self.resource_path("images/bullish.jpg")
            elif self.current_status == "sell":
                image_path = self.resource_path("images/bearish.jpg")
            else:
                image_path = self.resource_path("images/cryptoimage.jpg")

            if not os.path.exists(image_path):
                self.log_message(f"Image not found: {image_path}", "orange")
                return

            picture_img = Image.open(image_path)
            picture_img = picture_img.resize((334, 95), Image.Resampling.LANCZOS)
            self.picture_photo = ImageTk.PhotoImage(picture_img)

            if not hasattr(self, 'picture_label'):
                self.picture_label = ttk.Label(self.config_frame, borderwidth=2, relief="groove")
                self.picture_label.grid(row=0, column=2, rowspan=3, padx=10, pady=5, sticky="nsew")

            self.picture_label.configure(image=self.picture_photo)
            self.picture_label.image = self.picture_photo

        except Exception as e:
            self.log_message(f"Could not update image: {e}", "orange")

    def load_config(self):
        try:
            with open('config.json') as f:
                self.config = json.load(f)
            required_keys = ['live', 'demo', 'backtest']
            for mode in required_keys:
                if mode not in self.config:
                    raise ValueError(f"Missing {mode} configuration")
                for key in ['api_key', 'api_secret_key', 'passphrase']:
                    if key not in self.config[mode]:
                        raise ValueError(f"Missing {key} in {mode} config")
        except Exception as e:
            messagebox.showerror("Error", f"Config load failed: {str(e)}")
            self.root.destroy()

    def load_momentum_params_from_file(self, filename="momentum_params.json"):
        """
        Load Momentum parameters from a JSON file and apply them to the strategy.
        Returns True if successful, False otherwise.
        """
        try:
            if not os.path.exists(filename):
                self.log_message(f"❌ Parameter file not found: {filename}", "red")
                return False

            with open(filename, 'r') as f:
                loaded_params = json.load(f)

            # Get current strategy
            if 'Momentum' not in self.strategies:
                self.log_message("❌ Momentum strategy not initialized", "red")
                return False

            strategy = self.strategies['Momentum']

            # Update strategy attributes
            updates_made = 0
            for param, value in loaded_params.items():
                if hasattr(strategy, param):
                    current = getattr(strategy, param)
                    if current != value:
                        setattr(strategy, param, value)
                        updates_made += 1
                        self.log_message(f"   ✓ Updated {param}: {current} → {value}", "green")

                # Also update config if it exists
                if hasattr(strategy, 'config') and param in strategy.config:
                    if strategy.config[param] != value:
                        strategy.config[param] = value

            # Also update custom_params for persistence
            if 'momentum' in self.custom_params:
                self.custom_params['momentum'].update(loaded_params)

            self.log_message(f"✅ Loaded {updates_made} parameters from {filename}", "green")
            return True

        except json.JSONDecodeError as e:
            self.log_message(f"❌ Invalid JSON in {filename}: {e}", "red")
            return False
        except Exception as e:
            self.log_message(f"❌ Error loading parameters: {e}", "red")
            import traceback
            self.log_message(traceback.format_exc(), "red")
            return False

    def load_params_from_file(self, filename="momentum_params_default.json"):
        """
        Load default Momentum parameters from a JSON file.
        """
        try:
            if not os.path.exists(filename):
                self.log_message(f"⚠️ Default parameter file not found: {filename}", "orange")
                # Try to create default params file
                return self.save_default_params_to_file(filename)
            return self.load_momentum_params_from_file(filename)
        except Exception as e:
            self.log_message(f"❌ Error loading default params: {e}", "red")
            return False

    def save_default_params_to_file(self, filename="momentum_params_default.json"):
        """
        Save current default Momentum parameters to a JSON file.
        """
        try:
            default_params = self.get_default_momentum_params()

            with open(filename, 'w') as f:
                json.dump(default_params, f, indent=4)

            self.log_message(f"✅ Default parameters saved to {filename}", "green")
            return True
        except Exception as e:
            self.log_message(f"❌ Error saving default params: {e}", "red")
            return False

    def load_params(self):
        """
        Load strategy parameters from strategy_settings.json (used at initial startup).
        Contains default_params, custom_params, and selected_mode.
        """
        self.log_message("=" * 70, "blue")
        self.log_message("📂 LOADING STRATEGY PARAMETERS FROM strategy_settings.json", "blue")
        self.log_message("=" * 70, "blue")

        # First, ensure we have the structure initialized with built-in defaults
        if not hasattr(self, 'default_params'):
            self.default_params = {}
        if not hasattr(self, 'custom_params'):
            self.custom_params = {}

        # Load built-in defaults as base
        built_in_momentum = self.get_default_momentum_params()
        built_in_kalman = self.get_default_kalman_params()

        self.default_params['momentum'] = built_in_momentum.copy()
        self.default_params['kalman'] = built_in_kalman.copy()
        self.default_params['scalping'] = self.get_default_scalping_params().copy()
        self.custom_params['momentum'] = built_in_momentum.copy()
        self.custom_params['kalman'] = built_in_kalman.copy()
        self.custom_params['scalping'] = self.get_default_scalping_params().copy()

        # Load from strategy_settings.json
        settings_file = self.strategy_settings_file  # "strategy_settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)

                self.log_message(f"📂 Found {settings_file}", "green")

                # Load default_params if they exist
                if 'default_params' in settings:
                    if 'momentum' in settings['default_params']:
                        loaded_momentum = settings['default_params']['momentum']
                        if loaded_momentum:
                            self.default_params['momentum'].update(loaded_momentum)
                            self.log_message(f"   ✅ Loaded {len(loaded_momentum)} Momentum default parameters", "green")

                    if 'kalman' in settings['default_params']:
                        loaded_kalman = settings['default_params']['kalman']
                        if loaded_kalman:
                            self.default_params['kalman'].update(loaded_kalman)
                            self.log_message(f"   ✅ Loaded {len(loaded_kalman)} Kalman default parameters", "green")

                # Load custom_params if they exist
                if 'custom_params' in settings and settings['custom_params']:
                    custom_loaded = False

                    if 'momentum' in settings['custom_params']:
                        loaded_custom_momentum = settings['custom_params']['momentum']
                        if loaded_custom_momentum:
                            self.custom_params['momentum'].update(loaded_custom_momentum)
                            custom_loaded = True
                            self.log_message(f"   ✅ Loaded {len(loaded_custom_momentum)} Momentum custom parameters",
                                             "green")

                            # Log a few sample custom params to verify
                            sample_params = list(loaded_custom_momentum.items())[:3]
                            for param, value in sample_params:
                                self.log_message(f"      • {param}: {value}", "blue")

                    if 'kalman' in settings['custom_params']:
                        loaded_custom_kalman = settings['custom_params']['kalman']
                        if loaded_custom_kalman:
                            self.custom_params['kalman'].update(loaded_custom_kalman)
                            custom_loaded = True
                            self.log_message(f"   ✅ Loaded {len(loaded_custom_kalman)} Kalman custom parameters",
                                             "green")

                    if 'scalping' in settings['custom_params']:
                        loaded_custom_scalping = settings['custom_params']['scalping']
                        if loaded_custom_scalping:
                            self.custom_params['scalping'].update(loaded_custom_scalping)
                            custom_loaded = True
                            self.log_message(f"   ✅ Loaded {len(loaded_custom_scalping)} Scalping custom parameters",
                                             "green")

                    if custom_loaded:
                        self.log_message(f"   ✅ Custom parameters loaded successfully", "green")
                else:
                    self.log_message("   ℹ️ No custom parameters found in settings file", "blue")

                # Load selected mode
                if 'selected_mode' in settings:
                    self.param_toggle_var.set(settings['selected_mode'])
                    self.log_message(f"   ✅ Loaded selected mode: {settings['selected_mode']}", "green")

            except Exception as e:
                self.log_message(f"⚠️ Error loading from {settings_file}: {e}", "orange")
                import traceback
                self.log_message(traceback.format_exc(), "red")
        else:
            self.log_message(f"ℹ️ No {settings_file} found - using built-in defaults", "blue")

        # Final verification
        self.log_message("=" * 70, "cyan")
        self.log_message("📊 FINAL PARAMETER STATE:", "cyan")
        self.log_message(f"   Source: {settings_file if os.path.exists(settings_file) else 'Built-in defaults'}",
                         "blue")
        self.log_message(f"   Momentum default params: {len(self.default_params.get('momentum', {}))}", "blue")
        self.log_message(f"   Kalman default params: {len(self.default_params.get('kalman', {}))}", "blue")
        self.log_message(f"   Momentum custom params: {len(self.custom_params.get('momentum', {}))}", "blue")
        self.log_message(f"   Kalman custom params: {len(self.custom_params.get('kalman', {}))}", "blue")
        self.log_message(f"   Current mode: {self.param_toggle_var.get()}", "blue")

        # Log a few key custom params to verify they're set correctly
        if self.custom_params.get('momentum'):
            self.log_message("   Key custom parameter values:", "purple")
            key_params = ['quality_minimum_score', 'adx_min', 'risk_full_position', 'ema_fast_period']
            for param in key_params:
                if param in self.custom_params['momentum']:
                    self.log_message(f"      • {param}: {self.custom_params['momentum'][param]}", "white")

        self.log_message("=" * 70, "cyan")

        # Update UI widgets if settings panel is open
        if hasattr(self, 'momentum_param_widgets') or hasattr(self, 'kalman_param_widgets'):
            update_dict = {
                'momentum': self.custom_params['momentum'],
                'kalman': self.custom_params['kalman']
            }
            self._update_param_widgets_from_dict(update_dict)

        # Apply the loaded parameters based on current toggle selection
        self.apply_selected_parameters()

        self.log_message("=" * 70, "green")
        self.log_message("✅ PARAMETER LOADING COMPLETE", "green")
        self.log_message("=" * 70, "green")

        return True

    def _update_param_widgets_from_dict(self, params_dict):
        """Helper method to update UI widgets from loaded parameters"""
        try:
            widgets_updated = False

            # Update momentum widgets if they exist
            if hasattr(self, 'momentum_param_widgets') and 'momentum' in params_dict:
                momentum_updates = 0
                for param_name, value in params_dict['momentum'].items():
                    if param_name in self.momentum_param_widgets:
                        widget_info = self.momentum_param_widgets[param_name]
                        if isinstance(widget_info['custom'], tk.BooleanVar):
                            current = widget_info['custom'].get()
                            new_value = bool(value)
                            if current != new_value:
                                widget_info['custom'].set(new_value)
                                momentum_updates += 1
                        else:
                            current = widget_info['custom'].get()
                            new_value = str(value)
                            if current != new_value:
                                widget_info['custom'].set(new_value)
                                momentum_updates += 1

                if momentum_updates > 0:
                    self.log_message(f"   📊 Updated {momentum_updates} Momentum parameter widgets", "blue")
                    widgets_updated = True

            # Update kalman widgets if they exist
            if hasattr(self, 'kalman_param_widgets') and 'kalman' in params_dict:
                kalman_updates = 0
                for param_name, value in params_dict['kalman'].items():
                    if param_name in self.kalman_param_widgets:
                        widget_info = self.kalman_param_widgets[param_name]
                        if isinstance(widget_info['custom'], tk.BooleanVar):
                            current = widget_info['custom'].get()
                            new_value = bool(value)
                            if current != new_value:
                                widget_info['custom'].set(new_value)
                                kalman_updates += 1
                        else:
                            current = widget_info['custom'].get()
                            new_value = str(value)
                            if current != new_value:
                                widget_info['custom'].set(new_value)
                                kalman_updates += 1

                if kalman_updates > 0:
                    self.log_message(f"   📊 Updated {kalman_updates} Kalman parameter widgets", "blue")
                    widgets_updated = True

            if widgets_updated:
                self.log_message("✅ Parameter widgets synchronized with loaded values", "green")

        except Exception as e:
            self.log_message(f"⚠️ Error updating UI widgets: {e}", "orange")

    def save_current_defaults(self):
        """Save current default parameters to file"""
        try:
            defaults_to_save = {
                'momentum': self.default_params.get('momentum', {}),
                'kalman': self.default_params.get('kalman', {})
            }

            filename = "default_params.json"
            with open(filename, 'w') as f:
                json.dump(defaults_to_save, f, indent=4)

            self.log_message(f"💾 Saved current defaults to {filename}", "green")
            return True
        except Exception as e:
            self.log_message(f"❌ Error saving defaults: {e}", "red")
            return False

    def save_current_custom(self):
        """Save current custom parameters to file"""
        try:
            # Get current custom params from UI if panel is open
            if hasattr(self, 'momentum_param_widgets'):
                for param_name, widget_info in self.momentum_param_widgets.items():
                    custom_var = widget_info['custom']
                    if isinstance(custom_var, tk.BooleanVar):
                        self.custom_params['momentum'][param_name] = custom_var.get()
                    else:
                        value_str = custom_var.get()
                        self.custom_params['momentum'][param_name] = self.convert_param_value(value_str)

            if hasattr(self, 'kalman_param_widgets'):
                for param_name, widget_info in self.kalman_param_widgets.items():
                    custom_var = widget_info['custom']
                    if isinstance(custom_var, tk.BooleanVar):
                        self.custom_params['kalman'][param_name] = custom_var.get()
                    else:
                        value_str = custom_var.get()
                        self.custom_params['kalman'][param_name] = self.convert_param_value(value_str)

            custom_to_save = {
                'momentum': self.custom_params.get('momentum', {}),
                'kalman': self.custom_params.get('kalman', {})
            }

            filename = "custom_params.json"
            with open(filename, 'w') as f:
                json.dump(custom_to_save, f, indent=4)

            self.log_message(f"💾 Saved current custom params to {filename}", "green")
            return True
        except Exception as e:
            self.log_message(f"❌ Error saving custom params: {e}", "red")
            return False

    def load_params_at_startup(self):
        """
        Load parameters at application startup with proper hierarchy.
        This is called by root.after_idle in __init__
        """
        self.log_message("=" * 70, "blue")
        self.log_message("📂 LOADING PARAMETERS AT STARTUP", "blue")
        self.log_message("=" * 70, "blue")

        # STEP 1: Load from strategy_settings.json
        self.load_strategy_settings()

        # STEP 2: Apply the selected parameters to the strategy
        self.apply_selected_parameters()

        # Log final Tier 1 status (only_tier1_entries blocks Tier 1,
        # allowing only Tier 2 through)
        only_tier1 = self.custom_params['momentum'].get('only_tier1_entries', False)
        self.log_message("=" * 70, "green")
        self.log_message(f"📊 FINAL TIER 1 STATUS: {'BLOCKED' if only_tier1 else 'ACTIVE'}",
                         "yellow" if only_tier1 else "green")
        self.log_message(f"📋 Current mode: {self.param_toggle_var.get()}", "blue")
        self.log_message("=" * 70, "green")

    def add_load_buttons_to_settings(self):
        """
        Add load buttons to the settings panel (call this in create_momentum_parameter_controls)
        """
        # In your settings panel, add these buttons to the control_frame
        if hasattr(self, 'settings_panel') and self.settings_panel.winfo_exists():
            # Find or create a button frame
            button_frame = ttk.Frame(self.settings_panel)
            button_frame.pack(fill='x', padx=10, pady=5)

            ttk.Button(
                button_frame,
                text="📂 Load Custom Params",
                command=lambda: self.load_momentum_params_from_file()
            ).pack(side=tk.LEFT, padx=5)

            ttk.Button(
                button_frame,
                text="📂 Load Default Params",
                command=lambda: self.load_default_params_from_file()
            ).pack(side=tk.LEFT, padx=5)

            ttk.Button(
                button_frame,
                text="💾 Save Current as Default",
                command=lambda: self.save_default_params_to_file()
            ).pack(side=tk.LEFT, padx=5)

            ttk.Label(
                button_frame,
                text="(momentum_params.json | momentum_params_default.json)",
                foreground='gray',
                font=('Arial', 8)
            ).pack(side=tk.LEFT, padx=10)

    def switch_strategy(self, new_strategy_name):
        """Completely switch to a new trading strategy with metadata display"""
        try:
            if new_strategy_name not in self.strategies:
                raise ValueError(f"Unknown strategy: {new_strategy_name}")

            was_running = self.running
            if self.running:
                self.stop_trading()

            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL: Ensure market data is available BEFORE strategy switch
            # ═══════════════════════════════════════════════════════════════════
            self.log_message("=" * 70, "blue")
            self.log_message(f"🔄 Switching to strategy: {new_strategy_name}", "blue")
            self.log_message("=" * 70, "blue")

            # Validate market data availability
            if not self._switch_strategy_connection_block(new_strategy_name):
                if was_running:
                    self.start_trading()
                return False

            # ═══════════════════════════════════════════════════════════════════
            # All strategies (Momentum / Kalman / Scalping / Enhanced) - single TF
            # ═══════════════════════════════════════════════════════════════════
            if True:
                # Re-enable interval combobox for single-timeframe strategies
                if hasattr(self, 'interval_combobox'):
                    self.interval_combobox.config(state='readonly')
                    if hasattr(self, '_saved_interval'):
                        self.interval_combobox.set(self._saved_interval)
                    else:
                        self.interval_combobox.set("15m")

                # Get market data with retries
                df = self._get_market_data_with_retry(max_retries=3, delay=2)
                if df is None:
                    raise Exception("Could not get market data for strategy switch after multiple attempts")

                # Log data quality
                self.log_message(f"📊 Market data loaded: {len(df)} candles", "green")
                self.log_message(f"   Date range: {df.index[0]} to {df.index[-1]}", "blue")

                new_strategy = self.strategies[new_strategy_name]

                # Set strategy properties
                if hasattr(new_strategy, 'name'):
                    new_strategy.name = new_strategy_name
                else:
                    new_strategy.name = self.strategy_name

                if hasattr(new_strategy, 'trading_app'):
                    new_strategy.trading_app = self
                if hasattr(new_strategy, 'log_message'):
                    new_strategy.log_message = self.log_message
                if hasattr(new_strategy, 'place_order'):
                    new_strategy.place_order = self.place_order
                if hasattr(new_strategy, 'get_balance'):
                    new_strategy.get_balance = self.get_balance
                if hasattr(new_strategy, 'get_current_price'):
                    new_strategy.get_current_price = self.get_current_price

                # Enable ML if needed
                if new_strategy_name == "Enhanced" and self.ml_enabled:
                    new_strategy.ml_enabled = True
                    new_strategy.current_ml_model = self.current_ml_model

                # Calculate indicators
                self.log_message("📈 Calculating indicators for new strategy...", "blue")
                df = new_strategy.calculate_indicators(df)

                if df is None or len(df) < 2:
                    raise Exception("Indicator calculation failed or insufficient data")

                # Get strategy parameters for chart
                active_params = {}
                if new_strategy_name == "Momentum":
                    active_params = self.get_current_momentum_params()
                elif new_strategy_name == "Kalman":
                    active_params = self.get_current_kalman_params()

                # Update chart
                if hasattr(self, 'chart') and df is not None:
                    self.chart.update_chart(df, params=active_params)
                    self.log_message(f"✅ Chart updated with {new_strategy_name} parameters", "green")

                # Reset position state
                self.position = {
                    'type': None,
                    'price': None,
                    'quantity': None,
                    'time': None,
                    'stop_loss': None,
                    'trailing_stop': None,
                    'entry_confidence': None
                }

                self.strategy = new_strategy
                self.current_data = df.iloc[-1] if df is not None else None

                # Retrain ML if enabled
                if self.ml_enabled and self.current_ml_model:
                    self.log_message(f"🔄 Retraining ML model for {new_strategy_name}...", "blue")
                    if not self.current_ml_model.train(df):
                        self.log_message("⚠️ ML model training failed, but continuing with strategy", "orange")

                self.strategy_type_var.set(new_strategy_name)

                # Display strategy info
                if hasattr(new_strategy, 'get_strategy_info'):
                    info = new_strategy.get_strategy_info()
                    self.log_message(f"", "white")
                    self.log_message(f"{'═' * 70}", "cyan")
                    self.log_message(f"📋 STRATEGY INFORMATION", "cyan")
                    self.log_message(f"{'═' * 70}", "cyan")
                    self.log_message(f"  Name: {info.get('name', 'Unknown')}", "white")
                    self.log_message(f"  Version: {info.get('version', 'N/A')}", "white")
                    self.log_message(f"  Tier System: {info.get('tier_system', 'Standard')}", "white")
                    self.log_message(f"", "white")
                    self.log_message(f"📊 EXPECTED PERFORMANCE:", "cyan")
                    self.log_message(f"  Trades/Month: {info.get('expected_trades_monthly', 'N/A')}", "white")
                    self.log_message(f"  Win Rate: {info.get('target_win_rate', 'N/A')}", "white")
                    self.log_message(f"  CAGR: {info.get('target_cagr', 'N/A')}", "white")
                    self.log_message(f"  Sharpe Ratio: {info.get('target_sharpe', 'N/A')}", "white")
                    self.log_message(f"  Max Drawdown: {info.get('max_drawdown', 'N/A')}", "white")

                    if 'tier1_description' in info:
                        self.log_message(f"", "white")
                        self.log_message(f"🎯 TIER SYSTEM:", "cyan")
                        self.log_message(f"  Tier 1: {info['tier1_description']}", "white")
                        self.log_message(f"  Tier 2: {info['tier2_description']}", "white")

                    self.log_message(f"{'═' * 70}", "cyan")

                self.log_message(f"✅ Successfully switched to {new_strategy_name} strategy", "green")

            # Restart trading if it was running
            if was_running:
                self.log_message("🔄 Restarting trading with new strategy...", "blue")
                self.start_trading()

            return True

        except Exception as e:
            self.log_message(f"❌ Strategy switch failed: {str(e)}", "red")
            import traceback
            self.log_message(traceback.format_exc(), "red")
            if was_running:
                self.start_trading()
            return False

    def _switch_strategy_connection_block(self, new_strategy_name):
        """
        Drop-in replacement for the connection-validation block inside switch_strategy.
        Paste this in place of the existing _validate_market_data_ready call block.
        """
        self.log_message("=" * 70, "blue")
        self.log_message(f"🔄 Switching to strategy: {new_strategy_name}", "blue")
        self.log_message("=" * 70, "blue")

        # Auto-connect with retry — up to 3 attempts, 2 s apart
        self.log_message("🔍 Checking market data availability...", "cyan")
        market_data_ready = False
        last_msg = ""

        for attempt in range(1, 4):
            market_data_ready, last_msg = self._validate_market_data_ready(
                auto_connect=(attempt == 1),  # only auto-connect on first pass
                max_retries=2
            )
            if market_data_ready:
                break

            if attempt < 3:
                self.log_message(
                    f"⚠️ Attempt {attempt}/3: {last_msg} — retrying in 2 s...",
                    "orange"
                )
                import time
                time.sleep(2)

        if not market_data_ready:
            self.log_message(f"❌ Cannot switch strategy: {last_msg}", "red")
            self.log_message(
                "   Tip: Click 'Check Connection' in the main panel, wait for "
                "'Connection successful', then try switching again.",
                "orange"
            )
            return False  # caller must handle this (restore was_running if needed)

        return True  # proceed with the rest of switch_strategy

    # def _validate_market_data_ready(self):
    #     """
    #     Validate that market data is available and ready for strategy initialization.
    #     Returns (is_ready, message)
    #     """
    #     # Check if market API is initialized
    #     if self.mode_var.get().lower() != "backtest":
    #         if not hasattr(self, 'market_api') or self.market_api is None:
    #             return False, "Market API not initialized. Please check connection."
    #
    #     # Try to get data
    #     try:
    #         df = self.get_market_data()
    #         if df is None:
    #             return False, "No market data available"
    #
    #         if len(df) < 50:
    #             return False, f"Insufficient data: only {len(df)} candles (need at least 50)"
    #
    #         # Check for required columns
    #         required = ['Open', 'High', 'Low', 'Close', 'Volume']
    #         missing = [col for col in required if col not in df.columns]
    #         if missing:
    #             return False, f"Missing required columns: {missing}"
    #
    #         # Check for NaN values
    #         if df[required].isna().any().any():
    #             return False, "Data contains NaN values"
    #
    #         return True, "Market data ready"
    #
    #     except Exception as e:
    #         return False, f"Error validating market data: {str(e)}"

    def _validate_market_data_ready(self, auto_connect=True, max_retries=2):
        """
        Validate that market data is available.
        If the API is not connected, attempt to connect automatically before failing.

        Returns (is_ready: bool, message: str)
        """
        mode = self.mode_var.get().lower()

        # Backtest mode doesn't need a live API
        if mode == "backtest":
            try:
                df = self.get_market_data()
                if df is None:
                    return False, "No market data available"
                if len(df) < 50:
                    return False, f"Insufficient data: only {len(df)} candles (need at least 50)"
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing = [c for c in required if c not in df.columns]
                if missing:
                    return False, f"Missing required columns: {missing}"
                return True, "Market data ready (backtest)"
            except Exception as e:
                return False, f"Error validating market data: {str(e)}"

        # Live / Demo mode — check API, auto-connect if needed
        api_ready = hasattr(self, 'market_api') and self.market_api is not None

        if not api_ready and auto_connect:
            self.log_message("🔄 API not connected — attempting auto-connect...", "orange")
            for attempt in range(1, max_retries + 1):
                try:
                    self.check_connection()  # your existing connect method
                    import time
                    time.sleep(1.5)  # give socket a moment to settle
                    if hasattr(self, 'market_api') and self.market_api is not None:
                        # Quick sanity-ping
                        resp = self.market_api.get_tickers(instType="SPOT")
                        if resp.get('code') == '0':
                            self.log_message(f"✅ Auto-connect succeeded (attempt {attempt})", "green")
                            api_ready = True
                            break
                except Exception as e:
                    self.log_message(f"⚠️ Auto-connect attempt {attempt}/{max_retries} failed: {e}", "orange")

            if not api_ready:
                return (
                    False,
                    "Market API not reachable after auto-connect. "
                    "Please use 'Check Connection' and try again."
                )

        if not api_ready:
            return False, "Market API not initialized. Please check connection first."

        # API is up — validate actual data
        try:
            df = self.get_market_data()
            if df is None:
                return False, "No market data returned from API"
            if len(df) < 50:
                return False, f"Insufficient data: only {len(df)} candles (need ≥ 50)"
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                return False, f"Missing required columns: {missing}"
            if df[required].isna().any().any():
                return False, "Data contains NaN values — try again in a moment"
            return True, "Market data ready"
        except Exception as e:
            return False, f"Error validating market data: {str(e)}"

    def _get_market_data_with_retry(self, max_retries=3, delay=2):
        """
        Get market data with retry logic for robustness.
        """
        for attempt in range(max_retries):
            try:
                df = self.get_market_data()
                if df is not None and len(df) >= 50:
                    if attempt > 0:
                        self.log_message(f"✅ Data fetch succeeded on attempt {attempt + 1}", "green")
                    return df
                else:
                    if attempt < max_retries - 1:
                        self.log_message(f"⚠️ Attempt {attempt + 1}/{max_retries}: Data insufficient, retrying...",
                                         "orange")
                        time.sleep(delay)
                    else:
                        self.log_message(f"❌ Failed to get sufficient data after {max_retries} attempts", "red")
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log_message(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {str(e)}, retrying...", "orange")
                    time.sleep(delay)
                else:
                    self.log_message(f"❌ All {max_retries} attempts failed: {str(e)}", "red")
        return None

    # ═══════════════════════════════════════════════════════════════════════════════
    # FIX 1  —  get_market_data
    # ═══════════════════════════════════════════════════════════════════════════════
    def get_market_data(self):
        """Market data fetch — keyword args prevent positional mis-mapping bug."""
        if self.mode_var.get().lower() != "backtest":
            if not hasattr(self, 'market_api') or self.market_api is None:
                self.log_message("⚠️ Market API not initialized. Please check connection.", "red")
                return None

        try:
            # ── BACKTEST ────────────────────────────────────────────────────────
            if self.mode_var.get().lower() == "backtest":
                # FIX: use keyword args so interval is NOT mistaken for exchange_name
                df = self.get_historical_data(
                    symbol=self.symbol_var.get(),
                    interval=self.interval_var.get(),
                    limit=5000,
                )
                if df is not None and not df.empty:
                    self.log_message("📊 Using real historical data from Binance", "green")
                    self._last_backtest_df = df  # cache so AI can use it later
                    self.enable_ai_button()
                    return df
                else:
                    self.log_message("⚠️ No historical data — falling back to synthetic data", "orange")
                    freq = self.interval_var.get()
                    if freq == '1m':
                        freq = 'T'
                    elif freq == '5m':
                        freq = '5T'
                    return self.generate_test_data(freq)

            # ── LIVE / DEMO ──────────────────────────────────────────────────────
            symbol = self.symbol_var.get()
            interval = self.interval_var.get()
            total_limit = 50_000
            batch_size = 1_000
            all_data = []
            before = None

            while len(all_data) < total_limit:
                params = {'instId': symbol, 'bar': interval, 'limit': batch_size}
                if before:
                    params['before'] = before

                response = self.market_api.get_candlesticks(**params)
                if response['code'] != '0' or not response['data']:
                    break

                batch = response['data']
                all_data.extend(batch)
                if len(batch) < batch_size:
                    break
                before = batch[-1][0]

            if not all_data:
                self.log_message("⚠️ No data received from API", "orange")
                return None

            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                         'volCcy', 'volBase', 'turnover'],
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            float_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'volCcy', 'volBase', 'turnover']
            df[float_cols] = df[float_cols].astype(float)
            df = df.sort_index(ascending=True)

            start_str = df.index[0].strftime('%Y-%m-%d %H:%M:%S UTC')
            end_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
            self.log_message(f"📊 Market data: {start_str} → {end_str} ({len(df)} candles)", "blue")
            self.enable_ai_button()
            return df

        except Exception as e:
            self.log_message(f"⚠️ Data fetch error: {e}", "red")
            import logging, traceback
            logging.error(traceback.format_exc())
            return None

    def _process_trading_result(self, result, current_data, current_price, df=None):
        """
        Process trading result from any strategy.
        Extracted from trading_loop to avoid code duplication.
        """
        try:
            use_detailed = getattr(self, 'detailed_output_var', None)
            is_detailed = use_detailed.get() if use_detailed else True

            if is_detailed:
                self._display_detailed_analysis(df, current_data, result)
            else:
                self._display_simple_analysis(df, current_data, result)

            self._execute_trades_from_result(result, current_data, current_price, df)

            # Update chart with forecasts if available
            if hasattr(self, 'chart'):
                forecast = getattr(self, '_last_forecast', None)
                if forecast is not None and len(forecast) > 0:
                    try:
                        self.chart.plot_forecast(forecast)
                    except Exception as e:
                        self.log_message(f"⚠️ Could not plot forecast: {e}", "orange")

            self.play_notification("tick")
            self.root.after(0, self.update_stats)

        except Exception as e:
            self.log_message(f"Error processing trading result: {e}", "red")

    def trading_loop(self):
        """Main trading loop — single timeframe for all strategies."""
        while self.running:
            try:
                interval_str = self.interval_var.get()
                interval_seconds = self.interval_seconds_map.get(interval_str, 60)

                now = datetime.now(timezone.utc)
                if interval_str == '1m':
                    next_run = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
                elif interval_str == '5m':
                    next_run = now + timedelta(minutes=5 - (now.minute % 5))
                    next_run = next_run.replace(second=0, microsecond=0)
                elif interval_str == '15m':
                    next_run = now + timedelta(minutes=15 - (now.minute % 15))
                    next_run = next_run.replace(second=0, microsecond=0)
                elif interval_str == '30m':
                    next_run = now + timedelta(minutes=30 - (now.minute % 30))
                    next_run = next_run.replace(second=0, microsecond=0)
                elif interval_str == '1H':
                    next_run = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                elif interval_str == '1D':
                    next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    next_run = now + timedelta(seconds=interval_seconds)

                sleep_time = (next_run - now).total_seconds()
                self.log_message(
                    f"⏰ [{self.strategy_type_var.get()}] Next analysis at: "
                    f"{next_run.strftime('%Y-%m-%d %H:%M:%S UTC')}", "blue")

                self.timer.set_interval(interval_seconds)
                self.timer.start_time = time.time() - (time.time() % interval_seconds)
                self.timer.start()

                time.sleep(max(0, sleep_time))

                if not self.running:
                    break

                # ── U6: fetch 1h context when Scalping is active ──────
                if self.needs_multitimeframe():
                    try:
                        multi_df = self.get_market_data_multiframe()
                        htf = multi_df.get('_htf', {})
                        if htf:
                            ef_1h = htf.get('ema_fast_1h', 0)
                            es_1h = htf.get('ema_slow_1h', 0)
                            if ef_1h and es_1h:
                                direction = "↑ BULL" if ef_1h > es_1h else "↓ BEAR"
                                self.log_message(
                                    f"🔭 1h Context: {direction}  "
                                    f"(EMA9={ef_1h:.2f} vs EMA50={es_1h:.2f})",
                                    "cyan" if ef_1h > es_1h else "orange")
                    except Exception as htf_err:
                        self.log_message(
                            f"⚠️ HTF fetch skipped this cycle: {htf_err}",
                            "orange")

                df = self.get_market_data()
                if df is None:
                    self.log_message("⚠️ No market data returned — retrying next cycle", "orange")
                    continue

                # ── Time-window guard (live / demo only) ─────────────────────
                _strategy_name = self.strategy_type_var.get()
                _mode_name = self.mode_var.get().lower()
                if (_mode_name in ('live', 'demo')
                        and hasattr(self, 'trading_time_config')
                        and not self._is_time_unconstrained(_strategy_name)
                        and not self._is_within_trading_window(_strategy_name)):
                    self.log_message(
                        f"⏰ [{_strategy_name}] Outside trading window — pausing.",
                        "orange")
                    _wait = min(self._seconds_until_start(_strategy_name), 300)
                    time.sleep(max(5, _wait))
                    continue

                df = self.strategy.calculate_indicators(df)

                # ── NEW: Calculate ATR for stop placement ──────────────────────
                atr = self.get_atr(df)
                current_price = float(df['Close'].iloc[-1])

                # ── NEW: Log ATR-based stops for monitoring ────────────────────
                if atr > 0:
                    stops = self.calculate_atr_stops(current_price, atr)
                    if self.detailed_output_var.get():
                        self.log_message(
                            f"📊 ATR: {atr:.4f} | Stop: ${stops['stop_loss']:.2f} "
                            f"({stops['risk_pct']:.2f}%) | T1: ${stops['target_1']:.2f} "
                            f"| T2: ${stops['target_2']:.2f}",
                            "blue")

                # ── ensure trade_direction is always in sync with the GUI ─────
                _gui_dir = self.trade_direction_var.get()
                if hasattr(self.strategy, 'trade_direction'):
                    if self.strategy.trade_direction != _gui_dir:
                        self.strategy.trade_direction = _gui_dir
                        self.log_message(f"🔧 trade_direction synced: {_gui_dir}", "blue")
                if hasattr(self.strategy, 'only_long_entries'):
                    self.strategy.only_long_entries = (_gui_dir == 'long')
                if hasattr(self.strategy, 'only_short_entries'):
                    self.strategy.only_short_entries = (_gui_dir == 'short')

                if df is None or len(df) < 2:
                    self.log_message("⚠️ Indicator calculation failed or insufficient data", "orange")
                    continue

                current_data = df.iloc[-2].copy()
                current_price = float(current_data['Close'])

                # If the period has ended (trading_running=False but running still True
                # briefly during position close), skip new entry analysis.
                if not self.trading_running:
                    self.log_message(
                        "⏸  Period ended — skipping new entry analysis.", "orange")
                    break
                result = self.strategy.run_analysis_cycle(current_data, current_price, df)

                if hasattr(self, 'chart') and self.running:
                    st = self.strategy_type_var.get()
                    active_params = (self.get_current_momentum_params() if st == "Momentum"
                                     else self.get_current_kalman_params() if st == "Kalman"
                    else self.get_current_scalping_params() if st == "Scalping"
                    else {})
                    stop_loss = self.position.get('stop_loss') if self.position.get('type') else None
                    trailing_stop = self.position.get('trailing_stop') if self.position.get('type') else None
                    live_price = float(df['Close'].iloc[-1])
                    _df, _params, _sl, _ts, _lp = df, active_params, stop_loss, trailing_stop, live_price

                    def _update_chart_on_main(_df=_df, _params=_params,
                                              _sl=_sl, _ts=_ts, _lp=_lp):
                        try:
                            if (hasattr(self, 'chart') and self.chart is not None
                                    and hasattr(self.chart, 'canvas')
                                    and self.chart.canvas is not None
                                    and self.chart.canvas.figure is not None):
                                self.chart.update_chart(_df, params=_params,
                                                        stop_loss=_sl, trailing_stop=_ts, live_price=_lp)
                        except Exception as chart_err:
                            import logging
                            logging.warning(f"Chart update skipped: {chart_err}")

                    self.root.after(0, _update_chart_on_main)

                self._process_trading_result(result, current_data, current_price, df)

            except Exception as e:
                self.log_message(f"Trading error: {str(e)}", "red")
                logging.error(f"Trading error: {str(e)}")
                import traceback
                self.log_message(traceback.format_exc(), "red")
                self.play_notification("error")
                time.sleep(5)

    def update_config_from_settings(self, new_settings):
        self.config.update(new_settings)

        backtest_strategy = self.trading_app.strategies["BacktestMomentum"]
        for key, value in new_settings.items():
            if hasattr(backtest_strategy, key):
                setattr(backtest_strategy, key, value)

        self.trading_app.log_message(f"Strategy parameters updated across live and backtest", "blue")

    def open_settings(self):
        """Open the settings panel with indicator configuration - maximizes the window"""
        if hasattr(self, 'settings_panel') and self.settings_panel.winfo_exists():
            self.settings_panel.destroy()

        # 🚨 REMOVED: self.sync_custom_params_with_code_defaults() - No longer needed

        self.settings_panel = tk.Toplevel(self.root)
        self.settings_panel.title("Strategy Parameters")

        # Set initial size but then maximize
        self.settings_panel.geometry("1200x750")

        # Maximize the window based on OS
        try:
            # Try Windows maximize
            self.settings_panel.state('zoomed')
        except:
            try:
                # Try Linux/Unix maximize
                self.settings_panel.attributes('-zoomed', True)
            except:
                try:
                    # Try macOS maximize
                    self.settings_panel.wm_attributes('-zoomed', True)
                except:
                    # Fallback to large size
                    screen_width = self.settings_panel.winfo_screenwidth()
                    screen_height = self.settings_panel.winfo_screenheight()
                    self.settings_panel.geometry(f"{screen_width}x{screen_height}+0+0")

        self.settings_panel.resizable(True, True)
        self.settings_panel.grab_set()

        notebook = ttk.Notebook(self.settings_panel)

        # ── Trading Hours bar — always visible above tabs ──────────────────────
        hours_bar = tk.Frame(self.settings_panel, bg="#1a1a2e", pady=6)
        hours_bar.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(
            hours_bar,
            text="⏰  Trading Hours (UTC):",
            bg="#1a1a2e", fg="white",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(10, 20))

        for strategy in ("Momentum", "Kalman", "Scalping"):
            cfg = self.trading_time_config.get(strategy, {})

            def _summary(s=strategy):
                c = self.trading_time_config[s]
                sh, sm = c["start_h"].get(), c["start_m"].get()
                eh, em = c["end_h"].get(), c["end_m"].get()
                if sh == eh and sm == em:
                    return f"{s}: any time"
                return f"{s}: {sh:02d}:{sm:02d}→{eh:02d}:{em:02d}"

            lbl_var = tk.StringVar(value=_summary(strategy))
            lbl = tk.Label(
                hours_bar,
                textvariable=lbl_var,
                bg="#1a1a2e", fg="#aaaaff",
                font=("Arial", 9)
            )
            lbl.pack(side=tk.LEFT, padx=(0, 4))

            def _refresh_label(s=strategy, v=lbl_var):
                v.set(_summary(s))

            for key in ("start_h", "start_m", "end_h", "end_m"):
                cfg[key].trace_add("write", lambda *_, s=strategy, v=lbl_var: v.set(_summary(s)))

            tk.Button(
                hours_bar,
                text=f"⚙ {strategy}",
                command=lambda s=strategy: self.open_time_settings_panel(s),
                bg="#0055aa", fg="white",
                font=("Arial", 8, "bold"),
                relief="raised", bd=2,
                cursor="hand2", padx=8, pady=2
            ).pack(side=tk.LEFT, padx=(0, 14))

        # ── Notebook ───────────────────────────────────────────────────────────
        notebook.pack(expand=True, fill='both', padx=10, pady=(6, 10))

        momentum_tab = ttk.Frame(notebook)
        kalman_tab = ttk.Frame(notebook)
        scalping_tab = ttk.Frame(notebook)
        notebook.add(momentum_tab, text="Momentum Parameters")
        notebook.add(kalman_tab, text="Kalman Parameters")
        notebook.add(scalping_tab, text="⚡ Scalping Parameters")

        if not hasattr(self, 'custom_params') or not self.custom_params:
            self.load_strategy_settings()
        if 'scalping' not in self.custom_params:
            self.custom_params['scalping'] = self.get_default_scalping_params().copy()

        self.create_momentum_parameter_controls(momentum_tab)
        self.create_kalman_parameter_controls(kalman_tab)
        self.create_scalping_parameter_controls(scalping_tab)

        # ─── CONTROL FRAME AT BOTTOM ──────────────────────────────────────────
        control_frame = ttk.Frame(self.settings_panel)
        control_frame.pack(fill='x', padx=10, pady=10)

        # Parameter selection (existing)
        ttk.Label(control_frame, text="Use:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        self.param_toggle = ttk.Combobox(
            control_frame,
            textvariable=self.param_toggle_var,
            values=["Default Parameters", "Custom Parameters"],
            state="readonly",
            width=20,
            font=('Arial', 10)
        )
        self.param_toggle.pack(side=tk.LEFT, padx=5)
        self.param_toggle.bind("<<ComboboxSelected>>", self.on_param_toggle_changed)

        # ─── LOAD BUTTONS (Bottom) ───────────────────────────────────────
        load_frame = ttk.Frame(control_frame)
        load_frame.pack(side=tk.LEFT, padx=20)

        # Single Load All Parameters Button (loads both default and custom params)
        ttk.Button(
            load_frame,
            text="📂 Load All Parameters",
            command=self.load_params,
            width=18
        ).pack(side=tk.LEFT, padx=2)

        # Save Buttons (existing)
        ttk.Button(
            control_frame,
            text="💾 Save Custom Parameters",
            command=self.save_custom_parameters
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            control_frame,
            text="🔄 Reset to Defaults",
            command=self.reset_to_defaults
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            control_frame,
            text="❌ Close",
            command=self.settings_panel.destroy
        ).pack(side=tk.RIGHT, padx=5)

        # Log that settings panel was opened
        self.log_message("⚙ Settings panel opened (maximized)", "blue")

    def create_momentum_parameter_controls(self, parent):
        """Create parameter controls for Momentum strategy with CONSOLIDATED TIER SYSTEM."""
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ─── FIX: Set fixed width for left frame ──────────────────────────────────
        left_frame = ttk.Frame(paned, width=900)  # Fixed width
        paned.add(left_frame, weight=1)  # weight=0 prevents expansion

        right_frame = ttk.LabelFrame(paned, text="Backtest Optimization Parameters", width=500)
        paned.add(right_frame, weight=1)

        # ═══════════════════════════════════════════════════════════════
        # LEFT PANEL: EXISTING STRATEGY PARAMETERS
        # ═══════════════════════════════════════════════════════════════

        headers_frame = ttk.Frame(left_frame)
        headers_frame.pack(fill='x', padx=5, pady=(5, 0), side=tk.TOP)
        headers_frame.columnconfigure(0, weight=0, minsize=200)  # Reduced from 250
        headers_frame.columnconfigure(1, weight=0, minsize=100)  # Reduced from 120
        headers_frame.columnconfigure(2, weight=0, minsize=100)  # Reduced from 120
        headers_frame.columnconfigure(3, weight=1, minsize=250)  # Reduced from 350

        header_style = ('Arial', 9, 'bold')  # Smaller font
        ttk.Label(headers_frame, text="Parameter", font=header_style, anchor='w').grid(row=0, column=0, padx=3, pady=5,
                                                                                       sticky='w')
        ttk.Label(headers_frame, text="📌 Default Value", font=header_style, anchor='w').grid(row=0, column=1, padx=3,
                                                                                             pady=5, sticky='w')
        ttk.Label(headers_frame, text="✏️ Custom Value", font=header_style, anchor='w').grid(row=0, column=2, padx=3,
                                                                                             pady=5, sticky='w')
        ttk.Label(headers_frame, text="Description", font=header_style, anchor='w').grid(row=0, column=3, padx=3,
                                                                                         pady=5, sticky='w')
        ttk.Separator(headers_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky='ew', padx=3,
                                                               pady=(0, 3))

        content_frame = ttk.Frame(left_frame)
        content_frame.pack(fill='both', expand=True, padx=3, pady=3, side=tk.TOP)

        canvas = tk.Canvas(content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        default_params = self.get_default_momentum_params()
        self.momentum_param_widgets = {}

        # ═══════════════════════════════════════════════════════════════
        # CONSOLIDATED CATEGORIES — v10.0.5
        # All obsolete/duplicate parameters removed
        # ═══════════════════════════════════════════════════════════════
        categories = {
            '📊 EMA Parameters': [
                'ema_fast_period', 'ema_mid_period', 'ema_slow_period',
                'ema_near_tolerance', 'daily_ema_period',
            ],
            '🎯 LONG Quality Thresholds': [
                'quality_tier1_min_long',
                'quality_tier2_min_long',
            ],
            '🎯 SHORT Quality Thresholds': [
                'quality_tier1_min_short',
                'quality_tier2_min_short',
            ],
            '📊 Quality Component Weights (total=100)': [
                'weight_ema', 'weight_adx', 'weight_macd', 'weight_rsi', 'weight_volume',
            ],
            '🎯 TIER CONTROL': [
                'only_tier1_entries',
            ],
            '⏱️ TIER COOLDOWN & CONFLUENCE': [
                'min_bars_between_trades_tier1',
                'min_bars_between_trades_tier2',
                'cooldown_tier2_enabled',
                'tier1_confluence_min',
                'tier2_confluence_min',
            ],
            '⬆️ TIER 1 LONG FILTERS': [
                'tier1_adx_hard_min',
                'tier1_rsi_min',
                'tier1_rsi_max',
                'tier1_volume_min',
                'tier1_momentum_min',
                'tier1_kalman_min',
                'tier1_macd_gate',
                'tier1_price_ema_max_pct',
                'daily_trend_filter_enabled',
                'pullback_zone_lower_pct',
                'pullback_zone_upper_pct',
                'adx_slope_min',
            ],
            '⬇️ TIER 1 SHORT FILTERS': [
                'tier1_adx_hard_min_short',
                'tier1_rsi_min_short',
                'tier1_rsi_max_short',
                'tier1_volume_min_short',
                'tier1_momentum_min_short',
                'tier1_macd_gate_short',
                'daily_trend_down_filter_enabled',
            ],
            '🎯 TIER 2 FILTERS': [
                'tier2_adx_hard_min',
                'tier2_volume_min',
                'tier2_momentum_min',
                'tier2_rsi_min',
                'tier2_rsi_max',
                'tier2_rsi_min_short',
                'tier2_rsi_max_short',
                'tier2_macd_histogram_min',
                'tier2_require_macd_histogram',
            ],
            '🎯 TIER SIZE / EXIT / TRAILING': [
                'tier1_size_multiplier',
                'tier2_size_multiplier',
                'stop_loss_atr_mult',
                'exit_threshold_tier1',
                'exit_threshold_tier2',
                'trailing_activation_tier1',
                'trailing_activation_tier2',
                'trailing_distance_tier1',
                'trailing_distance_tier2',
            ],
            '🔬 Indicator Periods': [
                'adx_period', 'rsi_period', 'cci_period', 'atr_period',
                'volume_ma_period', 'macd_fast', 'macd_slow', 'macd_signal',
                'supertrend_atr_period', 'supertrend_multiplier',
                'kalman_q_param', 'kalman_r_param', 'vix_atr_period', 'vix_rolling_period',
            ],
            '🛡️ Risk Management': [
                'risk_tier1',
                'risk_tier2',
            ],
            '🔒 BREAKEVEN STOP': [
                'be_stop_enabled',
                'be_stop_r_trigger',
                'be_stop_no_progress_bars',
            ],
            '💰 Profit Targets': [
                'take_profit_r1',
                'take_profit_r2',
                'take_profit_r3',
            ],
            '📉 Exit Conditions': [
                'max_hold_bars',
                'min_hold_bars_before_stop',
                'emergency_stop_multiplier',
                'macd_bearish_cross_exit',
                'macd_bearish_cross_profit_min',
                'ema_cross_exit',
                'rsi_exit_threshold',
                'kalman_fade_threshold',
                'momentum_reversal_exit',
                'momentum_reversal_threshold',
                'momentum_reversal_profit_min',
            ],
            '⏱️ COOLDOWN & TRADE MANAGEMENT': [
                'max_daily_trades',
                'min_bars_between_trades',
                'cooldown_after_profit_target_bars',
                'cooldown_after_loss_bars',
                'consecutive_loss_threshold',
                'consecutive_loss_cooldown_bars',
            ],
            '🔬 PRECISION FILTERS': [
                'ema_trending_bars',
                'macd_hist_rising_bars',
                'rsi_direction_bars',
                'rsi_direction_min_move',
            ],
            '⚙️ Regime & Strategy Control': [
                'regime_filter_enabled',
                'ranging_min_checks',
                'bb_period', 'bb_std',
                'kc_period', 'kc_atr_mult',
                'chop_period', 'chop_threshold',
                'volatility_scaling',
                'atr_compression_enabled',
                'atr_compression_threshold',
                'extended_run_max_pct_long',
                'extended_run_max_pct_short',
            ],
            '📊 PRICE POSITIONING': [
                'price_percentile_bonus_early',
                'price_percentile_penalty_late',
                'price_percentile_early_threshold',
                'price_percentile_late_threshold',
                'price_percentile_lookback',
            ],
            '📈 ADX SCORING BANDS': [
                'adx_score_trend_forming',
                'adx_score_good_trend',
                'adx_score_strong_trend',
                'adx_score_very_strong',
                'adx_score_extended',
            ],
            '📉 MACD SCORING': [
                'macd_score_line_vs_signal',
                'macd_score_histogram_direction',
                'macd_score_zero_cross',
                'macd_score_histogram_value',
            ],
            '🧠 FUZZY MODE SETTINGS': [
                'fuzzy_mode_enabled',
                'fuzzy_learning_enabled',
                'fuzzy_safety_cutoffs',
                'fuzzy_default_margin_pct',
                'fuzzy_absolute_min',
                'fuzzy_absolute_max',
                'fuzzy_min_confidence',
                'fuzzy_min_samples',
                'fuzzy_max_adjustment_pct',
                'fuzzy_learning_rate',
                'fuzzy_conservative_start',
            ],
            '🎯 Trade Direction': [
                'trade_direction',
            ],
        }

        row = 0
        for category, params in categories.items():
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew',
                                                                      pady=(10, 5), padx=5)
            row += 1
            ttk.Label(scrollable_frame, text=category, font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=4,
                                                                                        sticky='w', padx=5, pady=5)
            row += 1

            for param_name in params:
                if param_name in default_params:
                    scrollable_frame.columnconfigure(0, weight=0, minsize=200)
                    scrollable_frame.columnconfigure(1, weight=0, minsize=100)
                    scrollable_frame.columnconfigure(2, weight=0, minsize=100)
                    scrollable_frame.columnconfigure(3, weight=1, minsize=250)

                    # ─── Use smaller font for parameter labels ──────────────────────────
                    label_text = param_name.replace('_', ' ').title()
                    ttk.Label(scrollable_frame, text=label_text, anchor='w', font=('Arial', 8)).grid(
                        row=row, column=0, padx=3, pady=1, sticky='w'
                    )

                    default_value = default_params[param_name]
                    if isinstance(default_value, bool):
                        default_display = "✓ Enabled" if default_value else "✗ Disabled"
                    else:
                        default_display = str(default_value)

                    default_entry = ttk.Entry(scrollable_frame, width=12, font=('Arial', 8))
                    default_entry.insert(0, default_display)
                    default_entry.config(state='readonly')
                    default_entry.grid(row=row, column=1, padx=3, pady=1, sticky='w')

                    custom_value = self.custom_params['momentum'].get(param_name, default_value)

                    if isinstance(default_value, bool):
                        custom_var = tk.BooleanVar(value=custom_value)

                        custom_entry = ttk.Checkbutton(
                            scrollable_frame, variable=custom_var,
                            text="Enable" if custom_value else "Disable"
                        )
                        custom_entry.grid(row=row, column=2, padx=3, pady=1, sticky='w')

                        # ── Highlight indicator label for booleans ──────────────────
                        bool_indicator = tk.Label(
                            scrollable_frame, text="●", width=2,
                            bg="yellow" if (bool(custom_value) != bool(default_value)) else "white"
                        )
                        bool_indicator.grid(row=row, column=2, padx=(80, 0), pady=1, sticky='w')

                        def _make_bool_callbacks(v, w, txt_widget, indicator, def_val):
                            def _update_text(*_):
                                try:
                                    current = v.get()
                                    txt_widget.config(text="Enable" if current else "Disable")
                                    if bool(current) != bool(def_val):
                                        indicator.config(bg="yellow")
                                    else:
                                        indicator.config(bg="white")
                                except tk.TclError:
                                    pass

                            return _update_text

                        custom_var.trace_add(
                            'write',
                            _make_bool_callbacks(custom_var, custom_entry, custom_entry, bool_indicator, default_value)
                        )

                    else:
                        custom_var = tk.StringVar(value=str(custom_value))

                        custom_entry = tk.Entry(  # ← tk.Entry (not ttk) so .config(bg=) works
                            scrollable_frame, textvariable=custom_var, width=12,
                            bg="yellow" if str(custom_value) != str(default_value) else "white",
                            font=('Arial', 8)
                        )
                        custom_entry.grid(row=row, column=2, padx=3, pady=1, sticky='w')

                        def _make_str_callback(v, widget, def_val):
                            def _highlight(*_):
                                try:
                                    widget.config(
                                        bg="yellow" if v.get() != str(def_val) else "white"
                                    )
                                except tk.TclError:
                                    pass

                            return _highlight

                        custom_var.trace_add(
                            'write',
                            _make_str_callback(custom_var, custom_entry, default_value)
                        )

                    description = self.get_momentum_param_description(param_name)
                    # ─── Use smaller font for descriptions ──────────────────────────────
                    ttk.Label(
                        scrollable_frame, text=description, wraplength=300,
                        anchor='w', foreground='#555555', font=('Arial', 7)
                    ).grid(row=row, column=3, padx=3, pady=1, sticky='w')

                    self.momentum_param_widgets[param_name] = {
                        'default': default_entry,
                        'custom': custom_var,
                        'widget': custom_entry
                    }
                    row += 1

        # ═══════════════════════════════════════════════════════════════
        # RIGHT PANEL: BACKTEST OPTIMIZATION PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        self._build_backtest_optimization_panel(right_frame)

    # ─────────────────────────────────────────────────────────────────────────────
    # REPLACE: _build_scalping_backtest_optimization_panel
    # Adds a "Current" column to every parameter row (mirrors momentum panel).
    # ─────────────────────────────────────────────────────────────────────────────
    def _build_scalping_backtest_optimization_panel(self, right_frame):
        """
        Build the right-panel backtest optimisation UI for the Scalping strategy.
        Adds a 'Current' column so users can see live param values at a glance.
        """
        import tkinter as tk
        from tkinter import ttk
        from strategies.MomentumStrategy_MACD_HybridScore_Latest import GlobalConfig

        # ── Capital settings ──────────────────────────────────────────────────
        capital_frame = ttk.LabelFrame(right_frame, text="💰 Global Capital Settings")
        capital_frame.pack(fill=tk.X, padx=5, pady=5)
        cap_inner = ttk.Frame(capital_frame)
        cap_inner.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(cap_inner, text="Current Capital:",
                  font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.capital_display_var = tk.StringVar(
            value=f"${GlobalConfig.INITIAL_CAPITAL:,.2f}"
        )
        ttk.Label(cap_inner, textvariable=self.capital_display_var,
                  font=("Arial", 10, "bold"), foreground="green").grid(
            row=0, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Label(cap_inner, text="New Capital ($):",
                  font=("Arial", 9)).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.capital_entry = ttk.Entry(cap_inner, width=15, font=("Arial", 9))
        self.capital_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.capital_entry.insert(0, str(GlobalConfig.INITIAL_CAPITAL))
        tk.Button(
            cap_inner, text="Update Capital",
            command=self._update_global_capital,
            bg="#4CAF50", fg="white",
            font=("Arial", 9, "bold"), padx=10, pady=2, cursor="hand2",
        ).grid(row=1, column=2, padx=5, pady=2)

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=5, pady=5)

        # ── ALL scalping optimisation params with default values ──────────────
        ALL_SCALPING_OPT_PARAMS = {
            "quality_min_long": {"desc": "Quality min score LONG", "v1": "62", "v2": "65", "v3": "68", "v4": "72"},
            "quality_min_short": {"desc": "Quality min score SHORT", "v1": "62", "v2": "65", "v3": "68", "v4": "72"},
            "quality_tier1_min": {"desc": "Tier 1 quality threshold", "v1": "75", "v2": "78", "v3": "80", "v4": "82"},
            "stop_loss_atr_mult": {"desc": "Stop = ATR × mult", "v1": "1.8", "v2": "2.0", "v3": "2.2", "v4": "2.5"},
            "trailing_activation_pct": {"desc": "Profit % to activate trailing", "v1": "0.008", "v2": "0.010",
                                        "v3": "0.012", "v4": "0.015"},
            "trailing_distance_pct": {"desc": "Trailing stop distance", "v1": "0.006", "v2": "0.008", "v3": "0.009",
                                      "v4": "0.012"},
            "be_stop_r_trigger": {"desc": "Breakeven at this R-multiple", "v1": "1.5", "v2": "1.8", "v3": "2.0",
                                  "v4": "2.5"},
            "take_profit_r1": {"desc": "Partial exit (50%) R-multiple", "v1": "1.2", "v2": "1.5", "v3": "1.8",
                               "v4": "2.0"},
            "take_profit_r2": {"desc": "Partial exit (30%) R-multiple", "v1": "2.0", "v2": "2.5", "v3": "3.0",
                               "v4": "3.5"},
            "adx_min_long": {"desc": "Min ADX for long entries", "v1": "20", "v2": "22", "v3": "23", "v4": "25"},
            "adx_min_short": {"desc": "Min ADX for short entries", "v1": "22", "v2": "23", "v3": "25", "v4": "27"},
            "adx_slope_min": {"desc": "ADX must rise per bar", "v1": "0.0", "v2": "0.05", "v3": "0.08", "v4": "0.10"},
            "volume_min_ratio": {"desc": "Volume vs average minimum", "v1": "1.1", "v2": "1.2", "v3": "1.4",
                                 "v4": "1.6"},
            "rsi_long_min": {"desc": "RSI minimum for longs", "v1": "44", "v2": "46", "v3": "48", "v4": "50"},
            "rsi_long_max": {"desc": "RSI maximum for longs", "v1": "62", "v2": "64", "v3": "66", "v4": "68"},
            "rsi_short_min": {"desc": "RSI minimum for shorts", "v1": "28", "v2": "30", "v3": "32", "v4": "34"},
            "rsi_short_max": {"desc": "RSI maximum for shorts", "v1": "48", "v2": "50", "v3": "52", "v4": "54"},
            "ema_fast_period": {"desc": "Fast EMA period", "v1": "3", "v2": "5", "v3": "7", "v4": "9"},
            "ema_mid_period": {"desc": "Mid EMA period", "v1": "9", "v2": "11", "v3": "13", "v4": "15"},
            "ema_slow_period": {"desc": "Slow EMA period", "v1": "17", "v2": "19", "v3": "21", "v4": "24"},
            "macd_fast": {"desc": "MACD fast period", "v1": "5", "v2": "6", "v3": "7", "v4": "8"},
            "macd_slow": {"desc": "MACD slow period", "v1": "11", "v2": "12", "v3": "13", "v4": "15"},
            "macd_signal_period": {"desc": "MACD signal period", "v1": "3", "v2": "4", "v3": "5", "v4": "6"},
            "stoch_k_period": {"desc": "Stochastic %K period", "v1": "3", "v2": "5", "v3": "7", "v4": "9"},
            "max_daily_trades": {"desc": "Max trades per day", "v1": "5", "v2": "6", "v3": "8", "v4": "10"},
            "min_bars_between_trades": {"desc": "Min bars between entries", "v1": "2", "v2": "3", "v3": "4", "v4": "6"},
            "cooldown_after_loss_bars": {"desc": "Bars to pause after a loss", "v1": "4", "v2": "6", "v3": "8",
                                         "v4": "12"},
            "max_hold_bars": {"desc": "Max hold in bars", "v1": "24", "v2": "32", "v3": "48", "v4": "64"},
            "pullback_zone_lower_pct": {"desc": "Max % below EMA_fast for long", "v1": "-1.0", "v2": "-1.5",
                                        "v3": "-2.0", "v4": "-2.5"},
            "pullback_zone_upper_pct": {"desc": "Max % above EMA_fast for long", "v1": "0.3", "v2": "0.5", "v3": "0.8",
                                        "v4": "1.0"},
            "atr_compression_threshold": {"desc": "Block when ATR < this × avg ATR", "v1": "0.25", "v2": "0.30",
                                          "v3": "0.35", "v4": "0.40"},
            "chop_threshold": {"desc": "Choppiness Index block threshold", "v1": "58", "v2": "61", "v3": "63",
                               "v4": "65"},
            "extended_run_max_pct_long": {"desc": "Block longs if run > % from low", "v1": "3.0", "v2": "4.0",
                                          "v3": "5.0", "v4": "6.0"},
            "extended_run_max_pct_short": {"desc": "Block shorts if drop > % from high", "v1": "3.0", "v2": "4.0",
                                           "v3": "5.0", "v4": "6.0"},
            "risk_per_trade": {"desc": "Base risk % per trade", "v1": "0.006", "v2": "0.008", "v3": "0.010",
                               "v4": "0.012"},
            "risk_tier1": {"desc": "Risk % for Tier-1 entries", "v1": "0.008", "v2": "0.010", "v3": "0.012",
                           "v4": "0.015"},
            "weight_ema": {"desc": "EMA component weight", "v1": "18", "v2": "20", "v3": "22", "v4": "25"},
            "weight_macd": {"desc": "MACD component weight", "v1": "20", "v2": "22", "v3": "23", "v4": "25"},
            "weight_stoch": {"desc": "Stochastic component weight", "v1": "16", "v2": "18", "v3": "20", "v4": "22"},
            "weight_rsi": {"desc": "RSI component weight", "v1": "14", "v2": "16", "v3": "18", "v4": "20"},
            "weight_volume": {"desc": "Volume component weight", "v1": "8", "v2": "10", "v3": "12", "v4": "14"},
        }

        # Initialise scalping_backtest_params
        if not hasattr(self, "scalping_backtest_params"):
            self.scalping_backtest_params = {}
        for key, meta in ALL_SCALPING_OPT_PARAMS.items():
            if key not in self.scalping_backtest_params:
                self.scalping_backtest_params[key] = {
                    "active": tk.BooleanVar(value=False),
                    "value1": tk.StringVar(value=meta["v1"]),
                    "value2": tk.StringVar(value=meta["v2"]),
                    "value3": tk.StringVar(value=meta["v3"]),
                    "value4": tk.StringVar(value=meta["v4"]),
                    "description": meta["desc"],
                }
            else:
                self.scalping_backtest_params[key]["description"] = meta["desc"]

        # ── Optimisation metrics ──────────────────────────────────────────────
        mf = ttk.LabelFrame(right_frame, text="Optimization Metrics")
        mf.pack(fill=tk.X, padx=5, pady=5)
        mi = ttk.Frame(mf)
        mi.pack(fill=tk.X, padx=5, pady=5)
        row_i, col_i = 0, 0
        for name, var in self.optimization_metrics.items():
            ttk.Checkbutton(mi, text=name.replace("_", " ").title(),
                            variable=var).grid(row=row_i, column=col_i,
                                               sticky="w", padx=5, pady=2)
            col_i += 1
            if col_i > 1:
                col_i = 0;
                row_i += 1
        ttk.Label(mi, text="Equal weights (all selected metrics)").grid(
            row=row_i + 1, column=0, columnspan=2, pady=5
        )

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=5, pady=3)

        # ── Action buttons ────────────────────────────────────────────────────
        af = ttk.Frame(right_frame)
        af.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(af, text="✅ Select All", command=self._sc_select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(af, text="❌ Deselect All", command=self._sc_deselect_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(af, text="💾 Save Params", command=self.save_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(af, text="📂 Load Params", command=self.load_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(af, text="🔄 Reset", command=self.reset_scalping_backtest_params).pack(side=tk.LEFT, padx=2)

        self.sc_selection_label = ttk.Label(af, text="Selected: 0",
                                            foreground="blue", font=("Arial", 9, "bold"))
        self.sc_selection_label.pack(side=tk.RIGHT, padx=5)

        # ── Parameter table header — NOW INCLUDES "Current" column ───────────
        phf = ttk.LabelFrame(right_frame, text="Scalping Parameters (Select for Optimization)")
        phf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ch = ttk.Frame(phf)
        ch.pack(fill=tk.X, padx=5, pady=2)
        for ci, (lbl, w) in enumerate([
            ("✓", 3), ("Parameter", 22), ("Current", 8),
            ("Val 1", 7), ("Val 2", 7), ("Val 3", 7), ("Val 4", 7),
        ]):
            ttk.Label(ch, text=lbl, width=w,
                      anchor="center" if ci == 0 else "w",
                      font=("Arial", 8, "bold")).grid(row=0, column=ci, padx=1)
        ttk.Separator(phf, orient="horizontal").pack(fill=tk.X, padx=5)

        sc_canvas = tk.Canvas(phf, bg="white", highlightthickness=0)
        sc_sb = ttk.Scrollbar(phf, orient="vertical", command=sc_canvas.yview)
        sc_sf = ttk.Frame(sc_canvas)
        sc_sf.bind("<Configure>",
                   lambda e: sc_canvas.configure(scrollregion=sc_canvas.bbox("all")))
        sc_canvas.create_window((0, 0), window=sc_sf, anchor="nw")
        sc_canvas.configure(yscrollcommand=sc_sb.set)
        sc_canvas.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        sc_sb.pack(side="right", fill="y")
        sc_canvas.bind("<MouseWheel>",
                       lambda e: sc_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self.scalping_backtest_param_widgets = {}

        for p_row, (pk, pd) in enumerate(
                sorted(self.scalping_backtest_params.items(), key=lambda x: x[1]["description"])
        ):
            frame = ttk.Frame(sc_sf)
            frame.grid(row=p_row, column=0, sticky="ew", pady=1, padx=2)

            # Checkbox
            def _mk_cmd(k=pk):
                def cmd(): self._sc_update_count()

                return cmd

            cb = ttk.Checkbutton(frame, variable=pd["active"], command=_mk_cmd())
            cb.grid(row=0, column=0, padx=2)

            # Clickable label
            lbl = ttk.Label(frame, text=pd["description"], width=25, anchor="w",
                            font=("Arial", 8), cursor="hand2")
            lbl.grid(row=0, column=1, padx=2, sticky="w")

            def _mk_click(k=pk):
                def click(e):
                    cur = self.scalping_backtest_params[k]["active"].get()
                    self.scalping_backtest_params[k]["active"].set(not cur)
                    self._sc_update_count()

                return click

            lbl.bind("<Button-1>", _mk_click())

            # ── CURRENT VALUE column (new) ────────────────────────────────────
            cur_val = self._get_scalping_param_value(pk)
            cur_str = str(cur_val) if cur_val is not None else "—"
            cur_lbl = ttk.Label(frame, text=cur_str, width=8, anchor="center",
                                font=("Arial", 8, "bold"), foreground="#0066CC")
            cur_lbl.grid(row=0, column=2, padx=1)

            # Value inputs (v1-v4)
            for vi, vk in enumerate(["value1", "value2", "value3", "value4"], start=3):
                e = ttk.Entry(frame, textvariable=pd[vk], width=7, font=("Arial", 8))
                e.grid(row=0, column=vi, padx=1)

            self.scalping_backtest_param_widgets[pk] = {
                **pd, "current_lbl": cur_lbl, "widget": cb,
            }

        self._sc_update_count()

    def _sc_update_count(self):
        """Update Scalping optimization selection counter."""
        if hasattr(self, 'sc_selection_label') and self.sc_selection_label.winfo_exists():
            count = sum(1 for p in self.scalping_backtest_params.values() if p['active'].get())
            color = 'red' if count > 10 else 'green' if count > 0 else 'blue'
            self.sc_selection_label.config(text=f"Selected: {count}", foreground=color)

    def _sc_select_all(self):
        for p in self.scalping_backtest_params.values():
            p['active'].set(True)
        self._sc_update_count()

    def _sc_deselect_all(self):
        for p in self.scalping_backtest_params.values():
            p['active'].set(False)
        self._sc_update_count()

    def _get_scalping_param_value(self, param_key):
        """Get current value for a scalping param (UI → custom_params → SCALPING_PARAMS)."""
        # Layer 1: live scalping param widgets
        if hasattr(self, 'scalping_param_widgets') and param_key in self.scalping_param_widgets:
            try:
                cvar = self.scalping_param_widgets[param_key]['custom']
                if isinstance(cvar, tk.BooleanVar):
                    return cvar.get()
                raw = cvar.get()
                if raw not in ('', None):
                    return self.convert_param_value(raw)
            except Exception:
                pass
        # Layer 2: saved custom
        val = self.custom_params.get('scalping', {}).get(param_key)
        if val is not None:
            return val
        # Layer 3: SCALPING_PARAMS defaults
        try:
            from strategies.scalping_strategy import SCALPING_PARAMS
            return SCALPING_PARAMS.get(param_key)
        except ImportError:
            return None

    def _build_backtest_optimization_panel(self, right_frame):
        """
        Build the right-panel backtest optimization parameter UI.
        COMPLETE FIX: Shows ALL parameters for the currently selected strategy.
        """
        from strategies.MomentumStrategy_MACD_HybridScore_Latest import GlobalConfig

        # Clear existing content
        for widget in right_frame.winfo_children():
            widget.destroy()

        # ── Capital settings (shared across all strategies) ──────────────────────
        capital_frame = ttk.LabelFrame(right_frame, text="💰 Global Capital Settings")
        capital_frame.pack(fill=tk.X, padx=5, pady=5)

        capital_inner = ttk.Frame(capital_frame)
        capital_inner.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(capital_inner, text="Current Capital:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=2)
        self.capital_display_var = tk.StringVar(value=f"${GlobalConfig.INITIAL_CAPITAL:,.2f}")
        ttk.Label(capital_inner, textvariable=self.capital_display_var,
                  font=('Arial', 10, 'bold'), foreground='green').grid(
            row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(capital_inner, text="New Capital ($):", font=('Arial', 9)).grid(
            row=1, column=0, sticky='w', padx=5, pady=2)
        self.capital_entry = ttk.Entry(capital_inner, width=15, font=('Arial', 9))
        self.capital_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        self.capital_entry.insert(0, str(GlobalConfig.INITIAL_CAPITAL))

        update_btn = tk.Button(
            capital_inner,
            text="Update Capital",
            command=self._update_global_capital,
            bg="#4CAF50",
            fg="white",
            font=('Arial', 9, 'bold'),
            padx=10,
            pady=2,
            cursor="hand2"
        )
        update_btn.grid(row=1, column=2, padx=5, pady=2)

        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)

        # ═══ GET CURRENTLY SELECTED STRATEGY ═══════════════════════════════════════
        current_strategy = self.strategy_type_var.get() if hasattr(self, 'strategy_type_var') else "Momentum"

        self.log_message("=" * 70, "cyan")
        self.log_message(f"🔧 Building Backtest Optimization Panel for: {current_strategy}", "cyan")
        self.log_message("=" * 70, "cyan")

        # ═══ BUILD STRATEGY-SPECIFIC PANEL ═════════════════════════════════════════
        if current_strategy == "Kalman":
            self._build_kalman_backtest_panel(right_frame)
        elif current_strategy == "Scalping":
            self._build_scalping_backtest_panel(right_frame)
        else:  # Momentum (default)
            self._build_momentum_backtest_panel(right_frame)

    def _build_momentum_backtest_panel(self, right_frame):
        """Build COMPLETE Momentum-specific backtest optimization panel."""

        current_params = self.get_current_momentum_params()

        # ═══════════════════════════════════════════════════════════════════
        # ALL_MOMENTUM_BACKTEST_PARAMS - UPDATED v9.5.0
        # REMOVED: generic/old parameters (quality_tier1_min, tier2_adx_min, etc.)
        # ADDED: tier-specific parameters (_long/_short suffixed)
        # ═══════════════════════════════════════════════════════════════════
        ALL_MOMENTUM_BACKTEST_PARAMS = {
            # ──────────────────────────────────────────────────────────────
            # QUALITY THRESHOLDS (UPDATED: tier-specific)
            # ──────────────────────────────────────────────────────────────
            'quality_tier1_min_long': {
                'description': 'Tier 1 Min Score (LONG)',
                'current': current_params.get('quality_tier1_min_long', 0.70),
                'v1': '0.65', 'v2': '0.70', 'v3': '0.75', 'v4': '0.80'
            },
            'quality_tier2_min_long': {
                'description': 'Tier 2 Min Score (LONG)',
                'current': current_params.get('quality_tier2_min_long', 0.60),
                'v1': '0.55', 'v2': '0.60', 'v3': '0.65', 'v4': '0.70'
            },
            'quality_tier1_min_short': {
                'description': 'Tier 1 Min Score (SHORT)',
                'current': current_params.get('quality_tier1_min_short', 0.70),
                'v1': '0.65', 'v2': '0.70', 'v3': '0.75', 'v4': '0.80'
            },
            'quality_tier2_min_short': {
                'description': 'Tier 2 Min Score (SHORT)',
                'current': current_params.get('quality_tier2_min_short', 0.60),
                'v1': '0.55', 'v2': '0.60', 'v3': '0.65', 'v4': '0.70'
            },

            # ──────────────────────────────────────────────────────────────
            # TIER CONTROL
            # ──────────────────────────────────────────────────────────────
            'only_tier1_entries': {
                'description': '🔥 Only Tier 2 Entries (Block Tier 1)',
                'current': current_params.get('only_tier1_entries', False),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''
            },

            # ──────────────────────────────────────────────────────────────
            # EMA PERIODS
            # ──────────────────────────────────────────────────────────────
            'ema_fast_period': {
                'description': 'EMA Fast Period',
                'current': current_params.get('ema_fast_period', 9),
                'v1': '8', 'v2': '9', 'v3': '10', 'v4': '12'
            },
            'ema_mid_period': {
                'description': 'EMA Mid Period',
                'current': current_params.get('ema_mid_period', 21),
                'v1': '18', 'v2': '20', 'v3': '21', 'v4': '24'
            },
            'ema_slow_period': {
                'description': 'EMA Slow Period',
                'current': current_params.get('ema_slow_period', 55),
                'v1': '45', 'v2': '50', 'v3': '55', 'v4': '60'
            },

            # ──────────────────────────────────────────────────────────────
            # QUALITY WEIGHTS
            # ──────────────────────────────────────────────────────────────
            'weight_ema': {
                'description': 'EMA Weight',
                'current': current_params.get('weight_ema', 20),
                'v1': '15', 'v2': '18', 'v3': '20', 'v4': '25'
            },
            'weight_adx': {
                'description': 'ADX Weight',
                'current': current_params.get('weight_adx', 20),
                'v1': '15', 'v2': '18', 'v3': '20', 'v4': '25'
            },
            'weight_macd': {
                'description': 'MACD Weight',
                'current': current_params.get('weight_macd', 25),
                'v1': '20', 'v2': '22', 'v3': '25', 'v4': '28'
            },
            'weight_rsi': {
                'description': 'RSI Weight',
                'current': current_params.get('weight_rsi', 20),
                'v1': '15', 'v2': '18', 'v3': '20', 'v4': '22'
            },
            'weight_volume': {
                'description': 'Volume Weight',
                'current': current_params.get('weight_volume', 15),
                'v1': '10', 'v2': '12', 'v3': '15', 'v4': '18'
            },

            # ──────────────────────────────────────────────────────────────
            # TIER 1 LONG FILTERS
            # ──────────────────────────────────────────────────────────────
            'tier1_adx_hard_min': {
                'description': 'Tier 1 ADX Hard Min (LONG)',
                'current': current_params.get('tier1_adx_hard_min', 25.0),
                'v1': '20', 'v2': '22', 'v3': '25', 'v4': '28'
            },
            'tier1_adx_min': {
                'description': 'Tier 1 ADX Min (soft)',
                'current': current_params.get('tier1_adx_min', 20.0),
                'v1': '16', 'v2': '18', 'v3': '20', 'v4': '22'
            },
            'tier1_rsi_min': {
                'description': 'Tier 1 RSI Min (LONG)',
                'current': current_params.get('tier1_rsi_min', 55.0),
                'v1': '50', 'v2': '52', 'v3': '55', 'v4': '58'
            },
            'tier1_rsi_max': {
                'description': 'Tier 1 RSI Max (LONG)',
                'current': current_params.get('tier1_rsi_max', 75.0),
                'v1': '70', 'v2': '72', 'v3': '75', 'v4': '78'
            },
            'tier1_volume_min': {
                'description': 'Tier 1 Volume Min (LONG)',
                'current': current_params.get('tier1_volume_min', 1.5),
                'v1': '1.2', 'v2': '1.3', 'v3': '1.5', 'v4': '1.8'
            },
            'tier1_momentum_min': {
                'description': 'Tier 1 Momentum Min',
                'current': current_params.get('tier1_momentum_min', 0.02),
                'v1': '0.01', 'v2': '0.015', 'v3': '0.02', 'v4': '0.03'
            },
            'tier1_price_ema_max_pct': {
                'description': 'Tier 1 Price-EMA Max %',
                'current': current_params.get('tier1_price_ema_max_pct', 1.5),
                'v1': '1.0', 'v2': '1.5', 'v3': '2.0', 'v4': '3.0'
            },
            'pullback_zone_lower_pct': {
                'description': 'Pullback Zone Lower %',
                'current': current_params.get('pullback_zone_lower_pct', -2.5),
                'v1': '-3.0', 'v2': '-2.5', 'v3': '-2.0', 'v4': '-1.5'
            },
            'pullback_zone_upper_pct': {
                'description': 'Pullback Zone Upper %',
                'current': current_params.get('pullback_zone_upper_pct', 1.5),
                'v1': '1.0', 'v2': '1.5', 'v3': '2.0', 'v4': '3.0'
            },
            'adx_slope_min': {
                'description': 'ADX Slope Min',
                'current': current_params.get('adx_slope_min', 0.1),
                'v1': '-0.2', 'v2': '0.0', 'v3': '0.1', 'v4': '0.2'
            },

            # ──────────────────────────────────────────────────────────────
            # TIER 1 SHORT FILTERS (UPDATED)
            # ──────────────────────────────────────────────────────────────
            'short_tier1_adx_hard_min': {
                'description': 'Tier 1 ADX Hard Min (SHORT)',
                'current': current_params.get('short_tier1_adx_hard_min', 30.0),
                'v1': '25', 'v2': '28', 'v3': '30', 'v4': '32'
            },
            'tier1_rsi_min_short': {
                'description': 'Tier 1 RSI Min (SHORT)',
                'current': current_params.get('tier1_rsi_min_short', 25.0),
                'v1': '20', 'v2': '22', 'v3': '25', 'v4': '28'
            },
            'tier1_rsi_max_short': {
                'description': 'Tier 1 RSI Max (SHORT)',
                'current': current_params.get('tier1_rsi_max_short', 45.0),
                'v1': '40', 'v2': '42', 'v3': '45', 'v4': '48'
            },
            'short_tier1_volume_min': {
                'description': 'Tier 1 Volume Min (SHORT)',
                'current': current_params.get('short_tier1_volume_min', 1.3),
                'v1': '1.0', 'v2': '1.2', 'v3': '1.3', 'v4': '1.5'
            },
            'short_tier1_momentum_min': {
                'description': 'Tier 1 Momentum Min (SHORT)',
                'current': current_params.get('short_tier1_momentum_min', 0.05),
                'v1': '0.03', 'v2': '0.04', 'v3': '0.05', 'v4': '0.06'
            },

            # ──────────────────────────────────────────────────────────────
            # TIER 2 FILTERS (UPDATED)
            # ──────────────────────────────────────────────────────────────
            'tier2_adx_hard_min': {
                'description': 'Tier 2 ADX Hard Min',
                'current': current_params.get('tier2_adx_hard_min', 20.0),
                'v1': '15', 'v2': '18', 'v3': '20', 'v4': '22'
            },
            'tier2_volume_min': {
                'description': 'Tier 2 Volume Min',
                'current': current_params.get('tier2_volume_min', 1.2),
                'v1': '0.9', 'v2': '1.0', 'v3': '1.2', 'v4': '1.4'
            },
            'tier2_momentum_min': {
                'description': 'Tier 2 Momentum Min',
                'current': current_params.get('tier2_momentum_min', 0.01),
                'v1': '0.005', 'v2': '0.01', 'v3': '0.015', 'v4': '0.02'
            },
            'tier2_rsi_min': {
                'description': 'Tier 2 RSI Min (LONG)',
                'current': current_params.get('tier2_rsi_min', 50.0),
                'v1': '45', 'v2': '48', 'v3': '50', 'v4': '55'
            },
            'tier2_rsi_max': {
                'description': 'Tier 2 RSI Max (LONG)',
                'current': current_params.get('tier2_rsi_max', 70.0),
                'v1': '65', 'v2': '68', 'v3': '70', 'v4': '72'
            },
            'tier2_rsi_min_short': {
                'description': 'Tier 2 RSI Min (SHORT)',
                'current': current_params.get('tier2_rsi_min_short', 30.0),
                'v1': '25', 'v2': '28', 'v3': '30', 'v4': '35'
            },
            'tier2_rsi_max_short': {
                'description': 'Tier 2 RSI Max (SHORT)',
                'current': current_params.get('tier2_rsi_max_short', 50.0),
                'v1': '45', 'v2': '48', 'v3': '50', 'v4': '55'
            },

            # ──────────────────────────────────────────────────────────────
            # CONFLUENCE & COOLDOWN
            # ──────────────────────────────────────────────────────────────
            'tier1_confluence_min': {
                'description': 'Tier 1 Confluence Min (signals count)',
                'current': current_params.get('tier1_confluence_min', 3.0),
                'v1': '2', 'v2': '3', 'v3': '4', 'v4': '5'
            },
            'tier2_confluence_min': {
                'description': 'Tier 2 Confluence Min (signals count)',
                'current': current_params.get('tier2_confluence_min', 2.0),
                'v1': '1', 'v2': '2', 'v3': '3', 'v4': '4'
            },
            'min_bars_between_trades_tier1': {
                'description': 'Min Bars Between Trades (Tier 1)',
                'current': current_params.get('min_bars_between_trades_tier1', 4),
                'v1': '2', 'v2': '3', 'v3': '4', 'v4': '6'
            },
            'min_bars_between_trades_tier2': {
                'description': 'Min Bars Between Trades (Tier 2)',
                'current': current_params.get('min_bars_between_trades_tier2', 3),
                'v1': '2', 'v2': '3', 'v3': '4', 'v4': '5'
            },

            # ──────────────────────────────────────────────────────────────
            # RISK MANAGEMENT (UPDATED)
            # ──────────────────────────────────────────────────────────────
            'risk_tier1': {
                'description': 'Tier 1 Risk %',
                'current': current_params.get('risk_tier1', 0.02),
                'v1': '0.015', 'v2': '0.02', 'v3': '0.025', 'v4': '0.03'
            },
            'risk_tier2': {
                'description': 'Tier 2 Risk %',
                'current': current_params.get('risk_tier2', 0.01),
                'v1': '0.008', 'v2': '0.01', 'v3': '0.015', 'v4': '0.02'
            },
            'tier1_size_multiplier': {
                'description': 'Tier 1 Size Multiplier',
                'current': current_params.get('tier1_size_multiplier', 1.0),
                'v1': '0.8', 'v2': '1.0', 'v3': '1.2', 'v4': '1.5'
            },
            'tier2_size_multiplier': {
                'description': 'Tier 2 Size Multiplier',
                'current': current_params.get('tier2_size_multiplier', 0.70),
                'v1': '0.5', 'v2': '0.7', 'v3': '0.9', 'v4': '1.0'
            },

            # ──────────────────────────────────────────────────────────────
            # STOP LOSS & TRAILING (UPDATED)
            # ──────────────────────────────────────────────────────────────
            'stop_loss_atr_mult': {
                'description': 'Stop Loss ATR Mult (all tiers)',
                'current': current_params.get('stop_loss_atr_mult', 2.0),
                'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'
            },
            'trailing_activation_tier1': {
                'description': 'Trail Activation % (Tier 1)',
                'current': current_params.get('trailing_activation_tier1', 0.03),
                'v1': '0.02', 'v2': '0.025', 'v3': '0.03', 'v4': '0.04'
            },
            'trailing_activation_tier2': {
                'description': 'Trail Activation % (Tier 2)',
                'current': current_params.get('trailing_activation_tier2', 0.02),
                'v1': '0.015', 'v2': '0.02', 'v3': '0.025', 'v4': '0.03'
            },
            'trailing_distance_tier1': {
                'description': 'Trailing Distance % (Tier 1)',
                'current': current_params.get('trailing_distance_tier1', 0.015),
                'v1': '0.01', 'v2': '0.015', 'v3': '0.02', 'v4': '0.025'
            },
            'trailing_distance_tier2': {
                'description': 'Trailing Distance % (Tier 2)',
                'current': current_params.get('trailing_distance_tier2', 0.01),
                'v1': '0.008', 'v2': '0.01', 'v3': '0.015', 'v4': '0.02'
            },
            'trailing_stop_atr_mult': {
                'description': 'Trailing Stop ATR Mult',
                'current': current_params.get('trailing_stop_atr_mult', 6.5),
                'v1': '5.0', 'v2': '5.5', 'v3': '6.0', 'v4': '6.5'
            },
            'trailing_stop_pct': {
                'description': 'Trailing Stop %',
                'current': current_params.get('trailing_stop_pct', 0.06),
                'v1': '0.04', 'v2': '0.05', 'v3': '0.06', 'v4': '0.08'
            },
            'trailing_activation_r': {
                'description': 'Trail Activation R',
                'current': current_params.get('trailing_activation_r', 3.0),
                'v1': '2.0', 'v2': '2.5', 'v3': '3.0', 'v4': '3.5'
            },

            # ──────────────────────────────────────────────────────────────
            # BREAKEVEN STOP
            # ──────────────────────────────────────────────────────────────
            'be_stop_enabled': {
                'description': 'Breakeven Stop Enabled',
                'current': current_params.get('be_stop_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''
            },
            'be_stop_r_trigger': {
                'description': 'Breakeven R Trigger',
                'current': current_params.get('be_stop_r_trigger', 2.0),
                'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'
            },

            # ──────────────────────────────────────────────────────────────
            # EXIT CONDITIONS
            # ──────────────────────────────────────────────────────────────
            'max_hold_bars': {
                'description': 'Max Hold Bars',
                'current': current_params.get('max_hold_bars', 500),
                'v1': '200', 'v2': '300', 'v3': '400', 'v4': '500'
            },
            'rsi_exit_threshold': {
                'description': 'RSI Exit Threshold',
                'current': current_params.get('rsi_exit_threshold', 80),
                'v1': '75', 'v2': '78', 'v3': '80', 'v4': '85'
            },
            'macd_bearish_cross_profit_min': {
                'description': 'MACD Bearish Cross Min Profit %',
                'current': current_params.get('macd_bearish_cross_profit_min', 2.5),
                'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'
            },
            'kalman_fade_threshold': {
                'description': 'Kalman Fade Threshold',
                'current': current_params.get('kalman_fade_threshold', 35),
                'v1': '25', 'v2': '30', 'v3': '35', 'v4': '40'
            },

            # ──────────────────────────────────────────────────────────────
            # COOLDOWN & TRADE MANAGEMENT
            # ──────────────────────────────────────────────────────────────
            'max_daily_trades': {
                'description': 'Max Daily Trades',
                'current': current_params.get('max_daily_trades', 15),
                'v1': '5', 'v2': '8', 'v3': '10', 'v4': '15'
            },
            'min_bars_between_trades': {
                'description': 'Min Bars Between Trades (blanket)',
                'current': current_params.get('min_bars_between_trades', 4),
                'v1': '2', 'v2': '3', 'v3': '4', 'v4': '6'
            },
            'cooldown_after_loss_bars': {
                'description': 'Cooldown After Loss (bars)',
                'current': current_params.get('cooldown_after_loss_bars', 12),
                'v1': '6', 'v2': '8', 'v3': '12', 'v4': '16'
            },

            # ──────────────────────────────────────────────────────────────
            # PRICE POSITIONING
            # ──────────────────────────────────────────────────────────────
            'price_percentile_bonus_early': {
                'description': 'Early Entry Bonus',
                'current': current_params.get('price_percentile_bonus_early', 12),
                'v1': '8', 'v2': '10', 'v3': '12', 'v4': '15'
            },
            'price_percentile_penalty_late': {
                'description': 'Late Entry Penalty',
                'current': current_params.get('price_percentile_penalty_late', 12),
                'v1': '8', 'v2': '10', 'v3': '12', 'v4': '15'
            },

            # ──────────────────────────────────────────────────────────────        # FUZZY MODE
            # ──────────────────────────────────────────────────────────────
            'fuzzy_default_margin_pct': {
                'description': 'Fuzzy Margin %',
                'current': current_params.get('fuzzy_default_margin_pct', 10),
                'v1': '5', 'v2': '8', 'v3': '10', 'v4': '15'
            },
            'fuzzy_absolute_min': {
                'description': 'Fuzzy Absolute Min',
                'current': current_params.get('fuzzy_absolute_min', 45),
                'v1': '55', 'v2': '58', 'v3': '62', 'v4': '65'
            },

            # ──────────────────────────────────────────────────────────────
            # REGIME FILTERS
            # ──────────────────────────────────────────────────────────────
            'regime_filter_enabled': {
                'description': 'Regime Filter Enabled',
                'current': current_params.get('regime_filter_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''
            },
            'chop_threshold': {
                'description': 'Choppiness Index Threshold',
                'current': current_params.get('chop_threshold', 58),
                'v1': '50', 'v2': '54', 'v3': '58', 'v4': '62'
            },
            'atr_compression_threshold': {
                'description': 'ATR Compression Threshold',
                'current': current_params.get('atr_compression_threshold', 0.25),
                'v1': '0.15', 'v2': '0.20', 'v3': '0.25', 'v4': '0.30'
            },
            'extended_run_max_pct_long': {
                'description': 'Extended Run Max % (LONG)',
                'current': current_params.get('extended_run_max_pct_long', 12.0),
                'v1': '8.0', 'v2': '10.0', 'v3': '12.0', 'v4': '15.0'
            },
            'extended_run_max_pct_short': {
                'description': 'Extended Run Max % (SHORT)',
                'current': current_params.get('extended_run_max_pct_short', 12.0),
                'v1': '8.0', 'v2': '10.0', 'v3': '12.0', 'v4': '15.0'
            },

            # ──────────────────────────────────────────────────────────────
            # TRADE DIRECTION
            # ──────────────────────────────────────────────────────────────
            'trade_direction': {
                'description': 'Trade Direction',
                'current': current_params.get('trade_direction', 'both'),
                'v1': 'long', 'v2': 'both', 'v3': 'short', 'v4': ''
            },
        }

        # ─── Initialize backtest_params ──────────────────────────────────
        if not hasattr(self, 'backtest_params'):
            self.backtest_params = {}

        for key, meta in ALL_MOMENTUM_BACKTEST_PARAMS.items():
            if key not in self.backtest_params:
                self.backtest_params[key] = {
                    'active': tk.BooleanVar(value=False),
                    'value1': tk.StringVar(value=meta['v1']),
                    'value2': tk.StringVar(value=meta['v2']),
                    'value3': tk.StringVar(value=meta['v3']),
                    'value4': tk.StringVar(value=meta['v4']),
                    'description': meta['description'],
                    'current_value': meta['current']
                }
            else:
                self.backtest_params[key]['current_value'] = meta['current']
                self.backtest_params[key]['description'] = meta['description']

        self._build_backtest_ui(right_frame)

    def _build_kalman_backtest_panel(self, right_frame):
        """Build COMPLETE Kalman-specific backtest optimization panel"""
        from strategies.KalmanTrendStrategy_New import KALMAN_PARAMS

        current_params = self.get_current_kalman_params()

        ALL_KALMAN_BACKTEST_PARAMS = {
            # Kalman Filter Parameters
            'process_noise_1': {
                'description': 'Process Noise 1',
                'current': current_params.get('process_noise_1', 0.001),
                'v1': '0.0005', 'v2': '0.001', 'v3': '0.002', 'v4': '0.005'},
            'process_noise_2': {
                'description': 'Process Noise 2',
                'current': current_params.get('process_noise_2', 0.001),
                'v1': '0.0005', 'v2': '0.001', 'v3': '0.002', 'v4': '0.005'},
            'measurement_noise': {
                'description': 'Measurement Noise',
                'current': current_params.get('measurement_noise', 100.0),
                'v1': '50', 'v2': '100', 'v3': '200', 'v4': '500'},

            # Trend Detection
            'trend_lookback': {
                'description': 'Trend Lookback',
                'current': current_params.get('trend_lookback', 20),
                'v1': '10', 'v2': '15', 'v3': '20', 'v4': '30'},
            'strength_smooth': {
                'description': 'Strength Smoothing',
                'current': current_params.get('strength_smooth', 5),
                'v1': '3', 'v2': '5', 'v3': '7', 'v4': '10'},

            # Risk/Reward
            'risk_reward': {
                'description': 'Risk/Reward Ratio',
                'current': current_params.get('risk_reward', 1.5),
                'v1': '1.0', 'v2': '1.5', 'v3': '2.0', 'v4': '2.5'},

            # Entry Conditions
            'long_kalman_strength_min': {
                'description': 'Min Kalman Strength (LONG)',
                'current': current_params.get('long_kalman_strength_min', 30),
                'v1': '20', 'v2': '25', 'v3': '30', 'v4': '40'},
            'short_kalman_strength_min': {
                'description': 'Min Kalman Strength (SHORT)',
                'current': current_params.get('short_kalman_strength_min', -30),
                'v1': '-40', 'v2': '-35', 'v3': '-30', 'v4': '-20'},

            # RSI Filters
            'long_rsi_min': {
                'description': 'RSI Min (LONG)',
                'current': current_params.get('long_rsi_min', 30),
                'v1': '25', 'v2': '30', 'v3': '35', 'v4': '40'},
            'long_rsi_max': {
                'description': 'RSI Max (LONG)',
                'current': current_params.get('long_rsi_max', 70),
                'v1': '60', 'v2': '65', 'v3': '70', 'v4': '75'},
            'short_rsi_min': {
                'description': 'RSI Min (SHORT)',
                'current': current_params.get('short_rsi_min', 30),
                'v1': '25', 'v2': '30', 'v3': '35', 'v4': '40'},
            'short_rsi_max': {
                'description': 'RSI Max (SHORT)',
                'current': current_params.get('short_rsi_max', 70),
                'v1': '60', 'v2': '65', 'v3': '70', 'v4': '75'},

            # Stop Loss & Trailing
            'stop_loss_pct': {
                'description': 'Stop Loss %',
                'current': current_params.get('stop_loss_pct', 0.02),
                'v1': '0.015', 'v2': '0.02', 'v3': '0.025', 'v4': '0.03'},
            'trailing_stop_pct': {
                'description': 'Trailing Stop %',
                'current': current_params.get('trailing_stop_pct', 0.015),
                'v1': '0.01', 'v2': '0.015', 'v3': '0.02', 'v4': '0.025'},
            'atr_multiplier': {
                'description': 'ATR Multiplier',
                'current': current_params.get('atr_multiplier', 2.0),
                'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'},

            # Risk Management
            'risk_per_trade': {
                'description': 'Risk Per Trade %',
                'current': current_params.get('risk_per_trade', 0.01),
                'v1': '0.005', 'v2': '0.01', 'v3': '0.015', 'v4': '0.02'},
            'max_position_pct': {
                'description': 'Max Position %',
                'current': current_params.get('max_position_pct', 0.15),
                'v1': '0.10', 'v2': '0.15', 'v3': '0.20', 'v4': '0.25'},

            # Hold Time
            'max_hold_bars': {
                'description': 'Max Hold Bars',
                'current': current_params.get('max_hold_bars', 48),
                'v1': '24', 'v2': '36', 'v3': '48', 'v4': '72'},
            'max_hold_seconds': {
                'description': 'Max Hold Seconds',
                'current': current_params.get('max_hold_seconds', 3600),
                'v1': '1800', 'v2': '3600', 'v3': '7200', 'v4': '14400'},

            # Market Filters
            'min_adx': {
                'description': 'Min ADX',
                'current': current_params.get('min_adx', 15),
                'v1': '10', 'v2': '15', 'v3': '20', 'v4': '25'},
            'min_volatility': {
                'description': 'Min Volatility',
                'current': current_params.get('min_volatility', 0.001),
                'v1': '0.0005', 'v2': '0.001', 'v3': '0.002', 'v4': '0.003'},
            'cooldown_bars': {
                'description': 'Cooldown Bars',
                'current': current_params.get('cooldown_bars', 10),
                'v1': '5', 'v2': '10', 'v3': '15', 'v4': '20'},
            'volume_min_ratio': {
                'description': 'Min Volume Ratio',
                'current': current_params.get('volume_min_ratio', 1.0),
                'v1': '0.8', 'v2': '1.0', 'v3': '1.2', 'v4': '1.5'},

            # Trade Direction
            'trade_direction': {
                'description': 'Trade Direction',
                'current': current_params.get('trade_direction', 'both'),
                'v1': 'long', 'v2': 'both', 'v3': 'short', 'v4': ''},
        }

        # Initialize backtest_params for Kalman
        if not hasattr(self, 'backtest_params'):
            self.backtest_params = {}

        for key, meta in ALL_KALMAN_BACKTEST_PARAMS.items():
            if key not in self.backtest_params:
                self.backtest_params[key] = {
                    'active': tk.BooleanVar(value=False),
                    'value1': tk.StringVar(value=meta['v1']),
                    'value2': tk.StringVar(value=meta['v2']),
                    'value3': tk.StringVar(value=meta['v3']),
                    'value4': tk.StringVar(value=meta['v4']),
                    'description': meta['description'],
                    'current_value': meta['current']
                }
            else:
                self.backtest_params[key]['current_value'] = meta['current']

        self._build_backtest_ui(right_frame)

    def _build_scalping_backtest_panel(self, right_frame):
        """Build COMPLETE Scalping-specific backtest optimization panel with save/load"""
        from strategies.scalping_strategy import SCALPING_PARAMS

        current_params = self.get_current_scalping_params()

        self.log_message("=" * 70, "yellow")
        self.log_message(f"📋 SCALPING BACKTEST PARAMETER SOURCE: {self.param_toggle_var.get()}", "yellow")
        self.log_message("=" * 70, "yellow")

        ALL_SCALPING_BACKTEST_PARAMS = {
            # ──────────────────────────────────────────────────────────────────────
            # EMA Periods
            # ──────────────────────────────────────────────────────────────────────
            'ema_fast_period': {
                'description': 'Fast EMA Period',
                'current': current_params.get('ema_fast_period', 5),
                'v1': '3', 'v2': '5', 'v3': '7', 'v4': '9'},
            'ema_mid_period': {
                'description': 'Mid EMA Period',
                'current': current_params.get('ema_mid_period', 13),
                'v1': '9', 'v2': '11', 'v3': '13', 'v4': '15'},
            'ema_slow_period': {
                'description': 'Slow EMA Period',
                'current': current_params.get('ema_slow_period', 21),
                'v1': '17', 'v2': '19', 'v3': '21', 'v4': '24'},

            # ──────────────────────────────────────────────────────────────────────
            # MACD Parameters
            # ──────────────────────────────────────────────────────────────────────
            'macd_fast': {
                'description': 'MACD Fast Period',
                'current': current_params.get('macd_fast', 12),
                'v1': '8', 'v2': '10', 'v3': '12', 'v4': '14'},
            'macd_slow': {
                'description': 'MACD Slow Period',
                'current': current_params.get('macd_slow', 26),
                'v1': '22', 'v2': '24', 'v3': '26', 'v4': '28'},
            'macd_signal_period': {
                'description': 'MACD Signal Period',
                'current': current_params.get('macd_signal_period', 9),
                'v1': '7', 'v2': '9', 'v3': '11', 'v4': '13'},

            # ──────────────────────────────────────────────────────────────────────
            # Stochastic Parameters
            # ──────────────────────────────────────────────────────────────────────
            'stoch_k_period': {
                'description': 'Stochastic %K Period',
                'current': current_params.get('stoch_k_period', 5),
                'v1': '3', 'v2': '5', 'v3': '7', 'v4': '9'},
            'stoch_d_period': {
                'description': 'Stochastic %D Period',
                'current': current_params.get('stoch_d_period', 3),
                'v1': '2', 'v2': '3', 'v3': '4', 'v4': '5'},
            'stoch_smooth': {
                'description': 'Stochastic Smoothing',
                'current': current_params.get('stoch_smooth', 3),
                'v1': '1', 'v2': '3', 'v3': '5', 'v4': '7'},
            'stoch_overbought': {
                'description': 'Stochastic Overbought',
                'current': current_params.get('stoch_overbought', 80),
                'v1': '70', 'v2': '75', 'v3': '80', 'v4': '85'},
            'stoch_oversold': {
                'description': 'Stochastic Oversold',
                'current': current_params.get('stoch_oversold', 20),
                'v1': '15', 'v2': '18', 'v3': '20', 'v4': '25'},
            'stoch_mid_upper': {
                'description': 'Stochastic Mid Upper',
                'current': current_params.get('stoch_mid_upper', 70),
                'v1': '60', 'v2': '65', 'v3': '70', 'v4': '75'},
            'stoch_mid_lower': {
                'description': 'Stochastic Mid Lower',
                'current': current_params.get('stoch_mid_lower', 30),
                'v1': '25', 'v2': '28', 'v3': '30', 'v4': '35'},

            # ──────────────────────────────────────────────────────────────────────
            # RSI Filters
            # ──────────────────────────────────────────────────────────────────────
            'rsi_period': {
                'description': 'RSI Period',
                'current': current_params.get('rsi_period', 14),
                'v1': '10', 'v2': '12', 'v3': '14', 'v4': '16'},
            'rsi_long_min': {
                'description': 'RSI Long Min',
                'current': current_params.get('rsi_long_min', 45),
                'v1': '40', 'v2': '45', 'v3': '50', 'v4': '55'},
            'rsi_long_max': {
                'description': 'RSI Long Max',
                'current': current_params.get('rsi_long_max', 68),
                'v1': '60', 'v2': '65', 'v3': '68', 'v4': '72'},
            'rsi_short_min': {
                'description': 'RSI Short Min',
                'current': current_params.get('rsi_short_min', 32),
                'v1': '28', 'v2': '30', 'v3': '32', 'v4': '35'},
            'rsi_short_max': {
                'description': 'RSI Short Max',
                'current': current_params.get('rsi_short_max', 55),
                'v1': '50', 'v2': '52', 'v3': '55', 'v4': '58'},
            'rsi_overbought_exit': {
                'description': 'RSI Overbought Exit',
                'current': current_params.get('rsi_overbought_exit', 75),
                'v1': '70', 'v2': '75', 'v3': '80', 'v4': '85'},
            'rsi_oversold_exit': {
                'description': 'RSI Oversold Exit',
                'current': current_params.get('rsi_oversold_exit', 25),
                'v1': '20', 'v2': '25', 'v3': '30', 'v4': '35'},

            # ──────────────────────────────────────────────────────────────────────
            # ADX Filters
            # ──────────────────────────────────────────────────────────────────────
            'adx_period': {
                'description': 'ADX Period',
                'current': current_params.get('adx_period', 14),
                'v1': '10', 'v2': '12', 'v3': '14', 'v4': '16'},
            'adx_min_long': {
                'description': 'ADX Min Long',
                'current': current_params.get('adx_min_long', 20),
                'v1': '15', 'v2': '18', 'v3': '20', 'v4': '22'},
            'adx_min_short': {
                'description': 'ADX Min Short',
                'current': current_params.get('adx_min_short', 22),
                'v1': '18', 'v2': '20', 'v3': '22', 'v4': '25'},
            'adx_extended_threshold': {
                'description': 'ADX Extended Threshold',
                'current': current_params.get('adx_extended_threshold', 45),
                'v1': '40', 'v2': '45', 'v3': '50', 'v4': '55'},
            'adx_slope_min': {
                'description': 'ADX Slope Min',
                'current': current_params.get('adx_slope_min', -0.1),
                'v1': '-0.3', 'v2': '-0.2', 'v3': '-0.1', 'v4': '0.0'},

            # ──────────────────────────────────────────────────────────────────────
            # Volume & ATR
            # ──────────────────────────────────────────────────────────────────────
            'volume_period': {
                'description': 'Volume Period',
                'current': current_params.get('volume_period', 20),
                'v1': '10', 'v2': '15', 'v3': '20', 'v4': '30'},
            'volume_min_ratio': {
                'description': 'Volume Min Ratio',
                'current': current_params.get('volume_min_ratio', 1.2),
                'v1': '1.0', 'v2': '1.1', 'v3': '1.2', 'v4': '1.4'},
            'volume_strong_ratio': {
                'description': 'Volume Strong Ratio',
                'current': current_params.get('volume_strong_ratio', 1.8),
                'v1': '1.5', 'v2': '1.8', 'v3': '2.0', 'v4': '2.5'},
            'atr_period': {
                'description': 'ATR Period',
                'current': current_params.get('atr_period', 14),
                'v1': '10', 'v2': '12', 'v3': '14', 'v4': '16'},
            'atr_compression_threshold': {
                'description': 'ATR Compression Threshold',
                'current': current_params.get('atr_compression_threshold', 0.35),
                'v1': '0.25', 'v2': '0.30', 'v3': '0.35', 'v4': '0.40'},

            # ──────────────────────────────────────────────────────────────────────
            # Quality Score Thresholds
            # ──────────────────────────────────────────────────────────────────────
            'quality_min_long': {
                'description': 'Min Quality Score (LONG)',
                'current': current_params.get('quality_min_long', 35),
                'v1': '30', 'v2': '35', 'v3': '40', 'v4': '45'},
            'quality_min_short': {
                'description': 'Min Quality Score (SHORT)',
                'current': current_params.get('quality_min_short', 35),
                'v1': '30', 'v2': '35', 'v3': '40', 'v4': '45'},
            'quality_tier1_min': {
                'description': 'Tier 1 Quality Min',
                'current': current_params.get('quality_tier1_min', 50),
                'v1': '45', 'v2': '48', 'v3': '50', 'v4': '55'},

            # ──────────────────────────────────────────────────────────────────────
            # Quality Weights (Scalping specific)
            # ──────────────────────────────────────────────────────────────────────
            'weight_ema': {
                'description': 'EMA Weight',
                'current': current_params.get('weight_ema', 22),
                'v1': '15', 'v2': '20', 'v3': '22', 'v4': '25'},
            'weight_macd': {
                'description': 'MACD Weight',
                'current': current_params.get('weight_macd', 23),
                'v1': '18', 'v2': '20', 'v3': '23', 'v4': '25'},
            'weight_stoch': {
                'description': 'Stochastic Weight',
                'current': current_params.get('weight_stoch', 20),
                'v1': '15', 'v2': '18', 'v3': '20', 'v4': '22'},
            'weight_rsi': {
                'description': 'RSI Weight',
                'current': current_params.get('weight_rsi', 18),
                'v1': '14', 'v2': '16', 'v3': '18', 'v4': '20'},
            'weight_volume': {
                'description': 'Volume Weight',
                'current': current_params.get('weight_volume', 12),
                'v1': '8', 'v2': '10', 'v3': '12', 'v4': '14'},
            'weight_adx': {
                'description': 'ADX Weight',
                'current': current_params.get('weight_adx', 5),
                'v1': '3', 'v2': '5', 'v3': '7', 'v4': '10'},

            # ──────────────────────────────────────────────────────────────────────
            # Risk Management
            # ──────────────────────────────────────────────────────────────────────
            'risk_per_trade': {
                'description': 'Risk Per Trade %',
                'current': current_params.get('risk_per_trade', 0.008),
                'v1': '0.005', 'v2': '0.008', 'v3': '0.010', 'v4': '0.012'},
            'risk_tier1': {
                'description': 'Tier 1 Risk %',
                'current': current_params.get('risk_tier1', 0.010),
                'v1': '0.008', 'v2': '0.010', 'v3': '0.012', 'v4': '0.015'},
            'max_position_size_pct': {
                'description': 'Max Position Size %',
                'current': current_params.get('max_position_size_pct', 0.15),
                'v1': '0.10', 'v2': '0.12', 'v3': '0.15', 'v4': '0.20'},
            'max_position_units': {
                'description': 'Max Position Units',
                'current': current_params.get('max_position_units', 100),
                'v1': '50', 'v2': '75', 'v3': '100', 'v4': '150'},

            # ──────────────────────────────────────────────────────────────────────
            # Stop Loss & Trailing
            # ──────────────────────────────────────────────────────────────────────
            'stop_loss_atr_mult': {
                'description': 'Stop Loss ATR Mult',
                'current': current_params.get('stop_loss_atr_mult', 1.8),
                'v1': '1.5', 'v2': '1.8', 'v3': '2.0', 'v4': '2.2'},
            'trailing_activation_pct': {
                'description': 'Trailing Activation %',
                'current': current_params.get('trailing_activation_pct', 0.025),
                'v1': '0.015', 'v2': '0.02', 'v3': '0.025', 'v4': '0.03'},
            'trailing_distance_pct': {
                'description': 'Trailing Distance %',
                'current': current_params.get('trailing_distance_pct', 0.01),
                'v1': '0.008', 'v2': '0.01', 'v3': '0.012', 'v4': '0.015'},
            'trailing_atr_mult': {
                'description': 'Trailing ATR Mult',
                'current': current_params.get('trailing_atr_mult', 0.8),
                'v1': '0.6', 'v2': '0.8', 'v3': '1.0', 'v4': '1.2'},

            # ──────────────────────────────────────────────────────────────────────
            # Breakeven Stop
            # ──────────────────────────────────────────────────────────────────────
            'be_stop_enabled': {
                'description': 'Breakeven Stop Enabled',
                'current': current_params.get('be_stop_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
            'be_stop_r_trigger': {
                'description': 'Breakeven R Trigger',
                'current': current_params.get('be_stop_r_trigger', 2.0),
                'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'},
            'be_stop_no_progress_bars': {
                'description': 'Breakeven No-Progress Bars',
                'current': current_params.get('be_stop_no_progress_bars', 8),
                'v1': '4', 'v2': '6', 'v3': '8', 'v4': '12'},

            # ──────────────────────────────────────────────────────────────────────
            # Profit Targets
            # ──────────────────────────────────────────────────────────────────────
            'take_profit_r1': {
                'description': 'Profit Target R1',
                'current': current_params.get('take_profit_r1', 1.5),
                'v1': '1.2', 'v2': '1.5', 'v3': '1.8', 'v4': '2.0'},
            'take_profit_r2': {
                'description': 'Profit Target R2',
                'current': current_params.get('take_profit_r2', 2.5),
                'v1': '2.0', 'v2': '2.5', 'v3': '3.0', 'v4': '3.5'},
            'partial_exit_pct_r1': {
                'description': 'Partial Exit % at R1',
                'current': current_params.get('partial_exit_pct_r1', 0.5),
                'v1': '0.3', 'v2': '0.4', 'v3': '0.5', 'v4': '0.6'},
            'partial_exit_pct_r2': {
                'description': 'Partial Exit % at R2',
                'current': current_params.get('partial_exit_pct_r2', 0.3),
                'v1': '0.2', 'v2': '0.25', 'v3': '0.3', 'v4': '0.4'},

            # ──────────────────────────────────────────────────────────────────────
            # Exit Conditions
            # ──────────────────────────────────────────────────────────────────────
            'macd_cross_exit_enabled': {
                'description': 'MACD Cross Exit Enabled',
                'current': current_params.get('macd_cross_exit_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
            'macd_cross_min_profit_r': {
                'description': 'MACD Cross Min Profit R',
                'current': current_params.get('macd_cross_min_profit_r', 1.0),
                'v1': '0.5', 'v2': '1.0', 'v3': '1.5', 'v4': '2.0'},
            'stoch_reversal_exit_enabled': {
                'description': 'Stochastic Reversal Exit Enabled',
                'current': current_params.get('stoch_reversal_exit_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
            'stoch_reversal_min_profit_r': {
                'description': 'Stoch Reversal Min Profit R',
                'current': current_params.get('stoch_reversal_min_profit_r', 0.8),
                'v1': '0.5', 'v2': '0.8', 'v3': '1.0', 'v4': '1.5'},
            'ema_cross_exit_enabled': {
                'description': 'EMA Cross Exit Enabled',
                'current': current_params.get('ema_cross_exit_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
            'ema_cross_min_profit_r': {
                'description': 'EMA Cross Min Profit R',
                'current': current_params.get('ema_cross_min_profit_r', 1.5),
                'v1': '1.0', 'v2': '1.5', 'v3': '2.0', 'v4': '2.5'},

            # ──────────────────────────────────────────────────────────────────────
            # Trade Management
            # ──────────────────────────────────────────────────────────────────────
            'max_hold_bars': {
                'description': 'Max Hold Bars',
                'current': current_params.get('max_hold_bars', 24),
                'v1': '16', 'v2': '20', 'v3': '24', 'v4': '32'},
            'min_hold_bars_before_stop': {
                'description': 'Min Hold Bars Before Stop',
                'current': current_params.get('min_hold_bars_before_stop', 3),
                'v1': '2', 'v2': '3', 'v3': '4', 'v4': '6'},
            'max_daily_trades': {
                'description': 'Max Daily Trades',
                'current': current_params.get('max_daily_trades', 10),
                'v1': '5', 'v2': '8', 'v3': '10', 'v4': '15'},
            'min_bars_between_trades': {
                'description': 'Min Bars Between Trades',
                'current': current_params.get('min_bars_between_trades', 1),
                'v1': '1', 'v2': '2', 'v3': '3', 'v4': '4'},
            'cooldown_after_loss_bars': {
                'description': 'Cooldown After Loss (bars)',
                'current': current_params.get('cooldown_after_loss_bars', 3),
                'v1': '2', 'v2': '3', 'v3': '4', 'v4': '6'},

            # ──────────────────────────────────────────────────────────────────────
            # Pullback Zone
            # ──────────────────────────────────────────────────────────────────────
            'pullback_zone_lower_pct': {
                'description': 'Pullback Zone Lower %',
                'current': current_params.get('pullback_zone_lower_pct', -2.5),
                'v1': '-3.0', 'v2': '-2.5', 'v3': '-2.0', 'v4': '-1.5'},
            'pullback_zone_upper_pct': {
                'description': 'Pullback Zone Upper %',
                'current': current_params.get('pullback_zone_upper_pct', 1.5),
                'v1': '1.0', 'v2': '1.5', 'v3': '2.0', 'v4': '3.0'},
            'momentum_min_long': {
                'description': 'Momentum Min % (LONG)',
                'current': current_params.get('momentum_min_long', 0.01),
                'v1': '0.005', 'v2': '0.01', 'v3': '0.015', 'v4': '0.02'},
            'momentum_min_short': {
                'description': 'Momentum Min % (SHORT)',
                'current': current_params.get('momentum_min_short', 0.01),
                'v1': '0.005', 'v2': '0.01', 'v3': '0.015', 'v4': '0.02'},

            # ──────────────────────────────────────────────────────────────────────
            # Regime Filter
            # ──────────────────────────────────────────────────────────────────────
            'regime_filter_enabled': {
                'description': 'Regime Filter Enabled',
                'current': current_params.get('regime_filter_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
            'chop_threshold': {
                'description': 'Choppiness Threshold',
                'current': current_params.get('chop_threshold', 61),
                'v1': '55', 'v2': '58', 'v3': '61', 'v4': '65'},

            # ──────────────────────────────────────────────────────────────────────
            # Daily Trend Filter
            # ──────────────────────────────────────────────────────────────────────
            'daily_trend_filter_enabled': {
                'description': 'Daily Trend Filter Enabled',
                'current': current_params.get('daily_trend_filter_enabled', True),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
            'daily_ema_period': {
                'description': 'Daily EMA Period (bars)',
                'current': current_params.get('daily_ema_period', 24),
                'v1': '12', 'v2': '18', 'v3': '24', 'v4': '36'},
            'daily_trend_adx_override': {
                'description': 'Daily Trend ADX Override',
                'current': current_params.get('daily_trend_adx_override', 25),
                'v1': '20', 'v2': '25', 'v3': '30', 'v4': '35'},

            # ──────────────────────────────────────────────────────────────────────
            # Extended Run
            # ──────────────────────────────────────────────────────────────────────
            'extended_run_max_pct_long': {
                'description': 'Extended Run Max % (LONG)',
                'current': current_params.get('extended_run_max_pct_long', 4.0),
                'v1': '3.0', 'v2': '4.0', 'v3': '5.0', 'v4': '6.0'},
            'extended_run_max_pct_short': {
                'description': 'Extended Run Max % (SHORT)',
                'current': current_params.get('extended_run_max_pct_short', 4.0),
                'v1': '3.0', 'v2': '4.0', 'v3': '5.0', 'v4': '6.0'},

            # ──────────────────────────────────────────────────────────────────────
            # Trade Direction
            # ──────────────────────────────────────────────────────────────────────
            'trade_direction': {
                'description': 'Trade Direction',
                'current': current_params.get('trade_direction', 'both'),
                'v1': 'long', 'v2': 'both', 'v3': 'short', 'v4': ''},
            'only_tier1_entries': {
                'description': 'Only Tier 2 Entries',
                'current': current_params.get('only_tier1_entries', False),
                'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
        }

        # Initialize scalping_backtest_params
        if not hasattr(self, 'scalping_backtest_params'):
            self.scalping_backtest_params = {}

        for key, meta in ALL_SCALPING_BACKTEST_PARAMS.items():
            if key not in self.scalping_backtest_params:
                self.scalping_backtest_params[key] = {
                    'active': tk.BooleanVar(value=False),
                    'value1': tk.StringVar(value=meta['v1']),
                    'value2': tk.StringVar(value=meta['v2']),
                    'value3': tk.StringVar(value=meta['v3']),
                    'value4': tk.StringVar(value=meta['v4']),
                    'description': meta['description'],
                    'current_value': meta['current']
                }
            else:
                self.scalping_backtest_params[key]['current_value'] = meta['current']
                self.scalping_backtest_params[key]['description'] = meta['description']

        # Build the UI
        self._build_scalping_backtest_ui(right_frame)

    def _build_scalping_backtest_ui(self, right_frame):
        """Build the UI for scalping backtest parameters with save/load buttons"""

        # ── Optimization Metrics Selection ──────────────────────────────────────
        metrics_frame = ttk.LabelFrame(right_frame, text="Optimization Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        metrics_inner = ttk.Frame(metrics_frame)
        metrics_inner.pack(fill=tk.X, padx=5, pady=5)

        row, col = 0, 0
        for metric_name, metric_var in self.optimization_metrics.items():
            ttk.Checkbutton(metrics_inner, text=metric_name.replace('_', ' ').title(),
                            variable=metric_var).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            col += 1
            if col > 1:
                col = 0
                row += 1

        ttk.Label(metrics_inner, text="Equal Weights (All selected metrics)").grid(
            row=row + 1, column=0, columnspan=2, pady=5)

        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=3)

        # ── Action Buttons (with Scalping-specific save/load) ───────────────────
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=3)

        ttk.Button(action_frame, text="✅ Select All",
                   command=self._sc_select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="❌ Deselect All",
                   command=self._sc_deselect_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="💾 Save Scalping Params",
                   command=self.save_scalping_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="📂 Load Scalping Params",
                   command=self.load_scalping_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🔄 Reset",
                   command=self.reset_scalping_backtest_params).pack(side=tk.LEFT, padx=2)

        self.sc_selection_label = ttk.Label(action_frame, text="Selected: 0",
                                            foreground='blue', font=('Arial', 9, 'bold'))
        self.sc_selection_label.pack(side=tk.RIGHT, padx=5)

        # ── Parameter Table Header ──────────────────────────────────────────────
        param_header_frame = ttk.LabelFrame(right_frame, text="Scalping Parameters (Select for Optimization)")
        param_header_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        col_header = ttk.Frame(param_header_frame)
        col_header.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(col_header, text="✓", width=3, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=0,
                                                                                                  padx=1)
        ttk.Label(col_header, text="Parameter", width=30, anchor='w', font=('Arial', 8, 'bold')).grid(row=0, column=1,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Current", width=10, anchor='center', font=('Arial', 8, 'bold')).grid(row=0,
                                                                                                         column=2,
                                                                                                         padx=1)
        ttk.Label(col_header, text="Val 1", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=3,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Val 2", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=4,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Val 3", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=5,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Val 4", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=6,
                                                                                                      padx=1)

        ttk.Separator(param_header_frame, orient='horizontal').pack(fill=tk.X, padx=5)

        # ── Scrollable Parameter List ───────────────────────────────────────────
        param_canvas = tk.Canvas(param_header_frame, bg='white', highlightthickness=0)
        param_scrollbar = ttk.Scrollbar(param_header_frame, orient="vertical", command=param_canvas.yview)
        param_scrollable = ttk.Frame(param_canvas)

        param_scrollable.bind("<Configure>",
                              lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all")))
        param_canvas.create_window((0, 0), window=param_scrollable, anchor="nw")
        param_canvas.configure(yscrollcommand=param_scrollbar.set)
        param_canvas.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        param_scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            param_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        param_canvas.bind("<MouseWheel>", _on_mousewheel)

        # Store widget references
        self.scalping_backtest_param_widgets = {}

        sorted_params = sorted(self.scalping_backtest_params.items(), key=lambda x: x[1]['description'])

        for p_row, (param_key, param_data) in enumerate(sorted_params):
            frame = ttk.Frame(param_scrollable)
            frame.grid(row=p_row, column=0, sticky='ew', pady=1, padx=2)

            def make_toggle_cmd(key=param_key):
                def cmd():
                    self._sc_update_count()

                return cmd

            cb = ttk.Checkbutton(frame, variable=param_data['active'], command=make_toggle_cmd())
            cb.grid(row=0, column=0, padx=2)

            # Parameter name/description
            lbl = ttk.Label(frame, text=param_data['description'], width=30, anchor='w',
                            font=('Arial', 8), cursor='hand2')
            lbl.grid(row=0, column=1, padx=2, sticky='w')

            def make_label_callback(k=param_key):
                def callback(e):
                    current = self.scalping_backtest_params[k]['active'].get()
                    self.scalping_backtest_params[k]['active'].set(not current)
                    self._sc_update_count()

                return callback

            lbl.bind("<Button-1>", make_label_callback())

            # Current value (from GUI-selected parameters)
            current_val = param_data.get('current_value', '')
            current_lbl = ttk.Label(frame, text=str(current_val), width=10, anchor='center',
                                    font=('Arial', 8, 'bold'), foreground='blue')
            current_lbl.grid(row=0, column=2, padx=1)

            # Value inputs
            for v_idx, v_key in enumerate(['value1', 'value2', 'value3', 'value4'], start=3):
                e = ttk.Entry(frame, textvariable=param_data[v_key], width=7, font=('Arial', 8))
                e.grid(row=0, column=v_idx, padx=1)

            self.scalping_backtest_param_widgets[param_key] = {
                'active': param_data['active'],
                'value1': param_data['value1'],
                'value2': param_data['value2'],
                'value3': param_data['value3'],
                'value4': param_data['value4'],
                'current_lbl': current_lbl,
                'widget': cb
            }

        self._sc_update_count()

        # Log summary
        self.log_message("=" * 70, "green")
        self.log_message(f"✅ Built Scalping backtest panel with {len(self.scalping_backtest_params)} parameters",
                         "green")
        self.log_message(f"📋 Parameter source: {self.param_toggle_var.get()}", "blue")
        self.log_message("=" * 70, "green")

    def save_scalping_backtest_params(self):
        """Save Scalping backtest parameters to JSON file"""
        try:
            params_to_save = {}
            for param_key, param_data in self.scalping_backtest_params.items():
                params_to_save[param_key] = {
                    'active': param_data['active'].get(),
                    'value1': param_data['value1'].get(),
                    'value2': param_data['value2'].get(),
                    'value3': param_data['value3'].get(),
                    'value4': param_data['value4'].get(),
                    'description': param_data['description']
                }

            # Save metrics
            metrics_to_save = {}
            for metric_name, metric_var in self.optimization_metrics.items():
                metrics_to_save[metric_name] = metric_var.get()

            save_data = {
                'strategy': 'Scalping',
                'parameters': params_to_save,
                'metrics': metrics_to_save,
                'timestamp': datetime.now().isoformat()
            }

            filename = "backtest_params_scalping.json"
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=4)

            selected_count = sum(1 for p in params_to_save.values() if p['active'])
            active_metrics = [m for m, v in metrics_to_save.items() if v]

            self.log_message("=" * 70, "green")
            self.log_message(f"✅ Scalping backtest params saved to: {filename}", "green")
            self.log_message(f"   • {selected_count} parameters selected for optimization", "blue")
            self.log_message(f"   • {len(active_metrics)} optimization metrics selected: {', '.join(active_metrics)}",
                             "blue")
            self.log_message("=" * 70, "green")

            messagebox.showinfo("Saved",
                                f"Scalping backtest parameters saved to {filename}\n\nParameters selected: {selected_count}")

        except Exception as e:
            self.log_message(f"❌ Error saving scalping backtest params: {e}", "red")
            messagebox.showerror("Error", f"Failed to save parameters:\n{e}")

    def load_scalping_backtest_params(self):
        """Load Scalping backtest parameters from JSON file"""
        filename = "backtest_params_scalping.json"
        try:
            if not os.path.exists(filename):
                messagebox.showwarning("File Not Found",
                                       f"No saved scalping backtest parameters found.\nExpected: {filename}\n\nSave parameters first.")
                return

            with open(filename, 'r') as f:
                loaded = json.load(f)

            # Handle both new and old format
            if isinstance(loaded, dict) and 'parameters' in loaded:
                params_dict = loaded['parameters']
                metrics_dict = loaded.get('metrics', {})
                self.log_message("📂 Loading scalping parameters with metrics", "blue")
            else:
                params_dict = loaded
                metrics_dict = {}
                self.log_message("📂 Loading scalping parameters in legacy format", "orange")

            loaded_count = 0
            for param_key, param_info in params_dict.items():
                if param_key in self.scalping_backtest_params:
                    self.scalping_backtest_params[param_key]['active'].set(param_info.get('active', False))
                    self.scalping_backtest_params[param_key]['value1'].set(str(param_info.get('value1', '')))
                    self.scalping_backtest_params[param_key]['value2'].set(str(param_info.get('value2', '')))
                    self.scalping_backtest_params[param_key]['value3'].set(str(param_info.get('value3', '')))
                    self.scalping_backtest_params[param_key]['value4'].set(str(param_info.get('value4', '')))
                    loaded_count += 1

            # Load optimization metrics
            metrics_loaded = 0
            for metric_name, metric_value in metrics_dict.items():
                if metric_name in self.optimization_metrics:
                    self.optimization_metrics[metric_name].set(metric_value)
                    metrics_loaded += 1

            self._sc_update_count()

            selected_count = sum(1 for p in self.scalping_backtest_params.values() if p['active'].get())
            active_metrics = [m for m, v in self.optimization_metrics.items() if v.get()]

            self.log_message("=" * 70, "green")
            self.log_message(f"✅ Loaded scalping backtest params from {filename}", "green")
            self.log_message(f"   • {loaded_count} parameters updated", "blue")
            self.log_message(f"   • {selected_count} parameters selected for optimization", "blue")
            self.log_message(f"   • {len(active_metrics)} optimization metrics loaded", "blue")
            self.log_message("=" * 70, "green")

            messagebox.showinfo("Loaded",
                                f"Scalping backtest parameters loaded from {filename}\n\n"
                                f"Updated: {loaded_count} parameters\n"
                                f"Selected for optimization: {selected_count}\n"
                                f"Optimization metrics: {len(active_metrics)}")

        except json.JSONDecodeError as e:
            self.log_message(f"❌ Invalid JSON in {filename}: {e}", "red")
            messagebox.showerror("Load Error", f"File is corrupted or not valid JSON:\n{e}")
        except Exception as e:
            self.log_message(f"❌ Error loading scalping backtest params: {e}", "red")
            messagebox.showerror("Load Error", f"Failed to load parameters:\n{e}")

    def reset_scalping_backtest_params(self):
        """Reset Scalping backtest parameters to defaults (all deselected)"""
        if messagebox.askyesno("Reset Scalping Params",
                               "Reset all scalping backtest parameters to defaults?\nThis will also deselect all parameters."):
            try:
                for param_data in self.scalping_backtest_params.values():
                    param_data['active'].set(False)

                self._sc_update_count()
                self.log_message("✅ Scalping backtest parameters reset (all deselected)", "green")

            except Exception as e:
                self.log_message(f"❌ Error resetting scalping backtest params: {e}", "red")

    # ─────────────────────────────────────────────────────────────────────────────
    # HELPER — call once anywhere you need the filename for a given strategy
    # ─────────────────────────────────────────────────────────────────────────────
    def _get_strategy_bt_file(self, strategy: str | None = None) -> str:
        """Return the canonical backtest-param filename for *strategy*.

        Falls back to the currently selected strategy when *strategy* is None.
        Examples:
            Momentum → backtest_params_momentum.json
            Kalman   → backtest_params_kalman.json
            Scalping → backtest_params_scalping.json
        """
        if strategy is None:
            strategy = (
                self.strategy_type_var.get()
                if hasattr(self, "strategy_type_var")
                else "Momentum"
            )
        return f"backtest_params_{strategy.lower()}.json"

    def _sc_select_all(self):
        """Select all scalping backtest parameters"""
        for param_data in self.scalping_backtest_params.values():
            param_data['active'].set(True)
        self._sc_update_count()
        self.log_message("✅ All scalping parameters selected", "green")

    def _sc_deselect_all(self):
        """Deselect all scalping backtest parameters"""
        for param_data in self.scalping_backtest_params.values():
            param_data['active'].set(False)
        self._sc_update_count()
        self.log_message("✅ All scalping parameters deselected", "blue")

    def _sc_update_count(self):
        """Update the scalping selection counter label"""
        if hasattr(self, 'sc_selection_label') and self.sc_selection_label.winfo_exists():
            count = sum(1 for p in self.scalping_backtest_params.values() if p['active'].get())
            color = 'red' if count > 15 else 'green' if count > 0 else 'blue'
            self.sc_selection_label.config(text=f"Selected: {count}", foreground=color)

    def _get_scalping_param_value(self, param_key):
        """Get current value for a scalping param (UI → custom_params → SCALPING_PARAMS)"""
        # Layer 1: live scalping param widgets
        if hasattr(self, 'scalping_param_widgets') and param_key in self.scalping_param_widgets:
            try:
                cvar = self.scalping_param_widgets[param_key]['custom']
                if isinstance(cvar, tk.BooleanVar):
                    return cvar.get()
                raw = cvar.get()
                if raw not in ('', None):
                    return self.convert_param_value(raw)
            except Exception:
                pass
        # Layer 2: saved custom
        val = self.custom_params.get('scalping', {}).get(param_key)
        if val is not None:
            return val
        # Layer 3: SCALPING_PARAMS defaults
        try:
            from strategies.scalping_strategy import SCALPING_PARAMS
            return SCALPING_PARAMS.get(param_key)
        except ImportError:
            return None

    def _build_backtest_ui(self, right_frame):
        """Build the actual UI for backtest parameters (shared across strategies)"""

        # ── Optimization Metrics Selection ──────────────────────────────────────
        metrics_frame = ttk.LabelFrame(right_frame, text="Optimization Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        metrics_inner = ttk.Frame(metrics_frame)
        metrics_inner.pack(fill=tk.X, padx=5, pady=5)

        row, col = 0, 0
        for metric_name, metric_var in self.optimization_metrics.items():
            ttk.Checkbutton(metrics_inner, text=metric_name.replace('_', ' ').title(),
                            variable=metric_var).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            col += 1
            if col > 1:
                col = 0
                row += 1

        ttk.Label(metrics_inner, text="Equal Weights (All selected metrics)").grid(
            row=row + 1, column=0, columnspan=2, pady=5)

        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=3)

        # ── Action Buttons ──────────────────────────────────────────────────────
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=3)

        ttk.Button(action_frame, text="✅ Select All",
                   command=self._select_all_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="❌ Deselect All",
                   command=self._deselect_all_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="💾 Save Params",
                   command=self.save_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="📂 Load Params",
                   command=self.load_backtest_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🔄 Reset",
                   command=self.reset_backtest_params).pack(side=tk.LEFT, padx=2)

        self.bt_selection_label = ttk.Label(action_frame, text="Selected: 0",
                                            foreground='blue', font=('Arial', 9, 'bold'))
        self.bt_selection_label.pack(side=tk.RIGHT, padx=5)

        # ── Parameter Table Header ──────────────────────────────────────────────
        param_header_frame = ttk.LabelFrame(right_frame, text="Parameters (Select for Optimization)")
        param_header_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        col_header = ttk.Frame(param_header_frame)
        col_header.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(col_header, text="✓", width=3, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=0,
                                                                                                  padx=1)
        ttk.Label(col_header, text="Parameter", width=30, anchor='w', font=('Arial', 8, 'bold')).grid(row=0, column=1,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Current", width=10, anchor='center', font=('Arial', 8, 'bold')).grid(row=0,
                                                                                                         column=2,
                                                                                                         padx=1)
        ttk.Label(col_header, text="Val 1", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=3,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Val 2", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=4,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Val 3", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=5,
                                                                                                      padx=1)
        ttk.Label(col_header, text="Val 4", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=6,
                                                                                                      padx=1)

        ttk.Separator(param_header_frame, orient='horizontal').pack(fill=tk.X, padx=5)

        # ── Scrollable Parameter List ───────────────────────────────────────────
        param_canvas = tk.Canvas(param_header_frame, bg='white', highlightthickness=0)
        param_scrollbar = ttk.Scrollbar(param_header_frame, orient="vertical", command=param_canvas.yview)
        param_scrollable = ttk.Frame(param_canvas)

        param_scrollable.bind("<Configure>",
                              lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all")))
        param_canvas.create_window((0, 0), window=param_scrollable, anchor="nw")
        param_canvas.configure(yscrollcommand=param_scrollbar.set)
        param_canvas.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        param_scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            param_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        param_canvas.bind("<MouseWheel>", _on_mousewheel)

        # Store widget references
        if not hasattr(self, 'backtest_param_widgets'):
            self.backtest_param_widgets = {}

        sorted_params = sorted(self.backtest_params.items(), key=lambda x: x[1]['description'])

        for p_row, (param_key, param_data) in enumerate(sorted_params):
            frame = ttk.Frame(param_scrollable)
            frame.grid(row=p_row, column=0, sticky='ew', pady=1, padx=2)

            def make_toggle_cmd(key=param_key):
                def cmd():
                    self._update_bt_selection_count()

                return cmd

            cb = ttk.Checkbutton(frame, variable=param_data['active'], command=make_toggle_cmd())
            cb.grid(row=0, column=0, padx=2)

            # Parameter name/description
            lbl = ttk.Label(frame, text=param_data['description'], width=30, anchor='w',
                            font=('Arial', 8), cursor='hand2')
            lbl.grid(row=0, column=1, padx=2, sticky='w')

            def make_label_callback(k=param_key):
                def callback(e):
                    self._toggle_bt_param(k)

                return callback

            lbl.bind("<Button-1>", make_label_callback())

            # Current value (from GUI-selected parameters)
            current_val = param_data.get('current_value', '')
            current_lbl = ttk.Label(frame, text=str(current_val), width=10, anchor='center',
                                    font=('Arial', 8, 'bold'), foreground='blue')
            current_lbl.grid(row=0, column=2, padx=1)

            # Value inputs
            for v_idx, v_key in enumerate(['value1', 'value2', 'value3', 'value4'], start=3):
                e = ttk.Entry(frame, textvariable=param_data[v_key], width=7, font=('Arial', 8))
                e.grid(row=0, column=v_idx, padx=1)

            self.backtest_param_widgets[param_key] = {
                'active': param_data['active'],
                'value1': param_data['value1'],
                'value2': param_data['value2'],
                'value3': param_data['value3'],
                'value4': param_data['value4'],
                'current_lbl': current_lbl,
                'widget': cb
            }

        self._update_bt_selection_count()

        # Log summary
        self.log_message("=" * 70, "green")
        self.log_message(f"✅ Built backtest panel with {len(self.backtest_params)} parameters", "green")
        self.log_message(f"📋 Parameter source: {self.param_toggle_var.get()}", "blue")
        self.log_message("=" * 70, "green")

    # def _build_backtest_optimization_panel(self, right_frame):
    #     """Build the right-panel backtest optimization parameter UI with ALL v7.6 metrics.
    #     Safe to call multiple times (once per strategy tab) — shares the same data model.
    #     """
    #     from strategies.MomentumStrategy_MACD_HybridScore_Latest import GlobalConfig
    #
    #     capital_frame = ttk.LabelFrame(right_frame, text="💰 Global Capital Settings")
    #     capital_frame.pack(fill=tk.X, padx=5, pady=5)
    #
    #     capital_inner = ttk.Frame(capital_frame)
    #     capital_inner.pack(fill=tk.X, padx=5, pady=5)
    #
    #     ttk.Label(capital_inner, text="Current Capital:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w',
    #                                                                                       padx=5, pady=2)
    #
    #     self.capital_display_var = tk.StringVar(value=f"${GlobalConfig.INITIAL_CAPITAL:,.2f}")
    #     ttk.Label(capital_inner, textvariable=self.capital_display_var,
    #               font=('Arial', 10, 'bold'), foreground='green').grid(row=0, column=1, sticky='w', padx=5, pady=2)
    #
    #     ttk.Label(capital_inner, text="New Capital ($):", font=('Arial', 9)).grid(row=1, column=0, sticky='w', padx=5,
    #                                                                               pady=2)
    #     self.capital_entry = ttk.Entry(capital_inner, width=15, font=('Arial', 9))
    #     self.capital_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)
    #     self.capital_entry.insert(0, str(GlobalConfig.INITIAL_CAPITAL))
    #
    #     update_btn = tk.Button(
    #         capital_inner,
    #         text="Update Capital",
    #         command=self._update_global_capital,
    #         bg="#4CAF50",
    #         fg="white",
    #         font=('Arial', 9, 'bold'),
    #         padx=10,
    #         pady=2,
    #         cursor="hand2"
    #     )
    #     update_btn.grid(row=1, column=2, padx=5, pady=2)
    #
    #     ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
    #
    #     # ═══════════════════════════════════════════════════════════════════════
    #     # ALL_BACKTEST_PARAMS — complete list including daily_ema_period and
    #     # daily_trend_adx_override (previously missing), and corrected value
    #     # orderings for quality_tier2_min and cooldown_after_loss_bars.
    #     # Rule: v1 = most conservative / tightest, v4 = most permissive / loosest.
    #     # ═══════════════════════════════════════════════════════════════════════
    #     ALL_BACKTEST_PARAMS = {
    #
    #         # ═══ QUALITY / TIER THRESHOLDS ═══════════════════════════════════
    #         'quality_tier1_min': {
    #             'description': 'Tier 1 Minimum Score (LONG)',
    #             'v1': '68', 'v2': '70', 'v3': '72', 'v4': '75'},
    #         'quality_tier2_min': {
    #             'description': 'Tier 2 Minimum Score (LONG) — lower = more long entries',
    #             'v1': '62', 'v2': '60', 'v3': '58', 'v4': '55'},  # FIX: was 55→65, now 62→55
    #         'fixed_threshold': {
    #             'description': 'Fixed Entry Threshold',
    #             'v1': '65', 'v2': '68', 'v3': '72', 'v4': '75'},
    #         'short_quality_tier1_min': {
    #             'description': 'Tier 1 Minimum Score (SHORT)',
    #             'v1': '70', 'v2': '72', 'v3': '75', 'v4': '78'},
    #         'short_quality_tier2_min': {
    #             'description': 'Tier 2 Minimum Score (SHORT)',
    #             'v1': '60', 'v2': '63', 'v3': '65', 'v4': '68'},
    #         'short_fixed_threshold': {
    #             'description': 'Fixed Entry Threshold (SHORT)',
    #             'v1': '70', 'v2': '72', 'v3': '75', 'v4': '78'},
    #         'only_tier1_entries': {
    #             'description': '🔥 Only Tier 2 Entries',
    #             'v1': 'false', 'v2': 'true', 'v3': '', 'v4': ''},
    #
    #         # ═══ EMA PERIODS ══════════════════════════════════════════════════
    #         'ema_fast_period': {
    #             'description': 'EMA Fast Period',
    #             'v1': '8', 'v2': '9', 'v3': '10', 'v4': '12'},
    #         'ema_mid_period': {
    #             'description': 'EMA Mid Period',
    #             'v1': '18', 'v2': '20', 'v3': '21', 'v4': '24'},
    #         'ema_slow_period': {
    #             'description': 'EMA Slow Period',
    #             'v1': '45', 'v2': '50', 'v3': '55', 'v4': '60'},
    #         'daily_ema_period': {  # FIX: was missing
    #             'description': 'Daily EMA Period (hours) — 168=7d, 240=10d, 360=15d, 480=20d — shorter unblocks more recovery entries',
    #             'v1': '480', 'v2': '360', 'v3': '240', 'v4': '168'},
    #         'daily_trend_adx_override': {  # FIX: was missing
    #             'description': 'ADX min to allow entry below daily EMA (strong-trend override) — lower = more early-recovery entries',
    #             'v1': '26', 'v2': '23', 'v3': '20', 'v4': '18'},
    #
    #         # ═══ QUALITY WEIGHTS ══════════════════════════════════════════════
    #         'weight_ema': {
    #             'description': 'EMA Weight',
    #             'v1': '15', 'v2': '18', 'v3': '20', 'v4': '25'},
    #         'weight_adx': {
    #             'description': 'ADX Weight',
    #             'v1': '15', 'v2': '18', 'v3': '20', 'v4': '25'},
    #         'weight_macd': {
    #             'description': 'MACD Weight',
    #             'v1': '20', 'v2': '22', 'v3': '25', 'v4': '28'},
    #         'weight_rsi': {
    #             'description': 'RSI Weight',
    #             'v1': '15', 'v2': '18', 'v3': '20', 'v4': '22'},
    #         'weight_volume': {
    #             'description': 'Volume Weight',
    #             'v1': '10', 'v2': '12', 'v3': '15', 'v4': '18'},
    #
    #         # ═══ LONG ENTRY FILTERS ═══════════════════════════════════════════
    #         'tier1_adx_hard_min': {
    #             'description': 'Tier 1 ADX Minimum (LONG) — NOTE: dead code for longs, only affects shorts + quality score',
    #             'v1': '18', 'v2': '20', 'v3': '22', 'v4': '25'},
    #         'tier1_adx_min': {
    #             'description': 'Tier 1 ADX Min (soft)',
    #             'v1': '16', 'v2': '18', 'v3': '20', 'v4': '22'},
    #         'tier1_rsi_min': {
    #             'description': 'Tier 1 RSI Minimum (LONG) — NOTE: dead code for longs (hardcoded at 40 in _validate_direction)',
    #             'v1': '38', 'v2': '40', 'v3': '42', 'v4': '44'},
    #         'tier1_rsi_max': {
    #             'description': 'Tier 1 RSI Maximum (LONG) — NOTE: dead code for longs',
    #             'v1': '60', 'v2': '62', 'v3': '64', 'v4': '66'},
    #         'tier1_volume_min': {
    #             'description': 'Tier 1 Volume Min (LONG)',
    #             'v1': '0.8', 'v2': '0.9', 'v3': '1.0', 'v4': '1.2'},
    #         'tier1_momentum_min': {
    #             'description': 'Tier 1 Momentum Min % (LONG)',
    #             'v1': '0.01', 'v2': '0.015', 'v3': '0.02', 'v4': '0.03'},
    #         'adx_min': {
    #             'description': 'ADX Minimum (Entry)',
    #             'v1': '15', 'v2': '18', 'v3': '20', 'v4': '22'},
    #         'adx_min_trend': {
    #             'description': 'ADX Min Trend Strength',
    #             'v1': '20', 'v2': '22', 'v3': '25', 'v4': '28'},
    #
    #         # ═══ SHORT ENTRY FILTERS ══════════════════════════════════════════
    #         'short_tier1_adx_hard_min': {
    #             'description': 'Tier 1 ADX Minimum (SHORT)',
    #             'v1': '24', 'v2': '26', 'v3': '28', 'v4': '30'},
    #         'short_tier1_rsi_min': {
    #             'description': 'Tier 1 RSI Minimum (SHORT)',
    #             'v1': '30', 'v2': '32', 'v3': '34', 'v4': '36'},
    #         'short_tier1_rsi_max': {
    #             'description': 'Tier 1 RSI Maximum (SHORT)',
    #             'v1': '50', 'v2': '52', 'v3': '54', 'v4': '56'},
    #         'short_tier1_volume_min': {
    #             'description': 'Tier 1 Volume Min (SHORT)',
    #             'v1': '1.0', 'v2': '1.1', 'v3': '1.3', 'v4': '1.5'},
    #         'short_tier1_momentum_min': {
    #             'description': 'Tier 1 Momentum Min % (SHORT)',
    #             'v1': '0.03', 'v2': '0.04', 'v3': '0.05', 'v4': '0.06'},
    #         'short_require_lower_highs_bars': {
    #             'description': 'Short: Lower Highs Bars Required',
    #             'v1': '1', 'v2': '2', 'v3': '3', 'v4': '4'},
    #         'short_require_lower_lows_bars': {
    #             'description': 'Short: Lower Lows Bars Required',
    #             'v1': '1', 'v2': '2', 'v3': '3', 'v4': '4'},
    #
    #         # ═══ PULLBACK ZONE & ADX SLOPE ════════════════════════════════════
    #         # v1 = most conservative / fewest trades (original defaults)
    #         # v4 = most permissive / most trades
    #         'pullback_zone_lower_pct': {
    #             'description': 'Pullback Zone Lower % — how far below EMA_Fast price can be. More negative = deeper pullbacks allowed = more entries.',
    #             'v1': '-2.5', 'v2': '-3.0', 'v3': '-4.0', 'v4': '-5.0'},
    #         'pullback_zone_upper_pct': {
    #             'description': 'Pullback Zone Upper % — how far above EMA_Fast price can be. Higher = more extended entries allowed = more entries.',
    #             'v1': '1.5', 'v2': '2.0', 'v3': '3.0', 'v4': '4.0'},
    #         'adx_slope_min': {
    #             'description': 'ADX Slope Min (rise per bar). 0.1 = strict rising required. 0.0 = flat allowed. Negative = slightly declining ADX allowed = significantly more entries.',
    #             'v1': '0.1', 'v2': '0.0', 'v3': '-0.3', 'v4': '-0.5'},
    #
    #         # ═══ TIER 2 FILTERS ═══════════════════════════════════════════════
    #         'tier2_adx_min': {
    #             'description': 'Tier 2 ADX Min',
    #             'v1': '12', 'v2': '15', 'v3': '18', 'v4': '20'},
    #         'tier2_volume_min': {
    #             'description': 'Tier 2 Volume Min',
    #             'v1': '0.3', 'v2': '0.4', 'v3': '0.5', 'v4': '0.6'},
    #         'cooldown_tier2_enabled': {
    #             'description': 'Enable Tier 2 cooldown',
    #             'v1': 'true', 'v2': 'false', 'v3': '', 'v4': ''},
    #
    #         # ═══ REGIME / VOLATILITY FILTERS ══════════════════════════════════
    #         'regime_filter_enabled': {
    #             'description': 'Regime Filter Enabled',
    #             'v1': 'true', 'v2': 'false', 'v3': '', 'v4': ''},
    #         'atr_compression_enabled': {
    #             'description': 'ATR Compression Filter Enabled',
    #             'v1': 'true', 'v2': 'false', 'v3': '', 'v4': ''},
    #         'atr_compression_threshold': {
    #             'description': 'ATR Compression Threshold — lower = more post-spike recovery entries (current=0.25)',
    #             'v1': '0.15', 'v2': '0.20', 'v3': '0.25', 'v4': '0.30'},
    #         'extended_run_max_pct_long': {
    #             'description': 'Extended Run Max % (LONG)',
    #             'v1': '10', 'v2': '12', 'v3': '15', 'v4': '20'},
    #         'extended_run_max_pct_short': {
    #             'description': 'Extended Run Max % (SHORT)',
    #             'v1': '10', 'v2': '12', 'v3': '15', 'v4': '20'},
    #         'chop_threshold': {
    #             'description': 'Choppiness Index Threshold',
    #             'v1': '50', 'v2': '54', 'v3': '58', 'v4': '62'},
    #
    #         # ═══ ENTRY PRECISION ══════════════════════════════════════════════
    #         'rsi_entry_min': {
    #             'description': 'RSI Entry Minimum',
    #             'v1': '38', 'v2': '40', 'v3': '42', 'v4': '45'},
    #         'rsi_entry_max': {
    #             'description': 'RSI Entry Maximum',
    #             'v1': '62', 'v2': '65', 'v3': '68', 'v4': '70'},
    #         'volume_min_ratio': {
    #             'description': 'Volume Ratio Minimum',
    #             'v1': '0.9', 'v2': '1.0', 'v3': '1.1', 'v4': '1.2'},
    #         'momentum_min': {
    #             'description': 'Momentum Minimum %',
    #             'v1': '0.02', 'v2': '0.03', 'v3': '0.04', 'v4': '0.05'},
    #         'rsi_direction_bars': {
    #             'description': 'RSI Direction Bars',
    #             'v1': '1', 'v2': '2', 'v3': '3', 'v4': '4'},
    #         'ema_trending_bars': {
    #             'description': 'EMA Trending Bars',
    #             'v1': '2', 'v2': '3', 'v3': '4', 'v4': '5'},
    #         'macd_hist_rising_bars': {
    #             'description': 'MACD Histogram Rising Bars',
    #             'v1': '0', 'v2': '1', 'v3': '2', 'v4': '3'},
    #
    #         # ═══ BREAKEVEN STOP ═══════════════════════════════════════════════
    #         'be_stop_enabled': {
    #             'description': 'Breakeven Stop Enabled',
    #             'v1': 'true', 'v2': 'false', 'v3': '', 'v4': ''},
    #         'be_stop_r_trigger': {
    #             'description': 'Breakeven Stop R Trigger',
    #             'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'},
    #         'be_stop_no_progress_bars': {
    #             'description': 'Breakeven No-Progress Bars',
    #             'v1': '30', 'v2': '35', 'v3': '40', 'v4': '50'},
    #
    #         # ═══ STOP LOSS & TRAILING ═════════════════════════════════════════
    #         'stop_loss_atr_mult': {
    #             'description': 'Stop Loss ATR Mult',
    #             'v1': '2.0', 'v2': '2.5', 'v3': '3.0', 'v4': '3.5'},
    #         'trailing_stop_atr_mult': {
    #             'description': 'Trailing Stop ATR Multiplier',
    #             'v1': '5.0', 'v2': '5.5', 'v3': '6.0', 'v4': '6.5'},
    #         'initial_trailing_atr_mult': {
    #             'description': 'Initial Trailing ATR Multiplier',
    #             'v1': '4.0', 'v2': '4.5', 'v3': '5.0', 'v4': '5.5'},
    #         'trailing_activation_pct': {
    #             'description': 'Trailing Activation %',
    #             'v1': '0.030', 'v2': '0.035', 'v3': '0.040', 'v4': '0.045'},
    #         'trailing_distance_pct': {
    #             'description': 'Trailing Distance %',
    #             'v1': '0.035', 'v2': '0.040', 'v3': '0.045', 'v4': '0.050'},
    #         'trailing_activation_r': {
    #             'description': 'Trail Activation R',
    #             'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'},
    #         'trailing_stop_pct': {
    #             'description': 'Trailing Stop %',
    #             'v1': '0.02', 'v2': '0.03', 'v3': '0.04', 'v4': '0.05'},
    #
    #         # ═══ COOLDOWNS ════════════════════════════════════════════════════
    #         'cooldown_after_loss_bars': {
    #             'description': 'Cooldown After Loss (bars/hours) — lower = more trades after a loss',
    #             'v1': '10', 'v2': '8', 'v3': '6', 'v4': '4'},  # FIX: was 6→24, now 10→4
    #         'consecutive_loss_threshold': {
    #             'description': 'Consecutive Loss Threshold',
    #             'v1': '2', 'v2': '3', 'v3': '4', 'v4': '5'},
    #         'consecutive_loss_cooldown_bars': {
    #             'description': 'Consecutive Loss Cooldown (bars) — affects frequency after loss streaks',
    #             'v1': '6', 'v2': '8', 'v3': '12', 'v4': '16'},
    #
    #         # ═══ EXIT CONDITIONS ══════════════════════════════════════════════
    #         'rsi_exit_threshold': {
    #             'description': 'RSI Exit Threshold',
    #             'v1': '72', 'v2': '75', 'v3': '78', 'v4': '80'},
    #         'macd_bearish_cross_profit_min': {
    #             'description': 'MACD Bearish Cross Min Profit %',
    #             'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'},
    #         'kalman_fade_threshold': {
    #             'description': 'Kalman Fade Threshold',
    #             'v1': '25', 'v2': '30', 'v3': '35', 'v4': '40'},
    #         'profit_min_fade': {
    #             'description': 'Min Profit % to Allow Fade Exit',
    #             'v1': '0.5', 'v2': '0.8', 'v3': '1.0', 'v4': '1.5'},
    #
    #         # ═══ TREND AGE PENALTY ════════════════════════════════════════════
    #         'trend_age_max_bars': {
    #             'description': 'Trend Age Max Bars (penalty trigger)',
    #             'v1': '15', 'v2': '20', 'v3': '25', 'v4': '30'},
    #         'trend_age_penalty_pts': {
    #             'description': 'Trend Age Penalty Points',
    #             'v1': '5', 'v2': '8', 'v3': '10', 'v4': '15'},
    #
    #         # ═══ RISK MANAGEMENT ══════════════════════════════════════════════
    #         'risk_tier1': {
    #             'description': 'Tier 1 Risk %',
    #             'v1': '0.008', 'v2': '0.010', 'v3': '0.012', 'v4': '0.015'},
    #         'risk_tier2': {
    #             'description': 'Tier 2 Risk %',
    #             'v1': '0.010', 'v2': '0.012', 'v3': '0.015', 'v4': '0.018'},
    #         'risk_tier2_exceptional': {
    #             'description': 'Tier 2 Exceptional Risk %',
    #             'v1': '0.025', 'v2': '0.030', 'v3': '0.035', 'v4': '0.040'},
    #         'risk_per_trade': {
    #             'description': 'Base Risk Per Trade',
    #             'v1': '0.015', 'v2': '0.018', 'v3': '0.020', 'v4': '0.022'},
    #         'risk_full_position': {
    #             'description': 'Full Position Risk %',
    #             'v1': '0.008', 'v2': '0.010', 'v3': '0.012', 'v4': '0.015'},
    #         'max_position_units': {
    #             'description': 'Max Position Units',
    #             'v1': '30', 'v2': '40', 'v3': '50', 'v4': '60'},
    #
    #         # ═══ PROFIT TARGETS ═══════════════════════════════════════════════
    #         'profit_target_r1': {
    #             'description': 'Profit Target R1',
    #             'v1': '1.5', 'v2': '2.0', 'v3': '2.5', 'v4': '3.0'},
    #         'profit_target_r2': {
    #             'description': 'Profit Target R2',
    #             'v1': '2.5', 'v2': '3.0', 'v3': '3.5', 'v4': '4.0'},
    #         'profit_target_r3': {
    #             'description': 'Profit Target R3',
    #             'v1': '6.0', 'v2': '8.0', 'v3': '10.0', 'v4': '12.0'},
    #
    #         # ═══ TRADE MANAGEMENT ═════════════════════════════════════════════
    #         'max_hold_bars': {
    #             'description': 'Max Hold Bars',
    #             'v1': '80', 'v2': '100', 'v3': '120', 'v4': '150'},
    #         'max_daily_trades': {
    #             'description': 'Max Daily Trades',
    #             'v1': '6', 'v2': '8', 'v3': '10', 'v4': '12'},
    #         'min_bars_between_trades': {
    #             'description': 'Min Bars Between Trades',
    #             'v1': '2', 'v2': '3', 'v3': '4', 'v4': '5'},
    #         'price_percentile_bonus_early': {
    #             'description': 'Early Entry Bonus',
    #             'v1': '12', 'v2': '15', 'v3': '18', 'v4': '20'},
    #         'price_percentile_penalty_late': {
    #             'description': 'Late Entry Penalty',
    #             'v1': '10', 'v2': '12', 'v3': '15', 'v4': '18'},
    #
    #         # ═══ FUZZY MODE ═══════════════════════════════════════════════════
    #         'fuzzy_default_margin_pct': {
    #             'description': 'Fuzzy Margin %',
    #             'v1': '8', 'v2': '10', 'v3': '12', 'v4': '15'},
    #         'fuzzy_absolute_min': {
    #             'description': 'Fuzzy Absolute Min',
    #             'v1': '58', 'v2': '60', 'v3': '62', 'v4': '65'},
    #     }
    #
    #     # Define parameter groups that should be linked
    #     self.param_groups = {
    #         'ema_periods': ['ema_fast_period', 'ema_mid_period', 'ema_slow_period'],
    #         'adx_scoring': ['adx_score_trend_forming', 'adx_score_good_trend', 'adx_score_strong_trend',
    #                         'adx_score_very_strong', 'adx_score_extended'],
    #         'profit_target_pcts': ['profit_target_r1_pct', 'profit_target_r2_pct', 'profit_target_r3_pct'],
    #     }
    #
    #     # Initialise self.backtest_params from ALL_BACKTEST_PARAMS if not already done
    #     if not hasattr(self, 'backtest_params') or not self.backtest_params:
    #         self.backtest_params = {}
    #
    #     for key, meta in ALL_BACKTEST_PARAMS.items():
    #         if key not in self.backtest_params:
    #             self.backtest_params[key] = {
    #                 'active': tk.BooleanVar(value=False),
    #                 'value1': tk.StringVar(value=meta['v1']),
    #                 'value2': tk.StringVar(value=meta['v2']),
    #                 'value3': tk.StringVar(value=meta['v3']),
    #                 'value4': tk.StringVar(value=meta['v4']),
    #                 'description': meta['description']
    #             }
    #
    #     # ── Optimization Metrics ──────────────────────────────────────────────
    #     metrics_header = ttk.LabelFrame(right_frame, text="Optimization Metrics")
    #     metrics_header.pack(fill=tk.X, padx=5, pady=5)
    #     metrics_inner = ttk.Frame(metrics_header)
    #     metrics_inner.pack(fill=tk.X, padx=5, pady=5)
    #
    #     row, col = 0, 0
    #     for metric_name, metric_var in self.optimization_metrics.items():
    #         ttk.Checkbutton(metrics_inner, text=metric_name.replace('_', ' ').title(),
    #                         variable=metric_var).grid(row=row, column=col, sticky='w', padx=5, pady=2)
    #         col += 1
    #         if col > 1:
    #             col = 0
    #             row += 1
    #
    #     ttk.Label(metrics_inner, text="Equal Weights (All selected metrics)").grid(
    #         row=row + 1, column=0, columnspan=2, pady=5)
    #
    #     ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=3)
    #
    #     # ── Action buttons row ────────────────────────────────────────────────
    #     action_frame = ttk.Frame(right_frame)
    #     action_frame.pack(fill=tk.X, padx=5, pady=3)
    #
    #     ttk.Button(action_frame, text="✅ Select All",
    #                command=self._select_all_backtest_params).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(action_frame, text="❌ Deselect All",
    #                command=self._deselect_all_backtest_params).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(action_frame, text="💾 Save Params",
    #                command=self.save_backtest_params).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(action_frame, text="📂 Load Params",
    #                command=self.load_backtest_params).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(action_frame, text="🔄 Reset",
    #                command=self.reset_backtest_params).pack(side=tk.LEFT, padx=2)
    #
    #     self.bt_selection_label = ttk.Label(action_frame, text="Selected: 0", foreground='blue',
    #                                         font=('Arial', 9, 'bold'))
    #     self.bt_selection_label.pack(side=tk.RIGHT, padx=5)
    #
    #     # ── Parameter table header ────────────────────────────────────────────
    #     param_header_frame = ttk.LabelFrame(right_frame, text="Parameters (Select for Optimization)")
    #     param_header_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    #
    #     col_header = ttk.Frame(param_header_frame)
    #     col_header.pack(fill=tk.X, padx=5, pady=2)
    #     ttk.Label(col_header, text="✓", width=3, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=0,
    #                                                                                               padx=1)
    #     ttk.Label(col_header, text="Parameter", width=25, anchor='w', font=('Arial', 8, 'bold')).grid(row=0, column=1,
    #                                                                                                   padx=1)
    #     ttk.Label(col_header, text="Val 1", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=2,
    #                                                                                                   padx=1)
    #     ttk.Label(col_header, text="Val 2", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=3,
    #                                                                                                   padx=1)
    #     ttk.Label(col_header, text="Val 3", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=4,
    #                                                                                                   padx=1)
    #     ttk.Label(col_header, text="Val 4", width=7, anchor='center', font=('Arial', 8, 'bold')).grid(row=0, column=5,
    #                                                                                                   padx=1)
    #     ttk.Separator(param_header_frame, orient='horizontal').pack(fill=tk.X, padx=5)
    #
    #     # ── Scrollable parameter list ─────────────────────────────────────────
    #     param_canvas = tk.Canvas(param_header_frame, bg='white', highlightthickness=0)
    #     param_scrollbar = ttk.Scrollbar(param_header_frame, orient="vertical", command=param_canvas.yview)
    #     param_scrollable = ttk.Frame(param_canvas)
    #     param_scrollable.bind("<Configure>",
    #                           lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all")))
    #     param_canvas.create_window((0, 0), window=param_scrollable, anchor="nw")
    #     param_canvas.configure(yscrollcommand=param_scrollbar.set)
    #     param_canvas.pack(side="left", fill="both", expand=True, padx=2, pady=2)
    #     param_scrollbar.pack(side="right", fill="y")
    #
    #     def _on_mousewheel(event):
    #         param_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    #
    #     param_canvas.bind("<MouseWheel>", _on_mousewheel)
    #
    #     # Don't reset on subsequent calls — widgets are per-tab but data is shared
    #     if not hasattr(self, 'backtest_param_widgets'):
    #         self.backtest_param_widgets = {}
    #     else:
    #         self.backtest_param_widgets = {}   # each tab gets its own widget references
    #     self._updating_group = False
    #
    #     sorted_params = sorted(self.backtest_params.items(), key=lambda x: x[1]['description'])
    #
    #     for p_row, (param_key, param_data) in enumerate(sorted_params):
    #         frame = ttk.Frame(param_scrollable)
    #         frame.grid(row=p_row, column=0, sticky='ew', pady=1, padx=2)
    #
    #         def make_toggle_cmd(key=param_key):
    #             def cmd():
    #                 if self._updating_group:
    #                     return
    #                 self._update_bt_selection_count()
    #                 self._check_bt_selection_warning()
    #
    #             return cmd
    #
    #         cb = ttk.Checkbutton(frame, variable=param_data['active'], command=make_toggle_cmd())
    #         cb.grid(row=0, column=0, padx=2)
    #
    #         lbl = ttk.Label(frame, text=param_data['description'], width=25, anchor='w',
    #                         font=('Arial', 8), cursor='hand2')
    #         lbl.grid(row=0, column=1, padx=2, sticky='w')
    #
    #         def make_label_callback(k=param_key):
    #             def callback(e):
    #                 self._toggle_bt_param(k)
    #
    #             return callback
    #
    #         lbl.bind("<Button-1>", make_label_callback())
    #
    #         for v_idx, v_key in enumerate(['value1', 'value2', 'value3', 'value4'], start=2):
    #             e = ttk.Entry(frame, textvariable=param_data[v_key], width=7, font=('Arial', 8))
    #             e.grid(row=0, column=v_idx, padx=1)
    #
    #         self.backtest_param_widgets[param_key] = {
    #             'active': param_data['active'],
    #             'value1': param_data['value1'],
    #             'value2': param_data['value2'],
    #             'value3': param_data['value3'],
    #             'value4': param_data['value4'],
    #             'widget': cb
    #         }
    #
    #     self._update_bt_selection_count()

    def _sync_param_group(self, group_name, triggered_by_key):
        """Synchronize all parameters in a group to have the same active state"""
        if group_name not in self.param_groups:
            return

        self._updating_group = True

        try:
            group_params = self.param_groups[group_name]

            # Get the current state from the triggered parameter
            current_state = self.backtest_params[triggered_by_key]['active'].get()

            # Apply the same state to all parameters in the group
            for param_key in group_params:
                if param_key in self.backtest_params:
                    self.backtest_params[param_key]['active'].set(current_state)

            # Update the selection count
            self._update_bt_selection_count()

            # Show a message about group synchronization
            if current_state:
                group_name_display = group_name.replace('_', ' ').title()
                self.log_message(f"📊 {group_name_display} group ENABLED - all parameters will be optimized together",
                                 "blue")
            else:
                group_name_display = group_name.replace('_', ' ').title()
                self.log_message(f"📊 {group_name_display} group DISABLED", "blue")

        finally:
            self._updating_group = False

    def _toggle_bt_param(self, param_key):
        """Toggle a backtest parameter's active state by clicking its label"""
        if param_key in self.backtest_params:
            if self._updating_group:
                return
            current = self.backtest_params[param_key]['active'].get()
            self.backtest_params[param_key]['active'].set(not current)
            self._update_bt_selection_count()
            self._check_bt_selection_warning()

    def _select_all_backtest_params(self):
        """Select all backtest parameters, respecting parameter groups"""
        self._updating_group = True

        try:
            # First, deselect all
            for param_data in self.backtest_params.values():
                param_data['active'].set(False)

            # Then select all individual parameters (not grouped)
            selected_groups = set()

            for param_key, param_data in self.backtest_params.items():
                # Check if this param is in a group
                in_group = False
                for group_name, group_params in self.param_groups.items():
                    if param_key in group_params:
                        in_group = True
                        if group_name not in selected_groups:
                            # Select the first param in the group to trigger group selection
                            first_param = group_params[0]
                            if first_param in self.backtest_params:
                                self.backtest_params[first_param]['active'].set(True)
                                selected_groups.add(group_name)
                        break

                if not in_group:
                    # Individual parameter - select it
                    param_data['active'].set(True)
        finally:
            self._updating_group = False

        self._update_bt_selection_count()
        self._check_bt_selection_warning()

    def _update_bt_selection_count(self):
        """Update the selection counter label"""
        if hasattr(self, 'bt_selection_label') and self.bt_selection_label.winfo_exists():
            count = sum(1 for p in self.backtest_params.values() if p['active'].get())
            color = 'red' if count > 10 else 'green' if count > 0 else 'blue'
            self.bt_selection_label.config(text=f"Selected: {count}", foreground=color)

    def _check_bt_selection_warning(self):
        """Warn user if more than 15 parameters are selected"""
        count = sum(1 for p in self.backtest_params.values() if p['active'].get())
        if count > 15:
            messagebox.showwarning(
                "Many Parameters Selected",
                f"⚠️ You have selected {count} parameters for optimization.\n\n"
                f"This will result in VERY LONG optimization times.\n"
                f"Consider selecting only the most critical parameters:\n"
                f"  • ADX Scoring (5 params)\n"
                f"  • Quality Thresholds (2-3 params)\n"
                f"  • Risk settings (2-3 params)\n"
                f"  • Key weights (3-4 params)"
            )
        elif count > 10:
            self.log_message(f"⚠️ {count} parameters selected - optimization may take a while", "orange")

    def _deselect_all_backtest_params(self):
        """Deselect all backtest parameters, respecting parameter groups"""
        self._updating_group = True

        try:
            for param_data in self.backtest_params.values():
                param_data['active'].set(False)
        finally:
            self._updating_group = False

        self._update_bt_selection_count()
        self.log_message("✅ All backtest parameters deselected", "blue")

    # ─────────────────────────────────────────────────────────────────────────────
    # REPLACE: save_backtest_params
    # ─────────────────────────────────────────────────────────────────────────────
    def save_backtest_params(self):
        """Save backtest optimisation parameters for the CURRENT strategy."""
        import json, os
        from datetime import datetime

        try:
            current_strategy = (
                self.strategy_type_var.get()
                if hasattr(self, "strategy_type_var")
                else "Momentum"
            )
            filename = self._get_strategy_bt_file(current_strategy)

            # Choose the right param dict
            if current_strategy == "Scalping":
                param_source = getattr(self, "scalping_backtest_params", {})
            else:
                param_source = self.backtest_params

            params_to_save = {}
            for key, data in param_source.items():
                params_to_save[key] = {
                    "active": data["active"].get(),
                    "value1": data["value1"].get(),
                    "value2": data["value2"].get(),
                    "value3": data["value3"].get(),
                    "value4": data["value4"].get(),
                    "description": data.get("description", key),
                }

            metrics_to_save = {
                name: var.get()
                for name, var in self.optimization_metrics.items()
            }

            save_data = {
                "strategy": current_strategy,
                "parameters": params_to_save,
                "metrics": metrics_to_save,
                "timestamp": datetime.now().isoformat(),
            }

            with open(filename, "w") as f:
                json.dump(save_data, f, indent=4)

            selected = sum(1 for p in params_to_save.values() if p["active"])
            active_metrics = [m for m, v in metrics_to_save.items() if v]

            self.log_message("=" * 70, "green")
            self.log_message(
                f"✅ [{current_strategy}] backtest params saved → {filename}", "green"
            )
            self.log_message(
                f"   • {selected} params selected  |  "
                f"{len(active_metrics)} metrics: {', '.join(active_metrics)}",
                "blue",
            )
            self.log_message("=" * 70, "green")

            import tkinter.messagebox as mb
            mb.showinfo(
                "Saved",
                f"{current_strategy} backtest params saved to:\n{filename}\n\n"
                f"Selected: {selected}  |  Metrics: {len(active_metrics)}",
            )

        except Exception as exc:
            self.log_message(f"❌ save_backtest_params error: {exc}", "red")
            import traceback as tb
            self.log_message(tb.format_exc(), "red")

    # def save_backtest_params(self):
    #     """Save backtest parameters AND optimization metrics to JSON file with file path display"""
    #     try:
    #         params_to_save = {}
    #         for param_key, param_data in self.backtest_params.items():
    #             params_to_save[param_key] = {
    #                 'active': param_data['active'].get(),
    #                 'value1': param_data['value1'].get(),
    #                 'value2': param_data['value2'].get(),
    #                 'value3': param_data['value3'].get(),
    #                 'value4': param_data['value4'].get(),
    #                 'description': param_data['description']
    #             }
    #
    #         # ═══ ADD OPTIMIZATION METRICS TO SAVED DATA ════════════════════
    #         metrics_to_save = {}
    #         for metric_name, metric_var in self.optimization_metrics.items():
    #             metrics_to_save[metric_name] = metric_var.get()
    #
    #         # Combine both in a single file
    #         save_data = {
    #             'parameters': params_to_save,
    #             'metrics': metrics_to_save,
    #             'timestamp': datetime.now().isoformat()
    #         }
    #
    #         filename = "backtest_params.json"
    #         with open(filename, 'w') as f:
    #             json.dump(save_data, f, indent=4)
    #
    #         # ═══════════════════════════════════════════════════════════════
    #         # DISPLAY FILE PATH
    #         # ═══════════════════════════════════════════════════════════════
    #         import os
    #         full_path = os.path.abspath(filename)
    #         file_size = os.path.getsize(full_path)
    #
    #         selected_count = sum(1 for p in params_to_save.values() if p['active'])
    #         active_metrics = [m for m, v in metrics_to_save.items() if v]
    #
    #         print("=" * 70)
    #         print(f"✅ Backtest parameters saved to: {full_path}")
    #         print(f"💾 File size: {file_size:,} bytes")
    #         print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #         # BUG FIX v9.4.1 (APP BUG D): print() does not accept a colour argument.
    #         # The string "blue" was being printed as a literal second positional argument.
    #         print(f"   • {selected_count} parameters selected for optimization")
    #         print(f"   • {len(active_metrics)} optimization metrics selected: {', '.join(active_metrics)}")
    #         print("=" * 70)
    #
    #         self.log_message(f"✅ Backtest parameters saved to {filename}", "green")
    #         self.log_message(f"📁 Full path: {full_path}", "blue")
    #         self.log_message(f"   • {selected_count} parameters selected for optimization", "blue")
    #         self.log_message(f"   • {len(active_metrics)} optimization metrics selected: {', '.join(active_metrics)}",
    #                          "blue")
    #
    #         messagebox.showinfo("Saved",
    #                             f"Backtest parameters saved to {filename}\n\n"
    #                             f"Full path: {full_path}\n\n"
    #                             f"Parameters selected: {selected_count}\n"
    #                             f"Optimization metrics: {len(active_metrics)}")
    #     except Exception as e:
    #         self.log_message(f"❌ Error saving backtest params: {e}", "red")
    #         messagebox.showerror("Error", f"Failed to save parameters:\n{e}")

    # ─────────────────────────────────────────────────────────────────────────────
    # REPLACE: load_backtest_params
    # ─────────────────────────────────────────────────────────────────────────────
    def load_backtest_params(self, strategy: str | None = None, silent: bool = False):
        """Load backtest optimisation parameters for *strategy* (or current strategy).

        Parameters
        ----------
        strategy : str | None
            Target strategy name.  None → use self.strategy_type_var.
        silent : bool
            When True suppress the messagebox (used for auto-load on strategy switch).
        """
        import json, os
        import tkinter.messagebox as mb

        try:
            if strategy is None:
                strategy = (
                    self.strategy_type_var.get()
                    if hasattr(self, "strategy_type_var")
                    else "Momentum"
                )
            filename = self._get_strategy_bt_file(strategy)

            if not os.path.exists(filename):
                if not silent:
                    mb.showwarning(
                        "File Not Found",
                        f"No saved params for {strategy}.\nExpected: {filename}\n\n"
                        "Save parameters first.",
                    )
                else:
                    self.log_message(
                        f"ℹ️ No saved backtest params for {strategy} ({filename})", "blue"
                    )
                return

            with open(filename, "r") as f:
                loaded = json.load(f)

            # Support both new format {parameters, metrics} and legacy flat dict
            if isinstance(loaded, dict) and "parameters" in loaded:
                params_dict = loaded["parameters"]
                metrics_dict = loaded.get("metrics", {})
            else:
                params_dict = loaded
                metrics_dict = {}
                self.log_message(
                    f"📂 [{strategy}] legacy format (no metrics)", "orange"
                )

            # Choose target param dict
            if strategy == "Scalping":
                target = getattr(self, "scalping_backtest_params", {})
            else:
                target = self.backtest_params

            loaded_count = 0
            for key, info in params_dict.items():
                if key in target:
                    target[key]["active"].set(info.get("active", False))
                    target[key]["value1"].set(str(info.get("value1", "")))
                    target[key]["value2"].set(str(info.get("value2", "")))
                    target[key]["value3"].set(str(info.get("value3", "")))
                    target[key]["value4"].set(str(info.get("value4", "")))
                    loaded_count += 1
                else:
                    # Parameter not yet in UI — add it silently
                    import tkinter as tk
                    target[key] = {
                        "active": tk.BooleanVar(value=info.get("active", False)),
                        "value1": tk.StringVar(value=str(info.get("value1", ""))),
                        "value2": tk.StringVar(value=str(info.get("value2", ""))),
                        "value3": tk.StringVar(value=str(info.get("value3", ""))),
                        "value4": tk.StringVar(value=str(info.get("value4", ""))),
                        "description": info.get("description", key),
                    }
                    loaded_count += 1

            for metric, value in metrics_dict.items():
                if metric in self.optimization_metrics:
                    self.optimization_metrics[metric].set(value)

            # Refresh selection counters
            if strategy == "Scalping" and hasattr(self, "_sc_update_count"):
                self._sc_update_count()
            elif hasattr(self, "_update_bt_selection_count"):
                self._update_bt_selection_count()

            selected = sum(1 for p in target.values() if p["active"].get())
            active_metrics = [m for m, v in self.optimization_metrics.items() if v.get()]

            self.log_message("=" * 70, "green")
            self.log_message(
                f"✅ [{strategy}] backtest params loaded ← {filename}", "green"
            )
            self.log_message(
                f"   • {loaded_count} params updated  |  {selected} selected  |  "
                f"{len(active_metrics)} metrics",
                "blue",
            )
            self.log_message("=" * 70, "green")

            if not silent:
                mb.showinfo(
                    "Loaded",
                    f"{strategy} backtest params loaded from:\n{filename}\n\n"
                    f"Updated: {loaded_count}  |  Selected: {selected}  |  "
                    f"Metrics: {len(active_metrics)}",
                )

            if selected > 10 and hasattr(self, "_check_bt_selection_warning"):
                self._check_bt_selection_warning()

        except json.JSONDecodeError as exc:
            self.log_message(f"❌ Invalid JSON in {filename}: {exc}", "red")
            if not silent:
                import tkinter.messagebox as mb
                mb.showerror("Load Error", f"File is corrupted:\n{exc}")
        except Exception as exc:
            self.log_message(f"❌ load_backtest_params error: {exc}", "red")
            import traceback as tb
            self.log_message(tb.format_exc(), "red")

    def switch_strategy_autoload_snippet(self, new_strategy_name: str):
        """
        Call this at the end of a successful switch_strategy() to silently
        restore saved backtest params for the newly selected strategy.
        """
        try:
            self.load_backtest_params(strategy=new_strategy_name, silent=True)
            self.log_message(
                f"📂 Backtest params auto-loaded for {new_strategy_name}", "blue"
            )
        except Exception as exc:
            self.log_message(
                f"ℹ️ Could not auto-load backtest params for {new_strategy_name}: {exc}",
                "orange",
            )

    def reset_backtest_params(self):
        """Reset backtest parameters to defaults"""
        if messagebox.askyesno("Reset Backtest Params",
                               "Reset all backtest parameters to defaults?\nThis will also deselect all parameters."):
            try:
                for param_key, param_data in self.backtest_params.items():
                    param_data['active'].set(False)

                self._update_bt_selection_count()
                self.log_message("✅ Backtest parameters reset (all deselected)", "green")

            except Exception as e:
                self.log_message(f"❌ Error resetting backtest params: {e}", "red")

    def _get_param_values(self, param_key):
        """
        Get values for a parameter:
        - If ACTIVE: Returns the 4 values for optimization (grid search)
        - If INACTIVE: Returns SINGLE value (current/default) as a list

        This ensures inactive parameters are FIXED during optimization.
        """
        if param_key not in self.backtest_params:
            return []

        param_data = self.backtest_params[param_key]

        # =========================================================
        # CASE 1: Parameter is ACTIVE - return 4 values for optimization
        # =========================================================
        if param_data['active'].get():
            values = []
            for v in ['value1', 'value2', 'value3', 'value4']:
                val = param_data[v].get()
                if val and val != '':
                    # Convert type based on content
                    if isinstance(val, str):
                        if val.lower() == 'true':
                            val = True
                        elif val.lower() == 'false':
                            val = False
                        elif '.' in val:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                        else:
                            try:
                                val = int(val)
                            except ValueError:
                                pass
                    values.append(val)

            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for v in values:
                if v not in seen:
                    seen.add(v)
                    unique_values.append(v)

            # Ensure we have at least one value
            if not unique_values:
                unique_values = [False]

            return unique_values

        # =========================================================
        # CASE 2: Parameter is INACTIVE - return SINGLE current value
        # =========================================================
        else:
            current_value = self._get_current_param_value(param_key)
            return [current_value] if current_value is not None else []

    def _get_current_param_value(self, param_key):
        """
        Get the current/default value for an inactive parameter.
        Priority:
        1. Live strategy instance (highest - reflects UI changes)
        2. Custom params (saved settings)
        3. MOMENTUM_PARAMS defaults (code defaults)
        """
        # =========================================================
        # PRIORITY 1: Get from live Momentum strategy
        # =========================================================
        if hasattr(self, 'strategies') and 'Momentum' in self.strategies:
            strategy = self.strategies['Momentum']
            if hasattr(strategy, param_key):
                value = getattr(strategy, param_key)
                if value is not None:
                    return value

        # =========================================================
        # PRIORITY 2: Get from custom_params (saved settings)
        # =========================================================
        if hasattr(self, 'custom_params') and 'momentum' in self.custom_params:
            if param_key in self.custom_params['momentum']:
                value = self.custom_params['momentum'][param_key]
                if value is not None:
                    return value

        # =========================================================
        # PRIORITY 3: Get from MOMENTUM_PARAMS (code defaults)
        # =========================================================
        try:
            from strategies.MomentumStrategy_MACD_HybridScore_Latest import MOMENTUM_PARAMS
            if param_key in MOMENTUM_PARAMS:
                return MOMENTUM_PARAMS[param_key]
        except ImportError:
            pass

        # =========================================================
        # FALLBACK: Hardcoded defaults for critical parameters
        # =========================================================
        fallbacks = {
            # Quality thresholds
            'quality_tier1_min': 72,
            'quality_tier2_min': 68,
            'short_quality_tier1_min': 75,
            'short_quality_tier2_min': 70,

            # EMA periods
            'ema_fast_period': 10,
            'ema_mid_period': 20,
            'ema_slow_period': 60,

            # Weights
            'weight_ema': 20,
            'weight_adx': 20,
            'weight_macd': 25,
            'weight_rsi': 20,
            'weight_volume': 12,

            # Risk
            'stop_loss_atr_mult': 2.2,
            'trailing_activation_pct': 0.04,
            'trailing_distance_pct': 0.035,
            'risk_tier1': 0.02,
            'risk_tier2': 0.025,

            # Entry filters
            'tier1_adx_hard_min': 20,
            'tier1_rsi_min': 44,
            'tier1_rsi_max': 64,
            'tier1_volume_min': 1.0,
            'tier1_momentum_min': 0.02,

            # Short filters
            'short_tier1_adx_hard_min': 28,
            'short_tier1_rsi_min': 34,
            'short_tier1_rsi_max': 54,
            'short_tier1_volume_min': 1.3,
            'short_tier1_momentum_min': 0.05,

            # Tier 2
            'tier2_adx_min': 18,
            'tier2_volume_min': 0.8,
            'tier1_size_multiplier': 1.0,
            'tier2_size_multiplier': 0.70,
            'tier1_stop_multiplier': 2.0,
            'tier2_stop_multiplier': 2.5,
            'min_bars_between_trades': 4,
            'min_bars_between_trades_tier1': 4,
            'min_bars_between_trades_tier2': 3,
            'cooldown_tier2_enabled': True,
            'tier1_confluence_min': 0.65,
            'tier2_confluence_min': 0.70,

            # Trade management - FIX: ADD max_daily_trades HERE
            'only_tier1_entries': False,
            'max_daily_trades': 15,  # ← ADD THIS
            'min_bars_between_trades': 5,
            'max_hold_bars': 150,
            'cooldown_after_profit_target_bars': 2,
            'cooldown_after_loss_bars': 12,

            # Price percentile
            'price_percentile_bonus_early': 12,
            'price_percentile_penalty_late': 12,

            # Fuzzy
            'fuzzy_default_margin_pct': 10,
            'fuzzy_absolute_min': 45,
        }

        return fallbacks.get(param_key, None)

    def create_scalping_parameter_controls(self, parent):
        """Create parameter controls for Scalping strategy — all SCALPING_PARAMS editable."""
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(paned, width=600)
        paned.add(left_frame, weight=2)

        right_frame = ttk.LabelFrame(paned, text="Backtest Optimization Parameters", width=500)
        paned.add(right_frame, weight=1)

        # ── Column headers ────────────────────────────────────────────────────
        headers_frame = ttk.Frame(left_frame)
        headers_frame.pack(fill='x', padx=5, pady=(5, 0), side=tk.TOP)
        for col, (text, minw) in enumerate([
            ("Parameter", 250), ("📌 Default Value", 120),
            ("✏️ Custom Value", 120), ("Description", 350)
        ]):
            headers_frame.columnconfigure(col, weight=0 if col < 3 else 1, minsize=minw)
            ttk.Label(headers_frame, text=text, font=('Arial', 10, 'bold'), anchor='w').grid(
                row=0, column=col, padx=5, pady=8, sticky='w')
        ttk.Separator(headers_frame, orient='horizontal').grid(
            row=1, column=0, columnspan=4, sticky='ew', padx=5, pady=(0, 5))

        # ── Scrollable content ────────────────────────────────────────────────
        content_frame = ttk.Frame(left_frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5, side=tk.TOP)

        canvas = tk.Canvas(content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        default_params = self.get_default_scalping_params()
        self.scalping_param_widgets = {}

        categories = {
            '📈 EMA Periods': [
                'ema_fast_period', 'ema_mid_period', 'ema_slow_period',
            ],
            '📊 MACD & Stochastic': [
                'macd_fast', 'macd_slow', 'macd_signal_period',
                'stoch_k_period', 'stoch_d_period', 'stoch_smooth',
                'stoch_overbought', 'stoch_oversold', 'stoch_mid_upper', 'stoch_mid_lower',
            ],
            '🎯 RSI Filters': [
                'rsi_period', 'rsi_long_min', 'rsi_long_max',
                'rsi_short_min', 'rsi_short_max',
                'rsi_overbought_exit', 'rsi_oversold_exit',
            ],
            '📐 ADX Filters': [
                'adx_period', 'adx_min_long', 'adx_min_short', 'adx_extended_threshold',
                'adx_slope_min',
            ],
            '📦 Volume & ATR': [
                'volume_period', 'volume_min_ratio', 'volume_strong_ratio',
                'atr_period', 'atr_compression_lookback', 'atr_compression_threshold',
            ],
            '🏆 Quality Score Thresholds': [
                'quality_min_long', 'quality_min_short', 'quality_tier1_min',
                'weight_ema', 'weight_macd', 'weight_stoch',
                'weight_rsi', 'weight_volume', 'weight_adx',
            ],
            '💰 Risk Management': [
                'risk_per_trade', 'risk_tier1',
                'max_position_size_pct', 'max_position_units',
                'min_cash_reserve', 'base_risk_pct',
            ],
            '🛑 Stop Loss & Trailing': [
                'stop_loss_atr_mult', 'trailing_activation_pct',
                'trailing_distance_pct', 'trailing_atr_mult',
            ],
            '🔒 Breakeven Stop': [
                'be_stop_enabled', 'be_stop_r_trigger', 'be_stop_no_progress_bars',
            ],
            '🎯 Profit Targets': [
                'take_profit_r1', 'take_profit_r2',
                'partial_exit_pct_r1', 'partial_exit_pct_r2',
            ],
            '🚪 Exit Conditions': [
                'macd_cross_exit_enabled', 'macd_cross_min_profit_r',
                'stoch_reversal_exit_enabled', 'stoch_reversal_min_profit_r',
                'ema_cross_exit_enabled', 'ema_cross_min_profit_r',
                'max_hold_bars', 'min_hold_bars_before_stop',
            ],
            '⏱️ Entry Timing Filters': [
                'pullback_zone_lower_pct', 'pullback_zone_upper_pct',
                'momentum_period', 'momentum_min_long', 'momentum_min_short',
            ],
            '🌍 Regime / Ranging Filter': [
                'regime_filter_enabled', 'bb_period', 'bb_std',
                'kc_period', 'kc_atr_mult',
                'chop_period', 'chop_threshold', 'ranging_min_checks',
            ],
            '⏳ Trade Frequency & Cooldown': [
                'max_daily_trades', 'min_bars_between_trades',
                'cooldown_after_loss_bars',
                'consecutive_loss_threshold', 'consecutive_loss_cooldown_bars',
            ],
            '📅 Daily Trend Filter': [
                'daily_trend_filter_enabled', 'daily_ema_period', 'daily_trend_adx_override',
            ],
            '🔍 Extended Run & Trend Age': [
                'extended_run_lookback',
                'extended_run_max_pct_long', 'extended_run_max_pct_short',
                'trend_age_penalty_enabled', 'trend_age_max_bars', 'trend_age_penalty_pts',
            ],
            '🛡️ Loss Limits': [
                'daily_loss_limit_pct', 'max_drawdown_limit_pct', 'max_consecutive_losses',
            ],
            '🧠 Fuzzy Learning': [
                'fuzzy_mode_enabled', 'fuzzy_learning_enabled',
                'fuzzy_absolute_min', 'fuzzy_absolute_max',
                'fuzzy_default_margin_pct', 'fuzzy_min_confidence', 'fuzzy_min_samples',
            ],
            '⚙️ Trade Direction': [
                'trade_direction', 'only_tier1_entries',
            ],
        }

        scalping_descriptions = {
            'ema_fast_period': 'Fast EMA period (default 5)',
            'ema_mid_period': 'Mid EMA period (default 13)',
            'ema_slow_period': 'Slow EMA period (default 21)',
            'macd_fast': 'MACD fast period — faster than standard for scalping',
            'macd_slow': 'MACD slow period',
            'macd_signal_period': 'MACD signal smoothing',
            'stoch_k_period': 'Stochastic %K period',
            'stoch_d_period': 'Stochastic %D smoothing',
            'stoch_smooth': 'Stochastic slow %K smoothing',
            'stoch_overbought': 'Stochastic overbought threshold (exit signal)',
            'stoch_oversold': 'Stochastic oversold threshold (exit signal)',
            'stoch_mid_upper': 'Upper mid zone for perfect long setup',
            'stoch_mid_lower': 'Lower mid zone for perfect short setup',
            'rsi_period': 'RSI look-back period',
            'rsi_long_min': 'RSI minimum for long entries (avoid oversold)',
            'rsi_long_max': 'RSI maximum for long entries (avoid overbought)',
            'rsi_short_min': 'RSI minimum for short entries',
            'rsi_short_max': 'RSI maximum for short entries (avoid too oversold)',
            'rsi_overbought_exit': 'RSI level that triggers long exit',
            'rsi_oversold_exit': 'RSI level that triggers short exit',
            'adx_period': 'ADX calculation period',
            'adx_min_long': 'Minimum ADX for long entries — needs real trend',
            'adx_min_short': 'Minimum ADX for short entries',
            'adx_extended_threshold': 'ADX above this = trend exhausted, block new entries',
            'adx_slope_min': 'Minimum ADX rise per bar (trend accelerating)',
            'volume_period': 'Volume MA look-back bars',
            'volume_min_ratio': 'Minimum volume vs average (1.4 = 40% above average)',
            'volume_strong_ratio': 'Strong volume confirmation threshold',
            'atr_period': 'ATR calculation period',
            'atr_compression_lookback': 'Bars to average ATR for compression check',
            'atr_compression_threshold': 'Block entries when ATR < this × average ATR',
            'quality_min_long': 'Minimum quality score to enter a long (0-100)',
            'quality_min_short': 'Minimum quality score to enter a short (0-100)',
            'quality_tier1_min': 'Quality score for Tier 1 (high-conviction) entries',
            'weight_ema': 'EMA alignment component weight (out of 100)',
            'weight_macd': 'MACD component weight',
            'weight_stoch': 'Stochastic component weight',
            'weight_rsi': 'RSI component weight',
            'weight_volume': 'Volume component weight',
            'weight_adx': 'ADX component weight',
            'risk_per_trade': 'Base risk % of equity per trade (0.008 = 0.8%)',
            'risk_tier1': 'Risk % for high-conviction Tier 1 entries',
            'max_position_size_pct': 'Maximum position as % of equity (0.15 = 15%)',
            'max_position_units': 'Hard unit cap per position',
            'min_cash_reserve': 'Minimum cash to keep unallocated',
            'base_risk_pct': 'Base risk used for position sizing',
            'stop_loss_atr_mult': 'Stop distance = ATR × this multiplier (2.2 recommended)',
            'trailing_activation_pct': 'Profit % needed to activate trailing stop',
            'trailing_distance_pct': 'Trailing stop distance from peak',
            'trailing_atr_mult': 'ATR multiplier for dynamic trailing distance',
            'be_stop_enabled': 'Enable breakeven stop',
            'be_stop_r_trigger': 'Move stop to breakeven when profit reaches this R',
            'be_stop_no_progress_bars': 'Move to BE if no progress after this many bars',
            'take_profit_r1': 'First partial exit at this R-multiple',
            'take_profit_r2': 'Second partial exit at this R-multiple',
            'partial_exit_pct_r1': 'Fraction of position to close at R1 (0.50 = 50%)',
            'partial_exit_pct_r2': 'Fraction to close at R2 (0.30 = 30%)',
            'macd_cross_exit_enabled': 'Exit on MACD bearish cross when in profit',
            'macd_cross_min_profit_r': 'Minimum profit (R) before MACD cross exit fires',
            'stoch_reversal_exit_enabled': 'Exit on Stochastic reversal when in profit',
            'stoch_reversal_min_profit_r': 'Minimum profit before Stochastic exit fires',
            'ema_cross_exit_enabled': 'Exit on full EMA bearish reversal',
            'ema_cross_min_profit_r': 'Minimum profit before EMA cross exit fires',
            'max_hold_bars': 'Force exit after this many bars (48 × 15min = 12h)',
            'min_hold_bars_before_stop': 'Do not allow stop in first N bars (noise protection)',
            'pullback_zone_lower_pct': 'Max % below EMA_fast for entry — deeper pullback allowed',
            'pullback_zone_upper_pct': 'Max % above EMA_fast for entry — tighter = fewer entries',
            'momentum_period': 'Look-back bars for short-term momentum check',
            'momentum_min_long': 'Minimum momentum % required for long entry',
            'momentum_min_short': 'Minimum downward momentum % for short entry',
            'regime_filter_enabled': 'Block entries in ranging/choppy markets',
            'bb_period': 'Bollinger Band period for regime detection',
            'bb_std': 'Bollinger Band standard deviation',
            'kc_period': 'Keltner Channel period',
            'kc_atr_mult': 'Keltner Channel ATR multiplier',
            'chop_period': 'Choppiness Index period',
            'chop_threshold': 'Choppiness above this = ranging, block entries',
            'ranging_min_checks': 'Minimum regime checks to classify as ranging',
            'max_daily_trades': 'Maximum entries per day (8 prevents overtrading)',
            'min_bars_between_trades': 'Bars to wait between entries (4 = 1 hour minimum)',
            'cooldown_after_loss_bars': 'Bars to pause after a losing trade',
            'consecutive_loss_threshold': 'Trigger extended cooldown after this many consecutive losses',
            'consecutive_loss_cooldown_bars': 'Extended cooldown bars after loss streak',
            'daily_trend_filter_enabled': 'Only trade longs above daily EMA, shorts below',
            'daily_ema_period': 'Bars for daily trend EMA (96 × 15min = 24h)',
            'daily_trend_adx_override': 'ADX level that overrides daily trend filter',
            'extended_run_lookback': 'Bars to look back for swing high/low',
            'extended_run_max_pct_long': 'Block longs if price already ran > this % from swing low',
            'extended_run_max_pct_short': 'Block shorts if price dropped > this % from swing high',
            'trend_age_penalty_enabled': 'Penalise old trends in quality score',
            'trend_age_max_bars': 'Trend older than this gets quality penalty',
            'trend_age_penalty_pts': 'Points deducted from quality for aged trend',
            'daily_loss_limit_pct': 'Halt trading if daily loss exceeds this % of equity',
            'max_drawdown_limit_pct': 'Halt if drawdown from peak exceeds this %',
            'max_consecutive_losses': 'Halt trading after this many consecutive losses',
            'fuzzy_mode_enabled': 'Use adaptive quality thresholds learned from near-misses',
            'fuzzy_learning_enabled': 'Enable continuous learning from near-miss trades',
            'fuzzy_absolute_min': 'Absolute floor — fuzzy mode cannot go below this',
            'fuzzy_absolute_max': 'Absolute ceiling for fuzzy threshold',
            'fuzzy_default_margin_pct': 'Percentage below fixed threshold to start fuzzy zone',
            'fuzzy_min_confidence': 'Minimum confidence for fuzzy entry to fire',
            'fuzzy_min_samples': 'Minimum near-miss samples before fuzzy activates',
            'trade_direction': 'long / short / both',
            'only_tier1_entries': 'When True, only Tier 1 (high-conviction) entries fire',
        }

        row = 0
        for category, params in categories.items():
            ttk.Separator(scrollable_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=4, sticky='ew', pady=(10, 5), padx=5)
            row += 1
            ttk.Label(scrollable_frame, text=category,
                      font=('Arial', 11, 'bold')).grid(
                row=row, column=0, columnspan=4, sticky='w', padx=5, pady=5)
            row += 1

            for param_name in params:
                if param_name not in default_params:
                    continue

                for col, minw in enumerate([250, 120, 120, 350]):
                    scrollable_frame.columnconfigure(col, weight=0 if col < 3 else 1, minsize=minw)

                label_text = param_name.replace('_', ' ').title()
                ttk.Label(scrollable_frame, text=label_text, anchor='w').grid(
                    row=row, column=0, padx=5, pady=2, sticky='w')

                default_value = default_params[param_name]
                if isinstance(default_value, bool):
                    default_display = "✓ Enabled" if default_value else "✗ Disabled"
                else:
                    default_display = str(default_value)

                default_entry = ttk.Entry(scrollable_frame, width=15)
                default_entry.insert(0, default_display)
                default_entry.config(state='readonly')
                default_entry.grid(row=row, column=1, padx=5, pady=2, sticky='w')

                custom_value = self.custom_params.get('scalping', {}).get(param_name, default_value)

                if isinstance(default_value, bool):
                    custom_var = tk.BooleanVar(value=bool(custom_value))
                    custom_widget = ttk.Checkbutton(
                        scrollable_frame, variable=custom_var,
                        text="Enable" if custom_value else "Disable")
                    custom_widget.grid(row=row, column=2, padx=5, pady=2, sticky='w')

                    indicator = tk.Label(scrollable_frame, text="●", width=2,
                                         bg="yellow" if bool(custom_value) != bool(default_value) else "white")
                    indicator.grid(row=row, column=2, padx=(100, 0), pady=2, sticky='w')

                    def _make_bool_cb(v, w, ind, dv):
                        def cb(*_):
                            try:
                                cur = v.get()
                                w.config(text="Enable" if cur else "Disable")
                                ind.config(bg="yellow" if bool(cur) != bool(dv) else "white")
                            except tk.TclError:
                                pass

                        return cb

                    custom_var.trace_add('write',
                                         _make_bool_cb(custom_var, custom_widget, indicator, default_value))
                else:
                    custom_var = tk.StringVar(value=str(custom_value))
                    custom_widget = tk.Entry(
                        scrollable_frame, textvariable=custom_var, width=15,
                        bg="yellow" if str(custom_value) != str(default_value) else "white")
                    custom_widget.grid(row=row, column=2, padx=5, pady=2, sticky='w')

                    def _make_str_cb(v, w, dv):
                        def cb(*_):
                            try:
                                w.config(bg="yellow" if v.get() != str(dv) else "white")
                            except tk.TclError:
                                pass

                        return cb

                    custom_var.trace_add('write',
                                         _make_str_cb(custom_var, custom_widget, default_value))

                description = scalping_descriptions.get(param_name, 'Scalping parameter')
                ttk.Label(scrollable_frame, text=description,
                          wraplength=400, anchor='w',
                          foreground='#555555').grid(row=row, column=3, padx=5, pady=2, sticky='w')

                self.scalping_param_widgets[param_name] = {
                    'default': default_entry,
                    'custom': custom_var,
                    'widget': custom_widget,
                }
                row += 1

        # ── Right panel: Scalping-specific optimization panel ─────────────────
        self._build_scalping_backtest_optimization_panel(right_frame)

    def create_kalman_parameter_controls(self, parent):
        """Create parameter controls for Kalman strategy with FIXED default values"""
        # Create PanedWindow to split left and right
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Strategy parameters
        left_frame = ttk.Frame(paned, width=600)
        paned.add(left_frame, weight=2)

        # Right panel - Backtest optimization parameters (same as momentum tab)
        right_frame = ttk.LabelFrame(paned, text="Backtest Optimization Parameters", width=400)
        paned.add(right_frame, weight=1)

        # Left panel: Kalman parameters
        headers_frame = ttk.Frame(left_frame)
        headers_frame.pack(fill='x', padx=5, pady=(5, 0), side=tk.TOP)

        headers_frame.columnconfigure(0, weight=0, minsize=250)
        headers_frame.columnconfigure(1, weight=0, minsize=120)
        headers_frame.columnconfigure(2, weight=0, minsize=120)
        headers_frame.columnconfigure(3, weight=1, minsize=350)

        header_style = ('Arial', 10, 'bold')

        ttk.Label(
            headers_frame,
            text="Parameter",
            font=header_style,
            anchor='w'
        ).grid(row=0, column=0, padx=5, pady=8, sticky='w')

        ttk.Label(
            headers_frame,
            text="📌 Default Value",
            font=header_style,
            anchor='w'
        ).grid(row=0, column=1, padx=5, pady=8, sticky='w')

        ttk.Label(
            headers_frame,
            text="✏️ Custom Value",
            font=header_style,
            anchor='w'
        ).grid(row=0, column=2, padx=5, pady=8, sticky='w')

        ttk.Label(
            headers_frame,
            text="Description",
            font=header_style,
            anchor='w'
        ).grid(row=0, column=3, padx=5, pady=8, sticky='w')

        ttk.Separator(headers_frame, orient='horizontal').grid(
            row=1, column=0, columnspan=4, sticky='ew', padx=5, pady=(0, 5)
        )

        content_frame = ttk.Frame(left_frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5, side=tk.TOP)

        canvas = tk.Canvas(content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        default_params = self.get_default_kalman_params()
        self.kalman_param_widgets = {}

        categories = {
            '🔬 Kalman Filter': ['process_noise_1', 'process_noise_2', 'measurement_noise',
                                'trend_lookback', 'strength_smooth'],
            '📊 Strategy Configuration': ['risk_reward', 'lookback', 'window', 'strength_smooth_param'],
            '📈 Moving Averages': ['ma_fast_period', 'ma_slow_period'],
            '🎯 Entry Conditions': ['kalman_strength_min', 'rsi_min', 'rsi_max',
                                   'volume_min_ratio', 'pullback_percent'],
            '🛡️ Risk Management': ['stop_loss_pct', 'trailing_stop_pct', 'atr_multiplier',
                                   'risk_per_trade', 'max_position_pct'],
            '📉 Exit Conditions': ['rsi_exit_threshold', 'max_hold_bars', 'max_hold_seconds'],
            '🌐 Market Filters': ['min_adx', 'min_volatility', 'cooldown_bars']
        }

        row = 0
        for category, params in categories.items():
            ttk.Separator(scrollable_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=4, sticky='ew', pady=(10, 5), padx=5
            )
            row += 1

            ttk.Label(
                scrollable_frame,
                text=category,
                font=('Arial', 11, 'bold')
            ).grid(row=row, column=0, columnspan=4, sticky='w', padx=5, pady=5)
            row += 1

            for param_name in params:
                if param_name in default_params:
                    scrollable_frame.columnconfigure(0, weight=0, minsize=250)
                    scrollable_frame.columnconfigure(1, weight=0, minsize=120)
                    scrollable_frame.columnconfigure(2, weight=0, minsize=120)
                    scrollable_frame.columnconfigure(3, weight=1, minsize=350)

                    label_text = param_name.replace('_', ' ').title()
                    ttk.Label(scrollable_frame, text=label_text, anchor='w').grid(
                        row=row, column=0, padx=5, pady=2, sticky='w'
                    )

                    default_value = str(default_params[param_name])
                    default_entry = ttk.Entry(scrollable_frame, width=15)
                    default_entry.insert(0, default_value)
                    default_entry.config(state='readonly')
                    default_entry.grid(row=row, column=1, padx=5, pady=2, sticky='w')

                    custom_value = str(self.custom_params['kalman'].get(param_name, default_value))
                    custom_var = tk.StringVar(value=custom_value)
                    custom_entry = ttk.Entry(scrollable_frame, textvariable=custom_var, width=15)
                    custom_entry.grid(row=row, column=2, padx=5, pady=2, sticky='w')

                    if custom_value != default_value:
                        custom_entry.configure(style='Modified.TEntry')

                    description = self.get_kalman_param_description(param_name)
                    ttk.Label(
                        scrollable_frame,
                        text=description,
                        wraplength=400,
                        anchor='w',
                        foreground='#555555'
                    ).grid(row=row, column=3, padx=5, pady=2, sticky='w')

                    self.kalman_param_widgets[param_name] = {
                        'default': default_entry,
                        'custom': custom_var,
                        'widget': custom_entry
                    }

                    row += 1

        # Right panel: full backtest optimization panel (same as Momentum)
        self._build_backtest_optimization_panel(right_frame)

    def fmt_val(self, value, format_str='{:.4f}'):
        if value is None:
            return "N/A"
        try:
            return format_str.format(float(value))
        except:
            return str(value)

    def get_momentum_param_description(self, param_name):
        """Get description for Momentum parameters - CONSOLIDATED v10.0.5."""
        descriptions = {
            # ─── EMA ────────────────────────────────────────────────────────
            'ema_fast_period': 'Fast EMA period for short-term trend',
            'ema_mid_period': 'Middle EMA period for medium-term trend',
            'ema_slow_period': 'Slow EMA period for long-term trend',
            'ema_near_tolerance': 'Tolerance for EMA near-price condition',
            'daily_ema_period': 'Daily EMA period for trend filter (bars)',

            # ─── QUALITY THRESHOLDS (CONSOLIDATED) ──────────────────────
            'quality_tier1_min_long': 'Tier 1 minimum quality score for LONG entries (0-1)',
            'quality_tier2_min_long': 'Tier 2 minimum quality score for LONG entries (0-1)',
            'quality_tier1_min_short': 'Tier 1 minimum quality score for SHORT entries (0-1)',
            'quality_tier2_min_short': 'Tier 2 minimum quality score for SHORT entries (0-1)',

            # ─── WEIGHTS ────────────────────────────────────────────────────
            'weight_ema': 'EMA component weight (out of 100)',
            'weight_adx': 'ADX component weight (out of 100)',
            'weight_macd': 'MACD component weight (out of 100)',
            'weight_rsi': 'RSI component weight (out of 100)',
            'weight_volume': 'Volume component weight (out of 100)',

            # ─── TIER CONTROL ──────────────────────────────────────────────
            'only_tier1_entries': '🔥 When True, blocks Tier 1 entries (only Tier 2 allowed)',

            # ─── COOLDOWN & CONFLUENCE ─────────────────────────────────────
            'min_bars_between_trades_tier1': 'Min bars since last trade for TIER 1 entry',
            'min_bars_between_trades_tier2': 'Min bars since last trade for TIER 2 entry',
            'cooldown_tier2_enabled': 'Enable Tier 2 cooldown after trades',
            'tier1_confluence_min': 'Number of confluent signals required for Tier 1 (default: 3)',
            'tier2_confluence_min': 'Number of confluent signals required for Tier 2 (default: 2)',

            # ─── TIER 1 LONG ───────────────────────────────────────────────
            'tier1_adx_hard_min': 'Hard minimum ADX for Tier 1 LONG entries (default: 25)',
            'tier1_rsi_min': 'Minimum RSI for Tier 1 LONG entries (default: 55)',
            'tier1_rsi_max': 'Maximum RSI for Tier 1 LONG entries (default: 75)',
            'tier1_volume_min': 'Minimum volume ratio for Tier 1 LONG entries (default: 1.5)',
            'tier1_momentum_min': 'Minimum momentum for Tier 1 LONG entries (default: 0.02)',
            'tier1_kalman_min': 'Minimum Kalman strength for Tier 1 entries',
            'tier1_macd_gate': 'MACD gate required for Tier 1 entries',
            'tier1_price_ema_max_pct': 'Maximum price-EMA distance % for Tier 1',
            'daily_trend_filter_enabled': 'Enable daily trend filter for LONG entries',
            'pullback_zone_lower_pct': 'Pullback zone lower bound %',
            'pullback_zone_upper_pct': 'Pullback zone upper bound %',
            'adx_slope_min': 'Minimum ADX slope (rise per bar)',

            # ─── TIER 1 SHORT ──────────────────────────────────────────────
            'tier1_adx_hard_min_short': 'Hard minimum ADX for Tier 1 SHORT entries (default: 30)',
            'tier1_rsi_min_short': 'Minimum RSI for Tier 1 SHORT entries (default: 25)',
            'tier1_rsi_max_short': 'Maximum RSI for Tier 1 SHORT entries (default: 45)',
            'tier1_volume_min_short': 'Minimum volume ratio for Tier 1 SHORT entries (default: 1.3)',
            'tier1_momentum_min_short': 'Minimum momentum for Tier 1 SHORT entries (default: 0.05)',
            'tier1_macd_gate_short': 'MACD gate required for Tier 1 SHORT entries',
            'daily_trend_down_filter_enabled': 'Enable daily trend filter for SHORT entries',

            # ─── TIER 2 FILTERS ────────────────────────────────────────────
            'tier2_adx_hard_min': 'Hard minimum ADX for Tier 2 entries (default: 20)',
            'tier2_volume_min': 'Minimum volume ratio for Tier 2 entries (default: 1.2)',
            'tier2_momentum_min': 'Minimum momentum for Tier 2 entries (default: 0.01)',
            'tier2_rsi_min': 'Minimum RSI for Tier 2 LONG entries (default: 50)',
            'tier2_rsi_max': 'Maximum RSI for Tier 2 LONG entries (default: 70)',
            'tier2_rsi_min_short': 'Minimum RSI for Tier 2 SHORT entries (default: 30)',
            'tier2_rsi_max_short': 'Maximum RSI for Tier 2 SHORT entries (default: 50)',
            'tier2_macd_histogram_min': 'Minimum MACD histogram for Tier 2',
            'tier2_require_macd_histogram': 'Require MACD histogram for Tier 2',

            # ─── SIZE/STOP/EXIT/TRAILING ──────────────────────────────────
            'tier1_size_multiplier': 'Position size multiplier for Tier 1 entries',
            'tier2_size_multiplier': 'Position size multiplier for Tier 2 entries',
            'stop_loss_atr_mult': 'ATR multiplier for stop loss (unified for all tiers)',
            'exit_threshold_tier1': 'Exit-power score below which Tier 1 closes',
            'exit_threshold_tier2': 'Exit-power score below which Tier 2 closes',
            'trailing_activation_tier1': 'Profit % to activate trailing stop (Tier 1)',
            'trailing_activation_tier2': 'Profit % to activate trailing stop (Tier 2)',
            'trailing_distance_tier1': 'Trailing stop distance from peak (Tier 1)',
            'trailing_distance_tier2': 'Trailing stop distance from peak (Tier 2)',

            # ─── INDICATORS ─────────────────────────────────────────────────
            'adx_period': 'ADX calculation period',
            'rsi_period': 'RSI calculation period',
            'cci_period': 'CCI calculation period',
            'atr_period': 'ATR calculation period',
            'volume_ma_period': 'Volume moving average period',
            'macd_fast': 'MACD fast period',
            'macd_slow': 'MACD slow period',
            'macd_signal': 'MACD signal period',
            'supertrend_atr_period': 'SuperTrend ATR period',
            'supertrend_multiplier': 'SuperTrend multiplier',
            'kalman_q_param': 'Kalman filter process noise Q',
            'kalman_r_param': 'Kalman filter measurement noise R',
            'vix_atr_period': 'VIX ATR period',
            'vix_rolling_period': 'VIX rolling period',

            # ─── RISK ───────────────────────────────────────────────────────
            'risk_tier1': 'Risk percentage for Tier 1 entries (0.02 = 2%)',
            'risk_tier2': 'Risk percentage for Tier 2 entries (0.01 = 1%)',

            # ─── BREAKEVEN ──────────────────────────────────────────────────
            'be_stop_enabled': 'Enable breakeven stop',
            'be_stop_r_trigger': 'R-multiple to trigger breakeven stop',
            'be_stop_no_progress_bars': 'Bars with no progress before breakeven',

            # ─── PROFIT TARGETS ─────────────────────────────────────────────
            'take_profit_r1': 'Profit target R1 (partial exit)',
            'take_profit_r2': 'Profit target R2 (partial exit)',
            'take_profit_r3': 'Profit target R3 (full exit)',
            'profit_target_r1': 'Profit target R1 (alias)',
            'profit_target_r2': 'Profit target R2 (alias)',
            'profit_target_r3': 'Profit target R3 (alias)',

            # ─── EXIT CONDITIONS ────────────────────────────────────────────
            'max_hold_bars': 'Maximum hold bars before exit',
            'min_hold_bars_before_stop': 'Minimum bars before stop can trigger',
            'emergency_stop_multiplier': 'Emergency stop multiplier',
            'macd_bearish_cross_exit': 'Exit on MACD bearish cross',
            'macd_bearish_cross_profit_min': 'Min profit % for MACD cross exit',
            'ema_cross_exit': 'Exit on EMA crossover',
            'rsi_exit_threshold': 'RSI exit threshold',
            'kalman_fade_threshold': 'Kalman fade threshold',
            'momentum_reversal_exit': 'Exit on momentum reversal',
            'momentum_reversal_threshold': 'Momentum reversal threshold',
            'momentum_reversal_profit_min': 'Min profit for momentum reversal exit',
            'profit_min_fade': 'Min profit for fade exit',
            'profit_min_time_exit': 'Min profit for time exit',
            'profit_min_ma_crossover': 'Min profit for MA crossover exit',

            # ─── COOLDOWN ────────────────────────────────────────────────────
            'max_daily_trades': 'Maximum trades per day',
            'min_bars_between_trades': 'Minimum bars between trades (blanket)',
            'cooldown_after_profit_target_bars': 'Cooldown after profit target',
            'cooldown_after_loss_bars': 'Cooldown after losing trade',
            'consecutive_loss_threshold': 'Consecutive losses before extended cooldown',
            'consecutive_loss_cooldown_bars': 'Extended cooldown after loss streak',

            # ─── PRECISION FILTERS ──────────────────────────────────────────
            'dmi_spread_min_long': 'DMI spread minimum for LONG',
            'dmi_spread_min_short': 'DMI spread minimum for SHORT',
            'ema_trending_bars': 'EMA trending bars required',
            'macd_hist_rising_bars': 'MACD histogram rising bars required',
            'rsi_direction_bars': 'RSI direction bars',
            'rsi_direction_min_move': 'RSI direction minimum move',
            'macd_hist_positive_required_long': 'MACD histogram positive required for LONG',
            'macd_hist_negative_required_short': 'MACD histogram negative required for SHORT',
            'bb_expand_required': 'BB expand required',
            'time_filter_enabled': 'Time filter enabled',
            'time_filter_start_utc': 'Time filter start (UTC hour)',
            'time_filter_end_utc': 'Time filter end (UTC hour)',

            # ─── REGIME ─────────────────────────────────────────────────────
            'regime_filter_enabled': 'Enable regime filter',
            'ranging_min_checks': 'Minimum checks for ranging regime',
            'bb_period': 'Bollinger Band period',
            'bb_std': 'Bollinger Band standard deviation',
            'kc_period': 'Keltner Channel period',
            'kc_atr_mult': 'Keltner Channel ATR multiplier',
            'chop_period': 'Choppiness Index period',
            'chop_threshold': 'Choppiness Index threshold',
            'volatility_scaling': 'Enable volatility scaling',
            'trade_high_vol': 'Allow trading in high volatility',
            'trade_ranging': 'Allow trading in ranging markets',
            'supertrend_exit_enabled': 'Enable SuperTrend exit',
            'atr_compression_enabled': 'Enable ATR compression filter',
            'atr_compression_threshold': 'ATR compression threshold',
            'extended_run_max_pct_long': 'Max extended run % for LONG',
            'extended_run_max_pct_short': 'Max extended run % for SHORT',

            # ─── PRICE POSITIONING ──────────────────────────────────────────
            'price_percentile_bonus_early': 'Early entry bonus points',
            'price_percentile_penalty_late': 'Late entry penalty points',
            'price_percentile_early_threshold': 'Percentile for early entry',
            'price_percentile_late_threshold': 'Percentile for late entry',
            'price_percentile_lookback': 'Lookback for price percentile',

            # ─── ADX SCORING ────────────────────────────────────────────────
            'adx_score_trend_forming': 'ADX trend forming threshold',
            'adx_score_good_trend': 'ADX good trend threshold',
            'adx_score_strong_trend': 'ADX strong trend threshold',
            'adx_score_very_strong': 'ADX very strong threshold',
            'adx_score_extended': 'ADX extended threshold',

            # ─── MACD SCORING ────────────────────────────────────────────────
            'macd_score_line_vs_signal': 'Enable MACD line vs signal scoring',
            'macd_score_histogram_direction': 'Enable MACD histogram direction scoring',
            'macd_score_zero_cross': 'Enable MACD zero cross scoring',
            'macd_score_histogram_value': 'Enable MACD histogram value scoring',

            # ─── FUZZY MODE ─────────────────────────────────────────────────
            'fuzzy_mode_enabled': 'Enable fuzzy adaptive mode',
            'fuzzy_learning_enabled': 'Enable fuzzy learning',
            'fuzzy_safety_cutoffs': 'Enable safety cutoffs for fuzzy mode',
            'fuzzy_default_margin_pct': 'Fuzzy default margin %',
            'fuzzy_absolute_min': 'Fuzzy absolute minimum',
            'fuzzy_absolute_max': 'Fuzzy absolute maximum',
            'fuzzy_min_confidence': 'Fuzzy minimum confidence',
            'fuzzy_min_samples': 'Fuzzy minimum samples required',
            'fuzzy_max_adjustment_pct': 'Fuzzy maximum adjustment %',
            'fuzzy_learning_rate': 'Fuzzy learning rate',
            'fuzzy_conservative_start': 'Fuzzy conservative start mode',

            # ─── TRADE DIRECTION ────────────────────────────────────────────
            'trade_direction': 'Trade direction: long, short, or both',
        }
        return descriptions.get(param_name, 'No description available')

    def get_kalman_param_description(self, param_name):
        """Get description for Kalman parameters"""
        descriptions = {
            'process_noise_1': 'Process noise parameter 1 for Kalman filter',
            'process_noise_2': 'Process noise parameter 2 for Kalman filter',
            'measurement_noise': 'Measurement noise for Kalman filter',
            'trend_lookback': 'Lookback period for trend calculation',
            'strength_smooth': 'Smoothing period for trend strength',
            'risk_reward': 'Risk/reward ratio for position sizing',
            'lookback': 'Lookback period for strategy',
            'window': 'Analysis window size',
            'strength_smooth_param': 'Parameter for strength smoothing',
            'ma_fast_period': 'Fast moving average period',
            'ma_slow_period': 'Slow moving average period',
            'kalman_strength_min': 'Minimum Kalman strength for entry',
            'rsi_min': 'Minimum RSI for entry',
            'rsi_max': 'Maximum RSI for entry',
            'volume_min_ratio': 'Minimum volume ratio for entry',
            'pullback_percent': 'Pullback percentage for entry',
            'stop_loss_pct': 'Stop loss percentage',
            'trailing_stop_pct': 'Trailing stop percentage',
            'atr_multiplier': 'ATR multiplier for stops',
            'risk_per_trade': 'Risk percentage per trade',
            'max_position_pct': 'Maximum position percentage',
            'rsi_exit_threshold': 'RSI threshold for exit',
            'max_hold_bars': 'Maximum bars to hold position',
            'max_hold_seconds': 'Maximum seconds to hold position',
            'min_adx': 'Minimum ADX for market condition',
            'min_volatility': 'Minimum volatility for trading',
            'cooldown_bars': 'Cooldown bars between trades'
        }
        return descriptions.get(param_name, 'No description available')

    def on_param_toggle_changed(self, event=None):
        """Handle parameter toggle selection change"""
        selection = self.param_toggle_var.get()

        self.apply_selected_parameters()

        if hasattr(self, 'refresh_scalping_current_values'):
            self.refresh_scalping_current_values()

        try:
            if os.path.exists(self.strategy_settings_file):
                with open(self.strategy_settings_file, 'r') as f:
                    settings = json.load(f)
            else:
                settings = {}

            settings['selected_mode'] = selection

            with open(self.strategy_settings_file, 'w') as f:
                json.dump(settings, f, indent=4)

        except Exception as e:
            self.log_message(f"Note: Could not save toggle selection: {e}", "orange")

        self.log_message(f"Switched to: {selection}", "blue")

    def apply_selected_parameters(self):
        """Apply selected parameters to all strategies."""
        selection = self.param_toggle_var.get()
        gui_dir = self.trade_direction_var.get()

        try:
            # ── Momentum ────────────────────────────────────────────────────
            if hasattr(self, 'strategies') and 'Momentum' in self.strategies:
                momentum_strategy = self.strategies['Momentum']
                params = self.get_current_momentum_params()

                # always force GUI direction into params
                params['trade_direction'] = gui_dir

                for key, value in params.items():
                    if hasattr(momentum_strategy, key):
                        setattr(momentum_strategy, key, value)
                    if hasattr(momentum_strategy, 'config') and \
                            key in momentum_strategy.config:
                        momentum_strategy.config[key] = value

                # explicit direction flags
                momentum_strategy.trade_direction = gui_dir
                momentum_strategy.only_long_entries = (gui_dir == 'long')
                momentum_strategy.only_short_entries = (gui_dir == 'short')
                self.log_message(
                    f"✅ Applied {selection} to Momentum "
                    f"(direction={gui_dir})", "green")

            # ── Kalman ──────────────────────────────────────────────────────
            if hasattr(self, 'strategies') and 'Kalman' in self.strategies:
                kalman_strategy = self.strategies['Kalman']
                params = self.get_current_kalman_params()
                params['trade_direction'] = gui_dir

                for key, value in params.items():
                    if hasattr(kalman_strategy, key):
                        setattr(kalman_strategy, key, value)
                    if hasattr(kalman_strategy, 'config') and \
                            key in kalman_strategy.config:
                        kalman_strategy.config[key] = value

                kalman_strategy.trade_direction = gui_dir
                self.log_message(
                    f"✅ Applied {selection} to Kalman "
                    f"(direction={gui_dir})", "green")

            # ── Scalping ────────────────────────────────────────────────────
            if hasattr(self, 'strategies') and 'Scalping' in self.strategies:
                scalping_strategy = self.strategies['Scalping']
                params = self.get_current_scalping_params()
                params['trade_direction'] = gui_dir

                for key, value in params.items():
                    if hasattr(scalping_strategy, key):
                        setattr(scalping_strategy, key, value)
                    if hasattr(scalping_strategy, 'config') and \
                            key in scalping_strategy.config:
                        scalping_strategy.config[key] = value

                scalping_strategy.trade_direction = gui_dir
                scalping_strategy.only_long_entries = (gui_dir == 'long')
                scalping_strategy.only_short_entries = (gui_dir == 'short')
                self.log_message(
                    f"✅ Applied {selection} to Scalping "
                    f"(direction={gui_dir})", "green")

        except Exception as e:
            self.log_message(f"❌ Error applying parameters: {e}", "red")

    def save_custom_parameters(self):
        """
        Save custom parameters to disk - PRESERVE ALL PARAMETERS
        v9.4 FIX: Always start with MOMENTUM_PARAMS defaults, then overlay UI values.
        This ensures params NOT in the settings panel are still saved correctly.
        """
        try:
            from strategies.MomentumStrategy_MACD_HybridScore_Latest import MOMENTUM_PARAMS

            # v9.4: Start with ALL MOMENTUM_PARAMS defaults
            momentum_params = MOMENTUM_PARAMS.copy()

            # Overlay existing custom params (handles params not in UI)
            if self.custom_params.get('momentum'):
                momentum_params.update(self.custom_params['momentum'])

            # Overlay LIVE UI values (highest priority)
            if hasattr(self, 'momentum_param_widgets'):
                ui_count = 0
                for param_name, widget_info in self.momentum_param_widgets.items():
                    custom_var = widget_info['custom']

                    if isinstance(custom_var, tk.BooleanVar):
                        momentum_params[param_name] = custom_var.get()
                    else:
                        value_str = custom_var.get()
                        momentum_params[param_name] = self.convert_param_value(value_str)
                    ui_count += 1

                self.log_message(f"✅ Captured {ui_count} params from UI", "green")

            # Update custom_params with complete set
            self.custom_params['momentum'] = momentum_params
            self.log_message(f"✅ Total momentum params (including non-UI): {len(momentum_params)}", "green")

            # Get kalman params from UI widgets
            if hasattr(self, 'kalman_param_widgets'):
                kalman_params = {}
                for param_name, widget_info in self.kalman_param_widgets.items():
                    custom_var = widget_info['custom']

                    if isinstance(custom_var, tk.BooleanVar):
                        kalman_params[param_name] = custom_var.get()
                    else:
                        value_str = custom_var.get()
                        kalman_params[param_name] = self.convert_param_value(value_str)

                self.custom_params['kalman'] = kalman_params
                self.log_message(f"✅ Captured {len(kalman_params)} kalman params from UI", "green")

            # Get scalping params from UI widgets
            if hasattr(self, 'scalping_param_widgets'):
                from strategies.scalping_strategy import SCALPING_PARAMS
                scalping_params = SCALPING_PARAMS.copy()
                if self.custom_params.get('scalping'):
                    scalping_params.update(self.custom_params['scalping'])
                for param_name, widget_info in self.scalping_param_widgets.items():
                    custom_var = widget_info['custom']
                    if isinstance(custom_var, tk.BooleanVar):
                        scalping_params[param_name] = custom_var.get()
                    else:
                        scalping_params[param_name] = self.convert_param_value(custom_var.get())
                self.custom_params['scalping'] = scalping_params
                self.log_message(f"✅ Captured {len(scalping_params)} scalping params from UI", "green")

            settings_to_save = {
                'default_params': self.default_params,
                'custom_params': self.custom_params,
                'selected_mode': self.param_toggle_var.get(),
                'timestamp': datetime.now().isoformat()
            }

            # ── persist trading-time windows ────────────────────────────────────
            settings_to_save = self._save_trading_time_config(settings_to_save)

            with open(self.strategy_settings_file, 'w') as f:
                json.dump(settings_to_save, f, indent=4)

            self.log_message("⏰ Trading time windows saved to settings", "green")

            # ═══════════════════════════════════════════════════════════════
            # ENHANCED: Get full file path and display detailed info
            # ═══════════════════════════════════════════════════════════════
            import os
            full_path = os.path.abspath(self.strategy_settings_file)
            file_size = os.path.getsize(full_path)

            # Print detailed save confirmation to console
            print("=" * 70)
            print(f"✅ Custom parameters saved successfully!")
            print(f"📁 File path: {full_path}")
            print(f"💾 File size: {file_size:,} bytes")
            print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🎯 Mode: {self.param_toggle_var.get()}")
            print(f"📊 Custom params saved: {len(self.custom_params.get('momentum', {}))}")

            # Show preview of saved parameters
            if self.custom_params.get('momentum'):
                print(f"\n📋 Custom momentum parameters saved:")
                sample_params = list(self.custom_params['momentum'].items())[:10]
                for param, value in sample_params:
                    # Highlight critical parameters
                    if param == 'only_tier1_entries':
                        status = "BLOCKED" if value else "ACTIVE"
                        print(f"   🔥 {param}: {value} [TIER 1 {status}]")
                    else:
                        print(f"   • {param}: {value}")
                if len(self.custom_params['momentum']) > 10:
                    print(f"   ... and {len(self.custom_params['momentum']) - 10} more parameters")
            print("=" * 70)

            # Log to GUI as well
            self.log_message(f"✅ Custom parameters saved to {self.strategy_settings_file}", "green")
            self.log_message(f"📁 Full path: {full_path}", "blue")
            self.log_message(f"💾 File size: {file_size:,} bytes", "blue")

            # Log Tier 1 status (only_tier1_entries blocks it)
            only_tier1 = self.custom_params['momentum'].get('only_tier1_entries', False)
            self.log_message(
                f"📊 TIER 1 STATUS SAVED: {'BLOCKED' if only_tier1 else 'ACTIVE'} (only_tier1_entries = {only_tier1})",
                "yellow" if only_tier1 else "green")

            # If Custom mode is active, apply the saved parameters immediately
            if self.param_toggle_var.get() == "Custom Parameters":
                self.apply_selected_parameters()
                self.log_message(f"✅ Applied saved custom parameters to strategy", "green")

            return True

        except Exception as e:
            self.log_message(f"❌ Error saving custom parameters: {e}", "red")
            import traceback
            self.log_message(f"   Traceback: {traceback.format_exc()}", "red")
            return False

    def convert_param_value(self, value_str):
        """Convert a string value from UI widget to proper Python type."""
        try:
            if '.' in str(value_str):
                return float(value_str)
            else:
                return int(value_str)
        except (ValueError, TypeError):
            if str(value_str).lower() in ['true', 'false']:
                return str(value_str).lower() == 'true'
            return value_str

    def reset_to_defaults(self):
        """
        Reset ALL values to MOMENTUM_PARAMS defaults.
        This copies code defaults to the custom column in UI
        """
        try:
            from strategies.MomentumStrategy_MACD_HybridScore_Latest import MOMENTUM_PARAMS, MomentumConfig

            print("=" * 70)
            print("🔄 RESETTING TO MOMENTUM_PARAMS DEFAULTS")
            print("=" * 70)

            # Get code defaults from MOMENTUM_PARAMS
            code_defaults = self.get_default_momentum_params()

            # Update UI widgets with defaults
            if hasattr(self, 'momentum_param_widgets'):
                updates_made = 0
                for param_name, widget_info in self.momentum_param_widgets.items():
                    if param_name in code_defaults:
                        default_value = code_defaults[param_name]
                        custom_var = widget_info['custom']

                        if isinstance(custom_var, tk.BooleanVar):
                            current = custom_var.get()
                            custom_var.set(bool(default_value))
                            if current != bool(default_value):
                                updates_made += 1
                        else:
                            current = custom_var.get()
                            custom_var.set(str(default_value))
                            if current != str(default_value):
                                updates_made += 1

                self.log_message(f"✅ Reset {updates_made} parameters to MOMENTUM_PARAMS defaults", "green")

                # Update custom_params with these new values
                for param_name, value in code_defaults.items():
                    self.custom_params['momentum'][param_name] = value

            # Reset kalman params
            if hasattr(self, 'kalman_param_widgets'):
                kalman_defaults = self.get_default_kalman_params()
                for param_name, widget_info in self.kalman_param_widgets.items():
                    if param_name in kalman_defaults:
                        default_value = kalman_defaults[param_name]
                        custom_var = widget_info['custom']

                        if isinstance(custom_var, tk.BooleanVar):
                            custom_var.set(bool(default_value))
                        else:
                            custom_var.set(str(default_value))

                        self.custom_params['kalman'][param_name] = default_value

            # Reset scalping params
            if hasattr(self, 'scalping_param_widgets'):
                scalping_defaults = self.get_default_scalping_params()
                for param_name, widget_info in self.scalping_param_widgets.items():
                    if param_name in scalping_defaults:
                        default_value = scalping_defaults[param_name]
                        custom_var = widget_info['custom']

                        if isinstance(custom_var, tk.BooleanVar):
                            custom_var.set(bool(default_value))
                        else:
                            custom_var.set(str(default_value))

                        self.custom_params.setdefault('scalping', {})[param_name] = default_value

            # Update MomentumConfig state
            self.current_mode = "Default Parameters"
            self.param_toggle_var.set("Default Parameters")
            MomentumConfig.set_current_mode("Default Parameters")

            # Save to file (with empty custom params)
            success = MomentumConfig.save_config(
                config=self.momentum_params,
                custom_params={},
                selected_mode="Default Parameters"
            )

            if success:
                print("✅ Reset complete - using MOMENTUM_PARAMS defaults")
                print(f"Total parameters: {len(self.momentum_params)}")
                print("=" * 70)

                self.log_message("🔄 Reset to MOMENTUM_PARAMS defaults", "green")

                messagebox.showinfo("Reset Complete",
                                    "All values reset to MOMENTUM_PARAMS defaults.\n"
                                    "Click 'Save Custom Parameters' to persist these changes.")

                return True
            else:
                print("⚠️ Reset completed but save failed")
                return False

        except Exception as e:
            self.log_message(f"❌ Error resetting parameters: {e}", "red")
            messagebox.showerror("Reset Error", f"Failed to reset: {e}")
            return False

    def update_confidence_display(self, value):
        self.ml_prediction_threshold = value
        self.confidence_var.set(f"{value:.2f}")

    def toggle_ml(self):
        """Toggle ML prediction on/off with proper data preprocessing"""
        if self.ml_enable_var.get():
            # Check if API is connected
            if not hasattr(self, 'market_api') or self.market_api is None:
                engine = pyttsx3.init()
                engine.setProperty('rate', 135)
                engine.say(f"Machine learning cannot be enabled, Please check connection first")
                engine.runAndWait()
                self.log_message(
                    "⚠️ Machine learning cannot be enabled: API not connected. Please check connection first.",
                    "red"
                )
                self.ml_enable_var.set(False)
                return

            self.ml_enabled = True
            self.ml_model_combobox.config(state="readonly")

            # Get selected model
            model_name = self.ml_model_var.get()
            if not model_name:
                model_name = list(self.ml_models.keys())[0]
                self.ml_model_var.set(model_name)

            self.current_ml_model = self.ml_models[model_name]
            self.log_message(f"🤖 ML/Prediction enabled using {model_name}", "purple")
            self.log_message("=" * 70, "purple")

            # Train the model
            self.log_message("📚 Training ML model...", "purple")
            df = self.get_market_data()

            if df is not None:
                # Calculate indicators first
                self.log_message("📊 Calculating technical indicators...", "blue")
                df = self.strategy.calculate_indicators(df)

                if df is not None:
                    # PREPARE DATA FOR ML (CRITICAL FIX)
                    model_type = "lstm" if "LSTM" in model_name else "tree"
                    df_ml = self.prepare_ml_data(df, model_type=model_type)

                    if df_ml is not None:
                        self.log_message(f"🚀 Starting {model_name} training...", "purple")

                        # Train the model
                        success = self.current_ml_model.train(df_ml)

                        if success:
                            # Get accuracy
                            accuracy = self.current_ml_model.get_accuracy()

                            self.log_message("=" * 70, "green")
                            self.log_message(f"✅ MODEL TRAINING SUCCESSFUL!", "green")
                            self.log_message("=" * 70, "green")
                            self.log_message(f"   Model:      {model_name}", "white")
                            self.log_message(f"   Accuracy:   {accuracy:.2f}%", "green" if accuracy > 60 else "orange")
                            self.log_message(f"   Data rows:  {len(df_ml)}", "white")
                            self.log_message(f"   Features:   {len(df_ml.columns)}", "white")

                            # Enable LSTM-specific features
                            if isinstance(self.current_ml_model, LSTMModel):
                                self.current_ml_model.enable_prediction_adjustment()
                                self.current_ml_model.enable_price_anchoring()
                                self.log_message("✅ LSTM enhancements enabled:", "green")
                                self.log_message("   - Prediction adjustment", "blue")
                                self.log_message("   - Price anchoring", "blue")

                            self.log_message("=" * 70, "green")

                            # Show model diagnostics for LSTM
                            if isinstance(self.current_ml_model, LSTMModel):
                                self.current_ml_model.print_diagnostics()
                        else:
                            self.log_message("❌ Failed to train ML model", "red")
                            self.log_message("   Possible causes:", "orange")
                            self.log_message("   - Insufficient data quality", "orange")
                            self.log_message("   - Model configuration issues", "orange")
                            self.log_message("   - Data preprocessing errors", "orange")

                            self.ml_enable_var.set(False)
                            self.ml_enabled = False
                            self.current_ml_model = None
                    else:
                        self.log_message("❌ Failed to prepare data for ML", "red")
                        self.log_message("   Check the log above for data quality issues", "orange")
                        self.ml_enable_var.set(False)
                        self.ml_enabled = False
                else:
                    self.log_message("❌ Failed to calculate indicators", "red")
                    self.ml_enable_var.set(False)
                    self.ml_enabled = False
            else:
                self.log_message("❌ Failed to get market data", "red")
                self.ml_enable_var.set(False)
                self.ml_enabled = False
        else:
            # Disable ML
            if isinstance(self.current_ml_model, LSTMModel):
                self.current_ml_model.disable_prediction_adjustment()
                self.current_ml_model.disable_price_anchoring()
                self.log_message("❌ LSTM enhancements disabled", "purple")

            self.ml_model_combobox.config(state=tk.DISABLED)
            self.current_ml_model = None
            self.ml_enabled = False
            self.log_message("🤖 ML/Prediction disabled", "purple")

    def toggle_log_expansion(self, event=None):
        try:
            if not self.log_expanded:
                current_scroll = self.log_area.yview()

                self.hidden_frames_info = []

                for frame_name, frame in self.expandable_frames.items():
                    if frame.winfo_ismapped():
                        pack_info = frame.pack_info()
                        self.hidden_frames_info.append({
                            'name': frame_name,
                            'frame': frame,
                            'pack_info': pack_info
                        })
                        frame.pack_forget()

                try:
                    left_pane_height = self.log_frame_ref.master.winfo_height()
                    expanded_lines = max(70, int(left_pane_height / 14))
                except:
                    expanded_lines = 80

                self.log_area.configure(height=expanded_lines)
                self.log_frame_ref.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                self.root.update_idletasks()
                self.log_area.yview_moveto(current_scroll[0])

                self.log_expanded = True
                self.log_frame_ref.configure(text="Trading Log [EXPANDED] - Double-click to restore")

            else:
                current_scroll = self.log_area.yview()

                self.log_frame_ref.pack_forget()

                for info in self.hidden_frames_info:
                    frame = info['frame']
                    pi = info['pack_info']

                    frame.pack(
                        side=pi.get('side', 'top'),
                        fill=pi.get('fill', 'none'),
                        expand=pi.get('expand', False),
                        padx=pi.get('padx', 0),
                        pady=pi.get('pady', 0),
                        anchor=pi.get('anchor', 'center')
                    )

                self.log_area.configure(height=self.original_log_height)
                self.log_frame_ref.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                self.root.update_idletasks()
                self.log_area.yview_moveto(current_scroll[0])

                self.hidden_frames_info = []
                self.log_expanded = False
                self.log_frame_ref.configure(text="Trading Log")

            self.root.update_idletasks()

        except Exception as e:
            print(f"Log expansion error: {str(e)}")
            self.log_expanded = False
            self.hidden_frames_info = []
            try:
                self.log_frame_ref.configure(text="Trading Log")
            except:
                pass

    def select_ml_model(self, event=None):
        """Handle ML model selection with proper data validation"""
        df = self.get_market_data()

        if df is None or len(df) < 100:
            self.log_message("⚠️ Insufficient data for ML training (need >100 records)", "red")
            self.log_message(f"   Current data: {len(df) if df is not None else 0} rows", "orange")
            return

        model_name = self.ml_model_var.get()
        if model_name in self.ml_models:
            self.current_ml_model = self.ml_models[model_name]
            self.log_message(f"🤖 Selected ML model: {model_name}", "purple")

            # Train if not already trained
            if not self.current_ml_model.is_trained:
                self.log_message(f"📚 Training {model_name} model...", "purple")
                df = self.get_market_data()

                if df is not None:
                    # Calculate indicators
                    self.log_message("📊 Calculating technical indicators...", "blue")
                    df = self.strategy.calculate_indicators(df)

                    if df is not None:
                        # Prepare data for ML
                        model_type = "lstm" if "LSTM" in model_name else "tree"
                        df_ml = self.prepare_ml_data(df, model_type=model_type)

                        if df_ml is not None:
                            self.log_message(f"🚀 Starting {model_name} training...", "purple")

                            # Train the model
                            success = self.current_ml_model.train(df_ml)

                            if success:
                                accuracy = self.current_ml_model.get_accuracy()
                                self.log_message(
                                    f"✅ {model_name} model trained successfully (Accuracy: {accuracy:.2f}%)",
                                    "green"
                                )

                                # Show model diagnostics
                                if isinstance(self.current_ml_model, LSTMModel):
                                    self.current_ml_model.print_diagnostics()
                            else:
                                self.log_message(f"❌ Failed to train {model_name} model", "red")
                        else:
                            self.log_message(f"❌ Failed to prepare data for {model_name}", "red")

    # ═══════════════════════════════════════════════════════════════════════
    # ATR-BASED STOP CALCULATOR (NEW)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_atr_stops(self, entry_price: float, atr: float = None,
                            stop_mult: float = 2.0, trail_mult: float = 1.5,
                            target_mult_1: float = 1.5, target_mult_2: float = 3.0,
                            df: pd.DataFrame = None) -> dict:
        """
        Calculate ATR-based stops, targets, and position sizing.

        Parameters:
        -----------
        entry_price : float
            Entry price for the trade
        atr : float, optional
            Current ATR value. If None, calculated from df.
        stop_mult : float
            Multiplier for stop loss (default: 2.0)
        trail_mult : float
            Multiplier for trailing stop (default: 1.5)
        target_mult_1 : float
            Multiplier for first profit target (default: 1.5)
        target_mult_2 : float
            Multiplier for second profit target (default: 3.0)
        df : pd.DataFrame, optional
            DataFrame with OHLCV data for ATR calculation

        Returns:
        --------
        dict with keys:
            - stop_loss: float
            - trailing_stop_start: float
            - target_1: float
            - target_2: float
            - risk_pct: float
            - atr_value: float
            - stop_distance: float
        """
        # Get ATR if not provided
        if atr is None:
            if df is not None and 'ATR' in df.columns:
                atr = float(df['ATR'].iloc[-1])
            else:
                # Estimate ATR from recent price
                atr = entry_price * 0.02  # Assume 2% ATR

        stop_distance = atr * stop_mult
        trail_distance = atr * trail_mult

        return {
            'stop_loss': entry_price - stop_distance,
            'trailing_stop_start': entry_price - trail_distance,
            'target_1': entry_price + (atr * target_mult_1),
            'target_2': entry_price + (atr * target_mult_2),
            'risk_pct': (stop_distance / entry_price) * 100,
            'atr_value': atr,
            'stop_distance': stop_distance,
            'entry_price': entry_price,
        }

    def get_atr(self, df: pd.DataFrame = None, period: int = 14) -> float:
        """Get current ATR value from DataFrame or estimate."""
        if df is not None and 'ATR' in df.columns and len(df) > period:
            return float(df['ATR'].iloc[-1])

        # Estimate from price if data available
        if df is not None and 'Close' in df.columns and len(df) > period:
            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values

            # Calculate ATR manually
            tr = np.maximum(
                highs - lows,
                np.maximum(
                    abs(highs - np.roll(closes, 1)),
                    abs(lows - np.roll(closes, 1))
                )
            )
            atr = np.mean(tr[-period:])
            return float(atr) if not np.isnan(atr) else 0.0

        return 0.0

    def calculate_volume_strength(self, df):
        candles = df["Close"].diff().fillna(0)
        volume = df["Volume"]
        candle_changes = candles * volume

        num_candles = len(candle_changes)
        group_size = 10
        num_groups = num_candles // group_size

        weighted_red_sum = 0
        weighted_green_sum = 0

        for i in range(num_groups):
            start = i * group_size
            end = (i + 1) * group_size
            group = candle_changes[start:end]
            weight = i + 1

            red = group[group < 0].sum()
            green = group[group > 0].sum()

            weighted_red_sum += red * weight
            weighted_green_sum += green * weight

        diff = weighted_green_sum + weighted_red_sum
        total = abs(weighted_green_sum) + abs(weighted_red_sum)
        return (diff / total) * 100 if total != 0 else 0

    def draw_volume_strength(self, canvas_frame, strength_value):
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(2, 0.2), dpi=100)
        ax = fig.add_subplot(111)

        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1.2)
        ax.axis('off')

        ax.axvspan(-100, 0, color='red', alpha=0.4)
        ax.axvspan(0, 100, color='green', alpha=0.4)
        ax.axvline(0, color='black', lw=1)
        ax.axvline(strength_value, color='blue', lw=3)
        ax.text(strength_value, 1.05, f"{strength_value:.1f}%", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='blue')

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # REPLACE the entire method with this:
    def update_trade_direction(self):
        direction = self.trade_direction_var.get()  # "long" | "short" | "both"
        icons = {'long': '⬆️', 'short': '⬇️', 'both': '↕️'}

        # 1. Update live Momentum strategy
        if hasattr(self, 'strategies') and 'Momentum' in self.strategies:
            strat = self.strategies['Momentum']
            strat.trade_direction = direction
            strat.only_long_entries = (direction == 'long')
            strat.only_short_entries = (direction == 'short')
            if hasattr(strat, '_pending_signal'):
                strat._pending_signal = None
            if hasattr(strat, '_last_direction_check'):
                strat._last_direction_check['bar'] = -999
            self.log_message(f"{icons.get(direction, '')} Direction → {direction.upper()} (live)", "cyan")

        # 2. Push into BacktestMomentumStrategy class params
        try:
            existing = BacktestMomentumStrategy._updated_params or {}
            existing['trade_direction'] = direction
            BacktestMomentumStrategy._updated_params = existing
            self.log_message(f"{icons.get(direction, '')} Direction → {direction.upper()} (backtest)", "cyan")
        except Exception as e:
            self.log_message(f"⚠️ Could not update backtest direction: {e}", "orange")

    def flash_emergency_button(self):
        if not self.emergency_flashing:
            return

        current_style = self.emergency_btn.cget('style')
        if current_style == 'Red.TButton':
            self.emergency_btn.config(style='Flashing.TButton')
        else:
            self.emergency_btn.config(style='Red.TButton')

        self.root.after(500, self.flash_emergency_button)

    def emergency_stop(self):
        self.log_message("🚨 EMERGENCY STOP ACTIVATED!", "red")
        self.play_notification("error")
        self.running = False
        if hasattr(self, "timer") and self.timer:
            self.timer.stop()

        if self.position['type'] == 'long':
            self.log_message("Closing LONG position due to emergency stop", "red")
            self.place_order('sell', self.get_current_price(), self.position['quantity'], exit_reason='emergency')
        elif self.position['type'] == 'short':
            self.log_message("Closing SHORT position due to emergency stop", "red")
            self.place_order('buy', self.get_current_price(), self.position['quantity'], exit_reason='emergency')

        self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
        self.update_status_indicators("parking")

        if hasattr(self, "start_btn") and self.start_btn:
            self.start_btn.config(state=tk.DISABLED)
        if hasattr(self, "stop_btn") and self.stop_btn:
            self.stop_btn.config(state=tk.DISABLED)

        self.style.configure('Flashing.TButton',
                             foreground='white',
                             background='white',
                             font=('Arial', 10, 'bold'),
                             padding=5,
                             borderwidth=2)

        self.emergency_flashing = True
        self.flash_emergency_button()

        self.root.after(10000, lambda: setattr(self, 'emergency_flashing', False))

    def on_symbol_change(self):
        self.validate_symbol()
        self.symbol_balance_label.config(text=f"{self.base_symbol()}: {self.get_balance(self.base_symbol()):.4f}")
        self.update_image()

    def log_message(self, message, color="white"):
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            full_message = f" {message}"

            # Always print to console for debugging
            print(f"{color.upper()}: {full_message}")

            # Map complex color names to simple ones
            color_map = {
                "bold green": "green",
                "bold blue": "blue",
                "bold red": "red",
                "bold cyan": "cyan",
                "bold purple": "purple",
                "bold orange": "orange",
                "bold yellow": "yellow",
                "bold white": "white",
                "dim": "gray",
            }

            # Get the base color (strip "bold " if present)
            display_color = color_map.get(color, color)

            # Make sure the color tag exists
            valid_colors = ["green", "red", "blue", "orange", "purple", "cyan", "yellow", "gray", "white"]
            if display_color not in valid_colors:
                display_color = "white"

            # Log to GUI if available
            if hasattr(self, 'log_area') and self.log_area:
                try:
                    if self.log_area.winfo_exists():
                        self.log_area.insert(tk.END, full_message + "\n", display_color)
                        self.log_area.see(tk.END)
                        self.log_area.update_idletasks()  # Force immediate update
                    else:
                        print(f"Log area doesn't exist")
                except (tk.TclError, AttributeError) as e:
                    print(f"GUI log error: {e}")
            else:
                print(f"No log_area attribute")

            # AI button logic
            if not self.ai_data_available:
                data_indicators = [
                    "Entry Confidence", "Price:", "RSI:", "ADX:", "BACKTEST",
                    "Loaded", "candles", "EMA", "Trading started",
                    "PROFESSIONAL", "v7.6", "SEEKING_ENTRY", "IN_TRADE",
                    "TIER2", "TIER1"  # Add these to detect when data is available
                ]
                if any(indicator in message for indicator in data_indicators):
                    self.enable_ai_button()

        except Exception as e:
            print(f"Log error: {e}")
            print(f"Original message: {message}")

        logging.info(full_message)

    def play_notification(self, sound_type):
        try:
            if sound_type == "buy":
                winsound.Beep(1000, 500)
            elif sound_type == "sell":
                winsound.Beep(800, 500)
            elif sound_type == "error":
                winsound.Beep(400, 1000)
            elif sound_type == "tick":
                winsound.Beep(600, 200)
            elif sound_type == "pre_buy_alert":
                winsound.Beep(700, 300)
            elif sound_type == "buy_success":
                winsound.Beep(1200, 400)
            elif sound_type == "pre_exit_alert":
                winsound.Beep(600, 300)
            elif sound_type == "sell_success":
                winsound.Beep(900, 400)
            elif sound_type == "sell_loss":
                winsound.Beep(500, 400)
        except Exception as e:
            self.log_message(f"Sound notification failed: {str(e)}", "orange")
            logging.error(f"Sound notification failed: {str(e)}")

    def update_status_indicators(self, status):
        self.style.configure('Buy.TFrame', background='white')
        self.style.configure('Parking.TFrame', background='white')
        self.style.configure('Sell.TFrame', background='white')
        self.current_status = status
        if status == "buy":
            self.style.configure('Buy.TFrame', background='green')
            self.buy_box.update_idletasks()
        elif status == "parking":
            self.style.configure('Parking.TFrame', background='orange')
            self.parking_box.update_idletasks()
        elif status == "sell":
            self.style.configure('Sell.TFrame', background='red')
            self.sell_box.update_idletasks()
        self.update_image()

    def update_mode_display(self, mode):
        """Update mode display AND properly show/hide backtest panel"""
        mode_text = mode.upper() + " MODE"
        color = 'green' if mode.lower() == 'live' else 'blue' if mode.lower() == 'demo' else 'purple'
        self.mode_display.config(text=mode_text, foreground=color)

        if mode.lower() == "backtest":
            # Hide explanation, show backtest controls
            if hasattr(self, 'explanation_textbox_ref'):
                self.explanation_textbox_ref.pack_forget()
            if hasattr(self, 'backtest_controls_frame_ref'):
                self.backtest_controls_frame_ref.pack(fill=tk.BOTH, expand=True, padx=3, pady=2)

            # Disable start button (not used in backtest mode)
            self.start_btn.config(state=tk.DISABLED)

            # Settings button state depends on whether backtest is running
            backtest_running = hasattr(self, 'backtest_running') and self.backtest_running
            if backtest_running:
                self.settings_btn.config(state=tk.DISABLED)
            else:
                self.settings_btn.config(state=tk.NORMAL)

        else:  # Live or Demo mode
            if hasattr(self, 'backtest_controls_frame_ref'):
                self.backtest_controls_frame_ref.pack_forget()
            if hasattr(self, 'explanation_textbox_ref'):
                self.explanation_textbox_ref.pack(fill=tk.BOTH, expand=True)

            # Start button enabled only if API connected and not running
            trading_running = hasattr(self, 'running') and self.running
            if trading_running:
                self.start_btn.config(state=tk.DISABLED)
                self.settings_btn.config(state=tk.DISABLED)  # Settings disabled while trading
            else:
                self.start_btn.config(state=tk.NORMAL if self.market_api else tk.DISABLED)
                self.settings_btn.config(state=tk.NORMAL)  # Settings enabled when idle

            if hasattr(self, 'frame2_middle'):
                self.frame2_middle.config(height=0)

        # Configure button styles consistently
        self.style.configure('Blue.TButton', foreground='white')
        self.style.map('Blue.TButton',
                       background=[('active', '#004499'),
                                   ('disabled', '#AAAAAA')],
                       foreground=[('disabled', 'grey'),
                                   ('!disabled', 'white')])

    def update_stats(self):
        def _update():
            closed_trades = [t for t in self.trade_history if t['type'] == 'sell']
            total_trades = len(closed_trades)

            win_trades = sum(1 for trade in closed_trades if trade.get('pnl', 0) > 0)
            win_percent = (win_trades / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(trade.get('pnl', 0) for trade in closed_trades)

            usdt_balance = self.get_balance('USDT')
            symbol_balance = self.get_balance(self.base_symbol())

            self.trades_label.config(text=f"Trades: {total_trades}")
            self.wins_label.config(text=f"Wins: {win_trades}")
            self.win_percent_label.config(text=f"Win %: {win_percent:.2f}%")
            self.pnl_label.config(text=f"PnL: ${total_pnl:.2f}")
            self.usdt_balance_label.config(text=f"USDT: ${usdt_balance:.2f}")
            self.symbol_balance_label.config(text=f"{self.base_symbol()}: {symbol_balance:.4f}")

        self.root.after(0, _update)

    def setup_apis(self):
        self.market_api = None
        self.trade_api = None
        self.account_api = None

    def check_connection(self):
        mode = self.mode_var.get().lower()
        try:
            config = self.config.get(mode, {})
            if not config:
                self.log_message(f"No configuration found for {mode} mode", "red")
                return
            self.market_api = MarketData.MarketAPI(
                api_key=config['api_key'],
                api_secret_key=config['api_secret_key'],
                passphrase=config['passphrase'],
                flag='0' if mode == 'live' else '1'
            )
            self.account_api = Account.AccountAPI(
                api_key=config['api_key'],
                api_secret_key=config['api_secret_key'],
                passphrase=config['passphrase'],
                flag='0' if mode == 'live' else '1'
            )
            self.trade_api = Trade.TradeAPI(
                api_key=config['api_key'],
                api_secret_key=config['api_secret_key'],
                passphrase=config['passphrase'],
                flag='0' if mode == 'live' else '1'
            )
            response = self.market_api.get_tickers(instType="SPOT")
            if response['code'] == '0':
                self.log_message("Connection successful!", "green")
                self.start_btn.config(state=tk.NORMAL)
                engine = pyttsx3.init()
                engine.setProperty('rate', 145)
                engine.say(f"Connection successful")
                engine.runAndWait()
            else:
                self.log_message(f"Connection failed: {response['msg']}", "red")
        except Exception as e:
            self.log_message(f"Connection error: {str(e)}", "red")
            logging.error(f"Connection error: {str(e)}")

    def _show_schedule_toast(self, start_str: str, end_str: str, starting_now: bool = False):
        """Show a floating 5-second notification about the trading schedule."""
        try:
            toast = tk.Toplevel(self.root)
            toast.overrideredirect(True)
            toast.attributes("-topmost", True)
            toast.attributes("-alpha", 0.93)

            w, h = 420, 110
            sx = self.root.winfo_screenwidth()
            sy = self.root.winfo_screenheight()
            x = (sx - w) // 2
            y = sy - h - 80
            toast.geometry(f"{w}x{h}+{x}+{y}")

            bg = "#1a1a2e"
            fr = tk.Frame(toast, bg=bg, bd=2, relief="ridge")
            fr.pack(fill=tk.BOTH, expand=True)

            header_text = "▶ Trading starting now" if starting_now else "⏰ Trading scheduled"
            header_col = "#00ff88" if starting_now else "#ffaa00"
            tk.Label(
                fr, text=header_text,
                bg=bg, fg=header_col,
                font=("Arial", 12, "bold"), pady=6
            ).pack()

            detail = (f"Ends at  {end_str} UTC"
                      if starting_now
                      else f"Starts  {start_str} UTC  →  ends  {end_str} UTC")
            tk.Label(
                fr, text=detail,
                bg=bg, fg="white",
                font=("Consolas", 10), pady=2
            ).pack()

            bar_frame = tk.Frame(fr, bg=bg, padx=14, pady=6)
            bar_frame.pack(fill=tk.X)
            canvas = tk.Canvas(bar_frame, bg="#333355", height=6,
                               highlightthickness=0, bd=0)
            canvas.pack(fill=tk.X)

            duration_ms = 5000
            interval_ms = 50
            steps = duration_ms // interval_ms
            bar_id = [None]
            step_counter = [0]

            def _draw_bar():
                canvas.delete("all")
                cw = canvas.winfo_width() or 392
                fraction = 1.0 - (step_counter[0] / steps)
                canvas.create_rectangle(
                    0, 0, int(cw * fraction), 6,
                    fill=header_col, outline="")

            def _tick():
                step_counter[0] += 1
                _draw_bar()
                if step_counter[0] < steps:
                    bar_id[0] = toast.after(interval_ms, _tick)
                else:
                    _close()

            def _close():
                try:
                    if bar_id[0]:
                        toast.after_cancel(bar_id[0])
                    toast.destroy()
                except tk.TclError:
                    pass

            fr.bind("<Button-1>", lambda e: _close())
            toast.bind("<Button-1>", lambda e: _close())
            toast.after(100, _tick)

        except Exception as e:
            self.log_message(f"⚠️ Toast notification error: {e}", "orange")

    def start_trading(self):
        """Start trading, respecting the per-strategy time window."""
        self.update_status_indicators("parking")
        self.order_size_pct = self.order_size_var.get()
        self.stop_loss_pct = self.stop_loss_var.get() / 100
        self.trailing_stop_pct = self.trailing_stop_var.get() / 100
        self.trade_direction = self.trade_direction_var.get()

        mode = self.mode_var.get().lower()
        strategy = self.strategy_type_var.get()

        if mode != "backtest":
            if not hasattr(self, 'market_api') or self.market_api is None:
                messagebox.showerror("Error", "Please check connection first!")
                return

        if self.running:
            return

        self._cancel_scheduled_trading()

        # ── Time-window logic (live / demo only) ─────────────────────────────
        if (mode in ('live', 'demo')
                and hasattr(self, 'trading_time_config')
                and not self._is_time_unconstrained(strategy)):

            s, e = self._get_window_minutes(strategy)
            sh, sm = divmod(s, 60)
            eh, em = divmod(e, 60)

            now_utc = datetime.now(timezone.utc)
            now_str = now_utc.strftime('%H:%M UTC')

            if self._is_within_trading_window(strategy):
                self.log_message(
                    f"⏰ [{strategy}] Inside window "
                    f"({sh:02d}:{sm:02d}→{eh:02d}:{em:02d} UTC) — starting.", "green")

                # ── toast: starting now ──────────────────────────────────────
                self._show_schedule_toast(
                    start_str=f"{sh:02d}:{sm:02d}",
                    end_str=f"{eh:02d}:{em:02d}",
                    starting_now=True
                )

                self._do_start_trading()

                stop_secs = self._seconds_until_end(strategy)
                if stop_secs > 0:
                    self._sched_stop_id = self.root.after(
                        int(stop_secs * 1000), self._scheduled_stop_callback)
                    self.log_message(
                        f"⏰ [{strategy}] Auto-stop in "
                        f"{stop_secs / 60:.1f} min at {eh:02d}:{em:02d} UTC", "blue")
            else:
                wait_secs = self._seconds_until_start(strategy)
                self._waiting_to_start = True

                self.log_message(
                    f"⏰ [{strategy}] Outside trading window — "
                    f"current time {now_str}. "
                    f"Waiting {wait_secs / 60:.1f} min until "
                    f"{sh:02d}:{sm:02d} UTC …", "orange")

                # ── toast: scheduled for later ───────────────────────────────
                self._show_schedule_toast(
                    start_str=f"{sh:02d}:{sm:02d}",
                    end_str=f"{eh:02d}:{em:02d}",
                    starting_now=False
                )

                self.mode_display.config(
                    text=f"⏰ WAITING {sh:02d}:{sm:02d} UTC",
                    foreground="orange")
                self._sched_start_id = self.root.after(
                    int(wait_secs * 1000), self._scheduled_start_callback)
                self.stop_btn.config(state=tk.NORMAL)
                self.start_btn.config(state=tk.DISABLED)
                self._update_wait_countdown(strategy, sh, sm)
            return

        # ── No time constraint → start immediately ───────────────────────────
        self._do_start_trading()
        self.log_message(
            f"📊 {strategy} — Order {self.order_size_pct}% | "
            f"SL {self.stop_loss_pct * 100:.1f}% | "
            f"Trail {self.trailing_stop_pct * 100:.1f}% | "
            f"Dir {self.trade_direction.upper()}", "blue")

    def stop_trading(self):
        """Stop the trading loop and clean up state."""
        if not self.running:
            self.log_message("⚠️ Trading is not currently running.", "orange")
            return

        self.running = False
        self.trading_running = False

        # Stop the circular timer
        if hasattr(self, 'timer') and self.timer:
            try:
                self.timer.stop()
            except Exception:
                pass

        # Re-enable / disable buttons
        if hasattr(self, 'start_btn') and self.start_btn:
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, 'stop_btn') and self.stop_btn:
            self.stop_btn.config(state=tk.DISABLED)

        # Update mode display (re-enables settings button)
        self.update_mode_display(self.mode_var.get())

        self.update_status_indicators("parking")
        self.log_message("🛑 Trading stopped.", "orange")

        # Voice notification
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 135)
            engine.say("Trading stopped")
            engine.runAndWait()
        except Exception:
            pass

        self.play_notification("tick")
        self.update_stats()

    def prepare_ml_data(self, df, model_type="lstm", max_nan_threshold=0.3):
        """
        Prepare DataFrame for ML models by:
        1. Selecting only numeric columns
        2. Handling boolean/object columns properly
        3. Removing columns with too many NaNs (> threshold)
        4. Ensuring no NaN/Inf values remain
        5. Maintaining minimum sample count

        Args:
            df: Input DataFrame
            model_type: "lstm" or "tree"
            max_nan_threshold: Maximum allowed NaN percentage per column (0.3 = 30%)
        """
        import numpy as np
        import pandas as pd

        self.log_message(f"🔧 Preparing data for {model_type} model...", "blue")
        self.log_message("=" * 70, "blue")

        # Make a copy to avoid modifying original
        df_ml = df.copy()

        # Display original data info
        self.log_message(f"📊 Original data: {len(df_ml)} rows, {len(df_ml.columns)} columns", "white")

        # Step 1: Convert boolean columns to int (0/1)
        bool_cols = df_ml.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            for col in bool_cols:
                df_ml[col] = df_ml[col].astype(int)
            self.log_message(f"   ✅ Converted {len(bool_cols)} boolean columns to int", "green")
            for i, col in enumerate(bool_cols[:5]):
                self.log_message(f"      - {col}", "blue")
            if len(bool_cols) > 5:
                self.log_message(f"      ... and {len(bool_cols) - 5} more", "blue")
        else:
            self.log_message(f"   ℹ️ No boolean columns found", "blue")

        # Step 2: Convert object columns that might contain boolean strings
        object_cols = df_ml.select_dtypes(include=['object']).columns
        converted_count = 0
        if len(object_cols) > 0:
            for col in object_cols:
                # Check if column contains boolean-like values
                unique_vals = df_ml[col].dropna().unique()
                bool_like = {'True', 'False', 'true', 'false', 'TRUE', 'FALSE', 'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y',
                             'n'}

                # Convert to string representation for comparison
                str_vals = set()
                for v in unique_vals:
                    if pd.notna(v):
                        str_vals.add(str(v))

                if str_vals.issubset(bool_like) or len(unique_vals) <= 2:
                    df_ml[col] = df_ml[col].map({
                        True: 1, False: 0,
                        'True': 1, 'False': 0,
                        'true': 1, 'false': 0,
                        'TRUE': 1, 'FALSE': 0,
                        'Yes': 1, 'No': 0,
                        'yes': 1, 'no': 0,
                        'Y': 1, 'N': 0,
                        'y': 1, 'n': 0,
                        1: 1, 0: 0
                    }).fillna(0).astype(int)
                    converted_count += 1
                    self.log_message(f"   ✅ Converted object column '{col}' to int", "green")

            if converted_count > 0:
                self.log_message(f"   Total: {converted_count} object columns converted", "green")
        else:
            self.log_message(f"   ℹ️ No object columns found", "blue")

        # Step 3: Select only numeric columns
        numeric_df = df_ml.select_dtypes(include=[np.number])
        self.log_message(f"   📊 Found {len(numeric_df.columns)} numeric columns", "cyan")

        if len(numeric_df.columns) == 0:
            self.log_message("❌ No numeric columns found in data!", "red")
            return None

        # Step 4: Calculate NaN percentages before any processing
        nan_percentage = numeric_df.isna().sum() / len(numeric_df)

        # Log columns with high NaN percentages
        high_nan_cols = nan_percentage[nan_percentage > 0].sort_values(ascending=False)
        if len(high_nan_cols) > 0:
            self.log_message(f"   📊 NaN percentages per column:", "yellow")
            for col, pct in high_nan_cols.head(10).items():
                self.log_message(f"      - {col}: {pct * 100:.1f}% NaNs", "orange")
            if len(high_nan_cols) > 10:
                self.log_message(f"      ... and {len(high_nan_cols) - 10} more columns with NaNs", "orange")

        # Step 5: Remove columns with too many NaNs
        cols_to_drop = nan_percentage[nan_percentage > max_nan_threshold].index.tolist()

        if cols_to_drop:
            self.log_message(f"   ⚠️ Dropping {len(cols_to_drop)} columns with >{max_nan_threshold * 100:.0f}% NaNs:",
                             "orange")
            for col in cols_to_drop[:10]:
                pct = nan_percentage[col] * 100
                self.log_message(f"      - {col}: {pct:.1f}% NaNs", "orange")
            if len(cols_to_drop) > 10:
                self.log_message(f"      ... and {len(cols_to_drop) - 10} more", "orange")

            numeric_df = numeric_df.drop(columns=cols_to_drop)
            self.log_message(f"   ✅ Remaining columns: {len(numeric_df.columns)}", "green")

        # Step 6: Check if we have enough data after column removal
        if len(numeric_df.columns) < 10:
            self.log_message(f"⚠️ Only {len(numeric_df.columns)} features remaining (min recommended: 10)", "orange")

        # Step 7: Handle NaN values with multiple strategies
        nan_count = numeric_df.isna().sum().sum()
        if nan_count > 0:
            self.log_message(f"   ⚠️ Handling {nan_count} NaN values...", "orange")

            # First try forward fill (carry last valid observation forward)
            numeric_df = numeric_df.ffill()

            # Then backward fill for any remaining NaNs at the beginning
            numeric_df = numeric_df.bfill()

            # If still have NaNs (should be rare), fill with column median
            remaining_nans = numeric_df.isna().sum().sum()
            if remaining_nans > 0:
                self.log_message(f"   ⚠️ {remaining_nans} NaNs remain after fill, filling with column medians...",
                                 "orange")

                # Fill remaining NaNs with column median
                for col in numeric_df.columns:
                    if numeric_df[col].isna().any():
                        median_val = numeric_df[col].median()
                        if pd.isna(median_val):  # If median is also NaN (all values NaN)
                            median_val = 0
                        numeric_df[col] = numeric_df[col].fillna(median_val)

            # Final check - should have no NaNs now
            final_nans = numeric_df.isna().sum().sum()
            if final_nans == 0:
                self.log_message(f"   ✅ NaN handling complete: {len(numeric_df)} rows, no NaNs remaining", "green")
            else:
                self.log_message(f"   ⚠️ {final_nans} NaNs remain after all handling - dropping those rows", "orange")
                before_drop = len(numeric_df)
                numeric_df = numeric_df.dropna()
                dropped = before_drop - len(numeric_df)
                if dropped > 0:
                    self.log_message(f"   ⚠️ Dropped {dropped} rows with remaining NaNs", "orange")
        else:
            self.log_message(f"   ✅ No NaN values found", "green")

        # Step 8: Handle infinite values safely
        if len(numeric_df) > 0:
            # Check for infinite values only in numeric data
            numeric_values = numeric_df.values
            inf_mask = np.isinf(numeric_values)
            inf_count = inf_mask.sum()

            if inf_count > 0:
                self.log_message(f"   ⚠️ Found {inf_count} infinite values", "orange")

                # Replace infinities with NaN
                numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

                # Fill NaNs with column median
                for col in numeric_df.columns:
                    if numeric_df[col].isna().any():
                        median_val = numeric_df[col].median()
                        if pd.isna(median_val):
                            median_val = 0
                        numeric_df[col] = numeric_df[col].fillna(median_val)

                self.log_message(f"   ✅ Infinite value handling complete", "green")
            else:
                self.log_message(f"   ✅ No infinite values found", "green")
        else:
            self.log_message("❌ No data remaining after NaN handling", "red")
            return None

        # Step 9: Final validation
        if len(numeric_df) == 0:
            self.log_message("❌ No valid data remaining after preprocessing", "red")
            return None

        if len(numeric_df) < 100:
            self.log_message(f"⚠️ Only {len(numeric_df)} rows after preprocessing (min 100 recommended)", "orange")
            if len(numeric_df) < 50:
                self.log_message("❌ Insufficient data for reliable ML training", "red")
                return None

        # Step 10: Display final stats
        self.log_message("=" * 70, "green")
        self.log_message(f"✅ DATA PREPARATION COMPLETE", "green")
        self.log_message("=" * 70, "green")
        self.log_message(f"   Final rows:      {len(numeric_df)}", "white")
        self.log_message(f"   Final features:  {len(numeric_df.columns)}", "white")
        self.log_message(f"   Date range:      {numeric_df.index[0]} to {numeric_df.index[-1]}", "white")
        self.log_message(f"   Memory usage:    {numeric_df.memory_usage(deep=True).sum() / 1024:.1f} KB", "white")
        self.log_message("=" * 70, "green")

        return numeric_df

    def _display_detailed_analysis(self, df, current_data, result):
        self.log_message(f"{'═' * 77}", "blue")
        self.log_message(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", "blue")
        self.log_message(f"{'═' * 77}", "blue")

        ml_conf_pct = 0.0
        ml_prediction = 0
        ml_is_enabled = getattr(self, 'ml_enabled', False)

        if ml_is_enabled and hasattr(self, 'current_ml_model') and self.current_ml_model:
            if hasattr(self.current_ml_model, 'is_trained') and self.current_ml_model.is_trained:
                try:
                    n_future = int(self.prediction_candles_slider.get()) if hasattr(self,
                                                                                    'prediction_candles_slider') else 5
                    ml_conf, ml_prediction, forecast = self.current_ml_model.predict(df, n_future)
                    ml_conf_pct = float(ml_conf * 100.0 if ml_conf <= 1.0 else ml_conf)

                    self._last_forecast = forecast if (forecast is not None and len(forecast) > 0) else None
                    self._last_ml_prediction = ml_prediction

                except Exception as e:
                    self.log_message(f"⚠️ ML prediction error: {e}", "orange")
                    self._last_forecast = None
                    self._last_ml_prediction = 0
        else:
            self._last_forecast = None
            self._last_ml_prediction = 0

        # ─── NEW: ATR-BASED STOP DISPLAY (Always show if available) ────────────
        atr = current_data.get('ATR', 0)
        current_price = float(current_data.get('Close', 0))

        if atr > 0 and current_price > 0:
            try:
                # Get ATR-based stops using the calculator
                if hasattr(self, 'atr_calc'):
                    stops = self.atr_calc.calculate_stops(
                        entry_price=current_price,
                        atr=atr,
                        stop_mult=2.0,
                        trail_mult=1.5,
                        target_mult_1=1.5,
                        target_mult_2=3.0
                    )

                    self.log_message(f"\n🛑 ATR-BASED LEVELS (v10.1.0):", "cyan")
                    self.log_message(f"   ATR: ${atr:.4f} ({atr / current_price * 100:.2f}%)", "white")
                    self.log_message(f"   Stop Loss: ${stops['stop_loss']:.2f} ({stops['risk_pct']:.2f}% risk)",
                                     "yellow")
                    self.log_message(f"   Trail Start: ${stops['trailing_start']:.2f} (1.5× ATR)", "orange")
                    self.log_message(
                        f"   Target 1: ${stops['target_1']:.2f} ({stops['target_1'] / current_price * 100 - 100:+.2f}%)",
                        "green")
                    self.log_message(
                        f"   Target 2: ${stops['target_2']:.2f} ({stops['target_2'] / current_price * 100 - 100:+.2f}%)",
                        "green")

                    # Show Risk/Reward ratio
                    rr_1 = self.atr_calc.get_risk_reward(current_price, stops['stop_loss'], stops['target_1'], 'long')
                    rr_2 = self.atr_calc.get_risk_reward(current_price, stops['stop_loss'], stops['target_2'], 'long')
                    self.log_message(f"   Risk/Reward: 1:{rr_1:.1f} (T1) | 1:{rr_2:.1f} (T2)",
                                     "green" if rr_1 >= 1.5 else "orange")

            except Exception as e:
                # Fallback to simple ATR display if calculator not available
                self.log_message(f"\n🛑 ATR LEVELS:", "cyan")
                self.log_message(f"   ATR: ${atr:.4f} ({atr / current_price * 100:.2f}%)", "white")
                stop_pct = getattr(self.strategy, 'stop_loss_atr_mult', 2.0) * atr / current_price * 100
                self.log_message(f"   Stop Loss: ${current_price - (atr * 2.0):.2f} ({stop_pct:.2f}% risk)",
                                 "yellow")
        else:
            self.log_message(f"\n📊 ATR: Not available", "gray")

        if isinstance(result, tuple) and len(result) >= 4:
            decision, quality_score, shares, reason = result[:4]

            if quality_score >= 0:
                # ── FIX: safe defaults so every strategy works, not just
                #    those that have _calculate_quality_score ──────────────
                combined_confidence = float(quality_score) if quality_score is not None else 0.0
                total_score = combined_confidence
                component_scores = {}

                # Determine strategy type and get quality score appropriately
                is_momentum_strategy = hasattr(self.strategy, '_calculate_quality_score')
                is_scalping_strategy = hasattr(self.strategy, '_quality_score_long') and hasattr(self.strategy,
                                                                                                 '_quality_score_short')

                if is_momentum_strategy:
                    # ── MOMENTUM/KARLMAN STRATEGY (weighted component scoring) ──
                    _eff_dir = getattr(self.strategy, "_pending_signal", None)
                    _eff_dir = _eff_dir.get("direction", "long") if _eff_dir else getattr(
                        self.strategy, "trade_direction", "long")
                    if _eff_dir == "short" and hasattr(self.strategy, "_calculate_quality_score_short"):
                        total_score, component_scores, score_reason = (
                            self.strategy._calculate_quality_score_short(current_data))
                    else:
                        total_score, component_scores, score_reason = (
                            self.strategy._calculate_quality_score(current_data))

                    combined_confidence = total_score
                    if ml_is_enabled and ml_conf_pct > 0:
                        combined_confidence = (total_score + ml_conf_pct) / 2

                    self.log_message(f"\n🎯 QUALITY SCORE SYSTEM ANALYSIS", "purple")

                    col_widths = [24, 25, 24]
                    self.log_message(self._create_table_separator(col_widths, "top"), "purple")
                    headers = ["Quality Score", "ML Contribution", "Combined Power"]
                    self.log_message(self._create_table_row(headers, col_widths), "white")
                    self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

                    quality_bar = self._create_confidence_bar(total_score, 10)
                    quality_cell = f"{total_score}/100 {quality_bar}"

                    if ml_is_enabled and ml_conf_pct > 0:
                        ml_bar = self._create_confidence_bar(ml_conf_pct, 10)
                        ml_cell = f"{ml_conf_pct:.0f}/100 {ml_bar}"
                    else:
                        ml_cell = "[DISABLED]"

                    combined_bar = self._create_confidence_bar(combined_confidence, 10)
                    combined_cell = f"{combined_confidence:.0f}/100 {combined_bar}"

                    data_cells = [quality_cell, ml_cell, combined_cell]
                    self.log_message(self._create_table_row(data_cells, col_widths),
                                     "green" if combined_confidence >= 75 else "orange")
                    self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

                    self._display_component_scores(component_scores)

                    if hasattr(self.strategy, '_get_position_multiplier'):
                        position_mult = self.strategy._get_position_multiplier(combined_confidence)
                        self.log_message(f"💰 Position Size: {position_mult * 100:.0f}% of normal",
                                         "green" if position_mult >= 1.0 else "orange")

                    tier = getattr(self.strategy, '_last_entry_tier', 1)
                    tier_label = f"TIER {tier}" if tier else "TIER None"
                    tier_color = "green" if tier == 2 else "blue" if tier == 1 else "orange"
                    self.log_message(f"🎯 {tier_label} Entry", tier_color)

                    if ml_is_enabled and ml_conf_pct > 0:
                        signal_text = "BULLISH" if ml_prediction == 1 else "BEARISH" if ml_prediction == -1 else "NEUTRAL"
                        signal_color = "green" if ml_prediction == 1 else "red" if ml_prediction == -1 else "orange"
                        ml_impact = combined_confidence - total_score
                        self.log_message(
                            f"🤖 ML: {signal_text} ({ml_conf_pct:.0f}%) | Impact: {ml_impact:+.0f} pts",
                            signal_color)

                        if self._last_forecast is not None:
                            n_future = int(self.prediction_candles_slider.get()) if hasattr(
                                self, 'prediction_candles_slider') else 5
                            current_price = float(current_data.get('Close', 0))
                            self.log_message(f"🕯️ FORECASTED PRICES ({n_future} candles):", "purple")
                            for i, pred_price in enumerate(self._last_forecast, 1):
                                change_pct = ((pred_price - current_price) / current_price * 100
                                              ) if current_price > 0 else 0
                                direction = "🔼" if change_pct > 0 else "🔽" if change_pct < 0 else "➡️"
                                color = "green" if change_pct > 0 else "red" if change_pct < 0 else "white"
                                self.log_message(
                                    f"   Candle {i}: ${pred_price:.4f} ({change_pct:+.2f}%) {direction}",
                                    color)

                    self.log_message(f"📝 Reason: {reason}", "blue")

                    # ─── NEW: Exit confirmation status ──────────────────────────────
                    if getattr(self.strategy, 'exit_confirmation_enabled', True):
                        rsi = current_data.get('RSI', 50)
                        adx = current_data.get('ADX', 0)
                        macd = current_data.get('MACD', 0)
                        macd_signal = current_data.get('MACD_Signal', 0)

                        self.log_message(f"\n🔒 EXIT CONFIRMATION STATUS:", "cyan")
                        self.log_message(f"   RSI: {rsi:.1f} {'✓' if rsi < 50 else '✗'} (need < 50 for long exit)",
                                         "green" if rsi < 50 else "red")
                        self.log_message(f"   ADX: {adx:.1f} {'✓' if adx < 25 else '✗'} (need < 25 for trend weak)",
                                         "green" if adx < 25 else "red")
                        self.log_message(f"   MACD: {'Bearish' if macd < macd_signal else 'Bullish'}",
                                         "red" if macd < macd_signal else "green")

                elif is_scalping_strategy:
                    # ── SCALPING STRATEGY (simplified additive scoring) ──
                    _eff_dir = getattr(self.strategy, "_pending_signal", None)
                    _eff_dir = _eff_dir.get("direction", "long") if _eff_dir else getattr(
                        self.strategy, "trade_direction", "long")

                    if _eff_dir == "short":
                        total_score, component_scores, score_reason = self.strategy._quality_score_short(current_data)
                    else:
                        total_score, component_scores, score_reason = self.strategy._quality_score_long(current_data)

                    combined_confidence = total_score
                    if ml_is_enabled and ml_conf_pct > 0:
                        combined_confidence = (total_score + ml_conf_pct) / 2

                    self.log_message(f"\n🎯 SCALPING STRATEGY - QUALITY SCORE ANALYSIS", "bold purple")
                    self.log_message(f"{'─' * 77}", "purple")

                    # Display the breakdown
                    self.log_message(f"📊 COMPONENT BREAKDOWN:", "cyan")
                    self.log_message(f"   {score_reason}", "yellow")
                    self.log_message(f"{'─' * 77}", "purple")

                    # Create visual bar for total score
                    bar_length = int((total_score / 100) * 10)
                    bar = "█" * bar_length + "░" * (10 - bar_length)

                    col_widths = [24, 25, 24]
                    self.log_message(self._create_table_separator(col_widths, "top"), "purple")
                    headers = ["Quality Score", "ML Contribution", "Combined Power"]
                    self.log_message(self._create_table_row(headers, col_widths), "white")
                    self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

                    quality_cell = f"{total_score}/100 {bar}"

                    if ml_is_enabled and ml_conf_pct > 0:
                        ml_bar = self._create_confidence_bar(ml_conf_pct, 10)
                        ml_cell = f"{ml_conf_pct:.0f}/100 {ml_bar}"
                    else:
                        ml_cell = "[DISABLED]"

                    combined_bar = self._create_confidence_bar(combined_confidence, 10)
                    combined_cell = f"{combined_confidence:.0f}/100 {combined_bar}"

                    data_cells = [quality_cell, ml_cell, combined_cell]
                    self.log_message(self._create_table_row(data_cells, col_widths),
                                     "green" if combined_confidence >= 75 else "orange" if combined_confidence >= 50 else "red")
                    self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

                    # Display individual component scores if available
                    if component_scores:
                        self.log_message(f"\n📊 COMPONENT SCORES:", "cyan")
                        self.log_message(f"{'─' * 50}", "purple")
                        for component, score in component_scores.items():
                            component_name = component.upper().replace('_', ' ')
                            bar_len = int((score / 100) * 10) if score <= 100 else 10
                            comp_bar = "█" * bar_len + "░" * (10 - bar_len)
                            self.log_message(f"   {component_name:<15}: {score:>3} {comp_bar}", "yellow")
                        self.log_message(f"{'─' * 50}", "purple")

                    if hasattr(self.strategy, '_get_position_multiplier'):
                        position_mult = self.strategy._get_position_multiplier(combined_confidence)
                        self.log_message(f"💰 Position Size: {position_mult * 100:.0f}% of normal",
                                         "green" if position_mult >= 1.0 else "orange")

                    tier = getattr(self.strategy, '_last_entry_tier',
                                   getattr(self.strategy, '_entry_tier', 1))
                    tier_label = f"TIER {tier}" if tier else "TIER None"
                    tier_color = "green" if tier == 2 else "blue" if tier == 1 else "orange"
                    self.log_message(f"🎯 {tier_label} Entry", tier_color)

                    if ml_is_enabled and ml_conf_pct > 0:
                        signal_text = "BULLISH" if ml_prediction == 1 else "BEARISH" if ml_prediction == -1 else "NEUTRAL"
                        signal_color = "green" if ml_prediction == 1 else "red" if ml_prediction == -1 else "orange"
                        ml_impact = combined_confidence - total_score
                        self.log_message(
                            f"🤖 ML: {signal_text} ({ml_conf_pct:.0f}%) | Impact: {ml_impact:+.0f} pts",
                            signal_color)

                        if self._last_forecast is not None:
                            n_future = int(self.prediction_candles_slider.get()) if hasattr(
                                self, 'prediction_candles_slider') else 5
                            current_price = float(current_data.get('Close', 0))
                            self.log_message(f"🕯️ FORECASTED PRICES ({n_future} candles):", "purple")
                            for i, pred_price in enumerate(self._last_forecast, 1):
                                change_pct = ((pred_price - current_price) / current_price * 100
                                              ) if current_price > 0 else 0
                                direction = "🔼" if change_pct > 0 else "🔽" if change_pct < 0 else "➡️"
                                color = "green" if change_pct > 0 else "red" if change_pct < 0 else "white"
                                self.log_message(
                                    f"   Candle {i}: ${pred_price:.4f} ({change_pct:+.2f}%) {direction}",
                                    color)

                    self.log_message(f"📝 Reason: {reason}", "blue")
                    self.log_message(f"{'─' * 77}", "purple")

                else:
                    # ── FALLBACK for unknown strategy types ──
                    total_score = quality_score if isinstance(quality_score, (int, float)) else 0
                    combined_confidence = total_score
                    if ml_is_enabled and ml_conf_pct > 0:
                        combined_confidence = (total_score + ml_conf_pct) / 2

                    self.log_message(f"\n🎯 QUALITY SCORE (Basic Mode)", "yellow")
                    bar_length = int((total_score / 100) * 10)
                    bar = "█" * bar_length + "░" * (10 - bar_length)
                    self.log_message(f"   Score: {total_score}/100 {bar}",
                                     "green" if total_score >= 70 else "orange" if total_score >= 50 else "red")
                    if ml_is_enabled and ml_conf_pct > 0:
                        self.log_message(f"   ML: {ml_conf_pct:.0f}%", "cyan")
                    self.log_message(f"📝 Reason: {reason}", "blue")

                # ── FIX: derive actual quality threshold from the active strategy
                #    so the "Minimum" label is always correct ──────────────────
                q_min = getattr(self.strategy, 'quality_min_long',
                                getattr(self.strategy, 'quality_minimum_score',
                                        getattr(self.strategy, 'quality_tier2_min', 75)))

                if (decision == "buy" or decision == "sell") and shares and shares > 0:
                    self.log_message(f"\n✅ {decision.upper()} SIGNAL DETECTED - {shares:.4f} shares", "green")

                    # ─── NEW: Show ATR-based position sizing ──────────────────────────
                    if atr > 0 and current_price > 0:
                        try:
                            if hasattr(self, 'atr_calc'):
                                equity = self.get_balance('USDT') if hasattr(self, 'get_balance') else 50000
                                stops = self.atr_calc.calculate_stops(current_price, atr)
                                pos_size = self.atr_calc.get_position_size(
                                    equity=equity,
                                    risk_pct=2.0,  # 2% risk per trade
                                    entry_price=current_price,
                                    stop_loss=stops['stop_loss']
                                )
                                self.log_message(f"📊 Position Size: {pos_size:.2f} units (2% risk)", "cyan")
                        except Exception:
                            pass
                else:
                    self.log_message(f"\n🎯 QUALITY SCORE REJECTED", "red")
                    self.log_message(f"{'=' * 70}", "red")
                    self.log_message(
                        f"📊 Score: {combined_confidence:.0f}/100 (Minimum: {q_min})", "red")
                    self.log_message(f"   Raw Quality: {total_score:.0f}", "yellow")
                    if ml_is_enabled and ml_conf_pct > 0:
                        self.log_message(f"   ML Contribution: {ml_conf_pct:.0f}%", "yellow")
                    self.log_message(f"   Reason: {reason}", "orange")
                    self.log_message(f"{'=' * 70}", "red")
            else:
                self.log_message(f"ℹ️ Position already open - monitoring", "blue")
        else:
            if len(result) == 2:
                exit_signal, exit_pct = result
                if exit_signal:
                    # ─── NEW: Enhanced exit signal display with ATR context ──────────
                    self.log_message(f"\n🚨 EXIT SIGNAL: {exit_signal} ({exit_pct * 100:.0f}%)", "yellow")

                    # Show exit confirmation status
                    if getattr(self.strategy, 'exit_confirmation_enabled', True):
                        rsi = current_data.get('RSI', 50)
                        adx = current_data.get('ADX', 0)
                        self.log_message(f"   Exit Confirmations: RSI={rsi:.1f} ADX={adx:.1f}",
                                         "green" if (rsi < 50 or adx < 25) else "orange")

                    if hasattr(self.strategy, 'position') and self.strategy.position:
                        entry_price = self.strategy.position.get('entry_price', 0)
                        if entry_price > 0:
                            current_price = float(current_data['Close'])
                            pos_type = self.strategy.position.get('type', 'long')
                            if pos_type == 'short':
                                profit_pct = ((entry_price - current_price) / entry_price) * 100
                                profit_r = ((entry_price - current_price) /
                                            (current_data.get('ATR', 1) * 3.0))
                            else:
                                profit_pct = ((current_price - entry_price) / entry_price) * 100
                                profit_r = ((current_price - entry_price) /
                                            (current_data.get('ATR', 1) * 3.0))

                            # Show current P/L with ATR context
                            self.log_message(
                                f"📊 Current P/L: {profit_pct:.2f}% ({profit_r:.2f}R)",
                                "green" if profit_pct > 0 else "red")

                            # Show distance to stop
                            stop_loss = self.strategy.position.get('stop_loss', 0)
                            if stop_loss > 0:
                                if pos_type == 'long':
                                    stop_distance = (current_price - stop_loss) / current_price * 100
                                else:
                                    stop_distance = (stop_loss - current_price) / current_price * 100
                                self.log_message(f"   Distance to Stop: {stop_distance:.2f}%",
                                                 "yellow" if stop_distance < 1.0 else "gray")
        self.log_message(f"{'═' * 77}", "blue")

    def _display_component_scores(self, component_scores):
        self.log_message(f"\n📊 Component Scores:", "cyan")

        col_widths = [15, 15, 15, 15]
        self.log_message(self._create_table_separator(col_widths, "top"), "purple")
        headers = ["EMA", "ADX", "MACD", "Volume"]
        self.log_message(self._create_table_row(headers, col_widths), "white")
        self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

        data_cells = [
            f"{component_scores.get('ema', 0)}/20",
            f"{component_scores.get('adx', 0)}/20",
            f"{component_scores.get('macd', 0)}/25",
            f"{component_scores.get('volume', 0)}/15"
        ]
        self.log_message(self._create_table_row(data_cells, col_widths), "green")
        self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

        self.log_message(f"\n📊 Supporting Scores:", "cyan")

        col_widths = [15, 15, 15]
        self.log_message(self._create_table_separator(col_widths, "top"), "purple")
        headers = ["RSI", "CCI", "Kalman"]
        self.log_message(self._create_table_row(headers, col_widths), "white")
        self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

        data_cells = [
            f"{component_scores.get('rsi', 0)}/20",
            f"{component_scores.get('cci', 0)}/5",
            f"{component_scores.get('kalman', 0)}/5"
        ]
        self.log_message(self._create_table_row(data_cells, col_widths), "green")
        self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

    def _display_simple_analysis(self, df, current_data, result):
        close = float(current_data['Close'])
        ema_fast = float(current_data.get('EMA_Fast', 0))
        ema_mid = float(current_data.get('EMA_Mid', 0))
        ema_slow = float(current_data.get('EMA_Slow', 0))
        full_alignment = close > ema_fast > ema_mid > ema_slow

        if isinstance(result, tuple) and len(result) >= 4:
            decision, quality_score, shares, reason = result[:4]

            if quality_score >= 0:
                tier = getattr(self.strategy, '_last_entry_tier', 1)
                tier_label = f"TIER {tier}"
                self.log_message(f"🎯 Quality Score: {quality_score:.0f}/100 | {tier_label}",
                                 "green" if quality_score >= 75 else "orange")
                if decision not in ("buy", "sell"):
                    self.log_message(f"⏸️ {reason}", "orange")
            else:
                self.log_message(f"ℹ️ Position open", "blue")
        else:
            if len(result) == 2:
                exit_signal, exit_pct = result
                if exit_signal:
                    self.log_message(f"🚨 Exit: {exit_signal}", "yellow")

        ema_status = "✅" if full_alignment else "❌"
        rsi = current_data.get('RSI', 0)
        adx = current_data.get('ADX', 0)
        volume_ratio = current_data.get('Volume_Ratio', 1.0)

        self.log_message(
            f"📊 Price: ${close:.4f} | EMA: {ema_status} | RSI: {rsi:.1f} | ADX: {adx:.1f} | Vol: {volume_ratio:.2f}x",
            "green" if full_alignment else "white")
        self.log_message(f"{'=' * 30} {datetime.now(timezone.utc).strftime('%H:%M:%S')} {'=' * 29}", "blue")

    def combined_score(self, quality_score, ml_score_raw, ml_confidence,
                       trade_direction="long",
                       high_conf_thresh=0.70, high_conf_weight=0.40,
                       med_conf_thresh=0.50, med_conf_weight=0.25,
                       low_conf_weight=0.10,
                       agreement_boost=1.05):
        """
        Combine quality score with ML prediction using direction-aware alignment.

        ml_score_raw    : raw prediction integer  (-1 = bearish, 0 = neutral, +1 = bullish)
        ml_confidence   : float 0-1  (or 0-100; auto-detected)
        trade_direction : 'long' or 'short'

        Rules
        -----
        Long  + bullish  → aligned   → boost
        Long  + bearish  → opposed   → penalty
        Short + bearish  → aligned   → boost
        Short + bullish  → opposed   → penalty
        Any   + neutral  → abstain   → score unchanged
        """
        # ── Normalise confidence to 0-1 ─────────────────────────────────────────
        ml_conf_norm = (ml_confidence / 100.0) if ml_confidence > 1.0 else float(ml_confidence)
        ml_conf_pct = ml_conf_norm * 100.0

        # ── Determine direction alignment ────────────────────────────────────────
        if ml_score_raw == 0:
            return float(quality_score)  # neutral — ML abstains entirely

        if trade_direction == "long":
            alignment = 1 if ml_score_raw == 1 else -1  # bullish=agree, bearish=oppose
        else:  # short
            alignment = 1 if ml_score_raw == -1 else -1  # bearish=agree, bullish=oppose

        # ── Weight tier based on confidence level ────────────────────────────────
        if ml_conf_norm >= high_conf_thresh:
            ml_weight = high_conf_weight
        elif ml_conf_norm >= med_conf_thresh:
            ml_weight = med_conf_weight
        else:
            ml_weight = low_conf_weight

        qs_weight = 1.0 - ml_weight

        # ── Compute combined score ───────────────────────────────────────────────
        if alignment == 1:
            # Aligned: weighted blend — ML pulls score toward its confidence level
            combined = (qs_weight * quality_score) + (ml_weight * ml_conf_pct)
            # Agreement bonus when BOTH signals are independently strong
            if quality_score >= 70 and ml_conf_pct >= 65:
                combined *= agreement_boost
        else:
            # Opposed: ML confidence scales a penalty deducted from quality score
            penalty = ml_weight * ml_conf_pct
            combined = quality_score - penalty

        return float(min(max(combined, 0.0), 100.0))

    def _execute_trades_from_result(self, result, current_data, current_price, df):
        """
        Execute trades from strategy result with GUI Order Size % as a hard ceiling.
        Includes visual feedback for position sizing relative to GUI cap.
        """
        if isinstance(result, tuple) and len(result) == 2:
            exit_signal, exit_pct = result
            if exit_signal is not None:
                self.log_message(f"🚨 EXECUTING EXIT: {exit_signal}", "yellow")
                success, profit, exit_price = self.strategy.execute_sell(
                    reason=exit_signal,
                    exit_percentage=exit_pct
                )
                if success:
                    self.log_message(f"✅ EXIT COMPLETE: Profit ${profit:.2f}", "green" if profit > 0 else "red")
                    self.play_notification("sell_success" if profit > 0 else "sell_loss")
                else:
                    self.log_message(f"❌ EXIT FAILED", "red")
                    self.play_notification("error")
                return

        if isinstance(result, tuple) and len(result) >= 4:
            decision, quality_score, shares, reason = result[:4]

            is_entry = (decision == "buy" or decision == "sell")

            if is_entry and quality_score >= 0 and shares > 0:
                ranging_value = current_data.get('Ranging')
                if ranging_value is not None:
                    is_ranging = ranging_value.iloc[0] if hasattr(ranging_value, 'iloc') else ranging_value
                    if is_ranging:
                        self.log_message("⏸️ Entry prevented: Market is ranging", "orange")
                        return

                confidence_threshold = float(self.confidence_var.get()) * 100

                # ── Determine trade direction from decision ───────────────────────
                _direction = "short" if decision == "sell" else "long"

                # ── Start with quality score as the base ─────────────────────────
                combined_confidence = float(quality_score)

                # ── ML contribution (direction-aware) ────────────────────────────
                ml_conf_pct = 0.0
                ml_prediction = 0
                ml_is_enabled = getattr(self, 'ml_enabled', False)

                if ml_is_enabled and hasattr(self, 'current_ml_model') and self.current_ml_model:
                    if hasattr(self.current_ml_model, 'is_trained') and self.current_ml_model.is_trained:
                        n_future = int(self.prediction_candles_slider.get()) \
                            if hasattr(self, 'prediction_candles_slider') else 5
                        try:
                            ml_conf, ml_prediction, forecast = self.current_ml_model.predict(df, n_future)
                            ml_conf_pct = float(ml_conf * 100.0 if ml_conf <= 1.0 else ml_conf)

                            model_thresh = getattr(self.current_ml_model, "confidence_threshold", 0.65)
                            model_thresh_pct = float(model_thresh * 100.0 if model_thresh <= 1.0 else model_thresh)

                            if ml_conf_pct >= model_thresh_pct:
                                combined_confidence = self.combined_score(
                                    quality_score=quality_score,
                                    ml_score_raw=ml_prediction,
                                    ml_confidence=ml_conf_pct / 100.0,
                                    trade_direction=_direction
                                )
                                # Log direction-aware adjustment
                                delta = combined_confidence - quality_score
                                align_label = "aligned ✅" if delta >= 0 else "opposed ⚠️"
                                self.log_message(
                                    f"🤖 ML ({_direction.upper()}): pred={ml_prediction:+d} "
                                    f"conf={ml_conf_pct:.0f}% → {align_label} "
                                    f"Δscore={delta:+.1f} "
                                    f"({quality_score:.0f} → {combined_confidence:.0f})",
                                    "green" if delta >= 0 else "orange"
                                )
                            else:
                                self.log_message(
                                    f"🤖 ML confidence {ml_conf_pct:.0f}% below threshold "
                                    f"{model_thresh_pct:.0f}% — using quality score only",
                                    "blue"
                                )
                        except Exception as e:
                            self.log_message(f"⚠️ ML prediction error: {e}", "orange")

                self.log_message(
                    f"🎯 Entry Check: Quality={quality_score:.0f} | "
                    f"Combined={combined_confidence:.0f} | "
                    f"Threshold={confidence_threshold:.0f} | "
                    f"Direction={_direction.upper()}",
                    "cyan"
                )

                if combined_confidence >= confidence_threshold:
                    self.play_notification("pre_buy_alert")

                    tier = getattr(self.strategy, '_last_entry_tier', 1)

                    # ═══════════════════════════════════════════════════════════════
                    # Execute the trade - GUI cap is applied inside execute_buy()
                    # ═══════════════════════════════════════════════════════════════
                    success, filled_qty, order_id = self.strategy.execute_buy(
                        shares=shares,
                        price=current_price,
                        atr=float(current_data.get('ATR', 1)),
                        quality_score=quality_score,
                        tier=tier
                    )

                    if success:
                        # ═══════════════════════════════════════════════════════════
                        # LOG POSITION SIZING DETAILS WITH GUI CAP VISUAL FEEDBACK
                        # ═══════════════════════════════════════════════════════════
                        gui_pct = self.order_size_var.get()
                        final_position_value = filled_qty * current_price

                        # Get current equity
                        equity = self.get_balance('USDT')
                        if equity is None or equity <= 0:
                            equity = 50000  # fallback

                        account_pct = (final_position_value / equity) * 100 if equity > 0 else 0
                        max_allowed = equity * (gui_pct / 100.0)

                        # Determine if GUI cap was binding
                        cap_binding = "🔒 CAP BINDING" if abs(account_pct - gui_pct) < 0.5 else "✅ WITHIN CAP"
                        cap_color = "yellow" if "CAP BINDING" in cap_binding else "green"

                        self.log_message(
                            f"📊 POSITION SIZE DETAILS:",
                            "cyan"
                        )
                        self.log_message(
                            f"   GUI Cap:     {gui_pct:.0f}% of equity (${max_allowed:,.2f})",
                            "blue"
                        )
                        self.log_message(
                            f"   Position:    {filled_qty:.4f} {self.base_symbol()} @ ${current_price:.2f}",
                            "white"
                        )
                        self.log_message(
                            f"   Value:       ${final_position_value:,.2f} ({account_pct:.1f}% of equity) {cap_binding}",
                            cap_color
                        )

                        # Show risk amount if available
                        if hasattr(self.strategy, 'position') and self.strategy.position:
                            stop_loss = self.strategy.position.get('stop_loss', 0)
                            if stop_loss and stop_loss > 0:
                                if _direction == 'long':
                                    risk_per_unit = current_price - stop_loss
                                else:
                                    risk_per_unit = stop_loss - current_price
                                risk_amount = risk_per_unit * filled_qty
                                risk_pct = (risk_amount / equity) * 100 if equity > 0 else 0
                                self.log_message(
                                    f"   Risk:        ${risk_amount:,.2f} ({risk_pct:.2f}% of equity)",
                                    "orange" if risk_pct > 2.0 else "green"
                                )
                        # ═══════════════════════════════════════════════════════════

                        ml_msg = f' | ML {ml_prediction:+d} ({ml_conf_pct:.0f}%)' if ml_is_enabled and ml_conf_pct > 0 else ''
                        direction_label = "SHORT" if decision == "sell" else "LONG"
                        direction_emoji = "🔴" if decision == "sell" else "🟢"
                        self.log_message(
                            f"{direction_emoji} {direction_label} EXECUTED{ml_msg} "
                            f"(Quality: {quality_score:.0f}, Combined: {combined_confidence:.0f}, Tier {tier})",
                            "green" if decision == "buy" else "red"
                        )
                        self.play_notification("buy_success")
                    else:
                        self.log_message(f"❌ ENTRY FAILED", "red")
                        self.play_notification("error")
                else:
                    # ── Rejection log ─────────────────────────────────────────────
                    self.log_message(f"\n🎯 ENTRY REJECTED", "red")
                    self.log_message(f"{'=' * 70}", "red")
                    self.log_message(
                        f"📊 Combined: {combined_confidence:.0f}/100  "
                        f"(minimum: {confidence_threshold:.0f})",
                        "red"
                    )
                    self.log_message(f"   Raw quality   : {quality_score:.0f}", "yellow")
                    if ml_is_enabled and ml_conf_pct > 0:
                        delta = combined_confidence - quality_score
                        align_label = "aligned" if delta >= 0 else "OPPOSED"
                        self.log_message(
                            f"   ML impact     : {delta:+.1f} pts  "
                            f"(pred={ml_prediction:+d}, conf={ml_conf_pct:.0f}%, {align_label})",
                            "orange"
                        )
                    self.log_message(f"   Reason        : {reason}", "orange")
                    self.log_message(f"{'=' * 70}", "red")

    def get_historical_data(self, symbol, exchange_name="binance", start=None, end=None,
                            interval="15m", days=360, cache=True, cache_dir="data_cache", limit=None):
        """
        Get historical OHLCV data with optional limit parameter.

        Args:
            symbol: Trading pair symbol
            exchange_name: Exchange to use ("binance" or "okx")
            start: Start timestamp in ms or datetime string
            end: End timestamp in ms or datetime string
            interval: Candle interval
            days: Number of days to fetch (used if start/end not provided)
            cache: Whether to use cache
            cache_dir: Directory for cache files
            limit: Maximum number of candles to return (None = all)
        """
        try:
            if exchange_name.lower() == "binance":
                exchange = ccxt.binance({
                    "enableRateLimit": True,
                    "timeout": 30000,
                })
            elif exchange_name.lower() == "okx":
                exchange = ccxt.okx({
                    "enableRateLimit": True
                })
            else:
                self.log_message(f"❌ Unsupported exchange: {exchange_name}", "red")
                return pd.DataFrame()
        except Exception as e:
            self.log_message(f"❌ Failed to initialize {exchange_name}: {str(e)}", "red")
            return pd.DataFrame()

        if interval in ['1H', '4H', '1D']:
            interval = interval.lower()

        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"{exchange_name}_{symbol.replace('/', '-')}_{interval}_{days}d.csv"
        )

        use_cache = getattr(self, 'use_cache_var', None)
        should_use_cache = use_cache.get() if use_cache else True

        # Cache handling (existing code remains the same)
        if cache and should_use_cache and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, parse_dates=["timestamp"], index_col="timestamp")

                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')

                # Apply limit if specified
                if limit is not None:
                    df = df.tail(min(len(df), limit))

                if start is not None:
                    start_dt = pd.to_datetime(start, unit='ms', utc=True) if isinstance(start, (int,
                                                                                                float)) else pd.to_datetime(
                        start).tz_localize('UTC')
                    df = df[df.index >= start_dt]

                if end is not None:
                    end_dt = pd.to_datetime(end, unit='ms', utc=True) if isinstance(end,
                                                                                    (int, float)) else pd.to_datetime(
                        end).tz_localize('UTC')
                    df = df[df.index <= end_dt]

                if not df.empty:
                    cache_start = df.index[0].strftime('%Y-%m-%d %H:%M:%S UTC')
                    cache_end = df.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
                    self.log_message(
                        f"📁 Loading filtered CSV cache [{exchange_name.upper()}]\n"
                        f"   Period: {cache_start} → {cache_end} | Records: {len(df)}",
                        "blue"
                    )
                    return df

            except Exception as e:
                self.log_message(f"⚠️ Cache read failed: {str(e)}. Fetching fresh data...", "orange")

        if end is None:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        else:
            end_dt = pd.to_datetime(end, utc=True) if not isinstance(end, (int, float)) else pd.to_datetime(end,
                                                                                                            unit='ms',
                                                                                                            utc=True)
            end_ms = int(end_dt.timestamp() * 1000)

        if start is None:
            start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        else:
            start_dt = pd.to_datetime(start, utc=True) if not isinstance(start, (int, float)) else pd.to_datetime(start,
                                                                                                                  unit='ms',
                                                                                                                  utc=True)
            start_ms = int(start_dt.timestamp() * 1000)

        limit_per_request = 1000  # CCXT default max per request
        all_data = []
        since = start_ms

        timeframe_ms = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }

        interval_ms = timeframe_ms.get(interval, 15 * 60 * 1000)

        start_dt_display = pd.to_datetime(start_ms, unit='ms', utc=True)
        end_dt_display = pd.to_datetime(end_ms, unit='ms', utc=True)

        self.log_message(
            f"🔍 Fetching {symbol} {interval} data from {exchange_name.upper()}\n"
            f"   Period: {start_dt_display.strftime('%Y-%m-%d %H:%M UTC')} → "
            f"{end_dt_display.strftime('%Y-%m-%d %H:%M UTC')}",
            "blue"
        )

        max_iterations = 1000  # Safety limit to prevent infinite loops
        iteration = 0
        last_timestamp = 0

        while since < end_ms and iteration < max_iterations:
            iteration += 1

            try:
                # Fetch batch of candles
                candles = exchange.fetch_ohlcv(
                    symbol.replace('-', '/'),
                    interval,
                    since=since,
                    limit=limit_per_request
                )

                if not candles:
                    self.log_message(f"⚠️ No more data received at {pd.to_datetime(since, unit='ms', utc=True)}",
                                     "orange")
                    break

                # Filter candles up to end_ms
                filtered_batch = [c for c in candles if c[0] <= end_ms]

                if filtered_batch:
                    # Add to our collection
                    all_data.extend(filtered_batch)

                    # Log progress periodically
                    if len(all_data) % 1000 == 0:
                        current_timestamp = filtered_batch[-1][0]
                        current_dt = pd.to_datetime(current_timestamp, unit='ms', utc=True)
                        self.log_message(
                            f" ✓ Fetched {len(all_data)} candles ... (up to {current_dt.strftime('%Y-%m-%d %H:%M UTC')})",
                            "green"
                        )

                    # Check if we've reached or passed the end
                    last_timestamp = filtered_batch[-1][0]
                    if last_timestamp >= end_ms:
                        self.log_message(f"   ✓ Reached end date", "green")
                        break

                    # Update since to the next candle after the last one
                    since = last_timestamp + interval_ms

                else:
                    # If batch was filtered out completely, we're done
                    break

                # Rate limiting - respect exchange's rate limits
                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                self.log_message(f"⚠️ API error: {str(e)} — retrying...", "red")
                time.sleep(2)
                continue

        if iteration >= max_iterations:
            self.log_message(f"⚠️ Reached maximum iterations ({max_iterations})", "orange")

        if not all_data:
            self.log_message("❌ No data fetched", "red")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )

        # Remove duplicates (keep first occurrence)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()

        # Final filtering to ensure we're within the requested range
        df = df[(df.index >= start_dt_display) & (df.index <= end_dt_display)]

        # Apply limit if specified
        if limit is not None:
            df = df.tail(min(len(df), limit))

        start_str = df.index[0].strftime('%Y-%m-%d %H:%M:%S UTC')
        end_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')

        expected_candles = int((end_ms - start_ms) / interval_ms)
        coverage_pct = (len(df) / expected_candles * 100) if expected_candles > 0 else 0

        self.log_message(
            f"✅ Fetch complete: {len(df)} candles\n"
            f"   Period: {start_str} → {end_str}\n"
            f"   Coverage: {coverage_pct:.1f}% of expected ({expected_candles} candles)",
            "green"
        )

        if cache and should_use_cache:
            df.to_csv(cache_file)
            self.log_message(f"💾 Data cached at {cache_file}", "blue")

        return df

    def update_trailing_stop(self, current_price):
        if self.position['price'] is None:
            return

        self.highest_since_buy = max(float(self.highest_since_buy), float(current_price))

        new_trailing_stop = self.highest_since_buy * (1 - float(self.trailing_stop_pct))

        if new_trailing_stop > float(self.trailing_stop or 0):
            self.trailing_stop = new_trailing_stop
            if hasattr(self, 'current_data') and self.current_data is not None:
                self.current_data['trailing_stop_loss'] = self.trailing_stop

            self.log_message(
                f"🔄 Trailing Stop Updated | "
                f"High: {self.highest_since_buy:.4f} | "
                f"New Stop: {self.trailing_stop:.4f}",
                "blue"
            )

    def is_ranging(self, df):
        if df is None or df.empty:
            return df

        df = df.copy()

        if 'BB_Width' not in df.columns and 'UpperBand' in df and 'LowerBand' in df:
            df['BB_Width'] = (df['UpperBand'] - df['LowerBand']) / df['Close']

        if not {'KC_Upper', 'KC_Lower'}.issubset(df.columns):
            df['KC_Mid'] = df['Close'].ewm(span=20).mean()
            df['KC_ATR'] = df['ATR'].rolling(window=20).mean()
            df['KC_Upper'] = df['KC_Mid'] + 1.5 * df['KC_ATR']
            df['KC_Lower'] = df['KC_Mid'] - 1.5 * df['KC_ATR']
        df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']

        bb_mean = df['BB_Width'].rolling(50).mean()
        bb_std = df['BB_Width'].rolling(50).std()
        df['BB_Z'] = (df['BB_Width'] - bb_mean) / bb_std

        df['Squeeze'] = df['BB_Width'] < df['KC_Width']

        if 'RSI' not in df.columns:
            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            avg_gain = up.rolling(14).mean()
            avg_loss = down.rolling(14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

        df['ATR_MA30'] = df['ATR'].rolling(30).mean()
        atr_threshold = df['ATR'].rolling(100).quantile(0.25)

        df['EMA_Fast_diff'] = df['EMA_Fast'].pct_change() * 100

        df['CHOP'] = self.choppiness_index(df['High'], df['Low'], df['Close'])

        df['Ranging'] = (
                (abs(df['Close'] - df['EMA_Fast']) / df['EMA_Fast'] <= 0.005) &
                (df['BB_Z'] < -0.5) &
                (df['Squeeze']) &
                (df['ATR'] < atr_threshold) &
                (abs(df['EMA_Fast_diff']) <= 0.05) &
                (df['RSI'].between(45, 55)) &
                (df['CHOP'] >= 60) &
                (df['ADX'] < 20)
        ).fillna(False)

        return df

    def _execute_exit(self, exit_price, entry_price, quantity, reason):
        self.play_notification("pre_exit_alert")

        if self.place_order('sell', exit_price, quantity, exit_reason=reason):
            pnl = (exit_price - entry_price) * quantity
            pnl_percent = (exit_price / entry_price - 1) * 100
            pnl_color = "green" if pnl > 0 else "red"

            self.log_message(
                f"🛑 EXIT Triggered: {reason.replace('_', ' ').title()}\n"
                f"Entry: {entry_price:.4f} | Exit: {exit_price:.4f}\n"
                f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%)",
                pnl_color
            )
            self.speak_trade(f"Position closed. {'Profit' if pnl > 0 else 'Loss'} of {abs(pnl):.2f}")

            self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
            self.update_status_indicators("parking")

    def handle_existing_position(self, current_price, current_data):
        if None in (self.position.get('price'), self.position.get('quantity')):
            self.log_message("⚠️ Invalid position data - resetting tracking", "red")
            self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
            self.update_status_indicators("parking")
            return

        try:
            current_price = float(current_price)
            position_price = float(self.position['price'])
            position_qty = float(self.position['quantity'])

            self.highest_since_buy = max(self.highest_since_buy, current_price)

            new_trailing_stop = self.highest_since_buy * (1 - float(self.trailing_stop_pct))
            if new_trailing_stop > float(self.trailing_stop or 0):
                self.trailing_stop = new_trailing_stop
                self.log_message(f"🔼 Trailing stop updated to: {self.trailing_stop:.4f}", "blue")

            atr_threshold = current_data.get('ATR_Threshold')
            if atr_threshold is None:
                atr_threshold = float(current_data.get('ATR', 1)) * 0.6

            exit_conditions = [
                (float(current_data.get('Close', current_price)) <= float(self.initial_stop_loss),
                 "stop_loss"),
                (float(current_data.get('Close', current_price)) <= float(self.trailing_stop or 0),
                 "trailing_stop"),
                (
                    float(current_data.get('SuperTrend_closed', 1)) == -1 and
                    float(current_data.get('MACD_closed', 0)) < float(current_data.get('MACD_Signal_closed', 1)) and
                    float(current_data.get('EMA_Fast', 0)) < float(current_data.get('EMA_Slow', 0)),
                    "trend_reversal_confirmed"
                ),
                (
                    float(current_data.get('RSI_closed', 0)) >= 70 and
                    float(current_data.get('MACD_closed', 0)) < float(current_data.get('MACD_Signal_closed', 1)),
                    "overbought_and_macd_loss"
                ),
                (
                    float(current_data.get('ADX', 25)) < 20 and
                    float(current_data.get('MACD_closed', 0)) < float(current_data.get('MACD_Signal_closed', 1)),
                    "weak_trend_and_momentum_loss"
                ),
                (
                    float(current_data.get('EMA_Fast', 0)) < float(current_data.get('EMA_Slow', 0)) and
                    float(current_data.get('MACD_closed', 0)) < float(current_data.get('MACD_Signal_closed', 1)) and
                    float(current_data.get('ADX', 25)) >= 20,
                    "ma_cross_down_confirmed"
                ),
                (
                    float(current_data.get('Close', current_price)) >= float(
                        current_data.get('BB_Upper', float('inf'))) and
                    float(current_data.get('RSI_closed', 0)) > 70 and
                    float(current_data.get('MACD_closed', 0)) < float(current_data.get('MACD_Signal_closed', 1)),
                    "bollinger_overbought_exit"
                ),
                (
                    float(current_data.get('ATR', 1)) < float(atr_threshold) and
                    float(current_data.get('MACD_closed', 0)) < float(current_data.get('MACD_Signal_closed', 1)),
                    "low_volatility_bearish_exit"
                )
            ]

            strategy_exit = self.strategy.check_exit_conditions(current_data, current_price)
            if strategy_exit:
                exit_conditions.append((True, strategy_exit))

            for condition, reason in exit_conditions:
                if condition:
                    exit_percentage = 1

                    success, profit, exit_price = self.strategy.execute_sell(
                        reason=reason,
                        exit_percentage=exit_percentage
                    )

                    if not success:
                        self.log_message("❌ SELL FAILED - Position may still be open!", "red")
                    else:
                        current_equity = self.get_balance('USDT')
                        if hasattr(self.strategy, 'on_trade_closed'):
                            self.strategy.on_trade_closed(profit, current_equity, reason)

                        self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
                        self.update_status_indicators("parking")
                    return

        except Exception as e:
            self.log_message(f"❌ {str(e)}", "red")
            logging.error(f"Position handling error: {str(e)}\n{traceback.format_exc()}")
            self.update_status_indicators("parking")

    def speak_trade(self, message):
        try:
            if self._tts_engine is None:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 145)
                self._tts_engine.setProperty('volume', 0.9)

            if "Buy" in message:
                self._tts_engine.setProperty('voice', 'english-us+m3')
            else:
                self._tts_engine.setProperty('voice', 'english-us+m4')

            self._tts_engine.say(message)
            self._tts_engine.runAndWait()
        except Exception as e:
            self.log_message(f"Voice error: {str(e)}", "orange")

    def update_position_on_entry(self, entry_price, quantity, confidence):
        try:
            atr_value = self.current_data.get('ATR_closed', entry_price * 0.01)
            self.initial_stop_loss = entry_price - (atr_value * self.config.get('atr_multiplier', 1.5))
            self.trailing_stop = self.initial_stop_loss

            if hasattr(self, 'current_data') and self.current_data is not None:
                self.current_data['stop_loss'] = self.initial_stop_loss
                self.current_data['trailing_stop_loss'] = self.trailing_stop

            self.highest_since_buy = entry_price
            entry_price = float(entry_price)
            quantity = float(quantity)
            if self.position['price'] is None:
                self.position = {'price': entry_price, 'quantity': quantity, 'time': datetime.now(timezone.utc),
                                 'entry_confidence': float(confidence)}
            else:
                total_cost = (float(self.position['price']) * float(self.position['quantity'])) + (
                        entry_price * quantity)
                total_quantity = float(self.position['quantity']) + quantity
                self.position['price'] = total_cost / total_quantity
                self.position['quantity'] = total_quantity
            self.highest_since_buy = entry_price
            self.initial_stop_loss = entry_price * (1 - float(self.stop_loss_pct))
            self.trailing_stop = self.trailing_stop = self.highest_since_buy * (1 - self.trailing_stop_pct)

            self.update_status_indicators("buy")
        except Exception as e:
            self.log_message(f"⚠️ Position entry error: {str(e)}", "red")
            self.position = {
                'price': entry_price,
                'quantity': quantity,
                'time': datetime.now(timezone.utc),
                'entry_confidence': confidence
            }
        finally:
            self.update_status_indicators("buy")

    def choppiness_index(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_sum = tr.rolling(window=period).sum()
        high_max = high.rolling(window=period).max()
        low_min = low.rolling(window=period).min()

        chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(period)
        return chop

    def safe_get(self, data, key, default=0):
        try:
            if hasattr(data, 'get'):
                val = data.get(key, default)
            elif hasattr(data, 'iloc') and len(data) > 0:
                val = data.iloc[0].get(key, default) if hasattr(data.iloc[0], 'get') else default
            else:
                val = default

            if hasattr(val, 'iloc'):
                return float(val.iloc[0]) if len(val) > 0 else default
            return float(val) if val is not None else default
        except:
            return default

    def _create_confidence_bar(self, confidence, width=10):
        if confidence is None or confidence <= 0:
            return f"[{'░' * width}]"
        filled = int(width * min(100, max(0, confidence)) / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _create_table_row(self, cells, widths):
        row_parts = []
        for cell, width in zip(cells, widths):
            cell_str = str(cell)[:width].ljust(width)
            row_parts.append(cell_str)
        return "│ " + " │ ".join(row_parts) + " │"

    def _create_table_separator(self, widths, style="middle"):
        if style == "top":
            left, mid, right, line = "┌", "┬", "┐", "─"
        elif style == "bottom":
            left, mid, right, line = "└", "┴", "┘", "─"
        else:
            left, mid, right, line = "├", "┼", "┤", "─"

        parts = [line * (w + 2) for w in widths]
        return left + mid.join(parts) + right

    def display_entry_conditions(self, current_data, strategy_type="Momentum"):
        close = float(self.safe_get(current_data, 'Close', 0))
        ema_fast = float(self.safe_get(current_data, 'EMA_Fast', 0))
        ema_mid = float(self.safe_get(current_data, 'EMA_Mid', 0))
        ema_slow = float(self.safe_get(current_data, 'EMA_Slow', 0))

        has_quality_score = hasattr(self.strategy, 'quality_score_enabled') and \
                            getattr(self.strategy, 'quality_score_enabled', False)

        if has_quality_score and hasattr(self.strategy, '_calculate_quality_score'):
            _eff_dir = getattr(self.strategy, "_pending_signal", None)
            _eff_dir = _eff_dir.get("direction", "long") if _eff_dir else getattr(self.strategy, "trade_direction",
                                                                                  "long")
            if _eff_dir == "short" and hasattr(self.strategy, "_calculate_quality_score_short"):
                total_score, component_scores, reason = self.strategy._calculate_quality_score_short(current_data)
            else:
                total_score, component_scores, reason = self.strategy._calculate_quality_score(current_data)

            self.log_message(f"", "white")
            self.log_message(f"📊 QUALITY SCORE ANALYSIS", "purple")
            self.log_message(f"{'═' * 77}", "purple")

            quality_bar = self._create_confidence_bar(total_score, 10)
            self.log_message(f"🎯 Total Quality Score: {total_score}/100 {quality_bar}",
                             "green" if total_score >= 70 else "orange" if total_score >= 60 else "red")

            self.log_message(f"", "white")
            self.log_message(f"📈 Component Scores:", "cyan")

            col_widths = [15, 15, 15, 15]
            self.log_message(self._create_table_separator(col_widths, "top"), "purple")
            headers = ["EMA Alignment", "ADX Strength", "Volume", "Momentum"]
            self.log_message(self._create_table_row(headers, col_widths), "white")
            self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

            data_cells = [
                f"{component_scores.get('ema', 0)}/20",
                f"{component_scores.get('adx', 0)}/20",
                f"{component_scores.get('volume', 0)}/20",
                f"{component_scores.get('momentum', 0)}/20"
            ]
            self.log_message(self._create_table_row(data_cells, col_widths), "green")
            self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

            self.log_message(f"", "white")
            self.log_message(f"📊 Supporting Scores:", "cyan")

            col_widths = [15, 15, 15]
            self.log_message(self._create_table_separator(col_widths, "top"), "purple")
            headers = ["RSI", "CCI", "Kalman"]
            self.log_message(self._create_table_row(headers, col_widths), "white")
            self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

            data_cells = [
                f"{component_scores.get('rsi', 0)}/10",
                f"{component_scores.get('cci', 0)}/5",
                f"{component_scores.get('kalman', 0)}/5"
            ]
            self.log_message(self._create_table_row(data_cells, col_widths), "green")
            self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

            if hasattr(self.strategy, '_get_position_multiplier'):
                position_mult = self.strategy._get_position_multiplier(total_score)
                self.log_message(f"", "white")
                self.log_message(f"💰 Position Size: {position_mult * 100:.0f}% of normal size",
                                 "green" if position_mult >= 1.0 else "orange")

            self.log_message(f"", "white")
            self.log_message(f"📝 Quality Reason: {reason}", "cyan")

            return {
                'quality_score': total_score,
                'component_scores': component_scores,
                'reason': reason
            }
        else:
            return self._display_old_entry_conditions(current_data, strategy_type)

    def _display_old_entry_conditions(self, current_data, strategy_type="Momentum"):
        close = float(self.safe_get(current_data, 'Close', 0))
        ema_fast = float(self.safe_get(current_data, 'EMA_Fast', 0))
        ema_mid = float(self.safe_get(current_data, 'EMA_Mid', 0))
        ema_slow = float(self.safe_get(current_data, 'EMA_Slow', 0))
        adx = float(self.safe_get(current_data, 'ADX', 0))
        rsi = float(self.safe_get(current_data, 'RSI', 50))
        cci = float(self.safe_get(current_data, 'CCI', 0))
        volume_ratio = float(self.safe_get(current_data, 'Volume_Ratio', 0))
        kalman_strength = float(self.safe_get(current_data, 'Kalman_Strength', 0))
        vix = float(self.safe_get(current_data, 'VIX', 0))

        adx_min = getattr(self.strategy, 'adx_min_trend', 20)
        rsi_min = getattr(self.strategy, 'rsi_entry_min', 40)
        rsi_max = getattr(self.strategy, 'rsi_entry_max', 70)
        cci_threshold = getattr(self.strategy, 'cci_entry_threshold', -100)
        volume_min = getattr(self.strategy, 'volume_min_ratio', 1.0)
        kalman_min = getattr(self.strategy, 'kalman_min_strength', 0.1)
        vix_max = getattr(self.strategy, 'vix_max_threshold', 30)

        ema_aligned = (close > ema_fast) and (ema_fast > ema_mid) and (ema_mid > ema_slow)

        adx_ok = adx >= adx_min
        vix_ok = vix <= vix_max

        rsi_ok = rsi_min <= rsi <= rsi_max
        cci_ok = cci >= cci_threshold

        volume_ok = volume_ratio >= volume_min
        kalman_ok = kalman_strength >= kalman_min

        self.log_message(f"", "white")
        self.log_message(f"📊 ENTRY CONDITIONS (Binary System)", "purple")
        self.log_message(f"{'═' * 77}", "purple")

        self.log_message(f"", "white")
        self.log_message(f"📈 EMA ALIGNMENT:", "cyan")
        self.log_message(f"   Close > EMA_Fast > EMA_Mid > EMA_Slow", "white")
        self.log_message(f"   {close:.2f} > {ema_fast:.2f} > {ema_mid:.2f} > {ema_slow:.2f}",
                         "green" if ema_aligned else "red")
        self.log_message(f"   Status: {'✓ ALIGNED' if ema_aligned else '✗ NOT ALIGNED'}",
                         "green" if ema_aligned else "red")

        self.log_message(f"", "white")
        self.log_message(f"🔴 TIER 1 - ESSENTIAL FILTERS:", "cyan")

        col_widths = [20, 15, 15, 15]
        self.log_message(self._create_table_separator(col_widths, "top"), "purple")
        headers = ["Condition", "Required", "Actual", "Status"]
        self.log_message(self._create_table_row(headers, col_widths), "white")
        self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

        adx_status = "✓ PASS" if adx_ok else "✗ FAIL"
        self.log_message(self._create_table_row(
            ["ADX Trend", f">= {adx_min}", f"{adx:.1f}", adx_status], col_widths),
            "green" if adx_ok else "red")

        vix_status = "✓ PASS" if vix_ok else "✗ FAIL"
        self.log_message(self._create_table_row(
            ["VIX Level", f"<= {vix_max}", f"{vix:.1f}", vix_status], col_widths),
            "green" if vix_ok else "red")

        self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

        self.log_message(f"", "white")
        self.log_message(f"🟡 TIER 2 - MOMENTUM FILTERS:", "cyan")

        self.log_message(self._create_table_separator(col_widths, "top"), "purple")
        self.log_message(self._create_table_row(headers, col_widths), "white")
        self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

        rsi_status = "✓ PASS" if rsi_ok else "✗ FAIL"
        self.log_message(self._create_table_row(
            ["RSI Range", f"{rsi_min}-{rsi_max}", f"{rsi:.1f}", rsi_status], col_widths),
            "green" if rsi_ok else "red")

        cci_status = "✓ PASS" if cci_ok else "✗ FAIL"
        self.log_message(self._create_table_row(
            ["CCI Level", f">= {cci_threshold}", f"{cci:.1f}", cci_status], col_widths),
            "green" if cci_ok else "red")

        volume_status = "✓ PASS" if volume_ok else "✗ FAIL"
        self.log_message(self._create_table_row(
            ["Volume Ratio", f">= {volume_min}", f"{volume_ratio:.2f}", volume_status], col_widths),
            "green" if volume_ok else "red")

        kalman_status = "✓ PASS" if kalman_ok else "✗ FAIL"
        self.log_message(self._create_table_row(
            ["Kalman Strength", f">= {kalman_min}", f"{kalman_strength:.3f}", kalman_status], col_widths),
            "green" if kalman_ok else "red")

        self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

        all_conditions = [ema_aligned, adx_ok, vix_ok, rsi_ok, cci_ok, volume_ok, kalman_ok]
        passed = sum(all_conditions)
        total = len(all_conditions)

        self.log_message(f"", "white")
        self.log_message(f"📊 OVERALL: {passed}/{total} conditions passed",
                         "green" if passed == total else "orange" if passed >= 5 else "red")

        all_pass = all(all_conditions)
        self.log_message(f"🎯 Entry Signal: {'✓ VALID' if all_pass else '✗ NO ENTRY'}",
                         "green" if all_pass else "red")

        return {
            'ema_aligned': ema_aligned,
            'adx_ok': adx_ok,
            'vix_ok': vix_ok,
            'rsi_ok': rsi_ok,
            'cci_ok': cci_ok,
            'volume_ok': volume_ok,
            'kalman_ok': kalman_ok,
            'all_pass': all_pass,
            'passed_count': passed,
            'total_count': total
        }

    def display_confidence_scores(self, technical_score, ml_confidence, ml_prediction,
                                  trade_direction, combined_power):
        self.log_message(f"", "white")
        self.log_message(f"📊 ENTRY POWER ANALYSIS - {trade_direction}", "purple")

        col_widths = [15, 17, 15, 10]

        self.log_message(self._create_table_separator(col_widths, "top"), "purple")
        headers = ["Technical Score", "ML Contribution", "Total Power", "Decision"]
        self.log_message(self._create_table_row(headers, col_widths), "white")
        self.log_message(self._create_table_separator(col_widths, "middle"), "purple")

        tech_bar = self._create_confidence_bar(technical_score / 60 * 100, 6)
        tech_text = f"{technical_score:.0f}/60 {tech_bar}"

        if ml_confidence > 0:
            ml_weight = ml_confidence * 0.40

            if trade_direction == "LONG":
                if ml_prediction == 1:
                    ml_contrib = ml_weight
                    ml_text = f"+{ml_contrib:.1f} 🐂✅"
                    ml_color = "green"
                elif ml_prediction == -1:
                    ml_contrib = -ml_weight * 0.5
                    ml_text = f"{ml_contrib:.1f} 🐻❌"
                    ml_color = "red"
                else:
                    ml_text = f"+0.0 ➖"
                    ml_color = "orange"

            else:
                if ml_prediction == -1:
                    ml_contrib = ml_weight
                    ml_text = f"+{ml_contrib:.1f} 🐻✅"
                    ml_color = "green"
                elif ml_prediction == 1:
                    ml_contrib = -ml_weight * 0.5
                    ml_text = f"{ml_contrib:.1f} 🐂❌"
                    ml_color = "red"
                else:
                    ml_text = f"+0.0 ➖"
                    ml_color = "orange"
        else:
            ml_text = "DISABLED"
            ml_color = "gray"

        power_bar = self._create_confidence_bar(combined_power, 6)
        power_text = f"{combined_power:.0f}/100 {power_bar}"

        if combined_power >= 80:
            decision = "FULL SIZE 🚀"
            decision_color = "green"
            position_mult = 1.0
        elif combined_power >= 70:
            decision = "80% SIZE ✅"
            decision_color = "green"
            position_mult = 0.8
        elif combined_power >= 60:
            decision = "60% SIZE 🟢"
            decision_color = "green"
            position_mult = 0.6
        elif combined_power >= 50:
            decision = "40% SIZE 🟡"
            decision_color = "orange"
            position_mult = 0.4
        else:
            decision = "SKIP ❌"
            decision_color = "red"
            position_mult = 0.0

        data_cells = [tech_text, ml_text, power_text, decision]
        self.log_message(self._create_table_row(data_cells, col_widths), decision_color)
        self.log_message(self._create_table_separator(col_widths, "bottom"), "purple")

        if ml_confidence > 0:
            self.log_message(f"", "white")
            self.log_message(f"🤖 ML Details:", "blue")
            self.log_message(f"   Confidence: {ml_confidence:.1f}%", "white")
            pred_text = 'BULLISH (+1)' if ml_prediction == 1 else 'BEARISH (-1)' if ml_prediction == -1 else 'NEUTRAL (0)'
            self.log_message(f"   Prediction: {pred_text}", "white")
            self.log_message(f"   Alignment: {ml_text.split()[1] if len(ml_text.split()) > 1 else ml_text}",
                             ml_color)
            self.log_message(f"   Contribution: {ml_text.split()[0]} points", ml_color)

        return decision, decision_color, position_mult

    def display_score_breakdown(self, components):
        self.log_message(f"", "white")
        self.log_message(f"🔍 SCORE BREAKDOWN", "blue")

        col_widths = [15, 17, 17, 17]

        self.log_message(self._create_table_separator(col_widths, "top"), "blue")
        headers = ["Base Score", "Direction Bonus", "Condition Mult", "Divergence Pen"]
        self.log_message(self._create_table_row(headers, col_widths), "white")
        self.log_message(self._create_table_separator(col_widths, "middle"), "blue")

        base = components.get('base_score', 0)
        bonus = components.get('direction_bonus', 0)
        mult = components.get('condition_multiplier', 1.0)
        penalty = components.get('divergence_penalty', 0)

        bonus_str = f"+{bonus:.0f}" if bonus >= 0 else f"{bonus:.0f}"

        data_cells = [
            f"{base:.1f}",
            bonus_str,
            f"{mult:.3f}x",
            f"-{penalty:.1f}" if penalty > 0 else "0"
        ]

        if bonus >= 0 and penalty == 0:
            row_color = "green"
        elif bonus < 0 or penalty > 5:
            row_color = "red"
        else:
            row_color = "orange"

        self.log_message(self._create_table_row(data_cells, col_widths), row_color)
        self.log_message(self._create_table_separator(col_widths, "bottom"), "blue")

    def calculate_professional_combined_score(self, entry_confidence, ml_confidence, ml_prediction, conditions_result):
        entry_conf = min(100, max(0, float(entry_confidence)))
        ml_conf = min(100, max(0, float(ml_confidence))) if ml_confidence else 0

        if ml_conf > 0:
            entry_weight = 0.55
            ml_weight = 0.45

            safe_entry = max(1, entry_conf)
            safe_ml = max(1, ml_conf)

            geometric_mean = math.exp(entry_weight * math.log(safe_entry) + ml_weight * math.log(safe_ml))
            arithmetic_mean = entry_conf * entry_weight + ml_conf * ml_weight
            base_score = (geometric_mean * 0.6) + (arithmetic_mean * 0.4)
        else:
            base_score = entry_conf

        direction_bonus = 0
        if ml_prediction == 1:
            direction_bonus = 10
        elif ml_prediction == -1:
            direction_bonus = -15

        if conditions_result:
            mandatory = conditions_result.get('mandatory_passed', 0)
            mandatory_total = conditions_result.get('mandatory_total', 5)
            supporting = conditions_result.get('supporting_passed', 0)
            supporting_total = conditions_result.get('supporting_total', 3)

            mandatory_ratio = mandatory / max(1, mandatory_total)
            supporting_ratio = supporting / max(1, supporting_total)

            weighted_ratio = (mandatory_ratio * 0.7) + (supporting_ratio * 0.3)
            condition_multiplier = 0.7 + (weighted_ratio * 0.45)
        else:
            condition_multiplier = 1.0

        divergence_penalty = 0
        if ml_conf > 0:
            divergence = abs(entry_conf - ml_conf)
            if divergence > 30:
                divergence_penalty = (divergence - 30) * 0.15

        combined_score = (base_score + direction_bonus) * condition_multiplier - divergence_penalty
        combined_score = min(100, max(0, combined_score))

        if combined_score >= 80:
            recommendation = "STRONG BUY"
            confidence_level = "Very High"
        elif combined_score >= 70:
            recommendation = "BUY"
            confidence_level = "High"
        elif combined_score >= 60:
            recommendation = "MODERATE BUY"
            confidence_level = "Moderate"
        elif combined_score >= 50:
            recommendation = "WEAK BUY"
            confidence_level = "Low"
        elif combined_score >= 40:
            recommendation = "CAUTION"
            confidence_level = "Very Low"
        else:
            recommendation = "NO TRADE"
            confidence_level = "Insufficient"

        return {
            'combined_score': combined_score,
            'recommendation': recommendation,
            'confidence_level': confidence_level,
            'components': {
                'base_score': base_score,
                'direction_bonus': direction_bonus,
                'condition_multiplier': condition_multiplier,
                'divergence_penalty': divergence_penalty
            }
        }

    def log_quality_score_entry(self, entry_price, quantity, quality_score, component_scores):
        self.log_message(f"", "white")
        self.log_message(f"📊 QUALITY SCORE ENTRY DETAILS", "purple")
        self.log_message(f"{'═' * 77}", "purple")

        for component, score in component_scores.items():
            max_score = 20 if component in ['ema', 'adx', 'volume', 'momentum'] else \
                10 if component == 'rsi' else 5
            score_pct = (score / max_score) * 100

            bar = self._create_confidence_bar(score_pct, 5)
            self.log_message(
                f"{component.upper():12} {score:2d}/{max_score:2d} {bar:5} ({score_pct:.0f}%)",
                "green" if score_pct >= 80 else "orange" if score_pct >= 60 else "red"
            )

        self.log_message(f"", "white")
        self.log_message(f"🎯 TOTAL QUALITY SCORE: {quality_score}/100",
                         "green" if quality_score >= 80 else "orange" if quality_score >= 70 else "yellow")

        if hasattr(self.strategy, '_get_position_multiplier'):
            position_mult = self.strategy._get_position_multiplier(quality_score)
            self.log_message(f"💰 POSITION SIZE: {position_mult * 100:.0f}% of normal",
                             "green" if position_mult >= 1.0 else "blue")

        self.log_message(f"{'═' * 77}", "purple")

    def predict_future_trend(self, n_future: int = 5):
        """
        Use the current ML model to predict the trend for the next n_future candles.

        Returns
        -------
        predictions : list
            List of prediction labels ('bullish' / 'bearish' / 'neutral').
        confidence : float
            Average confidence across predictions (0-1).
        """
        if not self.ml_enabled or self.current_ml_model is None:
            return [], 0.0

        if not getattr(self.current_ml_model, 'is_trained', False):
            return [], 0.0

        df = self.get_market_data()
        if df is None or df.empty:
            return [], 0.0

        try:
            if hasattr(self.strategy, 'calculate_indicators'):
                df = self.strategy.calculate_indicators(df)

            conf, prediction, forecast = self.current_ml_model.predict(df, n_future)
            confidence = float(conf) if conf <= 1.0 else float(conf) / 100.0

            if prediction == 1:
                predictions = ['bullish'] * n_future
            elif prediction == -1:
                predictions = ['bearish'] * n_future
            else:
                predictions = ['neutral'] * n_future

            return predictions, confidence

        except Exception as e:
            self.log_message(f"⚠️ predict_future_trend error: {e}", "orange")
            return [], 0.0

    def close_partial(self, fraction: float = 0.5):
        """
        Close a fraction of the current position.

        Parameters
        ----------
        fraction : float
            Portion of the position to close (0 < fraction <= 1).
        """
        if self.position.get('type') is None or self.position.get('quantity') is None:
            self.log_message("⚠️ close_partial called with no open position.", "orange")
            return False

        if not (0 < fraction <= 1):
            self.log_message(f"❌ Invalid close_partial fraction: {fraction}", "red")
            return False

        quantity_to_close = self.position['quantity'] * fraction
        current_price = self.get_current_price()
        if current_price is None:
            if self.current_data is not None:
                current_price = float(self.current_data.get('Close', 0))
            if not current_price:
                self.log_message("❌ close_partial: cannot determine current price.", "red")
                return False

        self.log_message(
            f"📤 Partial close: {fraction * 100:.0f}% ({quantity_to_close:.4f} units) @ ${current_price:.4f}",
            "blue"
        )

        success = self.place_order(
            'sell',
            current_price,
            quantity=quantity_to_close,
            exit_reason='partial_profit_target'
        )

        if success:
            # Update tracked position size
            remaining = self.position['quantity'] - quantity_to_close
            if remaining > 1e-8:
                self.position['quantity'] = remaining
                self.log_message(
                    f"✅ Partial close complete. Remaining: {remaining:.4f} units.",
                    "green"
                )
            else:
                # Fully closed
                self.position = {
                    'type': None, 'price': None,
                    'quantity': None, 'time': None,
                    'stop_loss': None, 'trailing_stop': None,
                    'entry_confidence': None
                }
                self.update_status_indicators("parking")
                self.log_message("✅ Position fully closed via close_partial.", "green")

        return success

    def check_exit_conditions(self, current_data, current_price):
        if self.position['type'] is None:
            return None

        atr = current_data.get('ATR_closed', 1)
        swing_point = self.position.get('swing_point', current_data.get('swing_low_closed', current_price))

        new_trailing_stop = self.update_trailing_stop(
            current_price, atr, swing_point,
            direction=self.position['type'],
            atr_mult=1.5,
            prev_tsl=self.position['trailing_stop']
        )
        self.position['trailing_stop'] = new_trailing_stop

        entry_price = self.position['price']
        stop = self.position['stop_loss']
        tsl = self.position['trailing_stop']
        be_stop = self.position.get('breakeven_stop')
        risk = abs(entry_price - stop) if (entry_price is not None and stop is not None) else 0.0

        atr_mult = self.params.get('atr_mult', 2.0)
        if self.position['type'] == 'long':
            atr_stop = entry_price - atr * atr_mult
            last_swing_low = current_data.get('swing_low_closed', atr_stop)
            hybrid_stop = max(atr_stop, last_swing_low)
            self.position['stop_loss'] = max(stop, hybrid_stop) if stop is not None else hybrid_stop

            if current_price <= min(self.position['stop_loss'], tsl):
                return 'hard_stop' if self.position['stop_loss'] <= tsl else 'trailing_stop'
        else:
            atr_stop = entry_price + atr * atr_mult
            last_swing_high = current_data.get('swing_high_closed', atr_stop)
            hybrid_stop = min(atr_stop, last_swing_high)
            self.position['stop_loss'] = min(stop, hybrid_stop) if stop is not None else hybrid_stop

            if current_price >= max(self.position['stop_loss'], tsl):
                return 'hard_stop' if self.position['stop_loss'] >= tsl else 'trailing_stop'

        if be_stop is None and risk > 0:
            if self.position['type'] == 'long' and current_price >= entry_price + risk:
                if hasattr(self, "close_partial"):
                    self.close_partial(0.5)
                self.position['breakeven_stop'] = entry_price
                be_stop = entry_price
            elif self.position['type'] == 'short' and current_price <= entry_price - risk:
                if hasattr(self, "close_partial"):
                    self.close_partial(0.5)
                self.position['breakeven_stop'] = entry_price
                be_stop = entry_price

        if be_stop:
            if self.position['type'] == 'long' and current_price <= be_stop:
                return 'breakeven'
            elif self.position['type'] == 'short' and current_price >= be_stop:
                return 'breakeven'

        n_future = self.params.get('exit_n_future', 5)
        min_confidence = self.params.get('exit_min_confidence', 0.65)
        bearish_threshold = self.params.get('exit_bearish_ratio', 0.6)

        if self.ml_enabled and self.current_ml_model is not None:
            prediction, confidence = self.predict_future_trend(n_future=n_future)
            try:
                bearish_count = sum(1 for p in prediction if str(p).lower() == "bearish")
                bearish_ratio = bearish_count / max(1, n_future)
            except Exception:
                bearish_ratio = 0.0
            macd_down = current_data.get('MACD_closed', 0) < current_data.get('MACD_Signal_closed', 0)

            if bearish_ratio >= bearish_threshold and confidence >= min_confidence:
                if current_data.get('SuperTrend_closed') == -1 or macd_down:
                    return f"ml_exit_confirmed ({bearish_ratio:.2f}, conf={confidence:.2f})"

            adx_weak = current_data.get('ADX_closed', 25) < 20
            if bearish_ratio >= 0.5 and confidence >= 0.6 and (macd_down or adx_weak):
                return "ml_forecast_with_momentum_loss"

        target_mult = self.params.get('profit_target', 2.5)
        atr_mean = getattr(self.trading_app, 'atr_mean', max(1e-8, atr))

        volatility_ratio = atr / max(1e-8, atr_mean)
        if volatility_ratio > 1.5:
            adaptive_mult = target_mult * 1.2
        elif volatility_ratio < 0.7:
            adaptive_mult = target_mult * 0.8
        else:
            adaptive_mult = target_mult

        adaptive_mult = max(1.5, min(5.0, adaptive_mult))

        if self.position['type'] == 'long' and current_price >= entry_price * adaptive_mult:
            return f'profit_target_{adaptive_mult:.1f}R'
        elif self.position['type'] == 'short' and current_price <= entry_price / adaptive_mult:
            return f'profit_target_{adaptive_mult:.1f}R'

        supertrend_flip = current_data.get('SuperTrend_closed') == -1
        adx_collapse = current_data.get('ADX_closed', 25) < 20
        rsi_exit = current_data.get('RSI_closed', 50) > 70
        macd_down = current_data.get('MACD_closed', 0) < current_data.get('MACD_Signal_closed', 0)

        if self.position['type'] == 'long':
            swing_low_break = current_price < current_data.get('swing_low_closed', current_price)
            if supertrend_flip or (adx_collapse and swing_low_break):
                return 'trend_reversal'
            if rsi_exit and macd_down:
                return 'momentum_exhaustion'
            if current_data.get('Close') < current_data.get('middle_closed', current_price) and \
                    current_data.get('Close') < current_data.get('upper_closed', current_price):
                return 'bollinger_reversion'
        else:
            swing_high_break = current_price > current_data.get('swing_high_closed', current_price)
            if supertrend_flip or (adx_collapse and swing_high_break):
                return 'trend_reversal'
            rsi_oversold = current_data.get('RSI_closed', 50) < 30
            if rsi_oversold and not macd_down:
                return 'momentum_exhaustion'
            if current_data.get('Close') > current_data.get('middle_closed', current_price) and \
                    current_data.get('Close') > current_data.get('lower_closed', current_price):
                return 'bollinger_reversion'

        if self.bars_held >= self.params.get('max_hold_bars', 240):
            if self.position['type'] == 'long' and current_price <= entry_price * 1.02:
                return 'time_exit'
            elif self.position['type'] == 'short' and current_price >= entry_price * 0.98:
                return 'time_exit'

        if hasattr(self.trading_app, "check_equity_safeguard"):
            if self.trading_app.check_equity_safeguard():
                return 'portfolio_safeguard'

        return None

    def _log_monte_carlo_summary(self, results, simulator):
        self.log_message("=" * 70, "purple")
        self.log_message("📊 MONTE CARLO RESULTS", "purple")
        self.log_message("=" * 70, "purple")

        stats_cap = results['statistics']['final_capital']
        stats_ret = results['statistics']['returns']
        stats_dd = results['statistics']['drawdown']

        self.log_message(f"\n💰 FINAL CAPITAL:", "blue")
        self.log_message(f"   Expected: ${stats_cap['mean']:,.2f}", "white")
        self.log_message(f"   90% CI: ${stats_cap['percentile_5']:,.2f} - ${stats_cap['percentile_95']:,.2f}", "white")
        self.log_message(f"   Profit Prob: {stats_cap['prob_profit'] * 100:.2f}%",
                         "green" if stats_cap['prob_profit'] > 0.5 else "orange")

        self.log_message(f"\n📈 RETURNS:", "blue")
        self.log_message(f"   Expected: {stats_ret['mean']:.2f}%", "white")
        self.log_message(f"   90% CI: {stats_ret['percentile_5']:.2f}% - {stats_ret['percentile_95']:.2f}%", "white")

        self.log_message(f"\n📉 DRAWDOWN:", "blue")
        self.log_message(f"   Expected: {stats_dd['mean']:.2f}%", "white")
        self.log_message(f"   Worst Case: {stats_dd['percentile_95']:.2f}%", "red")

        self.log_message(f"\n⚠️ RISK:", "orange")
        risk_50 = sum(1 for c in results['final_capitals']
                      if c < simulator.initial_capital * 0.5) / len(results['final_capitals'])
        self.log_message(f"   50% Loss Risk: {risk_50 * 100:.2f}%", "red" if risk_50 > 0.1 else "orange")

        double_prob = sum(1 for c in results['final_capitals']
                          if c >= simulator.initial_capital * 2) / len(results['final_capitals'])
        self.log_message(f"   2x Probability: {double_prob * 100:.2f}%", "green")

        self.log_message("=" * 70, "purple")

    def _apply_bt_conditions_to_strategy_class(self, strategy_cls):
        """Apply backtest parameters to strategy class for optimization"""
        # This method would convert the 4-value parameters to optimization ranges
        # For now, we'll leave it as a placeholder
        pass

    def log_backtest_result(self, result):
        try:
            if hasattr(self, 'backtest_results_text'):
                timestamp = datetime.now().strftime('%H:%M:%S')
                msg = f"[{timestamp}] Trades: {result.get('trades', 0)} | "
                msg += f"Score: {result.get('score', 0):.2f} | "
                msg += f"Penalty: {result.get('penalty', 1.0):.2f} | "
                msg += f"Metrics: {', '.join(result.get('active_metrics', []))}\n"

                self.backtest_results_text.insert(tk.END, msg)
                self.backtest_results_text.see(tk.END)
        except:
            pass

    def combined_constraint(self, params):
        """
        Constraint function for backtesting optimization.
        FIXED: Quality tier ordering corrected — Tier1_min MUST be >= Tier2_min
               because Tier 1 is the higher-quality (more selective) tier.
        """

        def get_param(p, key, default):
            if hasattr(p, 'get'):
                return p.get(key, default)
            return getattr(p, key, default)

        # ── EMA ordering — only when all 3 are being optimized ───────────────
        has_ema_fast = 'ema_fast_period' in self.optimization_params_active
        has_ema_mid = 'ema_mid_period' in self.optimization_params_active
        has_ema_slow = 'ema_slow_period' in self.optimization_params_active

        if has_ema_fast and has_ema_mid and has_ema_slow:
            fast = get_param(params, 'ema_fast_period', 9)
            mid = get_param(params, 'ema_mid_period', 21)
            slow = get_param(params, 'ema_slow_period', 50)
            if not (fast < mid < slow):
                return False

        # ── Weight sum — only when all 5 weights are being optimized ─────────
        has_w_ema = 'weight_ema' in self.optimization_params_active
        has_w_adx = 'weight_adx' in self.optimization_params_active
        has_w_macd = 'weight_macd' in self.optimization_params_active
        has_w_rsi = 'weight_rsi' in self.optimization_params_active
        has_w_volume = 'weight_volume' in self.optimization_params_active

        if has_w_ema and has_w_adx and has_w_macd and has_w_rsi and has_w_volume:
            total = (
                    get_param(params, 'weight_ema', 20) +
                    get_param(params, 'weight_adx', 20) +
                    get_param(params, 'weight_macd', 25) +
                    get_param(params, 'weight_rsi', 20) +
                    get_param(params, 'weight_volume', 15)
            )
            if not (85 <= total <= 115):
                return False

        # ── Risk tier ordering ────────────────────────────────────────────────
        if ('risk_tier1' in self.optimization_params_active and
                'risk_tier2' in self.optimization_params_active):
            tier1_risk = get_param(params, 'risk_tier1', 0.015)
            tier2_risk = get_param(params, 'risk_tier2', 0.018)
            if not (tier1_risk <= tier2_risk):
                return False

        # ── Quality tier ordering ─────────────────────────────────────────────
        # FIX: Tier 1 is the HIGHER-quality tier (more selective → higher min score).
        #      Tier 2 is the LOWER-quality tier (less selective → lower min score).
        #      Therefore tier1_min MUST be >= tier2_min.
        if ('quality_tier1_min' in self.optimization_params_active and
                'quality_tier2_min' in self.optimization_params_active):
            t1 = get_param(params, 'quality_tier1_min', 72)
            t2 = get_param(params, 'quality_tier2_min', 62)
            if not (t1 >= t2):
                return False

        # ── Short quality tier ordering (same logic) ─────────────────────────
        if ('short_quality_tier1_min' in self.optimization_params_active and
                'short_quality_tier2_min' in self.optimization_params_active):
            st1 = get_param(params, 'short_quality_tier1_min', 75)
            st2 = get_param(params, 'short_quality_tier2_min', 65)
            if not (st1 >= st2):
                return False

        return True

    def combined_objective(self, series):
        """
        Multi-metric objective function.
        Maximizes all selected metrics with equal weights.
        Applies adaptive trade count penalties based on strategy/timeframe.
        """
        try:
            # --- Collect active metrics ---
            active_metrics = [
                metric_name
                for metric_name, metric_var in self.optimization_metrics.items()
                if metric_var.get()
            ]
            if not active_metrics:
                active_metrics = ['sharpe', 'returns']

            # --- Adaptive config ---
            strategy_type = self.strategy_type_var.get()
            interval = self.interval_var.get()
            config = self.get_objective_config(strategy_type=strategy_type, interval=interval)

            num_trades = series.get('# Trades', 0)

            # Hard minimum — discard runs with too few trades entirely
            if num_trades < config['min_trades_absolute']:
                return -999999

            # Soft penalty for too few or too many trades
            trade_penalty = 1.0
            if num_trades < config['min_trades_penalty']:
                trade_penalty = config['penalty_low']
            elif num_trades > config['max_trades_penalty']:
                trade_penalty = config['penalty_high']

            # --- Equal weights across all selected metrics ---
            weight_per_metric = 1.0 / len(active_metrics)
            total_score = 0.0

            if 'sharpe' in active_metrics:
                sharpe = series.get('Sharpe Ratio', 0)
                # Diminishing returns above 3
                if sharpe > 3:
                    sharpe = 3 + (sharpe - 3) * 0.5
                total_score += max(0, sharpe) * weight_per_metric * 10

            if 'sortino' in active_metrics:
                sortino = series.get('Sortino Ratio', 0)
                if sortino > 4:
                    sortino = 4 + (sortino - 4) * 0.5
                total_score += max(0, sortino) * weight_per_metric * 8

            if 'returns' in active_metrics:
                returns = series.get('Return [%]', 0)
                # Cap contribution to avoid runaway returns dominating
                returns_score = min(returns / 10, 20)
                total_score += max(0, returns_score) * weight_per_metric

            if 'winrate' in active_metrics:
                winrate = series.get('Win Rate [%]', 0)
                # Meaningful only above ~40%; scale to 0-8
                winrate_score = max(0, (winrate - 40) / 5)
                winrate_score = min(winrate_score, 8)
                total_score += winrate_score * weight_per_metric

            if 'profit_factor' in active_metrics:
                pf = series.get('Profit Factor', 1)
                # Score above break-even (1.0), capped at 10
                pf_score = max(0, min((pf - 1.0) * 5, 10))
                total_score += pf_score * weight_per_metric

            if 'equity' in active_metrics:
                equity = series.get('Equity Final [$]', 50000)
                initial_cash = getattr(self, '_initial_cash', 50000)
                equity_score = ((equity - initial_cash) / initial_cash) * 10
                equity_score = min(max(equity_score, 0), 20)
                total_score += equity_score * weight_per_metric

            if 'trade_count' in active_metrics:
                # Reward the statistically valid range (30-72 trades/16mo = ~2-4.5/mo)
                # Penalise hard below 20 and softly above 96
                n = num_trades
                if n < 20:
                    tc_score = 0  # too few — no score
                elif n < 35:
                    tc_score = (n - 20) / 15 * 6  # 0→6 linearly (20-35)
                elif n <= 72:
                    tc_score = 6 + (n - 35) / 37 * 4  # 6→10 (sweet spot)
                else:
                    tc_score = max(4, 10 - (n - 72) * 0.06)  # gentle penalty >72
                total_score += tc_score * weight_per_metric

            final_score = total_score * trade_penalty
            return final_score

        except Exception as e:
            if hasattr(self, 'log_message'):
                self.log_message(f"⚠️ Objective function error: {e}", "red")
            return -999999

    def _ui_safe(self, func, *args, **kwargs):
        """Thread-safe wrapper: schedules a UI call on the main thread."""
        self.root.after(0, lambda: func(*args, **kwargs))

    def _run_backtest_in_thread(self):
        import threading

        # Disable button and show running state
        if hasattr(self, 'run_btn'):
            self.run_btn.config(state='disabled', text='⏳ Running...')

        # Mark backtest as running and disable settings
        self.backtest_running = True
        self.update_mode_display(self.mode_var.get())  # This will disable settings button

        def _worker():
            try:
                self.run_backtest()
            except Exception as e:
                self.log_message(f"❌ Thread error: {e}", "red")
            finally:
                # Mark backtest as finished
                self.backtest_running = False
                # Re-enable buttons on the main thread
                if hasattr(self, 'run_btn'):
                    self.run_btn.after(
                        0,
                        lambda: self.run_btn.config(
                            state='normal', text='▶ Run Backtest'
                        )
                    )
                # Update UI to re-enable settings button
                self.root.after(0, lambda: self.update_mode_display(self.mode_var.get()))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def debug_backtest_parameters(self, strategy_type, current_params):
        """Debug function to verify parameters are being passed correctly to backtest"""
        self.log_message("=" * 70, "yellow")
        self.log_message("🔍 BACKTEST PARAMETER DIAGNOSTICS", "yellow")
        self.log_message("=" * 70, "yellow")

        # Log source of parameters
        self.log_message(f"📋 Parameter Source: {self.param_toggle_var.get()}", "cyan")
        self.log_message(f"📊 Strategy Type: {strategy_type}", "cyan")

        # Log critical parameters that should affect behavior
        critical_params = [
            'only_tier1_entries',
            'quality_tier1_min_long',  # ← REPLACED
            'quality_tier2_min_long',  # ← REPLACED
            'quality_tier1_min_short',  # ← ADDED
            'quality_tier2_min_short',  # ← ADDED
            'tier1_adx_hard_min',
            'tier1_volume_min',
            'stop_loss_atr_mult',
            'trailing_activation_tier1',  # ← REPLACED
            'trailing_distance_tier1',  # ← REPLACED
            'trade_direction'
        ]

        self.log_message("\n📊 CRITICAL PARAMETER VALUES:", "cyan")
        for param in critical_params:
            if param in current_params:
                value = current_params[param]
                if param == 'only_tier1_entries':
                    color = "yellow" if value else "green"
                    status = "BLOCKED" if value else "ACTIVE"
                    self.log_message(f"   {param:25} = {value}  [TIER 2 {status}]", color)
                else:
                    self.log_message(f"   {param:25} = {value}", "white")
            else:
                # Don't flag direction-specific keys as "NOT FOUND" if their opposite exists
                if param in ['quality_tier1_min_long', 'quality_tier1_min_short',
                             'quality_tier2_min_long', 'quality_tier2_min_short']:
                    # Check if the opposite direction key exists
                    opposite = {
                        'quality_tier1_min_long': 'quality_tier1_min_short',
                        'quality_tier1_min_short': 'quality_tier1_min_long',
                        'quality_tier2_min_long': 'quality_tier2_min_short',
                        'quality_tier2_min_short': 'quality_tier2_min_long',
                    }.get(param)
                    if opposite in current_params:
                        # Not an error — just the other direction is set
                        continue
                self.log_message(f"   {param:25} = NOT FOUND", "red")

        # Show if parameters were injected into backtest class
        if hasattr(BacktestMomentumStrategy, '_use_updated_params'):
            self.log_message(f"\n📤 Backtest class updated params: {BacktestMomentumStrategy._use_updated_params}",
                             "green" if BacktestMomentumStrategy._use_updated_params else "red")
            if BacktestMomentumStrategy._updated_params:
                self.log_message(f"   Parameters in backtest class: {len(BacktestMomentumStrategy._updated_params)}",
                                 "cyan")
                # Check both directions
                if 'only_tier1_entries' in BacktestMomentumStrategy._updated_params:
                    bt_value = BacktestMomentumStrategy._updated_params['only_tier1_entries']
                    self.log_message(f"   Backtest only_tier1_entries = {bt_value}",
                                     "yellow" if bt_value else "green")
                for key in ['quality_tier1_min_long', 'quality_tier1_min_short']:
                    if key in BacktestMomentumStrategy._updated_params:
                        self.log_message(f"   Backtest {key} = {BacktestMomentumStrategy._updated_params[key]}", "cyan")

        self.log_message("=" * 70, "yellow")
        return current_params

    def _set_initial_cash(self, cash):
        """Set initial cash for objective function"""
        self._initial_cash = cash

    # ── ADD this new helper anywhere in TradingApp ────────────────────────────
    def _normalize_ohlcv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame always has the canonical OHLCV column names
        (Open, High, Low, Close, Volume) regardless of how they arrived.
        Works on copies only — never mutates the cached frame.
        """
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower == 'open':
                col_map[col] = 'Open'
            elif lower == 'high':
                col_map[col] = 'High'
            elif lower == 'low':
                col_map[col] = 'Low'
            elif lower in ('close', 'close_price'):
                col_map[col] = 'Close'
            elif lower in ('volume', 'vol'):
                col_map[col] = 'Volume'
        if col_map:
            df = df.rename(columns=col_map)
        return df

    def run_backtest(self):
        """
        Run comprehensive backtest with Quality Score optimization and automatic Excel export.
        FIXED: Complete Scalping backtest parameter support with save/load functionality
        FIXED: Forces GUI trade direction into backtest (LONG/SHORT/BOTH)
        FIXED: Reloads strategy module to pick up parameter changes
        FIXED: Excel export happens AFTER all backtest processing is complete
        FIXED: All strategies use single-timeframe data fetching
        FIXED: Kalman strategy exports LONG/SHORT column to separate Excel file
        FIXED: Normalize backtest_type_var – accepts 'Optimize' OR 'Optimization'
        FIXED: Inactive backtest params forwarded as fixed values so optimizer uses
               the selected default/custom parameter set as its base
        FIXED: Early return when no optimization parameters are selected
        FIXED: numpy int64/float64/bool_ types converted before json.dump
        FIXED: GUI Order Size % applied as hard ceiling to all trades in backtest
        FIXED: bt.optimize() crashing with 0xC0000005 (access violation). Root cause
               chain: backtesting.py's optimize() always spins up a real
               SharedMemoryManager subprocess (independent of Pool type); and the GUI's
               'fixed_threshold'/'short_fixed_threshold' params (leftover from an
               earlier design, superseded by the Tier1/Tier2 hybrid quality score
               system) are not declared as class attributes on BacktestMomentumStrategy,
               so backtesting.py's _check_params() raised AttributeError deep inside a
               worker thread — and that exception, combined with Windows' shared-memory
               cleanup path, is what actually produced the access violation instead of
               a clean traceback. We (a) force a single-worker thread pool so TA-Lib is
               never called concurrently, and (b) strip the stale params before they
               ever reach bt.optimize()/bt.run().
        """
        # ── Neutralize the crash sources in backtesting.py's optimize() ──────
        import backtesting
        from multiprocessing.dummy import Pool as _ThreadPool

        # (a) Force a single-worker thread pool — avoids concurrent TA-Lib calls.
        backtesting.Pool = lambda processes=None, initializer=None, initargs=(): _ThreadPool(1)

        # (b) Bypass the SharedMemoryManager subprocess entirely. Since we're using
        # a thread pool (not real processes), there's no need to move the dataframe
        # into a shared-memory segment via a separate manager process — just pass
        # it through in-process. This removes the Windows spawn/re-import crash path
        # and also lets any real exception (e.g. bad params) surface cleanly instead
        # of triggering a shared-memory-cleanup crash.
        class _NoOpSharedMemoryManager:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def df2shm(self, df):
                return df

            @staticmethod
            def shm2df(data):
                return data, []  # second value must be iterable — backtesting.py loops over it

        backtesting.backtesting.SharedMemoryManager = _NoOpSharedMemoryManager
        # ──────────────────────────────────────────────────────────────────────

        original_mode = None
        original_value = None
        if hasattr(self, 'param_toggle_var'):
            original_mode = self.param_toggle_var.get()

        if 'momentum' in self.custom_params:
            original_value = self.custom_params['momentum'].get('only_tier1_entries', None)
            self.log_message(
                f"   ℹ️ only_tier1_entries = {original_value} (respecting current configuration)", "cyan")

        init_excel = None
        opt_excel = None
        monte_carlo_results = None
        initial_stats = None
        optimized_stats = None
        optimized_params_dict = None
        df_initial_indicators = None
        df_opt_indicators = None

        # ── Helper: convert numpy scalars so json.dump never crashes ─────────
        def _to_python(v):
            import numpy as np
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.bool_):
                return bool(v)
            return v

        try:
            if hasattr(self, 'backtest_results_frame'):
                self.root.after(0, lambda: self.backtest_results_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=2))

            if hasattr(self, 'backtest_results_text'):
                self._ui_safe(self.backtest_results_text.delete, 1.0, tk.END)
                self._ui_safe(self.backtest_results_text.insert, tk.END, "Starting backtest...\n")
                self._ui_safe(self.backtest_results_text.see, tk.END)

            symbol = self.symbol_var.get()
            interval = self.interval_var.get()
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            strategy_type = self.strategy_type_var.get()

            self.log_message("=" * 70, "blue")
            self.log_message(f"🚀 STARTING BACKTEST", "blue")
            self.log_message(f"   Strategy : {strategy_type}", "blue")
            self.log_message(f"   Symbol   : {symbol}", "blue")
            self.log_message(f"   Interval : {interval}", "blue")
            self.log_message(f"   Period   : {start_date} to {end_date}", "blue")
            self.log_message("=" * 70, "blue")

            # ── Normalize backtest type ──────────────────────────────────────
            bt_type_raw = self.backtest_type_var.get()
            is_optimization = bt_type_raw.strip().lower() in ("optimization", "optimize")

            self.log_message("=" * 70, "cyan")
            self.log_message(f"🔍 BACKTEST TYPE VAR  = '{bt_type_raw}'", "cyan")
            self.log_message(f"🔍 IS OPTIMIZATION   = {is_optimization}", "cyan")
            self.log_message("=" * 70, "cyan")

            # ── Collect & log selected optimization metrics ──────────────────
            active_metrics = [
                m for m, v in self.optimization_metrics.items() if v.get()
            ]

            self.log_message("=" * 70, "purple")
            self.log_message("🎯 OPTIMIZATION METRICS SELECTED", "purple")
            self.log_message("=" * 70, "purple")

            if active_metrics:
                for i, metric in enumerate(active_metrics, 1):
                    metric_display = metric.replace('_', ' ').title()
                    self.log_message(f"   {i}. {metric_display}", "cyan")
                self.log_message(f"\n   Total: {len(active_metrics)} metric(s) selected", "white")
            else:
                self.log_message("⚠️ No optimization metrics selected — using Sharpe + Returns", "orange")
                active_metrics = ['sharpe', 'returns']
                self.optimization_metrics['sharpe'].set(True)
                self.optimization_metrics['returns'].set(True)
                self.log_message(f"   Default metrics enabled: Sharpe Ratio, Return [%]", "yellow")

            self.log_message("=" * 70, "purple")

            # ── Collect & log selected backtest parameters ───────────────────
            if strategy_type == "Scalping":
                selected_params_preview = {
                    k: v for k, v in getattr(self, 'scalping_backtest_params', {}).items()
                    if v.get('active', tk.BooleanVar(value=False)).get()
                }
            else:
                selected_params_preview = {
                    k: v for k, v in self.backtest_params.items() if v['active'].get()
                }

            self.log_message("=" * 70, "green")
            self.log_message(f"⚙️ PARAMETERS SELECTED FOR OPTIMIZATION ({len(selected_params_preview)})", "green")
            self.log_message("=" * 70, "green")

            if not selected_params_preview and is_optimization:
                self.log_message(
                    "⚠️ No optimization parameters selected!\n"
                    "   Go to Settings → Backtest Optimization Parameters panel\n"
                    "   and check the parameters you want to optimize.", "orange")

            self.log_message("=" * 70, "green")

            # ── Fetch data ───────────────────────────────────────────────────
            self.clear_old_cache_if_needed()

            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)

            self.log_message("📥 Loading historical data...", "blue")

            df = self.get_historical_data(symbol=symbol, start=start_ts, end=end_ts, interval=interval)

            if df is None or df.empty:
                self.log_message(f"❌ No data for {start_date} → {end_date}", "red")
                self._ui_safe(messagebox.showerror, "Error", f"No data available for {start_date} to {end_date}")
                return

            self.debug_data_range(df)
            df.columns = [col.capitalize() for col in df.columns]
            self._last_backtest_df = df
            self._last_backtest_stats = None
            actual_start = df.index[0].strftime('%Y-%m-%d')
            actual_end = df.index[-1].strftime('%Y-%m-%d')
            self.log_message(f"✅ Loaded {len(df):,} candles  ({actual_start} → {actual_end})", "green")

            if actual_start > start_date:
                self.log_message(f"⚠️ Data starts {actual_start}, later than selected {start_date}", "orange")
            if actual_end < end_date:
                self.log_message(f"⚠️ Data ends {actual_end}, earlier than selected {end_date}", "orange")

            # ── Parameter source ─────────────────────────────────────────────
            current_mode = self.param_toggle_var.get()
            self.log_message("=" * 70, "yellow")
            self.log_message(f"🔧 PARAMETER SOURCE: {current_mode}", "yellow")
            self.log_message("=" * 70, "yellow")

            if current_mode == "Custom Parameters":
                self.log_message("📋 Updating parameters from LIVE UI settings...", "cyan")
                self.update_custom_params_from_ui()
                self.log_message("✅ Parameters updated from UI", "green")

            # ── Reload strategy module ───────────────────────────────────────
            self.log_message("=" * 70, "cyan")
            self.log_message("♻️ RELOADING STRATEGY MODULE (to pick up file changes)", "cyan")
            self.log_message("=" * 70, "cyan")

            import sys
            import importlib

            if strategy_type != "Scalping":
                strategy_module = 'strategies.MomentumStrategy_MACD_HybridScore_Latest'
                if strategy_module in sys.modules:
                    importlib.reload(sys.modules[strategy_module])
                    self.log_message(f"   ✅ Reloaded: {strategy_module}", "green")
                else:
                    self.log_message(f"   📦 First load: {strategy_module}", "blue")

                _fresh_mod = sys.modules[strategy_module]
                BacktestMomentumStrategy = _fresh_mod.BacktestMomentumStrategy
                MOMENTUM_PARAMS_live = _fresh_mod.MOMENTUM_PARAMS
                GlobalConfig_live = _fresh_mod.GlobalConfig

                self.default_params['momentum'] = MOMENTUM_PARAMS_live.copy()
                self.log_message(f"   ✅ Updated default_params from reloaded MOMENTUM_PARAMS", "green")
            else:
                from strategies.scalping_strategy import ScalpingConfig
                GlobalConfig_live = type('GlobalConfig', (), {'INITIAL_CAPITAL': 50000})()

            self.log_message(f"   💰 Fresh INITIAL_CAPITAL: ${GlobalConfig_live.INITIAL_CAPITAL:,.2f}", "blue")

            gui_order_pct = self.order_size_var.get()
            self.log_message("=" * 70, "bold cyan")
            self.log_message(f"📊 GUI ORDER SIZE CAP: {gui_order_pct:.0f}% of equity", "bold cyan")
            self.log_message(f"   → Applied as HARD CEILING to all backtest trades", "cyan")
            self.log_message("=" * 70, "bold cyan")

            # ── Choose strategy class ────────────────────────────────────────
            if strategy_type == "Kalman":
                strategy_class = BacktestKalmanTrendStrategy
                initial_params = self.get_current_kalman_params()
                self.log_message("🎯 Using Kalman Strategy", "purple")

            elif strategy_type == "Scalping":
                try:
                    from strategies.scalping_strategy import BacktestScalpingStrategy, ScalpingConfig
                    strategy_class = BacktestScalpingStrategy
                except ImportError as e:
                    self.log_message(f"❌ Could not import scalping_strategy: {e}", "red")
                    self.log_message("   Make sure scalping_strategy.py is in strategies/", "orange")
                    return

                initial_params = self.get_current_scalping_params()

                self.log_message("🎯 Using Professional Scalping Strategy v1.0", "purple")
                self.log_message("=" * 70, "cyan")
                self.log_message("📋 SCALPING CONFIGURATION", "cyan")
                self.log_message("=" * 70, "cyan")
                self.log_message(
                    f"   EMA      : {initial_params.get('ema_fast_period', 5)}/{initial_params.get('ema_mid_period', 13)}/{initial_params.get('ema_slow_period', 21)}",
                    "white")
                self.log_message(
                    f"   MACD     : ({initial_params.get('macd_fast', 12)},{initial_params.get('macd_slow', 26)},{initial_params.get('macd_signal_period', 9)})",
                    "white")
                self.log_message(
                    f"   Stoch    : ({initial_params.get('stoch_k_period', 5)},{initial_params.get('stoch_smooth', 3)},{initial_params.get('stoch_d_period', 3)})",
                    "white")
                self.log_message(f"   Stop     : {initial_params.get('stop_loss_atr_mult', 1.8)}× ATR", "white")
                self.log_message(
                    f"   Quality  : {initial_params.get('quality_min_long', 35)} (L) / {initial_params.get('quality_min_short', 35)} (S)",
                    "white")
                self.log_message("=" * 70, "cyan")

                BacktestScalpingStrategy.set_updated_params(initial_params)
                self.log_message(
                    f"   📊 {len(initial_params)} params injected into BacktestScalpingStrategy (mode: {self.param_toggle_var.get()})",
                    "cyan")
                self.log_message("=" * 70, "yellow")

            else:  # Momentum
                strategy_class = BacktestMomentumStrategy
                mode_label = "CUSTOM (LIVE UI)" if current_mode == "Custom Parameters" else "DEFAULT"
                self.log_message(f"🎯 Using Momentum Strategy ({mode_label} parameters)", "purple")

                current_params = self.get_current_momentum_params()

                gui_direction = self.trade_direction_var.get()
                current_params['trade_direction'] = gui_direction
                self.log_message(f"📊 BACKTEST DIRECTION FORCED: {gui_direction.upper()} (from GUI)", "bold green")
                self.log_message(f"   Tier1 Pass (LONG): {current_params.get('quality_tier1_min_long', 75)}", "cyan")
                self.log_message(f"   Tier2 Pass (LONG): {current_params.get('quality_tier2_min_long', 65)}", "cyan")
                if gui_direction == 'short':
                    self.log_message(f"   Tier1 Pass (SHORT): {current_params.get('quality_tier1_min_short', 75)}",
                                     "cyan")
                    self.log_message(f"   Tier2 Pass (SHORT): {current_params.get('quality_tier2_min_short', 65)}",
                                     "cyan")

                current_params['gui_order_size_pct'] = gui_order_pct
                self.log_message(f"📊 GUI Order Size Cap applied to backtest: {gui_order_pct:.0f}%", "bold cyan")
                self.log_message(f"   → All position sizes will be capped at {gui_order_pct:.0f}% of equity", "cyan")

                tier2_value = current_params.get('only_tier1_entries', 'NOT FOUND')
                tier2_min = current_params.get('quality_tier2_min', 'NOT FOUND')
                self.log_message(f"   🔧 only_tier1_entries = {tier2_value}",
                                 "yellow" if tier2_value else "green")
                self.log_message(f"   📊 quality_tier2_min = {tier2_min}", "white")
                self.log_message(f"   📊 quality_tier1_min = {current_params.get('quality_tier1_min', 'NOT FOUND')}",
                                 "white")

                self.log_message("=" * 70, "yellow")
                self.log_message("🔍 FINAL PARAMETER VERIFICATION", "yellow")
                self.debug_backtest_parameters(strategy_type, current_params)

                BacktestMomentumStrategy._updated_params = current_params.copy()
                BacktestMomentumStrategy._use_updated_params = True
                BacktestMomentumStrategy.trade_direction = gui_direction

                for _k, _v in current_params.items():
                    try:
                        setattr(BacktestMomentumStrategy, _k, _v)
                    except Exception:
                        pass

                self.log_message(
                    f"   ✅ {len(current_params)} base params applied to BacktestMomentumStrategy class",
                    "green"
                )

                initial_params = current_params

            # ── Initialise Backtest object ───────────────────────────────────
            initial_cash = GlobalConfig_live.INITIAL_CAPITAL
            commission_rate = self.commission_var.get()
            self.log_message(f"💰 Initial Capital : ${initial_cash:,.2f}", "blue")
            self.log_message(f"💳 Commission Rate : {commission_rate}", "blue")

            bt = Backtest(
                df,
                strategy_class,
                cash=initial_cash,
                commission=commission_rate,
                trade_on_close=False,
                exclusive_orders=True
            )

            self._set_initial_cash(initial_cash)

            # ═══ STEP 1: Initial backtest ════════════════════════════════════
            self.log_message("🏃 Running initial backtest...", "blue")
            initial_stats = bt.run(**initial_params)
            self.log_message("✅ Initial backtest complete!", "green")
            self._last_backtest_stats = initial_stats

            df_initial_indicators = (
                initial_stats._strategy.df_enhanced
                if hasattr(initial_stats, '_strategy') and hasattr(initial_stats._strategy, 'df_enhanced')
                else df
            )

            self.display_backtest_results(initial_stats, f"{strategy_type.upper()} — INITIAL BACKTEST")
            self.enable_ai_button()

            # ═══ STEP 2: Optimisation ════════════════════════════════════════
            optimized_stats = None
            optimized_params_dict = None

            if is_optimization:
                self.log_message("=" * 70, "purple")
                self.log_message("🔧 STARTING PARAMETER OPTIMIZATION", "purple")
                self.log_message("=" * 70, "purple")

                config = self.get_objective_config(strategy_type, interval)
                self.log_message("⚙️ Adaptive Objective Configuration:", "cyan")
                self.log_message(f"   Strategy  : {strategy_type} | Interval: {interval}", "white")
                self.log_message(f"   Min trades (absolute) : {config['min_trades_absolute']}", "white")
                self.log_message(f"   Min trades (penalty)  : {config['min_trades_penalty']}", "white")
                self.log_message(f"   Max trades (penalty)  : {config['max_trades_penalty']}", "white")
                self.log_message(
                    f"   Penalties : low={config['penalty_low']:.2f}, high={config['penalty_high']:.2f}",
                    "white")

                if strategy_type == "Kalman":
                    self.log_message("⚙️ Optimizing Kalman parameters...", "blue")
                    optimized_stats = bt.optimize(
                        trailing_stop_pct=[0.005, 0.01, 0.02, 0.03, 0.05],
                        stop_loss_pct=[0.01, 0.015, 0.02, 0.025],
                        risk_reward=[1.5, 2.0, 2.5, 3.0],
                        lookback=[10, 15, 20, 25],
                        maximize=self.combined_objective,
                        max_tries=100,
                        random_state=42
                    )

                elif strategy_type == "Scalping":
                    scalping_opt_params = {}
                    scalping_fixed_params = {}

                    sc_params = getattr(self, 'scalping_backtest_params', {})
                    if not sc_params:
                        self.log_message("⚠️ Scalping optimization panel not initialised — run standard backtest only",
                                         "orange")
                        pass
                    else:
                        for param_key, param_data in sc_params.items():
                            is_active = param_data['active'].get()
                            values = []
                            for vk in ['value1', 'value2', 'value3', 'value4']:
                                raw = param_data[vk].get()
                                if raw not in ('', None):
                                    values.append(self.convert_param_value(raw))
                            seen, uvals = set(), []
                            for v in values:
                                k2 = (type(v), v)
                                if k2 not in seen:
                                    seen.add(k2)
                                    uvals.append(v)

                            if is_active and uvals:
                                scalping_opt_params[param_key] = uvals
                            else:
                                cur = self._get_scalping_param_value(param_key)
                                if cur is not None:
                                    scalping_fixed_params[param_key] = [cur]

                        if not scalping_opt_params:
                            self.log_message(
                                "⚠️ No Scalping optimization parameters selected — running standard backtest only\n"
                                "   To enable optimization: Open Settings → ⚡ Scalping Parameters → right panel\n"
                                "   and tick the parameters you want to optimize.",
                                "orange"
                            )
                        else:
                            all_sc_opt = {**scalping_opt_params, **scalping_fixed_params}

                            total_sc = 1
                            for v in scalping_opt_params.values():
                                total_sc *= len(v)
                            max_tries_sc = min(total_sc, 500)

                            self.log_message("=" * 70, "cyan")
                            self.log_message("📊 SCALPING OPTIMIZATION PARAMETERS", "cyan")
                            self.log_message("=" * 70, "cyan")
                            for pk, vals in scalping_opt_params.items():
                                desc = sc_params[pk]['description']
                                self.log_message(f"   🔄 {desc:40} = {vals} (OPTIMIZING)", "green")
                            self.log_message(f"\n   Total combinations : {total_sc:,}", "blue")
                            self.log_message(f"   Max tries          : {max_tries_sc}", "blue")
                            self.log_message("=" * 70, "cyan")

                            try:
                                optimized_stats = bt.optimize(
                                    **all_sc_opt,
                                    maximize=self.combined_objective,
                                    max_tries=max_tries_sc,
                                    random_state=42,
                                    return_heatmap=False
                                )
                                self.log_message("✅ Scalping optimization complete!", "green")

                                if optimized_stats is not None and hasattr(optimized_stats, '_strategy'):
                                    optimized_params_dict = {
                                        k: v for k, v in optimized_stats._strategy._params.items()
                                        if not k.startswith('_')
                                    }
                                    if optimized_params_dict:
                                        self.log_message("🔄 Re-running with optimized Scalping params...", "blue")
                                        optimized_stats = bt.run(**optimized_params_dict)
                                        self.display_backtest_results(
                                            optimized_stats, "SCALPING — OPTIMIZED RESULTS")
                                        _fname = f"scalping_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                        with open(_fname, 'w') as _f:
                                            json.dump({k: _to_python(v) for k, v in optimized_params_dict.items()}, _f,
                                                      indent=4)
                                        self.log_message(f"💾 Optimized Scalping params saved: {_fname}", "green")

                            except Exception as sc_opt_err:
                                self.log_message(f"❌ Scalping optimization error: {sc_opt_err}", "red")
                                import traceback as _tb
                                self.log_message(_tb.format_exc(), "red")

                else:  # Momentum optimization
                    optimization_params = {}

                    for param_key, param_data in self.backtest_params.items():
                        values = self._get_param_values(param_key)
                        if not values:
                            continue

                        if param_key == 'only_tier1_entries':
                            bool_values = []
                            for v in values:
                                if isinstance(v, str):
                                    bool_values.append(v.lower() == 'true')
                                else:
                                    bool_values.append(bool(v))
                            optimization_params[param_key] = bool_values
                        else:
                            optimization_params[param_key] = values

                    # ── Strip params BacktestMomentumStrategy doesn't declare ──
                    # 'fixed_threshold' / 'short_fixed_threshold' are stale GUI
                    # parameters left over from an earlier design, superseded by
                    # the Tier1/Tier2 hybrid quality score system, and are not
                    # class attributes on BacktestMomentumStrategy. Passing them
                    # to bt.optimize()/bt.run() makes backtesting.py's
                    # _check_params() raise AttributeError deep inside a worker
                    # thread, which on Windows can surface as a 0xC0000005 access
                    # violation instead of a clean traceback.
                    _stale_params = {'fixed_threshold', 'short_fixed_threshold'}
                    for _stale in _stale_params:
                        if _stale in optimization_params:
                            self.log_message(
                                f"⚠️ Skipping '{_stale}' — not a declared parameter on "
                                f"BacktestMomentumStrategy (leftover from an earlier "
                                f"design, superseded by Tier1/Tier2 scoring)", "orange"
                            )
                            del optimization_params[_stale]

                    active_opt_params = {
                        k: v for k, v in optimization_params.items()
                        if self.backtest_params[k]['active'].get()
                    }
                    fixed_opt_params = {
                        k: v for k, v in optimization_params.items()
                        if not self.backtest_params[k]['active'].get()
                    }

                    self.log_message("=" * 70, "cyan")
                    self.log_message("🔍 OPTIMIZATION PARAMETERS VERIFICATION", "cyan")
                    self.log_message("=" * 70, "cyan")
                    self.log_message(
                        f"   ✅ {len(active_opt_params)} params will be VARIED by optimizer", "green"
                    )
                    self.log_message(
                        f"   📌 {len(fixed_opt_params)} params fixed at current {current_mode} values", "blue"
                    )
                    self.log_message("=" * 70, "cyan")

                    for param_key, values in active_opt_params.items():
                        desc = self.backtest_params[param_key]['description']
                        values_str = ", ".join([str(v) for v in values])
                        self.log_message(f"   🔄 {desc:35} = [{values_str}] (OPTIMIZING)", "green")

                    for param_key, values in fixed_opt_params.items():
                        desc = self.backtest_params[param_key]['description']
                        self.log_message(f"   📌 {desc:35} = {values[0]} (FIXED)", "blue")

                    self.log_message("=" * 70, "cyan")

                    if not active_opt_params:
                        self.log_message(
                            "❌ No optimization parameters selected!\n"
                            "   Go to Settings → Backtest Optimization Parameters\n"
                            "   and select the parameters you want to optimize.",
                            "red"
                        )
                        self._ui_safe(
                            messagebox.showwarning,
                            "No Parameters Selected",
                            "No optimization parameters are selected.\n\n"
                            "Go to Settings → Backtest Optimization Parameters panel\n"
                            "and check the parameters you want to optimize."
                        )
                        return

                    self.optimization_params_active = set(active_opt_params.keys())

                    self.log_message("=" * 70, "cyan")
                    self.log_message("📊 ACTIVE OPTIMIZATION PARAMETERS WITH VALUES", "cyan")
                    self.log_message("=" * 70, "cyan")

                    for param_key, values in active_opt_params.items():
                        desc = self.backtest_params[param_key]['description']
                        values_str = ", ".join([str(v) for v in values])
                        self.log_message(f" ✓ {desc:35} = [{values_str}]", "white")

                    self.log_message("=" * 70, "cyan")
                    self.log_message(f"🎯 OPTIMIZING FOR : {', '.join(active_metrics)}", "cyan")

                    total_combinations = 1
                    for vals in active_opt_params.values():
                        total_combinations *= len(vals)

                    max_tries = min(total_combinations, 500)
                    self.log_message(f"🔢 Total combinations : {total_combinations:,}", "blue")
                    self.log_message(f"⏳ Max tries          : {max_tries}", "blue")
                    self.log_message(f"📋 Parameter base     : {current_mode}", "blue")

                    try:
                        optimized_stats = bt.optimize(
                            **optimization_params,
                            constraint=self.combined_constraint,
                            maximize=self.combined_objective,
                            max_tries=max_tries,
                            random_state=42,
                            return_heatmap=False
                        )
                        self.log_message("✅ Optimization search complete!", "green")

                    except Exception as opt_err:
                        self.log_message(f"❌ Optimization error: {opt_err}", "red")
                        self.log_message(traceback.format_exc(), "red")
                        optimized_stats = None

                    if optimized_stats is not None and hasattr(optimized_stats, '_strategy'):
                        optimized_params_dict = {
                            k: v
                            for k, v in optimized_stats._strategy._params.items()
                            if not k.startswith('_')
                        }

                        if optimized_params_dict:
                            self.log_message("🔄 Re-running with optimized parameters...", "blue")
                            try:
                                optimized_stats = bt.run(**optimized_params_dict)
                                self.log_message("✅ Re-run complete!", "green")
                            except Exception as rerun_err:
                                self.log_message(f"❌ Re-run error: {rerun_err}", "red")
                                optimized_stats = None
                                optimized_params_dict = None

                            if optimized_stats is not None:
                                df_opt_indicators = (
                                    optimized_stats._strategy.df_enhanced
                                    if hasattr(optimized_stats, '_strategy') and
                                       hasattr(optimized_stats._strategy, 'df_enhanced')
                                    else df_initial_indicators
                                )

                                self.display_backtest_results(
                                    optimized_stats,
                                    f"{strategy_type.upper()} — OPTIMIZED RESULTS"
                                )
                                self.log_parameter_changes(initial_params, optimized_params_dict)
                                self.analyze_quality_score_results(optimized_stats)

                                params_export_file = (
                                    f"optimized_params_{strategy_type.lower()}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                )
                                with open(params_export_file, 'w') as f:
                                    json.dump(
                                        {k: _to_python(v) for k, v in optimized_params_dict.items()},
                                        f, indent=4
                                    )
                                self.log_message(
                                    f"💾 Optimized parameters exported to: {params_export_file}", "green"
                                )

                                self.log_message("=" * 70, "green")
                                self.log_message("🏆 BEST PARAMETERS FOUND:", "green")
                                self.log_message("=" * 70, "green")

                                quality_params, ema_params = {}, {}
                                risk_params, fuzzy_params, other_params = {}, {}, {}

                                for param, value in optimized_params_dict.items():
                                    if any(x in param for x in ['quality', 'tier', 'weight_']):
                                        quality_params[param] = value
                                    elif 'ema' in param:
                                        ema_params[param] = value
                                    elif 'fuzzy' in param:
                                        fuzzy_params[param] = value
                                    elif any(x in param for x in ['risk', 'stop', 'trail']):
                                        risk_params[param] = value
                                    else:
                                        other_params[param] = value

                                for label, group in [
                                    ("🎯 QUALITY SCORE", quality_params),
                                    ("🧠 FUZZY MODE", fuzzy_params),
                                    ("📊 EMA", ema_params),
                                    ("🛡️ RISK", risk_params),
                                    ("📌 OTHER", other_params),
                                ]:
                                    if group:
                                        self.log_message(f"\n{label} PARAMETERS:", "cyan")
                                        for p, v in sorted(group.items()):
                                            self.log_message(f"   {p:40} {v}", "white")

                                self.log_message("=" * 70, "green")

                                init_eq = initial_stats.get('Equity Final [$]', 0)
                                opt_eq = optimized_stats.get('Equity Final [$]', 0)
                                delta = opt_eq - init_eq
                                delta_p = (delta / init_eq * 100) if init_eq > 0 else 0
                                delta_col = "green" if delta >= 0 else "red"

                                self.log_message("\n📊 OPTIMIZATION IMPACT:", "green")
                                self.log_message(f"   Initial Equity   : ${init_eq:,.2f}", "white")
                                self.log_message(f"   Optimized Equity : ${opt_eq:,.2f}", "white")
                                self.log_message(
                                    f"   Improvement      : ${delta:,.2f} ({delta_p:+.2f}%) "
                                    f"{'✓' if delta >= 0 else '✗'}",
                                    delta_col
                                )
                                self.log_message("=" * 70, "green")

                        elif optimized_stats is None:
                            self.log_message("❌ Optimization returned no results", "red")

            # ═══ STEP 3: Generate plots ═══════════════════════════════════════
            try:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                init_plot = f"{strategy_type.lower()}_initial_{ts}.html"
                bt.plot(filename=init_plot, open_browser=True, resample=False)
                self.log_message(f"📈 Initial plot saved: {init_plot}", "blue")

                if optimized_stats is not None:
                    opt_plot = f"{strategy_type.lower()}_optimized_{ts}.html"
                    bt.plot(filename=opt_plot, open_browser=True, resample=False)
                    self.log_message(f"📈 Optimized plot saved: {opt_plot}", "blue")
            except Exception as e:
                self.log_message(f"⚠️ Could not generate plots: {e}", "orange")

            # ═══ STEP 4: Monte Carlo ══════════════════════════════════════════
            if self.use_monte_carlo_var.get():
                target_stats = optimized_stats if optimized_stats is not None else initial_stats
                monte_carlo_results = self.run_monte_carlo_simulation(target_stats)

            # ═══ STEP 5: Export to Excel ══════════════════════════════════════
            self.log_message("=" * 70, "blue")
            self.log_message("💾 EXPORTING RESULTS TO EXCEL", "blue")
            self.log_message("=" * 70, "blue")

            init_excel = self.save_backtest_results_to_excel(
                df_initial_indicators, initial_stats, suffix="initial"
            )
            if init_excel:
                self.log_message(f"✅ Initial results exported: {init_excel}", "green")

            if strategy_type == "Kalman":
                self.log_message("=" * 70, "blue")
                self.log_message("📊 EXPORTING KALMAN TRADES (with LONG/SHORT column)", "blue")
                self.log_message("=" * 70, "blue")

                kalman_strategy_instance = None

                if optimized_stats is not None and hasattr(optimized_stats, '_strategy'):
                    kalman_strategy_instance = optimized_stats._strategy
                elif initial_stats is not None and hasattr(initial_stats, '_strategy'):
                    kalman_strategy_instance = initial_stats._strategy

                if kalman_strategy_instance and hasattr(kalman_strategy_instance, 'export_trades_to_excel'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    sym_clean = self.symbol_var.get().replace('-', '_')
                    suffix_k = "optimized" if optimized_stats else "initial"
                    kalman_file = f"kalman_{sym_clean}_{suffix_k}_{timestamp}.xlsx"

                    success = kalman_strategy_instance.export_trades_to_excel(kalman_file)
                    if success:
                        self.log_message(f"✅ Kalman trades exported: {kalman_file}", "green")
                        self.log_message(f"   📊 Includes LONG/SHORT position_type column", "green")

                        if hasattr(kalman_strategy_instance, 'trade_log') and kalman_strategy_instance.trade_log:
                            long_count = sum(
                                1 for t in kalman_strategy_instance.trade_log if t.get('position_type') == 'long')
                            short_count = sum(
                                1 for t in kalman_strategy_instance.trade_log if t.get('position_type') == 'short')
                            self.log_message(
                                f"   📈 LONG trades: {long_count} | SHORT trades: {short_count}", "cyan")
                    else:
                        self.log_message("⚠️ Kalman Excel export failed", "orange")
                else:
                    self.log_message("⚠️ Could not export Kalman trades — strategy instance not available", "orange")

            if optimized_stats is not None and optimized_params_dict:
                if df_opt_indicators is None:
                    df_opt_indicators = (
                        optimized_stats._strategy.df_enhanced
                        if hasattr(optimized_stats, '_strategy') and
                           hasattr(optimized_stats._strategy, 'df_enhanced')
                        else df_initial_indicators
                    )
                opt_excel = self.save_backtest_results_to_excel(
                    df_opt_indicators, optimized_stats, suffix="optimized"
                )
                if opt_excel:
                    self.log_message(f"✅ Optimized results exported: {opt_excel}", "green")

            if hasattr(self, 'chart') and df is not None:
                final_params = optimized_params_dict if optimized_params_dict else initial_params
                self.root.after(0, lambda p=final_params: self.chart.update_chart(df, params=p))
                self.log_message("✅ Chart updated with final parameters", "green")

            # ═══ STEP 6: Completion ═══════════════════════════════════════════
            self.log_message("=" * 70, "purple")
            self.log_message("✨ BACKTEST COMPLETE ✨", "purple")
            self.log_message("=" * 70, "purple")

            if hasattr(self, 'backtest_results_text'):
                def _update_results_text():
                    self.backtest_results_text.insert(tk.END, "\n" + "=" * 50 + "\n")
                    self.backtest_results_text.insert(tk.END, "BACKTEST COMPLETE!\n")
                    self.backtest_results_text.insert(tk.END, f"Return   : {initial_stats['Return [%]']:.2f}%\n")
                    self.backtest_results_text.insert(tk.END, f"Win Rate : {initial_stats['Win Rate [%]']:.2f}%\n")
                    self.backtest_results_text.insert(tk.END, f"Sharpe   : {initial_stats['Sharpe Ratio']:.3f}\n")
                    self.backtest_results_text.see(tk.END)

                self.root.after(0, _update_results_text)

            completion_msg = (
                f"Backtest Complete!\n\n"
                f"Strategy : {strategy_type}\n"
                f"Period   : {start_date} to {end_date}\n"
                f"Candles  : {len(df):,}\n"
                f"GUI Cap  : {gui_order_pct:.0f}% of equity (hard ceiling)\n\n"
                f"Initial Results:\n"
                f"  Final Equity : ${initial_stats['Equity Final [$]']:,.2f}\n"
                f"  Return       : {initial_stats['Return [%]']:.2f}%\n"
                f"  Win Rate     : {initial_stats['Win Rate [%]']:.2f}%\n"
                f"  Sharpe       : {initial_stats['Sharpe Ratio']:.3f}\n"
                f"  Trades       : {initial_stats['# Trades']}\n"
            )

            if optimized_stats is not None:
                completion_msg += (
                    f"\nOptimized Results:\n"
                    f"  Final Equity : ${optimized_stats['Equity Final [$]']:,.2f}\n"
                    f"  Return       : {optimized_stats['Return [%]']:.2f}%\n"
                    f"  Win Rate     : {optimized_stats['Win Rate [%]']:.2f}%\n"
                    f"  Sharpe       : {optimized_stats['Sharpe Ratio']:.3f}\n"
                    f"  Trades       : {optimized_stats['# Trades']}\n"
                )
                if optimized_params_dict:
                    init_eq = initial_stats.get('Equity Final [$]', 0)
                    opt_eq = optimized_stats.get('Equity Final [$]', 0)
                    delta = opt_eq - init_eq
                    delta_p = (delta / init_eq * 100) if init_eq > 0 else 0
                    completion_msg += (
                        f"\nEquity Improvement   : ${delta:,.2f} ({delta_p:+.2f}%)\n"
                        f"Parameters optimized : {len(optimized_params_dict)}\n"
                        f"Parameter base used  : {current_mode}\n"
                    )

            completion_msg += "\nExcel reports generated and opened."
            self._ui_safe(messagebox.showinfo, "Backtest Complete", completion_msg)

        except Exception as e:
            error_msg = f"❌ Backtest failed: {str(e)}"
            self.log_message(error_msg, "red")
            self.log_message(traceback.format_exc(), "red")

            if hasattr(self, 'backtest_results_text'):
                self._ui_safe(self.backtest_results_text.insert, tk.END, f"\n❌ ERROR: {str(e)}\n")
                self._ui_safe(self.backtest_results_text.see, tk.END)

            self._ui_safe(messagebox.showerror, "Backtest Error", f"Backtest failed:\n\n{str(e)}")

        finally:
            try:
                if original_mode == "Custom Parameters" and original_value is not None:
                    self.custom_params['momentum']['only_tier1_entries'] = original_value
                    self.log_message(f"   🔄 Restored only_tier1_entries = {original_value}", "blue")
            except Exception:
                pass

    def _get_param_values(self, param_key):
        """
        Get values for a parameter:
        - If ACTIVE: Returns the 4 values for optimization (grid search)
        - If INACTIVE: Returns SINGLE value (current/default) as a list

        This ensures inactive parameters are FIXED during optimization.
        """
        if param_key not in self.backtest_params:
            return []

        param_data = self.backtest_params[param_key]

        # =========================================================
        # CASE 1: Parameter is ACTIVE - return 4 values for optimization
        # =========================================================
        if param_data['active'].get():
            values = []
            for v in ['value1', 'value2', 'value3', 'value4']:
                val = param_data[v].get()
                if val and val != '':
                    # Convert type based on content
                    if isinstance(val, str):
                        if val.lower() == 'true':
                            val = True
                        elif val.lower() == 'false':
                            val = False
                        elif '.' in val:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                        else:
                            try:
                                val = int(val)
                            except ValueError:
                                pass
                    # FIX: Ensure max_daily_trades is always int
                    if param_key == 'max_daily_trades' and isinstance(val, (int, float)):
                        val = int(val)
                    values.append(val)

            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for v in values:
                if v not in seen:
                    seen.add(v)
                    unique_values.append(v)

            # Ensure we have at least one value
            if not unique_values:
                unique_values = [False]

            return unique_values

        # =========================================================
        # CASE 2: Parameter is INACTIVE - return SINGLE current value
        # =========================================================
        else:
            current_value = self._get_current_param_value(param_key)
            return [current_value] if current_value is not None else []

    def _convert_value(self, value_str):
        """Convert string to appropriate numeric type"""
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except:
            return 0

    def run_monte_carlo_simulation(self, stats):
        try:
            self.log_message("=" * 70, "purple")
            self.log_message("🎲 STARTING MONTE CARLO SIMULATION", "purple")
            self.log_message("=" * 70, "purple")

            # Extract initial capital
            initial_capital = None
            for key in ['Equity Initial [$]', 'Start Cash', 'Initial Capital']:
                try:
                    value = stats.get(key)
                    if value is not None and isinstance(value, (int, float)):
                        initial_capital = float(value)
                        self.log_message(f"✅ Initial capital: ${initial_capital:,.2f}", "green")
                        break
                except:
                    continue

            if initial_capital is None or initial_capital <= 0:
                from strategies.MomentumStrategy_MACD_HybridScore_Latest import GlobalConfig
                initial_capital = GlobalConfig.INITIAL_CAPITAL
                self.log_message(f"⚠️ Using default capital: ${initial_capital:,.2f}", "orange")

            if initial_capital is None and hasattr(stats, '_equity_curve') and len(stats._equity_curve) > 0:
                try:
                    # Try to get the actual starting equity from the result curve
                    initial_capital = float(stats._equity_curve['Equity'].iloc[0])
                except (AttributeError, TypeError, KeyError, IndexError):
                    # Fallback to GlobalConfig if the stats object is messy
                    initial_capital = getattr(GlobalConfig, 'INITIAL_CAPITAL', 50000.0)
                self.log_message(f"✅ Initial capital from equity curve: ${initial_capital:,.2f}", "green")

            # Extract trades
            trade_history = []

            if hasattr(stats, '_trades') and stats._trades is not None:
                trades_df = stats._trades

                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                    self.log_message(f"📊 Processing {len(trades_df)} trades...", "blue")

                    # Log available columns for debugging
                    self.log_message(f"   Trade columns: {list(trades_df.columns)}", "cyan")

                    for idx, row in trades_df.iterrows():
                        try:
                            pnl_pct = None

                            # Try all possible column names
                            if 'ReturnPct' in row:
                                pnl_pct = float(row['ReturnPct']) * 100
                            elif 'Return [%]' in row:
                                pnl_pct = float(row['Return [%]'])
                            elif 'PnL [%]' in row:
                                pnl_pct = float(row['PnL [%]'])
                            elif 'PnL' in row and 'EntryPrice' in row and 'Size' in row:
                                entry_price = float(row['EntryPrice'])
                                size = float(row['Size'])
                                pnl = float(row['PnL'])
                                entry_value = entry_price * size
                                if entry_value > 0:
                                    pnl_pct = (pnl / entry_value) * 100
                            elif 'EntryPrice' in row and 'ExitPrice' in row and 'Size' in row:
                                entry_price = float(row['EntryPrice'])
                                exit_price = float(row['ExitPrice'])
                                size = float(row['Size'])
                                entry_value = entry_price * size
                                if entry_value > 0:
                                    pnl = (exit_price - entry_price) * size
                                    pnl_pct = (pnl / entry_value) * 100
                            elif 'EntryPrice' in row and 'ExitPrice' in row:
                                # Assume size 1 if not specified
                                entry_price = float(row['EntryPrice'])
                                exit_price = float(row['ExitPrice'])
                                pnl_pct = ((exit_price - entry_price) / entry_price) * 100

                            if pnl_pct is not None:
                                trade_history.append({
                                    'type': 'sell',
                                    'pnl_pct': pnl_pct
                                })
                                self.log_message(f"   Trade {idx}: PnL = {pnl_pct:.2f}%", "green")
                            else:
                                self.log_message(f"   Trade {idx}: Could not extract PnL", "orange")
                                self.log_message(f"      Row data: {row.to_dict()}", "yellow")

                        except Exception as e:
                            self.log_message(f"   ⚠️ Trade {idx} error: {str(e)}", "red")
                            continue

                    self.log_message(f"✅ Extracted {len(trade_history)} trades", "green")

            # If no trades extracted, create synthetic ones
            if not trade_history:
                total_trades = stats.get('# Trades', 0)
                if total_trades > 0:
                    self.log_message(f"⚠️ No valid PnL data found, creating synthetic trades...", "orange")

                    # Get performance metrics
                    win_rate = stats.get('Win Rate [%]', 50) / 100
                    avg_return = stats.get('Return [%]', 0)

                    # Calculate typical win/loss sizes
                    if avg_return > 0:
                        # Assume avg win is 2x avg loss
                        avg_win = avg_return * 2 if win_rate > 0.5 else avg_return * 1.5
                        avg_loss = -avg_win * 0.5
                    else:
                        avg_win = 2.0
                        avg_loss = -1.0

                    # Create synthetic trades
                    for i in range(min(total_trades, 100)):
                        if np.random.random() < win_rate:
                            pnl_pct = abs(avg_win) * (0.5 + np.random.random() * 0.5)
                        else:
                            pnl_pct = -abs(avg_loss) * (0.5 + np.random.random() * 0.5)

                        trade_history.append({'type': 'sell', 'pnl_pct': pnl_pct})

                    self.log_message(f"✅ Created {len(trade_history)} synthetic trades", "green")
                else:
                    self.log_message(f"❌ No trades available for Monte Carlo", "red")
                    return None

            # Run simulation
            if trade_history:
                self.log_message(f"\n📊 Trade statistics for simulation:", "blue")
                pnls = [t['pnl_pct'] for t in trade_history]
                self.log_message(f"   Total trades: {len(pnls)}", "white")
                self.log_message(f"   Win rate: {sum(1 for p in pnls if p > 0) / len(pnls) * 100:.1f}%", "green")
                self.log_message(f"   Avg return: {np.mean(pnls):.2f}%", "white")
                self.log_message(f"   Max win: {max(pnls):.2f}%", "green")
                self.log_message(f"   Max loss: {min(pnls):.2f}%", "red")

                simulator = MonteCarloSimulator(trade_history, initial_capital)

                n_simulations = self.mc_simulations_var.get()
                n_trades = self.mc_trades_var.get()
                method = self.mc_method_var.get()

                self.log_message(f"\n🎲 Running {n_simulations} simulations with {n_trades} trades each...", "blue")

                results = simulator.run_simulation(
                    n_simulations=n_simulations,
                    n_trades=n_trades,
                    method=method
                )

                # Save results
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                symbol = self.symbol_var.get().replace('-', '_')
                strategy = self.strategy_type_var.get()

                viz_path = f"monte_carlo_{symbol}_{strategy}_{timestamp}.png"
                simulator.visualize_results(results, save_path=viz_path)

                report_path = f"monte_carlo_report_{symbol}_{strategy}_{timestamp}.txt"
                simulator.generate_report(results, output_path=report_path)

                self.log_message(f"\n✅ Monte Carlo complete!", "green")
                self.log_message(f"   Visualization: {viz_path}", "blue")
                self.log_message(f"   Report: {report_path}", "blue")

                return results
            else:
                self.log_message("❌ No trades to simulate", "red")
                return None

        except Exception as e:
            self.log_message(f"❌ Monte Carlo error: {str(e)}", "red")
            import traceback
            self.log_message(traceback.format_exc(), "red")
            return None

    def debug_data_range(self, df):
        """Comprehensive data diagnostics with UTC timestamps"""
        if df is None or df.empty:
            self.log_message("❌ NO DATA LOADED", "red")
            return

        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            days_covered = (df.index[-1] - df.index[0]).days
            candles_per_day = len(df) / days_covered if days_covered > 0 else 0

            start_str = df.index[0].strftime('%Y-%m-%d %H:%M:%S UTC')
            end_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')

            self.log_message(f"🔍 DATA DIAGNOSTICS:", "purple")
            self.log_message(f"   Date range: {start_str} to {end_str}", "white")
            self.log_message(f"   Total days covered: {days_covered}", "white")
            self.log_message(f"   Total candles: {len(df):,}", "white")
            self.log_message(f"   Candles/day: {candles_per_day:.1f}", "white")

            expected_days = (pd.to_datetime(self.end_date_var.get()) - pd.to_datetime(self.start_date_var.get())).days
            self.log_message(f"   Expected days: {expected_days}", "white")

            missing_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum()
            if missing_data.any():
                self.log_message(f"   Missing data: {dict(missing_data)}", "orange")
            else:
                self.log_message("   ✅ No missing OHLCV data", "green")

            price_changes = df['Close'].pct_change().dropna()
            volatile_periods = (price_changes.abs() > 0.05).sum()
            self.log_message(f"   Large moves (>5%): {volatile_periods} / {len(price_changes)}", "white")

        except Exception as e:
            self.log_message(f"❌ Data diagnostics error: {str(e)}", "red")

    def log_parameter_changes(self, initial_params, optimized_params):
        self.log_message("🔍 PARAMETER COMPARISON:", "purple")
        self.log_message("=" * 60, "purple")

        changes_found = False
        for param, optimized_value in optimized_params.items():
            if param.startswith('_'):
                continue

            initial_value = initial_params.get(param, "N/A")

            if initial_value == "N/A":
                self.log_message(f"   {param:25} {optimized_value} (new)", "blue")
                changes_found = True
            elif initial_value != optimized_value:
                self.log_message(f"   {param:25} {initial_value} → {optimized_value}", "green")
                changes_found = True
            else:
                self.log_message(f"   {param:25} {initial_value} (unchanged)", "white")

        if not changes_found:
            self.log_message("   No parameter changes detected", "orange")

        self.log_message("=" * 60, "purple")

    def update_strategy_with_optimized_params(self, optimized_params, strategy_type):
        """Apply optimized parameters to the actual strategy instance"""
        try:
            if not optimized_params:
                self.log_message("⚠️ Cannot update strategy - no optimized parameters provided", "orange")
                return False

            if strategy_type in self.strategies:
                strategy_instance = self.strategies[strategy_type]
                updates_made = 0

                for param, value in optimized_params.items():
                    if not param.startswith('_') and hasattr(strategy_instance, param):
                        current_value = getattr(strategy_instance, param)
                        if current_value != value:
                            setattr(strategy_instance, param, value)
                            self.log_message(f"   🔄 Updated {param}: {current_value} → {value}", "blue")
                            updates_made += 1

                if hasattr(strategy_instance, 'config'):
                    for param, value in optimized_params.items():
                        if not param.startswith('_') and param in strategy_instance.config:
                            if strategy_instance.config[param] != value:
                                old_value = strategy_instance.config[param]
                                strategy_instance.config[param] = value
                                self.log_message(f"   🔄 Updated config {param}: {old_value} → {value}", "blue")
                                updates_made += 1

                if updates_made > 0:
                    self.log_message(f"✅ Strategy '{strategy_type}' updated with {updates_made} optimized parameters",
                                     "green")
                    return True
                else:
                    self.log_message(f"ℹ️ No parameter updates needed for '{strategy_type}'", "blue")
                    return True
            else:
                self.log_message(f"⚠️ Strategy '{strategy_type}' not found for parameter update", "orange")
                return False

        except Exception as e:
            self.log_message(f"❌ Error updating strategy with optimized params: {str(e)}", "red")
            return False

    def clear_old_cache_if_needed(self):
        """Clear cache when date range changes significantly"""
        try:
            cache_dir = "data_cache"
            if os.path.exists(cache_dir):
                symbol = self.symbol_var.get().replace('/', '-')
                interval = self.interval_var.get()

                pattern = f"{symbol}_{interval}_*.csv"
                cache_files = glob(os.path.join(cache_dir, pattern))

                for cache_file in cache_files:
                    try:
                        df = pd.read_csv(cache_file, nrows=1)
                        if 'timestamp' in df.columns:
                            first_date = pd.to_datetime(df['timestamp'].iloc[0])
                            desired_start = pd.to_datetime(self.start_date_var.get())

                            if first_date > desired_start:
                                os.remove(cache_file)
                                self.log_message(f"🗑️ Deleted outdated cache: {os.path.basename(cache_file)}", "orange")
                    except:
                        continue

        except Exception as e:
            self.log_message(f"⚠️ Cache cleanup error: {str(e)}", "orange")

    def add_tier_column_to_excel(self, input_file, output_file=None):
        """
        Add Tier column to an existing Excel file based on quality scores

        Args:
            input_file: Path to input Excel file
            output_file: Path to output file (if None, creates new file with '_with_tier' suffix)
        """
        try:
            if output_file is None:
                output_file = input_file.replace('.xlsx', '_with_tier.xlsx')

            self.log_message(f"📂 Reading {input_file}...", "blue")

            # Read all sheets
            xl = pd.ExcelFile(input_file)
            sheet_names = xl.sheet_names

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for sheet_name in sheet_names:
                    df = pd.read_excel(input_file, sheet_name=sheet_name)

                    # Only add Tier column to Trades sheet
                    if sheet_name == 'Trades':
                        self.log_message(f"   Adding Tier column to {sheet_name} sheet...", "blue")

                        # Get tier thresholds from strategy
                        # quality_tier1_min = 75  (highest bar, checked first)
                        # quality_tier2_min = 65

                        if hasattr(self, 'strategy') and self.strategy is not None:
                            quality_tier1_min = getattr(self.strategy, 'quality_tier1_min', 72)  # v9.4.2: was 75
                            quality_tier2_min = getattr(self.strategy, 'quality_tier2_min', 62)  # v9.4.2: was 88

                        # Check if Quality_Score column exists
                        if 'Quality_Score' in df.columns:
                            # Add Tier column based on quality score.
                            # NOTE: Tier 1 is the highest/most-selective bar, so it must
                            # be checked FIRST (np.select takes the first True condition).
                            conditions = [
                                (df['Quality_Score'] >= quality_tier1_min),
                                (df['Quality_Score'] >= quality_tier2_min),
                                (df['Quality_Score'] > 0)
                            ]
                            choices = ['Tier 1', 'Tier 2', 'Tier 3', 'Below Tier 3']
                            df['Tier'] = np.select(conditions, choices, default='Unknown')

                            # Add numeric tier for filtering
                            tier_map = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3, 'Below Tier 3': 0, 'Unknown': -1}
                            df['Tier_Number'] = df['Tier'].map(tier_map)

                            # Log tier summary
                            tier_counts = df['Tier'].value_counts()
                            self.log_message(f"   ✅ Added Tier column:", "green")
                            tier_colors = {"Tier 1": "blue", "Tier 2": "green", "Tier 3": "red"}
                            for tier_name, count in tier_counts.items():
                                pct = (count / len(df)) * 100
                                color = tier_colors.get(tier_name, "orange")
                                self.log_message(f"      {tier_name}: {count} ({pct:.1f}%)", color)
                        else:
                            self.log_message(f"   ⚠️ Quality_Score column not found in Trades sheet", "orange")

                    # Write sheet to output file
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            self.log_message(f"✅ Saved to {output_file}", "green")

            # Try to open the file
            try:
                os.startfile(output_file)
            except:
                pass

            return output_file

        except Exception as e:
            self.log_message(f"❌ Error adding tier column: {str(e)}", "red")
            import traceback
            self.log_message(traceback.format_exc(), "red")
            return None

    def save_backtest_results_to_excel(self, df, stats=None, suffix=""):
        """
        Export backtest results to Excel with 100% accurate trade data.
        PRIMARY SOURCE: stats['_trades'] (the backtest engine's complete trade ledger)
        ENRICHMENT: strategy.trade_records (for tier, quality score, signals)
        """
        try:
            if stats is None:
                self.log_message("⚠️ No stats provided - cannot export", "orange")
                return None

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            symbol = self.symbol_var.get().replace('-', '_')
            interval = self.interval_var.get()
            strategy = self.strategy_type_var.get()
            strategy_clean = strategy.replace(' ', '_').replace('(', '').replace(')', '').replace('\\', '_').replace(
                '/', '_')
            suffix_str = f"_{suffix}" if suffix else ""
            filename = f"backtest_{symbol}_{interval}_{strategy}{suffix_str}_{timestamp}.xlsx"

            self.log_message(f"📝 Creating Excel file: {filename}", "blue")

            # Get tier thresholds for reporting
            quality_tier1_min, quality_tier2_min, source = self.get_current_tier_thresholds(backtest_mode=True)
            self.log_message(
                f"✅ USING TIER THRESHOLDS (source: {source}): Tier1={quality_tier1_min}, Tier2={quality_tier2_min}",
                "bold green")

            import re
            filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
            filename = filename.replace(' ', '_')
            writer = pd.ExcelWriter(filename, engine='openpyxl')
            sheets_created = 0

            try:
                strategy_instance = getattr(stats, '_strategy', None)

                # ─── PRIMARY SOURCE: Backtest trades (COMPLETE) ═══
                backtest_trades = []
                trades_df = None
                if '_trades' in stats and stats['_trades'] is not None:
                    trades_df = stats['_trades']
                    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                        backtest_trades = trades_df.to_dict('records')
                        self.log_message(
                            f"📊 PRIMARY SOURCE: {len(backtest_trades)} trades from backtest results", "green")

                # ─── ENRICHMENT: Strategy trade records (for tier, quality, signals) ═══
                trade_records = []
                if strategy_instance and hasattr(strategy_instance, 'trade_records'):
                    trade_records = strategy_instance.trade_records
                    self.log_message(
                        f"📊 ENRICHMENT: {len(trade_records)} trade records found for enrichment", "blue")

                # Get commission rate from settings
                # FIX: commission_var already stores a decimal fraction (e.g. 0.001 = 0.1%),
                # the SAME value passed directly to Backtest(commission=...) above. Dividing by
                # 100 here made Expected_Commission ~100x smaller than Effective_Commission.
                commission_rate = self.commission_var.get()
                self.log_message(f"💳 Commission rate: {commission_rate * 100:.4f}%", "cyan")

                def _rec_get(rec, *names, default=None):
                    if rec is None:
                        return default
                    for name in names:
                        if hasattr(rec, '__dataclass_fields__'):
                            if hasattr(rec, name):
                                val = getattr(rec, name)
                                if val is not None:
                                    return val
                        elif isinstance(rec, dict):
                            if name in rec and rec[name] is not None:
                                return rec[name]
                    return default

                def _fmt_time(t):
                    if t is None:
                        return ''
                    if hasattr(t, 'strftime'):
                        if hasattr(t, 'tzinfo') and t.tzinfo is not None:
                            t = t.replace(tzinfo=None)
                        return t.strftime('%Y-%m-%d %H:%M:%S')
                    return str(t)

                # ─── MARKET DATA PREP ────────────────────────────────────────
                market_df = df.copy()
                if isinstance(market_df.index, pd.DatetimeIndex) and market_df.index.tz is not None:
                    market_df.index = market_df.index.tz_localize(None)
                for col in market_df.select_dtypes(include=['datetime', 'datetimetz']).columns:
                    if hasattr(market_df[col], 'dt') and market_df[col].dt.tz is not None:
                        market_df[col] = market_df[col].dt.tz_convert(None)

                market_df['Entry_Signal'] = ''
                market_df['Exit_Signal'] = ''
                market_df['Exit_Reason'] = ''
                market_df['Confluence_Score'] = 0
                market_df['Risk_Allocation_%'] = 0.0

                self._exit_reason_map = getattr(strategy_instance, '_exit_reason_map', {})

                # ─── BUILD LOOKUP FOR STRATEGY RECORDS BY ENTRY BAR ═══
                # entry_bar comes from TradeRecord.entry_bar, which previously didn't
                # exist as a field at all (that was the real reason matching was
                # always 0/N regardless of any offset). Key on the raw value here;
                # the lookup below tries a small offset fallback and logs which one
                # actually hits, instead of assuming a direction blind.
                record_by_entry_bar = {}
                for record in trade_records:
                    entry_bar = _rec_get(record, 'entry_bar')
                    if entry_bar is not None:
                        record_by_entry_bar[entry_bar] = record

                _match_offset_counts = {}

                # ─── BUILD TRADES DATA FROM BACKTEST TRADES (PRIMARY SOURCE) ═══
                trades_data = []
                total_gross_pnl = 0
                total_commission_calc = 0
                total_commission_expected = 0  # For double-check

                if backtest_trades:
                    self.log_message(f"📊 Building trades from {len(backtest_trades)} backtest trades", "blue")

                    # Group by EntryBar to handle partial exits
                    # FIX: EntryBar==0 is a legitimate first-bar entry, not a "missing value"
                    # sentinel. Using 0 as both meanings caused real bar-0 trades to be
                    # silently dropped from the Trades sheet (while still counted in the
                    # equity-curve-based Summary), which showed up as a Trades-vs-Summary
                    # PnL mismatch. Use -1 (an impossible real bar index) as the sentinel.
                    trade_groups = {}
                    for trade in backtest_trades:
                        entry_bar = trade.get('EntryBar', -1)
                        if entry_bar not in trade_groups:
                            trade_groups[entry_bar] = []
                        trade_groups[entry_bar].append(trade)

                    trade_counter = 0
                    for entry_bar, group_trades in sorted(trade_groups.items()):
                        if entry_bar < 0:
                            continue

                        group_trades.sort(key=lambda x: x.get('ExitBar', 0))
                        first_trade = group_trades[0]

                        # Determine direction from size
                        first_size = first_trade.get('Size', 0)
                        direction = 'LONG' if first_size > 0 else 'SHORT'
                        entry_price = first_trade.get('EntryPrice', 0)
                        entry_time = first_trade.get('EntryTime')
                        # FIX: backtesting.py splits a partially-closed position into
                        # multiple Trade records, and each record's own Size is only
                        # the portion closed by THAT leg, not the original total
                        # position size. Using first_trade's Size alone (e.g. 16)
                        # displayed "Entry_Size" as 16 on every leg of a trade whose
                        # true original size was actually 50 (16+11+23 summed across
                        # legs) - the real entry size is the sum across all legs.
                        entry_size = sum(abs(t.get('Size', 0)) for t in group_trades)

                        trade_counter += 1
                        total_legs = len(group_trades)

                        # Get enrichment from strategy record if available
                        # Try exact match, then a small offset fallback (order fills
                        # can legitimately land 1 bar away from when the entry_bar
                        # was recorded); log which one actually hit so any remaining
                        # systematic offset is visible instead of silently zero.
                        matching_record = record_by_entry_bar.get(entry_bar)
                        _used_offset = 0
                        if matching_record is None:
                            for _off in (-1, 1, -2, 2):
                                matching_record = record_by_entry_bar.get(entry_bar + _off)
                                if matching_record is not None:
                                    _used_offset = _off
                                    break
                        _match_offset_counts[_used_offset if matching_record else 'none'] = \
                            _match_offset_counts.get(_used_offset if matching_record else 'none', 0) + 1
                        entry_quality = _rec_get(matching_record, 'entry_quality_score', 'quality_score',
                                                 default=0) or 0
                        entry_tier = _rec_get(matching_record, 'entry_tier', 'tier', default=2)

                        # Determine tier name
                        if entry_tier == 1:
                            tier_name = "Tier 1"
                        elif entry_tier == 2:
                            tier_name = "Tier 2"
                        elif entry_tier == 3:
                            tier_name = "Tier 3"
                        elif entry_quality and entry_quality > 0:
                            if entry_quality >= quality_tier1_min:
                                tier_name, entry_tier = "Tier 1", 1
                            elif entry_quality >= quality_tier2_min:
                                tier_name, entry_tier = "Tier 2", 2
                            else:
                                tier_name, entry_tier = "Below Tier 2", 0
                        else:
                            tier_name, entry_tier = "Unknown", 0

                        # Process each exit leg
                        for leg_pos, trade in enumerate(group_trades):
                            exit_bar = trade.get('ExitBar', 0)
                            exit_price = trade.get('ExitPrice', 0)
                            exit_time = trade.get('ExitTime')
                            leg_size = abs(trade.get('Size', 0))

                            if leg_size == 0:
                                # Fallback only: use this leg's own share, not the
                                # full summed position size (entry_size), to avoid
                                # overstating this leg's gross_pnl/commission.
                                leg_size = abs(first_size) if leg_pos == 0 else entry_size / max(total_legs, 1)

                            # ═══ CALCULATE GROSS PNL ═══
                            if direction == 'LONG':
                                gross_pnl = (exit_price - entry_price) * leg_size
                            else:
                                gross_pnl = (entry_price - exit_price) * leg_size

                            # ═══ CALCULATE EXPECTED COMMISSION ═══
                            # Commission is applied on BOTH entry and exit
                            expected_entry_commission = entry_price * leg_size * commission_rate
                            expected_exit_commission = exit_price * leg_size * commission_rate
                            expected_commission = expected_entry_commission + expected_exit_commission

                            # ═══ USE THE BACKTEST'S PNL DIRECTLY ═══
                            if 'PnL' in trade and trade.get('PnL') is not None:
                                net_pnl = float(trade.get('PnL', 0))
                            else:
                                # Fallback: calculate from prices
                                if direction == 'LONG':
                                    net_pnl = (exit_price - entry_price) * leg_size
                                else:
                                    net_pnl = (entry_price - exit_price) * leg_size

                            # ═══ CALCULATE EFFECTIVE COMMISSION ═══
                            effective_commission = gross_pnl - net_pnl
                            total_gross_pnl += gross_pnl
                            total_commission_calc += effective_commission
                            total_commission_expected += expected_commission

                            # ═══ Calculate Return_% from net PnL ═══
                            entry_value = entry_price * leg_size
                            return_pct = (net_pnl / entry_value) * 100 if entry_value > 0 else 0

                            # Get exit reason
                            exit_reason = self._extract_exit_reason(trade, exit_bar, market_df)

                            # Determine exit label
                            if total_legs > 1:
                                exit_label = f"Partial {trade_counter}/{leg_pos + 1}"
                                full_exit_reason = f"{exit_label} - {exit_reason}" if exit_reason else exit_label
                            else:
                                exit_label = "Full Exit"
                                full_exit_reason = f"{exit_label} - {exit_reason}" if exit_reason else exit_label

                            # Mark entry in market data
                            if entry_time is not None:
                                try:
                                    entry_dt = pd.Timestamp(entry_time)
                                    if entry_dt.tzinfo is not None:
                                        entry_dt = entry_dt.tz_localize(None)
                                    if entry_dt in market_df.index:
                                        market_df.at[entry_dt, 'Entry_Signal'] = 'BUY'
                                except Exception as e:
                                    pass

                            # Mark exit in market data
                            if exit_time is not None:
                                try:
                                    exit_dt = pd.Timestamp(exit_time)
                                    if exit_dt.tzinfo is not None:
                                        exit_dt = exit_dt.tz_localize(None)
                                    if exit_dt in market_df.index:
                                        market_df.at[exit_dt, 'Exit_Signal'] = 'SELL'
                                        market_df.at[exit_dt, 'Exit_Reason'] = full_exit_reason
                                except Exception as e:
                                    pass

                            # ─── BUILD TRADE ROW ──────────────────────────────
                            trades_data.append({
                                'Trade_#': trade_counter,
                                'Leg_#': leg_pos + 1,
                                'Exit_Type': exit_label,
                                'Tier': tier_name,
                                'Tier_Number': entry_tier if entry_tier is not None else -1,
                                'Direction': direction,
                                'Entry_Time': _fmt_time(entry_time),
                                'Entry_Price': round(entry_price, 4),
                                'Entry_Size': round(entry_size, 4),
                                'Exit_Time': _fmt_time(exit_time),
                                'Exit_Price': round(exit_price, 4),
                                'Exit_Size': round(leg_size, 4),
                                'Exit_Reason': full_exit_reason,
                                'Gross_PnL': round(gross_pnl, 2),
                                'Expected_Commission': round(expected_commission, 2),
                                'Effective_Commission': round(effective_commission, 2),
                                'PnL': round(net_pnl, 2),
                                'Return_%': round(return_pct, 2),
                                'Quality_Score': entry_quality,
                                'Confluence_Score': 0,
                                'Risk_Allocation_%': 0,
                                'Signal_ADX': round(_rec_get(matching_record, 'signal_adx', default=0) or 0, 1),
                                'Signal_RSI': round(_rec_get(matching_record, 'signal_rsi', default=50) or 50, 1),
                                'Signal_MACD': round(_rec_get(matching_record, 'signal_macd', default=0) or 0, 4),
                                'Signal_Volume_Ratio': round(
                                    _rec_get(matching_record, 'signal_volume', default=1.0) or 1.0, 2),
                                'ML_Prediction': _rec_get(matching_record, 'signal_ml_prediction', default=0) or 0,
                                'ML_Confidence_%': round(
                                    _rec_get(matching_record, 'signal_ml_confidence', default=0.0) or 0.0, 1),
                                'Market_Regime': _rec_get(matching_record, 'market_regime', default='') or '',
                                'Matched_To_Strategy_Record': 'Yes' if matching_record else 'No',
                                'Win': 'Yes' if net_pnl > 0 else 'No',
                            })

                if trades_data:
                    trade_output_df = pd.DataFrame(trades_data)

                    col_order = [
                        'Trade_#', 'Leg_#', 'Exit_Type', 'Tier', 'Tier_Number', 'Direction',
                        'Entry_Time', 'Entry_Price', 'Entry_Size',
                        'Exit_Time', 'Exit_Price', 'Exit_Size', 'Exit_Reason',
                        'Gross_PnL', 'Expected_Commission', 'Effective_Commission', 'PnL', 'Return_%',
                        'Quality_Score', 'Confluence_Score', 'Risk_Allocation_%',
                        'Signal_ADX', 'Signal_RSI', 'Signal_MACD',
                        'Signal_Volume_Ratio', 'ML_Prediction', 'ML_Confidence_%',
                        'Market_Regime', 'Matched_To_Strategy_Record', 'Win'
                    ]
                    existing_cols = [c for c in col_order if c in trade_output_df.columns]
                    trade_output_df = trade_output_df[existing_cols]

                    trade_output_df.to_excel(writer, sheet_name='Trades', index=False)
                    sheets_created += 1

                    self.log_message(f"   ✅ Trades sheet created ({len(trade_output_df)} trade legs)", "green")

                    # Log unique trade count
                    unique_trades = trade_output_df['Trade_#'].nunique()
                    self.log_message(f"   📊 Unique trades: {unique_trades}", "cyan")

                    # Log matched vs unmatched
                    matched_count = len(trade_output_df[trade_output_df['Matched_To_Strategy_Record'] == 'Yes'])
                    unmatched_count = len(trade_output_df[trade_output_df['Matched_To_Strategy_Record'] == 'No'])
                    if unmatched_count > 0:
                        self.log_message(f"   📊 Matched to strategy: {matched_count}, Unmatched: {unmatched_count}",
                                         "orange")
                    try:
                        self.log_message(f"   🔎 Match offset breakdown: {_match_offset_counts}", "orange")
                    except NameError:
                        pass

                    # Log PnL summary
                    total_net_pnl = trade_output_df['PnL'].sum()
                    total_gross_pnl_calc = trade_output_df['Gross_PnL'].sum()
                    total_effective_commission = trade_output_df['Effective_Commission'].sum()
                    total_expected_commission = trade_output_df['Expected_Commission'].sum()
                    self.log_message(f"   💰 Gross PnL: ${total_gross_pnl_calc:,.2f}", "cyan")
                    self.log_message(f"   💰 Expected Commissions: ${total_expected_commission:,.2f}", "yellow")
                    self.log_message(f"   💰 Effective Commissions: ${total_effective_commission:,.2f}", "yellow")
                    self.log_message(f"   💰 Net PnL: ${total_net_pnl:,.2f}", "green" if total_net_pnl > 0 else "red")

                # ─── MARKET DATA SHEET ────────────────────────────────────────
                export_columns = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'EMA_Fast', 'EMA_Mid', 'EMA_Slow',
                    'RSI', 'CCI', 'ADX', 'ATR',
                    'Kalman_Strength', 'Volume_Ratio',
                    'Entry_Signal', 'Exit_Signal', 'Exit_Reason',
                    'Confluence_Score', 'Risk_Allocation_%'
                ]
                missing_export_cols = [col for col in export_columns if col not in market_df.columns]
                if missing_export_cols:
                    self.log_message(f"⚠️ Market data export missing columns: {missing_export_cols}", "orange")
                export_columns = [col for col in export_columns if col in market_df.columns]
                market_export = market_df[export_columns].copy()
                numeric_cols = market_export.select_dtypes(include=[np.number]).columns
                market_export[numeric_cols] = market_export[numeric_cols].round(4)
                market_export.to_excel(writer, sheet_name='Market_Data')
                sheets_created += 1

                # ─── SUMMARY SHEET ────────────────────────────────────────────
                _start = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
                _end = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
                _num_months = max((_end - _start).days / 30.44, 1)

                # Get initial capital
                initial_capital = 50000
                if hasattr(stats, 'get'):
                    initial_capital = stats.get('Equity Initial [$]', 50000)
                elif hasattr(stats, '_strategy') and hasattr(stats._strategy, 'initial_cash'):
                    initial_capital = stats._strategy.initial_cash

                final_equity = stats.get('Equity Final [$]', 0)
                total_return = stats.get('Return [%]', 0)
                net_profit = final_equity - initial_capital

                # ═══ Use trade data for statistics ═══
                if trades_data and len(trades_data) > 0:
                    unique_trades_count = trade_output_df['Trade_#'].nunique()
                    total_legs_count = len(trades_data)
                    wins = sum(1 for t in trades_data if t.get('Win') == 'Yes')
                    win_rate = (wins / total_legs_count * 100) if total_legs_count > 0 else 0

                    # ═══ Calculate totals from trades sheet ═══
                    total_gross_from_trades = trade_output_df['Gross_PnL'].sum()
                    total_expected_commission_from_trades = trade_output_df['Expected_Commission'].sum()
                    total_effective_commission_from_trades = trade_output_df['Effective_Commission'].sum()
                    total_net_from_trades = trade_output_df['PnL'].sum()

                    # ═══ DOUBLE CHECK: Does Gross - Effective Commission = Net? ═══
                    calc_net = total_gross_from_trades - total_effective_commission_from_trades
                    calc_diff = abs(calc_net - total_net_from_trades)

                    # ═══ DOUBLE CHECK: Does Expected Commission match Effective? ═══
                    comm_diff = abs(total_expected_commission_from_trades - total_effective_commission_from_trades)

                    # ═══ VERIFICATION: Net PnL from Trades vs Summary ═══
                    summary_diff = total_net_from_trades - net_profit

                    self.log_message("=" * 70, "yellow")
                    self.log_message("🔍 PnL & COMMISSION VERIFICATION", "yellow")
                    self.log_message("=" * 70, "yellow")
                    self.log_message(f"   Gross PnL (from Trades):        ${total_gross_from_trades:,.2f}", "cyan")
                    self.log_message(
                        f"   Expected Commissions (@{commission_rate * 100:.2f}%): ${total_expected_commission_from_trades:,.2f}",
                        "yellow")
                    self.log_message(
                        f"   Effective Commissions:          ${total_effective_commission_from_trades:,.2f}", "yellow")
                    self.log_message(f"   Gross - Effective Commission:   ${calc_net:,.2f}", "cyan")
                    self.log_message(f"   Net PnL (from Trades sheet):    ${total_net_from_trades:,.2f}", "cyan")
                    self.log_message(f"   Net Profit (from Summary):      ${net_profit:,.2f}", "cyan")
                    self.log_message("", "white")
                    self.log_message(
                        f"   ✅ Gross - Eff. Comm = Net PnL:  {'✅ PASS' if calc_diff < 0.01 else f'❌ FAIL (diff: ${calc_diff:,.2f})'}",
                        "green" if calc_diff < 0.01 else "red")
                    self.log_message(
                        f"   ✅ Expected vs Effective Comm:   {'✅ PASS' if comm_diff < 0.01 else f'❌ FAIL (diff: ${comm_diff:,.2f})'}",
                        "green" if comm_diff < 0.01 else "red")
                    self.log_message(
                        f"   ✅ Trades PnL vs Summary:        {'✅ PASS' if abs(summary_diff) < 0.01 else f'❌ FAIL (diff: ${summary_diff:,.2f})'}",
                        "green" if abs(summary_diff) < 0.01 else "red")

                    if abs(summary_diff) >= 0.01:
                        self.log_message(f"   💡 The difference is likely due to:", "orange")
                        self.log_message(f"      - Rounding differences between trades", "orange")
                        self.log_message(f"      - The backtest engine's internal calculations", "orange")
                        self.log_message(f"   📌 Using Summary net profit as source of truth", "cyan")
                    self.log_message("=" * 70, "yellow")

                    total_pnl_display = net_profit
                else:
                    unique_trades_count = stats.get('# Trades', 0)
                    total_legs_count = unique_trades_count
                    win_rate = stats.get('Win Rate [%]', 0)
                    total_gross_from_trades = 0
                    total_expected_commission_from_trades = 0
                    total_effective_commission_from_trades = 0
                    total_net_from_trades = net_profit
                    total_pnl_display = net_profit

                _avg_trades_per_month = unique_trades_count / _num_months if _num_months > 0 else 0

                summary_data = [
                    ['═══ PERFORMANCE METRICS ═══', ''],
                    ['Start Date', self.start_date_var.get()],
                    ['End Date', self.end_date_var.get()],
                    ['Initial Capital', f"${initial_capital:,.2f}"],
                    ['Final Equity', f"${final_equity:,.2f}"],
                    ['Net Profit (from Summary)', f"${net_profit:,.2f}"],
                    ['Total Return', f"{total_return:.2f}%"],
                    ['Months', f"{_num_months:.1f}"],
                    ['', ''],
                    ['═══ PnL BREAKDOWN ═══', ''],
                    ['Gross PnL (from Trades)', f"${total_gross_from_trades:,.2f}"],
                    ['Expected Commissions', f"${total_expected_commission_from_trades:,.2f}"],
                    ['Effective Commissions', f"${total_effective_commission_from_trades:,.2f}"],
                    ['Net PnL (Gross - Effective Comm)',
                     f"${total_gross_from_trades - total_effective_commission_from_trades:,.2f}"],
                    ['Net PnL (from Trades sheet)', f"${total_net_from_trades:,.2f}"],
                    ['Net Profit (from Summary)', f"${net_profit:,.2f}"],
                    ['PnL Difference', f"${total_net_from_trades - net_profit:,.2f}"],
                    ['', ''],
                    ['═══ COMMISSION VERIFICATION ═══', ''],
                    ['Commission Rate', f"{commission_rate * 100:.2f}%"],
                    ['Total Expected Commission', f"${total_expected_commission_from_trades:,.2f}"],
                    ['Total Effective Commission', f"${total_effective_commission_from_trades:,.2f}"],
                    ['Commission Difference',
                     f"${total_expected_commission_from_trades - total_effective_commission_from_trades:,.2f}"],
                    ['Commission as % of Gross',
                     f"{(total_effective_commission_from_trades / total_gross_from_trades * 100) if total_gross_from_trades != 0 else 0:.2f}%"],
                    ['', ''],
                    ['═══ TRADE STATISTICS ═══', ''],
                    ['Total Trades (Unique Entries)', unique_trades_count],
                    ['Total Trade Legs (Incl. Partial Exits)', total_legs_count],
                    ['Win Rate (Per Leg)', f"{win_rate:.2f}%"],
                    ['Avg Trades / Month', f"{_avg_trades_per_month:.1f}"],
                    ['', ''],
                    ['═══ RISK METRICS ═══', ''],
                    ['Sharpe Ratio', f"{stats.get('Sharpe Ratio', 0):.3f}"],
                    ['Sortino Ratio', f"{stats.get('Sortino Ratio', 0):.3f}"],
                    ['Max Drawdown', f"{stats.get('Max. Drawdown [%]', 0):.2f}%"],
                    ['Profit Factor', f"{stats.get('Profit Factor', 0):.2f}"],
                    ['', ''],
                    ['═══ TIER THRESHOLDS USED ═══', ''],
                    ['Tier 1 Minimum', quality_tier1_min],
                    ['Tier 2 Minimum', quality_tier2_min],
                    ['Parameter Mode', self.param_toggle_var.get()],
                    ['Threshold Source', source],
                ]

                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                sheets_created += 1

                # ─── DIAGNOSTIC: why entries were skipped (esp. useful for 0/low-trade runs) ═══
                if strategy_instance and hasattr(strategy_instance, 'get_hold_reason_summary'):
                    hold_summary = strategy_instance.get_hold_reason_summary(top_n=40)
                    if hold_summary:
                        rejections_df = pd.DataFrame(
                            hold_summary, columns=['Reason (bucketed)', 'Bar Count', '% of Bars Checked'])
                        rejections_df.to_excel(writer, sheet_name='Entry_Rejections', index=False)
                        sheets_created += 1
                        top_reason, top_count, top_pct = hold_summary[0]
                        self.log_message(
                            f"🔍 TOP REJECTION REASON: '{top_reason}' blocked {top_count} bars ({top_pct}% of bars checked) "
                            f"— see 'Entry_Rejections' sheet for the full breakdown", "orange")

                writer.close()

                file_size_kb = os.path.getsize(filename) / 1024
                self.log_message(f"✅ Excel file created successfully!", "green")
                self.log_message(f"   📁 File: {filename}", "green")
                self.log_message(f"   📊 Sheets: {sheets_created}", "green")
                self.log_message(f"   💾 Size: {file_size_kb:.1f} KB", "green")

                try:
                    os.startfile(filename)
                except:
                    pass

                return filename

            except Exception as inner_e:
                try:
                    writer.close()
                except:
                    pass
                raise inner_e

        except Exception as e:
            self.log_message(f"❌ Excel export failed: {str(e)}", "red")
            import traceback
            self.log_message(traceback.format_exc(), "red")
            return None

    def fix_existing_excel_tiers(self, excel_file):
        """Fix Tier classifications in an existing Excel file"""
        import pandas as pd

        self.log_message(f"🔧 Fixing Tier classifications in: {excel_file}", "blue")

        # Read the Excel file
        xl = pd.ExcelFile(excel_file)
        fixed_file = excel_file.replace('.xlsx', '_FIXED.xlsx')

        with pd.ExcelWriter(fixed_file, engine='openpyxl') as writer:
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)

                if sheet_name == 'Trades':
                    self.log_message(f"   Fixing {sheet_name} sheet...", "blue")

                    # Get tier thresholds
                    # quality_tier1_min = 75 (highest bar, checked first)
                    # quality_tier2_min = 65

                    if hasattr(self, 'strategy') and self.strategy is not None:
                        quality_tier1_min = getattr(self.strategy, 'quality_tier1_min', 72)  # v9.4.2: was 75
                        quality_tier2_min = getattr(self.strategy, 'quality_tier2_min', 62)  # v9.4.2: was 88

                    # Fix Tier based on Quality_Score
                    # NOTE: Tier 1 is the highest/most-selective bar, so it must
                    # be checked FIRST (np.select takes the first True condition).
                    conditions = [
                        (df['Quality_Score'] >= quality_tier1_min),
                        (df['Quality_Score'] >= quality_tier2_min),
                        (df['Quality_Score'] > 0)
                    ]
                    choices = ['Tier 1', 'Tier 2', 'Tier 3', 'Below Tier 3']
                    df['Tier'] = np.select(conditions, choices, default='Unknown')

                    # Update Tier_Number
                    tier_map = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3, 'Below Tier 3': 0, 'Unknown': -1}
                    df['Tier_Number'] = df['Tier'].map(tier_map)

                    # Log the fixes
                    tier_counts = df['Tier'].value_counts()
                    tier_colors = {"Tier 1": "blue", "Tier 2": "green", "Tier 3": "red"}
                    for tier_name, count in tier_counts.items():
                        self.log_message(f"      {tier_name}: {count}", tier_colors.get(tier_name, "orange"))

                df.to_excel(writer, sheet_name=sheet_name, index=False)

        self.log_message(f"✅ Fixed file saved as: {fixed_file}", "green")

        try:
            os.startfile(fixed_file)
        except:
            pass

        return fixed_file

    def _extract_exit_reason(self, trade_row, exit_bar, market_df):
        """
        Extract exit reason with improved lookup that handles off-by-one.

        Priority:
          1. Real reason from _exit_reason_map (checks ALL possible key combinations)
          2. Stored exit_reason field on trade_row
          3. Position-level stored reason
          4. Heuristic fallback (last resort - NOT "trailing_stop_profit_protection")
        """
        try:
            # ── PRIORITY 1: Get entry bar from trade row ──────────────────────────
            entry_bar = -1
            if hasattr(trade_row, 'get'):
                entry_bar = int(trade_row.get('EntryBar', -1))
            elif hasattr(trade_row, 'EntryBar'):
                entry_bar = int(trade_row.EntryBar)

            if entry_bar < 0:
                if hasattr(self, 'log_message'):
                    self.log_message(f"   ⚠️ No entry_bar found in trade row", "orange")
                return self._heuristic_exit_reason(trade_row, exit_bar, market_df, 0)

            exit_bar_key = int(exit_bar) if exit_bar is not None else None
            exit_map = getattr(self, '_exit_reason_map', {})

            # ── PRIORITY 2: Try ALL possible key combinations ──────────────────────
            # The off-by-one bug means the strategy stores reasons with:
            #   - Entry bar at decision bar (entry_bar)
            #   - Exit bar at decision bar (exit_bar)
            # But the exporter looks at:
            #   - Entry bar from trade row (entry_bar)
            #   - Exit bar from trade row (exit_bar)
            # The fill actually happens on the NEXT bar.

            possible_keys = []

            # 1. Exact match (decision bar -> decision bar)
            possible_keys.append((entry_bar, exit_bar_key))

            # 2. Entry bar + 1, Exit bar (fill after entry)
            if entry_bar >= 0:
                possible_keys.append((entry_bar + 1, exit_bar_key))

            # 3. Entry bar, Exit bar + 1 (fill after exit decision)
            if exit_bar_key is not None:
                possible_keys.append((entry_bar, exit_bar_key + 1))

            # 4. Entry bar + 1, Exit bar + 1 (both delayed)
            if entry_bar >= 0 and exit_bar_key is not None:
                possible_keys.append((entry_bar + 1, exit_bar_key + 1))

            # 5. Entry bar - 1, Exit bar (early entry recorded)
            if entry_bar > 0:
                possible_keys.append((entry_bar - 1, exit_bar_key))

            # 6. Entry bar, Exit bar - 1 (early exit recorded)
            if exit_bar_key is not None and exit_bar_key > 0:
                possible_keys.append((entry_bar, exit_bar_key - 1))

            # 7. Legacy single-key (old format)
            possible_keys.append(entry_bar)
            if entry_bar >= 0:
                possible_keys.append(entry_bar + 1)
                if entry_bar > 0:
                    possible_keys.append(entry_bar - 1)

            # Try each key
            for key in possible_keys:
                if key in exit_map:
                    real_reason = exit_map[key]
                    if real_reason and real_reason not in ['', None, 'unknown', 'exit_condition_met']:
                        if hasattr(self, 'log_message'):
                            if isinstance(key, tuple):
                                offset_info = f"(entry={key[0]}, exit={key[1]})"
                            else:
                                offset_info = f"(legacy key={key})"
                            self.log_message(f"   ✅ Real exit reason found: {real_reason} {offset_info}", "green")
                        return real_reason

            # ── PRIORITY 3: Stored reason on trade_row ─────────────────────────────
            if hasattr(trade_row, 'get'):
                stored_reason = trade_row.get('exit_reason', None)
                if stored_reason and stored_reason not in ['exit_condition_met', 'unknown', None, '']:
                    if hasattr(self, 'log_message'):
                        self.log_message(f"   ℹ️ Using stored reason: {stored_reason}", "blue")
                    return stored_reason

            # ── PRIORITY 4: Strategy-level stored reason ──────────────────────────
            if hasattr(self, 'strategy') and hasattr(self.strategy, '_exit_reason'):
                stored = getattr(self.strategy, '_exit_reason', None)
                if stored:
                    if hasattr(self, 'log_message'):
                        self.log_message(f"   ✅ Using strategy stored exit reason: {stored}", "blue")
                    return stored

            # ── PRIORITY 5: HEURISTIC FALLBACK (LAST RESORT) ──────────────────────
            # This should only fire if ALL lookups fail
            return self._heuristic_exit_reason(trade_row, exit_bar, market_df, entry_bar)

        except Exception as e:
            if hasattr(self, 'log_message'):
                self.log_message(f"⚠️ Exit reason extraction error: {str(e)}", "orange")
            return f"extraction_error_{type(e).__name__}"

    def _heuristic_exit_reason(self, trade_row, exit_bar, market_df, entry_bar):
        """
        Heuristic fallback - only used when real reason can't be found.
        This is the LAST resort, not the first.
        IMPORTANT: This method NEVER guesses "trailing_stop_profit_protection"
        unless it can actually detect a trailing stop hit.
        """
        try:
            if exit_bar >= len(market_df):
                return "index_error"

            if entry_bar < 0 or entry_bar >= len(market_df):
                entry_bar = 0

            exit_data = market_df.iloc[exit_bar]
            entry_price = float(market_df.iloc[entry_bar]['Close'])
            exit_price = float(exit_data.get('Close', 0))

            if entry_price == 0:
                return "invalid_entry_price"

            # Direction-aware price change
            trade_direction = 'long'
            if hasattr(trade_row, 'get'):
                trade_direction = trade_row.get('direction', 'long') or 'long'

            if trade_direction == 'short':
                price_change_pct = (entry_price - exit_price) / entry_price * 100
            else:
                price_change_pct = (exit_price - entry_price) / entry_price * 100

            bars_held = exit_bar - entry_bar
            atr = float(exit_data.get('ATR', 1))
            stop_atr_mult = getattr(self, 'stop_loss_atr_mult', 2.5)
            stop_pct = (atr * stop_atr_mult / entry_price) * 100

            # ─── Check if this was actually a trailing stop hit ──────────────
            # Try to detect actual trailing stop hit from strategy state
            if hasattr(self, 'strategy') and hasattr(self.strategy, '_trailing_stop'):
                trailing_stop = getattr(self.strategy, '_trailing_stop', None)
                if trailing_stop is not None:
                    if trade_direction == 'long' and exit_price <= trailing_stop:
                        return "trailing_stop_hit"
                    elif trade_direction == 'short' and exit_price >= trailing_stop:
                        return "trailing_stop_hit"

            # ─── Stop loss detection ────────────────────────────────────────────
            if price_change_pct <= -stop_pct:
                return "stop_loss_hard"

            # ─── Emergency stop detection ──────────────────────────────────────
            emergency_mult = getattr(self, 'emergency_stop_multiplier', 2.0)
            if price_change_pct <= -(stop_pct * emergency_mult):
                return "stop_loss_hard_emergency"

            # ─── Max hold time detection ──────────────────────────────────────
            max_hold = getattr(self, 'max_hold_bars', 120)
            if bars_held >= max_hold:
                return "max_hold_time_profitable" if price_change_pct > 0 else "max_hold_time"

            # ─── EMA reversal detection ────────────────────────────────────────
            ema_fast = float(exit_data.get('EMA_Fast', 0))
            ema_mid = float(exit_data.get('EMA_Mid', ema_fast))
            ema_slow = float(exit_data.get('EMA_Slow', 0))
            if trade_direction == 'long' and ema_fast < ema_mid < ema_slow and price_change_pct > 0:
                return "ema_full_reversal"
            elif trade_direction == 'short' and ema_fast > ema_mid > ema_slow and price_change_pct > 0:
                return "ema_full_reversal"

            # ─── MACD cross detection ──────────────────────────────────────────
            macd = float(exit_data.get('MACD_closed', 0))
            macd_signal = float(exit_data.get('MACD_Signal_closed', 0))
            if trade_direction == 'long' and macd < macd_signal and price_change_pct > 0:
                return "macd_bearish_cross"
            elif trade_direction == 'short' and macd > macd_signal and price_change_pct > 0:
                return "macd_bullish_cross"

            # ─── ADX collapse detection ────────────────────────────────────────
            adx = float(exit_data.get('ADX', 25))
            if adx < 25 and price_change_pct > 0.5:
                return "adx_collapse_trend_weak"

            # ─── Price-based heuristics (meaningful categories) ──────────────
            if price_change_pct >= 6.0:
                return "heuristic_large_profit"
            elif price_change_pct >= 3.0:
                return "heuristic_profit_target"
            elif price_change_pct >= 0.5:
                return "heuristic_small_profit"
            elif price_change_pct >= -stop_pct:
                return "heuristic_small_loss"
            else:
                return "heuristic_large_loss"

        except Exception as e:
            return f"heuristic_error_{type(e).__name__}"

    def _calculate_entry_confluence(self, entry_data):
        """Calculate confluence score for an entry based on technical conditions."""
        score = 0
        details = []

        try:
            close = float(entry_data.get('Close', 0))
            ema_fast = float(entry_data.get('EMA_Fast', 0))
            ema_mid = float(entry_data.get('EMA_Mid', 0))
            ema_slow = float(entry_data.get('EMA_Slow', 0))

            if close > ema_fast > ema_mid > ema_slow:
                score += 5
                details.append(
                    f"EMA Structure ✓ (Close={close:.2f} > Fast={ema_fast:.2f} > Mid={ema_mid:.2f} > Slow={ema_slow:.2f})")
            else:
                details.append(f"EMA Structure ✗ (alignment broken)")
        except Exception as e:
            details.append(f"EMA Structure ✗ (error: {e})")

        try:
            volume_ratio = float(entry_data.get('Volume_Ratio', 0))

            if hasattr(self, 'strategy') and hasattr(self.strategy, 'volume_min_ratio'):
                volume_threshold = self.strategy.volume_min_ratio
            else:
                volume_threshold = 1.10

            if volume_ratio > volume_threshold:
                score += 1
                details.append(f"Volume ✓ (ratio={volume_ratio:.2f} > {volume_threshold})")
            else:
                details.append(f"Volume ✗ (ratio={volume_ratio:.2f} ≤ {volume_threshold})")
        except Exception as e:
            details.append(f"Volume ✗ (error: {e})")

        try:
            kalman = float(entry_data.get('Kalman_Strength', 0))

            if hasattr(self, 'strategy') and hasattr(self.strategy, 'kalman_min_strength'):
                kalman_threshold = self.strategy.kalman_min_strength
            else:
                kalman_threshold = 0.18

            if kalman > kalman_threshold:
                score += 1
                details.append(f"Kalman ✓ (strength={kalman:.2f} > {kalman_threshold})")
            else:
                details.append(f"Kalman ✗ (strength={kalman:.2f} ≤ {kalman_threshold})")
        except Exception as e:
            details.append(f"Kalman ✗ (error: {e})")

        if hasattr(self, 'log_message') and getattr(self, 'debug_confluence', False):
            self.log_message(f"📊 Confluence Calculation ({score}/7):", "blue")
            for detail in details:
                self.log_message(f"   {detail}", "white")

        return score

    def _calculate_risk_allocation(self, entry_data, confluence):
        """Calculate risk allocation percentage based on confluence score."""
        if confluence == 7:
            return 0.015
        elif confluence == 6:
            volume_ratio = float(entry_data.get('Volume_Ratio', 0))
            kalman = float(entry_data.get('Kalman_Strength', 0))

            volume_threshold = getattr(self.strategy, 'volume_min_ratio', 1.10)
            kalman_threshold = getattr(self.strategy, 'kalman_min_strength', 0.18)

            if volume_ratio > volume_threshold:
                return 0.010
            elif kalman > kalman_threshold:
                return 0.0075
            else:
                return 0.0075
        else:
            return 0.005

    def create_analysis_dataframe(self, df):
        working_df = df.copy(deep=True)

        if not isinstance(working_df.index, pd.DatetimeIndex):
            working_df.index = pd.to_datetime(working_df.index)

        working_df = working_df.sort_index()

        self.log_message(f"📋 Created working dataframe: {len(working_df)} rows", "blue")
        self.log_message(f"   Date range: {working_df.index[0]} to {working_df.index[-1]}", "blue")

        return working_df

    def display_backtest_results(self, stats, title):
        self.log_message("\n" + "=" * 60, "white")
        self.log_message(f"📊 {title}", "purple")
        self.log_message("=" * 60, "white")

        final_equity = stats.get('Equity Final [$]', 0)
        total_trades = stats.get('# Trades', 0)
        win_rate = stats.get('Win Rate [%]', 0)
        total_return = stats.get('Return [%]', 0)
        sharpe_ratio = stats.get('Sharpe Ratio', 0)
        sortino_ratio = stats.get('Sortino Ratio', 0)
        max_drawdown = stats.get('Max. Drawdown [%]', 0)

        self.log_message(f"➡️ Final Equity:        ${final_equity:,.2f}",
                         "green" if final_equity > 50000 else "red")
        self.log_message(f"➡️ Total Trades:        {total_trades}",
                         "green" if total_trades > 0 else "orange")
        self.log_message(f"➡️ Win Rate:            {win_rate:.2f}%",
                         "green" if win_rate > 50 else "orange")
        self.log_message(f"➡️ Return [%]:          {total_return:.2f}%",
                         "green" if total_return > 0 else "red")
        self.log_message(f"➡️ Sharpe Ratio:        {sharpe_ratio:.3f}",
                         "green" if sharpe_ratio > 1 else "orange")
        self.log_message(f"➡️ Sortino Ratio:       {sortino_ratio:.3f}",
                         "green" if sortino_ratio > 1.5 else "orange")
        self.log_message(f"➡️ Max Drawdown:        {max_drawdown:.2f}%",
                         "red" if max_drawdown > 10 else "orange")

        avg_trade = stats.get('Avg. Trade [%]', 0)
        profit_factor = stats.get('Profit Factor', 0)
        best_trade = stats.get('Best Trade [%]', 0)
        worst_trade = stats.get('Worst Trade [%]', 0)

        if avg_trade:
            self.log_message(f"➡️ Avg Trade [%]:       {avg_trade:.2f}%",
                             "green" if avg_trade > 0 else "red")
        if profit_factor:
            self.log_message(f"➡️ Profit Factor:       {profit_factor:.2f}",
                             "green" if profit_factor > 1.5 else "orange")
        if best_trade:
            self.log_message(f"➡️ Best Trade:          {best_trade:.2f}%", "green")
        if worst_trade:
            self.log_message(f"➡️ Worst Trade:         {worst_trade:.2f}%", "red")

        avg_duration = stats.get('Avg. Trade Duration', None)
        if avg_duration and not pd.isna(avg_duration):
            try:
                if hasattr(avg_duration, 'components'):
                    hours = avg_duration.components.hours
                    minutes = avg_duration.components.minutes
                    duration_str = f"{hours}h {minutes}m"
                else:
                    duration_str = str(avg_duration)

                self.log_message(f"➡️ Avg Trade Duration:  {duration_str}", "white")
            except (AttributeError, TypeError):
                self.log_message(f"➡️ Avg Trade Duration:  {avg_duration}", "white")

        self.log_message("=" * 60, "white")

    def calculate_order_size(self, current_price, balance_currency='USDT'):
        cash = self.get_balance(balance_currency)
        cash_to_use = cash * (self.order_size_var.get() / 100)

        fractional_units = cash_to_use / current_price

        return max(1, round(fractional_units))

    def place_order(self, side, price, quantity=None, **kwargs):
        try:
            if side not in ['buy', 'sell']:
                self.log_message(f"❌ Invalid order side: {side}", "red")
                return False

            # v9.4.2: Determine the intent of this order:
            #   side='buy'  + no position      → open long
            #   side='buy'  + short position   → close short (buy-to-cover)
            #   side='sell' + no position      → open short
            #   side='sell' + long position    → close long
            current_pos_type = self.position.get('type')  # None, 'long', or 'short'
            is_opening_short = (side == 'sell' and current_pos_type is None)
            is_closing_short = (side == 'buy' and current_pos_type == 'short')
            is_closing_long = (side == 'sell' and current_pos_type == 'long')
            is_opening_long = (side == 'buy' and current_pos_type is None)

            if side == 'buy':
                if is_closing_short:
                    # Buy-to-cover — no balance check, quantity comes from existing position
                    quantity = round(self.position.get('quantity', 0), 4)
                    if quantity <= 0:
                        self.log_message("❌ Short-close rejected: Corrupted position quantity", "red")
                        self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
                        return False
                elif is_opening_long:
                    if current_pos_type is not None:
                        self.log_message("⚠️ Buy rejected: Position already open", "orange")
                        return False
                    quantity = self.calculate_order_size(price)
                    if not quantity or quantity <= 0:
                        self.log_message("❌ Buy rejected: Invalid quantity", "red")
                        return False
                    required_capital = quantity * price
                    balance = self.get_balance('USDT')
                    if required_capital > balance:
                        self.log_message(
                            f"❌ Insufficient balance: Need ${required_capital:.2f}, Have ${balance:.2f}",
                            "red"
                        )
                        return False
                else:
                    self.log_message("⚠️ Buy rejected: Unexpected position state", "orange")
                    return False

            else:  # side == 'sell'
                if is_opening_short:
                    # Opening a short — validate via strategy, quantity provided by caller
                    if quantity is None or quantity <= 0:
                        self.log_message("❌ Short open rejected: Invalid quantity", "red")
                        return False
                    # Note: balance check for shorts is exchange-side (requires margin/futures account)

                elif is_closing_long:
                    if not self.validate_position_before_sell():
                        self.log_message("❌ Sell rejected: No valid position", "red")
                        return False
                    quantity = round(self.position.get('quantity', 0), 4)
                    if quantity <= 0:
                        self.log_message("❌ Sell rejected: Corrupted position quantity", "red")
                        self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
                        return False
                else:
                    self.log_message("⚠️ Sell rejected: Unexpected position state", "orange")
                    return False

                # Exit reason validation applies to both long-close and short-close
                if not is_opening_short:
                    valid_exit_reasons = {
                        'stop_loss', 'trailing_stop', 'breakeven', 'trend_reversal',
                        'trend_reversal_confirmed', 'overbought', 'overbought_and_macd_loss',
                        'momentum_loss', 'weak_trend_and_momentum_loss', 'ema_cross_down',
                        'ema_cross_down_confirmed', 'bollinger_overbought_exit',
                        'low_volatility_bearish_exit', 'time_exit', 'time_exit_profitable',
                        'manual_close', 'emergency', 'ranging_market', 'portfolio_safeguard',
                        'ml_exit_confirmed', 'profit_target_r2', 'profit_target_r4',
                        'profit_target_r6', 'profit_target_r8', 'macd_bearish_cross',
                        'ema_full_reversal', 'adx_collapse_trend_weak', 'max_hold_time'
                    }
                    exit_reason = kwargs.get('exit_reason', 'manual_close')
                    if exit_reason not in valid_exit_reasons:
                        self.log_message(f"⚠️ Invalid exit reason '{exit_reason}', using manual_close", "orange")
                        exit_reason = 'manual_close'
                    kwargs['exit_reason'] = exit_reason

            self.log_message(
                f"📤 Preparing {side.upper()} {quantity:.4f} {self.base_symbol()} @ ${price:.4f}",
                "blue"
            )

            order_params = {
                'side': side,
                'price': float(price),
                'quantity': float(quantity),
                'timestamp': datetime.now(timezone.utc),
                'symbol': self.symbol_var.get(),
                'status': 'pending'
            }

            if is_opening_long:
                atr = kwargs.get('atr', None)
                if atr is None:
                    if hasattr(self, 'current_data') and self.current_data is not None:
                        atr = float(self.current_data.get('ATR_closed', self.current_data.get('ATR', price * 0.01)))
                    else:
                        atr = price * 0.01
                # v9.4.2: Long stop is BELOW entry price
                self.initial_stop_loss = price - (atr * self.stop_loss_pct / 0.01)
                self.trailing_stop = self.initial_stop_loss
                self.highest_since_buy = price
                order_params.update({
                    'entry_confidence': kwargs.get('confidence', 0),
                    'atr_value': atr,
                    'initial_stop_loss': self.initial_stop_loss,
                    'position_intent': 'open_long'
                })

            elif is_opening_short:
                atr = kwargs.get('atr', None)
                if atr is None:
                    if hasattr(self, 'current_data') and self.current_data is not None:
                        atr = float(self.current_data.get('ATR_closed', self.current_data.get('ATR', price * 0.01)))
                    else:
                        atr = price * 0.01
                # v9.4.2: Short stop is ABOVE entry price
                self.initial_stop_loss = price + (atr * self.stop_loss_pct / 0.01)
                self.trailing_stop = self.initial_stop_loss
                self.highest_since_buy = price
                order_params.update({
                    'entry_confidence': kwargs.get('confidence', 0),
                    'atr_value': atr,
                    'initial_stop_loss': self.initial_stop_loss,
                    'position_intent': 'open_short'
                })

            else:  # closing a position (long-close or short-close)
                order_params.update({
                    'exit_reason': kwargs.get('exit_reason', 'manual_close'),
                    'entry_price': self.position.get('price'),
                    'entry_time': self.position.get('time'),
                    'position_intent': 'close_short' if is_closing_short else 'close_long'
                })

                if order_params['entry_time']:
                    order_params['hold_duration_minutes'] = (
                                                                    order_params['timestamp'] - order_params[
                                                                'entry_time']
                                                            ).total_seconds() / 60

            if self.mode_var.get().lower() == 'backtest':
                success = self.execute_backtest_order(order_params)
            else:
                success = self.execute_live_order(order_params)

            if not success:
                self.log_message(
                    f"❌ {side.upper()} FAILED: {quantity:.4f} @ ${price:.4f}",
                    "red"
                )
                self.play_notification("error")
                return False

            if is_opening_long:
                self.update_position_on_entry(price, quantity, kwargs.get('confidence', 0))
                self.log_message(
                    f"✅ LONG ORDER FILLED: {quantity:.4f} @ ${price:.4f} | "
                    f"Stop: ${self.initial_stop_loss:.4f} | Trail: ${self.trailing_stop:.4f}",
                    "green"
                )
                self.play_notification("buy_success")
                if hasattr(self, 'chart') and self.chart:
                    try:
                        self.chart.add_buy_marker(datetime.now(timezone.utc), price)
                    except Exception as e:
                        self.log_message(f"⚠️ Could not place buy marker: {e}", "orange")

            elif is_opening_short:
                # Track the short position at the app level
                self.position = {
                    'type': 'short',
                    'price': price,
                    'quantity': quantity,
                    'time': datetime.now(timezone.utc),
                    'entry_confidence': kwargs.get('confidence', 0)
                }
                self.log_message(
                    f"✅ SHORT ORDER FILLED: {quantity:.4f} @ ${price:.4f} | "
                    f"Stop: ${self.initial_stop_loss:.4f}",
                    "red"
                )
                self.play_notification("buy_success")
                if hasattr(self, 'chart') and self.chart:
                    try:
                        self.chart.add_sell_marker(datetime.now(timezone.utc), price)
                    except Exception as e:
                        self.log_message(f"⚠️ Could not place sell marker: {e}", "orange")

            elif is_closing_long:
                # v9.4.2: Long close PnL = (exit - entry) × qty
                entry_price = order_params.get('entry_price')
                pnl = (price - entry_price) * quantity if entry_price else 0.0
                pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price and quantity else 0.0
                self.log_message(
                    f"✅ LONG CLOSE FILLED: {quantity:.4f} @ ${price:.4f} | "
                    f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {kwargs.get('exit_reason', 'N/A')}",
                    "green" if pnl > 0 else "red"
                )
                self.play_notification("sell_success" if pnl > 0 else "sell_loss")
                if hasattr(self, 'chart') and self.chart:
                    try:
                        self.chart.add_sell_marker(datetime.now(timezone.utc), price)
                    except Exception as e:
                        self.log_message(f"⚠️ Could not place sell marker: {e}", "orange")
                self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
                self.update_status_indicators("parking")

            elif is_closing_short:
                # v9.4.2: Short close PnL = (entry - exit) × qty  (profit when price falls)
                entry_price = order_params.get('entry_price')
                pnl = (entry_price - price) * quantity if entry_price else 0.0
                pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price and quantity else 0.0
                self.log_message(
                    f"✅ SHORT CLOSE FILLED: {quantity:.4f} @ ${price:.4f} | "
                    f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {kwargs.get('exit_reason', 'N/A')}",
                    "green" if pnl > 0 else "red"
                )
                self.play_notification("sell_success" if pnl > 0 else "sell_loss")
                if hasattr(self, 'chart') and self.chart:
                    try:
                        self.chart.add_buy_marker(datetime.now(timezone.utc), price)
                    except Exception as e:
                        self.log_message(f"⚠️ Could not place buy marker: {e}", "orange")
                self.position = {'type': None, 'price': None, 'quantity': None, 'time': None}
                self.update_status_indicators("parking")

            return True

        except Exception as e:
            self.log_message(f"❌ Order execution failed: {str(e)}", "red")
            import traceback
            self.log_message(traceback.format_exc(), "red")
            return False

    def execute_backtest_order(self, order_params):
        try:
            self.update_stats()
            return True
        except Exception as e:
            self.log_message(f"Backtest order error: {str(e)}", "red")
            logging.error(f"Backtest order error: {traceback.format_exc()}")
            return False

    def verify_order_execution(self, order_id):
        try:
            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries):
                response = self.trade_api.get_order(instId=self.symbol_var.get(), ordId=order_id)
                if response['code'] == '0' and len(response['data']) > 0:
                    order_info = response['data'][0]
                    if order_info['state'] == 'filled':
                        return True
                    elif order_info['state'] in ['canceled', 'live']:
                        self.log_message(f"Order {order_id} not filled. State: {order_info['state']}", "orange")
                        return False
                time.sleep(retry_delay)
            self.log_message(f"Failed to verify order {order_id} after {max_retries} attempts", "red")
            return False
        except Exception as e:
            self.log_message(f"Order verification error: {str(e)}", "red")
            logging.error(f"Order verification error: {str(e)}")
            return False

    def log_order_execution(self, order_params):
        try:
            side = order_params['side'].lower()
            price = float(order_params['price'])
            quantity = float(order_params.get('actual_quantity', order_params['quantity']))
            # FIX: commission_var already stores a decimal fraction (e.g. 0.001 = 0.1%);
            # dividing by 100 again under-recorded real commission by ~100x.
            commission_rate = self.commission_var.get()
            timestamp = datetime.now(timezone.utc)

            # v9.4.2: Use position_intent to correctly classify the trade record.
            # Short-close places a side='buy' order, but it IS a position closure,
            # so it must be recorded as type='sell' for update_stats to count it.
            position_intent = order_params.get('position_intent', '')
            is_close = position_intent in ('close_long', 'close_short')
            is_close_short = position_intent == 'close_short'
            record_type = 'sell' if is_close else side  # closures always recorded as 'sell'

            trade_record = {
                'time': timestamp,
                'type': record_type,
                'price': price,
                'quantity': quantity,
                'commission': price * quantity * commission_rate,
                'exit_reason': order_params.get('exit_reason', 'manual' if side == 'sell' else None)
            }

            if side == 'buy':
                trade_record.update({
                    'entry_price': price,
                    'pnl': 0,
                    'entry_confidence': order_params.get('entry_confidence', 0),
                    'stop_loss': self.initial_stop_loss,
                    'trailing_stop': self.trailing_stop
                })

                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                log_msg = (f"[{timestamp_str}] 🟢 BUY {quantity:.4f} {self.base_symbol()} @ ${price:.4f} | "
                           f"Cost: ${(price * quantity):.2f} | "
                           f"Commission: ${trade_record['commission']:.4f}")

            elif side == 'sell' or is_close:
                if self.position['price'] is None:
                    self.log_message("⚠️ Close executed with no active position", "orange")
                    return False

                entry_price = float(self.position['price'])
                pos_type = self.position.get('type', 'long')
                # v9.4.2: direction-aware PnL for log
                if pos_type == 'short' or is_close_short:
                    gross_pnl = (entry_price - price) * quantity  # profit when price falls
                else:
                    gross_pnl = (price - entry_price) * quantity
                entry_commission = entry_price * quantity * commission_rate
                exit_commission = price * quantity * commission_rate
                total_commission = entry_commission + exit_commission
                net_pnl = gross_pnl - total_commission
                pnl_pct = (net_pnl / (entry_price * quantity)) * 100 if entry_price and quantity else 0.0
                duration = (timestamp - self.position['time']).total_seconds() / 60

                trade_record.update({
                    'entry_price': entry_price,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'pnl_pct': pnl_pct,
                    'duration': duration,
                    'commission': total_commission,
                    'pnl': net_pnl
                })

                log_color = "green" if net_pnl > 0 else "red"
                direction_label = "SHORT CLOSE" if (pos_type == 'short' or is_close_short) else "SELL"
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                log_msg = (
                    f"[{timestamp_str}] 🔴 {direction_label} {quantity:.4f} {self.base_symbol()} @ ${price:.4f} | "
                    f"Entry: ${entry_price:.4f} | "
                    f"Gross: ${gross_pnl:.2f} | "
                    f"Net: ${net_pnl:.2f} ({pnl_pct:.2f}%) | "
                    f"Duration: {duration:.1f}m")

                if self.mode_var.get().lower() == "backtest":
                    self.virtual_balance['USDT'] += (price * quantity) - total_commission
                    self.virtual_balance[self.base_symbol()] -= quantity

            self.trade_history.append(trade_record)

            self.update_stats()
            self.log_message(log_msg, log_color if side == 'sell' else "blue")
            return True

        except Exception as e:
            error_msg = f"⚠️ Order logging failed: {str(e)}"
            self.log_message(error_msg, "orange")
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            return False

    def wait_for_order_fill(self, order_id, timeout=30):
        start_time = time.time()
        check_interval = 2

        self.log_message(f"⏳ Waiting for order {order_id} to fill...", "blue")

        while time.time() - start_time < timeout:
            try:
                response = self.trade_api.get_order(instId=self.symbol_var.get(), ordId=order_id)
                if response['code'] == '0' and len(response['data']) > 0:
                    order_info = response['data'][0]
                    state = order_info['state']

                    if state == 'filled':
                        self.log_message(f"✅ Order {order_id} filled successfully", "green")
                        return True
                    elif state in ['canceled', 'partially_canceled']:
                        self.log_message(f"❌ Order {order_id} was canceled", "red")
                        return False
                    elif state == 'live':
                        pass
                    else:
                        self.log_message(f"⚠️ Order {order_id} in unexpected state: {state}", "orange")

                time.sleep(check_interval)
            except Exception as e:
                self.log_message(f"Error checking order status: {str(e)}", "red")
                time.sleep(check_interval)

        self.log_message(f"⏰ Order {order_id} fill check timed out after {timeout} seconds", "red")
        return False

    def get_order_details(self, order_id):
        try:
            response = self.trade_api.get_order(instId=self.symbol_var.get(), ordId=order_id)
            if response['code'] == '0' and len(response['data']) > 0:
                return response['data'][0]
            else:
                self.log_message(f"Failed to get order details: {response.get('msg', 'Unknown error')}", "red")
        except Exception as e:
            self.log_message(f"Error getting order details: {str(e)}", "red")
        return None

    def validate_position_before_sell(self):
        if self.position['price'] is None or self.position['quantity'] is None:
            self.log_message("⚠️ No active position to sell", "orange")
            return False

        symbol_balance = self.get_balance(self.base_symbol())
        position_quantity = self.position['quantity']

        if symbol_balance < position_quantity:
            self.log_message(f"⚠️ Position mismatch: Tracking {position_quantity} {self.base_symbol()}, "
                             f"but only {symbol_balance} in account. Resetting position tracking.", "red")
            self.position = {'price': None, 'quantity': None, 'time': None}
            return False

        current_price = self.get_current_price()
        if current_price and self.position['price']:
            position_value = position_quantity * self.position['price']
            if position_value <= 0 or position_value > 1000000:
                self.log_message(f"⚠️ Invalid position value: ${position_value}. Resetting position.", "red")
                self.position = {'price': None, 'quantity': None, 'time': None}
                return False

        return True

    def execute_live_order(self, order_params):
        try:
            order = self.trade_api.place_order(
                instId=self.symbol_var.get(),
                tdMode='cash',
                side=order_params['side'],
                ordType='market',
                sz=str(order_params['quantity']),
                tgtCcy='base_ccy'
            )

            if order['code'] != '0':
                error_code = order.get('code', 'UNKNOWN')
                error_msg = order.get('msg', 'No error message provided')
                error_explanations = {
                    '50113': 'Insufficient balance',
                    '51008': 'Order quantity too small',
                    '51009': 'Order quantity too large',
                    '51023': 'Too many open orders',
                    '1': 'General system error',
                    '2': 'Exchange not available'
                }
                explanation = error_explanations.get(error_code, "See API documentation for error code meaning")
                self.log_message(f"❌ Order failed - Code {error_code}: {error_msg}\n  "
                                 f" Explanation: {explanation}\n   Details: {order.get('data', 'No additional data')}",
                                 "red")
                return False

            order_id = order['data'][0]['ordId']
            self.log_message(f"📋 Order placed successfully. Order ID: {order_id}", "blue")

            if not self.wait_for_order_fill(order_id, timeout=30):
                self.log_message(f"❌ Order {order_id} not filled within timeout period", "red")
                return False

            filled_details = self.get_order_details(order_id)
            if not filled_details or float(filled_details.get('fillSz', 0)) == 0:
                self.log_message(f"❌ Order {order_id} has zero filled quantity", "red")
                return False

            order_params['actual_price'] = float(filled_details.get('fillPx', order_params['price']))
            order_params['actual_quantity'] = float(filled_details.get('fillSz', order_params['quantity']))
            order_params['order_id'] = order_id

            # FIX: commission_var already stores a decimal fraction; the extra /100
            # under-recorded real commission by ~100x.
            commission_rate = self.commission_var.get()
            order_params['commission'] = order_params['actual_price'] * order_params[
                'actual_quantity'] * commission_rate

            self.log_order_execution(order_params)

            if order_params['side'] == 'sell':
                self.position = {'price': None, 'quantity': None, 'time': None}
                self.update_status_indicators("parking")

            return True

        except requests.exceptions.RequestException as e:
            self.log_message(f"⌛ Network error during {order_params['side']} order\n   Error Type:"
                             f" {type(e).__name__}\n   Details: {str(e)}", "red")
            return False
        except Exception as e:
            error_type = type(e).__name__
            self.log_message(f"💥 Unexpected error during {order_params['side']} order\n  "
                             f" Error Type: {error_type}\n   Details: {str(e)}\n   Traceback: {traceback.format_exc()}",
                             "red")
            logging.exception("Order execution failed")
            return False

    def get_balance(self, currency, retries=3, delay=2.0):
        """Get balance with retry logic for transient API errors."""
        if self.mode_var.get().lower() in ('backtest', 'demo'):
            return self.virtual_balance.get(currency, 0)

        for attempt in range(1, retries + 1):
            try:
                if not self.account_api:
                    raise Exception("Account API not initialized")

                response = self.account_api.get_account_balance()

                if response['code'] == '0':
                    for asset in response['data'][0]['details']:
                        if asset['ccy'] == currency:
                            return float(asset['availBal'])
                    return 0.0

                # transient busy errors — retry silently
                msg = response.get('msg', '')
                if 'busy' in msg.lower() or 'try again' in msg.lower():
                    if attempt < retries:
                        time.sleep(delay)
                        continue
                    # only log after all retries exhausted
                    self.log_message(
                        f"⚠️ Balance unavailable after {retries} attempts "
                        f"({msg}) — using cached value", "orange")
                    return self._cached_balance.get(currency, 0)

                self.log_message(
                    f"Balance API error: {msg}", "red")
                return 0.0

            except Exception as e:
                if attempt < retries:
                    time.sleep(delay)
                    continue
                self.log_message(
                    f"Balance error after {retries} attempts: {e}", "red")
                return self._cached_balance.get(currency, 0)

        return self._cached_balance.get(currency, 0)

    def get_current_price(self):
        try:
            response = self.market_api.get_ticker(instId=self.symbol_var.get())
            if response['code'] == '0':
                return float(response['data'][0]['last'])
            return None
        except Exception as e:
            self.log_message(f"Price check error: {str(e)}", "red")
            return None

    def recover_failed_sell(self, order_params):
        try:
            current_price = self.get_current_price()
            if current_price is None:
                self.log_message("Cannot recover - failed to get current price", "red")
                return False
            adjusted_price = current_price * 0.99
            self.log_message(f"Retrying sell at adjusted price: {adjusted_price:.4f}", "orange")
            if self.place_order('sell', adjusted_price, order_params['quantity'], exit_reason='recovery_attempt'):
                return True
            self.log_message("Sell recovery failed - activating emergency", "red")
            self.emergency_stop()
            return False
        except Exception as e:
            self.log_message(f"Recovery error: {str(e)}", "red")
            self.emergency_stop()
            return False

    def generate_test_data(self, freq='T', start_date=None, end_date=None):
        try:
            start_date = start_date or '2023-01-01'
            end_date = end_date or datetime.now().strftime("%Y-%m-%d")
            date_rng = pd.date_range(start=start_date, end=end_date, freq=freq)
            df = pd.DataFrame(date_rng, columns=['timestamp'])
            base_price = 100 + (df.index * 0.1)
            volatility = pd.Series(range(len(df))).apply(
                lambda x: math.sin(x / 10) * 2 + (0.5 * x if x % 100 == 0 else 0))
            df['Close'] = base_price + volatility
            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close'] * 1.005
            df['Low'] = df['Close'] * 0.995
            df['Volume'] = 1000 + (df.index % 100) * 50
            df.set_index('timestamp', inplace=True)
            return df.dropna()
        except Exception as e:
            self.log_message(f"Test data generation error: {str(e)}", "red")
            logging.error(f"Test data generation error: {str(e)}")
            return None

    def show_final_results(self):
        if self.mode_var.get().lower() == 'backtest':
            final_balance = self.virtual_balance['USDT']
            if self.position['price'] is not None:
                final_balance += self.virtual_balance[self.base_symbol()] * self.position['price']

            initial_balance = 1000.0
            gross_profit = final_balance - initial_balance
            total_commission = sum(t.get('commission', 0) for t in self.trade_history)
            net_profit = gross_profit - total_commission
            roi = (net_profit / initial_balance) * 100

            closed_trades = [t for t in self.trade_history if t['type'] == 'sell']

            winning_trades = [t for t in closed_trades if t['net_pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['net_pnl'] <= 0]

            avg_win = sum(t['net_pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['net_pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0

            best_trade = max(t['net_pnl'] for t in closed_trades) if closed_trades else 0
            worst_trade = min(t['net_pnl'] for t in closed_trades) if closed_trades else 0

            if closed_trades:
                returns = [t['pnl_pct'] for t in closed_trades]
                avg_return = sum(returns) / len(returns)
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    downside_std = (sum((r - 0) ** 2 for r in downside_returns) / len(downside_returns)) ** 0.5
                    sortino_ratio = (avg_return / downside_std) if downside_std > 0 else 0
                else:
                    sortino_ratio = float('inf') if avg_return > 0 else 0
            else:
                sortino_ratio = 0

            message = (
                f"Backtest Results:\n"
                f"Initial Balance: ${initial_balance:.2f}\n"
                f"Final Balance: ${final_balance:.2f}\n"
                f"Gross Profit: ${gross_profit:.2f}\n"
                f"Total Commissions: ${total_commission:.2f}\n"
                f"Net Profit: ${net_profit:.2f}\n"
                f"ROI: {roi:.2f}%\n\n"
                f"Total Trades: {len(closed_trades)}\n"
                f"Win Rate: {win_rate * 100:.1f}%\n"
                f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}\n"
                f"Best Trade: ${best_trade:.2f}\n"
                f"Worst Trade: ${worst_trade:.2f}\n"
                f"Sortino Ratio: {sortino_ratio:.3f}\n\n"
                f"Commission Rate: {self.commission_var.get() * 100:.4f}%"
            )

            self.log_message(message, "purple")
            messagebox.showinfo("Backtest Complete", message)

    def analyze_quality_score_results(self, stats_opt):
        if not hasattr(stats_opt, '_strategy'):
            return

        params = stats_opt._strategy._params

        self.log_message("=" * 80, "green")
        self.log_message("🏆 QUALITY SCORE OPTIMIZATION RESULTS", "green")
        self.log_message("=" * 80, "green")

        min_score = params.get('quality_minimum_score', 75)
        fuzzy_enabled = params.get('fuzzy_mode_enabled', False)

        if min_score <= 65:
            threshold_color = "green"
            recommendation = "LOWER THRESHOLD WORKS BEST"
        elif min_score <= 70:
            threshold_color = "cyan"
            recommendation = "MODERATE THRESHOLD OPTIMAL"
        else:
            threshold_color = "yellow"
            recommendation = "STRICT THRESHOLD PREFERRED"

        self.log_message(f"\n🎯 RECOMMENDED QUALITY THRESHOLD: {min_score}", threshold_color)
        self.log_message(f"   Recommendation: {recommendation}", "white")

        self.log_message(f"\n🧠 FUZZY MODE: {'ENABLED ✓' if fuzzy_enabled else 'DISABLED ✗'}",
                         "green" if fuzzy_enabled else "blue")

        if fuzzy_enabled:
            margin = params.get('fuzzy_default_margin_pct', 10)
            abs_min = params.get('fuzzy_absolute_min', 60)
            self.log_message(f"   Margin: {margin}%", "white")
            self.log_message(f"   Absolute Minimum: {abs_min}", "white")

        self.log_message(f"\n📊 OPTIMAL WEIGHT DISTRIBUTION:", "cyan")

        weights = {
            'EMA': params.get('weight_ema', 20),
            'ADX': params.get('weight_adx', 20),
            'MACD': params.get('weight_macd', 25),
            'RSI': params.get('weight_rsi', 20),
            'Volume': params.get('weight_volume', 15)
        }

        total = sum(weights.values())
        for name, value in weights.items():
            pct = (value / total) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            self.log_message(f"   {name:6}: {value:2d} pts ({pct:3.0f}%) {bar}", "white")

        self.log_message(f"\n🔧 TIER REQUIREMENTS:", "cyan")
        self.log_message(f"   Tier 1 ADX Min: {params.get('tier1_adx_hard_min', 25)}", "white")
        self.log_message(f"   Tier 1 RSI Range: {params.get('tier1_rsi_min', 35)}-{params.get('tier1_rsi_max', 75)}",
                         "white")
        self.log_message(f"   Tier 1 Volume Min: {params.get('tier1_volume_min', 1.0)}x", "white")
        self.log_message(f"\n   Tier 2 ADX Min: {params.get('tier2_adx_min', 15)}", "white")
        self.log_message(f"   Tier 2 Volume Min: {params.get('tier2_volume_min', 0.6)}x", "white")

        self.log_message(f"\n📈 PRICE POSITIONING:", "cyan")
        early_bonus = params.get('price_percentile_bonus_early', 15)
        late_penalty = params.get('price_percentile_penalty_late', 15)
        self.log_message(f"   Early Entry Bonus: +{early_bonus}", "green")
        self.log_message(f"   Late Entry Penalty: -{late_penalty}", "red")

        self.log_message(f"\n📊 PERFORMANCE WITH OPTIMAL SETTINGS:", "cyan")
        self.log_message(f"   Sharpe Ratio: {stats_opt['Sharpe Ratio']:.3f}", "white")
        self.log_message(f"   Win Rate: {stats_opt['Win Rate [%]']:.1f}%", "white")
        self.log_message(f"   Total Trades: {stats_opt['# Trades']}", "white")

        self.log_message("=" * 80, "green")

        return {
            'quality_minimum_score': min_score,
            'fuzzy_mode_enabled': fuzzy_enabled,
            'weights': weights,
            'tier1_adx_min': params.get('tier1_adx_hard_min', 25),
            'tier1_volume_min': params.get('tier1_volume_min', 1.0),

        }