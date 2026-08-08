import sys
import time
import threading
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os

import ccxt
import pandas as pd
from backtesting import Backtest, Strategy

# =====================================================================
# 1. IMPORT REAL STRATEGY FROM YOUR STRATEGY FILE
# =====================================================================
try:
    from strategies.MomentumStrategy_MACD_HybridScore_Claude import (
        BacktestMomentumStrategy,
        MomentumConfig,
        MOMENTUM_PARAMS
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import BacktestMomentumStrategy. {e}")
    print("Ensure 'MomentumStrategy_MACD_HybridScore_Claude.py' is in the same folder.")
    sys.exit(1)

# =====================================================================
# 2. STRATEGY CONFIGURATION
# =====================================================================

STRATEGY_CONFIG = {
    'Momentum': {
        'class': BacktestMomentumStrategy,
        'param_prefix': 'momentum',
        'param_defaults': MOMENTUM_PARAMS,
        'optimization_params': {
            # Core Quality Thresholds
            'quality_tier1_min_long': [70, 75, 80],
            'quality_tier2_min_long': [60, 65, 70],

            # Trend & Momentum Filters
            'tier1_adx_hard_min_long': [22, 25, 28],
            'tier1_rsi_max_long': [60, 65, 70],
            'tier1_confluence_min': [0.55, 0.65, 0.75],

            # Direction-Specific Stop Loss Multipliers
            'tier1_stop_multiplier_long': [1.5, 2.0, 2.5],
            'tier1_stop_multiplier_short': [2.5, 3.0, 3.5],

            # Risk Management
            'risk_tier1': [0.015, 0.020, 0.025],
            'risk_tier2': [0.010, 0.015, 0.020],
        },
        'param_mapping': {
            'quality_tier1_min_long': 'quality_tier1_min_long',
            'quality_tier2_min_long': 'quality_tier2_min_long',
            'tier1_adx_hard_min_long': 'tier1_adx_hard_min_long',
            'tier1_rsi_max_long': 'tier1_rsi_max_long',
            'tier1_confluence_min': 'tier1_confluence_min',
            'tier1_stop_multiplier_long': 'tier1_stop_multiplier_long',
            'tier1_stop_multiplier_short': 'tier1_stop_multiplier_short',
            'risk_tier1': 'risk_tier1',
            'risk_tier2': 'risk_tier2',
        },
        'description': 'Two-tier MACD Hybrid Momentum (Real Strategy)',
        'color': '#4CAF50',
        'icon': '📈'
    }
}


# =====================================================================
# 3. CCXT DATA FETCHING LOGIC
# =====================================================================

def fetch_ccxt_historical_data(
        symbol: str, timeframe: str, start_str: str, end_str: str, log_func, exchange_name: str = 'binance'
) -> pd.DataFrame:
    log_func(f"Connecting to {exchange_name.upper()} for {symbol} ({timeframe})...")
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({'enableRateLimit': True})

    start_dt = pd.to_datetime(start_str, utc=True)
    end_dt = pd.to_datetime(end_str, utc=True)

    since = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    all_ohlcv = []
    log_func(f"Fetching candles from {start_str} to {end_str}...")

    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break

            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)

            if ohlcv[-1][0] >= end_ts or len(ohlcv) < 1000:
                break

            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            log_func(f"Fetch warning: {e}. Retrying in 2s...")
            time.sleep(2)

    if not all_ohlcv:
        raise ValueError(f"No OHLCV data returned for {symbol} in specified date range.")

    log_func(f"Downloaded {len(all_ohlcv):,} total candles successfully.")

    df = pd.DataFrame(
        all_ohlcv,
        columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'],
    )
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df.set_index('Timestamp', inplace=True)

    df = df[df.index <= end_dt]
    df.index = df.index.tz_localize(None)
    return df


# =====================================================================
# 4. PARAMETER FILE HELPERS
# =====================================================================

def load_strategy_settings() -> dict:
    settings_file = "strategy_settings.json"
    default_settings = {
        'default_params': {},
        'custom_params': {'momentum': {}},
        'selected_mode': 'Default Parameters'
    }

    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except Exception:
            return default_settings
    return default_settings


def save_strategy_settings(settings: dict) -> bool:
    settings_file = "strategy_settings.json"
    try:
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def update_custom_params(strategy_type: str, optimized_params: dict) -> tuple:
    try:
        settings = load_strategy_settings()
        config = STRATEGY_CONFIG.get(strategy_type, {})
        prefix = config.get('param_prefix', 'momentum')

        if 'custom_params' not in settings:
            settings['custom_params'] = {}
        if prefix not in settings['custom_params']:
            settings['custom_params'][prefix] = {}

        current_custom = settings['custom_params'][prefix]
        param_mapping = config.get('param_mapping', {})

        mapped_params = {}
        changes = {}

        for opt_key, value in optimized_params.items():
            # Explicitly cast to string to satisfy type checkers
            actual_key = str(param_mapping.get(opt_key, opt_key))
            mapped_params[actual_key] = value

            if actual_key in current_custom:
                if current_custom[actual_key] != value:
                    changes[actual_key] = {'old': current_custom[actual_key], 'new': value}
            else:
                changes[actual_key] = {'old': 'Not Set', 'new': value}

        if not changes:
            return True, {}

        settings['custom_params'][prefix].update(mapped_params)
        settings['selected_mode'] = 'Custom Parameters'

        if save_strategy_settings(settings):
            return True, changes
        return False, None

    except Exception as e:
        print(f"Error updating custom params: {e}")
        return False, None


def get_current_strategy_from_app() -> str:
    env_strategy = os.environ.get('BEST_PARAM_STRATEGY', '')
    if env_strategy and env_strategy in STRATEGY_CONFIG:
        return env_strategy
    return 'Momentum'


# =====================================================================
# 5. RESULTS TABLE DIALOG
# =====================================================================

class ResultsTableDialog:
    def __init__(self, parent, stats, strategy_type, symbol, timeframe, start_date, end_date):
        self.parent = parent
        self.stats = stats
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self._build_dialog()

    def _build_dialog(self):
        self.dialog = tk.Toplevel(self.parent)
        config = STRATEGY_CONFIG.get(self.strategy_type, {})
        icon = config.get('icon', '📊')
        color = config.get('color', '#4CAF50')

        self.dialog.title(f"{icon} {self.strategy_type} Professional Optimization Results")
        self.dialog.geometry("720x640")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        self.dialog.update_idletasks()
        width, height = 720, 640
        x = (self.dialog.winfo_screenwidth() - width) // 2
        y = (self.dialog.winfo_screenheight() - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

        header_frame = tk.Frame(self.dialog, bg="#1a1a2e", pady=12)
        header_frame.pack(fill=tk.X)

        tk.Label(header_frame, text=f"{icon} {self.strategy_type} OPTIMIZATION RESULTS",
                 font=('Helvetica', 16, 'bold'), bg="#1a1a2e", fg=color).pack()
        tk.Label(header_frame, text=f"{self.symbol} • {self.timeframe} • {self.start_date} → {self.end_date}",
                 font=('Helvetica', 10), bg="#1a1a2e", fg="#aaaaaa").pack()

        main_frame = ttk.Frame(self.dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        columns = ('Metric', 'Value', 'Grade', 'Target')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=13)
        tree.heading('Metric', text='Metric')
        tree.heading('Value', text='Value')
        tree.heading('Grade', text='Performance')
        tree.heading('Target', text='Target Range')

        tree.column('Metric', width=180, anchor='w')
        tree.column('Value', width=120, anchor='center')
        tree.column('Grade', width=150, anchor='center')
        tree.column('Target', width=150, anchor='center')

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        metrics = [
            ('📈 Sharpe Ratio', f"{self.stats.get('Sharpe Ratio', 0.0):.3f}"),
            ('💰 Total Return', f"{self.stats.get('Return [%]', 0.0):.2f}%"),
            ('🎯 Win Rate', f"{self.stats.get('Win Rate [%]', 0.0):.2f}%"),
            ('📉 Max Drawdown', f"{self.stats.get('Max. Drawdown [%]', 0.0):.2f}%"),
            ('📋 Total Trades', f"{self.stats.get('# Trades', 0)}"),
            ('💵 Final Equity', f"${self.stats.get('Equity Final [$]', 0.0):,.2f}"),
        ]

        for metric, value in metrics:
            tree.insert('', tk.END, values=(metric, value, '', 'N/A'))

        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        ttk.Button(btn_frame, text="✅ Close", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)


# =====================================================================
# 6. TKINTER GUI INTERFACE
# =====================================================================

class StrategyOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Strategy Parameter Optimizer")
        self.root.geometry("820x850")
        self.root.minsize(750, 750)

        self.root.attributes('-topmost', True)
        self.root.focus_force()
        self.root.grab_set()

        self._center_window()
        self.start_time = None
        self.timer_running = False
        self.timer_id = None

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.optimization_complete = False
        self.best_params = None
        self.opt_stats = None
        self.current_strategy = get_current_strategy_from_app()

        self._build_ui()
        self._load_env_vars()

        if self.current_strategy in STRATEGY_CONFIG:
            self.strategy_var.set(self.current_strategy)
            self._on_strategy_change()

        self.log(f"📊 Auto-detected strategy: {self.current_strategy}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self.root.grab_release()
        except:
            pass
        self.root.destroy()

    def _center_window(self):
        self.root.update_idletasks()
        width, height = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _load_env_vars(self):
        symbol = os.environ.get('BEST_PARAM_SYMBOL', 'SOL/USDT')
        interval = os.environ.get('BEST_PARAM_INTERVAL', '15m')
        start_date = os.environ.get('BEST_PARAM_START_DATE', '2022-01-01')
        end_date = os.environ.get('BEST_PARAM_END_DATE', datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        self.symbol_var.set(symbol)
        self.timeframe_var.set(interval)
        self.start_date_var.set(start_date)
        self.end_date_var.set(end_date)

    def _build_ui(self):
        ttk.Label(self.root, text="🚀 Professional Strategy Optimization Suite",
                  font=('Helvetica', 16, 'bold')).pack(pady=12)

        config_frame = ttk.LabelFrame(self.root, text=" Target Configuration ")
        config_frame.pack(fill="x", padx=15, pady=5)

        ttk.Label(config_frame, text="Strategy:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.strategy_var = tk.StringVar(value="Momentum")
        self.strategy_combo = ttk.Combobox(config_frame, textvariable=self.strategy_var,
                                           values=["Momentum"], width=15, state="readonly")
        self.strategy_combo.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        self.strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy_change)

        ttk.Label(config_frame, text="Symbol:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.symbol_var = tk.StringVar(value="SOL/USDT")
        ttk.Entry(config_frame, textvariable=self.symbol_var, width=15).grid(row=1, column=1, sticky="w", padx=10,
                                                                             pady=5)

        ttk.Label(config_frame, text="Timeframe:").grid(row=1, column=2, sticky="w", padx=10, pady=5)
        self.timeframe_var = tk.StringVar(value="15m")
        ttk.Combobox(config_frame, textvariable=self.timeframe_var,
                     values=["1m", "5m", "15m", "1h", "4h", "1d"], width=10, state="readonly").grid(
            row=1, column=3, sticky="w", padx=10, pady=5)

        ttk.Label(config_frame, text="Start Date:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.start_date_var = tk.StringVar(value="2022-01-01")
        ttk.Entry(config_frame, textvariable=self.start_date_var, width=15).grid(row=2, column=1, sticky="w", padx=10,
                                                                                 pady=5)

        ttk.Label(config_frame, text="End Date:").grid(row=2, column=2, sticky="w", padx=10, pady=5)
        self.end_date_var = tk.StringVar(value=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        ttk.Entry(config_frame, textvariable=self.end_date_var, width=15).grid(row=2, column=3, sticky="w", padx=10,
                                                                               pady=5)

        self.info_frame = ttk.LabelFrame(self.root, text=" Strategy Info ")
        self.info_frame.pack(fill="x", padx=15, pady=5)
        self.strategy_info_label = ttk.Label(self.info_frame, text="", font=('Helvetica', 10, 'bold'))
        self.strategy_info_label.pack(padx=10, pady=5, anchor="w")

        self.params_frame = ttk.LabelFrame(self.root, text=" Optimization Parameters ")
        self.params_frame.pack(fill="x", padx=15, pady=5)
        self.params_text = scrolledtext.ScrolledText(self.params_frame, height=5, font=("Consolas", 9), wrap="word")
        self.params_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._update_params_display()

        dir_frame = ttk.LabelFrame(self.root, text=" Execution Direction ")
        dir_frame.pack(fill="x", padx=15, pady=5)
        self.direction_var = tk.StringVar(value="BOTH")  # Default to BOTH for realistic market regimes
        ttk.Radiobutton(dir_frame, text="Long Only", variable=self.direction_var, value="LONG_ONLY").pack(side="left",
                                                                                                          padx=20,
                                                                                                          pady=5)
        ttk.Radiobutton(dir_frame, text="Short Only", variable=self.direction_var, value="SHORT_ONLY").pack(side="left",
                                                                                                            padx=20,
                                                                                                            pady=5)
        ttk.Radiobutton(dir_frame, text="Both", variable=self.direction_var, value="BOTH").pack(side="left", padx=20,
                                                                                                pady=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=15, pady=10)

        self.run_btn = ttk.Button(control_frame, text="🚀 Run Optimization", command=self.start_optimization_thread)
        self.run_btn.pack(side="left", padx=5)

        self.progress_bar = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=10)

        self.save_btn = ttk.Button(control_frame, text="💾 Save Params", command=self.save_to_custom_params,
                                   state=tk.DISABLED)
        self.save_btn.pack(side="left", padx=5)

        output_frame = ttk.LabelFrame(self.root, text=" Results & Optimum Clause ")
        output_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.output_text = scrolledtext.ScrolledText(output_frame, font=("Consolas", 10), wrap="word", bg="#1e1e1e",
                                                     fg="#d4d4d4")
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _on_strategy_change(self, event=None):
        self.current_strategy = self.strategy_var.get()
        config = STRATEGY_CONFIG.get(self.current_strategy, {})
        self.strategy_info_label.config(
            text=f"{config.get('icon', '📊')} {self.current_strategy}: {config.get('description', '')}",
            foreground=config.get('color', '#4CAF50')
        )
        self._update_params_display()
        self.log(f"📊 Switched to {self.current_strategy} strategy")

    def _update_params_display(self):
        config = STRATEGY_CONFIG.get(self.current_strategy, {})
        opt_params = config.get('optimization_params', {})

        lines = [f"📊 Optimizing {len(opt_params)} core parameters for {self.current_strategy}:", ""]
        for param, values in opt_params.items():
            lines.append(f"  • {param}: {values}")

        total_combos = 1
        for values in opt_params.values():
            total_combos *= len(values)
        lines.append("")
        lines.append(f"  Total combinations: {total_combos:,} (Using random search max_tries=200)")

        self.params_text.config(state=tk.NORMAL)
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, "\n".join(lines))
        self.params_text.config(state=tk.DISABLED)

    def log(self, message: str):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def start_optimization_thread(self):
        self.output_text.delete("1.0", tk.END)
        self.run_btn.config(state="disabled")
        self.save_btn.config(state=tk.DISABLED)
        self.progress_bar.start(10)
        self.optimization_complete = False
        self.best_params = None

        thread = threading.Thread(target=self.run_optimization_process, daemon=True)
        thread.start()

    def save_to_custom_params(self):
        if not self.best_params or not self.optimization_complete:
            messagebox.showwarning("No Results", "Please run optimization first.")
            return

        param_list = "\n".join(f"  • {k}: {v}" for k, v in self.best_params.items())
        confirm = messagebox.askyesnocancel(
            "Save Parameters",
            f"Overwrite current custom parameters with these optimized values?\n\n{param_list}"
        )

        if not confirm:
            return

        success, changes = update_custom_params(self.current_strategy, self.best_params)
        if success:
            self.log("✅ Custom parameters updated successfully!")
            messagebox.showinfo("Success", "Optimized parameters saved to custom_params!")
            self.save_btn.config(state=tk.DISABLED, text="✅ Saved!")
        else:
            messagebox.showerror("Error", "Could not update custom parameters.")

    def run_optimization_process(self):
        try:
            symbol = self.symbol_var.get().strip().upper()
            timeframe = self.timeframe_var.get().strip().lower()
            start_date = self.start_date_var.get().strip()
            end_date = self.end_date_var.get().strip()
            trade_direction = self.direction_var.get().lower()
            strategy_type = self.strategy_var.get()

            self.log(f"📊 Strategy: {strategy_type} | Direction: {trade_direction}")

            config = STRATEGY_CONFIG.get(strategy_type)
            strategy_class = config['class']
            opt_params = config['optimization_params']
            param_mapping = config['param_mapping']

            # Fetch Data
            df = fetch_ccxt_historical_data(symbol, timeframe, start_date, end_date, self.log)

            # Initialize Backtest with standard settings matching the real strategy
            bt = Backtest(
                df,
                strategy_class,
                cash=50000.0,
                commission=0.001,  # 0.1% commission
                exclusive_orders=True,
                trade_on_close=True
            )

            self.log("Executing Grid Search Optimization...")

            # Add fixed parameters (like trade_direction) to the optimizer call
            fixed_params = {
                'trade_direction': trade_direction
            }

            # Run Optimization
            opt_stats, heatmap = bt.optimize(
                **opt_params,
                **fixed_params,
                maximize='Sharpe Ratio',
                return_heatmap=True,
                max_tries=200,
                random_state=42,
                constraint=lambda p: p.quality_tier1_min_long > p.quality_tier2_min_long
            )

            # Extract Best Parameters
            best_strat = opt_stats._strategy
            self.best_params = {}

            for opt_key in opt_params.keys():
                # Explicitly cast to string to satisfy type checkers
                actual_key = str(param_mapping.get(opt_key, opt_key))
                value = getattr(best_strat, opt_key, None)
                if value is not None:
                    self.best_params[actual_key] = value

            self.opt_stats = opt_stats
            self.optimization_complete = True

            sharpe = opt_stats.get('Sharpe Ratio', 0.0)
            self.log("\n" + "=" * 60)
            self.log(f"          {strategy_type} OPTIMIZATION COMPLETE          ")
            self.log("=" * 60)
            self.log(f"Sharpe Ratio    : {sharpe:.2f}")
            self.log(f"Total Return    : {opt_stats.get('Return [%]', 0.0):.2f}%")
            self.log(f"Max Drawdown    : {opt_stats.get('Max. Drawdown [%]', 0.0):.2f}%")
            self.log(f"Win Rate        : {opt_stats.get('Win Rate [%]', 0.0):.2f}%")
            self.log(f"Total Trades    : {opt_stats.get('# Trades', 0)}")

            self.log("\n🏆 BEST PARAMETERS FOUND:")
            self.log("-" * 40)
            for actual_key, value in self.best_params.items():
                self.log(f"   {actual_key}: {value}")

            # Show Results Dialog
            self.root.after(100, lambda: ResultsTableDialog(
                self.root, opt_stats, strategy_type, symbol, timeframe, start_date, end_date
            ))

            self.save_btn.config(state=tk.NORMAL, text=f"💾 Save {strategy_type} Params")
            messagebox.showinfo("Optimization Complete",
                                f"Sharpe: {sharpe:.3f}\nReturn: {opt_stats.get('Return [%]', 0.0):.2f}%\nTrades: {opt_stats.get('# Trades', 0)}")

        except Exception as err:
            self.log(f"\n❌ Error during optimization: {str(err)}")
            messagebox.showerror("Optimization Error", str(err))

        finally:
            self.progress_bar.stop()
            self.run_btn.config(state="normal")


# =====================================================================
# 7. MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = StrategyOptimizerGUI(root)
    root.mainloop()