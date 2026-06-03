import time
import math
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FormatStrFormatter
import mplfinance as mpf
from matplotlib.dates import AutoDateLocator
import tkinter as tk
from tkinter import ttk


def unix_to_readable(unix_timestamp):
    """Convert Unix timestamp to a datetime object."""
    date_time = datetime.fromtimestamp(unix_timestamp / 1000.0)
    readable_date_time = date_time.strftime('%Y-%m-%d %H:%M:%S')
    return readable_date_time


def convert_to_unix_timestamp(date_str, date_format="%Y-%m-%dT%H:%M:%S.%fZ"):
    dt = datetime.strptime(date_str, date_format)
    return int(dt.timestamp() * 1000)


class FinancialChartWidget:
    def __init__(self, parent, width: int = 800, height: int = 500, max_bars: int = 60):
        self.parent = parent
        self.width = width
        self.height = height
        self.max_bars = max_bars
        self.df: pd.DataFrame | None = None
        self.current_forecast = None
        self.forecast_length = 0
        self.root = None

        self.view_state = "normal"
        self.bottom_chart_mode = tk.StringVar(value="volume")
        self._last_known_time = None

        # ============================================================
        # TRADE SIGNAL MARKERS
        # ============================================================
        self.buy_markers = []
        self.sell_markers = []

        # ============================================================
        # CURSOR TRACKING AND TOOLTIP VARIABLES
        # ============================================================
        self.cursor_annotation = None
        self.cursor_line_vertical = None
        self.cursor_line_horizontal = None
        self.hover_connection = None
        self.motion_connection = None
        self.leave_connection = None

        self.indicator_map = {
            'Volume': 'volume',
            'Kalman Filter': 'kalman',
            'RCI': 'rci',
            'CCI': 'cci',
            'RSI': 'rsi',
            'MACD': 'macd',
            'ADX': 'adx',
            'ATR': 'atr'
        }

        self.ema_settings = [
            {'period': 9,  'col': 'EMA_Fast', 'color': 'cyan',    'label': 'EMA Fast', 'param_key': 'ema_fast_period'},
            {'period': 21, 'col': 'EMA_Mid',  'color': 'yellow',  'label': 'EMA Mid',  'param_key': 'ema_mid_period'},
            {'period': 60, 'col': 'EMA_Slow', 'color': 'magenta', 'label': 'EMA Slow', 'param_key': 'ema_slow_period'},
        ]

        self.market_colors = mpf.make_marketcolors(
            up='#00ff00', down='#ff3333', edge='inherit', wick='inherit', volume='inherit', ohlc='white'
        )
        self.custom_style = mpf.make_mpf_style(
            marketcolors=self.market_colors,
            facecolor='#1e1e1e',
            gridcolor='#ffffff',
            gridstyle='--',
            rc={'axes.labelcolor': 'white', 'axes.edgecolor': 'white'}
        )

        self.frame = ttk.Frame(parent)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.top_info_frame = tk.Frame(self.frame, bg='#1e1e1e')
        self.top_info_frame.pack(side=tk.TOP, fill=tk.X)

        self.legend_frame = tk.Frame(self.top_info_frame, bg='#1e1e1e')
        self.legend_frame.pack(side=tk.TOP, fill=tk.X)

        self.ohlc_frame = tk.Frame(self.top_info_frame, bg='#1e1e1e')
        self.ohlc_frame.pack(side=tk.TOP, fill=tk.X, pady=(2, 2))

        self.ohlc_labels = {}
        ohlc_info = [('O', 'Open', 'white'), ('H', 'High', '#00ff00'),
                     ('L', 'Low', '#ff5555'), ('C', 'Close', '#ffff00')]
        for prefix, label, color in ohlc_info:
            lbl = tk.Label(self.ohlc_frame, text=f"{label}: --", fg=color, bg='#1e1e1e',
                           font=('Arial', 10, 'bold'))
            lbl.pack(side=tk.LEFT, padx=15)
            self.ohlc_labels[prefix] = lbl

        self.change_label = tk.Label(self.ohlc_frame, text="Chg: --", fg='white', bg='#1e1e1e',
                                     font=('Arial', 10, 'bold'))
        self.change_label.pack(side=tk.LEFT, padx=15)

        self._create_ema_labels()
        self._create_bottom_dropdown()

        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax1, self.ax2 = None, None
        self._create_axes_normal()
        self.fig.patch.set_facecolor('#1e1e1e')

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._connect_hover_events()

    # ============================================================
    # HELPER
    # ============================================================
    def get_root(self):
        try:
            return self.frame.winfo_toplevel()
        except:
            return None

    # ============================================================
    # TRADE SIGNAL MARKERS
    # ============================================================
    def add_buy_marker(self, timestamp, price):
        """Add a buy marker to the chart"""
        try:
            if hasattr(self, 'ax'):
                self.ax.plot(timestamp, price, '^', color='green', markersize=10,
                             markeredgecolor='white', markeredgewidth=1)
                self.canvas.draw()
        except Exception as e:
            print(f"Error adding buy marker: {e}")

    def add_sell_marker(self, timestamp, price):
        """Add a sell marker to the chart"""
        try:
            if hasattr(self, 'ax'):
                self.ax.plot(timestamp, price, 'v', color='red', markersize=10,
                             markeredgecolor='white', markeredgewidth=1)
                self.canvas.draw()
        except Exception as e:
            print(f"Error adding sell marker: {e}")
    def add_sell_marker(self, index, price=None):
        if self.ax1 is None or self.df is None:
            return False

        if isinstance(index, pd.Timestamp) or hasattr(index, 'timestamp'):
            try:
                idx_pos = self.df.index.get_loc(index)
            except:
                idx_pos = np.argmin(np.abs(self.df.index - index))
        else:
            try:
                idx_pos = int(index)
            except:
                return False

        if idx_pos < 0 or idx_pos >= len(self.df):
            return False

        candle = self.df.iloc[idx_pos]
        candle_low = candle.get('Low', price if price else 0)
        if candle_low == 0:
            candle_low = candle.get('Close', 100)

        marker_y = candle_low * 0.97
        marker = self.ax1.annotate(
            'S', xy=(idx_pos, marker_y), xytext=(0, 0),
            textcoords='offset points', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff3333',
                      edgecolor='white', linewidth=2, alpha=0.9),
            color='black', fontsize=11, fontweight='bold', zorder=100
        )
        self.sell_markers.append(marker)
        self.canvas.draw_idle()
        return True

    def clear_trade_markers(self):
        for marker in self.buy_markers:
            try:
                marker.remove()
            except:
                pass
        for marker in self.sell_markers:
            try:
                marker.remove()
            except:
                pass
        self.buy_markers = []
        self.sell_markers = []
        self.canvas.draw_idle()

    def set_trade_signals_from_dataframe(self, df_with_signals):
        if self.ax1 is None or self.df is None:
            return
        self.clear_trade_markers()
        if 'Entry_Signal' in df_with_signals.columns:
            for idx in df_with_signals[df_with_signals['Entry_Signal'] == 'BUY'].index:
                if idx in self.df.index:
                    self.add_buy_marker(idx)
        if 'Exit_Signal' in df_with_signals.columns:
            for idx in df_with_signals[df_with_signals['Exit_Signal'] == 'SELL'].index:
                if idx in self.df.index:
                    self.add_sell_marker(idx)

    # ============================================================
    # HOVER EVENTS
    # ============================================================
    def _connect_hover_events(self):
        self.hover_connection = self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.motion_connection = self.fig.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)
        self.leave_connection = self.fig.canvas.mpl_connect('figure_leave_event', self._on_leave_figure)
        try:
            self.root = self.frame.winfo_toplevel()
        except:
            self.root = None

    def _disconnect_hover_events(self):
        for attr in ('hover_connection', 'motion_connection', 'leave_connection'):
            conn = getattr(self, attr, None)
            if conn is not None:
                try:
                    self.fig.canvas.mpl_disconnect(conn)
                except:
                    pass
                setattr(self, attr, None)

    def _on_hover(self, event):
        if not hasattr(event, 'inaxes') or event.inaxes != self.ax1:
            self._remove_cursor_lines()
            return
        if self.df is None or self.df.empty:
            return
        x = event.xdata
        if x is None:
            return

        try:
            from matplotlib.dates import num2date
            hover_dt = num2date(x)
            hover_timestamp = pd.Timestamp(hover_dt)
            if hover_timestamp.tz is not None:
                hover_timestamp = hover_timestamp.tz_localize(None)
            df_index = self.df.index
            if hasattr(df_index, 'tz') and df_index.tz is not None:
                df_index_naive = df_index.tz_localize(None)
            else:
                df_index_naive = df_index
            time_diffs = abs(df_index_naive - hover_timestamp)
            idx = time_diffs.argmin()
            if time_diffs.iloc[idx] > pd.Timedelta(minutes=30):
                self._remove_cursor_lines()
                return
        except Exception as e:
            try:
                idx = int(round(x))
                if idx < 0 or idx >= len(self.df):
                    return
            except:
                return

        candle = self.df.iloc[idx]
        candle_time = self.df.index[idx]
        close_price = float(candle.get('Close', 0))
        open_price  = float(candle.get('Open',  close_price))
        high_price  = float(candle.get('High',  close_price))
        low_price   = float(candle.get('Low',   close_price))

        if isinstance(candle_time, pd.Timestamp):
            if candle_time.tz is not None:
                time_str = candle_time.tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S UTC')
            else:
                time_str = candle_time.strftime('%Y-%m-%d %H:%M:%S') + " UTC"
        else:
            time_str = str(candle_time)

        is_up = close_price >= open_price
        color = '#00ff00' if is_up else '#ff5555'
        tooltip_text = (
            f"📅 {time_str}\n"
            f"━━━━━━━━━━━━━━\n"
            f"O: ${open_price:.4f} -  H: ${high_price:.4f}\n"
            f"C: ${close_price:.4f} -  L: ${low_price:.4f}\n"
        )
        if 'RSI' in candle and not pd.isna(candle['RSI']):
            tooltip_text += f"\nRSI: {float(candle['RSI']):.1f}"
        if 'Volume' in candle and not pd.isna(candle['Volume']):
            tooltip_text += f"  Vol: {float(candle['Volume']):.0f}"
        if 'ADX' in candle and not pd.isna(candle['ADX']):
            tooltip_text += f"\nADX: {float(candle['ADX']):.1f}"

        self._remove_cursor_lines()
        self.cursor_line_vertical   = self.ax1.axvline(x=x, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        self.cursor_line_horizontal = self.ax1.axhline(y=close_price, color='white', linestyle='--',
                                                       linewidth=0.8, alpha=0.7)
        # ── Auto-flip tooltip left when near right edge ──────────────────────────
        xmin, xmax = self.ax1.get_xlim()
        x_range = xmax - xmin if xmax != xmin else 1
        # If cursor is in the right 35% of the chart, flip tooltip to the left
        near_right_edge = ((x - xmin) / x_range) > 0.65

        if near_right_edge:
            # Tooltip to the LEFT of the cursor
            x_offset = -160
            y_offset = 20
            arrow_style = dict(arrowstyle='->', color=color, linewidth=1.5,
                               connectionstyle='arc3,rad=0.0')
        else:
            # Tooltip to the RIGHT of the cursor (default)
            x_offset = 10
            y_offset = 20
            arrow_style = dict(arrowstyle='->', color=color, linewidth=1.5)

        self.cursor_annotation = self.ax1.annotate(
            tooltip_text,
            xy=(x, close_price),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e1e1e',
                      edgecolor=color, linewidth=2, alpha=0.95),
            color='white', fontsize=9, family='monospace',
            arrowprops=arrow_style
        )
        # ─────────────────────────────────────────────────────────────────────────
        self.canvas.draw_idle()

    def _on_leave_axes(self, event):
        self._remove_cursor_lines()
        self.canvas.draw_idle()

    def _on_leave_figure(self, event):
        self._remove_cursor_lines()
        self.canvas.draw_idle()

    def _remove_cursor_lines(self):
        for attr in ('cursor_annotation', 'cursor_line_vertical', 'cursor_line_horizontal'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except:
                    pass
                setattr(self, attr, None)

    # ============================================================
    # DROPDOWN
    # ============================================================
    def _create_bottom_dropdown(self):
        self.bottom_control_frame = tk.Frame(self.frame, bg='#1A3C8C')
        self.bottom_control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        gradient_canvas = tk.Canvas(self.bottom_control_frame, height=30, bg='#1A3C8C',
                                    highlightthickness=0, bd=0)
        gradient_canvas.pack(fill=tk.X)

        width = 800
        for i in range(width):
            r = int(26  + (74  - 26)  * i / width)
            g = int(60  + (144 - 60)  * i / width)
            b = int(140 + (226 - 140) * i / width)
            gradient_canvas.create_line(i, 0, i, 30, fill=f'#{r:02x}{g:02x}{b:02x}')

        control_container = tk.Frame(gradient_canvas, bg='#2A4C9C')
        gradient_canvas.create_window(0, 0, window=control_container, anchor='nw', width=width, height=30)

        tk.Label(control_container, text="📊 Bottom Chart:", fg='white', bg='#2A4C9C',
                 font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(15, 5), pady=5)

        self.bottom_chart_combobox = ttk.Combobox(
            control_container, textvariable=self.bottom_chart_mode,
            values=list(self.indicator_map.keys()), state="readonly",
            width=18, font=('Arial', 10)
        )
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('ModernBlue.TCombobox',
                        background='#2A5CBC', foreground='white',
                        fieldbackground='#2A5CBC', selectbackground='#1A4CAC',
                        selectforeground='white', borderwidth=1, relief='solid',
                        padding=5, arrowsize=12)
        style.map('ModernBlue.TCombobox',
                  fieldbackground=[('readonly', '#2A5CBC')],
                  background=[('readonly', '#2A5CBC')],
                  foreground=[('readonly', 'white')],
                  selectbackground=[('readonly', '#1A4CAC')],
                  selectforeground=[('readonly', 'white')])
        self.bottom_chart_combobox.configure(style='ModernBlue.TCombobox')
        self.bottom_chart_combobox.pack(side=tk.LEFT, padx=5, pady=3)
        self.bottom_chart_combobox.bind("<<ComboboxSelected>>", self._on_bottom_chart_change)

        self.current_bottom_label = tk.Label(
            control_container, text="[Volume]", fg='#E6F3FF', bg='#1A4CAC',
            font=('Arial', 10, 'bold'), relief='raised', bd=1, padx=10, pady=2
        )
        self.current_bottom_label.pack(side=tk.LEFT, padx=(20, 5), pady=3)

        info_label = tk.Label(control_container, text="ℹ️", fg='white', bg='#2A4C9C',
                              font=('Arial', 10), cursor='hand2')
        info_label.pack(side=tk.RIGHT, padx=(0, 15))

        def show_tooltip(event):
            from tkinter import messagebox
            messagebox.showinfo("Bottom Chart Selector",
                                "Select which indicator to display in the bottom chart.\n\n"
                                "Volume: Trading volume with color coding\n"
                                "Kalman Filter: Trend following indicator\n"
                                "RCI: Rank Correlation Index\n"
                                "CCI: Commodity Channel Index\n"
                                "RSI: Relative Strength Index\n"
                                "MACD: Moving Average Convergence Divergence\n"
                                "ADX: Average Directional Index\n"
                                "ATR: Average True Range")
        info_label.bind("<Button-1>", show_tooltip)

    def _on_bottom_chart_change(self, event=None):
        selected_display = self.bottom_chart_mode.get()
        if hasattr(self, 'current_bottom_label'):
            self.current_bottom_label.config(text=selected_display)
        if selected_display in self.indicator_map:
            internal_mode = self.indicator_map[selected_display]
            self.bottom_chart_mode.set(internal_mode)
        if self.df is not None:
            self.update_chart(self.df)

    # ============================================================
    # INDICATOR CALCULATIONS
    # ============================================================
    def _calculate_indicators(self):
        if self.df is None or self.df.empty:
            return

        if 'RSI' not in self.df.columns:
            delta = self.df['Close'].diff()
            gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs    = gain / loss
            self.df['RSI'] = 100 - (100 / (1 + rs))

        if 'MACD' not in self.df.columns:
            exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
            self.df['MACD']           = exp1 - exp2
            self.df['MACD_Signal']    = self.df['MACD'].ewm(span=9, adjust=False).mean()
            self.df['MACD_Histogram'] = self.df['MACD'] - self.df['MACD_Signal']

        if 'ATR' not in self.df.columns:
            high_low    = self.df['High'] - self.df['Low']
            high_close  = np.abs(self.df['High'] - self.df['Close'].shift())
            low_close   = np.abs(self.df['Low']  - self.df['Close'].shift())
            true_range  = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
            self.df['ATR'] = true_range.rolling(14).mean()

    # ============================================================
    # CHART CREATION AND UPDATING
    # ============================================================
    def _apply_axis_styling(self, ax):
        if ax is None:
            return
        ax.set_facecolor('#1e1e1e')
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.grid(color='#ffffff', linestyle='--', linewidth=0.6, alpha=0.35)

    def _create_axes_normal(self):
        self.fig.clear()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
        self.fig.subplots_adjust(left=0.03, right=0.88, top=0.92, bottom=0.12)

    def _create_axes_single(self, which='ax1'):
        self.fig.clear()
        if which == 'ax1':
            self.ax1 = self.fig.add_subplot(111)
            self.ax2 = None
        else:
            self.ax2 = self.fig.add_subplot(111)
            self.ax1 = None
        self.fig.subplots_adjust(left=0.03, right=0.88, top=0.92, bottom=0.12)

    def update_chart(self, df: pd.DataFrame, *, params: dict = None, stop_loss=None, trailing_stop=None,
                     live_price=None):
        if df is None or df.empty:
            return

        self.df = df.copy()

        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                self.df.index = pd.to_datetime(self.df.index)
            except:
                self.df.index = pd.date_range(
                    end=datetime.now(timezone.utc), periods=len(self.df), freq='1min'
                )

        if self.df.index.tz is not None:
            self.df.index = self.df.index.tz_localize(None)

        self.df = self.df.sort_index()
        self._calculate_indicators()

        if params:
            for ema in self.ema_settings:
                key = ema['param_key']
                if key in params:
                    ema['period'] = int(params[key])

        if self.max_bars and len(self.df) > self.max_bars:
            self.df = self.df.iloc[-self.max_bars:]

        for ema in self.ema_settings:
            self.df[ema['col']] = self.df['Close'].ewm(span=ema['period'], adjust=False).mean()

        # ── FIX 1: Remove cursor annotation BEFORE clearing axes ─────────────────
        # Without this, ax.clear() sets cursor_annotation.axes = None but the
        # object survives in memory. canvas.draw() then finds it and crashes on
        # self.axes.bbox.
        self._disconnect_hover_events()
        self._remove_cursor_lines()
        # ─────────────────────────────────────────────────────────────────────────

        if self.ax1:
            self.ax1.clear()
        if self.ax2:
            self.ax2.clear()

        self._update_ema_labels()
        self._update_ohlc_display()

        add_plots = []
        if self.ax1:
            for ema in self.ema_settings:
                add_plots.append(mpf.make_addplot(self.df[ema['col']], ax=self.ax1,
                                                  color=ema['color'], width=1.2))
            if stop_loss is not None:
                sl = pd.Series(stop_loss, index=self.df.index) if isinstance(stop_loss, (int, float)) else stop_loss
                add_plots.append(mpf.make_addplot(sl, ax=self.ax1, color='white', linestyle=':', width=1.0))
            if trailing_stop is not None:
                ts = pd.Series(trailing_stop, index=self.df.index) if isinstance(trailing_stop,
                                                                                 (int, float)) else trailing_stop
                add_plots.append(mpf.make_addplot(ts, ax=self.ax1, color='yellow', linestyle=':', width=1.0))

        try:
            w_cfg = dict(candle_linewidth=1.0, candle_width=0.7)

            if self.view_state == "ax1_max":
                mpf.plot(self.df, type='candle', style=self.custom_style,
                         ax=self.ax1, addplot=add_plots, update_width_config=w_cfg,
                         datetime_format='%H:%M:%S\n%Y-%m-%d')

            elif self.view_state == "ax2_max":
                self._plot_indicator_on_ax(self.ax2, xlim=None)

            else:
                show_vol = (self.bottom_chart_mode.get() == 'volume')
                mpf.plot(self.df, type='candle', style=self.custom_style,
                         ax=self.ax1,
                         volume=(self.ax2 if show_vol else False),
                         addplot=add_plots, update_width_config=w_cfg,
                         datetime_format='%H:%M:%S\n%Y-%m-%d')

                if not show_vol:
                    ax1_xlim = self.ax1.get_xlim()
                    self._plot_indicator_on_ax(self.ax2, xlim=ax1_xlim)

        except Exception as e:
            print('Plot Error:', e)

        self._apply_axis_styling(self.ax1)
        self._apply_axis_styling(self.ax2)
        if self.ax1:
            self.ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        self._set_chart_titles()

        # ── Live Forming Candle Overlay ──────────────────────────────────────────
        if self.ax1 is not None and len(self.df) >= 1:
            try:
                live = self.df.iloc[-1]
                live_o = float(live['Open'])
                live_c = float(live['Close'])
                live_h = float(live['High'])
                live_l = float(live['Low'])

                display_price = float(live_price) if live_price is not None else live_c
                is_bull = display_price >= live_o
                price_col = '#26a69a' if is_bull else '#ef5350'
                x_live = len(self.df) - 1

                # ── FIX 2: Snapshot lists before removing to avoid mutating-while-iterating ──
                live_lines = [l for l in self.ax1.get_lines()
                              if getattr(l, '_live_overlay', False)]
                live_texts = [t for t in self.ax1.texts
                              if getattr(t, '_live_overlay', False)]
                for obj in live_lines + live_texts:
                    try:
                        obj.remove()
                    except Exception:
                        pass
                # ────────────────────────────────────────────────────────────────────────────

                xmin, xmax = self.ax1.get_xlim()
                padded_xmax = xmax + 1.0
                self.ax1.set_xlim(xmin, padded_xmax)

                price_line, = self.ax1.plot(
                    [xmin, padded_xmax], [display_price, display_price],
                    color=price_col, linewidth=0.8, linestyle='--', alpha=0.75, zorder=6
                )
                price_line._live_overlay = True

                v_line, = self.ax1.plot(
                    [x_live, x_live], [live_l, live_h],
                    color=price_col, linewidth=1.0, linestyle='-', alpha=0.5, zorder=5
                )
                v_line._live_overlay = True

                # ── FIX 3: Use get_yaxis_transform() instead of xycoords=('axes fraction','data')
                # The tuple form requires self.axes.bbox during draw, which can be None
                # when axes have been recently cleared. get_yaxis_transform() is equivalent
                # but resolves the transform at creation time, avoiding the crash.
                label_ann = self.ax1.annotate(
                    f' ${display_price:.4f}',
                    xy=(1.0, display_price),
                    xycoords=self.ax1.get_yaxis_transform(),
                    fontsize=8, color=price_col,
                    va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e1e',
                              edgecolor=price_col, linewidth=1, alpha=0.85),
                    zorder=7
                )
                label_ann._live_overlay = True

                direction = '▲' if is_bull else '▼'
                chg = display_price - live_o
                chg_pct = (chg / live_o * 100) if live_o != 0 else 0
                self.change_label.config(
                    text=f"{direction} {chg:+.4f} ({chg_pct:+.2f}%)",
                    fg='#00ff00' if is_bull else '#ff5555'
                )

            except Exception as e:
                print(f'Live overlay error: {e}')
        # ─────────────────────────────────────────────────────────────────────────

        if self.current_forecast is not None:
            self.plot_forecast(self.current_forecast)

        self._format_axis_labels()

        root = self.get_root()
        if root:
            root.after(100, self._connect_hover_events)
        else:
            self._connect_hover_events()

        # ── FIX 4: Guard draw() against any remaining stale annotations ──────────
        # Defensive last resort: if any annotation still has axes=None, remove it
        # before draw() so a single bad artist can't abort the entire render.
        for ax in [self.ax1, self.ax2]:
            if ax is None:
                continue
            stale = [a for a in ax.texts if a.axes is None]
            for a in stale:
                try:
                    a.remove()
                except Exception:
                    pass
        # ─────────────────────────────────────────────────────────────────────────

        self.canvas.draw()
    # ============================================================
    # BOTTOM CHART INDICATOR PLOTTING  ← FIXED
    # ============================================================
    def _plot_indicator_on_ax(self, ax, xlim=None):
        """
        Plot the selected indicator on the bottom axis.

        `xlim` should be the (xmin, xmax) tuple taken from ax1 *after* mpf.plot
        has run.  When provided, ax2 is forced to the same x-limits so both
        charts are always in sync — regardless of which indicator is shown.

        NOTE: ax.clear() is NOT called here.  The caller (update_chart) is
        responsible for clearing ax2 before this method runs, which avoids
        breaking the sharex link created in _create_axes_normal.
        """
        if ax is None or self.df is None:
            return

        mode = self.bottom_chart_mode.get()
        # Use the same integer x-coordinates that mpf uses internally
        x = np.arange(len(self.df))

        if mode == 'volume':
            colors = ['#00ff00' if c >= o else '#ff3333'
                      for o, c in zip(self.df['Open'], self.df['Close'])]
            ax.bar(x, self.df['Volume'], color=colors, alpha=0.8)
            ax.set_ylabel('Volume', color='white')
            ax.set_yscale('log')

        elif mode == 'kalman' and 'Kalman' in self.df.columns:
            ax.plot(x, self.df['Kalman'].values, color='#00FFFF', linewidth=1.5)
            ax.set_ylabel('Kalman', color='white')

        elif mode == 'rci' and 'RCI' in self.df.columns:
            ax.plot(x, self.df['RCI'].values, color='#FF00FF', linewidth=1.5)
            ax.axhline(80,  color='red',   linestyle='--', alpha=0.3)
            ax.axhline(-80, color='green', linestyle='--', alpha=0.3)
            ax.set_ylabel('RCI', color='white')

        elif mode == 'cci' and 'CCI' in self.df.columns:
            ax.plot(x, self.df['CCI'].values, color='#F1C40F', linewidth=1.5)
            ax.axhline(100,  color='red',   linestyle='--', alpha=0.3)
            ax.axhline(-100, color='green', linestyle='--', alpha=0.3)
            ax.axhline(0,    color='white', linestyle='-',  alpha=0.5)
            ax.set_ylabel('CCI', color='white')

        elif mode == 'rsi' and 'RSI' in self.df.columns:
            ax.plot(x, self.df['RSI'].values, color='#FFA500', linewidth=1.5)
            ax.axhline(70, color='red',   linestyle='--', alpha=0.5)
            ax.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax.axhline(50, color='white', linestyle='-',  alpha=0.3)
            ax.set_ylabel('RSI', color='white')
            ax.set_ylim(0, 100)

        elif mode == 'macd' and 'MACD' in self.df.columns and 'MACD_Signal' in self.df.columns:
            ax.plot(x, self.df['MACD'].values,        color='#00FF00', linewidth=1.5, label='MACD')
            ax.plot(x, self.df['MACD_Signal'].values, color='#FF0000', linewidth=1.5, label='Signal')
            if 'MACD_Histogram' in self.df.columns:
                colors = ['#00FF00' if h >= 0 else '#FF0000' for h in self.df['MACD_Histogram']]
                ax.bar(x, self.df['MACD_Histogram'].values, color=colors, alpha=0.5, width=0.8)
            ax.axhline(0, color='white', linestyle='-', alpha=0.5)
            ax.legend(fontsize=8, facecolor='#1e1e1e', edgecolor='white')
            ax.set_ylabel('MACD', color='white')

        elif mode == 'adx' and 'ADX' in self.df.columns:
            ax.plot(x, self.df['ADX'].values, color='#9B59B6', linewidth=1.5, label='ADX')
            ax.axhline(25, color='yellow', linestyle='--', alpha=0.5, label='Trend Threshold')
            ax.set_ylabel('ADX', color='white')
            ax.legend(fontsize=8, facecolor='#1e1e1e', edgecolor='white')

        elif mode == 'atr' and 'ATR' in self.df.columns:
            ax.plot(x, self.df['ATR'].values, color='#3498DB', linewidth=1.5)
            ax.set_ylabel('ATR', color='white')

        else:
            # Fallback: volume
            colors = ['#00ff00' if c >= o else '#ff3333'
                      for o, c in zip(self.df['Open'], self.df['Close'])]
            ax.bar(x, self.df['Volume'], color=colors, alpha=0.8)
            ax.set_ylabel('Volume', color='white')
            ax.set_yscale('log')

        self._apply_axis_styling(ax)
        ax.set_facecolor('#1e1e1e')

        # ─── KEY FIX: force ax2 to use exactly the same x-range as ax1 ────────
        if xlim is not None:
            ax.set_xlim(xlim)
        # ────────────────────────────────────────────────────────────────────────

    # ============================================================
    # OHLC / EMA DISPLAY
    # ============================================================
    def _update_ohlc_display(self):
        if self.df is None or self.df.empty:
            return
        l = self.df.iloc[-1]
        for p, k in [('O', 'Open'), ('H', 'High'), ('L', 'Low'), ('C', 'Close')]:
            if k in l:
                self.ohlc_labels[p].config(text=f"{p}: {l[k]:.4f}")
        if len(self.df) > 1:
            pc  = self.df.iloc[-2]['Close']
            chg = l['Close'] - pc
            pct = (chg / pc) * 100 if pc != 0 else 0
            self.change_label.config(
                text=f"{'▲' if chg >= 0 else '▼'} {chg:+.4f} ({pct:+.2f}%)",
                fg='#00ff00' if chg >= 0 else '#ff5555'
            )

    def _create_ema_labels(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        for ema in self.ema_settings:
            f = tk.Frame(self.legend_frame, bg='#1e1e1e')
            f.pack(side=tk.LEFT, padx=10, pady=3)
            tk.Label(f, text='━━', fg=ema['color'], bg='#1e1e1e',
                     font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
            ema['label_widget'] = tk.Label(f, text=f"{ema['label']} ({ema['period']})",
                                           fg='white', bg='#1e1e1e', font=('Arial', 10))
            ema['label_widget'].pack(side=tk.LEFT, padx=5)
            ema['value_label'] = tk.Label(f, text='--', fg=ema['color'], bg='#1e1e1e',
                                          font=('Arial', 10, 'bold'))
            ema['value_label'].pack(side=tk.LEFT)

    def _update_ema_labels(self):
        if self.df is None or self.df.empty:
            return
        for ema in self.ema_settings:
            if 'label_widget' in ema:
                ema['label_widget'].config(text=f"{ema['label']} ({ema['period']})")
            if ema['col'] in self.df.columns:
                val = self.df[ema['col']].iloc[-1]
                ema['value_label'].config(text=f'{val:.4f}' if pd.notna(val) else '--')

    def _on_click(self, event):
        if not event.dblclick:
            return
        if self.view_state == "normal":
            self.view_state = "ax1_max" if event.inaxes == self.ax1 else "ax2_max"
            self._create_axes_single('ax1' if self.view_state == "ax1_max" else 'ax2')
        else:
            self.view_state = "normal"
            self._create_axes_normal()
        if self.df is not None:
            self.update_chart(self.df)

    def _set_chart_titles(self):
        if self.ax1:
            self.ax1.set_title('Price Chart', color='white', fontsize=12,
                               fontweight='bold', pad=10, loc='left')
        if self.ax2:
            display_name = "Volume"
            for disp_name, mode in self.indicator_map.items():
                if mode == self.bottom_chart_mode.get():
                    display_name = disp_name
                    break
            self.ax2.set_title(display_name, color='white', fontsize=12,
                               fontweight='bold', pad=10, loc='left')

    def plot_forecast(self, forecast):
        if self.ax1 is None or forecast is None:
            return
        self.current_forecast = np.array(forecast)
        for line in [l for l in self.ax1.lines if l.get_label() == 'Forecast (ML)']:
            line.remove()
        idx = range(len(self.df), len(self.df) + len(forecast))
        self.ax1.plot(idx, forecast, '--', color='white', label='Forecast (ML)',
                      linewidth=1.5, zorder=100)
        self.ax1.set_xlim(self.ax1.get_xlim()[0], len(self.df) + len(forecast) + 1)
        self.ax1.legend(loc='upper left', fontsize=8, facecolor='#1e1e1e',
                        edgecolor='white', labelcolor='white')

    def _format_axis_labels(self):
        for ax in [self.ax1, self.ax2]:
            if ax:
                for lbl in ax.get_xticklabels():
                    lbl.set_color('white')
                    lbl.set_rotation(45)
                    lbl.set_horizontalalignment('right')

    def set_max_bars(self, n: int):
        self.max_bars = max(20, int(n))


# ============================================================
# CIRCLE TIMER (unchanged)
# ============================================================
class CircleTimer:
    def __init__(self, parent, size=150, bg_color='light gray', fg_color='black'):
        self.parent = parent
        self.size = size
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.angle = 0
        self.active = False
        self.current_interval = 60
        self.start_time = None
        self.first_start_time = None
        self.elapsed = 0
        self.interval_strings = ['1m', '5m', '15m', '30m', '1H', '1D']
        self.interval_seconds_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1H': 3600, '1D': 86400
        }
        self.current_interval_index = 0

        self.canvas = tk.Canvas(parent, width=size, height=size,
                                bg=bg_color, highlightthickness=0)
        self.canvas.pack()
        self.draw_background()

        self.sector = self.canvas.create_arc(0, 0, 0, 0, start=90, extent=0,
                                              fill='#3498db', outline='black', style=tk.PIESLICE)
        self.hand   = self.canvas.create_line(0, 0, 0, 0, width=3, fill='red', arrow=tk.LAST)
        self.interval_text = self.canvas.create_text(size // 2, size // 2 + 30,
                                                      text="1m", fill="white", font=("Arial", 16))
        self.canvas.itemconfigure(self.sector, state=tk.HIDDEN)
        self.canvas.itemconfigure(self.hand,   state=tk.HIDDEN)

        self.digital_timer = tk.Label(parent, text="00:00:00",
                                      font=("DS-Digital", 20), bg="black", fg="lime")
        self.digital_timer.pack()
        self.animation_id = None

    def draw_background(self):
        center = self.size // 2
        radius = self.size // 2 - 10
        self.canvas.create_oval(center - radius, center - radius,
                                center + radius, center + radius,
                                outline=self.fg_color, width=2)
        self.canvas.create_oval(center - 3, center - 3, center + 3, center + 3,
                                fill=self.fg_color, outline=self.fg_color)
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            inner = (center + (radius - 10) * math.cos(angle),
                     center + (radius - 10) * math.sin(angle))
            outer = (center + radius * math.cos(angle),
                     center + radius * math.sin(angle))
            self.canvas.create_line(*inner, *outer, fill=self.fg_color, width=2)

    def set_interval(self, interval):
        if isinstance(interval, str):
            if interval in self.interval_strings:
                self.current_interval = self.interval_seconds_map[interval]
                self.current_interval_index = self.interval_strings.index(interval)
        elif isinstance(interval, (int, float)):
            diffs   = {k: abs(v - interval) for k, v in self.interval_seconds_map.items()}
            closest = min(diffs, key=diffs.get)
            self.current_interval       = self.interval_seconds_map[closest]
            self.current_interval_index = self.interval_strings.index(closest)
        self.canvas.itemconfigure(self.interval_text,
                                  text=self.interval_strings[self.current_interval_index])
        if self.active:
            self.restart()

    def update_timer(self):
        if not self.active:
            return
        elapsed  = time.time() - self.start_time
        progress = min(elapsed / self.current_interval, 1.0)
        self.angle  = progress * 360
        center      = self.size // 2
        radius      = self.size // 2 - 15
        self.canvas.coords(self.sector,
                           center - radius, center - radius,
                           center + radius, center + radius)
        self.canvas.itemconfigure(self.sector, extent=-self.angle)
        angle_rad = math.radians(self.angle - 90)
        end_x     = center + (radius - 5) * math.cos(angle_rad)
        end_y     = center + (radius - 5) * math.sin(angle_rad)
        self.canvas.coords(self.hand, center, center, end_x, end_y)
        if elapsed >= self.current_interval - 0.1:
            self.complete_cycle()
        else:
            self.animation_id = self.parent.after(50, self.update_timer)

    def complete_cycle(self):
        self.canvas.itemconfigure(self.sector, fill='green')
        self.parent.after(200, lambda: self.canvas.itemconfigure(self.sector, fill='#3498db'))
        self.start_time = time.time()
        self.angle = 0
        self.update_timer()

    def start(self):
        if not self.active:
            self.active = True
            if not self.first_start_time:
                self.first_start_time = time.time()
                self.update_digital_timer()
            self.start_time = time.time() - (time.time() % self.current_interval)
            self.canvas.itemconfigure(self.sector, state=tk.NORMAL)
            self.canvas.itemconfigure(self.hand,   state=tk.NORMAL)
            self.update_timer()

    def update_digital_timer(self):
        if self.first_start_time:
            elapsed_seconds = int(time.time() - self.first_start_time)
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.digital_timer.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.parent.after(1000, self.update_digital_timer)

    def stop(self):
        if self.active:
            self.active = False
            if self.animation_id:
                self.parent.after_cancel(self.animation_id)
            self.canvas.itemconfigure(self.sector, state=tk.HIDDEN)
            self.canvas.itemconfigure(self.hand,   state=tk.HIDDEN)

    def restart(self):
        self.stop()
        self.start()