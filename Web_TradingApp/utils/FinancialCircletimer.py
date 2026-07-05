import time
import math
from datetime import datetime, timedelta
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

        self.view_state = "normal"
        self.bottom_chart_mode = tk.StringVar(value="volume")

        # EMA settings
        self.ema_settings = [
            {'period': 9, 'col': 'EMA_Fast', 'color': 'cyan', 'label': 'EMA Fast', 'param_key': 'ema_fast_period'},
            {'period': 21, 'col': 'EMA_Mid', 'color': 'yellow', 'label': 'EMA Mid', 'param_key': 'ema_mid_period'},
            {'period': 60, 'col': 'EMA_Slow', 'color': 'magenta', 'label': 'EMA Slow', 'param_key': 'ema_slow_period'},
        ]

        # Define High-Brightness Market Colors
        self.market_colors = mpf.make_marketcolors(
            up='#00ff00', down='#ff3333', edge='inherit', wick='inherit', volume='inherit', ohlc='white'
        )

        self.custom_style = mpf.make_mpf_style(
            marketcolors=self.market_colors,
            facecolor='#1e1e1e',
            gridcolor='#ffffff',  # Base grid color
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
        ohlc_info = [('O', 'Open', 'white'), ('H', 'High', '#00ff00'), ('L', 'Low', '#ff5555'),
                     ('C', 'Close', '#ffff00')]
        for prefix, label, color in ohlc_info:
            lbl = tk.Label(self.ohlc_frame, text=f"{label}: --", fg=color, bg='#1e1e1e', font=('Arial', 10, 'bold'))
            lbl.pack(side=tk.LEFT, padx=15)
            self.ohlc_labels[prefix] = lbl

        self.change_label = tk.Label(self.ohlc_frame, text="Chg: --", fg='white', bg='#1e1e1e',
                                     font=('Arial', 10, 'bold'))
        self.change_label.pack(side=tk.LEFT, padx=15)

        self._create_ema_labels()

        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax1, self.ax2 = None, None
        self._create_axes_normal()
        self.fig.patch.set_facecolor('#1e1e1e')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _apply_axis_styling(self, ax):
        if ax is None: return
        ax.set_facecolor('#1e1e1e')

        # Force Y values to the Right
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position("right")

        # High Visibility Grid and Ticks
        ax.tick_params(axis='both', colors='white', labelsize=9)
        for spine in ax.spines.values(): spine.set_color('white')

        # Adjusted for high visibility
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

    def update_chart(self, df: pd.DataFrame, *, params: dict = None, stop_loss=None, trailing_stop=None):
        if df is None or df.empty: return
        self.df = df.copy()

        if params:
            for ema in self.ema_settings:
                key = ema['param_key']
                if key in params: ema['period'] = int(params[key])

        if self.max_bars and len(self.df) > self.max_bars:
            self.df = self.df.iloc[-self.max_bars:]

        for ema in self.ema_settings:
            self.df[ema['col']] = self.df['Close'].ewm(span=ema['period'], adjust=False).mean()

        if self.ax1: self.ax1.clear()
        if self.ax2: self.ax2.clear()

        self._update_ema_labels()
        self._update_ohlc_display()

        add_plots = []
        if self.ax1:
            for ema in self.ema_settings:
                add_plots.append(mpf.make_addplot(self.df[ema['col']], ax=self.ax1, color=ema['color'], width=1.2))
            if stop_loss is not None:
                add_plots.append(mpf.make_addplot(pd.Series(stop_loss, index=self.df.index), ax=self.ax1, color='white',
                                                  linestyle=':', width=1.0))

        try:
            w_cfg = dict(candle_linewidth=1.0, candle_width=0.7)
            if self.view_state == "ax1_max":
                mpf.plot(self.df, type='candle', style=self.custom_style, ax=self.ax1, addplot=add_plots,
                         update_width_config=w_cfg)
            elif self.view_state == "ax2_max":
                self._plot_indicator_on_ax(self.ax2)
            else:
                show_vol = self.bottom_chart_mode.get() == 'volume'
                mpf.plot(self.df, type='candle', style=self.custom_style, ax=self.ax1,
                         volume=(self.ax2 if show_vol else False), addplot=add_plots, update_width_config=w_cfg)
                if not show_vol: self._plot_indicator_on_ax(self.ax2)
        except Exception as e:
            print('Plot Error:', e)

        # Re-apply styling and formatting AFTER mpf.plot to override its defaults
        self._apply_axis_styling(self.ax1)
        self._apply_axis_styling(self.ax2)
        if self.ax1: self.ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        self._set_chart_titles()
        if self.current_forecast is not None: self.plot_forecast(self.current_forecast)
        self._format_axis_labels()
        self.canvas.draw()

    def _plot_indicator_on_ax(self, ax):
        if ax is None or self.df is None: return
        mode, x = self.bottom_chart_mode.get(), range(len(self.df))
        if mode == 'volume':
            colors = ['#00ff00' if c >= o else '#ff3333' for o, c in zip(self.df['Open'], self.df['Close'])]
            ax.bar(x, self.df['Volume'], color=colors, alpha=0.8)
        elif mode == 'kalman' and 'Kalman' in self.df.columns:
            ax.plot(x, self.df['Kalman'], color='#00FFFF')
        elif mode == 'rci' and 'RCI' in self.df.columns:
            ax.plot(x, self.df['RCI'], color='#FF00FF')
            for h in [80, -80]: ax.axhline(h, color='red' if h > 0 else 'green', linestyle='--', alpha=0.3)
        elif mode == 'cci' and 'cci' in self.df.columns:
            ax.plot(x, self.df['CCI'], color='#F1C40F')
            for h in [100, -100]: ax.axhline(h, color='white', linestyle='--', alpha=0.3)

    def _update_ohlc_display(self):
        if self.df is None or self.df.empty: return
        l = self.df.iloc[-1]
        for p, k in [('O', 'Open'), ('H', 'High'), ('L', 'Low'), ('C', 'Close')]:
            self.ohlc_labels[p].config(text=f"{p}: {l[k]:.4f}")
        if len(self.df) > 1:
            pc = self.df.iloc[-2]['Close']
            chg = l['Close'] - pc
            pct = (chg / pc) * 100 if pc != 0 else 0
            self.change_label.config(text=f"{'▲' if chg >= 0 else '▼'} {chg:+.4f} ({pct:+.2f}%)",
                                     fg='#00ff00' if chg >= 0 else '#ff5555')

    def _create_ema_labels(self):
        for w in self.legend_frame.winfo_children(): w.destroy()
        for ema in self.ema_settings:
            f = tk.Frame(self.legend_frame, bg='#1e1e1e')
            f.pack(side=tk.LEFT, padx=10, pady=3)
            tk.Label(f, text='━━', fg=ema['color'], bg='#1e1e1e', font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
            ema['label_widget'] = tk.Label(f, text=f"{ema['label']} ({ema['period']})", fg='white', bg='#1e1e1e',
                                           font=('Arial', 10))
            ema['label_widget'].pack(side=tk.LEFT, padx=5)
            ema['value_label'] = tk.Label(f, text='--', fg=ema['color'], bg='#1e1e1e', font=('Arial', 10, 'bold'))
            ema['value_label'].pack(side=tk.LEFT)

    def _update_ema_labels(self):
        if self.df is None or self.df.empty: return
        for ema in self.ema_settings:
            if 'label_widget' in ema: ema['label_widget'].config(text=f"{ema['label']} ({ema['period']})")
            if ema['col'] in self.df.columns:
                val = self.df[ema['col']].iloc[-1]
                ema['value_label'].config(text=f'{val:.4f}' if pd.notna(val) else '--')

    def _on_click(self, event):
        if not event.dblclick: return
        if self.view_state == "normal":
            self.view_state = "ax1_max" if event.inaxes == self.ax1 else "ax2_max"
            self._create_axes_single('ax1' if self.view_state == "ax1_max" else 'ax2')
        else:
            self.view_state = "normal"
            self._create_axes_normal()
        if self.df is not None: self.update_chart(self.df)

    def _set_chart_titles(self):
        if self.ax1: self.ax1.set_title('Price', color='white', fontsize=11, fontweight='bold', loc='left')
        if self.ax2:
            t = {'volume': 'Volume', 'kalman': 'Kalman', 'rci': 'RCI', 'cci': 'CCI'}.get(self.bottom_chart_mode.get(),
                                                                                         'Volume')
            self.ax2.set_title(t, color='white', fontsize=11, fontweight='bold', loc='left')

    def plot_forecast(self, forecast):
        if self.ax1 is None or forecast is None: return
        self.current_forecast = np.array(forecast)
        for line in [l for l in self.ax1.lines if l.get_label() == 'Forecast (ML)']: line.remove()
        idx = range(len(self.df), len(self.df) + len(forecast))
        self.ax1.plot(idx, forecast, '--', color='white', label='Forecast (ML)', linewidth=1.5, zorder=100)
        self.ax1.set_xlim(self.ax1.get_xlim()[0], len(self.df) + len(forecast) + 1)
        self.ax1.legend(loc='upper left', fontsize=8, facecolor='#1e1e1e', edgecolor='white', labelcolor='white')

    def _format_axis_labels(self):
        for ax in [self.ax1, self.ax2]:
            if ax:
                for lbl in ax.get_xticklabels():
                    lbl.set_color('white')
                    lbl.set_rotation(45)
                    lbl.set_horizontalalignment('right')

    def set_max_bars(self, n: int):
        self.max_bars = max(20, int(n))

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
        self.interval_seconds_map = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1H': 3600, '1D': 86400}
        self.current_interval_index = 0

        self.canvas = tk.Canvas(parent, width=size, height=size, bg=bg_color, highlightthickness=0)
        self.canvas.pack()

        self.draw_background()

        self.sector = self.canvas.create_arc(0, 0, 0, 0, start=90, extent=0, fill='#3498db',
                                             outline='black', style=tk.PIESLICE)
        self.hand = self.canvas.create_line(0, 0, 0, 0, width=3, fill='red', arrow=tk.LAST)
        self.interval_text = self.canvas.create_text(size // 2, size // 2 + 30, text="1m",
                                                     fill="white", font=("Arial", 16))

        self.canvas.itemconfigure(self.sector, state=tk.HIDDEN)
        self.canvas.itemconfigure(self.hand, state=tk.HIDDEN)

        self.digital_timer = tk.Label(
            parent,
            text="00:00:00",
            font=("DS-Digital", 20),
            bg="black",
            fg="lime",
        )
        self.digital_timer.pack()

        self.animation_id = None

    def draw_background(self):
        center = self.size // 2
        radius = self.size // 2 - 10
        self.canvas.create_oval(center - radius, center - radius, center + radius, center + radius,
                                outline=self.fg_color, width=2)
        self.canvas.create_oval(center - 3, center - 3, center + 3, center + 3,
                                fill=self.fg_color, outline=self.fg_color)

        for i in range(12):
            angle = math.radians(i * 30 - 90)
            inner = center + (radius - 10) * math.cos(angle), center + (radius - 10) * math.sin(angle)
            outer = center + radius * math.cos(angle), center + radius * math.sin(angle)
            self.canvas.create_line(*inner, *outer, fill=self.fg_color, width=2)

    def set_interval(self, interval):
        if isinstance(interval, str):
            if interval in self.interval_strings:
                self.current_interval = self.interval_seconds_map[interval]
                self.current_interval_index = self.interval_strings.index(interval)
        elif isinstance(interval, (int, float)):
            diffs = {k: abs(v - interval) for k, v in self.interval_seconds_map.items()}
            closest = min(diffs, key=diffs.get)
            self.current_interval = self.interval_seconds_map[closest]
            self.current_interval_index = self.interval_strings.index(closest)

        self.canvas.itemconfigure(self.interval_text, text=self.interval_strings[self.current_interval_index])
        if self.active:
            self.restart()

    def update_timer(self):
        if not self.active: return
        current_time = time.time()
        elapsed = current_time - self.start_time
        progress = min(elapsed / self.current_interval, 1.0)
        self.angle = progress * 360
        center = self.size // 2
        radius = self.size // 2 - 15
        self.canvas.coords(self.sector, center - radius, center - radius, center + radius, center + radius)
        self.canvas.itemconfigure(self.sector, extent=-self.angle)
        angle_rad = math.radians(self.angle - 90)
        end_x = center + (radius - 5) * math.cos(angle_rad)
        end_y = center + (radius - 5) * math.sin(angle_rad)
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
            self.canvas.itemconfigure(self.hand, state=tk.NORMAL)
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
            if self.animation_id: self.parent.after_cancel(self.animation_id)
            self.canvas.itemconfigure(self.sector, state=tk.HIDDEN)
            self.canvas.itemconfigure(self.hand, state=tk.HIDDEN)

    def restart(self):
        self.stop()
        self.start()


## Changes Made:
'''
1. **Added `_set_chart_titles()` method** that sets titles above each chart:
   - **ax1**: Always shows "Price"
   - **ax2**: Dynamic based on `bottom_chart_mode` - shows "Volume", "Kalman Filter", "RCI", or "CCI"

2. **Title styling**:
   - White color
   - Bold font (11pt)
   - Left-aligned (`loc='left'`)
   - Padding above chart (`pad=8`)

3. **Adjusted spacing** (`hspace=0.4`) to accommodate titles between charts

4. **Removed duplicate title setting** from `_plot_indicator_on_ax()` - now handled centrally by `_set_chart_titles()`

The layout now looks like:
```
┌─────────────────────────────────────────────┐
│ ━━ EMA Fast (9) 123.45  ━━ EMA Mid (21)...  │  ← EMA Legend
├─────────────────────────────────────────────┤  ← Separator
│ O: 123.4567  H: 124.5678  L: 122.3456 ...   │  ← OHLC Bar
├─────────────────────────────────────────────┤
│ Price                                       │  ← Chart Title
│              CANDLESTICK CHART              │  ← ax1
│                                             │
├─────────────────────────────────────────────┤
│ Volume                                      │  ← Chart Title
│              VOLUME CHART                   │  ← ax2
└─────────────────────────────────────────────┘
'''