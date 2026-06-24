import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# STRATEGY CLASS (unchanged – no look‑ahead bias)
# ============================================================================

class TrendRetestStrategy:
    """
    Trend‑Following Breakout‑Retest Strategy
    Matches TradingView Pine Script logic.
    """

    def __init__(self,
                 ma_len=50,
                 ma_type='EMA',
                 use_htf_filter=True,
                 htf_tf='4H',
                 htf_ma_len=50,
                 pivot_left_bars=10,
                 pivot_right_bars=5,
                 retest_bars=10,
                 retest_tol_pct=0.3,
                 vol_len=20,
                 vol_mult=1.2,
                 use_weak_exit=True,
                 use_range_filter=True,
                 use_range_exit=True,
                 adx_len=14,
                 adx_smoothing=14,
                 adx_threshold=20,
                 atr_len=14,
                 min_atr_pct=0.1,
                 atr_stop_mult=2.0,
                 atr_trail_mult=2.5,
                 support_buffer_pct=0.1,
                 entry_confirm_pct=0.05,
                 risk_per_trade=1.0,
                 max_pos_pct=20.0,
                 initial_capital=50000,
                 commission=0.001):

        self.params = {k: v for k, v in locals().items() if k != 'self'}
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset_state()

    def reset_state(self):
        self.position = 0
        self.equity = self.initial_capital
        self.trade_log = []
        self.entry_price = None
        self.entry_bar = None
        self.key_level = None
        self.awaiting_retest = False
        self.breakout_bar = None
        self.initial_stop = None
        self.trail_stop = None
        self.highest_since_entry = None
        self.level_active = False
        self.resistance = None
        self.resistance_bar = None
        self.df_with_indicators = None
        self.original_df = None
        self.results_df = pd.DataFrame()
        self.metrics = {}

    def calculate_indicators(self, df):
        df = df.copy()
        # Moving Average
        if self.params['ma_type'] == 'EMA':
            df['ma'] = ta.ema(df['Close'], length=self.params['ma_len'])
        else:
            df['ma'] = ta.sma(df['Close'], length=self.params['ma_len'])

        # ATR
        df['atr'] = ta.atr(df['High'], df['Low'], df['Close'],
                           length=self.params['atr_len'])
        df['atr_pct'] = (df['atr'] / df['Close']) * 100

        # Volume MA
        df['vol_ma'] = ta.sma(df['Volume'], length=int(self.params['vol_len']))
        df['vol_confirmed'] = df['Volume'] > df['vol_ma'] * self.params['vol_mult']
        df['vol_weak'] = df['Volume'] < df['vol_ma']

        # ADX
        adx_result = ta.adx(df['High'], df['Low'], df['Close'],
                            length=self.params['adx_len'])
        df['adx'] = adx_result['ADX_' + str(self.params['adx_len'])]
        df['is_ranging'] = df['adx'] < self.params['adx_threshold']

        # Pivots (no look‑ahead)
        df = self.calculate_pivots_without_lookahead(df)

        df['volatility_ok'] = df['atr_pct'] >= self.params['min_atr_pct']

        # Forward fill
        for col in ['ma', 'atr', 'atr_pct', 'vol_ma', 'adx', 'pivot_high']:
            if col in df.columns:
                df[col] = df[col].ffill()
        return df

    def calculate_pivots_without_lookahead(self, df):
        left = self.params['pivot_left_bars']
        right = self.params['pivot_right_bars']
        df['pivot_high'] = np.nan
        df['pivot_bar_index'] = np.nan
        highs = df['High'].values
        n = len(df)
        for i in range(left, n - right):
            is_pivot = True
            for j in range(1, left + 1):
                if i - j < 0 or highs[i] <= highs[i - j]:
                    is_pivot = False
                    break
            if is_pivot:
                for j in range(1, right + 1):
                    if i + j >= n or highs[i] <= highs[i + j]:
                        is_pivot = False
                        break
            if is_pivot:
                df.loc[df.index[i + right], 'pivot_high'] = highs[i]
                df.loc[df.index[i + right], 'pivot_bar_index'] = i
        df['pivot_high'] = df['pivot_high'].ffill()
        df['pivot_bar_index'] = df['pivot_bar_index'].ffill()
        return df

    def run_backtest(self, df):
        self.reset_state()
        self.original_df = df.copy()
        self.df_with_indicators = self.calculate_indicators(df)
        df = self.df_with_indicators

        results = []
        start_idx = max(50, self.params['pivot_left_bars'] +
                        self.params['pivot_right_bars'])

        for i in range(start_idx, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i - 1] if i > 0 else None
            if pd.isna(current_bar['ma']) or pd.isna(current_bar['atr']):
                continue

            # Trailing stop update
            if self.position > 0:
                self.highest_since_entry = max(self.highest_since_entry, current_bar['High'])

                # Break-even logic: move stop to entry if price reaches 1 * ATR profit
                if self.highest_since_entry > self.entry_price + current_bar['atr']:
                    self.trail_stop = max(self.trail_stop, self.entry_price)

                self.trail_stop = max(self.trail_stop,
                                      self.highest_since_entry -
                                      self.params['atr_trail_mult'] * current_bar['atr'])

            self.detect_breakout(df, i, current_bar, prev_bar)

            # Entry
            if self.check_entry_conditions(df, i, current_bar, prev_bar) and self.position == 0:
                self.execute_entry(df, i, current_bar)

            # Exit
            if self.check_exit_conditions(df, i, current_bar, prev_bar) and self.position > 0:
                self.execute_exit(df, i, current_bar)

            self.update_support_line(df, i, current_bar)

            results.append({
                'date': df.index[i],
                'Close': current_bar['Close'],
                'position': self.position,
                'equity': self.equity,
                'key_level': self.key_level,
                'resistance': self.resistance,
                'pivot_high': current_bar['pivot_high'],
                'trail_stop': self.trail_stop
            })

        if results:
            self.results_df = pd.DataFrame(results).set_index('date')
        self.calculate_metrics(df)
        return self.results_df, self.trade_log

    def detect_breakout(self, df, i, current_bar, prev_bar):
        if not pd.isna(current_bar['pivot_high']):
            self.resistance = current_bar['pivot_high']
            self.resistance_bar = i - self.params['pivot_right_bars']
        if (self.resistance is not None and prev_bar is not None and
                prev_bar['Close'] <= self.resistance and current_bar['Close'] > self.resistance):
            self.key_level = self.resistance
            self.awaiting_retest = True
            self.breakout_bar = i
            self.level_active = True
            return True
        if (self.awaiting_retest and self.breakout_bar is not None and
                i - self.breakout_bar > self.params['retest_bars']):
            self.awaiting_retest = False
        return False

    def check_entry_conditions(self, df, i, current_bar, prev_bar):
        if self.position != 0:
            return False
        bullish_trend = current_bar['Close'] > current_bar['ma']
        bullish_htf = True  # placeholder
        retest_zone = (self.key_level is not None and
                       current_bar['Low'] <= self.key_level * (1 + self.params['retest_tol_pct'] / 100))

        # Stronger confirmation:
        # 1. Close must be above key level + confirm pct
        # 2. Candle must be bullish (Close > Open)
        # 3. Market must be trending (ADX > threshold)
        retest_bounce = (self.awaiting_retest and retest_zone and
                         current_bar['Close'] > self.key_level * (1 + self.params['entry_confirm_pct'] / 100) and
                         current_bar['Close'] > current_bar['Open'])

        vol_confirmed = current_bar['vol_confirmed']
        volatility_ok = current_bar['volatility_ok']
        not_ranging = not self.params['use_range_filter'] or not current_bar['is_ranging']

        return (bullish_trend and bullish_htf and retest_bounce and vol_confirmed and
                volatility_ok and not_ranging)

    def update_support_line(self, df, i, current_bar):
        if self.key_level is not None and self.position > 0 and current_bar['Close'] < self.key_level:
            self.level_active = False

    def execute_entry(self, df, i, current_bar):
        atr_val = current_bar['atr']
        stop_dist = self.params['atr_stop_mult'] * atr_val
        risk_amt = self.equity * (self.params['risk_per_trade'] / 100)
        qty_raw = risk_amt / stop_dist
        qty_cap = (self.equity * self.params['max_pos_pct'] / 100) / current_bar['Close']
        qty = min(qty_raw, qty_cap)
        entry_price = current_bar['Close']
        commission_cost = entry_price * qty * self.commission

        self.position = qty
        self.equity -= commission_cost
        self.entry_price = entry_price
        self.entry_bar = i
        self.initial_stop = entry_price - stop_dist
        self.trail_stop = self.initial_stop
        self.highest_since_entry = entry_price
        self.awaiting_retest = False

        self.trade_log.append({
            'entry_date': df.index[i],
            'entry_price': entry_price,
            'quantity': qty,
            'stop_loss': self.initial_stop,
            'risk_amount': risk_amt
        })
        print(f"BUY @ {entry_price:.2f} | Qty: {qty:.4f} | Stop: {self.initial_stop:.2f}")

    def check_exit_conditions(self, df, i, current_bar, prev_bar):
        if self.position <= 0:
            return False
        support_broken = (self.key_level is not None and current_bar['Close'] < self.key_level * (1 - self.params['support_buffer_pct'] / 100))
        trail_hit = (self.trail_stop is not None and current_bar['Close'] < self.trail_stop)
        return (support_broken or trail_hit)

    def execute_exit(self, df, i, current_bar):
        exit_price = current_bar['Close']
        pnl = (exit_price - self.entry_price) * self.position
        commission_cost = exit_price * self.position * self.commission
        net_pnl = pnl - commission_cost
        self.equity += net_pnl

        exit_reason = self.get_exit_reason(df, i, current_bar)
        if self.trade_log:
            self.trade_log[-1].update({
                'exit_date': df.index[i],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': net_pnl,
                'pnl_pct': (net_pnl / (self.entry_price * self.position)) * 100
            })
        print(f"SELL @ {exit_price:.2f} | PnL: {net_pnl:.2f} ({exit_reason})")
        self.position = 0
        self.entry_price = None
        self.initial_stop = None
        self.trail_stop = None
        self.highest_since_entry = None

    def get_exit_reason(self, df, i, current_bar):
        if self.key_level is not None and current_bar['Close'] < self.key_level * (1 - self.params['support_buffer_pct'] / 100):
            return 'support_broken'
        elif self.trail_stop is not None and current_bar['Close'] < self.trail_stop:
            return 'trailing_stop'
        else:
            return 'unknown'

    def calculate_metrics(self, df):
        trades_df = pd.DataFrame(self.trade_log)
        if trades_df.empty:
            print("\n" + "=" * 50)
            print("NO TRADES EXECUTED")
            print("=" * 50)
            print("Adjust parameters to generate signals.")
            self.metrics = {'total_trades': 0, 'final_equity': self.equity}
            return

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()

        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': trades_df['pnl'].mean(),
            'max_pnl': trades_df['pnl'].max(),
            'min_pnl': trades_df['pnl'].min(),
            'final_equity': self.equity,
            'total_return': ((self.equity - self.initial_capital) / self.initial_capital) * 100
        }

        print("\n" + "=" * 50)
        print("PERFORMANCE METRICS")
        print("=" * 50)
        for k, v in self.metrics.items():
            if isinstance(v, float):
                if 'rate' in k or 'return' in k:
                    print(f"{k.replace('_', ' ').title()}: {v:.2f}%")
                else:
                    print(f"{k.replace('_', ' ').title()}: {v:.2f}")
            else:
                print(f"{k.replace('_', ' ').title()}: {v}")
        print("\nTrade Summary:")
        print(trades_df[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl', 'pnl_pct',
                         'exit_reason']].to_string())

    def plot_results(self):
        if self.results_df.empty or self.original_df is None:
            print("No data to plot.")
            return

        df = self.original_df
        results = self.results_df

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=('Price & Signals', 'Position', 'Equity'))

        # Price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue')), row=1,
                      col=1)

        # MA
        if self.df_with_indicators is not None and 'ma' in self.df_with_indicators.columns:
            fig.add_trace(go.Scatter(x=self.df_with_indicators.index, y=self.df_with_indicators['ma'],
                                     mode='lines', name='MA50', line=dict(color='orange', dash='dash')), row=1, col=1)

        # Key level (becomes support once retested)
        if 'key_level' in results.columns:
            fig.add_trace(go.Scatter(x=results.index, y=results['key_level'],
                                     mode='lines', name='Key Level (Support)', line=dict(color='red', dash='dot')), row=1, col=1)

        # Active resistance level (most recent pivot high)
        if 'resistance' in results.columns:
            fig.add_trace(go.Scatter(x=results.index, y=results['resistance'],
                                     mode='lines', name='Resistance', line=dict(color='purple', dash='dot')), row=1, col=1)

        # Pivot highs themselves (markers at the actual swing points)
        pivots = self.df_with_indicators[['pivot_high']].dropna()
        if not pivots.empty:
            fig.add_trace(go.Scatter(x=pivots.index, y=pivots['pivot_high'],
                                     mode='markers', name='Pivot Highs',
                                     marker=dict(symbol='diamond', size=7, color='magenta')), row=1, col=1)

        # Trail stop
        if 'trail_stop' in results.columns:
            fig.add_trace(go.Scatter(x=results.index, y=results['trail_stop'],
                                     mode='lines', name='Trail Stop', line=dict(color='green', dash='dash')), row=1,
                          col=1)

        # Trades
        trades_df = pd.DataFrame(self.trade_log)
        if not trades_df.empty:
            fig.add_trace(go.Scatter(x=trades_df['entry_date'], y=trades_df['entry_price'],
                                     mode='markers', name='Entry',
                                     marker=dict(symbol='triangle-up', size=15, color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=trades_df['exit_date'], y=trades_df['exit_price'],
                                     mode='markers', name='Exit',
                                     marker=dict(symbol='triangle-down', size=15, color='red')), row=1, col=1)

        # Position
        fig.add_trace(go.Scatter(x=results.index, y=results['position'],
                                 mode='lines', name='Position', fill='tozeroy', line=dict(color='purple')), row=2,
                      col=1)
        # Equity
        fig.add_trace(go.Scatter(x=results.index, y=results['equity'],
                                 mode='lines', name='Equity', line=dict(color='green')), row=3, col=1)
        fig.add_hline(y=self.initial_capital, line_dash="dash", line_color="gray", row=3, col=1,
                      annotation_text="Initial Capital")

        fig.update_layout(height=900, title_text="Trend Retest Strategy Backtest", showlegend=True)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Position", row=2, col=1)
        fig.update_yaxes(title_text="Equity", row=3, col=1)
        fig.show()


# ============================================================================
# DATA LOADING HELPER (NOW WITH 'timestamp' SUPPORT)
# ============================================================================

def load_data(file_path, date_column=None):
    """
    Load CSV and automatically set a datetime index.
    Recognizes 'timestamp' as a valid date column.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # If date_column not provided, try to detect it
    if date_column is None:
        date_names = ['date', 'Date', 'DATE', 'datetime', 'DateTime',
                      'timestamp', 'Timestamp', 'time', 'Time', 'index']
        for col in date_names:
            if col in df.columns:
                date_column = col
                break
        # If still None, try to infer from first column
        if date_column is None:
            try:
                pd.to_datetime(df[df.columns[0]])
                date_column = df.columns[0]
                print(f"Inferred date column: '{date_column}'")
            except:
                print("No date column found. Using row index.")
                return df

    # Parse and set index
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        # Drop rows where date parsing failed
        df = df.dropna(subset=[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main routine: tries to load 'your_data.csv' with timestamp column,
    falls back to sample data if file not found.
    """
    # Try to load your actual data file
    try:
        df = load_data('C:/Users/dahab/PyCharm_2026.2.23/New_Bollinger_bands/9.00Trading_ML_Prediction_Widget_Demo_Live_backtrade/data_cache/2026.csv', date_column='timestamp')
        print("Using your data file.")
    except FileNotFoundError:
        print("File 'your_data.csv' not found. Generating sample data instead.")
        # Sample data generation
        np.random.seed(42)
        dates = pd.date_range('01-05-2026', '31-12-2029', freq='D')
        n = len(dates)
        time = np.arange(n)
        trend = time * 0.02
        cycle = 10 * np.sin(time * 2 * np.pi / 252)
        noise = np.random.randn(n) * 3
        price = 100 + trend + cycle + noise
        volatility = 1 + np.abs(np.random.randn(n) * 0.5)
        price = price + np.cumsum(np.random.randn(n) * volatility * 0.2)
        df = pd.DataFrame({
            'open': price * (1 + np.random.randn(n) * 0.005),
            'High': price * (1 + np.abs(np.random.randn(n) * 0.015)),
            'Low': price * (1 - np.abs(np.random.randn(n) * 0.015)),
            'Close': price,
            'Volume': 1000000 + np.random.randint(0, 2000000, n)
        }, index=dates)
        # Ensure OHLC validity
        df['High'] = df[['open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['open', 'Close', 'Low']].min(axis=1)
        print("Sample data generated.")

    # Initialize strategy (you can tweak parameters)
    strategy = TrendRetestStrategy(
        pivot_left_bars=10,
        pivot_right_bars=5,
        retest_tol_pct=0.3,
        vol_mult=1.2,
        min_atr_pct=0.05,
        use_range_filter=True,
        atr_stop_mult=2.0,
        atr_trail_mult=3.0,
        support_buffer_pct=0.1,
        entry_confirm_pct=0.05,
        # ... other parameters as needed
    )

    # Run backtest
    results_df, trade_log = strategy.run_backtest(df)

    # Show results
    if strategy.trade_log:
        strategy.plot_results()
    else:
        print("\nNo trades executed. Try adjusting parameters.")
        # Optionally show diagnostic info
        if strategy.df_with_indicators is not None:
            d = strategy.df_with_indicators
            print("\nDiagnostic stats:")
            print(f"Pivots detected: {d['pivot_high'].notna().sum()}")
            print(f"Volume confirmed: {d['vol_confirmed'].sum()}")
            print(f"Volatility OK: {d['volatility_ok'].sum()}")
            print(f"Not ranging: {(~d['is_ranging']).sum()}")

    return strategy, results_df, trade_log


if __name__ == "__main__":
    strategy, results_df, trade_log = main()