"""
Standalone baseline backtest - runs without GUI to quickly measure
current strategy performance on ETH-USDT 15m data.

Targets:
  - Win Rate  > 65%
  - Sharpe    > 1.0
  - Annual Rtn > 25%ETH
  - Win/Sotena > 2.0
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

# Force UTF-8 to handle emoji log messages from strategy
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import talib

# ── Inline minimal versions of the key strategy components ──────────────────

from backtesting import Backtest, Strategy
from strategies.MomentumStrategy_MACD_HybridScore_Latest import (
    MOMENTUM_PARAMS,
    BacktestMomentumStrategy,
    IndicatorCalculator,
    MomentumConfig,
)

# ── Load data ────────────────────────────────────────────────────────────────

DATA_PATH = r"C:\Users\dahab\OneDrive\Desktop\binance_ETH-USDT_1h_360d.csv"

df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], dayfirst=True)
df.rename(columns={'timestamp': 'Date'}, inplace=True)
df.set_index('Date', inplace=True)
# re-parse datetime index properly
df.index = pd.to_datetime(df.index, dayfirst=True)
# ensure OHLCV names match what the indicator calculator expects
df.rename(columns={
    'Open': 'open', 'High': 'high', 'Low': 'low',
    'Close': 'close', 'Volume': 'volume'
}, inplace=True)
df.sort_index(inplace=True)

print(f"Loaded {len(df):,} bars  {df.index[0]} -> {df.index[-1]}")
print(f"Columns: {list(df.columns)}")

# ── Add required indicators (same as IndicatorCalculator) ───────────────────

def add_indicators(df, params):
    df = df.copy()
    p = params
    df['EMA_Fast']      = talib.EMA(df['close'], p['ema_fast_period'])
    df['EMA_Mid']       = talib.EMA(df['close'], p['ema_mid_period'])
    df['EMA_Slow']      = talib.EMA(df['close'], p['ema_slow_period'])
    df['EMA_Daily_50']  = talib.EMA(df['close'], p.get('daily_ema_period', 4800))
    df['Above_Daily_50']= (df['close'] > df['EMA_Daily_50']).astype(bool)
    df['ADX']           = talib.ADX(df['high'], df['low'], df['close'], p['adx_period'])
    df['RSI']           = talib.RSI(df['close'], p['rsi_period'])
    df['CCI']           = talib.CCI(df['high'], df['low'], df['close'], p['cci_period'])
    vol_ma              = talib.SMA(df['volume'], p['volume_period'])
    df['Volume_Ratio']  = (df['volume'] / vol_ma.replace(0, 1)).clip(0.01, 10)
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(
        df['close'], fastperiod=p['macd_fast'], slowperiod=p['macd_slow'],
        signalperiod=p['macd_signal'])
    df['MACD_Above_Signal']    = df['MACD'] > df['MACD_Signal']
    df['MACD_Above_Zero']      = df['MACD'] > 0
    df['MACD_Histogram_Rising'] = df['MACD_Histogram'] > df['MACD_Histogram'].shift(1)
    df['Momentum']        = df['close'].pct_change(5) * 100
    df['Momentum_1']      = df['close'].pct_change(1) * 100
    df['ATR']             = talib.ATR(df['high'], df['low'], df['close'], p['atr_period'])
    df['DMP']             = talib.PLUS_DI(df['high'], df['low'], df['close'], p['adx_period'])
    df['DMM']             = talib.MINUS_DI(df['high'], df['low'], df['close'], p['adx_period'])
    lookback = p.get('price_percentile_lookback', 20)
    df['High_20bar']   = df['high'].rolling(lookback).max()
    df['Low_20bar']    = df['low'].rolling(lookback).min()
    df['Price_Range_20bar'] = df['High_20bar'] - df['Low_20bar']
    df['Price_Percentile_20bar'] = np.where(
        df['Price_Range_20bar'] > 0,
        ((df['close'] - df['Low_20bar']) / df['Price_Range_20bar']) * 100,
        50.0).clip(0, 100)

    # Bollinger + Keltner for squeeze detection
    df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(
        df['close'], timeperiod=p.get('bb_period',20),
        nbdevup=p.get('bb_std',2.0), nbdevdn=p.get('bb_std',2.0))
    df['BB_Width'] = (df['UpperBand'] - df['LowerBand']) / df['close']
    kc_period = p.get('kc_period', 20)
    kc_mult   = p.get('kc_atr_mult', 1.5)
    df['KC_Mid']   = df['close'].ewm(span=kc_period).mean()
    df['KC_ATR']   = df['ATR'].rolling(kc_period).mean()
    df['KC_Upper'] = df['KC_Mid'] + kc_mult * df['KC_ATR']
    df['KC_Lower'] = df['KC_Mid'] - kc_mult * df['KC_ATR']
    df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']
    bb_mean = df['BB_Width'].rolling(50).mean()
    bb_std  = df['BB_Width'].rolling(50).std().replace(0, 1)
    df['BB_Z']     = (df['BB_Width'] - bb_mean) / bb_std
    df['Squeeze']  = df['BB_Width'] < df['KC_Width']
    df['ATR_MA30'] = df['ATR'].rolling(30).mean()

    # Choppiness
    tr1  = df['high'] - df['low']
    tr2  = abs(df['high'] - df['close'].shift(1))
    tr3  = abs(df['low']  - df['close'].shift(1))
    tr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum   = tr.rolling(window=p['chop_period']).sum()
    high_max  = df['high'].rolling(window=p['chop_period']).max()
    low_min   = df['low'].rolling(window=p['chop_period']).min()
    df['CHOP']= 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(p['chop_period'])
    chop_threshold = p.get('chop_threshold', 60)
    min_checks    = p.get('ranging_min_checks', 5)
    c1 = (abs(df['close'] - df['EMA_Fast']) / df['EMA_Fast'] <= 0.005).fillna(False)
    c2 = (df['BB_Z'] < -0.5).fillna(False)
    c3 = df['Squeeze'].fillna(False)
    c4 = (df['ATR'] < df['ATR'].rolling(100).quantile(0.25)).fillna(False)
    c5 = (df['EMA_Fast'].pct_change().abs() <= 0.0005).fillna(False)
    c6 = df['RSI'].between(45, 55).fillna(False)
    c7 = (df['CHOP'] >= chop_threshold).fillna(False)
    c8 = (df['ADX'] < 20).fillna(False)
    ranging_score = (c1.astype(int)+c2.astype(int)+c3.astype(int)+
                     c4.astype(int)+c5.astype(int)+c6.astype(int)+
                     c7.astype(int)+c8.astype(int))
    df['Ranging'] = (ranging_score >= min_checks).fillna(False)

    # Closed-shift versions for "previous bar" lookups
    for col in ['MACD','MACD_Signal','MACD_Histogram','EMA_Fast','ADX','RSI',
                'Volume_Ratio','Momentum','ATR','Ranging','Price_Percentile_20bar']:
        if col in df.columns:
            df[f'{col}_closed'] = df[col].shift(1)

    return df

params = MOMENTUM_PARAMS.copy()
df = add_indicators(df, params)

# Ensure column names match what Strategy.next() reads (capitalise)
df.rename(columns={c: c.capitalize() for c in df.columns}, inplace=True)

# Align with backtesting library expectations (needs Open High Low Close Volume)
df.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'}, inplace=True)

# Fix index name
df.index.name = 'Date'

print(f"Indicators added. Shape: {df.shape}")

# ── Run backtest ─────────────────────────────────────────────────────────────

bt = Backtest(
    df,
    BacktestMomentumStrategy,
    cash=50_000,
    commission=0.001,
    trade_on_close=True,
    exclusive_orders=True,
)

print("\nRunning baseline backtest with default MOMENTUM_PARAMS ...")
stats = bt.run()

print("\n" + "="*60)
print("BASELINE BACKTEST RESULTS")
print("="*60)
for key, val in stats.items():
    print(f"  {key:<30} {val}")

# ── Compute extra metrics not in standard backtesting stats ──────────────────

trades = stats._trades
print(f"\nTrade columns: {list(trades.columns)}")
if len(trades) > 0:
    pnl_col = 'PnL' if 'PnL' in trades.columns else ('Profit' if 'Profit' in trades.columns else None)
    if pnl_col is None:
        print("ERROR: cannot find PnL column in trades. Available:", list(trades.columns))
    else:
        wins   = trades[trades[pnl_col] > 0]
        losses = trades[trades[pnl_col] <= 0]
        win_rate = len(wins) / len(trades) * 100
        gross_profit = wins[pnl_col].sum()
        gross_loss   = abs(losses[pnl_col].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = wins[pnl_col].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses[pnl_col].mean()) if len(losses) > 0 else 0
        sotiena  = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Annualise - use stats['Return (Ann.) [%]'] directly
        start_date = df.index[0]
        end_date   = df.index[-1]
        years = (end_date - start_date).days / 365.25
        annual_return = float(stats['Return (Ann.) [%]'])

        # Sharpe - use equity curve numeric values only
        equity_curve = stats._equity_curve
        eq_vals = equity_curve['Equity'].dropna().values
        if len(eq_vals) > 1:
            rets = np.diff(eq_vals) / eq_vals[:-1]
            sharpe = (rets.mean() / rets.std()) * np.sqrt(96 * 252) if rets.std() > 0 else 0.0
        else:
            sharpe = 0.0

        max_dd = stats['Max. Drawdown [%]']

        print(f"\n{'='*60}")
        print("EXTRA METRICS")
        print(f"{'='*60}")
        print(f"  Total Trades              : {len(trades)}")
        print(f"  Win Rate                  : {win_rate:.2f}%")
        print(f"  Profit Factor             : {profit_factor:.3f}" + (" (inf)" if profit_factor == float('inf') else ""))
        print(f"  Avg Win                   : ${avg_win:,.2f}")
        print(f"  Avg Loss                  : ${avg_loss:,.2f}")
        print(f"  Win/Sotena (Avg Win/Loss) : {sotiena:.3f}" + (" (inf)" if sotiena == float('inf') else ""))
        print(f"  Annual Return             : {annual_return:.2f}%")
        print(f"  Annualised Sharpe         : {sharpe:.3f}")
        print(f"  Max Drawdown              : {max_dd:.2f}%")
        print(f"  Best Trade                : ${trades[pnl_col].max():,.2f}")
        print(f"  Worst Trade               : ${trades[pnl_col].min():,.2f}")

        print(f"\n{'='*60}")
        print("TARGET CHECK")
        print(f"{'='*60}")
        print(f"  Win Rate   > 65%  : {'PASS' if win_rate > 65 else 'FAIL'}  ({win_rate:.2f}%)")
        print(f"  Win/Sotena > 2.0  : {'PASS' if sotiena > 2.0 else 'FAIL'}  ({sotiena:.3f})")
        print(f"  Sharpe     > 1.0  : {'PASS' if sharpe > 1.0 else 'FAIL'}  ({sharpe:.3f})")
        print(f"  Annual Rtn  > 25% : {'PASS' if annual_return > 25 else 'FAIL'}  ({annual_return:.2f}%)")
else:
    print("No trades generated!")

print("\nDone.")
