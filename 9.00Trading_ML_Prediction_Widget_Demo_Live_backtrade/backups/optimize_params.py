"""
Systematic parameter optimization for the momentum strategy.
Tests a wide range of stop loss and quality threshold combinations
to find parameters that hit all 4 targets:
  - Win Rate  > 65%
  - Sharpe    > 1.0
  - Annual Rtn > 25%
  - Win/Sotena > 2.0
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from strategies.MomentumStrategy_MACD_HybridScore_Latest import (
    MOMENTUM_PARAMS, BacktestMomentumStrategy, IndicatorCalculator
)

DATA_PATH = r"C:\Users\dahab\OneDrive\Desktop\binance_SOL-USDT_15m_360d.csv"

# ── Load and prep data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], dayfirst=True)
df.rename(columns={'timestamp': 'Date'}, inplace=True)
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index, dayfirst=True)
df.rename(columns={c: c.capitalize() for c in ['open','high','low','close','volume']}, inplace=True)
print(f"Loaded {len(df):,} bars  {df.index[0]} -> {df.index[-1]}")

# ── Add indicators ─────────────────────────────────────────────────────────────
params = MOMENTUM_PARAMS.copy()
p = params
df['EMA_Fast']      = talib.EMA(df['Close'], p['ema_fast_period'])
df['EMA_Mid']       = talib.EMA(df['Close'], p['ema_mid_period'])
df['EMA_Slow']      = talib.EMA(df['Close'], p['ema_slow_period'])
df['EMA_Daily_50']  = talib.EMA(df['Close'], p.get('daily_ema_period', 4800))
df['Above_Daily_50']= (df['Close'] > df['EMA_Daily_50']).astype(bool)
df['ADX']           = talib.ADX(df['High'], df['Low'], df['Close'], p['adx_period'])
df['RSI']           = talib.RSI(df['Close'], p['rsi_period'])
df['CCI']           = talib.CCI(df['High'], df['Low'], df['Close'], p['cci_period'])
df['Volume_Ratio']  = (df['Volume'] / talib.SMA(df['Volume'], p['volume_period']).replace(0,1)).clip(0.01, 10)
df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(
    df['Close'], fastperiod=p['macd_fast'], slowperiod=p['macd_slow'], signalperiod=p['macd_signal'])
df['Momentum']        = df['Close'].pct_change(5) * 100
df['Momentum_1']      = df['Close'].pct_change(1) * 100
df['ATR']             = talib.ATR(df['High'], df['Low'], df['Close'], p['atr_period'])
df['DMP']             = talib.PLUS_DI(df['High'], df['Low'], df['Close'], p['adx_period'])
df['DMM']             = talib.MINUS_DI(df['High'], df['Low'], df['Close'], p['adx_period'])
lookback = p.get('price_percentile_lookback', 20)
df['High_20bar']   = df['High'].rolling(lookback).max()
df['Low_20bar']    = df['Low'].rolling(lookback).min()
df['Price_Range_20bar'] = df['High_20bar'] - df['Low_20bar']
df['Price_Percentile_20bar'] = np.where(
    df['Price_Range_20bar'] > 0,
    ((df['Close'] - df['Low_20bar']) / df['Price_Range_20bar']) * 100, 50.0).clip(0, 100)
df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(
    df['Close'], timeperiod=p.get('bb_period',20), nbdevup=p.get('bb_std',2.0), nbdevdn=p.get('bb_std',2.0))
df['BB_Width'] = (df['UpperBand'] - df['LowerBand']) / df['Close']
kc_period = p.get('kc_period', 20); kc_mult = p.get('kc_atr_mult', 1.5)
df['KC_Mid']   = df['Close'].ewm(span=kc_period).mean()
df['KC_ATR']   = df['ATR'].rolling(kc_period).mean()
df['KC_Upper'] = df['KC_Mid'] + kc_mult * df['KC_ATR']
df['KC_Lower'] = df['KC_Mid'] - kc_mult * df['KC_ATR']
df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']
bb_mean = df['BB_Width'].rolling(50).mean(); bb_std = df['BB_Width'].rolling(50).std().replace(0, 1)
df['BB_Z']     = (df['BB_Width'] - bb_mean) / bb_std
df['Squeeze']  = df['BB_Width'] < df['KC_Width']
tr1 = df['High'] - df['Low']; tr2 = abs(df['High'] - df['Close'].shift(1)); tr3 = abs(df['Low'] - df['Close'].shift(1))
tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr_sum = tr.rolling(window=p['chop_period']).sum()
high_max = df['High'].rolling(window=p['chop_period']).max(); low_min = df['Low'].rolling(window=p['chop_period']).min()
df['CHOP'] = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(p['chop_period'])
chop_threshold = p.get('chop_threshold', 60); min_checks = p.get('ranging_min_checks', 5)
c1 = (abs(df['Close'] - df['EMA_Fast']) / df['EMA_Fast'] <= 0.005).fillna(False)
c2 = (df['BB_Z'] < -0.5).fillna(False); c3 = df['Squeeze'].fillna(False)
c4 = (df['ATR'] < df['ATR'].rolling(100).quantile(0.25)).fillna(False)
c5 = (df['EMA_Fast'].pct_change().abs() <= 0.0005).fillna(False)
c6 = df['RSI'].between(45, 55).fillna(False); c7 = (df['CHOP'] >= chop_threshold).fillna(False)
c8 = (df['ADX'] < 20).fillna(False)
ranging_score = (c1.astype(int)+c2.astype(int)+c3.astype(int)+c4.astype(int)+c5.astype(int)+c6.astype(int)+c7.astype(int)+c8.astype(int))
df['Ranging'] = (ranging_score >= min_checks).fillna(False)
for col in ['MACD','MACD_Signal','MACD_Histogram','EMA_Fast','ADX','RSI','Volume_Ratio','Momentum','ATR','Ranging','Price_Percentile_20bar']:
    if col in df.columns:
        df[f'{col}_closed'] = df[col].shift(1)
df.rename(columns={c: c.capitalize() for c in df.columns}, inplace=True)
print(f"Indicators added. Shape: {df.shape}")

# ── Parameter grid ──────────────────────────────────────────────────────────────
# Test: stop_loss_atr_mult × quality_tier2_min × trade_direction
stop_mults     = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
quality_t2_mins = [55, 60, 65, 70, 75]
trade_dirs      = ['long', 'both']

results = []

print(f"\nRunning {len(stop_mults)*len(quality_t2_mins)*len(trade_dirs)} parameter combinations...")
print("="*80)

for direction in trade_dirs:
    for stop_mult in stop_mults:
        for quality_min in quality_t2_mins:
            # Build param override dict
            override = {
                'stop_loss_atr_mult': stop_mult,
                'trade_direction': direction,
                'quality_tier2_min': quality_min,
                'short_quality_tier2_min': quality_min,
                'fixed_threshold': quality_min + 5,
                'short_fixed_threshold': quality_min + 5,
                'tier1_adx_hard_min': 20,
                'tier1_rsi_min': 38,
                'tier1_rsi_max': 75,
                'tier1_volume_min': 0.9,
                'dmi_spread_min_long': 3.0,
                'dmi_spread_min_short': 3.0,
                'ema_trending_bars': 3,
                'macd_hist_rising_bars': 1,
                'bb_expand_required': False,
                'time_filter_enabled': False,
                'tier2_adx_min': 18,
                'tier2_volume_min': 0.6,
                'tier2_momentum_min': 0.08,
                'tier2_macd_histogram_min': 0.0,
                'tier2_require_macd_histogram': False,
                'trailing_activation_pct': 0.005,
                'trailing_distance_pct': 0.003,
                'be_stop_enabled': True,
                'be_stop_r_trigger': 2.0,
                'be_stop_no_progress_bars': 25,
                'max_hold_bars': 200,
            }

            BacktestMomentumStrategy.set_updated_params(override)

            bt = Backtest(df, BacktestMomentumStrategy, cash=50_000, commission=0.001,
                          trade_on_close=True, exclusive_orders=True)
            try:
                stats = bt.run()
            except Exception as e:
                continue

            trades = stats._trades
            if len(trades) < 5:
                continue

            pnl_col = 'PnL'
            wins   = trades[trades[pnl_col] > 0]
            losses = trades[trades[pnl_col] <= 0]
            win_rate = len(wins) / len(trades) * 100
            gross_profit = wins[pnl_col].sum()
            gross_loss   = abs(losses[pnl_col].sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_win = wins[pnl_col].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses[pnl_col].mean()) if len(losses) > 0 else 0
            sotiena = avg_win / avg_loss if avg_loss > 0 else float('inf')
            annual_return = float(stats['Return (Ann.) [%]'])
            eq_vals = stats._equity_curve['Equity'].dropna().values
            sharpe = (np.diff(eq_vals) / eq_vals[:-1]).mean() / (np.diff(eq_vals) / eq_vals[:-1]).std() * np.sqrt(96 * 252) if len(eq_vals) > 1 and np.diff(eq_vals).std() > 0 else 0

            hits = []
            if win_rate > 65: hits.append('WR')
            if sotiena > 2.0: hits.append('Sot')
            if sharpe > 1.0: hits.append('Sh')
            if annual_return > 25: hits.append('Ann')
            hit_count = len(hits)

            results.append({
                'direction': direction,
                'stop_mult': stop_mult,
                'quality_min': quality_min,
                'trades': len(trades),
                'win_rate': win_rate,
                'sotiena': sotiena,
                'sharpe': sharpe,
                'annual_return': annual_return,
                'profit_factor': profit_factor,
                'max_dd': float(stats['Max. Drawdown [%]']),
                'hits': hit_count,
                'hit_tags': '/'.join(hits),
            })

# ── Print results sorted by hits then Sharpe ────────────────────────────────────
results.sort(key=lambda x: (x['hits'], x['sharpe']), reverse=True)

print(f"\n{'DIR':<6} {'STOP':<6} {'QUAL':<6} {'N':<4} {'WR%':<7} {'Sot':<6} {'Sharpe':<8} {'Ann%':<8} {'PF':<6} {'DD%':<7} {'HITS'}")
print("="*90)
for r in results:
    row = "{:<6} {:<6.1f} {:<6} {:<4} {:<7.1f} {:<6.2f} {:<8.3f} {:<8.2f} {:<6.2f} {:<7.2f} {}".format(
        r['direction'], r['stop_mult'], r['quality_min'], r['trades'],
        r['win_rate'], r['sotiena'], r['sharpe'], r['annual_return'],
        r['profit_factor'], r['max_dd'], r['hit_tags'])
    print(row)

print(f"\nTotal combinations tested: {len(results)}")
best = results[0] if results else None
if best:
    print(f"\nBEST: direction={best['direction']}, stop_mult={best['stop_mult']}, "
          f"quality_min={best['quality_min']}, hits={best['hits']}/4, "
          f"Sharpe={best['sharpe']:.3f}, WR={best['win_rate']:.1f}%, "
          f"Ann={best['annual_return']:.2f}%, Sotiena={best['sotiena']:.2f}")
    print(f"\nRecommended MOMENTUM_PARAMS updates:")
    print(f"  'trade_direction': \"{best['direction']}\"")
    print(f"  'stop_loss_atr_mult': {best['stop_mult']}")
    print(f"  'quality_tier2_min': {best['quality_min']}")
    print(f"  'fixed_threshold': {best['quality_min'] + 5}")
