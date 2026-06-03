#!/usr/bin/env python3
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROFESSIONAL SCALPING TRADER v5.1                         │
│                                                                              │
│  VWAP CORE STRATEGY - FULLY WORKING WITH TRADES & BACKTEST                   │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import threading
import time
import tkinter as tk
import matplotlib.dates as mdates
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto

import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class GlobalConfig:
    INITIAL_CAPITAL = 50000.0
    DEFAULT_SYMBOL = "ETH-USDT"
    ACTIVE_TIMEFRAME = "5m"
    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
    TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


VWAP_STRATEGY_CONFIG = {
    "risk_reclaim": 0.004,
    "risk_bounce": 0.005,
    "risk_discount": 0.003,
    "atr_period": 7,
    "stop_atr_mult": 1.0,
    "trailing_atr_mult": 0.8,
    "target_atr_mult": 1.0,
    "supertrend_period": 7,
    "supertrend_multiplier": 2.0,
    "adx_min": 5,
    "ema_fast": 9,
    "ema_slow": 21,
    "trend_signal_age_min": 0, #2
    "stochastic_k": 5,
    "stochastic_d": 3,
    "rsi_period": 9,
    "volume_period": 20,
    "volume_min_ratio": 0.30,# 0.55
    "poc_avoidance_pct": 0.0025,
    "rsi_moderate_min": 38,
    "rsi_moderate_max": 72,
    "adx_baseline": 14,
    "adx_bounce_extra": 1,
    "vwap_band_stdev": 1.0,
    "partial_profit_pct": 0.55,
    "max_hold_bars": 60,
    "cooldown_bars": 5,
    "max_consecutive_losses": 4,
    "loss_cooldown_bars": 6,
    "max_leverage": 8,
    "enable_reclaim": True,
    "enable_bounce": True,
    "enable_discount": True,
    "trade_direction": "both",
}


class StrategyState(Enum):
    SEEKING_ENTRY = auto()
    IN_TRADE = auto()
    COOLDOWN = auto()


class SetupType(Enum):
    VWAP_RECLAIM = "VWAP_Reclaim"
    VWAP_BAND_BOUNCE = "VWAP_Band_Bounce"
    DISCOUNT_PULLBACK = "Discount_Pullback"


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorCalculator:

    @staticmethod
    def ema(data, period):
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)

        atr = IndicatorCalculator.atr(high, low, close, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k = 100 * ((close - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k.fillna(50), d.fillna(50)

    @staticmethod
    def supertrend(high, low, close, period=10, multiplier=3):

        atr = IndicatorCalculator.atr(high, low, close, period)

        hl2 = (high + low) / 2

        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)

        direction = pd.Series(index=close.index, dtype=int)

        direction.iloc[0] = 1

        for i in range(1, len(close)):

            if close.iloc[i] > upperband.iloc[i - 1]:
                direction.iloc[i] = 1

            elif close.iloc[i] < lowerband.iloc[i - 1]:
                direction.iloc[i] = -1

            else:
                direction.iloc[i] = direction.iloc[i - 1]

        return direction.fillna(1)

    @staticmethod
    def calculate_all(df, config):
        df = df.copy()
        if len(df) < 50:
            return df

        # EMAs
        df['EMA_9'] = IndicatorCalculator.ema(df['Close'], config['ema_fast'])
        df['EMA_21'] = IndicatorCalculator.ema(df['Close'], config['ema_slow'])

        # RSI
        df['RSI'] = IndicatorCalculator.rsi(df['Close'], config['rsi_period'])

        # ATR
        df['ATR'] = IndicatorCalculator.atr(df['High'], df['Low'], df['Close'], config['atr_period'])

        # ADX
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = IndicatorCalculator.adx(df['High'], df['Low'], df['Close'], 14)

        # Volume Ratio
        df['Volume_MA'] = df['Volume'].rolling(window=config['volume_period']).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, 1)

        # Supertrend
        df['Super_Direction'] = IndicatorCalculator.supertrend(
            df['High'], df['Low'], df['Close'], config['supertrend_period'], config['supertrend_multiplier']
        )

        # Trend Age
        df['Trend_Age'] = 0
        for i in range(1, len(df)):
            if df['Super_Direction'].iloc[i] == df['Super_Direction'].iloc[i - 1]:
                df.loc[df.index[i], 'Trend_Age'] = df['Trend_Age'].iloc[i - 1] + 1
            else:
                df.loc[df.index[i], 'Trend_Age'] = 1

        # Stochastic
        k, d = IndicatorCalculator.stochastic(df['High'], df['Low'], df['Close'], 14, 3)
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        # VWAP
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Cum_TPV'] = (df['Typical'] * df['Volume']).cumsum()
        df['Cum_Vol'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cum_TPV'] / df['Cum_Vol']
        df['VWAP_Dev'] = df['Typical'] - df['VWAP']
        df['VWAP_Var'] = ((df['VWAP_Dev'] ** 2) * df['Volume']).cumsum() / df['Cum_Vol']
        df['VWAP_Stdev'] = np.sqrt(df['VWAP_Var'].clip(lower=0))
        df['VWAP_High'] = df['VWAP'] + df['VWAP_Stdev'] * config['vwap_band_stdev']
        df['VWAP_Low'] = df['VWAP'] - df['VWAP_Stdev'] * config['vwap_band_stdev']

        # POC
        df['POC'] = df['Close'].rolling(window=20).mean()

        # Shifted values for signal detection
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_VWAP_Low'] = df['VWAP_Low'].shift(1)
        df['Prev_VWAP_High'] = df['VWAP_High'].shift(1)
        df['Prev_RSI'] = df['RSI'].shift(1)

        return df.fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
# VWAP CORE STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class VWAPCoreStrategy:

    def __init__(self, config: dict):
        self.config = config.copy()
        self.balance = GlobalConfig.INITIAL_CAPITAL
        self.position = None
        self.trades = []
        self.state = StrategyState.SEEKING_ENTRY
        self.consecutive_losses = 0
        self.cooldown_counter = 0
        self.log_callback = None
        self.current_data = None

    def set_log_callback(self, callback):
        self.log_callback = callback

    def log(self, message, color='white'):
        if self.log_callback:
            self.log_callback(message, color)

    def check_entry(self, data):
        """RELAXED entry logic - generates real trades"""

        if self.state != StrategyState.SEEKING_ENTRY:
            return 'hold', f"State: {self.state}", 0

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return 'hold', f"Cooldown: {self.cooldown_counter}", 0

        price = data.get('Close', 0)

        vwap = data.get('VWAP', price)
        prev_vwap = data.get('Prev_VWAP', vwap)

        vwap_low = data.get('VWAP_Low', price)
        vwap_high = data.get('VWAP_High', price)

        prev_close = data.get('Prev_Close', price)

        rsi = data.get('RSI', 50)
        prev_rsi = data.get('Prev_RSI', 50)

        vol_ratio = data.get('Volume_Ratio', 1)

        adx = data.get('ADX', 20)

        super_dir = data.get('Super_Direction', 1)

        ema9 = data.get('EMA_9', price)
        ema21 = data.get('EMA_21', price)

        stoch_k = data.get('Stoch_K', 50)
        stoch_d = data.get('Stoch_D', 50)

        # MUCH MORE RELAXED FILTERS
        is_bullish = (
                super_dir == 1
                or ema9 > ema21
        )

        is_bearish = (
                super_dir == -1
                or ema9 < ema21
        )

        trade_dir = self.config.get('trade_direction', 'both')

        # =========================================================
        # LONG ENTRIES
        # =========================================================

        if is_bullish and trade_dir in ('both', 'long'):

            # VWAP reclaim
            if (
                    price > vwap
                    and prev_close <= prev_vwap
                    and rsi > 40
                    and vol_ratio > 0.3
            ):
                return 'buy', "VWAP Reclaim LONG", 70

            # Bounce
            if (
                    price >= vwap_low
                    and rsi > 35
                    and stoch_k >= stoch_d
            ):
                return 'buy', "Band Bounce LONG", 60

            # EMA continuation
            if (
                    ema9 > ema21
                    and rsi > 45
                    and adx > 5
            ):
                return 'buy', "EMA Trend LONG", 55

        # =========================================================
        # SHORT ENTRIES
        # =========================================================

        if is_bearish and trade_dir in ('both', 'short'):

            # VWAP reclaim short
            if (
                    price < vwap
                    and prev_close >= prev_vwap
                    and rsi < 60
                    and vol_ratio > 0.3
            ):
                return 'sell_short', "VWAP Reclaim SHORT", 70

            # Bounce short
            if (
                    price <= vwap_high
                    and rsi < 65
                    and stoch_k <= stoch_d
            ):
                return 'sell_short', "Band Bounce SHORT", 60

            # EMA continuation short
            if (
                    ema9 < ema21
                    and rsi < 55
                    and adx > 5
            ):
                return 'sell_short', "EMA Trend SHORT", 55

        return 'hold', "No signal", 0

    def calculate_position_size(self, price, atr, setup_type):
        if setup_type == SetupType.VWAP_RECLAIM:
            risk_pct = self.config['risk_reclaim']
        elif setup_type == SetupType.VWAP_BAND_BOUNCE:
            risk_pct = self.config['risk_bounce']
        else:
            risk_pct = self.config['risk_discount']

        risk_amount = self.balance * risk_pct
        stop_distance = atr * self.config['stop_atr_mult']

        if stop_distance <= 0:
            stop_distance = price * 0.01

        size = risk_amount / stop_distance
        max_notional = self.balance * self.config['max_leverage']
        max_size = max_notional / price

        return max(min(size, max_size), 0.001)

    def execute_entry(self, action, price, setup_type, reason):
        atr = self.current_data.get('ATR', price * 0.01) if self.current_data else price * 0.01
        if atr <= 0:
            atr = price * 0.01

        quantity = self.calculate_position_size(price, atr, setup_type)

        if quantity <= 0:
            return False

        stop_distance = atr * self.config['stop_atr_mult']

        if action == 'buy':
            stop_loss = price - stop_distance
            pos_type = 'long'
            target = price + atr * self.config['target_atr_mult']
        else:
            stop_loss = price + stop_distance
            pos_type = 'short'
            target = price - atr * self.config['target_atr_mult']

        self.position = {
            'type': pos_type,
            'entry': price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'target': target,
            'setup_type': setup_type,
            'entry_time': datetime.now(),
            'bars_held': 0,
            'partial_taken': False,
            'trailing_stop': 0,
            'highest': price,
            'lowest': price
        }
        self.state = StrategyState.IN_TRADE

        arrow = '📈' if action == 'buy' else '📉'
        self.log(f"{arrow} ENTRY {pos_type.upper()} @ ${price:.2f} | {setup_type.value} | {reason}",
                 'green' if action == 'buy' else 'red')
        return True

    def check_exit(self, data, current_price):
        if self.state != StrategyState.IN_TRADE or not self.position:
            return None, ""

        pos = self.position
        atr = data.get('ATR', 0.01)

        if pos['type'] == 'long':
            pos['highest'] = max(pos['highest'], current_price)
        else:
            pos['lowest'] = min(pos['lowest'], current_price)

        pos['bars_held'] += 1

        # Stop loss
        if pos['type'] == 'long' and current_price <= pos['stop_loss']:
            return 'sell', "Stop loss"
        if pos['type'] == 'short' and current_price >= pos['stop_loss']:
            return 'buy_cover', "Stop loss"

        # Partial profit
        if not pos['partial_taken']:
            if pos['type'] == 'long' and current_price >= pos['target']:
                return 'partial', "Target 1 hit"
            if pos['type'] == 'short' and current_price <= pos['target']:
                return 'partial', "Target 1 hit"

        # Trailing stop
        if pos['partial_taken']:
            trail_dist = atr * self.config['trailing_atr_mult']
            if pos['type'] == 'long':
                new_trail = pos['highest'] - trail_dist
                if new_trail > pos['trailing_stop']:
                    pos['trailing_stop'] = new_trail
                if current_price <= pos['trailing_stop']:
                    return 'sell', "Trailing stop"
            else:
                new_trail = pos['lowest'] + trail_dist
                if new_trail < pos['trailing_stop'] or pos['trailing_stop'] == 0:
                    pos['trailing_stop'] = new_trail
                if current_price >= pos['trailing_stop']:
                    return 'buy_cover', "Trailing stop"

        # Max hold
        if pos['bars_held'] >= self.config['max_hold_bars']:
            return 'sell' if pos['type'] == 'long' else 'buy_cover', "Max hold"

        return None, ""

    def execute_exit(self, action, price, reason, is_partial=False):
        if not self.position:
            return 0

        pos = self.position
        entry = pos['entry']
        quantity = pos['quantity'] if not is_partial else pos['quantity'] * self.config['partial_profit_pct']

        if pos['type'] == 'long':
            pnl = (price - entry) * quantity
        else:
            pnl = (entry - price) * quantity

        if is_partial:
            pos['quantity'] -= quantity
            pos['partial_taken'] = True

            if pos['type'] == 'long':
                pos['trailing_stop'] = price - (self.current_data.get('ATR', 0.01) * self.config['trailing_atr_mult'])
                pos['stop_loss'] = price - (self.current_data.get('ATR', 0.01) * self.config['stop_atr_mult'] * 1.5)
            else:
                pos['trailing_stop'] = price + (self.current_data.get('ATR', 0.01) * self.config['trailing_atr_mult'])
                pos['stop_loss'] = price + (self.current_data.get('ATR', 0.01) * self.config['stop_atr_mult'] * 1.5)

            self.log(f"📊 PARTIAL EXIT @ ${price:.2f} | PnL: ${pnl:.2f} | Remaining: {pos['quantity']:.4f} | {reason}",
                     'orange')
            return pnl

        # Full exit
        self.trades.append({
            'type': pos['type'],
            'entry': entry,
            'exit': price,
            'quantity': pos['quantity'],
            'setup_type': pos['setup_type'].value if pos['setup_type'] else 'Unknown',
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_pct': (pnl / (entry * pos['quantity'])) * 100 if entry * pos['quantity'] > 0 else 0,
            'exit_reason': reason,
            'bars_held': pos['bars_held']
        })

        self.balance += pnl

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config['max_consecutive_losses']:
                self.state = StrategyState.COOLDOWN
                self.cooldown_counter = self.config['loss_cooldown_bars']
        else:
            self.consecutive_losses = 0

        self.position = None

        if self.state != StrategyState.COOLDOWN:
            self.state = StrategyState.SEEKING_ENTRY

        color = 'green' if pnl > 0 else 'red'
        arrow = '✅' if pnl > 0 else '❌'
        self.log(f"{arrow} EXIT {pos['type'].upper()} @ ${price:.2f} | PnL: ${pnl:.2f} | {reason}", color)

        return pnl

    def update_data(self, data):
        self.current_data = data

    def get_stats(self):
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'balance': self.balance,
                'roi': 0,
                'profit_factor': 0,
                'consecutive_losses': self.consecutive_losses
            }

        winning = [t for t in self.trades if t['pnl'] > 0]
        losing = [t for t in self.trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0

        total_wins = sum(t['pnl'] for t in winning) if winning else 0
        total_losses = abs(sum(t['pnl'] for t in losing)) if losing else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'balance': self.balance,
            'roi': (self.balance - GlobalConfig.INITIAL_CAPITAL) / GlobalConfig.INITIAL_CAPITAL * 100,
            'profit_factor': profit_factor,
            'consecutive_losses': self.consecutive_losses
        }


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════

class DataFetcher:

    @staticmethod
    def generate_synthetic_data(timeframe='5m', periods=200, start_price=2000):
        """
        Generate NON-DETERMINISTIC synthetic market data.
        Fixed:
        - Removed fixed random seed
        - Timeframe now properly affects candles
        - More realistic volatility/trend behavior
        """

        freq_map = {
            '1m': 'min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': 'h',
            '2h': '2h',
            '4h': '4h',
            '1d': 'D'
        }

        freq = freq_map.get(timeframe, '5min')

        # DIFFERENT DATA EVERY RUN
        np.random.seed(int(time.time() * 1000) % 1000000)

        dates = pd.date_range(
            end=datetime.now(),
            periods=periods,
            freq=freq
        )

        # Timeframe volatility scaling
        tf_volatility = {
            '1m': 0.002,
            '5m': 0.004,
            '15m': 0.007,
            '30m': 0.010,
            '1h': 0.015,
            '2h': 0.020,
            '4h': 0.030,
            '1d': 0.050
        }

        volatility = tf_volatility.get(timeframe, 0.004)

        # Symbol-style trending behavior
        price = [start_price]

        current_trend = np.random.choice([-1, 1])
        trend_strength = np.random.uniform(0.0001, 0.001)

        for i in range(1, periods):

            # Occasionally flip trend
            if np.random.random() < 0.03:
                current_trend *= -1
                trend_strength = np.random.uniform(0.0001, 0.0015)

            drift = current_trend * trend_strength
            shock = np.random.normal(0, volatility)

            change = drift + shock

            new_price = price[-1] * (1 + change)

            # Prevent invalid prices
            new_price = max(new_price, start_price * 0.2)

            price.append(new_price)

        price = np.array(price)

        # Create realistic OHLC candles
        open_prices = np.roll(price, 1)
        open_prices[0] = price[0]

        candle_noise = volatility * 1.5

        highs = np.maximum(open_prices, price) * (
                1 + np.abs(np.random.normal(0, candle_noise, periods))
        )

        lows = np.minimum(open_prices, price) * (
                1 - np.abs(np.random.normal(0, candle_noise, periods))
        )

        volumes = np.random.uniform(10000, 100000, periods)

        # Volume spikes during volatility
        returns = np.abs(np.diff(np.concatenate([[0], price])))
        volumes *= (1 + returns / np.mean(returns + 1e-9))

        df = pd.DataFrame({
            'Open': open_prices,
            'High': highs,
            'Low': lows,
            'Close': price,
            'Volume': volumes
        }, index=dates)

        # Ensure valid OHLC structure
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

        return df

    @staticmethod
    def fetch_live_data(symbol, timeframe, limit=200):
        """Fetch real data from OKX"""
        tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                  '1h': '1H', '2h': '2H', '4h': '4H', '1d': '1D'}
        try:
            url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={tf_map.get(timeframe, '5m')}&limit={limit}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '0' and data.get('data'):
                    df = pd.DataFrame(data['data'],
                                      columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume', 'volCcy', 'volBase',
                                               'turnover'])
                    df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
                    df.set_index('ts', inplace=True)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                    return df.sort_index()
            return None
        except Exception as e:
            print(f"Fetch error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class BacktestEngine:

    @staticmethod
    def run(df, config, initial_capital=50000):
        """Run simple backtest on historical data"""
        strategy = VWAPCoreStrategy(config.copy())
        strategy.balance = initial_capital

        df = IndicatorCalculator.calculate_all(df, config)

        trades = []

        for i in range(50, len(df)):
            current_data = df.iloc[i].to_dict()
            prev_data = df.iloc[i - 1].to_dict() if i > 0 else current_data

            data_point = {
                'Close': current_data['Close'],
                'Prev_Close': prev_data['Close'],
                'VWAP': current_data.get('VWAP', current_data['Close']),
                'Prev_VWAP': prev_data.get('VWAP', prev_data['Close']),

                'VWAP_High': current_data.get('VWAP_High', current_data['Close']),
                'VWAP_Low': current_data.get('VWAP_Low', current_data['Close']),

                'Prev_VWAP_Low': prev_data.get('VWAP_Low', prev_data['Close']),
                'Prev_VWAP_High': prev_data.get('VWAP_High', prev_data['Close']),
                'EMA_9': current_data.get('EMA_9', 0),
                'EMA_21': current_data.get('EMA_21', 0),
                'RSI': current_data.get('RSI', 50),
                'Prev_RSI': prev_data.get('RSI', 50),
                'ATR': current_data.get('ATR', 0.01),
                'ADX': current_data.get('ADX', 0),
                'Volume_Ratio': current_data.get('Volume_Ratio', 0.5),
                'Super_Direction': current_data.get('Super_Direction', 1),
                'Trend_Age': current_data.get('Trend_Age', 1),
                'Stoch_K': current_data.get('Stoch_K', 50),
                'Stoch_D': current_data.get('Stoch_D', 50),
            }

            strategy.update_data(data_point)
            current_price = current_data['Close']

            if i % 100 == 0:
                print(
                    f"Bar {i} | "
                    f"Price={current_price:.2f} "
                    f"VWAP={data_point['VWAP']:.2f} "
                    f"ADX={data_point['ADX']:.2f} "
                    f"RSI={data_point['RSI']:.2f} "
                    f"TrendAge={data_point['Trend_Age']} "
                    f"VolRatio={data_point['Volume_Ratio']:.2f}"
                )

            # Check exit
            if strategy.state == StrategyState.IN_TRADE and strategy.position:
                exit_action, reason = strategy.check_exit(data_point, current_price)
                if exit_action:
                    strategy.execute_exit(exit_action, current_price, reason, is_partial=(exit_action == 'partial'))

            # Check entry
            if strategy.state == StrategyState.SEEKING_ENTRY:
                action, reason, quality = strategy.check_entry(data_point)
                if action in ('buy', 'sell_short'):
                    setup_type = SetupType.VWAP_RECLAIM
                    if "Band Bounce" in reason:
                        setup_type = SetupType.VWAP_BAND_BOUNCE
                    elif "Discount" in reason:
                        setup_type = SetupType.DISCOUNT_PULLBACK
                    strategy.execute_entry(action, current_price, setup_type, reason)
                    print(f"ENTRY: {action} {current_price:.2f} {reason}")

        return strategy.get_stats(), strategy.trades


# ═══════════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingTraderApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Professional Scalping Trader v5.1 - VWAP Core Strategy")
        self.root.geometry("1400x850")
        self.root.configure(bg='#1a1a2e')

        self.running = False
        self.mode = 'demo'
        self.strategy = VWAPCoreStrategy(VWAP_STRATEGY_CONFIG.copy())
        self.strategy.set_log_callback(self.log_message)
        self.current_data = None
        self.df = None
        self.trading_thread = None

        self.setup_ui()
        self.update_stats_display()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_panel = tk.Frame(main_frame, width=420, bg='#1a1a2e')
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        right_panel = tk.Frame(main_frame, bg='#1a1a2e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.configure('TLabelframe', background='#1a1a2e', foreground='white')
        style.configure('TLabelframe.Label', background='#1a1a2e', foreground='white')
        style.configure('TLabel', background='#1a1a2e', foreground='white')

        # Settings Frame
        settings_frame = ttk.LabelFrame(left_panel, text="⚙️ Trading Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Row 0
        tk.Label(settings_frame, text="Mode:", bg='#1a1a2e', fg='white').grid(row=0, column=0, sticky='w', padx=5,
                                                                              pady=2)
        self.mode_var = tk.StringVar(value="Demo")
        mode_combo = ttk.Combobox(settings_frame, textvariable=self.mode_var, values=['Demo', 'Live'], width=12)
        mode_combo.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)

        tk.Label(settings_frame, text="Symbol:", bg='#1a1a2e', fg='white').grid(row=0, column=2, sticky='w', padx=5,
                                                                                pady=2)
        self.symbol_var = tk.StringVar(value=GlobalConfig.DEFAULT_SYMBOL)
        ttk.Entry(settings_frame, textvariable=self.symbol_var, width=12).grid(row=0, column=3, sticky='w', padx=5,
                                                                               pady=2)

        # Row 1
        tk.Label(settings_frame, text="Timeframe:", bg='#1a1a2e', fg='white').grid(row=1, column=0, sticky='w', padx=5,
                                                                                   pady=2)
        self.timeframe_var = tk.StringVar(value=GlobalConfig.ACTIVE_TIMEFRAME)
        tf_combo = ttk.Combobox(settings_frame, textvariable=self.timeframe_var, values=GlobalConfig.TIMEFRAMES,
                                width=12)
        tf_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        tf_combo.bind('<<ComboboxSelected>>', self.on_timeframe_change)

        tk.Label(settings_frame, text="Direction:", bg='#1a1a2e', fg='white').grid(row=1, column=2, sticky='w', padx=5,
                                                                                   pady=2)
        self.direction_var = tk.StringVar(value=VWAP_STRATEGY_CONFIG['trade_direction'])
        dir_combo = ttk.Combobox(settings_frame, textvariable=self.direction_var, values=['both', 'long', 'short'],
                                 width=12)
        dir_combo.grid(row=1, column=3, sticky='w', padx=5, pady=2)
        dir_combo.bind('<<ComboboxSelected>>', self.on_direction_change)

        # Row 2 - Setup Types
        tk.Label(settings_frame, text="Setup Types:", bg='#1a1a2e', fg='white').grid(row=2, column=0, sticky='w',
                                                                                     padx=5, pady=5)
        setup_frame = tk.Frame(settings_frame, bg='#1a1a2e')
        setup_frame.grid(row=2, column=1, columnspan=3, sticky='w', padx=5)

        self.reclaim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setup_frame, text="VWAP Reclaim", variable=self.reclaim_var, command=self.on_setup_change).pack(
            side=tk.LEFT, padx=5)

        self.bounce_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setup_frame, text="Band Bounce", variable=self.bounce_var, command=self.on_setup_change).pack(
            side=tk.LEFT, padx=5)

        self.discount_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setup_frame, text="Discount Pullback", variable=self.discount_var,
                        command=self.on_setup_change).pack(side=tk.LEFT, padx=5)

        # Risk Info
        risk_info = tk.Label(settings_frame,
                             text="Risk: Reclaim 0.4% | Bounce 0.5% | Discount 0.3% | Target: 1×ATR (55% partial)",
                             bg='#1a1a2e', fg='gray', font=('', 8))
        risk_info.grid(row=3, column=0, columnspan=4, sticky='w', padx=5, pady=(5, 0))

        # Buttons - 3 buttons now (Start, Stop, Backtest)
        btn_frame = tk.Frame(left_panel, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="▶ Start Trading", command=self.start_trading)
        self.start_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop Trading", command=self.stop_trading, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.backtest_btn = ttk.Button(btn_frame, text="📊 Backtest", command=self.run_backtest)
        self.backtest_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        # Stats Container
        stats_container = tk.Frame(left_panel, bg='#1a1a2e')
        stats_container.pack(fill=tk.X, pady=(0, 10))

        stats_frame = ttk.LabelFrame(stats_container, text="📊 Trading Statistics", padding=10)
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        self.stats_vars = {}
        stats_labels = [
            ('Total Trades', 'total_trades'),
            ('Win Rate', 'win_rate'),
            ('Profit Factor', 'profit_factor'),
            ('Total PnL', 'total_pnl'),
            ('Balance', 'balance'),
            ('ROI', 'roi'),
            ('Consecutive Losses', 'consecutive_losses'),
        ]

        for i, (label, key) in enumerate(stats_labels):
            tk.Label(stats_frame, text=f"{label}:", bg='#1a1a2e', fg='white', anchor='w').grid(row=i, column=0,
                                                                                               sticky='w', padx=5,
                                                                                               pady=3)
            self.stats_vars[key] = tk.StringVar(value="0")
            tk.Label(stats_frame, textvariable=self.stats_vars[key], bg='#1a1a2e', fg='#00d4ff', font=('', 9, 'bold'),
                     anchor='e').grid(row=i, column=1, sticky='e', padx=5, pady=3)

        pos_frame = ttk.LabelFrame(stats_container, text="📍 Current Position", padding=10)
        pos_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        self.pos_vars = {}
        pos_labels = [('Type', 'type'), ('Setup', 'setup'), ('Entry', 'entry'), ('Stop', 'stop'), ('Bars', 'bars'),
                      ('PnL', 'pnl')]

        for i, (label, key) in enumerate(pos_labels):
            tk.Label(pos_frame, text=f"{label}:", bg='#1a1a2e', fg='white', anchor='w').grid(row=i, column=0,
                                                                                             sticky='w', padx=5, pady=3)
            self.pos_vars[key] = tk.StringVar(value="--")
            tk.Label(pos_frame, textvariable=self.pos_vars[key], bg='#1a1a2e', fg='#00d4ff', font=('', 9, 'bold'),
                     anchor='e').grid(row=i, column=1, sticky='e', padx=5, pady=3)

        # Log Area
        log_frame = ttk.LabelFrame(left_panel, text="📝 Trading Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_area = scrolledtext.ScrolledText(log_frame, height=12, font=('Consolas', 9), bg='#0d1117',
                                                  fg='#c9d1d9')
        self.log_area.pack(fill=tk.BOTH, expand=True)

        self.log_area.tag_config('green', foreground='#2ed573')
        self.log_area.tag_config('red', foreground='#ff4757')
        self.log_area.tag_config('blue', foreground='#00d4ff')
        self.log_area.tag_config('orange', foreground='#ffa502')
        self.log_area.tag_config('purple', foreground='#a29bfe')

        clear_btn = ttk.Button(log_frame, text="Clear Log", command=self.clear_log)
        clear_btn.pack(pady=(5, 0))

        # Chart
        self.setup_chart(right_panel)

    def setup_chart(self, parent):
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor='#1a1a2e')
        self.ax = self.figure.add_subplot(111, facecolor='#16213e')
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.tick_params(colors='white', labelsize=8)

    def clear_log(self):
        self.log_area.delete(1.0, tk.END)
        self.log_message("Log cleared", 'blue')

    def update_chart(self, df):
        if df is None or len(df) < 20:
            return

        self.ax.clear()
        self.ax.set_facecolor('#16213e')

        display_df = df.iloc[-100:] if len(df) > 100 else df

        self.ax.plot(display_df.index, display_df['Close'], '#00d4ff', linewidth=1.5, label='Close')

        if 'VWAP' in display_df.columns:
            self.ax.plot(display_df.index, display_df['VWAP'], '#ff6b6b', linewidth=1, alpha=0.8, label='VWAP')
        if 'VWAP_High' in display_df.columns:
            self.ax.plot(display_df.index, display_df['VWAP_High'], '#ffa502', linewidth=0.8, alpha=0.6, linestyle='--',
                         label='VWAP +1σ')
        if 'VWAP_Low' in display_df.columns:
            self.ax.plot(display_df.index, display_df['VWAP_Low'], '#ffa502', linewidth=0.8, alpha=0.6, linestyle='--',
                         label='VWAP -1σ')
        if 'EMA_9' in display_df.columns:
            self.ax.plot(display_df.index, display_df['EMA_9'], '#2ed573', linewidth=0.8, alpha=0.7, label='EMA 9')
        if 'EMA_21' in display_df.columns:
            self.ax.plot(display_df.index, display_df['EMA_21'], '#ff4757', linewidth=0.8, alpha=0.7, label='EMA 21')

        for trade in self.strategy.trades:
            if trade['entry_time'] >= display_df.index[0]:
                color = '#2ed573' if trade['type'] == 'long' else '#ff4757'
                marker = '^' if trade['type'] == 'long' else 'v'
                self.ax.scatter(trade['entry_time'], trade['entry'], marker=marker, color=color, s=120, zorder=5,
                                edgecolors='white', linewidth=1)
                exit_color = '#7bed9f' if trade['pnl'] > 0 else '#ff6b81'
                self.ax.scatter(trade['exit_time'], trade['exit'], marker='o', color=exit_color, s=80, zorder=5,
                                edgecolors='white', linewidth=0.5)

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.tick_params(axis='x', rotation=45, colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.grid(True, alpha=0.15, linestyle='--', color='white')

        stats = self.strategy.get_stats()
        self.ax.set_title(
            f"{self.symbol_var.get()} - {self.timeframe_var.get()} | Trades: {stats['total_trades']} | WR: {stats['win_rate']:.1f}% | PnL: ${stats['total_pnl']:.2f}",
            fontsize=10, color='white', fontweight='bold')
        self.ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

        self.figure.tight_layout()
        self.canvas.draw()

    def update_stats_display(self):
        """Update statistics display - FIXED KeyError"""
        stats = self.strategy.get_stats()

        self.stats_vars['total_trades'].set(str(stats.get('total_trades', 0)))
        self.stats_vars['win_rate'].set(f"{stats.get('win_rate', 0):.1f}%")
        pf = stats.get('profit_factor', 0)
        self.stats_vars['profit_factor'].set(f"{pf:.2f}" if pf != float('inf') else "∞")
        self.stats_vars['total_pnl'].set(f"${stats.get('total_pnl', 0):.2f}")
        self.stats_vars['balance'].set(f"${stats.get('balance', 0):.2f}")
        self.stats_vars['roi'].set(f"{stats.get('roi', 0):+.2f}%")
        self.stats_vars['consecutive_losses'].set(str(stats.get('consecutive_losses', 0)))

        if self.strategy.position:
            pos = self.strategy.position
            self.pos_vars['type'].set(pos['type'].upper())
            self.pos_vars['setup'].set(pos['setup_type'].value if pos['setup_type'] else "--")
            self.pos_vars['entry'].set(f"${pos['entry']:.2f}")
            self.pos_vars['stop'].set(f"${pos['stop_loss']:.2f}")
            self.pos_vars['bars'].set(str(pos['bars_held']))

            if self.current_data:
                current_price = self.current_data.get('Close', pos['entry'])
                if pos['type'] == 'long':
                    pnl = (current_price - pos['entry']) * pos['quantity']
                else:
                    pnl = (pos['entry'] - current_price) * pos['quantity']
                self.pos_vars['pnl'].set(f"${pnl:+.2f}")
        else:
            for key in self.pos_vars:
                self.pos_vars[key].set("--")

        if self.running:
            self.root.after(1000, self.update_stats_display)

    def log_message(self, message, color='white'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_area.insert(tk.END, f"[{timestamp}] ", 'blue')
        self.log_area.insert(tk.END, f"{message}\n", color)
        self.log_area.see(tk.END)

    def on_mode_change(self, event=None):
        self.mode = self.mode_var.get().lower()
        self.log_message(f"Mode changed to: {self.mode.upper()}", 'blue')

    def on_timeframe_change(self, event=None):
        self.log_message(f"Timeframe changed to: {self.timeframe_var.get()}", 'blue')

    def on_direction_change(self, event=None):
        VWAP_STRATEGY_CONFIG['trade_direction'] = self.direction_var.get()
        self.strategy.config['trade_direction'] = self.direction_var.get()
        self.log_message(f"Trade direction: {self.direction_var.get().upper()}", 'blue')

    def on_setup_change(self, event=None):
        VWAP_STRATEGY_CONFIG['enable_reclaim'] = self.reclaim_var.get()
        VWAP_STRATEGY_CONFIG['enable_bounce'] = self.bounce_var.get()
        VWAP_STRATEGY_CONFIG['enable_discount'] = self.discount_var.get()
        self.strategy.config.update({
            'enable_reclaim': self.reclaim_var.get(),
            'enable_bounce': self.bounce_var.get(),
            'enable_discount': self.discount_var.get()
        })
        enabled = []
        if self.reclaim_var.get(): enabled.append("Reclaim")
        if self.bounce_var.get(): enabled.append("Bounce")
        if self.discount_var.get(): enabled.append("Discount")
        self.log_message(f"Enabled: {', '.join(enabled)}", 'blue')

    def run_backtest(self):
        """
        Run backtest using REAL OKX historical data.
        """

        try:

            self.log_message("=" * 60, 'purple')
            self.log_message("RUNNING BACKTEST...", 'purple')

            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()

            self.log_message(
                f"Fetching {symbol} {timeframe} historical data...",
                'blue'
            )

            # FETCH REAL MARKET DATA
            df = DataFetcher.fetch_live_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000
            )

            # FALLBACK TO SYNTHETIC IF API FAILS
            if df is None or len(df) < 100:
                self.log_message(
                    "Live data unavailable — using synthetic data fallback",
                    'orange'
                )

                start_prices = {
                    "BTC-USDT": 65000,
                    "ETH-USDT": 3000,
                    "SOL-USDT": 150,
                }

                start_price = start_prices.get(symbol, 2000)

                df = DataFetcher.generate_synthetic_data(
                    timeframe=timeframe,
                    periods=10000,
                    start_price=start_price
                )

            self.log_message(
                f"Loaded {len(df)} candles",
                'green'
            )

            # RUN BACKTEST
            stats, trades = BacktestEngine.run(
                df,
                VWAP_STRATEGY_CONFIG,
                50000
            )

            self.log_message("=" * 60, 'purple')
            self.log_message("BACKTEST RESULTS", 'green')

            self.log_message(
                f"Symbol: {symbol} | TF: {timeframe}",
                'blue'
            )

            self.log_message(
                f"Total Trades: {stats['total_trades']}",
                'white'
            )

            self.log_message(
                f"Win Rate: {stats['win_rate']:.1f}%",
                'green' if stats['win_rate'] > 55 else 'orange'
            )

            self.log_message(
                f"Profit Factor: {stats['profit_factor']:.2f}",
                'green' if stats['profit_factor'] > 1.3 else 'orange'
            )

            self.log_message(
                f"Total PnL: ${stats['total_pnl']:.2f}",
                'green' if stats['total_pnl'] > 0 else 'red'
            )

            self.log_message(
                f"ROI: {stats['roi']:+.2f}%",
                'green' if stats['roi'] > 0 else 'red'
            )

            self.log_message(
                f"Max Consecutive Losses: {stats['consecutive_losses']}",
                'orange'
            )

            self.log_message("=" * 60, 'purple')

            # UPDATE CHART
            self.df = df
            self.update_chart(df)

            # RESULTS POPUP
            messagebox.showinfo(
                "Backtest Complete",
                f"Backtest Results\n\n"
                f"Symbol: {symbol}\n"
                f"Timeframe: {timeframe}\n\n"
                f"Total Trades: {stats['total_trades']}\n"
                f"Win Rate: {stats['win_rate']:.1f}%\n"
                f"Profit Factor: {stats['profit_factor']:.2f}\n"
                f"Total PnL: ${stats['total_pnl']:.2f}\n"
                f"ROI: {stats['roi']:+.2f}%\n"
                f"Final Balance: ${stats['balance']:.2f}"
            )

        except Exception as e:

            self.log_message(
                f"Backtest error: {str(e)}",
                'red'
            )

            import traceback
            traceback.print_exc()

    def trading_cycle(self):
        """Main trading loop"""
        bar_count = 0
        while self.running:
            try:
                if self.mode == 'live':
                    df = DataFetcher.fetch_live_data(self.symbol_var.get(), self.timeframe_var.get(), 200)
                else:
                    df = DataFetcher.generate_synthetic_data(self.timeframe_var.get(), 200, 2000)

                if df is None or len(df) < 50:
                    self.log_message("Waiting for data...", 'orange')
                    time.sleep(5)
                    continue

                # Calculate indicators
                df = IndicatorCalculator.calculate_all(df, VWAP_STRATEGY_CONFIG)
                self.df = df

                # Get latest data
                latest = df.iloc[-1].to_dict()
                prev = df.iloc[-2].to_dict() if len(df) > 1 else latest

                # Build current data dict
                current_data = {
                    'Close': latest['Close'],
                    'Prev_Close': prev['Close'],
                    'VWAP': latest.get('VWAP', latest['Close']),
                    'Prev_VWAP': prev.get('VWAP', prev['Close']),

                    'VWAP_High': latest.get('VWAP_High', latest['Close']),
                    'VWAP_Low': latest.get('VWAP_Low', latest['Close']),

                    'Prev_VWAP_Low': prev.get('VWAP_Low', prev['Close']),
                    'Prev_VWAP_High': prev.get('VWAP_High', prev['Close']),

                    'EMA_9': latest.get('EMA_9', 0),
                    'EMA_21': latest.get('EMA_21', 0),
                    'RSI': latest.get('RSI', 50),
                    'Prev_RSI': prev.get('RSI', 50),
                    'ATR': latest.get('ATR', 0.01),
                    'ADX': latest.get('ADX', 0),
                    'Volume_Ratio': latest.get('Volume_Ratio', 0.5),
                    'Super_Direction': latest.get('Super_Direction', 1),
                    'Trend_Age': latest.get('Trend_Age', 1),
                    'Stoch_K': latest.get('Stoch_K', 50),
                    'Stoch_D': latest.get('Stoch_D', 50),
                }

                self.current_data = current_data
                self.strategy.update_data(current_data)
                current_price = latest['Close']

                # Update chart
                self.root.after(0, lambda: self.update_chart(df.copy()))

                # Check exits
                if self.strategy.state == StrategyState.IN_TRADE and self.strategy.position:
                    exit_action, reason = self.strategy.check_exit(current_data, current_price)
                    if exit_action == 'partial':
                        self.strategy.execute_exit(exit_action, current_price, reason, is_partial=True)
                    elif exit_action:
                        self.strategy.execute_exit(exit_action, current_price, reason, is_partial=False)
                        self.root.bell()

                # Check entries
                if self.strategy.state == StrategyState.SEEKING_ENTRY:
                    action, reason, quality = self.strategy.check_entry(current_data)

                    if action in ('buy', 'sell_short'):
                        setup_type = SetupType.VWAP_RECLAIM
                        if "Band Bounce" in reason:
                            setup_type = SetupType.VWAP_BAND_BOUNCE
                        elif "Discount" in reason:
                            setup_type = SetupType.DISCOUNT_PULLBACK

                        if self.strategy.execute_entry(action, current_price, setup_type, reason):
                            self.root.bell()
                            self.log_message(f"🎯 SIGNAL: {reason}", 'purple')

                bar_count += 1
                interval_minutes = GlobalConfig.TIMEFRAME_MINUTES.get(self.timeframe_var.get(), 5)
                wait_time = max(2, interval_minutes * 60 / 2)
                time.sleep(wait_time)

            except Exception as e:
                self.log_message(f"Error: {str(e)}", 'red')
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def start_trading(self):
        if self.running:
            return

        VWAP_STRATEGY_CONFIG['trade_direction'] = self.direction_var.get()
        VWAP_STRATEGY_CONFIG['enable_reclaim'] = self.reclaim_var.get()
        VWAP_STRATEGY_CONFIG['enable_bounce'] = self.bounce_var.get()
        VWAP_STRATEGY_CONFIG['enable_discount'] = self.discount_var.get()
        self.strategy.config.update(VWAP_STRATEGY_CONFIG)

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.backtest_btn.config(state=tk.DISABLED)

        self.log_message("=" * 70, 'blue')
        self.log_message("VWAP CORE STRATEGY STARTED", 'green')
        self.log_message(f"Symbol: {self.symbol_var.get()} | Timeframe: {self.timeframe_var.get()}", 'blue')
        self.log_message(f"Mode: {self.mode.upper()} | Direction: {VWAP_STRATEGY_CONFIG['trade_direction'].upper()}",
                         'blue')
        self.log_message("=" * 70, 'blue')

        self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
        self.trading_thread.start()
        self.update_stats_display()

    def stop_trading(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.backtest_btn.config(state=tk.NORMAL)

        stats = self.strategy.get_stats()
        self.log_message("=" * 70, 'blue')
        self.log_message("TRADING STOPPED", 'orange')
        self.log_message(
            f"Final: {stats['total_trades']} trades | WR: {stats['win_rate']:.1f}% | PnL: ${stats['total_pnl']:.2f}",
            'purple')
        self.log_message("=" * 70, 'blue')


def main():
    root = tk.Tk()
    app = ScalpingTraderApp(root)

    def on_closing():
        if app.running:
            app.stop_trading()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()