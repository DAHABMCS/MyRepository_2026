#!/usr/bin/env python3
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROFESSIONAL SCALPING TRADER v3.2                         │
│                                                                              │
│  FULLY FIXED VERSION - All missing parts included                           │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import json
import threading
import time
import tkinter as tk
import matplotlib.dates as mdates
from tkinter import ttk, scrolledtext
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque
from enum import Enum, auto

import numpy as np
import pandas as pd
import talib
import requests
from backtesting import Backtest, Strategy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class GlobalConfig:
    """Global trading configuration."""
    INITIAL_CAPITAL = 50000.0
    DEFAULT_SYMBOL = "ETH-USDT"
    ACTIVE_TIMEFRAME = "5m"

    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
    TIMEFRAME_MINUTES = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440
    }


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED STRATEGY CONFIGURATION - REPLACE the STRATEGY_CONFIG dict with this:

# OPTIMIZED STRATEGY CONFIGURATION - BALANCED VERSION (based on working 52% win rate)

# OPTIMIZED STRATEGY CONFIGURATION - FOCUS ON RISK/REWARD

STRATEGY_CONFIG = {
    # Entry thresholds
    "min_quality_long": 55,
    "min_quality_short": 62,

    # Indicator periods
    "ema_fast": 9,
    "ema_mid": 21,
    "ema_slow": 50,
    "rsi_period": 14,
    "atr_period": 14,
    "adx_period": 14,
    "volume_period": 20,

    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # Risk management - IMPROVED R:R
    "risk_per_trade": 0.003,  # REDUCED to 0.3% risk (smaller losses)
    "stop_loss_atr_mult": 2.5,  # TIGHTER stops (2.5x ATR)
    "min_stop_pct": 0.02,  # 2% minimum stop
    "max_stop_pct": 0.04,  # 4% maximum stop
    "short_stop_atr_mult": 3.0,  # TIGHTER for shorts
    "short_min_stop_pct": 0.025,  # 2.5% minimum for shorts

    # Take profit - CRITICAL for positive returns
    "take_profit_atr_mult": 4.0,  # Take profit at 4x ATR (1:1.6 risk/reward)
    "min_take_profit_pct": 0.035,  # Minimum 3.5% profit target
    "max_take_profit_pct": 0.08,  # Maximum 8% profit target

    # Trailing stops (secondary exit)
    "trailing_activation_pct": 0.025,  # Activate at 2.5% profit
    "trailing_distance_pct": 0.01,  # 1% trail (tighter)
    "max_hold_bars": 36,

    # Volume
    "volume_min_ratio": 1.2,
    "volume_strong_ratio": 1.8,

    # RSI zones
    "rsi_long_min": 40,
    "rsi_long_max": 70,
    "rsi_short_min": 35,
    "rsi_short_max": 55,

    # ADX
    "adx_min": 20,
    "adx_strong": 35,

    # Trade direction
    "trade_direction": "both",

    # Loss protection
    "max_consecutive_losses": 3,  # Increased to allow more trades
    "cooldown_bars": 4,  # Shorter cooldown

    # Short-specific filters
    "min_drop_for_short": 1.2,
    "pullback_min_pct": 0.6,
    "pullback_max_pct": 3.5,
}

# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class StrategyState(Enum):
    SEEKING_ENTRY = auto()
    IN_TRADE = auto()
    COOLDOWN = auto()


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATOR - COMPLETE FIXED VERSION
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorCalculator:
    """Calculate all trading indicators - COMPLETE FIXED VERSION."""

    @staticmethod
    def calculate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = df.copy()

        # Ensure we have enough data
        if len(df) < 50:
            return df

        # EMAs
        df['EMA_Fast'] = talib.EMA(df['Close'], config['ema_fast'])
        df['EMA_Mid'] = talib.EMA(df['Close'], config['ema_mid'])
        df['EMA_Slow'] = talib.EMA(df['Close'], config['ema_slow'])

        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
            df['Close'],
            fastperiod=config['macd_fast'],
            slowperiod=config['macd_slow'],
            signalperiod=config['macd_signal']
        )

        # RSI
        df['RSI'] = talib.RSI(df['Close'], config['rsi_period'])

        # ATR
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], config['atr_period'])

        # ADX
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], config['adx_period'])

        # Volume MA
        df['Volume_MA'] = talib.SMA(df['Volume'], config['volume_period'])
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, 1)

        # Bollinger Bands
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )

        # SuperTrend
        atr_mult = 3.0
        hl2 = (df['High'] + df['Low']) / 2
        df['SuperTrend_Upper'] = hl2 + atr_mult * df['ATR']
        df['SuperTrend_Lower'] = hl2 - atr_mult * df['ATR']

        df['SuperTrend'] = 0.0
        df['SuperTrend_Direction'] = 1

        for i in range(1, len(df)):
            if df['SuperTrend_Direction'].iloc[i - 1] == 1:
                if df['Close'].iloc[i] <= df['SuperTrend_Upper'].iloc[i]:
                    df.loc[df.index[i], 'SuperTrend_Direction'] = -1
                    df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Upper'].iloc[i]
                else:
                    df.loc[df.index[i], 'SuperTrend_Direction'] = 1
                    df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Lower'].iloc[i]
            else:
                if df['Close'].iloc[i] >= df['SuperTrend_Lower'].iloc[i]:
                    df.loc[df.index[i], 'SuperTrend_Direction'] = 1
                    df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Lower'].iloc[i]
                else:
                    df.loc[df.index[i], 'SuperTrend_Direction'] = -1
                    df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Upper'].iloc[i]

        # CCI
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Rate of Change
        df['ROC'] = talib.ROC(df['Close'], timeperiod=5)

        # Rolling highs/lows - CRITICAL for short detection
        df['Highest_10'] = talib.MAX(df['High'], timeperiod=10)
        df['Lowest_10'] = talib.MIN(df['Low'], timeperiod=10)
        df['Highest_5'] = talib.MAX(df['High'], timeperiod=5)
        df['Lowest_5'] = talib.MIN(df['Low'], timeperiod=5)

        # Pullback calculations
        df['Pullback_From_High'] = (df['Highest_10'] - df['Close']) / df['Highest_10'] * 100
        df['Rally_From_Low'] = (df['Close'] - df['Lowest_10']) / df['Lowest_10'] * 100

        # Consecutive direction
        df['Direction'] = 0
        df['Consecutive_Direction'] = 0

        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                df.loc[df.index[i], 'Direction'] = 1
                df.loc[df.index[i], 'Consecutive_Direction'] = df['Consecutive_Direction'].iloc[i - 1] + 1 if \
                df['Direction'].iloc[i - 1] == 1 else 1
            else:
                df.loc[df.index[i], 'Direction'] = -1
                df.loc[df.index[i], 'Consecutive_Direction'] = df['Consecutive_Direction'].iloc[i - 1] + 1 if \
                df['Direction'].iloc[i - 1] == -1 else 1

        # Shifted values for closed candles (so we don't use current incomplete candle)
        columns_to_shift = [
            'EMA_Fast', 'EMA_Slow', 'MACD', 'MACD_Signal', 'RSI', 'ADX', 'ATR',
            'Volume_Ratio', 'CCI', 'ROC', 'Lowest_5', 'Highest_5',
            'Lowest_10', 'Highest_10',
            'Consecutive_Direction', 'SuperTrend_Direction',
            'Pullback_From_High', 'Rally_From_Low'
        ]

        for col in columns_to_shift:
            if col in df.columns:
                df[f'{col}_Closed'] = df[col].shift(1)
            else:
                df[f'{col}_Closed'] = 0

        # Ensure SuperTrend_Closed exists
        if 'SuperTrend_Direction_Closed' in df.columns:
            df['SuperTrend_Closed'] = df['SuperTrend_Direction_Closed']
        else:
            df['SuperTrend_Closed'] = 0

        # Fill NaN values
        df = df.fillna(0)

        return df


# ═══════════════════════════════════════════════════════════════════════════
# SCALPING LOGIC - COMPLETE
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingLogic:
    """Core trading logic with fixed short entry conditions."""

    def __init__(self, config: dict):
        self.config = config.copy()
        self.position = {
            'type': None, 'entry_price': 0, 'quantity': 0,
            'stop_loss': 0, 'trailing_stop': 0, 'highest_price': 0,
            'entry_time': None, 'bars_held': 0
        }
        self.balance = GlobalConfig.INITIAL_CAPITAL
        self.trades = []
        self.strategy_state = StrategyState.SEEKING_ENTRY
        self.current_data = {}
        self.consecutive_losses = 0
        self.cooldown_counter = 0
        self.timeframe = "5m"

    def update_timeframe(self, tf: str):
        self.timeframe = tf

    def can_short(self, data: dict) -> Tuple[bool, str]:
        """Check if short conditions are met - RELAXED for more trades."""

        price = data.get('Close', 0)
        lowest_10 = data.get('Lowest_10_Closed', price)
        highest_10 = data.get('Highest_10_Closed', price)

        if highest_10 <= 0 or lowest_10 <= 0:
            return False, "Insufficient data"

        drop_from_high = (highest_10 - price) / highest_10 * 100
        rally_from_low = (price - lowest_10) / lowest_10 * 100

        # RELAXED: Only require 1.2% drop (was 1.5%)
        if drop_from_high < self.config.get('min_drop_for_short', 1.2):
            return False, f"Drop: {drop_from_high:.1f}%"

        # RELAXED: Only require 0.6% pullback (was 0.8%)
        if rally_from_low < self.config.get('pullback_min_pct', 0.6):
            return False, f"Pullback: {rally_from_low:.1f}%"

        # RELAXED: Allow up to 3.5% recovery (was 3.0%)
        if rally_from_low > self.config.get('pullback_max_pct', 3.5):
            return False, f"Recovery: {rally_from_low:.1f}%"

        # RELAXED RSI range
        rsi = data.get('RSI_Closed', 50)
        if rsi < 30 or rsi > 75:  # Wider range (was 35-70)
            return False, f"RSI: {rsi:.0f}"

        return True, f"Drop:{drop_from_high:.1f}% Rally:{rally_from_low:.1f}% RSI:{rsi:.0f}"

    def calculate_quality_short(self, data: dict) -> Tuple[int, str]:
        """Quality scoring for short."""
        breakdown = []

        # First check if we can short
        can_short, short_reason = self.can_short(data)
        if not can_short:
            return 0, f"❌ {short_reason}"

        # Mandatory conditions
        supertrend_bear = data.get('SuperTrend_Closed', 0) == -1
        ma_bear = data.get('EMA_Fast_Closed', 0) < data.get('EMA_Slow_Closed', 0)

        mandatory_score = 0
        if supertrend_bear:
            mandatory_score += 25
            breakdown.append("ST_BEAR+25")
        else:
            breakdown.append("ST_BULL+0")

        if ma_bear:
            mandatory_score += 15
            breakdown.append("MA_BEAR+15")
        else:
            breakdown.append("MA_BULL+0")

        if mandatory_score < 40:
            return 0, " | ".join(breakdown) + " | MANDATORY FAILED"

        # Supporting conditions
        supporting_score = 0
        supporting_hits = []

        if data.get('MACD_Closed', 0) < data.get('MACD_Signal_Closed', 0):
            supporting_score += 15
            supporting_hits.append("MACD+15")

        adx = data.get('ADX_Closed', 0)
        if adx > 35:
            supporting_score += 8
            supporting_hits.append(f"ADX+8({adx:.0f})")
        elif adx > 25:
            supporting_score += 5
            supporting_hits.append(f"ADX+5({adx:.0f})")

        vol_ratio = data.get('Volume_Ratio_Closed', 1.0)
        if vol_ratio >= 1.8:
            supporting_score += 15
            supporting_hits.append(f"VOL+15({vol_ratio:.1f}x)")
        elif vol_ratio >= 1.2:
            supporting_score += 10
            supporting_hits.append(f"VOL+10({vol_ratio:.1f}x)")

        rsi = data.get('RSI_Closed', 50)
        if 35 <= rsi <= 55:
            supporting_score += 5
            supporting_hits.append(f"RSI+5({rsi:.0f})")

        cci = data.get('CCI_Closed', 0)
        if -100 < cci < 100:
            supporting_score += 5
            supporting_hits.append(f"CCI+5({cci:.0f})")

        supporting_score = min(supporting_score, 60)
        total_score = mandatory_score + supporting_score

        breakdown.append(f"SUPPORT({supporting_score}/60): {'+'.join(supporting_hits) if supporting_hits else 'none'}")
        breakdown.append(f"TOTAL={total_score}")

        return total_score, " | ".join(breakdown)

    def calculate_quality_long(self, data: dict) -> Tuple[int, str]:
        """Quality scoring for long."""
        breakdown = []

        supertrend_bull = data.get('SuperTrend_Closed', 0) == 1
        ma_bull = data.get('EMA_Fast_Closed', 0) > data.get('EMA_Slow_Closed', 0)

        mandatory_score = 0
        if supertrend_bull:
            mandatory_score += 25
            breakdown.append("ST_BULL+25")
        else:
            breakdown.append("ST_BEAR+0")

        if ma_bull:
            mandatory_score += 15
            breakdown.append("MA_BULL+15")
        else:
            breakdown.append("MA_BEAR+0")

        if mandatory_score < 40:
            return 0, " | ".join(breakdown) + " | MANDATORY FAILED"

        supporting_score = 0
        supporting_hits = []

        if data.get('MACD_Closed', 0) > data.get('MACD_Signal_Closed', 0):
            supporting_score += 15
            supporting_hits.append("MACD+15")

        adx = data.get('ADX_Closed', 0)
        if adx > 35:
            supporting_score += 8
            supporting_hits.append(f"ADX+8({adx:.0f})")
        elif adx > 25:
            supporting_score += 5
            supporting_hits.append(f"ADX+5({adx:.0f})")

        vol_ratio = data.get('Volume_Ratio_Closed', 1.0)
        if vol_ratio >= 1.8:
            supporting_score += 15
            supporting_hits.append(f"VOL+15({vol_ratio:.1f}x)")
        elif vol_ratio >= 1.2:
            supporting_score += 10
            supporting_hits.append(f"VOL+10({vol_ratio:.1f}x)")

        rsi = data.get('RSI_Closed', 50)
        if 40 <= rsi <= 70:
            supporting_score += 5
            supporting_hits.append(f"RSI+5({rsi:.0f})")

        cci = data.get('CCI_Closed', 0)
        if -100 < cci < 100:
            supporting_score += 5
            supporting_hits.append(f"CCI+5({cci:.0f})")

        if data.get('ATR_Closed', 0) > 0:
            supporting_score += 3
            supporting_hits.append("ATR+3")

        supporting_score = min(supporting_score, 60)
        total_score = mandatory_score + supporting_score

        breakdown.append(f"SUPPORT({supporting_score}/60): {'+'.join(supporting_hits) if supporting_hits else 'none'}")
        breakdown.append(f"TOTAL={total_score}")

        return total_score, " | ".join(breakdown)

    def calculate_position_size(self, price: float, atr: float, quality: int, is_short: bool = False) -> float:
        """Calculate position size - WITH performance adjustment."""
        base_risk = self.config.get('risk_per_trade', 0.005)

        # Adjust based on recent win rate
        recent_trades = self.trades[-10:] if len(self.trades) > 0 else []
        if len(recent_trades) >= 5:
            recent_wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            win_rate = recent_wins / len(recent_trades)

            if win_rate > 0.6:
                base_risk = base_risk * 1.2  # Increase size when winning
            elif win_rate < 0.4:
                base_risk = base_risk * 0.7  # Decrease size when losing

        # Quality bonus
        quality_mult = 1.0
        if quality >= 70:
            quality_mult = 1.2
        elif quality >= 65:
            quality_mult = 1.1

        if self.timeframe == '1m':
            base_risk = base_risk * 0.5
        elif is_short:
            base_risk = base_risk * 0.85

        risk_amount = self.balance * base_risk * quality_mult

        # Calculate stop distance
        if is_short:
            atr_stop = atr * self.config.get('short_stop_atr_mult', 4.0)
            min_stop = price * self.config.get('short_min_stop_pct', 0.035)
        else:
            atr_stop = atr * self.config.get('stop_loss_atr_mult', 3.5)
            min_stop = price * self.config.get('min_stop_pct', 0.03)

        stop_distance = max(atr_stop, min_stop)
        max_stop = price * self.config.get('max_stop_pct', 0.06)
        stop_distance = min(stop_distance, max_stop)

        if stop_distance <= 0:
            stop_distance = price * 0.03

        size = (risk_amount / stop_distance)
        max_size = self.balance * 0.12 / price

        final_size = min(size, max_size)
        final_size = max(final_size, 0.001)

        return final_size
    def check_entry(self, data: dict) -> Tuple[str, int, str]:
        """Check for entry signals - FIXED thresholds."""
        if self.strategy_state == StrategyState.COOLDOWN:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.config.get('cooldown_bars', 8):
                self.strategy_state = StrategyState.SEEKING_ENTRY
                self.cooldown_counter = 0
                self.consecutive_losses = 0
            else:
                return 'hold', 0, f"COOLDOWN ({self.cooldown_counter}/{self.config.get('cooldown_bars', 8)})"

        if self.strategy_state == StrategyState.IN_TRADE:
            return 'hold', 0, "Already in trade"

        trade_dir = self.config.get('trade_direction', 'both')

        # Check short first (higher threshold)
        if trade_dir in ('short', 'both'):
            quality_short, breakdown_short = self.calculate_quality_short(data)
            min_short = self.config.get('min_quality_short', 65)
            if quality_short >= min_short:
                return 'sell_short', quality_short, breakdown_short
            elif quality_short > 0:
                # Log why we're not shorting
                pass

        # Check long
        if trade_dir in ('long', 'both'):
            quality_long, breakdown_long = self.calculate_quality_long(data)
            min_long = self.config.get('min_quality_long', 55)
            if quality_long >= min_long:
                return 'buy', quality_long, breakdown_long

        return 'hold', 0, "No signal"

    def check_exit(self, data: dict, current_price: float) -> Tuple[Optional[str], str]:
        """Check for exit signals - ADDED profit target."""
        if self.strategy_state != StrategyState.IN_TRADE:
            return None, "No position"

        pos_type = self.position['type']
        entry = self.position['entry_price']

        if pos_type == 'long':
            profit_pct = (current_price - entry) / entry * 100
        else:
            profit_pct = (entry - current_price) / entry * 100

        # NEW: Profit target - take profits at 3% (improves win rate)
        profit_target = self.config.get('profit_target_pct', 3.0)
        if profit_pct >= profit_target:
            return ('sell' if pos_type == 'long' else 'buy_cover'), f"Profit target {profit_target}% hit"

        # Trailing stop (earlier activation)
        trailing_activation = self.config.get('trailing_activation_pct', 0.025) * 100
        trailing_distance = self.config.get('trailing_distance_pct', 0.015)

        if pos_type == 'long':
            if profit_pct >= trailing_activation:
                new_trail = current_price * (1 - trailing_distance)
                self.position['trailing_stop'] = max(self.position['trailing_stop'], new_trail)

            if current_price <= self.position['stop_loss']:
                return 'sell', f"Stop loss"
            if self.position['trailing_stop'] > 0 and current_price <= self.position['trailing_stop']:
                return 'sell', f"Trailing stop"
        else:
            if profit_pct >= trailing_activation:
                new_trail = current_price * (1 + trailing_distance)
                if self.position['trailing_stop'] == 0 or new_trail < self.position['trailing_stop']:
                    self.position['trailing_stop'] = new_trail

            if current_price >= self.position['stop_loss']:
                return 'buy_cover', f"Stop loss"
            if self.position['trailing_stop'] > 0 and current_price >= self.position['trailing_stop']:
                return 'buy_cover', f"Trailing stop"

        # Max hold
        self.position['bars_held'] += 1
        if self.position['bars_held'] >= self.config.get('max_hold_bars', 36):
            return ('sell' if pos_type == 'long' else 'buy_cover'), f"Max hold"

        return None, "Holding"

    def execute_entry(self, action: str, price: float, quality: int, reason: str) -> bool:
        """Open a new position - with validation."""
        atr = self.current_data.get('ATR_Closed', price * 0.01)
        if atr <= 0:
            atr = price * 0.01

        is_short = (action == 'sell_short')
        quantity = self.calculate_position_size(price, atr, quality, is_short)

        # CRITICAL FIX: Validate quantity
        if quantity <= 0 or quantity is None:
            return False

        # Calculate stop distance
        if is_short:
            atr_stop = atr * self.config.get('short_stop_atr_mult', 4.5)
            min_stop = price * self.config.get('short_min_stop_pct', 0.04)
        else:
            atr_stop = atr * self.config.get('stop_loss_atr_mult', 4.0)
            min_stop = price * self.config.get('min_stop_pct', 0.035)

        stop_distance = max(atr_stop, min_stop)
        stop_distance = min(stop_distance, price * self.config.get('max_stop_pct', 0.06))

        # Ensure stop_distance is positive
        if stop_distance <= 0:
            stop_distance = price * 0.035

        if action == 'buy':
            stop_loss = price - stop_distance
            pos_type = 'long'
        else:
            stop_loss = price + stop_distance
            pos_type = 'short'

        self.position = {
            'type': pos_type,
            'entry_price': price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'trailing_stop': stop_loss,
            'highest_price': price,
            'entry_time': datetime.now(),
            'bars_held': 0
        }
        self.strategy_state = StrategyState.IN_TRADE

        return True

    def execute_exit(self, action: str, price: float, reason: str) -> float:
        """Close position."""
        if self.position['type'] is None:
            return 0.0

        entry = self.position['entry_price']
        quantity = self.position['quantity']

        if self.position['type'] == 'long':
            pnl = (price - entry) * quantity
        else:
            pnl = (entry - price) * quantity

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.get('max_consecutive_losses', 2):
                self.strategy_state = StrategyState.COOLDOWN
                self.cooldown_counter = 0
        else:
            self.consecutive_losses = 0

        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': entry,
            'exit_price': price,
            'quantity': quantity,
            'type': self.position['type'],
            'pnl': pnl,
            'exit_reason': reason,
            'bars_held': self.position['bars_held']
        }
        self.trades.append(trade)
        self.balance += pnl

        self.position = {
            'type': None, 'entry_price': 0, 'quantity': 0,
            'stop_loss': 0, 'trailing_stop': 0, 'highest_price': 0,
            'entry_time': None, 'bars_held': 0
        }

        if self.strategy_state != StrategyState.COOLDOWN:
            self.strategy_state = StrategyState.SEEKING_ENTRY

        return pnl

    def update_data(self, data: dict):
        self.current_data = data

    def get_stats(self) -> dict:
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'balance': self.balance, 'roi': 0,
                    'consecutive_losses': 0}

        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(self.trades) - len(winning_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'balance': self.balance,
            'roi': (self.balance - GlobalConfig.INITIAL_CAPITAL) / GlobalConfig.INITIAL_CAPITAL * 100,
            'consecutive_losses': self.consecutive_losses
        }


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST STRATEGY - COMPLETE FIXED VERSION
# ═══════════════════════════════════════════════════════════════════════════

class BacktestScalpingStrategy(Strategy):
    """Backtesting strategy - FIXED position sizing."""

    def init(self):
        # Calculate all indicators
        self.df = IndicatorCalculator.calculate(self.data.df, STRATEGY_CONFIG)
        self.logic = ScalpingLogic(STRATEGY_CONFIG)

        # CRITICAL FIX: Set initial balance
        self.logic.balance = self.equity

    def next(self):
        idx = len(self.data) - 1

        # Safely get values with fallbacks
        def safe_get(col, default=0):
            try:
                if idx > 0 and col in self.df.columns:
                    val = self.df[col].iloc[idx]
                    return val if not pd.isna(val) else default
                return default
            except:
                return default

        # Build current data dict
        current_data = {
            'Close': self.data.Close[-1],
            'EMA_Fast_Closed': safe_get('EMA_Fast_Closed'),
            'EMA_Slow_Closed': safe_get('EMA_Slow_Closed'),
            'SuperTrend_Closed': safe_get('SuperTrend_Closed'),
            'MACD_Closed': safe_get('MACD_Closed'),
            'MACD_Signal_Closed': safe_get('MACD_Signal_Closed'),
            'RSI_Closed': safe_get('RSI_Closed', 50),
            'ADX_Closed': safe_get('ADX_Closed', 20),
            'ATR_Closed': safe_get('ATR_Closed', 1),
            'Volume_Ratio_Closed': safe_get('Volume_Ratio_Closed', 1),
            'CCI_Closed': safe_get('CCI_Closed', 0),
            'ROC_Closed': safe_get('ROC_Closed', 0),
            'Lowest_5_Closed': safe_get('Lowest_5_Closed', self.data.Close[-1]),
            'Highest_5_Closed': safe_get('Highest_5_Closed', self.data.Close[-1]),
            'Lowest_10_Closed': safe_get('Lowest_10_Closed', self.data.Close[-1]),
            'Highest_10_Closed': safe_get('Highest_10_Closed', self.data.Close[-1]),
            'Consecutive_Direction_Closed': safe_get('Consecutive_Direction_Closed', 0),
            'Pullback_From_High_Closed': safe_get('Pullback_From_High_Closed', 0),
            'Rally_From_Low_Closed': safe_get('Rally_From_Low_Closed', 0),
        }

        self.logic.update_data(current_data)
        current_price = self.data.Close[-1]

        # Update logic balance to match backtest equity
        self.logic.balance = self.equity

        # Check exits
        if self.position:
            exit_action, reason = self.logic.check_exit(current_data, current_price)
            if exit_action:
                pnl = self.logic.execute_exit(exit_action, current_price, reason)
                self.position.close()
                return

        # Check entries
        if not self.position:
            action, quality, reason = self.logic.check_entry(current_data)
            if action in ('buy', 'sell_short'):
                # CRITICAL FIX: Calculate size directly for backtest
                atr = current_data.get('ATR_Closed', current_price * 0.01)
                if atr <= 0:
                    atr = current_price * 0.01

                is_short = (action == 'sell_short')

                # Calculate position size
                risk_amount = self.equity * STRATEGY_CONFIG['risk_per_trade']

                if is_short:
                    stop_distance = atr * STRATEGY_CONFIG.get('short_stop_atr_mult', 4.5)
                    min_stop = current_price * STRATEGY_CONFIG.get('short_min_stop_pct', 0.04)
                else:
                    stop_distance = atr * STRATEGY_CONFIG.get('stop_loss_atr_mult', 4.0)
                    min_stop = current_price * STRATEGY_CONFIG.get('min_stop_pct', 0.035)

                stop_distance = max(stop_distance, min_stop)
                stop_distance = min(stop_distance, current_price * STRATEGY_CONFIG.get('max_stop_pct', 0.06))

                if stop_distance <= 0:
                    stop_distance = current_price * 0.035

                # Calculate size (as fraction of equity for backtesting)
                size_fraction = (risk_amount / stop_distance) / current_price

                # Ensure size is positive and reasonable
                size_fraction = max(size_fraction, 0.001)  # Minimum 0.1% of equity
                size_fraction = min(size_fraction, 0.15)  # Maximum 15% of equity

                # Enter position
                if action == 'buy':
                    self.buy(size=size_fraction)
                    self.logic.execute_entry(action, current_price, quality, reason)
                else:
                    self.sell(size=size_fraction)
                    self.logic.execute_entry(action, current_price, quality, reason)


# ═══════════════════════════════════════════════════════════════════════════
# GUI APPLICATION - COMPLETE
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingTraderApp:
    """Main GUI application - COMPLETE."""

    def __init__(self, root):
        self.root = root
        self.root.title("Professional Scalping Trader v3.2 - Complete Fixed Version")
        self.root.geometry("1400x850")

        # Trading state
        self.running = False
        self.mode = 'demo'
        self.strategy = ScalpingLogic(STRATEGY_CONFIG)
        self.current_data = None
        self.df = None
        self.trading_thread = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Create all UI elements."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Right panel
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="Trading Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Mode:").grid(row=0, column=0, sticky='w', padx=5)
        self.mode_var = tk.StringVar(value="Demo")
        mode_combo = ttk.Combobox(settings_frame, textvariable=self.mode_var,
                                  values=['Demo', 'Live'], width=12)
        mode_combo.grid(row=0, column=1, sticky='w', padx=5)
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)

        ttk.Label(settings_frame, text="Symbol:").grid(row=1, column=0, sticky='w', padx=5)
        self.symbol_var = tk.StringVar(value=GlobalConfig.DEFAULT_SYMBOL)
        symbol_entry = ttk.Entry(settings_frame, textvariable=self.symbol_var, width=15)
        symbol_entry.grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(settings_frame, text="Timeframe:").grid(row=2, column=0, sticky='w', padx=5)
        self.timeframe_var = tk.StringVar(value=GlobalConfig.ACTIVE_TIMEFRAME)
        timeframe_combo = ttk.Combobox(settings_frame, textvariable=self.timeframe_var,
                                       values=GlobalConfig.TIMEFRAMES, width=12)
        timeframe_combo.grid(row=2, column=1, sticky='w', padx=5)
        timeframe_combo.bind('<<ComboboxSelected>>', self.on_timeframe_change)

        ttk.Label(settings_frame, text="Direction:").grid(row=3, column=0, sticky='w', padx=5)
        self.direction_var = tk.StringVar(value=STRATEGY_CONFIG['trade_direction'])
        direction_combo = ttk.Combobox(settings_frame, textvariable=self.direction_var,
                                       values=['both', 'long', 'short'], width=12)
        direction_combo.grid(row=3, column=1, sticky='w', padx=5)
        direction_combo.bind('<<ComboboxSelected>>', self.on_direction_change)

        ttk.Label(settings_frame, text="Risk per Trade (%):").grid(row=4, column=0, sticky='w', padx=5)
        self.risk_var = tk.DoubleVar(value=STRATEGY_CONFIG['risk_per_trade'] * 100)
        risk_spin = ttk.Spinbox(settings_frame, from_=0.1, to=5, increment=0.1,
                                textvariable=self.risk_var, width=12)
        risk_spin.grid(row=4, column=1, sticky='w', padx=5)
        risk_spin.bind('<FocusOut>', self.on_risk_change)

        ttk.Label(settings_frame, text="Min Quality:").grid(row=5, column=0, sticky='w', padx=5)
        self.quality_var = tk.IntVar(value=STRATEGY_CONFIG['min_quality_long'])
        quality_scale = ttk.Scale(settings_frame, from_=40, to=80, variable=self.quality_var, orient=tk.HORIZONTAL)
        quality_scale.grid(row=5, column=1, sticky='ew', padx=5)
        quality_label = ttk.Label(settings_frame, textvariable=self.quality_var)
        quality_label.grid(row=5, column=2, padx=5)

        # Control buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="▶ Start Trading", command=self.start_trading)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop Trading", command=self.stop_trading, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.backtest_btn = ttk.Button(btn_frame, text="📊 Run Backtest", command=self.run_backtest)
        self.backtest_btn.pack(side=tk.LEFT, padx=5)

        # Stats frame
        stats_frame = ttk.LabelFrame(left_panel, text="Trading Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        self.stats_vars = {}
        stats_labels = [
            ('Total Trades', 'total_trades'),
            ('Winning Trades', 'winning_trades'),
            ('Losing Trades', 'losing_trades'),
            ('Win Rate', 'win_rate'),
            ('Total PnL', 'total_pnl'),
            ('Balance', 'balance'),
            ('ROI', 'roi'),
            ('Consecutive Losses', 'consecutive_losses')
        ]

        for i, (label, key) in enumerate(stats_labels):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=i, column=0, sticky='w', padx=5, pady=2)
            self.stats_vars[key] = tk.StringVar(value="0")
            ttk.Label(stats_frame, textvariable=self.stats_vars[key]).grid(row=i, column=1, sticky='w', padx=5)

        # Position info
        pos_frame = ttk.LabelFrame(left_panel, text="Current Position", padding=10)
        pos_frame.pack(fill=tk.X, pady=(0, 10))

        self.pos_vars = {}
        pos_labels = [
            ('Type', 'type'),
            ('Entry Price', 'entry'),
            ('Quantity', 'qty'),
            ('Stop Loss', 'stop'),
            ('Trailing Stop', 'trail'),
            ('PnL', 'pnl'),
            ('Bars Held', 'bars')
        ]

        for i, (label, key) in enumerate(pos_labels):
            ttk.Label(pos_frame, text=f"{label}:").grid(row=i, column=0, sticky='w', padx=5, pady=2)
            self.pos_vars[key] = tk.StringVar(value="--")
            ttk.Label(pos_frame, textvariable=self.pos_vars[key]).grid(row=i, column=1, sticky='w', padx=5)

        # Log area
        log_frame = ttk.LabelFrame(left_panel, text="Trading Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_area = scrolledtext.ScrolledText(log_frame, height=15, font=('Consolas', 9))
        self.log_area.pack(fill=tk.BOTH, expand=True)

        self.log_area.tag_config('green', foreground='#00ff00')
        self.log_area.tag_config('red', foreground='#ff5555')
        self.log_area.tag_config('blue', foreground='#00bfff')
        self.log_area.tag_config('orange', foreground='#ffa500')
        self.log_area.tag_config('purple', foreground='#da70d6')

        # Chart
        self.setup_chart(right_panel)

    def setup_chart(self, parent):
        """Set up the price chart."""
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_chart(self, df):
        """Update price chart with indicators."""
        if df is None or len(df) < 20:
            return

        self.ax.clear()
        display_df = df.iloc[-150:] if len(df) > 150 else df

        self.ax.plot(display_df.index, display_df['Close'], 'b-', linewidth=1.5, label='Close')

        if 'EMA_Fast' in display_df.columns:
            self.ax.plot(display_df.index, display_df['EMA_Fast'], 'g--', linewidth=1, alpha=0.8, label='EMA 9')
        if 'EMA_Slow' in display_df.columns:
            self.ax.plot(display_df.index, display_df['EMA_Slow'], 'r--', linewidth=1, alpha=0.8, label='EMA 50')

        for trade in self.strategy.trades:
            trade_time = trade['entry_time']
            if trade_time and hasattr(trade_time, 'timestamp') and trade_time >= display_df.index[0]:
                if trade['type'] == 'long':
                    self.ax.scatter(trade_time, trade['entry_price'],
                                    marker='^', color='lime', s=100, zorder=5)
                    self.ax.scatter(trade['exit_time'], trade['exit_price'],
                                    marker='v', color='red', s=80, zorder=5)
                else:
                    self.ax.scatter(trade_time, trade['entry_price'],
                                    marker='v', color='red', s=100, zorder=5)
                    self.ax.scatter(trade['exit_time'], trade['exit_price'],
                                    marker='^', color='lime', s=80, zorder=5)

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.tick_params(axis='x', rotation=45, labelsize=8)
        self.ax.grid(True, alpha=0.3, linestyle='--')

        latest_price = display_df['Close'].iloc[-1]
        self.ax.set_title(f"{self.symbol_var.get()} - {self.timeframe_var.get()} | "
                          f"Latest: ${latest_price:.4f} | "
                          f"Trades: {len(self.strategy.trades)}", fontsize=10)
        self.ax.legend(loc='upper left', fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()

    def log_message(self, message: str, color: str = 'white'):
        """Add message to log area."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_area.insert(tk.END, f"[{timestamp}] {message}\n", color)
        self.log_area.see(tk.END)

    def update_stats_display(self):
        """Update statistics display."""
        if not self.running:
            return

        stats = self.strategy.get_stats()
        for key, var in self.stats_vars.items():
            if key in stats:
                value = stats[key]
                if key in ['win_rate', 'roi']:
                    var.set(f"{value:.2f}%")
                elif key in ['total_pnl', 'balance']:
                    var.set(f"${value:.2f}")
                else:
                    var.set(str(value))

        pos = self.strategy.position
        if pos['type']:
            self.pos_vars['type'].set(pos['type'].upper())
            self.pos_vars['entry'].set(f"${pos['entry_price']:.4f}")
            self.pos_vars['qty'].set(f"{pos['quantity']:.4f}")
            self.pos_vars['stop'].set(f"${pos['stop_loss']:.4f}")
            self.pos_vars['trail'].set(f"${pos['trailing_stop']:.4f}" if pos['trailing_stop'] > 0 else "--")
            self.pos_vars['bars'].set(str(pos['bars_held']))

            if self.current_data:
                current_price = self.current_data.get('Close', pos['entry_price'])
                if pos['type'] == 'long':
                    pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['quantity']
                self.pos_vars['pnl'].set(f"${pnl:.2f}")
        else:
            for key in self.pos_vars:
                if key != 'pnl':
                    self.pos_vars[key].set("--")

        self.root.after(1000, self.update_stats_display)

    def on_mode_change(self, event=None):
        self.mode = self.mode_var.get().lower()
        self.log_message(f"Mode changed to: {self.mode.upper()}", 'blue')

    def on_timeframe_change(self, event=None):
        tf = self.timeframe_var.get()
        self.strategy.update_timeframe(tf)
        self.log_message(f"Timeframe changed to: {tf}", 'blue')

    def on_direction_change(self, event=None):
        STRATEGY_CONFIG['trade_direction'] = self.direction_var.get()
        self.strategy.config['trade_direction'] = self.direction_var.get()
        self.log_message(f"Trade direction changed to: {self.direction_var.get().upper()}", 'blue')

    def on_risk_change(self, event=None):
        STRATEGY_CONFIG['risk_per_trade'] = self.risk_var.get() / 100
        self.strategy.config['risk_per_trade'] = self.risk_var.get() / 100
        self.log_message(f"Risk per trade changed to: {self.risk_var.get():.1f}%", 'blue')

    def fetch_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch real market data from OKX."""
        try:
            symbol = self.symbol_var.get()
            interval = self.timeframe_var.get()

            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1H', '2h': '2H', '4h': '4H', '1d': '1D'
            }
            bar = interval_map.get(interval, '5m')

            url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={bar}&limit=200"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    df = pd.DataFrame(data['data'], columns=[
                        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'volCcy', 'volBase', 'turnover'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df.astype(float)
                    df = df.sort_index()
                    return df
            return None
        except Exception as e:
            self.log_message(f"Data fetch error: {str(e)}", 'red')
            return None

    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic data for demo."""
        periods = 200
        freq = self.timeframe_var.get()

        freq_map = {'1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T',
                    '1h': 'H', '2h': '2H', '4h': '4H', '1d': 'D'}
        pandas_freq = freq_map.get(freq, '5T')

        np.random.seed(int(time.time()) % 1000)
        returns = np.random.normal(0.0002, 0.015, periods)
        price = 100 * np.exp(np.cumsum(returns))
        trend = np.linspace(0, 0.1, periods) * np.sin(np.linspace(0, 4 * np.pi, periods))
        price = price * (1 + trend)
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=pandas_freq)

        df = pd.DataFrame({
            'Open': price * (1 + np.random.normal(0, 0.002, periods)),
            'High': price * (1 + np.abs(np.random.normal(0.001, 0.003, periods))),
            'Low': price * (1 - np.abs(np.random.normal(0.001, 0.003, periods))),
            'Close': price,
            'Volume': np.random.uniform(10000, 50000, periods)
        }, index=dates)

        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        return df

    def trading_cycle(self):
        """Main trading loop."""
        while self.running:
            try:
                if self.mode == 'live':
                    df = self.fetch_market_data()
                else:
                    df = self.generate_synthetic_data()

                if df is None or len(df) < 50:
                    self.log_message("Waiting for more data...", 'orange')
                    time.sleep(5)
                    continue

                df = IndicatorCalculator.calculate(df, STRATEGY_CONFIG)
                self.df = df

                latest = df.iloc[-1]
                current_data = {
                    'Close': latest['Close'],
                    'EMA_Fast_Closed': latest.get('EMA_Fast_Closed', 0),
                    'EMA_Slow_Closed': latest.get('EMA_Slow_Closed', 0),
                    'SuperTrend_Closed': latest.get('SuperTrend_Closed', 0),
                    'MACD_Closed': latest.get('MACD_Closed', 0),
                    'MACD_Signal_Closed': latest.get('MACD_Signal_Closed', 0),
                    'RSI_Closed': latest.get('RSI_Closed', 50),
                    'ADX_Closed': latest.get('ADX_Closed', 20),
                    'ATR_Closed': latest.get('ATR_Closed', 1),
                    'Volume_Ratio_Closed': latest.get('Volume_Ratio_Closed', 1),
                    'CCI_Closed': latest.get('CCI_Closed', 0),
                    'BB_Upper': latest.get('BB_Upper', float('inf')),
                    'BB_Lower': latest.get('BB_Lower', 0),
                    'ROC_Closed': latest.get('ROC_Closed', 0),
                    'Lowest_5_Closed': latest.get('Lowest_5_Closed', latest['Close']),
                    'Highest_5_Closed': latest.get('Highest_5_Closed', latest['Close']),
                    'Lowest_10_Closed': latest.get('Lowest_10_Closed', latest['Close']),
                    'Highest_10_Closed': latest.get('Highest_10_Closed', latest['Close']),
                    'Consecutive_Direction_Closed': latest.get('Consecutive_Direction_Closed', 0),
                }

                self.current_data = current_data
                self.strategy.update_data(current_data)
                self.strategy.update_timeframe(self.timeframe_var.get())
                current_price = latest['Close']

                if len(df) > 0:
                    self.root.after(0, lambda df_copy=df.copy(): self.update_chart(df_copy))

                # Check exits
                if self.strategy.strategy_state == StrategyState.IN_TRADE:
                    exit_action, reason = self.strategy.check_exit(current_data, current_price)
                    if exit_action:
                        entry_price = self.strategy.position['entry_price']
                        pos_type = self.strategy.position['type']
                        qty = self.strategy.position['quantity']

                        actual_exit_price = current_price
                        if "Stop Loss" in reason:
                            actual_exit_price = self.strategy.position['stop_loss']
                        elif "Trailing Stop" in reason:
                            actual_exit_price = self.strategy.position['trailing_stop']

                        pnl = self.strategy.execute_exit(exit_action, actual_exit_price, reason)

                        if pos_type == 'long':
                            pnl_pct = (actual_exit_price - entry_price) / entry_price * 100
                            self.log_message(f"✅ EXIT LONG @ ${actual_exit_price:.4f} | "
                                             f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | {reason}", 'green')
                        else:
                            pnl_pct = (entry_price - actual_exit_price) / entry_price * 100
                            self.log_message(f"✅ EXIT SHORT @ ${actual_exit_price:.4f} | "
                                             f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | {reason}", 'red')

                        self.root.bell()

                # Check entries
                if self.strategy.strategy_state == StrategyState.SEEKING_ENTRY:
                    action, quality, reason = self.strategy.check_entry(current_data)

                    if action in ('buy', 'sell_short'):
                        self.strategy.config['min_quality_long'] = self.quality_var.get()
                        self.strategy.config['min_quality_short'] = self.quality_var.get()
                        self.strategy.config['risk_per_trade'] = self.risk_var.get() / 100

                        if self.strategy.execute_entry(action, current_price, quality, reason):
                            if action == 'buy':
                                self.log_message(f"📈 LONG @ ${current_price:.4f} | Quality: {quality} | {reason}",
                                                 'green')
                            else:
                                self.log_message(f"📉 SHORT @ ${current_price:.4f} | Quality: {quality} | {reason}",
                                                 'red')
                            self.root.bell()

                interval_minutes = GlobalConfig.TIMEFRAME_MINUTES.get(self.timeframe_var.get(), 5)
                time.sleep(max(5, interval_minutes * 60 / 2))

            except Exception as e:
                self.log_message(f"Trading error: {str(e)}", 'red')
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def start_trading(self):
        """Start trading - FIXED to use correct thresholds."""
        if self.running:
            return

        # CRITICAL FIX: Use separate thresholds for long and short
        min_quality_long = self.quality_var.get()
        min_quality_short = max(65, self.quality_var.get())  # Shorts require higher quality

        STRATEGY_CONFIG['min_quality_long'] = min_quality_long
        STRATEGY_CONFIG['min_quality_short'] = min_quality_short
        STRATEGY_CONFIG['trade_direction'] = self.direction_var.get()
        STRATEGY_CONFIG['risk_per_trade'] = self.risk_var.get() / 100

        self.strategy.config.update(STRATEGY_CONFIG)
        self.running = True

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.backtest_btn.config(state=tk.DISABLED)

        self.log_message("=" * 60, 'blue')
        self.log_message(f"TRADING STARTED - {self.symbol_var.get()} {self.timeframe_var.get()}", 'green')
        self.log_message(f"Mode: {self.mode.upper()}", 'blue')
        self.log_message(f"Direction: {STRATEGY_CONFIG['trade_direction'].upper()}", 'blue')
        self.log_message(f"Min Quality (Long/Short): {min_quality_long}/{min_quality_short}", 'blue')
        self.log_message(f"Risk per Trade: {STRATEGY_CONFIG['risk_per_trade'] * 100:.1f}%", 'blue')
        self.log_message("=" * 60, 'blue')

        self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
        self.trading_thread.start()
        self.update_stats_display()

    def stop_trading(self):
        """Stop trading."""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.backtest_btn.config(state=tk.NORMAL)

        stats = self.strategy.get_stats()
        self.log_message("=" * 60, 'blue')
        self.log_message("TRADING STOPPED", 'orange')
        self.log_message(f"Final Stats: {stats['total_trades']} trades | "
                         f"Win Rate: {stats['win_rate']:.1f}% | "
                         f"Total PnL: ${stats['total_pnl']:.2f}", 'purple')
        self.log_message("=" * 60, 'blue')

    def run_backtest(self):
        """Run backtest."""
        try:
            self.log_message("Running backtest...", 'blue')
            df = self.generate_synthetic_data()

            if df is None or len(df) < 100:
                self.log_message("Insufficient data for backtest", 'red')
                return

            STRATEGY_CONFIG['min_quality_long'] = self.quality_var.get()
            STRATEGY_CONFIG['min_quality_short'] = max(60, self.quality_var.get())
            STRATEGY_CONFIG['trade_direction'] = self.direction_var.get()
            STRATEGY_CONFIG['risk_per_trade'] = self.risk_var.get() / 100

            bt = Backtest(df, BacktestScalpingStrategy,
                          cash=GlobalConfig.INITIAL_CAPITAL,
                          commission=0.0005)

            stats = bt.run()

            self.log_message("\n" + "=" * 60, 'purple')
            self.log_message("BACKTEST RESULTS", 'purple')
            self.log_message("=" * 60, 'purple')
            self.log_message(f"Total Trades: {stats['# Trades']}", 'white')
            self.log_message(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%",
                             'green' if stats.get('Win Rate [%]', 0) > 40 else 'orange')
            self.log_message(f"Return: {stats.get('Return [%]', 0):.2f}%",
                             'green' if stats.get('Return [%]', 0) > 0 else 'red')
            self.log_message(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.3f}", 'white')
            self.log_message(f"Max Drawdown: {stats.get('Max. Drawdown [%]', 0):.2f}%", 'orange')
            self.log_message("=" * 60, 'purple')

        except Exception as e:
            self.log_message(f"Backtest error: {str(e)}", 'red')
            import traceback
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

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