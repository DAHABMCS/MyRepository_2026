"""
EMA + CCI + MACD Histogram Strategy
Based on: traderversity.com EMA-CCI-MACD article

Indicators:
  - EMA 50   : Primary dynamic support/resistance
  - EMA 110  : Secondary support/resistance
  - EMA 250  : Major trend boundary (last line of defense)
  - CCI      : Overbought/oversold timing oscillator
  - MACD Histogram : Trend direction/momentum filter

Entry Logic:
  BUY  : Price near EMA support + CCI crossed above 0 (from below -100) + MACD hist > 0
  SELL : Price near EMA resistance + CCI crossed below 0 (from above +100) + MACD hist < 0

Exit Logic:
  BUY  exit: CCI reaches overbought (+100), OR MACD hist turns negative, OR price breaks below EMA 250
  SELL exit: CCI reaches oversold (-100), OR MACD hist turns positive, OR price breaks above EMA 250
"""

import json
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np
import pandas as pd
import talib
from strategies.base3_New import BaseStrategy
from backtesting import Strategy


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION - Single source of truth for capital
# ═══════════════════════════════════════════════════════════════════════════

class GlobalConfig:
    """Single source of truth for global trading parameters"""

    # INITIAL CAPITAL - Change this value and EVERYTHING updates automatically
    INITIAL_CAPITAL = 50000.0  # <-- CHANGE THIS VALUE

    # Other global settings can go here too
    COMMISSION_RATE = 0.001
    DEFAULT_SYMBOL = "SOL-USDT"

    @classmethod
    def update_capital(cls, new_capital):
        """Update capital and log the change"""
        old = cls.INITIAL_CAPITAL
        cls.INITIAL_CAPITAL = float(new_capital)
        logging.info(f"💰 GLOBAL CAPITAL UPDATED: ${old:,.2f} → ${cls.INITIAL_CAPITAL:,.2f}")
        return cls.INITIAL_CAPITAL


# Create a convenience reference
CAPITAL = GlobalConfig.INITIAL_CAPITAL


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR ROBUST METRICS
# ═══════════════════════════════════════════════════════════════════════════

def safe_profit_factor(gross_profit: float, gross_loss: float) -> float:
    """
    Safely calculate profit factor, handling zero-loss cases
    Returns infinity if no losses, otherwise profit factor
    """
    if gross_loss <= 0:
        return float('inf')  # No losses = infinite profit factor
    return gross_profit / gross_loss


def summarize_performance(trades: List[Any], initial_capital: float = None) -> Dict:
    """
    Generate comprehensive performance summary with safe handling of edge cases
    """
    if initial_capital is None:
        initial_capital = GlobalConfig.INITIAL_CAPITAL

    num_trades = len(trades)
    if num_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_pct": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "avg_profit_per_trade": 0.0,
            "max_drawdown_pct": 0.0,
            "warning": "No trades to analyze"
        }

    # Calculate basic metrics
    total_profit = sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades)
    total_profit_pct = (total_profit / initial_capital) * 100

    wins = sum(1 for t in trades if getattr(t, 'profit', t.get('profit', 0)) > 0)
    losses = sum(1 for t in trades if getattr(t, 'profit', t.get('profit', 0)) <= 0)
    win_rate = (wins / num_trades) if num_trades > 0 else 0.0

    gross_profit = sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades
                       if getattr(t, 'profit', t.get('profit', 0)) > 0)
    gross_loss = -sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades
                      if getattr(t, 'profit', t.get('profit', 0)) < 0)

    # Safe profit factor
    profit_factor = safe_profit_factor(gross_profit, gross_loss)

    # Expectancy
    avg_win = (gross_profit / wins) if wins > 0 else 0.0
    avg_loss = (gross_loss / losses) if losses > 0 else 0.0
    loss_rate = losses / num_trades if num_trades > 0 else 0.0
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    # Max drawdown (simplified - you may want to enhance this)
    max_dd = 0.0
    if hasattr(trades[0], 'max_drawdown_pct'):
        max_dd = max(getattr(t, 'max_drawdown_pct', 0) for t in trades)
    elif isinstance(trades[0], dict):
        max_dd = max(t.get('max_drawdown_pct', 0) for t in trades)

    summary = {
        "total_trades": num_trades,
        "win_rate": round(win_rate, 4),
        "profit_pct": round(total_profit_pct, 4),
        "profit_factor": None if profit_factor == float('inf') else round(profit_factor, 4),
        "expectancy": round(expectancy, 6),
        "avg_profit_per_trade": round(total_profit / num_trades, 6) if num_trades else 0.0,
        "max_drawdown_pct": round(max_dd, 4),
        "wins": wins,
        "losses": losses,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }

    # Add warning for small sample size
    if num_trades < 30:
        summary["warning"] = f"⚠️ Small sample size ({num_trades} trades) - interpret with caution"
    else:
        summary["warning"] = ""

    return summary


def bootstrap_win_rate(trades: List[Any], n_iter: int = 1000, alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    """
    Calculate bootstrap confidence interval for win rate
    Returns (lower_bound, upper_bound) or None if insufficient data
    """
    num_trades = len(trades)
    if num_trades < 5:
        return None

    results = []
    for _ in range(n_iter):
        # Sample with replacement
        sample = [random.choice(trades) for _ in range(num_trades)]
        wins = sum(1 for t in sample if getattr(t, 'profit', t.get('profit', 0)) > 0)
        results.append(wins / num_trades)

    results.sort()
    lower_idx = int(n_iter * (alpha / 2))
    upper_idx = int(n_iter * (1 - alpha / 2))

    return results[lower_idx], results[upper_idx]


def load_params_with_validation(param_dict: Dict, defaults: Dict) -> Dict:
    """
    Load parameters with validation to prevent NaN/Inf values
    """
    if not isinstance(param_dict, dict):
        logging.warning("Momentum params file malformed; using defaults.")
        return defaults.copy()

    # Start with defaults
    updated = defaults.copy()

    # Update with provided values, but only for keys that exist in defaults
    for k, v in param_dict.items():
        if k in defaults:
            # Check for NaN/Inf in numeric values
            if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)):
                logging.warning(f"Param {k} is invalid (NaN/Inf); resetting to default {defaults[k]}")
                updated[k] = defaults[k]
            else:
                updated[k] = v
        else:
            logging.debug(f"Ignoring unknown parameter: {k}")

    return updated


# ═══════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

class StrategyState(Enum):
    SEEKING_ENTRY = auto()
    IN_TRADE = auto()


POSITION_ALREADY_OPEN_SENTINEL = -1


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    entry_time: datetime
    entry_price: float
    entry_size: float
    entry_tier: int
    entry_quality_score: int
    entry_reason: str
    entry_direction: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_size: Optional[float] = None
    exit_reason: Optional[str] = None
    profit: float = 0.0
    profit_pct: float = 0.0
    profit_r: float = 0.0
    max_profit: float = 0.0
    max_drawdown_pct: float = 0.0
    hold_duration: float = 0.0
    market_regime: str = "UNKNOWN"
    vix_level: float = 0.0
    partial_exits_taken: int = 0
    partial_pnl_realised: float = 0.0
    original_size: float = 0.0

    def to_dict(self) -> dict:
        return {
            'trade_id': self.trade_id, 'symbol': self.symbol,
            'entry_time': self.entry_time, 'entry_price': self.entry_price,
            'entry_size': self.entry_size, 'entry_tier': self.entry_tier,
            'entry_quality_score': self.entry_quality_score,
            'entry_direction': self.entry_direction,
            'exit_time': self.exit_time, 'exit_price': self.exit_price,
            'profit': self.profit, 'profit_pct': self.profit_pct,
            'profit_r': self.profit_r, 'exit_reason': self.exit_reason,
            'hold_duration': self.hold_duration, 'market_regime': self.market_regime,
            'partial_exits_taken': self.partial_exits_taken,
            'partial_pnl_realised': self.partial_pnl_realised,
        }


@dataclass
class RiskMetrics:
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_recovery_days: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST OPTIMIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

BACKTEST_PARAMS = {
    'ema_fast': {'active': True, 'values': [50], 'description': 'Fast EMA Period'},
    'ema_mid': {'active': True, 'values': [110], 'description': 'Mid EMA Period'},
    'ema_slow': {'active': True, 'values': [250], 'description': 'Slow EMA Period'},
    'cci_oversold': {'active': True, 'values': [-100], 'description': 'CCI Oversold Level'},
    'cci_overbought': {'active': True, 'values': [100], 'description': 'CCI Overbought Level'},
    'ema_proximity_pct': {'active': True, 'values': [0.003], 'description': 'EMA Proximity %'},
}


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: RISK MANAGEMENT — IMPROVED WITH ROBUST POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

class ProfessionalRiskController:
    def __init__(self, starting_equity: float = None):
        # Use global capital if not specified
        if starting_equity is None:
            starting_equity = GlobalConfig.INITIAL_CAPITAL

        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.peak_equity = starting_equity
        self.daily_loss_limit = starting_equity * 0.02
        self.weekly_loss_limit = starting_equity * 0.05
        self.monthly_loss_limit = starting_equity * 0.10
        self.max_drawdown_limit = starting_equity * 0.20
        self.max_consecutive_losses = 5
        self.max_position_size_pct = 0.10
        self.max_concentration_pct = 0.10
        self.max_active_trades = 10
        self.min_cash_reserve = 0.20
        self.base_risk_pct = 0.015
        self.trades: List[TradeRecord] = []
        self.daily_trades: deque = deque(maxlen=500)
        self.today_loss = 0.0
        self.this_week_loss = 0.0
        self.this_month_loss = 0.0
        self.today_date = datetime.now().date()
        self.equity_curve = [starting_equity]
        self.drawdown_curve = [0.0]
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.risk_metrics = RiskMetrics()

    def calculate_position_size(self, entry_price, stop_loss_price, win_rate=0.50,
                                profit_factor=1.0, quality_score=75, tier=1, adx=25):
        """
        IMPROVED: Robust position sizing with guard rails to prevent over-optimistic sizes
        """
        # Guard rails
        if entry_price <= 0:
            return 0

        risk_per_trade = abs(entry_price - stop_loss_price) / entry_price
        if risk_per_trade <= 0 or risk_per_trade > 0.25:  # Slightly increased max risk guard
            return 0

        # Expected profit per unit risk
        # Keep it simple to avoid over-sizing in low-liquidity regimes
        expected_profit = risk_per_trade * 2.0

        # Kelly-like sizing with bounds
        if expected_profit <= 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = ((win_rate * expected_profit) - ((1 - win_rate) * risk_per_trade)) / expected_profit
        kelly_fraction = max(0.0, min(kelly_fraction, 0.05))  # cap to avoid over-leveraging

        quality_weight = max(0.5, min((quality_score / 75) ** 0.5, 1.5))

        # Simple ADX-based risk adjustment (avoid extreme sizing in weak trends)
        if adx < 20:
            adx_multiplier = 0.6
        elif adx < 25:
            adx_multiplier = 0.9
        elif adx < 35:
            adx_multiplier = 1.0
        elif adx < 45:
            adx_multiplier = 0.8
        else:
            adx_multiplier = 0.5

        losing_streak_multiplier = 0.7 if self.consecutive_losses >= 3 else 1.0

        equity_health = self.current_equity / max(1.0, self.peak_equity)
        equity_adjustment = min(equity_health ** 2, 1.0)

        # Final risk percentage per trade
        risk_pct = (self.base_risk_pct * quality_weight * adx_multiplier *
                    losing_streak_multiplier * equity_adjustment)

        # Ensure risk_pct is within reasonable bounds
        risk_pct = max(0.001, min(risk_pct, 0.05))  # Between 0.1% and 5%

        risk_amount = self.current_equity * risk_pct
        # Position size in units of asset
        position_size = (risk_amount / (entry_price * risk_per_trade))
        max_position_amount = self.current_equity * self.max_position_size_pct

        # Ensure we don't exceed max position size per trade
        position_size = min(position_size, max_position_amount / entry_price)

        # Floor to at least 1 unit if the size is meaningful, else 0
        return max(0, int(position_size))

    def validate_trade_entry(self, position_size, entry_price):
        if self.today_loss <= -self.daily_loss_limit:
            return False, f"daily_loss_limit_reached_{self.today_loss:.2f}"
        if self.this_week_loss <= -self.weekly_loss_limit:
            return False, f"weekly_loss_limit_reached_{self.this_week_loss:.2f}"
        if self.this_month_loss <= -self.monthly_loss_limit:
            return False, f"monthly_loss_limit_reached_{self.this_month_loss:.2f}"
        drawdown_pct = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown_pct >= self.max_drawdown_limit:
            return False, f"max_drawdown_limit_{drawdown_pct:.1%}"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"consecutive_losses_{self.consecutive_losses}"
        position_cost = position_size * entry_price
        available_cash = self.current_equity * (1 - self.min_cash_reserve)
        if position_cost > available_cash:
            return False, f"insufficient_cash_{available_cash:.2f}"
        position_pct = (position_size * entry_price) / self.current_equity
        if position_pct > self.max_position_size_pct:
            return False, f"position_too_large_{position_pct:.1%}"
        return True, "passed_all_checks"

    def record_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self.total_trades += 1
        if trade.profit > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
        if trade.exit_time and trade.exit_time.date() == self.today_date:
            self.today_loss += trade.profit
        self.current_equity += trade.profit
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        self.equity_curve.append(self.current_equity)
        self._calculate_performance_metrics()

    def _calculate_performance_metrics(self):
        if self.total_trades == 0:
            return
        self.risk_metrics.win_rate = (self.winning_trades / self.total_trades) * 100
        wins = sum(t.profit for t in self.trades if t.profit > 0)
        losses = abs(sum(t.profit for t in self.trades if t.profit < 0))
        self.risk_metrics.profit_factor = safe_profit_factor(wins, losses)
        peak = self.starting_equity
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > self.risk_metrics.max_drawdown:
                self.risk_metrics.max_drawdown = dd
        self.risk_metrics.expectancy = sum(t.profit for t in self.trades) / self.total_trades
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            if len(returns) > 0 and np.std(returns) > 0:
                self.risk_metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

    def get_stats(self):
        return {
            'total_trades': self.total_trades,
            'win_rate': f"{self.risk_metrics.win_rate:.1f}%",
            'profit_factor': f"{self.risk_metrics.profit_factor:.2f}" if self.risk_metrics.profit_factor != float(
                'inf') else "∞",
            'max_drawdown': f"{self.risk_metrics.max_drawdown:.1%}",
            'sharpe_ratio': f"{self.risk_metrics.sharpe_ratio:.2f}",
            'expectancy': f"${self.risk_metrics.expectancy:.2f}",
            'current_equity': f"${self.current_equity:,.2f}",
            'total_profit': f"${self.current_equity - self.starting_equity:,.2f}",
            'roi': f"{((self.current_equity - self.starting_equity) / self.starting_equity):.1%}",
            'consecutive_losses': self.consecutive_losses,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: MACRO REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class MacroRegimeDetector:
    def __init__(self):
        self.current_regime = "UNKNOWN"
        self.regime_confidence = 0.0
        self.regime_history = deque(maxlen=100)
        self.adx_threshold = 25
        self.tier1_adx_hard_min = 25
        self.vix_low = 15
        self.vix_high = 30
        self.bb_squeeze_threshold = 0.4

    def detect_regime(self, ema_fast, ema_slow, adx, vix=20, bb_width_percentile=50):
        is_uptrend = ema_fast > ema_slow
        trend_strong = adx > self.adx_threshold
        is_low_vol = vix < self.vix_low
        is_high_vol = vix > self.vix_high
        is_range_bound = bb_width_percentile < self.bb_squeeze_threshold
        if is_uptrend and trend_strong and is_low_vol:
            regime, confidence = "BULLISH_LOW_VOL", 0.95
        elif is_uptrend and trend_strong and is_high_vol:
            regime, confidence = "BULLISH_HIGH_VOL", 0.85
        elif is_uptrend and not trend_strong:
            regime, confidence = "BULLISH_WEAK", 0.70
        elif not is_uptrend and trend_strong:
            regime, confidence = "BEARISH_DECLINING", 0.90
        elif is_range_bound:
            regime, confidence = "RANGING_VOLATILE", 0.80
        else:
            regime, confidence = "UNDEFINED", 0.50
        self.current_regime = regime
        self.regime_confidence = confidence
        self.regime_history.append(regime)
        return regime, confidence

    def get_position_multiplier(self, regime):
        return {'BULLISH_LOW_VOL': 1.3, 'BULLISH_HIGH_VOL': 0.8, 'BULLISH_WEAK': 0.7,
                'BEARISH_DECLINING': 0.3, 'RANGING_VOLATILE': 0.5, 'UNDEFINED': 0.6}.get(regime, 0.6)


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: EXIT MANAGER — EMA/CCI/MACD EXITS
# ═══════════════════════════════════════════════════════════════════════════

class ProfessionalExitManager:
    def __init__(self, config: dict):
        self.config = config

        # EMA CCI MACD Strategy Parameters
        self.ema_fast = config.get('ema_fast_period', 50)
        self.ema_mid = config.get('ema_mid_period', 110)
        self.ema_slow = config.get('ema_slow_period', 250)
        self.cci_oversold = config.get('cci_oversold', -100)
        self.cci_overbought = config.get('cci_overbought', 100)
        self.ema_proximity_pct = config.get('ema_proximity_pct', 0.003)

        # Exit conditions enabled flags
        self.cci_exit_enabled = config.get('cci_exit_enabled', True)
        self.macd_exit_enabled = config.get('macd_exit_enabled', True)
        self.ema_slow_exit_enabled = config.get('ema_slow_exit_enabled', True)

        # Risk parameters
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 2.5)
        self.max_hold_bars = config.get('max_hold_bars', 120)
        self.min_hold_bars_before_stop = config.get('min_hold_bars_before_stop', 6)
        self.emergency_stop_multiplier = config.get('emergency_stop_multiplier', 2.0)

    def _near_level(self, price: float, level: float) -> bool:
        """Returns True if price is within proximity % of a given level."""
        if level <= 0:
            return False
        return abs(price - level) / level <= self.ema_proximity_pct

    def _cci_crossed_above_zero(self, cci_prev: float, cci_curr: float) -> bool:
        """CCI crossed from negative to positive (zero-line cross up)."""
        return cci_prev < 0 < cci_curr

    def _cci_crossed_below_zero(self, cci_prev: float, cci_curr: float) -> bool:
        """CCI crossed from positive to negative (zero-line cross down)."""
        return cci_prev > 0 > cci_curr

    def _near_ema_support(self, price: float, ema50: float, ema110: float, ema250: float) -> bool:
        """Price is considered 'near EMA support' if it is close to any EMA level."""
        return (self._near_level(price, ema50) or
                self._near_level(price, ema110) or
                self._near_level(price, ema250))

    def _near_ema_resistance(self, price: float, ema50: float, ema110: float, ema250: float) -> bool:
        """Price is considered 'near EMA resistance' if it approaches any EMA from below."""
        return (self._near_level(price, ema50) or
                self._near_level(price, ema110) or
                self._near_level(price, ema250))

    def evaluate_exit(self, current_price, entry_price, stop_loss, highest_price, lowest_price,
                      bars_held, partial_exits, ema_fast, ema_mid, ema_slow,
                      macd, macd_signal, macd_prev, signal_prev, adx, atr,
                      position_type='long', trailing_activated=False, trailing_stop=None,
                      cci_curr=None, macd_hist=None):
        """
        Evaluate exit conditions for EMA + CCI + MACD Histogram Strategy

        Exit Logic:
          LONG exit: CCI reaches overbought (+100), OR MACD hist turns negative,
                     OR price breaks below EMA 250
          SHORT exit: CCI reaches oversold (-100), OR MACD hist turns positive,
                      OR price breaks above EMA 250
        """

        # 🔍 DEBUG: Print version marker on first call
        if not hasattr(self, '_version_printed'):
            print("=" * 70)
            print("🎯 EXIT MANAGER VERSION: EMA + CCI + MACD Histogram Strategy")
            print("=" * 70)
            self._version_printed = True

        stop_distance = atr * self.stop_loss_atr_mult

        # Calculate profit based on position type
        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            profit_r = (current_price - entry_price) / stop_distance if stop_distance > 0 else 0
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
            profit_r = (entry_price - current_price) / stop_distance if stop_distance > 0 else 0

        # ═══ 1. HARD STOP — Always protect capital ═══════════════════════
        if position_type == 'long':
            if current_price <= stop_loss:
                if bars_held < self.min_hold_bars_before_stop:
                    emergency_stop = entry_price - (stop_distance * self.emergency_stop_multiplier)
                    if current_price <= emergency_stop:
                        return "stop_loss_hard_emergency", 1.0
                else:
                    return "stop_loss_hard", 1.0
        else:  # short
            if current_price >= stop_loss:
                if bars_held < self.min_hold_bars_before_stop:
                    emergency_stop = entry_price + (stop_distance * self.emergency_stop_multiplier)
                    if current_price >= emergency_stop:
                        return "stop_loss_hard_emergency", 1.0
                else:
                    return "stop_loss_hard", 1.0

        # ═══ 2. LONG EXIT CONDITIONS ═══════════════════════════════════════
        if position_type == 'long':
            # Condition A: CCI reaches overbought (+100)
            if self.cci_exit_enabled and cci_curr is not None:
                if cci_curr >= self.cci_overbought:
                    print(f"  🎯 LONG EXIT: CCI overbought ({cci_curr:.1f} ≥ +{self.cci_overbought})")
                    return "cci_overbought", 1.0

            # Condition B: MACD histogram turns negative
            if self.macd_exit_enabled and macd_hist is not None:
                if macd_hist < 0:
                    print(f"  🎯 LONG EXIT: MACD hist turned negative ({macd_hist:.5f})")
                    return "macd_hist_negative", 1.0

            # Condition C: Price breaks below EMA 250 (major structure break)
            if self.ema_slow_exit_enabled and ema_slow > 0:
                if current_price < ema_slow:
                    print(f"  🎯 LONG EXIT: Price broke below EMA 250 (${current_price:.2f} < ${ema_slow:.2f})")
                    return "ema_slow_break_below", 1.0

        # ═══ 3. SHORT EXIT CONDITIONS ══════════════════════════════════════
        else:  # short position
            # Condition A: CCI reaches oversold (-100)
            if self.cci_exit_enabled and cci_curr is not None:
                if cci_curr <= self.cci_oversold:
                    print(f"  🎯 SHORT EXIT: CCI oversold ({cci_curr:.1f} ≤ {self.cci_oversold})")
                    return "cci_oversold", 1.0

            # Condition B: MACD histogram turns positive
            if self.macd_exit_enabled and macd_hist is not None:
                if macd_hist > 0:
                    print(f"  🎯 SHORT EXIT: MACD hist turned positive ({macd_hist:.5f})")
                    return "macd_hist_positive", 1.0

            # Condition C: Price breaks above EMA 250 (major structure break)
            if self.ema_slow_exit_enabled and ema_slow > 0:
                if current_price > ema_slow:
                    print(f"  🎯 SHORT EXIT: Price broke above EMA 250 (${current_price:.2f} > ${ema_slow:.2f})")
                    return "ema_slow_break_above", 1.0

        # ═══ 4. MAX HOLD TIME — Prevent capital being locked too long ═══
        if bars_held >= self.max_hold_bars:
            return "max_hold_time", 1.0

        return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: CONFIGURATION — EMA/CCI/MACD PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

MOMENTUM_PARAMS = {
    # ═══════════════════════════════════════════════════════════════════
    # EMA + CCI + MACD Histogram Strategy Parameters
    # Based on: traderversity.com EMA-CCI-MACD article
    # ═══════════════════════════════════════════════════════════════════

    # EMA Settings
    "ema_fast_period": 50,
    "ema_mid_period": 110,
    "ema_slow_period": 250,

    # CCI Settings
    "cci_period": 20,
    "cci_oversold": -100,
    "cci_overbought": 100,

    # EMA Proximity
    "ema_proximity_pct": 0.003,  # price within 0.3% of EMA counts as "near"

    # Exit Control Flags
    "cci_exit_enabled": True,
    "macd_exit_enabled": True,
    "ema_slow_exit_enabled": True,

    # MACD Settings (standard)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # ADX Settings (for trend filtering - optional)
    "adx_min": 20,
    "adx_period": 14,
    "adx_filter_enabled": False,  # Optional ADX filter


    # Risk Management
    "risk_per_trade": 0.012,
    "risk_full_position": 0.012,
    "risk_reduced_position": 0.008,
    "risk_aggressive_position": 0.015,
    "stop_loss_pct": 0.02,  # <-- ADD THIS LINE (2% stop loss)

    # Stop Loss & Trailing (optional - can be disabled)
    "stop_loss_atr_mult": 2.5,
    "atr_period": 14,
    "trailing_activation_pct": 0.015,  # 1.5% activation (optional)
    "trailing_distance_pct": 0.01,  # 1.0% trail distance (optional)
    "trailing_stop_enabled": False,  # Disabled by default for EMA/CCI/MACD strategy

    # Trade Management
    "max_daily_trades": 8,
    "min_bars_between_trades": 2,
    "cooldown_after_loss_bars": 3,
    "max_hold_bars": 120,
    "min_hold_bars_before_stop": 6,
    "emergency_stop_multiplier": 2.0,

    # ═══ DIRECTION CONTROL ════════════════════════════════════════════
    "trade_direction": "both",  # "long" | "short" | "both"

    # Quality scoring (simplified for this strategy)
    "quality_score_enabled": False,
    "quality_tier1_min": 60,
    "quality_tier2_min": 75,

    # Tier control (not used in this strategy)
    "only_tier2_entries": False,
    "backtest_only_tier2_active": True,
    "backtest_only_tier2_values": [True, False],

    # RSI (optional filter)
    "rsi_period": 14,
    "rsi_filter_enabled": False,

    # Volume (optional filter)
    "volume_period": 20,
    "volume_filter_enabled": False,


    # Regime Detection
    "regime_filter_enabled": False,


}


# ═══════════════════════════════════════════════════════════════════════════
# PART 6: CONFIGURATION MANAGER — UPDATED WITH PROPER HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════

class MomentumConfig:
    CONFIG_FILE = "strategy_settings.json"
    _custom_params = {}
    _current_mode = "Default Parameters"
    _saved_mode = "Default Parameters"

    @classmethod
    def get_config(cls, momentum_params_override=None):
        """
        Load config with proper hierarchy (SINGLE SOURCE OF TRUTH):
        1. ALWAYS start with MOMENTUM_PARAMS defaults (code)
        2. If Custom mode, overlay saved custom params
        3. Apply runtime overrides if provided
        """
        try:
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: ALWAYS START WITH MOMENTUM_PARAMS (SINGLE SOURCE OF TRUTH)
            # ═══════════════════════════════════════════════════════════════
            config = MOMENTUM_PARAMS.copy()
            param_sources = {}

            for key in config.keys():
                param_sources[key] = {
                    'value': config[key],
                    'source': 'MOMENTUM_PARAMS',
                    'overridden': False
                }

            logging.info("=" * 80)
            logging.info("📋 STEP 1: Loading MOMENTUM_PARAMS (Single Source of Truth)")
            logging.info("=" * 80)
            logging.info(f"Loaded {len(config)} default parameters from code")

            # ═══════════════════════════════════════════════════════════════
            # STEP 2: LOAD CUSTOM PARAMS FROM FILE (IF EXISTS)
            # ═══════════════════════════════════════════════════════════════
            cls._custom_params = {}
            custom_params_applied = 0

            if os.path.exists(cls.CONFIG_FILE):
                try:
                    with open(cls.CONFIG_FILE, 'r') as f:
                        saved = json.load(f)
                        cls._custom_params = saved.get('custom_params', {})

                        # Load mode preference
                        if 'selected_mode' in saved:
                            cls._saved_mode = saved['selected_mode']
                            cls._current_mode = cls._saved_mode

                    logging.info("")
                    logging.info("=" * 80)
                    logging.info("📋 STEP 2: Loading Custom Parameters from File")
                    logging.info("=" * 80)
                    logging.info(f"File: {cls.CONFIG_FILE}")
                    logging.info(f"Mode: {cls._current_mode}")
                    logging.info(f"Custom params available: {len(cls._custom_params)}")

                    # Apply custom params ONLY if in Custom mode
                    if cls._current_mode == "Custom Parameters" and cls._custom_params:
                        logging.info("")
                        logging.info("Applying custom parameters (mode = Custom):")
                        for key, value in sorted(cls._custom_params.items()):
                            if key in config:
                                old_value = config[key]
                                config[key] = value
                                param_sources[key] = {
                                    'value': value,
                                    'source': 'Custom params (file)',
                                    'overridden': True
                                }
                                if old_value != value:
                                    logging.info(f"  ✓ {key}: {old_value} → {value}")
                                    custom_params_applied += 1
                            else:
                                logging.warning(f"  ⚠️ Unknown parameter: {key}")
                    else:
                        logging.info("Mode is Default - NOT applying custom params")

                except Exception as e:
                    logging.error(f"Error loading saved config: {e}")
                    cls._custom_params = {}
            else:
                logging.info(f"ℹ️ No saved settings file found")

            # ═══════════════════════════════════════════════════════════════
            # STEP 3: APPLY RUNTIME OVERRIDES (FROM APP STATE)
            # ═══════════════════════════════════════════════════════════════
            runtime_overrides_applied = 0

            if momentum_params_override and isinstance(momentum_params_override, dict):
                logging.info("")
                logging.info("=" * 80)
                logging.info("📋 STEP 3: Applying Runtime Overrides")
                logging.info("=" * 80)
                logging.info(f"Overrides provided: {len(momentum_params_override)}")
                logging.info("")

                for key, value in sorted(momentum_params_override.items()):
                    if key in config:
                        old_value = config[key]
                        config[key] = value

                        param_sources[key] = {
                            'value': value,
                            'source': 'Runtime override',
                            'overridden': True
                        }

                        if old_value != value:
                            logging.info(f"  ✓ {key}: {old_value} → {value}")
                            runtime_overrides_applied += 1
                    else:
                        logging.warning(f"  ⚠️ Unknown parameter: {key}")

            # ═══════════════════════════════════════════════════════════════
            # FINAL SUMMARY
            # ═══════════════════════════════════════════════════════════════
            logging.info("")
            logging.info("=" * 80)
            logging.info("📊 FINAL CONFIGURATION SUMMARY")
            logging.info("=" * 80)
            logging.info(f"Total parameters: {len(config)}")
            logging.info(f"Mode: {cls._current_mode}")
            logging.info(f"Custom params applied: {custom_params_applied}")
            logging.info(f"Runtime overrides applied: {runtime_overrides_applied}")

            # Show critical parameters with sources
            logging.info("")
            logging.info("Critical parameters (EMA + CCI + MACD Strategy):")
            critical_params = [
                'ema_fast_period',
                'ema_mid_period',
                'ema_slow_period',
                'cci_oversold',
                'cci_overbought',
                'ema_proximity_pct',
                'trade_direction',
                'stop_loss_atr_mult'
            ]

            for param in critical_params:
                if param in param_sources:
                    info = param_sources[param]
                    source = info['source']
                    value = info['value']

                    if info['overridden']:
                        prefix = "🔴"
                    else:
                        prefix = "🔵"

                    logging.info(f"  {prefix} {param} = {value} [{source}]")

            logging.info("=" * 80)

            return config

        except Exception as e:
            logging.error(f"❌ Config load error: {e}")
            logging.error("Falling back to MOMENTUM_PARAMS")
            return MOMENTUM_PARAMS.copy()

    @classmethod
    def get_custom_params(cls):
        """Get custom parameters loaded from file"""
        return cls._custom_params

    @classmethod
    def get_current_mode(cls):
        """Get the saved mode preference (Default/Custom)"""
        return cls._current_mode

    @classmethod
    def set_current_mode(cls, mode):
        """Set the current mode preference"""
        cls._current_mode = mode
        cls._saved_mode = mode

    @classmethod
    def save_config(cls, config, custom_params=None, selected_mode=None):
        """
        Save configuration to file.

        CRITICAL: Only save CUSTOM params (deltas from MOMENTUM_PARAMS)
        NOT the entire config!
        """
        try:
            os.makedirs("strategy_configs", exist_ok=True)

            # Determine which mode we're saving
            mode = selected_mode if selected_mode else cls._current_mode

            # Calculate deltas (only params that differ from MOMENTUM_PARAMS)
            if mode == "Custom Parameters" and custom_params:
                # Save only the custom params (deltas)
                params_to_save = custom_params
                logging.info(f"💾 Saving {len(params_to_save)} custom parameters")
            else:
                # Default mode - save empty custom params
                params_to_save = {}
                logging.info("💾 Saving Default mode (no custom params)")

            save_data = {
                'timestamp': datetime.now().isoformat(),
                'selected_mode': mode,
                'custom_params': params_to_save,
                'note': 'MOMENTUM_PARAMS in code is the single source of truth for defaults'
            }

            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(save_data, f, indent=4)

            # Update class state
            cls._custom_params = params_to_save
            cls._current_mode = mode
            cls._saved_mode = mode

            logging.info(f"✅ Settings saved to {cls.CONFIG_FILE}")
            logging.info(f"   Mode: {mode}")
            logging.info(f"   Custom params: {len(params_to_save)}")

            return True

        except Exception as e:
            logging.error(f"❌ Config save error: {e}")
            return False

    @classmethod
    def reset_to_defaults(cls):
        """Reset to code defaults (MOMENTUM_PARAMS) - single source of truth"""
        logging.info(f"🔄 Reset to MOMENTUM_PARAMS defaults (single source of truth)")
        return MOMENTUM_PARAMS.copy()

    @classmethod
    def validate_config(cls, config):
        """Validate and fix any inconsistencies in the config"""
        modified = False

        # Ensure trade_direction is valid
        if 'trade_direction' in config:
            if config['trade_direction'] not in ['long', 'short', 'both']:
                config['trade_direction'] = 'both'
                modified = True

        return config, modified


# ═══════════════════════════════════════════════════════════════════════════
# PART 7: INDICATOR CALCULATOR (EMA/CCI/MACD)
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorCalculator:
    @staticmethod
    def calculate(df, params):
        df = df.copy()
        try:
            # EMA Indicators
            df['EMA_Fast'] = talib.EMA(df['Close'], params['ema_fast_period'])
            df['EMA_Mid'] = talib.EMA(df['Close'], params['ema_mid_period'])
            df['EMA_Slow'] = talib.EMA(df['Close'], params['ema_slow_period'])

            # CCI Indicator
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], params['cci_period'])

            # MACD Indicator
            df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(
                df['Close'],
                fastperiod=params['macd_fast'],
                slowperiod=params['macd_slow'],
                signalperiod=params['macd_signal']
            )

            # ADX (optional trend filter)
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], params['adx_period'])

            # ATR for stop loss
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], params['atr_period'])

            # Optional RSI
            if params.get('rsi_filter_enabled', False):
                df['RSI'] = talib.RSI(df['Close'], params['rsi_period'])

            # Optional Volume MA
            if params.get('volume_filter_enabled', False):
                df['Volume_MA'] = talib.SMA(df['Volume'], params['volume_period'])
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['Volume_Ratio'] = np.where(df['Volume_MA'] > 0,
                                                  df['Volume'] / df['Volume_MA'], 1.0)
                df['Volume_Ratio'] = (df['Volume_Ratio']
                                      .replace([np.inf, -np.inf], np.nan)
                                      .fillna(1.0).clip(0.01, 10.0))

            # Shift indicators for closed values (needed for cross detection)
            indicators_to_shift = ['EMA_Fast', 'EMA_Mid', 'EMA_Slow', 'CCI',
                                   'MACD', 'MACD_Signal', 'MACD_Histogram', 'ADX', 'ATR']

            for col in indicators_to_shift:
                if col in df.columns:
                    df[f'{col}_closed'] = df[col].shift(1)

            return df
        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
            raise


# ═══════════════════════════════════════════════════════════════════════════
# PART 8: CORE MOMENTUM LOGIC — EMA + CCI + MACD Strategy
# ═══════════════════════════════════════════════════════════════════════════

class MomentumLogic:
    """
    EMA + CCI + MACD Histogram Strategy Logic
    Based on: traderversity.com EMA-CCI-MACD article
    """

    def __init__(self, config=None, trading_app=None):
        # Get momentum_params from trading_app if available (for overrides)
        momentum_params_override = None
        if trading_app and hasattr(trading_app, 'get_current_momentum_params'):
            momentum_params_override = trading_app.get_current_momentum_params()

        # Load config with proper hierarchy
        if config is None:
            self.config = MomentumConfig.get_config(momentum_params_override)
        else:
            self.config = config

        self.trading_app = trading_app

        # Store custom params and mode for reference
        self.custom_params = MomentumConfig.get_custom_params()
        self.current_mode = MomentumConfig.get_current_mode()

        # Set all config values as attributes
        for key, value in self.config.items():
            setattr(self, key, value)

        # Ensure critical parameters exist
        required_params = {
            'ema_fast_period': 50,
            'ema_mid_period': 110,
            'ema_slow_period': 250,
            'cci_oversold': -100,
            'cci_overbought': 100,
            'ema_proximity_pct': 0.003,
            'trade_direction': 'both',
            'stop_loss_atr_mult': 2.5,
            'cci_exit_enabled': True,
            'macd_exit_enabled': True,
            'ema_slow_exit_enabled': True,
        }

        for param_name, default_value in required_params.items():
            if not hasattr(self, param_name) or getattr(self, param_name) is None:
                setattr(self, param_name, default_value)
                if self.trading_app and hasattr(self.trading_app, 'log_message'):
                    self.trading_app.log_message(
                        f"⚠️ Added missing param: {param_name} = {default_value}",
                        "orange"
                    )

        # Initialize components
        self.risk_controller = ProfessionalRiskController()
        self.regime_detector = MacroRegimeDetector()
        self.exit_manager = ProfessionalExitManager(self.config)
        self.strategy_state = StrategyState.SEEKING_ENTRY
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.equity_curve = [GlobalConfig.INITIAL_CAPITAL]
        self.trade_history = []
        self.last_trade_bar = -999
        self.bar_count = 0
        self.bars_held = 0
        self.current_regime = "UNKNOWN"
        self.regime_changes = []
        self.trade_counter = 0
        self.tier1_trades = 0
        self.tier2_trades = 0
        self._last_quality_score = 0
        self._last_entry_tier = None
        self._current_df = None
        self.backtest_params = BACKTEST_PARAMS.copy()

        # Pending signal tracking for next-bar execution
        self._pending_signal = None
        self._signal_bar = -999
        self._signal_price = None

        self.trade_direction = (
            trading_app.trade_direction_var.get()
            if trading_app and hasattr(trading_app, 'trade_direction_var')
            else self.config.get('trade_direction', 'both')
        )
        self.only_long_entries = (self.trade_direction == 'long')
        self.only_short_entries = (self.trade_direction == 'short')

        self._last_direction_check = {
            'bar': -999,
            'result': True,
            'reason': '',
            'action': 'hold'
        }
        self._suggested_action = None

        # Log parameter source
        self._log_parameter_source()

    def _log_parameter_source(self):
        """Log where parameters came from"""
        logging.info("=" * 70)
        logging.info("📊 PARAMETER SOURCE (EMA + CCI + MACD Strategy)")
        logging.info("=" * 70)
        logging.info(f"   Base source: MOMENTUM_PARAMS (code defaults)")

        if self.custom_params:
            logging.info(f"   Custom params loaded: {len(self.custom_params)} parameters")
        else:
            logging.info(f"   Custom params: None")

        logging.info(f"   Current mode: {self.current_mode}")
        logging.info(
            f"   EMA Periods: Fast={self.ema_fast_period}, Mid={self.ema_mid_period}, Slow={self.ema_slow_period}")
        logging.info(f"   CCI Levels: Oversold={self.cci_oversold}, Overbought={self.cci_overbought}")
        logging.info(f"   Direction: {self.trade_direction.upper()}")
        logging.info("=" * 70)

    def _log(self, message, color="white"):
        """Log message through trading app if available"""
        if self.trading_app and hasattr(self.trading_app, 'log_message'):
            self.trading_app.log_message(message, color)
        else:
            print(f"[{color}] {message}")

    # ── State helpers ────────────────────────────────────────────────
    def _transition_to_in_trade(self):
        self.strategy_state = StrategyState.IN_TRADE

    def _transition_to_seeking_entry(self):
        self.strategy_state = StrategyState.SEEKING_ENTRY

    # ── Helper functions for EMA/CCI/MACD Strategy ──────────────────
    def _near_level(self, price: float, level: float) -> bool:
        """Returns True if price is within proximity % of a given level."""
        if level <= 0:
            return False
        return abs(price - level) / level <= self.ema_proximity_pct

    def _cci_crossed_above_zero(self, cci_prev: float, cci_curr: float) -> bool:
        """CCI crossed from negative to positive (zero-line cross up)."""
        return cci_prev < 0 < cci_curr

    def _cci_crossed_below_zero(self, cci_prev: float, cci_curr: float) -> bool:
        """CCI crossed from positive to negative (zero-line cross down)."""
        return cci_prev > 0 > cci_curr

    def _cci_was_oversold(self, cci_prev: float) -> bool:
        """CCI was below oversold threshold."""
        return cci_prev <= self.cci_oversold

    def _cci_was_overbought(self, cci_prev: float) -> bool:
        """CCI was above overbought threshold."""
        return cci_prev >= self.cci_overbought

    def _near_ema_support(self, price: float, ema_fast: float, ema_mid: float, ema_slow: float) -> bool:
        """Price is considered 'near EMA support' if it is close to any EMA level."""
        return (self._near_level(price, ema_fast) or
                self._near_level(price, ema_mid) or
                self._near_level(price, ema_slow))

    def _near_ema_resistance(self, price: float, ema_fast: float, ema_mid: float, ema_slow: float) -> bool:
        """Price is considered 'near EMA resistance' if it approaches any EMA from below."""
        return (self._near_level(price, ema_fast) or
                self._near_level(price, ema_mid) or
                self._near_level(price, ema_slow))

    def _validate_direction(self, data):
        """
        Centralized direction validation with comprehensive checks
        """
        direction = getattr(self, 'trade_direction', 'both').lower()

        # Cache check - only validate once per bar
        if self._last_direction_check['bar'] == self.bar_count:
            return (self._last_direction_check['result'],
                    self._last_direction_check['reason'],
                    self._last_direction_check['action'])

        # Get current market data
        close = data.get('Close', 0)
        ema_fast = data.get('EMA_Fast', 0)
        ema_slow = data.get('EMA_Slow', 0)
        cci_curr = data.get('CCI', 0)
        macd_hist = data.get('MACD_Histogram', 0)

        # LONG-ONLY MODE
        if direction == 'long':
            # Check: Price should be above EMA Slow for long bias
            if close <= ema_slow:
                result = (False, f"LONG-ONLY: Price below EMA 250 ({close:.2f} <= {ema_slow:.2f})", "hold")
                self._last_direction_check = {'bar': self.bar_count, 'result': False, 'reason': result[1],
                                              'action': 'hold'}
                return result

            result = (True, "LONG-ONLY: Valid bullish setup", "buy")
            self._last_direction_check = {'bar': self.bar_count, 'result': True, 'reason': result[1], 'action': 'buy'}
            return result

        # SHORT-ONLY MODE
        elif direction == 'short':
            # Check: Price should be below EMA Slow for short bias
            if close >= ema_slow:
                result = (False, f"SHORT-ONLY: Price above EMA 250 ({close:.2f} >= {ema_slow:.2f})", "hold")
                self._last_direction_check = {'bar': self.bar_count, 'result': False, 'reason': result[1],
                                              'action': 'hold'}
                return result

            result = (True, "SHORT-ONLY: Valid bearish setup", "sell")
            self._last_direction_check = {'bar': self.bar_count, 'result': True, 'reason': result[1], 'action': 'sell'}
            return result

        # BOTH MODE - allow any direction
        else:
            # Determine if setup is bullish or bearish based on EMA alignment
            is_bullish = (ema_fast > ema_slow)
            is_bearish = (ema_fast < ema_slow)

            # Also check CCI and MACD for additional confirmation
            if is_bullish and cci_curr > 0 and macd_hist > 0:
                result = (True, "BOTH: Bullish setup (EMA uptrend, CCI>0, MACD hist>0)", "buy")
                self._last_direction_check = {'bar': self.bar_count, 'result': True, 'reason': result[1],
                                              'action': 'buy'}
                return result
            elif is_bearish and cci_curr < 0 and macd_hist < 0:
                result = (True, "BOTH: Bearish setup (EMA downtrend, CCI<0, MACD hist<0)", "sell")
                self._last_direction_check = {'bar': self.bar_count, 'result': True, 'reason': result[1],
                                              'action': 'sell'}
                return result
            elif is_bullish:
                result = (True, "BOTH: Bullish bias (EMA uptrend)", "buy")
                self._last_direction_check = {'bar': self.bar_count, 'result': True, 'reason': result[1],
                                              'action': 'buy'}
                return result
            elif is_bearish:
                result = (True, "BOTH: Bearish bias (EMA downtrend)", "sell")
                self._last_direction_check = {'bar': self.bar_count, 'result': True, 'reason': result[1],
                                              'action': 'sell'}
                return result
            else:
                result = (False, "BOTH: No clear direction", "hold")
                self._last_direction_check = {'bar': self.bar_count, 'result': False, 'reason': result[1],
                                              'action': 'hold'}
                return result

    def _check_entry_conditions(self, data):
        """
        Entry conditions for EMA + CCI + MACD Histogram Strategy

        BUY  : Price near EMA support + CCI crossed above 0 (from below -100) + MACD hist > 0
        SELL : Price near EMA resistance + CCI crossed below 0 (from above +100) + MACD hist < 0
        """

        # ═══ STEP 0: DIRECTION VALIDATION — MUST PASS FIRST ═══════════════
        direction_valid, direction_reason, suggested_action = self._validate_direction(data)

        if not direction_valid:
            return "hold", 0, None, direction_reason, {}

        # Store the suggested action for later use
        self._suggested_action = suggested_action

        # Get current data
        price = data.get('Close', 0)
        ema_fast = data.get('EMA_Fast', 0)
        ema_mid = data.get('EMA_Mid', 0)
        ema_slow = data.get('EMA_Slow', 0)
        cci_curr = data.get('CCI', 0)
        cci_prev = data.get('CCI_closed', 0)
        macd_hist = data.get('MACD_Histogram', 0)

        # Get previous MACD histogram for cross detection
        macd_hist_prev = data.get('MACD_Histogram_closed', 0)

        # ── BUY conditions ──────────────────────────────────────────────────────
        # 1. Price near a known EMA support level
        # 2. CCI was below -100 previously and has now crossed back above 0
        # 3. MACD histogram is positive (bullish trend filter)

        cci_was_oversold = self._cci_was_oversold(cci_prev)
        cci_zero_cross_up = self._cci_crossed_above_zero(cci_prev, cci_curr)
        price_at_support = self._near_ema_support(price, ema_fast, ema_mid, ema_slow)
        macd_bullish = macd_hist > 0

        buy_signal = (
                price_at_support
                and cci_was_oversold
                and cci_zero_cross_up
                and macd_bullish
        )

        # ── SELL conditions ─────────────────────────────────────────────────────
        # 1. Price near a known EMA resistance level
        # 2. CCI was above +100 previously and has now crossed back below 0
        # 3. MACD histogram is negative (bearish trend filter)

        cci_was_overbought = self._cci_was_overbought(cci_prev)
        cci_zero_cross_down = self._cci_crossed_below_zero(cci_prev, cci_curr)
        price_at_resistance = self._near_ema_resistance(price, ema_fast, ema_mid, ema_slow)
        macd_bearish = macd_hist < 0

        sell_signal = (
                price_at_resistance
                and cci_was_overbought
                and cci_zero_cross_down
                and macd_bearish
        )

        # Determine which signal to use based on direction
        decision = None
        reason = None

        if self.trade_direction == 'both':
            if buy_signal:
                decision = "buy"
                reason = f"BUY: Price near EMA support + CCI crossed above 0 from {cci_prev:.1f} to {cci_curr:.1f} + MACD hist={macd_hist:.5f}>0"
            elif sell_signal:
                decision = "sell"
                reason = f"SELL: Price near EMA resistance + CCI crossed below 0 from {cci_prev:.1f} to {cci_curr:.1f} + MACD hist={macd_hist:.5f}<0"
        elif self.trade_direction == 'long' and buy_signal:
            decision = "buy"
            reason = f"BUY: Price near EMA support + CCI crossed above 0 from {cci_prev:.1f} to {cci_curr:.1f} + MACD hist={macd_hist:.5f}>0"
        elif self.trade_direction == 'short' and sell_signal:
            decision = "sell"
            reason = f"SELL: Price near EMA resistance + CCI crossed below 0 from {cci_prev:.1f} to {cci_curr:.1f} + MACD hist={macd_hist:.5f}<0"

        if decision:
            # Store signal for next-bar execution
            quality_score = 75  # Default quality score for this strategy
            self._pending_signal = {
                'decision': decision,
                'quality_score': quality_score,
                'tier': 1,
                'position_mult': 1.0,
                'breakdown': reason,
                'component_scores': {},
                'signal_price': price,
                'signal_cci': cci_curr,
                'signal_macd_hist': macd_hist,
                'signal_ema_fast': ema_fast,
                'signal_ema_slow': ema_slow,
                'signal_bar': self.bar_count,
                'signal_time': datetime.now(timezone.utc),
            }
            self._signal_bar = self.bar_count
            self._signal_price = price

            self._log(f"📊 {decision.upper()} SIGNAL PENDING: {reason} @ ${price:.2f} (will execute next bar)", "purple")

            return ("hold", quality_score, None, f"{decision.upper()}_SIGNAL_PENDING_execute_next_bar", {})

        return ("hold", 0, None, "No entry signal — conditions not fully met.", {})

    def calculate_position_size(self, equity, atr, price,
                                quality_score=75, tier=1, position_mult=1.0):
        stop_distance = atr * getattr(self, 'stop_loss_atr_mult', 2.5)
        stop_loss = price - stop_distance if self._suggested_action == 'buy' else price + stop_distance
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 50
        pf = (self.risk_controller.risk_metrics.profit_factor
              if self.risk_controller.risk_metrics.profit_factor > 0 else 1.0)
        adx = 25
        if hasattr(self, '_current_df') and self._current_df is not None:
            try:
                adx = float(self._current_df['ADX'].iloc[-1])
            except:
                pass
        base_size = self.risk_controller.calculate_position_size(
            entry_price=price, stop_loss_price=stop_loss,
            win_rate=win_rate / 100, profit_factor=pf,
            quality_score=quality_score, tier=tier, adx=adx
        )
        regime_mult = self.regime_detector.get_position_multiplier(self.current_regime)
        return max(1, int(base_size * regime_mult * position_mult))

    def record_trade(self, profit, exit_reason="unknown", tier=None, size=None,
                     direction=None, entry_quality=None, entry_price=None,
                     exit_price=None, hold_duration=None, entry_bar=None, exit_bar=None,
                     signal_cci=None, signal_macd_hist=None, signal_ema_fast=None,
                     signal_ema_slow=None, signal_price=None, signal_time=None, signal_bar=None):
        """
        Record a completed trade with accurate signal data
        """
        corrected_size = size
        trade_direction = direction or self.trade_direction

        if size is not None and size < 0 and trade_direction == 'long':
            self._log(f"⚠️ WARNING: Negative size ({size}) recorded in LONG-ONLY mode! "
                      f"Auto-correcting to {abs(size)} for analysis", "bold red")
            corrected_size = abs(size)

        self.total_trades += 1
        self.total_profit += profit
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Track tier counts
        if tier == 1:
            self.tier1_trades += 1
        elif tier == 2:
            self.tier2_trades += 1

        # Create comprehensive trade record
        trade_record = {
            'profit': profit,
            'exit_reason': exit_reason,
            'tier': tier,
            'size': corrected_size,
            'original_size': size,
            'direction': trade_direction,
            'entry_quality': entry_quality,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'hold_duration': hold_duration,
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            # Signal data
            'signal_cci': signal_cci,
            'signal_macd_hist': signal_macd_hist,
            'signal_ema_fast': signal_ema_fast,
            'signal_ema_slow': signal_ema_slow,
            'signal_price': signal_price,
            'signal_time': signal_time,
            'signal_bar': signal_bar,
            'timestamp': datetime.now(timezone.utc)
        }

        self.trade_history.append(trade_record)

        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def validate_trade_log(self, trades_df=None):
        """
        Validate trade log for direction consistency.
        """
        if trades_df is None and hasattr(self, 'trade_history'):
            trades_df = pd.DataFrame(self.trade_history)

        if trades_df is None or len(trades_df) == 0:
            return True

        if self.trade_direction == 'long':
            if 'size' in trades_df.columns:
                negative_sizes = trades_df[trades_df['size'] < 0]
                if len(negative_sizes) > 0:
                    self._log(f"⚠️ Found {len(negative_sizes)} trades with negative size in LONG-ONLY mode!",
                              "bold red")
                    return False

        elif self.trade_direction == 'short':
            if 'size' in trades_df.columns:
                positive_sizes = trades_df[trades_df['size'] > 0]
                if len(positive_sizes) > 0:
                    self._log(f"⚠️ Found {len(positive_sizes)} trades with positive size in SHORT-ONLY mode!",
                              "bold red")
                    return False

        self._log(f"✅ Trade log validated for {self.trade_direction.upper()} mode", "green")
        return True

    def get_performance_stats(self):
        if self.total_trades == 0:
            return {'total_trades': 0, 'win_rate': 0, 'total_profit': 0,
                    'avg_win': 0, 'avg_loss': 0, 'tier1_trades': 0, 'tier2_trades': 0}
        wr = (self.winning_trades / self.total_trades) * 100
        wins = [t['profit'] for t in self.trade_history if t['profit'] > 0]
        losses = [t['profit'] for t in self.trade_history if t['profit'] < 0]
        return {
            'total_trades': self.total_trades, 'win_rate': wr,
            'total_profit': self.total_profit,
            'avg_profit': self.total_profit / self.total_trades,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': abs(np.mean(losses)) if losses else 0,
            'winning_trades': self.winning_trades, 'losing_trades': self.losing_trades,
            'tier1_trades': self.tier1_trades, 'tier2_trades': self.tier2_trades,
        }

    def clear_pending_signals(self):
        """Clear any pending signals when direction changes"""
        self._suggested_action = None
        self._last_direction_check['bar'] = -999
        self._log(f"🧹 Cleared pending signals due to direction change", "purple")

    def update_params(self, new_params, is_custom_mode=False):
        """Update parameters dynamically"""
        if not new_params:
            return

        self.config.update(new_params)
        for key, value in new_params.items():
            setattr(self, key, value)

        self.exit_manager = ProfessionalExitManager(self.config)

        mode_str = "CUSTOM" if is_custom_mode else "DEFAULT"
        self._log(f"📊 PARAMETER MODE: {mode_str}", "bold yellow" if is_custom_mode else "blue")
        self._log(f"✅ Parameters updated from settings panel ({mode_str} mode)", "green")

    def get_active_params(self):
        """Get the currently active parameters"""
        if self.current_mode == "Custom Parameters" and self.custom_params:
            active = self.config.copy()
            active.update(self.custom_params)
            return active
        else:
            return self.config.copy()

    def get_trade_records_for_export(self):
        """Generate accurate trade records for Excel export"""
        trades = []

        for i, trade in enumerate(self.trade_history):
            actual_tier = trade.get('tier', 0)

            # Signal data
            signal_cci = trade.get('signal_cci', 0)
            signal_macd_hist = trade.get('signal_macd_hist', 0)
            signal_ema_fast = trade.get('signal_ema_fast', 0)
            signal_ema_slow = trade.get('signal_ema_slow', 0)
            signal_price = trade.get('signal_price', 0)
            signal_bar = trade.get('signal_bar', trade.get('entry_bar', 0) - 1)
            signal_time = trade.get('signal_time', '')

            # Execution data
            entry_price = trade.get('entry_price', 0)
            entry_bar = trade.get('entry_bar', 0)
            entry_time = trade.get('entry_time', '')

            # Calculate returns
            if trade.get('direction') == 'long':
                return_pct = ((trade.get('exit_price', 0) - entry_price) /
                              entry_price) * 100 if entry_price > 0 else 0
            else:
                return_pct = ((entry_price - trade.get('exit_price', 0)) /
                              entry_price) * 100 if entry_price > 0 else 0

            trade_record = {
                'Trade_#': i + 1,
                'Tier': f"Tier {actual_tier}",
                'Tier_Number': actual_tier,
                'Signal_Bar': signal_bar,
                'Signal_Time': signal_time,
                'Signal_Price': signal_price,
                'Signal_CCI': signal_cci,
                'Signal_MACD_Hist': signal_macd_hist,
                'Signal_EMA_Fast': signal_ema_fast,
                'Signal_EMA_Slow': signal_ema_slow,
                'Entry_Bar': entry_bar,
                'Entry_Time': entry_time,
                'Entry_Price': entry_price,
                'Exit_Time': trade.get('exit_time', ''),
                'Exit_Bar': trade.get('exit_bar', 0),
                'Exit_Price': trade.get('exit_price', 0),
                'Exit_Reason': trade.get('exit_reason', ''),
                'Size': abs(trade.get('size', 0)),
                'PnL': trade.get('profit', 0),
                'Return_%': round(return_pct, 2),
                'Duration': trade.get('hold_duration', 0),
                'Win': 'Yes' if trade.get('profit', 0) > 0 else 'No',
            }
            trades.append(trade_record)

        return trades


# ═══════════════════════════════════════════════════════════════════════════
# PART 9: LIVE TRADING STRATEGY — EMA + CCI + MACD Strategy
# ═══════════════════════════════════════════════════════════════════════════

class MomentumStrategy(BaseStrategy, MomentumLogic):
    """EMA + CCI + MACD Histogram Strategy - Live Trading"""

    def __init__(self, trading_app=None):
        # Call BaseStrategy.__init__ first
        BaseStrategy.__init__(self, trading_app)

        # Get momentum_params from trading_app if available
        momentum_params = None
        if trading_app and hasattr(trading_app, 'get_current_momentum_params'):
            momentum_params = trading_app.get_current_momentum_params()

        # Load config with proper hierarchy
        params = MomentumConfig.get_config(momentum_params)

        # Now call MomentumLogic.__init__
        MomentumLogic.__init__(self, config=params, trading_app=trading_app)

        self.name = "EMA + CCI + MACD Histogram Strategy v1.0"

        # Log startup configuration
        logging.info("=" * 70)
        logging.info("EMA + CCI + MACD HISTOGRAM STRATEGY STARTUP")
        logging.info(f"📁 Config source: MOMENTUM_PARAMS (code)")
        logging.info(f"📁 Custom params file: {MomentumConfig.CONFIG_FILE}")
        logging.info(f"🔄 momentum_params override: {'YES' if momentum_params else 'NO'}")
        logging.info(
            f"📊 EMA Periods: Fast={self.ema_fast_period}, Mid={self.ema_mid_period}, Slow={self.ema_slow_period}")
        logging.info(f"📊 CCI Levels: Oversold={self.cci_oversold}, Overbought={self.cci_overbought}")
        logging.info(f"📊 Direction: {self.trade_direction.upper()}")
        logging.info("=" * 70)

        self.position = {
            'type': None, 'entry_price': None, 'quantity': None,
            'stop_loss': None, 'trailing_stop': None,
            'trailing_activated': False,
            'highest_price': None,
            'lowest_price': None,
            'entry_bar': None, 'partial_exits': 0, 'original_quantity': None,
            'tier': None, 'entry_time': None, 'entry_quality_score': None,
            'entry_reason': None, 'trade_id': None,
            'partial_pnl_realised': 0.0,
            'signal_cci': None, 'signal_macd_hist': None,
            'signal_ema_fast': None, 'signal_ema_slow': None,
            'signal_price': None, 'signal_time': None, 'signal_bar': None,
        }
        self.bars_held = 0
        self.last_signal = None
        self.signal_history = []
        self._account_quantity = 0
        self.trade_counter = 0

        if self.trading_app:
            self._log("=" * 70, "cyan")
            self._log("EMA + CCI + MACD HISTOGRAM STRATEGY v1.0", "bold green")
            self._log(
                f"✅ EMA Periods: Fast={self.ema_fast_period}, Mid={self.ema_mid_period}, Slow={self.ema_slow_period}",
                "green")
            self._log(f"✅ CCI Levels: Oversold={self.cci_oversold}, Overbought={self.cci_overbought}", "green")
            self._log(f"✅ Direction: {self.trade_direction.upper()}", "bold blue")
            self._log("✅ Entry: EMA support + CCI cross up + MACD hist > 0 (BUY)", "green")
            self._log("✅ Entry: EMA resistance + CCI cross down + MACD hist < 0 (SELL)", "green")
            self._log("✅ Exit: CCI extreme OR MACD hist reversal OR EMA 250 break", "yellow")
            self._log("=" * 70, "cyan")

    def run_analysis_cycle(self, current_data, current_price, df=None):
        self._current_df = df
        if self.strategy_state == StrategyState.SEEKING_ENTRY:
            return self.check_entry_conditions(current_data)
        return self.check_exit_conditions(current_data, current_price)

    def check_entry_conditions(self, current_data):
        if self.strategy_state != StrategyState.SEEKING_ENTRY:
            return "hold", POSITION_ALREADY_OPEN_SENTINEL, 0, "state_is_IN_TRADE"
        return self._check_entry_conditions(current_data)

    def check_exit_conditions(self, current_data, current_price):
        """Live trading exit logic"""
        if self.strategy_state != StrategyState.IN_TRADE:
            return None, 1.0

        # Update highest/lowest price based on position direction
        if self.position['type'] == 'long':
            if self.position['highest_price'] is None:
                self.position['highest_price'] = current_price
            else:
                self.position['highest_price'] = max(self.position['highest_price'], current_price)
        else:  # short position
            if self.position['lowest_price'] is None:
                self.position['lowest_price'] = current_price
            else:
                self.position['lowest_price'] = min(self.position['lowest_price'], current_price)

        # Get ATR
        atr = current_data.get('ATR')
        if not atr or atr <= 0:
            raise ValueError(f"Invalid ATR at bar {self.bars_held}: {atr}")

        return self.exit_manager.evaluate_exit(
            current_price=current_price,
            entry_price=self.position['entry_price'],
            stop_loss=self.position['stop_loss'],
            highest_price=self.position.get('highest_price', current_price),
            lowest_price=self.position.get('lowest_price', current_price),
            bars_held=self.bars_held,
            partial_exits=self.position.get('partial_exits', 0),
            ema_fast=current_data.get('EMA_Fast', 0),
            ema_mid=current_data.get('EMA_Mid', 0),
            ema_slow=current_data.get('EMA_Slow', 0),
            macd=current_data.get('MACD', 0),
            macd_signal=current_data.get('MACD_Signal', 0),
            macd_prev=current_data.get('MACD_closed', 0),
            signal_prev=current_data.get('MACD_Signal_closed', 0),
            adx=current_data.get('ADX', 0),
            atr=atr,
            position_type=self.position['type'],
            cci_curr=current_data.get('CCI', 0),
            macd_hist=current_data.get('MACD_Histogram', 0),
            trailing_activated=self.position.get('trailing_activated', False),
            trailing_stop=self.position.get('trailing_stop')
        )

    def sync_position_with_account(self):
        if not self.trading_app:
            return
        try:
            if hasattr(self.trading_app, 'get_account_balance'):
                bal = self.trading_app.get_account_balance()
                actual_qty = float(bal.get('quantity', 0))
            elif hasattr(self.trading_app, 'account_balance'):
                actual_qty = float(getattr(self.trading_app.account_balance, 'quantity', 0))
            else:
                return
            self._account_quantity = actual_qty
            if self.position['type'] is not None:
                tracked = self.position.get('quantity', 0)
                if abs(actual_qty - tracked) > 0.001:
                    self._log(f"Position mismatch — correcting", "orange")
                    self.position['quantity'] = actual_qty
            return actual_qty
        except Exception as e:
            self._log(f"Error syncing position: {e}", "red")
            return None

    def execute_buy(self, shares, price, atr, quality_score, tier):
        """Execute buy order"""
        try:
            if self.trade_direction == 'short':
                self._log(f"❌ BLOCKED: Attempted LONG trade in SHORT-ONLY mode", "bold red")
                return False, 0, None

            stop_loss_price = price - (atr * getattr(self, 'stop_loss_atr_mult', 2.5))

            if stop_loss_price >= price:
                self._log(f"REJECTED: stop {stop_loss_price:.4f} >= entry {price:.4f}", "red")
                return False, 0, None

            if self.trading_app and hasattr(self.trading_app, 'place_order'):
                order_result = self.trading_app.place_order(
                    side='buy', quantity=shares, price=price,
                    confidence=quality_score, atr=atr)
                if isinstance(order_result, dict):
                    if not order_result.get('success', False):
                        return False, 0, None
                    filled_qty = order_result.get('filled_quantity', shares)
                    actual_price = order_result.get('filled_price', price)
                else:
                    filled_qty, actual_price = (shares, price) if order_result else (0, price)
                    if not order_result:
                        return False, 0, None
            else:
                filled_qty, actual_price = shares, price

            self.trade_counter = getattr(self, 'trade_counter', 0) + 1

            # Copy signal data from pending signal if available
            signal_data = {}
            if hasattr(self, '_pending_signal') and self._pending_signal:
                signal_data = {
                    'signal_cci': self._pending_signal.get('signal_cci', 0),
                    'signal_macd_hist': self._pending_signal.get('signal_macd_hist', 0),
                    'signal_ema_fast': self._pending_signal.get('signal_ema_fast', 0),
                    'signal_ema_slow': self._pending_signal.get('signal_ema_slow', 0),
                    'signal_price': self._pending_signal.get('signal_price', 0),
                    'signal_time': self._pending_signal.get('signal_time', datetime.now(timezone.utc)),
                    'signal_bar': self._pending_signal.get('signal_bar', getattr(self, 'bar_count', 0) - 1),
                }

            self.position = {
                'type': 'long',
                'entry_price': actual_price,
                'quantity': filled_qty,
                'original_quantity': filled_qty,
                'stop_loss': stop_loss_price,
                'trailing_stop': None,
                'trailing_activated': False,
                'highest_price': actual_price,
                'lowest_price': None,
                'entry_bar': getattr(self, 'bar_count', 0),
                'partial_exits': 0,
                'tier': tier,
                'entry_time': datetime.now(timezone.utc),
                'entry_quality_score': quality_score,
                'entry_reason': '',
                'trade_id': self.trade_counter,
                'partial_pnl_realised': 0.0,
                **signal_data,
            }
            self.bars_held = 0
            self._transition_to_in_trade()

            self._log(f"POSITION #{self.trade_counter} OPENED: {filled_qty} @ ${actual_price:.4f} "
                      f"Tier {tier} Q={quality_score} Dir={self.trade_direction.upper()} → IN_TRADE", "bold green")
            self._log(
                f"  ├─ Stop Loss: ${stop_loss_price:.4f} ({(actual_price - stop_loss_price) / actual_price * 100:.1f}%)",
                "yellow")
            if signal_data:
                self._log(f"  └─ Signal Data: CCI={signal_data.get('signal_cci', 0):.1f}, "
                          f"MACD Hist={signal_data.get('signal_macd_hist', 0):.5f}", "magenta")

            return True, filled_qty, None
        except Exception as e:
            self._log(f"ERROR execute_buy: {e}", "bold red")
            return False, 0, None

    def execute_sell(self, reason="manual", exit_percentage=1.0):
        """Execute sell order"""
        if self.position['type'] is None:
            return False, 0, 0
        try:
            self.sync_position_with_account()
            current_qty = self.position.get('quantity', 0)
            if current_qty <= 0:
                self.position['type'] = None
                self._transition_to_seeking_entry()
                return False, 0, 0

            if self.trading_app and hasattr(self.trading_app, 'get_current_price'):
                current_price = self.trading_app.get_current_price()
            else:
                current_price = self.position['entry_price'] * 1.01

            exit_qty = current_qty * exit_percentage
            exit_qty = min(exit_qty, current_qty)

            if self.trading_app and hasattr(self.trading_app, 'place_order'):
                res = self.trading_app.place_order(side='sell', quantity=exit_qty, price=current_price)
                filled_qty = res.get('filled_quantity', exit_qty) if isinstance(res, dict) else exit_qty
                exit_price = res.get('filled_price', current_price) if isinstance(res, dict) else current_price
            else:
                filled_qty, exit_price = exit_qty, current_price

            leg_profit = (exit_price - self.position['entry_price']) * filled_qty
            profit_pct = (exit_price - self.position['entry_price']) / self.position['entry_price'] * 100
            stop_dist = self.position['stop_loss'] - self.position['entry_price']
            profit_r = (exit_price - self.position['entry_price']) / abs(stop_dist) if stop_dist != 0 else 0

            if exit_percentage >= 0.99:
                total_pnl = leg_profit + self.position.get('partial_pnl_realised', 0.0)
                self.record_trade(
                    profit=total_pnl,
                    exit_reason=reason,
                    tier=self.position.get('tier'),
                    size=current_qty,
                    direction=self.trade_direction,
                    entry_quality=self.position.get('entry_quality_score'),
                    entry_price=self.position['entry_price'],
                    exit_price=exit_price,
                    hold_duration=(datetime.now(timezone.utc) - self.position['entry_time']).total_seconds() / 60,
                    entry_bar=self.position.get('entry_bar'),
                    exit_bar=getattr(self, 'bar_count', 0),
                    signal_cci=self.position.get('signal_cci'),
                    signal_macd_hist=self.position.get('signal_macd_hist'),
                    signal_ema_fast=self.position.get('signal_ema_fast'),
                    signal_ema_slow=self.position.get('signal_ema_slow'),
                    signal_price=self.position.get('signal_price'),
                    signal_time=self.position.get('signal_time'),
                    signal_bar=self.position.get('signal_bar')
                )

                trade_rec = TradeRecord(
                    trade_id=self.position['trade_id'], symbol="SOL/USD",
                    entry_time=self.position['entry_time'],
                    entry_price=self.position['entry_price'],
                    entry_size=self.position['original_quantity'],
                    entry_tier=self.position.get('tier'),
                    entry_quality_score=self.position['entry_quality_score'],
                    entry_reason=self.position.get('entry_reason', ''),
                    entry_direction=self.trade_direction,
                    exit_time=datetime.now(timezone.utc),
                    exit_price=exit_price, exit_size=filled_qty, exit_reason=reason,
                    profit=total_pnl, profit_pct=profit_pct, profit_r=profit_r,
                    hold_duration=(datetime.now(timezone.utc) -
                                   self.position['entry_time']).total_seconds() / 60,
                    market_regime=self.current_regime,
                    partial_exits_taken=self.position.get('partial_exits', 0),
                    partial_pnl_realised=self.position.get('partial_pnl_realised', 0.0),
                    original_size=self.position['original_quantity'],
                )
                self.risk_controller.record_trade(trade_rec)

                self.position = {
                    'type': None,
                    'entry_price': None,
                    'quantity': None,
                    'stop_loss': None,
                    'trailing_stop': None,
                    'trailing_activated': False,
                    'highest_price': None,
                    'lowest_price': None,
                    'entry_bar': None,
                    'partial_exits': 0,
                    'original_quantity': None,
                    'tier': None,
                    'entry_time': None,
                    'entry_quality_score': None,
                    'entry_reason': None,
                    'trade_id': None,
                    'partial_pnl_realised': 0.0,
                    'signal_cci': None, 'signal_macd_hist': None,
                    'signal_ema_fast': None, 'signal_ema_slow': None,
                    'signal_price': None, 'signal_time': None, 'signal_bar': None,
                }
                self._account_quantity = 0
                self.bars_held = 0
                self._transition_to_seeking_entry()
                self._log(
                    f"POSITION CLOSED → SEEKING_ENTRY | PnL: ${total_pnl:.2f} ({profit_pct:.2f}%) | Reason: {reason}",
                    "green" if total_pnl > 0 else "red")
            else:
                self.position['quantity'] -= filled_qty
                self.position['partial_exits'] = self.position.get('partial_exits', 0) + 1
                self.position['partial_pnl_realised'] = (
                        self.position.get('partial_pnl_realised', 0.0) + leg_profit)
                self._log(
                    f"Partial exit #{self.position['partial_exits']}: "
                    f"{filled_qty:.4f} @ ${exit_price:.4f} "
                    f"leg=${leg_profit:.2f} cumulative_partial=${self.position['partial_pnl_realised']:.2f} "
                    f"remaining={self.position['quantity']:.4f} — state stays IN_TRADE",
                    "blue")
            return True, leg_profit, exit_price
        except Exception as e:
            self._log(f"ERROR execute_sell: {e}", "bold red")
            return False, 0, 0

    def execute_short(self, shares, price, atr, quality_score, tier):
        """Execute short order"""
        try:
            if self.trade_direction == 'long':
                self._log(f"❌ BLOCKED: Attempted SHORT trade in LONG-ONLY mode", "bold red")
                return False, 0, None

            stop_loss_price = price + (atr * getattr(self, 'stop_loss_atr_mult', 2.5))

            if stop_loss_price <= price:
                self._log(f"REJECTED: stop {stop_loss_price:.4f} <= entry {price:.4f}", "red")
                return False, 0, None

            if self.trading_app and hasattr(self.trading_app, 'place_order'):
                order_result = self.trading_app.place_order(
                    side='sell', quantity=shares, price=price,
                    confidence=quality_score, atr=atr)
                if isinstance(order_result, dict):
                    if not order_result.get('success', False):
                        return False, 0, None
                    filled_qty = order_result.get('filled_quantity', shares)
                    actual_price = order_result.get('filled_price', price)
                else:
                    filled_qty, actual_price = (shares, price) if order_result else (0, price)
                    if not order_result:
                        return False, 0, None
            else:
                filled_qty, actual_price = shares, price

            self.trade_counter = getattr(self, 'trade_counter', 0) + 1

            signal_data = {}
            if hasattr(self, '_pending_signal') and self._pending_signal:
                signal_data = {
                    'signal_cci': self._pending_signal.get('signal_cci', 0),
                    'signal_macd_hist': self._pending_signal.get('signal_macd_hist', 0),
                    'signal_ema_fast': self._pending_signal.get('signal_ema_fast', 0),
                    'signal_ema_slow': self._pending_signal.get('signal_ema_slow', 0),
                    'signal_price': self._pending_signal.get('signal_price', 0),
                    'signal_time': self._pending_signal.get('signal_time', datetime.now(timezone.utc)),
                    'signal_bar': self._pending_signal.get('signal_bar', getattr(self, 'bar_count', 0) - 1),
                }

            self.position = {
                'type': 'short',
                'entry_price': actual_price,
                'quantity': filled_qty,
                'original_quantity': filled_qty,
                'stop_loss': stop_loss_price,
                'trailing_stop': None,
                'trailing_activated': False,
                'highest_price': None,
                'lowest_price': actual_price,
                'entry_bar': getattr(self, 'bar_count', 0),
                'partial_exits': 0,
                'tier': tier,
                'entry_time': datetime.now(timezone.utc),
                'entry_quality_score': quality_score,
                'entry_reason': '',
                'trade_id': self.trade_counter,
                'partial_pnl_realised': 0.0,
                **signal_data,
            }
            self.bars_held = 0
            self._transition_to_in_trade()

            self._log(f"POSITION #{self.trade_counter} OPENED: {filled_qty} @ ${actual_price:.4f} "
                      f"Tier {tier} Q={quality_score} Dir={self.trade_direction.upper()} → IN_TRADE", "bold green")
            self._log(
                f"  ├─ Stop Loss: ${stop_loss_price:.4f} ({(stop_loss_price - actual_price) / actual_price * 100:.1f}%)",
                "yellow")
            if signal_data:
                self._log(f"  └─ Signal Data: CCI={signal_data.get('signal_cci', 0):.1f}, "
                          f"MACD Hist={signal_data.get('signal_macd_hist', 0):.5f}", "magenta")

            return True, filled_qty, None
        except Exception as e:
            self._log(f"ERROR execute_short: {e}", "bold red")
            return False, 0, None

    def execute_cover(self, reason="manual", exit_percentage=1.0):
        """Execute cover order for short positions"""
        if self.position['type'] is None:
            return False, 0, 0
        try:
            self.sync_position_with_account()
            current_qty = self.position.get('quantity', 0)
            if current_qty <= 0:
                self.position['type'] = None
                self._transition_to_seeking_entry()
                return False, 0, 0

            if self.trading_app and hasattr(self.trading_app, 'get_current_price'):
                current_price = self.trading_app.get_current_price()
            else:
                current_price = self.position['entry_price'] * 0.99

            exit_qty = current_qty * exit_percentage
            exit_qty = min(exit_qty, current_qty)

            if self.trading_app and hasattr(self.trading_app, 'place_order'):
                res = self.trading_app.place_order(side='buy', quantity=exit_qty, price=current_price)
                filled_qty = res.get('filled_quantity', exit_qty) if isinstance(res, dict) else exit_qty
                exit_price = res.get('filled_price', current_price) if isinstance(res, dict) else current_price
            else:
                filled_qty, exit_price = exit_qty, current_price

            leg_profit = (self.position['entry_price'] - exit_price) * filled_qty
            profit_pct = (self.position['entry_price'] - exit_price) / self.position['entry_price'] * 100
            stop_dist = self.position['stop_loss'] - self.position['entry_price']
            profit_r = (self.position['entry_price'] - exit_price) / abs(stop_dist) if stop_dist != 0 else 0

            if exit_percentage >= 0.99:
                total_pnl = leg_profit + self.position.get('partial_pnl_realised', 0.0)
                self.record_trade(
                    profit=total_pnl,
                    exit_reason=reason,
                    tier=self.position.get('tier'),
                    size=current_qty,
                    direction=self.trade_direction,
                    entry_quality=self.position.get('entry_quality_score'),
                    entry_price=self.position['entry_price'],
                    exit_price=exit_price,
                    hold_duration=(datetime.now(timezone.utc) - self.position['entry_time']).total_seconds() / 60,
                    entry_bar=self.position.get('entry_bar'),
                    exit_bar=getattr(self, 'bar_count', 0),
                    signal_cci=self.position.get('signal_cci'),
                    signal_macd_hist=self.position.get('signal_macd_hist'),
                    signal_ema_fast=self.position.get('signal_ema_fast'),
                    signal_ema_slow=self.position.get('signal_ema_slow'),
                    signal_price=self.position.get('signal_price'),
                    signal_time=self.position.get('signal_time'),
                    signal_bar=self.position.get('signal_bar')
                )

                trade_rec = TradeRecord(
                    trade_id=self.position['trade_id'], symbol="SOL/USD",
                    entry_time=self.position['entry_time'],
                    entry_price=self.position['entry_price'],
                    entry_size=self.position['original_quantity'],
                    entry_tier=self.position.get('tier'),
                    entry_quality_score=self.position['entry_quality_score'],
                    entry_reason=self.position.get('entry_reason', ''),
                    entry_direction=self.trade_direction,
                    exit_time=datetime.now(timezone.utc),
                    exit_price=exit_price, exit_size=filled_qty, exit_reason=reason,
                    profit=total_pnl, profit_pct=profit_pct, profit_r=profit_r,
                    hold_duration=(datetime.now(timezone.utc) -
                                   self.position['entry_time']).total_seconds() / 60,
                    market_regime=self.current_regime,
                    partial_exits_taken=self.position.get('partial_exits', 0),
                    partial_pnl_realised=self.position.get('partial_pnl_realised', 0.0),
                    original_size=self.position['original_quantity'],
                )
                self.risk_controller.record_trade(trade_rec)

                self.position = {
                    'type': None,
                    'entry_price': None,
                    'quantity': None,
                    'stop_loss': None,
                    'trailing_stop': None,
                    'trailing_activated': False,
                    'highest_price': None,
                    'lowest_price': None,
                    'entry_bar': None,
                    'partial_exits': 0,
                    'original_quantity': None,
                    'tier': None,
                    'entry_time': None,
                    'entry_quality_score': None,
                    'entry_reason': None,
                    'trade_id': None,
                    'partial_pnl_realised': 0.0,
                    'signal_cci': None, 'signal_macd_hist': None,
                    'signal_ema_fast': None, 'signal_ema_slow': None,
                    'signal_price': None, 'signal_time': None, 'signal_bar': None,
                }
                self._account_quantity = 0
                self.bars_held = 0
                self._transition_to_seeking_entry()
                self._log(
                    f"POSITION CLOSED → SEEKING_ENTRY | PnL: ${total_pnl:.2f} ({profit_pct:.2f}%) | Reason: {reason}",
                    "green" if total_pnl > 0 else "red")
            else:
                self.position['quantity'] -= filled_qty
                self.position['partial_exits'] = self.position.get('partial_exits', 0) + 1
                self.position['partial_pnl_realised'] = (
                        self.position.get('partial_pnl_realised', 0.0) + leg_profit)
                self._log(
                    f"Partial exit #{self.position['partial_exits']}: "
                    f"{filled_qty:.4f} @ ${exit_price:.4f} "
                    f"leg=${leg_profit:.2f} cumulative_partial=${self.position['partial_pnl_realised']:.2f} "
                    f"remaining={self.position['quantity']:.4f} — state stays IN_TRADE",
                    "blue")
            return True, leg_profit, exit_price
        except Exception as e:
            self._log(f"ERROR execute_cover: {e}", "bold red")
            return False, 0, 0

    def calculate_indicators(self, df):
        return IndicatorCalculator.calculate(df, self.config)

    def on_bar_update(self, current_equity):
        self.equity_curve.append(current_equity)
        self.bar_count += 1
        if self.strategy_state == StrategyState.IN_TRADE:
            self.bars_held += 1
        self.risk_controller.current_equity = current_equity
        if current_equity > self.risk_controller.peak_equity:
            self.risk_controller.peak_equity = current_equity

    def get_strategy_stats(self):
        stats = self.get_performance_stats()
        risk_stats = self.risk_controller.get_stats()
        return {
            'strategy_name': self.name,
            'total_trades': stats['total_trades'],
            'win_rate': stats['win_rate'],
            'total_profit': stats['total_profit'],
            'tier1_trades': stats['tier1_trades'],
            'tier2_trades': stats['tier2_trades'],
            'profit_factor': risk_stats['profit_factor'],
            'max_drawdown': risk_stats['max_drawdown'],
            'sharpe_ratio': risk_stats['sharpe_ratio'],
            'current_regime': self.current_regime,
            'strategy_state': self.strategy_state.name,
            'trade_direction': self.trade_direction,
        }

    def _build_current_data(self):
        """Build current data dict for validation"""
        if hasattr(self, '_current_df') and self._current_df is not None and len(self._current_df) > 0:
            last_row = self._current_df.iloc[-1]
            return {
                'Close': last_row.get('Close', 0),
                'EMA_Fast': last_row.get('EMA_Fast', 0),
                'EMA_Slow': last_row.get('EMA_Slow', 0),
                'EMA_Mid': last_row.get('EMA_Mid', 0),
                'CCI': last_row.get('CCI', 0),
                'MACD': last_row.get('MACD', 0),
                'MACD_Signal': last_row.get('MACD_Signal', 0),
                'MACD_Histogram': last_row.get('MACD_Histogram', 0),
                'ADX': last_row.get('ADX', 0),
                'ATR': last_row.get('ATR', 0),
            }
        return {}

# ═══════════════════════════════════════════════════════════════════════════
# PART 10: BACKTEST STRATEGY — EMA + CCI + MACD Strategy
# ═══════════════════════════════════════════════════════════════════════════

class BacktestMomentumStrategy(Strategy, MomentumLogic):
    """EMA + CCI + MACD Histogram Strategy - Backtest"""

    # ═══ FORCE PARAMETER DEFINITION FOR BACKTESTING FRAMEWORK ═══
    # EMA Settings
    ema_fast_period = 50
    ema_mid_period = 110
    ema_slow_period = 250

    # CCI Settings
    cci_period = 20
    cci_oversold = -100
    cci_overbought = 100

    # EMA Proximity
    ema_proximity_pct = 0.003

    # Exit Control Flags
    cci_exit_enabled = True
    macd_exit_enabled = True
    ema_slow_exit_enabled = True

    # MACD Settings
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # ADX Settings
    adx_min = 20
    adx_period = 14
    adx_filter_enabled = False

    # Risk Management
    risk_per_trade = 0.012
    risk_full_position = 0.012
    risk_reduced_position = 0.008
    risk_aggressive_position = 0.015
    stop_loss_pct = 0.02

    # Stop Loss & Trailing
    stop_loss_atr_mult = 2.5
    atr_period = 14
    trailing_activation_pct = 0.015
    trailing_distance_pct = 0.01
    trailing_stop_enabled = False

    # Trade Management
    max_daily_trades = 8
    min_bars_between_trades = 2
    cooldown_after_loss_bars = 3
    max_hold_bars = 120
    min_hold_bars_before_stop = 6
    emergency_stop_multiplier = 2.0

    # Direction Control
    trade_direction = "both"

    # Quality scoring (simplified for this strategy)
    quality_score_enabled = False
    quality_tier1_min = 60
    quality_tier2_min = 75

    # Tier control
    only_tier2_entries = False
    backtest_only_tier2_active = True
    backtest_only_tier2_values = [True, False]

    # RSI (optional)
    rsi_period = 14
    rsi_filter_enabled = False

    # Volume (optional)
    volume_period = 20
    volume_filter_enabled = False

    # Regime Detection
    regime_filter_enabled = False

    # ═══ Class variables to receive updated parameters from app ═══════
    _use_updated_params = False
    _updated_params = {}

    @classmethod
    def set_updated_params(cls, params):
        """Call this from your app to inject updated parameters into backtest"""
        if params and isinstance(params, dict):
            cls._use_updated_params = True
            cls._updated_params = params.copy()
            logging.info(f"🔴 BACKTEST USING UPDATED PARAMETERS: {len(params)} params")
        else:
            cls._use_updated_params = False
            cls._updated_params = {}
            logging.info("🔵 BACKTEST USING DEFAULT MOMENTUM_PARAMS")

    @classmethod
    def reset_to_defaults(cls):
        """Reset backtest to use MOMENTUM_PARAMS defaults"""
        cls._use_updated_params = False
        cls._updated_params = {}
        logging.info("🔵 BACKTEST RESET TO MOMENTUM_PARAMS DEFAULTS")

    # ═══ OPTIMIZATION RANGES ════════════════════════════════════
    backtest_ema_fast_active = True
    backtest_ema_fast_values = [40, 50, 60]

    backtest_ema_mid_active = True
    backtest_ema_mid_values = [100, 110, 120]

    backtest_ema_slow_active = True
    backtest_ema_slow_values = [200, 250, 300]

    backtest_cci_period_active = True
    backtest_cci_period_values = [14, 20, 30]

    backtest_ema_proximity_active = True
    backtest_ema_proximity_values = [0.002, 0.003, 0.005]

    backtest_stop_loss_mult_active = True
    backtest_stop_loss_mult_values = [2.0, 2.5, 3.0]

    def __init__(self, broker, data, params):
        Strategy.__init__(self, broker, data, params)

        # STEP 1: LOAD BASE CONFIG VIA MomentumConfig
        if self.__class__._use_updated_params and self.__class__._updated_params:
            config = MomentumConfig.get_config(
                momentum_params_override=self.__class__._updated_params
            )
            source = "MomentumConfig + Runtime Overrides"
            logging.info("🔴 BACKTEST: Using updated params from app")
        else:
            config = MomentumConfig.get_config()
            source = "MomentumConfig (defaults + mode-based params)"
            logging.info("🔵 BACKTEST: Using MomentumConfig defaults")

        # STEP 2: APPLY OPTIMIZATION PARAMETERS
        optimization_count = 0
        if params:
            logging.info("")
            logging.info("Applying optimization parameters:")
            for key, value in params.items():
                if key in config:
                    old_value = config[key]
                    config[key] = value
                    if old_value != value:
                        logging.info(f"  ✓ {key}: {old_value} → {value}")
                        optimization_count += 1
                else:
                    logging.warning(f"  ⚠️ Unknown parameter: {key}")

        # STEP 3: SET AS INSTANCE ATTRIBUTES
        for key, value in config.items():
            setattr(self, key, value)

        # Ensure critical attributes exist
        critical_defaults = {
            'trailing_activation_pct': 0.015,
            'trailing_distance_pct': 0.01,
            'trade_direction': 'both',
            'stop_loss_atr_mult': 2.5,
            'only_tier2_entries': False,
            'quality_tier1_min': 60,
            'quality_tier2_min': 75,
            'tier1_adx_hard_min': 18,
            'tier1_volume_min': 0.8,
            'cci_oversold': -100,
            'cci_overbought': 100,
            'ema_fast_period': 50,
            'ema_mid_period': 110,
            'ema_slow_period': 250,
            'ema_proximity_pct': 0.003,
            'cci_exit_enabled': True,
            'macd_exit_enabled': True,
            'ema_slow_exit_enabled': True,
        }

        for attr, default_val in critical_defaults.items():
            if not hasattr(self, attr) or getattr(self, attr) is None:
                setattr(self, attr, default_val)
                logging.warning(f"⚠️ Added missing attribute: {attr} = {default_val}")

        # STEP 4: INITIALIZE MomentumLogic
        MomentumLogic.__init__(self, config=config, trading_app=None)

        # LOG FINAL CONFIGURATION
        print(f"\n{'=' * 70}")
        print(f"BACKTEST CONFIGURATION LOADED")
        print(f"{'=' * 70}")
        print(f"Source: {source}")
        print(f"Total params: {len(config)}")
        print(f"Optimization overrides: {optimization_count}")
        print(f"Mode: {MomentumConfig._current_mode}")
        print(f"Direction: {self.trade_direction.upper()}")
        print(f"{'=' * 70}\n")

        # Backtest-specific state
        self._entry_price = np.nan
        self._stop_loss = np.nan
        self._highest_price = np.nan
        self._lowest_price = np.nan
        self._bars_held = 0
        self._partial_exits = 0
        self._entry_bar = -999
        self._entry_tier = None
        self._entry_quality = 0
        self._params_dict = params
        self._partial_pnl_realised = 0.0
        self._exit_reason_map = {}
        self._position_direction = self.trade_direction

        # Pending signal tracking
        self._pending_signal = None
        self._signal_bar = -999
        self._signal_price = None

        # Signal data storage
        self._signal_cci = None
        self._signal_macd_hist = None
        self._signal_ema_fast = None
        self._signal_ema_slow = None
        self._signal_price = None
        self._signal_time = None
        self._signal_bar = None

    def init(self):
        self.df_indicators = IndicatorCalculator.calculate(self.data.df.copy(), self.config)
        self.df_enhanced = self.df_indicators.copy()

        self.ema_fast = self.I(lambda: self.df_indicators['EMA_Fast'].values, name='EMA_Fast')
        self.ema_mid = self.I(lambda: self.df_indicators['EMA_Mid'].values, name='EMA_Mid')
        self.ema_slow = self.I(lambda: self.df_indicators['EMA_Slow'].values, name='EMA_Slow')
        self.cci = self.I(lambda: self.df_indicators['CCI'].values, name='CCI')
        self.macd_line = self.I(lambda: self.df_indicators['MACD'].values, name='MACD')
        self.macd_sig = self.I(lambda: self.df_indicators['MACD_Signal'].values, name='MACD_Signal')
        self.macd_hist = self.I(lambda: self.df_indicators['MACD_Histogram'].values, name='MACD_Histogram')
        self.adx = self.I(lambda: self.df_indicators['ADX'].values, name='ADX')
        self.atr = self.I(lambda: self.df_indicators['ATR'].values, name='ATR')

    def _build_current_data(self):
        def safe(arr, default=0.0):
            try:
                v = arr[-1]
                return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            except Exception:
                return default

        return {
            'Close': self.data.Close[-1],
            'EMA_Fast': safe(self.ema_fast),
            'EMA_Mid': safe(self.ema_mid),
            'EMA_Slow': safe(self.ema_slow),
            'CCI': safe(self.cci),
            'CCI_closed': float(self.cci[-2]) if len(self.cci) > 1 and not np.isnan(self.cci[-2]) else 0.0,
            'MACD': safe(self.macd_line),
            'MACD_Signal': safe(self.macd_sig),
            'MACD_Histogram': safe(self.macd_hist),
            'MACD_Histogram_closed': float(self.macd_hist[-2]) if len(self.macd_hist) > 1 and not np.isnan(
                self.macd_hist[-2]) else 0.0,
            'ADX': safe(self.adx),
            'ATR': safe(self.atr, 1.0),
        }

    def _bt_open_position(self, current_data, quality_score, tier, position_mult, current_price):
        size = self.calculate_position_size(
            self.equity, current_data['ATR'], current_price,
            quality_score, tier, position_mult)
        if size <= 0:
            return

        stop = current_price - (self.stop_loss_atr_mult * current_data['ATR'])
        if stop >= current_price:
            print(f"REJECTED: stop {stop:.4f} >= price {current_price:.4f}")
            return

        self._position_direction = self.trade_direction

        if self._position_direction == 'long' or self._position_direction == 'both':
            self.buy(size=size)
        else:
            self.sell(size=size)

        self._entry_price = current_price
        self._stop_loss = stop
        self._highest_price = current_price
        self._lowest_price = current_price
        self._bars_held = 0
        self._partial_exits = 0
        self._entry_bar = len(self.data) - 1
        self._entry_tier = tier
        self._entry_quality = quality_score
        self._partial_pnl_realised = 0.0
        self._transition_to_in_trade()

        if self._pending_signal:
            self._signal_cci = self._pending_signal.get('signal_cci')
            self._signal_macd_hist = self._pending_signal.get('signal_macd_hist')
            self._signal_ema_fast = self._pending_signal.get('signal_ema_fast')
            self._signal_ema_slow = self._pending_signal.get('signal_ema_slow')
            self._signal_price = self._pending_signal.get('signal_price')
            self._signal_time = self._pending_signal.get('signal_time')
            self._signal_bar = self._pending_signal.get('signal_bar')

        direction_icon = "⬆️" if self._position_direction == 'long' else "⬇️"
        print(
            f"{direction_icon} {self._position_direction.upper()} T{tier} Q={quality_score} @ ${current_price:.2f} → IN_TRADE")

    def _bt_close_position(self, current_price, exit_signal, profit_pct):
        size_at_close = self.position.size
        self._exit_reason_map[self._entry_bar] = exit_signal
        self.position.close()

        abs_size = abs(size_at_close)

        if self._position_direction == 'long' or self._position_direction == 'both':
            final_leg = (current_price - self._entry_price) * abs_size
            profit_pct_calc = (current_price - self._entry_price) / self._entry_price * 100
        else:
            final_leg = (self._entry_price - current_price) * abs_size
            profit_pct_calc = (self._entry_price - current_price) / self._entry_price * 100

        total_profit = final_leg + self._partial_pnl_realised

        self.record_trade(
            profit=total_profit,
            exit_reason=exit_signal,
            tier=self._entry_tier,
            size=abs_size,
            direction=self._position_direction,
            entry_quality=self._entry_quality,
            entry_price=self._entry_price,
            exit_price=current_price,
            hold_duration=self._bars_held,
            entry_bar=self._entry_bar,
            exit_bar=len(self.data) - 1,
            signal_cci=self._signal_cci,
            signal_macd_hist=self._signal_macd_hist,
            signal_ema_fast=self._signal_ema_fast,
            signal_ema_slow=self._signal_ema_slow,
            signal_price=self._signal_price,
            signal_time=self._signal_time,
            signal_bar=self._signal_bar
        )

        direction_icon = "⬆️" if self._position_direction == 'long' else "⬇️"
        win_loss_icon = "✅" if total_profit > 0 else "❌"

        print(f"{win_loss_icon} {direction_icon} CLOSE @ ${current_price:.2f} {profit_pct_calc:+.2f}% → SEEKING_ENTRY")

        self._entry_price = np.nan
        self._stop_loss = np.nan
        self._highest_price = np.nan
        self._lowest_price = np.nan
        self._bars_held = 0
        self._partial_exits = 0
        self._partial_pnl_realised = 0.0
        self._entry_tier = None
        self._position_direction = self.trade_direction
        self._transition_to_seeking_entry()

    def next(self):
        try:
            if any(np.isnan(x[-1]) for x in [self.ema_fast, self.ema_slow, self.cci, self.macd_hist, self.atr]):
                return
        except Exception:
            return

        idx = len(self.data) - 1
        current_data = self._build_current_data()
        current_price = self.data.Close[-1]
        self._current_df = self.df_enhanced.iloc[:idx + 1]
        self.bar_count = idx

        # Check for pending signal
        if self._pending_signal is not None and self.bar_count > self._signal_bar:
            signal = self._pending_signal
            execution_price = self.data.Open[-1]

            size = self.calculate_position_size(
                self.equity, current_data['ATR'], execution_price,
                signal['quality_score'], signal['tier'], signal['position_mult']
            )

            if size > 0:
                if signal['decision'] == "buy":
                    stop = execution_price - (self.stop_loss_atr_mult * current_data['ATR'])
                    if stop < execution_price:
                        self._position_direction = 'long'
                        self.buy(size=size)
                    else:
                        self._pending_signal = None
                        return
                else:
                    stop = execution_price + (self.stop_loss_atr_mult * current_data['ATR'])
                    if stop > execution_price:
                        self._position_direction = 'short'
                        self.sell(size=size)
                    else:
                        self._pending_signal = None
                        return

                self._entry_price = execution_price
                self._stop_loss = stop
                self._highest_price = execution_price if self._position_direction == 'long' else None
                self._lowest_price = execution_price if self._position_direction == 'short' else None
                self._bars_held = 0
                self._partial_exits = 0
                self._entry_bar = idx
                self._entry_tier = signal['tier']
                self._entry_quality = signal['quality_score']
                self._partial_pnl_realised = 0.0
                self._signal_cci = signal.get('signal_cci')
                self._signal_macd_hist = signal.get('signal_macd_hist')
                self._signal_ema_fast = signal.get('signal_ema_fast')
                self._signal_ema_slow = signal.get('signal_ema_slow')
                self._signal_price = signal.get('signal_price')
                self._signal_time = signal.get('signal_time')
                self._signal_bar = signal.get('signal_bar')
                self._transition_to_in_trade()

                print(f"  ✅ FILLED: {size} shares @ ${execution_price:.2f}")
            else:
                print(f"  ❌ REJECTED: Position size {size} <= 0")

            self._pending_signal = None
            self._signal_bar = -999
            self._signal_price = None
            return

        if self.strategy_state == StrategyState.SEEKING_ENTRY:
            self._check_entry_conditions(current_data)
        else:
            self._bars_held += 1

            exit_signal, exit_pct = self.exit_manager.evaluate_exit(
                current_price=current_price,
                entry_price=self._entry_price,
                stop_loss=self._stop_loss,
                highest_price=self._highest_price,
                lowest_price=self._lowest_price,
                bars_held=self._bars_held,
                partial_exits=self._partial_exits,
                ema_fast=current_data['EMA_Fast'],
                ema_mid=current_data['EMA_Mid'],
                ema_slow=current_data['EMA_Slow'],
                macd=current_data['MACD'],
                macd_signal=current_data['MACD_Signal'],
                macd_prev=current_data['MACD_closed'],
                signal_prev=current_data['MACD_Signal_closed'],
                adx=current_data['ADX'],
                atr=current_data['ATR'],
                position_type=self._position_direction,
                cci_curr=current_data['CCI'],
                macd_hist=current_data['MACD_Histogram']
            )

            if exit_signal:
                if self._position_direction == 'long':
                    profit_pct = (current_price - self._entry_price) / self._entry_price * 100
                else:
                    profit_pct = (self._entry_price - current_price) / self._entry_price * 100

                self._bt_close_position(current_price, exit_signal, profit_pct)