# ═══════════════════════════════════════════════════════════════════════════
# v9.5.1
# ═══════════════════════════════════════════════════════════════════════════


import json
import logging
import os

os.environ['BACKTESTING_DISABLE_MULTIPROCESSING'] = '1'  # future-proof hint
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


# BUG FIX: 'CAPITAL = GlobalConfig.INITIAL_CAPITAL' captured the value at import
# time. Any subsequent call to GlobalConfig.update_capital() left CAPITAL stale.
# All code must use GlobalConfig.INITIAL_CAPITAL directly.
# This alias is kept for backward compatibility but marked deprecated.
def _get_capital():
    return GlobalConfig.INITIAL_CAPITAL


CAPITAL = GlobalConfig.INITIAL_CAPITAL  # deprecated — use GlobalConfig.INITIAL_CAPITAL


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR ROBUST METRICS
# ═══════════════════════════════════════════════════════════════════════════

def safe_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss <= 0:
        return float('inf')
    return gross_profit / gross_loss


def summarize_performance(trades: List[Any], initial_capital: float = None) -> Dict:
    if initial_capital is None:
        initial_capital = GlobalConfig.INITIAL_CAPITAL

    num_trades = len(trades)
    if num_trades == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_pct": 0.0,
            "profit_factor": None, "expectancy": 0.0, "avg_profit_per_trade": 0.0,
            "max_drawdown_pct": 0.0, "warning": "No trades to analyze"
        }

    total_profit = sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades)
    total_profit_pct = (total_profit / initial_capital) * 100

    wins = sum(1 for t in trades if getattr(t, 'profit', t.get('profit', 0)) > 0)
    losses = sum(1 for t in trades if getattr(t, 'profit', t.get('profit', 0)) <= 0)
    win_rate = (wins / num_trades) if num_trades > 0 else 0.0

    gross_profit = sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades
                       if getattr(t, 'profit', t.get('profit', 0)) > 0)
    gross_loss = -sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades
                      if getattr(t, 'profit', t.get('profit', 0)) < 0)

    profit_factor = safe_profit_factor(gross_profit, gross_loss)

    avg_win = (gross_profit / wins) if wins > 0 else 0.0
    avg_loss = (gross_loss / losses) if losses > 0 else 0.0
    loss_rate = losses / num_trades if num_trades > 0 else 0.0
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    max_dd = 0.0
    if hasattr(trades[0], 'max_drawdown_pct'):
        max_dd = max(getattr(t, 'max_drawdown_pct', 0) for t in trades)
    elif isinstance(trades[0], dict):
        max_dd = max(t.get('max_drawdown_pct', 0) for t in trades)

    summary = {
        "total_trades": num_trades, "win_rate": round(win_rate, 4),
        "profit_pct": round(total_profit_pct, 4),
        "profit_factor": None if profit_factor == float('inf') else round(profit_factor, 4),
        "expectancy": round(expectancy, 6),
        "avg_profit_per_trade": round(total_profit / num_trades, 6) if num_trades else 0.0,
        "max_drawdown_pct": round(max_dd, 4),
        "wins": wins, "losses": losses,
        "gross_profit": round(gross_profit, 2), "gross_loss": round(gross_loss, 2),
    }

    if num_trades < 30:
        summary["warning"] = f"⚠️ Small sample size ({num_trades} trades) - interpret with caution"
    else:
        summary["warning"] = ""

    return summary


def bootstrap_win_rate(trades: List[Any], n_iter: int = 1000, alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    num_trades = len(trades)
    if num_trades < 5:
        return None
    results = []
    for _ in range(n_iter):
        sample = [random.choice(trades) for _ in range(num_trades)]
        wins = sum(1 for t in sample if getattr(t, 'profit', t.get('profit', 0)) > 0)
        results.append(wins / num_trades)
    results.sort()
    lower_idx = int(n_iter * (alpha / 2))
    upper_idx = int(n_iter * (1 - alpha / 2))
    return results[lower_idx], results[upper_idx]


def compute_sortino(returns, target=0.0, periods_per_year=6048):
    # periods_per_year default = 252 trading days × 24 hours = 6048 for 1H bars.
    # Pass periods_per_year=252*6=1512 for 4H, 252 for daily.
    # Old default of 252 was for daily returns — caused Sortino to be ~5x understated on 1H.
    import numpy as _np
    if len(returns) < 2: return 0.0
    downside = returns[returns < target]
    if len(downside) == 0: return float('inf')
    d = _np.sqrt(_np.mean(downside ** 2))
    return (_np.mean(returns) / d) * _np.sqrt(periods_per_year) if d > 0 else float('inf')


def compute_sharpe(returns, risk_free=0.0, periods_per_year=6048):
    # periods_per_year default = 252 trading days × 24 hours = 6048 for 1H bars.
    # Pass periods_per_year=252*6=1512 for 4H, 252 for daily.
    # Old default of 252 was for daily returns — caused Sharpe to be ~5x understated on 1H.
    import numpy as _np
    if len(returns) < 2: return 0.0
    s = _np.std(returns)
    return ((_np.mean(returns) - risk_free) / s) * _np.sqrt(periods_per_year) if s > 0 else 0.0


def load_params_with_validation(param_dict: Dict, defaults: Dict) -> Dict:
    if not isinstance(param_dict, dict):
        logging.warning("Momentum params file malformed; using defaults.")
        return defaults.copy()
    updated = defaults.copy()
    for k, v in param_dict.items():
        if k in defaults:
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
    'quality': {'active': True, 'values': [60, 65, 70, 75], 'description': 'Quality Score Threshold'},
    'adx': {'active': True, 'values': [20, 22, 25, 28], 'description': 'ADX Minimum'},
    'rsi': {'active': True, 'values': [30, 35, 40, 45], 'description': 'RSI Minimum'},
    'volume': {'active': True, 'values': [0.9, 1.0, 1.1, 1.2], 'description': 'Volume Ratio'},
    'momentum': {'active': True, 'values': [0.1, 0.2, 0.3, 0.4], 'description': 'Momentum Minimum'},
    'ema_fast': {'active': True, 'values': [5, 8, 9, 12], 'description': 'EMA Fast Period'},
    'ema_mid': {'active': True, 'values': [18, 20, 21, 26], 'description': 'EMA Mid Period'},
    'ema_slow': {'active': True, 'values': [40, 45, 50, 55], 'description': 'EMA Slow Period'},
    'weight_ema': {'active': True, 'values': [15, 18, 20, 22], 'description': 'EMA Weight'},
    'weight_adx': {'active': True, 'values': [15, 18, 20, 22], 'description': 'ADX Weight'},
    'weight_macd': {'active': True, 'values': [20, 22, 25, 28], 'description': 'MACD Weight'},
    'weight_rsi': {'active': True, 'values': [15, 18, 20, 22], 'description': 'RSI Weight'},
    'weight_volume': {'active': True, 'values': [10, 12, 15, 18], 'description': 'Volume Weight'},
    'risk_tier1': {'active': True, 'values': [0.010, 0.012, 0.015, 0.018], 'description': 'Tier 1 Risk %'},
    'risk_tier2': {'active': True, 'values': [0.012, 0.015, 0.018, 0.022], 'description': 'Tier 2 Risk %'},
    'stop_loss_mult': {'active': True, 'values': [2.5, 3.0, 3.5, 4.0], 'description': 'Stop Loss ATR Multiplier'},
}


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: RISK MANAGEMENT — v9.5: HARD CAP ON POSITION SIZE
# ═══════════════════════════════════════════════════════════════════════════

class ProfessionalRiskController:
    def __init__(self, starting_equity: float = None):
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
        self.max_position_size_pct = 0.18
        self.max_concentration_pct = 0.10
        self.max_active_trades = 10
        self.min_cash_reserve = 0.15
        self.base_risk_pct = 0.035
        # ═══ v9.5 FIX #3: Hard cap on position size in units ═══
        self.max_position_units = None  # Never exceed 50 units regardless of price
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
        if entry_price <= 0:
            return 0

        risk_per_trade = abs(entry_price - stop_loss_price) / entry_price
        if risk_per_trade <= 0 or risk_per_trade > 0.25:
            return 0

        reward_multiple = max(1.0, min(profit_factor, 4.0))
        expected_profit = risk_per_trade * reward_multiple

        if expected_profit <= 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = ((win_rate * expected_profit) - ((1 - win_rate) * risk_per_trade)) / expected_profit
        kelly_fraction = max(0.0, min(kelly_fraction, 0.08))

        quality_weight = max(0.5, min((quality_score / 75) ** 0.5, 1.5))

        tier_weight = 1.1 if tier == 2 else 1.0

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

        losing_streak_multiplier = 0.7 if self.consecutive_losses >= 2 else 1.0

        trailing_wr = 0.50
        if len(self.daily_trades) >= 5:
            recent = list(self.daily_trades)[-20:]
            wins = sum(1 for t in recent if t.get('profit', 0) > 0)
            trailing_wr = wins / len(recent) if recent else 0.50
        performance_adjust = max(0.5, trailing_wr) if len(self.daily_trades) >= 5 else 1.0

        equity_health = self.current_equity / max(1.0, self.peak_equity)
        # FROM:
        equity_adjustment = min(equity_health ** 2, 1.0)
        # TO:
        equity_adjustment = min(equity_health, 1.0)  # was squaring the penalty

        risk_pct = (self.base_risk_pct * quality_weight * adx_multiplier *
                    losing_streak_multiplier * performance_adjust * equity_adjustment *
                    tier_weight)

        risk_pct = max(0.001, min(risk_pct, 0.07))

        risk_amount = self.current_equity * risk_pct
        position_size = (risk_amount / (entry_price * risk_per_trade))
        max_position_amount = self.current_equity * self.max_position_size_pct

        position_size = min(position_size, max_position_amount / entry_price)

        if entry_price >= 1000:
            return max(0.0, round(position_size, 6))
        elif entry_price >= 100:
            return max(0.0, round(position_size, 4))
        else:
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
                self.risk_metrics.sharpe_ratio = compute_sharpe(returns)
                self.risk_metrics.sortino_ratio = compute_sortino(returns)

    def get_stats(self):
        return {
            'total_trades': self.total_trades,
            'win_rate': f"{self.risk_metrics.win_rate:.1f}%",
            'profit_factor': f"{self.risk_metrics.profit_factor:.2f}" if self.risk_metrics.profit_factor != float(
                'inf') else "∞",
            'max_drawdown': f"{self.risk_metrics.max_drawdown:.1%}",
            'sharpe_ratio': f"{self.risk_metrics.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.risk_metrics.sortino_ratio:.2f}",
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

    def is_tradeable(self, regime: str) -> bool:
        return regime not in ('RANGING_VOLATILE', 'UNDEFINED')

    def get_position_multiplier(self, regime):
        return {'BULLISH_LOW_VOL': 1.3, 'BULLISH_HIGH_VOL': 0.8, 'BULLISH_WEAK': 0.7,
                'BEARISH_DECLINING': 0.3, 'RANGING_VOLATILE': 0.5, 'UNDEFINED': 0.6}.get(regime, 0.6)


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: EXIT MANAGER — WITH FULL LONG/SHORT TRAILING STOP SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

class ProfessionalExitManager:
    def __init__(self, config: dict):
        self.config = config
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 1.5)
        self.profit_targets = {}
        self.macd_bearish_cross_enabled = config.get('macd_bearish_cross_exit', True)
        self.macd_cross_profit_min = config.get('macd_bearish_cross_profit_min', 1.0)
        self.ema_cross_exit_enabled = config.get('ema_cross_exit', True)
        self.ema_cross_profit_min = 2.0
        self.trailing_activation_pct = config.get('trailing_activation_pct', 0.005)
        self.trailing_distance_pct = config.get('trailing_distance_pct', 0.01)
        self.trailing_activation_r = config.get('trailing_activation_r', 3.0)
        self.initial_trailing_atr_mult = config.get('initial_trailing_atr_mult', 4.0)
        self.max_hold_bars = config.get('max_hold_bars', 120)
        self.min_hold_bars_before_stop = config.get('min_hold_bars_before_stop', 6)
        self.emergency_stop_multiplier = config.get('emergency_stop_multiplier', 2.0)

        self.take_profit_r1 = config.get('take_profit_r1', 3.0)
        self.take_profit_r2 = config.get('take_profit_r2', 5.0)
        self.take_profit_r3 = config.get('take_profit_r3', 8.0)
        self.partial_exit_pct = config.get('partial_exit_pct', 0.33)

    def get_initial_trailing_stop(self, entry_price, atr=None, position_type='long'):
        return entry_price

    def evaluate_exit(self, current_price, entry_price, stop_loss, highest_price, lowest_price,
                      bars_held, partial_exits, ema_fast, ema_mid, ema_slow,
                      macd, macd_signal, macd_prev, signal_prev, adx, atr,
                      position_type='long', trailing_activated=False, trailing_stop=None):

        if not hasattr(self, '_version_printed'):
            print("=" * 70)
            print("🎯 EXIT MANAGER VERSION: v9.5 - FULL LONG/SHORT/BOTH SUPPORT")
            print("=" * 70)
            self._version_printed = True

        stop_distance = atr * self.stop_loss_atr_mult

        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            profit_r = (current_price - entry_price) / stop_distance if stop_distance > 0 else 0
        else:
            profit_pct = (entry_price - current_price) / entry_price
            profit_r = (entry_price - current_price) / stop_distance if stop_distance > 0 else 0

        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0.01
        activation_threshold = max(atr_pct * 1.5, 0.015)
        trailing_distance_pct = max(atr_pct * 0.5, 0.01)

        if not hasattr(self, '_trade_debug_printed') or self._trade_debug_printed != entry_price:
            print(f"  🔬 VOLATILITY PARAMS: ATR={atr_pct:.2f}% | "
                  f"Activate@{activation_threshold:.2%} | Trail={trailing_distance_pct:.2%}")
            self._trade_debug_printed = entry_price

        # ═══ 1. HARD STOP ═══
        if position_type == 'long':
            if current_price <= stop_loss:
                if bars_held < self.min_hold_bars_before_stop:
                    emergency_stop = entry_price - (stop_distance * self.emergency_stop_multiplier)
                    if current_price <= emergency_stop:
                        return "stop_loss_hard_emergency", 1.0
                else:
                    return "stop_loss_hard", 1.0
        else:
            if current_price >= stop_loss:
                if bars_held < self.min_hold_bars_before_stop:
                    emergency_stop = entry_price + (stop_distance * self.emergency_stop_multiplier)
                    if current_price >= emergency_stop:
                        return "stop_loss_hard_emergency", 1.0
                else:
                    return "stop_loss_hard", 1.0

        # ═══ 2. TRAILING STOP HIT ═══
        if trailing_activated and trailing_stop is not None:
            if position_type == 'long' and current_price <= trailing_stop:
                print(f"  🎯 LONG TRAILING STOP HIT: ${current_price:.2f} <= ${trailing_stop:.2f}")
                return "trailing_stop_hit", 1.0
            elif position_type == 'short' and current_price >= trailing_stop:
                print(f"  🎯 SHORT TRAILING STOP HIT: ${current_price:.2f} >= ${trailing_stop:.2f}")
                return "trailing_stop_hit", 1.0

        # ═══ 3. PARTIAL TAKE-PROFIT LADDER (R-multiple scale-out) ═══
        # Checked highest-R-first so a trade that gapped straight past r1
        # lands on the correct tier instead of always taking r1 first.
        # `partial_exits` gates each tier so it only fires once per trade.
        if profit_r >= self.take_profit_r3 and partial_exits < 3:
            print(f"  💰 TAKE PROFIT R3 HIT: {profit_r:.2f}R >= {self.take_profit_r3:.2f}R")
            return "take_profit_r3", self.partial_exit_pct
        if profit_r >= self.take_profit_r2 and partial_exits < 2:
            print(f"  💰 TAKE PROFIT R2 HIT: {profit_r:.2f}R >= {self.take_profit_r2:.2f}R")
            return "take_profit_r2", self.partial_exit_pct
        if profit_r >= self.take_profit_r1 and partial_exits < 1:
            print(f"  💰 TAKE PROFIT R1 HIT: {profit_r:.2f}R >= {self.take_profit_r1:.2f}R")
            return "take_profit_r1", self.partial_exit_pct

        # ═══ 4. MACD CROSS EXIT ═══
        if self.macd_bearish_cross_enabled and profit_r >= self.macd_cross_profit_min:
            if position_type == 'long':
                if macd_prev >= signal_prev and macd < macd_signal:
                    if ema_fast > ema_slow and adx >= 30:
                        pass
                    else:
                        return "macd_bearish_cross", 1.0
            else:
                if macd_prev <= signal_prev and macd > macd_signal:
                    if ema_fast < ema_slow and adx >= 30:
                        pass
                    else:
                        return "macd_bullish_cross", 1.0

        # ═══ 5. EMA FULL REVERSAL (>= 2R) ═══
        if self.ema_cross_exit_enabled and profit_r >= 2.0:
            if position_type == 'long':
                if ema_fast < ema_mid < ema_slow:
                    return "ema_full_reversal", 1.0
            else:
                if ema_fast > ema_mid > ema_slow:
                    return "ema_full_reversal", 1.0

        # ═══ 6. ADX COLLAPSE + MACD INVERSION ═══
        if adx < 25 and profit_r >= 1.5:
            if position_type == 'long':
                if macd < macd_signal and not (current_price > ema_mid and ema_fast > ema_slow):
                    return "adx_collapse_trend_weak", 1.0
            else:
                if macd > macd_signal and not (current_price < ema_mid and ema_fast < ema_slow):
                    return "adx_collapse_trend_weak", 1.0

        # ═══ 7. MAX HOLD TIME ═══
        if bars_held >= self.max_hold_bars:
            return "max_hold_time", 1.0

        return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: CONFIGURATION — WITH DIRECTION CONTROL (v9.5 UPDATED)
# ═══════════════════════════════════════════════════════════════════════════
#
MOMENTUM_PARAMS = {
    # ═══ DIRECTION CONTROL ════════════════════════════════════════════
    "trade_direction": "both",

    # ═══ LONG THRESHOLDS ═══════════════════════════════════════════════
    "quality_tier1_min": 62,  # updated: 72 → 68
    "quality_tier2_min": 65,
    "fixed_threshold": 72,

    # ═══ SHORT THRESHOLDS ═══════════════════════════════════════════════
    "short_quality_tier1_min": 65,  # updated: 75 → 70
    "short_quality_tier2_min": 70,
    "short_fixed_threshold": 75,

    # Fuzzy Learning
    "fuzzy_mode_enabled": False,
    "fuzzy_learning_enabled": True,
    "fuzzy_safety_cutoffs": True,
    "fuzzy_default_margin_pct": 10,
    "fuzzy_absolute_min": 45,
    "fuzzy_absolute_max": 65,
    "fuzzy_min_confidence": 0.6,
    "fuzzy_min_samples": 5,
    "fuzzy_max_adjustment_pct": 15,
    "fuzzy_learning_rate": 0.3,
    "fuzzy_conservative_start": True,

    # EMA Settings
    "ema_fast_period": 10,
    "ema_mid_period": 18,
    "ema_slow_period": 45,

    # Regime Detection
    "regime_filter_enabled": True,
    "ranging_min_checks": 4,
    "bb_period": 20,
    "bb_std": 2.0,
    "kc_period": 20,
    "kc_atr_mult": 1.5,
    "chop_period": 14,
    "chop_threshold": 58,

    # Quality Component Weights — TOTAL 100 POINTS
    # 20 + 20 + 25 + 20 + 15 = 100
    "weight_ema": 25,
    "weight_adx": 15,
    "weight_macd": 28,
    "weight_rsi": 18,
    "weight_volume": 18,  # was 12 — corrected to make weights sum to exactly 100

    "ema_near_tolerance": 0.005,
    "rsi_dynamic_enabled": True,

    # Price Percentile Adjustments
    "price_percentile_bonus_early": 12,
    "price_percentile_penalty_late": 12,
    "price_percentile_early_threshold": 25,
    "price_percentile_late_threshold": 80,
    "price_percentile_lookback": 20,

    # ═══ LONG FILTERS ═══════════════════════════════════════════════════
    "tier1_adx_hard_min": 22,  # updated: 20 → 25
    "tier1_adx_min": 20,
    "tier1_rsi_min": 42,  # updated: 44 → 40
    "tier1_rsi_max": 68,  # updated: 64 → 68
    "tier1_volume_min": 1.0,  # hard default; relaxes to 0.2 when EMA stacked + ADX>=25
    "tier1_momentum_min": 0.01,
    "tier1_kalman_min": 0.0,
    "tier1_macd_gate": True,
    "tier1_price_ema_max_pct": 1.5,
    "daily_trend_filter_enabled": False,
    "daily_ema_period": 720,  # optimised: 1920→720 (30-day, less lag on recovery)
    "daily_trend_adx_override": 28,  # ADX min for strong-trend override below daily EMA

    "extended_run_max_pct_long": 12.0,  # updated: 8.0 → 12.0
    "extended_run_max_pct_short": 12.0,  # updated: 8.0 → 12.0
    "extended_run_lookback": 20,

    "atr_compression_enabled": True,
    "atr_compression_threshold": 0.25,  # fix: inflated spike avg blocks recoveries
    "atr_compression_lookback": 50,

    "trend_age_penalty_enabled": True,
    "trend_age_max_bars": 20,
    "trend_age_penalty_pts": 10,

    "consecutive_loss_cooldown_enabled": True,
    "consecutive_loss_threshold": 3,
    "consecutive_loss_cooldown_bars": 12,

    # === PRECISION FILTERS ================================================
    "dmi_spread_min_long": 0.0,
    "dmi_spread_min_short": 0.0,
    "ema_trending_bars": 3,
    "macd_hist_rising_bars": 0,
    "rsi_direction_bars": 3,
    "rsi_direction_min_move": 1.0,
    "macd_hist_positive_required_long": False,
    "macd_hist_negative_required_short": False,
    "bb_expand_required": False,
    "time_filter_enabled": False,
    "time_filter_start_utc": 6,
    "time_filter_end_utc": 23,

    # === BREAKEVEN STOP ===================================================
    "be_stop_enabled": True,
    "be_stop_r_trigger": 2.0,  # updated: 1.5 → 2.0
    "be_stop_no_progress_bars": 50,  # updated: 30 → 50

    # ═══ SHORT FILTERS ════════════════════════════════════════════════════
    "short_tier1_adx_hard_min": 25,  # updated: 28 → 30
    "short_tier1_rsi_max": 48,
    "short_tier1_rsi_min": 32,
    "short_tier1_volume_min": 1.2,
    "short_tier1_momentum_min": 0.04,
    "short_tier1_macd_gate": True,
    "daily_trend_down_filter_enabled": True,
    "short_require_lower_highs_bars": 2,
    "short_require_lower_lows_bars": 2,

    # ADX Scoring Bands
    "adx_score_trend_forming": 15,
    "adx_score_good_trend": 20,
    "adx_score_strong_trend": 25,
    "adx_score_very_strong": 32,
    "adx_score_extended": 38,

    # Tier 2 Filters
    "tier2_adx_min": 18,
    "tier2_volume_min": 0.5,  # tier2 is less strict than tier1 by design
    "tier2_volume_min_ratio": 1.1,
    "tier2_momentum_min": 0.05,
    "tier2_macd_histogram_min": 0.001,
    "tier2_require_macd_histogram": True,

    # MACD Settings
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "macd_score_line_vs_signal": 7,
    "macd_score_histogram_direction": 7,
    "macd_score_zero_cross": 4,
    "macd_score_histogram_value": 3,

    # ADX Settings
    "adx_min": 18,
    "adx_min_trend": 25,  # updated: 22 → 25
    "adx_period": 14,

    # RSI Settings
    "rsi_entry_min": 40,  # updated: 42 → 40
    "rsi_entry_max": 68,
    "rsi_period": 14,

    # Volume Settings
    "volume_min_ratio": 1.1,
    "volume_period": 20,
    "volume_ma_period": 20,

    # Momentum & Other Indicators
    "momentum_min": 0.05,
    "kalman_min_strength": 0.0,
    "cci_period": 20,
    "cci_filter_enabled": False,
    "vix_max_threshold": 40,

    # Risk Management
    "risk_per_trade": 0.028,
    "risk_full_position": 0.025,
    "risk_reduced_position": 0.016,
    "risk_aggressive_position": 0.03,
    "risk_tier1": 0.030,  # updated: 0.02 → 0.025
    "risk_tier2": 0.038,  # updated: 0.025 → 0.03
    "risk_tier2_exceptional": 0.035,  # updated: 0.03 → 0.035

    # ═══ STOP LOSS & TRAILING ═════════════════════════════════════════
    "stop_loss_atr_mult": 3.5,
    "trailing_stop_atr_mult": 6.5,  # updated: 5.5 → 6.5
    "atr_period": 14,
    "supertrend_atr_period": 10,
    "supertrend_multiplier": 3.0,
    "kalman_q_param": 0.05,
    "kalman_r_param": 0.8,
    "vix_atr_period": 14,
    "vix_rolling_period": 20,

    # Profit Targets (legacy - disabled)
    "take_profit_r1": 3.0,
    "take_profit_r2": 5.0,
    "take_profit_r3": 8.0,
    "profit_target_r1": 9999.0,
    "profit_target_r2": 9999.0,
    "profit_target_r3": 9999.0,

    # ═══ TRAILING STOP ACTIVATION ═════════════════════════════════════
    "trailing_activation_pct": 0.03,  # updated: 0.035 → 0.03
    "trailing_distance_pct": 0.035,

    # Backward compatibility
    "trailing_stop_pct": 0.1,
    "trailing_activation_r": 2.5,
    "initial_trailing_atr_mult": 5.5,

    # ═══ EXIT MANAGEMENT ═══════════════════════════════════════════════
    "min_hold_bars_before_stop": 4,
    "emergency_stop_multiplier": 1.5,
    "max_hold_bars": 500,  # updated: 300 → 500
    "cooldown_after_profit_target_bars": 2,

    # ═══ TIER CONTROL ═════════════════════════════════════════════════
    "only_tier2_entries": False,
    "backtest_only_tier2_active": True,
    "backtest_only_tier2_values": [True, False],

    # ═══ TRADE MANAGEMENT ═══════════════════════════════════════════════
    "max_daily_trades": 15,
    "min_bars_between_trades": 4,  # updated: 8 → 4
    "min_bars_between_trades_tier2": 3,
    "cooldown_tier2_enabled": True,
    "cooldown_after_loss_bars": 12,  # updated: 24 → 12

    # Exit Conditions
    "macd_bearish_cross_exit": True,
    "macd_bearish_cross_profit_min": 2.5,
    "ema_cross_exit": True,
    "momentum_reversal_exit": True,
    "momentum_reversal_threshold": -0.3,
    "momentum_reversal_profit_min": 0.05,
    "rsi_exit_threshold": 80,
    "kalman_fade_threshold": 35,
    "profit_min_fade": 1.0,
    "profit_min_time_exit": 0.5,
    "profit_min_ma_crossover": 0.5,

    # Advanced Features
    "volatility_scaling": False,
    "trade_high_vol": False,
    "trade_ranging": False,
    "supertrend_exit_enabled": False,

    # v9.5 FIX #3: Position size hard cap
    "max_position_units": 50,

    # ═══ PULLBACK ZONE & ADX SLOPE — parameterized (previously hardcoded) ═══
    # Defaults reproduce original v9.5 behaviour exactly.
    # These were previously hardcoded in _check_tier1_filters (shorts only).
    # Now also enforced for longs via _validate_direction.
    "pullback_zone_lower_pct": -2.5,  # price % below EMA_Fast allowed at entry
    "pullback_zone_upper_pct": 1.5,  # price % above EMA_Fast allowed at entry
    "adx_slope_min": 0.1,  # min ADX rise per bar required to enter

    # ═══ DIRECTION CONFLUENCE GATE ════════════════════════════════════════
    # _validate_direction() used to require ALL ~8-9 long-side / ~7 short-side
    # conditions to pass simultaneously (a hard AND-chain). Backtest replay
    # showed each condition individually passes only ~45-55% of the time, so
    # the AND-chain joint-probability made no_valid_direction fire on ~80% of
    # bars regardless of which single threshold was tuned. Replaced with a
    # confluence-vote: only "full bearish/bullish EMA stack" remains a hard
    # block (trading straight against full EMA structure is genuinely risky);
    # everything else is scored, and a direction is valid once it clears
    # direction_confluence_min of its checks. 0.75 (75%) tested well in
    # backtest replay: no_valid_direction dropped from ~79.5% to ~35%, while
    # long_valid rose from ~9.5% to ~36.6% and short_valid from ~11% to ~28%.
    "direction_confluence_min": 0.75,

    # ═══ KALMAN STRATEGY PARAMETERS ══════════════════════════════════════
    # KalmanTrendStrategy uses config['key'] direct access — all keys it
    # references must exist here. Keys that conflict with momentum values
    # (adx_period, volume_period, max_hold_bars, risk_per_trade,
    #  trailing_stop_pct, volume_min_ratio) are supplied by the JSON
    # kalman section and are NOT duplicated here to avoid overwriting
    # momentum values. Only Kalman-unique keys are added below.
    "trading_direction": "both",  # Kalman direction flag
    "process_noise_1": 0.001,
    "process_noise_2": 0.001,
    "measurement_noise": 100.0,
    "trend_lookback": 20,
    "strength_smooth": 5,
    "strength_smooth_param": 5,
    "risk_reward": 1.5,
    "lookback": 20,
    "window": 10,
    "ma_fast_period": 20,
    "ma_slow_period": 50,
    "long_kalman_strength_min": 30,
    "long_rsi_min": 30,
    "long_rsi_max": 70,
    "long_pullback_percent": 0.1,
    "long_rsi_exit_threshold": 80,
    "short_kalman_strength_min": -30,
    "short_rsi_min": 30,
    "short_rsi_max": 70,
    "short_rally_percent": 0.1,
    "short_rsi_exit_threshold": 20,
    "stop_loss_pct": 0.02,  # Kalman stop loss %
    "atr_multiplier": 2.0,
    "max_position_pct": 0.15,
    "min_hold_bars": 2,  # Kalman min hold (vs min_hold_bars_before_stop)
    "max_hold_seconds": 3600,
    "min_adx": 15,  # Kalman ADX min (vs tier1_adx_hard_min)
    "min_volatility": 0.001,
    "max_spread_pct": 0.001,
    "cooldown_bars": 10,
}


# ═══════════════════════════════════════════════════════════════════════════
# PART 6: CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class MomentumConfig:
    CONFIG_FILE = "strategy_settings.json"
    _custom_params = {}
    _current_mode = "Default Parameters"
    _saved_mode = "Default Parameters"

    @classmethod
    def get_config(cls, momentum_params_override=None):
        try:
            config = MOMENTUM_PARAMS.copy()
            param_sources = {}

            for key in config.keys():
                param_sources[key] = {'value': config[key], 'source': 'MOMENTUM_PARAMS', 'overridden': False}

            logging.info("=" * 80)
            logging.info("📋 STEP 1: Loading MOMENTUM_PARAMS (Single Source of Truth)")
            logging.info("=" * 80)

            cls._custom_params = {}
            custom_params_applied = 0

            if os.path.exists(cls.CONFIG_FILE):
                try:
                    with open(cls.CONFIG_FILE, 'r') as f:
                        saved = json.load(f)
                        raw_custom = saved.get('custom_params', {})
                        if 'momentum' in raw_custom and isinstance(raw_custom.get('momentum'), dict):
                            cls._custom_params = raw_custom['momentum']
                        elif raw_custom and not any(isinstance(v, dict) for v in raw_custom.values()):
                            cls._custom_params = raw_custom
                        else:
                            cls._custom_params = {}
                        if 'selected_mode' in saved:
                            cls._saved_mode = saved['selected_mode']
                            cls._current_mode = cls._saved_mode

                    if cls._current_mode == "Custom Parameters" and cls._custom_params:
                        for key, value in sorted(cls._custom_params.items()):
                            if key in config:
                                old_value = config[key]
                                config[key] = value
                                param_sources[key] = {'value': value, 'source': 'Custom params (file)',
                                                      'overridden': True}
                                if old_value != value:
                                    custom_params_applied += 1
                except Exception as e:
                    logging.error(f"Error loading saved config: {e}")
                    cls._custom_params = {}

            runtime_overrides_applied = 0
            if momentum_params_override and isinstance(momentum_params_override, dict):
                for key, value in sorted(momentum_params_override.items()):
                    if key in config:
                        old_value = config[key]
                        config[key] = value
                        param_sources[key] = {'value': value, 'source': 'Runtime override', 'overridden': True}
                        if old_value != value:
                            runtime_overrides_applied += 1

            logging.info(
                f"Total parameters: {len(config)} | Custom: {custom_params_applied} | Runtime: {runtime_overrides_applied}")
            return config

        except Exception as e:
            logging.error(f"❌ Config load error: {e}")
            return MOMENTUM_PARAMS.copy()

    @classmethod
    def get_custom_params(cls):
        return cls._custom_params

    @classmethod
    def get_current_mode(cls):
        return cls._current_mode

    @classmethod
    def set_current_mode(cls, mode):
        cls._current_mode = mode
        cls._saved_mode = mode

    @classmethod
    def save_config(cls, config, custom_params=None, selected_mode=None):
        try:
            os.makedirs("strategy_configs", exist_ok=True)
            mode = selected_mode if selected_mode else cls._current_mode
            if mode == "Custom Parameters" and custom_params:
                params_to_save = custom_params
            else:
                params_to_save = {}

            save_data = {
                'timestamp': datetime.now().isoformat(),
                'selected_mode': mode,
                'custom_params': params_to_save,
                'note': 'MOMENTUM_PARAMS in code is the single source of truth for defaults'
            }
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(save_data, f, indent=4)

            cls._custom_params = params_to_save
            cls._current_mode = mode
            cls._saved_mode = mode
            return True
        except Exception as e:
            logging.error(f"❌ Config save error: {e}")
            return False

    @classmethod
    def reset_to_defaults(cls):
        return MOMENTUM_PARAMS.copy()

    @classmethod
    def validate_config(cls, config):
        modified = False
        if 'only_tier2_entries' in config:
            if isinstance(config['only_tier2_entries'], str):
                config['only_tier2_entries'] = config['only_tier2_entries'].lower() == 'true'
                modified = True
        return config, modified

    @classmethod
    def get_timeframe_aware_params(cls, timeframe: str) -> dict:
        """
        Return a param-override dict that scales all bar-count parameters to the
        given timeframe so real-time durations stay consistent across assets and
        intervals.

        Root cause: daily_ema_period=720 was hardcoded for SOL-USDT 1H where
        720 bars == 30 days. On 4H bars 720 == 120 days (far too slow); on 1D
        bars 720 == 720 days (~2 years, broken). This caused 0 trades on BTC/ETH
        at any non-1H interval because the daily trend filter blocked everything.

        Usage
        -----
        params = {**your_custom_params,
                  **MomentumConfig.get_timeframe_aware_params('4H')}
        BacktestMomentumStrategy.set_updated_params(params)

        Supported timeframes: '1m','3m','5m','15m','30m','1H','2H','4H',
                              '6H','8H','12H','1D','3D','1W'
        """
        BARS_PER_DAY: dict = {
            '1M': 1440, '3M': 480, '5M': 288, '15M': 96,
            '30M': 48, '1H': 24, '2H': 12, '4H': 6,
            '6H': 4, '8H': 3, '12H': 2, '1D': 1,
            '3D': 1 / 3, '1W': 1 / 7,
        }
        tf = timeframe.upper().replace('MIN', 'M').replace('HOUR', 'H').replace('DAY', 'D')
        bpd = BARS_PER_DAY.get(tf)
        if bpd is None:
            logging.warning(
                f"get_timeframe_aware_params: unknown timeframe '{timeframe}'. "
                f"Returning empty dict — MOMENTUM_PARAMS defaults will be used as-is."
            )
            return {}

        def bars(days: float) -> int:
            return max(1, int(round(days * bpd)))

        params = {
            # Trend filter: always 30-day EMA regardless of interval
            # On 1H (bpd=24) = 720  → matches MOMENTUM_PARAMS default
            # On 4H (bpd=6)  = 180  → was 720 = 120 days (broken)
            # On 1D (bpd=1)  = 30   → was 720 = 720 days (broken)
            'daily_ema_period': bars(30),

            # ATR compression lookback: 2 weeks of bars
            'atr_compression_lookback': bars(14),

            # Trend age penalty: 5 days of bars
            'trend_age_max_bars': bars(5),

            # Cooldowns: ~12 real hours regardless of timeframe
            'cooldown_after_loss_bars': max(1, bars(0.5)),
            'consecutive_loss_cooldown_bars': max(1, bars(0.5)),
            'min_bars_between_trades': max(1, bars(0.17)),  # ~4h

            # BE stop no-progress window: 2 days of bars
            'be_stop_no_progress_bars': bars(2),

            # Max hold bars: 3 weeks of bars
            'max_hold_bars': bars(21),
        }

        logging.info(
            f"⏱  TIMEFRAME PARAMS for '{timeframe}' (bpd={bpd}):\n"
            + "\n".join(f"   {k}: {v}" for k, v in params.items())
        )
        return params


# ═══════════════════════════════════════════════════════════════════════════
# PART 7: INDICATOR CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorCalculator:
    @staticmethod
    def calculate(df, params):
        df = df.copy()
        try:
            df['EMA_Fast'] = talib.EMA(df['Close'], params['ema_fast_period'])
            df['EMA_Mid'] = talib.EMA(df['Close'], params['ema_mid_period'])
            df['EMA_Slow'] = talib.EMA(df['Close'], params['ema_slow_period'])
            # default 720 matches MOMENTUM_PARAMS (30 days on 1H)
            daily_ema_period = params.get('daily_ema_period', 720)
            df['EMA_Daily_50'] = talib.EMA(df['Close'], daily_ema_period)
            df['Above_Daily_50'] = (df['Close'] > df['EMA_Daily_50']).astype(bool)
            df['EMA_200'] = talib.EMA(df['Close'], 200)
            df['Above_EMA200'] = (df['Close'] > df['EMA_200']).astype(bool)
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], params['adx_period'])
            df['RSI'] = talib.RSI(df['Close'], params['rsi_period'])
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], params['cci_period'])
            df['Volume_MA'] = talib.SMA(df['Volume'], params['volume_period'])
            with np.errstate(divide='ignore', invalid='ignore'):
                df['Volume_Ratio'] = np.where(df['Volume_MA'] > 0, df['Volume'] / df['Volume_MA'], 1.0)
            df['Volume_Ratio'] = (df['Volume_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.01, 10.0))
            df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(
                df['Close'], fastperiod=params['macd_fast'],
                slowperiod=params['macd_slow'], signalperiod=params['macd_signal'])
            df['ADX_prev'] = df['ADX'].shift(1)
            df['MACD_Above_Signal'] = df['MACD'] > df['MACD_Signal']
            df['MACD_Above_Zero'] = df['MACD'] > 0
            df['MACD_Histogram_Rising'] = df['MACD_Histogram'] > df['MACD_Histogram'].shift(1)
            df['MACD_Histogram_Positive'] = df['MACD_Histogram'] > 0
            df['Momentum'] = df['Close'].pct_change(5) * 100
            df['Momentum_1'] = df['Close'].pct_change(1) * 100
            df['Momentum_prev'] = df['Momentum'].shift(1)
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], params['atr_period'])
            df['DMP'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], params['adx_period'])
            df['DMM'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], params['adx_period'])
            df['Kalman_Strength'] = IndicatorCalculator._calculate_kalman_strength(df, params)
            df = IndicatorCalculator._detect_ranging_market(df, params)
            lookback = params.get('price_percentile_lookback', 20)
            df['High_20bar'] = df['High'].rolling(window=lookback).max()
            df['Low_20bar'] = df['Low'].rolling(window=lookback).min()
            df['Price_Range_20bar'] = df['High_20bar'] - df['Low_20bar']
            with np.errstate(divide='ignore', invalid='ignore'):
                df['Price_Percentile_20bar'] = np.where(
                    df['Price_Range_20bar'] > 0,
                    ((df['Close'] - df['Low_20bar']) / df['Price_Range_20bar']) * 100, 50.0)
            df['Price_Percentile_20bar'] = df['Price_Percentile_20bar'].clip(0, 100).fillna(50)
            df['Price_Percentile_20bar_closed'] = df['Price_Percentile_20bar'].shift(1)

            # v9.5: Pre-compute ATR compression indicator
            atr_lookback = params.get('atr_compression_lookback', 50)
            df['ATR_MA50'] = df['ATR'].rolling(window=atr_lookback).mean()
            # BUG FIX: default was 0.70, MOMENTUM_PARAMS defines 0.25.
            # Using 0.70 here while _is_atr_compressed uses 0.25 caused
            # ATR_Compressed column to mismatch the live filter logic.
            df['ATR_Compressed'] = (df['ATR'] < df['ATR_MA50'] * params.get('atr_compression_threshold', 0.25))

            # v9.5: Pre-compute swing high/low for extended run filter
            run_lookback = params.get('extended_run_lookback', 20)
            df['Swing_Low_20'] = df['Low'].rolling(window=run_lookback).min()
            df['Swing_High_20'] = df['High'].rolling(window=run_lookback).max()
            df['Run_From_Low_Pct'] = ((df['Close'] - df['Swing_Low_20']) / df['Swing_Low_20'] * 100).fillna(0)
            df['Run_From_High_Pct'] = ((df['Swing_High_20'] - df['Close']) / df['Swing_High_20'] * 100).fillna(0)

            # v9.5: Pre-compute trend age (bars since EMA fast crossed above/below slow)
            ema_bullish = (df['EMA_Fast'] > df['EMA_Slow']).astype(int)
            ema_bearish = (df['EMA_Fast'] < df['EMA_Slow']).astype(int)
            # Count consecutive bars in current trend direction
            bullish_groups = ema_bullish.ne(ema_bullish.shift()).cumsum()
            bearish_groups = ema_bearish.ne(ema_bearish.shift()).cumsum()
            df['Trend_Age_Bullish'] = ema_bullish.groupby(bullish_groups).cumsum()
            df['Trend_Age_Bearish'] = ema_bearish.groupby(bearish_groups).cumsum()

            indicators_to_export = [
                'EMA_Fast', 'EMA_Mid', 'EMA_Slow', 'EMA_Daily_50', 'Above_Daily_50',
                'EMA_200', 'Above_EMA200',
                'DMP', 'DMM',
                'ADX', 'RSI', 'CCI', 'Volume_Ratio', 'Momentum', 'Momentum_1', 'Momentum_prev',
                'MACD', 'MACD_Signal', 'MACD_Histogram',
                'MACD_Above_Signal', 'MACD_Above_Zero',
                'MACD_Histogram_Rising', 'MACD_Histogram_Positive',
                'ATR', 'Kalman_Strength', 'Price_Percentile_20bar',
                'UpperBand', 'MiddleBand', 'LowerBand', 'BB_Width', 'BB_Z',
                'KC_Upper', 'KC_Mid', 'KC_Lower', 'KC_Width',
                'Squeeze', 'ATR_MA30', 'EMA_Fast_diff', 'CHOP', 'Ranging',
                'ATR_MA50', 'ATR_Compressed',  # v9.5
                'Swing_Low_20', 'Swing_High_20', 'Run_From_Low_Pct', 'Run_From_High_Pct',  # v9.5
                'Trend_Age_Bullish', 'Trend_Age_Bearish',  # v9.5
            ]
            for col in indicators_to_export:
                if col in df.columns:
                    df[f'{col}_closed'] = df[col].shift(1)
            return df
        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
            raise

    @staticmethod
    def _calculate_kalman_strength(df, params):
        spread = (df['EMA_Fast'] - df['EMA_Slow']) / df['EMA_Slow'] * 100
        return np.clip(np.abs(spread) / 5.0, 0, 1)

    @staticmethod
    def _detect_ranging_market(df, params):
        if df is None or df.empty:
            return df
        df = df.copy()
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        if 'UpperBand' not in df.columns:
            df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(
                df['Close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std, matype=0)
        df['BB_Width'] = (df['UpperBand'] - df['LowerBand']) / df['Close']
        kc_period = params.get('kc_period', 20)
        kc_mult = params.get('kc_atr_mult', 1.5)
        if 'KC_Upper' not in df.columns:
            df['KC_Mid'] = df['Close'].ewm(span=kc_period).mean()
            df['KC_ATR'] = df['ATR'].rolling(window=kc_period).mean()
            df['KC_Upper'] = df['KC_Mid'] + kc_mult * df['KC_ATR']
            df['KC_Lower'] = df['KC_Mid'] - kc_mult * df['KC_ATR']
        df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']
        bb_mean = df['BB_Width'].rolling(50).mean()
        bb_std_rolling = df['BB_Width'].rolling(50).std()
        df['BB_Z'] = (df['BB_Width'] - bb_mean) / bb_std_rolling.replace(0, 1)
        df['Squeeze'] = df['BB_Width'] < df['KC_Width']
        df['ATR_MA30'] = df['ATR'].rolling(30).mean()
        atr_threshold = df['ATR'].rolling(100).quantile(0.25)
        df['EMA_Fast_diff'] = df['EMA_Fast'].pct_change() * 100
        chop_period = params.get('chop_period', 14)
        df['CHOP'] = IndicatorCalculator._choppiness_index(df['High'], df['Low'], df['Close'], period=chop_period)
        chop_threshold = params.get('chop_threshold', 60)
        min_checks = params.get('ranging_min_checks', 4)  # default matches MOMENTUM_PARAMS (was wrong 5)
        c1 = (abs(df['Close'] - df['EMA_Fast']) / df['EMA_Fast'] <= 0.005).fillna(False)
        c2 = (df['BB_Z'] < -0.5).fillna(False)
        c3 = df['Squeeze'].fillna(False)
        c4 = (df['ATR'] < atr_threshold).fillna(False)
        c5 = (abs(df['EMA_Fast_diff']) <= 0.05).fillna(False)
        c6 = df['RSI'].between(45, 55).fillna(False)
        c7 = (df['CHOP'] >= chop_threshold).fillna(False)
        c8 = (df['ADX'] < 20).fillna(False)
        ranging_score = (c1.astype(int) + c2.astype(int) + c3.astype(int) +
                         c4.astype(int) + c5.astype(int) + c6.astype(int) +
                         c7.astype(int) + c8.astype(int))
        df['Ranging'] = (ranging_score >= min_checks).fillna(False)
        return df

    @staticmethod
    def _choppiness_index(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_sum = tr.rolling(window=period).sum()
        high_max = high.rolling(window=period).max()
        low_min = low.rolling(window=period).min()
        chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(period)
        return chop.fillna(50)


# ═══════════════════════════════════════════════════════════════════════════
# PART 8: CORE MOMENTUM LOGIC — v9.5 WITH ALL FIXES
# ═══════════════════════════════════════════════════════════════════════════

class MomentumLogic:
    def __init__(self, config=None, trading_app=None):
        momentum_params_override = None
        if trading_app and hasattr(trading_app, 'get_current_momentum_params'):
            momentum_params_override = trading_app.get_current_momentum_params()

        if config is None:
            self.config = MomentumConfig.get_config(momentum_params_override)
        else:
            self.config = config

        self.trading_app = trading_app
        self.custom_params = MomentumConfig.get_custom_params()
        self.current_mode = MomentumConfig.get_current_mode()

        for key, value in self.config.items():
            setattr(self, key, value)

        self.ai_data_available = False

        required_params = {
            'trailing_activation_pct': 0.03, 'trailing_distance_pct': 0.035,
            'trade_direction': 'both', 'only_tier2_entries': False,
            'quality_tier2_min': 65, 'quality_tier1_min': 68,
            'short_quality_tier2_min': 70, 'short_quality_tier1_min': 70,
            'short_fixed_threshold': 75, 'fixed_threshold': 72,
            'tier1_adx_hard_min': 25, 'short_tier1_adx_hard_min': 30,
            'tier1_volume_min': 0.8, 'stop_loss_atr_mult': 3.5,
            'max_position_units': 50,
        }

        for param_name, default_value in required_params.items():
            if not hasattr(self, param_name) or getattr(self, param_name) is None:
                setattr(self, param_name, default_value)

        self.risk_controller = ProfessionalRiskController()
        # v9.5 FIX #3: sync hard cap from config
        self.risk_controller.max_position_units = getattr(self, 'max_position_units', 50)
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
        self.near_miss_trades = []
        self._previous_fuzzy_lower = None
        self.fuzzy_mode_enabled = getattr(self, 'fuzzy_mode_enabled', False)
        self._current_df = None
        self.backtest_params = BACKTEST_PARAMS.copy()

        self.trailing_activated = False
        self.highest_price = 0
        self.trailing_stop = None

        self._pending_signal = None
        self._signal_bar = -999
        self._signal_price = None

        self._last_profit_target_bar = -999
        self.trade_direction = (
            trading_app.trade_direction_var.get()
            if trading_app and hasattr(trading_app, 'trade_direction_var')
            else self.config.get('trade_direction', 'long')
        )
        self.only_long_entries = (self.trade_direction == 'long')
        self.only_short_entries = (self.trade_direction == 'short')

        self._last_direction_check = {
            'bar': -999, 'direction': None, 'result': True, 'reason': '', 'action': 'hold'
        }
        self._suggested_action = None

        # v9.5 FIX #6: Track consecutive losses for extended cooldown
        self._consecutive_loss_count = 0
        self._last_loss_bar = -999

        self._log_parameter_source()

    def _log_parameter_source(self):
        logging.info("=" * 70)
        logging.info("📊 PARAMETER SOURCE v9.5 (SINGLE SOURCE OF TRUTH)")
        logging.info("=" * 70)
        logging.info(f"   Direction: {self.trade_direction.upper()}")
        logging.info(f"   ATR Compression Filter: {getattr(self, 'atr_compression_enabled', True)}")
        logging.info(f"   Extended Run Filter: max {getattr(self, 'extended_run_max_pct_long', 12.0)}%")
        logging.info(f"   Max Position Units: {getattr(self, 'max_position_units', 50)}")
        logging.info(f"   Short ADX Min: {getattr(self, 'short_tier1_adx_hard_min', 30)}")
        logging.info(f"   Short Volume Min: {getattr(self, 'short_tier1_volume_min', 1.3)}")
        logging.info(f"   Consecutive Loss Cooldown: {getattr(self, 'consecutive_loss_cooldown_bars', 12)} bars")
        logging.info("=" * 70)

    def _log(self, message, color="white"):
        if self.trading_app and hasattr(self.trading_app, 'log_message'):
            self.trading_app.log_message(message, color)
        else:
            print(f"[{color}] {message}")

    def _transition_to_in_trade(self):
        self.strategy_state = StrategyState.IN_TRADE

    def _transition_to_seeking_entry(self):
        self.strategy_state = StrategyState.SEEKING_ENTRY

    def _near_or_above(self, a, b, tolerance=0.005):
        return True if a > b else (b > 0 and abs(a - b) / b <= tolerance)

    def _get_position_multiplier(self, quality_score):
        if quality_score >= 90:
            return getattr(self, 'position_100_90', 1.5)
        elif quality_score >= 80:
            return getattr(self, 'position_89_80', 1.3)
        elif quality_score >= 70:
            return getattr(self, 'position_79_70', 1.1)
        elif quality_score >= 60:
            return getattr(self, 'position_69_60', 0.9)
        elif quality_score >= 50:
            return getattr(self, 'position_59_50', 0.6)
        else:
            return getattr(self, 'position_below_50', 0.0)

    # ═══════════════════════════════════════════════════════════════════════
    # v9.5 FIX #1: ATR COMPRESSION FILTER
    # ═══════════════════════════════════════════════════════════════════════

    def _is_atr_compressed(self, data):
        """Block ALL entries when ATR is compressed (chop/range)"""
        if not getattr(self, 'atr_compression_enabled', True):
            return False

        df = getattr(self, '_current_df', None)
        if df is None or len(df) < 50:
            return False

        try:
            atr_now = data.get('ATR', 0)
            if 'ATR_MA50' in df.columns:
                atr_avg = float(df['ATR_MA50'].iloc[-1])
            else:
                atr_avg = float(df['ATR'].iloc[-50:].mean())

            if atr_avg <= 0:
                return False

            threshold = getattr(self, 'atr_compression_threshold', 0.25)  # matches MOMENTUM_PARAMS
            is_compressed = atr_now < (atr_avg * threshold)

            # Strong-trend override: when EMA fast>mid>slow AND ADX>=25,
            # skip the compression block — post-spike recovery in a confirmed
            # trend must not be blocked by an ATR average inflated by the spike
            if is_compressed and self._current_df is not None:
                try:
                    _row = self._current_df.iloc[-1]
                    _ef = float(_row.get('EMA_Fast', 0))
                    _em = float(_row.get('EMA_Mid', 0))
                    _es = float(_row.get('EMA_Slow', 0))
                    _ad = float(_row.get('ADX', 0))
                    if _ef > _em > _es and _ad >= 25:
                        return False  # confirmed trend: override compression block
                except Exception:
                    pass

            if is_compressed:
                ratio = atr_now / atr_avg if atr_avg > 0 else 0
                self._log(f"🔴 ATR COMPRESSED: {atr_now:.4f} < {atr_avg:.4f} × {threshold} "
                          f"(ratio={ratio:.2f}) — NO ENTRIES", "red")
            return is_compressed
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════════════
    # v9.5 FIX #2: EXTENDED RUN FILTER
    # ═══════════════════════════════════════════════════════════════════════

    def _is_extended_run_long(self, data):
        """Block longs when price has run >5% from recent swing low"""
        df = getattr(self, '_current_df', None)
        if df is None or len(df) < 20:
            return False

        try:
            close = data.get('Close', 0)
            if 'Swing_Low_20' in df.columns:
                swing_low = float(df['Swing_Low_20'].iloc[-1])
            else:
                lookback = getattr(self, 'extended_run_lookback', 20)
                swing_low = float(df['Low'].iloc[-lookback:].min())

            if swing_low <= 0:
                return False

            run_pct = (close - swing_low) / swing_low * 100
            max_run = getattr(self, 'extended_run_max_pct_long', 12.0)  # matches MOMENTUM_PARAMS

            if run_pct > max_run:
                self._log(f"🔴 EXTENDED RUN (LONG): Price ran {run_pct:.1f}% from swing low "
                          f"${swing_low:.2f} → ${close:.2f} (max={max_run}%)", "red")
                return True
            return False
        except Exception:
            return False

    def _is_extended_run_short(self, data):
        """Block shorts when price has dropped >5% from recent swing high"""
        df = getattr(self, '_current_df', None)
        if df is None or len(df) < 20:
            return False

        try:
            close = data.get('Close', 0)
            if 'Swing_High_20' in df.columns:
                swing_high = float(df['Swing_High_20'].iloc[-1])
            else:
                lookback = getattr(self, 'extended_run_lookback', 20)
                swing_high = float(df['High'].iloc[-lookback:].max())

            if swing_high <= 0:
                return False

            run_pct = (swing_high - close) / swing_high * 100
            max_run = getattr(self, 'extended_run_max_pct_short', 12.0)  # matches MOMENTUM_PARAMS

            if run_pct > max_run:
                self._log(f"🔴 EXTENDED RUN (SHORT): Price dropped {run_pct:.1f}% from swing high "
                          f"${swing_high:.2f} → ${close:.2f} (max={max_run}%)", "red")
                return True
            return False
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════════════
    # v9.5 FIX #5: TREND AGE PENALTY FOR QUALITY SCORE
    # ═══════════════════════════════════════════════════════════════════════

    def _get_trend_age_penalty(self, data, direction='long'):
        """Return quality score penalty if trend has been running too long"""
        if not getattr(self, 'trend_age_penalty_enabled', True):
            return 0

        df = getattr(self, '_current_df', None)
        if df is None or len(df) < 5:
            return 0

        try:
            max_bars = getattr(self, 'trend_age_max_bars', 20)
            penalty_pts = getattr(self, 'trend_age_penalty_pts', 10)

            if direction == 'long' and 'Trend_Age_Bullish' in df.columns:
                age = int(df['Trend_Age_Bullish'].iloc[-1])
            elif direction == 'short' and 'Trend_Age_Bearish' in df.columns:
                age = int(df['Trend_Age_Bearish'].iloc[-1])
            else:
                # Fallback: count consecutive bars where EMA_Fast > EMA_Slow (long) or vice versa
                ema_f = df['EMA_Fast'].iloc[-max_bars * 2:] if len(df) > max_bars * 2 else df['EMA_Fast']
                ema_s = df['EMA_Slow'].iloc[-max_bars * 2:] if len(df) > max_bars * 2 else df['EMA_Slow']
                if direction == 'long':
                    aligned = (ema_f > ema_s).astype(int)
                else:
                    aligned = (ema_f < ema_s).astype(int)
                # Count consecutive True from the end
                age = 0
                for v in reversed(aligned.values):
                    if v == 1:
                        age += 1
                    else:
                        break

            if age >= max_bars:
                return penalty_pts
            return 0
        except Exception:
            return 0

    # ═══════════════════════════════════════════════════════════════════════
    # v9.5 FIX #6: CONSECUTIVE LOSS COOLDOWN
    # ═══════════════════════════════════════════════════════════════════════

    def _is_consecutive_loss_cooldown_active(self):
        """Check if we should skip entries due to consecutive losses"""
        if not getattr(self, 'consecutive_loss_cooldown_enabled', True):
            return False

        threshold = getattr(self, 'consecutive_loss_threshold', 3)
        cooldown_bars = getattr(self, 'consecutive_loss_cooldown_bars', 12)  # default updated in v9.5.1

        if self._consecutive_loss_count >= threshold:
            bars_since_loss = self.bar_count - self._last_loss_bar
            if bars_since_loss < cooldown_bars:
                remaining = cooldown_bars - bars_since_loss
                # BUG FIX v9.5.1: was hard-coded to 15-min bars; use actual bar interval
                _bih = getattr(self, '_bar_interval_hours', 1.0)
                self._log(f"🔴 CONSECUTIVE LOSS COOLDOWN: {self._consecutive_loss_count} losses, "
                          f"{remaining} bars remaining ({remaining * _bih:.1f}h)", "red")
                return True
            else:
                # Cooldown expired, reset
                self._consecutive_loss_count = 0
        return False

    # ═══════════════════════════════════════════════════════════════════════
    # v9.5 FIX #4: STRICTER SHORT PRICE STRUCTURE CHECK
    # ═══════════════════════════════════════════════════════════════════════

    def _has_bearish_price_structure(self, data):
        """Check for required lower highs AND lower lows for short entries"""
        df = getattr(self, '_current_df', None)
        if df is None or len(df) < 10:
            return True  # Allow if no data

        try:
            required_lh = getattr(self, 'short_require_lower_highs_bars', 2)  # default updated in v9.5.1
            required_ll = getattr(self, 'short_require_lower_lows_bars', 2)

            recent_highs = df['High'].iloc[-required_lh - 1:].values
            recent_lows = df['Low'].iloc[-required_ll - 1:].values

            # Check lower highs
            lower_highs = True
            for i in range(1, len(recent_highs)):
                if recent_highs[i] >= recent_highs[i - 1]:
                    lower_highs = False
                    break

            # Check lower lows
            lower_lows = True
            for i in range(1, len(recent_lows)):
                if recent_lows[i] >= recent_lows[i - 1]:
                    lower_lows = False
                    break

            if not lower_highs:
                self._log(f"🔴 SHORT BLOCKED: No {required_lh} consecutive lower highs", "orange")
                return False

            if not lower_lows:
                self._log(f"🔴 SHORT BLOCKED: No {required_ll} consecutive lower lows", "orange")
                return False

            return True
        except Exception:
            return True

    def _momentum_is_sustainable(self, data):
        df = getattr(self, '_current_df', None)
        if df is None or len(df) < 15:
            return True
        try:
            recent_closes = df['Close'].iloc[-6:-1].values
            recent_highs = df['High'].iloc[-6:-1].values
            recent_lows = df['Low'].iloc[-6:-1].values
            recent_volumes = df['Volume'].iloc[-6:-1].values if 'Volume' in df.columns else None
            if len(recent_closes) < 3:
                return True
            score = 0
            if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                higher_highs = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
                higher_lows = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
                if higher_highs and higher_lows:
                    score += 3
                elif higher_highs:
                    score += 2
                elif higher_lows:
                    score += 1
            if recent_volumes is not None and len(recent_volumes) >= 3:
                avg_volume = df['Volume'].iloc[-21:-1].mean() if 'Volume' in df.columns and len(
                    df) >= 20 else recent_volumes.mean()
                if avg_volume > 0:
                    latest_volume_ratio = recent_volumes[-1] / avg_volume
                    if latest_volume_ratio > 1.8:
                        score += 3
                    elif latest_volume_ratio > 1.4:
                        score += 2
                    elif latest_volume_ratio > 1.1:
                        score += 1
            if len(recent_closes) >= 3:
                pct_changes = [(recent_closes[i] - recent_closes[i - 1]) / recent_closes[i - 1] * 100 for i in
                               range(1, len(recent_closes))]
                latest_pct = pct_changes[-1] if pct_changes else 0
                if latest_pct > 1.5:
                    score += 3
                elif latest_pct > 0.8:
                    score += 2
                elif latest_pct > 0.3:
                    score += 1
            adx = data.get('ADX', 0)
            if adx >= 30:
                score += 2
            elif adx >= 25:
                score += 1
            elif adx < 20:
                score -= 1
            return score >= 3
        except Exception:
            return True

    def _momentum_is_sustainable_down(self, data):
        df = getattr(self, '_current_df', None)
        if df is None or len(df) < 15:
            return True
        try:
            recent_closes = df['Close'].iloc[-6:-1].values
            recent_highs = df['High'].iloc[-6:-1].values
            recent_lows = df['Low'].iloc[-6:-1].values
            recent_volumes = df['Volume'].iloc[-6:-1].values if 'Volume' in df.columns else None
            if len(recent_closes) < 3:
                return True
            score = 0
            if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                lower_highs = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
                lower_lows = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))
                if lower_highs and lower_lows:
                    score += 3
                elif lower_highs:
                    score += 2
                elif lower_lows:
                    score += 1
            if recent_volumes is not None and len(recent_volumes) >= 3:
                avg_volume = df['Volume'].iloc[-21:-1].mean() if 'Volume' in df.columns and len(
                    df) >= 20 else recent_volumes.mean()
                if avg_volume > 0:
                    latest_volume_ratio = recent_volumes[-1] / avg_volume
                    if latest_volume_ratio > 1.8:
                        score += 3
                    elif latest_volume_ratio > 1.4:
                        score += 2
                    elif latest_volume_ratio > 1.1:
                        score += 1
            if len(recent_closes) >= 3:
                pct_changes = [(recent_closes[i] - recent_closes[i - 1]) / recent_closes[i - 1] * 100 for i in
                               range(1, len(recent_closes))]
                latest_pct = pct_changes[-1] if pct_changes else 0
                if latest_pct < -1.5:
                    score += 3
                elif latest_pct < -0.8:
                    score += 2
                elif latest_pct < -0.3:
                    score += 1
            adx = data.get('ADX', 0)
            if adx >= 30:
                score += 2
            elif adx >= 25:
                score += 1
            elif adx < 20:
                score -= 1
            return score >= 3
        except Exception:
            return True

    def _daily_trend_is_up(self, data):
        if not getattr(self, 'daily_trend_filter_enabled', True):
            return True

        # Primary check: close > EMA_Daily (uses daily_ema_period, now 30d = 720h)
        above_daily = data.get('Above_Daily_50', None)
        if above_daily is None:
            df = getattr(self, '_current_df', None)
            if df is not None and 'Above_Daily_50' in df.columns and len(df) > 0:
                above_daily = df['Above_Daily_50'].iloc[-1]
            else:
                return True
        if above_daily is None or (isinstance(above_daily, float) and np.isnan(above_daily)):
            return True
        if bool(above_daily):
            return True  # Primary check passed

        # Secondary override: strong short-term trend confirmation
        # Allows early-recovery entries when EMA stack is fully ordered
        # and ADX is above the override threshold — even if the 30d EMA
        # hasn't turned yet. Catches the Jul 2025 SOL recovery scenario.
        ema_fast = data.get('EMA_Fast', 0)
        ema_mid = data.get('EMA_Mid', 0)
        ema_slow = data.get('EMA_Slow', 0)
        adx = data.get('ADX', 0)
        adx_min_ov = getattr(self, 'daily_trend_adx_override', 28)
        if ema_fast > ema_mid > ema_slow and adx >= adx_min_ov:
            return True  # Strong momentum override

        return False

    def _daily_trend_is_down(self, data):
        if not getattr(self, 'daily_trend_down_filter_enabled', True):
            return True
        above_daily = data.get('Above_Daily_50', None)
        if above_daily is None:
            df = getattr(self, '_current_df', None)
            if df is not None and 'Above_Daily_50' in df.columns and len(df) > 0:
                above_daily = df['Above_Daily_50'].iloc[-1]
            else:
                return True
        if above_daily is None or (isinstance(above_daily, float) and np.isnan(above_daily)):
            return True
        return not bool(above_daily)

    def _score_macd_momentum(self, data):
        macd = data.get('MACD', 0) or 0
        signal = data.get('MACD_Signal', 0) or 0
        histogram = data.get('MACD_Histogram', 0) or 0
        hist_rising = data.get('MACD_Histogram_Rising', False)
        score = 0;
        parts = []
        pts_line = getattr(self, 'macd_score_line_vs_signal', 7)
        if macd > signal:
            score += pts_line; parts.append(f"L>S+{pts_line}")
        elif signal != 0 and abs(macd - signal) / abs(signal) < 0.01:
            p = round(pts_line * 0.5);
            score += p;
            parts.append(f"L~S+{p}")
        else:
            parts.append("L<S+0")
        pts_dir = getattr(self, 'macd_score_histogram_direction', 7)
        if hist_rising:
            score += pts_dir; parts.append(f"H↑+{pts_dir}")
        else:
            parts.append("H↓+0")
        pts_zero = getattr(self, 'macd_score_zero_cross', 4)
        if macd > 0:
            score += pts_zero; parts.append(f"Z>0+{pts_zero}")
        else:
            parts.append("Z≤0+0")
        pts_val = getattr(self, 'macd_score_histogram_value', 3)
        close_price = data.get('Close', 0) if isinstance(data, dict) else 0
        hist_prev = data.get('MACD_Histogram_prev', None) if isinstance(data, dict) else None
        hist_pct = abs(histogram) / close_price * 100 if close_price > 0 else 0
        fresh_cross = (histogram > 0 and hist_prev is not None and hist_prev <= 0)
        hist_small = hist_pct < 0.25
        hist_large = hist_pct > 0.70
        if fresh_cross:
            bonus = pts_val + 6;
            score += bonus;
            parts.append(f"H_FRESH+{bonus}")
        elif histogram > 0 and hist_small:
            score += pts_val;
            parts.append(f"H_SMALL+{pts_val}")
        elif histogram > 0 and hist_large:
            penalty = max(0, pts_val - 6);
            score += penalty;
            parts.append(f"H_EXHAUST+{penalty}")
        elif histogram > 0:
            p = round(pts_val * 0.6);
            score += p;
            parts.append(f"H>0+{p}")
        elif histogram > -0.001:
            p = round(pts_val * 0.3);
            score += p;
            parts.append(f"H~0+{p}")
        else:
            parts.append("H<0+0")
        return score, ",".join(parts)

    def _score_macd_momentum_short(self, data):
        macd = data.get('MACD', 0) or 0
        signal = data.get('MACD_Signal', 0) or 0
        histogram = data.get('MACD_Histogram', 0) or 0
        hist_falling = not data.get('MACD_Histogram_Rising', False)
        score = 0;
        parts = []
        pts_line = getattr(self, 'macd_score_line_vs_signal', 7)
        if macd < signal:
            score += pts_line; parts.append(f"L<S+{pts_line}")
        elif signal != 0 and abs(macd - signal) / abs(signal) < 0.01:
            p = round(pts_line * 0.5);
            score += p;
            parts.append(f"L~S+{p}")
        else:
            parts.append("L>S+0")
        pts_dir = getattr(self, 'macd_score_histogram_direction', 7)
        if hist_falling:
            score += pts_dir; parts.append(f"H↓+{pts_dir}")
        else:
            parts.append("H↑+0")
        pts_zero = getattr(self, 'macd_score_zero_cross', 4)
        if macd < 0:
            score += pts_zero; parts.append(f"Z<0+{pts_zero}")
        else:
            parts.append("Z≥0+0")
        pts_val = getattr(self, 'macd_score_histogram_value', 3)
        close_price_s = data.get('Close', 0) if isinstance(data, dict) else 0
        hist_prev_s = data.get('MACD_Histogram_prev', None) if isinstance(data, dict) else None
        hist_pct_s = abs(histogram) / close_price_s * 100 if close_price_s > 0 else 0
        fresh_cross_s = (histogram < 0 and hist_prev_s is not None and hist_prev_s >= 0)
        hist_small_s = hist_pct_s < 0.25
        hist_large_s = hist_pct_s > 0.70
        if fresh_cross_s:
            bonus_s = pts_val + 6;
            score += bonus_s;
            parts.append(f"H_FRESH_S+{bonus_s}")
        elif histogram < 0 and hist_small_s:
            score += pts_val;
            parts.append(f"H_SMALL_S+{pts_val}")
        elif histogram < 0 and hist_large_s:
            penalty_s = max(0, pts_val - 6);
            score += penalty_s;
            parts.append(f"H_EXHAUST_S+{penalty_s}")
        elif histogram < 0:
            p = round(pts_val * 0.6);
            score += p;
            parts.append(f"H<0+{p}")
        elif histogram < 0.001:
            p = round(pts_val * 0.3);
            score += p;
            parts.append(f"H~0+{p}")
        else:
            parts.append("H>0+0")
        return score, ",".join(parts)

    def _calculate_quality_score(self, data):
        """
        v9.5.2 FIX: Score range now spreads ~16-95 instead of 68-81.
        Changes from v9.5:
        - EMA partial credit floors reduced (fast>slow alone: 60%→25%, partial: 40%→10%)
        - Added intermediate EMA tier: all three ordered but close below fast = 50%
        - RSI overbought zone (>75) now scores 0 (was missing, allowing overbought entries)
        - RSI 70-75 zone now scores 25% (was missing)
        - Volume below 0.8x now near-zero (was 30-60%)
        - ADX below 18 now scores 0, 18-22 scores 15% (new tier)
        - ADX extended (>40) penalty increased: 50%→40%
        """
        if self.strategy_state != StrategyState.SEEKING_ENTRY:
            raise RuntimeError("_calculate_quality_score() called while IN_TRADE")

        component_scores = {}
        breakdown_parts = []

        close = data.get('Close', 0)
        ema_fast = data.get('EMA_Fast', 0)
        ema_mid = data.get('EMA_Mid', 0)
        ema_slow = data.get('EMA_Slow', 0)
        adx = data.get('ADX', 0)
        rsi = data.get('RSI', 50)
        volume_ratio = data.get('Volume_Ratio', 1.0)
        price_pct = data.get('Price_Percentile_20bar', 50)

        # ── EMA ALIGNMENT ────────────────────────────────────────────────
        tolerance = getattr(self, 'ema_near_tolerance', 0.005) or 0.005
        weight_ema = getattr(self, 'weight_ema', 20)

        if close > ema_fast > ema_mid > ema_slow:
            ema_score = weight_ema
            breakdown_parts.append(f"EMA={ema_score}/20(Perfect)")
        elif (self._near_or_above(ema_fast, ema_mid, tolerance) and
              self._near_or_above(ema_mid, ema_slow, tolerance) and
              close > ema_fast * (1 - tolerance)):
            ema_score = round(weight_ema * 0.75)
            breakdown_parts.append(f"EMA={ema_score}/20(Near)")
        elif ema_fast > ema_mid > ema_slow:
            # Trend structure intact but close below fast EMA — bad timing
            ema_score = round(weight_ema * 0.50)
            breakdown_parts.append(f"EMA={ema_score}/20(Ordered,BadTiming)")
        elif ema_fast > ema_slow:
            # Only fast>slow — weak signal, was 60% (now 25%)
            ema_score = round(weight_ema * 0.25)
            breakdown_parts.append(f"EMA={ema_score}/20(Fast>Slow_Only)")
        elif ema_fast > ema_mid:
            # Was 40%, now 10% — very partial alignment
            ema_score = round(weight_ema * 0.10)
            breakdown_parts.append(f"EMA={ema_score}/20(Partial)")
        else:
            ema_score = 0
            breakdown_parts.append("EMA=0/20")

        component_scores['ema'] = ema_score

        # ── ADX STRENGTH ─────────────────────────────────────────────────
        weight_adx = getattr(self, 'weight_adx', 20)

        if adx < 18:
            adx_score = 0;                          adx_label = "NoTrend"
        elif adx < 22:
            adx_score = round(weight_adx * 0.15);   adx_label = "VeryWeak"
        elif adx < 26:
            adx_score = round(weight_adx * 0.40);   adx_label = "Forming"
        elif adx < 30:
            adx_score = round(weight_adx * 0.70);   adx_label = "Good"
        elif adx < 35:
            adx_score = round(weight_adx * 0.90);   adx_label = "Strong"
        elif adx < 40:
            adx_score = round(weight_adx * 0.75);   adx_label = "PeakCaution"
        else:
            adx_score = round(weight_adx * 0.40);   adx_label = "Extended"

        breakdown_parts.append(f"ADX={adx_score}/{weight_adx}({adx:.1f},{adx_label})")
        component_scores['adx'] = adx_score

        # ── MACD MOMENTUM ────────────────────────────────────────────────
        weight_macd = getattr(self, 'weight_macd', 25)
        macd_score, macd_breakdown = self._score_macd_momentum(data)
        if weight_macd != 25:
            macd_score = round(macd_score * weight_macd / 25)
        component_scores['macd'] = macd_score
        breakdown_parts.append(f"MACD={macd_score}/{weight_macd}({macd_breakdown})")

        # ── RSI ZONE ─────────────────────────────────────────────────────
        weight_rsi = getattr(self, 'weight_rsi', 20)

        if 60 <= rsi <= 67:
            rsi_score = weight_rsi;               rsi_label = "PrimeLong"
        elif 55 <= rsi < 60:
            rsi_score = round(weight_rsi * 0.80); rsi_label = "StrongLong"
        elif 67 < rsi <= 70:
            rsi_score = round(weight_rsi * 0.55); rsi_label = "NearPeak"
        elif 70 < rsi <= 75:
            rsi_score = round(weight_rsi * 0.25); rsi_label = "Overbought"
        elif rsi > 75:
            rsi_score = 0;                        rsi_label = "TooHigh"
        elif 48 <= rsi < 55:
            rsi_score = round(weight_rsi * 0.45); rsi_label = "EarlyLong"
        elif 42 <= rsi < 48:
            rsi_score = round(weight_rsi * 0.20); rsi_label = "Borderline"
        else:
            rsi_score = 0;                        rsi_label = "OutOfRange"

        _rdb = getattr(self, 'rsi_direction_bars', 3)
        _rdm = getattr(self, 'rsi_direction_min_move', 1.0)
        _dfr = getattr(self, '_current_df', None)
        if _dfr is not None and 'RSI' in _dfr.columns and len(_dfr) > _rdb + 1:
            _delta = float(_dfr['RSI'].iloc[-1]) - float(_dfr['RSI'].iloc[-(_rdb + 1)])
            if _delta >= _rdm:
                rsi_score = min(rsi_score + 5, weight_rsi); rsi_label += "+Rising"
            elif _delta <= -_rdm:
                rsi_score = max(rsi_score - 5, 0);           rsi_label += "-Falling"

        component_scores['rsi'] = rsi_score
        breakdown_parts.append(f"RSI={rsi_score}/{weight_rsi}({rsi:.1f},{rsi_label})")

        # ── VOLUME ───────────────────────────────────────────────────────
        weight_volume = getattr(self, 'weight_volume', 12)

        if volume_ratio >= 2.0:
            volume_score = weight_volume
        elif volume_ratio >= 1.5:
            volume_score = round(weight_volume * 0.85)
        elif volume_ratio >= 1.2:
            volume_score = round(weight_volume * 0.70)
        elif volume_ratio >= 1.0:
            volume_score = round(weight_volume * 0.50)
        elif volume_ratio >= 0.8:
            volume_score = round(weight_volume * 0.30)
        elif volume_ratio >= 0.6:
            volume_score = round(weight_volume * 0.15)
        else:
            volume_score = 0

        component_scores['volume'] = volume_score
        breakdown_parts.append(f"Vol={volume_score}/{weight_volume}({volume_ratio:.2f}x)")

        # ── PRICE PERCENTILE ADJUSTMENT ──────────────────────────────────
        if price_pct < 20:
            adj, txt = 15, "EarlyEntry+15"
        elif price_pct < 40:
            adj, txt = 8, "EarlyEntry+8"
        elif price_pct < 60:
            adj, txt = 0, "MidRange+0"
        elif price_pct < 80:
            adj, txt = -5, "LateEntry-5"
        else:
            adj, txt = -15, "PeakEntry-15"
        breakdown_parts.append(txt)

        # ── TREND AGE PENALTY (v9.5) ──────────────────────────────────────
        trend_penalty = self._get_trend_age_penalty(data, direction='long')
        if trend_penalty > 0:
            adj -= trend_penalty
            breakdown_parts.append(f"TrendAge-{trend_penalty}")

        total_score = max(0, min(sum(component_scores.values()) + adj, 100))
        return int(total_score), component_scores, " | ".join(breakdown_parts)

    def _calculate_quality_score_short(self, data):
        """
        v9.5.2 FIX: Mirror of _calculate_quality_score — same floor reductions applied.
        Score range now spreads ~16-95 instead of compressed 68-81.
        """
        if self.strategy_state != StrategyState.SEEKING_ENTRY:
            raise RuntimeError("_calculate_quality_score_short() called while IN_TRADE")

        component_scores = {}
        breakdown_parts = []

        close = data.get('Close', 0)
        ema_fast = data.get('EMA_Fast', 0)
        ema_mid = data.get('EMA_Mid', 0)
        ema_slow = data.get('EMA_Slow', 0)
        adx = data.get('ADX', 0)
        rsi = data.get('RSI', 50)
        volume_ratio = data.get('Volume_Ratio', 1.0)
        price_pct = data.get('Price_Percentile_20bar', 50)

        # ── EMA ALIGNMENT (BEARISH) ──────────────────────────────────────
        tolerance = getattr(self, 'ema_near_tolerance', 0.005) or 0.005
        weight_ema = getattr(self, 'weight_ema', 20)

        if close < ema_fast < ema_mid < ema_slow:
            ema_score = weight_ema
            breakdown_parts.append(f"EMA={ema_score}/20(PerfectBearish)")
        elif (self._near_or_above(ema_mid, ema_fast, tolerance) and
              self._near_or_above(ema_slow, ema_mid, tolerance) and
              close < ema_fast * (1 + tolerance)):
            ema_score = round(weight_ema * 0.75)
            breakdown_parts.append(f"EMA={ema_score}/20(NearBearish)")
        elif ema_fast < ema_mid < ema_slow:
            # All three ordered bearish but close above fast — bad timing
            ema_score = round(weight_ema * 0.50)
            breakdown_parts.append(f"EMA={ema_score}/20(Ordered,BadTiming)")
        elif ema_fast < ema_slow:
            ema_score = round(weight_ema * 0.25)
            breakdown_parts.append(f"EMA={ema_score}/20(Fast<Slow_Only)")
        elif ema_fast < ema_mid:
            ema_score = round(weight_ema * 0.10)
            breakdown_parts.append(f"EMA={ema_score}/20(PartialBearish)")
        else:
            ema_score = 0
            breakdown_parts.append("EMA=0/20")

        component_scores['ema'] = ema_score

        # ── ADX STRENGTH ─────────────────────────────────────────────────
        weight_adx = getattr(self, 'weight_adx', 20)

        if adx < 18:
            adx_score = 0;                          adx_label = "NoTrend"
        elif adx < 22:
            adx_score = round(weight_adx * 0.15);   adx_label = "VeryWeak"
        elif adx < 26:
            adx_score = round(weight_adx * 0.40);   adx_label = "Forming"
        elif adx < 30:
            adx_score = round(weight_adx * 0.70);   adx_label = "Good"
        elif adx < 35:
            adx_score = round(weight_adx * 0.90);   adx_label = "Strong"
        elif adx < 40:
            adx_score = round(weight_adx * 0.75);   adx_label = "PeakCaution"
        else:
            adx_score = round(weight_adx * 0.40);   adx_label = "Extended"

        breakdown_parts.append(f"ADX={adx_score}/{weight_adx}({adx:.1f},{adx_label})")
        component_scores['adx'] = adx_score

        # ── MACD MOMENTUM (SHORT) ────────────────────────────────────────
        weight_macd = getattr(self, 'weight_macd', 25)
        macd_score, macd_breakdown = self._score_macd_momentum_short(data)
        if weight_macd != 25:
            macd_score = round(macd_score * weight_macd / 25)
        component_scores['macd'] = macd_score
        breakdown_parts.append(f"MACD={macd_score}/{weight_macd}({macd_breakdown})")

        # ── RSI ZONE (SHORT) ─────────────────────────────────────────────
        weight_rsi = getattr(self, 'weight_rsi', 20)

        if 38 <= rsi <= 45:
            rsi_score = weight_rsi;               rsi_label = "PrimeShort"
        elif 45 < rsi <= 52:
            rsi_score = round(weight_rsi * 0.75); rsi_label = "ShortMomentum"
        elif 52 < rsi <= 57:
            rsi_score = round(weight_rsi * 0.45); rsi_label = "WeakeningShort"
        elif 34 <= rsi < 38:
            rsi_score = round(weight_rsi * 0.45); rsi_label = "ShortEarly"
        elif 30 <= rsi < 34:
            rsi_score = round(weight_rsi * 0.20); rsi_label = "Borderline"
        elif rsi < 30:
            rsi_score = 0;                        rsi_label = "Oversold"
        else:
            rsi_score = 0;                        rsi_label = "OutOfRange"

        _rdb = getattr(self, 'rsi_direction_bars', 3)
        _rdm = getattr(self, 'rsi_direction_min_move', 1.0)
        _dfr = getattr(self, '_current_df', None)
        if _dfr is not None and 'RSI' in _dfr.columns and len(_dfr) > _rdb + 1:
            _fall = float(_dfr['RSI'].iloc[-(_rdb + 1)]) - float(_dfr['RSI'].iloc[-1])
            if _fall >= _rdm:
                rsi_score = min(rsi_score + 5, weight_rsi); rsi_label += "+Falling"
            elif _fall <= -_rdm:
                rsi_score = max(rsi_score - 5, 0);           rsi_label += "-Rising"

        component_scores['rsi'] = rsi_score
        breakdown_parts.append(f"RSI={rsi_score}/{weight_rsi}({rsi:.1f},{rsi_label})")

        # ── VOLUME ───────────────────────────────────────────────────────
        weight_volume = getattr(self, 'weight_volume', 12)

        if volume_ratio >= 2.0:
            volume_score = weight_volume
        elif volume_ratio >= 1.5:
            volume_score = round(weight_volume * 0.85)
        elif volume_ratio >= 1.2:
            volume_score = round(weight_volume * 0.70)
        elif volume_ratio >= 1.0:
            volume_score = round(weight_volume * 0.50)
        elif volume_ratio >= 0.8:
            volume_score = round(weight_volume * 0.30)
        elif volume_ratio >= 0.6:
            volume_score = round(weight_volume * 0.15)
        else:
            volume_score = 0

        component_scores['volume'] = volume_score
        breakdown_parts.append(f"Vol={volume_score}/{weight_volume}({volume_ratio:.2f}x)")

        # ── PRICE PERCENTILE ADJUSTMENT (SHORT — inverted) ───────────────
        if price_pct > 80:
            adj, txt = 15, "LateEntryShort+15"
        elif price_pct > 60:
            adj, txt = 8, "LateEntryShort+8"
        elif price_pct > 40:
            adj, txt = 0, "MidRange+0"
        elif price_pct > 20:
            adj, txt = -5, "EarlyEntry-5"
        else:
            adj, txt = -15, "EarlyEntry-15"
        breakdown_parts.append(txt)

        # ── TREND AGE PENALTY (v9.5) ──────────────────────────────────────
        trend_penalty = self._get_trend_age_penalty(data, direction='short')
        if trend_penalty > 0:
            adj -= trend_penalty
            breakdown_parts.append(f"TrendAge-{trend_penalty}")

        total_score = max(0, min(sum(component_scores.values()) + adj, 100))
        return int(total_score), component_scores, " | ".join(breakdown_parts)

    def _check_tier1_filters(self, data):
        ema_fast = data.get('EMA_Fast', 0)
        ema_mid = data.get('EMA_Mid', 0)
        ema_slow = data.get('EMA_Slow', 0)
        adx = data.get('ADX', 0);
        rsi = data.get('RSI', 50)
        momentum = data.get('Momentum', 0)
        macd = data.get('MACD', 0);
        macd_signal = data.get('MACD_Signal', 0)
        if hasattr(self, 'trading_app') and hasattr(self.trading_app, 'get_volume_ratio'):
            volume_ratio = self.trading_app.get_volume_ratio(df=None, current_data=data, default=1.0)
        else:
            volume_ratio = float(data.get('Volume_Ratio', 1.0))
            volume_ratio = max(0.01, min(10.0, volume_ratio))

        # ── EMA TREND ─────────────────────────────────────────────────
        if not (ema_fast > ema_slow):
            return "tier1_ema_not_trending"

        # ── ADX HARD MIN ───────────────────────────────────────────────
        adx_hard_min = getattr(self, 'tier1_adx_hard_min', 25)
        if adx < adx_hard_min:
            return f"tier1_adx_too_weak_{adx:.1f}_need_{adx_hard_min}"

        # ── RSI RANGE ──────────────────────────────────────────────────
        t1_rsi_min = getattr(self, 'tier1_rsi_min', 40)
        t1_rsi_max = getattr(self, 'tier1_rsi_max', 68)
        if not (t1_rsi_min <= rsi <= t1_rsi_max):
            return f"tier1_rsi_out_{rsi:.1f}_need_{t1_rsi_min}-{t1_rsi_max}"

        # ── VOLUME — relax to 0.2 on strong confirmed trend ────────────
        t1_vol_min = getattr(self, 'tier1_volume_min', 0.8)
        if (ema_fast > ema_mid > ema_slow and adx >= 25):
            t1_vol_min = min(t1_vol_min, 0.2)
        if volume_ratio < t1_vol_min:
            return f"tier1_volume_low_{volume_ratio:.2f}_need_{t1_vol_min}"

        # ── MOMENTUM ───────────────────────────────────────────────────
        t1_mom_min = getattr(self, 'tier1_momentum_min', 0.02)
        if momentum < t1_mom_min:
            return f"tier1_momentum_weak_{momentum:.2f}_need_{t1_mom_min}"

        # ── PRICE EXTENSION FROM EMA_FAST ─────────────────────────────
        price_ext_max = getattr(self, 'tier1_price_ema_max_pct', 1.5)
        if price_ext_max is None: price_ext_max = 1.5
        ema_fast_val = data.get('EMA_Fast', 0);
        close_val = data.get('Close', 0)
        if ema_fast_val > 0 and close_val > 0:
            ext_pct = (close_val - ema_fast_val) / ema_fast_val * 100
            if ext_pct > price_ext_max:
                return f"tier1_price_too_extended_{ext_pct:.1f}pct_above_ema"

        # ── MACD GATE ─────────────────────────────────────────────────
        if getattr(self, 'tier1_macd_gate', True):
            if macd <= macd_signal:
                return f"tier1_macd_below_signal_{macd:.3f}_vs_{macd_signal:.3f}"

        # ── DMI SPREAD ────────────────────────────────────────────────
        dmi_spread = getattr(self, 'dmi_spread_min_long', 0.0)
        if dmi_spread > 0:
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'DMP' in _df.columns and len(_df) > 0:
                try:
                    dmp = float(_df['DMP'].iloc[-1]);
                    dmm = float(_df['DMM'].iloc[-1])
                    if (dmp - dmm) < dmi_spread:
                        return f"tier1_dmi_spread_{dmp - dmm:.1f}<{dmi_spread}"
                except Exception:
                    pass

        # ── MACD HISTOGRAM RISING ─────────────────────────────────────
        if getattr(self, 'macd_hist_rising_bars', 0) >= 2:
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'MACD_Histogram' in _df.columns and len(_df) > 2:
                h = _df['MACD_Histogram']
                if not (float(h.iloc[-1]) > float(h.iloc[-2]) > float(h.iloc[-3])):
                    return "tier1_macd_hist_not_rising_2bars"

        # ── BB EXPANSION ──────────────────────────────────────────────
        if getattr(self, 'bb_expand_required', False):
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'BB_Width' in _df.columns and len(_df) > 40:
                if float(_df['BB_Width'].iloc[-1]) < float(_df['BB_Width'].iloc[-40:].mean()):
                    return "tier1_bb_squeeze"

        # ── TIME FILTER ───────────────────────────────────────────────
        if getattr(self, 'time_filter_enabled', False):
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'timestamp' in _df.columns and len(_df) > 0:
                import pandas as _pd
                try:
                    hour = _pd.to_datetime(_df['timestamp'].iloc[-1]).hour
                    if not (getattr(self, 'time_filter_start_utc', 6) <= hour <= getattr(self, 'time_filter_end_utc',
                                                                                         23)):
                        return f"tier1_dead_hour_{hour}utc"
                except Exception:
                    pass

        # ── ADX RISING SLOPE — now parameterized ──────────────────────
        # Default 0.1 preserves original behaviour.
        # Set adx_slope_min to -0.3 to allow flat/slightly-declining ADX.
        adx_now = data.get('ADX', 0)
        adx_prev_val = data.get('ADX_prev', adx_now)
        adx_slope_min = getattr(self, 'adx_slope_min', 0.1)
        if (adx_now - adx_prev_val) < adx_slope_min:
            return f"v9_adx_not_rising_slope_{adx_now - adx_prev_val:.3f}_need_{adx_slope_min}"

        # ── PULLBACK ZONE — now parameterized ─────────────────────────
        # Defaults preserve original behaviour (-2.5 / +1.5).
        # Widen to pullback_zone_lower=-4.0, pullback_zone_upper=3.0
        # to significantly increase entry frequency.
        ema_fast_val2 = data.get('EMA_Fast', 0);
        close_val2 = data.get('Close', 0)
        if ema_fast_val2 > 0 and close_val2 > 0:
            dist_pct = (close_val2 - ema_fast_val2) / ema_fast_val2 * 100
            pz_lower = getattr(self, 'pullback_zone_lower_pct', -2.5)
            pz_upper = getattr(self, 'pullback_zone_upper_pct', 1.5)
            if not (pz_lower <= dist_pct <= pz_upper):
                return f"v9_not_in_pullback_zone_dist_{dist_pct:.2f}pct_need_{pz_lower}/{pz_upper}"

        # ── DAILY TREND ───────────────────────────────────────────────
        if not self._daily_trend_is_up(data):
            return "tier1_daily_trend_not_up_price_below_50day_ema"
        return "pass"

    def _check_tier1_filters_short(self, data):
        ema_fast = data.get('EMA_Fast', 0);
        ema_slow = data.get('EMA_Slow', 0)
        adx = data.get('ADX', 0);
        rsi = data.get('RSI', 50)
        momentum = data.get('Momentum', 0)
        macd = data.get('MACD', 0);
        macd_signal = data.get('MACD_Signal', 0)
        if hasattr(self, 'trading_app') and hasattr(self.trading_app, 'get_volume_ratio'):
            volume_ratio = self.trading_app.get_volume_ratio(df=None, current_data=data, default=1.0)
        else:
            volume_ratio = float(data.get('Volume_Ratio', 1.0))
            volume_ratio = max(0.01, min(10.0, volume_ratio))

        if not (ema_fast < ema_slow):
            return "short_tier1_ema_not_trending_bearish"

        # v9.5 FIX #4: Stricter short ADX minimum
        adx_hard_min = getattr(self, 'short_tier1_adx_hard_min', 30)
        if adx < adx_hard_min:
            return f"short_tier1_adx_too_weak_{adx:.1f}_need_{adx_hard_min}"

        t1_rsi_min = getattr(self, 'short_tier1_rsi_min', 34)
        t1_rsi_max = getattr(self, 'short_tier1_rsi_max', 54)
        if not (t1_rsi_min <= rsi <= t1_rsi_max):
            return f"short_tier1_rsi_out_{rsi:.1f}_need_{t1_rsi_min}-{t1_rsi_max}"

        # v9.5 FIX #4: Stricter short volume minimum
        t1_vol_min = getattr(self, 'short_tier1_volume_min', 1.3)
        if volume_ratio < t1_vol_min:
            return f"short_tier1_volume_low_{volume_ratio:.2f}_need_{t1_vol_min}"

        # v9.5 FIX #4: Stricter short momentum
        t1_mom_min = getattr(self, 'short_tier1_momentum_min', 0.05)
        if momentum > -t1_mom_min:
            return f"short_tier1_momentum_not_bearish_{momentum:.2f}_need_negative"

        if getattr(self, 'short_tier1_macd_gate', True):
            if macd >= macd_signal:
                return f"short_tier1_macd_above_signal_{macd:.3f}_vs_{macd_signal:.3f}"

        dmi_spread = getattr(self, 'dmi_spread_min_short', 0.0)
        if dmi_spread > 0:
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'DMM' in _df.columns and len(_df) > 0:
                try:
                    dmp = float(_df['DMP'].iloc[-1]);
                    dmm = float(_df['DMM'].iloc[-1])
                    if (dmm - dmp) < dmi_spread:
                        return f"short_tier1_dmi_spread_{dmm - dmp:.1f}<{dmi_spread}"
                except Exception:
                    pass

        if getattr(self, 'macd_hist_rising_bars', 0) >= 2:
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'MACD_Histogram' in _df.columns and len(_df) > 2:
                h = _df['MACD_Histogram']
                if not (float(h.iloc[-1]) < float(h.iloc[-2]) < float(h.iloc[-3])):
                    return "short_tier1_macd_hist_not_falling_2bars"

        if getattr(self, 'bb_expand_required', False):
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'BB_Width' in _df.columns and len(_df) > 40:
                if float(_df['BB_Width'].iloc[-1]) < float(_df['BB_Width'].iloc[-40:].mean()):
                    return "short_tier1_bb_squeeze"

        if getattr(self, 'time_filter_enabled', False):
            _df = getattr(self, '_current_df', None)
            if _df is not None and 'timestamp' in _df.columns and len(_df) > 0:
                import pandas as _pd2
                try:
                    hour = _pd2.to_datetime(_df['timestamp'].iloc[-1]).hour
                    if not (getattr(self, 'time_filter_start_utc', 6) <= hour <= getattr(self, 'time_filter_end_utc',
                                                                                         23)):
                        return f"short_tier1_dead_hour_{hour}utc"
                except Exception:
                    pass

        adx_now_s = data.get('ADX', 0)
        adx_prev_s = data.get('ADX_prev', adx_now_s)
        if (adx_now_s - adx_prev_s) < 0.1:
            return f"v9_short_adx_not_rising_slope_{adx_now_s - adx_prev_s:.3f}"

        ema_fast_s = data.get('EMA_Fast', 0);
        close_s = data.get('Close', 0)
        if ema_fast_s > 0 and close_s > 0:
            dist_s = (close_s - ema_fast_s) / ema_fast_s * 100
            if not (-2.5 <= dist_s <= 0.5):
                return f"v9_short_not_in_bounce_zone_dist_{dist_s:.2f}pct"

        if not self._daily_trend_is_down(data):
            return "short_tier1_daily_trend_not_down_price_above_50day_ema"

        # v9.5 FIX #4: Require bearish price structure (lower highs + lower lows)
        if not self._has_bearish_price_structure(data):
            return "v94_short_no_bearish_price_structure"

        return "pass"

    def _validate_tier2_conditions(self, data):
        if getattr(self, 'regime_filter_enabled', True) and data.get('Ranging', False):
            return False
        rsi = data.get('RSI', 50)
        if rsi > 90: return False
        macd_hist = data.get('MACD_Histogram', 0)
        if macd_hist <= 0: return False
        bars_since = self.bar_count - self.last_trade_bar
        cooldown_bars = getattr(self, 'cooldown_after_loss_bars', 12)
        if (bars_since < cooldown_bars and self.trade_history and self.trade_history[-1]['profit'] < 0):
            return False
        return True

    def _validate_tier2_conditions_short(self, data):
        if getattr(self, 'regime_filter_enabled', True) and data.get('Ranging', False):
            return False
        rsi = data.get('RSI', 50)
        if rsi < 30: return False
        macd_hist = data.get('MACD_Histogram', 0)
        if macd_hist >= 0: return False
        bars_since = self.bar_count - self.last_trade_bar
        cooldown_bars = getattr(self, 'cooldown_after_loss_bars', 12)
        if (bars_since < cooldown_bars and self.trade_history and self.trade_history[-1]['profit'] < 0):
            return False
        return True

    def _calculate_dynamic_fuzzy_threshold(self):
        safety_config = {
            'absolute_minimum': getattr(self, 'fuzzy_absolute_min', 45),
            'absolute_maximum': getattr(self, 'fuzzy_absolute_max', 65),
            'default_fuzzy': 75 * (1 - getattr(self, 'fuzzy_default_margin_pct', 10) / 100),
            'min_confidence': getattr(self, 'fuzzy_min_confidence', 0.6),
            'max_adjustment_pct': getattr(self, 'fuzzy_max_adjustment_pct', 15),
            'min_samples': getattr(self, 'fuzzy_min_samples', 5),
            'learning_rate': getattr(self, 'fuzzy_learning_rate', 0.3),
            'conservative_mode': getattr(self, 'fuzzy_conservative_start', True),
            'outlier_rejection': 2.5,
        }
        default_cfg = {
            'fuzzy_lower': safety_config['default_fuzzy'],
            'fuzzy_margin_pct': getattr(self, 'fuzzy_default_margin_pct', 10),
            'confidence': 0.5, 'sample_size': 0, 'using_default': True, 'safety_status': 'default'
        }
        if not hasattr(self, 'near_miss_trades') or len(self.near_miss_trades) < safety_config['min_samples']:
            return default_cfg
        profitable = [t for t in self.near_miss_trades
                      if t.get('would_have_been_profitable', False) and
                      safety_config['absolute_minimum'] <= t['score'] <= 75]
        if len(profitable) < safety_config['min_samples']:
            return default_cfg
        scores = [t['score'] for t in profitable]
        mean_s = sum(scores) / len(scores)
        std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5
        filtered = [s for s in scores if abs(s - mean_s) <= safety_config['outlier_rejection'] * std_s]
        if len(filtered) < safety_config['min_samples']:
            return {**default_cfg, 'reason': 'too_many_outliers'}
        fm = sum(filtered) / len(filtered)
        fs = (sum((s - fm) ** 2 for s in filtered) / len(filtered)) ** 0.5
        mult = 1.5 if safety_config['conservative_mode'] else 2.0
        candidate = fm - (mult * fs)
        if candidate < safety_config['absolute_minimum']:
            fuzzy_lower, status = safety_config['absolute_minimum'], 'absolute_min_enforced'
        elif candidate > safety_config['absolute_maximum']:
            fuzzy_lower, status = safety_config['absolute_maximum'], 'absolute_max_enforced'
        else:
            fuzzy_lower, status = candidate, 'normal'
        default_lower = safety_config['default_fuzzy']
        max_dev = default_lower * (safety_config['max_adjustment_pct'] / 100)
        if abs(fuzzy_lower - default_lower) > max_dev:
            fuzzy_lower = (default_lower + max_dev if fuzzy_lower > default_lower else default_lower - max_dev)
            status = 'deviation_capped'
        sample_conf = min(1.0, len(filtered) / 50)
        consistency = 1.0 - min(1.0, fs / 8)
        raw_conf = sample_conf * 0.5 + consistency * 0.5
        if self._previous_fuzzy_lower is not None:
            fuzzy_lower = (self._previous_fuzzy_lower * (1 - safety_config['learning_rate']) +
                           fuzzy_lower * safety_config['learning_rate'])
        self._previous_fuzzy_lower = fuzzy_lower
        confidence = raw_conf * 0.8 if status != 'normal' else raw_conf
        if confidence < safety_config['min_confidence']:
            return {**default_cfg, 'reason': 'low_confidence',
                    'candidate_lower': round(fuzzy_lower, 1), 'raw_confidence': round(raw_conf, 2)}
        return {
            'fuzzy_lower': round(fuzzy_lower, 1),
            'fuzzy_margin_pct': round(((75 - fuzzy_lower) / 75) * 100, 1),
            'avg_score': round(fm, 1), 'std_dev': round(fs, 1),
            'min_score': round(min(filtered), 1), 'max_score': round(max(filtered), 1),
            'confidence': round(confidence, 2), 'raw_confidence': round(raw_conf, 2),
            'sample_size': len(filtered), 'total_samples': len(scores),
            'safety_status': status, 'using_default': False,
            'deviation_from_default': round(fuzzy_lower - default_lower, 1),
        }

    def _track_near_miss(self, score, data, rejected_reason, market_context=None):
        if not hasattr(self, 'near_miss_trades'): self.near_miss_trades = []
        atr = data.get('ATR', 0)
        atr_pct = (atr / data.get('Close', 1)) * 100 if data.get('Close', 0) > 0 else 0
        entry = {
            'timestamp': datetime.now(timezone.utc), 'score': score,
            'rejected_reason': rejected_reason, 'would_have_been_profitable': None,
            'adx': data.get('ADX', 0), 'rsi': data.get('RSI', 50),
            'macd': data.get('MACD', 0), 'macd_signal': data.get('MACD_Signal', 0),
            'ema_alignment': data.get('EMA_Fast', 0) > data.get('EMA_Slow', 0),
            'price': data.get('Close', 0), 'atr_percent': round(atr_pct, 2),
            'volume_ratio': data.get('Volume_Ratio', 1.0),
            'market_regime': self.current_regime,
            'price_percentile': data.get('Price_Percentile_20bar', 50),
        }
        if market_context: entry.update(market_context)
        self.near_miss_trades.append(entry)
        if len(self.near_miss_trades) > 200:
            self.near_miss_trades = self.near_miss_trades[-200:]

    def _validate_direction(self, data, direction_hint=None):
        direction = getattr(self, 'trade_direction', 'both').lower()
        effective_direction = direction_hint if direction_hint else direction

        cached = self._last_direction_check
        if (cached['bar'] == self.bar_count and cached.get('direction') == effective_direction):
            return (cached['result'], cached['reason'], cached['action'])

        close = data.get('Close', 0);
        ema_fast = data.get('EMA_Fast', 0)
        ema_slow = data.get('EMA_Slow', 0);
        ema_mid = data.get('EMA_Mid', 0)
        macd = data.get('MACD', 0);
        macd_signal = data.get('MACD_Signal', 0)
        macd_hist = data.get('MACD_Histogram', 0)
        rsi = data.get('RSI', 50);
        volume_ratio = data.get('Volume_Ratio', 1.0)
        df = getattr(self, '_current_df', None)

        def _cache(res, reason, action):
            self._last_direction_check = {
                'bar': self.bar_count, 'direction': effective_direction,
                'result': res, 'reason': reason, 'action': action,
            }
            return (res, reason, action)

        if effective_direction == 'long':
            ema200 = data.get('EMA_200', 0)

            # ── Non-negotiable structural guard: never go long against a
            #    fully bearish EMA stack. Everything else below is scored,
            #    not hard-gated. ──────────────────────────────────────────
            if ema_fast < ema_mid < ema_slow:
                return _cache(False, f"LONG: Full bearish EMA stack (hard block)", "hold")

            ema_mid_d = data.get('EMA_Mid', 0)
            ema_slow_d = data.get('EMA_Slow', 0)
            adx_d = data.get('ADX', 0)
            strong_trend_d = (ema_fast > ema_mid_d > ema_slow_d and adx_d >= 25)
            vol_gate = 0.2 if strong_trend_d else 0.8

            adx_now_l = data.get('ADX', 0)
            adx_prev_l = data.get('ADX_prev', adx_now_l)
            adx_slope_min_l = getattr(self, 'adx_slope_min', 0.1)

            pz_lower_l = getattr(self, 'pullback_zone_lower_pct', -2.5)
            pz_upper_l = getattr(self, 'pullback_zone_upper_pct', 1.5)
            dist_pct_l = (close - ema_fast) / ema_fast * 100 if ema_fast > 0 else 0

            pattern_ok = True
            if df is not None and len(df) > 10:
                recent_highs = df['High'].iloc[-5:].values
                recent_lows = df['Low'].iloc[-5:].values
                if len(recent_highs) >= 3 and recent_highs[-1] < recent_highs[-2] < recent_highs[-3]:
                    pattern_ok = False
                if len(recent_lows) >= 3 and recent_lows[-1] < recent_lows[-2] < recent_lows[-3]:
                    pattern_ok = False

            checks = [
                ("EMA200_macro_trend", not (ema200 > 0 and close < ema200)),
                ("EMA_fast_gt_slow", ema_fast > ema_slow),
                ("price_above_fastEMA", close > ema_fast),
                ("MACD_not_bearish", not (macd < macd_signal and macd_hist < 0)),
                ("RSI_above_40", rsi >= 40),
                ("volume_gate", not (volume_ratio < vol_gate and close > ema_fast)),
                ("no_bearish_pattern", pattern_ok),
                ("ADX_slope", (adx_now_l - adx_prev_l) >= adx_slope_min_l),
                ("pullback_zone", pz_lower_l <= dist_pct_l <= pz_upper_l),
            ]
            passed = sum(1 for _, ok in checks if ok)
            total = len(checks)
            confluence = passed / total
            min_confluence = getattr(self, 'direction_confluence_min', 0.75)

            if confluence >= min_confluence:
                failed = [name for name, ok in checks if not ok]
                note = f" (soft fail: {', '.join(failed)})" if failed else ""
                return _cache(True, f"LONG: Valid bullish setup ({passed}/{total} confluence){note}", "buy")

            failed = [name for name, ok in checks if not ok]
            return _cache(False,
                          f"LONG: confluence {passed}/{total} ({confluence:.0%}) < min {min_confluence:.0%} "
                          f"| failed: {', '.join(failed)}",
                          "hold")

        elif effective_direction == 'short':
            ema200 = data.get('EMA_200', 0)

            # ── Non-negotiable structural guard: never go short against a
            #    fully bullish EMA stack. Everything else below is scored. ──
            if ema_fast > ema_mid > ema_slow:
                return _cache(False, f"SHORT: Full bullish EMA stack (hard block)", "hold")

            pattern_ok = True
            if df is not None and len(df) > 10:
                recent_highs = df['High'].iloc[-5:].values
                recent_lows = df['Low'].iloc[-5:].values
                if len(recent_highs) >= 3 and recent_highs[-1] > recent_highs[-2] > recent_highs[-3]:
                    pattern_ok = False
                if len(recent_lows) >= 3 and recent_lows[-1] > recent_lows[-2] > recent_lows[-3]:
                    pattern_ok = False

            checks = [
                ("EMA200_macro_trend", not (ema200 > 0 and close > ema200)),
                ("EMA_fast_lt_slow", ema_fast < ema_slow),
                ("price_below_fastEMA", close < ema_fast),
                ("MACD_not_bullish", not (macd > macd_signal and macd_hist > 0)),
                ("RSI_below_65", rsi <= 65),
                ("volume_gate", not (volume_ratio < 0.8 and close < ema_fast)),
                ("no_bullish_pattern", pattern_ok),
            ]
            passed = sum(1 for _, ok in checks if ok)
            total = len(checks)
            confluence = passed / total
            min_confluence = getattr(self, 'direction_confluence_min', 0.75)

            if confluence >= min_confluence:
                failed = [name for name, ok in checks if not ok]
                note = f" (soft fail: {', '.join(failed)})" if failed else ""
                return _cache(True, f"SHORT: Valid bearish setup ({passed}/{total} confluence){note}", "sell")

            failed = [name for name, ok in checks if not ok]
            return _cache(False,
                          f"SHORT: confluence {passed}/{total} ({confluence:.0%}) < min {min_confluence:.0%} "
                          f"| failed: {', '.join(failed)}",
                          "hold")

        else:
            is_bullish = (ema_fast > ema_slow and close > ema_fast and macd > macd_signal)
            is_bearish = (ema_fast < ema_slow and close < ema_fast and macd < macd_signal)
            if is_bullish:
                # Apply same geometry filters as long-only path
                adx_now_b = data.get('ADX', 0)
                adx_prev_b = data.get('ADX_prev', adx_now_b)
                adx_slope_b = getattr(self, 'adx_slope_min', 0.1)
                if (adx_now_b - adx_prev_b) < adx_slope_b:
                    return _cache(False, f"BOTH/LONG: ADX slope {adx_now_b - adx_prev_b:.3f} < {adx_slope_b}", "hold")
                ef_b = data.get('EMA_Fast', 0);
                cl_b = data.get('Close', 0)
                if ef_b > 0 and cl_b > 0:
                    dp_b = (cl_b - ef_b) / ef_b * 100
                    pzlo_b = getattr(self, 'pullback_zone_lower_pct', -2.5)
                    pzhi_b = getattr(self, 'pullback_zone_upper_pct', 1.5)
                    if not (pzlo_b <= dp_b <= pzhi_b):
                        return _cache(False, f"BOTH/LONG: dist {dp_b:.2f}pct outside [{pzlo_b}/{pzhi_b}]", "hold")
                return _cache(True, "BOTH: Bullish setup detected", "buy")
            elif is_bearish:
                return _cache(True, "BOTH: Bearish setup detected", "sell")
            else:
                return _cache(False, "BOTH: No clear direction", "hold")

    def _create_pending_signal(self, direction, quality_score, tier,
                               component_scores, breakdown, data):
        decision = 'buy' if direction == 'long' else 'sell'
        self._pending_signal = {
            'direction': direction, 'decision': decision,
            'quality_score': quality_score, 'tier': tier,
            'position_mult': 1.0, 'breakdown': breakdown,
            'component_scores': component_scores,
            'signal_price': data.get('Close', 0),
            'signal_adx': data.get('ADX', 0),
            'signal_rsi': data.get('RSI', 50),
            'signal_macd': data.get('MACD', 0),
            'signal_volume': data.get('Volume_Ratio', 1.0),
            'signal_price_pct': data.get('Price_Percentile_20bar', 50),
            'signal_bar': self.bar_count,
            'signal_time': datetime.now(timezone.utc),
        }
        self._signal_bar = self.bar_count
        self._signal_price = data.get('Close', 0)
        direction_emoji = "📈" if direction == 'long' else "📉"
        self._log(f"{direction_emoji} TIER{tier} {direction.upper()} SIGNAL PENDING: Q={quality_score} "
                  f"@ ${self._signal_price:.2f} (will execute next bar)", "purple")
        return ("hold", quality_score, None,
                f"{direction.upper()}_TIER{tier}_SIGNAL_PENDING_execute_next_bar",
                component_scores)

    def _check_entry_conditions(self, data):
        cooldown_bars = getattr(self, 'cooldown_after_profit_target_bars', 2)
        bars_since_profit_target = self.bar_count - self._last_profit_target_bar
        if bars_since_profit_target < cooldown_bars:
            remaining = cooldown_bars - bars_since_profit_target
            return ("hold", 0, None, f"profit_target_cooldown_{remaining}_bars_remaining", {})

        # ═══ MIN BARS BETWEEN TRADES — now enforced (last_trade_bar fixed) ═══
        min_bars = getattr(self, 'min_bars_between_trades', 4)
        bars_since_last = self.bar_count - self.last_trade_bar
        if bars_since_last < min_bars:
            return ("hold", 0, None,
                    f"min_bars_between_trades_{bars_since_last}_of_{min_bars}", {})

        # ═══ v9.5 FIX #1: ATR COMPRESSION FILTER — blocks ALL directions ═══
        if self._is_atr_compressed(data):
            return ("hold", 0, None, "v94_atr_compressed_no_entries", {})

        # ═══ v9.5 FIX #6: CONSECUTIVE LOSS COOLDOWN ═══
        if self._is_consecutive_loss_cooldown_active():
            return ("hold", 0, None, "v94_consecutive_loss_cooldown", {})

        effective_direction = self.trade_direction

        if effective_direction == 'long':
            return self._check_long_entry_conditions(data)
        elif effective_direction == 'short':
            return self._check_short_entry_conditions(data)
        else:
            return self._check_both_entry_conditions(data)

    def _check_long_entry_conditions(self, data):
        direction_valid, direction_reason, suggested_action = self._validate_direction(data, direction_hint='long')
        if not direction_valid:
            return "hold", 0, None, direction_reason, {}

        # STRUCTURAL FIX: daily trend enforced directly in long entry path
        # (_daily_trend_is_up was only in _check_tier1_filters, which is never
        #  called for long entries — this moves the check into the live path)
        if not self._daily_trend_is_up(data):
            return "hold", 0, None, "long_daily_trend_not_up", {}

        # v9.5 FIX #2: Extended run filter for longs
        if self._is_extended_run_long(data):
            return "hold", 0, None, "v94_extended_run_long_blocked", {}

        self._suggested_action = 'buy'
        quality_score, component_scores, breakdown = self._calculate_quality_score(data)
        self._last_quality_score = quality_score
        min_for_signal = self.quality_tier2_min

        regime, confidence = self.regime_detector.detect_regime(
            ema_fast=data.get('EMA_Fast', 0), ema_slow=data.get('EMA_Slow', 0),
            adx=data.get('ADX', 0), vix=20,
            bb_width_percentile=data.get('BB_Width_Percentile', 50))
        self.current_regime = regime

        # BUG FIX v9.5.1 (#11): Regime filter was applied to shorts but silently omitted
        # from longs. Ranging/untradeable markets could still produce long entries.
        # This is symmetric with _check_short_entry_conditions (line ~2363).
        if getattr(self, 'regime_filter_enabled', True):
            if data.get('Ranging', False) or not self.regime_detector.is_tradeable(regime):
                return "hold", quality_score, None, f"long_regime_blocked_{regime}", component_scores

        if quality_score < min_for_signal:
            return ("hold", quality_score, None,
                    f"quality_too_low_{quality_score}_need_{min_for_signal}", component_scores)

        only_tier2 = getattr(self, 'only_tier2_entries', False)

        if only_tier2:
            if quality_score >= self.quality_tier2_min:
                if self._validate_tier2_conditions(data):
                    return self._create_pending_signal('long', quality_score, 2, component_scores, breakdown, data)
            return ("hold", quality_score, None,
                    f"TIER1_BLOCKED_by_only_tier2_flag", component_scores)
        else:
            if quality_score >= self.fixed_threshold:
                if self._validate_tier2_conditions(data):
                    return self._create_pending_signal('long', quality_score, 1, component_scores, breakdown, data)
                self._log(f"📉 TIER 1 strict filter failed, entering as Tier 2: Q={quality_score:.0f}", "yellow")
                return self._create_pending_signal('long', quality_score, 2, component_scores, breakdown, data)

            if quality_score >= self.quality_tier2_min:
                # BUG FIX v9.5.1 (#12): Both branches previously returned a Tier 2 signal,
                # meaning _validate_tier2_conditions had zero effect for Tier 2 entries.
                # Now, failing conditions (negative MACD histogram, ranging, loss cooldown)
                # actually block the entry instead of silently entering anyway.
                if self._validate_tier2_conditions(data):
                    return self._create_pending_signal('long', quality_score, 2, component_scores, breakdown, data)
                return ("hold", quality_score, None,
                        "tier2_conditions_not_met_macd_histogram_or_ranging", component_scores)

            elif getattr(self, 'fuzzy_learning_enabled', True) and 60 <= quality_score <= 74:
                self._track_near_miss(quality_score, data, f"quality_below_75_{quality_score}")

            return ("hold", quality_score, None,
                    f"quality_too_low_{quality_score}_need_{self.quality_tier2_min}", component_scores)

    def _check_short_entry_conditions(self, data):
        direction_valid, direction_reason, suggested_action = self._validate_direction(data, direction_hint='short')
        if not direction_valid:
            return "hold", 0, None, direction_reason, {}

        # v9.5 FIX #2: Extended run filter for shorts
        if self._is_extended_run_short(data):
            return "hold", 0, None, "v94_extended_run_short_blocked", {}

        self._suggested_action = 'sell'
        quality_score, component_scores, breakdown = self._calculate_quality_score_short(data)
        self._last_quality_score = quality_score

        if quality_score < self.short_quality_tier2_min:
            return ("hold", quality_score, None,
                    f"short_quality_too_low_{quality_score}_need_{self.short_quality_tier2_min}", component_scores)

        tier1_result = self._check_tier1_filters_short(data)
        # BUG FIX: tier1_result was computed but never checked — short Tier 1 filters
        # (ADX hard min, RSI range, volume, momentum, bearish price structure, ADX slope,
        #  pullback zone) were ALL dead code. Now enforced.
        if tier1_result != "pass":
            return ("hold", quality_score, None,
                    f"short_tier1_failed_{tier1_result}", component_scores)

        regime, confidence = self.regime_detector.detect_regime(
            ema_fast=data.get('EMA_Fast', 0), ema_slow=data.get('EMA_Slow', 0),
            adx=data.get('ADX', 0), vix=20,
            bb_width_percentile=data.get('BB_Width_Percentile', 50))
        self.current_regime = regime

        if getattr(self, 'regime_filter_enabled', True):
            if data.get('Ranging', False) or not self.regime_detector.is_tradeable(regime):
                return "hold", quality_score, None, f"regime_blocked_{regime}", component_scores

        only_tier2 = getattr(self, 'only_tier2_entries', False)

        if only_tier2:
            if quality_score >= self.short_quality_tier2_min:
                if self._validate_tier2_conditions_short(data):
                    return self._create_pending_signal('short', quality_score, 2, component_scores, breakdown, data)
            return ("hold", quality_score, None,
                    f"SHORT_TIER1_BLOCKED_by_only_tier2_flag", component_scores)
        else:
            if quality_score >= self.short_fixed_threshold:
                if self._validate_tier2_conditions_short(data):
                    return self._create_pending_signal('short', quality_score, 1, component_scores, breakdown, data)
                return self._create_pending_signal('short', quality_score, 2, component_scores, breakdown, data)

            if quality_score >= self.short_quality_tier2_min:
                # BUG FIX v9.5.1 (#12): Mirror fix for short Tier 2 entries.
                if self._validate_tier2_conditions_short(data):
                    return self._create_pending_signal('short', quality_score, 2, component_scores, breakdown, data)
                return ("hold", quality_score, None,
                        "short_tier2_conditions_not_met_macd_histogram_or_ranging", component_scores)

            elif getattr(self, 'fuzzy_learning_enabled', True) and 60 <= quality_score <= 74:
                self._track_near_miss(quality_score, data,
                                      f"short_quality_below_{self.short_fixed_threshold}_{quality_score}")

            return ("hold", quality_score, None,
                    f"short_quality_too_low_{quality_score}_need_{self.short_quality_tier2_min}", component_scores)

    def _check_both_entry_conditions(self, data):
        long_result = self._check_long_entry_conditions(data)
        long_decision = long_result[0] if long_result else "hold"

        short_result = self._check_short_entry_conditions(data)
        short_decision = short_result[0] if short_result else "hold"

        if long_decision == "hold" and short_decision == "hold":
            long_reason = long_result[3] if long_result and len(long_result) > 3 else "unknown"
            short_reason = short_result[3] if short_result and len(short_result) > 3 else "unknown"
            self._log(f"🚫 no_valid_direction | LONG blocked: {long_reason} | SHORT blocked: {short_reason}", "gray")
            return "hold", 0, None, f"no_valid_direction (long={long_reason} | short={short_reason})", {}

        if long_decision != "hold" and short_decision != "hold":
            long_quality = long_result[1] if len(long_result) > 1 else 0
            short_quality = short_result[1] if len(short_result) > 1 else 0

            if abs(long_quality - short_quality) > 5:
                if long_quality > short_quality:
                    return long_result
                else:
                    return short_result

            ema_fast_val = data.get('EMA_Fast', 0)
            ema_slow_val = data.get('EMA_Slow', 0)
            trend_is_up = ema_fast_val > ema_slow_val

            if trend_is_up:
                return long_result
            else:
                return short_result

        return long_result if long_decision != "hold" else short_result

    def calculate_position_size(self, equity, atr, price,
                                quality_score=75, tier=1, position_mult=1.0):
        stop_distance = atr * getattr(self, 'stop_loss_atr_mult', 3.5)
        stop_loss = price - stop_distance
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
            quality_score=quality_score, tier=tier, adx=adx)
        regime_mult = self.regime_detector.get_position_multiplier(self.current_regime)
        raw_size = base_size * regime_mult * position_mult

        # ── Value-based floor instead of forced min=1 ────────────────────────
        # max(1, int(raw)) breaks BTC/ETH: 1 unit at $60k = $60k > $50k equity
        # = trade silently rejected by backtesting engine = 0 trades.
        # Floor is 0.1% of equity expressed in units — affordable on any asset.
        min_notional_units = (equity * 0.001) / price if price > 0 else 0
        size = max(min_notional_units, raw_size)

        # Precision: fractional for high-price assets, whole units otherwise
        if price >= 1000:
            size = round(size, 6)
        elif price >= 100:
            size = round(size, 4)
        else:
            size = max(0, int(size))

        # v9.5 FIX #3: Hard cap enforced at logic level too
        max_units = getattr(self, 'max_position_units', 50)
        return min(size, max_units)

    def record_trade(self, profit, exit_reason="unknown", tier=None, size=None,
                     direction=None, entry_quality=None, entry_price=None,
                     exit_price=None, hold_duration=None, entry_bar=None, exit_bar=None,
                     signal_adx=None, signal_rsi=None, signal_macd=None,
                     signal_volume=None, signal_price_pct=None, signal_price=None,
                     signal_time=None, signal_bar=None):
        corrected_size = size
        trade_direction = direction or self.trade_direction

        if size is not None and size < 0 and trade_direction == 'long':
            corrected_size = abs(size)

        self.total_trades += 1
        self.total_profit += profit
        if profit > 0:
            self.winning_trades += 1
            self._consecutive_loss_count = 0  # v9.5: reset on win
        else:
            self.losing_trades += 1
            self._consecutive_loss_count += 1  # v9.5: increment on loss
            self._last_loss_bar = self.bar_count  # v9.5: track last loss bar

        if tier == 1:
            self.tier1_trades += 1
        elif tier == 2:
            self.tier2_trades += 1

        # BUG FIX: last_trade_bar was initialised to -999 and never updated,
        # so min_bars_between_trades (defined in MOMENTUM_PARAMS as 4) was
        # silently never enforced in backtest or live trading.
        self.last_trade_bar = self.bar_count

        trade_record = {
            'profit': profit, 'exit_reason': exit_reason, 'tier': tier,
            'size': corrected_size, 'original_size': size,
            'direction': trade_direction, 'entry_quality': entry_quality,
            'entry_price': entry_price, 'exit_price': exit_price,
            'hold_duration': hold_duration, 'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'signal_adx': signal_adx, 'signal_rsi': signal_rsi,
            'signal_macd': signal_macd, 'signal_volume': signal_volume,
            'signal_price_pct': signal_price_pct, 'signal_price': signal_price,
            'signal_time': signal_time, 'signal_bar': signal_bar,
            'timestamp': datetime.now(timezone.utc)
        }

        self.trade_history.append(trade_record)
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        if exit_reason and 'profit_target' in exit_reason:
            self._last_profit_target_bar = self.bar_count

    def validate_trade_log(self, trades_df=None):
        if trades_df is None and hasattr(self, 'trade_history'):
            trades_df = pd.DataFrame(self.trade_history)
        if trades_df is None or len(trades_df) == 0:
            return True
        if self.trade_direction == 'long':
            if 'size' in trades_df.columns:
                negative_sizes = trades_df[trades_df['size'] < 0]
                if len(negative_sizes) > 0: return False
        elif self.trade_direction == 'short':
            if 'size' in trades_df.columns:
                positive_sizes = trades_df[trades_df['size'] > 0]
                if len(positive_sizes) > 0: return False
        elif self.trade_direction == 'both':
            if 'direction' in trades_df.columns:
                invalid = trades_df[~trades_df['direction'].isin(['long', 'short', 'both'])]
                if len(invalid) > 0: return False
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
            'tier1_pct': (self.tier1_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'tier2_pct': (self.tier2_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
        }

    def log_fuzzy_threshold_stats(self):
        if not self.fuzzy_mode_enabled:
            self._log("RIGID MODE ACTIVE", "blue");
            return
        config = self._calculate_dynamic_fuzzy_threshold()
        self._log("FUZZY MODE STATS", "cyan")
        if config.get('using_default', False):
            self._log(f"Fuzzy Lower: {config['fuzzy_lower']:.1f} (DEFAULT)", "yellow")
        else:
            self._log(f"Fuzzy Lower: {config['fuzzy_lower']:.1f} conf:{config['confidence'] * 100:.0f}%", "green")

    def reset_fuzzy_learning(self):
        self.near_miss_trades = []
        self._previous_fuzzy_lower = None

    def clear_pending_signals(self):
        self._suggested_action = None
        self._last_direction_check['bar'] = -999

    def update_params(self, new_params, is_custom_mode=False):
        if not new_params: return
        self.config.update(new_params)
        for key, value in new_params.items():
            setattr(self, key, value)
        self.exit_manager = ProfessionalExitManager(self.config)
        # v9.5: sync hard cap
        self.risk_controller.max_position_units = getattr(self, 'max_position_units', 50)

        if 'trade_direction' in new_params:
            direction = new_params['trade_direction']
            self.trade_direction = direction
            self.only_long_entries = (direction == 'long')
            self.only_short_entries = (direction == 'short')

    def get_active_params(self):
        if self.current_mode == "Custom Parameters" and self.custom_params:
            active = self.config.copy()
            active.update(self.custom_params)
            return active
        else:
            return self.config.copy()


# ═══════════════════════════════════════════════════════════════════════════
# PART 9: LIVE TRADING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class MomentumStrategy(BaseStrategy, MomentumLogic):
    """v9.5 Live trading — WITH ALL FIXES"""

    def __init__(self, trading_app=None):
        BaseStrategy.__init__(self, trading_app)
        momentum_params = None
        if trading_app and hasattr(trading_app, 'get_current_momentum_params'):
            momentum_params = trading_app.get_current_momentum_params()
        params = MomentumConfig.get_config(momentum_params)
        MomentumLogic.__init__(self, config=params, trading_app=trading_app)

        self.trade_counter = 0
        self.name = "Professional Momentum Strategy v9.5 — ATR_FILTER | EXTENDED_RUN | SIZE_CAP"

        self.position = {
            'type': None, 'entry_price': None, 'quantity': None,
            'stop_loss': None, 'trailing_stop': None,
            'trailing_activated': False, 'highest_price': None,
            'lowest_price': None, 'entry_bar': None, 'partial_exits': 0,
            'original_quantity': None, 'tier': None, 'entry_time': None,
            'entry_quality_score': None, 'entry_reason': None,
            'trade_id': None, 'partial_pnl_realised': 0.0,
            'signal_adx': None, 'signal_rsi': None, 'signal_macd': None,
            'signal_volume': None, 'signal_price_pct': None, 'signal_price': None,
            'signal_time': None, 'signal_bar': None,
        }
        self.bars_held = 0
        self.last_signal = None
        self.signal_history = []
        self._account_quantity = 0
        self.trade_counter = 0

        if self.trading_app:
            self._log("=" * 70, "cyan")
            self._log("MOMENTUM STRATEGY v9.5.1 — ATR_FILTER | EXTENDED_RUN | SIZE_CAP | STRICT_SHORTS", "bold green")
            self._log(
                f"✅ ATR Compression Filter: ENABLED (threshold={getattr(self, 'atr_compression_threshold', 0.4)})",
                "green")
            self._log(f"✅ Extended Run Filter: max {getattr(self, 'extended_run_max_pct_long', 12.0)}% from swing",
                      "green")
            self._log(f"✅ Position Size Cap: {getattr(self, 'max_position_units', 50)} units max", "green")
            self._log(f"✅ Short ADX Min: {getattr(self, 'short_tier1_adx_hard_min', 30)}", "green")
            self._log(f"✅ Short Volume Min: {getattr(self, 'short_tier1_volume_min', 1.3)}x", "green")
            self._log(
                f"✅ Trend Age Penalty: -{getattr(self, 'trend_age_penalty_pts', 10)}pts after {getattr(self, 'trend_age_max_bars', 20)} bars",
                "green")
            self._log(
                f"✅ Consecutive Loss Cooldown: {getattr(self, 'consecutive_loss_cooldown_bars', 12)} bars after {getattr(self, 'consecutive_loss_threshold', 3)} losses",
                "green")
            self._log("=" * 70, "cyan")

    def run_analysis_cycle(self, current_data, current_price, df=None):
        self._current_df = df
        if self.strategy_state == StrategyState.SEEKING_ENTRY:
            return self.check_entry_conditions(current_data)
        return self.check_exit_conditions(current_data, current_price)

    def check_entry_conditions(self, current_data):
        if self.strategy_state != StrategyState.SEEKING_ENTRY:
            return "hold", POSITION_ALREADY_OPEN_SENTINEL, 0, "state_is_IN_TRADE"
        if 'Price_Percentile_20bar' not in current_data or current_data['Price_Percentile_20bar'] is None:
            current_data['Price_Percentile_20bar'] = 50
        return self._check_entry_conditions(current_data)

    def check_exit_conditions(self, current_data, current_price):
        if self.strategy_state != StrategyState.IN_TRADE:
            return None, 1.0

        if self.position['type'] == 'long':
            if self.position['highest_price'] is None:
                self.position['highest_price'] = current_price
            else:
                self.position['highest_price'] = max(self.position['highest_price'], current_price)
        else:
            if self.position['lowest_price'] is None:
                self.position['lowest_price'] = current_price
            else:
                self.position['lowest_price'] = min(self.position['lowest_price'], current_price)

        atr = current_data.get('ATR')
        if not atr or atr <= 0:
            raise ValueError(f"Invalid ATR at bar {self.bars_held}: {atr}")

        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0.01
        activation_threshold = max(atr_pct * 1.5, 0.015)
        trailing_distance_pct = max(atr_pct * 0.5, 0.01)

        if self.position['type'] == 'long':
            if not self.position.get('trailing_activated', False):
                profit_pct = (current_price - self.position['entry_price']) / self.position['entry_price']
                if profit_pct >= activation_threshold:
                    self.position['trailing_activated'] = True
                    self.position['trailing_stop'] = current_price * (1 - trailing_distance_pct)
            if self.position.get('trailing_activated', False):
                new_stop = self.position['highest_price'] * (1 - trailing_distance_pct)
                if new_stop > (self.position.get('trailing_stop') or 0):
                    self.position['trailing_stop'] = new_stop
        else:
            if not self.position.get('trailing_activated', False):
                profit_pct = (self.position['entry_price'] - current_price) / self.position['entry_price']
                if profit_pct >= activation_threshold:
                    self.position['trailing_activated'] = True
                    self.position['trailing_stop'] = current_price * (1 + trailing_distance_pct)
            if self.position.get('trailing_activated', False):
                new_stop = self.position['lowest_price'] * (1 + trailing_distance_pct)
                if new_stop < (self.position.get('trailing_stop') or float('inf')):
                    self.position['trailing_stop'] = new_stop

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
            trailing_activated=self.position.get('trailing_activated', False),
            trailing_stop=self.position.get('trailing_stop'))

    def sync_position_with_account(self):
        if not self.trading_app: return
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
                    self.position['quantity'] = actual_qty
            return actual_qty
        except Exception as e:
            self._log(f"Error syncing position: {e}", "red")
            return None

    def execute_buy(self, shares, price, atr, quality_score, tier):
        try:
            if self.trade_direction == 'short':
                return False, 0, None
            if self.trade_direction == 'both':
                pending = getattr(self, '_pending_signal', None)
                if pending and pending.get('direction') == 'short':
                    return False, 0, None
            if self.trade_direction == 'long':
                current_data = self._build_current_data() if hasattr(self, '_build_current_data') else None
                if current_data:
                    ema_fast = current_data.get('EMA_Fast', 0)
                    ema_slow = current_data.get('EMA_Slow', 0)
                    if ema_fast <= ema_slow:
                        return False, 0, None

            # v9.5 FIX #3: Cap shares
            max_units = getattr(self, 'max_position_units', 50)
            shares = min(shares, max_units)

            stop_loss_price = price - (atr * getattr(self, 'stop_loss_atr_mult', 3.5))
            if stop_loss_price >= price:
                return False, 0, None

            if self.trading_app and hasattr(self.trading_app, 'place_order'):
                order_result = self.trading_app.place_order(
                    side='buy', quantity=shares, price=price,
                    confidence=quality_score, atr=atr)
                if isinstance(order_result, dict):
                    if not order_result.get('success', False): return False, 0, None
                    filled_qty = order_result.get('filled_quantity', shares)
                    actual_price = order_result.get('filled_price', price)
                else:
                    filled_qty, actual_price = (shares, price) if order_result else (0, price)
                    if not order_result: return False, 0, None
            else:
                filled_qty, actual_price = shares, price

            self.trade_counter = getattr(self, 'trade_counter', 0) + 1

            signal_data = {}
            if hasattr(self, '_pending_signal') and self._pending_signal:
                signal_data = {
                    'signal_adx': self._pending_signal.get('signal_adx', 0),
                    'signal_rsi': self._pending_signal.get('signal_rsi', 50),
                    'signal_macd': self._pending_signal.get('signal_macd', 0),
                    'signal_volume': self._pending_signal.get('signal_volume', 1.0),
                    'signal_price_pct': self._pending_signal.get('signal_price_pct', 50),
                    'signal_price': self._pending_signal.get('signal_price', 0),
                    'signal_time': self._pending_signal.get('signal_time', datetime.now(timezone.utc)),
                    'signal_bar': self._pending_signal.get('signal_bar', getattr(self, 'bar_count', 0) - 1),
                }

            self.position = {
                'type': 'long', 'entry_price': actual_price,
                'quantity': filled_qty, 'original_quantity': filled_qty,
                'stop_loss': stop_loss_price, 'trailing_stop': None,
                'trailing_activated': False, 'highest_price': actual_price,
                'lowest_price': None, 'entry_bar': getattr(self, 'bar_count', 0),
                'partial_exits': 0, 'tier': tier,
                'entry_time': datetime.now(timezone.utc),
                'entry_quality_score': quality_score,
                'entry_reason': '', 'trade_id': self.trade_counter,
                'partial_pnl_realised': 0.0, **signal_data,
            }
            self.bars_held = 0
            self._transition_to_in_trade()
            return True, filled_qty, None
        except Exception as e:
            self._log(f"ERROR execute_buy: {e}", "bold red")
            return False, 0, None

    def execute_sell(self, reason="manual", exit_percentage=1.0):
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

            exit_qty = min(current_qty * exit_percentage, current_qty)

            if self.trading_app and hasattr(self.trading_app, 'place_order'):
                res = self.trading_app.place_order(side='sell', quantity=exit_qty, price=current_price)
                filled_qty = res.get('filled_quantity', exit_qty) if isinstance(res, dict) else exit_qty
                exit_price = res.get('filled_price', current_price) if isinstance(res, dict) else current_price
            else:
                filled_qty, exit_price = exit_qty, current_price

            # BUG FIX: long formula only — short P&L must flip sign
            is_short = self.position.get('type') == 'short'
            if is_short:
                leg_profit = (self.position['entry_price'] - exit_price) * filled_qty
                profit_pct = (self.position['entry_price'] - exit_price) / self.position['entry_price'] * 100
            else:
                leg_profit = (exit_price - self.position['entry_price']) * filled_qty
                profit_pct = (exit_price - self.position['entry_price']) / self.position['entry_price'] * 100
            # stop_dist: use abs() so profit_r is always positive when profitable
            stop_dist = abs(self.position['entry_price'] - self.position['stop_loss'])
            profit_r = (abs(exit_price - self.position['entry_price'])) / stop_dist if stop_dist != 0 else 0
            profit_r = profit_r if leg_profit >= 0 else -profit_r

            if exit_percentage >= 0.99:
                total_pnl = leg_profit + self.position.get('partial_pnl_realised', 0.0)
                self.record_trade(
                    profit=total_pnl, exit_reason=reason,
                    tier=self.position.get('tier'), size=current_qty,
                    direction=self.trade_direction,
                    entry_quality=self.position.get('entry_quality_score'),
                    entry_price=self.position['entry_price'],
                    exit_price=exit_price,
                    hold_duration=(datetime.now(timezone.utc) - self.position['entry_time']).total_seconds() / 60,
                    entry_bar=self.position.get('entry_bar'),
                    exit_bar=getattr(self, 'bar_count', 0),
                    signal_adx=self.position.get('signal_adx'),
                    signal_rsi=self.position.get('signal_rsi'),
                    signal_macd=self.position.get('signal_macd'),
                    signal_volume=self.position.get('signal_volume'),
                    signal_price_pct=self.position.get('signal_price_pct'),
                    signal_price=self.position.get('signal_price'),
                    signal_time=self.position.get('signal_time'),
                    signal_bar=self.position.get('signal_bar'))

                trade_rec = TradeRecord(
                    trade_id=self.position['trade_id'],
                    symbol=getattr(self, 'symbol', GlobalConfig.DEFAULT_SYMBOL),
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
                    hold_duration=(datetime.now(timezone.utc) - self.position['entry_time']).total_seconds() / 60,
                    market_regime=self.current_regime,
                    partial_exits_taken=self.position.get('partial_exits', 0),
                    partial_pnl_realised=self.position.get('partial_pnl_realised', 0.0),
                    original_size=self.position['original_quantity'])
                self.risk_controller.record_trade(trade_rec)

                self.position = {
                    'type': None, 'entry_price': None, 'quantity': None,
                    'stop_loss': None, 'trailing_stop': None,
                    'trailing_activated': False, 'highest_price': None,
                    'lowest_price': None, 'entry_bar': None, 'partial_exits': 0,
                    'original_quantity': None, 'tier': None, 'entry_time': None,
                    'entry_quality_score': None, 'entry_reason': None,
                    'trade_id': None, 'partial_pnl_realised': 0.0,
                    'signal_adx': None, 'signal_rsi': None, 'signal_macd': None,
                    'signal_volume': None, 'signal_price_pct': None, 'signal_price': None,
                    'signal_time': None, 'signal_bar': None,
                }
                self._account_quantity = 0
                self.bars_held = 0
                self._transition_to_seeking_entry()
            else:
                self.position['quantity'] -= filled_qty
                self.position['partial_exits'] = self.position.get('partial_exits', 0) + 1
                self.position['partial_pnl_realised'] = self.position.get('partial_pnl_realised', 0.0) + leg_profit
            return True, leg_profit, exit_price
        except Exception as e:
            self._log(f"ERROR execute_sell: {e}", "bold red")
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

    def set_fuzzy_mode(self, enabled):
        self.fuzzy_mode_enabled = enabled
        if enabled: self.log_fuzzy_threshold_stats()

    def reset_fuzzy_learning(self):
        super().reset_fuzzy_learning()

    def _build_current_data(self):
        if hasattr(self, '_current_df') and self._current_df is not None and len(self._current_df) > 0:
            last_row = self._current_df.iloc[-1]
            return {
                'Close': last_row.get('Close', 0),
                'EMA_Fast': last_row.get('EMA_Fast', 0),
                'EMA_Slow': last_row.get('EMA_Slow', 0),
                'EMA_Mid': last_row.get('EMA_Mid', 0),
                'MACD': last_row.get('MACD', 0),
                'MACD_Signal': last_row.get('MACD_Signal', 0),
                'MACD_Histogram': last_row.get('MACD_Histogram', 0),
                'RSI': last_row.get('RSI', 50),
                'Volume_Ratio': last_row.get('Volume_Ratio', 1.0),
                'ATR': last_row.get('ATR', 0),
                'ADX_prev': last_row.get('ADX_prev', last_row.get('ADX', 0)),
                'MACD_Histogram_prev': last_row.get('MACD_Histogram_closed', 0),
                'EMA_200': last_row.get('EMA_200', 0),
                'Above_EMA200': last_row.get('Above_EMA200', True),
            }
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# PART 10: BACKTEST STRATEGY — v9.5
# ═══════════════════════════════════════════════════════════════════════════

class BacktestMomentumStrategy(Strategy, MomentumLogic):
    """v9.5 Backtest — with ALL FIXES"""

    _use_updated_params = False
    _updated_params = {}

    @classmethod
    def set_updated_params(cls, params):
        if params and isinstance(params, dict):
            cls._use_updated_params = True
            cls._updated_params = params.copy()
        else:
            cls._use_updated_params = False
            cls._updated_params = {}

    @classmethod
    def reset_to_defaults(cls):
        cls._use_updated_params = False
        cls._updated_params = {}

    @staticmethod
    def _create_param_attributes():
        attrs = {}
        for key in MOMENTUM_PARAMS.keys():
            attrs[key] = None
        return attrs

    locals().update(_create_param_attributes.__func__())

    backtest_quality_active = True
    backtest_quality_values = [70, 75, 80]
    backtest_adx_active = True
    backtest_adx_values = [18, 20, 22, 25]
    backtest_rsi_active = True
    backtest_rsi_values = [35, 40, 45]
    backtest_volume_active = True
    backtest_volume_values = [0.9, 1.0, 1.1, 1.2]
    backtest_momentum_active = True
    backtest_momentum_values = [0.1, 0.2, 0.3]
    backtest_ema_fast_active = True
    backtest_ema_fast_values = [5, 8, 9, 12]
    backtest_ema_mid_active = True
    backtest_ema_mid_values = [18, 20, 21, 26]
    backtest_ema_slow_active = True
    backtest_ema_slow_values = [40, 45, 50, 55]
    backtest_weight_ema_active = True
    backtest_weight_ema_values = [15, 18, 20, 22]
    backtest_weight_adx_active = True
    backtest_weight_adx_values = [15, 18, 20, 22]
    backtest_weight_macd_active = True
    backtest_weight_macd_values = [20, 22, 25, 28]
    backtest_weight_rsi_active = True
    backtest_weight_rsi_values = [15, 18, 20, 22]
    backtest_weight_volume_active = True
    backtest_weight_volume_values = [10, 12, 15, 18]
    backtest_risk_tier1_active = True
    backtest_risk_tier1_values = [0.010, 0.012, 0.015, 0.018]
    backtest_risk_tier2_active = True
    backtest_risk_tier2_values = [0.012, 0.015, 0.018, 0.022]
    backtest_stop_loss_mult_active = True
    backtest_stop_loss_mult_values = [2.0, 2.5, 3.0, 3.5]
    backtest_only_tier2_active = True
    backtest_only_tier2_values = [True, False]

    def __init__(self, broker, data, params):
        Strategy.__init__(self, broker, data, params)

        if self.__class__._use_updated_params and self.__class__._updated_params:
            config = MomentumConfig.get_config(momentum_params_override=self.__class__._updated_params)
        else:
            config = MomentumConfig.get_config()

        if params:
            for key, value in params.items():
                if key in config:
                    config[key] = value

        for key, value in config.items():
            setattr(self, key, value)

        critical_defaults = {
            'trailing_activation_pct': 0.03, 'trailing_distance_pct': 0.035,
            'trade_direction': 'both', 'stop_loss_atr_mult': 3.5,
            'only_tier2_entries': False, 'quality_tier1_min': 68,
            'quality_tier2_min': 65, 'short_quality_tier2_min': 70,
            'short_fixed_threshold': 75, 'fixed_threshold': 72,
            'tier1_adx_hard_min': 25, 'short_tier1_adx_hard_min': 30,
            'tier1_volume_min': 0.8, 'max_position_units': 50,
        }
        for attr, default_val in critical_defaults.items():
            if not hasattr(self, attr) or getattr(self, attr) is None:
                setattr(self, attr, default_val)

        MomentumLogic.__init__(self, config=config, trading_app=None)

        print(f"\n{'=' * 70}")
        print(f"BACKTEST v9.5 CONFIGURATION LOADED")
        print(f"{'=' * 70}")
        print(f"Direction: {self.trade_direction.upper()}")
        print(f"ATR Compression Filter: {getattr(self, 'atr_compression_enabled', True)}")
        print(f"Extended Run Filter: max {getattr(self, 'extended_run_max_pct_long', 12.0)}%")
        print(f"Max Position Units: {getattr(self, 'max_position_units', 50)}")
        print(f"Short ADX Min: {getattr(self, 'short_tier1_adx_hard_min', 30)}")
        print(
            f"Trend Age Penalty: -{getattr(self, 'trend_age_penalty_pts', 10)}pts after {getattr(self, 'trend_age_max_bars', 20)} bars")
        print(f"{'=' * 70}\n")

        self._entry_price = np.nan
        self._stop_loss = np.nan
        self._highest_price = np.nan
        self._lowest_price = np.nan
        self._trailing_activated = False
        self._trailing_stop = None
        self._bars_held = 0
        self._partial_exits = 0
        self._entry_bar = -999
        self._entry_tier = None
        self._entry_quality = 0
        self._entry_reason = ""
        self._params_dict = params
        self._partial_pnl_realised = 0.0
        self._exit_reason_map = {}
        _cfg_direction = self.config.get('trade_direction', 'long')
        self._position_direction = _cfg_direction if _cfg_direction in ('long', 'short') else 'long'
        self._pending_signal = None
        self._signal_bar = -999
        self._signal_price = None
        self._signal_adx = None
        self._signal_rsi = None
        self._signal_macd = None
        self._signal_volume = None
        self._signal_price_pct = None
        self._signal_time = None

        # v9.5.1 FIX: Store bar interval in hours for correct cooldown display.
        # Detect from data if available, default to 1.0 (1H bars).
        try:
            idx_freq = data.df.index.freq
            if idx_freq is not None:
                self._bar_interval_hours = idx_freq.nanos / 3_600_000_000_000
            else:
                delta = (data.df.index[1] - data.df.index[0]).total_seconds() / 3600
                self._bar_interval_hours = delta
        except Exception:
            self._bar_interval_hours = 1.0  # default 1H

    def _get_optimization_ranges(self):
        ranges = {}
        optimization_map = {
            'backtest_quality_values': ['quality_minimum_score', 'quality_tier1_min'],
            'backtest_adx_values': ['tier1_adx_hard_min', 'adx_min'],
            'backtest_rsi_values': ['tier1_rsi_min', 'rsi_entry_min'],
            'backtest_volume_values': ['tier1_volume_min', 'volume_min_ratio'],
            'backtest_momentum_values': ['tier1_momentum_min', 'momentum_min'],
            'backtest_ema_fast_values': ['ema_fast_period'],
            'backtest_ema_mid_values': ['ema_mid_period'],
            'backtest_ema_slow_values': ['ema_slow_period'],
            'backtest_weight_ema_values': ['weight_ema'],
            'backtest_weight_adx_values': ['weight_adx'],
            'backtest_weight_macd_values': ['weight_macd'],
            'backtest_weight_rsi_values': ['weight_rsi'],
            'backtest_weight_volume_values': ['weight_volume'],
            'backtest_risk_tier1_values': ['risk_tier1'],
            'backtest_risk_tier2_values': ['risk_tier2'],
            'backtest_stop_loss_mult_values': ['stop_loss_atr_mult'],
            'backtest_only_tier2_values': ['only_tier2_entries'],
        }
        for attr_name, param_names in optimization_map.items():
            active_attr = attr_name.replace('_values', '_active')
            if hasattr(self, attr_name) and getattr(self, active_attr, True):
                values = getattr(self, attr_name)
                for param_name in param_names:
                    ranges[param_name] = values
        return ranges

    def _bt_safe_size(self, units, price):
        """Convert a unit-count size to a backtesting.py-compatible size value.

        backtesting.py assertion:  0 < size < 1  OR  round(size)==size >= 1
          • 0 < size < 1  → treated as equity fraction (e.g. 0.15 = 15 % of equity)
          • size >= 1, whole int → treated as number of units

        Our calculate_position_size returns fractional units for high-price assets
        (e.g. 0.14 BTC, 1.23 ETH). 1.23 is neither a valid fraction nor a whole
        integer, which triggers the AssertionError.  This method normalises any size
        to a value the library accepts.
        """
        if units <= 0:
            return 0
        # Already a valid whole integer — use directly
        if units >= 1 and units == round(units):
            return int(units)
        # Everything else (fractional BTC, fractional ETH, e.g. 0.14 or 1.23):
        # convert to equity fraction so backtesting.py knows the correct notional.
        notional = units * price
        fraction = notional / max(self.equity, 1.0)
        # Clamp to (0, 0.9999) — never request > 99 % of equity in one trade
        return max(0.0001, min(fraction, 0.9999))

    def _bt_open_position(self, current_data, quality_score, tier, position_mult, current_price):
        if self.only_tier2_entries and tier != 2:
            print(f"❌ BLOCKED: Tier {tier} entry attempted but only_tier2_entries=True")
            return

        size = self.calculate_position_size(
            self.equity, current_data['ATR'], current_price,
            quality_score, tier, position_mult)
        if size <= 0:
            return

        if self._position_direction == 'long':
            stop = current_price - (self.stop_loss_atr_mult * current_data['ATR'])
            if stop >= current_price:
                print(f"REJECTED: stop {stop:.4f} >= price {current_price:.4f}")
                return
            self.buy(size=self._bt_safe_size(size, current_price))
        else:
            stop = current_price + (self.stop_loss_atr_mult * current_data['ATR'])
            if stop <= current_price:
                print(f"REJECTED: stop {stop:.4f} <= price {current_price:.4f}")
                return
            self.sell(size=self._bt_safe_size(size, current_price))

        self._entry_price = current_price
        self._stop_loss = stop
        self._highest_price = current_price if self._position_direction == 'long' else None
        self._lowest_price = current_price if self._position_direction == 'short' else None
        self._bars_held = 0
        self._partial_exits = 0
        self._entry_bar = len(self.data) - 1
        self._entry_tier = tier
        self._entry_quality = quality_score
        self._partial_pnl_realised = 0.0
        self._trailing_activated = False
        self._trailing_stop = None
        self._transition_to_in_trade()

        if self._pending_signal:
            self._signal_adx = self._pending_signal.get('signal_adx')
            self._signal_rsi = self._pending_signal.get('signal_rsi')
            self._signal_macd = self._pending_signal.get('signal_macd')
            self._signal_volume = self._pending_signal.get('signal_volume')
            self._signal_price_pct = self._pending_signal.get('signal_price_pct')
            self._signal_price = self._pending_signal.get('signal_price')
            self._signal_time = self._pending_signal.get('signal_time')
            self._signal_bar = self._pending_signal.get('signal_bar')

        direction_icon = "⬆️" if self._position_direction == 'long' else "⬇️"
        print(f"{direction_icon} ENTER T{tier} Q={quality_score} @ ${current_price:.2f} "
              f"ADX={current_data['ADX']:.1f} RSI={current_data['RSI']:.1f} "
              f"Dir={self.trade_direction.upper()} → IN_TRADE")

    def _calculate_actual_tier(self, quality_score):
        # BUG FIX v9.5.1: Must check the HIGHER threshold (Tier 1) first.
        # Old code checked quality_tier2_min (68) first, making Tier 1 unreachable
        # since any score >= 68 also satisfies >= 68 and returns 2 immediately.
        if self.only_tier2_entries:
            return 2 if quality_score >= self.quality_tier2_min else 0
        else:
            if quality_score >= self.quality_tier1_min:  # >= 72 first
                return 1
            elif quality_score >= self.quality_tier2_min:  # 68-71 second
                return 2
            else:
                return 0

    def init(self):
        self.df_indicators = IndicatorCalculator.calculate(self.data.df.copy(), self.config)
        self.df_enhanced = self.df_indicators.copy()
        self.ema_fast = self.I(lambda: self.df_indicators['EMA_Fast'].values, name='EMA_Fast')
        self.ema_mid = self.I(lambda: self.df_indicators['EMA_Mid'].values, name='EMA_Mid')
        self.ema_slow = self.I(lambda: self.df_indicators['EMA_Slow'].values, name='EMA_Slow')
        self.ema_daily_50 = self.I(lambda: self.df_indicators['EMA_Daily_50'].values, name='EMA_Daily_50')
        self.above_daily_50 = self.I(lambda: self.df_indicators['Above_Daily_50'].astype(float).values,
                                     name='Above_Daily_50')
        self.macd_line = self.I(lambda: self.df_indicators['MACD'].values, name='MACD')
        self.macd_sig = self.I(lambda: self.df_indicators['MACD_Signal'].values, name='MACD_Signal')
        self.macd_hist = self.I(lambda: self.df_indicators['MACD_Histogram'].values, name='MACD_Histogram')
        self.adx = self.I(lambda: self.df_indicators['ADX'].values, name='ADX')
        self.rsi = self.I(lambda: self.df_indicators['RSI'].values, name='RSI')
        self.volume_ratio = self.I(lambda: self.df_indicators['Volume_Ratio'].values, name='Volume_Ratio')
        self.momentum = self.I(lambda: self.df_indicators['Momentum'].values, name='Momentum')
        self.atr = self.I(lambda: self.df_indicators['ATR'].values, name='ATR')
        self.adx_prev = self.I(
            lambda: self.df_indicators['ADX_prev'].fillna(self.df_indicators['ADX_prev'].bfill()).values,
            name='ADX_prev')
        self.ranging = self.I(lambda: self.df_indicators['Ranging'].astype(float).values, name='Ranging')
        self.price_pct = self.I(lambda: self.df_indicators['Price_Percentile_20bar'].values, name='Price_Pct')
        self.ema_200 = self.I(lambda: self.df_indicators['EMA_200'].values, name='EMA_200')
        self.above_ema200 = self.I(lambda: self.df_indicators['Above_EMA200'].astype(float).values, name='Above_EMA200')

    def _build_current_data(self):
        def safe(arr, default=0.0):
            try:
                v = arr[-1]
                return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            except Exception:
                return default

        def safe_bool(arr, default=True):
            try:
                v = arr[-1]
                if v is None or (isinstance(v, float) and np.isnan(v)): return default
                return bool(v)
            except Exception:
                return default

        return {
            'Close': self.data.Close[-1],
            'EMA_Fast': safe(self.ema_fast), 'EMA_Mid': safe(self.ema_mid),
            'EMA_Slow': safe(self.ema_slow),
            'ADX': safe(self.adx), 'RSI': safe(self.rsi, 50.0),
            'Volume_Ratio': safe(self.volume_ratio, 1.0),
            'Momentum': safe(self.momentum),
            'ATR': safe(self.atr, 1.0),
            'ADX_prev': safe(self.adx_prev),
            'Ranging': safe_bool(self.ranging, False),
            'MACD': safe(self.macd_line),
            'MACD_Signal': safe(self.macd_sig),
            'MACD_Histogram': safe(self.macd_hist),
            'MACD_closed': float(self.macd_line[-2]) if len(self.macd_line) > 1 and not np.isnan(
                self.macd_line[-2]) else 0.0,
            'MACD_Signal_closed': float(self.macd_sig[-2]) if len(self.macd_sig) > 1 and not np.isnan(
                self.macd_sig[-2]) else 0.0,
            'MACD_Histogram_Rising': (float(self.macd_hist[-1]) > float(self.macd_hist[-2])) if len(
                self.macd_hist) > 1 else False,
            'MACD_Histogram_prev': float(self.macd_hist[-2]) if len(self.macd_hist) > 1 and not np.isnan(
                self.macd_hist[-2]) else 0.0,
            'Price_Percentile_20bar': safe(self.price_pct, 50.0),
            'EMA_Daily_50': safe(self.ema_daily_50),
            'Above_Daily_50': safe_bool(self.above_daily_50, True),
            'EMA_200': safe(self.ema_200),
            'Above_EMA200': safe_bool(self.above_ema200, True),
        }

    def _bt_close_position(self, current_price, exit_signal, profit_pct):
        size_at_close = abs(self.position.size)
        self._exit_reason_map[self._entry_bar] = exit_signal
        self.position.close()

        if self._position_direction == 'long':
            final_leg = (current_price - self._entry_price) * size_at_close
            profit_pct_calc = (current_price - self._entry_price) / self._entry_price * 100
        else:
            final_leg = (self._entry_price - current_price) * size_at_close
            profit_pct_calc = (self._entry_price - current_price) / self._entry_price * 100

        total_profit = final_leg + self._partial_pnl_realised

        self.record_trade(
            profit=total_profit, exit_reason=exit_signal,
            tier=self._entry_tier, size=size_at_close,
            direction=self._position_direction,
            entry_quality=self._entry_quality,
            entry_price=self._entry_price, exit_price=current_price,
            hold_duration=self._bars_held,
            entry_bar=self._entry_bar, exit_bar=len(self.data) - 1,
            signal_adx=self._signal_adx, signal_rsi=self._signal_rsi,
            signal_macd=self._signal_macd, signal_volume=self._signal_volume,
            signal_price_pct=self._signal_price_pct,
            signal_price=self._signal_price,
            signal_time=self._signal_time, signal_bar=self._signal_bar)

        direction_icon = "⬆️" if self._position_direction == 'long' else "⬇️"
        win_loss_icon = "✅" if total_profit > 0 else "❌"
        print(f"{win_loss_icon} {direction_icon} EXIT @ ${current_price:.2f} {profit_pct_calc:+.2f}% "
              f"hold={self._bars_held}bars reason={exit_signal} → SEEKING_ENTRY")

        self._entry_price = np.nan;
        self._stop_loss = np.nan
        self._highest_price = np.nan;
        self._lowest_price = np.nan
        self._bars_held = 0;
        self._partial_exits = 0
        self._partial_pnl_realised = 0.0;
        self._entry_tier = None
        # BUG FIX v9.5.1 (STRAT BUG 10): self.trade_direction can be 'both', which is
        # not a valid position direction. Reset to 'long' as a safe neutral sentinel.
        # _position_direction will be properly set from the next pending signal.
        self._position_direction = 'long'
        self._trailing_activated = False;
        self._trailing_stop = None
        self._be_stop_set = False
        self._signal_adx = None;
        self._signal_rsi = None
        self._signal_macd = None;
        self._signal_volume = None
        self._signal_price_pct = None;
        self._signal_price = None
        self._signal_time = None;
        self._signal_bar = None
        self._transition_to_seeking_entry()

    def _extract_exit_reason(self, entry_bar):
        return self._exit_reason_map.get(entry_bar, "unknown")

    def get_trade_records_for_export(self):
        trades = []
        for i, trade in enumerate(self.trade_history):
            actual_tier = trade.get('tier', 0)
            quality_score = trade.get('entry_quality', 0)
            entry_price = trade.get('entry_price', 0)

            if trade.get('direction') == 'long':
                return_pct = ((trade.get('exit_price', 0) - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            else:
                return_pct = ((entry_price - trade.get('exit_price', 0)) / entry_price) * 100 if entry_price > 0 else 0

            trade_record = {
                'Trade_#': i + 1, 'Tier': f"Tier {actual_tier}",
                'Tier_Number': actual_tier,
                'Signal_Bar': trade.get('signal_bar', trade.get('entry_bar', 0) - 1),
                'Signal_Time': trade.get('signal_time', ''),
                'Signal_Price': trade.get('signal_price', 0),
                'Signal_Quality_Score': quality_score,
                'Signal_ADX': trade.get('signal_adx', 0),
                'Signal_RSI': trade.get('signal_rsi', 50),
                'Signal_MACD': trade.get('signal_macd', 0),
                'Signal_Volume_Ratio': trade.get('signal_volume', 1.0),
                'Signal_Price_Percentile': trade.get('signal_price_pct', 50),
                'Entry_Bar': trade.get('entry_bar', 0),
                'Entry_Time': trade.get('entry_time', ''),
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
                'Confluence_Score': trade.get('confluence_score', 0),
                'Risk_Allocation_%': trade.get('risk_allocation', 0) * 100,
            }
            trades.append(trade_record)
        return trades

    def next(self):
        try:
            if any(np.isnan(x[-1]) for x in [
                self.ema_fast, self.adx, self.rsi, self.macd_line, self.atr, self.ema_daily_50
            ]):
                return
        except Exception:
            return

        idx = len(self.data) - 1
        current_data = self._build_current_data()
        current_price = self.data.Close[-1]
        self._current_df = self.df_enhanced.iloc[:idx + 1]
        self.bar_count = idx

        # Check for pending signal from previous bar
        if self._pending_signal is not None and self.bar_count > self._signal_bar:
            signal = self._pending_signal
            execution_price = self.data.Open[-1]

            self._position_direction = 'long' if signal['decision'] == "buy" else 'short'

            size = self.calculate_position_size(
                self.equity, current_data['ATR'], execution_price,
                signal['quality_score'], signal['tier'], signal['position_mult'])

            if size > 0:
                if signal['decision'] == "buy":
                    stop = execution_price - (self.stop_loss_atr_mult * current_data['ATR'])
                    if stop < execution_price:
                        self.buy(size=self._bt_safe_size(size, execution_price))
                    else:
                        self._pending_signal = None
                        return
                else:
                    stop = execution_price + (self.stop_loss_atr_mult * current_data['ATR'])
                    if stop > execution_price:
                        self.sell(size=self._bt_safe_size(size, execution_price))
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
                self._trailing_activated = False
                self._trailing_stop = None
                self._be_stop_set = False
                self._signal_adx = signal.get('signal_adx')
                self._signal_rsi = signal.get('signal_rsi')
                self._signal_macd = signal.get('signal_macd')
                self._signal_volume = signal.get('signal_volume')
                self._signal_price_pct = signal.get('signal_price_pct')
                self._signal_price = signal.get('signal_price')
                self._signal_time = signal.get('signal_time')
                self._signal_bar = signal.get('signal_bar')
                self._transition_to_in_trade()

                direction_icon = "⬆️" if self._position_direction == 'long' else "⬇️"
                print(f"{direction_icon} ENTER T{signal['tier']} Q={signal['quality_score']} "
                      f"@ ${execution_price:.2f} Size={size} Stop=${stop:.2f}")

            self._pending_signal = None
            self._signal_bar = -999
            self._signal_price = None
            return

        # =========================================================
        # SEEKING ENTRY STATE
        # =========================================================
        if self.strategy_state == StrategyState.SEEKING_ENTRY:
            result = self._check_entry_conditions(current_data)

            # Check if we got a pending signal
            if hasattr(self, '_pending_signal') and self._pending_signal is not None:
                # Signal will execute on next bar
                return
            return

        # =========================================================
        # IN_TRADE STATE - MANAGE EXITS
        # =========================================================
        if self.strategy_state == StrategyState.IN_TRADE:
            self._bars_held += 1

            # Update highest/lowest prices for trailing
            if self._position_direction == 'long':
                if current_price > self._highest_price:
                    self._highest_price = current_price
            else:
                if self._lowest_price is None or current_price < self._lowest_price:
                    self._lowest_price = current_price

            atr = current_data.get('ATR', 0)
            if atr <= 0:
                return

            # =========================================================
            # FIXED: CORRECT VOLATILITY-BASED THRESHOLDS
            # =========================================================
            # Calculate ATR as percentage of price (keep as decimal, NOT multiplied by 100)
            atr_pct = (atr / current_price) if current_price > 0 else 0.001

            # Get configured thresholds (already as decimals)
            trailing_activation = getattr(self, 'trailing_activation_pct', 0.03)
            trailing_distance = getattr(self, 'trailing_distance_pct', 0.035)

            # Use max of ATR-based or configured values (both as decimals)
            activation_threshold = max(atr_pct * 1.5, trailing_activation)
            distance_threshold = max(atr_pct * 0.5, trailing_distance)

            # Debug output for first few bars of a new trade
            if self._bars_held <= 3:
                print(f"🔧 VOLATILITY DEBUG: ATR={atr:.4f}, ATR%={atr_pct:.2%}, "
                      f"Activation={activation_threshold:.2%}, Distance={distance_threshold:.2%}")

            # =========================================================
            # TRAILING STOP LOGIC (FIXED)
            # =========================================================
            if self._position_direction == 'long':
                # Check if we should activate trailing stop
                if not self._trailing_activated:
                    profit_pct = (current_price - self._entry_price) / self._entry_price
                    if profit_pct >= activation_threshold:
                        self._trailing_activated = True
                        self._trailing_stop = current_price * (1 - distance_threshold)
                        print(f"🔒 TRAILING ACTIVATED @ {profit_pct:.2%} "
                              f"(threshold={activation_threshold:.2%}) Stop=${self._trailing_stop:.2f}")

                # Update trailing stop if activated
                if self._trailing_activated:
                    new_stop = self._highest_price * (1 - distance_threshold)
                    if new_stop > (self._trailing_stop or 0):
                        self._trailing_stop = new_stop

            else:  # short position
                # Check if we should activate trailing stop
                if not self._trailing_activated:
                    profit_pct = (self._entry_price - current_price) / self._entry_price
                    if profit_pct >= activation_threshold:
                        self._trailing_activated = True
                        self._trailing_stop = current_price * (1 + distance_threshold)
                        print(f"🔒 TRAILING ACTIVATED (SHORT) @ {profit_pct:.2%} "
                              f"Stop=${self._trailing_stop:.2f}")

                # Update trailing stop if activated
                if self._trailing_activated:
                    new_stop = self._lowest_price * (1 + distance_threshold)
                    if new_stop < (self._trailing_stop or float('inf')):
                        self._trailing_stop = new_stop

            # =========================================================
            # BREAKEVEN STOP LOGIC
            # =========================================================
            if getattr(self, 'be_stop_enabled', True) and not getattr(self, '_be_stop_set', False):
                stop_distance = abs(self._entry_price - self._stop_loss)
                if stop_distance > 0:
                    if self._position_direction == 'long':
                        profit_amount = current_price - self._entry_price
                        be_trigger_r = getattr(self, 'be_stop_r_trigger', 2.0)
                        no_progress_bars = getattr(self, 'be_stop_no_progress_bars', 50)

                        if profit_amount >= be_trigger_r * stop_distance:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BREAKEVEN STOP ACTIVATED @ {profit_amount / stop_distance:.1f}R")
                        elif self._bars_held >= no_progress_bars and profit_amount / self._entry_price < 0.003:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BREAKEVEN STOP (NO PROGRESS) after {self._bars_held} bars")

                    else:  # short
                        profit_amount = self._entry_price - current_price
                        be_trigger_r = getattr(self, 'be_stop_r_trigger', 2.0)
                        no_progress_bars = getattr(self, 'be_stop_no_progress_bars', 50)

                        if profit_amount >= be_trigger_r * stop_distance:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BREAKEVEN STOP ACTIVATED (SHORT) @ {profit_amount / stop_distance:.1f}R")
                        elif self._bars_held >= no_progress_bars and profit_amount / self._entry_price < 0.003:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BREAKEVEN STOP (NO PROGRESS) after {self._bars_held} bars")

            # =========================================================
            # EVALUATE EXIT CONDITIONS
            # =========================================================
            exit_signal, exit_pct = self.exit_manager.evaluate_exit(
                current_price=current_price,
                entry_price=self._entry_price,
                stop_loss=self._stop_loss,
                highest_price=self._highest_price if self._position_direction == 'long' else None,
                lowest_price=self._lowest_price if self._position_direction == 'short' else None,
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
                atr=atr,
                position_type=self._position_direction,
                trailing_activated=self._trailing_activated,
                trailing_stop=self._trailing_stop
            )

            # =========================================================
            # HARD STOP CHECK (additional safety)
            # =========================================================
            if exit_signal is None:
                if self._position_direction == 'long':
                    if current_price <= self._stop_loss:
                        exit_signal = "stop_loss_hard"
                        exit_pct = 1.0
                        print(f"🛑 HARD STOP HIT: ${current_price:.2f} <= ${self._stop_loss:.2f}")
                else:
                    if current_price >= self._stop_loss:
                        exit_signal = "stop_loss_hard"
                        exit_pct = 1.0
                        print(f"🛑 HARD STOP HIT (SHORT): ${current_price:.2f} >= ${self._stop_loss:.2f}")

            # =========================================================
            # TRAILING STOP CHECK (additional safety)
            # =========================================================
            if exit_signal is None and self._trailing_activated and self._trailing_stop is not None:
                if self._position_direction == 'long' and current_price <= self._trailing_stop:
                    exit_signal = "trailing_stop_hit"
                    exit_pct = 1.0
                    print(f"🎯 TRAILING STOP HIT: ${current_price:.2f} <= ${self._trailing_stop:.2f}")
                elif self._position_direction == 'short' and current_price >= self._trailing_stop:
                    exit_signal = "trailing_stop_hit"
                    exit_pct = 1.0
                    print(f"🎯 TRAILING STOP HIT (SHORT): ${current_price:.2f} >= ${self._trailing_stop:.2f}")

            # =========================================================
            # FORCE EXIT AT END OF BACKTEST (CRITICAL FIX v9.5.1)
            # =========================================================
            # BUG FIX: idx IS already len(self.data)-1, so the old check
            # was ALWAYS True, closing every trade after exactly 1 bar.
            # Correct approach: compare against the full df_enhanced length.
            total_bars = len(self.df_enhanced)
            is_last_bar = (idx == total_bars - 1)
            if exit_signal is None and is_last_bar and self._bars_held > 0:
                if self._position_direction == 'long':
                    profit_pct = (current_price - self._entry_price) / self._entry_price * 100
                else:
                    profit_pct = (self._entry_price - current_price) / self._entry_price * 100
                exit_signal = "end_of_backtest"
                exit_pct = 1.0
                print(f"🏁 FORCED EXIT AT END OF BACKTEST: Profit={profit_pct:.2f}%")

            # =========================================================
            # EXECUTE EXIT
            # =========================================================
            if exit_signal:
                if self._position_direction == 'long':
                    profit_pct = (current_price - self._entry_price) / self._entry_price * 100
                else:
                    profit_pct = (self._entry_price - current_price) / self._entry_price * 100

                # Handle partial exits
                if exit_pct < 1.0 and self._partial_exits < 4:
                    self._partial_exits += 1
                    raw_partial = abs(self.position.size) * exit_pct
                    # BUG FIX: int() zeroed out fractional BTC positions (e.g. int(0.07)=0)
                    # Use price-aware rounding matching calculate_position_size logic
                    if current_price >= 1000:
                        partial_size = round(raw_partial, 6)
                    elif current_price >= 100:
                        partial_size = round(raw_partial, 4)
                    else:
                        partial_size = int(raw_partial)
                    # Convert to backtesting.py-safe size
                    partial_size = self._bt_safe_size(raw_partial, current_price)
                    if partial_size > 0:
                        if self._position_direction == 'long':
                            self.sell(size=partial_size)
                            partial_profit = (current_price - self._entry_price) * raw_partial
                        else:
                            self.buy(size=partial_size)
                            partial_profit = (self._entry_price - current_price) * raw_partial
                        self._partial_pnl_realised += partial_profit
                        print(
                            f"📊 PARTIAL EXIT {self._partial_exits}: {raw_partial:.6f} units, Profit=${partial_profit:.2f}")
                        return

                # Full exit
                self._bt_close_position(current_price, exit_signal, profit_pct)

# End of file