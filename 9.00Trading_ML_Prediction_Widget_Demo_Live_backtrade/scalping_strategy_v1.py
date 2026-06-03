# ═══════════════════════════════════════════════════════════════════════════
# PROFESSIONAL SCALPING STRATEGY v1.0
# Target: 15-min bars | Win Rate >65% | Sharpe >1.5 | Max DD <10%
# ─────────────────────────────────────────────────────────────────────────
# Core Logic:
#   • EMA 5/13/21 trend alignment
#   • MACD (6,13,5) momentum confirmation — faster than standard
#   • Stochastic (5,3,3) entry timing
#   • RSI 14 zone filter
#   • ATR-based stops + adaptive trailing
#   • Volume confirmation (1.2x minimum)
#   • Quality score gate (0–100) — must pass before entry
#   • Pending-signal execution (signal bar N → fill bar N+1 open)
# ═══════════════════════════════════════════════════════════════════════════


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
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class GlobalConfig:
    """Single source of truth for global trading parameters."""

    INITIAL_CAPITAL = 50000.0      # ← change this, everything updates

    COMMISSION_RATE  = 0.0005      # 0.05 % taker fee (typical crypto)
    DEFAULT_SYMBOL   = "SOL-USDT"

    @classmethod
    def update_capital(cls, new_capital: float) -> float:
        old = cls.INITIAL_CAPITAL
        cls.INITIAL_CAPITAL = float(new_capital)
        logging.info(f"💰 CAPITAL UPDATED: ${old:,.2f} → ${cls.INITIAL_CAPITAL:,.2f}")
        return cls.INITIAL_CAPITAL


# Deprecated alias — always use GlobalConfig.INITIAL_CAPITAL directly
CAPITAL = GlobalConfig.INITIAL_CAPITAL


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def safe_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss <= 0:
        return float('inf')
    return gross_profit / gross_loss


def compute_sharpe(returns, risk_free: float = 0.0, periods_per_year: int = 26280) -> float:
    # 26 280 = 252 × 24 × 4 for 15-min bars.  Adjust for other timeframes.
    import numpy as _np
    if len(returns) < 2:
        return 0.0
    s = _np.std(returns)
    return ((_np.mean(returns) - risk_free) / s) * _np.sqrt(periods_per_year) if s > 0 else 0.0


def compute_sortino(returns, target: float = 0.0, periods_per_year: int = 26280) -> float:
    import numpy as _np
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < target]
    if len(downside) == 0:
        return float('inf')
    d = _np.sqrt(_np.mean(downside ** 2))
    return (_np.mean(returns) / d) * _np.sqrt(periods_per_year) if d > 0 else float('inf')


def summarize_performance(trades: List[Any], initial_capital: float = None) -> Dict:
    if initial_capital is None:
        initial_capital = GlobalConfig.INITIAL_CAPITAL
    n = len(trades)
    if n == 0:
        return {"total_trades": 0, "win_rate": 0.0, "profit_pct": 0.0,
                "profit_factor": None, "expectancy": 0.0, "warning": "No trades"}
    total_profit   = sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades)
    wins           = sum(1 for t in trades if getattr(t, 'profit', t.get('profit', 0)) > 0)
    losses         = n - wins
    gross_profit   = sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades
                         if getattr(t, 'profit', t.get('profit', 0)) > 0)
    gross_loss     = -sum(getattr(t, 'profit', t.get('profit', 0)) for t in trades
                          if getattr(t, 'profit', t.get('profit', 0)) < 0)
    win_rate       = wins / n
    avg_win        = gross_profit / wins   if wins   > 0 else 0.0
    avg_loss       = gross_loss  / losses  if losses > 0 else 0.0
    expectancy     = win_rate * avg_win - (1 - win_rate) * avg_loss
    return {
        "total_trades"        : n,
        "win_rate"            : round(win_rate, 4),
        "profit_pct"          : round(total_profit / initial_capital * 100, 4),
        "profit_factor"       : round(safe_profit_factor(gross_profit, gross_loss), 4),
        "expectancy"          : round(expectancy, 6),
        "avg_profit_per_trade": round(total_profit / n, 6),
        "gross_profit"        : round(gross_profit, 2),
        "gross_loss"          : round(gross_loss,   2),
        "wins"                : wins,
        "losses"              : losses,
        "warning"             : f"⚠️ Small sample ({n})" if n < 30 else "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

class StrategyState(Enum):
    SEEKING_ENTRY = auto()
    IN_TRADE      = auto()


POSITION_ALREADY_OPEN_SENTINEL = -1


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    trade_id             : int
    symbol               : str
    entry_time           : datetime
    entry_price          : float
    entry_size           : float
    entry_tier           : int
    entry_quality_score  : int
    entry_reason         : str
    entry_direction      : str
    exit_time            : Optional[datetime] = None
    exit_price           : Optional[float]    = None
    exit_size            : Optional[float]    = None
    exit_reason          : Optional[str]      = None
    profit               : float = 0.0
    profit_pct           : float = 0.0
    profit_r             : float = 0.0
    max_profit           : float = 0.0
    max_drawdown_pct     : float = 0.0
    hold_duration        : float = 0.0
    market_regime        : str   = "UNKNOWN"
    partial_exits_taken  : int   = 0
    partial_pnl_realised : float = 0.0
    original_size        : float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RiskMetrics:
    daily_loss          : float = 0.0
    max_drawdown        : float = 0.0
    consecutive_losses  : int   = 0
    win_rate            : float = 0.0
    profit_factor       : float = 0.0
    sharpe_ratio        : float = 0.0
    sortino_ratio       : float = 0.0
    expectancy          : float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SCALPING PARAMETERS  — Single source of truth
# ═══════════════════════════════════════════════════════════════════════════

SCALPING_PARAMS = {

    # ─── DIRECTION ────────────────────────────────────────────────────────
    "trade_direction"               : "both",   # "long" | "short" | "both"

    # ─── EMA PERIODS (fast trend) ─────────────────────────────────────────
    "ema_fast_period"               : 5,
    "ema_mid_period"                : 13,
    "ema_slow_period"               : 21,

    # ─── MACD (faster settings for scalping) ──────────────────────────────
    "macd_fast"                     : 6,
    "macd_slow"                     : 13,
    "macd_signal_period"            : 5,

    # ─── STOCHASTIC (tight, fast) ─────────────────────────────────────────
    "stoch_k_period"                : 5,
    "stoch_d_period"                : 3,
    "stoch_smooth"                  : 3,
    "stoch_overbought"              : 80,
    "stoch_oversold"                : 20,
    "stoch_mid_upper"               : 70,
    "stoch_mid_lower"               : 30,

    # ─── RSI ──────────────────────────────────────────────────────────────
    "rsi_period"                    : 14,
    "rsi_long_min"                  : 45,
    "rsi_long_max"                  : 68,
    "rsi_short_min"                 : 32,
    "rsi_short_max"                 : 55,
    "rsi_overbought_exit"           : 75,
    "rsi_oversold_exit"             : 25,

    # ─── ADX ──────────────────────────────────────────────────────────────
    "adx_period"                    : 14,
    "adx_min_long"                  : 20,
    "adx_min_short"                 : 22,
    "adx_extended_threshold"        : 45,   # above = trend exhausted

    # ─── VOLUME ───────────────────────────────────────────────────────────
    "volume_period"                 : 20,
    "volume_min_ratio"              : 1.2,  # 1.2× average minimum
    "volume_strong_ratio"           : 1.8,  # 1.8× = strong confirmation

    # ─── ATR ──────────────────────────────────────────────────────────────
    "atr_period"                    : 10,
    "atr_compression_lookback"      : 40,
    "atr_compression_threshold"     : 0.35, # block when ATR < 35% of 40-bar avg

    # ─── QUALITY SCORE THRESHOLDS ─────────────────────────────────────────
    "quality_min_long"              : 62,
    "quality_min_short"             : 64,
    "quality_tier1_min"             : 75,   # high-conviction tier 1

    # ─── QUALITY COMPONENT WEIGHTS (must sum ≤ 100) ───────────────────────
    "weight_ema"                    : 22,
    "weight_macd"                   : 23,
    "weight_stoch"                  : 20,
    "weight_rsi"                    : 18,
    "weight_volume"                 : 12,
    "weight_adx"                    : 5,

    # ─── RISK MANAGEMENT ──────────────────────────────────────────────────
    "risk_per_trade"                : 0.008,  # 0.8% of equity per trade
    "risk_tier1"                    : 0.010,  # 1.0% for high-conviction
    "max_position_size_pct"         : 0.15,   # never > 15% equity in one trade
    "max_position_units"            : 100,    # hard unit cap
    "min_cash_reserve"              : 0.20,
    "base_risk_pct"                 : 0.008,

    # ─── STOP LOSS & TRAILING ─────────────────────────────────────────────
    "stop_loss_atr_mult"            : 1.5,    # tight stop for scalping
    "trailing_activation_pct"       : 0.008,  # 0.8% profit → activate trail
    "trailing_distance_pct"         : 0.006,  # trail 0.6% from peak
    "trailing_atr_mult"             : 0.8,    # ATR-based trail distance

    # ─── BREAKEVEN STOP ───────────────────────────────────────────────────
    "be_stop_enabled"               : True,
    "be_stop_r_trigger"             : 1.5,    # move to BE at 1.5R profit
    "be_stop_no_progress_bars"      : 15,     # BE if no progress in 15 bars (~3.75h)

    # ─── PROFIT TARGETS (R-multiples) ─────────────────────────────────────
    "take_profit_r1"                : 1.5,    # partial exit at 1.5R
    "take_profit_r2"                : 2.5,    # partial exit at 2.5R
    "partial_exit_pct_r1"           : 0.50,   # exit 50% at R1
    "partial_exit_pct_r2"           : 0.30,   # exit 30% at R2
    # remaining 20% runs with trailing stop

    # ─── EXIT CONDITIONS ──────────────────────────────────────────────────
    "macd_cross_exit_enabled"       : True,
    "macd_cross_min_profit_r"       : 1.0,    # only exit on MACD cross if >= 1R profit
    "stoch_reversal_exit_enabled"   : True,
    "stoch_reversal_min_profit_r"   : 0.8,
    "ema_cross_exit_enabled"        : True,
    "ema_cross_min_profit_r"        : 1.5,
    "max_hold_bars"                 : 32,     # 32 × 15min = 8 hours max
    "min_hold_bars_before_stop"     : 3,

    # ─── ENTRY TIMING FILTERS ─────────────────────────────────────────────
    "pullback_zone_lower_pct"       : -1.5,   # max % below EMA_fast at entry
    "pullback_zone_upper_pct"       : 0.8,    # max % above EMA_fast at entry
    "adx_slope_min"                 : 0.05,   # ADX must be rising
    "momentum_period"               : 3,      # short look-back for 15-min momentum
    "momentum_min_long"             : 0.01,   # 0.01% momentum needed for long
    "momentum_min_short"            : 0.01,   # negative

    # ─── REGIME / RANGING FILTER ──────────────────────────────────────────
    "regime_filter_enabled"         : True,
    "bb_period"                     : 20,
    "bb_std"                        : 2.0,
    "kc_period"                     : 20,
    "kc_atr_mult"                   : 1.5,
    "chop_period"                   : 14,
    "chop_threshold"                : 61,
    "ranging_min_checks"            : 4,

    # ─── TRADE FREQUENCY / COOLDOWN ───────────────────────────────────────
    "max_daily_trades"              : 20,
    "min_bars_between_trades"       : 2,      # 2 × 15min = 30 min minimum gap
    "cooldown_after_loss_bars"      : 4,      # 4 × 15min = 1 hour after loss
    "consecutive_loss_threshold"    : 3,
    "consecutive_loss_cooldown_bars": 8,      # 8 × 15min = 2 hours

    # ─── DAILY TREND FILTER ───────────────────────────────────────────────
    "daily_trend_filter_enabled"    : True,
    "daily_ema_period"              : 96,     # 96 × 15min = 24 hours
    "daily_trend_adx_override"      : 25,     # skip daily filter if ADX >= 25 + stacked EMA

    # ─── EXTENDED RUN FILTER ──────────────────────────────────────────────
    "extended_run_lookback"         : 12,
    "extended_run_max_pct_long"     : 4.0,    # block longs if > 4% from swing low
    "extended_run_max_pct_short"    : 4.0,

    # ─── TREND AGE PENALTY ────────────────────────────────────────────────
    "trend_age_penalty_enabled"     : True,
    "trend_age_max_bars"            : 12,     # 12 × 15min = 3 hours
    "trend_age_penalty_pts"         : 8,

    # ─── LOSS LIMITS (daily / drawdown) ───────────────────────────────────
    "daily_loss_limit_pct"          : 0.02,   # stop trading if -2% equity today
    "max_drawdown_limit_pct"        : 0.10,   # halt if -10% from peak
    "max_consecutive_losses"        : 5,

    # ─── BACKTESTING ONLY TIER 2 ──────────────────────────────────────────
    "only_tier2_entries"            : False,

    # ─── TIMEFRAME HINT ───────────────────────────────────────────────────
    "bar_interval_minutes"          : 15,

    # ─── FUZZY LEARNING (optional) ────────────────────────────────────────
    "fuzzy_mode_enabled"            : False,
    "fuzzy_learning_enabled"        : True,
    "fuzzy_absolute_min"            : 45,
    "fuzzy_absolute_max"            : 62,
    "fuzzy_default_margin_pct"      : 10,
    "fuzzy_min_confidence"          : 0.60,
    "fuzzy_min_samples"             : 8,
}


# ═══════════════════════════════════════════════════════════════════════════
# RISK CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingRiskController:
    def __init__(self, starting_equity: float = None):
        if starting_equity is None:
            starting_equity = GlobalConfig.INITIAL_CAPITAL

        self.starting_equity     = starting_equity
        self.current_equity      = starting_equity
        self.peak_equity         = starting_equity
        self.daily_loss_limit    = starting_equity * SCALPING_PARAMS['daily_loss_limit_pct']
        self.max_drawdown_limit  = starting_equity * SCALPING_PARAMS['max_drawdown_limit_pct']
        self.max_consecutive     = SCALPING_PARAMS['max_consecutive_losses']
        self.max_position_pct    = SCALPING_PARAMS['max_position_size_pct']
        self.max_position_units  = SCALPING_PARAMS['max_position_units']
        self.min_cash_reserve    = SCALPING_PARAMS['min_cash_reserve']
        self.base_risk_pct       = SCALPING_PARAMS['base_risk_pct']

        self.trades              : List[TradeRecord] = []
        self.today_loss          = 0.0
        self.today_date          = datetime.now().date()
        self.equity_curve        = [starting_equity]
        self.consecutive_losses  = 0
        self.total_trades        = 0
        self.winning_trades      = 0
        self.losing_trades       = 0
        self.risk_metrics        = RiskMetrics()

    # ── Position sizing ───────────────────────────────────────────────────
    def calculate_position_size(self, entry_price: float, stop_loss_price: float,
                                quality_score: int = 70, adx: float = 25.0) -> float:
        if entry_price <= 0:
            return 0
        risk_per_trade = abs(entry_price - stop_loss_price) / entry_price
        if risk_per_trade <= 0 or risk_per_trade > 0.15:
            return 0

        quality_weight = max(0.6, min((quality_score / 70) ** 0.5, 1.4))

        if adx < 20:   adx_mult = 0.6
        elif adx < 25: adx_mult = 0.85
        elif adx < 35: adx_mult = 1.0
        elif adx < 45: adx_mult = 0.9
        else:          adx_mult = 0.6

        streak_mult  = 0.7 if self.consecutive_losses >= 2 else 1.0
        health       = min((self.current_equity / max(1.0, self.peak_equity)) ** 2, 1.0)
        risk_pct     = self.base_risk_pct * quality_weight * adx_mult * streak_mult * health
        risk_pct     = max(0.002, min(risk_pct, 0.025))

        risk_amount  = self.current_equity * risk_pct
        position_sz  = risk_amount / (entry_price * risk_per_trade)
        max_sz       = self.current_equity * self.max_position_pct / entry_price
        position_sz  = min(position_sz, max_sz, self.max_position_units)

        if entry_price >= 1000: return max(0.0, round(position_sz, 6))
        elif entry_price >= 100: return max(0.0, round(position_sz, 4))
        else: return max(0, int(position_sz))

    # ── Validation ────────────────────────────────────────────────────────
    def validate_entry(self, position_size: float, entry_price: float) -> Tuple[bool, str]:
        if self.today_loss <= -self.daily_loss_limit:
            return False, f"daily_loss_limit_{self.today_loss:.2f}"
        dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if dd >= self.max_drawdown_limit:
            return False, f"max_drawdown_{dd:.1%}"
        if self.consecutive_losses >= self.max_consecutive:
            return False, f"consecutive_losses_{self.consecutive_losses}"
        cost = position_size * entry_price
        if cost > self.current_equity * (1 - self.min_cash_reserve):
            return False, f"insufficient_cash"
        if (position_size * entry_price) / self.current_equity > self.max_position_pct:
            return False, f"position_too_large"
        return True, "pass"

    def record_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self.total_trades += 1
        if trade.profit > 0:
            self.winning_trades    += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades     += 1
            self.consecutive_losses += 1
        if trade.exit_time and trade.exit_time.date() == self.today_date:
            self.today_loss += trade.profit
        self.current_equity += trade.profit
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        self.equity_curve.append(self.current_equity)
        self._recalculate_metrics()

    def _recalculate_metrics(self):
        if self.total_trades == 0: return
        self.risk_metrics.win_rate = self.winning_trades / self.total_trades
        wins   = sum(t.profit for t in self.trades if t.profit > 0)
        losses = abs(sum(t.profit for t in self.trades if t.profit < 0))
        self.risk_metrics.profit_factor = safe_profit_factor(wins, losses)
        peak = self.starting_equity
        for eq in self.equity_curve:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak
            if dd > self.risk_metrics.max_drawdown:
                self.risk_metrics.max_drawdown = dd
        self.risk_metrics.expectancy = sum(t.profit for t in self.trades) / self.total_trades
        if len(self.equity_curve) > 1:
            rets = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            if len(rets) > 0 and np.std(rets) > 0:
                self.risk_metrics.sharpe_ratio  = compute_sharpe(rets)
                self.risk_metrics.sortino_ratio = compute_sortino(rets)

    def get_stats(self) -> dict:
        rm = self.risk_metrics
        return {
            'total_trades'    : self.total_trades,
            'win_rate'        : f"{rm.win_rate * 100:.1f}%",
            'profit_factor'   : f"{rm.profit_factor:.2f}" if rm.profit_factor != float('inf') else "∞",
            'max_drawdown'    : f"{rm.max_drawdown:.1%}",
            'sharpe_ratio'    : f"{rm.sharpe_ratio:.2f}",
            'sortino_ratio'   : f"{rm.sortino_ratio:.2f}",
            'expectancy'      : f"${rm.expectancy:.2f}",
            'current_equity'  : f"${self.current_equity:,.2f}",
            'total_profit'    : f"${self.current_equity - self.starting_equity:,.2f}",
            'roi'             : f"{(self.current_equity - self.starting_equity) / self.starting_equity:.1%}",
            'consecutive_loss': self.consecutive_losses,
        }


# ═══════════════════════════════════════════════════════════════════════════
# REGIME DETECTOR  (lightweight — same interface as momentum version)
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    def __init__(self):
        self.current_regime    = "UNKNOWN"
        self.regime_confidence = 0.0
        self.regime_history    = deque(maxlen=100)

    def detect_regime(self, ema_fast, ema_slow, adx, bb_width_pct=50):
        up   = ema_fast > ema_slow
        strg = adx > 25
        sqz  = bb_width_pct < 30

        if up and strg:   regime, conf = "TRENDING_UP",   0.90
        elif not up and strg: regime, conf = "TRENDING_DOWN", 0.90
        elif sqz:         regime, conf = "RANGING_TIGHT", 0.80
        else:             regime, conf = "CHOPPY",        0.55

        self.current_regime    = regime
        self.regime_confidence = conf
        self.regime_history.append(regime)
        return regime, conf

    def is_tradeable(self, regime: str) -> bool:
        return regime not in ("RANGING_TIGHT", "CHOPPY")

    def get_position_multiplier(self, regime: str) -> float:
        return {"TRENDING_UP": 1.2, "TRENDING_DOWN": 1.0,
                "RANGING_TIGHT": 0.4, "CHOPPY": 0.5}.get(regime, 0.6)


# ═══════════════════════════════════════════════════════════════════════════
# EXIT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingExitManager:
    def __init__(self, config: dict):
        self.cfg = config

    def evaluate_exit(self, current_price, entry_price, stop_loss,
                      highest_price, lowest_price, bars_held, partial_exits,
                      ema_fast, ema_mid, ema_slow,
                      macd, macd_signal, macd_prev, signal_prev,
                      stoch_k, stoch_d, rsi, adx, atr,
                      position_type='long',
                      trailing_activated=False, trailing_stop=None):

        stop_dist  = atr * self.cfg.get('stop_loss_atr_mult', 1.5)
        if position_type == 'long':
            profit_r = (current_price - entry_price) / stop_dist if stop_dist > 0 else 0
        else:
            profit_r = (entry_price - current_price) / stop_dist if stop_dist > 0 else 0

        min_hold = self.cfg.get('min_hold_bars_before_stop', 3)

        # ── 1. HARD STOP ─────────────────────────────────────────────────
        if position_type == 'long':
            if current_price <= stop_loss and bars_held >= min_hold:
                return "stop_loss_hard", 1.0
            if current_price <= stop_loss and bars_held < min_hold:
                emergency = entry_price - stop_dist * self.cfg.get('emergency_stop_mult', 2.0)
                if current_price <= emergency:
                    return "stop_loss_emergency", 1.0
        else:
            if current_price >= stop_loss and bars_held >= min_hold:
                return "stop_loss_hard", 1.0
            if current_price >= stop_loss and bars_held < min_hold:
                emergency = entry_price + stop_dist * self.cfg.get('emergency_stop_mult', 2.0)
                if current_price >= emergency:
                    return "stop_loss_emergency", 1.0

        # ── 2. TRAILING STOP HIT ─────────────────────────────────────────
        if trailing_activated and trailing_stop is not None:
            if position_type == 'long' and current_price <= trailing_stop:
                return "trailing_stop", 1.0
            if position_type == 'short' and current_price >= trailing_stop:
                return "trailing_stop", 1.0

        # ── 3. PARTIAL EXITS (R-based) — only if less than 2 partials ────
        r1 = self.cfg.get('take_profit_r1', 1.5)
        r2 = self.cfg.get('take_profit_r2', 2.5)
        if partial_exits == 0 and profit_r >= r1:
            return "partial_r1", self.cfg.get('partial_exit_pct_r1', 0.50)
        if partial_exits == 1 and profit_r >= r2:
            return "partial_r2", self.cfg.get('partial_exit_pct_r2', 0.30)

        # ── 4. RSI EXTREME EXIT ───────────────────────────────────────────
        if position_type == 'long' and rsi > self.cfg.get('rsi_overbought_exit', 75) and profit_r >= 0.5:
            return "rsi_overbought", 1.0
        if position_type == 'short' and rsi < self.cfg.get('rsi_oversold_exit', 25) and profit_r >= 0.5:
            return "rsi_oversold", 1.0

        # ── 5. STOCHASTIC REVERSAL ────────────────────────────────────────
        if self.cfg.get('stoch_reversal_exit_enabled', True) and profit_r >= self.cfg.get('stoch_reversal_min_profit_r', 0.8):
            if position_type == 'long':
                ob = self.cfg.get('stoch_overbought', 80)
                if stoch_k > ob and stoch_d > ob and stoch_k < stoch_d:
                    return "stoch_overbought_cross", 1.0
            else:
                os_ = self.cfg.get('stoch_oversold', 20)
                if stoch_k < os_ and stoch_d < os_ and stoch_k > stoch_d:
                    return "stoch_oversold_cross", 1.0

        # ── 6. MACD CROSS EXIT ────────────────────────────────────────────
        if self.cfg.get('macd_cross_exit_enabled', True) and profit_r >= self.cfg.get('macd_cross_min_profit_r', 1.0):
            if position_type == 'long':
                if macd_prev >= signal_prev and macd < macd_signal:
                    if not (ema_fast > ema_slow and adx >= 28):
                        return "macd_bearish_cross", 1.0
            else:
                if macd_prev <= signal_prev and macd > macd_signal:
                    if not (ema_fast < ema_slow and adx >= 28):
                        return "macd_bullish_cross", 1.0

        # ── 7. EMA FULL REVERSAL ──────────────────────────────────────────
        if self.cfg.get('ema_cross_exit_enabled', True) and profit_r >= self.cfg.get('ema_cross_min_profit_r', 1.5):
            if position_type == 'long' and ema_fast < ema_mid < ema_slow:
                return "ema_full_reversal", 1.0
            if position_type == 'short' and ema_fast > ema_mid > ema_slow:
                return "ema_full_reversal", 1.0

        # ── 8. MAX HOLD TIME ──────────────────────────────────────────────
        if bars_held >= self.cfg.get('max_hold_bars', 32):
            return "max_hold_time", 1.0

        return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingConfig:
    CONFIG_FILE      = "scalping_settings.json"
    _custom_params   = {}
    _current_mode    = "Default Parameters"

    @classmethod
    def get_config(cls, override: dict = None) -> dict:
        config = SCALPING_PARAMS.copy()
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r') as f:
                    saved = json.load(f)
                if saved.get('selected_mode') == "Custom Parameters":
                    for k, v in saved.get('custom_params', {}).items():
                        if k in config:
                            config[k] = v
            except Exception as e:
                logging.warning(f"Could not load scalping config: {e}")
        if override and isinstance(override, dict):
            for k, v in override.items():
                if k in config:
                    config[k] = v
        return config

    @classmethod
    def save_config(cls, custom_params: dict, selected_mode: str = "Custom Parameters"):
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump({'timestamp': datetime.now().isoformat(),
                           'selected_mode': selected_mode,
                           'custom_params': custom_params}, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Config save error: {e}")
            return False

    @classmethod
    def reset_to_defaults(cls) -> dict:
        return SCALPING_PARAMS.copy()


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorCalculator:

    @staticmethod
    def calculate(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        try:
            # ── EMAs ──────────────────────────────────────────────────────
            df['EMA_Fast'] = talib.EMA(df['Close'], params['ema_fast_period'])
            df['EMA_Mid']  = talib.EMA(df['Close'], params['ema_mid_period'])
            df['EMA_Slow'] = talib.EMA(df['Close'], params['ema_slow_period'])
            daily_period   = params.get('daily_ema_period', 96)
            df['EMA_Daily']     = talib.EMA(df['Close'], daily_period)
            df['Above_Daily']   = (df['Close'] > df['EMA_Daily']).astype(bool)

            # ── MACD (faster settings for scalping) ───────────────────────
            df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(
                df['Close'],
                fastperiod  = params['macd_fast'],
                slowperiod  = params['macd_slow'],
                signalperiod= params['macd_signal_period'])
            df['MACD_Hist_Rising'] = df['MACD_Histogram'] > df['MACD_Histogram'].shift(1)

            # ── Stochastic ────────────────────────────────────────────────
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
                df['High'], df['Low'], df['Close'],
                fastk_period  = params['stoch_k_period'],
                slowk_period  = params['stoch_smooth'],
                slowk_matype  = 0,
                slowd_period  = params['stoch_d_period'],
                slowd_matype  = 0)
            df['Stoch_K_Rising'] = df['Stoch_K'] > df['Stoch_K'].shift(1)

            # ── ADX / DMI ─────────────────────────────────────────────────
            df['ADX']     = talib.ADX(df['High'], df['Low'], df['Close'], params['adx_period'])
            df['ADX_prev']= df['ADX'].shift(1)
            df['DMP']     = talib.PLUS_DI(df['High'], df['Low'], df['Close'], params['adx_period'])
            df['DMM']     = talib.MINUS_DI(df['High'], df['Low'], df['Close'], params['adx_period'])

            # ── RSI ───────────────────────────────────────────────────────
            df['RSI'] = talib.RSI(df['Close'], params['rsi_period'])

            # ── Volume ────────────────────────────────────────────────────
            df['Volume_MA'] = talib.SMA(df['Volume'], params['volume_period'])
            with np.errstate(divide='ignore', invalid='ignore'):
                df['Volume_Ratio'] = np.where(
                    df['Volume_MA'] > 0, df['Volume'] / df['Volume_MA'], 1.0)
            df['Volume_Ratio'] = (df['Volume_Ratio']
                                  .replace([np.inf, -np.inf], np.nan)
                                  .fillna(1.0).clip(0.01, 10.0))

            # ── ATR ───────────────────────────────────────────────────────
            df['ATR']    = talib.ATR(df['High'], df['Low'], df['Close'], params['atr_period'])
            atr_lb       = params.get('atr_compression_lookback', 40)
            df['ATR_MA'] = df['ATR'].rolling(atr_lb).mean()
            df['ATR_Compressed'] = (
                df['ATR'] < df['ATR_MA'] * params.get('atr_compression_threshold', 0.35))

            # ── Momentum (short, 3-bar) ────────────────────────────────────
            mp = params.get('momentum_period', 3)
            df['Momentum'] = df['Close'].pct_change(mp) * 100

            # ── BB + KC squeeze (ranging detection) ───────────────────────
            df = IndicatorCalculator._detect_ranging(df, params)

            # ── Price percentile (20-bar range) ───────────────────────────
            df['High_20'] = df['High'].rolling(20).max()
            df['Low_20']  = df['Low'].rolling(20).min()
            df['Price_Range_20'] = df['High_20'] - df['Low_20']
            with np.errstate(divide='ignore', invalid='ignore'):
                df['Price_Pct_20'] = np.where(
                    df['Price_Range_20'] > 0,
                    (df['Close'] - df['Low_20']) / df['Price_Range_20'] * 100, 50.0)
            df['Price_Pct_20'] = df['Price_Pct_20'].clip(0, 100).fillna(50)

            # ── Extended run filter pre-compute ───────────────────────────
            run_lb = params.get('extended_run_lookback', 12)
            df['Swing_Low']       = df['Low'].rolling(run_lb).min()
            df['Swing_High']      = df['High'].rolling(run_lb).max()
            df['Run_From_Low']    = ((df['Close'] - df['Swing_Low']) / df['Swing_Low'] * 100).fillna(0)
            df['Run_From_High']   = ((df['Swing_High'] - df['Close']) / df['Swing_High'] * 100).fillna(0)

            # ── Trend age pre-compute ─────────────────────────────────────
            bull = (df['EMA_Fast'] > df['EMA_Slow']).astype(int)
            bear = (df['EMA_Fast'] < df['EMA_Slow']).astype(int)
            df['Trend_Age_Bull'] = bull.groupby(bull.ne(bull.shift()).cumsum()).cumsum()
            df['Trend_Age_Bear'] = bear.groupby(bear.ne(bear.shift()).cumsum()).cumsum()

            # ── Closed (shifted) copies ───────────────────────────────────
            for col in ['EMA_Fast', 'EMA_Mid', 'EMA_Slow', 'ADX', 'RSI',
                        'MACD', 'MACD_Signal', 'MACD_Histogram',
                        'Stoch_K', 'Stoch_D', 'Volume_Ratio', 'ATR',
                        'Momentum', 'Price_Pct_20', 'Above_Daily',
                        'ATR_Compressed', 'Ranging',
                        'Trend_Age_Bull', 'Trend_Age_Bear']:
                if col in df.columns:
                    df[f'{col}_closed'] = df[col].shift(1)

            return df

        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
            raise

    @staticmethod
    def _detect_ranging(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        bb_period = params.get('bb_period', 20)
        bb_std    = params.get('bb_std', 2.0)
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std, matype=0)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close'].replace(0, 1)

        kc_p = params.get('kc_period', 20)
        kc_m = params.get('kc_atr_mult', 1.5)
        df['KC_Mid']   = df['Close'].ewm(span=kc_p).mean()
        df['KC_ATR']   = df['ATR'].rolling(kc_p).mean() if 'ATR' in df.columns else (df['BB_Upper'] - df['BB_Lower']) / 4
        df['KC_Upper'] = df['KC_Mid'] + kc_m * df['KC_ATR']
        df['KC_Lower'] = df['KC_Mid'] - kc_m * df['KC_ATR']
        df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']

        chop_p   = params.get('chop_period', 14)
        df['CHOP'] = IndicatorCalculator._choppiness(df['High'], df['Low'], df['Close'], chop_p)

        chop_thr  = params.get('chop_threshold', 61)
        min_chk   = params.get('ranging_min_checks', 4)

        c1 = (abs(df['Close'] - df['EMA_Fast']) / df['EMA_Fast'].replace(0, 1) <= 0.004).fillna(False)
        c2 = (df['BB_Width'] < df['KC_Width']).fillna(False)          # BB inside KC = squeeze
        c3 = (df['ADX'] < 20).fillna(False) if 'ADX' in df.columns else pd.Series(False, index=df.index)
        c4 = df['RSI'].between(44, 56).fillna(False)  if 'RSI' in df.columns else pd.Series(False, index=df.index)
        c5 = (df['CHOP'] >= chop_thr).fillna(False)
        c6 = (df['Volume_Ratio'] < 0.8).fillna(False) if 'Volume_Ratio' in df.columns else pd.Series(False, index=df.index)

        score = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int) + c5.astype(int) + c6.astype(int)
        df['Ranging'] = (score >= min_chk).fillna(False)
        return df

    @staticmethod
    def _choppiness(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low  - close.shift(1))
        tr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_s = tr.rolling(period).sum()
        hn    = high.rolling(period).max()
        ln    = low.rolling(period).min()
        chop  = 100 * np.log10(atr_s / (hn - ln).replace(0, 1)) / np.log10(period)
        return chop.fillna(50)


# ═══════════════════════════════════════════════════════════════════════════
# CORE SCALPING LOGIC
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingLogic:

    def __init__(self, config: dict = None, trading_app=None):
        self.config       = config or ScalpingConfig.get_config()
        self.trading_app  = trading_app

        for k, v in self.config.items():
            setattr(self, k, v)

        self.risk_controller    = ScalpingRiskController()
        self.risk_controller.max_position_units = getattr(self, 'max_position_units', 100)
        self.regime_detector    = RegimeDetector()
        self.exit_manager       = ScalpingExitManager(self.config)

        self.strategy_state     = StrategyState.SEEKING_ENTRY
        self.bar_count          = 0
        self.bars_held          = 0
        self.last_trade_bar     = -999
        self.trade_counter      = 0
        self.total_trades       = 0
        self.winning_trades     = 0
        self.losing_trades      = 0
        self.total_profit       = 0.0
        self.trade_history      : List[dict] = []
        self.current_regime     = "UNKNOWN"
        self.equity_curve       = [GlobalConfig.INITIAL_CAPITAL]

        self._current_df        = None
        self._pending_signal    = None
        self._signal_bar        = -999
        self._signal_price      = None
        self._last_profit_target_bar = -999
        self._consecutive_loss_count = 0
        self._last_loss_bar     = -999
        self._last_quality_score = 0
        self._near_miss_trades  = []

        trade_dir = (
            trading_app.trade_direction_var.get()
            if trading_app and hasattr(trading_app, 'trade_direction_var')
            else self.config.get('trade_direction', 'both')
        )
        self.trade_direction = trade_dir

        self._last_dir_cache    = {'bar': -999, 'dir': None, 'result': True, 'reason': ''}

    # ─────────────────────────────────────────────────────────────────────
    # STATE TRANSITIONS
    # ─────────────────────────────────────────────────────────────────────
    def _to_in_trade(self):      self.strategy_state = StrategyState.IN_TRADE
    def _to_seeking_entry(self): self.strategy_state = StrategyState.SEEKING_ENTRY

    # ─────────────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────────────
    def _log(self, msg: str, color: str = "white"):
        if self.trading_app and hasattr(self.trading_app, 'log_message'):
            self.trading_app.log_message(msg, color)
        else:
            print(f"[{color.upper()}] {msg}")

    # ─────────────────────────────────────────────────────────────────────
    # GUARD FILTERS
    # ─────────────────────────────────────────────────────────────────────
    def _is_atr_compressed(self, data: dict) -> bool:
        df = self._current_df
        if df is None or len(df) < 40:
            return False
        atr_now = data.get('ATR', 0)
        atr_avg = float(df['ATR_MA'].iloc[-1]) if 'ATR_MA' in df.columns else float(df['ATR'].iloc[-40:].mean())
        if atr_avg <= 0:
            return False
        thr = getattr(self, 'atr_compression_threshold', 0.35)
        compressed = atr_now < atr_avg * thr
        # Strong trend override
        if compressed:
            row = df.iloc[-1]
            if (float(row.get('EMA_Fast', 0)) > float(row.get('EMA_Mid', 0)) >
                    float(row.get('EMA_Slow', 0)) and float(row.get('ADX', 0)) >= 25):
                return False
        return compressed

    def _is_extended_run_long(self, data: dict) -> bool:
        df = self._current_df
        if df is None or len(df) < 12:
            return False
        close = data.get('Close', 0)
        swing_low = float(df['Swing_Low'].iloc[-1]) if 'Swing_Low' in df.columns else float(df['Low'].iloc[-12:].min())
        if swing_low <= 0:
            return False
        run_pct = (close - swing_low) / swing_low * 100
        max_run = getattr(self, 'extended_run_max_pct_long', 4.0)
        if run_pct > max_run:
            self._log(f"🔴 EXT_RUN_LONG: +{run_pct:.1f}% from low (max={max_run}%)", "red")
            return True
        return False

    def _is_extended_run_short(self, data: dict) -> bool:
        df = self._current_df
        if df is None or len(df) < 12:
            return False
        close = data.get('Close', 0)
        swing_high = float(df['Swing_High'].iloc[-1]) if 'Swing_High' in df.columns else float(df['High'].iloc[-12:].max())
        if swing_high <= 0:
            return False
        run_pct = (swing_high - close) / swing_high * 100
        max_run = getattr(self, 'extended_run_max_pct_short', 4.0)
        if run_pct > max_run:
            self._log(f"🔴 EXT_RUN_SHORT: -{run_pct:.1f}% from high (max={max_run}%)", "red")
            return True
        return False

    def _is_consecutive_loss_cooldown(self) -> bool:
        thresh  = getattr(self, 'consecutive_loss_threshold', 3)
        cd_bars = getattr(self, 'consecutive_loss_cooldown_bars', 8)
        if self._consecutive_loss_count >= thresh:
            if (self.bar_count - self._last_loss_bar) < cd_bars:
                return True
            else:
                self._consecutive_loss_count = 0
        return False

    def _trend_age_penalty(self, direction: str = 'long') -> int:
        if not getattr(self, 'trend_age_penalty_enabled', True):
            return 0
        df  = self._current_df
        if df is None or len(df) < 5:
            return 0
        max_bars  = getattr(self, 'trend_age_max_bars', 12)
        pen_pts   = getattr(self, 'trend_age_penalty_pts', 8)
        col = 'Trend_Age_Bull' if direction == 'long' else 'Trend_Age_Bear'
        if col in df.columns:
            age = int(df[col].iloc[-1])
            if age >= max_bars:
                return pen_pts
        return 0

    def _daily_trend_is_up(self, data: dict) -> bool:
        if not getattr(self, 'daily_trend_filter_enabled', True):
            return True
        above = data.get('Above_Daily', None)
        if above is None:
            df = self._current_df
            above = bool(df['Above_Daily'].iloc[-1]) if df is not None and 'Above_Daily' in df.columns else True
        if bool(above):
            return True
        # Short-term momentum override
        ef  = data.get('EMA_Fast', 0)
        em  = data.get('EMA_Mid',  0)
        es  = data.get('EMA_Slow', 0)
        adx = data.get('ADX', 0)
        if ef > em > es and adx >= getattr(self, 'daily_trend_adx_override', 25):
            return True
        return False

    def _daily_trend_is_down(self, data: dict) -> bool:
        if not getattr(self, 'daily_trend_filter_enabled', True):
            return True
        above = data.get('Above_Daily', None)
        if above is None:
            df = self._current_df
            above = bool(df['Above_Daily'].iloc[-1]) if df is not None and 'Above_Daily' in df.columns else False
        return not bool(above)

    # ─────────────────────────────────────────────────────────────────────
    # QUALITY SCORE — LONG
    # ─────────────────────────────────────────────────────────────────────
    def _quality_score_long(self, data: dict) -> Tuple[int, dict, str]:
        if self.strategy_state != StrategyState.SEEKING_ENTRY:
            raise RuntimeError("Quality score called while IN_TRADE")

        parts = []
        scores = {}

        close       = data.get('Close', 0)
        ema_fast    = data.get('EMA_Fast', 0)
        ema_mid     = data.get('EMA_Mid', 0)
        ema_slow    = data.get('EMA_Slow', 0)
        macd        = data.get('MACD', 0)       or 0
        macd_sig    = data.get('MACD_Signal', 0) or 0
        macd_hist   = data.get('MACD_Histogram', 0) or 0
        hist_rising = data.get('MACD_Hist_Rising', False)
        stoch_k     = data.get('Stoch_K', 50)
        stoch_d     = data.get('Stoch_D', 50)
        rsi         = data.get('RSI', 50)
        volume_r    = data.get('Volume_Ratio', 1.0)
        adx         = data.get('ADX', 0)
        price_pct   = data.get('Price_Pct_20', 50)

        w_ema  = getattr(self, 'weight_ema',   22)
        w_macd = getattr(self, 'weight_macd',  23)
        w_stch = getattr(self, 'weight_stoch', 20)
        w_rsi  = getattr(self, 'weight_rsi',   18)
        w_vol  = getattr(self, 'weight_volume', 12)
        w_adx  = getattr(self, 'weight_adx',    5)

        # ── EMA ──────────────────────────────────────────────────────────
        if close > ema_fast > ema_mid > ema_slow:
            e_s = w_ema;                        e_l = "Perfect"
        elif ema_fast > ema_mid > ema_slow:
            e_s = round(w_ema * 0.65);          e_l = "Ordered"
        elif ema_fast > ema_mid and close > ema_fast:
            e_s = round(w_ema * 0.45);          e_l = "Partial"
        elif ema_fast > ema_slow:
            e_s = round(w_ema * 0.20);          e_l = "Weak"
        else:
            e_s = 0;                            e_l = "Bear"
        scores['ema'] = e_s
        parts.append(f"EMA={e_s}/{w_ema}({e_l})")

        # ── MACD ─────────────────────────────────────────────────────────
        m_s = 0
        if macd > macd_sig:    m_s += round(w_macd * 0.45)
        if hist_rising:        m_s += round(w_macd * 0.35)
        if macd > 0:           m_s += round(w_macd * 0.20)
        if macd_hist > 0 and macd > macd_sig and not hist_rising:
            m_s = round(m_s * 0.7)  # histogram positive but not rising — partial
        m_s = min(m_s, w_macd)
        scores['macd'] = m_s
        parts.append(f"MACD={m_s}/{w_macd}")

        # ── STOCHASTIC ────────────────────────────────────────────────────
        ob = getattr(self, 'stoch_overbought', 80)
        os_ = getattr(self, 'stoch_oversold',  20)
        mid_upper = getattr(self, 'stoch_mid_upper', 70)
        if stoch_k > stoch_d and os_ < stoch_k < mid_upper:
            st_s = w_stch;                      st_l = "Perfect"
        elif stoch_k > stoch_d and stoch_k < ob:
            st_s = round(w_stch * 0.70);        st_l = "Good"
        elif stoch_k > stoch_d:
            st_s = round(w_stch * 0.40);        st_l = "Overbought"
        elif stoch_k > os_:
            st_s = round(w_stch * 0.15);        st_l = "Neutral"
        else:
            st_s = 0;                           st_l = "Oversold"
        scores['stoch'] = st_s
        parts.append(f"STOCH={st_s}/{w_stch}({st_l})")

        # ── RSI ───────────────────────────────────────────────────────────
        rsi_min = getattr(self, 'rsi_long_min', 45)
        rsi_max = getattr(self, 'rsi_long_max', 68)
        if 55 <= rsi <= 65:   rs_s = w_rsi;                rs_l = "Prime"
        elif rsi_min <= rsi < 55: rs_s = round(w_rsi * 0.65); rs_l = "Early"
        elif 65 < rsi <= rsi_max: rs_s = round(w_rsi * 0.40); rs_l = "Late"
        elif rsi > rsi_max:   rs_s = 0;                    rs_l = "Overbought"
        else:                 rs_s = 0;                    rs_l = "TooWeak"
        scores['rsi'] = rs_s
        parts.append(f"RSI={rs_s}/{w_rsi}({rsi:.1f},{rs_l})")

        # ── VOLUME ────────────────────────────────────────────────────────
        strong = getattr(self, 'volume_strong_ratio', 1.8)
        if   volume_r >= strong:  v_s = w_vol
        elif volume_r >= 1.5:     v_s = round(w_vol * 0.80)
        elif volume_r >= 1.2:     v_s = round(w_vol * 0.60)
        elif volume_r >= 1.0:     v_s = round(w_vol * 0.35)
        elif volume_r >= 0.8:     v_s = round(w_vol * 0.15)
        else:                     v_s = 0
        scores['volume'] = v_s
        parts.append(f"Vol={v_s}/{w_vol}({volume_r:.2f}x)")

        # ── ADX ───────────────────────────────────────────────────────────
        if   adx >= 35:  a_s = round(w_adx * 0.80);  a_l = "Strong"
        elif adx >= 25:  a_s = w_adx;                 a_l = "Good"
        elif adx >= 20:  a_s = round(w_adx * 0.60);  a_l = "Forming"
        else:            a_s = 0;                     a_l = "Weak"
        scores['adx'] = a_s
        parts.append(f"ADX={a_s}/{w_adx}({adx:.1f},{a_l})")

        # ── PRICE PERCENTILE ADJUSTMENT ──────────────────────────────────
        if   price_pct < 20:  adj, pa = 12,  "Early+12"
        elif price_pct < 40:  adj, pa = 5,   "Good+5"
        elif price_pct < 65:  adj, pa = 0,   "Mid+0"
        elif price_pct < 80:  adj, pa = -6,  "Late-6"
        else:                 adj, pa = -15, "Peak-15"
        parts.append(pa)

        # ── TREND AGE PENALTY ────────────────────────────────────────────
        pen = self._trend_age_penalty('long')
        if pen > 0:
            adj -= pen
            parts.append(f"TrendAge-{pen}")

        total = max(0, min(sum(scores.values()) + adj, 100))
        return int(total), scores, " | ".join(parts)

    # ─────────────────────────────────────────────────────────────────────
    # QUALITY SCORE — SHORT  (mirror of long with bearish logic)
    # ─────────────────────────────────────────────────────────────────────
    def _quality_score_short(self, data: dict) -> Tuple[int, dict, str]:
        if self.strategy_state != StrategyState.SEEKING_ENTRY:
            raise RuntimeError("Quality score called while IN_TRADE")

        parts = []
        scores = {}

        close       = data.get('Close', 0)
        ema_fast    = data.get('EMA_Fast', 0)
        ema_mid     = data.get('EMA_Mid', 0)
        ema_slow    = data.get('EMA_Slow', 0)
        macd        = data.get('MACD', 0)       or 0
        macd_sig    = data.get('MACD_Signal', 0) or 0
        macd_hist   = data.get('MACD_Histogram', 0) or 0
        hist_rising = data.get('MACD_Hist_Rising', False)
        stoch_k     = data.get('Stoch_K', 50)
        stoch_d     = data.get('Stoch_D', 50)
        rsi         = data.get('RSI', 50)
        volume_r    = data.get('Volume_Ratio', 1.0)
        adx         = data.get('ADX', 0)
        price_pct   = data.get('Price_Pct_20', 50)

        w_ema  = getattr(self, 'weight_ema',   22)
        w_macd = getattr(self, 'weight_macd',  23)
        w_stch = getattr(self, 'weight_stoch', 20)
        w_rsi  = getattr(self, 'weight_rsi',   18)
        w_vol  = getattr(self, 'weight_volume', 12)
        w_adx  = getattr(self, 'weight_adx',    5)

        # ── EMA (BEARISH) ─────────────────────────────────────────────────
        if close < ema_fast < ema_mid < ema_slow:
            e_s = w_ema;                        e_l = "PerfectBear"
        elif ema_fast < ema_mid < ema_slow:
            e_s = round(w_ema * 0.65);          e_l = "OrderedBear"
        elif ema_fast < ema_mid and close < ema_fast:
            e_s = round(w_ema * 0.45);          e_l = "PartialBear"
        elif ema_fast < ema_slow:
            e_s = round(w_ema * 0.20);          e_l = "WeakBear"
        else:
            e_s = 0;                            e_l = "Bull"
        scores['ema'] = e_s
        parts.append(f"EMA={e_s}/{w_ema}({e_l})")

        # ── MACD (SHORT) ──────────────────────────────────────────────────
        m_s = 0
        if macd < macd_sig:             m_s += round(w_macd * 0.45)
        if not hist_rising:             m_s += round(w_macd * 0.35)
        if macd < 0:                    m_s += round(w_macd * 0.20)
        if macd_hist < 0 and macd < macd_sig and hist_rising:
            m_s = round(m_s * 0.7)
        m_s = min(m_s, w_macd)
        scores['macd'] = m_s
        parts.append(f"MACD={m_s}/{w_macd}")

        # ── STOCHASTIC (SHORT) ────────────────────────────────────────────
        ob  = getattr(self, 'stoch_overbought', 80)
        os_ = getattr(self, 'stoch_oversold',   20)
        mid_lower = getattr(self, 'stoch_mid_lower', 30)
        if stoch_k < stoch_d and mid_lower < stoch_k < ob:
            st_s = w_stch;                      st_l = "PerfectShort"
        elif stoch_k < stoch_d and stoch_k > os_:
            st_s = round(w_stch * 0.70);        st_l = "GoodShort"
        elif stoch_k < stoch_d:
            st_s = round(w_stch * 0.40);        st_l = "Oversold"
        else:
            st_s = 0;                           st_l = "Bullish"
        scores['stoch'] = st_s
        parts.append(f"STOCH={st_s}/{w_stch}({st_l})")

        # ── RSI (SHORT) ────────────────────────────────────────────────────
        rsi_min = getattr(self, 'rsi_short_min', 32)
        rsi_max = getattr(self, 'rsi_short_max', 55)
        if 38 <= rsi <= 47:  rs_s = w_rsi;                rs_l = "PrimeShort"
        elif rsi_max >= rsi > 47: rs_s = round(w_rsi * 0.60); rs_l = "LateShort"
        elif rsi_min <= rsi < 38: rs_s = round(w_rsi * 0.40); rs_l = "EarlyShort"
        elif rsi < rsi_min:  rs_s = 0;                    rs_l = "Oversold"
        else:                rs_s = 0;                    rs_l = "TooStrong"
        scores['rsi'] = rs_s
        parts.append(f"RSI={rs_s}/{w_rsi}({rsi:.1f},{rs_l})")

        # ── VOLUME ────────────────────────────────────────────────────────
        strong = getattr(self, 'volume_strong_ratio', 1.8)
        if   volume_r >= strong:  v_s = w_vol
        elif volume_r >= 1.5:     v_s = round(w_vol * 0.80)
        elif volume_r >= 1.2:     v_s = round(w_vol * 0.60)
        elif volume_r >= 1.0:     v_s = round(w_vol * 0.35)
        elif volume_r >= 0.8:     v_s = round(w_vol * 0.15)
        else:                     v_s = 0
        scores['volume'] = v_s
        parts.append(f"Vol={v_s}/{w_vol}({volume_r:.2f}x)")

        # ── ADX ───────────────────────────────────────────────────────────
        adx_min_s = getattr(self, 'adx_min_short', 22)
        if   adx >= 35:      a_s = round(w_adx * 0.80)
        elif adx >= 25:      a_s = w_adx
        elif adx >= adx_min_s: a_s = round(w_adx * 0.50)
        else:                a_s = 0
        scores['adx'] = a_s
        parts.append(f"ADX={a_s}/{w_adx}({adx:.1f})")

        # ── PRICE PERCENTILE ADJUSTMENT (SHORT — inverted) ────────────────
        if   price_pct > 80:  adj, pa = 12,  "HighShort+12"
        elif price_pct > 60:  adj, pa = 5,   "MidHigh+5"
        elif price_pct > 35:  adj, pa = 0,   "Mid+0"
        elif price_pct > 20:  adj, pa = -6,  "Low-6"
        else:                 adj, pa = -15, "TooLow-15"
        parts.append(pa)

        pen = self._trend_age_penalty('short')
        if pen > 0:
            adj -= pen
            parts.append(f"TrendAge-{pen}")

        total = max(0, min(sum(scores.values()) + adj, 100))
        return int(total), scores, " | ".join(parts)

    # ─────────────────────────────────────────────────────────────────────
    # TIER 1 HARD FILTERS — LONG
    # ─────────────────────────────────────────────────────────────────────
    def _check_long_filters(self, data: dict) -> str:
        ef  = data.get('EMA_Fast', 0)
        es  = data.get('EMA_Slow', 0)
        em  = data.get('EMA_Mid',  0)
        adx = data.get('ADX', 0)
        rsi = data.get('RSI', 50)
        mom = data.get('Momentum', 0)
        macd = data.get('MACD', 0)
        macd_s = data.get('MACD_Signal', 0)
        vr  = float(data.get('Volume_Ratio', 1.0))
        sk  = data.get('Stoch_K', 50)
        sd  = data.get('Stoch_D', 50)

        # ── Trend alignment ───────────────────────────────────────────────
        if not (ef > es):
            return "long_ema_not_bull"

        # ── ADX hard minimum ──────────────────────────────────────────────
        adx_min = getattr(self, 'adx_min_long', 20)
        if adx < adx_min:
            return f"long_adx_weak_{adx:.1f}_need_{adx_min}"

        # ── ADX not extended ──────────────────────────────────────────────
        adx_ext = getattr(self, 'adx_extended_threshold', 45)
        if adx > adx_ext:
            return f"long_adx_extended_{adx:.1f}"

        # ── RSI range ─────────────────────────────────────────────────────
        rsi_min = getattr(self, 'rsi_long_min', 45)
        rsi_max = getattr(self, 'rsi_long_max', 68)
        if not (rsi_min <= rsi <= rsi_max):
            return f"long_rsi_out_{rsi:.1f}_need_{rsi_min}-{rsi_max}"

        # ── Volume ────────────────────────────────────────────────────────
        vol_min = getattr(self, 'volume_min_ratio', 1.2)
        # Relax volume gate in strong confirmed trend
        if ef > em > es and adx >= 25:
            vol_min = min(vol_min, 0.6)
        if vr < vol_min:
            return f"long_volume_low_{vr:.2f}_need_{vol_min}"

        # ── MACD gate ─────────────────────────────────────────────────────
        if macd <= macd_s:
            return f"long_macd_below_signal"

        # ── Stochastic: K > D and not overbought ──────────────────────────
        ob = getattr(self, 'stoch_overbought', 80)
        if sk <= sd:
            return f"long_stoch_k_below_d"
        if sk > ob and sd > ob:
            return f"long_stoch_overbought_{sk:.1f}"

        # ── Momentum ──────────────────────────────────────────────────────
        mom_min = getattr(self, 'momentum_min_long', 0.01)
        if mom < mom_min:
            return f"long_momentum_weak_{mom:.3f}"

        # ── Pullback zone ─────────────────────────────────────────────────
        if ef > 0:
            dist = (data.get('Close', 0) - ef) / ef * 100
            pzl = getattr(self, 'pullback_zone_lower_pct', -1.5)
            pzu = getattr(self, 'pullback_zone_upper_pct',  0.8)
            if not (pzl <= dist <= pzu):
                return f"long_pullback_zone_{dist:.2f}pct_need_{pzl}/{pzu}"

        # ── ADX rising slope ─────────────────────────────────────────────
        adx_now  = data.get('ADX', 0)
        adx_prev = data.get('ADX_prev', adx_now)
        if (adx_now - adx_prev) < getattr(self, 'adx_slope_min', 0.05):
            return f"long_adx_not_rising"

        # ── Daily trend ───────────────────────────────────────────────────
        if not self._daily_trend_is_up(data):
            return "long_daily_trend_down"

        return "pass"

    # ─────────────────────────────────────────────────────────────────────
    # TIER 1 HARD FILTERS — SHORT
    # ─────────────────────────────────────────────────────────────────────
    def _check_short_filters(self, data: dict) -> str:
        ef  = data.get('EMA_Fast', 0)
        es  = data.get('EMA_Slow', 0)
        adx = data.get('ADX', 0)
        rsi = data.get('RSI', 50)
        mom = data.get('Momentum', 0)
        macd = data.get('MACD', 0)
        macd_s = data.get('MACD_Signal', 0)
        vr  = float(data.get('Volume_Ratio', 1.0))
        sk  = data.get('Stoch_K', 50)
        sd  = data.get('Stoch_D', 50)

        if not (ef < es):
            return "short_ema_not_bear"

        adx_min = getattr(self, 'adx_min_short', 22)
        if adx < adx_min:
            return f"short_adx_weak_{adx:.1f}_need_{adx_min}"

        adx_ext = getattr(self, 'adx_extended_threshold', 45)
        if adx > adx_ext:
            return f"short_adx_extended_{adx:.1f}"

        rsi_min = getattr(self, 'rsi_short_min', 32)
        rsi_max = getattr(self, 'rsi_short_max', 55)
        if not (rsi_min <= rsi <= rsi_max):
            return f"short_rsi_out_{rsi:.1f}_need_{rsi_min}-{rsi_max}"

        vol_min = getattr(self, 'volume_min_ratio', 1.2)
        if vr < vol_min:
            return f"short_volume_low_{vr:.2f}_need_{vol_min}"

        if macd >= macd_s:
            return f"short_macd_above_signal"

        os_ = getattr(self, 'stoch_oversold', 20)
        if sk >= sd:
            return f"short_stoch_k_above_d"
        if sk < os_ and sd < os_:
            return f"short_stoch_oversold_{sk:.1f}"

        mom_min = getattr(self, 'momentum_min_short', 0.01)
        if mom > -mom_min:
            return f"short_momentum_not_bearish_{mom:.3f}"

        if ef > 0:
            dist = (data.get('Close', 0) - ef) / ef * 100
            # Short entries want price slightly above fast EMA (bounce zone)
            if not (-1.5 <= dist <= 0.5):
                return f"short_bounce_zone_{dist:.2f}pct"

        adx_now  = data.get('ADX', 0)
        adx_prev = data.get('ADX_prev', adx_now)
        if (adx_now - adx_prev) < getattr(self, 'adx_slope_min', 0.05):
            return f"short_adx_not_rising"

        if not self._daily_trend_is_down(data):
            return "short_daily_trend_up"

        return "pass"

    # ─────────────────────────────────────────────────────────────────────
    # PENDING SIGNAL FACTORY
    # ─────────────────────────────────────────────────────────────────────
    def _create_pending_signal(self, direction: str, quality: int, tier: int,
                               component_scores: dict, breakdown: str,
                               data: dict) -> tuple:
        decision = 'buy' if direction == 'long' else 'sell'
        self._pending_signal = {
            'direction': direction, 'decision': decision,
            'quality_score': quality, 'tier': tier,
            'position_mult': 1.0, 'breakdown': breakdown,
            'component_scores': component_scores,
            'signal_price': data.get('Close', 0),
            'signal_adx': data.get('ADX', 0),
            'signal_rsi': data.get('RSI', 50),
            'signal_macd': data.get('MACD', 0),
            'signal_volume': data.get('Volume_Ratio', 1.0),
            'signal_price_pct': data.get('Price_Pct_20', 50),
            'signal_bar': self.bar_count,
            'signal_time': datetime.now(timezone.utc),
        }
        self._signal_bar   = self.bar_count
        self._signal_price = data.get('Close', 0)
        icon = "📈" if direction == 'long' else "📉"
        self._log(f"{icon} T{tier} {direction.upper()} SIGNAL @ ${self._signal_price:.4f} Q={quality} "
                  f"→ executes next bar", "purple")
        return ("hold", quality, None,
                f"{direction.upper()}_T{tier}_PENDING", component_scores)

    # ─────────────────────────────────────────────────────────────────────
    # ENTRY CONDITIONS DISPATCHER
    # ─────────────────────────────────────────────────────────────────────
    def _check_entry_conditions(self, data: dict) -> tuple:
        # ── Cooldowns & guards ────────────────────────────────────────────
        min_bars = getattr(self, 'min_bars_between_trades', 2)
        if (self.bar_count - self.last_trade_bar) < min_bars:
            return ("hold", 0, None, f"min_bar_gap", {})

        if self._is_atr_compressed(data):
            return ("hold", 0, None, "atr_compressed", {})

        if self._is_consecutive_loss_cooldown():
            return ("hold", 0, None, "consecutive_loss_cooldown", {})

        if data.get('Ranging', False) and getattr(self, 'regime_filter_enabled', True):
            return ("hold", 0, None, "market_ranging", {})

        d = self.trade_direction.lower()
        if d == 'long':   return self._entry_long(data)
        if d == 'short':  return self._entry_short(data)
        return self._entry_both(data)

    def _entry_long(self, data: dict) -> tuple:
        if self._is_extended_run_long(data):
            return ("hold", 0, None, "extended_run_long", {})

        quality, comp, bkd = self._quality_score_long(data)
        self._last_quality_score = quality
        q_min = getattr(self, 'quality_min_long', 62)
        if quality < q_min:
            if 55 <= quality < q_min:
                self._near_miss_trades.append({'quality': quality, 'direction': 'long',
                                               'bar': self.bar_count})
            return ("hold", quality, None, f"quality_low_{quality}<{q_min}", comp)

        filters = self._check_long_filters(data)
        if filters != "pass":
            return ("hold", quality, None, f"filter_failed:{filters}", comp)

        regime, _ = self.regime_detector.detect_regime(
            data.get('EMA_Fast', 0), data.get('EMA_Slow', 0),
            data.get('ADX', 0))
        self.current_regime = regime
        if not self.regime_detector.is_tradeable(regime):
            return ("hold", quality, None, f"regime_blocked_{regime}", comp)

        tier = 1 if quality >= getattr(self, 'quality_tier1_min', 75) else 2
        return self._create_pending_signal('long', quality, tier, comp, bkd, data)

    def _entry_short(self, data: dict) -> tuple:
        if self._is_extended_run_short(data):
            return ("hold", 0, None, "extended_run_short", {})

        quality, comp, bkd = self._quality_score_short(data)
        self._last_quality_score = quality
        q_min = getattr(self, 'quality_min_short', 64)
        if quality < q_min:
            return ("hold", quality, None, f"quality_low_{quality}<{q_min}", comp)

        filters = self._check_short_filters(data)
        if filters != "pass":
            return ("hold", quality, None, f"filter_failed:{filters}", comp)

        regime, _ = self.regime_detector.detect_regime(
            data.get('EMA_Fast', 0), data.get('EMA_Slow', 0),
            data.get('ADX', 0))
        self.current_regime = regime
        if not self.regime_detector.is_tradeable(regime):
            return ("hold", quality, None, f"regime_blocked_{regime}", comp)

        tier = 1 if quality >= getattr(self, 'quality_tier1_min', 75) else 2
        return self._create_pending_signal('short', quality, tier, comp, bkd, data)

    def _entry_both(self, data: dict) -> tuple:
        long_r  = self._entry_long(data)
        short_r = self._entry_short(data)

        ld = long_r[0]
        sd = short_r[0]

        if ld == "hold" and sd == "hold":
            return ("hold", 0, None, "no_clear_direction", {})

        if ld != "hold" and sd != "hold":
            # Tie-break: higher quality wins; EMA trend as secondary
            lq = long_r[1];  sq = short_r[1]
            if abs(lq - sq) > 5:
                return long_r if lq > sq else short_r
            ef = data.get('EMA_Fast', 0);  es = data.get('EMA_Slow', 0)
            return long_r if ef > es else short_r

        return long_r if ld != "hold" else short_r

    # ─────────────────────────────────────────────────────────────────────
    # POSITION SIZE
    # ─────────────────────────────────────────────────────────────────────
    def calculate_position_size(self, equity: float, atr: float, price: float,
                                quality_score: int = 70, tier: int = 1,
                                position_mult: float = 1.0) -> float:
        stop_dist  = atr * getattr(self, 'stop_loss_atr_mult', 1.5)
        stop_price = price - stop_dist
        adx = 25.0
        if self._current_df is not None:
            try: adx = float(self._current_df['ADX'].iloc[-1])
            except: pass

        base = self.risk_controller.calculate_position_size(
            price, stop_price, quality_score, adx)

        regime_mult = self.regime_detector.get_position_multiplier(self.current_regime)
        raw = base * regime_mult * position_mult

        min_units = (equity * 0.001) / price if price > 0 else 0
        size = max(min_units, raw)

        max_units = getattr(self, 'max_position_units', 100)
        size = min(size, max_units)

        if price >= 1000: return round(size, 6)
        elif price >= 100: return round(size, 4)
        else: return max(0, int(size))

    # ─────────────────────────────────────────────────────────────────────
    # TRADE RECORDING
    # ─────────────────────────────────────────────────────────────────────
    def record_trade(self, profit: float, exit_reason: str = "unknown",
                     tier: int = None, size: float = None, direction: str = None,
                     entry_quality: int = None, entry_price: float = None,
                     exit_price: float = None, hold_duration: float = None,
                     entry_bar: int = None, exit_bar: int = None, **kwargs):
        self.total_trades += 1
        self.total_profit += profit
        if profit > 0:
            self.winning_trades += 1
            self._consecutive_loss_count = 0
        else:
            self.losing_trades += 1
            self._consecutive_loss_count += 1
            self._last_loss_bar = self.bar_count

        self.last_trade_bar = self.bar_count

        self.trade_history.append({
            'profit': profit, 'exit_reason': exit_reason,
            'tier': tier, 'size': size, 'direction': direction,
            'entry_quality': entry_quality, 'entry_price': entry_price,
            'exit_price': exit_price, 'hold_duration': hold_duration,
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'timestamp': datetime.now(timezone.utc), **kwargs})
        if len(self.trade_history) > 200:
            self.trade_history = self.trade_history[-200:]

    def get_performance_stats(self) -> dict:
        n = self.total_trades
        if n == 0:
            return {'total_trades': 0, 'win_rate': 0, 'total_profit': 0}
        wr = self.winning_trades / n * 100
        wins   = [t['profit'] for t in self.trade_history if t['profit'] > 0]
        losses = [t['profit'] for t in self.trade_history if t['profit'] < 0]
        return {
            'total_trades'   : n,
            'win_rate'       : wr,
            'total_profit'   : self.total_profit,
            'avg_profit'     : self.total_profit / n,
            'avg_win'        : float(np.mean(wins))        if wins   else 0,
            'avg_loss'       : float(abs(np.mean(losses))) if losses else 0,
            'winning_trades' : self.winning_trades,
            'losing_trades'  : self.losing_trades,
        }


# ═══════════════════════════════════════════════════════════════════════════
# LIVE STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class ScalpingStrategy(BaseStrategy, ScalpingLogic):
    """v1.0 Live Scalping — 15-min bars, EMA/MACD/Stoch momentum bursts."""

    def __init__(self, trading_app=None):
        BaseStrategy.__init__(self, trading_app)
        override = None
        if trading_app and hasattr(trading_app, 'get_current_momentum_params'):
            override = trading_app.get_current_momentum_params()
        params = ScalpingConfig.get_config(override)
        ScalpingLogic.__init__(self, config=params, trading_app=trading_app)

        self.name = "Professional Scalping Strategy v1.0 — EMA|MACD|STOCH|ATR"
        self.position = {
            'type': None, 'entry_price': None, 'quantity': None,
            'stop_loss': None, 'trailing_stop': None,
            'trailing_activated': False, 'highest_price': None,
            'lowest_price': None, 'entry_bar': None, 'partial_exits': 0,
            'original_quantity': None, 'tier': None, 'entry_time': None,
            'entry_quality_score': None, 'entry_reason': None,
            'trade_id': None, 'partial_pnl_realised': 0.0,
        }
        self.bars_held   = 0
        self.trade_counter = 0

        if self.trading_app:
            self._log("=" * 70, "cyan")
            self._log("SCALPING STRATEGY v1.0 — LIVE", "bold green")
            self._log(f"  ✅ EMA {params['ema_fast_period']}/{params['ema_mid_period']}/{params['ema_slow_period']}", "green")
            self._log(f"  ✅ MACD ({params['macd_fast']},{params['macd_slow']},{params['macd_signal_period']})", "green")
            self._log(f"  ✅ Stoch ({params['stoch_k_period']},{params['stoch_smooth']},{params['stoch_d_period']})", "green")
            self._log(f"  ✅ Stop: {params['stop_loss_atr_mult']}× ATR | Trail activates @ {params['trailing_activation_pct']:.1%}", "green")
            self._log(f"  ✅ Max hold: {params['max_hold_bars']} bars  |  Quality min: {params['quality_min_long']}", "green")
            self._log("=" * 70, "cyan")

    def run_analysis_cycle(self, current_data, current_price, df=None):
        self._current_df = df
        if self.strategy_state == StrategyState.SEEKING_ENTRY:
            return self.check_entry_conditions(current_data)
        return self.check_exit_conditions(current_data, current_price)

    def check_entry_conditions(self, current_data):
        if self.strategy_state != StrategyState.SEEKING_ENTRY:
            return "hold", POSITION_ALREADY_OPEN_SENTINEL, 0, "in_trade"
        return self._check_entry_conditions(current_data)

    def check_exit_conditions(self, current_data, current_price):
        if self.strategy_state != StrategyState.IN_TRADE:
            return None, 1.0

        if self.position['type'] == 'long':
            self.position['highest_price'] = max(
                self.position.get('highest_price') or current_price, current_price)
        else:
            lp = self.position.get('lowest_price')
            self.position['lowest_price'] = min(lp, current_price) if lp is not None else current_price

        atr = current_data.get('ATR')
        if not atr or atr <= 0:
            raise ValueError(f"Invalid ATR at bar {self.bars_held}: {atr}")

        # ── Trailing stop update ──────────────────────────────────────────
        ta_pct = max((atr / current_price) * 1.5,
                     getattr(self, 'trailing_activation_pct', 0.008))
        td_pct = max((atr / current_price) * 0.8,
                     getattr(self, 'trailing_distance_pct', 0.006))

        if self.position['type'] == 'long':
            profit_pct = (current_price - self.position['entry_price']) / self.position['entry_price']
            if not self.position['trailing_activated'] and profit_pct >= ta_pct:
                self.position['trailing_activated'] = True
                self.position['trailing_stop'] = current_price * (1 - td_pct)
            if self.position['trailing_activated']:
                new_stop = self.position['highest_price'] * (1 - td_pct)
                if new_stop > (self.position['trailing_stop'] or 0):
                    self.position['trailing_stop'] = new_stop
        else:
            profit_pct = (self.position['entry_price'] - current_price) / self.position['entry_price']
            if not self.position['trailing_activated'] and profit_pct >= ta_pct:
                self.position['trailing_activated'] = True
                self.position['trailing_stop'] = current_price * (1 + td_pct)
            if self.position['trailing_activated']:
                new_stop = self.position['lowest_price'] * (1 + td_pct)
                if new_stop < (self.position['trailing_stop'] or float('inf')):
                    self.position['trailing_stop'] = new_stop

        return self.exit_manager.evaluate_exit(
            current_price    = current_price,
            entry_price      = self.position['entry_price'],
            stop_loss        = self.position['stop_loss'],
            highest_price    = self.position.get('highest_price', current_price),
            lowest_price     = self.position.get('lowest_price', current_price),
            bars_held        = self.bars_held,
            partial_exits    = self.position.get('partial_exits', 0),
            ema_fast         = current_data.get('EMA_Fast', 0),
            ema_mid          = current_data.get('EMA_Mid',  0),
            ema_slow         = current_data.get('EMA_Slow', 0),
            macd             = current_data.get('MACD', 0),
            macd_signal      = current_data.get('MACD_Signal', 0),
            macd_prev        = current_data.get('MACD_closed', 0),
            signal_prev      = current_data.get('MACD_Signal_closed', 0),
            stoch_k          = current_data.get('Stoch_K', 50),
            stoch_d          = current_data.get('Stoch_D', 50),
            rsi              = current_data.get('RSI', 50),
            adx              = current_data.get('ADX', 0),
            atr              = atr,
            position_type    = self.position['type'],
            trailing_activated = self.position.get('trailing_activated', False),
            trailing_stop    = self.position.get('trailing_stop'))

    def calculate_indicators(self, df):
        return IndicatorCalculator.calculate(df, self.config)

    def on_bar_update(self, current_equity: float):
        self.equity_curve.append(current_equity)
        self.bar_count += 1
        if self.strategy_state == StrategyState.IN_TRADE:
            self.bars_held += 1
        self.risk_controller.current_equity = current_equity
        if current_equity > self.risk_controller.peak_equity:
            self.risk_controller.peak_equity = current_equity

    def get_strategy_stats(self) -> dict:
        stats = self.get_performance_stats()
        risk  = self.risk_controller.get_stats()
        return {
            'strategy_name'  : self.name,
            'total_trades'   : stats['total_trades'],
            'win_rate'       : stats.get('win_rate', 0),
            'total_profit'   : stats.get('total_profit', 0),
            'profit_factor'  : risk['profit_factor'],
            'max_drawdown'   : risk['max_drawdown'],
            'sharpe_ratio'   : risk['sharpe_ratio'],
            'current_regime' : self.current_regime,
            'strategy_state' : self.strategy_state.name,
            'trade_direction': self.trade_direction,
        }


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class BacktestScalpingStrategy(Strategy, ScalpingLogic):
    """v1.0 Backtest — pending-signal execution on next bar open."""

    _use_updated_params = False
    _updated_params     = {}

    @classmethod
    def set_updated_params(cls, params: dict):
        if params and isinstance(params, dict):
            cls._use_updated_params = True
            cls._updated_params     = params.copy()
        else:
            cls._use_updated_params = False
            cls._updated_params     = {}

    @classmethod
    def reset_to_defaults(cls):
        cls._use_updated_params = False
        cls._updated_params     = {}

    # Expose all SCALPING_PARAMS as backtesting.py optimisable attributes
    for _k in SCALPING_PARAMS:
        locals()[_k] = None

    def __init__(self, broker, data, params):
        Strategy.__init__(self, broker, data, params)

        override = self.__class__._updated_params if self.__class__._use_updated_params else {}
        config   = ScalpingConfig.get_config(override)
        if params:
            for k, v in params.items():
                if k in config:
                    config[k] = v
        for k, v in config.items():
            setattr(self, k, v)

        ScalpingLogic.__init__(self, config=config, trading_app=None)

        print(f"\n{'=' * 70}")
        print("BACKTEST SCALPING STRATEGY v1.0")
        print(f"{'=' * 70}")
        print(f"Direction  : {self.trade_direction.upper()}")
        print(f"EMA        : {config['ema_fast_period']}/{config['ema_mid_period']}/{config['ema_slow_period']}")
        print(f"MACD       : ({config['macd_fast']},{config['macd_slow']},{config['macd_signal_period']})")
        print(f"Stoch      : ({config['stoch_k_period']},{config['stoch_smooth']},{config['stoch_d_period']})")
        print(f"Stop mult  : {config['stop_loss_atr_mult']}× ATR")
        print(f"Max hold   : {config['max_hold_bars']} bars")
        print(f"Quality min: {config['quality_min_long']} (L) / {config['quality_min_short']} (S)")
        print(f"{'=' * 70}\n")

        self._entry_price          = np.nan
        self._stop_loss            = np.nan
        self._highest_price        = np.nan
        self._lowest_price         = np.nan
        self._trailing_activated   = False
        self._trailing_stop        = None
        self._be_stop_set          = False
        self._bars_held            = 0
        self._partial_exits        = 0
        self._entry_bar            = -999
        self._entry_tier           = None
        self._entry_quality        = 0
        self._partial_pnl_realised = 0.0
        self._position_direction   = config.get('trade_direction', 'long')
        if self._position_direction == 'both':
            self._position_direction = 'long'
        self._exit_reason_map      = {}
        self._pending_signal       = None
        self._signal_bar           = -999

        # Bar interval detection for display purposes
        try:
            delta = (data.df.index[1] - data.df.index[0]).total_seconds() / 60
            self._bar_interval_minutes = delta
        except Exception:
            self._bar_interval_minutes = float(config.get('bar_interval_minutes', 15))

    # ─────────────────────────────────────────────────────────────────────
    def _bt_safe_size(self, units: float, price: float) -> float:
        """Convert units to a backtesting.py-safe size value."""
        if units <= 0:
            return 0
        if units >= 1 and units == round(units):
            return int(units)
        fraction = (units * price) / max(self.equity, 1.0)
        return max(0.0001, min(fraction, 0.9999))

    # ─────────────────────────────────────────────────────────────────────
    def init(self):
        df = IndicatorCalculator.calculate(self.data.df.copy(), self.config)
        self.df_indicators = df
        self.df_enhanced   = df.copy()

        def _I(col, default=0.0):
            vals = df[col].fillna(default).values if col in df.columns else np.full(len(df), default)
            return self.I(lambda v=vals: v, name=col)

        self.i_ema_fast    = _I('EMA_Fast')
        self.i_ema_mid     = _I('EMA_Mid')
        self.i_ema_slow    = _I('EMA_Slow')
        self.i_ema_daily   = _I('EMA_Daily')
        self.i_above_daily = _I('Above_Daily', 1.0)
        self.i_macd        = _I('MACD')
        self.i_macd_sig    = _I('MACD_Signal')
        self.i_macd_hist   = _I('MACD_Histogram')
        self.i_stoch_k     = _I('Stoch_K',  50.0)
        self.i_stoch_d     = _I('Stoch_D',  50.0)
        self.i_adx         = _I('ADX')
        self.i_adx_prev    = _I('ADX_prev')
        self.i_rsi         = _I('RSI',  50.0)
        self.i_volume_r    = _I('Volume_Ratio', 1.0)
        self.i_atr         = _I('ATR', 1.0)
        self.i_momentum    = _I('Momentum')
        self.i_ranging     = _I('Ranging',  0.0)
        self.i_price_pct   = _I('Price_Pct_20', 50.0)

    # ─────────────────────────────────────────────────────────────────────
    def _build_current_data(self) -> dict:
        def sv(arr, d=0.0):
            try:
                v = arr[-1]
                return d if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            except: return d

        def sb(arr, d=True):
            try:
                v = arr[-1]
                if v is None or (isinstance(v, float) and np.isnan(v)): return d
                return bool(v)
            except: return d

        return {
            'Close'              : float(self.data.Close[-1]),
            'EMA_Fast'           : sv(self.i_ema_fast),
            'EMA_Mid'            : sv(self.i_ema_mid),
            'EMA_Slow'           : sv(self.i_ema_slow),
            'ADX'                : sv(self.i_adx),
            'ADX_prev'           : sv(self.i_adx_prev),
            'RSI'                : sv(self.i_rsi, 50.0),
            'Volume_Ratio'       : sv(self.i_volume_r, 1.0),
            'Momentum'           : sv(self.i_momentum),
            'ATR'                : sv(self.i_atr, 1.0),
            'MACD'               : sv(self.i_macd),
            'MACD_Signal'        : sv(self.i_macd_sig),
            'MACD_Histogram'     : sv(self.i_macd_hist),
            'MACD_Hist_Rising'   : (float(self.i_macd_hist[-1]) > float(self.i_macd_hist[-2]))
                                   if len(self.i_macd_hist) > 1 else False,
            'MACD_closed'        : float(self.i_macd[-2])     if len(self.i_macd)     > 1 and not np.isnan(self.i_macd[-2])     else 0.0,
            'MACD_Signal_closed' : float(self.i_macd_sig[-2]) if len(self.i_macd_sig) > 1 and not np.isnan(self.i_macd_sig[-2]) else 0.0,
            'Stoch_K'            : sv(self.i_stoch_k,  50.0),
            'Stoch_D'            : sv(self.i_stoch_d,  50.0),
            'Ranging'            : sb(self.i_ranging, False),
            'Price_Pct_20'       : sv(self.i_price_pct, 50.0),
            'Above_Daily'        : sb(self.i_above_daily, True),
        }

    # ─────────────────────────────────────────────────────────────────────
    def _bt_close_position(self, current_price: float, exit_signal: str, profit_pct: float):
        size_at_close = abs(self.position.size)
        self._exit_reason_map[self._entry_bar] = exit_signal
        self.position.close()

        if self._position_direction == 'long':
            final_leg = (current_price - self._entry_price) * size_at_close
        else:
            final_leg = (self._entry_price - current_price) * size_at_close
        total_profit = final_leg + self._partial_pnl_realised

        self.record_trade(
            profit=total_profit, exit_reason=exit_signal,
            tier=self._entry_tier, size=size_at_close,
            direction=self._position_direction,
            entry_quality=self._entry_quality,
            entry_price=self._entry_price, exit_price=current_price,
            hold_duration=self._bars_held,
            entry_bar=self._entry_bar, exit_bar=len(self.data) - 1)

        icon = "⬆️" if self._position_direction == 'long' else "⬇️"
        win  = "✅" if total_profit > 0 else "❌"
        print(f"{win} {icon} EXIT @ ${current_price:.4f} {profit_pct:+.2f}% "
              f"hold={self._bars_held}bars reason={exit_signal} → SEEKING")

        self._entry_price  = np.nan; self._stop_loss = np.nan
        self._highest_price = np.nan; self._lowest_price = np.nan
        self._bars_held    = 0; self._partial_exits = 0
        self._partial_pnl_realised = 0.0
        self._trailing_activated   = False; self._trailing_stop = None
        self._be_stop_set  = False
        self._position_direction = 'long'
        self._transition_to_seeking_entry()

    # ─────────────────────────────────────────────────────────────────────
    def next(self):
        # Guard: wait for all indicators to warm up
        try:
            if any(np.isnan(x[-1]) for x in [
                self.i_ema_fast, self.i_adx, self.i_rsi,
                self.i_macd, self.i_atr, self.i_stoch_k
            ]):
                return
        except Exception:
            return

        idx           = len(self.data) - 1
        current_data  = self._build_current_data()
        current_price = float(self.data.Close[-1])
        self._current_df = self.df_enhanced.iloc[:idx + 1]
        self.bar_count   = idx

        # ── Execute pending signal on next bar's open ─────────────────────
        if self._pending_signal is not None and self.bar_count > self._signal_bar:
            sig            = self._pending_signal
            exec_price     = float(self.data.Open[-1])
            self._position_direction = 'long' if sig['decision'] == 'buy' else 'short'

            size = self.calculate_position_size(
                self.equity, current_data['ATR'], exec_price,
                sig['quality_score'], sig['tier'], sig['position_mult'])

            if size > 0:
                stop_dist = getattr(self, 'stop_loss_atr_mult', 1.5) * current_data['ATR']
                if self._position_direction == 'long':
                    stop = exec_price - stop_dist
                    if stop < exec_price:
                        self.buy(size=self._bt_safe_size(size, exec_price))
                    else:
                        self._pending_signal = None; return
                else:
                    stop = exec_price + stop_dist
                    if stop > exec_price:
                        self.sell(size=self._bt_safe_size(size, exec_price))
                    else:
                        self._pending_signal = None; return

                self._entry_price          = exec_price
                self._stop_loss            = stop
                self._highest_price        = exec_price if self._position_direction == 'long' else None
                self._lowest_price         = exec_price if self._position_direction == 'short' else None
                self._bars_held            = 0
                self._partial_exits        = 0
                self._entry_bar            = idx
                self._entry_tier           = sig['tier']
                self._entry_quality        = sig['quality_score']
                self._partial_pnl_realised = 0.0
                self._trailing_activated   = False
                self._trailing_stop        = None
                self._be_stop_set          = False
                self._transition_to_in_trade()

                icon = "⬆️" if self._position_direction == 'long' else "⬇️"
                print(f"{icon} ENTER T{sig['tier']} Q={sig['quality_score']} "
                      f"@ ${exec_price:.4f} Stop=${stop:.4f} Size={size}")

            self._pending_signal = None
            self._signal_bar     = -999
            return

        # ══════════════════════════════════════════════════════════════════
        # SEEKING ENTRY
        # ══════════════════════════════════════════════════════════════════
        if self.strategy_state == StrategyState.SEEKING_ENTRY:
            self._check_entry_conditions(current_data)
            return

        # ══════════════════════════════════════════════════════════════════
        # IN TRADE — MANAGE POSITION
        # ══════════════════════════════════════════════════════════════════
        if self.strategy_state == StrategyState.IN_TRADE:
            self._bars_held += 1

            # Track high/low
            if self._position_direction == 'long':
                if self._highest_price is None or current_price > self._highest_price:
                    self._highest_price = current_price
            else:
                if self._lowest_price is None or current_price < self._lowest_price:
                    self._lowest_price = current_price

            atr = current_data.get('ATR', 0)
            if atr <= 0:
                return

            # ── Trailing stop ─────────────────────────────────────────────
            atr_pct = atr / current_price if current_price > 0 else 0.001
            ta_pct  = max(atr_pct * 1.5, getattr(self, 'trailing_activation_pct', 0.008))
            td_pct  = max(atr_pct * 0.8, getattr(self, 'trailing_distance_pct', 0.006))

            if self._position_direction == 'long':
                if not self._trailing_activated:
                    pp = (current_price - self._entry_price) / self._entry_price
                    if pp >= ta_pct:
                        self._trailing_activated = True
                        self._trailing_stop = current_price * (1 - td_pct)
                        print(f"🔒 TRAILING ON @ {pp:.2%} stop=${self._trailing_stop:.4f}")
                if self._trailing_activated:
                    ns = self._highest_price * (1 - td_pct)
                    if ns > (self._trailing_stop or 0):
                        self._trailing_stop = ns
            else:
                if not self._trailing_activated:
                    pp = (self._entry_price - current_price) / self._entry_price
                    if pp >= ta_pct:
                        self._trailing_activated = True
                        self._trailing_stop = current_price * (1 + td_pct)
                        print(f"🔒 TRAILING ON (S) @ {pp:.2%} stop=${self._trailing_stop:.4f}")
                if self._trailing_activated:
                    ns = self._lowest_price * (1 + td_pct)
                    if ns < (self._trailing_stop or float('inf')):
                        self._trailing_stop = ns

            # ── Breakeven stop ────────────────────────────────────────────
            if getattr(self, 'be_stop_enabled', True) and not self._be_stop_set:
                sd = abs(self._entry_price - self._stop_loss)
                if sd > 0:
                    be_r  = getattr(self, 'be_stop_r_trigger', 1.5)
                    be_b  = getattr(self, 'be_stop_no_progress_bars', 15)
                    if self._position_direction == 'long':
                        pa = current_price - self._entry_price
                        if pa >= be_r * sd:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BE STOP @ {pa/sd:.1f}R")
                        elif self._bars_held >= be_b and pa / self._entry_price < 0.002:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BE STOP (no progress, {self._bars_held}b)")
                    else:
                        pa = self._entry_price - current_price
                        if pa >= be_r * sd:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BE STOP (S) @ {pa/sd:.1f}R")
                        elif self._bars_held >= be_b and pa / self._entry_price < 0.002:
                            self._stop_loss = self._entry_price
                            self._be_stop_set = True
                            print(f"🔒 BE STOP (S, no progress)")

            # ── Evaluate exit ─────────────────────────────────────────────
            exit_signal, exit_pct = self.exit_manager.evaluate_exit(
                current_price   = current_price,
                entry_price     = self._entry_price,
                stop_loss       = self._stop_loss,
                highest_price   = self._highest_price,
                lowest_price    = self._lowest_price,
                bars_held       = self._bars_held,
                partial_exits   = self._partial_exits,
                ema_fast        = current_data['EMA_Fast'],
                ema_mid         = current_data['EMA_Mid'],
                ema_slow        = current_data['EMA_Slow'],
                macd            = current_data['MACD'],
                macd_signal     = current_data['MACD_Signal'],
                macd_prev       = current_data['MACD_closed'],
                signal_prev     = current_data['MACD_Signal_closed'],
                stoch_k         = current_data['Stoch_K'],
                stoch_d         = current_data['Stoch_D'],
                rsi             = current_data['RSI'],
                adx             = current_data['ADX'],
                atr             = atr,
                position_type   = self._position_direction,
                trailing_activated = self._trailing_activated,
                trailing_stop   = self._trailing_stop)

            # ── Safety hard stop check ────────────────────────────────────
            if exit_signal is None:
                if self._position_direction == 'long' and current_price <= self._stop_loss:
                    exit_signal = "stop_loss_hard"; exit_pct = 1.0
                elif self._position_direction == 'short' and current_price >= self._stop_loss:
                    exit_signal = "stop_loss_hard"; exit_pct = 1.0

            # ── Safety trailing check ─────────────────────────────────────
            if (exit_signal is None and self._trailing_activated
                    and self._trailing_stop is not None):
                if self._position_direction == 'long' and current_price <= self._trailing_stop:
                    exit_signal = "trailing_stop"; exit_pct = 1.0
                elif self._position_direction == 'short' and current_price >= self._trailing_stop:
                    exit_signal = "trailing_stop"; exit_pct = 1.0

            # ── Force close on last bar ───────────────────────────────────
            total_bars  = len(self.df_enhanced)
            is_last_bar = (idx == total_bars - 1)
            if exit_signal is None and is_last_bar and self._bars_held > 0:
                exit_signal = "end_of_backtest"; exit_pct = 1.0
                print("🏁 FORCED EXIT — end of backtest data")

            # ── Execute exit ──────────────────────────────────────────────
            if exit_signal:
                if self._position_direction == 'long':
                    pct = (current_price - self._entry_price) / self._entry_price * 100
                else:
                    pct = (self._entry_price - current_price) / self._entry_price * 100

                # Partial exit handling
                if exit_pct < 1.0 and self._partial_exits < 3:
                    self._partial_exits += 1
                    raw_p = abs(self.position.size) * exit_pct
                    ps    = self._bt_safe_size(raw_p, current_price)
                    if ps > 0:
                        if self._position_direction == 'long':
                            self.sell(size=ps)
                            pp = (current_price - self._entry_price) * raw_p
                        else:
                            self.buy(size=ps)
                            pp = (self._entry_price - current_price) * raw_p
                        self._partial_pnl_realised += pp
                        print(f"📊 PARTIAL EXIT {self._partial_exits} @ ${current_price:.4f} P&L=${pp:.2f}")
                        return

                self._bt_close_position(current_price, exit_signal, pct)


# ═══════════════════════════════════════════════════════════════════════════
# END OF FILE
# ═══════════════════════════════════════════════════════════════════════════
