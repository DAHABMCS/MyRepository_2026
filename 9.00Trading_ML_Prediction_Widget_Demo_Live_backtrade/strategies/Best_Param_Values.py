#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
BEST PARAMETER VALUES v7.1 — PRIORITY-BASED OPTIMIZER (FIXED VERIFICATION)
================================================================================

A complete standalone GUI application for strategy parameter optimization with
dynamic priority selection.

FIXES:
    - Verification now properly validates selected parameters against priority hierarchy
    - Shows CRITICAL params: selected ✅ or missing ❌
    - Shows HIGH priority as "Recommended" (not critical)
    - Shows MEDIUM priority as "Optional"
    - Flags any selected params not in priority tree

FEATURES:
    - Select strategy: Momentum, Kalman, Scalping
    - Select optimization metric: Sharpe, Return, Win Rate, etc.
    - Configure symbol, timeframe, date range
    - Dynamic priority tree: CRITICAL / HIGH / MEDIUM
    - Select individual parameters with checkboxes
    - Web verification button validates against benchmarks
    - Run optimization with selected parameters only
    - Auto-update strategy_settings.json
    - Export results to JSON

USAGE:
    python Best_Param_Values.py

OR from the app:
    Click "🏆 Best Params" button
================================================================================
"""

import os
import sys
import json
import argparse
import warnings
import time
import threading
import re
import webbrowser
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

# ─── Suppress warnings ──────────────────────────────────────────────────────
warnings.filterwarnings('ignore')

# ─── Environment setup ─────────────────────────────────────────────────────
os.environ['TA_LIBRARY'] = 'talib'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BACKTESTING_DISABLE_MULTIPROCESSING'] = '1'
os.environ['TQDM_DISABLE'] = '1'

# ─── Path setup ────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ─── Imports ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import ccxt
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# ─── Try to import OpenAI for AI advisor ──────────────────────────────────
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# 1. PARAMETER DESCRIPTIONS
# ═══════════════════════════════════════════════════════════════════════════

PARAM_DESCRIPTIONS = {
    # ─── Momentum Strategy ──────────────────────────────────────────────────
    "quality_tier1_min_long": "Minimum quality score for Tier 1 long entries (0-100)",
    "quality_tier2_min_long": "Minimum quality score for Tier 2 long entries",
    "quality_tier1_min_short": "Minimum quality score for Tier 1 short entries",
    "quality_tier2_min_short": "Minimum quality score for Tier 2 short entries",
    "tier1_adx_hard_min_long": "Minimum ADX for long entries",
    "tier1_adx_hard_min_short": "Minimum ADX for short entries",
    "tier2_adx_hard_min": "Minimum ADX for Tier 2 entries",
    "tier1_rsi_min_long": "Minimum RSI for long entries",
    "tier1_rsi_max_long": "Maximum RSI for long entries",
    "tier1_rsi_min_short": "Minimum RSI for short entries",
    "tier1_rsi_max_short": "Maximum RSI for short entries",
    "tier2_rsi_min_long": "Minimum RSI for Tier 2 long entries",
    "tier2_rsi_max_long": "Maximum RSI for Tier 2 long entries",
    "tier1_volume_min_long": "Minimum volume ratio for long entries",
    "tier1_volume_min_short": "Minimum volume ratio for short entries",
    "tier2_volume_min": "Minimum volume ratio for Tier 2 entries",
    "tier1_momentum_min": "Minimum price momentum for entries",
    "tier2_momentum_min": "Minimum price momentum for Tier 2 entries",
    "tier1_confluence_min": "Minimum confluence score for Tier 1",
    "tier2_confluence_min": "Minimum confluence score for Tier 2",
    "risk_tier1": "Risk percentage for Tier 1 trades",
    "risk_tier2": "Risk percentage for Tier 2 trades",
    "stop_loss_atr_mult": "ATR multiplier for stop loss",
    "trailing_activation_tier1": "Profit % to activate trailing stop for Tier 1",
    "trailing_activation_tier2": "Profit % to activate trailing stop for Tier 2",
    "trailing_distance_tier1": "Trailing stop distance from peak for Tier 1",
    "trailing_distance_tier2": "Trailing stop distance from peak for Tier 2",
    "exit_threshold_tier1": "Exit power score threshold for Tier 1",
    "exit_threshold_tier2": "Exit power score threshold for Tier 2",
    "weight_ema": "Weight for EMA component in quality score",
    "weight_adx": "Weight for ADX component in quality score",
    "weight_macd": "Weight for MACD component in quality score",
    "weight_rsi": "Weight for RSI component in quality score",
    "weight_volume": "Weight for Volume component in quality score",
    "ema_fast_period": "Fast EMA period",
    "ema_mid_period": "Middle EMA period",
    "ema_slow_period": "Slow EMA period",
    "min_bars_between_trades": "Minimum bars between trades",
    "be_stop_r_trigger": "R-multiple at which to move stop to breakeven",
    "take_profit_r1": "R-multiple for first profit target",
    "take_profit_r2": "R-multiple for second profit target",
    "take_profit_r3": "R-multiple for third profit target",
    "only_tier1_entries": "When True, only Tier 1 entries are allowed",
    "trade_direction": "Trading direction: 'long', 'short', or 'both'",
    "ml_weight": "ML prediction weight factor",

    # ─── Kalman Strategy ────────────────────────────────────────────────────
    "process_noise_1": "Kalman filter process noise 1",
    "process_noise_2": "Kalman filter process noise 2",
    "measurement_noise": "Kalman filter measurement noise",
    "trend_lookback": "Lookback period for trend calculation",
    "strength_smooth": "Smoothing period for trend strength",
    "long_kalman_strength_min": "Minimum Kalman strength for long entries",
    "short_kalman_strength_min": "Minimum Kalman strength for short entries",
    "long_rsi_min": "Minimum RSI for long entries",
    "long_rsi_max": "Maximum RSI for long entries",
    "short_rsi_min": "Minimum RSI for short entries",
    "short_rsi_max": "Maximum RSI for short entries",
    "stop_loss_pct": "Stop loss percentage",
    "trailing_stop_pct": "Trailing stop percentage",
    "atr_multiplier": "ATR multiplier for dynamic stops",
    "risk_reward": "Risk/reward ratio target",
    "risk_per_trade": "Risk percentage per trade",
    "max_position_pct": "Maximum position as percentage of equity",
    "volume_min_ratio": "Minimum volume ratio for entries",
    "min_adx": "Minimum ADX for entries",
    "pullback_percent": "Pullback percentage for long entries",
    "rally_percent": "Rally percentage for short entries",
    "max_hold_bars": "Maximum bars to hold a position",
    "cooldown_bars": "Cooldown bars between trades",

    # ─── Scalping Strategy ──────────────────────────────────────────────────
    "quality_min_long": "Minimum quality score for long entries",
    "quality_min_short": "Minimum quality score for short entries",
    "quality_tier1_min": "Tier 1 quality threshold for scalping",
    "quality_tier2_min": "Tier 2 quality threshold for scalping",
    "rsi_long_min": "Minimum RSI for long entries",
    "rsi_long_max": "Maximum RSI for long entries",
    "rsi_short_min": "Minimum RSI for short entries",
    "rsi_short_max": "Maximum RSI for short entries",
    "adx_min_long": "Minimum ADX for long entries",
    "adx_min_short": "Minimum ADX for short entries",
    "adx_extended_threshold": "ADX threshold for extended trend blocking",
    "volume_min_ratio": "Minimum volume ratio",
    "volume_strong_ratio": "Strong volume confirmation threshold",
    "trailing_activation_pct": "Profit % to activate trailing stop",
    "trailing_distance_pct": "Trailing stop distance from peak",
    "take_profit_r1": "First profit target (R-multiple)",
    "take_profit_r2": "Second profit target (R-multiple)",
    "partial_exit_pct_r1": "Percentage to exit at first target",
    "partial_exit_pct_r2": "Percentage to exit at second target",
    "pullback_zone_lower_pct": "Lower pullback zone for long entries",
    "pullback_zone_upper_pct": "Upper pullback zone for long entries",
    "momentum_min_long": "Minimum momentum for long entries",
    "momentum_min_short": "Minimum momentum for short entries",
    "trend_age_min_bars": "Minimum trend age in bars",
    "max_daily_trades": "Maximum trades per day",
    "macd_cross_exit_enabled": "Enable MACD cross exit signal",
    "ema_cross_exit_enabled": "Enable EMA cross exit signal",
    "stoch_reversal_exit_enabled": "Enable Stochastic reversal exit",
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. PRIORITY ENGINE - DYNAMIC RULE-BASED (NO AI REQUIRED)
# ═══════════════════════════════════════════════════════════════════════════

class PriorityEngine:
    """Dynamically calculates parameter priorities based on strategy, metric, direction, and tier."""

    # Strategy-specific priority definitions
    STRATEGY_PRIORITIES = {
        "momentum": {
            "Win Rate [%]": {
                "CRITICAL": [
                    {"name": "quality_tier2_min_long", "tier": 2, "desc": "Primary entry gate for longs"},
                    {"name": "quality_tier2_min_short", "tier": 2, "desc": "Primary entry gate for shorts"},
                    {"name": "tier2_confluence_min", "tier": 2, "desc": "Multi-signal confirmation"},
                    {"name": "tier2_adx_hard_min", "tier": 2, "desc": "Trend strength filter"},
                    {"name": "tier2_momentum_min", "tier": 2, "desc": "Momentum threshold"}
                ],
                "HIGH": [
                    {"name": "quality_tier1_min_long", "tier": 1, "desc": "Tier 1 quality gate"},
                    {"name": "quality_tier1_min_short", "tier": 1, "desc": "Tier 1 quality gate (short)"},
                    {"name": "tier2_rsi_min_long", "tier": 2, "desc": "RSI lower bound"},
                    {"name": "tier2_rsi_max_long", "tier": 2, "desc": "RSI upper bound"},
                    {"name": "tier2_volume_min", "tier": 2, "desc": "Volume confirmation"}
                ],
                "MEDIUM": [
                    {"name": "tier1_adx_hard_min_long", "tier": 1, "desc": "Tier 1 ADX filter"},
                    {"name": "tier1_momentum_min", "tier": 1, "desc": "Tier 1 momentum"},
                    {"name": "weight_macd", "tier": "both", "desc": "MACD scoring weight (24%)"},
                    {"name": "weight_ema", "tier": "both", "desc": "EMA scoring weight (22%)"},
                    {"name": "weight_rsi", "tier": "both", "desc": "RSI scoring weight (16%)"}
                ]
            },
            "Sharpe Ratio": {
                "CRITICAL": [
                    {"name": "stop_loss_atr_mult", "tier": "both", "desc": "Risk/reward controller"},
                    {"name": "risk_tier1", "tier": 1, "desc": "Tier 1 risk %"},
                    {"name": "risk_tier2", "tier": 2, "desc": "Tier 2 risk %"},
                    {"name": "trailing_activation_tier1", "tier": 1, "desc": "Trailing trigger"},
                    {"name": "trailing_distance_tier1", "tier": 1, "desc": "Trailing distance"}
                ],
                "HIGH": [
                    {"name": "quality_tier1_min_long", "tier": 1, "desc": "Entry quality"},
                    {"name": "quality_tier2_min_long", "tier": 2, "desc": "Entry quality (fallback)"},
                    {"name": "exit_threshold_tier1", "tier": 1, "desc": "Exit power threshold"},
                    {"name": "tier1_adx_hard_min_long", "tier": 1, "desc": "Trend strength"},
                    {"name": "weight_macd", "tier": "both", "desc": "MACD scoring weight"}
                ],
                "MEDIUM": [
                    {"name": "weight_ema", "tier": "both", "desc": "EMA scoring weight"},
                    {"name": "weight_rsi", "tier": "both", "desc": "RSI scoring weight"},
                    {"name": "tier1_confluence_min", "tier": 1, "desc": "Signal confluence"},
                    {"name": "tier2_confluence_min", "tier": 2, "desc": "Signal confluence"},
                    {"name": "weight_adx", "tier": "both", "desc": "ADX scoring weight"}
                ]
            },
            "Return [%]": {
                "CRITICAL": [
                    {"name": "quality_tier1_min_long", "tier": 1, "desc": "High-quality entries"},
                    {"name": "quality_tier2_min_long", "tier": 2, "desc": "Quality entries"},
                    {"name": "tier1_adx_hard_min_long", "tier": 1, "desc": "Trend strength"},
                    {"name": "tier1_momentum_min", "tier": 1, "desc": "Momentum filter"},
                    {"name": "tier2_momentum_min", "tier": 2, "desc": "Momentum filter (T2)"}
                ],
                "HIGH": [
                    {"name": "quality_tier1_min_short", "tier": 1, "desc": "Short quality"},
                    {"name": "quality_tier2_min_short", "tier": 2, "desc": "Short quality (T2)"},
                    {"name": "tier1_confluence_min", "tier": 1, "desc": "Signal confirmation"},
                    {"name": "tier2_confluence_min", "tier": 2, "desc": "Signal confirmation"},
                    {"name": "weight_macd", "tier": "both", "desc": "MACD scoring weight"}
                ],
                "MEDIUM": [
                    {"name": "weight_ema", "tier": "both", "desc": "EMA scoring weight"},
                    {"name": "risk_tier1", "tier": 1, "desc": "Position sizing"},
                    {"name": "stop_loss_atr_mult", "tier": "both", "desc": "Stop loss width"},
                    {"name": "tier2_adx_hard_min", "tier": 2, "desc": "Tier 2 ADX"},
                    {"name": "weight_rsi", "tier": "both", "desc": "RSI scoring weight"}
                ]
            },
            "Profit Factor": {
                "CRITICAL": [
                    {"name": "risk_tier1", "tier": 1, "desc": "Risk management"},
                    {"name": "risk_tier2", "tier": 2, "desc": "Risk management"},
                    {"name": "stop_loss_atr_mult", "tier": "both", "desc": "Stop loss width"},
                    {"name": "trailing_distance_tier1", "tier": 1, "desc": "Trailing distance"},
                    {"name": "exit_threshold_tier1", "tier": 1, "desc": "Exit threshold"}
                ],
                "HIGH": [
                    {"name": "quality_tier1_min_long", "tier": 1, "desc": "Entry quality"},
                    {"name": "quality_tier2_min_long", "tier": 2, "desc": "Entry quality"},
                    {"name": "tier1_confluence_min", "tier": 1, "desc": "Signal confluence"},
                    {"name": "tier2_confluence_min", "tier": 2, "desc": "Signal confluence"},
                    {"name": "weight_macd", "tier": "both", "desc": "MACD scoring"}
                ],
                "MEDIUM": [
                    {"name": "weight_ema", "tier": "both", "desc": "EMA scoring"},
                    {"name": "tier1_momentum_min", "tier": 1, "desc": "Momentum"},
                    {"name": "tier2_momentum_min", "tier": 2, "desc": "Momentum"},
                    {"name": "trailing_activation_tier1", "tier": 1, "desc": "Trailing trigger"},
                    {"name": "weight_rsi", "tier": "both", "desc": "RSI scoring"}
                ]
            }
        },
        "scalping": {
            "Win Rate [%]": {
                "CRITICAL": [
                    {"name": "quality_min_long", "tier": "both", "desc": "Entry quality gate"},
                    {"name": "quality_min_short", "tier": "both", "desc": "Entry quality gate (short)"},
                    {"name": "volume_min_ratio", "tier": "both", "desc": "Volume confirmation"},
                    {"name": "rsi_long_min", "tier": "both", "desc": "RSI lower bound"},
                    {"name": "rsi_long_max", "tier": "both", "desc": "RSI upper bound"}
                ],
                "HIGH": [
                    {"name": "adx_min_long", "tier": "both", "desc": "Trend strength"},
                    {"name": "quality_tier1_min", "tier": 1, "desc": "Tier 1 quality"},
                    {"name": "pullback_zone_lower_pct", "tier": "both", "desc": "Pullback zone lower"},
                    {"name": "pullback_zone_upper_pct", "tier": "both", "desc": "Pullback zone upper"},
                    {"name": "momentum_min_long", "tier": "both", "desc": "Momentum filter"}
                ],
                "MEDIUM": [
                    {"name": "stop_loss_atr_mult", "tier": "both", "desc": "Stop width"},
                    {"name": "weight_ema", "tier": "both", "desc": "EMA scoring"},
                    {"name": "weight_macd", "tier": "both", "desc": "MACD scoring"},
                    {"name": "ema_fast_period", "tier": "both", "desc": "Fast EMA period"},
                    {"name": "trend_age_min_bars", "tier": "both", "desc": "Trend age min"}
                ]
            },
            "Sharpe Ratio": {
                "CRITICAL": [
                    {"name": "stop_loss_atr_mult", "tier": "both", "desc": "Risk control"},
                    {"name": "trailing_activation_pct", "tier": "both", "desc": "Trailing trigger"},
                    {"name": "trailing_distance_pct", "tier": "both", "desc": "Trailing distance"},
                    {"name": "risk_per_trade", "tier": "both", "desc": "Risk per trade"},
                    {"name": "take_profit_r1", "tier": "both", "desc": "Profit target R1"}
                ],
                "HIGH": [
                    {"name": "quality_min_long", "tier": "both", "desc": "Entry quality"},
                    {"name": "volume_min_ratio", "tier": "both", "desc": "Volume filter"},
                    {"name": "adx_min_long", "tier": "both", "desc": "Trend filter"},
                    {"name": "max_hold_bars", "tier": "both", "desc": "Max hold time"},
                    {"name": "rsi_long_min", "tier": "both", "desc": "RSI filter"}
                ],
                "MEDIUM": [
                    {"name": "weight_macd", "tier": "both", "desc": "MACD scoring"},
                    {"name": "weight_ema", "tier": "both", "desc": "EMA scoring"},
                    {"name": "macd_fast", "tier": "both", "desc": "MACD fast period"},
                    {"name": "atr_period", "tier": "both", "desc": "ATR period"},
                    {"name": "bb_period", "tier": "both", "desc": "BB period"}
                ]
            },
            "Return [%]": {
                "CRITICAL": [
                    {"name": "quality_min_long", "tier": "both", "desc": "Entry quality"},
                    {"name": "quality_tier1_min", "tier": 1, "desc": "Tier 1 quality"},
                    {"name": "pullback_zone_lower_pct", "tier": "both", "desc": "Entry zone lower"},
                    {"name": "adx_min_long", "tier": "both", "desc": "Trend filter"},
                    {"name": "volume_strong_ratio", "tier": "both", "desc": "Strong volume"}
                ],
                "HIGH": [
                    {"name": "take_profit_r1", "tier": "both", "desc": "Profit target R1"},
                    {"name": "take_profit_r2", "tier": "both", "desc": "2nd profit target"},
                    {"name": "rsi_long_min", "tier": "both", "desc": "RSI min"},
                    {"name": "rsi_long_max", "tier": "both", "desc": "RSI max"},
                    {"name": "stop_loss_atr_mult", "tier": "both", "desc": "Stop width"}
                ],
                "MEDIUM": [
                    {"name": "weight_ema", "tier": "both", "desc": "EMA scoring"},
                    {"name": "weight_macd", "tier": "both", "desc": "MACD scoring"},
                    {"name": "ema_fast_period", "tier": "both", "desc": "Fast EMA"},
                    {"name": "macd_fast", "tier": "both", "desc": "MACD fast"},
                    {"name": "volume_min_ratio", "tier": "both", "desc": "Volume min"}
                ]
            }
        },
        "kalman": {
            "Win Rate [%]": {
                "CRITICAL": [
                    {"name": "long_kalman_strength_min", "tier": "both", "desc": "Kalman strength long"},
                    {"name": "short_kalman_strength_min", "tier": "both", "desc": "Kalman strength short"},
                    {"name": "long_rsi_min", "tier": "both", "desc": "RSI lower bound"},
                    {"name": "volume_min_ratio", "tier": "both", "desc": "Volume filter"},
                    {"name": "min_adx", "tier": "both", "desc": "ADX filter"}
                ],
                "HIGH": [
                    {"name": "stop_loss_pct", "tier": "both", "desc": "Stop loss %"},
                    {"name": "trailing_stop_pct", "tier": "both", "desc": "Trailing stop %"},
                    {"name": "long_pullback_percent", "tier": "both", "desc": "Pullback %"},
                    {"name": "short_rally_percent", "tier": "both", "desc": "Rally %"},
                    {"name": "atr_multiplier", "tier": "both", "desc": "ATR multiplier"}
                ],
                "MEDIUM": [
                    {"name": "trend_lookback", "tier": "both", "desc": "Trend lookback"},
                    {"name": "strength_smooth", "tier": "both", "desc": "Smoothing"},
                    {"name": "process_noise_1", "tier": "both", "desc": "Kalman noise 1"},
                    {"name": "measurement_noise", "tier": "both", "desc": "Kalman measurement noise"},
                    {"name": "cooldown_bars", "tier": "both", "desc": "Cooldown bars"}
                ]
            },
            "Sharpe Ratio": {
                "CRITICAL": [
                    {"name": "stop_loss_pct", "tier": "both", "desc": "Risk control"},
                    {"name": "atr_multiplier", "tier": "both", "desc": "Risk sizing"},
                    {"name": "risk_reward", "tier": "both", "desc": "Risk/reward target"},
                    {"name": "trailing_stop_pct", "tier": "both", "desc": "Trailing stop"},
                    {"name": "risk_per_trade", "tier": "both", "desc": "Risk %"}
                ],
                "HIGH": [
                    {"name": "long_kalman_strength_min", "tier": "both", "desc": "Signal strength"},
                    {"name": "long_rsi_min", "tier": "both", "desc": "RSI filter"},
                    {"name": "volume_min_ratio", "tier": "both", "desc": "Volume filter"},
                    {"name": "min_adx", "tier": "both", "desc": "Trend filter"},
                    {"name": "max_hold_bars", "tier": "both", "desc": "Max hold"}
                ],
                "MEDIUM": [
                    {"name": "trend_lookback", "tier": "both", "desc": "Lookback"},
                    {"name": "strength_smooth", "tier": "both", "desc": "Smoothing"},
                    {"name": "process_noise_1", "tier": "both", "desc": "Kalman noise"},
                    {"name": "measurement_noise", "tier": "both", "desc": "Kalman noise"},
                    {"name": "cooldown_bars", "tier": "both", "desc": "Cooldown"}
                ]
            },
            "Return [%]": {
                "CRITICAL": [
                    {"name": "long_kalman_strength_min", "tier": "both", "desc": "Signal strength"},
                    {"name": "short_kalman_strength_min", "tier": "both", "desc": "Signal strength short"},
                    {"name": "long_pullback_percent", "tier": "both", "desc": "Entry zone"},
                    {"name": "short_rally_percent", "tier": "both", "desc": "Entry zone short"},
                    {"name": "long_rsi_min", "tier": "both", "desc": "RSI filter"}
                ],
                "HIGH": [
                    {"name": "risk_reward", "tier": "both", "desc": "Risk/reward"},
                    {"name": "stop_loss_pct", "tier": "both", "desc": "Stop loss"},
                    {"name": "volume_min_ratio", "tier": "both", "desc": "Volume filter"},
                    {"name": "min_adx", "tier": "both", "desc": "Trend filter"},
                    {"name": "atr_multiplier", "tier": "both", "desc": "ATR sizing"}
                ],
                "MEDIUM": [
                    {"name": "trend_lookback", "tier": "both", "desc": "Lookback"},
                    {"name": "strength_smooth", "tier": "both", "desc": "Smoothing"},
                    {"name": "process_noise_1", "tier": "both", "desc": "Kalman noise"},
                    {"name": "measurement_noise", "tier": "both", "desc": "Kalman noise"},
                    {"name": "cooldown_bars", "tier": "both", "desc": "Cooldown"}
                ]
            }
        }
    }

    def __init__(self):
        self.strategy = "momentum"
        self.metric = "Win Rate [%]"
        self.direction = "both"
        self.tier = "both"
        self.context = {}

    def get_priorities(self, strategy: str, metric: str, direction: str = "both",
                       tier: str = "both", context: dict = None) -> Dict[str, List[Dict]]:
        """
        Get dynamic priority list based on strategy, metric, direction, and tier.
        Returns dict with CRITICAL, HIGH, MEDIUM lists.
        """
        self.strategy = strategy.lower()
        self.metric = metric
        self.direction = direction
        self.tier = tier
        self.context = context or {}

        # Get base priorities for strategy + metric
        strategy_data = self.STRATEGY_PRIORITIES.get(self.strategy, {})
        base_priorities = strategy_data.get(self.metric, {})

        # If no specific priorities for this metric, use default
        if not base_priorities:
            if strategy_data:
                base_priorities = list(strategy_data.values())[0]
            else:
                return {"CRITICAL": [], "HIGH": [], "MEDIUM": []}

        # Filter by direction
        filtered = self._filter_by_direction(base_priorities)

        # Filter by tier
        filtered = self._filter_by_tier(filtered)

        # Apply context adjustments
        if self.context:
            filtered = self._apply_context_adjustments(filtered)

        return filtered

    def _filter_by_direction(self, priorities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Filter parameters by trading direction."""
        if self.direction == "both":
            return priorities

        filtered = {}
        direction_suffix = "_long" if self.direction == "long" else "_short"

        for level, params in priorities.items():
            filtered[level] = []
            for p in params:
                name = p['name']
                # Check if param matches direction
                if name.endswith(direction_suffix):
                    filtered[level].append(p)
                elif "_long" not in name and "_short" not in name:
                    # Direction-agnostic param
                    filtered[level].append(p)
                elif self.direction in name:
                    filtered[level].append(p)
        return filtered

    def _filter_by_tier(self, priorities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Filter parameters by tier."""
        if self.tier == "both":
            return priorities

        filtered = {}
        try:
            tier_num = int(self.tier.replace("tier", ""))
        except ValueError:
            return priorities

        for level, params in priorities.items():
            filtered[level] = []
            for p in params:
                p_tier = p.get('tier', 'both')
                if p_tier == 'both' or p_tier == tier_num:
                    filtered[level].append(p)
        return filtered

    def _apply_context_adjustments(self, priorities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Adjust priorities based on context (timeframe, volatility, etc.)."""
        timeframe = self.context.get('timeframe', '1h')
        volatility = self.context.get('volatility', 'normal')

        adjusted = {k: list(v) for k, v in priorities.items()}

        # Short timeframe: boost stop loss and trailing params
        if timeframe in ['1m', '5m']:
            stop_params = ['stop_loss_atr_mult', 'stop_loss_pct', 'trailing_activation_pct',
                           'trailing_distance_pct', 'trailing_activation_tier1']
            for p_name in stop_params:
                for i, p in enumerate(adjusted.get('MEDIUM', [])):
                    if p['name'] == p_name:
                        adjusted['MEDIUM'].pop(i)
                        adjusted['HIGH'].append(p)
                        break
                for i, p in enumerate(adjusted.get('HIGH', [])):
                    if p['name'] == p_name:
                        adjusted['HIGH'].pop(i)
                        adjusted['CRITICAL'].append(p)
                        break

        # High volatility: boost risk params
        if volatility == 'high':
            risk_params = ['risk_tier1', 'risk_tier2', 'risk_per_trade', 'stop_loss_atr_mult']
            for p_name in risk_params:
                for level in ['MEDIUM', 'HIGH']:
                    for i, p in enumerate(adjusted.get(level, [])):
                        if p['name'] == p_name:
                            adjusted[level].pop(i)
                            target = 'CRITICAL' if level == 'HIGH' else 'HIGH'
                            adjusted[target].append(p)
                            break

        return adjusted


# ═══════════════════════════════════════════════════════════════════════════
# 3. WEB VERIFICATION ENGINE (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

class PriorityVerifier:
    """
    Verifies selected parameters against the dynamic priority hierarchy.
    Properly validates that CRITICAL params are selected, and shows HIGH/MEDIUM as suggestions.
    """

    def __init__(self):
        self.cache = {}
        self.engine = PriorityEngine()

    def verify_parameters(self, strategy: str, selected_params: List[str],
                          metric: str, direction: str = "both", tier: str = "both") -> Dict:
        """
        Verify selected parameters against dynamic priority engine.

        Returns:
            - critical_selected: CRITICAL params that ARE selected ✅
            - critical_missing: CRITICAL params that are NOT selected ❌
            - high_suggested: HIGH priority params not selected (recommended)
            - medium_suggested: MEDIUM priority params not selected (optional)
            - extra_selected: params selected but not in priority tree ⚠️
            - all_critical_selected: boolean
            - coverage_score: % of CRITICAL params selected
        """
        # Get priorities from engine (SAME logic as tree)
        priorities = self.engine.get_priorities(strategy, metric, direction, tier)

        # Build categorized lists
        critical_params = [p['name'] for p in priorities.get('CRITICAL', [])]
        high_params = [p['name'] for p in priorities.get('HIGH', [])]
        medium_params = [p['name'] for p in priorities.get('MEDIUM', [])]
        all_priority_params = set(critical_params + high_params + medium_params)

        # Categorize selected params
        critical_selected = [p for p in selected_params if p in critical_params]
        critical_missing = [p for p in critical_params if p not in selected_params]

        high_selected = [p for p in selected_params if p in high_params]
        high_suggested = [p for p in high_params if p not in selected_params]

        medium_selected = [p for p in selected_params if p in medium_params]
        medium_suggested = [p for p in medium_params if p not in selected_params]

        extra_selected = [p for p in selected_params if p not in all_priority_params]

        # Calculate scores
        all_critical_selected = len(critical_missing) == 0
        coverage_score = len(critical_selected) / len(critical_params) if critical_params else 1.0

        # Determine status
        if not critical_params:
            status = "info"
            status_text = "ℹ️ No CRITICAL parameters defined for this configuration"
        elif all_critical_selected:
            status = "success"
            status_text = "✅ All CRITICAL parameters are selected!"
        else:
            status = "warning"
            status_text = f"⚠️ Missing {len(critical_missing)} CRITICAL parameter(s)"

        return {
            'strategy': strategy,
            'metric': metric,
            'direction': direction,
            'tier': tier,
            'status': status,
            'status_text': status_text,
            'critical_selected': critical_selected,
            'critical_missing': critical_missing,
            'high_selected': high_selected,
            'high_suggested': high_suggested,
            'medium_selected': medium_selected,
            'medium_suggested': medium_suggested,
            'extra_selected': extra_selected,
            'all_critical_selected': all_critical_selected,
            'coverage_score': coverage_score,
            'total_critical': len(critical_params),
            'total_high': len(high_params),
            'total_medium': len(medium_params),
            'total_selected': len(selected_params),
            'timestamp': datetime.now().isoformat(),
            'source': 'Dynamic Priority Engine'
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. PRIORITY SELECTION UI FRAME
# ═══════════════════════════════════════════════════════════════════════════

class PrioritySelectionFrame(tk.Frame):
    """Frame with checkable priority tree for parameter selection."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.engine = PriorityEngine()
        self.verifier = PriorityVerifier()
        self.priority_vars = {}
        self.param_vars = {}
        self.current_priorities = {}
        self.priority_items = {}

        self._build_ui()

    def _build_ui(self):
        # Main frame
        main_frame = ttk.LabelFrame(self, text="🎯 Parameter Priority Selection", padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top controls
        controls = ttk.Frame(main_frame)
        controls.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(controls, text="Strategy:").pack(side=tk.LEFT, padx=(0, 5))
        self.strategy_combo = ttk.Combobox(controls,
                                           values=["Momentum", "Scalping", "Kalman"],
                                           state="readonly", width=10)
        self.strategy_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.strategy_combo.bind('<<ComboboxSelected>>', self._on_change)

        ttk.Label(controls, text="Metric:").pack(side=tk.LEFT, padx=(0, 5))
        self.metric_combo = ttk.Combobox(controls,
                                         values=["Win Rate [%]", "Sharpe Ratio", "Return [%]", "Profit Factor"],
                                         state="readonly", width=12)
        self.metric_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.metric_combo.bind('<<ComboboxSelected>>', self._on_change)

        ttk.Label(controls, text="Direction:").pack(side=tk.LEFT, padx=(0, 5))
        self.direction_combo = ttk.Combobox(controls,
                                            values=["both", "long", "short"],
                                            state="readonly", width=6)
        self.direction_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.direction_combo.bind('<<ComboboxSelected>>', self._on_change)

        ttk.Label(controls, text="Tier:").pack(side=tk.LEFT, padx=(0, 5))
        self.tier_combo = ttk.Combobox(controls,
                                       values=["both", "tier1", "tier2"],
                                       state="readonly", width=6)
        self.tier_combo.pack(side=tk.LEFT)
        self.tier_combo.bind('<<ComboboxSelected>>', self._on_change)

        # Button row
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 5))

        self.select_all_btn = tk.Button(btn_frame, text="✅ Select All",
                                        command=self._select_all,
                                        bg="#4CAF50", fg="white",
                                        font=('Arial', 9, 'bold'),
                                        padx=10, pady=2, cursor="hand2")
        self.select_all_btn.pack(side=tk.LEFT, padx=2)

        self.deselect_all_btn = tk.Button(btn_frame, text="❌ Deselect All",
                                          command=self._deselect_all,
                                          bg="#FF5555", fg="white",
                                          font=('Arial', 9, 'bold'),
                                          padx=10, pady=2, cursor="hand2")
        self.deselect_all_btn.pack(side=tk.LEFT, padx=2)

        self.verify_btn = tk.Button(btn_frame, text="🔍 Verify Selection",
                                    command=self._verify_params,
                                    bg="#FF6B35", fg="white",
                                    font=('Arial', 9, 'bold'),
                                    padx=10, pady=2, cursor="hand2")
        self.verify_btn.pack(side=tk.LEFT, padx=2)

        self.verify_status = ttk.Label(btn_frame, text="", foreground="#888888")
        self.verify_status.pack(side=tk.LEFT, padx=10)

        # Scrollable tree
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(tree_frame, yscrollcommand=scrollbar.set,
                                 selectmode='none',height= 6)
        scrollbar.config(command=self.tree.yview)

        self.tree['columns'] = ('check', 'param', 'tier', 'description')
        self.tree.column('#0', width=0, stretch=False)
        self.tree.column('check', width=40, anchor='center')
        self.tree.column('param', width=200, anchor='w')
        self.tree.column('tier', width=60, anchor='center')
        self.tree.column('description', width=250, anchor='w')

        self.tree.heading('check', text='')
        self.tree.heading('param', text='Parameter')
        self.tree.heading('tier', text='Tier')
        self.tree.heading('description', text='Description')

        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind('<ButtonRelease-1>', self._on_tree_click)

        # Bottom summary
        summary_frame = ttk.Frame(main_frame)
        summary_frame.pack(fill=tk.X, pady=(5, 0))

        self.summary_label = ttk.Label(summary_frame,
                                       text="0 parameters selected",
                                       font=('Arial', 9, 'bold'))
        self.summary_label.pack(side=tk.LEFT)

        # Load defaults
        self._load_defaults()
        self._update_priorities()

    def _load_defaults(self):
        """Load default values from app."""
        if self.app:
            self.strategy_combo.set("Momentum")
            self.metric_combo.set("Win Rate [%]")
            self.direction_combo.set("both")
            self.tier_combo.set("both")

    def _on_change(self, event=None):
        """Handle any change in selection."""
        self._update_priorities()

    def _update_priorities(self):
        """Update the priority tree based on current selections."""
        # Clear existing tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.priority_vars = {}
        self.param_vars = {}
        self.priority_items = {}

        # Get priorities
        strategy = self.strategy_combo.get().lower()
        metric = self.metric_combo.get()
        direction = self.direction_combo.get()
        tier = self.tier_combo.get()

        priorities = self.engine.get_priorities(strategy, metric, direction, tier)
        self.current_priorities = priorities

        total_params = 0

        # Color mapping for levels
        level_colors = {
            'CRITICAL': '#FF4444',
            'HIGH': '#FFA500',
            'MEDIUM': '#4CAF50'
        }

        # Add each priority level
        for level in ['CRITICAL', 'HIGH', 'MEDIUM']:
            params = priorities.get(level, [])
            if not params:
                continue

            # Add level header
            level_var = tk.BooleanVar(value=True)
            self.priority_vars[level] = level_var

            level_item = self.tree.insert('', 'end', text='',
                                          values=('☑', f"[{level}] ({len(params)})", '', ''),
                                          open=True)
            self.priority_items[level] = level_item
            self.tree.item(level_item, tags=('level', level))

            # Add each parameter
            for p in params:
                param_name = p['name']
                tier_label = f"T{p['tier']}" if p['tier'] != 'both' else 'Both'
                desc = p['desc']

                param_var = tk.BooleanVar(value=True)
                self.param_vars[param_name] = param_var

                item = self.tree.insert(level_item, 'end', text='',
                                        values=('☑', param_name, tier_label, desc),
                                        tags=('param', param_name, level))
                total_params += 1

        # Update summary
        self.summary_label.config(text=f"{total_params} parameters selected")

        # Save to app
        if self.app:
            self.app._selected_params = list(self.param_vars.keys())

    def _on_tree_click(self, event):
        """Handle clicks on tree items."""
        item = self.tree.identify_row(event.y)
        if not item:
            return

        tags = self.tree.item(item, 'tags')
        values = list(self.tree.item(item, 'values'))  # Convert to list for modification

        if not values:
            return

        # Check if click was on checkbox area
        region = self.tree.identify_region(event.x, event.y)
        if region != 'cell' and region != 'tree':
            return

        # If level item
        if 'level' in tags:
            level = tags[1] if len(tags) > 1 else tags[0]
            if level in self.priority_vars:
                current = self.priority_vars[level].get()
                new_val = not current
                self.priority_vars[level].set(new_val)

                for child in self.tree.get_children(item):
                    child_values = list(self.tree.item(child, 'values'))
                    if child_values:
                        param_name = child_values[1]
                        if param_name in self.param_vars:
                            self.param_vars[param_name].set(new_val)
                            child_values[0] = '☑' if new_val else '☐'
                            self.tree.item(child, values=child_values)

                values[0] = '☑' if new_val else '☐'
                self.tree.item(item, values=values)
                self._update_summary()
            return

        # If param item
        if 'param' in tags:
            param_name = tags[1] if len(tags) > 1 else values[1]
            if param_name in self.param_vars:
                current = self.param_vars[param_name].get()
                new_val = not current
                self.param_vars[param_name].set(new_val)

                values[0] = '☑' if new_val else '☐'
                self.tree.item(item, values=values)

                parent = self.tree.parent(item)
                if parent:
                    self._update_level_checkbox(parent)

                self._update_summary()

    def _update_level_checkbox(self, level_item):
        """Update level checkbox based on child states."""
        children = self.tree.get_children(level_item)
        all_checked = True
        any_checked = False

        for child in children:
            child_values = self.tree.item(child, 'values')
            if child_values and child_values[0] == '☑':
                any_checked = True
            else:
                all_checked = False

        values = list(self.tree.item(level_item, 'values'))
        if all_checked and any_checked:
            values[0] = '☑'
        elif any_checked:
            values[0] = '☐'
        else:
            values[0] = '☐'
        self.tree.item(level_item, values=values)

    def _update_summary(self):
        """Update the summary label."""
        selected = sum(1 for v in self.param_vars.values() if v.get())
        total = len(self.param_vars)
        self.summary_label.config(text=f"{selected} of {total} parameters selected")

        if self.app:
            self.app._selected_params = [name for name, var in self.param_vars.items() if var.get()]

    def _select_all(self):
        """Select all parameters."""
        for var in self.param_vars.values():
            var.set(True)
        for item in self.tree.get_children():
            for child in self.tree.get_children(item):
                child_values = list(self.tree.item(child, 'values'))
                if child_values:
                    child_values[0] = '☑'
                    self.tree.item(child, values=child_values)
            self._update_level_checkbox(item)
        self._update_summary()

    def _deselect_all(self):
        """Deselect all parameters."""
        for var in self.param_vars.values():
            var.set(False)
        for item in self.tree.get_children():
            for child in self.tree.get_children(item):
                child_values = list(self.tree.item(child, 'values'))
                if child_values:
                    child_values[0] = '☐'
                    self.tree.item(child, values=child_values)
            self._update_level_checkbox(item)
        self._update_summary()

    def _verify_params(self):
        """Run verification on selected parameters using dynamic priority engine."""
        strategy = self.strategy_combo.get().lower()
        metric = self.metric_combo.get()
        direction = self.direction_combo.get()
        tier = self.tier_combo.get()

        selected = [name for name, var in self.param_vars.items() if var.get()]

        if not selected:
            self.verify_status.config(text="⚠️ No parameters selected", foreground="#FF5555")
            return

        self.verify_btn.config(state=tk.DISABLED, text="⏳ Verifying...")
        self.verify_status.config(text="Checking priority hierarchy...", foreground="#FFA500")

        def worker():
            # Use SAME engine as priority tree
            report = self.verifier.verify_parameters(
                strategy=strategy,
                selected_params=selected,
                metric=metric,
                direction=direction,
                tier=tier
            )
            self.app.root.after(0, lambda: self._show_verification_results(report))
            self.app.root.after(0, lambda: self.verify_btn.config(
                state=tk.NORMAL, text="🔍 Verify Selection"))
            # Update status based on report
            status_color = "#4CAF50" if report['status'] == 'success' else "#FFA500" if report[
                                                                                            'status'] == 'warning' else "#888888"
            self.app.root.after(0, lambda: self.verify_status.config(
                text=report['status_text'], foreground=status_color))

        threading.Thread(target=worker, daemon=True).start()

    def _show_verification_results(self, report):
        """Display verification results in a popup with proper hierarchy validation."""
        results_window = tk.Toplevel(self.app.root)
        results_window.title("🔍 Parameter Verification Results")
        results_window.geometry("800x700")
        results_window.transient(self.app.root)
        results_window.grab_set()

        w, h = 800, 700
        sw = results_window.winfo_screenwidth()
        sh = results_window.winfo_screenheight()
        results_window.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        # Header with status
        status_color = "#4CAF50" if report['status'] == 'success' else "#FFA500" if report[
                                                                                        'status'] == 'warning' else "#888888"

        tk.Label(results_window, text="🔍 Parameter Verification Report",
                 font=('Helvetica', 14, 'bold'), pady=10).pack()

        # Status banner
        status_frame = tk.Frame(results_window, bg=status_color, padx=20, pady=10)
        status_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(status_frame, text=report['status_text'],
                 font=('Arial', 12, 'bold'), bg=status_color, fg="white").pack()

        # Config info
        info = tk.Frame(results_window)
        info.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(info, text=f"Strategy: {report['strategy'].upper()}",
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Label(info, text=f"Metric: {report['metric']}",
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        tk.Label(info, text=f"Direction: {report.get('direction', 'both')}",
                 font=('Arial', 9), fg='#888888').pack(side=tk.LEFT, padx=5)
        tk.Label(info, text=f"Tier: {report.get('tier', 'both')}",
                 font=('Arial', 9), fg='#888888').pack(side=tk.LEFT, padx=5)

        # Coverage score
        coverage = report.get('coverage_score', 0) * 100
        coverage_color = "#4CAF50" if coverage >= 100 else "#FFA500" if coverage >= 50 else "#FF5555"
        tk.Label(results_window,
                 text=f"CRITICAL Parameter Coverage: {coverage:.0f}%  ({len(report.get('critical_selected', []))} of {report.get('total_critical', 0)})",
                 font=('Arial', 11), fg=coverage_color).pack(pady=5)

        text = tk.Text(results_window, wrap=tk.WORD, font=('Consolas', 9),
                       bg="#1e1e1e", fg="#d4d4d4")
        text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        text.tag_config("green", foreground="#4CAF50")
        text.tag_config("red", foreground="#FF5555")
        text.tag_config("yellow", foreground="#FFD700")
        text.tag_config("white", foreground="#FFFFFF")
        text.tag_config("cyan", foreground="#00FFFF")
        text.tag_config("orange", foreground="#FF8C00")
        text.tag_config("blue", foreground="#58a6ff")
        text.tag_config("gray", foreground="#888888")

        # ─── CRITICAL Section ──────────────────────────────────────────────
        text.insert(tk.END, "\n🔴 CRITICAL PARAMETERS (Must Have)\n", "red")
        text.insert(tk.END, "-" * 70 + "\n", "white")

        critical_selected = report.get('critical_selected', [])
        critical_missing = report.get('critical_missing', [])

        if critical_selected:
            text.insert(tk.END, f"  ✅ SELECTED ({len(critical_selected)}):\n", "green")
            for p in critical_selected:
                text.insert(tk.END, f"    ✓ {p}\n", "green")
        else:
            text.insert(tk.END, "  ⚠️ No CRITICAL params selected!\n", "red")

        if critical_missing:
            text.insert(tk.END, f"\n  ❌ MISSING ({len(critical_missing)}):\n", "red")
            for p in critical_missing:
                text.insert(tk.END, f"    ✗ {p}\n", "red")

        # ─── HIGH Section ──────────────────────────────────────────────────
        text.insert(tk.END, "\n🟠 HIGH PRIORITY (Recommended)\n", "orange")
        text.insert(tk.END, "-" * 70 + "\n", "white")

        high_selected = report.get('high_selected', [])
        high_suggested = report.get('high_suggested', [])

        if high_selected:
            text.insert(tk.END, f"  ✅ SELECTED ({len(high_selected)}):\n", "green")
            for p in high_selected:
                text.insert(tk.END, f"    ✓ {p}\n", "green")
        else:
            text.insert(tk.END, "  (None selected)\n", "gray")

        if high_suggested:
            text.insert(tk.END, f"\n  💡 SUGGESTED ({len(high_suggested)}):\n", "orange")
            for p in high_suggested:
                text.insert(tk.END, f"    → {p}\n", "orange")

        # ─── MEDIUM Section ─────────────────────────────────────────────────
        text.insert(tk.END, "\n🟡 MEDIUM PRIORITY (Optional)\n", "yellow")
        text.insert(tk.END, "-" * 70 + "\n", "white")

        medium_selected = report.get('medium_selected', [])
        medium_suggested = report.get('medium_suggested', [])

        if medium_selected:
            text.insert(tk.END, f"  ✅ SELECTED ({len(medium_selected)}):\n", "green")
            for p in medium_selected:
                text.insert(tk.END, f"    ✓ {p}\n", "green")
        else:
            text.insert(tk.END, "  (None selected)\n", "gray")

        if medium_suggested:
            text.insert(tk.END, f"\n  💡 SUGGESTED ({len(medium_suggested)}):\n", "yellow")
            for p in medium_suggested:
                text.insert(tk.END, f"    → {p}\n", "yellow")

        # ─── EXTRA Section ──────────────────────────────────────────────────
        extra_selected = report.get('extra_selected', [])
        if extra_selected:
            text.insert(tk.END, "\n⚠️ EXTRA SELECTIONS (Not in Priority Tree)\n", "red")
            text.insert(tk.END, "-" * 70 + "\n", "white")
            text.insert(tk.END, "  These parameters are not in the priority hierarchy:\n", "gray")
            for p in extra_selected:
                text.insert(tk.END, f"    ⚠️ {p}\n", "red")

        # ─── Summary ───────────────────────────────────────────────────────
        text.insert(tk.END, "\n📊 SUMMARY\n", "cyan")
        text.insert(tk.END, "-" * 70 + "\n", "white")
        text.insert(tk.END, f"  Total CRITICAL: {report.get('total_critical', 0)}\n", "white")
        text.insert(tk.END, f"  Total HIGH:     {report.get('total_high', 0)}\n", "white")
        text.insert(tk.END, f"  Total MEDIUM:   {report.get('total_medium', 0)}\n", "white")
        text.insert(tk.END, f"  Total Selected: {report.get('total_selected', 0)}\n", "white")

        # Recommendation
        if report['all_critical_selected']:
            text.insert(tk.END, "\n✅ RECOMMENDATION: All CRITICAL params selected. Ready to optimize!\n", "green")
        else:
            text.insert(tk.END,
                        f"\n⚠️ RECOMMENDATION: Add the {len(critical_missing)} missing CRITICAL params for best results.\n",
                        "red")

        tk.Button(results_window, text="Close", command=results_window.destroy,
                  bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'),
                  padx=20, pady=5).pack(pady=10)

    def get_selected_params(self) -> List[str]:
        """Get list of currently selected parameter names."""
        return [name for name, var in self.param_vars.items() if var.get()]

# ═══════════════════════════════════════════════════════════════════════════
# 5. STRATEGY IMPORTER
# ═══════════════════════════════════════════════════════════════════════════

def import_strategy_module(strategy_name: str):
    """Dynamically import the REAL strategy module."""
    strategy_name = strategy_name.lower()

    try:
        if strategy_name == "momentum":
            from strategies.MomentumStrategy_MACD_HybridScore_Claude import (
                BacktestMomentumStrategy, MOMENTUM_PARAMS, BACKTEST_PARAMS
            )
            return BacktestMomentumStrategy, MOMENTUM_PARAMS, BACKTEST_PARAMS, "REAL"
        elif strategy_name == "kalman":
            from strategies.KalmanTrendStrategy_New import (
                BacktestKalmanTrendStrategy, KALMAN_PARAMS
            )
            return BacktestKalmanTrendStrategy, KALMAN_PARAMS, {}, "REAL"
        elif strategy_name == "scalping":
            from strategies.scalping_strategy import (
                BacktestScalpingStrategy, SCALPING_PARAMS
            )
            return BacktestScalpingStrategy, SCALPING_PARAMS, {}, "REAL"
    except ImportError as e:
        print(f"⚠️ Absolute import failed: {e}")

    try:
        import importlib.util
        if strategy_name == "momentum":
            filename = os.path.join(current_dir, "MomentumStrategy_MACD_HybridScore_Claude.py")
        elif strategy_name == "kalman":
            filename = os.path.join(current_dir, "KalmanTrendStrategy_New.py")
        elif strategy_name == "scalping":
            filename = os.path.join(current_dir, "scalping_strategy.py")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        spec = importlib.util.spec_from_file_location(strategy_name, filename)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[strategy_name] = module
            spec.loader.exec_module(module)

            if strategy_name == "momentum":
                return (module.BacktestMomentumStrategy,
                        module.MOMENTUM_PARAMS,
                        module.BACKTEST_PARAMS,
                        "FILE")
            elif strategy_name == "kalman":
                return (module.BacktestKalmanTrendStrategy,
                        module.KALMAN_PARAMS,
                        {},
                        "FILE")
            elif strategy_name == "scalping":
                return (module.BacktestScalpingStrategy,
                        module.SCALPING_PARAMS,
                        {},
                        "FILE")
    except Exception as e:
        print(f"⚠️ File import failed: {e}")

    return None, None, None, "NONE"


# ═══════════════════════════════════════════════════════════════════════════
# 6. STRATEGY SETTINGS LOADER
# ═══════════════════════════════════════════════════════════════════════════

def find_settings_file() -> Optional[str]:
    """Locate strategy_settings.json."""
    candidates = []
    if os.environ.get('BEST_PARAM_SETTINGS_FILE'):
        candidates.append(os.environ['BEST_PARAM_SETTINGS_FILE'])
    candidates.append(os.path.join(os.getcwd(), "strategy_settings.json"))
    candidates.append(os.path.join(current_dir, "strategy_settings.json"))
    candidates.append(os.path.join(project_root, "strategy_settings.json"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def load_effective_default_params(strategy_name: str, module_default_params: dict,
                                  log_callback=None) -> Tuple[dict, str]:
    """Build effective default parameter set."""

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    effective = dict(module_default_params)
    prefix = strategy_name.lower()

    settings_file = find_settings_file()
    if not settings_file:
        log("ℹ️ No strategy_settings.json found — using strategy module defaults only.")
        return effective, ""

    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)

        default_section = settings.get('default_params', {}).get(prefix, {})
        effective.update(default_section)

        mode = settings.get('selected_mode', 'Default Parameters')
        if mode == 'Custom Parameters':
            custom_section = settings.get('custom_params', {}).get(prefix, {})
            if custom_section:
                effective.update(custom_section)
                log(f"📄 Loaded CUSTOM params for '{prefix}' from {settings_file}")
            else:
                log(f"📄 Mode is 'Custom Parameters' but no custom section for '{prefix}'")
        else:
            log(f"📄 Loaded DEFAULT params for '{prefix}' from {settings_file}")

        direction = effective.get('trade_direction', 'long')
        log(f"   ↳ trade_direction: '{direction}'")

        return effective, settings_file

    except Exception as e:
        log(f"⚠️ Could not read {settings_file}: {e}")
        return effective, ""


# ═══════════════════════════════════════════════════════════════════════════
# 7. PARAMETER RANGES
# ═══════════════════════════════════════════════════════════════════════════

def get_optimization_ranges_from_defaults(default_params: dict,
                                          param_names: List[str],
                                          variance_pct: float = 0.25) -> dict:
    """Generate optimization ranges from default parameter values."""
    ranges = {}
    for param in param_names:
        if param not in default_params:
            continue
        val = default_params[param]

        if isinstance(val, bool):
            ranges[param] = [False, True]
            continue

        if isinstance(val, int):
            # Sign-aware: previously any val <= 0 was silently skipped entirely,
            # permanently excluding legitimate negative-default params (e.g.
            # short_kalman_strength_min = -30) from ever being optimized.
            if val == 0:
                # No natural scale to anchor a percentage spread on; use a
                # small fixed symmetric window instead of skipping.
                lower, upper = -2, 3
            elif val > 0:
                lower = max(0, int(val * (1 - variance_pct)))
                upper = int(val * (1 + variance_pct)) + 1
                if upper - lower < 3:
                    lower = max(0, val - 2)
                    upper = val + 3
            else:  # val < 0: mirror the positive-side logic, bounding at 0
                upper = min(0, int(val * (1 - variance_pct)))
                lower = int(val * (1 + variance_pct)) - 1
                if upper - lower < 3:
                    upper = min(0, val + 2)
                    lower = val - 3
            ranges[param] = list(range(lower, upper + 1, max(1, (upper - lower) // 3)))
            if val not in ranges[param]:
                ranges[param].append(val)
                ranges[param].sort()
            continue

        if isinstance(val, float):
            # Sign-aware: previously any val <= 0 was silently skipped entirely
            # (e.g. momentum_reversal_threshold = -0.3, dmi_spread_min_long = 0.0).
            if val == 0:
                step = 0.01
            else:
                step = max(0.01, abs(val) * variance_pct)
            if param in ['risk_tier1', 'risk_tier2', 'risk_per_trade']:
                step = 0.005
            vals = [val - step * 2, val - step, val, val + step, val + step * 2]
            ranges[param] = sorted(set([round(v, 4) for v in vals]))
            continue

    return ranges


# ═══════════════════════════════════════════════════════════════════════════
# 8. DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_ccxt_data(symbol: str, interval: str, start: str, end: str,
                    exchange: str = 'binance', max_retries: int = 3,
                    log_callback=None) -> pd.DataFrame:
    """Fetch OHLCV data with retry logic."""

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    if '-' in symbol and '/' not in symbol:
        symbol = symbol.replace('-', '/')

    for attempt in range(max_retries):
        try:
            exch = getattr(ccxt, exchange)({'enableRateLimit': True})
            start_dt = pd.to_datetime(start, utc=True)
            end_dt = pd.to_datetime(end, utc=True)
            since = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)

            all_data = []
            while since < end_ts:
                ohlcv = exch.fetch_ohlcv(symbol, interval, since=since, limit=1000)
                if not ohlcv:
                    break
                since = ohlcv[-1][0] + 1
                all_data.extend(ohlcv)
                if ohlcv[-1][0] >= end_ts or len(ohlcv) < 1000:
                    break
                time.sleep(exch.rateLimit / 1000)

            if not all_data:
                continue

            df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df[df.index <= end_dt]
            return df.tz_localize(None)

        except ccxt.BadSymbol as e:
            if attempt == 0:
                log(f"⚠️ Symbol '{symbol}' not found, trying alternatives...")
                alt_symbols = []
                if '/' in symbol:
                    alt_symbols.append(symbol.replace('/', '-'))
                if '-' in symbol:
                    alt_symbols.append(symbol.replace('-', '/'))
                if 'USDT' not in symbol and 'USDC' not in symbol:
                    alt_symbols.append(f"{symbol}/USDT")
                if symbol.endswith('/USDT'):
                    alt_symbols.append(symbol.replace('/USDT', ''))
                if symbol.endswith('-USDT'):
                    alt_symbols.append(symbol.replace('-USDT', ''))

                for alt in alt_symbols:
                    if alt != symbol:
                        try:
                            log(f"   Trying: {alt}")
                            ohlcv = exch.fetch_ohlcv(alt, interval, limit=1)
                            if ohlcv:
                                symbol = alt
                                log(f"   ✅ Using: {symbol}")
                                break
                        except:
                            continue
                continue
            else:
                raise

        except Exception as e:
            if attempt < max_retries - 1:
                log(f"⚠️ Data fetch attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                raise

    raise RuntimeError(f"Failed to fetch data for {symbol} after {max_retries} attempts")


# ═══════════════════════════════════════════════════════════════════════════
# 9. OPTIMIZATION ENGINE (MODIFIED)
# ═══════════════════════════════════════════════════════════════════════════

def run_optimization(strategy_name: str,
                     symbol: str,
                     interval: str,
                     start_date: str,
                     end_date: str,
                     metric: str,
                     param_names: List[str] = None,
                     max_tries: int = 200,
                     api_key: str = None,
                     log_callback=None,
                     progress_callback=None) -> Dict[str, Any]:
    """
    Run optimization on specified parameter names.
    If param_names is None, uses all available params.
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    log(f"\n{'=' * 70}")
    log(f"🚀 OPTIMIZATION RUN: {strategy_name} → {metric}")
    log(f"{'=' * 70}")
    log(f"  Symbol:   {symbol}")
    log(f"  Interval: {interval}")
    log(f"  Period:   {start_date} → {end_date}")
    log(f"  Max tries: {max_tries}")
    if param_names:
        log(f"  Params to optimize: {len(param_names)} (user-selected)")
    log(f"{'=' * 70}\n")

    progress(5, "Loading strategy...")

    StrategyClass, module_default_params, backtest_params, source = import_strategy_module(strategy_name)
    if StrategyClass is None:
        raise RuntimeError(f"Could not import {strategy_name} strategy")

    log(f"✅ Loaded {strategy_name} from {source} source")

    progress(7, "Reading strategy_settings.json...")
    default_params, settings_file = load_effective_default_params(strategy_name, module_default_params, log)
    log(f"   Effective params: {len(default_params)} parameters"
        + (f" (from {settings_file})" if settings_file else " (module defaults)"))
    log(f"   Trade direction: {default_params.get('trade_direction', 'long').upper()}")

    # Determine which params to optimize
    if param_names is None:
        # Use all tunable params
        all_params = [p for p in default_params.keys()
                      if isinstance(default_params[p], (int, float, bool))
                      and p != 'trade_direction']
        param_names = all_params
        log(f"📊 Optimizing all {len(param_names)} parameters")
    else:
        # Filter to only those that exist in defaults
        valid_params = [p for p in param_names if p in default_params]
        if len(valid_params) < len(param_names):
            invalid = set(param_names) - set(valid_params)
            log(f"⚠️ Ignoring invalid params: {invalid}")
        param_names = valid_params
        log(f"📊 Optimizing {len(param_names)} user-selected parameters")

    if not param_names:
        raise RuntimeError("No valid parameters to optimize")

    # Generate ranges
    progress(15, "Generating parameter ranges...")
    param_ranges = get_optimization_ranges_from_defaults(
        default_params, param_names, variance_pct=0.25
    )

    param_ranges = {k: v for k, v in param_ranges.items() if len(v) >= 2}

    log(f"\n📊 Generated ranges for {len(param_ranges)} parameters:")
    for p, vals in param_ranges.items():
        display_vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals[:3]]
        if len(vals) > 3:
            display_vals.append("...")
        log(f"   {p}: {', '.join(display_vals)}")

    if not param_ranges:
        raise RuntimeError("No valid parameter ranges generated")

    # Fetch data
    progress(20, "Fetching market data...")
    df = fetch_ccxt_data(symbol, interval, start_date, end_date, log_callback=log)
    log(f"📊 Loaded {len(df)} candles")

    from backtesting import Backtest

    progress(25, "Running baseline...")
    log("\n🏃 Running baseline...")
    bt = Backtest(df, StrategyClass, cash=50000, commission=0.0006, exclusive_orders=True)
    baseline_stats = bt.run(**default_params)

    baseline_metric = baseline_stats.get(metric, 0)
    baseline_sharpe = baseline_stats.get('Sharpe Ratio', 0)
    baseline_return = baseline_stats.get('Return [%]', 0)
    baseline_winrate = baseline_stats.get('Win Rate [%]', 0)
    baseline_trades = baseline_stats.get('# Trades', 0)

    log(f"   Baseline {metric}: {baseline_metric:.4f}")
    log(f"   Sharpe: {baseline_sharpe:.3f} | Return: {baseline_return:.2f}% | WR: {baseline_winrate:.1f}% | Trades: {baseline_trades}")

    # Run optimization
    progress(30, f"Running optimization ({max_tries} combinations)...")
    log(f"\n🔍 Running random search ({max_tries} combinations)...")

    try:
        optimized_stats = bt.optimize(
            **param_ranges,
            maximize=metric,
            max_tries=max_tries,
            random_state=42,
            return_heatmap=False
        )

        best_params = {}
        for p in param_ranges.keys():
            if hasattr(optimized_stats._strategy, p):
                best_params[p] = getattr(optimized_stats._strategy, p)

        progress(90, "Calculating results...")

        optimized_metric = optimized_stats.get(metric, 0)
        optimized_sharpe = optimized_stats.get('Sharpe Ratio', 0)
        optimized_return = optimized_stats.get('Return [%]', 0)
        optimized_winrate = optimized_stats.get('Win Rate [%]', 0)
        optimized_trades = optimized_stats.get('# Trades', 0)

        metric_improvement = optimized_metric - baseline_metric
        metric_pct_improvement = (metric_improvement / abs(baseline_metric) * 100
                                  if baseline_metric != 0 else float('inf'))
        sharpe_improvement = optimized_sharpe - baseline_sharpe
        return_improvement = optimized_return - baseline_return

        progress(100, "Complete!")

        log(f"\n🏆 BEST RESULTS:")
        log(f"   Best {metric}: {optimized_metric:.4f} ({metric_improvement:+.4f}, {metric_pct_improvement:+.1f}%)")
        log(f"   Sharpe: {optimized_sharpe:.3f} ({sharpe_improvement:+.3f})")
        log(f"   Return: {optimized_return:.2f}% ({return_improvement:+.2f}%)")
        log(f"   Win Rate: {optimized_winrate:.1f}%")
        log(f"   Trades: {optimized_trades}")

        log(f"\n📋 BEST PARAMETERS:")
        for p, v in best_params.items():
            default = default_params.get(p, "N/A")
            if default != "N/A" and default != v:
                log(f"   {p}: {v}  (was {default})")
            else:
                log(f"   {p}: {v}  (unchanged)")

        return {
            'best_params': best_params,
            'optimized_params': param_names,
            'trade_direction': default_params.get('trade_direction', 'long'),
            'settings_file': settings_file,
            'baseline_stats': {
                metric: baseline_metric,
                'Sharpe Ratio': baseline_sharpe,
                'Return [%]': baseline_return,
                'Win Rate [%]': baseline_winrate,
                '# Trades': baseline_trades,
            },
            'optimized_stats': {
                metric: optimized_metric,
                'Sharpe Ratio': optimized_sharpe,
                'Return [%]': optimized_return,
                'Win Rate [%]': optimized_winrate,
                '# Trades': optimized_trades,
            },
            'improvements': {
                metric: metric_improvement,
                f'{metric}_pct': metric_pct_improvement,
                'Sharpe Ratio': sharpe_improvement,
                'Return [%]': return_improvement,
            },
            'metric_used': metric,
            'params_optimized': len(param_ranges),
            'tries_used': max_tries,
        }

    except Exception as e:
        log(f"❌ Optimization error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'best_params': {},
            'baseline_stats': {},
            'optimized_stats': {},
            'improvements': {},
            'metric_used': metric,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 10. UPDATE STRATEGY SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

def update_strategy_settings(strategy_name: str, best_params: Dict[str, Any]) -> bool:
    """Update the strategy_settings.json with optimized parameters."""
    settings_file = "strategy_settings.json"
    try:
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        else:
            settings = {'default_params': {}, 'custom_params': {}, 'selected_mode': 'Default Parameters'}

        prefix = 'momentum' if strategy_name.lower() == 'momentum' else strategy_name.lower()

        if 'custom_params' not in settings:
            settings['custom_params'] = {}
        if prefix not in settings['custom_params']:
            settings['custom_params'][prefix] = {}

        for k, v in best_params.items():
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            settings['custom_params'][prefix][k] = v

        settings['selected_mode'] = 'Custom Parameters'

        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)

        return True
    except Exception as e:
        print(f"⚠️ Could not update settings: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 11. LOAD API KEY
# ═══════════════════════════════════════════════════════════════════════════

def load_api_key() -> Optional[str]:
    """Load DeepSeek API key from config.json if available."""
    config_file = "config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get('deepseek_api_key', None)
        except:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 12. MAIN GUI APPLICATION (MODIFIED)
# ═══════════════════════════════════════════════════════════════════════════

class OptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏆 Strategy Parameter Optimizer v7.1")
        self.root.geometry("1110x700")
        self.root.minsize(1110, 700)

        # Center window
        self.root.update_idletasks()
        w, h = 1110, 800
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - (h+100)) // 2}")

        # Variables
        self.strategy_var = tk.StringVar(value="Momentum")
        self.metric_var = tk.StringVar(value="Sharpe Ratio")
        self.symbol_var = tk.StringVar(value="SOL/USDT")
        self.interval_var = tk.StringVar(value="15m")
        self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"))
        self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.max_tries_var = tk.IntVar(value=200)
        self.auto_update_var = tk.BooleanVar(value=True)

        self.results = None
        self.is_running = False
        self._selected_params = []

        # Timer
        self.run_start_time = None
        self.timer_job = None

        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        # ─── Title ──────────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg="#1a1a2e", pady=12)
        title_frame.pack(fill=tk.X)

        tk.Label(title_frame, text="🏆 STRATEGY PARAMETER OPTIMIZER v7.1",
                 font=('Helvetica', 18, 'bold'), bg="#1a1a2e", fg="#4CAF50").pack()
        tk.Label(title_frame, text="Priority-Based Parameter Selection & Optimization",
                 font=('Helvetica', 10), bg="#1a1a2e", fg="#aaaaaa").pack()

        # ─── Timer ──────────────────────────────────────────────────────────
        self.timer_frame = tk.Frame(title_frame, bg="#0d1117", bd=1, relief=tk.SOLID,
                                    padx=10, pady=6)
        self.timer_frame.place(relx=1.0, rely=0.5, anchor="e", x=-15)

        self.elapsed_time_label = tk.Label(self.timer_frame, text="⏳ Elapsed: 00:00",
                                           font=('Consolas', 10, 'bold'),
                                           bg="#0d1117", fg="#3fb950")
        self.elapsed_time_label.pack(anchor="w")

        self.est_time_label = tk.Label(self.timer_frame, text="⏱ Est. remaining: --:--",
                                       font=('Consolas', 10, 'bold'),
                                       bg="#0d1117", fg="#58a6ff")
        self.est_time_label.pack(anchor="w")

        # ─── Main container ──────────────────────────────────────────────────
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)

        # ─── Left panel ─────────────────────────────────────────────────────
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # ─── Configuration Frame ────────────────────────────────────────────
        config_frame = ttk.LabelFrame(left_panel, text="⚙️ Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Strategy + Metric
        row1 = ttk.Frame(config_frame)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Strategy:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Combobox(row1, textvariable=self.strategy_var,
                     values=["Momentum", "Kalman", "Scalping"],
                     width=12, state="readonly").pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row1, text="Metric:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Combobox(row1, textvariable=self.metric_var,
                     values=["Sharpe Ratio", "Return [%]", "Win Rate [%]",
                             "Profit Factor", "Sortino Ratio"],
                     width=16, state="readonly").pack(side=tk.LEFT)

        # Row 2: Symbol + Interval
        row2 = ttk.Frame(config_frame)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="Symbol:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2, textvariable=self.symbol_var, width=12).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row2, text="Interval:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Combobox(row2, textvariable=self.interval_var,
                     values=["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
                     width=8, state="readonly").pack(side=tk.LEFT)

        # Row 3: Date Range
        row3 = ttk.Frame(config_frame)
        row3.pack(fill=tk.X, pady=3)
        ttk.Label(row3, text="Start:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row3, textvariable=self.start_date_var, width=12).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(row3, text="End:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row3, textvariable=self.end_date_var, width=12).pack(side=tk.LEFT)

        # Row 4: Max Tries + Options
        row4 = ttk.Frame(config_frame)
        row4.pack(fill=tk.X, pady=3)
        ttk.Label(row4, text="Max Tries:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(row4, from_=20, to=500, textvariable=self.max_tries_var,
                    width=6).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Checkbutton(row4, text="💾 Auto-Update Settings",
                        variable=self.auto_update_var).pack(side=tk.LEFT)

        # ─── Priority Selection ─────────────────────────────────────────────
        self.priority_frame = PrioritySelectionFrame(left_panel, self)
        self.priority_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # ─── Buttons ────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        self.run_btn = tk.Button(btn_frame, text="🚀 Run Optimization",
                                 command=self._run_optimization,
                                 bg="#4CAF50", fg="white",
                                 font=('Arial', 11, 'bold'),
                                 padx=20, pady=4, cursor="hand2")
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop",
                                  command=self._stop_optimization,
                                  bg="#FF5555", fg="white",
                                  font=('Arial', 11, 'bold'),
                                  padx=20, pady=4, cursor="hand2",
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(btn_frame, text="💾 Save Results",
                                  command=self._save_results,
                                  bg="#2196F3", fg="white",
                                  font=('Arial', 11, 'bold'),
                                  padx=15, pady=4, cursor="hand2",
                                  state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # ─── Progress Bar ────────────────────────────────────────────────────
        progress_frame = ttk.Frame(left_panel)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(pady=2)

        # ─── Log Output ─────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(left_panel, text="📋 Output Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD,
                                                  font=("Consolas", 9),
                                                  bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_text.tag_config("green", foreground="#00FF00")
        self.log_text.tag_config("red", foreground="#FF5555")
        self.log_text.tag_config("yellow", foreground="#FFD700")
        self.log_text.tag_config("cyan", foreground="#00FFFF")
        self.log_text.tag_config("white", foreground="#FFFFFF")
        self.log_text.tag_config("purple", foreground="#DA70D6")

        # ─── Right panel: Results ──────────────────────────────────────────
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.results_frame = ttk.LabelFrame(right_panel, text="📊 Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD,
                                                      font=("Consolas", 9),
                                                      bg="#0d1117", fg="#c9d1d9")
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.results_text.tag_config("header", foreground="#58a6ff", font=("Consolas", 11, "bold"))
        self.results_text.tag_config("green", foreground="#3fb950")
        self.results_text.tag_config("red", foreground="#f85149")
        self.results_text.tag_config("yellow", foreground="#d29922")
        self.results_text.tag_config("cyan", foreground="#79c0ff")

        # Initial message
        self.results_text.insert(tk.END, "📊 Optimization Results\n", "header")
        self.results_text.insert(tk.END, "=" * 57 + "\n\n", "header")
        self.results_text.insert(tk.END, "Configure the parameters above,\n", "white")
        self.results_text.insert(tk.END, "select parameters in the priority tree,\n", "white")
        self.results_text.insert(tk.END, "then click 🚀 Run Optimization to begin.\n\n", "white")
        self.results_text.insert(tk.END, "The priority tree shows parameters grouped by\n", "white")
        self.results_text.insert(tk.END, "CRITICAL → HIGH → MEDIUM influence on the\n", "white")
        self.results_text.insert(tk.END, "selected metric. Check/uncheck individual\n", "white")
        self.results_text.insert(tk.END, "parameters to control what gets optimized.\n\n", "white")
        self.results_text.insert(tk.END, "Click 🔍 Verify Selection to check if you have\n", "white")
        self.results_text.insert(tk.END, "all the CRITICAL parameters selected.", "white")

        # ─── Status Bar ─────────────────────────────────────────────────────
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        self.status_label = ttk.Label(status_bar, text="✅ Ready", font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)

        ttk.Label(status_bar, text="v7.1 | Priority-Based Optimizer", font=('Arial', 8)).pack(side=tk.RIGHT)

    def _load_defaults(self):
        """Load defaults from environment."""
        if os.environ.get('BEST_PARAM_STRATEGY'):
            self.strategy_var.set(os.environ['BEST_PARAM_STRATEGY'])
        if os.environ.get('BEST_PARAM_SYMBOL'):
            self.symbol_var.set(os.environ['BEST_PARAM_SYMBOL'])
        if os.environ.get('BEST_PARAM_INTERVAL'):
            self.interval_var.set(os.environ['BEST_PARAM_INTERVAL'])
        if os.environ.get('BEST_PARAM_START_DATE'):
            self.start_date_var.set(os.environ['BEST_PARAM_START_DATE'])
        if os.environ.get('BEST_PARAM_END_DATE'):
            self.end_date_var.set(os.environ['BEST_PARAM_END_DATE'])

    def _log(self, message, color="white"):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", color)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _update_progress(self, value, message):
        """Update progress bar."""
        self.progress_var.set(value)
        self.progress_label.config(text=message)

        if self.run_start_time is not None and value and value > 0:
            elapsed = time.time() - self.run_start_time
            est_total = elapsed / (value / 100.0)
            remaining = max(0, est_total - elapsed)
            self.est_time_label.config(text=f"⏱ Est. remaining: {self._format_time(remaining)}")

        self.root.update_idletasks()

    @staticmethod
    def _format_time(seconds) -> str:
        """Format seconds as MM:SS."""
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _start_timers(self):
        """Start the elapsed timer."""
        self.run_start_time = time.time()
        self.elapsed_time_label.config(text="⏳ Elapsed: 00:00")
        self.est_time_label.config(text="⏱ Est. remaining: calculating...")
        self._tick_timer()

    def _tick_timer(self):
        """Update elapsed timer."""
        if not self.is_running or self.run_start_time is None:
            self.timer_job = None
            return
        elapsed = time.time() - self.run_start_time
        self.elapsed_time_label.config(text=f"⏳ Elapsed: {self._format_time(elapsed)}")
        self.timer_job = self.root.after(1000, self._tick_timer)

    def _stop_timers(self, final_message: Optional[str] = None):
        """Stop timer."""
        if self.timer_job is not None:
            try:
                self.root.after_cancel(self.timer_job)
            except Exception:
                pass
            self.timer_job = None
        if self.run_start_time is not None:
            elapsed = time.time() - self.run_start_time
            self.elapsed_time_label.config(text=f"⏳ Elapsed: {self._format_time(elapsed)}")
        self.est_time_label.config(text=final_message or "⏱ Est. remaining: 00:00")

    def _update_results(self, results):
        """Update results display."""
        try:
            self.results_text.delete(1.0, tk.END)

            if not results:
                self.results_text.insert(tk.END, "⚠️ No results returned.\n", "red")
                return

            if 'error' in results and results['error']:
                self.results_text.insert(tk.END, "❌ OPTIMIZATION FAILED\n", "header")
                self.results_text.insert(tk.END, "=" * 60 + "\n\n", "header")
                self.results_text.insert(tk.END, str(results['error']), "red")
                return

            metric = results.get('metric_used', 'N/A')
            baseline = results.get('baseline_stats', {}) or {}
            optimized = results.get('optimized_stats', {}) or {}
            direction = results.get('trade_direction', 'N/A')
            params_opt = results.get('params_optimized', 0)

            self.results_text.insert(tk.END, "🏆 OPTIMIZATION COMPLETE\n", "header")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n", "header")
            self.results_text.insert(tk.END,
                                     f"Direction: {str(direction).upper()}   |   Params optimized: {params_opt}\n\n",
                                     "white")

            # Performance comparison
            self.results_text.insert(tk.END, "📊 PERFORMANCE COMPARISON\n", "cyan")
            self.results_text.insert(tk.END, "-" * 60 + "\n", "white")

            self.results_text.insert(tk.END, f"{'Metric':<22} {'Baseline':>12} {'Optimized':>12} {'Change':>12}\n",
                                     "white")
            self.results_text.insert(tk.END, "-" * 60 + "\n", "white")

            for key in ['Sharpe Ratio', 'Return [%]', 'Win Rate [%]', '# Trades', metric]:
                b = baseline.get(key, 0)
                o = optimized.get(key, 0)
                if b == 0 and o == 0:
                    continue
                if key == '# Trades':
                    chg = o - b
                    color = "green" if chg > 0 else "red" if chg < 0 else "white"
                    arrow = "✅" if chg > 0 else "❌" if chg < 0 else "➡️"
                    self.results_text.insert(tk.END, f"{str(key):<22} {b:>12.0f} {o:>12.0f} {arrow} {chg:>+9.0f}\n",
                                             color)
                else:
                    chg = o - b
                    color = "green" if chg > 0 else "red" if chg < 0 else "white"
                    arrow = "✅" if chg > 0 else "❌" if chg < 0 else "➡️"
                    self.results_text.insert(tk.END, f"{str(key):<22} {b:>12.3f} {o:>12.3f} {arrow} {chg:>+9.3f}\n",
                                             color)

            # Best parameters
            self.results_text.insert(tk.END, "\n📋 BEST PARAMETERS\n", "cyan")
            self.results_text.insert(tk.END, "-" * 60 + "\n", "white")

            best_params = results.get('best_params', {}) or {}
            if best_params:
                for k, v in best_params.items():
                    self.results_text.insert(tk.END, f"  {str(k):<30} = {v}\n", "yellow")
            else:
                self.results_text.insert(tk.END, "  No parameters optimized\n", "white")

            # Optimized params list
            opt_params = results.get('optimized_params', [])
            if opt_params:
                self.results_text.insert(tk.END, f"\n📊 OPTIMIZED PARAMETERS ({len(opt_params)})\n", "cyan")
                self.results_text.insert(tk.END, "-" * 60 + "\n", "white")
                for i, p in enumerate(opt_params, 1):
                    self.results_text.insert(tk.END, f"  {i}. {p}\n", "white")

            self.results_text.insert(tk.END, "\n" + "=" * 60 + "\n", "header")
            self.results_text.insert(tk.END, "✅ Optimization complete!\n", "green")

        except Exception as e:
            import traceback
            self._log(f"❌ Error rendering results: {e}", "red")
            self._log(traceback.format_exc(), "red")
            try:
                self.results_text.insert(tk.END, "❌ ERROR DISPLAYING RESULTS\n", "header")
                self.results_text.insert(tk.END, "=" * 60 + "\n\n", "header")
                self.results_text.insert(tk.END, f"{e}\n\n", "red")
                self.results_text.insert(tk.END, "See the Output Log for details.\n", "yellow")
            except Exception:
                pass
        finally:
            self.root.update_idletasks()

    def _run_optimization(self):
        """Run optimization with selected parameters."""
        if self.is_running:
            return

        # Get selected parameters from priority frame
        selected_params = self.priority_frame.get_selected_params()

        if not selected_params:
            self._log("⚠️ No parameters selected for optimization", "yellow")
            messagebox.showwarning("No Parameters", "Please select at least one parameter to optimize.")
            return

        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)

        self.log_text.delete(1.0, tk.END)
        self.status_label.config(text="⏳ Running...")
        self._start_timers()

        # Get parameters
        strategy = self.strategy_var.get()
        metric = self.metric_var.get()
        symbol = self.symbol_var.get()
        interval = self.interval_var.get()
        start = self.start_date_var.get()
        end = self.end_date_var.get()
        max_tries = self.max_tries_var.get()
        auto_update = self.auto_update_var.get()

        api_key = load_api_key()

        self._log(f"🚀 Starting optimization with {len(selected_params)} parameters")
        self._log(f"   Strategy: {strategy}")
        self._log(f"   Metric: {metric}")
        self._log(f"   Params: {', '.join(selected_params[:10])}" + (
            f" ... and {len(selected_params) - 10} more" if len(selected_params) > 10 else ""))

        def worker():
            try:
                results = run_optimization(
                    strategy_name=strategy,
                    symbol=symbol,
                    interval=interval,
                    start_date=start,
                    end_date=end,
                    metric=metric,
                    param_names=selected_params,
                    max_tries=max_tries,
                    api_key=api_key,
                    log_callback=self._log,
                    progress_callback=self._update_progress,
                )

                self.results = results

                self.root.after(0, lambda: self._update_results(results))

                if 'error' not in results and results.get('best_params'):
                    if auto_update:
                        success = update_strategy_settings(strategy, results['best_params'])
                        if success:
                            self._log("✅ Auto-updated strategy_settings.json", "green")

                    self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.status_label.config(text="✅ Complete!"))
                    self.root.after(0, lambda: self._stop_timers("⏱ Est. remaining: 00:00 (done)"))
                else:
                    self.root.after(0, lambda: self.status_label.config(text="❌ Failed"))
                    self.root.after(0, lambda: self._stop_timers("⏱ Est. remaining: -- (failed)"))

            except Exception as e:
                self.root.after(0, lambda: self._log(f"❌ Error: {e}", "red"))
                self.root.after(0, lambda: self.status_label.config(text="❌ Error"))
                self.root.after(0, lambda: self._stop_timers("⏱ Est. remaining: -- (error)"))

            finally:
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                self.is_running = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _stop_optimization(self):
        """Stop optimization."""
        self.is_running = False
        self.status_label.config(text="⏹ Stopped")
        self._log("⏹ Optimization stopped by user", "yellow")
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._stop_timers("⏱ Est. remaining: -- (stopped)")

    def _save_results(self):
        """Save results to JSON."""
        if not self.results or 'error' in self.results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"optimized_{self.strategy_var.get().lower()}_{timestamp}.json"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                def convert(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    return obj

                save_data = {
                    'strategy': self.strategy_var.get(),
                    'timestamp': timestamp,
                    'metric': self.metric_var.get(),
                    'optimized_params': self.results.get('optimized_params', []),
                    'baseline': convert(self.results.get('baseline_stats', {})),
                    'optimized': convert(self.results.get('optimized_stats', {})),
                    'improvements': convert(self.results.get('improvements', {})),
                    'best_params': convert(self.results.get('best_params', {})),
                }

                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=4)

                self._log(f"💾 Results saved to: {file_path}", "green")
                self.status_label.config(text=f"💾 Saved: {os.path.basename(file_path)}")

            except Exception as e:
                self._log(f"❌ Failed to save: {e}", "red")

    def _on_close(self):
        """Handle window close."""
        if self.is_running:
            if messagebox.askyesno("Confirm", "Optimization is running. Stop and exit?"):
                self.is_running = False
                self.root.destroy()
        else:
            self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# 13. MAIN ENTRY
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer v7.1")
    parser.add_argument("--no-gui", action="store_true", help="Run in console mode")
    parser.add_argument("--strategy", default="Momentum", help="Strategy to optimize")
    parser.add_argument("--metric", default="Sharpe Ratio", help="Optimization metric")
    parser.add_argument("--symbol", default="SOL/USDT", help="Trading pair")
    parser.add_argument("--interval", default="15m", help="Timeframe")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-tries", type=int, default=200, help="Max combinations")
    parser.add_argument("--auto-update", action="store_true", help="Auto-update settings")
    parser.add_argument("--params", nargs="+", help="Specific parameters to optimize")
    args = parser.parse_args()

    # Console mode
    if args.no_gui:
        print("🏆 Running in console mode...")

        if args.start is None:
            args.start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        if args.end is None:
            args.end = datetime.now().strftime("%Y-%m-%d")

        # Load defaults to get available params
        StrategyClass, module_default_params, _, _ = import_strategy_module(args.strategy)
        if StrategyClass is None:
            print(f"❌ Could not import {args.strategy} strategy")
            sys.exit(1)

        default_params, _ = load_effective_default_params(args.strategy, module_default_params, print)

        # Determine which params to optimize
        if args.params:
            param_names = [p for p in args.params if p in default_params]
        else:
            # Use all params
            param_names = [p for p in default_params.keys()
                           if isinstance(default_params[p], (int, float, bool))
                           and p != 'trade_direction']

        print(f"📊 Optimizing {len(param_names)} parameters")

        results = run_optimization(
            strategy_name=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
            metric=args.metric,
            param_names=param_names,
            max_tries=args.max_tries,
            log_callback=print
        )

        if 'error' not in results and results.get('best_params') and args.auto_update:
            update_strategy_settings(args.strategy, results['best_params'])

        sys.exit(0)

    # GUI mode
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()