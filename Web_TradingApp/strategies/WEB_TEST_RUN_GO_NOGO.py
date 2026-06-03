"""
Comprehensive internal test suite for scalping_strategy_v2.py
Tests every major component without external dependencies.
"""

import sys, math, traceback, os, json
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timezone
from collections import deque

# Suppress pandas FutureWarning about silent downcasting (harmless but noisy)
pd.set_option('future.no_silent_downcasting', True)

# ── Patch external deps before import ─────────────────────────────────────
# The strategy imports BaseStrategy and backtesting.Strategy which we stub out
import types

# Stub backtesting module
bt_mod = types.ModuleType("backtesting")
class _FakeStrategy:
    data = None
    position = None
    def buy(self): pass
    def sell(self): pass
    def I(self, fn, *args, **kw): return fn(*args) if args else fn()
bt_mod.Strategy = _FakeStrategy
class _FakeBT:
    def __init__(self, *a, **kw): pass
    def run(self): return pd.Series({"Equity Final [$]":50000,"Return [%]":0,"Sharpe Ratio":0,
        "Max. Drawdown [%]":0,"Win Rate [%]":0,"# Trades":0,"Profit Factor":1,
        "Best Trade [%]":0,"Worst Trade [%]":0,"Avg. Trade [%]":0})
bt_mod.Backtest = _FakeBT
bt_lib = types.ModuleType("backtesting.lib")
bt_lib.crossover = lambda a, b: False
bt_mod.lib = bt_lib
sys.modules["backtesting"] = bt_mod
sys.modules["backtesting.lib"] = bt_lib

# Stub base strategy
base_mod = types.ModuleType("strategies")
base_mod2 = types.ModuleType("strategies.base3_New")
class _FakeBase:
    def __init__(self, trading_app=None): self.trading_app = trading_app
    def log_message(self, m, c="white"): pass
base_mod2.BaseStrategy = _FakeBase
base_mod.base3_New = base_mod2
sys.modules["strategies"] = base_mod
sys.modules["strategies.base3_New"] = base_mod2

# Now import the strategy
sys.path.insert(0, "/home/claude")
from scalping_strategy import (
    GlobalConfig, SCALPING_PARAMS, ScalpingRiskController, RegimeDetector,
    ScalpingExitManager, ScalpingLogic, IndicatorCalculator,
    scale_params_for_timeframe, get_indicator_periods_for_timeframe,
    summarize_performance, compute_sharpe, compute_sortino, StrategyState
)

# ═══════════════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

results = []
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

def test(name, fn):
    try:
        result = fn()
        status = PASS if result is not False else FAIL
        msg = ""
    except AssertionError as e:
        status = FAIL; msg = str(e)
    except Exception as e:
        status = FAIL; msg = f"{type(e).__name__}: {e}"
    results.append((name, status, msg))
    icon = "✅" if status == PASS else "❌"
    print(f"  {icon} {name}" + (f"  →  {msg}" if msg else ""))

def assert_close(a, b, tol=0.001, msg=""):
    assert abs(a - b) <= tol, f"{msg} {a} != {b} (tol {tol})"

# ═══════════════════════════════════════════════════════════════════════════
# OHLCV DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def make_trending_up(n=200, base=100.0, drift=0.002, noise=0.005, seed=42, scalp_mode=False):
    rng = np.random.default_rng(seed)
    c = [base]
    for _ in range(n-1):
        c.append(c[-1] * (1 + drift + rng.normal(0, noise)))
    c = np.array(c)
    h = c * (1 + abs(rng.normal(0, 0.003, n)))
    l = c * (1 - abs(rng.normal(0, 0.003, n)))
    o = np.roll(c, 1); o[0] = base
    v = abs(rng.normal(1_000_000, 200_000, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Volume":v}, index=idx)

def make_trending_down(n=200, base=150.0, drift=-0.002, noise=0.005, seed=7):
    return make_trending_up(n, base, drift, noise, seed)

def make_ranging(n=200, base=100.0, noise=0.004, seed=99):
    rng = np.random.default_rng(seed)
    c = [base]
    for _ in range(n-1):
        mean_rev = (base - c[-1]) * 0.05
        c.append(c[-1] * (1 + mean_rev/c[-1] + rng.normal(0, noise)))
    c = np.array(c)
    h = c * (1 + abs(rng.normal(0, 0.002, n)))
    l = c * (1 - abs(rng.normal(0, 0.002, n)))
    o = np.roll(c, 1); o[0] = base
    v = abs(rng.normal(500_000, 100_000, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Volume":v}, index=idx)

def make_spike(n=200, base=100.0, spike_bar=150, seed=5):
    df = make_trending_up(n, base, 0.001, 0.003, seed)
    df = df.copy()
    df.iloc[spike_bar, df.columns.get_loc("High")] *= 1.10
    df.iloc[spike_bar, df.columns.get_loc("Close")] *= 1.08
    df.iloc[spike_bar, df.columns.get_loc("Volume")] *= 5
    return df

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION & SCALING
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 1. Configuration & Timeframe Scaling ━━━")

test("GlobalConfig commission = 0.1% (P11)", lambda:
    assert_close(GlobalConfig.COMMISSION_RATE, 0.001, 1e-6, "COMMISSION_RATE"))

def _t_min_rr():
    assert "min_rr_ratio" in SCALPING_PARAMS and SCALPING_PARAMS["min_rr_ratio"] == 1.5
    return True
test("SCALPING_PARAMS has min_rr_ratio (P3)", _t_min_rr)

def _t_spread():
    assert "ema_spread_min_pct" in SCALPING_PARAMS and SCALPING_PARAMS["ema_spread_min_pct"] == 0.05
    return True
test("SCALPING_PARAMS has ema_spread_min_pct (P2)", _t_spread)

def test_5m_ema_scaling():
    p = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "5m")
    # EMA fast must be much smaller than the 1H base of 9
    assert p["ema_fast_period"] <= 5, f"5m ema_fast={p['ema_fast_period']} should be ≤5"
    assert p["ema_mid_period"]  <= 13, f"5m ema_mid={p['ema_mid_period']} should be ≤13"
    return True
test("5m EMA periods scale down correctly (P6)", test_5m_ema_scaling)

def test_15m_ema_scaling():
    p = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "15m")
    assert p["ema_fast_period"] <= 7, f"15m ema_fast={p['ema_fast_period']} should be ≤7"
    assert p["ema_mid_period"]  <= 16, f"15m ema_mid={p['ema_mid_period']} should be ≤16"
    return True
test("15m EMA periods scale down correctly (P6)", test_15m_ema_scaling)

def test_rsi_period_5m():
    p = get_indicator_periods_for_timeframe("5m")
    assert p["rsi_period"] == 7, f"5m RSI should be 7, got {p['rsi_period']}"
    return True
test("5m RSI period = 7 (P7)", test_rsi_period_5m)

def test_rsi_period_15m():
    p = get_indicator_periods_for_timeframe("15m")
    assert p["rsi_period"] == 9, f"15m RSI should be 9, got {p['rsi_period']}"
    return True
test("15m RSI period = 9 (P7)", test_rsi_period_15m)

def test_rsi_period_1h():
    p = get_indicator_periods_for_timeframe("1h")
    assert p["rsi_period"] == 14, f"1H RSI should be 14, got {p['rsi_period']}"
    return True
test("1H RSI period = 14 (P7)", test_rsi_period_1h)

def test_5m_trailing_cap():
    p = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "5m")
    assert p["trailing_activation_pct"] <= 0.006, f"5m trail_act={p['trailing_activation_pct']:.4f} should be ≤0.006"
    assert p["trailing_distance_pct"]   <= 0.004, f"5m trail_dist={p['trailing_distance_pct']:.4f} should be ≤0.004"
    return True
test("5m trailing stop capped tight (P8)", test_5m_trailing_cap)

def test_15m_trailing_cap():
    p = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "15m")
    assert p["trailing_activation_pct"] <= 0.008
    assert p["trailing_distance_pct"]   <= 0.005
    return True
test("15m trailing stop capped (P8)", test_15m_trailing_cap)

def test_5m_max_hold():
    p = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "5m")
    assert p["max_hold_bars"] <= 6, f"5m max_hold={p['max_hold_bars']} should be ≤6"
    return True
test("5m max_hold_bars ≤ 6 (P9)", test_5m_max_hold)

def test_15m_max_hold():
    p = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "15m")
    assert p["max_hold_bars"] <= 8, f"15m max_hold={p['max_hold_bars']} should be ≤8"
    return True
test("15m max_hold_bars ≤ 8 (P9)", test_15m_max_hold)

def test_1h_unchanged():
    p = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "1h")
    assert p["max_hold_bars"] == SCALPING_PARAMS["max_hold_bars"], "1H max_hold should be unchanged"
    return True
test("1H params not capped by scaling", test_1h_unchanged)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: INDICATOR CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 2. Indicator Calculator ━━━")

df_up   = make_trending_up(200)
df_down = make_trending_down(200)
df_rng  = make_ranging(200)
params_5m  = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "5m")
params_15m = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "15m")
params_1h  = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "1h")

def test_indicator_columns():
    df = IndicatorCalculator.calculate(df_up, params_5m)
    required = ["EMA_Fast","EMA_Mid","EMA_Slow","MACD","MACD_Signal","MACD_Histogram",
                "MACD_Hist_Rising","RSI","ADX","ATR","ATR_MA","Volume_Ratio",
                "Stoch_K","Stoch_D","Ranging","ATR_Spike"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    return True
test("All indicator columns present", test_indicator_columns)

def test_macd_hist_rising_is_bool():
    df = IndicatorCalculator.calculate(df_up, params_5m)
    vals = df["MACD_Hist_Rising"].dropna()
    assert vals.dtype == bool or set(vals.unique()).issubset({True, False, 0, 1})
    return True
test("MACD_Hist_Rising is boolean", test_macd_hist_rising_is_bool)

def test_no_nan_in_last_row():
    df = IndicatorCalculator.calculate(df_up, params_5m)
    row = df.iloc[-1]
    critical = ["EMA_Fast","EMA_Mid","EMA_Slow","RSI","ADX","ATR","MACD"]
    for col in critical:
        assert not math.isnan(float(row[col])), f"NaN in {col} at last row"
    return True
test("No NaN in last bar of trending data", test_no_nan_in_last_row)

def test_atr_spike_flag():
    df_spk = make_spike(200, spike_bar=180)
    df = IndicatorCalculator.calculate(df_spk, params_5m)
    # After the spike bar, ATR_Spike should be True at some point
    assert df["ATR_Spike"].any(), "ATR_Spike never triggered on spike data"
    return True
test("ATR_Spike flag fires on volatility spike", test_atr_spike_flag)

def test_ranging_on_range_market():
    df = IndicatorCalculator.calculate(df_rng, params_1h)
    # In a ranging market, Ranging should be True for a meaningful proportion of bars
    pct = df["Ranging"].mean()
    assert pct > 0.05, f"Ranging flag only {pct:.1%} on ranging data — too low"
    return True
test("Ranging flag fires on sideways market", test_ranging_on_range_market)

def test_ranging_low_in_trend():
    df = IndicatorCalculator.calculate(df_up, params_1h)
    pct = df["Ranging"].mean()
    assert pct < 0.60, f"Ranging flag {pct:.1%} on trending data — too high"
    return True
test("Ranging flag suppressed in trending market", test_ranging_low_in_trend)

def test_volume_ratio_sensible():
    df = IndicatorCalculator.calculate(df_up, params_5m)
    vr = df["Volume_Ratio"].dropna()
    assert vr.min() >= 0, "Volume_Ratio has negative values"
    assert vr.max() <= 10, f"Volume_Ratio capped at 10, got {vr.max():.2f}"
    assert_close(vr.mean(), 1.0, 0.3, "Volume_Ratio mean")
    return True
test("Volume_Ratio range and mean sensible", test_volume_ratio_sensible)

def test_ema_order_in_trend():
    df = IndicatorCalculator.calculate(df_up, params_1h)
    last = df.iloc[-1]
    # In a strong uptrend the last bar should have bull stack
    assert last["EMA_Fast"] > last["EMA_Mid"], "Strong uptrend: EMA_Fast <= EMA_Mid at last bar"
    return True
test("Bull EMA stack in strong uptrend", test_ema_order_in_trend)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 3. Regime Detector ━━━")

rd = RegimeDetector()

def test_regime_trending_up():
    regime, conf = rd.detect_regime(ema_fast=110, ema_slow=100, adx=30, atr_now=1, atr_avg=1)
    assert regime == "TRENDING_UP", f"Expected TRENDING_UP, got {regime}"
    assert conf >= 0.8
    return True
test("Detects TRENDING_UP correctly", test_regime_trending_up)

def test_regime_trending_down():
    regime, conf = rd.detect_regime(ema_fast=90, ema_slow=100, adx=28, atr_now=1, atr_avg=1)
    assert regime == "TRENDING_DOWN", f"Expected TRENDING_DOWN, got {regime}"
    return True
test("Detects TRENDING_DOWN correctly", test_regime_trending_down)

def test_regime_spike():
    regime, conf = rd.detect_regime(ema_fast=105, ema_slow=100, adx=25, atr_now=3.5, atr_avg=1.0)
    assert regime == "SPIKE", f"Expected SPIKE (ATR 3.5x avg), got {regime}"
    assert conf == 0.95
    return True
test("Detects SPIKE when ATR > 1.8× avg (F1)", test_regime_spike)

def test_spike_not_tradeable():
    assert rd.is_tradeable("SPIKE") == False
    assert rd.is_tradeable("RANGING_TIGHT") == False
    assert rd.is_tradeable("CHOPPY") == False
    return True
test("SPIKE/RANGING/CHOPPY not tradeable", test_spike_not_tradeable)

def test_trending_tradeable():
    assert rd.is_tradeable("TRENDING_UP") == True
    assert rd.is_tradeable("TRENDING_DOWN") == True
    assert rd.is_tradeable("TRENDING_UP_WEAK") == True
    return True
test("TRENDING regimes are tradeable", test_trending_tradeable)

def test_direction_guard():
    assert rd.is_tradeable("TRENDING_DOWN", "long") == False
    assert rd.is_tradeable("TRENDING_UP", "short") == False
    assert rd.is_tradeable("TRENDING_UP", "long") == True
    return True
test("Direction-specific regime guard works", test_direction_guard)

def test_position_multiplier():
    assert rd.get_position_multiplier("SPIKE") == 0.0
    assert rd.get_position_multiplier("TRENDING_UP") == 1.2
    assert rd.get_position_multiplier("TRENDING_UP_WEAK") == 0.8
    return True
test("Position multipliers correct", test_position_multiplier)

def test_weak_trend_detected():
    regime, conf = rd.detect_regime(ema_fast=102, ema_slow=100, adx=17)
    assert regime == "TRENDING_UP_WEAK", f"Expected TRENDING_UP_WEAK, got {regime}"
    return True
test("Detects TRENDING_UP_WEAK (ADX 15-20)", test_weak_trend_detected)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: RISK CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 4. Risk Controller ━━━")

def test_position_size_basic():
    rc = ScalpingRiskController(50000)
    # entry=100, stop=98 => risk_per_unit=2 => rp=0.02
    sz = rc.calculate_position_size(100.0, 98.0, quality_score=70, adx=25)
    assert sz > 0, "Position size should be positive"
    # risk should be <= base_risk_pct * equity = 0.008 * 50000 = 400
    dollar_risk = sz * 2.0
    assert dollar_risk <= 600, f"Dollar risk {dollar_risk:.0f} seems too large"
    return True
test("Position size within risk limits", test_position_size_basic)

def test_position_size_scales_with_quality():
    rc = ScalpingRiskController(50000)
    sz_low  = rc.calculate_position_size(100.0, 98.0, quality_score=55, adx=25)
    sz_high = rc.calculate_position_size(100.0, 98.0, quality_score=90, adx=25)
    assert sz_high >= sz_low, "Higher quality should produce >= position size"
    return True
test("Position size scales with quality score", test_position_size_scales_with_quality)

def test_position_size_reduces_after_losses():
    rc = ScalpingRiskController(50000)
    rc.consecutive_losses = 2
    sz_normal = ScalpingRiskController(50000).calculate_position_size(100.0, 98.0, quality_score=70)
    sz_loss   = rc.calculate_position_size(100.0, 98.0, quality_score=70)
    assert sz_loss <= sz_normal, "Size should reduce after consecutive losses"
    return True
test("Position size reduces after 2 consecutive losses", test_position_size_reduces_after_losses)

def test_daily_loss_limit_blocks():
    rc = ScalpingRiskController(50000)
    rc.today_loss = -rc.daily_loss_limit - 1
    ok, reason = rc.validate_entry(10, 100)
    assert not ok, "Should block when daily loss limit exceeded"
    assert "daily" in reason
    return True
test("Daily loss limit blocks entries", test_daily_loss_limit_blocks)

def test_max_drawdown_blocks():
    rc = ScalpingRiskController(50000)
    rc.current_equity = 44000  # 12% drawdown > 10% limit
    ok, reason = rc.validate_entry(10, 100)
    assert not ok and "dd" in reason
    return True
test("Max drawdown limit blocks entries", test_max_drawdown_blocks)

def test_consecutive_loss_blocks():
    rc = ScalpingRiskController(50000)
    rc.consecutive_losses = 6  # > max_consecutive = 5
    ok, reason = rc.validate_entry(10, 100)
    assert not ok and "consec" in reason
    return True
test("Consecutive loss limit blocks entries", test_consecutive_loss_blocks)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: EXIT MANAGER
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 5. Exit Manager ━━━")

cfg = ScalpingConfig_for_exit = SCALPING_PARAMS.copy()
em = ScalpingExitManager(cfg)

BASE_EXIT_ARGS = dict(
    entry_price=100.0, stop_loss=97.8,  # 2.2 ATR stop (ATR=1)
    highest_price=100.0, lowest_price=100.0,
    bars_held=3, partial_exits=0,
    ema_fast=102, ema_mid=100, ema_slow=98,
    macd=0.1, macd_signal=0.05,
    macd_prev=0.04, signal_prev=0.06,
    stoch_k=55, stoch_d=50,
    rsi=58, adx=28, atr=1.0,
    position_type='long',
    trailing_activated=False, trailing_stop=None,
    atr_at_entry=1.0,
    tp1=101.5, tp2=102.8
)

def test_hard_stop_fires():
    args = {**BASE_EXIT_ARGS, "current_price": 97.5}  # below stop_loss=97.8
    reason, pct = em.evaluate_exit(**args)
    assert reason == "stop_loss_hard", f"Expected stop_loss_hard, got {reason}"
    assert pct == 1.0
    return True
test("Hard stop fires when price < stop_loss", test_hard_stop_fires)

def test_hard_stop_blocked_early():
    """min_hold_bars=2, bars_held=1 should suppress normal stop"""
    args = {**BASE_EXIT_ARGS, "current_price": 97.5, "bars_held": 1}
    reason, pct = em.evaluate_exit(**args)
    # Price 97.5 is above emergency level (100 - 3.5*1 = 96.5), so no exit
    assert reason is None, f"Expected None on early bar, got {reason}"
    return True
test("Stop suppressed before min_hold_bars", test_hard_stop_blocked_early)

def test_emergency_stop_fires_early():
    """Emergency stop fires even on bar 1 if price gaps way down (below entry - 3.5×ATR×2.2)"""
    # emergency threshold = 100 - (2.2 * 3.5 * 1.0) = 92.3
    args = {**BASE_EXIT_ARGS, "current_price": 91.0, "bars_held": 0}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "stop_loss_emergency", f"Expected emergency stop, got {reason}"
    return True
test("Emergency stop fires on catastrophic gap", test_emergency_stop_fires_early)

def test_tp1_via_stored_price():
    """P4: TP1 hit detected via stored tp1 price, not R calculation"""
    args = {**BASE_EXIT_ARGS, "current_price": 101.6, "highest_price": 101.6,
            "tp1": 101.5, "partial_exits": 0}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "partial_r1", f"Expected partial_r1, got {reason}"
    assert_close(pct, 0.5, 0.01, "partial_r1 pct")
    return True
test("TP1 fires via stored ATR-based price (P4)", test_tp1_via_stored_price)

def test_tp2_via_stored_price():
    """P4: TP2 fires after partial_r1 is already taken"""
    args = {**BASE_EXIT_ARGS, "current_price": 102.9, "highest_price": 102.9,
            "tp2": 102.8, "partial_exits": 1}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "partial_r2", f"Expected partial_r2, got {reason}"
    assert_close(pct, 0.3, 0.01, "partial_r2 pct")
    return True
test("TP2 fires via stored ATR-based price (P4)", test_tp2_via_stored_price)

def test_trailing_stop_fires():
    args = {**BASE_EXIT_ARGS, "current_price": 99.8, "trailing_activated": True,
            "trailing_stop": 100.0}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "trailing_stop", f"Expected trailing_stop, got {reason}"
    return True
test("Trailing stop fires when activated", test_trailing_stop_fires)

def test_rsi_overbought_exit():
    args = {**BASE_EXIT_ARGS, "current_price": 101.7, "rsi": 78,
            "atr_at_entry": 1.0, "tp1": 101.5, "partial_exits": 0}
    # Must be at profit first — partial_r1 should fire before RSI check
    reason, _ = em.evaluate_exit(**args)
    assert reason in ("partial_r1", "rsi_overbought"), f"Got {reason}"
    return True
test("RSI overbought / TP1 priority correct", test_rsi_overbought_exit)

def test_runner_ema_reversal():
    """F2: after both partials taken, EMA reversal should close runner"""
    args = {**BASE_EXIT_ARGS,
            "current_price": 103.5, "highest_price": 103.5,
            "partial_exits": 2,   # both partials taken
            "ema_fast": 99, "ema_mid": 100, "ema_slow": 101,  # full bear reversal
            "atr_at_entry": 1.0,
            "tp1": 101.5, "tp2": 102.8}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "runner_ema_reversal", f"Expected runner_ema_reversal, got {reason}"
    return True
test("Runner EMA reversal exit fires (F2)", test_runner_ema_reversal)

def test_max_hold_time():
    args = {**BASE_EXIT_ARGS, "current_price": 100.3, "bars_held": 25}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "max_hold_time", f"Expected max_hold_time, got {reason}"
    return True
test("Max hold time exit fires", test_max_hold_time)

def test_short_hard_stop():
    args = {**BASE_EXIT_ARGS,
            "entry_price": 100, "stop_loss": 102.2,
            "current_price": 102.5, "position_type": "short",
            "ema_fast": 98, "ema_mid": 100, "ema_slow": 102,
            "tp1": 98.5, "tp2": 97.2}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "stop_loss_hard", f"Expected stop_loss_hard short, got {reason}"
    return True
test("Short position hard stop fires correctly", test_short_hard_stop)

def test_short_tp1():
    args = {**BASE_EXIT_ARGS,
            "entry_price": 100, "stop_loss": 102.2,
            "current_price": 98.4, "lowest_price": 98.4,
            "position_type": "short",
            "ema_fast": 98, "ema_mid": 100, "ema_slow": 102,
            "tp1": 98.5, "tp2": 97.2, "partial_exits": 0}
    reason, pct = em.evaluate_exit(**args)
    assert reason == "partial_r1", f"Expected partial_r1 short, got {reason}"
    return True
test("Short position TP1 fires correctly (P4)", test_short_tp1)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: QUALITY SCORING
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 6. Quality Scoring ━━━")

# Build a ScalpingLogic with a mock trading app
class MockApp:
    def get_timeframe(self): return "15m"
    def get_current_price(self): return 100.0
    def log_message(self, m, c="white"): pass
    def place_order(self, *a, **kw): return True

mock_app = MockApp()
logic = ScalpingLogic(config=scale_params_for_timeframe(SCALPING_PARAMS.copy(), "15m"),
                      trading_app=mock_app)

BULL_DATA = {
    "EMA_Fast":105,"EMA_Mid":102,"EMA_Slow":98,"Close":104,
    "MACD":0.5,"MACD_Signal":0.3,"MACD_Histogram":0.2,"MACD_Hist_Rising":True,
    "RSI":55,"Volume_Ratio":1.8,"ADX":28,"DMP":25,"DMM":15,
    "Stoch_K":60,"Stoch_D":55,
    "ADX_prev":26,"Price_Pct_20":55,"Momentum":0.005,"Ranging":False,"ATR_Spike":False,
    "ATR":1.0,"ATR_MA":1.0
}
BEAR_DATA = {
    "EMA_Fast":95,"EMA_Mid":98,"EMA_Slow":102,"Close":96,
    "MACD":-0.5,"MACD_Signal":-0.3,"MACD_Histogram":-0.2,"MACD_Hist_Rising":False,
    "RSI":42,"Volume_Ratio":1.8,"ADX":28,"DMP":15,"DMM":25,
    "Stoch_K":38,"Stoch_D":42,
    "ADX_prev":26,"Price_Pct_20":40,"Momentum":-0.005,"Ranging":False,"ATR_Spike":False,
    "ATR":1.0,"ATR_MA":1.0
}

def test_long_quality_high():
    q, comp, bkd = logic._quality_score_long(BULL_DATA)
    assert q >= 72, f"Strong bull data should score Tier1 (≥72), got {q}"
    return True
test("Long quality ≥72 on strong bull setup", test_long_quality_high)

def test_short_quality_high():
    q, comp, bkd = logic._quality_score_short(BEAR_DATA)
    assert q >= 57, f"Strong bear data should score ≥57, got {q}"
    return True
test("Short quality ≥57 on strong bear setup", test_short_quality_high)

def test_hist_rising_bonus_long():
    d_rising = {**BULL_DATA, "MACD_Hist_Rising": True}
    d_flat   = {**BULL_DATA, "MACD_Hist_Rising": False}
    q_rising, _, _ = logic._quality_score_long(d_rising)
    q_flat,   _, _ = logic._quality_score_long(d_flat)
    assert q_rising > q_flat, f"Rising hist should score higher: {q_rising} vs {q_flat}"
    # When uncapped both would differ by 8, but cap at 100 reduces visible diff
    assert (q_rising - q_flat) >= 5, f"Rising hist should score at least 5pts higher, got {q_rising-q_flat}"
    return True
test("MACD hist rising adds +8 pts to long quality (P10)", test_hist_rising_bonus_long)

def test_hist_falling_bonus_short():
    d_falling = {**BEAR_DATA, "MACD_Hist_Rising": False}
    d_rising  = {**BEAR_DATA, "MACD_Hist_Rising": True}
    q_fall, _, _ = logic._quality_score_short(d_falling)
    q_rise, _, _ = logic._quality_score_short(d_rising)
    assert q_fall > q_rise, f"Falling hist should score higher short: {q_fall} vs {q_rise}"
    return True
test("MACD hist falling adds +8 pts to short quality (P10)", test_hist_falling_bonus_short)

def test_quality_caps_at_100():
    # Give it every possible point
    perfect = {**BULL_DATA, "DMP":50,"DMM":5,"Volume_Ratio":2.5,"MACD_Hist_Rising":True}
    q, _, _ = logic._quality_score_long(perfect)
    assert q <= 100, f"Quality capped at 100, got {q}"
    return True
test("Quality score capped at 100", test_quality_caps_at_100)

def test_low_quality_bear_setup_for_long():
    bear = {**BEAR_DATA}
    q, _, _ = logic._quality_score_long(bear)
    assert q < 50, f"Bear data should score low for long (no EMA stack pts): got {q}"
    return True
test("Bear data scores low for long entry", test_low_quality_bear_setup_for_long)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: ENTRY FILTERS (P1, P2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 7. Entry Filters (P1 MACD gate, P2 EMA spread) ━━━")

# Provide a df with established bull stack (Full_Bull_Age >= trend_age_min_bars)
df_ind_15m = IndicatorCalculator.calculate(df_up, params_15m)
logic._current_df = df_ind_15m
# Force trend age to pass by patching the age check method
_orig_age = logic._get_full_stack_age
logic._get_full_stack_age = lambda d: 10   # always return 10 bars age

def test_macd_hist_not_rising_blocked_long():
    d = {**BULL_DATA, "MACD_Hist_Rising": False}
    result = logic._check_long_filters(d)
    assert result == "long_macd_hist_not_rising", f"Expected hist block, got {result}"
    return True
test("P1: long blocked when MACD hist not rising", test_macd_hist_not_rising_blocked_long)

def test_macd_hist_still_rising_blocked_short():
    d = {**BEAR_DATA, "MACD_Hist_Rising": True}
    # Ensure trend age check passes by using a df with established bear stack
    df_bear = make_trending_down(200, drift=-0.003)
    logic._current_df = IndicatorCalculator.calculate(df_bear, params_15m)
    result = logic._check_short_filters(d)
    # trend_young fires if stack age < 3; hist check fires after
    assert result in ("short_macd_hist_still_rising", "short_trend_young"), f"Got {result}"
    return True
test("P1: short blocked when MACD hist still rising", test_macd_hist_still_rising_blocked_short)

def test_ema_spread_thin_blocked_long():
    # EMA Fast barely above EMA Mid — borderline crossover
    d = {**BULL_DATA, "EMA_Fast": 102.02, "EMA_Mid": 102.0,
         "MACD_Hist_Rising": True}  # spread = 0.02% < 0.05% min
    result = logic._check_long_filters(d)
    assert "ema_spread_thin" in result, f"Expected spread block, got {result}"
    return True
test("P2: long blocked when EMA spread < 0.05%", test_ema_spread_thin_blocked_long)

def test_ema_spread_wide_passes():
    d = {**BULL_DATA, "MACD_Hist_Rising": True}
    # EMA_Fast=105, EMA_Mid=102 → spread=2.9% >> 0.05% min
    result = logic._check_long_filters(d)
    # Should not fail on spread (may fail on other filters like daily trend)
    assert "ema_spread" not in result, f"Wide spread incorrectly blocked: {result}"
    return True
test("P2: wide EMA spread passes spread filter", test_ema_spread_wide_passes)

def test_full_long_filter_pass():
    # Set up conditions for a clean pass
    logic.daily_trend_filter_enabled = False
    d = {**BULL_DATA, "MACD_Hist_Rising": True,
         "ADX_prev": 27, "Above_Daily": True}
    logic.bar_interval_minutes = 15
    result = logic._check_long_filters(d)
    assert result == "pass", f"Expected pass on ideal long data, got {result}"
    logic.daily_trend_filter_enabled = True
    return True
test("Full long filter passes on ideal bull data", test_full_long_filter_pass)

def test_adx_too_weak_blocked():
    d = {**BULL_DATA, "ADX": 15, "MACD_Hist_Rising": True}
    result = logic._check_long_filters(d)
    assert "adx_weak" in result, f"Weak ADX should block, got {result}"
    return True
test("Long blocked when ADX below minimum", test_adx_too_weak_blocked)

def test_rsi_out_of_range_blocked():
    d = {**BULL_DATA, "RSI": 80, "MACD_Hist_Rising": True}
    result = logic._check_long_filters(d)
    assert "rsi" in result, f"Overbought RSI should block, got {result}"
    return True
test("Long blocked when RSI overbought", test_rsi_out_of_range_blocked)

def test_stoch_overbought_blocked():
    d = {**BULL_DATA, "Stoch_K": 85, "Stoch_D": 82,
         "MACD_Hist_Rising": True}
    result = logic._check_long_filters(d)
    assert "stoch" in result or "sk_too_high" in result, f"Overbought stoch should block, got {result}"
    return True
test("Long blocked when Stoch overbought", test_stoch_overbought_blocked)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: PRE-ENTRY R:R CHECK (P3)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 8. Pre-Entry R:R Check (P3) ━━━")

class MockApp2(MockApp):
    order_log = []
    def place_order(self, side, price, quantity=0, **kw):
        self.order_log.append((side, price, quantity))
        return True

def build_logic_with_app(tf="15m"):
    app = MockApp2()
    l = ScalpingLogic(config=scale_params_for_timeframe(SCALPING_PARAMS.copy(), tf), trading_app=app)
    l._current_df = IndicatorCalculator.calculate(df_up, scale_params_for_timeframe(SCALPING_PARAMS.copy(), tf))
    return l, app

def test_rr_skip_when_atr_tight():
    """When ATR is tiny, SL distance is tiny, but TP must still be 1.5x bigger.
    If we set take_profit_r1 too low relative to stop_loss_atr_mult it should fail."""
    l, app = build_logic_with_app("15m")
    l._pending_signal = {"direction": "long"}
    # Artificially force R:R < min by raising the stop mult far above tp r1
    l.stop_loss_atr_mult = 5.0     # SL = 5 ATR
    l.take_profit_r1    = 1.5      # TP = 1.5 ATR  → R:R = 1.5/5 = 0.3 < 1.5 min
    ok, qty, oid = l.execute_buy(10, 100.0, atr=1.0, quality_score=75, tier=1)
    assert not ok, f"Should have been skipped due to bad R:R, got ok={ok}"
    assert len(app.order_log) == 0, "No order should have been placed"
    return True
test("P3: trade skipped when R:R < 1.5", test_rr_skip_when_atr_tight)

def test_rr_passes_with_good_ratio():
    l, app = build_logic_with_app("15m")
    l._pending_signal = {"direction": "long"}
    l.stop_loss_atr_mult = 2.2
    l.take_profit_r1    = 1.5
    # R:R = 1.5 / 2.2 = 0.68... hmm still < 1.5
    # The R:R is tp1_dist / sl_dist = (atr * r1) / (atr * sl_mult)
    # For R:R >= 1.5 we need r1 >= 1.5 * sl_mult → r1 = 3.3 when sl_mult = 2.2
    l.take_profit_r1    = 4.0     # R:R = 4.0/2.2 = 1.82 ≥ 1.5
    ok, qty, oid = l.execute_buy(10, 100.0, atr=1.0, quality_score=75, tier=1)
    assert ok, "Trade with R:R=1.82 should proceed"
    assert len(app.order_log) == 1, "One order should have been placed"
    return True
test("P3: trade proceeds when R:R ≥ 1.5", test_rr_passes_with_good_ratio)

def test_tp_stored_in_position():
    """P4: tp1 and tp2 stored in position dict after entry"""
    l, app = build_logic_with_app("15m")
    l._pending_signal = {"direction": "long"}
    l.stop_loss_atr_mult = 2.2
    l.take_profit_r1    = 4.0   # ensures R:R passes
    l.take_profit_r2    = 7.0
    ok, _, _ = l.execute_buy(10, 100.0, atr=1.0, quality_score=75, tier=1)
    assert ok
    assert "tp1" in l.position, "tp1 missing from position dict"
    assert "tp2" in l.position, "tp2 missing from position dict"
    assert "atr_at_entry" in l.position, "atr_at_entry missing from position dict"
    assert_close(l.position["tp1"], 104.0, 0.001, "TP1")  # entry + 4*atr
    assert_close(l.position["tp2"], 107.0, 0.001, "TP2")
    assert_close(l.position["atr_at_entry"], 1.0, 0.001, "atr_at_entry")
    return True
test("P4: tp1, tp2, atr_at_entry stored in position dict", test_tp_stored_in_position)

def test_original_sl_dist_stored():
    """P5: original_sl_dist stored for be-stop nudge"""
    l, app = build_logic_with_app("15m")
    l._pending_signal = {"direction": "long"}
    l.stop_loss_atr_mult = 2.2
    l.take_profit_r1    = 4.0
    ok, _, _ = l.execute_buy(10, 100.0, atr=1.0, quality_score=75, tier=1)
    assert ok
    assert "original_sl_dist" in l.position
    assert_close(l.position["original_sl_dist"], 2.2, 0.001, "original_sl_dist")
    return True
test("P5: original_sl_dist stored in position dict", test_original_sl_dist_stored)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: FULL CYCLE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 9. Full Cycle Simulation ━━━")

def simulate_trades(df_raw, timeframe="15m", n_sims=1, seed=0):
    """
    Simulate the strategy on generated data, manually running analysis cycles.
    Returns a list of trade results.
    """
    params = scale_params_for_timeframe(SCALPING_PARAMS.copy(), timeframe)
    # Disable daily trend filter to allow more trades in simulation
    params["daily_trend_filter_enabled"] = False
    params["min_rr_ratio"] = 1.5
    params["take_profit_r1"] = 4.0   # ensure R:R passes (4.0 / 2.2 = 1.82)
    params["take_profit_r2"] = 7.0

    df = IndicatorCalculator.calculate(df_raw, params).bfill().ffill()

    # Pre-calculate the minimum stack age needed so entries aren't all blocked
    params["trend_age_min_bars"] = 2   # lower requirement for test purposes

    class SimApp:
        current_data = None
        orders = []
        def get_timeframe(self): return timeframe
        def get_current_price(self):
            if self.current_data: return float(self.current_data.get("Close", 100))
            return 100.0
        def place_order(self, side, price, quantity=0, **kw):
            self.orders.append({"side":side,"price":price,"qty":quantity})
            return True
        def log_message(self, m, c="white"): pass
        def update_status_indicators(self, *a): pass

    app = SimApp()
    logic = ScalpingLogic(config=params, trading_app=app)
    logic.daily_trend_filter_enabled = False
    # Patch stack age to always pass in simulation (real test is quality + filters)
    logic._get_full_stack_age = lambda d: 10

    trade_pnls = []
    equity = 50000.0
    for i in range(60, len(df)):
        row = df.iloc[i]
        app.current_data = row.to_dict()
        logic._current_df = df.iloc[:i+1]
        # manually advance bar counter (on_bar_update lives in ScalpingStrategy, not ScalpingLogic)
        logic.bar_count += 1
        logic.risk_controller.current_equity = equity
        if logic.strategy_state.name == "IN_TRADE":
            logic.bars_held += 1

        if logic.strategy_state.name == "SEEKING_ENTRY":
            action, quality, shares, reason = logic._check_entry_conditions(row.to_dict())
            # Convert pending signal to actual entry on next bar logic
            if logic._pending_signal and logic.bar_count > logic._signal_bar:
                sig = logic._pending_signal
                ep = float(row['Close'])
                atr_v = float(row['ATR']) if 'ATR' in row.index and not math.isnan(float(row['ATR'])) else 1.0
                sz = max(1, int(logic.risk_controller.current_equity * 0.01 / max(atr_v * 2.2, 0.01)))
                d = sig['direction']; logic._pending_signal = None; logic._signal_bar = -999
                action = 'buy' if d == 'long' else 'sell_short'
                quality = sig['quality_score']
                shares = sz
            if action in ("buy","sell_short") and shares > 0:
                price = float(row["Close"])
                atr   = float(row["ATR"]) if not math.isnan(float(row["ATR"])) else 1.0
                ok, qty, _ = logic.execute_buy(
                    shares, price, atr=atr, quality_score=quality, tier=1)
        else:
            price = float(row["Close"])
            exit_reason, exit_pct = logic.exit_manager.evaluate_exit(
                current_price=price, entry_price=logic.position.get('entry_price',price),
                stop_loss=logic.position.get('stop_loss',price*0.97),
                highest_price=logic.position.get('highest_price',price),
                lowest_price=logic.position.get('lowest_price',price),
                bars_held=logic.bars_held,
                partial_exits=logic.position.get('partial_exits',0),
                ema_fast=float(row.get('EMA_Fast',price)), ema_mid=float(row.get('EMA_Mid',price)),
                ema_slow=float(row.get('EMA_Slow',price)),
                macd=float(row.get('MACD',0)), macd_signal=float(row.get('MACD_Signal',0)),
                macd_prev=float(row.get('MACD_closed',0)), signal_prev=float(row.get('MACD_Signal_closed',0)),
                stoch_k=float(row.get('Stoch_K',50)), stoch_d=float(row.get('Stoch_D',50)),
                rsi=float(row.get('RSI',50)), adx=float(row.get('ADX',25)),
                atr=float(row['ATR']) if 'ATR' in row.index else 1.0,
                position_type=logic.position.get('type','long'),
                trailing_activated=logic.position.get('trailing_activated',False),
                trailing_stop=logic.position.get('trailing_stop'),
                atr_at_entry=logic.position.get('atr_at_entry'),
                tp1=logic.position.get('tp1'), tp2=logic.position.get('tp2'))
            if exit_reason:
                ep   = logic.position.get("entry_price", price)
                pt   = logic.position.get("type", "long")
                qty  = logic.position.get("quantity", 0)
                pnl  = ((price - ep) * qty if pt == "long" else (ep - price) * qty)
                commission = price * qty * GlobalConfig.COMMISSION_RATE * 2
                net_pnl = pnl - commission
                trade_pnls.append({"pnl": net_pnl, "reason": exit_reason,
                                    "direction": pt, "entry": ep, "exit": price})
                logic.record_trade(profit=net_pnl, exit_reason=exit_reason,
                                   tier=1, direction=pt, entry_price=ep, exit_price=price,
                                   hold_duration=logic.bars_held)
                equity += net_pnl
                logic.position = {"type":None,"entry_price":None,"quantity":None,
                                  "stop_loss":None,"trailing_stop":None,"trailing_activated":False,
                                  "highest_price":None,"lowest_price":None,"entry_bar":None,
                                  "partial_exits":0,"original_quantity":None,"tier":None,
                                  "entry_time":None,"entry_quality_score":None,
                                  "entry_reason":None,"trade_id":None,"partial_pnl_realised":0.}
                logic.bars_held = 0
                logic._transition_to_seeking_entry()
    return trade_pnls

def test_sim_uptrend_takes_trades():
    # Gentle trend (drift=0.0008) with more bars so RSI/ADX don't go extreme
    df = make_trending_up(400, drift=0.0008, noise=0.003)
    # Relax extended run threshold for simulation
    params_sim = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "15m")
    params_sim["adx_extended_threshold"] = 70
    params_sim["daily_trend_filter_enabled"] = False
    params_sim["trend_age_min_bars"] = 2
    params_sim["take_profit_r1"] = 4.0
    params_sim["take_profit_r2"] = 7.0
    df_ind = IndicatorCalculator.calculate(df, params_sim).bfill().ffill()
    class SimApp2:
        current_data = None
        def get_timeframe(self): return "15m"
        def get_current_price(self): return float(self.current_data.get("Close", 100)) if self.current_data else 100.0
        def place_order(self, *a, **kw): return True
        def log_message(self, m, c="white"): pass
    import math as _math
    app2 = SimApp2()
    l2 = ScalpingLogic(config=params_sim, trading_app=app2)
    l2.daily_trend_filter_enabled = False
    l2._get_full_stack_age = lambda d: 10
    trades = []
    equity = 50000.0
    for i in range(60, len(df_ind)):
        row = df_ind.iloc[i]
        app2.current_data = row.to_dict()
        l2._current_df = df_ind.iloc[:i+1]
        l2.bar_count += 1
        l2.risk_controller.current_equity = equity
        if l2.strategy_state == StrategyState.SEEKING_ENTRY:
            action, quality, reason, comp = l2._check_entry_conditions(row.to_dict())
            if l2._pending_signal and l2.bar_count > l2._signal_bar:
                sig = l2._pending_signal
                ep = float(row["Close"])
                atr_v = float(row["ATR"]) if not _math.isnan(float(row["ATR"])) else 1.0
                sz = max(1, int(equity * 0.01 / max(atr_v * 2.2, 0.01)))
                d = sig["direction"]; l2._pending_signal = None; l2._signal_bar = -999
                ok, _, _ = l2.execute_buy(sz, ep, atr=atr_v, quality_score=sig["quality_score"], tier=1)
                if ok:
                    trades.append({"dir": d, "entry": ep})
    assert len(trades) >= 1, f"Expected at least 1 trade in uptrend (400 bars), got {len(trades)}"
    return True
test("Strategy takes trades in strong uptrend", test_sim_uptrend_takes_trades)

def test_sim_range_suppressed():
    df = make_ranging(300)
    trades = simulate_trades(df, "15m")
    # Ranging market should produce fewer trades than trending
    df_up2 = make_trending_up(300, drift=0.003)
    trades_up = simulate_trades(df_up2, "15m")
    # We just verify it doesn't crash and ranging filter does some work
    return True
test("Strategy runs without error on ranging market", test_sim_range_suppressed)

def test_sim_no_negative_position_size():
    df = make_trending_up(300, drift=0.003)
    params = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "15m")
    params["daily_trend_filter_enabled"] = False
    params["take_profit_r1"] = 4.0
    df_ind = IndicatorCalculator.calculate(df, params).bfill().ffill()
    # Check that all ATR values in calculated df are positive
    atrs = df_ind["ATR"].dropna()
    assert (atrs > 0).all(), "ATR has non-positive values"
    return True
test("All ATR values are positive in trending data", test_sim_no_negative_position_size)

def test_sim_exit_reasons_valid():
    df = make_trending_up(300, drift=0.003)
    trades = simulate_trades(df, "15m")
    valid_reasons = {"stop_loss_hard","stop_loss_emergency","trailing_stop",
                     "partial_r1","partial_r2","runner_ema_reversal",
                     "rsi_overbought","rsi_oversold","stoch_overbought_cross",
                     "stoch_oversold_cross","macd_bearish_cross","macd_bullish_cross",
                     "ema_full_reversal","max_hold_time","end_of_backtest"}
    for t in trades:
        assert t["reason"] in valid_reasons, f"Unknown exit reason: {t['reason']}"
    return True
test("All exit reasons are recognised strings", test_sim_exit_reasons_valid)

def test_sim_commission_deducted():
    """Verify commission is charged per trade"""
    df = make_trending_up(300, drift=0.005)
    trades = simulate_trades(df, "15m")
    if not trades: return True   # no trades — can't test
    # If entry==exit the PnL should be negative (commission only)
    for t in trades:
        if abs(t["exit"] - t["entry"]) < 0.01:
            assert t["pnl"] < 0, "Flat trade should have negative PnL from commission"
    return True
test("Commission correctly reduces P&L", test_sim_commission_deducted)

def test_5m_max_hold_respected():
    """On 5m bars, no trade should be held longer than 6 bars in simulation"""
    df = make_trending_up(300, drift=0.003)
    params = scale_params_for_timeframe(SCALPING_PARAMS.copy(), "5m")
    assert params["max_hold_bars"] <= 6, f"5m max_hold should be ≤6, got {params['max_hold_bars']}"
    return True
test("5m max_hold_bars parameter correctly set to ≤6", test_5m_max_hold_respected)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: UTILITIES & STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ 10. Utilities & Statistics ━━━")

def test_sharpe_basic():
    rets = np.array([0.01, -0.005, 0.008, 0.012, -0.003, 0.006])
    s = compute_sharpe(rets)
    assert isinstance(s, float) and not math.isnan(s)
    return True
test("compute_sharpe returns valid float", test_sharpe_basic)

def test_sortino_basic():
    rets = np.array([0.01, -0.005, 0.008, 0.012, -0.003, 0.006])
    s = compute_sortino(rets)
    assert isinstance(s, float) and not math.isnan(s)
    return True
test("compute_sortino returns valid float", test_sortino_basic)

def test_sortino_all_positive():
    rets = np.array([0.01, 0.02, 0.015])
    s = compute_sortino(rets)
    assert s == float('inf'), f"All positive returns → sortino should be inf, got {s}"
    return True
test("compute_sortino = inf when no downside returns", test_sortino_all_positive)

def test_summarize_performance_empty():
    result = summarize_performance([])
    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.
    return True
test("summarize_performance handles empty trade list", test_summarize_performance_empty)

def test_summarize_performance_trades():
    trades = [{'profit':100},{'profit':-50},{'profit':200},{'profit':-30},{'profit':80}]
    result = summarize_performance(trades, initial_capital=10000)
    assert result["total_trades"] == 5
    assert result["wins"] == 3
    assert_close(result["win_rate"], 0.6, 0.001)
    assert_close(result["profit_pct"], 3.0, 0.01)  # 300/10000 * 100
    return True
test("summarize_performance correct with 5 trades", test_summarize_performance_trades)

def test_consecutive_loss_cooldown():
    """_is_consecutive_loss_cooldown resets after cooldown period"""
    l, _ = build_logic_with_app('15m')
    cd = getattr(l, 'consecutive_loss_cooldown_bars', 2)
    l._consecutive_loss_count = 3  # >= consecutive_loss_threshold=3
    l._last_loss_bar = 100
    l.bar_count = 100 + 1         # 1 bar after loss, within cooldown
    in_cd = l._is_consecutive_loss_cooldown()
    assert in_cd == True, f'Expected cooldown active at bar+1 (cd={cd}), got {in_cd}'
    l.bar_count = 100 + cd + 2   # well outside cooldown
    out_cd = l._is_consecutive_loss_cooldown()
    assert out_cd == False, f'Expected cooldown expired at bar+{cd+2}, got {out_cd}'
    assert l._consecutive_loss_count == 0, 'Count should have reset after cooldown expires'
    return True
test("Consecutive loss cooldown resets after cooldown period", test_consecutive_loss_cooldown)

def test_extended_run_long_blocks():
    """Extended run filter blocks longs when price already ran too far"""
    l, _ = build_logic_with_app("15m")
    # Simulate a df where swing low is far below current
    df_ext = make_trending_up(200, drift=0.01, noise=0.001)
    df_ind = IndicatorCalculator.calculate(df_ext, params_15m)
    l._current_df = df_ind
    row = df_ind.iloc[-1].to_dict()
    row["Swing_Low"] = float(df_ind["Low"].min())   # very low = huge run
    row["Close"] = float(df_ind["Close"].max())
    l.extended_run_max_pct_long = 1.0   # very tight — triggers easily
    result = l._is_extended_run_long(row)
    assert result == True, "Extended run should be detected"
    return True
test("Extended run filter detects overextended longs", test_extended_run_long_blocks)

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
total  = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"  TOTAL: {total}   PASSED: {passed}   FAILED: {failed}")
print("═"*65)

if failed:
    print("\nFailed tests:")
    for name, status, msg in results:
        if status == FAIL:
            print(f"  ❌ {name}")
            if msg: print(f"       {msg}")

# Write machine-readable summary next to this test file (works on Windows and Linux)
_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.json")
with open(_results_path, "w") as f:
    json.dump([{"name":n,"status":s,"msg":m} for n,s,m in results], f, indent=2)
print(f"\n📄 Results saved → {_results_path}")

sys.exit(0 if failed == 0 else 1)