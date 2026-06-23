"""
optimize_momentum_sol_15m.py
══════════════════════════════════════════════════════════════════════════
Staged in-sample / out-of-sample optimizer for BacktestMomentumStrategy
(MomentumStrategy_MACD_HybridScore_Latest.py) on Binance SOL-USDT 15m data.

WHY STAGED, NOT ONE GIANT GRID SEARCH
--------------------------------------
ALL_MOMENTUM_BACKTEST_PARAMS has ~50 tunable parameters. Optimizing all of
them at once against 6-12 months of 15m data (your stated data window)
gives the optimizer vastly more degrees of freedom than your trade count
can support, which reliably produces a parameter set that fits historical
noise rather than real signal ("overfitting"). It will look fantastic on
the data you tested and can fail badly on data it hasn't seen.

This script instead:
  1. Splits your CSV chronologically into an IN-SAMPLE (IS) chunk (first
     ~70%) and an OUT-OF-SAMPLE (OOS) chunk (last ~30%), never touching
     OOS during optimization.
  2. Optimizes in three small waves (signal shape -> entry quality ->
     risk/exit), locking in each wave's winner before moving to the next,
     keeping the search space per wave small relative to your data size.
  3. After each wave, re-runs the locked-in params on OOS (no
     optimization, just bt.run()) and reports the degradation. If a
     wave's "winner" falls apart out-of-sample, the script automatically
     falls back to the default value for that wave's parameters.
  4. At the end, runs the final combined parameter set across the FULL
     period (IS+OOS) once, as the realistic report -- not the inflated
     IS-only number.

HOW TO RUN
----------
1. Place this file anywhere convenient, e.g. next to your App file.
2. Make sure this script can import your strategy module. Easiest path:
   put this file in the SAME folder as
   App_MACD_AI_HybridScore_Latest1.py (i.e.
   .../9.00Trading_ML_Prediction_Widget_Demo_Live_backtrade/), so that
   "from strategies.MomentumStrategy_MACD_HybridScore_Latest import ..."
   resolves the same way it does for the App.
3. Edit CSV_PATH below to point at your SOL 15m CSV/Excel file.
4. Edit the column-name mapping in load_data() if your file's headers
   differ from Open/High/Low/Close/Volume + a timestamp column.
5. Run:  python optimize_momentum_sol_15m.py
6. Read the printed report. It will tell you, per wave, the IS winner,
   its OOS performance, and whether it was kept or rejected. At the end
   it prints the final recommended parameter dict and the full-period
   (realistic) backtest stats for that combination.

This script does NOT modify your live strategy file, your GUI app, or
strategy_settings.json. It only imports BacktestMomentumStrategy and
MOMENTUM_PARAMS and runs backtests in-memory.
"""

import sys
import os
import json
import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# CONFIG — EDIT THESE FOUR THINGS FOR YOUR SETUP
# ──────────────────────────────────────────────────────────────────────────

# 1. Path to your local SOL 15m OHLCV file (CSV or XLSX).
CSV_PATH = r"C:\Users\dahab\PyCharm_2026.2.23\New_Bollinger_bands\9.00Trading_ML_Prediction_Widget_Demo_Live_backtrade\data_cache\2026.csv"

# 2. Path to the folder containing App_MACD_AI_HybridScore_Latest1.py,
#    so "strategies.MomentumStrategy_MACD_HybridScore_Latest" can be
#    imported exactly like the App imports it.
PROJECT_ROOT = r"C:\Users\dahab\PyCharm_2026.2.23\New_Bollinger_bands\9.00Trading_ML_Prediction_Widget_Demo_Live_backtrade"

# 3. Initial capital for the backtest (matches GlobalConfig.INITIAL_CAPITAL
#    unless you want to override it here).
INITIAL_CASH = 50_000.0

# 4. In-sample fraction (rest becomes out-of-sample). 0.70 = 70/30 split.
IS_FRACTION = 0.70

# Commission rate (matches GlobalConfig.COMMISSION_RATE in the strategy file)
COMMISSION = 0.001

sys.path.insert(0, PROJECT_ROOT)

from backtesting import Backtest  # noqa: E402

from strategies.MomentumStrategy_MACD_HybridScore_Latest import (  # noqa: E402
    BacktestMomentumStrategy,
    MOMENTUM_PARAMS,
)


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Loads OHLCV data and normalizes it to the column names backtesting.py
    expects: Open, High, Low, Close, Volume, with a DatetimeIndex sorted
    ascending. Adjust the column-rename map below if your file's headers
    differ.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # ── Normalize column names (edit this map if your headers differ) ──
    col_map = {}
    for col in df.columns:
        lower = str(col).strip().lower()
        if lower in ("timestamp", "time", "date", "datetime", "open_time"):
            col_map[col] = "timestamp"
        elif lower == "open":
            col_map[col] = "Open"
        elif lower == "high":
            col_map[col] = "High"
        elif lower == "low":
            col_map[col] = "Low"
        elif lower in ("close", "close_price"):
            col_map[col] = "Close"
        elif lower in ("volume", "vol"):
            col_map[col] = "Volume"
    df = df.rename(columns=col_map)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns after rename: {missing}. "
            f"Found columns: {list(df.columns)}. Edit load_data()'s col_map."
        )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "No timestamp column found and index is not already a "
            "DatetimeIndex. Add a 'timestamp' column or adjust load_data()."
        )

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df


def split_in_out_sample(df: pd.DataFrame, is_fraction: float):
    split_idx = int(len(df) * is_fraction)
    is_df = df.iloc[:split_idx].copy()
    oos_df = df.iloc[split_idx:].copy()
    return is_df, oos_df


# ──────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION — equal-weighted Sharpe + Returns + Win Rate + Equity
# Mirrors combined_objective() in App_MACD_AI_HybridScore_Latest1.py,
# adapted for the 15m timeframe per get_objective_config()'s 15m branch.
# ──────────────────────────────────────────────────────────────────────────

# 15m-specific trade-count guardrails (matches the 1.5x/1.5x/2.0x scaling
# the App applies for 1m/5m/15m/30m intervals against a "default" base of
# min_trades_absolute=10, min_trades_penalty=20, max_trades_penalty=80).
MIN_TRADES_ABSOLUTE = 15   # below this, the run is discarded outright
MIN_TRADES_PENALTY = 30    # below this, soft penalty applied
MAX_TRADES_PENALTY = 160   # above this, soft penalty applied (overtrading)
PENALTY_LOW = 0.6
PENALTY_HIGH = 0.8


def combined_objective(stats: pd.Series) -> float:
    num_trades = stats.get("# Trades", 0)

    if num_trades < MIN_TRADES_ABSOLUTE:
        return -999999.0

    trade_penalty = 1.0
    if num_trades < MIN_TRADES_PENALTY:
        trade_penalty = PENALTY_LOW
    elif num_trades > MAX_TRADES_PENALTY:
        trade_penalty = PENALTY_HIGH

    active_metrics = ["sharpe", "returns", "winrate", "equity"]
    weight_per_metric = 1.0 / len(active_metrics)
    total_score = 0.0

    sharpe = stats.get("Sharpe Ratio", 0) or 0
    if sharpe > 3:
        sharpe = 3 + (sharpe - 3) * 0.5
    total_score += max(0, sharpe) * weight_per_metric * 10

    returns = stats.get("Return [%]", 0) or 0
    returns_score = min(returns / 10, 20)
    total_score += max(0, returns_score) * weight_per_metric

    winrate = stats.get("Win Rate [%]", 0) or 0
    winrate_score = max(0, (winrate - 40) / 5)
    winrate_score = min(winrate_score, 8)
    total_score += winrate_score * weight_per_metric

    equity = stats.get("Equity Final [$]", INITIAL_CASH) or INITIAL_CASH
    equity_score = ((equity - INITIAL_CASH) / INITIAL_CASH) * 10
    equity_score = min(max(equity_score, 0), 20)
    total_score += equity_score * weight_per_metric

    return total_score * trade_penalty


# ──────────────────────────────────────────────────────────────────────────
# CONSTRAINT FUNCTION — mirrors combined_constraint() in the App
# ──────────────────────────────────────────────────────────────────────────

def make_constraint(active_param_names):
    active = set(active_param_names)

    def constraint(p):
        def g(key, default):
            return p[key] if key in p else default

        if {"ema_fast_period", "ema_mid_period", "ema_slow_period"} <= active:
            fast = g("ema_fast_period", 9)
            mid = g("ema_mid_period", 21)
            slow = g("ema_slow_period", 50)
            if not (fast < mid < slow):
                return False

        weight_keys = {"weight_ema", "weight_adx", "weight_macd",
                        "weight_rsi", "weight_volume"}
        if weight_keys <= active:
            total = sum(g(k, {
                "weight_ema": 20, "weight_adx": 20, "weight_macd": 25,
                "weight_rsi": 20, "weight_volume": 15,
            }[k]) for k in weight_keys)
            if not (85 <= total <= 115):
                return False

        if {"risk_tier1", "risk_tier2"} <= active:
            t1 = g("risk_tier1", 0.015)
            t2 = g("risk_tier2", 0.018)
            if not (t1 <= t2):
                return False

        if {"quality_tier1_min", "quality_tier2_min"} <= active:
            q1 = g("quality_tier1_min", 68)
            q2 = g("quality_tier2_min", 65)
            if not (q1 >= q2):
                return False

        return True

    return constraint


# ──────────────────────────────────────────────────────────────────────────
# WAVE DEFINITIONS
# Each wave lists the parameters to optimize and the candidate values to
# try. Values are drawn from ALL_MOMENTUM_BACKTEST_PARAMS /
# BACKTEST_PARAMS in your source files. Edit ranges here if you want to
# widen/narrow the search.
# ──────────────────────────────────────────────────────────────────────────

WAVE_1_SIGNAL_SHAPE = {
    "ema_fast_period": [8, 9, 10, 12],
    "ema_mid_period": [18, 20, 21, 24],
    "ema_slow_period": [45, 50, 55, 60],
    "weight_ema": [15, 18, 20, 22],
    "weight_adx": [15, 18, 20, 22],
    "weight_macd": [20, 22, 25, 28],
    "weight_rsi": [15, 18, 20, 22],
    "weight_volume": [10, 12, 15, 18],
}

WAVE_2_ENTRY_QUALITY = {
    "quality_tier1_min": [62, 65, 68, 72],
    "quality_tier2_min": [55, 58, 62, 65],
    "tier1_adx_hard_min": [20, 22, 25, 28],
    "tier1_rsi_min": [38, 40, 42, 44],
    "tier1_rsi_max": [62, 65, 68, 72],
    "tier1_volume_min": [0.7, 0.8, 1.0, 1.2],
    "tier1_momentum_min": [0.01, 0.015, 0.02, 0.03],
}

WAVE_3_RISK_EXIT = {
    "risk_tier1": [0.015, 0.020, 0.025, 0.030],
    "risk_tier2": [0.020, 0.025, 0.030, 0.035],
    "stop_loss_atr_mult": [2.5, 3.0, 3.5, 4.0],
    "trailing_activation_r": [1.5, 2.0, 2.5, 3.0],
    "take_profit_r1": [2.0, 2.5, 3.0, 3.5],
    "take_profit_r2": [3.0, 4.0, 5.0, 6.0],
    "take_profit_r3": [6.0, 7.0, 8.0, 10.0],
}

WAVES = [
    ("Wave 1 - Signal Shape (EMA periods + score weights)", WAVE_1_SIGNAL_SHAPE),
    ("Wave 2 - Entry Quality (thresholds/filters)", WAVE_2_ENTRY_QUALITY),
    ("Wave 3 - Risk & Exit (sizing, stops, targets)", WAVE_3_RISK_EXIT),
]

# Max acceptable degradation from IS to OOS before we reject a wave's
# winner and fall back to defaults for that wave (per the plan: ~30-40%).
MAX_OOS_DEGRADATION = 0.40


# ──────────────────────────────────────────────────────────────────────────
# CORE RUN HELPERS
# ──────────────────────────────────────────────────────────────────────────

def run_fixed(df: pd.DataFrame, params: dict) -> pd.Series:
    """Run a single backtest with fixed (non-optimized) params."""
    bt = Backtest(df, BacktestMomentumStrategy, cash=INITIAL_CASH,
                  commission=COMMISSION, exclusive_orders=True)
    stats = bt.run(**params)
    return stats


def run_optimize(df: pd.DataFrame, base_params: dict, wave_params: dict):
    """
    Run bt.optimize() over wave_params (grid), holding base_params fixed
    for everything not in this wave. Returns (best_stats, best_param_dict).
    """
    bt = Backtest(df, BacktestMomentumStrategy, cash=INITIAL_CASH,
                  commission=COMMISSION, exclusive_orders=True)

    fixed = {k: v for k, v in base_params.items() if k not in wave_params}
    opt_kwargs = dict(wave_params)
    opt_kwargs.update({k: v for k, v in fixed.items()})
    # backtesting.py's optimize() takes scalars as fixed and lists as
    # ranges to sweep -- fixed params above are scalars, wave_params are
    # lists, so this single call sweeps only the wave's parameters.

    constraint_fn = make_constraint(wave_params.keys())

    stats = bt.optimize(
        **opt_kwargs,
        maximize=combined_objective,
        constraint=constraint_fn,
        method="grid",
        max_tries=None,
    )

    best_params = {
        k: getattr(stats._strategy, k) for k in wave_params.keys()
    }
    return stats, best_params


def pct_degradation(is_value: float, oos_value: float) -> float:
    """Fractional drop from IS to OOS. Positive = got worse."""
    if is_value == 0:
        return 0.0 if oos_value >= 0 else 1.0
    return (is_value - oos_value) / abs(is_value)


def summarize(stats: pd.Series, label: str) -> str:
    return (
        f"{label}: Return={stats.get('Return [%]', 0):.2f}% | "
        f"WinRate={stats.get('Win Rate [%]', 0):.2f}% | "
        f"Sharpe={stats.get('Sharpe Ratio', 0):.2f} | "
        f"Sortino={stats.get('Sortino Ratio', 0):.2f} | "
        f"EquityFinal=${stats.get('Equity Final [$]', 0):,.2f} | "
        f"MaxDD={stats.get('Max. Drawdown [%]', 0):.2f}% | "
        f"Trades={stats.get('# Trades', 0)} | "
        f"ProfitFactor={stats.get('Profit Factor', float('nan')):.2f}"
    )


# ──────────────────────────────────────────────────────────────────────────
# MAIN STAGED OPTIMIZATION LOOP
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("STAGED IS/OOS OPTIMIZATION — Momentum MACD HybridScore — SOL/USDT 15m")
    print("=" * 90)

    print(f"\nLoading data from: {CSV_PATH}")
    df = load_data(CSV_PATH)
    print(f"Loaded {len(df)} bars  |  {df.index[0]}  ->  {df.index[-1]}")

    is_df, oos_df = split_in_out_sample(df, IS_FRACTION)
    print(f"\nIn-sample:     {len(is_df)} bars  ({is_df.index[0]} -> {is_df.index[-1]})")
    print(f"Out-of-sample: {len(oos_df)} bars  ({oos_df.index[0]} -> {oos_df.index[-1]})")

    # Running parameter set, starts at strategy defaults, gets updated
    # wave-by-wave with whatever wins validation.
    current_params = MOMENTUM_PARAMS.copy()

    report_rows = []

    for wave_name, wave_grid in WAVES:
        print("\n" + "-" * 90)
        print(wave_name)
        print("-" * 90)

        n_combos = 1
        for v in wave_grid.values():
            n_combos *= len(v)
        print(f"Grid size for this wave: {n_combos} combinations "
              f"({len(wave_grid)} params)")

        print("Optimizing on IN-SAMPLE data...")
        is_stats, wave_winner = run_optimize(is_df, current_params, wave_grid)
        print("IS winner params:", json.dumps(wave_winner, indent=2, default=str))
        print(summarize(is_stats, "IS (optimized)"))

        # Validate the winner on OOS, fixed (no further optimization).
        candidate_params = current_params.copy()
        candidate_params.update(wave_winner)

        print("\nValidating winner on OUT-OF-SAMPLE data (no optimization)...")
        oos_stats = run_fixed(oos_df, candidate_params)
        print(summarize(oos_stats, "OOS (validation)"))

        is_obj = combined_objective(is_stats)
        oos_obj = combined_objective(oos_stats)
        degradation = pct_degradation(is_obj, oos_obj)

        print(f"\nObjective score: IS={is_obj:.3f}  OOS={oos_obj:.3f}  "
              f"Degradation={degradation * 100:.1f}%")

        if oos_stats.get("# Trades", 0) < MIN_TRADES_ABSOLUTE:
            decision = "REJECTED (too few OOS trades — unreliable sample)"
            keep = False
        elif degradation > MAX_OOS_DEGRADATION:
            decision = (f"REJECTED (OOS degraded {degradation*100:.1f}% > "
                        f"{MAX_OOS_DEGRADATION*100:.0f}% threshold — likely overfit)")
            keep = False
        else:
            decision = "KEPT (held up reasonably well out-of-sample)"
            keep = True

        print(f"\nDecision: {decision}")

        if keep:
            current_params.update(wave_winner)
        # else: current_params keeps the prior (default/earlier-wave) values
        # for this wave's parameters — no update applied.

        report_rows.append({
            "wave": wave_name,
            "winner": wave_winner,
            "is_score": is_obj,
            "oos_score": oos_obj,
            "degradation_pct": degradation * 100,
            "kept": keep,
        })

    # ── Final full-period report on the locked-in combined parameter set ──
    print("\n" + "=" * 90)
    print("FINAL COMBINED PARAMETER SET (after staged validation)")
    print("=" * 90)
    print(json.dumps(current_params, indent=2, default=str))

    print("\nRunning final backtest over the FULL period (IS+OOS combined) "
          "for a realistic read — this is NOT another optimization pass.")
    full_stats = run_fixed(df, current_params)
    print("\n" + summarize(full_stats, "FULL PERIOD (realistic estimate)"))

    print("\n" + "-" * 90)
    print("Per-wave summary:")
    for row in report_rows:
        status = "KEPT" if row["kept"] else "REJECTED"
        print(f"  [{status:8}] {row['wave']:55} "
              f"IS={row['is_score']:.2f}  OOS={row['oos_score']:.2f}  "
              f"Degradation={row['degradation_pct']:.1f}%")

    out_path = os.path.join(os.getcwd(), "optimized_momentum_params.json")
    with open(out_path, "w") as f:
        json.dump(current_params, f, indent=2, default=str)
    print(f"\nFinal parameter set saved to: {out_path}")
    print("\nIMPORTANT: the FULL PERIOD numbers above (not the IS-only wave\n"
          "numbers) are the honest estimate of what this parameter set\n"
          "would have produced. Treat them as a ceiling, not a guarantee —\n"
          "live/forward performance on data after your file's end date is\n"
          "still unknown and should be paper-traded before risking capital.")


if __name__ == "__main__":
    main()