# ═══════════════════════════════════════════════════════════════════════════
# ATR Calculator Helper - v1.0.0
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class ATRCalculator:
    """ATR-based stop and target calculator for trading strategies."""

    def __init__(self, period: int = 14):
        self.period = period

    def calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR from DataFrame."""
        if df is None or len(df) < self.period + 1:
            return 0.0

        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values

            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - np.roll(close, 1)),
                    abs(low - np.roll(close, 1))
                )
            )
            # Remove first element (NaN from shift)
            tr = tr[1:]
            atr = np.mean(tr[-self.period:])
            return float(atr) if not np.isnan(atr) else 0.0
        except Exception:
            return 0.0

    def calculate_stops(self, entry_price: float, atr: float,
                        stop_mult: float = 2.0,
                        trail_mult: float = 1.5,
                        target_mult_1: float = 1.5,
                        target_mult_2: float = 3.0) -> Dict:
        """
        Calculate all stop and target levels.

        Parameters:
        -----------
        entry_price : float
            Entry price of the trade
        atr : float
            Current ATR value
        stop_mult : float
            Multiplier for stop loss (default: 2.0)
        trail_mult : float
            Multiplier for trailing stop start (default: 1.5)
        target_mult_1 : float
            First profit target multiplier (default: 1.5)
        target_mult_2 : float
            Second profit target multiplier (default: 3.0)

        Returns:
        --------
        Dictionary with all calculated levels
        """
        if atr <= 0:
            return {
                'stop_loss': entry_price * 0.98,
                'trailing_start': entry_price * 0.985,
                'target_1': entry_price * 1.015,
                'target_2': entry_price * 1.03,
                'atr_value': 0.0,
                'stop_distance': 0.0,
                'risk_pct': 2.0,
                'entry_price': entry_price
            }

        stop_distance = atr * stop_mult
        trail_distance = atr * trail_mult

        return {
            'stop_loss': round(entry_price - stop_distance, 4),
            'trailing_start': round(entry_price - trail_distance, 4),
            'target_1': round(entry_price + (atr * target_mult_1), 4),
            'target_2': round(entry_price + (atr * target_mult_2), 4),
            'atr_value': round(atr, 4),
            'stop_distance': round(stop_distance, 4),
            'risk_pct': round((stop_distance / entry_price) * 100, 2) if entry_price > 0 else 0,
            'entry_price': entry_price
        }

    def get_risk_reward(self, entry_price: float, stop_loss: float,
                        target: float, direction: str = 'long') -> float:
        """Calculate risk-reward ratio for a trade."""
        if direction == 'long':
            risk = entry_price - stop_loss
            reward = target - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - target

        if risk <= 0:
            return 0.0
        return reward / risk

    def get_position_size(self, equity: float, risk_pct: float,
                          entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk."""
        risk_amount = equity * (risk_pct / 100)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            return 0.0

        return risk_amount / risk_per_unit


# Quick test function
def test_atr_calculator():
    """Test ATR calculator with sample data."""
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='15min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        'High': prices * 1.002,
        'Low': prices * 0.998,
        'Close': prices,
        'Open': np.roll(prices, 1)
    }, index=dates)

    calc = ATRCalculator(period=14)
    atr = calc.calculate_atr(df)
    stops = calc.calculate_stops(entry_price=100, atr=atr)

    print("=" * 60)
    print("ATR CALCULATOR TEST")
    print("=" * 60)
    print(f"ATR: {atr:.4f}")
    print(f"Stop Loss: ${stops['stop_loss']:.2f}")
    print(f"Trailing Start: ${stops['trailing_start']:.2f}")
    print(f"Target 1: ${stops['target_1']:.2f}")
    print(f"Target 2: ${stops['target_2']:.2f}")
    print(f"Risk: {stops['risk_pct']:.2f}%")
    print("=" * 60)

    return stops


if __name__ == "__main__":
    test_atr_calculator()