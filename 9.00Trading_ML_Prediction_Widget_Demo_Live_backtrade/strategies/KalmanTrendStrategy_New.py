"""
═══════════════════════════════════════════════════════════════════════════
KALMAN TREND STRATEGY - FULLY FIXED WITH LONG & SHORT TRADING
═══════════════════════════════════════════════════════════════════════════
FIXES APPLIED:
1. SHORT TRADING ENABLED (trading_direction = "both")
2. LOWERED THRESHOLDS FOR BETTER SIGNAL DETECTION
3. SIMPLIFIED ENTRY CONDITIONS (2 mandatory conditions)
4. IMPROVED KALMAN COLOR THRESHOLDS (30/-30 instead of 70/-70)
5. ADDED DEBUG OUTPUT FOR SHORT SIGNALS
"""

from datetime import datetime, timezone
import pandas_ta as ta
from backtesting import Strategy
from .base3_New import BaseStrategy
import numpy as np
import logging
import json
import os

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED STRATEGY PARAMETERS - SHORT TRADING ENABLED
# ═══════════════════════════════════════════════════════════════════════════
KALMAN_PARAMS = {
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRADING DIRECTION - SET TO "both" FOR SHORT TRADING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "trading_direction": "both",  # Options: "long_only", "short_only", "both"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # KALMAN FILTER PARAMETERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "process_noise_1": 0.001,
    "process_noise_2": 0.001,
    "measurement_noise": 100.0,
    "trend_lookback": 20,
    "strength_smooth": 5,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY CONFIGURATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "risk_reward": 1.5,
    "lookback": 20,
    "window": 10,
    "strength_smooth_param": 5,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MOVING AVERAGES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "ma_fast_period": 20,
    "ma_slow_period": 50,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TECHNICAL INDICATORS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "rsi_period": 14,
    "atr_period": 14,
    "adx_period": 25,
    "bb_period": 20,
    "bb_std": 2,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LONG ENTRY CONDITIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "long_kalman_strength_min": 30,
    "long_rsi_min": 30,
    "long_rsi_max": 70,
    "long_pullback_percent": 0.1,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SHORT ENTRY CONDITIONS - EASIER TO TRIGGER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "short_kalman_strength_min": -30,
    "short_rsi_max": 70,
    "short_rsi_min": 30,
    "short_rally_percent": 0.1,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMMON ENTRY CONDITIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "volume_min_ratio": 1.0,
    "volume_period": 50,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RISK MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "stop_loss_pct": 0.02,
    "trailing_stop_pct": 0.015,
    "atr_multiplier": 2.0,
    "risk_per_trade": 0.01,
    "max_position_pct": 0.15,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EXIT CONDITIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "long_rsi_exit_threshold": 80,
    "short_rsi_exit_threshold": 20,
    "max_hold_bars": 48,
    "min_hold_bars": 2,
    "max_hold_seconds": 3600,

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MARKET STATE FILTERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "min_adx": 15,
    "min_volatility": 0.001,
    "max_spread_pct": 0.001,
    "cooldown_bars": 10,
}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════
class KalmanConfig:
    """Configuration management for Kalman strategy"""

    CONFIG_FILE = "strategy_configs/kalman_settings.json"

    @classmethod
    def get_config(cls):
        """Load configuration"""
        try:
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, 'r') as f:
                    saved = json.load(f)
                    config = KALMAN_PARAMS.copy()
                    config.update(saved)
                    return config
        except Exception as e:
            logging.error(f"Config load error: {e}")

        return KALMAN_PARAMS.copy()

    @classmethod
    def save_config(cls, config):
        """Save configuration"""
        try:
            os.makedirs("strategy_configs", exist_ok=True)
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Config save error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
class KalmanIndicatorCalculator:
    """Calculate all Kalman strategy indicators"""

    @staticmethod
    def calculate(df, params):
        """Calculate all indicators including Kalman filter"""
        df = df.copy()

        try:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # BASIC INDICATORS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # Moving averages
            df['MA_Fast'] = df['Close'].rolling(params['ma_fast_period']).mean()
            df['MA_Slow'] = df['Close'].rolling(params['ma_slow_period']).mean()

            # RSI
            df['RSI'] = ta.rsi(df['Close'], length=params['rsi_period'])

            # ATR
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=params['atr_period'])

            # Bollinger Bands
            bbands = ta.bbands(df['Close'], length=params['bb_period'], std=params['bb_std'])
            df['UpperBand'] = bbands['BBU_20_2.0']
            df['MiddleBand'] = bbands['BBM_20_2.0']
            df['LowerBand'] = bbands['BBL_20_2.0']

            # ADX for trend strength
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=params['adx_period'])
            df['ADX'] = adx_df['ADX_25']

            # Volume analysis
            df['Volume_MA'] = df['Volume'].rolling(params['volume_period']).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

            # Volatility
            df['Volatility'] = df['Close'].pct_change().rolling(20).std()

            # Pullback for LONG entries (retracement from high)
            df['Recent_High'] = df['High'].rolling(10).max()
            df['Pullback_Pct'] = (df['Recent_High'] - df['Close']) / df['Recent_High'] * 100

            # Rally for SHORT entries (retracement from low)
            df['Recent_Low'] = df['Low'].rolling(10).min()
            df['Rally_Pct'] = (df['Close'] - df['Recent_Low']) / df['Recent_Low'] * 100

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # KALMAN FILTER STRENGTH
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df = KalmanIndicatorCalculator._kalman_trend_strength(df, params)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STOP LOSS CALCULATIONS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            atr_stop = df['ATR'] * params['atr_multiplier']
            df['stop_loss_atr'] = df['Close'] - atr_stop
            df['stop_loss_pct'] = df['Close'] * (1 - params['stop_loss_pct'])
            df['stop_loss'] = df[['stop_loss_atr', 'stop_loss_pct']].min(axis=1)

            # Trailing stop (for longs)
            rolling_high = df['High'].rolling(window=20).max()
            df['trailing_stop_loss'] = rolling_high * (1 - params['trailing_stop_pct'])

            # SHORT stop loss calculations
            df['short_stop_loss_atr'] = df['Close'] + atr_stop
            df['short_stop_loss_pct'] = df['Close'] * (1 + params['stop_loss_pct'])
            df['short_stop_loss'] = df[['short_stop_loss_atr', 'short_stop_loss_pct']].max(axis=1)

            # Trailing stop for shorts
            rolling_low = df['Low'].rolling(window=20).min()
            df['short_trailing_stop'] = rolling_low * (1 + params['trailing_stop_pct'])

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CREATE "CLOSED" VERSIONS FOR BACKTESTING (shift by 1)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            indicators_to_shift = [
                'MA_Fast', 'MA_Slow', 'RSI', 'ATR', 'ADX', 'Volatility',
                'Kalman_Osc', 'Kalman_Strength', 'Kalman_Color',
                'UpperBand', 'MiddleBand', 'LowerBand',
                'Pullback_Pct', 'Rally_Pct', 'Recent_High', 'Recent_Low',
                'Volume_Ratio'
            ]

            for col in indicators_to_shift:
                if col in df.columns:
                    df[f'{col}_closed'] = df[col].shift(1)

            return df

        except Exception as e:
            logging.error(f"Kalman indicator calculation error: {e}")
            raise

    @staticmethod
    def _kalman_trend_strength(df, params):
        """
        Calculate Kalman filter trend strength indicator
        Returns oscillator, strength, and color signals
        """
        process_noise_1 = params['process_noise_1']
        process_noise_2 = params['process_noise_2']
        measurement_noise = params['measurement_noise']
        trend_lookback = params['trend_lookback']
        strength_smooth = params['strength_smooth']

        n = len(df)
        close_prices = df['Close'].values
        oscillator = np.zeros(n)
        filtered = np.zeros(n)
        raw_strength_arr = np.zeros(n)
        strength = np.zeros(n)

        # Initialize Kalman filter
        X = np.array([close_prices[0], 0.0])
        P = np.eye(2)

        Q = np.array([
            [process_noise_1, process_noise_1 * process_noise_2],
            [process_noise_2 * process_noise_1, process_noise_2]
        ])

        R = measurement_noise
        H = np.array([1, 0])
        F = np.array([[1, 1], [0, 1]])
        I = np.eye(2)

        # Run Kalman filter
        for i in range(n):
            # Predict
            X = F @ X
            P = F @ P @ F.T + Q

            # Update
            y = close_prices[i] - H @ X
            S = H @ P @ H.T + R
            K = (P @ H.T) / S
            X = X + K * y
            P = (I - np.outer(K, H)) @ P

            filtered[i] = X[0]
            oscillator[i] = X[1]

        # Calculate raw strength
        for i in range(trend_lookback, n):
            window_osc = oscillator[i - trend_lookback:i]
            max_abs = np.max(np.abs(window_osc))
            if max_abs > 1e-8:
                raw_strength_arr[i] = (oscillator[i] / max_abs) * 100

        # Apply weighted smoothing
        weights = np.arange(1, strength_smooth + 1)
        weight_sum = np.sum(weights)

        for i in range(trend_lookback + strength_smooth - 1, n):
            window = raw_strength_arr[i - strength_smooth + 1:i + 1]
            strength[i] = np.sum(weights * window) / weight_sum

        # Assign colors based on strength - LOWERED THRESHOLDS FOR BETTER SIGNALS
        colors = np.full(n, 'gray', dtype=object)
        for i in range(n):
            if strength[i] > 30:  # Changed from 70 to 30
                colors[i] = 'green'
            elif strength[i] < -30:  # Changed from -70 to -30
                colors[i] = 'red'
            elif strength[i] > 0:
                colors[i] = 'blue'
            else:
                colors[i] = 'orange'

        df['Kalman_Osc'] = oscillator
        df['Kalman_Strength'] = strength
        df['Kalman_Color'] = colors

        return df


# ═══════════════════════════════════════════════════════════════════════════
# CORE STRATEGY LOGIC - WITH WORKING SHORT TRADES
# ═══════════════════════════════════════════════════════════════════════════
class KalmanLogic:
    """Core Kalman trend strategy logic - LONG & SHORT SUPPORT"""

    def __init__(self, config=None, trading_app=None):
        self.config = config or KalmanConfig.get_config()
        self.trading_app = trading_app

        # Set all parameters as attributes
        for key, value in self.config.items():
            setattr(self, key, value)

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.long_trades = 0
        self.short_trades = 0
        self.winning_longs = 0
        self.winning_shorts = 0
        self.equity_curve = [50000]
        self.trade_history = []
        self.consecutive_losses = 0

        # Position tracking
        self.position_state = {
            'type': None,
            'price': None,
            'quantity': None,
            'time': None,
            'stop_loss': None,
            'trailing_stop': None,
            'entry_confidence': None
        }

        # Cooldown tracking
        self.last_trade_bar = -1000
        self.cooldown_bars = self.config.get('cooldown_bars', 10)

        # Entry tracking
        self._entry_i = None
        self._entry_price = None
        self._position_type = None
        self._trailing_stop = None

    def check_market_conditions(self, data):
        """Check overall market conditions before considering entry"""
        conditions = [
            data.get('ADX_closed', 0) > self.min_adx,
            data.get('Volatility_closed', 0) > self.min_volatility,
        ]
        return all(conditions)

    def check_long_entry_conditions(self, data, current_bar):
        """Check LONG entry conditions - Pullback on uptrend"""
        if current_bar - self.last_trade_bar < self.cooldown_bars:
            return 0, 0

        if not self.check_market_conditions(data):
            return 0, 0

        kalman_strength = data.get('Kalman_Strength_closed', 0)
        kalman_color = data.get('Kalman_Color_closed', 'gray')

        # Simplified LONG conditions
        mandatory = [
            kalman_color == 'green',
            kalman_strength > self.long_kalman_strength_min,
        ]

        supporting = [
            data.get('MA_Fast_closed', 0) > data.get('MA_Slow_closed', 0),
            data.get('Close', 0) > data.get('MA_Fast_closed', 0),
            self.long_rsi_min < data.get('RSI_closed', 50) < self.long_rsi_max,
            data.get('Volume_Ratio', 1) > self.volume_min_ratio,
            data.get('Pullback_Pct_closed', 0) > self.long_pullback_percent,
        ]

        return sum(mandatory), sum(supporting)

    def check_short_entry_conditions(self, data, current_bar):
        """Check SHORT entry conditions - Rally on downtrend"""
        if current_bar - self.last_trade_bar < self.cooldown_bars:
            return 0, 0

        if not self.check_market_conditions(data):
            return 0, 0

        kalman_strength = data.get('Kalman_Strength_closed', 0)
        kalman_color = data.get('Kalman_Color_closed', 'gray')

        # Simplified SHORT conditions - EASIER TO TRIGGER
        mandatory = [
            kalman_color == 'red',
            kalman_strength < self.short_kalman_strength_min,
        ]

        supporting = [
            data.get('MA_Fast_closed', 0) < data.get('MA_Slow_closed', 0),
            data.get('Close', 0) < data.get('MA_Fast_closed', 0),
            self.short_rsi_min < data.get('RSI_closed', 50) < self.short_rsi_max,
            data.get('Volume_Ratio', 1) > self.volume_min_ratio,
            data.get('Rally_Pct_closed', 0) > self.short_rally_percent,
        ]

        mandatory_count = sum(mandatory)
        supporting_count = sum(supporting)

        # Debug output for short signals (every 100 bars)
        if mandatory_count >= 1 and current_bar % 50 == 0:
            print(f"  🔍 SHORT CHECK: mandatory={mandatory_count}, supporting={supporting_count}, kalman_color={kalman_color}, kalman_strength={kalman_strength:.1f}")

        return mandatory_count, supporting_count

    def check_entry_conditions(self, data, current_bar):
        """
        Check entry conditions based on trading direction setting
        Returns: (direction, mandatory_count, supporting_count)
        """
        long_mandatory = 0
        long_supporting = 0
        short_mandatory = 0
        short_supporting = 0

        # Check LONG if enabled
        if self.trading_direction in ['long_only', 'both']:
            long_mandatory, long_supporting = self.check_long_entry_conditions(data, current_bar)

        # Check SHORT if enabled
        if self.trading_direction in ['short_only', 'both']:
            short_mandatory, short_supporting = self.check_short_entry_conditions(data, current_bar)

        # Allow entry with just 2 mandatory conditions
        if self.trading_direction in ['long_only', 'both'] and long_mandatory >= 2:
            return 'long', long_mandatory, long_supporting

        if self.trading_direction in ['short_only', 'both'] and short_mandatory >= 2:
            return 'short', short_mandatory, short_supporting

        return None, 0, 0

    def check_exit_conditions(self, data, entry_price, current_price, position_type, entry_time=None, current_bar=None):
        """
        Check exit conditions for both LONG and SHORT positions
        """
        if entry_price is None or entry_price == 0:
            return None

        # Minimum hold time check
        if current_bar is not None and self._entry_i is not None:
            bars_held = current_bar - self._entry_i
            if bars_held < self.min_hold_bars:
                return None

        if position_type == 'long':
            # Profit target
            if current_price >= entry_price * (1 + self.risk_reward * 0.01):
                return "profit_target"
            # Stop loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                return "stop_loss"
            # Trailing stop
            trailing_stop = getattr(self, '_trailing_stop', entry_price * (1 - self.trailing_stop_pct))
            if current_price <= trailing_stop:
                return "trailing_stop"
            # Trend reversal
            if data.get('Kalman_Color_closed') == 'red' and data.get('Kalman_Strength_closed', 0) < -20:
                return "trend_reversal"
            # Overbought
            if data.get('RSI_closed', 50) >= self.long_rsi_exit_threshold:
                return "overbought"

        else:  # short
            # Profit target
            if current_price <= entry_price * (1 - self.risk_reward * 0.01):
                return "profit_target"
            # Stop loss
            if current_price >= entry_price * (1 + self.stop_loss_pct):
                return "stop_loss"
            # Trailing stop
            trailing_stop = getattr(self, '_trailing_stop', entry_price * (1 + self.trailing_stop_pct))
            if current_price >= trailing_stop:
                return "trailing_stop"
            # Trend reversal
            if data.get('Kalman_Color_closed') == 'green' and data.get('Kalman_Strength_closed', 0) > 20:
                return "trend_reversal"
            # Oversold
            if data.get('RSI_closed', 50) <= self.short_rsi_exit_threshold:
                return "oversold"

        # Time exit
        if current_bar is not None and self._entry_i is not None:
            bars_held = current_bar - self._entry_i
            if bars_held >= self.max_hold_bars:
                return "time_exit"

        return None

    def calculate_position_size(self, equity, atr, price, position_type='long'):
        """Calculate position size based on risk"""
        if atr <= 0:
            return 1

        # Reduce position size after consecutive losses
        risk_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.1))
        risk_amount = equity * self.risk_per_trade * risk_multiplier

        if position_type == 'long':
            stop_distance = max(atr * self.atr_multiplier, price * self.stop_loss_pct)
        else:
            stop_distance = max(atr * self.atr_multiplier, price * self.stop_loss_pct)

        if stop_distance > 0:
            risk_based_shares = int(risk_amount / stop_distance)
        else:
            risk_based_shares = 1

        max_position_value = equity * self.max_position_pct
        max_shares = int(max_position_value / price)

        shares = min(risk_based_shares, max_shares)
        return max(shares, 1)

    def record_trade(self, profit, exit_reason="unknown", position_type="long"):
        """Record trade results"""
        self.total_trades += 1
        self.total_profit += profit

        if profit > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1

        if position_type == 'long':
            self.long_trades += 1
            if profit > 0:
                self.winning_longs += 1
        else:
            self.short_trades += 1
            if profit > 0:
                self.winning_shorts += 1

        self.trade_history.append({
            'profit': profit,
            'exit_reason': exit_reason,
            'position_type': position_type,
            'timestamp': datetime.now(timezone.utc)
        })

        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def get_performance_stats(self):
        """Get current performance statistics"""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'long_trades': 0,
                'short_trades': 0,
                'long_win_rate': 0,
                'short_win_rate': 0,
            }

        win_rate = (self.winning_trades / self.total_trades) * 100
        long_win_rate = (self.winning_longs / self.long_trades * 100) if self.long_trades > 0 else 0
        short_win_rate = (self.winning_shorts / self.short_trades * 100) if self.short_trades > 0 else 0

        wins = [t['profit'] for t in self.trade_history if t['profit'] > 0]
        losses = [t['profit'] for t in self.trade_history if t['profit'] < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
        }


# ═══════════════════════════════════════════════════════════════════════════
# LIVE TRADING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════
class KalmanTrendStrategy(BaseStrategy, KalmanLogic):
    """Live trading implementation - FULLY FIXED"""

    def __init__(self, trading_app=None, risk_reward=1.5, lookback=20, window=10, strength_smooth=5):
        BaseStrategy.__init__(self, trading_app)

        # Get parameters based on toggle selection
        if trading_app and hasattr(trading_app, 'get_current_kalman_params'):
            config = trading_app.get_current_kalman_params()
        else:
            config = KALMAN_PARAMS.copy()

        # Update with provided parameters
        config.update({
            'risk_reward': risk_reward,
            'lookback': lookback,
            'window': window,
            'strength_smooth_param': strength_smooth
        })

        # Ensure required config keys exist with defaults
        default_config = {
            'stop_loss_pct': 0.02,  # 2% default stop loss
            'trailing_stop_pct': 0.01,  # 1% trailing stop
            'atr_multiplier': 2.0,  # 2x ATR multiplier
            'cooldown_bars': 5,  # 5 bar cooldown
            'trading_direction': 'both',  # both, long_only, short_only
            'kalman_q': 0.001,
            'kalman_r': 0.1,
            'trend_period': 20,
            'signal_threshold': 2.0,
        }

        # Merge defaults with config (config takes precedence)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value

        # Get trading direction from app if available
        if trading_app and hasattr(trading_app, 'get_trading_direction'):
            config['trading_direction'] = trading_app.get_trading_direction()

        KalmanLogic.__init__(self, config=config, trading_app=trading_app)

        self.name = "Kalman-Fixed"
        self.risk_reward = risk_reward
        self.lookback = lookback
        self.window = window
        self.strength_smooth = strength_smooth

        # For compatibility - now safe to access
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 0.01)
        self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
        self.cooldown_bars = self.config.get('cooldown_bars', 5)
        self.trading_direction = self.config.get('trading_direction', 'both')

        # ML/Prediction integration
        self.ml_enabled = False
        self.current_ml_model = None

        # Confidence scoring
        self.quality_score_enabled = True
        self.quality_minimum_score = 50

        if self.trading_app:
            try:
                mode = 'Custom' if hasattr(trading_app,
                                           'param_toggle_var') and trading_app.param_toggle_var.get() == 'Custom Parameters' else 'Default'
                direction_display = {'long_only': 'LONG ONLY', 'short_only': 'SHORT ONLY', 'both': 'LONG & SHORT'}.get(
                    self.trading_direction, 'LONG & SHORT')
                self._log(f"🚀 KALMAN FIXED Strategy Initialized", "green")
                self._log(f"  - Trading Direction: {direction_display}", "cyan")
                self._log(f"  - Mode: {mode} Parameters", "yellow")
                self._log(f"  - Stop Loss: {self.stop_loss_pct * 100}%", "yellow")
                self._log(f"  - Trailing Stop: {self.trailing_stop_pct * 100}%", "yellow")
                self._log(f"  - Risk/Reward: {self.risk_reward}:1", "yellow")
                self._log(f"  - Cooldown: {self.cooldown_bars} bars", "yellow")
            except Exception as e:
                print(f"Kalman Strategy Initialized (log failed: {e})")
    def get_strategy_info(self):
        """Return strategy information for display in the app"""
        direction_display = {
            'long_only': 'LONG ONLY',
            'short_only': 'SHORT ONLY',
            'both': 'LONG & SHORT (Dual)'
        }.get(self.trading_direction, 'LONG & SHORT')

        return {
            'name': 'Kalman Fixed Strategy',
            'version': 'v4.0 - Fixed Trend Following with Short Trading',
            'tier_system': 'Kalman Strength + Pullback/Rally Entry',
            'expected_trades_monthly': '30-60',
            'target_win_rate': '45-55%',
            'target_cagr': '15-25%',
            'target_sharpe': '1.0-1.5',
            'max_drawdown': '10-15%',
            'trading_direction': direction_display,
            'tier1_description': f'{direction_display} - Strong signals (quality >= 70)',
            'tier2_description': f'{direction_display} - Moderate signals (quality >= 50)',
        }

    def _log(self, message, color="white"):
        """Helper for logging"""
        try:
            if self.trading_app and hasattr(self.trading_app, 'log_message'):
                self.trading_app.log_message(message, color)
            else:
                print(f"[Kalman] {color}: {message}")
        except Exception:
            print(f"[Kalman] {message}")

    def calculate_indicators(self, df):
        """Calculate indicators for live trading"""
        try:
            return KalmanIndicatorCalculator.calculate(df, self.config)
        except Exception as e:
            if self.trading_app:
                self.trading_app.log_message(f"Kalman indicator error: {str(e)}", "red")
            logging.error(f"Kalman indicator error: {str(e)}")
            return None

    def _calculate_quality_score(self, current_data, direction='long'):
        """Calculate quality score for LONG or SHORT entries"""
        kalman_strength = float(current_data.get('Kalman_Strength_closed', 0))
        kalman_color = str(current_data.get('Kalman_Color_closed', 'gray'))
        ma_fast = float(current_data.get('MA_Fast_closed', 0))
        ma_slow = float(current_data.get('MA_Slow_closed', 0))
        rsi = float(current_data.get('RSI_closed', 50))
        volume_ratio = float(current_data.get('Volume_Ratio', 1.0))
        adx = float(current_data.get('ADX_closed', 0))

        score = 50

        if direction == 'long':
            if kalman_color == 'green':
                score += 20
                if kalman_strength > 50:
                    score += 10
            else:
                score -= 20
            if ma_fast > ma_slow:
                score += 15
            if 30 < rsi < 70:
                score += 10
            if volume_ratio > 1.2:
                score += 10
            if adx > 20:
                score += 5

        else:  # short
            if kalman_color == 'red':
                score += 20
                if kalman_strength < -50:
                    score += 10
            else:
                score -= 20
            if ma_fast < ma_slow:
                score += 15
            if 30 < rsi < 70:
                score += 10
            if volume_ratio > 1.2:
                score += 10
            if adx > 20:
                score += 5

        return max(0, min(100, score))

    def _get_position_multiplier(self, quality_score):
        """Determine position size multiplier based on quality score"""
        if quality_score >= 80:
            return 1.0
        elif quality_score >= 70:
            return 0.8
        elif quality_score >= 60:
            return 0.6
        elif quality_score >= 50:
            return 0.4
        else:
            return 0.0

    def run_analysis_cycle(self, current_data, current_price, df=None):
        """Main analysis method - FULLY FIXED"""
        # Check if we have an active position
        if self.position_state and self.position_state.get('type') is not None:
            exit_reason = self.check_exit_conditions(
                current_data,
                self.position_state.get('price', 0),
                current_price,
                self.position_state.get('type'),
                self.position_state.get('time'),
                None
            )

            if exit_reason:
                return (exit_reason, 1.0, self.position_state.get('type'))
            else:
                return (-1, 0, 0, "IN_TRADE", self.position_state.get('type'))

        # No position - check entry
        else:
            direction, mandatory_count, supporting_count = self.check_entry_conditions(current_data, 0)

            if direction is None:
                return ("hold", 0, 0, "No signal", None)

            # Calculate quality score
            quality_score = self._calculate_quality_score(current_data, direction)

            # Only enter if quality score >= 50
            if quality_score >= 50:
                position_mult = self._get_position_multiplier(quality_score)
                if position_mult > 0:
                    atr = current_data.get('ATR', 1)
                    equity = 50000

                    if self.trading_app:
                        equity = self.trading_app.get_balance('USDT')

                    base_shares = self.calculate_position_size(equity, atr, current_price, direction)
                    shares = int(base_shares * position_mult)

                    reason = f"{direction.upper()} entry - Quality: {quality_score}"
                    return (direction, quality_score, shares, reason)
                else:
                    return ("hold", quality_score, 0, f"Weak {direction} signal ({quality_score})", direction)
            else:
                return ("hold", quality_score, 0, f"Poor {direction} signal ({quality_score})", direction)

    def execute_buy(self, shares, price, atr, quality_score, tier=1):
        """Execute a LONG order"""
        try:
            if not self.trading_app:
                return False, 0, None

            success = self.trading_app.place_order(
                'buy',
                price,
                quantity=shares,
                confidence=quality_score
            )

            if success:
                self.position_state = {
                    'type': 'long',
                    'price': price,
                    'quantity': shares,
                    'time': datetime.now(timezone.utc),
                    'stop_loss': price * (1 - self.stop_loss_pct),
                    'trailing_stop': price * (1 - self.trailing_stop_pct),
                    'entry_confidence': quality_score
                }
                self._entry_price = price
                self._position_type = 'long'
                self._trailing_stop = price * (1 - self.trailing_stop_pct)
                self._entry_i = 0
                return True, shares, f"order_{datetime.now().timestamp()}"
            return False, 0, None
        except Exception as e:
            if self.trading_app:
                self.trading_app.log_message(f"❌ Buy execution error: {str(e)}", "red")
            return False, 0, None

    def execute_short(self, shares, price, atr, quality_score, tier=1):
        """Execute a SHORT order"""
        try:
            if not self.trading_app:
                return False, 0, None

            success = self.trading_app.place_order(
                'sell',
                price,
                quantity=shares,
                confidence=quality_score,
                order_type='short'
            )

            if success:
                self.position_state = {
                    'type': 'short',
                    'price': price,
                    'quantity': shares,
                    'time': datetime.now(timezone.utc),
                    'stop_loss': price * (1 + self.stop_loss_pct),
                    'trailing_stop': price * (1 + self.trailing_stop_pct),
                    'entry_confidence': quality_score
                }
                self._entry_price = price
                self._position_type = 'short'
                self._trailing_stop = price * (1 + self.trailing_stop_pct)
                self._entry_i = 0
                return True, shares, f"order_{datetime.now().timestamp()}"
            return False, 0, None
        except Exception as e:
            if self.trading_app:
                self.trading_app.log_message(f"❌ Short execution error: {str(e)}", "red")
            return False, 0, None

    def execute_cover(self, reason, exit_percentage=1.0):
        """Execute a SHORT cover (buy to close)"""
        try:
            if not self.trading_app:
                return False, 0, 0

            if not self.position_state or self.position_state.get('type') != 'short':
                return False, 0, 0

            quantity = self.position_state.get('quantity', 0) * exit_percentage
            current_price = self.trading_app.get_current_price()
            if not current_price:
                return False, 0, 0

            entry_price = self.position_state.get('price', 0)
            pnl = (entry_price - current_price) * quantity

            success = self.trading_app.place_order(
                'buy',
                current_price,
                quantity=quantity,
                exit_reason=reason,
                order_type='cover'
            )

            if success:
                self.record_trade(pnl, reason, 'short')
                self.position_state = {
                    'type': None,
                    'price': None,
                    'quantity': None,
                    'time': None,
                    'stop_loss': None,
                    'trailing_stop': None,
                    'entry_confidence': None
                }
                return True, pnl, current_price
            return False, 0, 0
        except Exception as e:
            if self.trading_app:
                self.trading_app.log_message(f"❌ Cover execution error: {str(e)}", "red")
            return False, 0, 0

    def check_entry_conditions(self, current_data, current_bar=0):
        """Wrapper for entry conditions"""
        return KalmanLogic.check_entry_conditions(self, current_data, current_bar)

    def check_exit_conditions(self, current_data, current_price):
        """Check exit conditions wrapper"""
        current_bar = 0
        if hasattr(self, 'data') and self.data is not None and len(self.data) > 0:
            current_bar = len(self.data) - 1

        if not self.position_state or self.position_state.get('price') is None:
            return None

        entry_price = self.position_state.get('price', 0)
        entry_time = self.position_state.get('time')
        position_type = self.position_state.get('type')

        return KalmanLogic.check_exit_conditions(
            self,
            current_data,
            entry_price,
            current_price,
            position_type,
            entry_time,
            current_bar
        )


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST STRATEGY - WITH WORKING SHORT TRADES
# ═══════════════════════════════════════════════════════════════════════════
class BacktestKalmanTrendStrategy(Strategy, KalmanLogic):
    """Backtest implementation - FULLY FIXED WITH SHORT TRADES"""

    # Kalman Filter Parameters
    process_noise_1 = KALMAN_PARAMS['process_noise_1']
    process_noise_2 = KALMAN_PARAMS['process_noise_2']
    measurement_noise = KALMAN_PARAMS['measurement_noise']
    trend_lookback = KALMAN_PARAMS['trend_lookback']
    strength_smooth = KALMAN_PARAMS['strength_smooth']

    # Strategy Configuration
    trading_direction = KALMAN_PARAMS['trading_direction']
    risk_reward = KALMAN_PARAMS['risk_reward']
    lookback = KALMAN_PARAMS['lookback']
    window = KALMAN_PARAMS['window']
    strength_smooth_param = KALMAN_PARAMS['strength_smooth_param']

    # Moving Averages
    ma_fast_period = KALMAN_PARAMS['ma_fast_period']
    ma_slow_period = KALMAN_PARAMS['ma_slow_period']

    # Technical Indicators
    rsi_period = KALMAN_PARAMS['rsi_period']
    atr_period = KALMAN_PARAMS['atr_period']
    adx_period = KALMAN_PARAMS['adx_period']
    bb_period = KALMAN_PARAMS['bb_period']
    bb_std = KALMAN_PARAMS['bb_std']

    # LONG Entry Conditions
    long_kalman_strength_min = KALMAN_PARAMS['long_kalman_strength_min']
    long_rsi_min = KALMAN_PARAMS['long_rsi_min']
    long_rsi_max = KALMAN_PARAMS['long_rsi_max']
    long_pullback_percent = KALMAN_PARAMS['long_pullback_percent']

    # SHORT Entry Conditions
    short_kalman_strength_min = KALMAN_PARAMS['short_kalman_strength_min']
    short_rsi_max = KALMAN_PARAMS['short_rsi_max']
    short_rsi_min = KALMAN_PARAMS['short_rsi_min']
    short_rally_percent = KALMAN_PARAMS['short_rally_percent']

    # Common Entry
    volume_min_ratio = KALMAN_PARAMS['volume_min_ratio']
    volume_period = KALMAN_PARAMS['volume_period']

    # Risk Management
    stop_loss_pct = KALMAN_PARAMS['stop_loss_pct']
    trailing_stop_pct = KALMAN_PARAMS['trailing_stop_pct']
    atr_multiplier = KALMAN_PARAMS['atr_multiplier']
    risk_per_trade = KALMAN_PARAMS['risk_per_trade']
    max_position_pct = KALMAN_PARAMS['max_position_pct']

    # Exit Conditions
    long_rsi_exit_threshold = KALMAN_PARAMS['long_rsi_exit_threshold']
    short_rsi_exit_threshold = KALMAN_PARAMS['short_rsi_exit_threshold']
    max_hold_bars = KALMAN_PARAMS['max_hold_bars']
    min_hold_bars = KALMAN_PARAMS.get('min_hold_bars', 2)
    max_hold_seconds = KALMAN_PARAMS['max_hold_seconds']

    # Market State Filters
    min_adx = KALMAN_PARAMS['min_adx']
    min_volatility = KALMAN_PARAMS['min_volatility']
    max_spread_pct = KALMAN_PARAMS['max_spread_pct']
    cooldown_bars = KALMAN_PARAMS['cooldown_bars']

    def __init__(self, broker, data, params):
        Strategy.__init__(self, broker, data, params)

        config = {}
        for key in KALMAN_PARAMS.keys():
            if hasattr(self, key):
                config[key] = getattr(self, key)

        KalmanLogic.__init__(self, config=config, trading_app=None)

        self._entry_price = np.nan
        self._entry_i = None
        self._position_type = None
        self._trailing_stop = None
        self.trade_log = []
        self.open_trade = None
        self.last_trade_bar = -1000
        self.consecutive_losses = 0

    def init(self):
        """Initialize backtest"""
        direction_display = {
            'long_only': 'LONG ONLY',
            'short_only': 'SHORT ONLY',
            'both': 'LONG & SHORT'
        }.get(self.trading_direction, 'LONG & SHORT')

        print(f"\n{'=' * 80}")
        print(f"🚀 KALMAN BACKTEST - LONG & SHORT TRADING ENABLED")
        print(f"{'=' * 80}")
        print(f"\n📊 TRADING DIRECTION: {direction_display}")
        print(f"\n🔧 LONG PARAMETERS:")
        print(f"   Kalman Strength Min: {self.long_kalman_strength_min}")
        print(f"   Pullback %: {self.long_pullback_percent}%")
        print(f"\n🔧 SHORT PARAMETERS:")
        print(f"   Kalman Strength Min: {self.short_kalman_strength_min}")
        print(f"   Rally %: {self.short_rally_percent}%")
        print(f"\n🔧 COMMON:")
        print(f"   Stop Loss: {self.stop_loss_pct * 100}%")
        print(f"   Trailing Stop: {self.trailing_stop_pct * 100}%")
        print(f"   Risk/Reward: {self.risk_reward}:1")
        print(f"   Cooldown Bars: {self.cooldown_bars}")
        print(f"   Min Hold Bars: {self.min_hold_bars}")
        print(f"{'=' * 80}\n")

        self.df = KalmanIndicatorCalculator.calculate(self.data.df.copy(), self.config)
        self.timestamps = self.df.index.to_numpy()
        self.df_enhanced = self.df.copy()

        # Register indicators
        self.ma_fast = self.I(self.get_values, 'MA_Fast_closed')
        self.ma_slow = self.I(self.get_values, 'MA_Slow_closed')
        self.rsi = self.I(self.get_values, 'RSI_closed')
        self.atr = self.I(self.get_values, 'ATR_closed')
        self.kalman_strength = self.I(self.get_values, 'Kalman_Strength_closed')
        self.kalman_color_num = self.I(self.get_color_values, 'Kalman_Color_closed')
        self.volume_ratio = self.I(self.get_values, 'Volume_Ratio')
        self.pullback_pct = self.I(self.get_values, 'Pullback_Pct_closed')
        self.rally_pct = self.I(self.get_values, 'Rally_Pct_closed')
        self.adx = self.I(self.get_values, 'ADX_closed')
        self.volatility = self.I(self.get_values, 'Volatility_closed')

    def get_values(self, column):
        return self.df[column].values if column in self.df.columns else np.zeros(len(self.data))

    def get_color_values(self, column):
        if column in self.df.columns:
            color_map = {'green': 1, 'red': -1, 'blue': 0.5, 'orange': -0.5, 'gray': 0}
            return np.array([color_map.get(str(x).lower(), 0) for x in self.df[column]])
        return np.zeros(len(self.data))

    def _calculate_quality_score(self, current_data, direction='long'):
        """Calculate quality score for backtest"""
        kalman_strength = current_data.get('Kalman_Strength_closed', 0)
        kalman_color = current_data.get('Kalman_Color_closed', 'gray')
        ma_fast = current_data.get('MA_Fast_closed', 0)
        ma_slow = current_data.get('MA_Slow_closed', 0)
        rsi = current_data.get('RSI_closed', 50)
        volume_ratio = current_data.get('Volume_Ratio', 1.0)
        adx = current_data.get('ADX_closed', 0)

        score = 50

        if direction == 'long':
            if kalman_color == 'green':
                score += 20
                if kalman_strength > 50:
                    score += 10
            else:
                score -= 20
            if ma_fast > ma_slow:
                score += 15
            if 30 < rsi < 70:
                score += 10
            if volume_ratio > 1.2:
                score += 10
            if adx > 20:
                score += 5

        else:  # short
            if kalman_color == 'red':
                score += 20
                if kalman_strength < -50:
                    score += 10
            else:
                score -= 20
            if ma_fast < ma_slow:
                score += 15
            if 30 < rsi < 70:
                score += 10
            if volume_ratio > 1.2:
                score += 10
            if adx > 20:
                score += 5

        return max(0, min(100, score))

    def next(self):
        """Main backtest loop - FULLY FIXED WITH SHORT TRADES"""
        if len(self.data) < 2:
            return

        idx = len(self.data) - 1
        price = self.data.Close[-1]
        current_bar = idx

        # Get indicator values
        kalman_strength = self.kalman_strength[idx]
        kalman_color_num = self.kalman_color_num[idx]
        atr = self.atr[idx]
        adx = self.adx[idx]
        volatility = self.volatility[idx]
        rsi = self.rsi[idx]
        volume_ratio = self.volume_ratio[idx]
        pullback_pct = self.pullback_pct[idx]
        rally_pct = self.rally_pct[idx]
        ma_fast = self.ma_fast[idx]
        ma_slow = self.ma_slow[idx]

        kalman_color = 'green' if kalman_color_num == 1 else ('red' if kalman_color_num == -1 else 'gray')

        data_dict = {
            'Close': price,
            'MA_Fast_closed': ma_fast,
            'MA_Slow_closed': ma_slow,
            'RSI_closed': rsi,
            'ATR_closed': atr,
            'Kalman_Strength_closed': kalman_strength,
            'Kalman_Color_closed': kalman_color,
            'Volume_Ratio': volume_ratio,
            'Pullback_Pct_closed': pullback_pct,
            'Rally_Pct_closed': rally_pct,
            'ADX_closed': adx,
            'Volatility_closed': volatility,
            'stop_loss': price * (1 - self.stop_loss_pct),
            'trailing_stop_loss': self.data.High[-20:].max() * (1 - self.trailing_stop_pct) if len(self.data) >= 20 else price * (1 - self.trailing_stop_pct),
            'short_stop_loss': price * (1 + self.stop_loss_pct),
            'short_trailing_stop': self.data.Low[-20:].min() * (1 + self.trailing_stop_pct) if len(self.data) >= 20 else price * (1 + self.trailing_stop_pct),
        }

        # CLOSE EXISTING POSITION
        if self.position:
            exit_reason = None

            if self._position_type == 'long':
                if price > self._entry_price:
                    new_trailing = price * (1 - self.trailing_stop_pct)
                    if new_trailing > getattr(self, '_trailing_stop', new_trailing):
                        self._trailing_stop = new_trailing

                if self._entry_price is not None and self._entry_price > 0:
                    exit_reason = self.check_exit_conditions(
                        data_dict, self._entry_price, price, 'long', None, current_bar
                    )

            else:  # short
                if price < self._entry_price:
                    new_trailing = price * (1 + self.trailing_stop_pct)
                    if new_trailing < getattr(self, '_trailing_stop', new_trailing):
                        self._trailing_stop = new_trailing

                if self._entry_price is not None and self._entry_price > 0:
                    exit_reason = self.check_exit_conditions(
                        data_dict, self._entry_price, price, 'short', None, current_bar
                    )

            if exit_reason:
                if self.open_trade:
                    profit_pct = (price / self._entry_price - 1) * 100 if self._position_type == 'long' else (self._entry_price / price - 1) * 100
                    self.open_trade.update({
                        "exit_time": self.timestamps[idx],
                        "exit_price": price,
                        "type": "SELL" if self._position_type == 'long' else "COVER",
                        "pnl": self.position.pl,
                        "pnl_pct": profit_pct,
                        "exit_reason": exit_reason,
                        "bars_held": current_bar - self._entry_i if self._entry_i is not None else 0
                    })
                    self.trade_log.append(self.open_trade)
                    self.open_trade = None

                self.record_trade(self.position.pl, exit_reason, self._position_type)
                self.last_trade_bar = current_bar
                self.position.close()
                self._entry_i = None
                self._entry_price = None
                self._position_type = None
                self._trailing_stop = None
                return

        # OPEN NEW POSITION
        if not self.position:
            if current_bar - self.last_trade_bar < self.cooldown_bars:
                return

            if adx <= self.min_adx or volatility <= self.min_volatility:
                return

            # Check LONG
            long_signal = False
            short_signal = False

            if self.trading_direction in ['long_only', 'both']:
                long_mandatory, _ = self.check_long_entry_conditions(data_dict, current_bar)
                long_score = self._calculate_quality_score(data_dict, 'long')
                if long_mandatory >= 2 and long_score >= 50:
                    long_signal = True

            # Check SHORT
            if self.trading_direction in ['short_only', 'both']:
                short_mandatory, _ = self.check_short_entry_conditions(data_dict, current_bar)
                short_score = self._calculate_quality_score(data_dict, 'short')
                if short_mandatory >= 2 and short_score >= 50:
                    short_signal = True

            # Choose signal
            if long_signal and short_signal:
                long_score = self._calculate_quality_score(data_dict, 'long')
                short_score = self._calculate_quality_score(data_dict, 'short')
                if long_score >= short_score:
                    short_signal = False
                else:
                    long_signal = False

            if long_signal:
                size = self.calculate_position_size(self.equity, atr, price, 'long')
                if size > 0:
                    self.buy(size=size)
                    self._entry_i = current_bar
                    self._entry_price = price
                    self._position_type = 'long'
                    self._trailing_stop = price * (1 - self.trailing_stop_pct)

                    self.open_trade = {
                        "time": self.timestamps[idx],
                        "type": "BUY",
                        "position_type": "long",
                        "price": price,
                        "entry_price": price,
                        "quantity": size,
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": 0,
                        "pnl_pct": 0,
                        "exit_reason": "",
                        "bars_held": 0,
                        "quality_score": long_score
                    }
                    print(f"  📈 LONG ENTRY at bar {current_bar}: price={price:.4f}, size={size:.4f}, score={long_score}")

            elif short_signal:
                size = self.calculate_position_size(self.equity, atr, price, 'short')
                if size > 0:
                    self.sell(size=size)
                    self._entry_i = current_bar
                    self._entry_price = price
                    self._position_type = 'short'
                    self._trailing_stop = price * (1 + self.trailing_stop_pct)

                    self.open_trade = {
                        "time": self.timestamps[idx],
                        "type": "SHORT",
                        "position_type": "short",
                        "price": price,
                        "entry_price": price,
                        "quantity": size,
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": 0,
                        "pnl_pct": 0,
                        "exit_reason": "",
                        "bars_held": 0,
                        "quality_score": short_score
                    }
                    print(f"  📉 SHORT ENTRY at bar {current_bar}: price={price:.4f}, size={size:.4f}, score={short_score}")