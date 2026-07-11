"""
LSTM Model - Professional Trading Model with Price Anchoring
Version: 3.0.0 (Fixed with enhanced diagnostics)
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
import traceback

warnings.filterwarnings('ignore')


@dataclass
class PredictionRecord:
    """Record of a single prediction for accuracy tracking"""
    timestamp: str
    predicted_direction: int  # -1, 0, 1
    predicted_price: float
    actual_price: float = None
    actual_direction: int = None
    was_correct: bool = None
    confidence: float = 0.0

    def update_actual(self, actual_price: float):
        """Update with actual outcome"""
        self.actual_price = actual_price
        if self.predicted_price > 0:
            actual_change = (actual_price - self.predicted_price) / self.predicted_price
            self.actual_direction = 1 if actual_change > 0.001 else (-1 if actual_change < -0.001 else 0)
            self.was_correct = (self.predicted_direction == self.actual_direction) or \
                               (self.predicted_direction != 0 and
                                np.sign(self.predicted_direction) == np.sign(self.actual_direction))


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    rolling_directional_accuracy: float = 0.5
    rolling_magnitude_accuracy: float = 0.5
    confidence_calibration: float = 1.0
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_error_pct: float = 0.0
    avg_magnitude_error: float = 0.0
    regime_accuracy: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'rolling_directional_accuracy': self.rolling_directional_accuracy,
            'rolling_magnitude_accuracy': self.rolling_magnitude_accuracy,
            'confidence_calibration': self.confidence_calibration,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'avg_error_pct': self.avg_error_pct,
            'avg_magnitude_error': self.avg_magnitude_error,
            'regime_accuracy': self.regime_accuracy
        }


class LSTMModel:
    """
    Professional LSTM Model with Price Anchoring and Magnitude Correction.
    Fixed version with enhanced diagnostics and data validation.
    """

    def __init__(
            self,
            seq_len: int = 60,
            n_future: int = 5,
            min_train_acc: float = 55.0,
            confidence_threshold: float = 0.30,
            features: Optional[List[str]] = None,
            close_idx: int = 0,
            scaler: Optional[MinMaxScaler] = None,
            use_robust_scaler: bool = False
    ):
        # ===== Core Configuration =====
        self.seq_len = int(seq_len)
        self.sequence_length = int(seq_len)
        self.n_future = int(n_future)
        self.min_train_acc = float(min_train_acc)
        self.confidence_threshold = float(confidence_threshold)

        # ===== Model State =====
        self.model: Optional[Sequential] = None
        self.history = None
        self.is_trained = False

        # ===== Scaler Configuration =====
        if scaler:
            self.scaler = scaler
        elif use_robust_scaler:
            self.scaler = RobustScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))

        # ===== Data Schema =====
        self.features: Optional[List[str]] = features or ["Close"]
        self.close_col: Optional[str] = None
        self.close_idx: Optional[int] = close_idx
        self.n_features: int = len(self.features)

        # ===== Training Metrics =====
        self.train_accuracy: Optional[float] = None
        self.val_accuracy: Optional[float] = None
        self.val_rmse: Optional[float] = None
        self.val_mape: Optional[float] = None
        self.directional_accuracy: Optional[float] = None

        # ===== ROLLING PERFORMANCE TRACKING =====
        self.prediction_history: deque = deque(maxlen=100)
        self.metrics = ModelMetrics()
        self.last_prediction_price: Optional[float] = None
        self.last_prediction_direction: Optional[int] = None
        self.last_forecast: Optional[np.ndarray] = None

        # ===== ADAPTIVE ADJUSTMENT =====
        self.predadjcoef_enabled = True
        self.adj_coef_strength = 2.0
        self.adj_coef_bounds = (0.2, 3.0)
        self.adjustment_history = deque(maxlen=50)
        self.error_history = deque(maxlen=50)
        self.magnitude_error_history = deque(maxlen=50)
        self.volatility_lookback = 20

        # ===== PRICE ANCHORING =====
        self.price_anchoring_enabled = True
        self.anchor_strength = 0.95
        self.max_prediction_change_pct = 2.0
        self.learned_scale_factor = 1.0
        self.scale_factor_history = deque(maxlen=100)

        # ===== MAGNITUDE CORRECTION =====
        self.magnitude_correction_enabled = True
        self.expected_volatility_pct = 0.5
        self.volatility_multiplier = 3.0

        # ===== DIRECTIONAL THRESHOLDS =====
        self.direction_threshold = 0.15
        self.strong_signal_threshold = 0.30
        self.noise_threshold = 0.08

        # ===== REGIME DETECTION =====
        self.current_regime = "unknown"
        self.regime_history = deque(maxlen=20)

        # ===== CONFIDENCE CALIBRATION =====
        self.confidence_history = deque(maxlen=50)
        self.outcome_history = deque(maxlen=50)

        # ===== TRAINING DATA REFERENCE =====
        self.training_price_mean: Optional[float] = None
        self.training_price_std: Optional[float] = None

        # ===== DIAGNOSTICS =====
        self.last_data_diagnostics = {}

    # ==================== ENHANCED DATA VALIDATION ====================

    def _ensure_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate input DataFrame with comprehensive diagnostics"""

        print(f"\n{'=' * 60}")
        print(f"🔍 DATA VALIDATION DIAGNOSTICS")
        print(f"{'=' * 60}")
        print(f"📊 Input shape: {df.shape}")
        print(f"📊 Columns ({len(df.columns)}): {list(df.columns)}")
        if len(df) > 0:
            print(f"📊 Index range: {df.index[0]} to {df.index[-1]}")

        # Make a copy
        df_clean = df.copy()

        # STEP 1: Convert boolean columns to float FIRST
        bool_cols = df_clean.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_clean[col] = df_clean[col].astype(float)
        if len(bool_cols) > 0:
            print(f"✅ Converted {len(bool_cols)} boolean columns to float")

        # STEP 2: Convert object columns with boolean values
        object_cols = df_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
            try:
                # Try to convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                if df_clean[col].dtype == 'object':
                    # Map common boolean representations
                    mapping = {
                        'True': 1.0, 'False': 0.0,
                        'true': 1.0, 'false': 0.0,
                        'TRUE': 1.0, 'FALSE': 0.0,
                        'Yes': 1.0, 'No': 0.0,
                        'yes': 1.0, 'no': 0.0,
                        'Y': 1.0, 'N': 0.0,
                        'y': 1.0, 'n': 0.0
                    }
                    df_clean[col] = df_clean[col].map(mapping).fillna(0.0).astype(float)
            except:
                # If conversion fails, drop the column
                df_clean = df_clean.drop(columns=[col])
                print(f"⚠️ Dropped non-numeric column: {col}")

        # STEP 3: Select only numeric columns
        numeric_df = df_clean.select_dtypes(include=[np.number])
        print(f"📊 Numeric columns: {len(numeric_df.columns)}")

        if len(numeric_df.columns) == 0:
            raise ValueError("No numeric columns found in data!")

        # STEP 4: Check for columns with TOO MANY NaNs (>50%)
        nan_percentages = numeric_df.isna().sum() / len(numeric_df)
        cols_to_drop = nan_percentages[nan_percentages > 0.5].index.tolist()

        if cols_to_drop:
            print(f"⚠️ Dropping {len(cols_to_drop)} columns with >50% NaNs:")
            for col in cols_to_drop[:5]:
                print(f"   - {col}: {nan_percentages[col] * 100:.1f}% NaNs")
            numeric_df = numeric_df.drop(columns=cols_to_drop)

        # STEP 5: Handle remaining NaNs - DON'T drop rows yet
        if numeric_df.isna().any().any():
            print(f"⚠️ Handling remaining NaNs...")

            # Forward fill (carry last value forward)
            numeric_df = numeric_df.ffill()

            # Backward fill for any remaining at the beginning
            numeric_df = numeric_df.bfill()

            # If still have NaNs, fill with column mean
            if numeric_df.isna().any().any():
                print(f"⚠️ Filling remaining NaNs with column means...")
                numeric_df = numeric_df.fillna(numeric_df.mean())

        # STEP 6: Handle infinite values
        inf_mask = np.isinf(numeric_df.values)
        if inf_mask.any():
            print(f"⚠️ Found {inf_mask.sum()} infinite values, replacing with NaN...")
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            numeric_df = numeric_df.fillna(numeric_df.mean())

        # STEP 7: Final check
        if len(numeric_df) == 0:
            # If we lost all rows, try a more aggressive approach
            print("⚠️ No rows left! Trying aggressive cleaning...")

            # Drop columns with ANY NaN
            numeric_df = numeric_df.dropna(axis=1)

            # If still no rows, use the original data with minimal cleaning
            if len(numeric_df) == 0:
                print("⚠️ Using original data with minimal cleaning...")
                numeric_df = df_clean.select_dtypes(include=[np.number])
                numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        print(f"✅ Final data: {len(numeric_df)} rows, {len(numeric_df.columns)} columns")
        print(f"{'=' * 60}\n")

        return numeric_df

    def _infer_close_col(self, cols: List[str]) -> str:
        """Find the close price column"""
        for c in cols:
            if c.lower() == "close":
                return c
        return cols[0]

    def _check_and_set_schema(self, df: pd.DataFrame, features: Optional[List[str]]):
        """Validate and set data schema"""
        print(f"\n{'=' * 60}")
        print(f"🔍 SCHEMA VALIDATION")
        print(f"{'=' * 60}")

        if features is None:
            cols = list(df.columns)
            print(f"📊 No features specified, using all columns: {cols}")
        else:
            missing = [c for c in features if c not in df.columns]
            if missing:
                print(f"❌ Missing required feature columns: {missing}")
                print(f"📊 Available columns: {list(df.columns)}")
                raise ValueError(f"Missing required feature columns: {missing}")
            cols = list(features)
            print(f"📊 Using specified features: {cols}")

        close_col = self._infer_close_col(cols)
        print(f"📊 Inferred close column: {close_col}")

        self.features = cols
        self.close_col = close_col
        self.close_idx = self.features.index(self.close_col)
        self.n_features = len(self.features)

        print(f"✅ Schema set: features={self.features}, close_idx={self.close_idx}")
        print(f"{'=' * 60}\n")

    def _create_sequences_multivar(self, scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for single-step prediction"""
        X, y = [], []
        N = scaled.shape[0]
        if N <= self.seq_len:
            return np.array([]), np.array([])

        for i in range(N - self.seq_len):
            X.append(scaled[i: i + self.seq_len, :])
            y.append(scaled[i + self.seq_len, self.close_idx])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    def _create_sequences_multistep(self, scaled: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for MULTI-STEP DIRECT prediction"""
        X, y = [], []
        N = scaled.shape[0]
        if N <= self.seq_len + n_steps:
            return np.array([]), np.array([])

        for i in range(N - self.seq_len - n_steps + 1):
            X.append(scaled[i: i + self.seq_len, :])
            future_closes = scaled[i + self.seq_len: i + self.seq_len + n_steps, self.close_idx]
            y.append(future_closes)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _inverse_close_batch(self, y_scaled: np.ndarray, ref_rows_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original price scale"""
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        m = y_scaled.shape[0]
        if ref_rows_scaled.shape[0] != m:
            ref_rows_scaled = np.tile(ref_rows_scaled[0:1, :], (m, 1))

        combo = ref_rows_scaled.copy()
        combo[:, self.close_idx] = y_scaled[:, 0] if y_scaled.shape[1] == 1 else y_scaled[:, -1]
        inv = self.scaler.inverse_transform(combo)
        return inv[:, self.close_idx]

    # ==================== PRICE ANCHORING ====================

    def anchor_predictions(self, raw_forecast: np.ndarray, current_price: float,
                           recent_prices: np.ndarray) -> np.ndarray:
        """
        Anchor predictions to current price to prevent unrealistic jumps.
        """
        if raw_forecast is None or len(raw_forecast) == 0:
            return raw_forecast

        # Calculate what the model THINKS the change should be
        model_start = raw_forecast[0]
        model_end = raw_forecast[-1]
        model_change_pct = (model_end - model_start) / model_start * 100 if model_start > 0 else 0

        # Calculate recent actual volatility
        if len(recent_prices) > 1:
            returns = np.diff(recent_prices) / recent_prices[:-1] * 100
            actual_volatility = np.std(returns) if len(returns) > 0 else 0.5
            self.expected_volatility_pct = max(0.1, actual_volatility)

        # Calculate maximum reasonable change
        max_change_pct = self.expected_volatility_pct * self.volatility_multiplier * len(raw_forecast)
        max_change_pct = min(max_change_pct, self.max_prediction_change_pct * len(raw_forecast))

        # Detect if model has systematic bias
        price_discrepancy = abs(model_start - current_price) / current_price * 100

        if price_discrepancy > 5:  # More than 5% off from current price
            # Model is systematically biased - need to recalibrate
            if model_start > 0:
                new_scale_factor = current_price / model_start
                self.scale_factor_history.append(new_scale_factor)
                self.learned_scale_factor = np.mean(list(self.scale_factor_history))

        # Extract the MODEL's predicted percentage change
        if model_start > 0:
            predicted_changes_pct = (raw_forecast - model_start) / model_start * 100
        else:
            predicted_changes_pct = np.zeros_like(raw_forecast)

        # Clamp the changes to reasonable bounds
        clamped_changes_pct = np.clip(predicted_changes_pct, -max_change_pct, max_change_pct)

        # Apply clamped changes to CURRENT price
        anchored_forecast = current_price * (1 + clamped_changes_pct / 100)

        # Blend with pure direction signal if discrepancy was huge
        if price_discrepancy > 20:
            direction = np.sign(model_end - model_start)
            minimal_change = self.expected_volatility_pct * direction
            minimal_forecast = np.array([
                current_price * (1 + minimal_change * (i + 1) / len(raw_forecast) / 100)
                for i in range(len(raw_forecast))
            ])
            blend_weight = min(1.0, (price_discrepancy - 20) / 30)
            anchored_forecast = (1 - blend_weight) * anchored_forecast + blend_weight * minimal_forecast

        return anchored_forecast.astype(np.float32)

    def correct_magnitude(self, forecast: np.ndarray, current_price: float,
                          recent_prices: np.ndarray) -> np.ndarray:
        """Apply magnitude correction based on historical accuracy."""
        if not self.magnitude_correction_enabled or len(self.magnitude_error_history) < 5:
            return forecast

        avg_magnitude_error = np.mean(list(self.magnitude_error_history))

        if abs(avg_magnitude_error) > 0.1:
            correction_factor = 1.0 - (avg_magnitude_error / 100)
            correction_factor = np.clip(correction_factor, 0.5, 1.5)

            changes = forecast - current_price
            corrected_changes = changes * correction_factor
            forecast = current_price + corrected_changes

        return forecast

    # ==================== REGIME DETECTION ====================

    def detect_regime(self, prices: np.ndarray, period: int = 20) -> str:
        """Detect market regime: trending_up, trending_down, ranging, volatile"""
        if len(prices) < period:
            return "unknown"

        recent = prices[-period:]

        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        normalized_slope = slope / np.mean(recent) * 100

        returns = np.diff(recent) / recent[:-1]
        volatility = np.std(returns) * np.sqrt(252) * 100

        range_pct = (np.max(recent) - np.min(recent)) / np.mean(recent) * 100

        if volatility > 30:
            regime = "volatile"
        elif abs(normalized_slope) > 0.5 and range_pct < 3:
            regime = "trending_up" if normalized_slope > 0 else "trending_down"
        elif range_pct < 1.5:
            regime = "ranging"
        else:
            regime = "choppy"

        self.current_regime = regime
        self.regime_history.append(regime)

        return regime

    # ==================== VOLATILITY & ANALYSIS ====================

    def calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate annualized volatility percentage"""
        if len(prices) < 2:
            return 0.0
        prices = prices[-period:] if len(prices) > period else prices
        returns = np.diff(prices) / prices[:-1]
        daily_vol = np.std(returns)
        return daily_vol * np.sqrt(252) * 100

    def calculate_momentum(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate price momentum"""
        if len(prices) < period:
            return 0.0
        return (prices[-1] - prices[-period]) / prices[-period] * 100

    def is_overextended(self, prices: np.ndarray, current_price: float,
                        lookback_period: int = 20) -> Tuple[bool, bool, float]:
        """Check if price is overextended"""
        if len(prices) < lookback_period:
            return False, False, 0.5

        recent = prices[-lookback_period:]
        recent_high = np.max(recent)
        recent_low = np.min(recent)
        range_size = recent_high - recent_low

        if range_size < 1e-10:
            return False, False, 0.5

        position = (current_price - recent_low) / range_size
        overbought = position > 0.85
        oversold = position < 0.15

        return overbought, oversold, position

    # ==================== ROLLING ACCURACY TRACKING ====================

    def update_prediction_outcome(self, actual_price: float):
        """
        Call this with the actual price to update rolling accuracy.
        """
        if self.last_prediction_price is None:
            return

        price_change = (actual_price - self.last_prediction_price) / self.last_prediction_price * 100

        if price_change > self.noise_threshold:
            actual_direction = 1
        elif price_change < -self.noise_threshold:
            actual_direction = -1
        else:
            actual_direction = 0

        was_correct = False
        if self.last_prediction_direction == 0:
            was_correct = abs(price_change) < self.direction_threshold
        else:
            was_correct = np.sign(self.last_prediction_direction) == np.sign(actual_direction)

        self.metrics.total_predictions += 1
        if was_correct:
            self.metrics.correct_predictions += 1

        self.outcome_history.append(1 if was_correct else 0)
        if len(self.outcome_history) >= 5:
            self.metrics.rolling_directional_accuracy = np.mean(list(self.outcome_history))

        self.error_history.append(price_change)
        self.metrics.avg_error_pct = np.mean(np.abs(list(self.error_history)))

        if self.last_forecast is not None and len(self.last_forecast) > 0:
            predicted_change = (self.last_forecast[0] - self.last_prediction_price) / self.last_prediction_price * 100
            magnitude_error = predicted_change - price_change
            self.magnitude_error_history.append(magnitude_error)
            self.metrics.avg_magnitude_error = np.mean(list(self.magnitude_error_history))

        if len(self.confidence_history) >= 10 and len(self.outcome_history) >= 10:
            avg_confidence = np.mean(list(self.confidence_history)[-10:])
            avg_accuracy = np.mean(list(self.outcome_history)[-10:])
            if avg_accuracy > 0:
                self.metrics.confidence_calibration = avg_confidence / avg_accuracy
            else:
                self.metrics.confidence_calibration = 2.0

        print(f"📊 Accuracy Update: {self.metrics.correct_predictions}/{self.metrics.total_predictions} "
              f"({self.metrics.rolling_directional_accuracy * 100:.1f}%) | "
              f"Mag Error: {self.metrics.avg_magnitude_error:.2f}%")

    # ==================== PREDICTION ADJUSTMENT ====================

    def calculate_prediction_adjustment(self, forecast: np.ndarray, actual_price: float,
                                        recent_prices: np.ndarray) -> float:
        """Calculate adaptive adjustment coefficient based on recent errors."""
        if forecast is None or len(forecast) == 0:
            return 1.0

        forecast_error_pct = (forecast[0] - actual_price) / actual_price * 100

        recent_error_bias = 0.0
        if len(self.error_history) >= 5:
            recent_errors = list(self.error_history)[-5:]
            recent_error_bias = np.mean(recent_errors)

        magnitude_bias = 0.0
        if len(self.magnitude_error_history) >= 5:
            magnitude_bias = np.mean(list(self.magnitude_error_history)[-5:])

        recent_vol = self.calculate_volatility(recent_prices[-self.volatility_lookback:])
        vol_normalizer = max(recent_vol, 1.0)

        error_adjustment = -forecast_error_pct / vol_normalizer * self.adj_coef_strength
        bias_adjustment = -recent_error_bias / vol_normalizer * 0.5
        magnitude_adjustment = -magnitude_bias / vol_normalizer * 0.3

        adj_coef = 1.0 + error_adjustment + bias_adjustment + magnitude_adjustment

        min_bound, max_bound = self.adj_coef_bounds
        adj_coef = np.clip(adj_coef, min_bound, max_bound)

        self.adjustment_history.append(adj_coef)

        return adj_coef

    # ==================== CONFIDENCE CALCULATION ====================

    def calculate_ml_confidence(self, forecast: np.ndarray, actual_price: float,
                                recent_prices: np.ndarray, predict_adj_coef: float = 1.0) -> float:
        """Calculate confidence based on ACTUAL ACCURACY, not internal consistency."""

        rolling_acc = self.metrics.rolling_directional_accuracy
        accuracy_score = rolling_acc

        regime = self.detect_regime(recent_prices)
        price_change_pct = (forecast[-1] - actual_price) / actual_price * 100

        regime_score = 0.5
        if regime == "trending_up" and price_change_pct > 0:
            regime_score = 0.8
        elif regime == "trending_down" and price_change_pct < 0:
            regime_score = 0.8
        elif regime == "ranging":
            regime_score = 0.7 if abs(price_change_pct) < 0.3 else 0.3
        elif regime == "volatile":
            regime_score = 0.4
        elif regime == "choppy":
            regime_score = 0.3

        if (regime == "trending_up" and price_change_pct < -0.2) or \
                (regime == "trending_down" and price_change_pct > 0.2):
            regime_score = 0.2

        abs_change = abs(price_change_pct)
        if abs_change < self.noise_threshold:
            signal_score = 0.3
        elif abs_change < self.direction_threshold:
            signal_score = 0.5
        elif abs_change < self.strong_signal_threshold:
            signal_score = 0.8
        elif abs_change < 1.0:
            signal_score = 0.6
        else:
            signal_score = 0.3

        recent_vol = self.calculate_volatility(recent_prices[-20:])
        forecast_vol = self.calculate_volatility(forecast) if len(forecast) > 1 else 0

        vol_ratio = forecast_vol / max(recent_vol, 0.1)
        vol_score = 1.0 if vol_ratio < 1.5 else max(0.3, 1.0 / vol_ratio)

        overbought, oversold, position = self.is_overextended(recent_prices, actual_price)
        extension_penalty = 1.0
        if overbought and price_change_pct > 0:
            extension_penalty = 0.5
        elif oversold and price_change_pct < 0:
            extension_penalty = 0.5

        tech_score = vol_score * extension_penalty

        raw_confidence = (
                accuracy_score * 0.40 +
                regime_score * 0.25 +
                signal_score * 0.15 +
                tech_score * 0.20
        )

        if self.metrics.confidence_calibration > 1.2:
            calibration_factor = 1.0 / self.metrics.confidence_calibration
            raw_confidence *= calibration_factor

        adjustment_distance = abs(predict_adj_coef - 1.0)
        if adjustment_distance > 0.3:
            raw_confidence *= (1.0 - adjustment_distance * 0.3)

        if abs(self.metrics.avg_magnitude_error) > 1.0:
            magnitude_penalty = 1.0 - min(0.3, abs(self.metrics.avg_magnitude_error) / 10)
            raw_confidence *= magnitude_penalty

        if self.metrics.total_predictions < 10:
            raw_confidence *= 0.7
        elif self.metrics.total_predictions < 20:
            raw_confidence *= 0.85

        final_confidence = np.clip(raw_confidence, 0.1, 0.95)

        return final_confidence

    # ==================== MODEL BUILDING ====================

    def build_model(self, multi_step: bool = True) -> Sequential:
        """Build LSTM model"""
        output_dim = self.n_future if multi_step else 1

        model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(self.seq_len, self.n_features),
                 kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),

            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),

            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            BatchNormalization(),

            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(32, activation='relu'),

            Dense(output_dim)
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='huber',
            metrics=['mae']
        )

        return model

    # ==================== TRAINING ====================

    def train(self, df: pd.DataFrame, features: Optional[List[str]] = None,
              epochs: int = 100, batch_size: int = 32, verbose: int = 1,
              multi_step: bool = True) -> bool:
        """Train the LSTM model with enhanced diagnostics"""
        try:
            print(f"\n{'=' * 60}")
            print(f"🚀 LSTM MODEL TRAINING STARTED")
            print(f"{'=' * 60}")

            # Validate and clean data
            df = self._ensure_dataframe(df)
            self._check_and_set_schema(df, features)

            # Extract data
            data = df[self.features].astype(float).values
            close_prices = data[:, self.close_idx]

            print(f"\n📊 Data statistics:")
            print(f"   Total rows: {len(data)}")
            print(f"   Close price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
            print(f"   Close price mean: ${close_prices.mean():.2f}")
            print(f"   Close price std: ${close_prices.std():.2f}")

            self.training_price_mean = np.mean(close_prices)
            self.training_price_std = np.std(close_prices)

            if len(close_prices) > 1:
                returns = np.diff(close_prices) / close_prices[:-1] * 100
                self.expected_volatility_pct = np.std(returns)
                print(f"   Expected volatility: {self.expected_volatility_pct:.3f}%")

            # Scale data
            scaled = self.scaler.fit_transform(data)
            print(f"\n📊 Scaled data shape: {scaled.shape}")

            # Create sequences
            if multi_step:
                X, y = self._create_sequences_multistep(scaled, self.n_future)
                print(
                    f"📊 Multi-step sequences: X={X.shape if X.size > 0 else 'empty'}, y={y.shape if y.size > 0 else 'empty'}")
            else:
                X, y = self._create_sequences_multivar(scaled)
                print(
                    f"📊 Single-step sequences: X={X.shape if X.size > 0 else 'empty'}, y={y.shape if y.size > 0 else 'empty'}")

            if X.size == 0:
                required_rows = self.seq_len + (self.n_future if multi_step else 1)
                raise ValueError(
                    f"Not enough data for sequences. Need > {required_rows} rows, have {len(data)} rows. "
                    f"Try reducing seq_len ({self.seq_len}) or n_future ({self.n_future})."
                )

            # Split data
            split = int(0.8 * len(X))
            if split == 0 or split >= len(X):
                raise ValueError(f"Not enough data for train/validation split. Total sequences: {len(X)}")

            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            print(f"\n📊 Train/Validation split:")
            print(f"   Training: {X_train.shape[0]} sequences")
            print(f"   Validation: {X_val.shape[0]} sequences")

            # Build and train model
            self.model = self.build_model(multi_step=multi_step)
            print(f"\n✅ Model built with {self.model.count_params():,} parameters")

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5, min_lr=1e-7, verbose=1)
            ]

            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks
            )

            # Evaluate
            y_train_pred = self.model.predict(X_train, verbose=0)
            y_val_pred = self.model.predict(X_val, verbose=0)

            train_rmse = float(np.sqrt(mean_squared_error(y_train.flatten(), y_train_pred.flatten())))
            val_rmse = float(np.sqrt(mean_squared_error(y_val.flatten(), y_val_pred.flatten())))

            self.train_accuracy = max(0.0, (1.0 - train_rmse) * 100.0)
            self.val_accuracy = max(0.0, (1.0 - val_rmse) * 100.0)

            # Calculate directional accuracy
            if multi_step:
                train_pred_dir = np.sign(y_train_pred[:, 0] - y_train[:, 0])
                val_pred_dir = np.sign(y_val_pred[:, 0] - y_val[:, 0])
                train_actual_dir = np.sign(np.diff(y_train[:, 0], prepend=y_train[0, 0]))
                val_actual_dir = np.sign(np.diff(y_val[:, 0], prepend=y_val[0, 0]))
            else:
                train_pred_dir = np.sign(np.diff(y_train_pred.flatten(), prepend=y_train_pred[0, 0]))
                val_pred_dir = np.sign(np.diff(y_val_pred.flatten(), prepend=y_val_pred[0, 0]))
                train_actual_dir = np.sign(np.diff(y_train.flatten(), prepend=y_train[0, 0]))
                val_actual_dir = np.sign(np.diff(y_val.flatten(), prepend=y_val[0, 0]))

            train_dir_acc = np.mean(train_pred_dir == train_actual_dir) * 100
            val_dir_acc = np.mean(val_pred_dir == val_actual_dir) * 100

            self.directional_accuracy = val_dir_acc
            self.val_rmse = val_rmse

            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETE")
            print("=" * 60)
            print(f"📊 Scaled Train RMSE:     {train_rmse:.6f}")
            print(f"📊 Scaled Val RMSE:       {val_rmse:.6f}")
            print(f"📈 Train Directional Acc: {train_dir_acc:.2f}%")
            print(f"📈 Val Directional Acc:   {val_dir_acc:.2f}% ⬅️ KEY METRIC")
            print("=" * 60 + "\n")

            self.metrics.rolling_directional_accuracy = val_dir_acc / 100.0
            self.is_trained = True

            if val_dir_acc < self.min_train_acc:
                print(f"⚠️ Directional accuracy {val_dir_acc:.1f}% below threshold {self.min_train_acc:.1f}%")
                print(f"   Model will operate with reduced confidence")
                self.metrics.rolling_directional_accuracy = max(0.3, val_dir_acc / 100.0)

            return True

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            print("\n📋 DIAGNOSTICS:")
            if hasattr(self, 'last_data_diagnostics'):
                for key, value in self.last_data_diagnostics.items():
                    print(f"   {key}: {value}")
            traceback.print_exc()
            self.is_trained = False
            return False

    # ==================== PREDICTION ====================

    def predict(self, df: pd.DataFrame, n_future: int = None) -> Tuple[float, int, Optional[np.ndarray]]:
        """Predict future prices with calibrated confidence and price anchoring."""
        try:
            if self.model is None or not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")

            print(f"\n{'=' * 60}")
            print(f"🔮 LSTM PREDICTION")
            print(f"{'=' * 60}")

            df_clean = self._ensure_dataframe(df)

            missing = [c for c in self.features if c not in df_clean.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")

            steps = int(n_future) if n_future is not None else self.n_future

            data = df_clean[self.features].astype(float).values
            if data.shape[0] < self.seq_len:
                raise ValueError(f"Need at least {self.seq_len} rows for prediction (have {data.shape[0]})")

            scaled = self.scaler.transform(data)
            window = scaled[-self.seq_len:, :].copy()

            last_close_actual = float(df_clean[self.features[self.close_idx]].iloc[-1])
            recent_prices = df_clean[self.features[self.close_idx]].iloc[-50:].values

            print(f"💰 Current price: ${last_close_actual:.4f}")
            print(f"📊 Using {len(recent_prices)} recent prices for analysis")

            if self.last_prediction_price is not None:
                self.update_prediction_outcome(last_close_actual)

            x_in = window.reshape(1, self.seq_len, self.n_features)
            pred_scaled = self.model.predict(x_in, verbose=0)

            ref_row = window[-1, :].copy()
            raw_forecast = []

            if pred_scaled.shape[1] >= steps:
                for i in range(steps):
                    ref_copy = ref_row.copy()
                    ref_copy[self.close_idx] = pred_scaled[0, i]
                    inv_price = self.scaler.inverse_transform(ref_copy.reshape(1, -1))[0, self.close_idx]
                    raw_forecast.append(float(inv_price))
            else:
                current = window.copy()
                for _ in range(steps):
                    x_in = current.reshape(1, self.seq_len, self.n_features)
                    pred_s = float(self.model.predict(x_in, verbose=0).flatten()[0])
                    new_row = ref_row.copy()
                    new_row[self.close_idx] = pred_s
                    inv_price = self.scaler.inverse_transform(new_row.reshape(1, -1))[0, self.close_idx]
                    raw_forecast.append(float(inv_price))
                    current = np.vstack([current[1:, :], new_row])
                    ref_row = new_row

            raw_forecast = np.array(raw_forecast, dtype=np.float32)

            # Price anchoring
            if self.price_anchoring_enabled:
                forecast = self.anchor_predictions(raw_forecast, last_close_actual, recent_prices)

                if len(raw_forecast) > 0:
                    raw_change = (raw_forecast[-1] - raw_forecast[0]) / raw_forecast[0] * 100 if raw_forecast[
                                                                                                     0] > 0 else 0
                    anchored_change = (forecast[-1] - last_close_actual) / last_close_actual * 100
                    print(f"🔗 Price Anchoring: Raw ${raw_forecast[0]:.2f}→${raw_forecast[-1]:.2f} ({raw_change:+.2f}%)")
                    print(f"   Anchored: ${last_close_actual:.2f}→${forecast[-1]:.2f} ({anchored_change:+.2f}%)")
            else:
                forecast = raw_forecast

            # Magnitude correction
            if self.magnitude_correction_enabled:
                forecast = self.correct_magnitude(forecast, last_close_actual, recent_prices)

            # Prediction adjustment
            predict_adj_coef = 1.0
            if self.predadjcoef_enabled:
                predict_adj_coef = self.calculate_prediction_adjustment(
                    forecast, last_close_actual, recent_prices
                )
                changes = forecast - last_close_actual
                adjusted_changes = changes * predict_adj_coef
                forecast = last_close_actual + adjusted_changes
                print(f"🔧 Adjustment coefficient: {predict_adj_coef:.4f}")

            # Calculate confidence
            ml_confidence = self.calculate_ml_confidence(
                forecast, last_close_actual, recent_prices, predict_adj_coef
            )

            # Determine direction
            price_change_pct = (forecast[-1] - last_close_actual) / last_close_actual * 100

            if price_change_pct > self.direction_threshold:
                ml_prediction = 1
            elif price_change_pct < -self.direction_threshold:
                ml_prediction = -1
            else:
                ml_prediction = 0

            if ml_confidence < self.confidence_threshold:
                ml_prediction = 0

            # Store for future updates
            self.last_prediction_price = last_close_actual
            self.last_prediction_direction = ml_prediction
            self.last_forecast = forecast.copy()
            self.confidence_history.append(ml_confidence)

            # Output
            direction_str = "BULLISH 🟢" if ml_prediction == 1 else "BEARISH 🔴" if ml_prediction == -1 else "NEUTRAL ⚪"
            regime_str = self.current_regime.upper()

            print("\n" + "=" * 60)
            print(f"🤖 ML PREDICTION RESULT")
            print("=" * 60)
            print(f"📈 Direction:     {direction_str}")
            print(f"📊 Confidence:    {ml_confidence * 100:.1f}%")
            print(f"💰 Current Price: ${last_close_actual:.4f}")
            print(f"🎯 Target Price:  ${forecast[-1]:.4f} ({price_change_pct:+.2f}%)")
            print(f"📉 Regime:        {regime_str}")
            print(f"✅ Rolling Acc:   {self.metrics.rolling_directional_accuracy * 100:.1f}%")
            print("=" * 60 + "\n")

            return ml_confidence, ml_prediction, forecast

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            traceback.print_exc()
            return 0.0, 0, None

    # ==================== CONFIGURATION ====================

    def enable_prediction_adjustment(self, strength: float = 2.0, bounds: tuple = (0.2, 3.0)):
        """Enable adaptive prediction adjustment"""
        self.predadjcoef_enabled = True
        self.adj_coef_strength = strength
        self.adj_coef_bounds = bounds
        print(f"✅ Prediction adjustment enabled: strength={strength}, bounds={bounds}")

    def disable_prediction_adjustment(self):
        """Disable prediction adjustment"""
        self.predadjcoef_enabled = False
        print("❌ Prediction adjustment disabled")

    def enable_price_anchoring(self, strength: float = 0.95, max_change: float = 2.0):
        """Enable price anchoring"""
        self.price_anchoring_enabled = True
        self.anchor_strength = strength
        self.max_prediction_change_pct = max_change
        print(f"✅ Price anchoring enabled: strength={strength}, max_change={max_change}%")

    def disable_price_anchoring(self):
        """Disable price anchoring"""
        self.price_anchoring_enabled = False
        print("❌ Price anchoring disabled")

    def enable_magnitude_correction(self):
        """Enable magnitude correction"""
        self.magnitude_correction_enabled = True
        print("✅ Magnitude correction enabled")

    def disable_magnitude_correction(self):
        """Disable magnitude correction"""
        self.magnitude_correction_enabled = False
        print("❌ Magnitude correction disabled")

    def set_direction_thresholds(self, noise: float = 0.08, direction: float = 0.15, strong: float = 0.30):
        """Set directional thresholds"""
        self.noise_threshold = noise
        self.direction_threshold = direction
        self.strong_signal_threshold = strong
        print(f"📊 Thresholds set: noise={noise}%, direction={direction}%, strong={strong}%")

    def set_volatility_params(self, expected_vol: float = 0.5, multiplier: float = 3.0):
        """Set volatility parameters for anchoring"""
        self.expected_volatility_pct = expected_vol
        self.volatility_multiplier = multiplier
        print(f"📊 Volatility params: expected={expected_vol}%, multiplier={multiplier}x")

    def reset_metrics(self):
        """Reset all tracking metrics"""
        self.prediction_history.clear()
        self.metrics = ModelMetrics()
        self.last_prediction_price = None
        self.last_prediction_direction = None
        self.last_forecast = None
        self.confidence_history.clear()
        self.outcome_history.clear()
        self.error_history.clear()
        self.magnitude_error_history.clear()
        self.adjustment_history.clear()
        self.scale_factor_history.clear()
        self.learned_scale_factor = 1.0
        print("🔄 Metrics reset")

    # ==================== METRICS & DIAGNOSTICS ====================

    def get_accuracy(self) -> float:
        """Get current rolling directional accuracy"""
        return self.metrics.rolling_directional_accuracy * 100

    def get_metrics(self) -> dict:
        """Get comprehensive metrics"""
        return {
            **self.metrics.to_dict(),
            'val_rmse': self.val_rmse,
            'directional_accuracy': self.directional_accuracy,
            'total_adjustments': len(self.adjustment_history),
            'avg_adjustment': np.mean(list(self.adjustment_history)) if self.adjustment_history else 1.0,
            'current_regime': self.current_regime,
            'learned_scale_factor': self.learned_scale_factor,
            'expected_volatility': self.expected_volatility_pct,
        }

    def get_adjustment_stats(self) -> dict:
        """Get adjustment statistics"""
        if not self.adjustment_history:
            return {'message': 'No adjustment history available'}

        adjustments = list(self.adjustment_history)
        return {
            'count': len(adjustments),
            'mean': float(np.mean(adjustments)),
            'std': float(np.std(adjustments)),
            'min': float(min(adjustments)),
            'max': float(max(adjustments)),
            'last': float(adjustments[-1]) if adjustments else None,
            'recent_5': [float(a) for a in adjustments[-5:]]
        }

    def get_last_diagnostics(self) -> dict:
        """Get diagnostics from last data validation"""
        return self.last_data_diagnostics

    def print_diagnostics(self):
        """Print comprehensive diagnostics"""
        print("\n" + "=" * 70)
        print("📊 MODEL DIAGNOSTICS")
        print("=" * 70)

        print(f"\n🎯 ACCURACY METRICS:")
        print(f"   Rolling Directional: {self.metrics.rolling_directional_accuracy * 100:.1f}%")
        print(f"   Total Predictions:   {self.metrics.total_predictions}")
        print(f"   Correct Predictions: {self.metrics.correct_predictions}")
        print(f"   Avg Error:           {self.metrics.avg_error_pct:.2f}%")
        print(f"   Avg Magnitude Error: {self.metrics.avg_magnitude_error:.2f}%")

        print(f"\n⚖️ CALIBRATION:")
        print(f"   Confidence/Accuracy: {self.metrics.confidence_calibration:.2f}")
        status = "✅ Well calibrated" if 0.8 < self.metrics.confidence_calibration < 1.2 else \
            "⚠️ Overconfident" if self.metrics.confidence_calibration > 1.2 else \
                "⚠️ Underconfident"
        print(f"   Status:              {status}")

        print(f"\n🔗 PRICE ANCHORING:")
        print(f"   Enabled:             {self.price_anchoring_enabled}")
        print(f"   Scale Factor:        {self.learned_scale_factor:.4f}")
        print(f"   Expected Volatility: {self.expected_volatility_pct:.3f}%")
        print(f"   Max Change/Candle:   {self.max_prediction_change_pct:.2f}%")

        print(f"\n🔧 ADJUSTMENT STATS:")
        adj_stats = self.get_adjustment_stats()
        if 'message' not in adj_stats:
            print(f"   Mean:    {adj_stats['mean']:.4f}")
            print(f"   Std:     {adj_stats['std']:.4f}")
            print(f"   Range:   [{adj_stats['min']:.4f}, {adj_stats['max']:.4f}]")

        print(f"\n📉 REGIME:")
        print(f"   Current: {self.current_regime}")
        if self.regime_history:
            regime_counts = {}
            for r in self.regime_history:
                regime_counts[r] = regime_counts.get(r, 0) + 1
            print(f"   Recent:  {dict(regime_counts)}")

        print(f"\n📋 LAST DATA DIAGNOSTICS:")
        if self.last_data_diagnostics:
            for key, value in self.last_data_diagnostics.items():
                if key != 'original_head':  # Skip large output
                    print(f"   {key}: {value}")

        print("=" * 70 + "\n")

    def show_adjustment_stats(self):
        """Backward compatibility method"""
        stats = self.get_adjustment_stats()
        if isinstance(stats, dict) and 'message' not in stats:
            print(f"Adjustment Stats:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Last: {stats['last']:.4f}")
        else:
            print("No adjustment history available")
