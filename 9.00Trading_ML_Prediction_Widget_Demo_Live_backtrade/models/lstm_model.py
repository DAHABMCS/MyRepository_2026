import numpy as np
import pandas as pd
from collections import deque
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class LSTMModel:
    def __init__(
            self,
            seq_len: int = 60,
            n_future: int = 5,
            min_train_acc: float = 55.0,
            confidence_threshold: float = 0.30,
            features: Optional[List[str]] = None,
            close_idx: int = 0,
            scaler: Optional[MinMaxScaler] = None
    ):
        # Core configuration
        self.seq_len = int(seq_len)
        self.sequence_length = int(seq_len)
        self.n_future = int(n_future)
        self.min_train_acc = float(min_train_acc)
        self.confidence_threshold = float(confidence_threshold)

        # Model state
        self.model: Optional[Sequential] = None
        self.history = None
        self.is_trained = False

        # Scaler
        self.scaler = scaler or MinMaxScaler(feature_range=(0, 1))

        # Learned data schema
        self.features: Optional[List[str]] = features or ["Close"]
        self.close_col: Optional[str] = None
        self.close_idx: Optional[int] = close_idx
        self.n_features: int = len(self.features)

        # Metrics
        self.train_accuracy: Optional[float] = None
        self.val_accuracy: Optional[float] = None
        self.val_rmse: Optional[float] = None
        self.val_mape: Optional[float] = None

        # Prediction adjustment configuration
        self.predadjcoef_enabled = False
        self.adj_coef_strength = 0.5
        self.adj_coef_bounds = (0.9, 1.1)
        self.adjustment_history = deque(maxlen=20)
        self.volatility_lookback = 10

    # ---------- Utilities ----------
    def _ensure_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        if df.empty:
            raise ValueError("No valid data after cleaning NaNs/Infs.")
        return df

    def _infer_close_col(self, cols: List[str]) -> str:
        for c in cols:
            if c.lower() == "close":
                return c
        return cols[0]

    def _check_and_set_schema(self, df: pd.DataFrame, features: Optional[List[str]]):
        if features is None:
            cols = list(df.columns)
        else:
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required feature columns in df: {missing}")
            cols = list(features)

        close_col = self._infer_close_col(cols)
        self.features = cols
        self.close_col = close_col
        self.close_idx = self.features.index(self.close_col)
        self.n_features = len(self.features)

    def _create_sequences_multivar(self, scaled: np.ndarray):
        X, y = [], []
        N = scaled.shape[0]
        if N <= self.seq_len:
            return np.array([]), np.array([])

        for i in range(N - self.seq_len):
            X.append(scaled[i: i + self.seq_len, :])
            y.append(scaled[i + self.seq_len, self.close_idx])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        return X, y

    def _inverse_close_batch(self, y_scaled: np.ndarray, ref_rows_scaled: np.ndarray) -> np.ndarray:
        m = y_scaled.shape[0]
        combo = ref_rows_scaled.copy()
        combo[:, self.close_idx] = y_scaled[:, 0]
        inv = self.scaler.inverse_transform(combo)
        return inv[:, self.close_idx]

    # ---------- Model ----------
    def build_model(self) -> Sequential:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.seq_len, self.n_features)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        return model

    def calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate annualized volatility percentage"""
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252)
        return annualized_vol * 100

    def is_overextended(self, prices: np.ndarray, current_price: float,
                        lookback_period: int = 14) -> tuple[bool, bool]:
        """Check if price is near recent highs (overbought) or lows (oversold)"""
        if len(prices) < lookback_period:
            return False, False

        recent_high = np.max(prices[-lookback_period:])
        recent_low = np.min(prices[-lookback_period:])
        range_size = recent_high - recent_low

        if range_size < 1e-10:
            return False, False

        position = (current_price - recent_low) / range_size
        overbought = position > 0.7
        oversold = position < 0.3

        return overbought, oversold

    def calculate_prediction_adjustment(self, forecast: np.ndarray, actual_price: float,
                                        recent_prices: np.ndarray) -> float:
        """
        Calculate adjustment coefficient based on forecast vs actual discrepancy
        """
        if forecast is None or len(forecast) == 0:
            return 1.0

        forecast_error_pct = (forecast[0] - actual_price) / actual_price
        recent_volatility = self.calculate_volatility(recent_prices[-self.volatility_lookback:])
        vol_normalizer = max(recent_volatility, 0.001)

        adj_coef = 1.0 + (forecast_error_pct / vol_normalizer) * self.adj_coef_strength
        min_bound, max_bound = self.adj_coef_bounds
        adj_coef = max(min_bound, min(max_bound, adj_coef))

        self.adjustment_history.append(adj_coef)
        return adj_coef

    def calculate_ml_confidence(self, forecast: np.ndarray, actual_price: float,
                                recent_prices: np.ndarray, predict_adj_coef: float = 1.0) -> float:
        """
        Enhanced confidence calculation with adjustment factor
        """
        adjustment_significance = abs(predict_adj_coef - 1.0) * 100

        if len(forecast) > 1:
            forecast_changes = np.diff(forecast)
            direction_consistency = np.std(forecast_changes) / (np.mean(np.abs(forecast_changes)) + 1e-10)
            consistency_score = 1.0 / (1.0 + direction_consistency)
        else:
            consistency_score = 0.5

        total_change_pct = (forecast[-1] - actual_price) / actual_price * 100
        signal_strength = min(1.0, abs(total_change_pct) / 0.5)

        recent_vol = self.calculate_volatility(recent_prices[-20:])
        forecast_vol = self.calculate_volatility(forecast) if len(forecast) > 1 else 0
        vol_ratio = forecast_vol / max(recent_vol, 0.01)
        volatility_score = 1.0 if vol_ratio < 2.0 else 1.0 / vol_ratio

        adjustment_confidence = 1.0 - min(1.0, adjustment_significance / 2.0)

        base_confidence = (
                consistency_score * 0.4 +
                signal_strength * 0.3 +
                volatility_score * 0.2 +
                adjustment_confidence * 0.1
        )

        overbought, oversold = self.is_overextended(recent_prices, actual_price)
        if (overbought and total_change_pct > 0) or (oversold and total_change_pct < 0):
            base_confidence *= 0.8

        return max(0.1, min(1.0, base_confidence))

    # ---------- Training ----------
    def train(self, df: pd.DataFrame, features: Optional[List[str]] = None, epochs: int = 100, batch_size: int = 32,
              verbose: int = 1) -> bool:
        try:
            df = self._ensure_dataframe(df)
            self._check_and_set_schema(df, features)

            data = df[self.features].astype(float).values
            scaled = self.scaler.fit_transform(data)

            X, y = self._create_sequences_multivar(scaled)
            if X.size == 0:
                raise ValueError(f"Not enough data to create sequences (need > seq_len={self.seq_len}).")

            split = int(0.8 * len(X))
            if split == 0 or split >= len(X):
                raise ValueError("Not enough data for a valid train/validation split.")

            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            self.model = self.build_model()
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            rlrop = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, min_lr=1e-10)

            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=[es, rlrop]
            )

            y_tr_pred_s = self.model.predict(X_train, verbose=0)
            y_va_pred_s = self.model.predict(X_val, verbose=0)

            tr_rmse_s = float(np.sqrt(mean_squared_error(y_train, y_tr_pred_s)))
            va_rmse_s = float(np.sqrt(mean_squared_error(y_val, y_va_pred_s)))

            self.train_accuracy = max(0.0, (1.0 - tr_rmse_s) * 100.0)
            self.val_accuracy = max(0.0, (1.0 - va_rmse_s) * 100.0)

            refs_train = scaled[self.seq_len: self.seq_len + len(y_train), :].copy()
            refs_val = scaled[self.seq_len + len(y_train): self.seq_len + len(y_train) + len(y_val), :].copy()

            y_tr_orig = self._inverse_close_batch(y_train, refs_train)
            y_tr_pred_orig = self._inverse_close_batch(y_tr_pred_s, refs_train)
            y_va_orig = self._inverse_close_batch(y_val, refs_val)
            y_va_pred_orig = self._inverse_close_batch(y_va_pred_s, refs_val)

            val_rmse = float(np.sqrt(mean_squared_error(y_va_orig, y_va_pred_orig)))
            val_mape = float(mean_absolute_percentage_error(y_va_orig, y_va_pred_orig) * 100.0)

            self.val_rmse = val_rmse
            self.val_mape = val_mape

            print("✅ Training Complete")
            print(f"📊 Scaled Train Accuracy (proxy):  {self.train_accuracy:.2f}%")
            print(f"📊 Scaled Val   Accuracy (proxy):  {self.val_accuracy:.2f}%")
            print(f"📉 Validation RMSE (original):     {val_rmse:.6f}")
            print(f"📉 Validation MAPE% (original):    {val_mape:.3f}%")

            self.is_trained = self.train_accuracy >= self.min_train_acc
            if not self.is_trained:
                print(f"⚠️ Model accuracy below {self.min_train_acc:.2f}% threshold")
                return False
            return True

        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.is_trained = False
            return False

    def predict(self, df: pd.DataFrame, n_future: int = 5):
        """
        Predict future CLOSE prices and return (ml_confidence, ml_prediction, forecast).
        """
        try:
            if self.model is None or not getattr(self, "is_trained", True):
                raise ValueError("Model not trained or not available. Call train() first.")

            if not hasattr(self, "features") or not hasattr(self, "close_idx"):
                raise AttributeError("Model schema not set. Set self.features/self.close_idx.")

            df_clean = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
            missing = [c for c in self.features if c not in df_clean.columns]
            if missing:
                raise ValueError(f"Prediction data missing feature columns: {missing}")

            steps = int(n_future) if n_future is not None else int(self.n_future)
            steps = max(1, steps)

            data = df_clean[self.features].astype(float).values
            if data.shape[0] < self.seq_len:
                raise ValueError(f"Need at least seq_len={self.seq_len} rows for prediction (have {data.shape[0]}).")

            if not hasattr(self, "scaler") or self.scaler is None:
                raise AttributeError("Scaler not found on model (self.scaler).")

            scaled = self.scaler.transform(data)
            window = scaled[-self.seq_len:, :].copy()
            current = window.copy()
            ref_row = current[-1, :].copy()
            last_close_actual = float(df_clean[self.features[self.close_idx]].iloc[-1])

            inv_preds = []
            for _ in range(steps):
                x_in = current.reshape(1, self.seq_len, self.n_features)
                pred_s = float(np.asarray(self.model.predict(x_in, verbose=0)).reshape(-1)[0])
                new_row = ref_row.copy()
                new_row[self.close_idx] = pred_s
                inv_price = self.scaler.inverse_transform(new_row.reshape(1, -1))[0, self.close_idx]
                inv_preds.append(float(inv_price))
                current = np.vstack([current[1:, :], new_row])
                ref_row = new_row

            forecast = np.array(inv_preds, dtype=np.float32).flatten()

            # ----- PREDICTION ADJUSTMENT COEFFICIENT -----
            predict_adj_coef = 1.0
            if self.predadjcoef_enabled and forecast is not None and len(forecast) > 0:
                recent_prices = df_clean[self.features[self.close_idx]].iloc[-20:].values
                predict_adj_coef = self.calculate_prediction_adjustment(
                    forecast, last_close_actual, recent_prices
                )

                original_forecast = forecast.copy()
                forecast = forecast * predict_adj_coef

                print(f"🔧 Forecast adjustment applied: {predict_adj_coef:.4f}")
                print(f"   Original first: {original_forecast[0]:.4f}, Adjusted first: {forecast[0]:.4f}")
                print(f"   Original last: {original_forecast[-1]:.4f}, Adjusted last: {forecast[-1]:.4f}")
                print(
                    f"   Actual price: {last_close_actual:.4f}, Error: {((forecast[0] - last_close_actual) / last_close_actual * 100):.4f}%")

            # ----- CONFIDENCE CALCULATION -----
            recent_prices = df_clean[self.features[self.close_idx]].iloc[-20:].values
            ml_confidence_frac = self.calculate_ml_confidence(
                forecast, last_close_actual, recent_prices, predict_adj_coef
            )

            # Determine prediction direction
            price_change_pct = (forecast[-1] - last_close_actual) / last_close_actual * 100

            if price_change_pct > 0.05:
                ml_prediction = 1
            elif price_change_pct < -0.05:
                ml_prediction = -1
            else:
                ml_prediction = 0

            # Apply confidence thresholds
            if ml_confidence_frac < 0.2:
                ml_prediction = 0
                ml_confidence_frac = 0.1

            if ml_confidence_frac < self.confidence_threshold:
                ml_prediction = 0
                ml_confidence_frac = self.confidence_threshold / 2

            print(
                f"📊 Final Prediction: {'BULLISH' if ml_prediction == 1 else 'BEARISH' if ml_prediction == -1 else 'NEUTRAL'}")
            print(f"📈 Confidence: {ml_confidence_frac * 100:.2f}%")
            print(f"💰 Price Change: {price_change_pct:.4f}%")
            print(f"🎯 Forecast Range: {forecast[0]:.4f} -> {forecast[-1]:.4f}")

            return ml_confidence_frac, ml_prediction, forecast

        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            return 0.0, 0, None

    def get_accuracy(self) -> float:
        return float(self.train_accuracy) if self.train_accuracy is not None else 0.0

    def metrics(self) -> dict:
        return {
            "val_rmse": self.val_rmse,
            "val_mape_percent": self.val_mape,
            "scaled_train_accuracy_percent": self.train_accuracy,
            "scaled_val_accuracy_percent": self.val_accuracy,
        }

    def show_adjustment_stats(self):
        if self.current_ml_model and hasattr(self.current_ml_model, 'get_adjustment_stats'):
            stats = self.current_ml_model.get_adjustment_stats()
            if isinstance(stats, dict):
                stats_text = f"Adjustment Stats:\nCount: {stats['count']}\nMean: {stats['mean']:.4f}\nLast: {stats['last']:.4f}"
                self.log_message(stats_text, "blue")
            else:
                self.log_message(stats, "blue")

    def enable_prediction_adjustment(self, strength: float = 0.5, bounds: tuple = (0.9, 1.1)):
        """Enable prediction adjustment coefficient"""
        self.predadjcoef_enabled = True
        self.adj_coef_strength = strength
        self.adj_coef_bounds = bounds
        print(f"✅ Prediction adjustment enabled: strength={strength}, bounds={bounds}")

    def disable_prediction_adjustment(self):
        """Disable prediction adjustment coefficient"""
        self.predadjcoef_enabled = False
        print("❌ Prediction adjustment disabled")

    def get_adjustment_stats(self):
        """Get statistics about recent adjustments"""
        if not self.adjustment_history:
            return "No adjustment history available"

        adjustments = list(self.adjustment_history)
        return {
            'count': len(adjustments),
            'mean': np.mean(adjustments),
            'std': np.std(adjustments),
            'min': min(adjustments),
            'max': max(adjustments),
            'last': adjustments[-1] if adjustments else None
        }