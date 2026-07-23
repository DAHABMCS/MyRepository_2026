import datetime
import  csv
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error


class PredictionLogger:
    def __init__(self, log_file='prediction_logs.csv'):
        self.log_file = log_file
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize the log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'symbol', 'interval', 'strategy',
                    'predicted', 'actual', 'confidence', 'model',
                    'position', 'entry_price', 'exit_reason', 'pnl'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def log_prediction(self, timestamp, symbol, interval, strategy,
                       predicted_value, actual_value, confidence, model_used,
                       position=None, entry_price=None, exit_reason=None, pnl=None):
        """Log prediction with trading context"""
        log_entry = {
            'timestamp': timestamp,
            'symbol': symbol,
            'interval': interval,
            'strategy': strategy,
            'predicted': predicted_value,
            'actual': actual_value,
            'confidence': confidence,
            'model': model_used,
            'position': position,
            'entry_price': entry_price,
            'exit_reason': exit_reason,
            'pnl': pnl
        }

        # Save to CSV
        with open(self.log_file, 'a', newline='') as csvfile:
            fieldnames = list(log_entry.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(log_entry)

        return log_entry

    def evaluate_predictions(self, symbol=None, model=None, strategy=None):
        """Evaluate prediction performance with detailed metrics"""
        df = pd.read_csv(self.log_file)

        # Filter by symbol, model, or strategy if specified
        if symbol:
            df = df[df['symbol'] == symbol]
        if model:
            df = df[df['model'] == model]
        if strategy:
            df = df[df['strategy'] == strategy]

        if len(df) == 0:
            print("No prediction data available for evaluation")
            return None

        # Calculate error metrics
        df['error'] = abs(df['predicted'] - df['actual'])
        df['error_pct'] = (df['error'] / df['actual']) * 100
        df['direction_correct'] = np.sign(df['predicted'] - df['actual'].shift(1)) == np.sign(
            df['actual'] - df['actual'].shift(1))

        # Basic metrics
        mae = mean_absolute_error(df['actual'], df['predicted'])
        mse = mean_squared_error(df['actual'], df['predicted'])
        rmse = np.sqrt(mse)
        mape = df['error_pct'].mean()
        r2 = r2_score(df['actual'], df['predicted'])

        # Direction accuracy
        direction_accuracy = df['direction_correct'].mean() * 100

        # Confidence analysis
        high_confidence = df[df['confidence'] > 0.7]
        low_confidence = df[df['confidence'] <= 0.7]

        print("=" * 60)
        print("PREDICTION PERFORMANCE EVALUATION")
        print("=" * 60)
        print(f"Total Predictions: {len(df)}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"R² Score: {r2:.4f}")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        print(f"High Confidence Predictions: {len(high_confidence)}")
        if len(high_confidence) > 0:
            print(
                f"  - High Confidence MAE: {mean_absolute_error(high_confidence['actual'], high_confidence['predicted']):.4f}")
            print(f"  - High Confidence Direction Accuracy: {high_confidence['direction_correct'].mean() * 100:.2f}%")

        return df

    def plot_predictions_vs_actuals(self, symbol=None, model=None, limit=100):
        """Plot predictions vs actual values"""
        df = pd.read_csv(self.log_file)

        if symbol:
            df = df[df['symbol'] == symbol]
        if model:
            df = df[df['model'] == model]

        if len(df) == 0:
            print("No data to plot")
            return

        # Limit data for better visualization
        df = df.tail(limit)

        plt.figure(figsize=(15, 8))

        # Plot actual and predicted values
        plt.plot(df['timestamp'], df['actual'], label='Actual', marker='o', linewidth=2)
        plt.plot(df['timestamp'], df['predicted'], label='Predicted', marker='x', linestyle='--')

        # Add confidence intervals
        confidence = df['confidence'] * 100
        plt.fill_between(df['timestamp'],
                         df['predicted'] * 0.95,
                         df['predicted'] * 1.05,
                         alpha=0.2, label='Confidence Band')

        plt.legend()
        plt.title(f'Predicted vs Actual Prices\n({symbol if symbol else "All Symbols"})')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_model_performance(self):
        """Compare performance across different models"""
        df = pd.read_csv(self.log_file)

        if len(df) == 0:
            return None

        performance = {}
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            mae = mean_absolute_error(model_data['actual'], model_data['predicted'])
            direction_accuracy = model_data['direction_correct'].mean() * 100
            avg_confidence = model_data['confidence'].mean() * 100

            performance[model] = {
                'predictions': len(model_data),
                'mae': mae,
                'direction_accuracy': direction_accuracy,
                'avg_confidence': avg_confidence
            }

        return performance

    def get_recent_predictions(self, limit=10):
        """Get recent predictions for analysis"""
        df = pd.read_csv(self.log_file)
        if len(df) == 0:
            return pd.DataFrame()
        return df.tail(limit)


class ConfidenceManager:
    """Manages prediction accuracy tracking and confidence weighting"""

    def __init__(self, history_size=100):
        self.prediction_history = deque(maxlen=history_size)
        self.accuracy_threshold = 0.60  # 60% minimum accuracy
        self.min_predictions = 10  # Minimum predictions before trusting accuracy

    def add_prediction_result(self, predicted_trend, actual_trend, confidence):
        """Add a prediction result to history"""
        self.prediction_history.append({
            'predicted': predicted_trend,
            'actual': actual_trend,
            'correct': predicted_trend == actual_trend,
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc)
        })

    def get_prediction_accuracy(self):
        """Calculate current prediction accuracy"""
        if len(self.prediction_history) < self.min_predictions:
            return 0.5  # Default neutral accuracy

        correct_predictions = sum(1 for p in self.prediction_history if p['correct'])
        return correct_predictions / len(self.prediction_history)

    def get_prediction_weight(self):
        """Get weight for predictions based on historical accuracy"""
        accuracy = self.get_prediction_accuracy()

        if accuracy < self.accuracy_threshold:
            return 0.0  # No weight if accuracy too low

        # Linear scaling from 0.6 (60%) to 1.0 (100%) accuracy
        # Maps to weight range 0.3 to 0.7
        weight = 0.3 + (accuracy - 0.6) * 1.0  # (0.7 - 0.3) / (1.0 - 0.6)
        return min(0.7, max(0.0, weight))

    def combine_confidence(self, indicator_confidence, ml_confidence):
        """Combine indicator and ML confidence based on ML accuracy"""
        ml_weight = self.get_prediction_weight()
        indicator_weight = 1.0 - ml_weight

        combined = (indicator_confidence * indicator_weight) + (ml_confidence * ml_weight)
        return combined

    def is_ml_reliable(self):
        """Check if ML predictions are reliable enough to use"""
        return self.get_prediction_accuracy() >= self.accuracy_threshold and len(
            self.prediction_history) >= self.min_predictions


class BaseMLModel:
    def __init__(self):
        self.name = "Base"
        self.features = [
            'Close', 'Volume', 'RSI_closed', 'MACD_closed', 'MACD_Signal_closed',
            'EMA_Fast_closed', 'EMA_Slow_closed', 'ATR_closed', 'SuperTrend_closed'
        ]
        self._accuracy = None
        self.scaler = None
        self.is_trained = False

    def get_accuracy(self):
        return self._accuracy if self._accuracy is not None else 0.0

    def score_prediction(self):
        return self.get_accuracy()

    def generate_target(self, df):
        """Default: Binary classification - will price go up next step?"""
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        return df

    def prepare_data(self, df):
        df = df.copy().dropna()  # Clean first
        df = self.generate_target(df)  # Add target
        df = df.dropna(subset=['Target'])  # Remove NaNs from target

        X = df[self.features]  # Features
        y = df['Target']  # Labels

        # Optional: Check shapes
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def calculate_stop_loss(entry_price, atr, swing_point, direction="long", atr_mult=2.0):
        """
        Hybrid Stop Loss logic:
        - Places SL at last swing point +/- ATR buffer
        - For longs: SL = min(swing_low, entry_price - atr_mult*atr)
        - For shorts: SL = max(swing_high, entry_price + atr_mult*atr)

        Args:
            entry_price (float): trade entry price
            atr (float): current ATR value
            swing_point (float): last swing high/low
            direction (str): "long" or "short"
            atr_mult (float): ATR multiplier

        Returns:
            float: calculated stop loss price
        """
        if direction == "long":
            sl = min(swing_point, entry_price - atr_mult * atr)
        else:
            sl = max(swing_point, entry_price + atr_mult * atr)
        return sl

    def update_trailing_stop(current_price, atr, swing_point, direction="long", atr_mult=1.5, prev_tsl=None):
        """
        Hybrid Trailing Stop logic:
        - Trails behind swing structure with ATR buffer
        - For longs: TSL = max(prev_tsl, swing_low - atr_mult*atr)
        - For shorts: TSL = min(prev_tsl, swing_high + atr_mult*atr)

        Args:
            current_price (float): latest price
            atr (float): current ATR value
            swing_point (float): last swing high/low
            direction (str): "long" or "short"
            atr_mult (float): ATR multiplier for buffer
            prev_tsl (float or None): previous trailing stop (if any)

        Returns:
            float: updated trailing stop price
        """
        if direction == "long":
            tsl = swing_point - atr_mult * atr
            if prev_tsl is not None:
                tsl = max(prev_tsl, tsl)  # only moves up
        else:
            tsl = swing_point + atr_mult * atr
            if prev_tsl is not None:
                tsl = min(prev_tsl, tsl)  # only moves down
        return tsl
class ConfidenceManager:
    """Manages prediction accuracy tracking and confidence weighting"""

    def __init__(self, history_size=100):
        self.prediction_history = deque(maxlen=history_size)
        self.accuracy_threshold = 0.60  # 60% minimum accuracy
        self.min_predictions = 10  # Minimum predictions before trusting accuracy

    def add_prediction_result(self, predicted_trend, actual_trend, confidence):
        """Add a prediction result to history"""
        self.prediction_history.append({
            'predicted': predicted_trend,
            'actual': actual_trend,
            'correct': predicted_trend == actual_trend,
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc)
        })

    def get_prediction_accuracy(self):
        """Calculate current prediction accuracy"""
        if len(self.prediction_history) < self.min_predictions:
            return 0.5  # Default neutral accuracy

        correct_predictions = sum(1 for p in self.prediction_history if p['correct'])
        return correct_predictions / len(self.prediction_history)

    def get_prediction_weight(self):
        """Get weight for predictions based on historical accuracy"""
        accuracy = self.get_prediction_accuracy()

        if accuracy < self.accuracy_threshold:
            return 0.0  # No weight if accuracy too low

        # Linear scaling from 0.6 (60%) to 1.0 (100%) accuracy
        # Maps to weight range 0.3 to 0.7
        weight = 0.3 + (accuracy - 0.6) * 1.0  # (0.7 - 0.3) / (1.0 - 0.6)
        return min(0.7, max(0.0, weight))

    def combine_confidence(self, indicator_confidence, ml_confidence):
        """Combine indicator and ML confidence based on ML accuracy"""
        ml_weight = self.get_prediction_weight()
        indicator_weight = 1.0 - ml_weight

        combined = (indicator_confidence * indicator_weight) + (ml_confidence * ml_weight)
        return combined

    def is_ml_reliable(self):
        """Check if ML predictions are reliable enough to use"""
        return self.get_prediction_accuracy() >= self.accuracy_threshold and len(
            self.prediction_history) >= self.min_predictions

