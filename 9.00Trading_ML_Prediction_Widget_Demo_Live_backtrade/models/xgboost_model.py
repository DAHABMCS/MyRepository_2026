from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from .base_ML import BaseMLModel
from sklearn.metrics import accuracy_score
import numpy as np


class XGBoostModel(BaseMLModel):
    def __init__(self):
        super().__init__()
        self.name = "XGBoost"
        # Define default features that should exist in your data
        self.features = [
            'RSI', 'ADX', 'MACD_Histogram', 'Volume_Ratio', 'Momentum',
            'ATR', 'Close', 'EMA_Fast', 'EMA_Mid', 'EMA_Slow',
            'CCI', 'Kalman_Strength', 'MACD', 'MACD_Signal',
            'Price_Percentile_20bar', 'BB_Width', 'CHOP'
        ]
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.scaler = None
        self.is_trained = False
        self._accuracy = 0.0
        self._used_features = []  # Store which features were actually used

    def train(self, df):
        """
        Train the XGBoost model with available features.

        Args:
            df: DataFrame with features and 'Target' column

        Returns:
            bool: True if training successful
        """
        self.is_trained = False

        # ─── FIX: Only use features that exist in the DataFrame ──────────────
        available_features = [f for f in self.features if f in df.columns]

        # Add 'Target' to the list to check
        if 'Target' not in df.columns:
            print("❌ XGBoost: 'Target' column not found in DataFrame")
            return False

        if len(available_features) < 3:
            print(f"⚠️ XGBoost: Only {len(available_features)} features available (need at least 3)")
            print(f"   Available: {available_features}")
            print(f"   Missing: {[f for f in self.features if f not in df.columns]}")
            return False

        # Store which features were actually used
        self._used_features = available_features

        # Prepare data - only use available features
        X = df[available_features].copy()
        y = df['Target'].copy()

        # Drop rows with NaN
        combined = pd.concat([X, y], axis=1).dropna()
        if len(combined) < 50:
            print(f"⚠️ XGBoost: Only {len(combined)} rows after cleaning (need at least 50)")
            return False

        X_clean = combined[available_features]
        y_clean = combined['Target']

        # Scale features
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Train model
        self.model.fit(X_scaled, y_clean)

        # Calculate accuracy
        y_pred = self.model.predict(X_scaled)
        self._accuracy = accuracy_score(y_clean, y_pred)
        self.is_trained = True

        print(f"✅ XGBoost trained successfully with {len(available_features)} features")
        print(f"   Accuracy: {self._accuracy:.2%}")
        print(f"   Features: {available_features[:5]}...")
        return True

    def predict(self, df, n_future=5):
        """
        Predict future trend direction.

        Args:
            df: DataFrame with features
            n_future: Number of future steps to forecast

        Returns:
            tuple: (confidence, prediction, forecast_prices)
        """
        if not self.is_trained or self.scaler is None:
            print("⚠️ XGBoost: Model not trained yet")
            return 0.0, 0, []

        # ─── FIX: Use only features that exist ──────────────────────────────
        available_features = [f for f in self._used_features if f in df.columns]
        if not available_features:
            print("⚠️ XGBoost: No available features for prediction")
            return 0.0, 0, []

        # Get the most recent complete row
        recent = df[available_features].iloc[-1:].copy()
        if recent.isna().any().any():
            print("⚠️ XGBoost: Recent data contains NaN values")
            return 0.0, 0, []

        # Scale and predict
        X_scaled = self.scaler.transform(recent)
        prediction = self.model.predict(X_scaled)[0]

        # Get confidence (probability of the predicted class)
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = proba.max()

        # Generate a simple forecast
        current_price = float(df['Close'].iloc[-1]) if 'Close' in df.columns else 100.0
        forecast = []

        # Directional movement based on prediction
        # +1 = bullish, -1 = bearish, 0 = neutral
        if prediction == 1:
            # Bullish: gradually increase
            for i in range(1, n_future + 1):
                change = 0.005 * (1 + 0.5 * np.random.random())
                forecast.append(current_price * (1 + change * i))
        elif prediction == -1:
            # Bearish: gradually decrease
            for i in range(1, n_future + 1):
                change = 0.005 * (1 + 0.5 * np.random.random())
                forecast.append(current_price * (1 - change * i))
        else:
            # Neutral: slight random walk
            for i in range(1, n_future + 1):
                change = 0.002 * (np.random.random() - 0.5)
                forecast.append(current_price * (1 + change * i))

        return confidence, prediction, forecast

    def get_accuracy(self):
        """Return the model's accuracy as a percentage."""
        return self._accuracy * 100 if self._accuracy else 0.0