from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from .base_ML import BaseMLModel

from sklearn.preprocessing import MinMaxScaler
class RandomForestModel(BaseMLModel):
    def __init__(self):
        super().__init__()
        self.name = "Random Forest Regressor"
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        self.close_idx: int = 0  # index of close price in features
        self.seq_len: int = 1  # sequence length for windowed features
        self.scaler = None
        self._accuracy = None
        self._mae = None

    def train(self, df, test_size=0.2):
        """
        Train RF regressor to predict actual close price.
        """
        self.is_trained = False
        df = df.copy().dropna()

        # Features = everything except Close
        self.features = [c for c in df.columns if c != "Close"]
        X = df[self.features]
        y = df["Close"].values

        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)

        # Validation metrics
        self._accuracy = r2_score(y_val, y_pred)*100  # R² score
        self._mae = mean_absolute_error(y_val, y_pred)

        self.is_trained = True
        return True

    def predict(self, data, n_future: int = 1):
        """
        Forecast closing prices.
        Returns:
        - ml_confidence: reliability of prediction (0-1)
        - trend_signal: 1 bullish / -1 bearish
        - forecast: array of future close prices
        """
        if not self.is_trained or self.scaler is None:
            raise Exception("Model not trained yet.")

        # --- Prepare input ---
        if isinstance(data, pd.Series):
            input_data = pd.DataFrame([data[self.features].values], columns=self.features)
        elif isinstance(data, pd.DataFrame):
            input_data = data[self.features]
        else:
            raise ValueError("Unsupported input type for prediction")

        input_scaled = self.scaler.transform(input_data)
        last_close = float(data["Close"].iloc[-1]) if "Close" in data else None

        # === Single-step prediction ===
        if n_future == 1:
            pred_close = float(self.model.predict(input_scaled)[0])

            # Confidence estimation from variance of trees
            preds_per_tree = np.array([t.predict(input_scaled)[0] for t in self.model.estimators_])
            pred_std = np.std(preds_per_tree)
            ml_confidence = float(np.clip(1 - pred_std / (pred_close + 1e-6), 0, 1))

            trend_signal = 1 if pred_close > last_close else -1
            return ml_confidence, trend_signal, np.array([pred_close])

        # === Multi-step forecasting ===
        steps = max(1, int(n_future))
        forecast = []

        current = input_scaled[-self.seq_len:, :].copy()

        for _ in range(steps):
            x_in = current.flatten().reshape(1, -1)
            pred_close = float(self.model.predict(x_in)[0])
            forecast.append(pred_close)

            # Build artificial next row: reuse last features but replace Close with predicted
            new_row = current[-1, :].copy()
            if self.close_idx < new_row.shape[0]:
                new_row[self.close_idx] = pred_close
            current = np.vstack([current[1:, :], new_row])

        forecast = np.array(forecast, dtype=np.float32)

        # Confidence = stability across trees on last step
        preds_per_tree = np.array([t.predict(x_in)[0] for t in self.model.estimators_])
        pred_std = np.std(preds_per_tree)
        ml_confidence = float(np.clip(1 - pred_std / (np.mean(preds_per_tree) + 1e-6), 0, 1))

        # Trend signal from last forecasted close vs last known close
        if forecast[-1] > last_close:
            trend_signal = 1
        else:
            trend_signal = -1

        return ml_confidence, trend_signal, forecast
