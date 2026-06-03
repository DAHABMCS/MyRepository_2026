from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from .base_ML import BaseMLModel
from sklearn.metrics import accuracy_score


class XGBoostModel(BaseMLModel):
    def __init__(self):
        super().__init__()
        self.name = "XGBoost"
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.is_trained = False

    def train(self, df):
        self.is_trained = False
        df = df.dropna()
        X = df[self.features]
        y = df['Target']

        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        y_pred = self.model.predict(X_scaled)
        self._accuracy = accuracy_score(y, y_pred)
        self.is_trained = True
        return True

    def predict(self, df, n_future=1):
        if self.scaler is None:
            raise Exception("Model not trained yet.")
        X = df[self.features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[-1]
