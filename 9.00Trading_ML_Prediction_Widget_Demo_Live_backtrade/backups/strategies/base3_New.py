import datetime
from datetime import datetime, timezone, timedelta
from collections import deque
import numpy as np
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field


class BaseStrategy:
    """Abstract base class that all trading strategies must inherit from.

    Providef log_message(self, message, color="white"):des core framework methods and decision-making logic that strategies
    can leverage or override. All trading strategies should implement the
    abstract methods to integrate with the trading application.
    """

    def __init__(self, trading_app, **params):
        """Initialize the base strategy with application reference and parameters.

        Args:
            trading_app: Reference to the main TradingApp instance
            **params: Strategy-specific parameters
        """
        self.trading_app = trading_app
        self.params = params
        self.confidence_manager = ConfidenceManager()  # Keep for backward compatibility
        self.min_indicator_threshold = 0.50  # 50% minimum indicator confidence
        self.min_combined_threshold = 0.65  # 65% minimum combined confidence

        # Initialize enhanced components (will be overridden by specific strategies)
        self.risk_manager = None
        self.performance_analytics = None
        self.alert_manager = None

    # ─────────────────────────────────────────────────────────────────────────
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ─────────────────────────────────────────────────────────────────────────

    def log_message(self, message, color="white"):
        """Log message to trading app"""
        if hasattr(self, 'trading_app') and self.trading_app:
            self.trading_app.log_message(message, color)
        else:
            print(message)

    def get_current_price(self):
        """Get current price from trading app"""
        if hasattr(self, 'trading_app') and self.trading_app:
            return self.trading_app.get_current_price()
        return None

    def place_order(self, side, quantity, price, **kwargs):
        """Place order through trading app"""
        if hasattr(self, 'trading_app') and self.trading_app:
            return self.trading_app.place_order(side, price, quantity, **kwargs)
        return False

    def play_notification(self, sound_type):
        """Play sound notification - must be implemented by subclass"""
        pass

    def get_balance(self, currency):
        """Get balance - must be implemented by subclass"""
        return 1000  # Default for testing

    def place_order(self, side, price, quantity, **kwargs):
        """Place order - must be implemented by subclass"""
        return True  # Default for testing

    def update_position_on_entry(self, price, quantity, confidence):
        """Update position on entry - must be implemented by subclass"""
        pass

    def handle_existing_position(self, current_price, current_data):
        """Handle existing position - must be implemented by subclass"""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# KEPT FOR BACKWARD COMPATIBILITY - Will be removed in future versions
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceManager:
    """DEPRECATED: Kept for backward compatibility only.
    Use MomentumLogic's fuzzy mode system instead."""

    def __init__(self, history_size=100):
        self.prediction_history = deque(maxlen=history_size)
        self.accuracy_threshold = 0.60
        self.min_predictions = 10

    def add_prediction_result(self, predicted_trend, actual_trend, confidence):
        self.prediction_history.append({
            'predicted': predicted_trend,
            'actual': actual_trend,
            'correct': predicted_trend == actual_trend,
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc)
        })

    def get_prediction_accuracy(self):
        if len(self.prediction_history) < self.min_predictions:
            return 0.5
        correct_predictions = sum(1 for p in self.prediction_history if p['correct'])
        return correct_predictions / len(self.prediction_history)

    def get_prediction_weight(self):
        accuracy = self.get_prediction_accuracy()
        if accuracy < self.accuracy_threshold:
            return 0.0
        weight = 0.3 + (accuracy - 0.6) * 1.0
        return min(0.7, max(0.0, weight))

    def combine_confidence(self, indicator_confidence, ml_confidence):
        ml_weight = self.get_prediction_weight()
        indicator_weight = 1.0 - ml_weight
        combined = (indicator_confidence * indicator_weight) + (ml_confidence * ml_weight)
        return combined

    def is_ml_reliable(self):
        return self.get_prediction_accuracy() >= self.accuracy_threshold and len(
            self.prediction_history) >= self.min_predictions