import numpy as np
from datetime import datetime, timezone
from .base3_New import BaseStrategy,  ConfidenceManager


class TradingStrategy(BaseStrategy):
    """Enhanced trading strategy with confidence management and guaranteed equal buy/sell quantities"""

    def __init__(self, trading_app, config):
        super().__init__(trading_app)
        self.name = "StrategyName"  # ← ADD THIS LINE
        self.config = config
        self.position = {
            'type': None,
            'price': None,
            'quantity': None,  # This is the EXACT quantity we bought
            'time': None,
            'stop_loss': None,
            'trailing_stop': None,
            'swing_point': None,
            'breakeven_stop': None,
            'used_ml': False,
            'ml_prediction': None,
            'ml_confidence': None
        }
        self.confidence_manager = ConfidenceManager()
        self.ml_enabled = config.get('ml_enabled', False)
        self.current_ml_model = None
        self.bars_held = 0

        # Confidence thresholds
        self.min_indicator_threshold = config.get('min_indicator_threshold', 0.50)
        self.min_combined_threshold = config.get('min_combined_threshold', 0.65)

    def execute_strategy(self, current_data, df):
        """
        Enhanced strategy execution with guaranteed equal buy/sell quantities.
        """
        if not self._validate_market_data(df):
            if hasattr(self, 'log_message'):
                self.log_message("⚠️ No valid data to analyze", "orange")
            return

        # Update statistics
        if len(df) > 0:
            rolling_vol = df['Volume'].rolling(50)
            self.volume_mean = rolling_vol.mean().iloc[-1]
            self.atr_mean = df['ATR_closed'].mean() if 'ATR_closed' in df else 0

        # Get current data
        if hasattr(df, 'iloc'):
            current_data = df.iloc[-1].copy()
        current_price = current_data.get('Close', 0)

        # === Validate position integrity ===
        if not self.validate_position_integrity():
            self.force_position_reset()
            return

        # === Get indicator confidence ===
        indicator_confidence, raw_score, meets_indicator_threshold = self.check_entry_conditions(current_data)

        # === ML Prediction and Confidence ===
        ml_confidence = 0.0
        ml_prediction = 0
        combined_confidence = indicator_confidence
        use_ml = False

        if self.ml_enabled and self.current_ml_model is not None:
            try:
                # Prepare data for ML
                if hasattr(current_data, 'to_frame'):
                    ml_data = current_data.to_frame().T
                else:
                    ml_data = df.tail(1) if hasattr(df, 'tail') else df

                n_future = getattr(self, 'prediction_candles_slider', type('obj', (object,), {'get': lambda: 5})).get()
                ml_conf, ml_prediction, forecast = self.current_ml_model.predict(df, n_future)

                # Normalize ML confidence
                ml_confidence = float(ml_conf * 100.0 if ml_conf <= 1.0 else ml_conf)
                model_threshold = getattr(self.current_ml_model, "confidence_threshold", 0.65) * 100

                # Determine ML signal
                signal_text = "BULLISH" if ml_prediction == 1 else "BEARISH" if ml_prediction == -1 else "WAIT"
                signal_color = "green" if ml_prediction == 1 else "red" if ml_prediction == -1 else "yellow"

                if hasattr(self, 'log_message'):
                    self.log_message(f"🤖 ML Prediction: {signal_text} (Confidence: {ml_confidence:.2f}%)", signal_color)
                    if forecast is not None:
                        self.log_message(f"🕯️ Forecast: {forecast}", signal_color)

                # Check ML reliability and confidence
                if self.confidence_manager.is_ml_reliable() and ml_confidence >= model_threshold:
                    # ML is reliable and confident
                    combined_confidence = self.confidence_manager.combine_confidence(indicator_confidence,
                                                                                     ml_confidence)
                    use_ml = True

                    if hasattr(self, 'log_message'):
                        ml_weight = self.confidence_manager.get_prediction_weight()
                        accuracy = self.confidence_manager.get_prediction_accuracy()
                        self.log_message(
                            f"✅ ML reliable: accuracy={accuracy:.2f}, weight={ml_weight:.2f}, "
                            f"combined confidence={combined_confidence:.1f}%", "green"
                        )
                else:
                    # ML not reliable or not confident enough
                    if not self.confidence_manager.is_ml_reliable():
                        accuracy = self.confidence_manager.get_prediction_accuracy()
                        pred_count = len(self.confidence_manager.prediction_history)

                        if pred_count < self.confidence_manager.min_predictions:
                            reason = f"insufficient data ({pred_count}/{self.confidence_manager.min_predictions})"
                        else:
                            reason = f"low accuracy ({accuracy:.1f}% < 60%)"

                        if hasattr(self, 'log_message'):
                            self.log_message(f"⚠️ ML disabled: {reason}, using indicators only", "orange")
                    else:
                        if hasattr(self, 'log_message'):
                            self.log_message(
                                f"⚠️ ML confidence {ml_confidence:.1f}% below threshold {model_threshold:.0f}%, "
                                f"using indicators only", "orange"
                            )

                    combined_confidence = indicator_confidence

            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"ML prediction error: {e}")
                combined_confidence = indicator_confidence

        # === Logging ===
        if hasattr(self, 'log_message'):
            self.log_message(f"\n{'=' * 80}", "blue")
            self.log_message(f"📊 Indicator Confidence: {indicator_confidence:.1f}%",
                             "green" if meets_indicator_threshold else "orange")

            if use_ml:
                ml_weight = self.confidence_manager.get_prediction_weight()
                self.log_message(f"🤖 ML Confidence: {ml_confidence:.1f}% (weight: {ml_weight:.2f})", "blue")

            self.log_message(f"🎯 Combined Confidence: {combined_confidence:.1f}%",
                             "green" if combined_confidence >= (self.min_combined_threshold * 100) else "orange")

        # === Handle existing position ===
        if self.position['type'] is not None:
            self.bars_held += 1
            exit_reason = self.check_exit_conditions(current_data, current_price)

            if exit_reason:
                # Track prediction accuracy if we used ML for entry
                if self.position.get('used_ml'):
                    # Determine actual trend based on exit reason and P&L
                    actual_trend = 1 if current_price > self.position['price'] else -1
                    entry_prediction = self.position.get('ml_prediction', 0)
                    entry_confidence = self.position.get('ml_confidence', 0)

                    self.confidence_manager.add_prediction_result(
                        entry_prediction, actual_trend, entry_confidence
                    )

                # Execute exit with EXACT quantity from position
                self.close_position_with_exact_quantity(exit_reason, current_price)

                if hasattr(self, 'log_message'):
                    self.log_message(f"🔴 POSITION CLOSED: {exit_reason}", "red")

            return

        # === Check for new entry ===
        if self.position['type'] is None:
            # Check if market is ranging
            if current_data.get('Ranging', False):
                if hasattr(self, 'log_message'):
                    self.log_message("⏸️ Entry prevented: Market is ranging", "orange")
                return

            # Check confidence thresholds
            confidence_threshold = getattr(self, 'confidence_var', type('obj', (object,), {
                'get': lambda: self.min_combined_threshold})).get() * 100

            if not meets_indicator_threshold:
                if hasattr(self, 'log_message'):
                    self.log_message(f"❌ Entry blocked: Indicator confidence {indicator_confidence:.1f}% "
                                     f"below minimum {self.min_indicator_threshold * 100:.0f}%", "red")
                return

            if combined_confidence < confidence_threshold:
                if hasattr(self, 'log_message'):
                    self.log_message(f"❌ Entry blocked: Combined confidence {combined_confidence:.1f}% "
                                     f"below threshold {confidence_threshold:.0f}%", "red")
                return

            # Calculate position size
            usdt_balance = getattr(self, 'get_balance', lambda x: 1000)('USDT')  # Default for testing
            order_size_pct = getattr(self, 'order_size_var', type('obj', (object,), {'get': lambda: 0.1})).get()
            quantity = (usdt_balance * order_size_pct) / current_price

            if quantity > 0:
                # Execute entry and store EXACT quantity
                executed_quantity = self.execute_buy_order(current_price, quantity, combined_confidence)

                if executed_quantity > 0:
                    # Update position with EXACT executed quantity
                    self.position = {
                        'type': 'long',
                        'price': current_price,
                        'quantity': executed_quantity,  # Store EXACT quantity bought
                        'time': datetime.now(timezone.utc),
                        'stop_loss': current_price * 0.98,  # 2% stop loss
                        'trailing_stop': current_price * 0.98,
                        'swing_point': current_data.get('swing_low_closed', current_price),
                        'breakeven_stop': None,
                        'used_ml': use_ml,
                        'ml_prediction': ml_prediction if use_ml else None,
                        'ml_confidence': ml_confidence if use_ml else None
                    }

                    self.bars_held = 0

                    if hasattr(self, 'log_message'):
                        ml_text = f" with ML ({ml_confidence:.1f}%)" if use_ml else ""
                        self.log_message(
                            f"🟢 BUY EXECUTED{ml_text}\n"
                            f"Quantity: {executed_quantity:.8f} at ${current_price:.4f}\n"
                            f"Combined Confidence: {combined_confidence:.1f}%", "green"
                        )

    def _validate_market_data(self, df):
        """Validate market data quality"""
        if df is None or len(df) == 0:
            return False

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            return False

        # Check for NaN values
        if df[required_columns].isna().any().any():
            return False

        return True

    def execute_buy_order(self, price, quantity, confidence):
        """
        Execute buy order and return the EXACT quantity that was filled.
        This ensures we know exactly how much we bought.
        """
        try:
            if hasattr(self, 'place_order'):
                # Place the order and get the actual executed quantity
                order_result = self.place_order('buy', price, quantity, confidence=confidence)

                if order_result:
                    # Extract actual filled quantity from order result
                    # This depends on your exchange API response format
                    if isinstance(order_result, dict) and 'filled_quantity' in order_result:
                        return float(order_result['filled_quantity'])
                    elif isinstance(order_result, dict) and 'executedQty' in order_result:
                        return float(order_result['executedQty'])
                    else:
                        # If we can't get exact quantity, return the requested quantity
                        # But log a warning
                        if hasattr(self, 'log_message'):
                            self.log_message("⚠️ Could not get exact filled quantity, using requested quantity",
                                             "orange")
                        return quantity
                else:
                    return 0  # Order failed
            else:
                # Fallback for testing - assume full quantity was filled
                return quantity

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error executing buy order: {e}")
            return 0

    def close_position_with_exact_quantity(self, exit_reason, current_price):
        """
        Close position using the EXACT quantity we bought.
        This guarantees sell quantity equals buy quantity.
        """
        if not self.validate_position_before_sell():
            return False

        try:
            # Use the EXACT quantity from when we bought
            exact_quantity = self.position['quantity']

            if hasattr(self, 'log_message'):
                self.log_message(f"🔄 Closing position: {exact_quantity:.8f} units at ${current_price:.4f}", "blue")

            # Execute sell order with exact quantity
            if hasattr(self, 'place_order'):
                sell_result = self.place_order('sell', current_price, exact_quantity)

                if sell_result:
                    # Calculate P&L
                    buy_price = self.position['price']
                    pnl = (current_price - buy_price) * exact_quantity
                    pnl_pct = ((current_price / buy_price) - 1) * 100

                    if hasattr(self, 'log_message'):
                        color = "green" if pnl > 0 else "red"
                        self.log_message(
                            f"💰 P&L: ${pnl:.4f} ({pnl_pct:.2f}%)\n"
                            f"Buy: {exact_quantity:.8f} @ ${buy_price:.4f}\n"
                            f"Sell: {exact_quantity:.8f} @ ${current_price:.4f}\n"
                            f"Reason: {exit_reason}", color
                        )

                    # Reset position
                    self._reset_position()
                    return True
                else:
                    if hasattr(self, 'log_message'):
                        self.log_message("❌ Failed to execute sell order", "red")
                    return False
            else:
                # Fallback for testing - assume order was successful
                if hasattr(self, 'log_message'):
                    self.log_message(f"✅ Position closed (testing mode): {exact_quantity:.8f} units", "green")

                # Reset position
                self._reset_position()
                return True

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error closing position: {e}")
            if hasattr(self, 'log_message'):
                self.log_message(f"❌ Error closing position: {str(e)}", "red")
            return False

    def validate_position_before_sell(self):
        """Validate that we actually have a position before selling"""
        if self.position['type'] is None or self.position['quantity'] is None:
            if hasattr(self, 'log_message'):
                self.log_message("⚠️ No active position to sell", "orange")
            return False

        # Check if we actually have the asset in our balance
        if hasattr(self, 'get_balance'):
            symbol_balance = self.get_balance(self._get_base_symbol())
            position_quantity = self.position['quantity']

            if symbol_balance < position_quantity:
                if hasattr(self, 'log_message'):
                    self.log_message(f"⚠️ Position mismatch: Tracking {position_quantity}, "
                                     f"but only {symbol_balance} in account. Resetting position tracking.", "red")
                # Reset position tracking to match reality
                self._reset_position()
                return False

        # Additional validation: check if the position makes sense
        current_price = getattr(self, 'get_current_price', lambda: None)()
        if current_price and self.position['price']:
            position_value = position_quantity * self.position['price']
            if position_value <= 0 or position_value > 1000000:  # Sanity check
                if hasattr(self, 'log_message'):
                    self.log_message(f"⚠️ Invalid position value: ${position_value}. Resetting position.", "red")
                self._reset_position()
                return False

        return True

    def _get_base_symbol(self):
        """Helper to get base symbol"""
        if hasattr(self, 'symbol_var'):
            return self.symbol_var.get().split('-')[0]
        return 'SOL'  # Default

    def get_position_quantity(self):
        """
        Get the exact quantity of the current position.
        Useful for external checks and validations.
        """
        return self.position.get('quantity', 0) if self.position['type'] is not None else 0

    def validate_position_integrity(self):
        """
        Validate that position data is consistent.
        Returns True if position is valid or None, False if corrupted.
        """
        if self.position['type'] is None:
            return True  # No position is valid

        required_fields = ['price', 'quantity', 'time']
        for field in required_fields:
            if self.position.get(field) is None:
                if hasattr(self, 'log_message'):
                    self.log_message(f"❌ Position integrity check failed: missing {field}", "red")
                return False

        if self.position['quantity'] <= 0:
            if hasattr(self, 'log_message'):
                self.log_message(f"❌ Position integrity check failed: invalid quantity {self.position['quantity']}",
                                 "red")
            return False

        return True

    def debug_position_state(self):
        """Debug helper to log current position state"""
        if hasattr(self, 'log_message'):
            self.log_message("🐛 POSITION DEBUG STATE:", "magenta")
            for key, value in self.position.items():
                if key == 'quantity' and value is not None:
                    self.log_message(f"  {key}: {value:.8f}", "magenta")
                else:
                    self.log_message(f"  {key}: {value}", "magenta")
            self.log_message(f"  bars_held: {self.bars_held}", "magenta")

    def force_position_reset(self):
        """Emergency position reset - use if position gets corrupted"""
        if hasattr(self, 'log_message'):
            self.log_message("🚨 FORCE RESETTING POSITION", "red")

        self._reset_position()

        if hasattr(self, 'log_message'):
            self.log_message("✅ Position force reset complete", "green")

    def _reset_position(self):
        """Internal method to reset position state"""
        self.position = {
            'type': None, 'price': None, 'quantity': None, 'time': None,
            'stop_loss': None, 'trailing_stop': None, 'swing_point': None,
            'breakeven_stop': None, 'used_ml': False, 'ml_prediction': None, 'ml_confidence': None
        }
        self.bars_held = 0

    def _validate_critical_data(self, current_data):
        """Validate that critical data points exist"""
        required_fields = ['Close', 'SuperTrend_closed', 'MA_Fast_closed', 'MA_Slow_closed']
        return all(field in current_data for field in required_fields)

    def _assess_current_market_regime(self, current_data):
        """Assess current market regime (trending, ranging, volatile)"""
        # Simplified implementation
        adx = current_data.get('ADX_closed', 25)
        atr = current_data.get('ATR_closed', 1)

        if adx > 30:
            return 'trending'
        elif atr > getattr(self, 'atr_mean', 1) * 1.5:
            return 'volatile'
        else:
            return 'ranging'

    def _get_regime_adjusted_thresholds(self, regime):
        """Get thresholds adjusted for market regime"""
        base_thresholds = {
            'cci_threshold': -100,
            'volume_threshold': 1.2,
            'rsi_lower': 30,
            'rsi_upper': 70,
            'adx_threshold': 25
        }

        if regime == 'trending':
            base_thresholds['adx_threshold'] = 20
            base_thresholds['volume_threshold'] = 1.0
        elif regime == 'volatile':
            base_thresholds['rsi_lower'] = 25
            base_thresholds['rsi_upper'] = 75
            base_thresholds['volume_threshold'] = 1.5

        return base_thresholds

    def _calculate_risk_adjustment(self, current_data, market_regime):
        """Calculate risk-based penalty"""
        penalty = 0.0

        # High volatility penalty
        atr = current_data.get('ATR_closed', 1)
        if atr > getattr(self, 'atr_mean', 1) * 2:
            penalty += 0.1

        # Market regime penalty
        if market_regime == 'volatile':
            penalty += 0.05

        return min(penalty, 0.2)  # Cap at 20%

    def _get_calibrated_confidence(self, score, max_score):
        """Convert raw score to calibrated confidence percentage"""
        if max_score == 0:
            return 0

        raw_confidence = (score / max_score) * 100

        # Apply calibration curve (sigmoid-like)
        # Maps 0-100 raw to more realistic 0-90 range
        calibrated = 90 / (1 + np.exp(-(raw_confidence - 50) / 15))

        return max(0, min(90, calibrated))

    def update_trailing_stop(self, current_price, atr, swing_point, direction='long',
                             atr_mult=1.5, prev_tsl=None):
        """Update trailing stop loss"""
        if direction == 'long':
            new_tsl = max(
                current_price - (atr * atr_mult),
                swing_point,
                prev_tsl or 0
            )
        else:  # short
            new_tsl = min(
                current_price + (atr * atr_mult),
                swing_point,
                prev_tsl or float('inf')
            )

        return new_tsl

    def predict_future_trend(self, n_future=5):
        """Placeholder for ML prediction - replace with actual implementation"""
        # This should be implemented based on your ML model
        return (['bullish'] * n_future, 0.75)  # Placeholder

    def check_entry_conditions(self, current_data):
        """Placeholder - implement your specific entry logic"""
        return 75.0, 75.0, True  # confidence, raw_score, meets_threshold

    def check_exit_conditions(self, current_data, current_price):
        """Placeholder - implement your specific exit logic"""
        return None  # return exit reason or None