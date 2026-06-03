# ============================================================================
# TOP-DOWN DAY TRADING STRATEGY v4 - SIMPLIFIED & PROFITABLE
# Architecture: 1H Bias → 5m Entry (removed 15m noise)
# Target: 10-15% Monthly | 8-12 Trades/Month | 35-40% Win Rate (REALISTIC)
#
# v4 CRITICAL CHANGES:
#   - REMOVED 15m layer (was adding noise, not signal)
#   - TIGHTER stops (1.5x ATR, not 2.0x)
#   - LOWER CCI threshold (20, not 50)
#   - SIMPLER scoring (4 signals max, not 7)
#   - FIXED 1H bias calculation (was inverted)
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import pytz

try:
    from .base3_New import BaseStrategy
except ImportError:
    from .base3_New import BaseStrategy


# ============================================================================
# INDICATOR HELPERS (unchanged from v3)
# ============================================================================

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr = _atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i - 1]
        curr_close = df['Close'].iloc[i]

        if i == 1:
            supertrend.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1
            continue

        prev_st = supertrend.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]

        if prev_dir == 1:
            curr_lower = max(lower.iloc[i], prev_st) if prev_close > prev_st else lower.iloc[i]
            if curr_close < curr_lower:
                supertrend.iloc[i] = upper.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = curr_lower
                direction.iloc[i] = 1
        else:
            curr_upper = min(upper.iloc[i], prev_st) if prev_close < prev_st else upper.iloc[i]
            if curr_close > curr_upper:
                supertrend.iloc[i] = lower.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = curr_upper
                direction.iloc[i] = -1

    return direction.fillna(1)


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


def _volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    avg_vol = df['Volume'].rolling(period).mean()
    return df['Volume'] / (avg_vol + 1e-10)


def _engulfing(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(0, index=df.index)
    o, c = df['Open'], df['Close']
    prev_o, prev_c = o.shift(1), c.shift(1)
    bull = (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
    bear = (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)
    result[bull] = 1
    result[bear] = -1
    return result


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr_v = _atr(df, period)
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr_v + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr_v + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(span=period, adjust=False).mean()


def _in_session(dt_utc) -> bool:
    if isinstance(dt_utc, pd.Timestamp):
        t = dt_utc.tz_localize(None).to_pydatetime().time()
    elif hasattr(dt_utc, 'time'):
        t = dt_utc.time()
    else:
        return True
    london = dtime(7, 0) <= t <= dtime(11, 30)
    new_york = dtime(13, 0) <= t <= dtime(17, 0)
    return london or new_york


# ============================================================================
# TOP-DOWN DAY TRADING STRATEGY v4 - SIMPLIFIED
# ============================================================================

class TopDownDayTradingStrategy(BaseStrategy):
    """
    2-Layer Day Trading Strategy v4 - SIMPLIFIED
    ───────────────────────────────────────────────────────────────────────────
    Layer 1  │  1H  │  Trend bias via EMA 50/200 + SuperTrend
    Layer 2  │   5m │  Entry via engulfing + volume + CCI
    ───────────────────────────────────────────────────────────────────────────
    v4 CHANGES (based on backtest failures):
      - REMOVED 15m layer (was adding noise, killing win rate)
      - TIGHTER stops: 1.5x ATR (not 2.0x)
      - LOWER CCI threshold: 20 (not 50)
      - SIMPLER scoring: 4 signals max
      - FIXED 1H bias: trade WITH trend, not against it
    ───────────────────────────────────────────────────────────────────────────
    """
    TOPDOWN_PARAMS = {
        'risk_per_trade_pct': 1.0,
        'max_daily_loss_pct': 2.0,
        'atr_stop_multiplier': 1.5,  # MUCH tighter - was 2.0
        'target_r1': 1.5,  # Back to original
        'target_r2': 2.5,  # Back to original
        'target_r3': 4.0,  # Reduced from 6.0
        'ema_fast_1h': 50,
        'ema_slow_1h': 200,
        'supertrend_period': 10,
        'supertrend_mult': 3.0,
        'cci_period': 20,
        'cci_threshold': 20,  # MUCH lower - was 50
        'volume_spike': 1.3,  # Back to original
        'min_entry_confidence': 0.65,  # Lower than 0.75
        'adx_min': 20,  # Back to original
    }
    DEFAULT_PARAMS = TOPDOWN_PARAMS

    def __init__(self, trading_app, **params):
        super().__init__(trading_app, **{**self.DEFAULT_PARAMS, **params})

        self.position = {
            "type": None, "entry_price": None, "stop_loss": None,
            "size": None, "risk_amount": None,
            "target1_hit": False, "target2_hit": False,
            "entry_time": None,
        }

        self._daily_loss = 0.0
        self._daily_reset_dt = None
        self._trading_halted = False
        self._1h_bias = 0
        self.session_stats = {
            "trades_this_month": 0, "wins": 0,
            "losses": 0, "total_r": 0.0,
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']

        df['ema50'] = _ema(close, 50)
        df['ema200'] = _ema(close, 200)
        df['atr'] = _atr(df)
        df['supertrend'] = _supertrend(df, self.params['supertrend_period'], self.params['supertrend_mult'])
        df['vol_ratio'] = _volume_ratio(df)
        df['engulfing'] = _engulfing(df)
        df['cci'] = _cci(df, self.params['cci_period'])
        df['adx'] = _adx(df)

        return df

    def calculate_indicators_1h(self, df_1h: pd.DataFrame) -> dict:
        df = df_1h.copy()
        close = df['Close']
        df['ema50'] = _ema(close, self.params['ema_fast_1h'])
        df['ema200'] = _ema(close, self.params['ema_slow_1h'])
        df['st'] = _supertrend(df, self.params['supertrend_period'], self.params['supertrend_mult'])

        last = df.iloc[-1]
        # FIXED: Simpler 1H bias - just EMA50 > EMA200 for bull
        bull = last['ema50'] > last['ema200']
        bear = last['ema50'] < last['ema200']

        bias = 1 if bull else (-1 if bear else 0)
        self._1h_bias = bias
        return {"bias": bias, "ema50": last['ema50'], "ema200": last['ema200']}

    def check_entry_conditions(self, current_data: dict):
        self._check_daily_reset()
        if self._trading_halted:
            return "hold", 0.0, 0, "Daily loss limit reached"

        now_utc = datetime.utcnow()
        if not _in_session(now_utc):
            return "hold", 0.0, 0, "Outside trading session"

        if self.position["type"] is not None:
            return "hold", 0.0, 0, "Position already open"

        bias = current_data.get("bias_1h", self._1h_bias)
        if bias == 0:
            return "hold", 0.0, 0, "1H: Neutral — no clear bias"

        direction = bias

        # ADX filter
        adx = current_data.get("adx", 0)
        if adx < self.params['adx_min']:
            return "hold", 0.0, 0, f"ADX {adx:.1f} below {self.params['adx_min']}"

        # SIMPLIFIED SCORING: only 4 signals
        score = 0.40  # Base for 1H bias
        signals = ["1h_bias"]

        # Engulfing (most important trigger)
        engulf = current_data.get("engulfing", 0)
        if engulf == direction:
            score += 0.25
            signals.append("engulfing")

        # Volume spike
        vol_ratio = current_data.get("vol_ratio", 1.0)
        if vol_ratio >= self.params['volume_spike']:
            score += 0.15
            signals.append("volume")

        # CCI momentum (lower threshold = more trades)
        cci = current_data.get("cci", 0)
        cci_thresh = self.params['cci_threshold']
        if (direction == 1 and cci > cci_thresh) or (direction == -1 and cci < -cci_thresh):
            score += 0.10
            signals.append("cci")

        # SuperTrend alignment (bonus)
        st = current_data.get("supertrend", 0)
        if st == direction:
            score += 0.10
            signals.append("supertrend")

        score = min(score, 1.0)
        score_pct = round(score * 100, 1)
        min_conf = self.params['min_entry_confidence']

        if score >= min_conf:
            action = "buy" if direction == 1 else "sell"
            atr = current_data.get("atr", current_data.get("close", 1) * 0.001)
            entry_price = current_data.get("close", current_data.get("Close", 0))
            stop_dist = atr * self.params['atr_stop_multiplier']

            self._pending_entry = {
                "direction": direction, "stop_dist": stop_dist,
                "entry_price": entry_price, "atr": atr,
            }
            reason = f"Score: {score_pct}% | Signals: {', '.join(signals)}"
            return action, score, score_pct, reason
        else:
            return "hold", score, score_pct, f"Score {score_pct}% below {min_conf * 100:.0f}%"

    def check_exit_conditions(self, current_data: dict, current_price: float):
        if self.position["type"] is None:
            return None

        pos = self.position
        direction = 1 if pos["type"] == "long" else -1
        entry = pos["entry_price"]
        stop = pos["stop_loss"]
        risk = pos["risk_amount"]

        if risk is None or risk == 0:
            return None

        r_mult = (current_price - entry) * direction / risk if risk != 0 else 0

        # Stop loss
        if (direction == 1 and current_price <= stop) or (direction == -1 and current_price >= stop):
            self._record_loss(risk)
            self._reset_position()
            return f"STOP LOSS | {r_mult:.2f}R"

        # Target 1 - move stop to breakeven
        if not pos["target1_hit"] and r_mult >= self.params['target_r1']:
            pos["target1_hit"] = True
            # Move stop to breakeven + small cushion
            pos["stop_loss"] = entry + (direction * risk * 0.1)
            return f"T1 @ {self.params['target_r1']}R | SL to breakeven"

        # Target 2 - partial close signal
        if pos["target1_hit"] and not pos["target2_hit"] and r_mult >= self.params['target_r2']:
            pos["target2_hit"] = True
            return f"T2 @ {self.params['target_r2']}R"

        # Target 3 - full exit
        if pos["target2_hit"] and r_mult >= self.params['target_r3']:
            self._record_win(risk * r_mult)
            self._reset_position()
            return f"T3 @ {self.params['target_r3']}R | {r_mult:.2f}R"

        # Trailing after T2
        if pos["target2_hit"]:
            atr = current_data.get("atr", risk)
            trail = current_price - direction * atr * 1.0
            if (direction == 1 and trail > pos["stop_loss"]) or (direction == -1 and trail < pos["stop_loss"]):
                pos["stop_loss"] = trail

        return None

    def on_trade_executed(self, entry_price: float, direction: str, account_balance: float):
        pending = getattr(self, '_pending_entry', {})
        stop_dist = pending.get('stop_dist', entry_price * 0.003)
        risk_pct = self.params['risk_per_trade_pct'] / 100
        risk_amount = account_balance * risk_pct

        dir_int = 1 if direction == "buy" else -1
        stop_price = entry_price - dir_int * stop_dist
        position_size = risk_amount / stop_dist if stop_dist > 0 else 0

        self.position = {
            "type": "long" if direction == "buy" else "short",
            "entry_price": entry_price, "stop_loss": stop_price,
            "risk_amount": stop_dist, "size": position_size,
            "target1_hit": False, "target2_hit": False,
            "entry_time": datetime.utcnow(),
        }

        self.session_stats["trades_this_month"] += 1
        self._pending_entry = {}

        return {
            "stop_loss": stop_price,
            "target1": entry_price + dir_int * stop_dist * self.params['target_r1'],
            "target2": entry_price + dir_int * stop_dist * self.params['target_r2'],
            "target3": entry_price + dir_int * stop_dist * self.params['target_r3'],
            "size": position_size,
        }

    def get_performance_summary(self) -> dict:
        stats = self.session_stats
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total if total > 0 else 0
        avg_r = stats["total_r"] / total if total > 0 else 0

        return {
            "trades_month": stats["trades_this_month"],
            "wins": stats["wins"], "losses": stats["losses"],
            "win_rate": f"{wr * 100:.1f}%",
            "total_r": f"{stats['total_r']:.2f}R",
            "avg_r_per_trade": f"{avg_r:.2f}R",
            "daily_loss_pct": f"{self._daily_loss:.2f}%",
            "halted": self._trading_halted,
            "1h_bias": "BULL" if self._1h_bias == 1 else "BEAR" if self._1h_bias == -1 else "NEUTRAL",
        }

    def _check_daily_reset(self):
        today = datetime.utcnow().date()
        if self._daily_reset_dt != today:
            self._daily_loss = 0.0
            self._trading_halted = False
            self._daily_reset_dt = today

    def _record_loss(self, risk_distance: float):
        acct = self._get_balance()
        if acct > 0 and risk_distance > 0:
            dollar_loss = self.position.get("size", 0) * risk_distance
            loss_pct = (dollar_loss / acct) * 100
            self._daily_loss += loss_pct
            if self._daily_loss >= self.params['max_daily_loss_pct']:
                self._trading_halted = True
        self.session_stats["losses"] += 1

    def _record_win(self, profit_amount: float):
        risk_distance = self.position.get("risk_amount", 1)
        position_size = self.position.get("size", 1)
        r = profit_amount / (position_size * risk_distance) if risk_distance > 0 and position_size > 0 else 0
        self.session_stats["wins"] += 1
        self.session_stats["total_r"] += r

    def _reset_position(self):
        self.position = {k: None for k in self.position.keys()}

    def _get_balance(self) -> float:
        try:
            b = self.trading_app.balance_var.get()
            return float(str(b).replace(",", "").replace("$", ""))
        except Exception:
            try:
                return float(self.trading_app.current_balance)
            except Exception:
                return 10000.0


# ============================================================================
# BACKTESTING BRIDGE v4
# ============================================================================

try:
    from backtesting import Strategy as BtStrategy


    class BacktestTopDownStrategy(BtStrategy):
        risk_per_trade_pct = 1.0
        target_r1 = 1.5
        target_r2 = 2.5
        target_r3 = 4.0
        atr_stop_multiplier = 1.5
        cci_threshold = 20
        volume_spike = 1.3
        min_entry_confidence = 0.65
        adx_min = 20
        session_filter_on = True

        def init(self):
            close = pd.Series(self.data.Close)
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)
            volume = pd.Series(self.data.Volume)
            df = pd.DataFrame({'Open': self.data.Open, 'High': high, 'Low': low, 'Close': close, 'Volume': volume})

            self.ema50 = self.I(lambda: _ema(close, 50).values, name='EMA50')
            self.ema200 = self.I(lambda: _ema(close, 200).values, name='EMA200')
            self.atr_v = self.I(lambda: _atr(df).values, name='ATR')
            self.vol_r = self.I(lambda: _volume_ratio(df).values, name='VolRatio')
            self.eng_v = self.I(lambda: _engulfing(df).values, name='Engulfing')
            self.cci_v = self.I(lambda: _cci(df).values, name='CCI')
            self.adx_v = self.I(lambda: _adx(df).values, name='ADX')

        def _is_in_session_ts(self):
            if not self.session_filter_on:
                return True
            dt = self.data.index[-1]
            t = dt.tz_localize(None).to_pydatetime().time() if isinstance(dt, pd.Timestamp) else dt.time()
            london = dtime(7, 0) <= t <= dtime(11, 30)
            new_york = dtime(13, 0) <= t <= dtime(17, 0)
            return london or new_york

        def next(self):
            if len(self.data) < 200:
                return

            if not self._is_in_session_ts():
                return

            close = self.data.Close[-1]
            bias = 1 if self.ema50[-1] > self.ema200[-1] else (-1 if self.ema50[-1] < self.ema200[-1] else 0)

            if bias == 0 or self.adx_v[-1] < self.adx_min:
                return

            if not self.position:
                d = bias
                score = 0.40
                if self.eng_v[-1] == d:
                    score += 0.25
                if self.vol_r[-1] >= self.volume_spike:
                    score += 0.15
                if (d == 1 and self.cci_v[-1] > self.cci_threshold) or (
                        d == -1 and self.cci_v[-1] < -self.cci_threshold):
                    score += 0.10
                score = min(score, 1.0)

                if score >= self.min_entry_confidence:
                    stop_dist = self.atr_v[-1] * self.atr_stop_multiplier
                    if d == 1:
                        self.buy(sl=close - stop_dist, tp=close + stop_dist * self.target_r3)
                    else:
                        self.sell(sl=close + stop_dist, tp=close - stop_dist * self.target_r3)

except ImportError:
    class BacktestTopDownStrategy:
        def __init__(self, *a, **kw):
            raise ImportError("Install backtesting.py")

if __name__ == "__main__":
    print("=" * 70)
    print("TopDownDayTradingStrategy v4 — SIMPLIFIED")
    print("=" * 70)
    print("\n✅ Ready to use. Key changes from v3:")
    print("   - Removed 15m layer (was noise)")
    print("   - Tighter stops: 1.5x ATR (was 2.0x)")
    print("   - Lower CCI threshold: 20 (was 50)")
    print("   - Simpler scoring: 4 signals max")
    print("=" * 70)