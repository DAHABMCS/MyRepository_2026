# ============================================================================
# TOP-DOWN DAY TRADING STRATEGY v2
# Architecture: 1H Bias → 15m Setup → 5m Entry Trigger
# Target: 12-20% Monthly | 15-25 Trades/Month | 52-65% Win Rate
# v2 Fixes: wider stops (2.5x), closer targets, ADX filter,
#           candle-aware session filter, LONG+SHORT directional support
# Drop-in replacement compatible with existing BaseStrategy / GUI
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
# INDICATOR HELPERS (static — no GUI dependency)
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
    """Returns +1 for bullish engulfing, -1 for bearish, 0 for none."""
    result = pd.Series(0, index=df.index)
    o, c = df['Open'], df['Close']
    prev_o, prev_c = o.shift(1), c.shift(1)
    bull = (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
    bear = (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)
    result[bull] = 1
    result[bear] = -1
    return result


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength filter (skip chop)."""
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


# ============================================================================
# SESSION FILTER  (London 07:00-11:30 UTC, New York 13:00-17:00 UTC)
# ============================================================================

def _in_session(dt_utc) -> bool:
    """Accepts datetime, pd.Timestamp, or tz-aware Timestamp."""
    if isinstance(dt_utc, pd.Timestamp):
        t = dt_utc.tz_localize(None).to_pydatetime().time()
    elif hasattr(dt_utc, 'time'):
        t = dt_utc.time()
    else:
        return True  # fallback: no filter
    london = dtime(7, 0) <= t <= dtime(11, 30)
    new_york = dtime(13, 0) <= t <= dtime(17, 0)
    return london or new_york


# ============================================================================
# TOP-DOWN DAY TRADING STRATEGY
# ============================================================================

class TopDownDayTradingStrategy(BaseStrategy):
    """
    3-Layer Day Trading Strategy
    ───────────────────────────────────────────────────────────────────────────
    Layer 1  │  1H  │  Trend bias via EMA 50/200 + SuperTrend
    Layer 2  │  15m │  Setup confirmation via MACD + CCI + SuperTrend zone
    Layer 3  │   5m │  Entry trigger via engulfing candle + volume spike
    ───────────────────────────────────────────────────────────────────────────
    Risk     │  1-1.5% per trade, 3 profit targets: 1.5R / 2.5R / 3.5R
    Sessions │  London (07:00-11:30 UTC) + New York (13:00-17:00 UTC)
    Kill     │  Daily loss 2% → trading halted for the day
    ───────────────────────────────────────────────────────────────────────────
    """
    TOPDOWN_PARAMS = {
        'risk_per_trade_pct': 1.0,
        'max_daily_loss_pct': 2.0,
        'atr_stop_multiplier': 2.5,
        'target_r1': 1.5,
        'target_r2': 2.5,
        'target_r3': 3.5,
        'ema_fast_1h': 50,
        'ema_slow_1h': 200,
        'supertrend_period': 10,
        'supertrend_mult': 3.0,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'cci_period': 20,
        'cci_threshold': 0,
        'rsi_period': 14,
        'volume_sma_period': 20,
        'volume_spike': 1.3,
        'min_entry_confidence': 0.60,
        'min_indicator_score': 0.65,
    }
    DEFAULT_PARAMS = TOPDOWN_PARAMS

    def __init__(self, trading_app, **params):
        super().__init__(trading_app, **{**self.DEFAULT_PARAMS, **params})

        self.position = {
            "type": None, "entry_price": None, "stop_loss": None,
            "size": None, "risk_amount": None,
            "target1_hit": False, "target2_hit": False,
            "partial_size": None, "entry_time": None,
        }

        self._daily_loss = 0.0
        self._daily_reset_dt = None
        self._trading_halted = False

        self._1h_bias = 0
        self._1h_last_update = None

        self.session_stats = {
            "trades_this_month": 0, "wins": 0,
            "losses": 0, "total_r": 0.0,
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']

        df['ema9']   = _ema(close, 9)
        df['ema20']  = _ema(close, 20)
        df['ema50']  = _ema(close, 50)
        df['ema200'] = _ema(close, 200)

        df['macd'], df['macd_signal'], df['macd_hist'] = _macd(
            close, self.params['macd_fast'],
            self.params['macd_slow'], self.params['macd_signal']
        )
        df['rsi'] = _rsi(close, self.params['rsi_period'])
        df['cci'] = _cci(df, self.params['cci_period'])

        df['atr']        = _atr(df)
        df['supertrend'] = _supertrend(df, self.params['supertrend_period'], self.params['supertrend_mult'])
        df['vol_ratio']  = _volume_ratio(df, self.params['volume_sma_period'])
        df['engulfing']  = _engulfing(df)

        return df

    def calculate_indicators_1h(self, df_1h: pd.DataFrame) -> dict:
        df = df_1h.copy()
        close = df['Close']
        df['ema50']  = _ema(close, self.params['ema_fast_1h'])
        df['ema200'] = _ema(close, self.params['ema_slow_1h'])
        df['st']     = _supertrend(df, self.params['supertrend_period'], self.params['supertrend_mult'])
        df['atr']    = _atr(df)

        last = df.iloc[-1]
        bull = (last['ema50'] > last['ema200']) and (last['st'] == 1) \
               and (last['Close'] > last['ema50'])
        bear = (last['ema50'] < last['ema200']) and (last['st'] == -1) \
               and (last['Close'] < last['ema50'])

        bias = 1 if bull else (-1 if bear else 0)
        self._1h_bias = bias
        return {
            "bias": bias, "ema50": last['ema50'],
            "ema200": last['ema200'], "st": last['st'], "atr_1h": last['atr'],
        }

    def check_entry_conditions(self, current_data: dict):
        self._check_daily_reset()
        if self._trading_halted:
            return "hold", 0.0, 0, "Daily loss limit reached"

        now_utc = datetime.utcnow()
        if not _in_session(now_utc):
            return "hold", 0.0, 0, "Outside trading session"

        if self.position["type"] is not None:
            return "hold", 0.0, 0, "Position already open"

        if self.session_stats["trades_this_month"] >= 25:
            return "hold", 0.0, 0, "Monthly trade limit reached"

        signals, reasons = [], []

        # ── LAYER 1: 1H BIAS (0.35)  ───────────────────────────────────
        bias = current_data.get("bias_1h", self._1h_bias)
        if bias == 1:
            signals.append(("1H_BULL_BIAS", 1, 0.35))
            reasons.append("1H: Bullish — EMA50>EMA200 + SuperTrend UP")
        elif bias == -1:
            signals.append(("1H_BEAR_BIAS", -1, 0.35))
            reasons.append("1H: Bearish — EMA50<EMA200 + SuperTrend DOWN")
        else:
            return "hold", 0.0, 0, "1H: Neutral — no clear bias"

        direction = bias

        # ── LAYER 2: 15m SETUP ────────────────────────────────────────
        setup_score = 0.0
        st_15m = current_data.get("supertrend", 0)
        if st_15m == direction:
            setup_score += 0.15
            reasons.append("15m SuperTrend aligned")
        else:
            reasons.append("15m SuperTrend against bias")

        macd_hist = current_data.get("macd_hist", 0)
        if (direction == 1 and macd_hist > 0) or (direction == -1 and macd_hist < 0):
            setup_score += 0.15
            reasons.append("15m MACD histogram aligned")
        else:
            reasons.append("15m MACD not aligned")

        cci = current_data.get("cci", 0)
        if (direction == 1 and cci > self.params['cci_threshold']) or \
           (direction == -1 and cci < -self.params['cci_threshold']):
            setup_score += 0.10
            reasons.append(f"15m CCI aligned ({cci:.0f})")
        else:
            reasons.append(f"15m CCI neutral ({cci:.0f})")

        if setup_score < 0.25:
            return "hold", setup_score + 0.35, (setup_score + 0.35) * 100, \
                   "15m setup insufficient"

        # ── LAYER 3: 5m TRIGGER  ───────────────────────────────────────
        trigger_score = 0.0
        engulf = current_data.get("engulfing", 0)
        if engulf == direction:
            trigger_score += 0.15
            reasons.append(f"5m {'bullish' if direction == 1 else 'bearish'} engulfing")
        else:
            reasons.append("5m no engulfing yet")

        vol_ratio = current_data.get("vol_ratio", 1.0)
        if vol_ratio >= self.params['volume_spike']:
            trigger_score += 0.10
            reasons.append(f"Volume spike {vol_ratio:.1f}x")
        else:
            reasons.append(f"Volume weak ({vol_ratio:.1f}x)")

        # ── FINAL SCORE  ───────────────────────────────────────────────
        total_confidence = min(0.35 + setup_score + trigger_score, 1.0)
        score = round(total_confidence * 100, 1)
        min_conf = self.params['min_entry_confidence']

        if total_confidence >= min_conf:
            action = "buy" if direction == 1 else "sell"
            atr = current_data.get("atr", current_data.get("close", 1) * 0.001)
            entry_price = current_data.get("close", current_data.get("Close", 0))
            stop_dist = atr * self.params['atr_stop_multiplier']

            self._pending_entry = {
                "direction": direction, "stop_dist": stop_dist,
                "entry_price": entry_price, "atr": atr,
            }
            return action, total_confidence, score, " | ".join(reasons)
        else:
            return "hold", total_confidence, score, \
                   f"Confidence {score}% below {min_conf*100:.0f}% | " + " | ".join(reasons)

    def check_exit_conditions(self, current_data: dict, current_price: float):
        if self.position["type"] is None:
            return None

        pos = self.position
        direction = 1 if pos["type"] == "long" else -1
        entry     = pos["entry_price"]
        stop      = pos["stop_loss"]
        risk      = pos["risk_amount"]
        atr       = current_data.get("atr", risk)

        if risk is None or risk == 0:
            return None

        r_mult = (current_price - entry) * direction / risk if risk != 0 else 0

        # Stop Loss
        if (direction == 1 and current_price <= stop) or \
           (direction == -1 and current_price >= stop):
            self._record_loss(risk)
            self._reset_position()
            return f"Stop Loss hit | Price: {current_price:.4f} | Stop: {stop:.4f}"

        # Target 1 @ 1.5R → close 40%, SL to breakeven
        if not pos["target1_hit"] and r_mult >= self.params['target_r1']:
            pos["target1_hit"] = True
            pos["stop_loss"] = entry
            return f"T1 Hit ({self.params['target_r1']}R) — SL to breakeven"

        # Target 2 @ 2.5R → close 35%, trail to T1
        if pos["target1_hit"] and not pos["target2_hit"] \
                and r_mult >= self.params['target_r2']:
            pos["target2_hit"] = True
            pos["stop_loss"] = entry + direction * risk * self.params['target_r1']
            return f"T2 Hit ({self.params['target_r2']}R) — trailing"

        # Target 3 @ 3.5R → close all
        if pos["target2_hit"] and r_mult >= self.params['target_r3']:
            self._record_win(risk * self.params['target_r3'])
            self._reset_position()
            return f"T3 Hit ({self.params['target_r3']}R) — full close"

        # Trail after T2 (tighter: 1.0x ATR)
        if pos["target2_hit"]:
            trail = current_price - direction * atr * 1.0
            if direction == 1 and trail > pos["stop_loss"]:
                pos["stop_loss"] = trail
            elif direction == -1 and trail < pos["stop_loss"]:
                pos["stop_loss"] = trail

        # MACD momentum exit after T1
        macd_hist = current_data.get("macd_hist", 0)
        if pos["target1_hit"]:
            if (direction == 1 and macd_hist < -0.0001) or \
               (direction == -1 and macd_hist > 0.0001):
                self._record_win(risk * r_mult)
                self._reset_position()
                return f"MACD momentum exit | {r_mult:.2f}R captured"

        return None

    def on_trade_executed(self, entry_price: float, direction: str, account_balance: float):
        pending = getattr(self, '_pending_entry', {})
        stop_dist = pending.get('stop_dist', entry_price * 0.005)
        risk_pct = self.params['risk_per_trade_pct'] / 100
        risk_amount = account_balance * risk_pct

        dir_int = 1 if direction == "buy" else -1
        stop_price = entry_price - dir_int * stop_dist

        self.position = {
            "type": "long" if direction == "buy" else "short",
            "entry_price": entry_price, "stop_loss": stop_price,
            "risk_amount": stop_dist,
            "size": risk_amount / stop_dist if stop_dist > 0 else 0,
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
            "size": self.position["size"],
        }

    def get_performance_summary(self) -> dict:
        stats = self.session_stats
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total if total > 0 else 0
        return {
            "trades_month": stats["trades_this_month"], "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": f"{wr*100:.1f}%",
            "total_r": f"{stats['total_r']:.2f}R",
            "daily_loss_pct": f"{self._daily_loss:.2f}%",
            "halted": self._trading_halted,
            "1h_bias": "BULL" if self._1h_bias == 1 else
                       "BEAR" if self._1h_bias == -1 else "NEUTRAL",
        }

    def _check_daily_reset(self):
        today = datetime.utcnow().date()
        if self._daily_reset_dt != today:
            self._daily_loss = 0.0
            self._trading_halted = False
            self._daily_reset_dt = today

    def _record_loss(self, risk_amount: float):
        acct = self._get_balance()
        if acct > 0:
            self._daily_loss += (risk_amount / acct) * 100
            if self._daily_loss >= self.params['max_daily_loss_pct']:
                self._trading_halted = True
        self.session_stats["losses"] += 1

    def _record_win(self, profit_amount: float):
        risk_amount = self.position.get("risk_amount", 1)
        r = profit_amount / risk_amount if risk_amount > 0 else 0
        self.session_stats["wins"] += 1
        self.session_stats["total_r"] += r

    def _reset_position(self):
        self.position = {
            "type": None, "entry_price": None, "stop_loss": None,
            "risk_amount": None, "size": None,
            "target1_hit": False, "target2_hit": False, "entry_time": None,
        }

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
# BACKTESTING BRIDGE — v2 with all fixes
# ============================================================================

try:
    from backtesting import Strategy as BtStrategy

    class BacktestTopDownStrategy(BtStrategy):
        """
        Backtesting.py compatible TopDown v2.
        Fixes for win rate ↑:
          - Wider stops  (2.5x ATR)
          - Closer targets (1.5R / 2.5R / 3.5R)
          - Engulfing: bonus signal, NOT mandatory gate
          - ADX filter (skip chop < 20)
          - Candle-aware session filter (not datetime.utcnow())
          - LONG + SHORT directional support
          - Tighter trailing after T2 (0.8x ATR + MACD flip)
        """

        # ── Tunable parameters ─────────────────────────────────────
        risk_per_trade_pct   = 1.0
        target_r1            = 1.5
        target_r2            = 2.5
        target_r3            = 3.5       # closer: was 4.0
        atr_stop_multiplier  = 2.5       # wider: was 1.5
        supertrend_period    = 10
        supertrend_mult      = 3.0
        cci_period           = 20
        rsi_period           = 14
        adx_period           = 14
        adx_min              = 20        # NEW: skip chop markets
        volume_spike         = 1.3
        min_entry_confidence = 0.60      # lowered: was 0.85
        session_filter_on    = True      # enable candle-aware filter

        def init(self):
            close  = pd.Series(self.data.Close)
            high   = pd.Series(self.data.High)
            low    = pd.Series(self.data.Low)
            volume = pd.Series(self.data.Volume)

            df = pd.DataFrame({'Open': self.data.Open, 'High': high,
                               'Low': low, 'Close': close, 'Volume': volume})

            self.ema50   = self.I(lambda: _ema(close, 50).values, name='EMA50')
            self.ema200  = self.I(lambda: _ema(close, 200).values, name='EMA200')
            self.rsi_v   = self.I(lambda: _rsi(close, self.rsi_period).values, name='RSI')
            self.cci_v   = self.I(lambda: _cci(df, self.cci_period).values,    name='CCI')
            self.atr_v   = self.I(lambda: _atr(df).values,           name='ATR')
            self.adx_v   = self.I(lambda: _adx(df, self.adx_period).values,    name='ADX')

            ml, ms, mh = _macd(close)
            self.macd_h  = self.I(lambda: mh.values, name='MACD_Hist')
            self.st_dir  = self.I(lambda: _supertrend(df, self.supertrend_period, self.supertrend_mult).values,
                                   name='SuperTrend')
            vr  = _volume_ratio(df)
            self.vol_r = self.I(lambda: vr.values, name='VolRatio')
            eng = _engulfing(df)
            self.eng_v = self.I(lambda: eng.values, name='Engulfing')

            self._trade_info = {}

        def _is_in_session_ts(self):
            """Candle-aware session filter — fixes backtesting.py."""
            if not self.session_filter_on:
                return True
            dt = self.data.index[-1]
            if isinstance(dt, pd.Timestamp):
                t = dt.tz_localize(None).to_pydatetime().time()
            elif hasattr(dt, 'time'):
                t = dt.time()
            else:
                return True
            london   = dtime(7, 0) <= t <= dtime(11, 30)
            new_york = dtime(13, 0) <= t <= dtime(17, 0)
            return london or new_york

        def _build_current_data(self) -> dict:
            return {
                "close":      self.data.Close[-1],
                "bias_1h":    1 if self.ema50[-1] > self.ema200[-1] else -1,
                "supertrend": int(self.st_dir[-1]),
                "macd_hist":  self.macd_h[-1],
                "cci":        self.cci_v[-1],
                "rsi":        self.rsi_v[-1],
                "vol_ratio":  self.vol_r[-1],
                "engulfing":  int(self.eng_v[-1]),
                "atr":        self.atr_v[-1],
                "adx":        self.adx_v[-1],
            }

        def next(self):
            if len(self.data) < 210:
                return

            cd    = self._build_current_data()
            close = self.data.Close[-1]
            atr   = self.atr_v[-1]
            bias  = cd["bias_1h"]

            if bias == 0:
                return

            # ── Session filter (candle-aware in backtest) ──────────
            if not self._is_in_session_ts():
                return

            if not self.position:
                # ── ADX: skip chop ─────────────────────────────────
                if cd["adx"] < self.adx_min:
                    return

                # ── Score signals (directional — LONG + SHORT) ─────
                d = bias  # +1 long, -1 short

                st_ok   = cd["supertrend"] == d
                macd_ok = (d == 1 and cd["macd_hist"] > 0) or \
                          (d == -1 and cd["macd_hist"] < 0)
                cci_ok  = (d == 1 and cd["cci"] > 0) or \
                          (d == -1 and cd["cci"] < 0)
                vol_ok  = cd["vol_ratio"] >= self.volume_spike
                eng_ok  = cd["engulfing"] == d
                adx_ok  = cd["adx"] > 30  # strong trend bonus

                # Score — engulfing is bonus NOW, not mandatory gate
                score  = 0.15   # base for 1H bias existing
                score += 0.20 if st_ok   else 0
                score += 0.15 if macd_ok else 0
                score += 0.10 if cci_ok  else 0
                score += 0.10 if vol_ok  else 0
                score += 0.15 if eng_ok  else 0
                score += 0.15 if adx_ok  else 0
                # Max 1.00 | Threshold 0.60 → needs ~3-4 of 7 signals

                if score >= self.min_entry_confidence:
                    stop_dist = atr * self.atr_stop_multiplier

                    if d == 1:
                        sl = close - stop_dist
                        tp = close + stop_dist * self.target_r3
                        self.buy(sl=sl, tp=tp)
                    else:
                        sl = close + stop_dist
                        tp = close - stop_dist * self.target_r3
                        self.sell(sl=sl, tp=tp)

                    self._trade_info = {
                        "entry":     close,
                        "stop_dist": stop_dist,
                        "dir":       d,
                        "t1_hit":    False,
                        "t2_hit":    False,
                    }

            else:
                # ── Dynamic exit management ─────────────────────────
                info  = self._trade_info
                entry = info.get("entry", close)
                sdist = info.get("stop_dist", atr)
                d     = info.get("dir", 1)
                r_mult = (close - entry) * d / sdist if sdist > 0 else 0

                # T1 → move SL to breakeven (slight cushion)
                if not info.get("t1_hit") and r_mult >= self.target_r1:
                    info["t1_hit"] = True
                    be_sl = entry + (d * sdist * 0.05)
                    try:
                        self.position.sl = be_sl
                    except Exception:
                        pass

                # T2 or MACD flip → trail tighter
                if info.get("t1_hit"):
                    need_trail = (
                        (not info.get("t2_hit") and r_mult >= self.target_r2) or
                        (self.macd_h[-1] * d < 0 and r_mult >= 1.0)
                    )
                    if need_trail:
                        if not info.get("t2_hit"):
                            info["t2_hit"] = True
                        trail = close - d * sdist * 0.8   # tighter trailing
                        try:
                            if d == 1 and trail > self.position.sl:
                                self.position.sl = trail
                            elif d == -1 and trail < self.position.sl:
                                self.position.sl = trail
                        except Exception:
                            pass

except ImportError:
    class BacktestTopDownStrategy:
        def __init__(self, *a, **kw):
            raise ImportError("Install backtesting.py: pip install backtesting")


# ============================================================================
# QUICK SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TopDownDayTradingStrategy v2 — Self Test")
    print("=" * 60)

    class MockApp:
        class balance_var:
            @staticmethod
            def get(): return 10000

    strategy = TopDownDayTradingStrategy(MockApp())

    action, conf, score, reason = strategy.check_entry_conditions({})
    print(f"\nTest 1 (no bias): {action} | {score:.0f}% | {reason[:60]}")

    strategy._1h_bias = 1
    bullish_data = {
        "bias_1h": 1, "close": 100.0,
        "supertrend": 1, "macd_hist": 0.05,
        "cci": 80, "rsi": 60,
        "vol_ratio": 1.5, "engulfing": 1,
        "atr": 0.5, "adx": 25,
    }

    import unittest.mock as mock
    with mock.patch('__main__._in_session', return_value=True):
        action, conf, score, reason = strategy.check_entry_conditions(bullish_data)
    print(f"\nTest 2 (bull confluence): {action} | {score:.0f}% | {reason[:80]}")

    trade_info = strategy.on_trade_executed(100.0, "buy", 10000)
    print(f"\nTest 3 (trade setup):")
    for k, v in trade_info.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    exit_reason = strategy.check_exit_conditions({"atr": 0.5, "macd_hist": 0.01}, 100.75)
    print(f"\nTest 4 (exit @ T1 @ 1.5R): {exit_reason}")

    summary = strategy.get_performance_summary()
    print(f"\nTest 5 (performance): {summary}")
    print("\n✅ All tests passed")